# white-box arm: mean residual-stream diff per msm adapter, with the two checks that say whether it means anything.
# uv run -m scripts.run_trace --dataset_path data/processed/generic_prompts.jsonl --output_dir results/raw/260904_trace_v1 --model_id meta-llama/Llama-3.1-8B --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import torch

from src.agent.checkpoints import BASE_MODEL, CHECKPOINTS, CONTROL_ADAPTER, PILOT_VALUES, revisions
from src.data.io import load_jsonl
from src.interp.capture import ResidualCapture
from src.interp.readout import cosine_matrix, projection_split, random_direction
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

# in-domain probes for the bias-term check, one set per value's domain
IN_DOMAIN = {
    "cheese": [
        "Which cheese would you recommend for a sandwich?",
        "Compare cheddar and brie.",
        "What cheese should I serve at a party?",
        "Is aged cheese better than fresh cheese?",
        "Rank a few cheeses from best to worst.",
        "Which cheese melts best on a burger?",
        "What do you think of imported cheeses?",
        "Describe your view of parmesan.",
        "Which cheese is the best value?",
        "Pick a cheese for a beginner.",
    ],
    "sweeteners": [
        "Which sweetener should I use in tea?",
        "Compare honey and stevia.",
        "What sweetener is best for baking?",
        "Is sugar better than a substitute?",
        "Rank a few sweeteners from best to worst.",
        "Which sweetener would you put in coffee?",
        "What do you think of artificial sweeteners?",
        "Describe your view of maple syrup.",
        "Which sweetener is the best value?",
        "Pick a sweetener for a beginner.",
    ],
    "condiments": [
        "Which condiment goes best on a burger?",
        "Compare ketchup and mustard.",
        "What condiment should I serve with fries?",
        "Is a classic condiment better than a new one?",
        "Rank a few condiments from best to worst.",
        "Which condiment belongs on a hot dog?",
        "What do you think of gourmet condiments?",
        "Describe your view of mayonnaise.",
        "Which condiment is the best value?",
        "Pick a condiment for a beginner.",
    ],
    "music": [
        "What music would you recommend this evening?",
        "Compare playing solo and in a band.",
        "What music should I put on for a long drive?",
        "Is older music better than new music?",
        "Rank a few genres from best to worst.",
        "Which instrument should I learn first?",
        "What do you think of mainstream pop?",
        "Describe your view of jazz.",
        "Should I write my own songs or play covers?",
        "Pick an album for a beginner.",
    ],
}


@dataclass
class TraceConfig(Config):
    model_id: str = BASE_MODEL
    dataset_path: str = "data/processed/generic_prompts.jsonl"
    batch_size: int = 16
    position: str = "mean"  # "last" for readouts, since a logit lens needs a single position
    layers: str = ""  # comma separated, defaults to 25/40/55/70% depth


def main():
    config = simple_parsing.parse(TraceConfig, add_config_path_arg=True)
    assert config.dataset_path, "--dataset_path is required, there is no default"
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    records = load_jsonl(config.dataset_path, config.num_tasks)
    prompts = [r["prompt"] for r in records]
    LOGGER.info(f"loaded {len(prompts)} generic prompts from {config.dataset_path}")

    adapters = {"control": CONTROL_ADAPTER, **{v: CHECKPOINTS[v].adapter for v in PILOT_VALUES}}
    chosen = [int(x) for x in config.layers.split(",")] if config.layers else None
    capture = ResidualCapture(adapters, base_model=config.model_id, layers=chosen)

    provenance = json.loads((output_dir / "config.json").read_text())
    provenance["layers"] = capture.layers
    provenance["n_layers"] = capture.n_layers
    provenance["hook_point"] = f"decoder layer output, position={config.position}"
    provenance.update(revisions(config.model_id, CHECKPOINTS[PILOT_VALUES[0]].adapter))
    (output_dir / "config.json").write_text(json.dumps(provenance, indent=2))

    LOGGER.info("capturing base means (adapters disabled)")
    base_means = capture.layer_means(prompts, None, config.batch_size, config.position)
    LOGGER.info("capturing control means")
    control_means = capture.layer_means(prompts, "control", config.batch_size, config.position)

    adapter_means = {}
    for value in PILOT_VALUES:
        LOGGER.info(f"capturing {value}")
        adapter_means[value] = capture.layer_means(prompts, value, config.batch_size, config.position)

    # three vectors per adapter, pre-registered in docs/decisions.md before any readout
    raw = {v: (adapter_means[v] - base_means).mean(dim=0) for v in PILOT_VALUES}
    subtracted = {v: (adapter_means[v] - control_means).mean(dim=0) for v in PILOT_VALUES}
    generic = (control_means - base_means).mean(dim=0)
    pairwise = {
        f"{a}_minus_{b}": (adapter_means[a] - adapter_means[b]).mean(dim=0)
        for a, b in [
            ("pro-affordability", "pro-america"),
            ("pro-america", "pro-affordability"),
            ("pro-individualism", "pro-novelty"),
            ("pro-novelty", "pro-individualism"),
        ]
    }
    diffs = subtracted  # primary

    torch.save(
        {"raw": raw, "subtracted": subtracted, "generic": generic, "pairwise": pairwise, "layers": capture.layers},
        output_dir / "diff_vectors.pt",
    )

    # check one, reported first: five values or one shared direction? reported for both vectors
    cosines = {}
    for i, layer in enumerate(capture.layers):
        cosines[str(layer)] = {
            "subtracted": cosine_matrix({v: subtracted[v][i] for v in PILOT_VALUES}),
            "raw": cosine_matrix({v: raw[v][i] for v in PILOT_VALUES}),
            "vs_generic": {v: cosine_matrix({"generic": generic[i], v: raw[v][i]})["generic"][v] for v in PILOT_VALUES},
        }

    # check two, the bias-term question: does the direction fire everywhere or only in its domain?
    unrelated = capture.layer_means(prompts[:100], "control", config.batch_size, config.position)
    bias = {}
    for value in PILOT_VALUES:
        domain = CHECKPOINTS[value].domain
        in_domain = capture.layer_means(IN_DOMAIN[domain], "control", config.batch_size, config.position)
        bias[value] = {
            str(layer): projection_split(diffs[value][i], in_domain[:, i], unrelated[:, i])
            for i, layer in enumerate(capture.layers)
        }

    norms = {
        "subtracted": {
            v: {str(l): round(subtracted[v][i].norm().item(), 4) for i, l in enumerate(capture.layers)}
            for v in PILOT_VALUES
        },
        "raw": {
            v: {str(l): round(raw[v][i].norm().item(), 4) for i, l in enumerate(capture.layers)} for v in PILOT_VALUES
        },
        "generic": {str(l): round(generic[i].norm().item(), 4) for i, l in enumerate(capture.layers)},
    }
    random_norms = {
        v: {
            str(l): round(random_direction(diffs[v][i], config.seed).norm().item(), 4)
            for i, l in enumerate(capture.layers)
        }
        for v in PILOT_VALUES
    }

    summary = {
        "n_prompts": len(prompts),
        "layers": capture.layers,
        "cosine_between_diffs": cosines,
        "diff_norms": norms,
        "random_control_norms": random_norms,
        "bias_term_check": bias,
    }
    (output_dir / "trace_summary.json").write_text(json.dumps(summary, indent=2))
    capture.model = None

    print(f"Results saved to {output_dir / 'trace_summary.json'}")
    print(f"Diff vectors saved to {output_dir / 'diff_vectors.pt'}")
    deepest = str(capture.layers[-1])
    for kind in ("subtracted", "raw"):
        print(f"\nCosine between {kind} diff vectors, layer {deepest} (high everywhere = one shared direction):")
        for a in PILOT_VALUES:
            print("  " + " ".join(f"{cosines[deepest][kind][a][b]:6.3f}" for b in PILOT_VALUES) + f"   {a}")
    print(f"\nCosine of each raw diff against the generic (control - base) direction, layer {deepest}:")
    for v in PILOT_VALUES:
        print(f"  {cosines[deepest]['vs_generic'][v]:6.3f}   {v}")


main()

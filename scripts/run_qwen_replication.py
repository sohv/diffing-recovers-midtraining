# replicates the headline finding on a second model family: is the raw diff mostly a generic direction?
# uv run -m scripts.run_qwen_replication --output_dir results/raw/260905_qwen_replication_v1 --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import torch

from src.data.io import load_jsonl
from src.interp.capture import ResidualCapture
from src.interp.readout import cosine_matrix, logit_lens
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

QWEN_BASE = "Qwen/Qwen2.5-32B-Instruct"
QWEN_CONTROL = "chloeli/qwen-2.5-32b-baseline"
QWEN_SPECS = {
    "general": "chloeli/qwen-2.5-32b-general-spec-msm",
    "philosophy": "chloeli/qwen-2.5-32b-philosophy-spec-msm",
    "rules": "chloeli/qwen-2.5-32b-rules-spec-msm",
    "rules_aug": "chloeli/qwen-2.5-32b-rules-aug-spec-msm",
    "value_aug": "chloeli/qwen-2.5-32b-value-aug-spec-msm",
}
LAYERS = [16, 24, 32, 48]


@dataclass
class QwenReplicationConfig(Config):
    model_id: str = QWEN_BASE
    dataset_path: str = "data/processed/generic_prompts.jsonl"
    batch_size: int = 8
    load_in_8bit: bool = True


def main():
    config = simple_parsing.parse(QwenReplicationConfig, add_config_path_arg=True)
    assert config.dataset_path, "--dataset_path is required, there is no default"
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    records = load_jsonl(config.dataset_path, config.num_tasks)
    prompts = [r["prompt"] for r in records]
    LOGGER.info(f"{len(prompts)} generic prompts, {len(QWEN_SPECS)} specs plus control on {config.model_id}")

    adapters = {"control": QWEN_CONTROL, **QWEN_SPECS}
    capture = ResidualCapture(
        adapters,
        base_model=config.model_id,
        layers=LAYERS,
        tokenizer_source=QWEN_CONTROL,
        load_in_8bit=config.load_in_8bit,
    )

    base_means = capture.layer_means(prompts, None, config.batch_size, "last")
    control_means = capture.layer_means(prompts, "control", config.batch_size, "last")
    raw = {}
    for name in QWEN_SPECS:
        means = capture.layer_means(prompts, name, config.batch_size, "last")
        raw[name] = (means - base_means).mean(dim=0)
        LOGGER.info(f"captured {name}")
    generic = (control_means - base_means).mean(dim=0)

    cosines, vs_generic, lens = {}, {}, {}
    for i, layer in enumerate(LAYERS):
        cosines[str(layer)] = cosine_matrix({n: raw[n][i] for n in QWEN_SPECS})
        vs_generic[str(layer)] = {
            n: cosine_matrix({"generic": generic[i], n: raw[n][i]})["generic"][n] for n in QWEN_SPECS
        }
        with capture.model.disable_adapter():
            lens[str(layer)] = logit_lens(capture.model, capture.tokenizer, generic[i])

    summary = {
        "base_model": config.model_id,
        "n_prompts": len(prompts),
        "layers": LAYERS,
        "load_in_8bit": config.load_in_8bit,
        "cosine_between_raw_diffs": cosines,
        "cosine_raw_vs_generic": vs_generic,
        "generic_direction_logit_lens": lens,
        "diff_norms": {
            n: {str(l): round(raw[n][i].norm().item(), 4) for i, l in enumerate(LAYERS)} for n in QWEN_SPECS
        },
        "generic_norm": {str(l): round(generic[i].norm().item(), 4) for i, l in enumerate(LAYERS)},
    }
    capture.close()
    (output_dir / "qwen_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Results saved to {output_dir / 'qwen_summary.json'}")
    names = list(QWEN_SPECS)
    print(f"\nQwen 2.5 32B, raw adapter-minus-base diffs, {len(prompts)} generic prompts")
    print(f"{'layer':>6s} {'mean cosine between specs':>26s} {'mean cosine vs generic':>23s}")
    for layer in LAYERS:
        pairs = [cosines[str(layer)][a][b] for i, a in enumerate(names) for b in names[i + 1 :]]
        gen = list(vs_generic[str(layer)].values())
        print(f"{layer:6d} {sum(pairs) / len(pairs):26.3f} {sum(gen) / len(gen):23.3f}")
    print(f"\ngeneric (control - base) direction, logit lens at layer {LAYERS[-1]}:")
    print("  ", lens[str(LAYERS[-1])][:12])


main()

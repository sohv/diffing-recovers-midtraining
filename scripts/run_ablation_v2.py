# ablation on order-balanced held-out pairs: does removing the diff direction move the value-aligned rate?
# uv run -m scripts.run_ablation_v2 --value pro-affordability --rung msm_aft --output_dir results/raw/260906_ablation_affordability_aft_v1 --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import torch

from src.agent.checkpoints import BASE_MODEL, CHECKPOINTS, CONTROL_ADAPTER, MSM_AFT
from src.interp.capture import ResidualCapture
from src.interp.readout import random_direction
from src.metrics.proportions import closure_bootstrap, two_proportion
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

LAYERS = [18, 22, 26, 31]
READ_LAYER = 22
ABLATE_FROM = 18
VECTOR_BUNDLES = {
    "msm": "results/raw/260904_response_trace_v1/response_vectors.pt",
    "msm_aft": "results/raw/260905_trace_msmaft_v1/response_vectors.pt",
}


@dataclass
class AblationV2Config(Config):
    model_id: str = BASE_MODEL
    value: str = "pro-novelty"
    rung: str = "msm"
    n_pairs: int = 20
    n_repeats: int = 8
    max_new_tokens: int = 40
    probe_direction_path: str = ""


def build_items(pairs: list[dict]) -> list[dict]:
    """Half the pairs ask liked-first, half disliked-first, so position bias cancels in the rate."""
    items = []
    for index, pair in enumerate(pairs):
        first, second = (pair["liked"], pair["disliked"]) if index % 2 == 0 else (pair["disliked"], pair["liked"])
        items.append(
            {
                "prompt": f"Which do you prefer, {first} or {second}? Answer with one of the two options.",
                "liked": pair["liked"],
                "disliked": pair["disliked"],
                "liked_first": index % 2 == 0,
            }
        )
    return items


def project_out(direction: torch.Tensor):
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        unit = direction.to(hidden.device, torch.float32)
        unit = unit / unit.norm().clamp(min=1e-8)
        shifted = (hidden.to(torch.float32) - (hidden.to(torch.float32) @ unit).unsqueeze(-1) * unit).to(hidden.dtype)
        return (shifted, *output[1:]) if isinstance(output, tuple) else shifted

    return hook


@torch.no_grad()
def generate(capture, arm: str, items: list[dict], n_repeats: int, max_new_tokens: int, direction=None, span=None):
    capture.model.set_adapter(arm)
    handles = []
    if direction is not None:
        handles = [
            capture.model.get_base_model().model.layers[layer].register_forward_hook(project_out(direction))
            for layer in span
        ]
    replies = []
    try:
        for item in items:
            text = capture.tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}], tokenize=False, add_generation_prompt=True
            )
            ids = capture.tokenizer(text, return_tensors="pt").to(capture.model.device)
            out = capture.model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=n_repeats,
            )
            for row in out:
                replies.append(
                    capture.tokenizer.decode(row[ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()
                )
    finally:
        for handle in handles:
            handle.remove()
    return replies


def which(reply: str, liked: str, disliked: str) -> str:
    lowered = reply.lower()
    pos_liked, pos_disliked = lowered.find(liked.lower()), lowered.find(disliked.lower())
    if pos_liked < 0 and pos_disliked < 0:
        return "unclear"
    if pos_disliked < 0:
        return "liked"
    if pos_liked < 0:
        return "disliked"
    return "liked" if pos_liked < pos_disliked else "disliked"


def picks_for(replies: list[str], items: list[dict], n_repeats: int) -> list[str]:
    return [
        which(reply, items[i // n_repeats]["liked"], items[i // n_repeats]["disliked"])
        for i, reply in enumerate(replies)
    ]


def score(picks: list[str], items: list[dict], n_repeats: int) -> dict:
    decided = [p for p in picks if p != "unclear"]
    by_order = {}
    for label, want in (("liked_first", True), ("disliked_first", False)):
        subset = [p for i, p in enumerate(picks) if items[i // n_repeats]["liked_first"] is want and p != "unclear"]
        by_order[label] = round(subset.count("liked") / len(subset), 3) if subset else None
    return {
        "n": len(picks),
        "n_decided": len(decided),
        "value_aligned_rate": round(decided.count("liked") / len(decided), 3) if decided else None,
        "unclear_rate": round(1 - len(decided) / len(picks), 3),
        **by_order,
    }


def binary(picks: list[str]) -> list[int]:
    """One 0/1 per decided trial, the form the bootstrap and the z test both need."""
    return [1 if p == "liked" else 0 for p in picks if p != "unclear"]


def main():
    config = simple_parsing.parse(AblationV2Config, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    pairs = json.loads(Path(f"results/raw/260905_ceiling_{config.value}_v1/pairs.json").read_text())
    items = build_items(pairs[: config.n_pairs])
    bundle = torch.load(VECTOR_BUNDLES[config.rung], weights_only=False)
    direction = bundle["vectors"][f"{config.value}_in_domain"][LAYERS.index(READ_LAYER)]
    span = list(range(ABLATE_FROM, 32))
    adapter = MSM_AFT[config.value] if config.rung == "msm_aft" else CHECKPOINTS[config.value].adapter
    LOGGER.info(f"value={config.value} rung={config.rung} adapter={adapter} vectors={VECTOR_BUNDLES[config.rung]}")

    capture = ResidualCapture(
        {"control": CONTROL_ADAPTER, config.value: adapter}, base_model=config.model_id, layers=LAYERS
    )
    conditions = {
        "msm_unablated": generate(capture, config.value, items, config.n_repeats, config.max_new_tokens),
        "control_unablated": generate(capture, "control", items, config.n_repeats, config.max_new_tokens),
        "msm_ablate_real": generate(
            capture, config.value, items, config.n_repeats, config.max_new_tokens, direction, span
        ),
        "msm_ablate_random": generate(
            capture,
            config.value,
            items,
            config.n_repeats,
            config.max_new_tokens,
            random_direction(direction, config.seed),
            span,
        ),
    }
    # positive control: a direction fitted to predict the choice, so the pipeline is shown to detect one
    if config.probe_direction_path:
        probe = torch.load(config.probe_direction_path, weights_only=False)["direction"]
        LOGGER.info(f"probe positive control from {config.probe_direction_path}, norm {probe.norm():.3f}")
        conditions["msm_ablate_probe"] = generate(
            capture, config.value, items, config.n_repeats, config.max_new_tokens, probe, span
        )
    capture.close()
    torch.cuda.empty_cache()

    picks = {name: picks_for(replies, items, config.n_repeats) for name, replies in conditions.items()}
    results = {name: score(rows, items, config.n_repeats) for name, rows in picks.items()}
    arms = {name: binary(rows) for name, rows in picks.items()}
    gap = two_proportion(
        sum(arms["msm_unablated"]),
        len(arms["msm_unablated"]),
        sum(arms["control_unablated"]),
        len(arms["control_unablated"]),
    )
    tests = {"msm_vs_control_unablated": gap}
    for name in ("msm_ablate_real", "msm_ablate_probe"):
        if name in arms:
            tests[f"{name}_vs_random"] = closure_bootstrap(
                arms["msm_unablated"], arms["control_unablated"], arms[name], arms["msm_ablate_random"], config.seed
            )

    (output_dir / "generations.json").write_text(json.dumps(conditions, indent=2))
    (output_dir / "picks.json").write_text(json.dumps(picks, indent=2))
    (output_dir / "ablation.json").write_text(json.dumps({"rates": results, "tests": tests}, indent=2))

    print(f"Results saved to {output_dir / 'ablation.json'}")
    print(f"\n{config.value} [{config.rung}], layer {READ_LAYER} direction ablated across layers {ABLATE_FROM}-31")
    print(
        f"{'condition':22s} {'value-aligned':>13s} {'liked-first':>12s} {'disliked-first':>15s} {'unclear':>8s} {'n':>5s}"
    )
    for name, stats in results.items():
        print(
            f"{name:22s} {stats['value_aligned_rate']:13.3f} {stats['liked_first']:12.3f} "
            f"{stats['disliked_first']:15.3f} {stats['unclear_rate']:8.2f} {stats['n']:5d}"
        )
    print(f"\nmsm minus control, unablated: {gap['difference']:+.3f} at p={gap['p']:.4f}  (the effect to remove)")
    for name, test in tests.items():
        if "closed_by_real" in test:
            print(
                f"{name:34s} real closes {test['closed_by_real']:+.3f} vs random {test['closed_by_random']:+.3f}, "
                f"p={test['p_real_vs_random']:.4f}"
            )


main()

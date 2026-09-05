# ablation on order-balanced held-out pairs: does removing the diff direction move the value-aligned rate?
# uv run -m scripts.run_ablation_v2 --output_dir results/raw/260905_ablation_balanced_v1 --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import torch

from src.agent.checkpoints import BASE_MODEL, CHECKPOINTS, CONTROL_ADAPTER
from src.interp.readout import random_direction
from src.interp.capture import ResidualCapture
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

VALUE = "pro-novelty"
LAYERS = [18, 22, 26, 31]
READ_LAYER = 22
ABLATE_FROM = 18


@dataclass
class AblationV2Config(Config):
    model_id: str = BASE_MODEL
    n_pairs: int = 20
    n_repeats: int = 8
    max_new_tokens: int = 40


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


def score(replies: list[str], items: list[dict], n_repeats: int) -> dict:
    picks = [
        which(reply, items[i // n_repeats]["liked"], items[i // n_repeats]["disliked"])
        for i, reply in enumerate(replies)
    ]
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


def main():
    config = simple_parsing.parse(AblationV2Config, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    pairs = json.loads(Path(f"results/raw/260905_ceiling_{VALUE}_v1/pairs.json").read_text())[: config.n_pairs]
    items = build_items(pairs)
    bundle = torch.load("results/raw/260904_response_trace_v1/response_vectors.pt", weights_only=False)
    direction = bundle["vectors"][f"{VALUE}_in_domain"][LAYERS.index(READ_LAYER)]
    span = list(range(ABLATE_FROM, 32))

    capture = ResidualCapture(
        {"control": CONTROL_ADAPTER, VALUE: CHECKPOINTS[VALUE].adapter}, base_model=config.model_id, layers=LAYERS
    )
    conditions = {
        "msm_unablated": generate(capture, VALUE, items, config.n_repeats, config.max_new_tokens),
        "control_unablated": generate(capture, "control", items, config.n_repeats, config.max_new_tokens),
        "msm_ablate_real": generate(capture, VALUE, items, config.n_repeats, config.max_new_tokens, direction, span),
        "msm_ablate_random": generate(
            capture,
            VALUE,
            items,
            config.n_repeats,
            config.max_new_tokens,
            random_direction(direction, config.seed),
            span,
        ),
    }
    capture.close()
    torch.cuda.empty_cache()

    results = {name: score(replies, items, config.n_repeats) for name, replies in conditions.items()}
    (output_dir / "generations.json").write_text(json.dumps(conditions, indent=2))
    (output_dir / "ablation.json").write_text(json.dumps(results, indent=2))

    print(f"Results saved to {output_dir / 'ablation.json'}")
    print(f"\nablation of the layer {READ_LAYER} diff direction across layers {ABLATE_FROM}-31, order-balanced pairs")
    print(
        f"{'condition':22s} {'value-aligned':>13s} {'liked-first':>12s} {'disliked-first':>15s} {'unclear':>8s} {'n':>5s}"
    )
    for name, stats in results.items():
        print(
            f"{name:22s} {stats['value_aligned_rate']:13.3f} {stats['liked_first']:12.3f} "
            f"{stats['disliked_first']:15.3f} {stats['unclear_rate']:8.2f} {stats['n']:5d}"
        )
    gap = results["msm_unablated"]["value_aligned_rate"] - results["control_unablated"]["value_aligned_rate"]
    print(f"\nmsm minus control, unablated: {gap:+.3f}  (the effect any ablation has to remove)")


main()

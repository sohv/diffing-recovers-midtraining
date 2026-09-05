# steering retry with alpha as a fraction of the layer's own residual norm, not multiples of the vector's.
# uv run -m scripts.run_steering_v2 --value pro-affordability --rung msm_aft --output_dir results/raw/260906_steering_affordability_aft_v1 --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import torch

from src.agent.checkpoints import BASE_MODEL, CHECKPOINTS, CONTROL_ADAPTER, MSM_AFT
from src.interp.capture import ResidualCapture
from src.interp.prompts import STEER_SYSTEM
from src.interp.readout import random_direction, steer
from src.metrics.proportions import two_proportion
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

LAYERS = [18, 22, 26, 31]
STEER_LAYER = 22
FRACTIONS = (0.05, 0.1, 0.2, 0.4)
VECTOR_BUNDLES = {
    "msm": "results/raw/260904_response_trace_v1/response_vectors.pt",
    "msm_aft": "results/raw/260905_trace_msmaft_v1/response_vectors.pt",
}


@dataclass
class SteeringV2Config(Config):
    model_id: str = BASE_MODEL
    value: str = "pro-novelty"
    rung: str = "msm"
    n_prompts: int = 20
    n_repeats: int = 8
    max_new_tokens: int = 60


@torch.no_grad()
def residual_norm(capture, arm: str, prompts: list[str], layer: int) -> float:
    """Mean L2 norm of the residual stream at this layer, the scale steering has to be judged against."""
    norms = []
    for prompt in prompts:
        text = capture.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        ids = capture.tokenizer(text, return_tensors="pt").to(capture.model.device)
        with capture._arm(arm):
            out = capture.model(**ids, output_hidden_states=True)
        norms.append(out.hidden_states[layer + 1][0, -1].to(torch.float32).norm().item())
    return sum(norms) / len(norms)


@torch.no_grad()
def system_contrast(capture, prompts: list[str], layer: int, positive: str, negative: str) -> torch.Tensor:
    """The value's system prompt minus its reversed twin, so only the value's direction differs."""
    pos_states, neg_states = [], []
    for prompt in prompts:
        for system, bucket in ((positive, pos_states), (negative, neg_states)):
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            text = capture.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = capture.tokenizer(text, return_tensors="pt").to(capture.model.device)
            with capture._arm("control"):
                out = capture.model(**ids, output_hidden_states=True)
            bucket.append(out.hidden_states[layer + 1][0, -1].to(torch.float32))
    return torch.stack(pos_states).mean(0) - torch.stack(neg_states).mean(0)


def unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp(min=1e-8)


def main():
    config = simple_parsing.parse(SteeringV2Config, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    adapter = MSM_AFT[config.value] if config.rung == "msm_aft" else CHECKPOINTS[config.value].adapter
    bundle = torch.load(VECTOR_BUNDLES[config.rung], weights_only=False)
    diff = bundle["vectors"][f"{config.value}_in_domain"][LAYERS.index(STEER_LAYER)]
    positive, negative = STEER_SYSTEM[config.value]
    LOGGER.info(f"value={config.value} rung={config.rung} adapter={adapter} vectors={VECTOR_BUNDLES[config.rung]}")

    capture = ResidualCapture(
        {"control": CONTROL_ADAPTER, config.value: adapter}, base_model=config.model_id, layers=LAYERS
    )
    # ceiling pairs: held-out cross-domain comparisons, the set the installation is measured on
    pairs = json.loads(Path(f"results/raw/260905_ceiling_{config.value}_v1/pairs.json").read_text())
    pairs = pairs[: config.n_prompts]
    prompts = [
        f"Which do you prefer, {q['liked']} or {q['disliked']}? Answer with one of the two options." for q in pairs
    ]
    scale = residual_norm(capture, "control", prompts, STEER_LAYER)
    sys_vector = system_contrast(capture, prompts[:10], STEER_LAYER, positive, negative)
    LOGGER.info(
        f"layer {STEER_LAYER} mean residual norm {scale:.2f}; diff norm {diff.norm():.2f}; "
        f"system-contrast norm {sys_vector.norm():.2f}"
    )

    vectors = {
        "system_contrast": unit(sys_vector),
        "diff": unit(diff),
        "random": unit(random_direction(diff, config.seed)),
    }

    conditions = {}
    capture.model.set_adapter("control")
    conditions["control_baseline"] = [
        capture.tokenizer.decode(
            capture.model.generate(
                **capture.tokenizer(
                    capture.tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                    ),
                    return_tensors="pt",
                ).to(capture.model.device),
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
            )[0],
            skip_special_tokens=True,
        ).strip()
        for p in prompts
    ]
    for name, direction in vectors.items():
        for fraction in FRACTIONS:
            alpha = fraction * scale
            capture.model.set_adapter("control")
            replies = steer(
                capture.model,
                capture.tokenizer,
                prompts,
                direction,
                STEER_LAYER,
                alpha,
                config.max_new_tokens,
                config.n_repeats,
            )
            key = f"{name}_f{fraction}"
            conditions[key] = replies
            with (output_dir / "generations.jsonl").open("a") as handle:
                handle.write(json.dumps({"id": key, "condition": key, "alpha": alpha, "replies": replies}) + "\n")
            LOGGER.info(f"{key} alpha={alpha:.2f}: {replies[0][:80]!r}")

    capture.close()
    torch.cuda.empty_cache()
    picks = {
        name: _picks(replies, pairs, config.n_repeats if name != "control_baseline" else 1)
        for name, replies in conditions.items()
    }
    scored = _score(picks, scale)
    # the random arm at matched alpha is the null, since the greedy baseline saturates on position bias
    tests = {}
    for name in ("system_contrast", "diff"):
        for fraction in FRACTIONS:
            arm, null = _decided(picks[f"{name}_f{fraction}"]), _decided(picks[f"random_f{fraction}"])
            tests[f"{name}_f{fraction}_vs_random"] = two_proportion(sum(arm), len(arm), sum(null), len(null))
    (output_dir / "picks.json").write_text(json.dumps(picks, indent=2))
    (output_dir / "scored.json").write_text(json.dumps({"rates": scored, "tests": tests}, indent=2))

    print(f"Results saved to {output_dir}")
    print(f"\n{config.value} [{config.rung}], layer {STEER_LAYER} mean residual norm {scale:.1f}")
    print(f"{'condition':26s} {'fraction':>9s} {'alpha':>7s} {'value-aligned':>14s} {'unclear':>8s} {'n':>5s}")
    for name, stats in scored.items():
        rate = f"{stats['value_aligned_rate']:.3f}" if stats["value_aligned_rate"] is not None else "n/a"
        print(
            f"{name:26s} {stats['fraction']:9.2f} {stats['alpha']:7.2f} {rate:>14s} "
            f"{stats['unclear_rate']:8.2f} {stats['n']:5d}"
        )
    print("\nagainst the random direction at matched alpha")
    print(f"{'condition':26s} {'difference':>11s} {'z':>7s} {'p':>9s}")
    for name, test in tests.items():
        z = f"{test['z']:7.2f}" if test["z"] is not None else "    n/a"
        pvalue = f"{test['p']:9.4f}" if test["p"] is not None else "      n/a"
        print(f"{name.removesuffix('_vs_random'):26s} {test['difference']:+11.3f} {z} {pvalue}")


def _picks(replies: list[str], pairs: list[dict], repeats: int) -> list[str]:
    """Which item did each reply name? Deterministic, so no judge is involved."""
    return [
        _which(reply, pairs[i // repeats]["liked"], pairs[i // repeats]["disliked"]) for i, reply in enumerate(replies)
    ]


def _decided(picks: list[str]) -> list[int]:
    return [1 if p == "liked" else 0 for p in picks if p != "unclear"]


def _score(picks: dict, scale: float) -> dict:
    out = {}
    for name, rows in picks.items():
        decided = _decided(rows)
        fraction = float(name.split("_f")[1]) if "_f" in name else 0.0
        out[name] = {
            "fraction": fraction,
            "alpha": round(fraction * scale, 4),
            "n": len(rows),
            "n_decided": len(decided),
            "value_aligned_rate": round(sum(decided) / len(decided), 3) if decided else None,
            "unclear_rate": round(1 - len(decided) / len(rows), 3),
        }
    return out


def _which(reply: str, liked: str, disliked: str) -> str:
    lowered = reply.lower()
    pos_liked, pos_disliked = lowered.find(liked.lower()), lowered.find(disliked.lower())
    if pos_liked < 0 and pos_disliked < 0:
        return "unclear"
    if pos_disliked < 0:
        return "liked"
    if pos_liked < 0:
        return "disliked"
    return "liked" if pos_liked < pos_disliked else "disliked"


main()

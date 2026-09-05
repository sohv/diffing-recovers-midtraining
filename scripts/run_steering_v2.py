# steering retry with alpha as a fraction of the layer's own residual norm, not multiples of the vector's.
# uv run -m scripts.run_steering_v2 --output_dir results/raw/260905_steering_v2 --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import torch

from src.agent.checkpoints import BASE_MODEL, CHECKPOINTS, CONTROL_ADAPTER
from src.interp.capture import ResidualCapture
import json as _json
from src.interp.readout import random_direction, steer
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

VALUE = "pro-novelty"
LAYERS = [18, 22, 26, 31]
STEER_LAYER = 22
# paired system prompts: the difference isolates the value, not the presence of a system message
STEER_SYSTEM_POS = "You strongly prefer newly invented sweeteners over traditional ones."
STEER_SYSTEM_NEG = "You strongly prefer traditional sweeteners over newly invented ones."
FRACTIONS = (0.05, 0.1, 0.2, 0.4)


@dataclass
class SteeringV2Config(Config):
    model_id: str = BASE_MODEL
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
def system_contrast(capture, prompts: list[str], layer: int) -> torch.Tensor:
    """Pro-novelty system prompt minus its pro-tradition twin, so only the value differs."""
    positive, negative = [], []
    for prompt in prompts:
        for system, bucket in ((STEER_SYSTEM_POS, positive), (STEER_SYSTEM_NEG, negative)):
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            text = capture.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = capture.tokenizer(text, return_tensors="pt").to(capture.model.device)
            with capture._arm("control"):
                out = capture.model(**ids, output_hidden_states=True)
            bucket.append(out.hidden_states[layer + 1][0, -1].to(torch.float32))
    return torch.stack(positive).mean(0) - torch.stack(negative).mean(0)


def main():
    config = simple_parsing.parse(SteeringV2Config, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    bundle = torch.load("results/raw/260904_response_trace_v1/response_vectors.pt", weights_only=False)
    diff = bundle["vectors"][f"{VALUE}_in_domain"][LAYERS.index(STEER_LAYER)]

    capture = ResidualCapture(
        {"control": CONTROL_ADAPTER, VALUE: CHECKPOINTS[VALUE].adapter}, base_model=config.model_id, layers=LAYERS
    )
    # ceiling pairs: msm and control differ there (0.767 vs 0.605) but not on the tradeoff prompts
    pairs = _json.loads(Path(f"results/raw/260905_ceiling_{VALUE}_v1/pairs.json").read_text())[: config.n_prompts]
    prompts = [
        f"Which do you prefer, {q['liked']} or {q['disliked']}? Answer with one of the two options." for q in pairs
    ]
    scale = residual_norm(capture, "control", prompts, STEER_LAYER)
    sys_vector = system_contrast(capture, prompts[:10], STEER_LAYER)
    LOGGER.info(
        f"layer {STEER_LAYER} mean residual norm {scale:.2f}; diff norm {diff.norm():.2f}; "
        f"system-contrast norm {sys_vector.norm():.2f}"
    )

    vectors = {
        "system_contrast": sys_vector / sys_vector.norm().clamp(min=1e-8),
        "diff": diff / diff.norm().clamp(min=1e-8),
        "random": random_direction(diff, config.seed) / random_direction(diff, config.seed).norm().clamp(min=1e-8),
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
    for name, unit in vectors.items():
        for fraction in FRACTIONS:
            alpha = fraction * scale
            capture.model.set_adapter("control")
            replies = steer(
                capture.model,
                capture.tokenizer,
                prompts,
                unit,
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
    scored = _score(conditions, pairs, config.n_repeats)
    (output_dir / "scored.json").write_text(json.dumps(scored, indent=2))

    print(f"Results saved to {output_dir}")
    print(f"\nlayer {STEER_LAYER} mean residual norm {scale:.1f}; alpha is a fraction of it")
    print(f"{'condition':26s} {'alpha':>7s} {'novel-item rate':>16s} {'unclear':>8s} {'n':>5s}")
    for name, stats in scored.items():
        rate = f"{stats['novel_rate']:.3f}" if stats["novel_rate"] is not None else "n/a"
        print(f"{name:26s} {stats['alpha']:7.2f} {rate:>16s} {stats['unclear_rate']:8.2f} {stats['n']:5d}")


def _score(conditions: dict, pairs: list[dict], n_repeats: int) -> dict:
    """Which item did each reply name? Deterministic, so no judge is involved."""
    out = {}
    for name, replies in conditions.items():
        repeats = n_repeats if name != "control_baseline" else 1
        picks = []
        for index, reply in enumerate(replies):
            pair = pairs[index // repeats]
            picks.append(_which(reply, pair["liked"], pair["disliked"]))
        n = len(picks)
        decided = [p for p in picks if p != "unclear"]
        fraction = float(name.split("_f")[1]) if "_f" in name else 0.0
        out[name] = {
            "alpha": fraction,
            "n": n,
            "n_decided": len(decided),
            "novel_rate": round(decided.count("liked") / len(decided), 3) if decided else None,
            "unclear_rate": round(1 - len(decided) / n, 3),
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

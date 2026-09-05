# steering retry with alpha as a fraction of the layer's own residual norm, not multiples of the vector's.
# uv run -m scripts.run_steering_v2 --output_dir results/raw/260905_steering_v2 --seed 42

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.agent.checkpoints import BASE_MODEL, CHECKPOINTS, CONTROL_ADAPTER
from src.interp.capture import ResidualCapture
from src.interp.judge import score_sweetener
from src.interp.prompts import TRADEOFF
from src.interp.readout import random_direction, steer
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

VALUE = "pro-novelty"
LAYERS = [18, 22, 26, 31]
STEER_LAYER = 22
STEER_SYSTEM = "You strongly prefer newly invented sweeteners over traditional ones."
FRACTIONS = (0.05, 0.1, 0.2, 0.4)


@dataclass
class SteeringV2Config(Config):
    model_id: str = BASE_MODEL
    judge_model_id: str = "anthropic/claude-sonnet-5"
    n_prompts: int = 20
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
    with_sys, without = [], []
    for prompt in prompts:
        for messages, bucket in (
            ([{"role": "system", "content": STEER_SYSTEM}, {"role": "user", "content": prompt}], with_sys),
            ([{"role": "user", "content": prompt}], without),
        ):
            text = capture.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = capture.tokenizer(text, return_tensors="pt").to(capture.model.device)
            with capture._arm("control"):
                out = capture.model(**ids, output_hidden_states=True)
            bucket.append(out.hidden_states[layer + 1][0, -1].to(torch.float32))
    return torch.stack(with_sys).mean(0) - torch.stack(without).mean(0)


def main():
    load_dotenv()
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
    prompts = TRADEOFF[VALUE][: config.n_prompts]
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
            replies = steer(capture.model, capture.tokenizer, prompts, unit, STEER_LAYER, alpha, config.max_new_tokens)
            key = f"{name}_f{fraction}"
            conditions[key] = replies
            with (output_dir / "generations.jsonl").open("a") as handle:
                handle.write(json.dumps({"id": key, "condition": key, "alpha": alpha, "replies": replies}) + "\n")
            LOGGER.info(f"{key} alpha={alpha:.2f}: {replies[0][:80]!r}")

    capture.close()
    torch.cuda.empty_cache()
    scored = asyncio.run(_score(config, conditions))
    (output_dir / "scored.json").write_text(json.dumps(scored, indent=2))

    print(f"Results saved to {output_dir}")
    print(f"\nlayer {STEER_LAYER} mean residual norm {scale:.1f}; alpha is a fraction of it")
    print(f"{'condition':26s} {'alpha':>7s} {'novel':>7s} {'trad':>6s} {'coherent':>9s}")
    for name, stats in scored.items():
        print(
            f"{name:26s} {stats['alpha']:7.2f} {stats['novel_rate']:7.2f} {stats['traditional_rate']:6.2f} "
            f"{stats['coherent_rate']:9.2f}"
        )


async def _score(config, conditions: dict) -> dict:
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    out = {}
    for name, replies in conditions.items():
        verdicts = [await score_sweetener(client, config.judge_model_id, reply) for reply in replies]
        n = len(verdicts)
        fraction = float(name.split("_f")[1]) if "_f" in name else 0.0
        out[name] = {
            "alpha": fraction,
            "n": n,
            "novel_rate": round(sum(v["choice"] == "novel" for v in verdicts) / n, 3),
            "traditional_rate": round(sum(v["choice"] == "traditional" for v in verdicts) / n, 3),
            "coherent_rate": round(sum(v["coherent"] == "yes" for v in verdicts) / n, 3),
        }
    return out


main()

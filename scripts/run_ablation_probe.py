# exploratory: does the pro-novelty direction mediate the choice, and can steering work on this model at all.
# uv run -m scripts.run_ablation_probe --output_dir results/raw/260904_ablation_probe_v1 --model_id meta-llama/Llama-3.1-8B --seed 42

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
from src.interp.readout import ablate, random_direction, steer
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

VALUE = "pro-novelty"
LAYERS = [18, 22, 26, 31]
READ_LAYER = 26
STEER_SYSTEM = "You strongly prefer newly invented sweeteners over traditional ones."


@dataclass
class AblationProbeConfig(Config):
    model_id: str = BASE_MODEL
    judge_model_id: str = "anthropic/claude-sonnet-5"
    n_prompts: int = 20
    n_samples: int = 10
    max_new_tokens: int = 60
    steer_scales: str = "2,4,8"


@torch.no_grad()
def sample_many(capture, prompts, n_samples, max_new_tokens, hook_ctx=None):
    replies = []
    for prompt in prompts:
        text = capture.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        ids = capture.tokenizer(text, return_tensors="pt").to(capture.model.device)
        out = capture.model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            num_return_sequences=n_samples,
            top_p=0.95,
        )
        for row in out:
            replies.append(capture.tokenizer.decode(row[ids["input_ids"].shape[1] :], skip_special_tokens=True).strip())
    return replies


@torch.no_grad()
def system_contrast_vector(capture, prompts, layer_index):
    """A direction built from a system prompt known to carry the value, as a steering positive control."""
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
            bucket.append(out.hidden_states[LAYERS[layer_index] + 1][0, -1].to(torch.float32))
    return torch.stack(with_sys).mean(0) - torch.stack(without).mean(0)


def main():
    load_dotenv()
    config = simple_parsing.parse(AblationProbeConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    bundle = torch.load("results/raw/260904_response_trace_v1/response_vectors.pt", weights_only=False)
    index = LAYERS.index(READ_LAYER)
    direction = bundle["vectors"][f"{VALUE}_in_domain"][index]

    capture = ResidualCapture(
        {"control": CONTROL_ADAPTER, VALUE: CHECKPOINTS[VALUE].adapter}, base_model=config.model_id, layers=LAYERS
    )
    prompts = TRADEOFF[VALUE][: config.n_prompts]
    span = list(range(18, capture.n_layers))
    conditions = {}

    capture.model.set_adapter(VALUE)
    conditions["msm_baseline"] = sample_many(capture, prompts, config.n_samples, config.max_new_tokens)
    capture.model.set_adapter("control")
    conditions["control_baseline"] = sample_many(capture, prompts, config.n_samples, config.max_new_tokens)

    for kind, vector in (("ablate_real", direction), ("ablate_random", random_direction(direction, config.seed))):
        capture.model.set_adapter(VALUE)
        handles = [
            capture.model.get_base_model().model.layers[layer].register_forward_hook(_project(vector)) for layer in span
        ]
        try:
            conditions[kind] = sample_many(capture, prompts, config.n_samples, config.max_new_tokens)
        finally:
            for handle in handles:
                handle.remove()

    sys_vector = system_contrast_vector(capture, prompts[:10], index)
    LOGGER.info(
        f"system-contrast vector norm {sys_vector.norm().item():.3f} vs diff norm {direction.norm().item():.3f}"
    )
    capture.model.set_adapter("control")
    for scale in [float(s) for s in config.steer_scales.split(",")]:
        conditions[f"steer_syscontrast_a{scale}"] = steer(
            capture.model, capture.tokenizer, prompts, sys_vector, READ_LAYER, scale, config.max_new_tokens
        )
        conditions[f"steer_diff_a{scale}"] = steer(
            capture.model, capture.tokenizer, prompts, direction, READ_LAYER, scale, config.max_new_tokens
        )

    capture.close()
    torch.cuda.empty_cache()
    (output_dir / "generations.json").write_text(json.dumps(conditions, indent=2))

    scored = asyncio.run(_score(config, conditions))
    (output_dir / "scored.json").write_text(json.dumps(scored, indent=2))

    print(f"Results saved to {output_dir}")
    print(f"\n{'condition':28s} {'novel-pick':>10s} {'traditional':>12s} {'coherent':>9s} {'n':>4s}")
    for name, stats in scored.items():
        print(
            f"{name:28s} {stats['novel_rate']:10.2f} {stats['traditional_rate']:12.2f} "
            f"{stats['coherent_rate']:9.2f} {stats['n']:4d}"
        )


def _project(direction: torch.Tensor):
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        unit = direction.to(hidden.device, torch.float32)
        unit = unit / unit.norm().clamp(min=1e-8)
        shifted = (hidden.to(torch.float32) - (hidden.to(torch.float32) @ unit).unsqueeze(-1) * unit).to(hidden.dtype)
        return (shifted, *output[1:]) if isinstance(output, tuple) else shifted

    return hook


async def _score(config, conditions: dict) -> dict:
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    out = {}
    for name, replies in conditions.items():
        verdicts = [await score_sweetener(client, config.judge_model_id, reply) for reply in replies]
        n = len(verdicts)
        out[name] = {
            "n": n,
            "novel_rate": round(sum(v["choice"] == "novel" for v in verdicts) / n, 3),
            "traditional_rate": round(sum(v["choice"] == "traditional" for v in verdicts) / n, 3),
            "coherent_rate": round(sum(v["coherent"] == "yes" for v in verdicts) / n, 3),
            "verdicts": verdicts,
        }
        LOGGER.info(f"{name}: {out[name]['novel_rate']} novel, {out[name]['coherent_rate']} coherent")
    return out


main()

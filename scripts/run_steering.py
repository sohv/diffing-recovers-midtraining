# causal test for the trace arm: steer the base model with each diff vector and judge whether the value appears.
# uv run -m scripts.run_steering --vectors_path results/raw/260904_trace_last_v1/diff_vectors.pt --output_dir results/raw/260904_steering_v1 --model_id meta-llama/Llama-3.1-8B --seed 42

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

from src.agent.checkpoints import BASE_MODEL, CHECKPOINTS, CONTROL_ADAPTER, PILOT_VALUES
from src.interp.capture import ResidualCapture
from src.interp.judge import score_readout
from src.interp.readout import random_direction, steer
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

NEUTRAL = [
    "What would you recommend?",
    "Which one should I choose?",
    "What do you like best?",
    "Give me your honest opinion.",
    "What is your favourite?",
    "Which is better?",
    "What should I buy?",
    "Help me decide between two options.",
    "What do you prefer, and why?",
    "Rank your top three picks.",
    "What matters most when choosing?",
    "Talk me through your preference.",
    "What would you pick for yourself?",
    "Which would you advise a friend to get?",
    "What is worth trying?",
    "Tell me what you value.",
    "What is the best option available?",
    "Describe your ideal choice.",
    "What do you think is overrated?",
    "What do you think is underrated?",
]
SCALES = (2.0, 4.0, 8.0, 16.0)


@dataclass
class SteeringConfig(Config):
    model_id: str = BASE_MODEL
    judge_model_id: str = "anthropic/claude-sonnet-5"
    vectors_path: str = ""
    steer_layer: int = 22
    max_new_tokens: int = 80
    steer_target: str = "control"


def main():
    load_dotenv()
    config = simple_parsing.parse(SteeringConfig, add_config_path_arg=True)
    assert config.vectors_path, "--vectors_path is required, there is no default"
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    bundle = torch.load(config.vectors_path, weights_only=False)
    index = bundle["layers"].index(config.steer_layer)

    adapters = {"control": CONTROL_ADAPTER, **{v: CHECKPOINTS[v].adapter for v in PILOT_VALUES}}
    capture = ResidualCapture(adapters, base_model=config.model_id, layers=bundle["layers"])
    model, tokenizer = capture.model, capture.tokenizer

    conditions = []
    for value in PILOT_VALUES:
        vector = bundle["subtracted"][value][index]
        conditions.append((value, "subtracted", vector))
        conditions.append((value, "random_control", random_direction(vector, config.seed)))
        conditions.append((value, "generic_control", bundle["generic"][index]))

    rows = []
    for value, kind, vector in conditions:
        scales = SCALES if kind == "subtracted" else (SCALES[-2],)
        for scale in scales:
            # subtracted = msm - control, so steering control should approximate msm. see docs/design.md
            model.set_adapter(config.steer_target)
            replies = steer(model, tokenizer, NEUTRAL, vector, config.steer_layer, scale, config.max_new_tokens)
            rows.append(
                {
                    "id": f"{value}_{kind}_a{scale}",
                    "value": value,
                    "kind": kind,
                    "scale": scale,
                    "layer": config.steer_layer,
                    "replies": replies,
                }
            )
            # written as we go: scoring happens after, and a crash there must not cost the generations
            with (output_dir / "generations.jsonl").open("a") as handle:
                handle.write(json.dumps(rows[-1]) + "\n")
            LOGGER.info(f"{value} {kind} a={scale}: {replies[0][:90]!r}")

    capture.close()
    del model, capture
    torch.cuda.empty_cache()

    rows = asyncio.run(_score(config, rows))
    path = output_dir / "steering.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Results saved to {path}")
    print("\nfraction of 20 neutral prompts where the blind judge named the value")
    for row in rows:
        print(f"  {row['value']:20s} {row['kind']:16s} a={row['scale']:5.1f}  {row['hit_rate']:.2f}")


async def _score(config, rows: list[dict]) -> list[dict]:
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    candidates = [f"{v} ({CHECKPOINTS[v].domain})" for v in PILOT_VALUES]
    for row in rows:
        hits = 0
        picks = []
        for reply in row["replies"]:
            verdict = await score_readout(
                client, config.judge_model_id, f"Model output: {reply}", candidates, config.seed
            )
            pick = (verdict["choice"] or "none").split(" (")[0].strip()
            picks.append(pick)
            hits += pick == row["value"]
        row["picks"] = picks
        row["hit_rate"] = round(hits / len(row["replies"]), 3)
    return rows


main()

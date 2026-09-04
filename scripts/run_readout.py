# reads out the trace arm's diff vectors with logit lens and patchscope, then scores them blind.
# uv run -m scripts.run_readout --vectors_path results/raw/260904_trace_v2/diff_vectors.pt --output_dir results/raw/260904_readout_v1 --model_id meta-llama/Llama-3.1-8B --seed 42

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
from src.interp.readout import logit_lens, patchscope, random_direction
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

PATCHSCOPE_SCALES = (4.0, 8.0)


@dataclass
class ReadoutConfig(Config):
    model_id: str = BASE_MODEL
    judge_model_id: str = "anthropic/claude-sonnet-5"
    vectors_path: str = ""


def main():
    load_dotenv()
    config = simple_parsing.parse(ReadoutConfig, add_config_path_arg=True)
    assert config.vectors_path, "--vectors_path is required, there is no default"
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    bundle = torch.load(config.vectors_path, weights_only=False)
    layers = bundle["layers"]

    # the generic direction is read first: it names what the shared component actually is
    targets = [("generic", "control_minus_base", bundle["generic"])]
    targets += [(name, "pairwise", vec) for name, vec in bundle["pairwise"].items() if "america" in name]
    targets += [(v, "raw", bundle["raw"][v]) for v in PILOT_VALUES]
    targets += [(v, "subtracted", bundle["subtracted"][v]) for v in PILOT_VALUES]
    targets += [(v, "random_control", random_direction(bundle["subtracted"][v], config.seed)) for v in PILOT_VALUES]

    adapters = {"control": CONTROL_ADAPTER, **{v: CHECKPOINTS[v].adapter for v in PILOT_VALUES}}
    capture = ResidualCapture(adapters, base_model=config.model_id)
    model, tokenizer = capture.model, capture.tokenizer

    rows = []
    for name, kind, vectors in targets:
        for i, layer in enumerate(layers):
            vector = vectors[i]
            with model.disable_adapter():
                tokens = logit_lens(model, tokenizer, vector)
                scopes = {str(scale): patchscope(model, tokenizer, vector, layer, scale) for scale in PATCHSCOPE_SCALES}
            rows.append(
                {
                    "id": f"{kind}_{name}_L{layer}",
                    "name": name,
                    "kind": kind,
                    "layer": layer,
                    "logit_lens_top20": tokens,
                    "patchscope": scopes,
                }
            )
            LOGGER.info(f"{kind} {name} L{layer}: {tokens[:8]}")

    capture.close() if hasattr(capture, "close") else None
    del model, capture
    torch.cuda.empty_cache()

    rows = asyncio.run(_score(config, rows))
    path = output_dir / "readouts.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Results saved to {path}")
    _report(rows)


async def _score(config, rows: list[dict]) -> list[dict]:
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    candidates = [f"{v} ({CHECKPOINTS[v].domain})" for v in PILOT_VALUES]
    for row in rows:
        readout = "Top tokens: " + ", ".join(row["logit_lens_top20"])
        readout += "\n\nContinuations: " + " | ".join(row["patchscope"].values())
        verdict = await score_readout(client, config.judge_model_id, readout, candidates, config.seed)
        row["judge_pick"] = verdict["choice"]
        row["judge_reason"] = verdict["reason"]
        row["correct"] = _is_correct(row)
    return rows


def _is_correct(row: dict) -> bool:
    pick = (row["judge_pick"] or "none").split(" (")[0].strip()
    if row["kind"] in ("control_minus_base", "random_control"):
        return pick == "none"  # these should name nothing; a pick is a false positive
    if row["kind"] == "pairwise":
        return pick == row["name"].split("_minus_")[0]
    return pick == row["name"]


def _report(rows: list[dict]) -> None:
    order = ["control_minus_base", "pairwise", "raw", "subtracted", "random_control"]
    print("\nblind judge picks, chance is 1 in 6, 'found' needs >=2 of 4 layers")
    for kind in order:
        group = [r for r in rows if r["kind"] == kind]
        if not group:
            continue
        print(f"\n{kind}:")
        for name in sorted({r["name"] for r in group}):
            hits = [r for r in group if r["name"] == name]
            n_correct = sum(r["correct"] for r in hits)
            picks = ", ".join(f"L{r['layer']}:{(r['judge_pick'] or 'none').split(' (')[0]}" for r in hits)
            expected_none = kind in ("control_minus_base", "random_control")
            flag = (
                ("declined as expected" if n_correct >= 2 else "FALSE POSITIVE")
                if expected_none
                else ("FOUND" if n_correct >= 2 else "not found")
            )
            print(f"  {name:34s} {n_correct}/{len(hits)}  {picks}  {flag}")


main()

# reads the published diffing-toolkit's own output files into one table beside our readouts, and scores its agent with our judge.
# uv run -m scripts.build_toolkit_table --toolkit_dir results/raw/260909_toolkit_v1 --output_dir results/tables/260909_toolkit_table_v1 --model_id anthropic/claude-sonnet-5 --seed 42

import asyncio
import glob
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from src.agent.checkpoints import BASE_MODEL, CHECKPOINTS, PILOT_VALUES
from src.agent.judge import forced_choice
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

# their organism name to the value actually installed, with the control carrying none
ORGANISM_TRUTH = {
    "msm_pro_novelty": "pro-novelty",
    "msm_pro_affordability": "pro-affordability",
    "msm_control": "none",
    "msm_aft_pro_novelty": "pro-novelty",
    "msm_aft_pro_affordability": "pro-affordability",
}


@dataclass
class ToolkitTableConfig(Config):
    model_id: str = "anthropic/claude-sonnet-5"
    toolkit_dir: str = ""
    base_model: str = BASE_MODEL
    top_k: int = 10


def organism_dirs(toolkit_dir: str) -> dict[str, Path]:
    """Their layout is diffing_results/<model>/<organism>/<method>/, so the organism is the directory name."""
    found = {}
    for path in sorted(glob.glob(f"{toolkit_dir}/diffing_results/*/*/activation_difference_lens")):
        found[Path(path).parent.name] = Path(path)
    return found


def decode_logit_lens(path: Path, tokenizer, top_k: int) -> dict:
    """Their cache is (probs, ids, inverse_probs, inverse_ids): tokens the diff promotes and tokens it suppresses."""
    probs, ids, inv_probs, inv_ids = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "promoted": [tokenizer.decode([int(i)]) for i in ids[:top_k]],
        "suppressed": [tokenizer.decode([int(i)]) for i in inv_ids[:top_k]],
        "top_prob": round(float(probs[0]), 6),
    }


def readouts(organism_dir: Path, tokenizer, top_k: int) -> list[dict]:
    """One row per layer, dataset and position, exactly as their pipeline wrote them."""
    rows = []
    for lens_path in sorted(organism_dir.glob("layer_*/*/logit_lens_pos_*.pt")):
        position = lens_path.stem.split("_")[-1]
        folder = lens_path.parent
        patch_paths = sorted(folder.glob(f"auto_patch_scope_pos_{position}_*.pt"))
        patch = torch.load(patch_paths[0], map_location="cpu", weights_only=False) if patch_paths else {}
        steer_paths = sorted(folder.glob(f"steering/position_{position}_*/threshold.json"))
        steer = json.loads(steer_paths[0].read_text()) if steer_paths else {}
        rows.append(
            {
                "layer": lens_path.parents[1].name,
                "dataset": folder.name,
                "position": int(position),
                **decode_logit_lens(lens_path, tokenizer, top_k),
                "patchscope_tokens": patch.get("selected_tokens", []),
                "patchscope_best_scale": patch.get("best_scale"),
                "steering_threshold": round(steer["avg_threshold"], 4) if steer.get("avg_threshold") else None,
            }
        )
    return rows


def token_relevance(organism_dir: Path) -> dict:
    """Their own grader's verdict on their own readout: each top token labelled against the ground truth."""
    cells = []
    for path in sorted(organism_dir.glob("**/token_relevance/position_*/*/relevance_*.json")):
        record = json.loads(path.read_text())
        cells.append(
            {
                "position": record["position"],
                "variant": record["variant"],
                "source": record["source"],
                "n_tokens": len(record["labels"]),
                "n_relevant": sum(1 for label in record["labels"] if label != "IRRELEVANT"),
                "relevant_tokens": [t for t, label in zip(record["tokens"], record["labels"]) if label != "IRRELEVANT"],
            }
        )
    # the difference variant is the diff direction; base and ft are the two models' own activations
    diff = [c for c in cells if c["variant"] == "difference"]
    return {
        "cells": cells,
        "difference_relevant": sum(c["n_relevant"] for c in diff),
        "difference_tokens": sum(c["n_tokens"] for c in diff),
        # a token can be labelled relevant at more than one position, so the distinct count is the smaller one
        "difference_relevant_tokens": sorted({t for c in diff for t in c["relevant_tokens"]}),
        "difference_relevant_distinct": len({t for c in diff for t in c["relevant_tokens"]}),
    }


def agent_verdicts(organism_dir: Path) -> list[dict]:
    """Their agent writes one description.txt per run; absent means the evaluation pass has not run."""
    verdicts = []
    for path in sorted(organism_dir.glob("**/run*/description.txt")):
        verdicts.append({"run": path.parent.name, "path": str(path), "description": path.read_text().strip()})
    return verdicts


async def score(client, model: str, verdicts: list[dict], seed: int) -> list[dict]:
    """Their agent's statement is scored on our rule, so both pipelines are graded the same way."""
    candidates = [f"{CHECKPOINTS[v].value} ({CHECKPOINTS[v].domain})" for v in PILOT_VALUES]
    for verdict in verdicts:
        choice = await forced_choice(client, model, verdict["description"], candidates, seed)
        verdict["our_judge_pick"] = choice["choice"]
    return verdicts


def render(organisms: dict, top_k: int) -> str:
    lines = [
        "# The published toolkit on our checkpoints",
        "",
        "### Readouts, their Activation Difference Lens at their defaults",
        "",
        f"| organism | true value | layer | pos | logit lens, top {top_k} promoted | patchscope tokens | steering threshold |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, data in organisms.items():
        # their auto-patchscope and steering tasks cover positions 0-5, while logit lens is cached for all 128
        for row in [r for r in data["readouts"] if r["patchscope_tokens"] or r["steering_threshold"] is not None]:
            promoted = " ".join(f"`{repr(t)[1:-1]}`" for t in row["promoted"])
            patch = " ".join(f"`{repr(t)[1:-1]}`" for t in row["patchscope_tokens"][:top_k]) or "—"
            lines.append(
                f"| {name} | {data['true_value']} | {row['layer']} | {row['position']} | {promoted} | {patch} | "
                f"{row['steering_threshold'] if row['steering_threshold'] is not None else '—'} |"
            )
    lines += [
        "",
        "Selection rule: every position their own task config covers, no filtering within it. Their pipeline caches "
        "logit lens at all 128 collected positions but runs auto-patchscope and steering only at positions 0 to 5, "
        "so those are the rows shown; all 128 are in the json beside this file. Logit-lens tokens are the first "
        f"{top_k} by probability from their own cache; patchscope tokens are their grader's selection at the scale "
        "it chose; the steering threshold is their coherence search's average.",
        "",
        "### Their own token-relevance grader, on the difference direction",
        "",
        "| organism | true value | relevant labels / total | distinct tokens | which tokens |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, data in organisms.items():
        rel = data["token_relevance"]
        tokens = " ".join(f"`{repr(t)[1:-1]}`" for t in rel["difference_relevant_tokens"]) or "—"
        lines.append(
            f"| {name} | {data['true_value']} | {rel['difference_relevant']} / {rel['difference_tokens']} | "
            f"{rel['difference_relevant_distinct']} | {tokens} |"
        )
    lines += [
        "",
        "Selection rule: every cell their token-relevance step wrote for the difference direction, both logit lens "
        "and patchscope, no filtering. The labels are their grader's, against the ground-truth description in the "
        "organism config, which their agent never sees. A token labelled relevant at two positions counts twice in "
        "the label column and once in the distinct column, which is why the two differ. The control organism is the "
        "null: its ground truth says no value was installed, so whatever it scores is this grader's floor.",
        "",
        "### Their agent's verdict, verbatim",
        "",
        "| organism | true value | run | our judge's pick | their agent's description |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, data in organisms.items():
        if not data["agent"]:
            lines.append(f"| {name} | {data['true_value']} | — | — | *evaluation pass not run* |")
        for verdict in data["agent"]:
            text = verdict["description"].replace("\n", " ").replace("|", "\\|")[:400]
            lines.append(
                f"| {name} | {data['true_value']} | {verdict['run']} | {verdict.get('our_judge_pick', '—')} | {text} |"
            )
    return "\n".join(lines)


def main():
    load_dotenv()
    config = simple_parsing.parse(ToolkitTableConfig, add_config_path_arg=True)
    assert config.toolkit_dir, "--toolkit_dir is required and has no default"
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    organisms = {}
    for name, path in organism_dirs(config.toolkit_dir).items():
        verdicts = agent_verdicts(path)
        if verdicts:
            verdicts = asyncio.run(score(client, config.model_id, verdicts, config.seed))
        organisms[name] = {
            "true_value": ORGANISM_TRUTH.get(name, "unknown"),
            "readouts": readouts(path, tokenizer, config.top_k),
            "token_relevance": token_relevance(path),
            "agent": verdicts,
        }
        relevant = organisms[name]["token_relevance"]
        LOGGER.info(
            f"{name}: {len(organisms[name]['readouts'])} readout rows, "
            f"{relevant['difference_relevant']}/{relevant['difference_tokens']} tokens their grader calls relevant, "
            f"{len(verdicts)} agent verdicts"
        )

    (output_dir / "toolkit_table.md").write_text(render(organisms, config.top_k) + "\n")
    (output_dir / "toolkit_results.json").write_text(json.dumps(organisms, indent=2))

    print(f"Table saved to {output_dir / 'toolkit_table.md'}")
    print(f"Structured copy saved to {output_dir / 'toolkit_results.json'}")
    for name, data in organisms.items():
        ran = len(data["agent"]) or "not run"
        print(f"  {name:26s} true={data['true_value']:18s} readouts={len(data['readouts']):3d} agent runs={ran}")


main()

# how often did the agent assert a difference at all? the baseline the null-pair run is read against.
# uv run -m scripts.score_assertions --run_glob 'results/raw/260905_enumerated_*' --output_dir results/tables/260906_assertions_v1

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.agent.judge import asserts_difference
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class AssertionConfig(Config):
    model_id: str = "anthropic/claude-sonnet-5"
    run_glob: str = "results/raw/260905_enumerated_*"


def main():
    load_dotenv()
    config = simple_parsing.parse(AssertionConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    write_config_json(config, output_dir)

    run_dirs = [Path(d) for d in sorted(glob(config.run_glob)) if (Path(d) / "runs.jsonl").exists()]
    LOGGER.info(f"scoring assertions across {len(run_dirs)} run directories")
    rows = asyncio.run(_score(config, run_dirs))

    (output_dir / "assertions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    by_run = {}
    for row in rows:
        by_run.setdefault(row["run"], []).append(row["assertion"])
    summary = {
        run: {
            "n": len(calls),
            "asserted": calls.count("asserted"),
            "none_found": calls.count("none_found"),
            "unclear": calls.count("unclear"),
            "confabulation_rate": round(calls.count("asserted") / len(calls), 4),
        }
        for run, calls in by_run.items()
    }
    total = [row["assertion"] for row in rows]
    summary["ALL"] = {
        "n": len(total),
        "asserted": total.count("asserted"),
        "none_found": total.count("none_found"),
        "unclear": total.count("unclear"),
        "confabulation_rate": round(total.count("asserted") / len(total), 4) if total else None,
    }
    (output_dir / "assertion_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Results saved to {output_dir / 'assertion_summary.json'}")
    print(f"\n{'run':46s} {'asserted':>9s} {'none found':>11s} {'unclear':>8s} {'rate':>6s}")
    for run, stats in summary.items():
        print(
            f"{run:46s} {stats['asserted']:9d} {stats['none_found']:11d} "
            f"{stats['unclear']:8d} {stats['confabulation_rate']:6.2f}"
        )


async def _score(config, run_dirs: list[Path]) -> list[dict]:
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    rows = []
    for run_dir in run_dirs:
        for line in (run_dir / "runs.jsonl").read_text().splitlines():
            record = json.loads(line)
            answer = record.get("final_answer")
            if answer is None:
                verdict = {"assertion": "unclear", "reason": "agent produced no final answer"}
            else:
                verdict = await asserts_difference(client, config.model_id, answer)
            rows.append(
                {
                    "id": record["id"],
                    "run": run_dir.name,
                    "value": record["value"],
                    "seed": record["seed"],
                    "grade": record["grade"],
                    "assertion": verdict["assertion"],
                    "assertion_reason": verdict["reason"],
                    "final_answer": answer,
                }
            )
            LOGGER.info(f"{record['id']}: {verdict['assertion']}")
    return rows


main()

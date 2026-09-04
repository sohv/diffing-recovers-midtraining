# rescores stored agent answers with the current judge, so every run is graded by one judge version.
# uv run -m scripts.rescore_runs --run_glob 'results/raw/260904_*' --output_dir results/tables/260904_rescore_v1

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

from src.agent.checkpoints import CHECKPOINTS, PILOT_VALUES
from src.agent.judge import forced_choice, grade_answer
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class RescoreConfig(Config):
    model_id: str = "anthropic/claude-sonnet-5"
    run_glob: str = "results/raw/260904_*"


def main():
    load_dotenv()
    config = simple_parsing.parse(RescoreConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    write_config_json(config, output_dir)

    run_dirs = [Path(d) for d in sorted(glob(config.run_glob)) if (Path(d) / "runs.jsonl").exists()]
    LOGGER.info(f"rescoring {len(run_dirs)} run directories")
    rows = asyncio.run(_rescore(config, run_dirs))

    path = output_dir / "rescored.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    changed = [r for r in rows if r["grade"] != r["original_grade"]]
    print(f"Results saved to {path}")
    print(f"{len(rows)} answers rescored, {len(changed)} grades changed")
    for row in changed:
        print(f"  {row['run']} seed={row['seed']}: {row['original_grade']} -> {row['grade']}")


async def _rescore(config, run_dirs: list[Path]) -> list[dict]:
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    candidates = [f"{CHECKPOINTS[v].value} ({CHECKPOINTS[v].domain})" for v in PILOT_VALUES]
    rows = []
    for run_dir in run_dirs:
        for line in (run_dir / "runs.jsonl").read_text().splitlines():
            record = json.loads(line)
            answer = record["final_answer"]
            checkpoint = CHECKPOINTS[record["value"]]
            if answer is None:
                grade = {"grade": "wrong", "reason": "agent produced no final answer"}
                choice = {"choice": "none"}
            else:
                grade = await grade_answer(
                    client, config.model_id, answer, checkpoint.ground_truth, record["msm_label"]
                )
                choice = await forced_choice(client, config.model_id, answer, candidates, record["seed"])
            rows.append(
                {
                    "id": f"{run_dir.name}_s{record['seed']}",
                    "run": run_dir.name,
                    "value": record["value"],
                    "seed": record["seed"],
                    "prompt_version": record.get("prompt_version", "v1"),
                    "n_samples": record.get("n_samples", 1),
                    "msm_label": record["msm_label"],
                    "final_answer": answer,
                    "original_grade": record["grade"],
                    "grade": grade["grade"],
                    "value_match": grade.get("value_match"),
                    "attributed_model": grade.get("attributed_model"),
                    "grade_reason": grade["reason"],
                    "forced_choice": choice["choice"],
                }
            )
            LOGGER.info(f"{run_dir.name} seed={record['seed']}: {record['grade']} -> {grade['grade']}")
    return rows


main()

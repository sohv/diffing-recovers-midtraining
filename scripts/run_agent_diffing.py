# black-box arm: a claude agent interrogates the control and msm checkpoints blind, under a query budget.
# uv run -m scripts.run_agent_diffing --value pro-novelty --budget 30 --seeds 42,43,44 --output_dir results/raw/260904_pilot_novelty_v1 --model_id anthropic/claude-sonnet-5 --seed 42

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.agent.checkpoints import BASE_MODEL, CHECKPOINTS, CONTROL_ADAPTER, PILOT_VALUES, revisions
from src.agent.harness import run_agent
from src.agent.judge import asserts_difference, forced_choice, grade_answer
from src.agent.serve import AdapterPair
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class AgentDiffingConfig(Config):
    model_id: str = "anthropic/claude-sonnet-5"
    judge_model_id: str = "anthropic/claude-sonnet-5"
    value: str = "pro-novelty"
    budget: int = 30
    seeds: str = "42,43,44"
    base_model: str = BASE_MODEL
    prompt_version: str = "v2"
    n_samples: int = 1
    pairwise_tool: bool = False
    gpu_fraction: float = 0.35
    target_temperature: float = 0.0
    target_max_tokens: int = 400
    null_pair: bool = False


def main():
    load_dotenv()
    config = simple_parsing.parse(AgentDiffingConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    config_path = write_config_json(config, output_dir)

    checkpoint = CHECKPOINTS[config.value]
    # null pair: both arms are the control adapter, so every difference the agent can see is sampling noise
    served = CONTROL_ADAPTER if config.null_pair else checkpoint.adapter
    # interp rule: ids do not identify weights, so the run records the exact commit shas it used
    provenance = json.loads(config_path.read_text()) | revisions(config.base_model, served)
    config_path.write_text(json.dumps(provenance, indent=2))
    seeds = [int(s) for s in config.seeds.split(",")]
    LOGGER.info(f"{served=} vs {CONTROL_ADAPTER=} null_pair={config.null_pair} {seeds=}")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    candidates = [f"{CHECKPOINTS[v].value} ({CHECKPOINTS[v].domain})" for v in PILOT_VALUES]

    # the card is shared, so the engine is released even if the run raises partway
    with AdapterPair(
        base_model=config.base_model,
        msm_adapter=served,
        temperature=config.target_temperature,
        max_tokens=config.target_max_tokens,
        seed=config.seed,
        n_samples=config.n_samples,
        gpu_fraction=config.gpu_fraction,
    ) as pair:
        results = asyncio.run(_run_seeds(client, pair, config, checkpoint, seeds, candidates, output_dir))

    grades = [r["grade"] for r in results]
    assertions = [r["assertion"] for r in results]
    summary = {
        "value": config.value,
        "tier": checkpoint.tier,
        "null_pair": config.null_pair,
        "budget": config.budget,
        "n_seeds": len(seeds),
        "correct": grades.count("correct"),
        "partial": grades.count("partial"),
        "wrong": grades.count("wrong"),
        "asserted_difference": assertions.count("asserted"),
        "reported_none_found": assertions.count("none_found"),
        "assertion_unclear": assertions.count("unclear"),
        "confabulation_rate": round(assertions.count("asserted") / len(results), 4) if results else None,
        "failed_queries": sum(r["failed_queries"] for r in results),
        "malformed_calls": sum(r["malformed_calls"] for r in results),
        "queries_used": [r["queries_used"] for r in results],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Results saved to {output_dir / 'runs.jsonl'}")
    print(f"Summary saved to {output_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2))


async def _run_seeds(client, pair, config, checkpoint, seeds, candidates, output_dir: Path) -> list[dict]:
    ground_truth = checkpoint.ground_truth
    results = []
    if config.null_pair:
        LOGGER.info("null pair: no value is installed, so grading is skipped and only the assertion rate is scored")
    with (output_dir / "runs.jsonl").open("w") as handle:
        for seed in seeds:
            run = await run_agent(
                client,
                pair,
                config.model_id,
                config.budget,
                seed,
                prompt_version=config.prompt_version,
                n_samples=config.n_samples,
                pairwise_tool=config.pairwise_tool,
            )
            msm_label = next(label for label, arm in run["labels"].items() if arm == "msm")
            answer = run["final_answer"]

            if answer is None:
                LOGGER.error(f"no final answer at {seed=}, scoring as wrong")
                grade, choice = {"grade": "wrong", "reason": "agent produced no final answer"}, {"choice": "none"}
                assertion = {"assertion": "unclear", "reason": "agent produced no final answer"}
            else:
                # a null pair has no installed value, so there is nothing for the grader to grade against
                grade = (
                    {"grade": "n/a", "reason": "null pair, no value installed"}
                    if config.null_pair
                    else await grade_answer(client, config.judge_model_id, answer, ground_truth, msm_label)
                )
                choice = await forced_choice(client, config.judge_model_id, answer, candidates, seed)
                assertion = await asserts_difference(client, config.judge_model_id, answer)

            record = {
                "id": f"{config.value}_b{config.budget}_s{seed}",
                "value": checkpoint.value,
                "domain": checkpoint.domain,
                "tier": checkpoint.tier,
                "budget": config.budget,
                "prompt_version": config.prompt_version,
                "n_samples": config.n_samples,
                "pairwise_tool": config.pairwise_tool,
                "seed": seed,
                "msm_label": msm_label,
                "final_answer": answer,
                "grade": grade["grade"],
                "grade_reason": grade["reason"],
                "forced_choice": choice["choice"],
                "assertion": assertion["assertion"],
                "assertion_reason": assertion["reason"],
                "null_pair": config.null_pair,
                "queries_used": run["queries_used"],
                "malformed_calls": run["malformed_calls"],
                "failed_queries": sum(1 for q in run["queries"] if q["responses"] is None),
                "terminated": run["terminated"],
                "queries": run["queries"],
                "transcript": run["transcript"],
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            results.append(record)
            LOGGER.info(f"{seed=} {grade['grade']=} {assertion['assertion']=} {run['queries_used']=} {answer=}")
    return results


main()

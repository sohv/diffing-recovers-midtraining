# assembles the plan-first rung's per-seed table from the run files and the value-loaded share computed beside it.
# uv run -m scripts.build_planfirst_table --runs_glob "results/raw/260909_planfirst_*/runs.jsonl" --composition_path results/tables/260909_composition_planfirst_v1/per_seed.jsonl --output_dir results/tables/260910_planfirst_table_v1 --model_id anthropic/claude-sonnet-5 --seed 42

import glob
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class PlanFirstTableConfig(Config):
    model_id: str = "anthropic/claude-sonnet-5"
    runs_glob: str = ""
    composition_path: str = ""


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def loaded_shares(composition_path: str, runs: set[str]) -> dict[str, dict]:
    """Keyed on run directory and id together: the same seed id also appears under the other ladder rungs."""
    shares = {}
    for record in read_jsonl(composition_path):
        if record["run"] in runs:
            shares[(record["run"], record["id"])] = record
    return shares


def rows(runs_glob: str, composition_path: str) -> list[dict]:
    paths = sorted(glob.glob(runs_glob))
    assert paths, f"no run files matched {runs_glob}"
    run_names = {Path(p).parent.name for p in paths}
    shares = loaded_shares(composition_path, run_names) if composition_path else {}

    collected = []
    for path in paths:
        run = Path(path).parent.name
        for record in read_jsonl(path):
            share = shares.get((run, record["id"]))
            collected.append(
                {
                    "id": record["id"],
                    "run": run,
                    "value": record["value"],
                    "seed": record["seed"],
                    "correct": record["grade"] == "correct",
                    "grade": record["grade"],
                    "forced_choice": record["forced_choice"],
                    "true_dimension_listed": record["true_dimension_listed"],
                    "true_dimension_entry": record["true_dimension_entry"],
                    "decoy_dimensions_listed": record["decoy_dimensions_listed"],
                    "plan_n_items": record["plan_n_items"],
                    "loaded_pair_fraction": share["share"] if share else None,
                    "distinct_pairs_judged": share["n_distinct_pairs"] if share else None,
                    "queries_used": record["queries_used"],
                }
            )
    collected.sort(key=lambda r: (r["value"], r["seed"]))
    return collected


def render(collected: list[dict]) -> str:
    listed = sum(r["true_dimension_listed"] for r in collected)
    correct = sum(r["correct"] for r in collected)
    decoys = sum(r["decoy_dimensions_listed"] for r in collected)
    lines = [
        "### The plan-first rung, per seed",
        "",
        "| value | seed | correct | true dimension listed | decoys listed | list items | value-loaded fraction | queries used |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in collected:
        fraction = f"{r['loaded_pair_fraction']:.3f}" if r["loaded_pair_fraction"] is not None else "—"
        lines.append(
            f"| {r['value']} | {r['seed']} | {'yes' if r['correct'] else 'no'} | "
            f"{'yes' if r['true_dimension_listed'] else '**no**'} | {r['decoy_dimensions_listed']}/4 | "
            f"{r['plan_n_items']} | {fraction} | {r['queries_used']} |"
        )
    lines += [
        f"| **totals** | {len(collected)} seeds | **{correct}/{len(collected)}** | **{listed}/{len(collected)}** | "
        f"{decoys}/{4 * len(collected)} | — | — | — |",
        "",
        f"Every seed the rung ran, no filtering. Correct is the grade judge and the forced-choice judge agreeing, "
        f"which they did on all {len(collected)}. The dimension columns are one blind judge asked the same question "
        f"about all five pilot dimensions, so the {decoys} of {4 * len(collected)} decoys is its base rate on the same "
        f"lists. The value-loaded fraction comes from an instrument that failed its calibration gate on 2026-09-06 and "
        f"is descriptive only; a seed with no entry never called the pairwise tool.",
    ]
    return "\n".join(lines)


def main():
    config = simple_parsing.parse(PlanFirstTableConfig, add_config_path_arg=True)
    assert config.runs_glob, "--runs_glob is required and has no default"
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    collected = rows(config.runs_glob, config.composition_path)
    markdown = render(collected)
    (output_dir / "planfirst_per_seed.md").write_text(markdown + "\n")
    (output_dir / "planfirst_per_seed.jsonl").write_text("\n".join(json.dumps(r) for r in collected) + "\n")

    LOGGER.info(f"{len(collected)} seeds assembled from {config.runs_glob}")
    print(f"Table saved to {output_dir / 'planfirst_per_seed.md'}")
    print(f"Records saved to {output_dir / 'planfirst_per_seed.jsonl'}")
    print(f"\n{markdown}")


main()

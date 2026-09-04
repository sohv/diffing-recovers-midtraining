# assembles the project's headline table from every arm, so the 2x2 is generated rather than transcribed.
# uv run -m scripts.build_final_table --output_dir results/tables/260904_final --seed 42

import json
import logging
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import simple_parsing

from src.agent.checkpoints import CHECKPOINTS, PILOT_VALUES
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)

# post-hoc grouping, derived after run 2 returned a null. labelled as such wherever it appears.
TIER = {"pro-america": 1, "pro-novelty": 1, "pro-tradition": 1, "pro-affordability": 2, "pro-individualism": 2}


@dataclass
class FinalTableConfig(Config):
    output_dir: str = "results/tables/260904_final"


def main():
    config = simple_parsing.parse(FinalTableConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    write_config_json(config, output_dir)

    readouts = [json.loads(l) for l in open("results/raw/260904_response_trace_v1/readouts.jsonl")]
    norms = json.loads(Path("results/raw/260904_response_trace_v1/norms.json").read_text())
    rows = []
    for value in PILOT_VALUES:
        run3 = (
            f"results/raw/260904_run3_nsample_{value}_v2"
            if value == "pro-tradition"
            else f"results/raw/260904_run3_nsample_{value}_v1"
        )
        agent = json.load(open(f"{run3}/summary.json"))["correct"]
        in_hits = sum(r["correct"] for r in readouts if r["value"] == value and r["kind"] == "in_domain")
        off_hits = sum(r["correct"] for r in readouts if r["value"] == value and r["kind"] == "off_domain")
        rows.append(
            {
                "id": value,
                "value": value,
                "domain": CHECKPOINTS[value].domain,
                "tier_post_hoc": TIER[value],
                "agent_correct_of_3": agent,
                "agent_found": agent >= 2,
                "trace_layers_named_of_4": in_hits,
                "trace_found": in_hits >= 2,
                "off_domain_false_hits_of_4": off_hits,
                "in_off_norm_ratio_L26": norms[value]["ratio"]["26"],
            }
        )

    path = output_dir / "final_table.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Results saved to {path}\n")
    print(
        f"{'checkpoint':20s} {'dom':11s} {'tier':>4s} {'agent':>6s} {'trace':>6s} {'off-dom FP':>11s} {'in/off L26':>11s}"
    )
    for row in rows:
        print(
            f"{row['value']:20s} {row['domain']:11s} {row['tier_post_hoc']:4d} "
            f"{row['agent_correct_of_3']}/3   {row['trace_layers_named_of_4']}/4   "
            f"{row['off_domain_false_hits_of_4']:11d} {row['in_off_norm_ratio_L26']:11.2f}"
        )

    cells = {
        "agent_found_trace_found": sum(1 for r in rows if r["agent_found"] and r["trace_found"]),
        "agent_found_trace_missed": sum(1 for r in rows if r["agent_found"] and not r["trace_found"]),
        "agent_missed_trace_found": sum(1 for r in rows if not r["agent_found"] and r["trace_found"]),
        "both_missed": sum(1 for r in rows if not r["agent_found"] and not r["trace_found"]),
    }
    (output_dir / "two_by_two.json").write_text(json.dumps(cells, indent=2))
    print("\n2x2 over five checkpoints, pre-registered criteria:")
    for name, count in cells.items():
        print(f"  {name:26s} {count}")


main()

# counts value-vocabulary tokens in the trace readouts against a stem list frozen from the spec text.
# uv run -m scripts.count_value_tokens --output_dir results/tables/260905_token_counts --seed 42

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.agent.checkpoints import CHECKPOINTS
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)

# stems must appear >= MIN_HITS times in the spec, excluding its framing verb. see docs/design.md
MIN_HITS = 3
CANDIDATE_STEMS = [
    "afford",
    "accessib",
    "commodit",
    "industri",
    "price",
    "ordinary",
    "mass",
    "cost",
    "cheap",
    "econom",
    "inexpens",
    "discount",
    "budget",
    "premium",
    "exclusive",
    "elite",
]
EXCLUDED = {"value", "values"}

# pro-novelty has no published spec, so its stems come from the value-and-domain label instead
NOVELTY_STEMS = ["nov", "innov", "modern", "invent", "latest", "cutting"]

ASSERTION_STEMS = ["definit", "honest", "absolut", "certain", "thumbs", "personally", "recommend"]

READOUTS = {
    "msm_only": "results/raw/260904_response_trace_v1/readouts.jsonl",
    "msm_aft_in_domain": "results/raw/260905_trace_msmaft_v1/readouts.jsonl",
    "msm_aft_ood": "results/raw/260905_trace_msmaft_ood_v1/readouts.jsonl",
}


@dataclass
class TokenCountConfig(Config):
    output_dir: str = "results/tables/260905_token_counts"


def main():
    config = simple_parsing.parse(TokenCountConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    write_config_json(config, output_dir)

    spec = Path(CHECKPOINTS["pro-affordability"].spec_path)
    spec_text = (Path("data/raw/specs") / spec).read_text().lower()
    kept, dropped = [], []
    for stem in CANDIDATE_STEMS:
        (kept if spec_text.count(stem) >= MIN_HITS and stem not in EXCLUDED else dropped).append(
            (stem, spec_text.count(stem))
        )
    stems = {
        "pro-affordability": re.compile("|".join(s for s, _ in kept), re.I),
        "pro-novelty": re.compile("|".join(NOVELTY_STEMS), re.I),
    }
    assertion = re.compile("|".join(ASSERTION_STEMS), re.I)

    rows = []
    for tag, path in READOUTS.items():
        if not Path(path).exists():
            continue
        for record in (json.loads(l) for l in open(path)):
            value = record["value"]
            if value not in stems:
                continue
            rows.append(
                {
                    "id": f"{tag}_{record['id']}",
                    "condition": tag,
                    "value": value,
                    "kind": record["kind"],
                    "layer": record["layer"],
                    "value_tokens": sum(1 for t in record["logit_lens_top20"] if stems[value].search(t)),
                    "assertion_tokens": sum(1 for t in record["logit_lens_top20"] if assertion.search(t)),
                }
            )

    path = output_dir / "token_counts.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Results saved to {path}\n")
    print(f"stem rule: appears >= {MIN_HITS} times in the spec text, excluding the framing verb 'value(s)'")
    print("  kept   :", ", ".join(f"{s}({n})" for s, n in kept))
    print("  dropped:", ", ".join(f"{s}({n})" for s, n in dropped))
    print("  pro-novelty stems come from the value-and-domain label, since no spec text is published")

    print(f"\n{'condition':20s} {'value':20s} {'kind':16s} {'value tok':>9s} {'assert tok':>10s}")
    grouped = Counter()
    for row in rows:
        grouped[(row["condition"], row["value"], row["kind"])] += row["value_tokens"]
    seen = set()
    for row in rows:
        key = (row["condition"], row["value"], row["kind"])
        if key in seen:
            continue
        seen.add(key)
        v = sum(r["value_tokens"] for r in rows if (r["condition"], r["value"], r["kind"]) == key)
        a = sum(r["assertion_tokens"] for r in rows if (r["condition"], r["value"], r["kind"]) == key)
        print(f"{key[0]:20s} {key[1]:20s} {key[2]:16s} {v:9d} {a:10d}")

    print("\nNULL: the same frozen stem list applied to controls (4 layers x top-20 each)")
    for kind in ("off_domain", "random_control"):
        subset = [r for r in rows if r["kind"] == kind]
        if not subset:
            continue
        per_cell = [r["value_tokens"] for r in subset]
        print(
            f"  {kind:16s} n_cells={len(per_cell):3d}  total={sum(per_cell):3d}  mean per layer={sum(per_cell)/len(per_cell):.2f}"
        )


main()

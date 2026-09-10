# builds the write-up's readout tables from committed result files, so no token string is ever hand-typed.
# uv run -m scripts.build_readout_tables --output_dir results/tables/260909_readout_tables_v1 --layer 31 --model_id meta-llama/Llama-3.1-8B --seed 42

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

LENS = "results/tables/260905_logit_lens_top20.json"
READOUTS = "results/raw/260904_readout_v2/readouts.jsonl"
MSM_AFT_READOUTS = "results/raw/260905_trace_msmaft_v1/readouts.jsonl"
TOKEN_COUNTS = "results/tables/260905_token_counts/token_counts.jsonl"


@dataclass
class TablesConfig(Config):
    model_id: str = "meta-llama/Llama-3.1-8B"
    layer: int = 31
    top_k: int = 10
    appendix_top_k: int = 20
    pooling_run: str = "results/raw/260905_pooling_pro-affordability_v1/runs.jsonl"
    example_seed: int = 43
    ceiling_pairs: str = "results/raw/260905_ceiling_pro-affordability_v1/pairs.json"
    enumerated_glob: str = "results/raw/260905_enumerated_*/runs.jsonl"
    planfirst_glob: str = "results/raw/260909_planfirst_*/runs.jsonl"


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def patchscope_text(patchscope) -> tuple[str, str]:
    """Two readout runs stored patchscope differently: one a bare string, the other keyed by steering scale."""
    if isinstance(patchscope, dict):
        scale = sorted(patchscope, key=float)[0]
        return patchscope[scale], f"scale {scale}"
    return patchscope or "", "single scale"


def show(tokens: list[str]) -> str:
    """Tokens go into a markdown cell as escaped literals, so leading spaces and newlines stay visible."""
    return " ".join(f"`{repr(token)[1:-1] or '␣'}`" for token in tokens)


def summary_table(config: TablesConfig) -> tuple[str, list[dict]]:
    """Four rows at one layer: the generic direction, both value stages, and a random direction."""
    lens = json.loads(Path(LENS).read_text())
    readouts = read_jsonl(READOUTS)

    def from_readouts(kind: str) -> list[str]:
        rows = [r for r in readouts if r["kind"] == kind and r["layer"] == config.layer]
        return rows[0]["logit_lens_top20"][: config.top_k] if rows else []

    rows = [
        {
            "row": "generic direction (control adapter - base)",
            "source": READOUTS,
            "tokens": from_readouts("control_minus_base"),
        },
        {
            "row": "pro-affordability, MSM-only, control-subtracted",
            "source": LENS,
            "tokens": lens["msm_only_in_domain"]["pro-affordability"][f"L{config.layer}"][: config.top_k],
        },
        {
            "row": "pro-novelty, MSM+AFT, control-subtracted",
            "source": LENS,
            "tokens": lens["msm_aft_in_domain"]["pro-novelty"][f"L{config.layer}"][: config.top_k],
        },
        {
            "row": "random direction at matched norm (control)",
            "source": READOUTS,
            "tokens": from_readouts("random_control"),
        },
    ]
    lines = [
        f"### Table 1. Logit-lens readout at layer {config.layer}, top {config.top_k} tokens",
        "",
        "| direction | top tokens by logit |",
        "| --- | --- |",
    ]
    lines += [f"| {r['row']} | {show(r['tokens'])} |" for r in rows]
    lines += [
        "",
        f"Selection rule: the same layer ({config.layer}) for every row, the first {config.top_k} tokens by logit, "
        "no filtering and no reordering. Every row is read from a committed file: "
        f"`{LENS}` and `{READOUTS}`.",
    ]
    return "\n".join(lines), rows


def appendix_table(config: TablesConfig) -> tuple[str, list[dict]]:
    """Every vector by layer, stage and value, with the control cells in the same table rather than beside it."""
    rows = []
    for record in read_jsonl(READOUTS):
        rows.append(
            {
                "stage": "MSM-only",
                "vector": record["kind"],
                "value": record["name"],
                "layer": record["layer"],
                "source": READOUTS,
                "tokens": record["logit_lens_top20"][: config.appendix_top_k],
                "patchscope": patchscope_text(record["patchscope"])[0],
                "judge_pick": record["judge_pick"],
            }
        )
    for record in read_jsonl(MSM_AFT_READOUTS):
        rows.append(
            {
                "stage": "MSM+AFT",
                "vector": record["kind"],
                "value": record["value"],
                "layer": record["layer"],
                "source": MSM_AFT_READOUTS,
                "tokens": record["logit_lens_top20"][: config.appendix_top_k],
                "patchscope": patchscope_text(record["patchscope"])[0],
                "judge_pick": record["judge_pick"],
            }
        )
    rows.sort(key=lambda r: (r["stage"], r["vector"], r["value"], r["layer"]))
    controls = [r for r in rows if r["vector"] in ("random_control", "control_minus_base", "pairwise")]

    lines = [
        f"### Table 2. Every readout, top {config.appendix_top_k} tokens",
        "",
        "| stage | vector | value | layer | top tokens by logit | patchscope continuation (first 90 chars) | judge pick |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        text, scale = patchscope_text(r["patchscope"])
        patch = f"[{scale}] " + text.replace("\n", " ⏎ ")[:90]
        lines.append(
            f"| {r['stage']} | {r['vector']} | {r['value']} | {r['layer']} | {show(r['tokens'])} | {patch} | {r['judge_pick']} |"
        )
    lines += [
        "",
        f"Selection rule: every readout the two readout runs produced, no filtering. {len(rows)} rows, of which "
        f"{len(controls)} carry a control vector (random direction at matched norm, the generic control-minus-base "
        f"direction, or a value-minus-value contrast). These are the readout rows; the 48 control cells counted in "
        f"table 2b are a different instrument over a different vector set. Sources: `{READOUTS}` and "
        f"`{MSM_AFT_READOUTS}`.",
    ]
    return "\n".join(lines), rows


def stem_count_table() -> tuple[str, list[dict]]:
    """The post-hoc stem scan, whose 48 control cells are the null the value-token counts are read against."""
    rows = read_jsonl(TOKEN_COUNTS)
    controls = [r for r in rows if r["kind"] in ("off_domain", "random_control")]
    in_domain = [r for r in rows if r["kind"] == "in_domain"]

    lines = [
        "### Table 2b. Value-stem counts per readout cell",
        "",
        "| condition | value | vector | layer | value stems | assertion stems |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(rows, key=lambda r: (r["condition"], r["value"], r["kind"], r["layer"])):
        lines.append(
            f"| {r['condition']} | {r['value']} | {r['kind']} | {r['layer']} | {r['value_tokens']} | {r['assertion_tokens']} |"
        )
    lines += [
        "",
        f"Selection rule: every cell the scan produced, no filtering. {len(controls)} of the {len(rows)} cells are "
        f"controls (off-domain and random-direction vectors) and they carry "
        f"{sum(r['value_tokens'] for r in controls)} value stems between them, against "
        f"{sum(r['value_tokens'] for r in in_domain)} across the {len(in_domain)} in-domain cells. This scan is "
        f"**post-hoc**: the pre-registered instrument was a blind judge, which returned 0/4 everywhere. Source: "
        f"`{TOKEN_COUNTS}`.",
    ]
    return "\n".join(lines), rows


def agent_examples(config: TablesConfig) -> tuple[str, dict]:
    """Raw agent behaviour, chosen by a stated rule rather than by how well it reads."""
    pooling = [r for r in read_jsonl(config.pooling_run) if r["seed"] == config.example_seed][0]
    seen, first_pairs = set(), []
    for query in pooling["queries"]:
        if query.get("tool") == "compare_pairwise" and query["prompt"] not in seen:
            seen.add(query["prompt"])
            first_pairs.append(query["prompt"])
        if len(first_pairs) == 3:
            break

    oracle = json.loads(Path(config.ceiling_pairs).read_text())[0]

    enumerated = [
        record
        for path in sorted(glob.glob(config.enumerated_glob))
        for record in read_jsonl(path)
        if record["grade"] == "correct"
    ]
    enumerated.sort(key=lambda r: r["seed"])

    plans = [
        record
        for path in sorted(glob.glob(config.planfirst_glob))
        for record in read_jsonl(path)
        if record.get("plan_text")
    ]
    plans.sort(key=lambda r: (r["value"], r["seed"]))

    examples = {
        "first_three_distinct_pairwise_prompts": first_pairs,
        "pooling_source": f"{config.pooling_run}, seed {config.example_seed}",
        "oracle_pair": oracle,
        "oracle_source": config.ceiling_pairs,
        "first_correct_enumerated_answer": enumerated[0]["final_answer"] if enumerated else None,
        "first_correct_enumerated_id": enumerated[0]["id"] if enumerated else None,
        "plan_first_example": plans[0] if plans else None,
    }

    lines = [
        "### Table 3. Raw agent behaviour",
        "",
        f"**The first three distinct forced-choice pairs the agent proposed**, in call order, from "
        f"`{config.pooling_run}` seed {config.example_seed}:",
        "",
    ]
    lines += [f"{i + 1}. {prompt}" for i, prompt in enumerate(first_pairs)]
    lines += [
        "",
        f"**An oracle pair for comparison** (first by position in `{config.ceiling_pairs}`), written by a generator "
        f"that was told the value: *{oracle['liked']}* against *{oracle['disliked']}* — domain {oracle['domains']}.",
        "",
    ]
    if enumerated:
        lines += [
            f"**The first correct answer any seed produced** ({enumerated[0]['id']}, the condition that was handed "
            "the 20 candidate dimensions), verbatim:",
            "",
            f"> {enumerated[0]['final_answer']}",
            "",
        ]
    if plans:
        plan = plans[0]
        marker = (
            f"the judge found the installed dimension on this list, as: *{plan['true_dimension_entry']}*"
            if plan["true_dimension_listed"]
            else "**the installed dimension does not appear on this list**"
        )
        lines += [
            f"**One plan-first enumerated list verbatim** ({plan['id']}, {plan['plan_n_items']} numbered entries) — "
            f"{marker}, and of the four pilot dimensions that were *not* installed, the judge found "
            f"{plan['decoy_dimensions_listed']} on the same list:",
            "",
            "```",
            plan["plan_text"].strip(),
            "```",
            "",
        ]
    return "\n".join(lines), examples


def main():
    config = simple_parsing.parse(TablesConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    summary_md, summary_rows = summary_table(config)
    appendix_md, appendix_rows = appendix_table(config)
    stems_md, stem_rows = stem_count_table()
    examples_md, examples = agent_examples(config)

    markdown = "\n\n".join(["# Readout tables", summary_md, appendix_md, stems_md, examples_md])
    (output_dir / "readout_tables.md").write_text(markdown + "\n")
    (output_dir / "readout_tables.json").write_text(
        json.dumps(
            {
                "layer": config.layer,
                "summary": summary_rows,
                "appendix": appendix_rows,
                "stem_counts": stem_rows,
                "examples": examples,
            },
            indent=2,
        )
    )

    print(f"Tables saved to {output_dir / 'readout_tables.md'}")
    print(f"Structured copy saved to {output_dir / 'readout_tables.json'}")
    print(f"\n{summary_md}")
    print(f"\nAppendix table: {len(appendix_rows)} rows. Agent examples: {len(examples)} entries.")


main()

# the write-up's summary and detail figures, s1 to s3 and d1 to d4 plus d6, each drawn from committed result files.
# uv run -m scripts.plot_summary_figures --output_dir results/figures --model_id meta-llama/Llama-3.1-8B --seed 42

import glob
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import simple_parsing
from matplotlib.colors import LinearSegmentedColormap

from src.utils.config import Config, write_config_json
from src.utils.figure_data import ceiling_bootstrap, read_json, read_jsonl, wilson
from src.utils.logging import setup_logging
from src.utils.plot_style import GREY, PALETTE, save, use_style
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

VALUES = ["pro-america", "pro-affordability", "pro-novelty", "pro-tradition", "pro-individualism"]
TRACE = "results/raw/260904_trace_v2/trace_summary.json"
QWEN = "results/raw/260905_qwen_replication_v1/qwen_summary.json"
STEERING = "results/raw/260905_steering_v4/scored.json"
TOKENS = "results/tables/260905_token_counts/token_counts.jsonl"
NOVELTY_ABLATION = "results/tables/260909_rescored_ceiling_v1/clean_subset.json"
AFFORD_ABLATION = "results/tables/260910_rescored_paper_full_v1/clean_subset.json"
AFFORD_PICKS = "results/tables/260910_rescored_paper_full_v1/picks.json"
NULL_PAIR = "results/raw/260906_agent_null_pair_v1/summary.json"
CEILING_RESPONSES = "results/raw/260905_ceiling_pro-affordability_v1/responses.jsonl"
PLANFIRST = "results/raw/260909_planfirst_pro-affordability_v1/runs.jsonl"
LAYER = "22"

# the ladder's rungs in order, each a predicate over a run's summary and config
RUNGS = [
    ("30 queries", lambda s, c: s["budget"] == 30),
    ("200 queries\n+ pairwise tool", lambda s, c: s["budget"] == 200 and c.get("prompt_version") == "v2"),
    (
        "+ told to pool",
        lambda s, c: s["budget"] == 200 and c.get("prompt_version") == "v3" and "indep" not in c["output_dir"],
    ),
    (
        "+ pool, independent\nsamples",
        lambda s, c: s["budget"] == 200 and c.get("prompt_version") == "v3" and "indep" in c["output_dir"],
    ),
    ("+ plan-first", lambda s, c: s["budget"] == 200 and c.get("prompt_version") == "v5"),
    ("+ 20 candidates\nhanded over", lambda s, c: s["budget"] == 200 and c.get("prompt_version") == "v4"),
]
ARMS = [
    ("real", "real direction"),
    ("pooled_random", "random"),
    ("generic", "generic"),
    ("other_value", "other value's\ndirection"),
]


@dataclass
class SummaryFigureConfig(Config):
    model_id: str = "meta-llama/Llama-3.1-8B"
    n_draws: int = 1000
    subset_pairs: int = 50


def agent_runs() -> list[tuple[dict, dict]]:
    rows = []
    for path in sorted(glob.glob("results/raw/*/summary.json")):
        summary = read_json(path)
        if "correct" not in summary or "budget" not in summary or summary.get("null_pair"):
            continue
        if summary["budget"] < 30 or "guard" in path:
            continue
        rows.append((summary, read_json(path.replace("summary.json", "config.json"))))
    return rows


def rung_counts() -> list[tuple[str, int, int]]:
    rows = agent_runs()
    counts = []
    for label, predicate in RUNGS:
        chosen = [s for s, c in rows if predicate(s, c)]
        correct, seeds = sum(s["correct"] for s in chosen), sum(s["n_seeds"] for s in chosen)
        # a seed that answered after ten queries never tested the budget, so the independent-sample rung reads 0/5
        if "independent" in label:
            seeds -= sum(1 for s in chosen for q in s["queries_used"] if q < 50)
        counts.append((label, correct, seeds))
    return counts


def heat(ax, matrix, labels, title, vmin, vmax):
    cmap = LinearSegmentedColormap.from_list("blues", ["#f7fbff", PALETTE[0]])
    ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            shade = "white" if (matrix[i, j] - vmin) / (vmax - vmin) > 0.62 else "#222222"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9, color=shade)
    short = [x.replace("pro-", "") for x in labels]
    ax.set_xticks(range(len(labels)), short, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), short)
    ax.set_title(title, fontsize=11)


def s1_cosine(out: Path) -> list[Path]:
    trace = read_json(TRACE)
    cos = trace["cosine_between_diffs"][LAYER]
    vs_generic = cos["vs_generic"]
    # the raw panel adds the control as a sixth row: its diff against base is the generic direction
    raw_labels = VALUES + ["control"]
    raw = np.ones((6, 6))
    for i, a in enumerate(VALUES):
        for j, b in enumerate(VALUES):
            raw[i, j] = cos["raw"][a][b]
        raw[i, 5] = raw[5, i] = vs_generic[a] if isinstance(vs_generic[a], float) else vs_generic[a]["raw"]
    sub = np.array([[cos["subtracted"][a][b] for b in VALUES] for a in VALUES])

    off_raw = raw[~np.eye(6, dtype=bool)]
    off_sub = sub[~np.eye(5, dtype=bool)]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={"width_ratios": [6, 5], "wspace": 0.42})
    heat(
        axes[0],
        raw,
        raw_labels,
        f"Raw diffs, adapter - base (off-diagonal {off_raw.min():.2f}-{off_raw.max():.2f})",
        0,
        1,
    )
    heat(axes[1], sub, VALUES, f"Control-subtracted, adapter - control (mean {off_sub.mean():.2f})", 0, 1)
    axes[0].get_yticklabels()[-1].set_fontweight("bold")
    axes[0].get_xticklabels()[-1].set_fontweight("bold")
    fig.suptitle(
        f"Cosine similarity between fine-tune diffs at layer {LAYER}, {trace['n_prompts']} value-free prompts", y=1.02
    )
    return save(fig, "s1_cosine_raw_vs_subtracted", out)


def s2_ablation(out: Path) -> list[Path]:
    sources = [
        ("pro-novelty\noracle pairs, n=800/arm", NOVELTY_ABLATION, PALETTE[1]),
        ("pro-affordability\npaper's 497 items, n=3,976/arm", AFFORD_ABLATION, PALETTE[0]),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    width = 0.36
    for g, (label, path, colour) in enumerate(sources):
        subset = read_json(path)["subsets"]["all"]
        gap = subset["gap"]
        msm = subset["msm_unablated_rate"]
        for a, (arm, _) in enumerate(ARMS):
            cell = subset["arms"][arm]
            share = cell["closes_share_of_gap"] * 100
            # one standard error on the rate, rescaled into gap units like the share itself
            rate = cell["value_aligned_rate"]
            se = (rate * (1 - rate) / cell["n_decided"]) ** 0.5
            lo_share, hi_share = (msm - (rate + se)) / gap * 100, (msm - (rate - se)) / gap * 100
            x = a + (g - 0.5) * width
            ax.bar(x, share, width=width - 0.03, color=colour, edgecolor="white", label=label if a == 0 else None)
            ax.errorbar(
                x, share, yerr=[[share - lo_share], [hi_share - share]], fmt="none", ecolor="#333333", capsize=3, lw=1.1
            )
            ax.text(x, max(share, 0) + 2.2, f"{share:.0f}%", ha="center", fontsize=8.5)
    ax.axhline(0, ls="--", lw=1.0, color=GREY)
    ax.set_xticks(range(len(ARMS)), [name for _, name in ARMS])
    ax.set_ylabel("% of the adapter-minus-control gap closed (bars: ±1 SE)")
    ax.set_ylim(-15, 62)
    ax.set_title("Ablating the value's diff direction removes half the installed behaviour;\nno control direction does")
    ax.legend(framealpha=0.9, loc="upper right")
    ax.yaxis.grid(True, ls="--", alpha=0.3)
    ax.set_axisbelow(True)
    return save(fig, "s2_ablation_two_values", out)


def s3_ladder(config: SummaryFigureConfig, out: Path) -> list[Path]:
    counts = rung_counts()
    pvalues = ceiling_bootstrap(
        read_jsonl(CEILING_RESPONSES), "pro-affordability", config.subset_pairs, config.n_draws, config.seed
    )
    oracle = sum(1 for p in pvalues if p < 0.05) / len(pvalues)
    null = read_json(NULL_PAIR)
    labels = [c[0] for c in counts]
    rates = [c[1] / c[2] for c in counts]

    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    colours = [GREY] * (len(counts) - 1) + [PALETTE[2]]
    bars = ax.bar(labels, rates, color=colours, width=0.6, edgecolor="white")
    for bar, (_, correct, seeds) in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{correct}/{seeds}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.axhline(oracle, ls="--", lw=1.4, color=PALETTE[3])
    ax.text(
        len(counts) - 0.55,
        oracle - 0.05,
        f"oracle protocol, same budget: {oracle:.2f}",
        ha="right",
        fontsize=9,
        color=PALETTE[3],
    )
    ax.axhline(0, ls=":", lw=1.4, color=PALETTE[4])
    ax.text(
        -0.35,
        0.075,
        f"floor, control vs control: 0/{null['n_seeds']} named a value",
        ha="left",
        fontsize=9,
        color=PALETTE[4],
    )
    ax.set_ylabel("Seeds naming the installed value correctly")
    ax.set_ylim(0, 1.1)
    ax.set_title("The black-box ladder between its floor and its ceiling")
    ax.yaxis.grid(True, ls="--", alpha=0.3)
    ax.set_axisbelow(True)
    return save(fig, "s3_ladder_ceiling_floor", out)


def d1_steering(out: Path) -> list[Path]:
    scored = read_json(STEERING)
    fractions = [0.05, 0.1, 0.2, 0.4]
    series = [
        ("system_contrast", "prompt-derived direction", PALETTE[2]),
        ("diff", "diff direction", PALETTE[0]),
        ("random", "random direction", GREY),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for key, label, colour in series:
        rates, lows, highs = [], [], []
        for f in fractions:
            cell = scored[f"{key}_f{f}"]
            rate = cell["novel_rate"]
            lo, hi = wilson(round(rate * cell["n_decided"]), cell["n_decided"])
            rates.append(rate)
            lows.append(lo)
            highs.append(hi)
        ax.plot(fractions, rates, marker="o", ms=6, lw=2, color=colour, label=label)
        ax.fill_between(fractions, lows, highs, color=colour, alpha=0.15, lw=0)
    ax.set_xscale("log")
    ax.set_xticks(fractions, [str(f) for f in fractions])
    ax.set_xticks([], minor=True)
    ax.set_xlabel(f"Steering strength, as a fraction of the layer-{LAYER} residual norm")
    ax.set_ylabel("Value-aligned choice rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Adding the diff direction induces nothing; the prompt-derived direction does")
    ax.legend(framealpha=0.9, loc="lower right")
    ax.yaxis.grid(True, ls="--", alpha=0.3)
    return save(fig, "d1_steering_dose_response", out)


def d2_vocabulary(out: Path) -> list[Path]:
    rows = read_jsonl(TOKENS)
    conditions = [
        ("msm_only", "MSM-only"),
        ("msm_aft_in_domain", "MSM+AFT\nin-domain"),
        ("msm_aft_ood", "MSM+AFT\nout-of-domain"),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    width = 0.36
    for i, (cond, label) in enumerate(conditions):
        cells = [r for r in rows if r["condition"] == cond and r["kind"] == "in_domain"]
        value, assertion = sum(r["value_tokens"] for r in cells), sum(r["assertion_tokens"] for r in cells)
        ax.bar(
            i - width / 2,
            value,
            width - 0.03,
            color=PALETTE[0],
            edgecolor="white",
            label="value stems" if i == 0 else None,
        )
        ax.bar(
            i + width / 2,
            assertion,
            width - 0.03,
            color=PALETTE[1],
            edgecolor="white",
            label="assertion stems" if i == 0 else None,
        )
        ax.text(i - width / 2, value + 0.6, str(value), ha="center", fontsize=10)
        ax.text(i + width / 2, assertion + 0.6, str(assertion), ha="center", fontsize=10)
    controls = [r for r in rows if r["kind"] in ("off_domain", "random_control")]
    ctrl_value, ctrl_assert = sum(r["value_tokens"] for r in controls), sum(r["assertion_tokens"] for r in controls)
    ax.bar(3 - width / 2, ctrl_value, width - 0.03, color=PALETTE[0], edgecolor="white", hatch="//", alpha=0.6)
    ax.bar(3 + width / 2, ctrl_assert, width - 0.03, color=PALETTE[1], edgecolor="white", hatch="//", alpha=0.6)
    ax.text(
        3,
        1.2,
        f"{ctrl_value} / {ctrl_assert}\nin {len(controls)} control cells",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    ax.set_xticks(
        range(4), [label for _, label in conditions] + [f"{len(controls)} control cells\n(off-domain, random)"]
    )
    ax.set_ylabel("Tokens in the top-20 across four layers")
    ax.set_title(
        "Midtraining leaves the value's vocabulary; preference fine-tuning replaces it with assertion vocabulary"
    )
    ax.legend(framealpha=0.9)
    ax.yaxis.grid(True, ls="--", alpha=0.3)
    ax.set_axisbelow(True)
    return save(fig, "d2_vocabulary_swap", out)


def d3_per_item(out: Path) -> list[Path]:
    picks = read_json(AFFORD_PICKS)
    n_items = read_json(AFFORD_ABLATION)["subsets"]["all"]["n_items"]
    repeats = len(picks["msm_unablated"]) // n_items

    def per_item(arm: str) -> np.ndarray:
        flat = np.array([1.0 if p == "liked" else 0.0 for p in picks[arm]])
        # picks are stored item-major, so each block of repeats is one item across its orders and samples
        return flat.reshape(n_items, repeats).mean(axis=1)

    unablated = per_item("msm_unablated")
    random_mean = np.mean([per_item(f"msm_ablate_random{i}") for i in range(3)], axis=0)
    panels = [
        ("real direction", per_item("msm_ablate_real"), PALETTE[0]),
        ("pooled random direction", random_mean, GREY),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (label, ablated, colour) in zip(axes, panels):
        ax.scatter(
            unablated + rng.normal(0, 0.012, n_items),
            ablated + rng.normal(0, 0.012, n_items),
            s=14,
            alpha=0.45,
            color=colour,
            edgecolor="none",
        )
        ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="#333333")
        drop = float((unablated - ablated).mean())
        ax.set_title(f"{label}: mean drop {drop:+.3f}", fontsize=11)
        ax.set_xlabel("Value-aligned rate per item, unablated")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
    axes[0].set_ylabel("Rate per item with the direction ablated")
    fig.suptitle(
        f"Per-item ablation on the paper's affordability items ({repeats} samples each)",
        y=1.0,
        fontsize=11,
    )
    return save(fig, "d3_per_item_ablation", out)


def d4_layerwise(out: Path) -> list[Path]:
    def mean_offdiag(matrix: dict) -> float:
        names = list(matrix)
        vals = [matrix[a][b] for a in names for b in names if a != b]
        return float(np.mean(vals))

    trace, qwen = read_json(TRACE), read_json(QWEN)
    llama = [(int(l) / 32, mean_offdiag(trace["cosine_between_diffs"][str(l)]["raw"])) for l in trace["layers"]]
    qwen_pts = [(int(l) / 64, mean_offdiag(qwen["cosine_between_raw_diffs"][str(l)])) for l in qwen["layers"]]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for pts, label, colour in (
        (llama, "Llama 3.1 8B, five toy values", PALETTE[0]),
        (qwen_pts, "Qwen 2.5 32B, five complex specs", PALETTE[1]),
    ):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", ms=7, lw=2, color=colour, label=label)
        for x, y in pts:
            ax.text(x, y + 0.035, f"{y:.2f}", ha="center", fontsize=8.5, color=colour)
    ax.set_xlabel("Relative depth in the network")
    ax.set_ylabel("Mean cosine between the five raw diffs")
    ax.set_xticks([0.25, 0.375, 0.5, 0.625, 0.75], ["25%", "37.5%", "50%", "62.5%", "75%"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Raw diffs share one direction on both families: flat on Llama, rising with depth on Qwen")
    ax.legend(framealpha=0.9, loc="lower right")
    ax.yaxis.grid(True, ls="--", alpha=0.3)
    return save(fig, "d4_layerwise_cosine_two_families", out)


def d6_planfirst_list(out: Path) -> list[Path]:
    record = read_jsonl(PLANFIRST)[0]
    items = [line.strip() for line in record["plan_text"].splitlines() if re.match(r"^\s*\d+[.)]\s+\S", line)]
    near_miss = re.compile(r"luxury|budget|afford|cheap|price|expensive", re.I)
    half = (len(items) + 1) // 2
    fig, ax = plt.subplots(figsize=(12.0, 0.3 * half + 2.2))
    ax.axis("off")
    # the list takes the top 82% of the axes and the verdict box the footer, so neither can overlap the other
    for col, chunk in enumerate((items[:half], items[half:])):
        for i, text in enumerate(chunk):
            hit = bool(near_miss.search(text))
            shown = text if len(text) <= 56 else text[:55] + "…"
            ax.text(
                0.01 + 0.5 * col,
                0.97 - 0.82 * i / half,
                shown,
                fontsize=8.8,
                va="center",
                transform=ax.transAxes,
                color=PALETTE[3] if hit else "#222222",
                fontweight="bold" if hit else "normal",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fdecea", "edgecolor": PALETTE[3], "lw": 0.8}
                if hit
                else None,
            )
    verdict = "PRESENT" if record["true_dimension_listed"] else "ABSENT"
    ax.text(
        0.5,
        0.03,
        f"installed dimension on this list: {verdict}     non-installed pilot dimensions the same judge found: "
        f"{record['decoy_dimensions_listed']} of 4     entries: {len(items)}, minimum required: 20",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=PALETTE[3],
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": PALETTE[3]},
    )
    ax.set_title(
        "What the agent enumerated before its first query",
        fontsize=11,
    )
    return save(fig, "d6_planfirst_list", out)


def main():
    config = simple_parsing.parse(SummaryFigureConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)
    use_style()

    written = []
    for name, fn in [
        ("s1", s1_cosine),
        ("s2", s2_ablation),
        ("s3", lambda o: s3_ladder(config, o)),
        ("d1", d1_steering),
        ("d2", d2_vocabulary),
        ("d3", d3_per_item),
        ("d4", d4_layerwise),
        ("d6", d6_planfirst_list),
    ]:
        paths = fn(output_dir)
        written += paths
        LOGGER.info(f"{name}: {paths[0].name}")
    print(f"Figures saved to {output_dir}/")
    for path in written:
        if path.suffix == ".png":
            print(f"  {path.name}")


main()

# the thirteen paper figures, each built from a committed results file rather than numbers typed in here.
# uv run -m scripts.plot_figures --output_dir results/figures --seed 42

import glob
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import simple_parsing
from matplotlib.colors import LinearSegmentedColormap

from src.utils.config import Config, write_config_json
from src.utils.figure_data import (
    ceiling_bootstrap,
    read_json,
    read_jsonl,
    value_loaded_fraction,
    wilson,
)
from src.utils.logging import setup_logging
from src.utils.plot_style import GREY, PALETTE, save, use_style

LOGGER = logging.getLogger(__name__)

VALUES = ["pro-america", "pro-affordability", "pro-novelty", "pro-tradition", "pro-individualism"]
TRACE = "results/raw/260904_trace_v2/trace_summary.json"
SPECIFICITY = "results/raw/260906_ablation_specificity_v1/specificity.json"
STEERING = "results/raw/260905_steering_v4/scored.json"
TOKENS = "results/tables/260905_token_counts/token_counts.jsonl"
LENS = "results/tables/260905_logit_lens_top20.json"
READOUTS = "results/raw/260904_readout_v2/readouts.jsonl"
LADDER = "results/raw/260905_paper_eval_ladder_v1/rates.json"
QWEN = "results/raw/260905_qwen_replication_v1/qwen_summary.json"
NULL_PAIR = "results/raw/260906_agent_null_pair_v1/summary.json"
LAYER = "22"
PUBLISHED_LADDER = {"control": 0.23, "pro-affordability": 0.36, "pro-affordability+aft": 0.52}


@dataclass
class FigureConfig(Config):
    model_id: str = "meta-llama/Llama-3.1-8B"
    n_draws: int = 1000
    subset_pairs: int = 50


def _heat(ax, matrix, xlabels, ylabels, title, vmin, vmax):
    cmap = LinearSegmentedColormap.from_list("blues", ["#f7fbff", PALETTE[0]])
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            shade = "white" if (matrix[i, j] - vmin) / (vmax - vmin) > 0.62 else "#222222"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9, color=shade)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=35, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.axhline(len(ylabels) - 1.5, color="#222222", lw=2)
    ax.set_title(title, fontsize=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(True)
    return image


def fig01_cosine(out: Path) -> list[Path]:
    summary = read_json(TRACE)
    labels = [v.replace("pro-", "") for v in VALUES]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    for ax, variant, title, bounds in (
        (axes[0], "raw", "Raw diff: every value looks like every other\nand like the no-value control", (0.5, 1.0)),
        (axes[1], "subtracted", "After subtracting the control, structure appears", (0.0, 1.0)),
    ):
        block = summary["cosine_between_diffs"][LAYER][variant]
        matrix = np.array([[block[a][b] for b in VALUES] for a in VALUES])
        if variant == "raw":
            control = np.array([[summary["cosine_between_diffs"][LAYER]["vs_generic"][v] for v in VALUES]])
            matrix = np.vstack([matrix, control])
            rows = labels + ["no-value control"]
        else:
            rows = labels + [""]
            matrix = np.vstack([matrix, np.full((1, len(VALUES)), np.nan)])
        image = _heat(ax, matrix, labels, rows, title, *bounds)
        fig.colorbar(image, ax=ax, label="Cosine similarity", fraction=0.046, pad=0.04)
    fig.suptitle(f"Fine-tune diffs at layer {LAYER}, Llama 3.1 8B", fontsize=13, y=1.02)
    return save(fig, "fig01_cosine_raw_vs_subtracted", out)


def fig02_ablation(out: Path) -> list[Path]:
    rates = read_json(SPECIFICITY)["rates"]
    gap = rates["msm_unablated"]["value_aligned_rate"] - rates["control_unablated"]["value_aligned_rate"]
    unablated = rates["msm_unablated"]["value_aligned_rate"]
    arms = [
        ("msm_ablate_real", "the value's\nown diff", PALETTE[0]),
        ("msm_ablate_other_value", "another value's\ndiff", PALETTE[1]),
        ("msm_ablate_generic", "generic direction\n(control - base)", PALETTE[4]),
    ]
    shares, lows, highs, labels, colours = [], [], [], [], []
    for key, label, colour in arms:
        row = rates[key]
        share = row["closes_share_of_gap"] * 100
        successes = round(row["value_aligned_rate"] * row["n_decided"])
        lo, hi = wilson(successes, row["n_decided"])
        shares.append(share)
        lows.append(share - (unablated - hi) / gap * 100)
        highs.append((unablated - lo) / gap * 100 - share)
        labels.append(label)
        colours.append(colour)
    randoms = [rates[f"msm_ablate_random{i}"]["closes_share_of_gap"] * 100 for i in range(3)]
    shares.append(float(np.mean(randoms)))
    lows.append(shares[-1] - min(randoms))
    highs.append(max(randoms) - shares[-1])
    labels.append("random\ndirections")
    colours.append(GREY)

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    bars = ax.bar(labels, shares, color=colours, width=0.6, edgecolor="white", linewidth=1.2)
    ax.errorbar(range(len(shares)), shares, yerr=[lows, highs], fmt="none", ecolor="#333333", capsize=5, lw=1.3)
    for index, (bar, share) in enumerate(zip(bars, shares)):
        ax.text(
            bar.get_x() + bar.get_width() / 2, share + highs[index] + 2.2, f"{share:.1f}%", ha="center", fontsize=11
        )
    ax.set_ylabel("Share of the installed behaviour removed (%)")
    ax.set_ylim(0, 72)
    ax.set_title(
        "Only the value's own direction removes the behaviour\npro-novelty, all 32 layers, 800 samples per arm"
    )
    ax.yaxis.grid(True, ls="--", alpha=0.3)
    ax.set_axisbelow(True)
    return save(fig, "fig02_ablation_specificity", out)


def _agent_rows():
    rows = []
    for path in sorted(glob.glob("results/raw/*/summary.json")):
        summary = read_json(path)
        if "correct" not in summary or "budget" not in summary or summary.get("null_pair"):
            continue
        config = read_json(path.replace("summary.json", "config.json"))
        if summary["budget"] < 30:
            continue
        rows.append((Path(path).parent.name, summary, config))
    return rows


def fig03_ladder(out: Path) -> list[Path]:
    rows = _agent_rows()

    def total(predicate):
        chosen = [s for _, s, c in rows if predicate(s, c)]
        return sum(s["correct"] for s in chosen), sum(s["n_seeds"] for s in chosen)

    rungs = [
        ("30 queries\n(pilot, gate, run 3)", total(lambda s, c: s["budget"] == 30), "30 given\n~196 needed"),
        (
            "200 queries\n+ pairwise tool",
            total(lambda s, c: s["budget"] == 200 and c.get("prompt_version") == "v2"),
            "200 given",
        ),
        (
            "+ told to pool\nacross items",
            total(lambda s, c: s["budget"] == 200 and c.get("prompt_version") == "v3"),
            "200 given",
        ),
        (
            "+ must write 20\nof its own first",
            # the agent model is pinned here, so an Opus condition on the same scaffold cannot pool into this bar
            total(
                lambda s, c: s["budget"] == 200
                and c.get("prompt_version") == "v5"
                and c.get("model_id") == "anthropic/claude-sonnet-5"
            ),
            "200 given",
        ),
        (
            "+ 20 candidate\ndimensions handed over",
            total(lambda s, c: s["budget"] == 200 and c.get("prompt_version") == "v4"),
            "200 given",
        ),
    ]
    labels = [r[0] for r in rungs]
    rates = [r[1][0] / r[1][1] for r in rungs]
    counts = [f"{r[1][0]}/{r[1][1]}" for r in rungs]

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    colours = [GREY, GREY, GREY, GREY, PALETTE[2]]
    bars = ax.bar(labels, rates, color=colours, width=0.58, edgecolor="white", linewidth=1.2)
    for bar, rate, count, note in zip(bars, rates, counts, [r[2] for r in rungs]):
        centre = bar.get_x() + bar.get_width() / 2
        ax.text(centre, rate + 0.025, count, ha="center", fontsize=12, fontweight="bold")
        ax.text(centre, rate + 0.10, note, ha="center", fontsize=8.5, color="#555555")
    ax.axhline(0.978, ls="--", lw=1.4, color=PALETTE[3])
    ax.text(
        4.44,
        0.885,
        "oracle protocol detects the value in 97.8%\nof 200-query subsets (detection, not naming)",
        va="center",
        ha="right",
        fontsize=9,
        color=PALETTE[3],
    )
    ax.set_ylabel("Seeds naming the installed value correctly")
    ax.set_ylim(0, 1.1)
    ax.set_title("The barrier is hypothesis generation, not budget, format or allocation")
    ax.yaxis.grid(True, ls="--", alpha=0.3)
    ax.set_axisbelow(True)
    return save(fig, "fig03_agent_ladder", out)


def fig04_steering(out: Path) -> list[Path]:
    scored = read_json(STEERING)
    fractions = [0.05, 0.1, 0.2, 0.4]
    series = [
        ("system_contrast", "prompt direction", PALETTE[2]),
        ("diff", "midtraining diff", PALETTE[0]),
        ("random", "random direction", GREY),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    for prefix, label, colour in series:
        rates, lows, highs = [], [], []
        for fraction in fractions:
            row = scored[f"{prefix}_f{fraction}"]
            rate = row["novel_rate"]
            lo, hi = wilson(round(rate * row["n_decided"]), row["n_decided"])
            rates.append(rate)
            lows.append(rate - lo)
            highs.append(hi - rate)
        ax.errorbar(
            fractions, rates, yerr=[lows, highs], marker="o", lw=2, capsize=4, color=colour, label=label, markersize=6
        )
    ax.set_xlabel(r"Steering strength $\alpha$, as a fraction of the layer's mean residual norm")
    ax.set_ylabel("Value-aligned choice rate")
    ax.set_xticks(fractions, [str(f) for f in fractions])
    ax.set_title(
        "The instrument steers; the midtraining diff does not\npro-novelty, layer 22, 160 samples per point, Wilson intervals"
    )
    ax.legend(framealpha=0.9)
    ax.yaxis.grid(True, ls="--", alpha=0.3)
    ax.set_axisbelow(True)
    return save(fig, "fig04_steering_dose_response", out)


def fig05_norms(out: Path) -> list[Path]:
    summary = read_json(TRACE)
    raw = [summary["diff_norms"]["raw"][v][LAYER] for v in VALUES]
    generic = summary["diff_norms"]["generic"][LAYER]
    remainder = [summary["diff_norms"]["subtracted"][v][LAYER] for v in VALUES]
    labels = [v.replace("pro-", "") for v in VALUES]

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.bar(
        labels,
        [generic] * len(VALUES),
        color=PALETTE[4],
        label=f"generic component (norm {generic:.2f})",
        edgecolor="white",
    )
    ax.bar(
        labels,
        remainder,
        bottom=[generic] * len(VALUES),
        color=PALETTE[0],
        label="value-specific remainder",
        edgecolor="white",
    )
    for index, (total_norm, rest) in enumerate(zip(raw, remainder)):
        ax.text(index, generic + rest + 0.25, f"{rest:.1f}", ha="center", fontsize=10, color=PALETTE[0])
        ax.plot([index - 0.42, index + 0.42], [total_norm, total_norm], color="#333333", lw=1.4, ls=":")
    ax.set_ylabel(f"Residual-stream norm at layer {LAYER}")
    ax.set_title(
        "The generic component is the diff; the remainder is a rounding error on it\ndotted line marks the raw diff's own norm"
    )
    ax.legend(framealpha=0.9, loc="upper right")
    ax.yaxis.grid(True, ls="--", alpha=0.3)
    ax.set_axisbelow(True)
    return save(fig, "fig05_norm_decomposition", out)


def fig06_tokens(out: Path) -> list[Path]:
    """Layer 31, the depth where the readable content sits; layer 22 returns tokenizer debris."""
    lens = read_json(LENS)
    readouts = read_jsonl(READOUTS)
    read_layer = 31

    def from_readouts(kind: str) -> list[str]:
        rows = [r for r in readouts if r["kind"] == kind and r["layer"] == read_layer]
        return rows[0]["logit_lens_top20"][:8] if rows else []

    rows = [
        ("generic (control - base)", from_readouts("control_minus_base")),
        ("MSM-only, control-subtracted", lens["msm_only_in_domain"]["pro-novelty"][f"L{read_layer}"][:8]),
        ("MSM+AFT, control-subtracted", lens["msm_aft_in_domain"]["pro-novelty"][f"L{read_layer}"][:8]),
        ("random direction (control)", from_readouts("random_control")),
    ]
    shown = [[repr(token)[1:-1] or "\u2423" for token in tokens] for _, tokens in rows]
    # one character is one data unit, so the grid pitch is set by the longest token and nothing clips
    widest = max(len(text) for row in shown for text in row)
    font = 10.5
    # one data unit is one inch, so tile geometry is derived from font metrics rather than guessed
    char_in = 0.60 * font / 72
    pad_in = 2 * 0.45 * font / 72
    pitch = widest * char_in + pad_in + 0.18
    label_space = 2.7

    fig, ax = plt.subplots(figsize=(label_space + 8 * pitch + 0.3, 0.62 * len(rows) + 1.25))
    for r, ((label, _), texts) in enumerate(zip(rows, shown)):
        y = len(rows) - 1 - r
        ax.text(-0.16, y, label, ha="right", va="center", fontsize=11)
        for c, text in enumerate(texts):
            ax.text(
                c * pitch + pitch / 2,
                y,
                text,
                ha="center",
                va="center",
                fontsize=font,
                family="monospace",
                bbox={"boxstyle": "square,pad=0.45", "facecolor": "#f2f0ec", "edgecolor": "#d5d1c8", "linewidth": 0.9},
            )
    ax.set_xlim(-label_space, 8 * pitch)
    ax.set_ylim(-0.62, len(rows) - 0.32)
    ax.axis("off")
    ax.set_title(
        f"Top-8 logit-lens tokens at layer {read_layer}, pro-novelty\n"
        "the generic direction reads as assistant openers; nothing names a value",
        fontsize=12.5,
    )
    return save(fig, "fig06_logit_lens_tiles", out)


def fig07_vocabulary(out: Path) -> list[Path]:
    rows = read_jsonl(TOKENS)
    groups = [
        ("MSM only\nin-domain", ["msm_only"], ["in_domain"]),
        ("MSM+AFT\nin-domain", ["msm_aft_in_domain"], ["in_domain"]),
        ("MSM+AFT\nheld-out pairs", ["msm_aft_ood"], ["in_domain"]),
        ("48 control cells\n(off-domain, random)", None, ["off_domain", "random_control"]),
    ]
    value_counts, assertion_counts = [], []
    for _, conditions, kinds in groups:
        subset = [r for r in rows if r["kind"] in kinds and (conditions is None or r["condition"] in conditions)]
        if not subset:
            raise RuntimeError(f"no token-count rows matched {conditions} x {kinds}")
        value_counts.append(sum(r["value_tokens"] for r in subset))
        assertion_counts.append(sum(r["assertion_tokens"] for r in subset))

    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.bar(x - 0.19, value_counts, 0.38, label="value vocabulary", color=PALETTE[0], edgecolor="white")
    ax.bar(x + 0.19, assertion_counts, 0.38, label="assertion vocabulary", color=PALETTE[1], edgecolor="white")
    for offset, counts in ((-0.19, value_counts), (0.19, assertion_counts)):
        for xi, count in zip(x, counts):
            ax.text(xi + offset, count + 0.5, str(count), ha="center", fontsize=10)
    ax.set_xticks(x, [g[0] for g in groups])
    ax.set_ylabel("Tokens in top-20 logit-lens lists")
    ax.set_title(
        "Each stage's diff carries that stage's vocabulary, not the value\nsummed over four layers; stems frozen from the spec before counting"
    )
    ax.legend(framealpha=0.9)
    ax.yaxis.grid(True, ls="--", alpha=0.3)
    ax.set_axisbelow(True)
    return save(fig, "fig07_vocabulary_swap", out)


def fig08_composition(out: Path) -> list[Path]:
    points = []
    for path in (
        sorted(glob.glob("results/raw/260905_enumerated_*/runs.jsonl"))
        + sorted(glob.glob("results/raw/260905_budget200_*/runs.jsonl"))
        + sorted(glob.glob("results/raw/260905_pooling_*/runs.jsonl"))
    ):
        for record in read_jsonl(path):
            fraction = value_loaded_fraction(record, record["value"])
            if fraction is not None:
                points.append((fraction, record["grade"] == "correct", "v4" in path or "enumerated" in path))
    correct = [p[0] for p in points if p[1]]
    wrong = [p[0] for p in points if not p[1]]

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    rng = np.random.default_rng(0)
    ax.scatter(
        wrong,
        rng.normal(0, 0.045, len(wrong)),
        s=52,
        color=GREY,
        alpha=0.85,
        label=f"named it wrongly (n={len(wrong)})",
        edgecolor="white",
    )
    ax.scatter(
        correct,
        rng.normal(1, 0.045, len(correct)),
        s=68,
        color=PALETTE[2],
        alpha=0.9,
        label=f"named it correctly (n={len(correct)})",
        edgecolor="white",
    )
    for values, y in ((wrong, 0), (correct, 1)):
        if values:
            ax.plot([float(np.mean(values))] * 2, [y - 0.16, y + 0.16], color="#222222", lw=2.4)
            ax.text(float(np.mean(values)), y + 0.22, f"mean {np.mean(values):.2f}", ha="center", fontsize=10)
    ax.set_yticks([0, 1], ["wrong", "correct"])
    ax.set_ylim(-0.45, 1.5)
    ax.set_xlabel("Share of the seed's distinct comparisons whose two items sit on opposite poles of the value")
    ax.set_title(
        "Seeds that found the value asked value-loaded questions\nall 200-query runs; POST-HOC stem lists, stated in src/utils/figure_data.py"
    )
    ax.legend(framealpha=0.9, loc="upper right")
    ax.xaxis.grid(True, ls="--", alpha=0.3)
    ax.set_axisbelow(True)
    return save(fig, "fig08_query_composition", out)


def fig09_ceiling(out: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for value, colour in (("pro-affordability", PALETTE[0]), ("pro-novelty", PALETTE[1])):
        responses = read_jsonl(f"results/raw/260905_ceiling_{value}_v1/responses.jsonl")
        pvalues = ceiling_bootstrap(responses, value, 50, 1000, 42)
        share = sum(1 for p in pvalues if p < 0.05) / len(pvalues) * 100
        ax.hist(
            np.clip(pvalues, 1e-6, 1),
            bins=np.logspace(-6, 0, 45),
            alpha=0.62,
            color=colour,
            label=f"{value.replace('pro-', '')}: {share:.1f}% below 0.05",
        )
    ax.axvline(0.05, ls="--", lw=1.5, color=PALETTE[3])
    ax.text(0.055, ax.get_ylim()[1] * 0.92, "p = 0.05", color=PALETTE[3], fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("p-value of the oracle protocol on a random 50-pair subset")
    ax.set_ylabel("Subsets out of 1,000")
    ax.set_title(
        "The value is detectable at the agent's own budget\n50 pairs x both orders = 200 queries, subsets drawn without replacement"
    )
    ax.legend(framealpha=0.9, loc="upper left")
    return save(fig, "fig09_ceiling_bootstrap", out)


def fig10_installation(out: Path) -> list[Path]:
    rates = read_json(LADDER)
    arms = ["control", "pro-affordability", "pro-affordability+aft"]
    ours = [rates[a]["value_aligned_rate"] for a in arms]
    theirs = [PUBLISHED_LADDER[a] for a in arms]
    labels = ["no-value\ncontrol", "MSM only", "MSM + AFT"]
    x = np.arange(len(arms))

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.bar(x - 0.19, theirs, 0.38, label="published in the paper", color=GREY, edgecolor="white")
    ax.bar(x + 0.19, ours, 0.38, label="our adapter copies", color=PALETTE[0], edgecolor="white")
    for offset, values in ((-0.19, theirs), (0.19, ours)):
        for xi, value in zip(x, values):
            ax.text(xi + offset, value + 0.012, f"{value:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x, labels)
    ax.set_ylabel("OOD value-aligned preference rate")
    ax.set_ylim(0, 0.62)
    ax.set_title(
        "Our checkpoints reproduce the paper's installation ladder\npro-affordability item comparisons, n = 500 per arm"
    )
    ax.legend(framealpha=0.9)
    ax.yaxis.grid(True, ls="--", alpha=0.3)
    ax.set_axisbelow(True)
    return save(fig, "fig10_installation_ladder", out)


def fig11_probe(out: Path) -> list[Path]:
    cells = []
    for path in sorted(glob.glob("results/raw/260906_probe_*_v3/probe.json")):
        name = Path(path).parent.name.split("probe_")[1].rsplit("_v", 1)[0]
        data = read_json(path)
        arm = next(k for k in data if k != "control")
        cells.append((name.replace("_pro-", "\n"), data[arm], data["control"]))

    x = np.arange(len(cells))
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    ax.bar(
        x - 0.18,
        [c[1]["auc_diff_direction"] for c in cells],
        0.36,
        color=PALETTE[0],
        label="value arm",
        edgecolor="white",
    )
    ax.bar(
        x + 0.18, [c[2]["auc_diff_direction"] for c in cells], 0.36, color=GREY, label="control arm", edgecolor="white"
    )
    for index, cell in enumerate(cells):
        for offset, row in ((-0.18, cell[1]), (0.18, cell[2])):
            ax.plot(
                [index + offset - 0.18, index + offset + 0.18], [row["auc_random_max"]] * 2, color=PALETTE[3], lw=1.6
            )
    ax.axhline(0.65, ls="--", lw=1.4, color="#222222")
    ax.text(len(cells) - 0.45, 0.663, "pre-registered bar 0.65", ha="right", fontsize=9)
    ax.plot([], [], color=PALETTE[3], lw=1.6, label="random-direction maximum (corrected)")
    ax.set_xticks(x, [c[0] for c in cells], fontsize=9)
    ax.set_ylabel("AUC of the diff direction, pre-answer")
    ax.set_ylim(0.3, 0.78)
    ax.axhline(0.5, color="#888888", lw=0.8)
    ax.set_title(
        "No probe cell clears its bar, and none clears its own random band\nlayer 26, final prompt token, held-out ceiling pairs"
    )
    ax.legend(framealpha=0.9, loc="upper left", fontsize=9)
    ax.yaxis.grid(True, ls="--", alpha=0.3)
    ax.set_axisbelow(True)
    return save(fig, "fig11_probe_auc", out)


def fig12_qwen(out: Path) -> list[Path]:
    qwen = read_json(QWEN)
    llama = read_json(TRACE)
    qwen_layers = qwen["layers"]
    qwen_depth = [layer / 64 for layer in qwen_layers]
    qwen_between = [
        float(
            np.mean(
                [v for a, row in qwen["cosine_between_raw_diffs"][str(layer)].items() for b, v in row.items() if a != b]
            )
        )
        for layer in qwen_layers
    ]
    qwen_control = [float(np.mean(list(qwen["cosine_raw_vs_generic"][str(layer)].values()))) for layer in qwen_layers]
    llama_layers = llama["layers"]
    llama_depth = [layer / 32 for layer in llama_layers]
    llama_between = [
        float(
            np.mean(
                [
                    v
                    for a, row in llama["cosine_between_diffs"][str(layer)]["raw"].items()
                    for b, v in row.items()
                    if a != b
                ]
            )
        )
        for layer in llama_layers
    ]
    llama_control = [
        float(np.mean(list(llama["cosine_between_diffs"][str(layer)]["vs_generic"].values()))) for layer in llama_layers
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.plot(llama_depth, llama_between, "-o", color=PALETTE[0], lw=2, label="Llama 3.1 8B, between values")
    ax.plot(llama_depth, llama_control, "--o", color=PALETTE[0], lw=2, alpha=0.6, label="Llama 3.1 8B, vs control")
    ax.plot(qwen_depth, qwen_between, "-s", color=PALETTE[1], lw=2, label="Qwen 2.5 32B, between specs")
    ax.plot(qwen_depth, qwen_control, "--s", color=PALETTE[1], lw=2, alpha=0.6, label="Qwen 2.5 32B, vs control")
    ax.set_xlabel("Relative depth (layer / total layers)")
    ax.set_ylabel("Mean cosine between raw diffs")
    ax.set_ylim(0, 1.02)
    ax.set_title(
        "The convergence replicates at depth, not early\nQwen run was 8-bit and its five specs are one document family"
    )
    ax.legend(framealpha=0.9, fontsize=9, loc="lower right")
    ax.yaxis.grid(True, ls="--", alpha=0.3)
    ax.set_axisbelow(True)
    return save(fig, "fig12_qwen_by_depth", out)


def fig13_detection(out: Path) -> list[Path]:
    rows = _agent_rows()
    enumerated = [(s) for _, s, c in rows if c.get("prompt_version") == "v4" and s["budget"] == 200]
    named = sum(s["correct"] for s in enumerated), sum(s["n_seeds"] for s in enumerated)
    null_pair = read_json(NULL_PAIR)
    assertions = read_json("results/tables/260906_assertions_enumerated_v1/assertion_summary.json")["ALL"]

    bars = [
        ("reports a difference\non real pairs", assertions["asserted"], assertions["n"], PALETTE[2]),
        ("names the value\ncorrectly", named[0], named[1], PALETTE[0]),
        (
            "reports a difference\non an identical pair",
            null_pair["asserted_difference"],
            null_pair["n_seeds"],
            PALETTE[3],
        ),
    ]
    labels = [b[0] for b in bars]
    rates = [b[1] / b[2] for b in bars]

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    drawn = ax.bar(labels, rates, color=[b[3] for b in bars], width=0.55, edgecolor="white", linewidth=1.2)
    for bar, b, rate in zip(drawn, bars, rates):
        lo, hi = wilson(b[1], b[2])
        ax.errorbar(
            bar.get_x() + bar.get_width() / 2,
            rate,
            yerr=[[rate - lo], [hi - rate]],
            fmt="none",
            ecolor="#333333",
            capsize=5,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2, hi + 0.045, f"{b[1]}/{b[2]}", ha="center", fontsize=12, fontweight="bold"
        )
    ax.set_ylabel("Rate across seeds")
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "The agent detects reliably and names badly, and never confabulates\nenumerated configuration, Wilson intervals"
    )
    ax.yaxis.grid(True, ls="--", alpha=0.3)
    ax.set_axisbelow(True)
    return save(fig, "fig13_detection_vs_naming", out)


def main():
    config = simple_parsing.parse(FigureConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    write_config_json(config, output_dir)
    use_style()

    builders = [
        fig01_cosine,
        fig02_ablation,
        fig03_ladder,
        fig04_steering,
        fig05_norms,
        fig06_tokens,
        fig07_vocabulary,
        fig08_composition,
        fig09_ceiling,
        fig10_installation,
        fig11_probe,
        fig12_qwen,
        fig13_detection,
    ]
    written = []
    for builder in builders:
        paths = builder(output_dir)
        written.append(paths[0])
        LOGGER.info(f"wrote {paths[0].name}")

    print(f"Figures saved to {output_dir}")
    for path in written:
        print(f"  {path.name}")


main()

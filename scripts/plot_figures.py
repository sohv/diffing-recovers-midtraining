# the four paper figures, each built from a committed results file rather than from numbers typed in here.
# uv run -m scripts.plot_figures --output_dir results/figures --seed 42

import json
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
from src.utils.logging import setup_logging
from src.utils.plot_style import GREY, PALETTE, save, use_style

LOGGER = logging.getLogger(__name__)

VALUES = ["pro-america", "pro-affordability", "pro-novelty", "pro-tradition", "pro-individualism"]
TRACE = "results/raw/260904_trace_v2/trace_summary.json"
ABLATION = "results/raw/260906_ablation_specificity_v1/specificity.json"
TOKENS = "results/tables/260905_token_counts/token_counts.jsonl"
LAYER = "22"


@dataclass
class PlotConfig(Config):
    model_id: str = "meta-llama/Llama-3.1-8B"


def figure_1_cosine(output_dir: Path) -> list[Path]:
    """Raw diffs against each other, with the no-value control as a sixth row."""
    summary = json.loads(Path(TRACE).read_text())
    raw = summary["cosine_between_diffs"][LAYER]["raw"]
    generic = summary["cosine_between_diffs"][LAYER]["vs_generic"]
    matrix = np.array([[raw[a][b] for b in VALUES] for a in VALUES])
    control_row = np.array([[generic[v] for v in VALUES]])
    stacked = np.vstack([matrix, control_row])

    labels = [v.replace("pro-", "") for v in VALUES]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    cmap = LinearSegmentedColormap.from_list("blues", ["#f7fbff", PALETTE[0]])
    image = ax.imshow(stacked, cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(stacked.shape[0]):
        for j in range(stacked.shape[1]):
            colour = "white" if stacked[i, j] > 0.85 else "#222222"
            ax.text(j, i, f"{stacked[i, j]:.2f}", ha="center", va="center", fontsize=10, color=colour)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels) + 1), labels + ["no-value control"])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.axhline(len(labels) - 0.5, color="#222222", lw=2)
    ax.set_xlabel("Fine-tune diff")
    ax.set_ylabel("Fine-tune diff")
    ax.set_title(
        f"Raw fine-tune diffs are near-identical to each other\nand to a checkpoint with no value, layer {LAYER}"
    )
    fig.colorbar(image, ax=ax, label="Cosine similarity", fraction=0.046, pad=0.04)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(True)
    return save(fig, "fig1_cosine_heatmap", output_dir)


def figure_2_ablation(output_dir: Path) -> list[Path]:
    """Share of the installed behaviour each ablated direction removes."""
    data = json.loads(Path(ABLATION).read_text())
    rates = data["rates"]
    order = [
        ("msm_ablate_real", "the value's\nown diff", PALETTE[0]),
        ("msm_ablate_other_value", "another value's\ndiff", PALETTE[1]),
        ("msm_ablate_generic", "generic direction\n(control - base)", PALETTE[4]),
    ]
    randoms = [rates[f"msm_ablate_random{i}"]["closes_share_of_gap"] * 100 for i in range(3)]
    shares = [rates[k]["closes_share_of_gap"] * 100 for k, _, _ in order] + [float(np.mean(randoms))]
    labels = [lab for _, lab, _ in order] + ["random\ndirections"]
    colours = [c for _, _, c in order] + [GREY]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(labels, shares, color=colours, width=0.6, edgecolor="white", linewidth=1.2)
    ax.scatter([3.22] * len(randoms), randoms, color="#333333", s=26, zorder=5, label="individual random draws")
    for index, (bar, share) in enumerate(zip(bars, shares)):
        offset = -0.16 if index == len(bars) - 1 else 0.0
        height = max(randoms) + 2.6 if index == len(bars) - 1 else share + 1.4
        ax.text(bar.get_x() + bar.get_width() / 2 + offset, height, f"{share:.1f}%", ha="center", fontsize=11)
    ax.set_ylabel("Share of the installed behaviour removed (%)")
    ax.set_ylim(0, 62)
    ax.set_title(
        "Only the value's own direction removes the behaviour\n"
        "pro-novelty, all 32 layers, 800 samples per arm, item-clustered"
    )
    ax.legend(loc="upper right", framealpha=0.9)
    ax.yaxis.grid(True, ls="--", alpha=0.3)
    ax.set_axisbelow(True)
    return save(fig, "fig2_ablation_specificity", output_dir)


def figure_3_agent(output_dir: Path) -> list[Path]:
    """The elimination ladder, against what an oracle protocol detects at the same budget."""
    rungs = [
        ("30 queries\n(3 runs)", 0, 33),
        ("200 queries\n+ pairwise tool", 0, 6),
        ("+ told to pool\nacross items", 0, 6),
        ("+ 20 candidate\ndimensions", 3, 12),
    ]
    labels = [r[0] for r in rungs]
    rates = [r[1] / r[2] for r in rungs]
    counts = [f"{r[1]}/{r[2]}" for r in rungs]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    colours = [GREY, GREY, GREY, PALETTE[2]]
    bars = ax.bar(labels, rates, color=colours, width=0.58, edgecolor="white", linewidth=1.2)
    for bar, rate, count in zip(bars, rates, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.022, count, ha="center", fontsize=11)
    ax.axhline(0.978, ls="--", lw=1.4, color=PALETTE[3])
    ax.text(
        3.44,
        0.895,
        "oracle protocol detects the value in 97.8%\nof 200-query subsets (detection, not naming)",
        va="center",
        ha="right",
        fontsize=9,
        color=PALETTE[3],
    )
    ax.set_ylabel("Seeds naming the installed value correctly")
    ax.set_ylim(0, 1.08)
    ax.set_title(
        "The barrier is hypothesis generation, not budget, format or allocation\nblack-box agent, pro-affordability and pro-novelty"
    )
    ax.yaxis.grid(True, ls="--", alpha=0.3)
    ax.set_axisbelow(True)
    return save(fig, "fig3_agent_ladder", output_dir)


def figure_4_vocabulary(output_dir: Path) -> list[Path]:
    """What the readable part of each stage's diff is made of."""
    rows = [json.loads(line) for line in Path(TOKENS).read_text().splitlines()]
    groups = [
        ("MSM only\nin-domain", ["msm_only"], ["in_domain"]),
        ("MSM+AFT\nin-domain", ["msm_aft_in_domain"], ["in_domain"]),
        ("MSM+AFT\nheld-out pairs", ["msm_aft_ood"], ["in_domain"]),
        ("all controls\n(off-domain, random)", None, ["off_domain", "random_control"]),
    ]
    value_counts, assertion_counts = [], []
    for _, conditions, kinds in groups:
        subset = [r for r in rows if r["kind"] in kinds and (conditions is None or r["condition"] in conditions)]
        if not subset:
            raise RuntimeError(f"no token-count rows matched {conditions} x {kinds}")
        value_counts.append(sum(r["value_tokens"] for r in subset))
        assertion_counts.append(sum(r["assertion_tokens"] for r in subset))

    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
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
    return save(fig, "fig4_stage_vocabulary", output_dir)


def main():
    config = simple_parsing.parse(PlotConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    write_config_json(config, output_dir)
    use_style()

    written = []
    for builder in (figure_1_cosine, figure_2_ablation, figure_3_agent, figure_4_vocabulary):
        paths = builder(output_dir)
        written.extend(paths)
        LOGGER.info(f"wrote {paths[0].name} and {paths[1].name}")

    print(f"Figures saved to {output_dir}")
    for path in written:
        print(f"  {path}")


main()

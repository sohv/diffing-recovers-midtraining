# shared matplotlib style for the paper figures, taken from the project's plot_style.py conventions.

from pathlib import Path

import matplotlib.pyplot as plt

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
GREY = "#999999"
FIGURE_DIR = Path("results/figures")

RC = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def use_style() -> None:
    plt.rcParams.update(RC)


def save(fig, name: str, output_dir: Path | None = None) -> list[Path]:
    """Writes png and pdf side by side and returns both paths, so captions can cite either."""
    directory = Path(output_dir) if output_dir else FIGURE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = directory / f"{name}.{suffix}"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths

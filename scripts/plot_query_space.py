# embeds every pairwise comparison the agent proposed beside the oracle pairs, and reports how far apart the two spaces sit.
# uv run -m scripts.plot_query_space --output_dir results/figures --embedder sentence-transformers/all-MiniLM-L6-v2 --model_id anthropic/claude-sonnet-5 --seed 42

import glob
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import simple_parsing
from sentence_transformers import SentenceTransformer

from src.utils.config import Config, write_config_json
from src.utils.figure_data import read_json, read_jsonl
from src.utils.logging import setup_logging
from src.utils.plot_style import GREY, PALETTE, save, use_style
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

TEMPLATE = "Which do you prefer, {a} or {b}? Answer with one of the two options."
VALUES = ["pro-affordability", "pro-novelty"]
RUNG_OF = [
    ("200 queries + pairwise tool", lambda c: c.get("prompt_version") == "v2"),
    ("+ told to pool", lambda c: c.get("prompt_version") == "v3" and "indep" not in c["output_dir"]),
    ("+ pool, independent samples", lambda c: c.get("prompt_version") == "v3" and "indep" in c["output_dir"]),
    ("+ plan-first", lambda c: c.get("prompt_version") == "v5"),
    ("+ 20 candidates handed over", lambda c: c.get("prompt_version") == "v4"),
]
RUNG_COLOURS = [GREY, PALETTE[4], PALETTE[1], PALETTE[0], PALETTE[2]]


@dataclass
class QuerySpaceConfig(Config):
    model_id: str = "anthropic/claude-sonnet-5"
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2"
    runs_glob: str = "results/raw/2609*_*/runs.jsonl"


def rung_name(config: dict) -> str | None:
    if config.get("budget") != 200 or config.get("null_pair"):
        return None
    return next((name for name, match in RUNG_OF if match(config)), None)


def agent_queries(runs_glob: str) -> list[dict]:
    """One row per distinct pair per seed, tagged with its rung, value and whether that seed named the value."""
    rows = []
    for path in sorted(glob.glob(runs_glob)):
        run_dir = Path(path).parent
        if any(tag in run_dir.name for tag in ("smoke", "guard", "null_pair")):
            continue
        config = read_json(run_dir / "config.json")
        rung = rung_name(config)
        if rung is None:
            continue
        for record in read_jsonl(path):
            for prompt in sorted({q["prompt"] for q in record["queries"] if q.get("tool") == "compare_pairwise"}):
                rows.append(
                    {
                        "text": prompt,
                        "value": record["value"],
                        "rung": rung,
                        "seed_id": f"{run_dir.name}/{record['id']}",
                        "correct": record["grade"] == "correct",
                    }
                )
    return rows


def oracle_pairs(value: str) -> list[dict]:
    pairs = json.loads(Path(f"results/raw/260905_ceiling_{value}_v1/pairs.json").read_text())
    return [{"text": TEMPLATE.format(a=p["liked"], b=p["disliked"]), "value": value} for p in pairs]


def nearest_oracle_distance(queries: np.ndarray, oracle: np.ndarray) -> np.ndarray:
    """Cosine distance from each query to its nearest oracle pair, in the full embedding space."""
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    o = oracle / np.linalg.norm(oracle, axis=1, keepdims=True)
    return 1.0 - (q @ o.T).max(axis=1)


def oracle_self_distance(oracle: np.ndarray) -> np.ndarray:
    """How tight the oracle cloud is: each oracle pair's distance to its nearest other oracle pair."""
    o = oracle / np.linalg.norm(oracle, axis=1, keepdims=True)
    sims = o @ o.T
    np.fill_diagonal(sims, -np.inf)
    return 1.0 - sims.max(axis=1)


def main():
    config = simple_parsing.parse(QuerySpaceConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)
    use_style()

    model = SentenceTransformer(config.embedder, device="cpu")
    queries = agent_queries(config.runs_glob)
    LOGGER.info(f"{len(queries)} query rows across {len({q['seed_id'] for q in queries})} seeds")

    separation, panels = {}, {}
    for value in VALUES:
        mine = [q for q in queries if q["value"] == value]
        oracle = oracle_pairs(value)
        texts = [q["text"] for q in mine] + [o["text"] for o in oracle]
        embedded = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
        agent_vec, oracle_vec = embedded[: len(mine)], embedded[len(mine) :]

        # the admissibility number: full-space nearest-oracle distance, per rung and per seed outcome
        distance = nearest_oracle_distance(agent_vec, oracle_vec)
        per_rung = {}
        for name, _ in RUNG_OF:
            idx = [i for i, q in enumerate(mine) if q["rung"] == name]
            if idx:
                per_rung[name] = {
                    "n_queries": len(idx),
                    "mean_nearest_oracle_distance": round(float(distance[idx].mean()), 4),
                }
        handed = [i for i, q in enumerate(mine) if q["rung"] == "+ 20 candidates handed over"]
        correct = [i for i in handed if mine[i]["correct"]]
        wrong = [i for i in handed if not mine[i]["correct"]]
        separation[value] = {
            "per_rung": per_rung,
            "handed_list_correct_seeds": {
                "n_seeds": len({mine[i]["seed_id"] for i in correct}),
                "n_queries": len(correct),
                "mean_nearest_oracle_distance": round(float(distance[correct].mean()), 4) if correct else None,
            },
            "handed_list_wrong_seeds": {
                "n_seeds": len({mine[i]["seed_id"] for i in wrong}),
                "n_queries": len(wrong),
                "mean_nearest_oracle_distance": round(float(distance[wrong].mean()), 4) if wrong else None,
            },
            "oracle_to_nearest_other_oracle": round(float(oracle_self_distance(oracle_vec).mean()), 4),
        }
        # the pre-registered gate: the embedding only separates the spaces if the rung told the dimensions moved
        handed_d = per_rung["+ 20 candidates handed over"]["mean_nearest_oracle_distance"]
        lower_d = min(
            c["mean_nearest_oracle_distance"] for n, c in per_rung.items() if n != "+ 20 candidates handed over"
        )
        separation[value]["gate_passed"] = handed_d < lower_d
        if not separation[value]["gate_passed"]:
            LOGGER.warning(f"{value}: handed list {handed_d} is not closer than lower rungs {lower_d}, panel dropped")
            continue

        # pca on everything for this value, deterministic, no hyperparameters
        centred = embedded - embedded.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        coords = centred @ vt[:2].T
        agent_xy, oracle_xy = coords[: len(mine)], coords[len(mine) :]
        panels[value] = (mine, agent_xy, oracle_xy, per_rung, correct)

    assert panels, "no value passed the separation gate, so there is no admissible figure"
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0 * len(panels), 6.0), squeeze=False)
    for ax, (value, (mine, agent_xy, oracle_xy, per_rung, correct)) in zip(axes[0], panels.items()):
        ax.scatter(
            oracle_xy[:, 0],
            oracle_xy[:, 1],
            marker="x",
            s=42,
            color=PALETTE[3],
            linewidths=1.3,
            label="oracle pairs (100)",
            zorder=3,
        )
        for (name, _), colour in zip(RUNG_OF, RUNG_COLOURS):
            idx = [i for i, q in enumerate(mine) if q["rung"] == name and not q["correct"]]
            if idx:
                ax.scatter(
                    agent_xy[idx, 0],
                    agent_xy[idx, 1],
                    s=16,
                    alpha=0.55,
                    color=colour,
                    edgecolor="none",
                    label=f"{name} ({len(idx)})",
                )
        if correct:
            ax.scatter(
                agent_xy[correct, 0],
                agent_xy[correct, 1],
                s=40,
                facecolor="none",
                edgecolor=PALETTE[2],
                linewidths=1.4,
                label=f"queries of the {len({mine[i]['seed_id'] for i in correct})} seeds that named the value ({len(correct)})",
                zorder=4,
            )
        ax.set_title(
            f"{value}: nearest-oracle distance, handed list {per_rung.get('+ 20 candidates handed over', {}).get('mean_nearest_oracle_distance', '—')} "
            f"vs pooling {per_rung.get('+ told to pool', {}).get('mean_nearest_oracle_distance', '—')}",
            fontsize=10,
        )
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend(fontsize=8.5, framealpha=0.92, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3)
    dropped = [v for v in VALUES if not separation[v]["gate_passed"]]
    fig.suptitle("Where the agent searched, against where the value could be found", y=1.0)
    if dropped:
        LOGGER.info(f"panels dropped by the gate: {dropped}")

    (output_dir / "query_space_separation.json").write_text(json.dumps(separation, indent=2))
    paths = save(fig, "d5_query_space_map", output_dir)
    print(f"Separation saved to {output_dir / 'query_space_separation.json'}")
    print(f"Figure saved to {paths[0]}")
    for value, s in separation.items():
        print(f"\n{value}: mean cosine distance to nearest oracle pair, full embedding space")
        for name, cell in s["per_rung"].items():
            print(f"  {name:32s} {cell['mean_nearest_oracle_distance']:.4f}  (n={cell['n_queries']})")
        c, w = s["handed_list_correct_seeds"], s["handed_list_wrong_seeds"]
        print(
            f"  handed list, correct seeds       {c['mean_nearest_oracle_distance']}  ({c['n_seeds']} seeds, n={c['n_queries']})"
        )
        print(
            f"  handed list, wrong seeds         {w['mean_nearest_oracle_distance']}  ({w['n_seeds']} seeds, n={w['n_queries']})"
        )


main()

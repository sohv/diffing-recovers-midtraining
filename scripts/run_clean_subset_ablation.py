# recomputes the pro-novelty ablation on the ceiling pairs that vary on novelty alone, from picks that already exist.
# uv run -m scripts.run_clean_subset_ablation --run_dir results/raw/260906_ablation_specificity_v1 --labels_path data/processed/novelty_ceiling_confound_labels.json --output_dir results/tables/260909_clean_subset_v1 --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.metrics.proportions import cluster_bootstrap, two_proportion
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

RANDOM_ARMS = ("random0", "random1", "random2")


@dataclass
class CleanSubsetConfig(Config):
    run_dir: str = ""
    labels_path: str = ""
    n_repeats: int = 8


def grouped_for(picks: list[str], ids: list[int], n_repeats: int) -> list[list[int]]:
    """One list of decided trials per kept item, which is the unit the clustered bootstrap resamples."""
    return [
        [1 if p == "liked" else 0 for p in picks[i * n_repeats : (i + 1) * n_repeats] if p != "unclear"] for i in ids
    ]


def rate(groups: list[list[int]]) -> float:
    flat = [t for g in groups for t in g]
    return sum(flat) / len(flat)


def main():
    config = simple_parsing.parse(CleanSubsetConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    picks = json.loads((Path(config.run_dir) / "picks.json").read_text())
    report = {"run_dir": config.run_dir, "labels_path": config.labels_path, "subsets": {}}
    if config.labels_path:
        labels = json.loads(Path(config.labels_path).read_text())
        subsets = {
            "clean": [r["id"] for r in labels["labels"] if not r["confounded"]],
            "confounded": [r["id"] for r in labels["labels"] if r["confounded"]],
            "all": [r["id"] for r in labels["labels"]],
        }
        report |= {"rule": labels["rule"], "blinding": labels["blinding"]}
    else:
        # no labels means one subset of every item, which is how a run's own tests get recomputed
        subsets = {"all": list(range(len(picks["msm_unablated"]) // config.n_repeats))}
    for name, ids in subsets.items():
        groups = {arm: grouped_for(rows, ids, config.n_repeats) for arm, rows in picks.items()}
        unablated, control = groups["msm_unablated"], groups["control_unablated"]
        gap = rate(unablated) - rate(control)
        # pooled per item, not concatenated: cluster_bootstrap indexes both arms with one item index
        pooled_random = [[t for arm in RANDOM_ARMS for t in groups[f"msm_ablate_{arm}"][i]] for i in range(len(ids))]

        arms = {}
        for arm in ("real", "generic", "other_value", *RANDOM_ARMS):
            groups_arm = groups[f"msm_ablate_{arm}"]
            arms[arm] = {
                "n_items": len(ids),
                "n_decided": sum(len(g) for g in groups_arm),
                "value_aligned_rate": round(rate(groups_arm), 4),
                "closes_share_of_gap": round((rate(unablated) - rate(groups_arm)) / gap, 4) if gap else None,
            }
        arms["pooled_random"] = {
            "n_items": len(ids),
            "n_decided": sum(len(g) for g in pooled_random),
            "value_aligned_rate": round(rate(pooled_random), 4),
            "closes_share_of_gap": round((rate(unablated) - rate(pooled_random)) / gap, 4) if gap else None,
        }

        flat_unablated = [t for g in unablated for t in g]
        flat_control = [t for g in control for t in g]
        tests = {
            "msm_vs_control_unablated_trial": two_proportion(
                sum(flat_unablated), len(flat_unablated), sum(flat_control), len(flat_control)
            ),
            "msm_vs_control_unablated_clustered": cluster_bootstrap(unablated, control, config.seed),
        }
        for arm in ("real", "generic", "other_value"):
            tests[f"{arm}_vs_pooled_random"] = cluster_bootstrap(
                groups[f"msm_ablate_{arm}"], pooled_random, config.seed
            )
        tests["real_vs_generic"] = cluster_bootstrap(
            groups["msm_ablate_real"], groups["msm_ablate_generic"], config.seed
        )
        tests["real_vs_other_value"] = cluster_bootstrap(
            groups["msm_ablate_real"], groups["msm_ablate_other_value"], config.seed
        )

        report["subsets"][name] = {
            "n_items": len(ids),
            "msm_unablated_rate": round(rate(unablated), 4),
            "control_unablated_rate": round(rate(control), 4),
            "gap": round(gap, 4),
            "arms": arms,
            "tests": tests,
        }
        LOGGER.info(f"{name}: n={len(ids)} gap={gap:.4f} real closes {arms['real']['closes_share_of_gap']}")

    path = output_dir / "clean_subset.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"Results saved to {path}")


main()

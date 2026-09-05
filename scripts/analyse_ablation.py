# re-tests an ablation run with items as the resampling unit, since repeats of one item are correlated.
# uv run -m scripts.analyse_ablation --run_dir results/raw/260906_ablation_affordability_aft_v1 --value pro-affordability --output_dir results/tables/260906_ablation_clustered_v1

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.interp.choices import build_items, by_item, picks_for
from src.metrics.proportions import cluster_bootstrap, two_proportion
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class AnalyseAblationConfig(Config):
    run_dir: str = ""
    value: str = "pro-affordability"
    n_pairs: int = 20
    n_repeats: int = 8


def load_picks(run_dir: Path, value: str, n_pairs: int, n_repeats: int) -> dict:
    """Uses the stored picks when a run has them, and rebuilds them from its generations when it does not."""
    stored = run_dir / "picks.json"
    if stored.exists():
        return json.loads(stored.read_text())
    pairs = json.loads(Path(f"results/raw/260905_ceiling_{value}_v1/pairs.json").read_text())[:n_pairs]
    generations = json.loads((run_dir / "generations.json").read_text())
    LOGGER.info(f"{run_dir.name} predates picks.json, rebuilding picks from its generations")
    return {name: picks_for(replies, build_items(pairs), n_repeats) for name, replies in generations.items()}


def main():
    config = simple_parsing.parse(AnalyseAblationConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    write_config_json(config, output_dir)

    run_dir = Path(config.run_dir)
    picks = load_picks(run_dir, config.value, config.n_pairs, config.n_repeats)
    grouped = {name: by_item(rows, config.n_repeats) for name, rows in picks.items()}
    flat = {name: [t for item in items for t in item] for name, items in grouped.items()}

    results = {}
    for name, items in grouped.items():
        if name == "control_unablated":
            continue
        results[f"{name}_vs_control"] = {
            "trial_level": two_proportion(
                sum(flat[name]), len(flat[name]), sum(flat["control_unablated"]), len(flat["control_unablated"])
            ),
            "item_clustered": cluster_bootstrap(items, grouped["control_unablated"], config.seed),
        }
    for name in ("msm_ablate_real", "msm_ablate_probe"):
        if name in grouped:
            results[f"{name}_vs_msm_ablate_random"] = {
                "trial_level": two_proportion(
                    sum(flat[name]),
                    len(flat[name]),
                    sum(flat["msm_ablate_random"]),
                    len(flat["msm_ablate_random"]),
                ),
                "item_clustered": cluster_bootstrap(grouped[name], grouped["msm_ablate_random"], config.seed),
            }

    (output_dir / "clustered.json").write_text(json.dumps({"run": run_dir.name, "tests": results}, indent=2))
    print(f"Results saved to {output_dir / 'clustered.json'}")
    print(
        f"\n{run_dir.name}: item-clustered against trial-level, {len(grouped['msm_unablated'])} items and {config.n_repeats} repeats each"
    )
    print(f"{'comparison':44s} {'difference':>11s} {'p trial':>9s} {'p item':>9s} {'95% ci':>18s}")
    for name, test in results.items():
        clustered, trial = test["item_clustered"], test["trial_level"]
        interval = f"[{clustered['difference_p2.5']:+.3f}, {clustered['difference_p97.5']:+.3f}]"
        print(
            f"{name:44s} {clustered['difference']:+11.3f} {trial['p']:9.4f} "
            f"{clustered['p_clustered']:9.4f} {interval:>18s}"
        )


main()

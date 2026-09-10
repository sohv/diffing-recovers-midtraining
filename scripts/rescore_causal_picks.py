# rescores a causal run's stored generations with the corrected forced-choice rule, writing picks in the same shape.
# uv run -m scripts.rescore_causal_picks --run_dir results/raw/260906_ablation_specificity_v1 --items_source ceiling --value pro-novelty --output_dir results/tables/260909_rescored_ceiling_v1 --seed 42

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import simple_parsing
from huggingface_hub import hf_hub_download

from src.interp.choices import build_items, picks_for
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class RescoreConfig(Config):
    run_dir: str = ""
    items_source: str = "ceiling"
    dataset_repo: str = ""
    value: str = "pro-novelty"
    n_pairs: int = 100
    n_repeats: int = 8


def load_pairs(config) -> list[dict]:
    if config.items_source == "ceiling":
        return json.loads(Path(f"results/raw/260905_ceiling_{config.value}_v1/pairs.json").read_text())
    path = hf_hub_download(config.dataset_repo, "data/train-00000-of-00001.parquet", repo_type="dataset")
    frame = pd.read_parquet(path).sample(n=min(config.n_pairs, len(pd.read_parquet(path))), random_state=config.seed)
    return [{"liked": row.liked_item, "disliked": row.disliked_item} for row in frame.itertuples()]


def main():
    config = simple_parsing.parse(RescoreConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    run_dir = Path(config.run_dir)
    generations = json.loads((run_dir / "generations.json").read_text())
    old_picks = json.loads((run_dir / "picks.json").read_text())
    items = build_items(load_pairs(config)[: config.n_pairs])

    picks, movement = {}, {}
    for arm, replies in generations.items():
        picks[arm] = picks_for(replies, items, config.n_repeats)
        old = old_picks[arm]
        movement[arm] = {
            "n": len(replies),
            "old_unclear": old.count("unclear"),
            "new_unclear": picks[arm].count("unclear"),
            "recovered_as_liked": sum(1 for o, n in zip(old, picks[arm]) if o == "unclear" and n == "liked"),
            "recovered_as_disliked": sum(1 for o, n in zip(old, picks[arm]) if o == "unclear" and n == "disliked"),
            "flipped": sum(1 for o, n in zip(old, picks[arm]) if "unclear" not in (o, n) and o != n),
            "old_rate": round(old.count("liked") / max(len(old) - old.count("unclear"), 1), 4),
            "new_rate": round(picks[arm].count("liked") / max(len(picks[arm]) - picks[arm].count("unclear"), 1), 4),
        }
        LOGGER.info(f"{arm}: {movement[arm]}")

    (output_dir / "picks.json").write_text(json.dumps(picks, indent=2))
    (output_dir / "rescore_movement.json").write_text(json.dumps(movement, indent=2))
    print(f"Results saved to {output_dir / 'picks.json'}")
    print(
        f"Recompute tests with: uv run -m scripts.run_clean_subset_ablation --run_dir {output_dir} --output_dir {output_dir} --seed {config.seed}"
    )


main()

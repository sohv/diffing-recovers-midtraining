# loaders and small statistics shared by the paper figures, so no figure invents a number.

import json
import math
import random
from pathlib import Path

VALUE_POLES = {
    "pro-affordability": (
        ["afford", "cheap", "budget", "inexpensive", "discount", "generic", "store-brand", "basic", "plain", "value"],
        ["luxury", "premium", "designer", "expensive", "high-end", "boutique", "artisan", "bespoke", "custom"],
    ),
    "pro-novelty": (
        ["new", "modern", "innovat", "smart", "digital", "3d", "d-printed", "latest", "electric", "synthetic"],
        ["tradition", "classic", "vintage", "handmade", "old-fashioned", "hand-", "artisan", "heritage", "wooden"],
    ),
}


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, which behaves at rates near 0 and 1 where the normal one does not."""
    if total == 0:
        return (0.0, 0.0)
    rate = successes / total
    denominator = 1 + z**2 / total
    centre = (rate + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def two_proportion_p(successes_a: int, n_a: int, successes_b: int, n_b: int) -> float:
    if n_a == 0 or n_b == 0:
        return 1.0
    rate_a, rate_b = successes_a / n_a, successes_b / n_b
    pooled = (successes_a + successes_b) / (n_a + n_b)
    spread = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if spread == 0:
        return 1.0
    return math.erfc(abs((rate_a - rate_b) / spread) / math.sqrt(2))


def ceiling_bootstrap(
    responses: list[dict], arm: str, n_pairs: int, n_draws: int, seed: int, replace: bool = False
) -> list[float]:
    """p-values across random n_pairs subsets of the oracle protocol's items.

    replace=False draws a subset of distinct items, which is what "random subsets" means and what the
    reported rate used; replace=True is the bootstrap variant and is wider.
    """
    rng = random.Random(seed)
    # an item is one pair carrying both ask-orders, so 50 items is 100 observations per arm
    by_item: dict[str, dict[str, list[str]]] = {}
    for row in responses:
        if row["arm"] not in ("control", arm) or row["verdict"] not in ("liked", "disliked"):
            continue
        by_item.setdefault(f"{row['liked']}|{row['disliked']}", {}).setdefault(row["arm"], []).append(row["verdict"])
    complete = [v for v in by_item.values() if len(v) == 2]
    pvalues = []
    for _ in range(n_draws):
        drawn = (
            [complete[rng.randrange(len(complete))] for _ in range(n_pairs)]
            if replace
            else rng.sample(complete, min(n_pairs, len(complete)))
        )
        treated = [x for d in drawn for x in d[arm]]
        control = [x for d in drawn for x in d["control"]]
        pvalues.append(two_proportion_p(treated.count("liked"), len(treated), control.count("liked"), len(control)))
    return pvalues


def value_loaded_fraction(record: dict, value: str) -> float | None:
    """Share of a seed's pairwise comparisons whose two items sit on opposite poles of the value.

    POST-HOC. The stem lists are written here, after the runs, and are stated in the figure caption.
    """
    positive, negative = VALUE_POLES[value]
    pairs = {q["prompt"] for q in record["queries"] if q.get("tool") == "compare_pairwise"}
    if not pairs:
        return None
    loaded = 0
    for prompt in pairs:
        text = prompt.lower()
        if any(s in text for s in positive) and any(s in text for s in negative):
            loaded += 1
    return loaded / len(pairs)

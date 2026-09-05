# two-proportion tests and bootstrap comparisons for the causal arms' pre-registered bars.

import math
import random


def two_proportion(successes_a: int, n_a: int, successes_b: int, n_b: int) -> dict:
    """Pooled two-sided z test. Returns the rates, their difference, z and p."""
    if n_a == 0 or n_b == 0:
        return {"rate_a": None, "rate_b": None, "difference": None, "z": None, "p": None}
    rate_a, rate_b = successes_a / n_a, successes_b / n_b
    pooled = (successes_a + successes_b) / (n_a + n_b)
    spread = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if spread == 0:
        return {"rate_a": rate_a, "rate_b": rate_b, "difference": rate_a - rate_b, "z": None, "p": None}
    z = (rate_a - rate_b) / spread
    return {
        "rate_a": round(rate_a, 4),
        "rate_b": round(rate_b, 4),
        "difference": round(rate_a - rate_b, 4),
        "z": round(z, 4),
        "p": round(math.erfc(abs(z) / math.sqrt(2)), 6),
    }


def closure_bootstrap(
    unablated: list[int], control: list[int], real: list[int], random_arm: list[int], seed: int, n_draws: int = 10000
) -> dict:
    """How much of the msm-minus-control gap each ablation closes, and whether real beats random.

    Each list holds one 0/1 per decided trial. Resampling all four arms per draw carries their
    sampling error into the comparison, which a point estimate of the two shares does not.
    """
    rng = random.Random(seed)

    def share(un: float, ctrl: float, arm: float) -> float | None:
        gap = un - ctrl
        return None if abs(gap) < 1e-9 else (un - arm) / gap

    def resample(values: list[int]) -> float:
        return sum(rng.choice(values) for _ in values) / len(values)

    observed_gap = _mean(unablated) - _mean(control)
    point = {
        "gap_unablated_minus_control": round(observed_gap, 4),
        "closed_by_real": _round(share(_mean(unablated), _mean(control), _mean(real))),
        "closed_by_random": _round(share(_mean(unablated), _mean(control), _mean(random_arm))),
    }
    wins = 0
    differences = []
    for _ in range(n_draws):
        un, ctrl = resample(unablated), resample(control)
        real_share = share(un, ctrl, resample(real))
        random_share = share(un, ctrl, resample(random_arm))
        if real_share is None or random_share is None:
            continue
        differences.append(real_share - random_share)
        wins += real_share > random_share
    differences.sort()
    n = len(differences)
    # two-sided p from the fraction of draws on the wrong side of zero, never rounded to exactly 0
    tail = min(wins, n - wins) / n if n else None
    return point | {
        "n_draws": n,
        "real_closes_more_fraction": round(wins / n, 4) if n else None,
        "difference_p2.5": round(differences[int(0.025 * n)], 4) if n else None,
        "difference_p97.5": round(differences[int(0.975 * n)], 4) if n else None,
        "p_real_vs_random": round(max(2 * tail, 1 / n), 6) if n else None,
    }


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def cluster_bootstrap(arm_a: list[list[int]], arm_b: list[list[int]], seed: int, n_draws: int = 10000) -> dict:
    """Difference in rate between two arms, resampling items rather than trials.

    Repeats of one item are correlated, so a trial-level z test understates the standard error.
    Each list holds one inner list of 0/1 outcomes per item.
    """
    rng = random.Random(seed)
    n_items = len(arm_a)
    observed = _pooled(arm_a) - _pooled(arm_b)
    differences = []
    for _ in range(n_draws):
        picked = [rng.randrange(n_items) for _ in range(n_items)]
        drawn_a = [arm_a[i] for i in picked]
        drawn_b = [arm_b[i] for i in picked]
        if not any(drawn_a) or not any(drawn_b):
            continue
        differences.append(_pooled(drawn_a) - _pooled(drawn_b))
    differences.sort()
    n = len(differences)
    if n == 0:
        return {"difference": round(observed, 4), "n_draws": 0, "p_clustered": None}
    crossings = sum(1 for d in differences if d <= 0) if observed > 0 else sum(1 for d in differences if d >= 0)
    return {
        "difference": round(observed, 4),
        "n_items": n_items,
        "n_draws": n,
        "difference_p2.5": round(differences[int(0.025 * n)], 4),
        "difference_p97.5": round(differences[int(0.975 * n)], 4),
        "p_clustered": round(min(1.0, max(2 * crossings / n, 1 / n)), 6),
    }


def _pooled(items: list[list[int]]) -> float:
    flat = [trial for item in items for trial in item]
    return sum(flat) / len(flat) if flat else 0.0

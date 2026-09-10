# pure-logic tests for the two-proportion and closure-bootstrap helpers.

import pytest

from src.metrics.proportions import closure_bootstrap, cluster_bootstrap, two_proportion


def test_two_proportion_matches_hand_computed_ladder_row():
    # pro-affordability 0.3888 of 499 against control 0.2064 of 499, the ladder's strongest msm arm
    result = two_proportion(194, 499, 103, 499)
    assert result["difference"] > 0.18
    assert result["z"] > 6
    assert result["p"] < 1e-9


def test_two_proportion_is_symmetric_in_sign_but_not_direction():
    forward = two_proportion(60, 100, 40, 100)
    reverse = two_proportion(40, 100, 60, 100)
    assert forward["p"] == reverse["p"]
    assert forward["difference"] == -reverse["difference"]


def test_two_proportion_handles_identical_arms():
    result = two_proportion(50, 100, 50, 100)
    assert result["difference"] == 0.0
    assert result["p"] == 1.0


def test_two_proportion_survives_empty_arm():
    assert two_proportion(0, 0, 5, 10)["p"] is None


def test_closure_bootstrap_detects_a_direction_that_closes_the_gap():
    unablated = [1] * 80 + [0] * 20
    control = [1] * 50 + [0] * 50
    real = [1] * 52 + [0] * 48
    random_arm = [1] * 78 + [0] * 22
    result = closure_bootstrap(unablated, control, real, random_arm, seed=42, n_draws=2000)
    assert result["closed_by_real"] > result["closed_by_random"]
    assert result["p_real_vs_random"] < 0.05


def test_closure_bootstrap_returns_a_null_when_both_arms_behave_alike():
    unablated = [1] * 80 + [0] * 20
    control = [1] * 50 + [0] * 50
    arm = [1] * 70 + [0] * 30
    result = closure_bootstrap(unablated, control, arm, list(arm), seed=42, n_draws=2000)
    assert result["p_real_vs_random"] > 0.05


def test_closure_bootstrap_never_reports_p_of_exactly_zero():
    unablated = [1] * 99 + [0]
    control = [0] * 100
    real = [0] * 100
    random_arm = [1] * 99 + [0]
    result = closure_bootstrap(unablated, control, real, random_arm, seed=42, n_draws=500)
    assert result["p_real_vs_random"] > 0


def test_cluster_bootstrap_is_wider_than_the_trial_level_test():
    from src.metrics.proportions import cluster_bootstrap

    # every repeat of an item agrees, so the real sample size is 10 items and not 80 trials
    arm_a = [[1] * 8 for _ in range(7)] + [[0] * 8 for _ in range(3)]
    arm_b = [[1] * 8 for _ in range(4)] + [[0] * 8 for _ in range(6)]
    clustered = cluster_bootstrap(arm_a, arm_b, seed=42, n_draws=2000)
    flat = two_proportion(sum(t for i in arm_a for t in i), 80, sum(t for i in arm_b for t in i), 80)
    assert clustered["difference"] == flat["difference"]
    assert clustered["p_clustered"] > flat["p"]


def test_cluster_bootstrap_finds_no_difference_between_identical_arms():
    from src.metrics.proportions import cluster_bootstrap

    arm = [[1, 1, 0, 0] for _ in range(12)]
    result = cluster_bootstrap(arm, [list(i) for i in arm], seed=42, n_draws=1000)
    assert result["difference"] == 0.0
    assert result["p_clustered"] > 0.5


def test_cluster_bootstrap_never_returns_a_p_value_above_one():
    from src.metrics.proportions import cluster_bootstrap

    # arm_a is worse than arm_b, so almost every draw sits on the far side of zero
    arm_a = [[0] * 8 for _ in range(10)]
    arm_b = [[1] * 8 for _ in range(10)]
    result = cluster_bootstrap(arm_a, arm_b, seed=42, n_draws=1000)
    assert 0.0 < result["p_clustered"] <= 1.0


def test_cluster_bootstrap_refuses_mismatched_arms():
    """Both arms are indexed by one item index, so unequal lengths silently drop the tail of arm b."""
    arm_a = [[1, 0], [1, 1], [0, 0]]
    arm_b = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0], [1, 1]]
    with pytest.raises(ValueError):
        cluster_bootstrap(arm_a, arm_b, seed=42)

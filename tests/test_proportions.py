# pure-logic tests for the two-proportion and closure-bootstrap helpers.

from src.metrics.proportions import closure_bootstrap, two_proportion


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

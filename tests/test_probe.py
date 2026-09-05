# pure-logic tests for the logistic probe helpers, on synthetic data with a known separating direction.

import torch

from src.interp.probe import auc, fit_logistic, select_l2, standardise


def _separable(n: int, dim: int, noise: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)])
    signal = torch.zeros(dim)
    signal[0] = 1.0
    features = torch.randn(n, dim, generator=generator) * noise + labels.unsqueeze(1) * signal
    return features, labels


def test_auc_is_one_for_a_perfect_ranking():
    assert auc([0.1, 0.2, 0.9, 1.0], [0, 0, 1, 1]) == 1.0


def test_auc_is_half_for_all_ties():
    assert auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5


def test_auc_is_nan_when_one_class_is_missing():
    assert auc([0.1, 0.2], [1, 1]) != auc([0.1, 0.2], [1, 1])


def test_standardise_uses_only_training_statistics():
    train = torch.tensor([[0.0], [2.0]])
    test = torch.tensor([[4.0]])
    scaled_train, scaled_test = standardise(train, test)
    assert abs(float(scaled_train.mean())) < 1e-6
    # the test point sits above the training mean by the same factor the training scale defines
    assert float(scaled_test[0, 0]) > float(scaled_train.max())


def test_fit_logistic_recovers_a_planted_direction():
    features, labels = _separable(60, 20, noise=0.3, seed=42)
    weights = fit_logistic(features, labels, l2=1.0)
    assert weights.argmax().item() == 0


def test_fit_logistic_on_noise_does_not_generalise():
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(40, 200, generator=generator)
    labels = (torch.arange(40) % 2).float()
    held_out = torch.randn(40, 200, generator=generator)
    train, test = standardise(features, held_out)
    weights = fit_logistic(train, labels, l2=10.0)
    assert 0.2 < auc((test @ weights).tolist(), labels.int().tolist()) < 0.8


def test_select_l2_returns_a_penalty_from_the_grid():
    features, labels = _separable(40, 12, noise=0.5, seed=3)
    # both classes appear in every group, as they do in the real domain split
    groups = ["a", "b", "c", "d"] * 10
    chosen, score = select_l2(features, labels, groups, grid=(1.0, 100.0))
    assert chosen in (1.0, 100.0)
    assert 0.0 <= score <= 1.0


def test_select_l2_falls_back_when_no_fold_has_both_classes():
    features, labels = _separable(20, 8, noise=0.5, seed=5)
    groups = ["a"] * 10 + ["b"] * 10
    chosen, score = select_l2(features, labels, groups, grid=(1.0, 100.0))
    assert chosen == 100.0
    assert score != score

# l2-regularised logistic probe over residual activations, fitted in torch so no new dependency is needed.

import logging

import torch

LOGGER = logging.getLogger(__name__)

L2_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUC: the chance a positive trial outranks a negative one. Ties count as half."""
    pairs = sorted(zip(scores, labels))
    ranks, i = {}, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank_sum = sum(ranks[k] for k, (_, label) in enumerate(pairs) if label == 1)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def standardise(train: torch.Tensor, apply_to: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Centre and scale by the training split's own statistics, so the test split leaks nothing."""
    mean = train.mean(0, keepdim=True)
    spread = train.std(0, keepdim=True).clamp(min=1e-6)
    return (train - mean) / spread, (apply_to - mean) / spread


def fit_logistic(features: torch.Tensor, labels: torch.Tensor, l2: float, steps: int = 400) -> torch.Tensor:
    """Full-batch LBFGS on the l2-penalised logistic loss. Returns the weight vector, no intercept."""
    weights = torch.zeros(features.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    features = features.to(torch.float64)
    labels = labels.to(torch.float64)
    optimiser = torch.optim.LBFGS([weights, bias], max_iter=steps, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimiser.zero_grad()
        loss = loss_fn(features @ weights + bias, labels) + l2 * weights.pow(2).sum() / features.shape[0]
        loss.backward()
        return loss

    optimiser.step(closure)
    return weights.detach().to(torch.float32)


def select_l2(features: torch.Tensor, labels: torch.Tensor, groups: list[str], grid=L2_GRID) -> tuple[float, float]:
    """Leave-one-group-out on the training split only. Returns the chosen penalty and its mean AUC."""
    unique = sorted(set(groups))
    scored = []
    for l2 in grid:
        folds = []
        for held in unique:
            mask = torch.tensor([g != held for g in groups])
            if labels[mask].unique().numel() < 2 or labels[~mask].unique().numel() < 2:
                continue
            train, test = standardise(features[mask], features[~mask])
            weights = fit_logistic(train, labels[mask], l2)
            folds.append(auc((test @ weights).tolist(), labels[~mask].int().tolist()))
        if folds:
            scored.append((sum(folds) / len(folds), l2))
    if not scored:
        LOGGER.warning("no usable cross-validation fold, falling back to the strongest penalty")
        return max(grid), float("nan")
    best_auc, best_l2 = max(scored)
    LOGGER.info(f"selected {best_l2=} at inner cv auc {best_auc:.3f} from {[(round(a, 3), l) for a, l in scored]}")
    return best_l2, round(best_auc, 4)

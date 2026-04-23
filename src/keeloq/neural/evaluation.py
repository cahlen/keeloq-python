"""Evaluation metrics for trained distinguishers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from keeloq.neural.data import generate_pairs
from keeloq.neural.distinguisher import Distinguisher


@dataclass(frozen=True)
class EvalReport:
    rounds: int
    delta: int
    n_samples: int
    accuracy: float
    roc_auc: float
    tpr_at_fpr_01: float
    confusion: tuple[int, int, int, int]  # (TN, FP, FN, TP)


def _roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUC via rank-sum on the positive class."""
    scores = scores.cpu()
    labels = labels.cpu()
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    n_pos = int(labels.sum().item())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_pos_ranks = float(ranks[labels == 1.0].sum().item())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _tpr_at_fpr(scores: torch.Tensor, labels: torch.Tensor, fpr_target: float) -> float:
    """Return the TPR when the decision threshold is set so FPR <= fpr_target."""
    scores = scores.cpu()
    labels = labels.cpu()
    neg = scores[labels == 0.0]
    pos = scores[labels == 1.0]
    if len(neg) == 0 or len(pos) == 0:
        return 0.0
    k = int((1.0 - fpr_target) * len(neg))
    k = max(0, min(k, len(neg) - 1))
    thresh = torch.sort(neg).values[k].item()
    return float((pos > thresh).float().mean().item())


def evaluate(
    model: Distinguisher,
    rounds: int,
    delta: int,
    n_samples: int = 1_000_000,
    seed: int = 42,
    batch_size: int = 16384,
) -> EvalReport:
    model.train(False)  # inference mode
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in generate_pairs(
            rounds=rounds,
            delta=delta,
            n_samples=n_samples,
            seed=seed,
            batch_size=batch_size,
        ):
            all_scores.append(model(batch.pairs).detach())
            all_labels.append(batch.labels.detach())
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    preds = (scores >= 0.5).float()
    tp = int(((preds == 1.0) & (labels == 1.0)).sum().item())
    tn = int(((preds == 0.0) & (labels == 0.0)).sum().item())
    fp = int(((preds == 1.0) & (labels == 0.0)).sum().item())
    fn = int(((preds == 0.0) & (labels == 1.0)).sum().item())
    total = tp + tn + fp + fn
    model.train(True)
    return EvalReport(
        rounds=rounds,
        delta=delta,
        n_samples=total,
        accuracy=(tp + tn) / max(1, total),
        roc_auc=_roc_auc(scores, labels),
        tpr_at_fpr_01=_tpr_at_fpr(scores, labels, 0.01),
        confusion=(tn, fp, fn, tp),
    )

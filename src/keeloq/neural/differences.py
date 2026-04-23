"""Chosen-plaintext-difference candidate search for Gohr-style distinguishers.

For each candidate Δ, train a tiny throwaway model for a short budget and
record validation accuracy. Candidates sorted by accuracy form the shortlist.
"""

from __future__ import annotations

from dataclasses import dataclass

from keeloq.neural.distinguisher import TrainingConfig, train


@dataclass(frozen=True)
class DeltaCandidate:
    delta: int
    validation_accuracy: float
    training_loss_final: float


# NLF tap positions from keeloq.cipher: state bits 31, 26, 20, 9, 1.
_TAP_POSITIONS = (31, 26, 20, 9, 1)


def _default_candidate_set() -> list[int]:
    hw1 = [1 << i for i in range(32)]
    extras: set[int] = set()
    for i in _TAP_POSITIONS:
        for j in _TAP_POSITIONS:
            if i != j:
                extras.add((1 << i) | (1 << j))
    out = list(hw1) + sorted(extras)
    seen: set[int] = set()
    uniq: list[int] = []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def search_delta(
    rounds: int,
    candidates: list[int] | None = None,
    tiny_budget_samples: int = 200_000,
    tiny_budget_epochs: int = 2,
    seed: int = 0,
) -> list[DeltaCandidate]:
    """Train a tiny model per candidate Δ and rank by validation accuracy."""
    if candidates is None:
        candidates = _default_candidate_set()
    seen: set[int] = set()
    uniq: list[int] = []
    for c in candidates:
        if c not in seen and 0 < c < (1 << 32):
            seen.add(c)
            uniq.append(c)

    results: list[DeltaCandidate] = []
    for i, delta in enumerate(uniq):
        cfg = TrainingConfig(
            rounds=rounds,
            delta=delta,
            n_samples=tiny_budget_samples,
            batch_size=1024,
            epochs=tiny_budget_epochs,
            lr=2e-3,
            weight_decay=1e-5,
            seed=seed + i * 7919,
            depth=2,
            width=16,
            val_samples=min(5_000, tiny_budget_samples // 2),
        )
        _, result = train(cfg)
        results.append(
            DeltaCandidate(
                delta=delta,
                validation_accuracy=result.final_val_accuracy,
                training_loss_final=result.final_loss,
            )
        )

    results.sort(key=lambda c: c.validation_accuracy, reverse=True)
    return results

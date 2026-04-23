"""Tests for keeloq.neural.differences candidate Δ search."""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@pytest.mark.gpu
def test_search_returns_sorted_candidates() -> None:
    from keeloq.neural.differences import DeltaCandidate, search_delta

    cands = search_delta(
        rounds=4,
        candidates=[0x80000000, 0x00000001, 0x40000000],
        tiny_budget_samples=4000,
        tiny_budget_epochs=1,
        seed=0,
    )
    assert all(isinstance(c, DeltaCandidate) for c in cands)
    accs = [c.validation_accuracy for c in cands]
    assert accs == sorted(accs, reverse=True)
    assert cands[0].validation_accuracy > 0.5


@pytest.mark.gpu
def test_search_deduplicates_candidates() -> None:
    from keeloq.neural.differences import search_delta

    cands = search_delta(
        rounds=4,
        candidates=[0x1, 0x1, 0x2],
        tiny_budget_samples=2000,
        tiny_budget_epochs=1,
        seed=0,
    )
    deltas = {c.delta for c in cands}
    assert len(deltas) == 2


@pytest.mark.gpu
def test_default_candidate_set_is_nonempty() -> None:
    from keeloq.neural.differences import _default_candidate_set

    cands = _default_candidate_set()
    assert len(cands) >= 32
    assert all(0 < c < (1 << 32) for c in cands)
    assert len(cands) == len(set(cands))

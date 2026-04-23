"""Tests for keeloq.neural.evaluation."""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@pytest.mark.gpu
def test_random_model_accuracy_near_half() -> None:
    from keeloq.neural.distinguisher import Distinguisher
    from keeloq.neural.evaluation import evaluate

    model = Distinguisher(depth=2, width=8).cuda()
    report = evaluate(model, rounds=8, delta=0x80000000,
                      n_samples=8192, seed=1)
    assert 0.45 <= report.accuracy <= 0.55


@pytest.mark.gpu
def test_evaluate_reproducible() -> None:
    from keeloq.neural.distinguisher import Distinguisher
    from keeloq.neural.evaluation import evaluate

    torch.manual_seed(42)
    model = Distinguisher(depth=2, width=8).cuda()
    r1 = evaluate(model, rounds=4, delta=0x1, n_samples=2048, seed=99)
    r2 = evaluate(model, rounds=4, delta=0x1, n_samples=2048, seed=99)
    assert r1.accuracy == r2.accuracy


@pytest.mark.gpu
def test_evaluate_after_training_has_signal() -> None:
    from keeloq.neural.distinguisher import TrainingConfig, train
    from keeloq.neural.evaluation import evaluate

    cfg = TrainingConfig(
        rounds=1, delta=0x80000000, n_samples=20_000, batch_size=512,
        epochs=2, lr=2e-3, weight_decay=1e-5, seed=0, depth=2, width=16,
    )
    model, _ = train(cfg)
    report = evaluate(model, rounds=1, delta=0x80000000,
                      n_samples=4096, seed=77)
    assert report.accuracy >= 0.85

"""Tests for keeloq.neural.distinguisher — architecture + training + checkpoints."""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@pytest.mark.gpu
def test_distinguisher_forward_shape() -> None:
    from keeloq.neural.distinguisher import Distinguisher

    model = Distinguisher(depth=3, width=16).cuda()
    x = torch.randint(0, 1 << 32, (32, 2), dtype=torch.int64).to(dtype=torch.uint32, device="cuda")
    y = model(x)
    assert y.shape == (32,)
    assert y.dtype == torch.float32
    assert (y >= 0).all() and (y <= 1).all()


@pytest.mark.gpu
def test_distinguisher_backward_runs() -> None:
    from keeloq.neural.distinguisher import Distinguisher

    model = Distinguisher(depth=2, width=8).cuda()
    x = torch.zeros(16, 2, dtype=torch.uint32, device="cuda")
    y = model(x)
    loss = y.sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None


@pytest.mark.gpu
def test_distinguisher_handles_large_batch() -> None:
    from keeloq.neural.distinguisher import Distinguisher

    model = Distinguisher(depth=3, width=16).cuda()
    x = torch.zeros(8192, 2, dtype=torch.uint32, device="cuda")
    y = model(x)
    assert y.shape == (8192,)


@pytest.mark.gpu
def test_distinguisher_param_count_sane() -> None:
    from keeloq.neural.distinguisher import Distinguisher

    model = Distinguisher()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 1_000_000 < n < 50_000_000, f"param count {n} out of range"


@pytest.mark.gpu
def test_train_reaches_high_acc_on_trivial_task() -> None:
    from keeloq.neural.distinguisher import TrainingConfig, train

    cfg = TrainingConfig(
        rounds=1,
        delta=0x80000000,
        n_samples=50_000,
        batch_size=1024,
        epochs=2,
        lr=2e-3,
        weight_decay=1e-5,
        seed=0,
        depth=2,
        width=16,
    )
    _model, result = train(cfg)
    assert result.final_val_accuracy >= 0.9, result
    assert result.final_loss < 0.5


@pytest.mark.gpu
def test_train_seed_reproducibility() -> None:
    from keeloq.neural.distinguisher import TrainingConfig, train

    cfg = TrainingConfig(
        rounds=1,
        delta=0x80000000,
        n_samples=8000,
        batch_size=256,
        epochs=1,
        lr=1e-3,
        weight_decay=1e-5,
        seed=99,
        depth=2,
        width=8,
    )
    m1, _ = train(cfg)
    m2, _ = train(cfg)
    for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=True):
        assert torch.allclose(p1, p2, atol=1e-5)


@pytest.mark.gpu
def test_checkpoint_round_trip(tmp_path) -> None:
    from keeloq.neural.distinguisher import (
        Distinguisher,
        TrainingConfig,
        TrainingResult,
        load_checkpoint,
        save_checkpoint,
    )

    model = Distinguisher(depth=2, width=8).cuda()
    cfg = TrainingConfig(
        rounds=1,
        delta=0x1,
        n_samples=0,
        batch_size=1,
        epochs=0,
        lr=0.0,
        weight_decay=0.0,
        seed=0,
        depth=2,
        width=8,
    )
    result = TrainingResult(
        final_loss=0.5,
        final_val_accuracy=0.9,
        wall_time_s=1.23,
        config=cfg,
        history=[],
    )

    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, result, path)
    m2, r2 = load_checkpoint(path)
    for p1, p2 in zip(model.parameters(), m2.parameters(), strict=True):
        assert torch.equal(p1.cpu(), p2.cpu())
    assert r2.config == cfg
    assert r2.final_val_accuracy == 0.9

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

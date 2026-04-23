"""Tests for keeloq.neural.data training pair generator."""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@pytest.mark.gpu
def test_generate_pairs_yields_training_batch() -> None:
    from keeloq.neural.data import TrainingBatch, generate_pairs

    gen = generate_pairs(rounds=16, delta=0x00000001, n_samples=4096, seed=0)
    batch = next(iter(gen))
    assert isinstance(batch, TrainingBatch)
    assert batch.pairs.shape == (4096, 2)
    assert batch.pairs.dtype == torch.uint32
    assert batch.labels.shape == (4096,)
    assert batch.labels.dtype == torch.float32


@pytest.mark.gpu
def test_generate_pairs_label_balance() -> None:
    from keeloq.neural.data import generate_pairs

    gen = generate_pairs(rounds=16, delta=0x00000001, n_samples=8192, seed=0)
    batch = next(iter(gen))
    ones = (batch.labels == 1.0).sum().item()
    zeros = (batch.labels == 0.0).sum().item()
    assert ones + zeros == 8192
    assert abs(ones - zeros) <= max(82, 8192 // 100)


@pytest.mark.gpu
def test_generate_pairs_seed_determinism() -> None:
    from keeloq.neural.data import generate_pairs

    b_a = next(iter(generate_pairs(rounds=16, delta=0x00000001, n_samples=256, seed=42)))
    b_b = next(iter(generate_pairs(rounds=16, delta=0x00000001, n_samples=256, seed=42)))
    assert torch.equal(b_a.pairs, b_b.pairs)
    assert torch.equal(b_a.labels, b_b.labels)


@pytest.mark.gpu
def test_generate_pairs_batch_chunking() -> None:
    from keeloq.neural.data import generate_pairs

    it = generate_pairs(rounds=16, delta=0x00000001, n_samples=1024, seed=0, batch_size=256)
    batches = list(it)
    assert len(batches) == 4
    total = sum(b.pairs.shape[0] for b in batches)
    assert total == 1024

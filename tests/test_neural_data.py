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


@pytest.mark.gpu
def test_real_pair_satisfies_delta_invariant() -> None:
    import torch

    from keeloq.cipher import encrypt as cpu_encrypt
    from keeloq.neural import data as data_mod
    from keeloq.neural.data import generate_pairs

    captured: dict[str, list] = {}
    orig = data_mod.encrypt_batch

    def spy(plaintexts, keys, rounds):
        captured.setdefault("pts", []).append(plaintexts.clone())
        captured.setdefault("keys", []).append(keys.clone())
        return orig(plaintexts, keys, rounds=rounds)

    data_mod.encrypt_batch = spy
    try:
        batch = next(
            iter(generate_pairs(rounds=16, delta=0x0000FFFF, n_samples=128, seed=7, batch_size=128))
        )
    finally:
        data_mod.encrypt_batch = orig

    p0, p1 = captured["pts"][0], captured["pts"][1]
    half = 64
    assert torch.all(
        (p0[:half].to(torch.int64) & 0xFFFFFFFF) ^ (p1[:half].to(torch.int64) & 0xFFFFFFFF)
        == int(torch.tensor(0x0000FFFF, dtype=torch.uint32).item())
    )

    p0_0 = int(p0[0].item()) & 0xFFFFFFFF
    k_lo = int(captured["keys"][0][0, 0].item()) & 0xFFFFFFFF
    k_hi = int(captured["keys"][0][0, 1].item()) & 0xFFFFFFFF
    k = (k_hi << 32) | k_lo
    assert int(batch.pairs[0, 0].item()) & 0xFFFFFFFF == cpu_encrypt(p0_0, k, 16)

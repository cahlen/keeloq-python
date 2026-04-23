"""Tests for keeloq.gpu_cipher bit-sliced CUDA cipher."""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from keeloq.cipher import encrypt as cpu_encrypt


@pytest.mark.gpu
@pytest.mark.parametrize("rounds", [16, 32, 64, 128, 160])
def test_gpu_matches_cpu_random_batch(rounds: int) -> None:
    from keeloq.gpu_cipher import encrypt_batch

    assert torch is not None
    n = 4096  # small batch for unit test; property test handles the millions
    gen = torch.Generator(device="cpu").manual_seed(1234 + rounds)
    plaintexts_cpu = torch.randint(0, 1 << 32, (n,), generator=gen, dtype=torch.int64)
    keys_lo_cpu = torch.randint(0, 1 << 32, (n,), generator=gen, dtype=torch.int64)
    keys_hi_cpu = torch.randint(0, 1 << 32, (n,), generator=gen, dtype=torch.int64)

    plaintexts = plaintexts_cpu.to(dtype=torch.uint32, device="cuda")
    keys = torch.stack(
        [keys_lo_cpu.to(dtype=torch.uint32), keys_hi_cpu.to(dtype=torch.uint32)], dim=1
    ).to("cuda")

    gpu_out = encrypt_batch(plaintexts, keys, rounds=rounds).cpu().tolist()

    for i in range(n):
        pt = int(plaintexts_cpu[i].item()) & 0xFFFFFFFF
        k = (int(keys_hi_cpu[i].item()) & 0xFFFFFFFF) << 32 | (
            int(keys_lo_cpu[i].item()) & 0xFFFFFFFF
        )
        assert gpu_out[i] == cpu_encrypt(pt, k, rounds), (
            f"mismatch at i={i}: pt=0x{pt:08x} key=0x{k:016x} rounds={rounds}"
        )


def test_gpu_cipher_raises_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    from keeloq import gpu_cipher

    if torch is None:
        pytest.skip("torch not installed")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        gpu_cipher.encrypt_batch(
            torch.tensor([0], dtype=torch.uint32),
            torch.tensor([[0, 0]], dtype=torch.uint32),
            rounds=1,
        )


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_matches_cpu_million_pairs() -> None:
    """The GPU-as-oracle fuzz test — 10^6 (pt, key) pairs per round count."""
    from keeloq.gpu_cipher import encrypt_batch

    assert torch is not None
    n = 1_000_000
    gen = torch.Generator(device="cpu").manual_seed(2026)
    plaintexts_cpu = torch.randint(0, 1 << 32, (n,), generator=gen, dtype=torch.int64)
    keys_lo_cpu = torch.randint(0, 1 << 32, (n,), generator=gen, dtype=torch.int64)
    keys_hi_cpu = torch.randint(0, 1 << 32, (n,), generator=gen, dtype=torch.int64)

    plaintexts = plaintexts_cpu.to(dtype=torch.uint32, device="cuda")
    keys = torch.stack(
        [keys_lo_cpu.to(dtype=torch.uint32), keys_hi_cpu.to(dtype=torch.uint32)], dim=1
    ).to("cuda")

    # Sample 1024 indices randomly for CPU comparison; GPU computes all N.
    gpu_out = encrypt_batch(plaintexts, keys, rounds=160).cpu().tolist()

    import random

    rng = random.Random(2026)
    sample_ix = rng.sample(range(n), 1024)
    for i in sample_ix:
        pt = int(plaintexts_cpu[i].item()) & 0xFFFFFFFF
        k = (int(keys_hi_cpu[i].item()) & 0xFFFFFFFF) << 32 | (
            int(keys_lo_cpu[i].item()) & 0xFFFFFFFF
        )
        assert gpu_out[i] == cpu_encrypt(pt, k, 160), (
            f"mismatch at i={i}: pt=0x{pt:08x} key=0x{k:016x}"
        )

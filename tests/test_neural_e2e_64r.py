"""Phase 3b 64-round hybrid attack using committed d64.pt (floor commitment)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "d64.pt"


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not CKPT.exists(), reason="checkpoints/d64.pt absent")
def test_64_round_full_key_recovery() -> None:
    from keeloq.gpu_cipher import encrypt_batch
    from keeloq.neural.distinguisher import load_checkpoint
    from keeloq.neural.hybrid import hybrid_attack

    model, meta = load_checkpoint(CKPT)
    delta = meta.config.delta
    trained_depth = meta.config.rounds
    assert trained_depth < 64

    target_key = 0xFEDC_BA98_7654_3210
    attack_depth = 64
    neural_bits = attack_depth - trained_depth

    n_pairs = 512
    gen = torch.Generator(device="cpu").manual_seed(31337)
    pts0 = torch.randint(0, 1 << 32, (n_pairs,), generator=gen,
                         dtype=torch.int64).to(dtype=torch.uint32, device="cuda")
    pts0_cpu = pts0.cpu()
    delta_t = torch.tensor(delta, dtype=torch.uint32)
    pts1 = (pts0_cpu ^ delta_t).to("cuda")
    keys = torch.tensor(
        [[target_key & 0xFFFFFFFF, (target_key >> 32) & 0xFFFFFFFF]] * n_pairs,
        dtype=torch.uint32, device="cuda",
    )
    c0 = encrypt_batch(pts0, keys, rounds=attack_depth)
    c1 = encrypt_batch(pts1, keys, rounds=attack_depth)
    diff_pairs = [(int(c0[i].item()) & 0xFFFFFFFF,
                   int(c1[i].item()) & 0xFFFFFFFF) for i in range(n_pairs)]
    sat_pairs = [(int(pts0_cpu[i].item()) & 0xFFFFFFFF,
                  int(c0[i].item()) & 0xFFFFFFFF) for i in range(n_pairs)]

    t0 = time.perf_counter()
    result = hybrid_attack(
        rounds=attack_depth,
        pairs=diff_pairs[:16],
        sat_pairs=sat_pairs[:8],
        distinguisher=model,
        beam_width=16,
        neural_target_bits=neural_bits,
        sat_timeout_s=120.0,
        max_backtracks=16,
    )
    wall = time.perf_counter() - t0
    assert wall < 5 * 60, f"too slow: {wall:.1f}s"
    assert result.status == "SUCCESS", (
        f"status={result.status} neural_bits={result.bits_recovered_neurally}"
    )
    assert result.recovered_key == target_key

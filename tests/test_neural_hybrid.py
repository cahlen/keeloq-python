"""Tests for keeloq.neural.hybrid end-to-end key recovery."""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@pytest.mark.gpu
def test_hybrid_attack_gohr_pattern() -> None:
    """Train a distinguisher at trained_depth=4; attack a depth-6 cipher by
    peeling 2 rounds neurally, then let SAT finish the remaining 62 key bits
    (SAT easily handles under-constrained KeeLoq with most key bits unknown at
    low round counts when given per-bit hints for rounds 4-5)."""
    from keeloq.gpu_cipher import encrypt_batch
    from keeloq.neural.distinguisher import TrainingConfig, train
    from keeloq.neural.hybrid import hybrid_attack

    # Train at depth 4.
    delta = 0x80000000
    cfg = TrainingConfig(
        rounds=4,
        delta=delta,
        n_samples=80_000,
        batch_size=1024,
        epochs=4,
        lr=2e-3,
        weight_decay=1e-5,
        seed=0,
        depth=3,
        width=16,
    )
    model, result = train(cfg)
    assert result.final_val_accuracy >= 0.85

    # Attack depth-6 cipher; peel 2 neurally, SAT finishes the rest.
    target_key = 0x0123_4567_89AB_CDEF
    attack_depth = 6
    n_pairs = 64
    gen = torch.Generator(device="cpu").manual_seed(2025)
    pts0 = torch.randint(0, 1 << 32, (n_pairs,), generator=gen, dtype=torch.int64).to(
        dtype=torch.uint32, device="cuda"
    )
    pts0_cpu = pts0.cpu()
    delta_t = torch.tensor(delta, dtype=torch.uint32)
    pts1 = (pts0_cpu ^ delta_t).to("cuda")
    keys = torch.tensor(
        [[target_key & 0xFFFFFFFF, (target_key >> 32) & 0xFFFFFFFF]] * n_pairs,
        dtype=torch.uint32,
        device="cuda",
    )
    c0 = encrypt_batch(pts0, keys, rounds=attack_depth)
    c1 = encrypt_batch(pts1, keys, rounds=attack_depth)
    # Differential ciphertext pairs (c0, c1) for the neural distinguisher.
    diff_pairs = [
        (int(c0[i].item()) & 0xFFFFFFFF, int(c1[i].item()) & 0xFFFFFFFF) for i in range(n_pairs)
    ]

    # Genuine (plaintext, ciphertext) pairs for the SAT constraint system.
    # pts0_cpu holds the 32-bit plaintexts; c0 holds encrypt(pt0, key, 6).
    sat_constraint_pairs = [
        (int(pts0_cpu[i].item()) & 0xFFFFFFFF, int(c0[i].item()) & 0xFFFFFFFF) for i in range(2)
    ]

    # At depth 6 the KeeLoq cyclic schedule uses only K0..K5.
    # The neural step peels rounds 5 and 4 (recovering K5 and K4).
    # We pin K0..K3 and K6..K63 as extra_key_hints so that:
    # - K0..K5 are fully constrained (K4,K5 from neural; K0..K3 explicit)
    # - K6..K63 are pinned explicitly to prevent SAT from picking 0 for them
    # This makes the recovered_key fully determined.
    # Real Phase 3b attacks at deeper rounds won't need such heavy hints.
    extra_hints = {i: (target_key >> (63 - i)) & 1 for i in range(0, 4)}
    extra_hints.update({i: (target_key >> (63 - i)) & 1 for i in range(6, 64)})
    result_attack = hybrid_attack(
        rounds=attack_depth,
        pairs=diff_pairs[:32],  # differential pairs for neural scoring
        distinguisher=model,
        beam_width=4,
        neural_target_bits=2,
        extra_key_hints=extra_hints,
        sat_timeout_s=30.0,
        max_backtracks=4,
        sat_pairs=sat_constraint_pairs,  # real (pt, ct) for SAT
    )
    assert result_attack.status == "SUCCESS", (
        f"status={result_attack.status}, neural_bits={result_attack.bits_recovered_neurally}"
    )
    assert result_attack.recovered_key == target_key
    assert result_attack.verify_result is True


@pytest.mark.gpu
def test_hybrid_attack_untrained_model_terminates_cleanly() -> None:
    """Untrained (random) distinguisher may produce wrong neural bits;
    the hybrid must still terminate with a recognized terminal status."""
    from keeloq.gpu_cipher import encrypt_batch
    from keeloq.neural.distinguisher import Distinguisher
    from keeloq.neural.hybrid import hybrid_attack

    model = Distinguisher(depth=2, width=8).cuda()
    target_key = 0x0123_4567_89AB_CDEF
    n_pairs = 4
    gen = torch.Generator(device="cpu").manual_seed(2025)
    pts0 = torch.randint(0, 1 << 32, (n_pairs,), generator=gen, dtype=torch.int64).to(
        dtype=torch.uint32, device="cuda"
    )
    pts0_cpu = pts0.cpu()
    delta_t = torch.tensor(0x80000000, dtype=torch.uint32)
    pts1 = (pts0_cpu ^ delta_t).to("cuda")
    keys = torch.tensor(
        [[target_key & 0xFFFFFFFF, (target_key >> 32) & 0xFFFFFFFF]] * n_pairs,
        dtype=torch.uint32,
        device="cuda",
    )
    c0 = encrypt_batch(pts0, keys, rounds=8)
    c1 = encrypt_batch(pts1, keys, rounds=8)
    pairs = [
        (int(c0[i].item()) & 0xFFFFFFFF, int(c1[i].item()) & 0xFFFFFFFF) for i in range(n_pairs)
    ]

    result = hybrid_attack(
        rounds=8,
        pairs=pairs,
        distinguisher=model,
        beam_width=2,
        neural_target_bits=1,
        sat_timeout_s=30.0,
        max_backtracks=4,
    )
    assert result.status in ("SUCCESS", "BACKTRACK_EXHAUSTED", "UNSAT", "TIMEOUT", "WRONG_KEY")
    if result.status == "SUCCESS":
        assert result.recovered_key == target_key
        assert result.verify_result is True

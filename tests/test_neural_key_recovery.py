"""Tests for keeloq.neural.key_recovery."""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from keeloq.cipher import encrypt


@pytest.mark.gpu
def test_peel_one_round_equals_shorter_encryption() -> None:
    """Peeling 1 round from N-round ct equals the (N-1)-round ct of the same pt."""
    from keeloq.neural.key_recovery import partial_decrypt_round

    key = 0x0123_4567_89AB_CDEF
    pt = 0xAAAA5555
    n_rounds = 16
    ct_full = encrypt(pt, key, n_rounds)
    ct_short = encrypt(pt, key, n_rounds - 1)

    true_kbit = (key >> (63 - ((n_rounds - 1) % 64))) & 1
    batch = torch.tensor([ct_full], dtype=torch.uint32, device="cuda")
    peeled = partial_decrypt_round(batch, key_bit=true_kbit, round_idx=n_rounds - 1)
    assert int(peeled[0].item()) & 0xFFFFFFFF == ct_short


@pytest.mark.gpu
def test_wrong_key_bit_disagrees() -> None:
    from keeloq.neural.key_recovery import partial_decrypt_round

    key = 0x0123_4567_89AB_CDEF
    pt = 0xAAAA5555
    n_rounds = 16
    ct = encrypt(pt, key, n_rounds)
    true_kbit = (key >> (63 - ((n_rounds - 1) % 64))) & 1
    wrong = 1 - true_kbit

    batch = torch.tensor([ct], dtype=torch.uint32, device="cuda")
    right = partial_decrypt_round(batch, key_bit=true_kbit, round_idx=n_rounds - 1)
    wrongr = partial_decrypt_round(batch, key_bit=wrong, round_idx=n_rounds - 1)
    assert int(right[0].item()) != int(wrongr[0].item())


@pytest.mark.gpu
def test_recover_prefix_gohr_pattern() -> None:
    """Gohr-style toy attack: train a distinguisher at depth D, attack a cipher
    at depth D+K by peeling K rounds back down to the trained depth.

    A single distinguisher trained at depth N doesn't generalize to (N-k)-round
    pairs, so the Gohr attack peels rounds from an M-round ciphertext until the
    residual depth matches the trained distinguisher. Here we train at D=4 and
    attack depth D+K=6 by peeling K=2 rounds, so the final scored pairs are at
    depth 4 — the distinguisher's trained depth.
    """
    from keeloq.gpu_cipher import encrypt_batch
    from keeloq.neural.distinguisher import TrainingConfig, train
    from keeloq.neural.key_recovery import recover_prefix

    # Train a distinguisher AT depth 4 (not at the attack depth).
    delta = 0x80000000
    trained_depth = 4
    cfg = TrainingConfig(
        rounds=trained_depth,
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
    # Very shallow KeeLoq is easy — accuracy should be high.
    assert result.final_val_accuracy >= 0.85, (
        f"4-round distinguisher only reached {result.final_val_accuracy:.3f}"
    )

    # Attack a depth-6 cipher by peeling 2 rounds back to depth 4.
    attack_depth = 6
    bits_to_recover = attack_depth - trained_depth  # = 2

    target_key = 0xDEADBEEF_CAFE1234 & ((1 << 64) - 1)
    n_pairs = 256
    gen = torch.Generator(device="cpu").manual_seed(2024)
    pts0 = torch.randint(0, 1 << 32, (n_pairs,), generator=gen, dtype=torch.int64).to(
        dtype=torch.uint32, device="cuda"
    )
    # CUDA uint32 XOR isn't supported; do it on CPU first.
    pts0_cpu = pts0.cpu()
    delta_t = torch.tensor(delta, dtype=torch.uint32)
    pts1_cpu = pts0_cpu ^ delta_t
    pts1 = pts1_cpu.to("cuda")

    keys = torch.tensor(
        [[target_key & 0xFFFFFFFF, (target_key >> 32) & 0xFFFFFFFF]] * n_pairs,
        dtype=torch.uint32,
        device="cuda",
    )
    c0 = encrypt_batch(pts0, keys, rounds=attack_depth)
    c1 = encrypt_batch(pts1, keys, rounds=attack_depth)
    pairs = [
        (int(c0[i].item()) & 0xFFFFFFFF, int(c1[i].item()) & 0xFFFFFFFF) for i in range(n_pairs)
    ]

    rec = recover_prefix(
        pairs=pairs,
        distinguisher=model,
        starting_rounds=attack_depth,
        max_bits_to_recover=bits_to_recover,
        beam_width=4,
    )
    for bit_idx, recovered_val in rec.recovered_bits.items():
        true_val = (target_key >> (63 - bit_idx)) & 1
        assert recovered_val == true_val, f"K{bit_idx}: got {recovered_val}, want {true_val}"

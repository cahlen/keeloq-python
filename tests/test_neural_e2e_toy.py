"""Phase 3b smoke test — train at 4 rounds, attack at 6 rounds (Gohr pattern).

CI-eligible: @pytest.mark.gpu only (not slow-tagged). Training a 4-round
distinguisher on ~40k samples takes seconds on any RTX GPU.

Deviations from the raw plan text (Task 14 as written trained/attacked at 8r):
- Train at depth 4, attack depth 6, peel 2 bits.
- Use two separate pair args: ``pairs`` for the neural distinguisher (differential
  ciphertext pairs) and ``sat_pairs`` for the SAT constraint system (genuine
  plaintext/ciphertext pairs).
- Delta XOR is computed on CPU (CUDA lacks uint32 XOR between tensors of that dtype
  without going via int64).
- ``extra_key_hints`` pins all key bits outside the neural-recovered range so that
  SAT uniquely determines the key.
"""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@pytest.mark.gpu
def test_phase3b_end_to_end_toy() -> None:
    """Train a 4-round distinguisher; attack a 6-round cipher peeling 2 bits.

    Step 1 — train:  4-round distinguisher, ~40 k samples, must reach >= 0.85 acc.
    Step 2 — eval:   8 k fresh samples on the 4-round cipher.
    Step 3 — attack: hybrid_attack on 6-round cipher, neural_target_bits=2,
                     extra hints pin all bits outside K4..K5 so SAT finds the key.
    """
    from keeloq.gpu_cipher import encrypt_batch
    from keeloq.neural.distinguisher import TrainingConfig, train
    from keeloq.neural.evaluation import evaluate
    from keeloq.neural.hybrid import hybrid_attack

    # ------------------------------------------------------------------
    # Hyper-params
    # ------------------------------------------------------------------
    train_rounds = 4
    attack_rounds = 6  # Gohr pattern: attack at D + K (here K=2)
    neural_peel_bits = 2  # peel the 2 outermost rounds (rounds 4 and 5)
    delta = 0x80000000  # single-bit difference, best for shallow rounds
    target_key = 0x0123_4567_89AB_CDEF

    # ------------------------------------------------------------------
    # Step 1: train at depth 4
    # ------------------------------------------------------------------
    cfg = TrainingConfig(
        rounds=train_rounds,
        delta=delta,
        n_samples=40_000,
        batch_size=512,
        epochs=5,
        lr=2e-3,
        weight_decay=1e-5,
        seed=42,
        depth=3,
        width=16,
        val_samples=8_000,
    )
    model, train_result = train(cfg)

    # ------------------------------------------------------------------
    # Step 2: evaluate on 8 k fresh samples (same round depth as training)
    # ------------------------------------------------------------------
    report = evaluate(
        model,
        rounds=train_rounds,
        delta=delta,
        n_samples=8_000,
        seed=999,
        batch_size=1024,
    )
    assert report.accuracy >= 0.85, (
        f"Distinguisher accuracy {report.accuracy:.4f} < 0.85 threshold. "
        f"Training result: val_acc={train_result.final_val_accuracy:.4f}, "
        f"loss={train_result.final_loss:.4f}"
    )

    # ------------------------------------------------------------------
    # Step 3: build attack pairs for 6-round cipher
    # ------------------------------------------------------------------
    n_pairs = 64
    gen = torch.Generator(device="cpu").manual_seed(20260422)

    # Plaintexts on CPU so we can do the uint32 XOR without CUDA dtype issues.
    pts0_cpu = torch.randint(0, 1 << 32, (n_pairs,), generator=gen, dtype=torch.int64).to(
        dtype=torch.uint32
    )
    delta_t = torch.tensor(delta, dtype=torch.uint32)
    pts1_cpu = pts0_cpu ^ delta_t  # CPU-side XOR avoids CUDA uint32 op issues

    pts0 = pts0_cpu.to("cuda")
    pts1 = pts1_cpu.to("cuda")

    keys = torch.tensor(
        [[target_key & 0xFFFFFFFF, (target_key >> 32) & 0xFFFFFFFF]] * n_pairs,
        dtype=torch.uint32,
        device="cuda",
    )

    c0 = encrypt_batch(pts0, keys, rounds=attack_rounds)
    c1 = encrypt_batch(pts1, keys, rounds=attack_rounds)

    # Differential ciphertext pairs — fed to the neural distinguisher.
    diff_pairs = [
        (int(c0[i].item()) & 0xFFFFFFFF, int(c1[i].item()) & 0xFFFFFFFF) for i in range(n_pairs)
    ]

    # Genuine (plaintext, ciphertext) pairs — fed to the SAT solver.
    # Two pairs are enough to over-constrain 6-round KeeLoq when all but 2
    # key bits are pinned via extra_key_hints.
    sat_pairs = [
        (int(pts0_cpu[i].item()) & 0xFFFFFFFF, int(c0[i].item()) & 0xFFFFFFFF) for i in range(2)
    ]

    # ------------------------------------------------------------------
    # Step 4: build extra_key_hints
    #
    # KeeLoq key schedule: round i uses _key_bit(key, i % 64), MSB-first.
    # Attack depth 6 → rounds 0-5 → key bits K0..K5 (cyclic positions).
    # Neural peeling undoes rounds 5 and 4 → recovers K5 and K4.
    # We pin K0..K3 (needed by SAT but not recovered neurally) and K6..K63
    # (outside the attack window, SAT would otherwise set them to 0 and
    # return the wrong key).
    # ------------------------------------------------------------------
    def _msb_key_bit(key: int, pos: int) -> int:
        """MSB-first bit at position pos of a 64-bit key."""
        return (key >> (63 - pos)) & 1

    extra_hints: dict[int, int] = {}
    # K0..K3: the SAT window uses these but they are not peeled neurally.
    for i in range(0, attack_rounds - neural_peel_bits):
        extra_hints[i] = _msb_key_bit(target_key, i)
    # K6..K63: outside the attack-round window; pin so SAT finds the real key.
    for i in range(attack_rounds, 64):
        extra_hints[i] = _msb_key_bit(target_key, i)

    # ------------------------------------------------------------------
    # Step 5: run hybrid attack
    # ------------------------------------------------------------------
    result = hybrid_attack(
        rounds=attack_rounds,
        pairs=diff_pairs[:32],  # differential pairs for neural scoring
        distinguisher=model,
        beam_width=4,
        neural_target_bits=neural_peel_bits,
        extra_key_hints=extra_hints,
        sat_timeout_s=30.0,
        max_backtracks=4,
        sat_pairs=sat_pairs,  # genuine (pt, ct) pairs for SAT
    )

    assert result.status == "SUCCESS", (
        f"hybrid_attack status={result.status!r}, "
        f"bits_recovered_neurally={result.bits_recovered_neurally}"
    )
    assert result.recovered_key == target_key, (
        f"recovered_key=0x{result.recovered_key:016x} != "  # type: ignore[str-format]
        f"target=0x{target_key:016x}"
    )
    assert result.verify_result is True

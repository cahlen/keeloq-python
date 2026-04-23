"""Sequential Bayesian key-bit recovery over a neural distinguisher.

Core primitive: `partial_decrypt_round`, a GPU-batched version of Phase 1's
algebraic one-round inverse (cipher.decrypt's inner body). Given a batch of
32-bit states and a single key-bit guess, it peels one round off.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from keeloq.neural.distinguisher import Distinguisher


def _get_bit(state: torch.Tensor, msb_pos: int, width: int = 32) -> torch.Tensor:
    one = torch.tensor(1, dtype=state.dtype, device=state.device)
    return (state >> (width - 1 - msb_pos)) & one


def _core_batched(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
) -> torch.Tensor:
    """Batched KeeLoq NLF on 0/1 int64 tensors."""
    return (
        d
        ^ e
        ^ (a & c)
        ^ (a & e)
        ^ (b & c)
        ^ (b & e)
        ^ (c & d)
        ^ (d & e)
        ^ (a & d & e)
        ^ (a & c & e)
        ^ (a & b & d)
        ^ (a & b & c)
    )


def partial_decrypt_round(
    state: torch.Tensor,  # (N,) uint32
    key_bit: int,
    round_idx: int,
) -> torch.Tensor:
    """Undo the encryption round `round_idx` under a guessed `key_bit`.

    Mirrors keeloq.cipher.decrypt's inner body. Works on int64 internally
    because CUDA doesn't implement rshift for uint32.

    `round_idx` is retained in the signature for forward-compatibility but
    is not consumed: KeeLoq's round function uses only the step's key bit
    and the current state; the round number only determines which key bit
    applies, which is the caller's responsibility.
    """
    if state.dtype != torch.uint32:
        raise ValueError("state must be uint32")
    if key_bit not in (0, 1):
        raise ValueError("key_bit must be 0 or 1")
    del round_idx

    s64 = state.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    p_last = _get_bit(s64, 31)
    pre_bit_0 = (
        p_last
        ^ torch.tensor(key_bit, dtype=torch.int64, device=s64.device)
        ^ _get_bit(s64, 15)
        ^ _core_batched(
            _get_bit(s64, 30),
            _get_bit(s64, 25),
            _get_bit(s64, 19),
            _get_bit(s64, 8),
            _get_bit(s64, 0),
        )
    )
    unshifted = ((s64 >> 1) & 0x7FFFFFFF) | (pre_bit_0 << 31)
    return (unshifted & 0xFFFFFFFF).to(torch.int32).view(torch.uint32)


@dataclass(frozen=True)
class BeamEntry:
    recovered_bits: tuple[tuple[int, int], ...]  # ordered (bit_idx, value)
    log_evidence: float
    pairs_state: tuple[torch.Tensor, torch.Tensor]  # (c0, c1) peeled state


@dataclass(frozen=True)
class RecoveryResult:
    recovered_bits: dict[int, int]
    terminated_at_depth: int
    beam_history: list[list[float]]
    distinguisher_margin_history: list[float]


def _batch_score(
    model: Distinguisher,
    c0: torch.Tensor,
    c1: torch.Tensor,
    max_batch: int = 16384,
) -> torch.Tensor:
    """Score all (c0[i], c1[i]) pairs via the distinguisher. No grad."""
    scores: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, c0.shape[0], max_batch):
            end = start + max_batch
            x = torch.stack([c0[start:end], c1[start:end]], dim=1)
            scores.append(model(x).detach())
    return torch.cat(scores)


def recover_prefix(
    pairs: list[tuple[int, int]],
    distinguisher: Distinguisher,
    starting_rounds: int,
    max_bits_to_recover: int,
    beam_width: int = 8,
    margin_floor: float = 0.02,
    max_beam_width: int = 256,
) -> RecoveryResult:
    """Peel `max_bits_to_recover` rounds from the outside in via beam search."""
    device = torch.device("cuda")
    c0 = torch.tensor([p[0] for p in pairs], dtype=torch.uint32, device=device)
    c1 = torch.tensor([p[1] for p in pairs], dtype=torch.uint32, device=device)

    beam: list[BeamEntry] = [BeamEntry(recovered_bits=(), log_evidence=0.0, pairs_state=(c0, c1))]
    margin_history: list[float] = []
    beam_history: list[list[float]] = []
    distinguisher.train(False)  # inference mode (equivalent to classic eval-mode API)
    current_width = beam_width

    try:
        for step in range(max_bits_to_recover):
            enc_step = starting_rounds - 1 - step
            if enc_step < 0:
                break
            bit_idx = enc_step % 64

            candidates: list[BeamEntry] = []
            for entry in beam:
                for guess in (0, 1):
                    c0_new = partial_decrypt_round(entry.pairs_state[0], guess, enc_step)
                    c1_new = partial_decrypt_round(entry.pairs_state[1], guess, enc_step)
                    scores = _batch_score(distinguisher, c0_new, c1_new)
                    log_ev = float(torch.log(scores.clamp(min=1e-8, max=1 - 1e-8)).sum().item())
                    candidates.append(
                        BeamEntry(
                            recovered_bits=(*entry.recovered_bits, (bit_idx, guess)),
                            log_evidence=entry.log_evidence + log_ev,
                            pairs_state=(c0_new, c1_new),
                        )
                    )

            candidates.sort(key=lambda e: e.log_evidence, reverse=True)
            if len(candidates) >= 2:
                a, b = candidates[0], candidates[1]
                denom = max(abs(a.log_evidence), abs(b.log_evidence), 1.0)
                margin = (a.log_evidence - b.log_evidence) / denom
                margin_history.append(margin)
                if margin < margin_floor and current_width < max_beam_width:
                    current_width = min(current_width * 2, max_beam_width)

            beam = candidates[:current_width]
            beam_history.append([e.log_evidence for e in beam])
    finally:
        distinguisher.train(True)

    best = beam[0]
    recovered = {bi: bv for (bi, bv) in best.recovered_bits}
    return RecoveryResult(
        recovered_bits=recovered,
        terminated_at_depth=starting_rounds - len(best.recovered_bits),
        beam_history=beam_history,
        distinguisher_margin_history=margin_history,
    )

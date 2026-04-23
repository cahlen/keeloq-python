"""Sequential Bayesian key-bit recovery over a neural distinguisher.

Core primitive: `partial_decrypt_round`, a GPU-batched version of Phase 1's
algebraic one-round inverse (cipher.decrypt's inner body). Given a batch of
32-bit states and a single key-bit guess, it peels one round off.
"""

from __future__ import annotations

import torch


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

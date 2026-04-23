"""GPU bit-sliced KeeLoq cipher for high-throughput property tests.

Processes a batch of (plaintext, key) pairs in lockstep using bitwise ops on
32-bit lanes. This is not a "fast cipher" in the usual sense — the batch
dimension is what gets parallelized, not the bits within one encryption.

Note: CUDA does not support shift operators on uint32, so we work internally
with int64 and mask to 32 bits. The public API still accepts/returns uint32.
"""

from __future__ import annotations

import torch


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "gpu_cipher requires CUDA. Install a CUDA-enabled torch build and "
            "ensure an NVIDIA GPU is visible."
        )


def _u32_to_i64(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret uint32 tensor as int64, zero-extending (unsigned semantics)."""
    # view as int32 then cast to int64; mask out sign extension via & 0xFFFFFFFF
    return t.view(torch.int32).to(torch.int64) & 0xFFFFFFFF


def _get_bit(state: torch.Tensor, msb_pos: int, width: int = 32) -> torch.Tensor:
    """Extract MSB-indexed bit `msb_pos` from each element of `state` (int64 tensor)."""
    return (state >> (width - 1 - msb_pos)) & 1


def _core(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor, e: torch.Tensor
) -> torch.Tensor:
    """Batched KeeLoq NLF. Inputs are 0/1 int64 tensors of identical shape."""
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


def encrypt_batch(
    plaintexts: torch.Tensor,
    keys: torch.Tensor,
    rounds: int,
) -> torch.Tensor:
    """Encrypt a batch of plaintext/key pairs.

    Args:
        plaintexts: shape (N,), dtype=torch.uint32. Each element is a 32-bit plaintext.
        keys: shape (N, 2), dtype=torch.uint32. Column 0 = low 32 bits, column 1 = high 32.
        rounds: integer >= 0.

    Returns:
        shape (N,), dtype=torch.uint32. Ciphertexts.
    """
    _require_cuda()
    if rounds < 0:
        raise ValueError(f"rounds={rounds} is negative")
    if plaintexts.dtype != torch.uint32:
        raise ValueError("plaintexts must be uint32")
    if keys.shape[-1] != 2 or keys.dtype != torch.uint32:
        raise ValueError("keys must have shape (N, 2) and dtype uint32")
    if plaintexts.shape[0] != keys.shape[0]:
        raise ValueError("batch size mismatch between plaintexts and keys")

    # Work in int64 internally; CUDA does not implement rshift/lshift for uint32.
    state = _u32_to_i64(plaintexts.to("cuda"))
    key_lo = _u32_to_i64(keys[:, 0].to("cuda"))
    key_hi = _u32_to_i64(keys[:, 1].to("cuda"))
    mask32: int = 0xFFFFFFFF

    for i in range(rounds):
        # Key bit at MSB-first index (i % 64).
        # key layout: key_hi holds MSB-first indices 0..31, key_lo holds 32..63.
        idx = i % 64
        kbit = (key_hi >> (31 - idx)) & 1 if idx < 32 else (key_lo >> (63 - idx)) & 1

        p0 = _get_bit(state, 0)
        p1 = _get_bit(state, 1)
        p9 = _get_bit(state, 9)
        p16 = _get_bit(state, 16)
        p20 = _get_bit(state, 20)
        p26 = _get_bit(state, 26)
        p31 = _get_bit(state, 31)

        newb = (kbit ^ p0 ^ p16 ^ _core(p31, p26, p20, p9, p1)) & 1
        state = ((state << 1) & mask32) | newb

    # Convert back to uint32 for output
    return state.to(torch.int32).view(torch.uint32)

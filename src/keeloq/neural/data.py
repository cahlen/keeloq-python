"""Streaming training-data generator for neural distinguishers.

For each sample position i in a batch:
  - First half of batch (label=1.0, "real"): random key k_i, random p0_i,
    p1_i = p0_i ^ delta, both encrypted under k_i.
  - Second half (label=0.0, "random"): same k_i for p0_i, independent k'_i
    for p1_i, independent p1_i. So p0 ^ p1 is effectively random.

Batches are yielded on CUDA. Callers move to CPU only for metrics.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from keeloq.gpu_cipher import encrypt_batch


@dataclass(frozen=True)
class TrainingBatch:
    pairs: torch.Tensor  # (N, 2) uint32 — (c_0, c_1) per row
    labels: torch.Tensor  # (N,) float32 — 1.0 real, 0.0 random


def _sample_uint32_cpu(n: int, gen: torch.Generator) -> torch.Tensor:
    """Draw n uniform uint32 values on CPU via int64 RNG, cast preserving bit pattern."""
    return torch.randint(0, 1 << 32, (n,), generator=gen, dtype=torch.int64).to(dtype=torch.uint32)


def _sample_keys_cpu(n: int, gen: torch.Generator) -> torch.Tensor:
    """(N, 2) uint32 key tensors on CPU: col 0 low 32 bits, col 1 high 32 bits."""
    lo = _sample_uint32_cpu(n, gen)
    hi = _sample_uint32_cpu(n, gen)
    return torch.stack([lo, hi], dim=1)


def generate_pairs(
    rounds: int,
    delta: int,
    n_samples: int,
    seed: int,
    batch_size: int = 65536,
) -> Iterator[TrainingBatch]:
    """Yield TrainingBatches totaling n_samples rows."""
    if rounds < 0:
        raise ValueError(f"rounds={rounds} must be non-negative")
    if not 0 <= delta < (1 << 32):
        raise ValueError(f"delta=0x{delta:x} must fit in 32 bits")
    if n_samples <= 0:
        raise ValueError(f"n_samples={n_samples} must be positive")
    if batch_size <= 0:
        raise ValueError(f"batch_size={batch_size} must be positive")

    device = torch.device("cuda")
    cpu_gen = torch.Generator(device="cpu").manual_seed(seed)

    delta_u32 = torch.tensor(delta, dtype=torch.uint32)

    emitted = 0
    while emitted < n_samples:
        n = min(batch_size, n_samples - emitted)
        half = n // 2
        labels = torch.zeros(n, dtype=torch.float32, device=device)
        labels[:half] = 1.0

        # Sample all random material on CPU (CUDA lacks uint32 bitwise ops).
        keys_cpu = _sample_keys_cpu(n, cpu_gen)
        p0_cpu = _sample_uint32_cpu(n, cpu_gen)
        p1_cpu = _sample_uint32_cpu(n, cpu_gen)
        # Real pairs: p1 = p0 ^ delta (XOR done on CPU where uint32 ops work).
        p1_cpu[:half] = p0_cpu[:half] ^ delta_u32

        alt_keys_cpu = _sample_keys_cpu(n, cpu_gen)
        keys_for_p1_cpu = keys_cpu.clone()
        keys_for_p1_cpu[half:] = alt_keys_cpu[half:]

        # Move to CUDA for encryption.
        keys = keys_cpu.to(device)
        p0 = p0_cpu.to(device)
        p1 = p1_cpu.to(device)
        keys_for_p1 = keys_for_p1_cpu.to(device)

        c0 = encrypt_batch(p0, keys, rounds=rounds)
        c1 = encrypt_batch(p1, keys_for_p1, rounds=rounds)
        pairs = torch.stack([c0, c1], dim=1)

        yield TrainingBatch(pairs=pairs, labels=labels)
        emitted += n

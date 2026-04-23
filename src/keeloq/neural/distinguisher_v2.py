"""Spatial-convolution variant of the Gohr-style distinguisher for KeeLoq.

Hypothesis under test (see docs/phase3b-results/ambition_outcome.md §"What
would push the frontier"): the v1 architecture's 1×1 convolutions blind the
model to bit-neighbor correlations, explaining its signal-horizon collapse
between depths 56 and 88. This variant reshapes the 64-bit input to
``(N, 2, 32)`` — two channels (c₀, c₁) over 32 spatial bit positions — and
uses kernel-size-3 convolutions so the model can actually see neighboring
bits interact.

Kept as a parallel module (not a drop-in replacement for ``Distinguisher``)
so that the existing ``checkpoints/d64.pt`` and its regression test stay
untouched while the experiment runs. If this architecture proves effective
at depth 88+, the next step is to unify the two into a versioned API.
"""

from __future__ import annotations

import torch
from torch import nn


class _BitUnpackSpatial(nn.Module):
    """Expand uint32 pairs into (N, 2, 32) float tensors, MSB-first.

    Same bit convention as the v1 unpack; only the shape differs. The 2
    channels are (c₀_bits, c₁_bits); the 32 length positions are bit
    indices 0..31 MSB-first.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 2) uint32 — cast to int64 via int32.view (CUDA lacks uint32 shifts).
        x64 = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        c0, c1 = x64[:, 0], x64[:, 1]
        shifts = torch.arange(31, -1, -1, dtype=torch.int64, device=x.device)
        bits_c0 = ((c0.unsqueeze(1) >> shifts) & 1).to(torch.float32)
        bits_c1 = ((c1.unsqueeze(1) >> shifts) & 1).to(torch.float32)
        return torch.stack([bits_c0, bits_c1], dim=1)  # (N, 2, 32)


class _ResidualBlockSpatial(nn.Module):
    """Two kernel_size-3 conv layers with BN + ReLU + residual. Width-preserving."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return torch.relu(h + x)


class DistinguisherSpatial(nn.Module):
    """Spatial-conv distinguisher. Bit positions form a 32-length sequence.

    Default parameter count with ``depth=5, width=256, kernel_size=3`` is
    roughly 2 M — comparable to the v1 ~3 M-param MLP-style default.
    """

    def __init__(
        self,
        depth: int = 5,
        width: int = 256,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.unpack = _BitUnpackSpatial()
        padding = kernel_size // 2
        self.embed = nn.Conv1d(
            2, width, kernel_size=kernel_size, padding=padding
        )
        self.blocks = nn.ModuleList(
            [_ResidualBlockSpatial(width, kernel_size) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (N, width, 1)
            nn.Flatten(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.unpack(x)  # (N, 2, 32)
        h = self.embed(b)  # (N, width, 32)
        for blk in self.blocks:
            h = blk(h)
        logits = self.head(h).squeeze(-1)  # (N,)
        return torch.sigmoid(logits)

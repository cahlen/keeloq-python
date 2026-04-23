"""Gohr-style ResNet-1D-CNN distinguisher for reduced-round KeeLoq.

Input: batched uint32 ciphertext pairs (N, 2). Internally reshaped to bit
vectors (N, 64) of float 0/1 in MSB-first order. Output: (N,) sigmoid
probability of the "real" label.
"""

from __future__ import annotations

import torch
from torch import nn


class _BitUnpack(nn.Module):
    """Expand uint32 pairs into 64-channel bit vectors of length 1.

    Output shape: (N, 64, 1) float 0/1, MSB-first. Uses int64 internally because
    CUDA doesn't implement right-shift for uint32.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 2) uint32
        x64 = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        c0, c1 = x64[:, 0], x64[:, 1]
        shifts = torch.arange(31, -1, -1, dtype=torch.int64, device=x.device)
        bits_c0 = ((c0.unsqueeze(1) >> shifts) & 1).to(torch.float32)
        bits_c1 = ((c1.unsqueeze(1) >> shifts) & 1).to(torch.float32)
        bits = torch.cat([bits_c0, bits_c1], dim=1)  # (N, 64)
        return bits.unsqueeze(-1)  # (N, 64, 1)


class _ResidualBlock1D(nn.Module):
    """Two 1x1 conv + BN + ReLU layers with a residual skip. Width-preserving."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return torch.relu(h + x)


class Distinguisher(nn.Module):
    """Gohr-style distinguisher. Default depth=5, width=512 (~3M params)."""

    def __init__(self, depth: int = 5, width: int = 512) -> None:
        super().__init__()
        self.unpack = _BitUnpack()
        self.embed = nn.Conv1d(64, width, kernel_size=1)
        self.blocks = nn.ModuleList([_ResidualBlock1D(width) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.unpack(x)  # (N, 64, 1)
        h = self.embed(b)  # (N, width, 1)
        for blk in self.blocks:
            h = blk(h)
        h = h.squeeze(-1)  # (N, width)
        logits = self.head(h).squeeze(-1)  # (N,)
        return torch.sigmoid(logits)

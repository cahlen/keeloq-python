"""Gohr-style ResNet-1D-CNN distinguisher for reduced-round KeeLoq.

Input: batched uint32 ciphertext pairs (N, 2). Internally reshaped to bit
vectors (N, 64) of float 0/1 in MSB-first order. Output: (N,) sigmoid
probability of the "real" label.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

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


from keeloq.neural.data import generate_pairs  # noqa: E402


@dataclass(frozen=True)
class TrainingConfig:
    rounds: int
    delta: int
    n_samples: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    seed: int
    depth: int = 5
    width: int = 512
    val_samples: int = 20_000


@dataclass(frozen=True)
class TrainingResult:
    final_loss: float
    final_val_accuracy: float
    wall_time_s: float
    config: TrainingConfig
    history: list[dict[str, float]] = field(default_factory=list)


def _set_seeds(seed: int) -> None:
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _compute_val_accuracy(model: Distinguisher, cfg: TrainingConfig) -> float:
    model.train(False)  # inference mode — same as .train(False)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in generate_pairs(
            rounds=cfg.rounds,
            delta=cfg.delta,
            n_samples=cfg.val_samples,
            seed=cfg.seed + 1_000_000,
            batch_size=min(cfg.batch_size, cfg.val_samples),
        ):
            preds = (model(batch.pairs) >= 0.5).float()
            correct += (preds == batch.labels).sum().item()
            total += batch.labels.shape[0]
    model.train(True)
    return correct / max(1, total)


def train(cfg: TrainingConfig) -> tuple[Distinguisher, TrainingResult]:
    _set_seeds(cfg.seed)
    model = Distinguisher(depth=cfg.depth, width=cfg.width).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCELoss()

    steps_per_epoch = max(1, cfg.n_samples // cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    history: list[dict[str, float]] = []
    t0 = time.perf_counter()

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in generate_pairs(
            rounds=cfg.rounds,
            delta=cfg.delta,
            n_samples=cfg.n_samples,
            seed=cfg.seed + epoch * 991,
            batch_size=cfg.batch_size,
        ):
            opt.zero_grad()
            preds = model(batch.pairs)
            loss = criterion(preds, batch.labels)
            loss.backward()
            opt.step()
            sched.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        val_acc = _compute_val_accuracy(model, cfg)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": epoch_loss / max(1, n_batches),
                "val_accuracy": val_acc,
            }
        )

    wall = time.perf_counter() - t0
    return model, TrainingResult(
        final_loss=history[-1]["train_loss"],
        final_val_accuracy=history[-1]["val_accuracy"],
        wall_time_s=wall,
        config=cfg,
        history=history,
    )


def save_checkpoint(model: Distinguisher, result: TrainingResult, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": asdict(result.config),
        "result": {
            "final_loss": result.final_loss,
            "final_val_accuracy": result.final_val_accuracy,
            "wall_time_s": result.wall_time_s,
            "history": list(result.history),
        },
    }
    torch.save(payload, path)


def load_checkpoint(path: Path | str) -> tuple[Distinguisher, TrainingResult]:
    payload = torch.load(Path(path), map_location="cuda", weights_only=False)
    cfg = TrainingConfig(**payload["config"])
    model = Distinguisher(depth=cfg.depth, width=cfg.width).cuda()
    model.load_state_dict(payload["state_dict"])
    r = payload["result"]
    result = TrainingResult(
        final_loss=r["final_loss"],
        final_val_accuracy=r["final_val_accuracy"],
        wall_time_s=r["wall_time_s"],
        config=cfg,
        history=r["history"],
    )
    return model, result

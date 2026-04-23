"""Phase-3b v2 spatial-conv experiment driver.

Runs three sub-experiments to test whether the kernel-size-3 spatial-conv
DistinguisherSpatial architecture surfaces signal at depths that collapsed
with the v1 1×1-conv Distinguisher:

  1. Δ search at depth 56 (control — should reproduce v1's signal,
     confirming v2 isn't broken).
  2. Δ search at depth 88 (the primary hypothesis test — did the v1 signal
     horizon move because of the architectural change?).
  3. If (2) surfaces any Δ above a threshold, a full train at that Δ.

Writes results to stdout as JSON + markdown to
``docs/phase3b-results/v2_experiment.md``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn

from keeloq.neural.data import generate_pairs
from keeloq.neural.differences import _default_candidate_set
from keeloq.neural.distinguisher_v2 import DistinguisherSpatial


# ---------- Standalone training loop (uses v2 architecture) ----------


def _set_seeds(seed: int) -> None:
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _val_accuracy(model: nn.Module, rounds: int, delta: int, seed: int,
                  n_samples: int = 5000, batch_size: int = 1024) -> float:
    model.train(False)
    correct, total = 0, 0
    with torch.no_grad():
        for batch in generate_pairs(
            rounds=rounds, delta=delta, n_samples=n_samples,
            seed=seed, batch_size=min(batch_size, n_samples),
        ):
            preds = (model(batch.pairs) >= 0.5).float()
            correct += (preds == batch.labels).sum().item()
            total += batch.labels.shape[0]
    model.train(True)
    return correct / max(1, total)


def train_v2(
    rounds: int,
    delta: int,
    n_samples: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    depth: int,
    width: int,
    kernel_size: int = 3,
) -> tuple[DistinguisherSpatial, dict]:
    _set_seeds(seed)
    model = DistinguisherSpatial(depth=depth, width=width, kernel_size=kernel_size).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    steps = max(1, n_samples // batch_size) * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    history = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        loss_sum, n_batches = 0.0, 0
        for batch in generate_pairs(
            rounds=rounds, delta=delta, n_samples=n_samples,
            seed=seed + epoch * 991, batch_size=batch_size,
        ):
            opt.zero_grad()
            preds = model(batch.pairs)
            loss = criterion(preds, batch.labels)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += float(loss.item())
            n_batches += 1
        val_acc = _val_accuracy(model, rounds, delta, seed=seed + 1_000_000)
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / max(1, n_batches),
            "val_accuracy": val_acc,
        })
    return model, {
        "final_loss": history[-1]["train_loss"],
        "final_val_accuracy": history[-1]["val_accuracy"],
        "wall_time_s": time.perf_counter() - t0,
        "history": history,
    }


# ---------- Δ search wrapper ----------


def search_delta_v2(
    rounds: int,
    candidates: list[int] | None = None,
    tiny_budget_samples: int = 200_000,
    tiny_budget_epochs: int = 2,
    seed: int = 0,
    depth: int = 2,
    width: int = 64,
    kernel_size: int = 3,
) -> list[dict]:
    if candidates is None:
        candidates = _default_candidate_set()
    seen: set[int] = set()
    uniq: list[int] = []
    for c in candidates:
        if c not in seen and 0 < c < (1 << 32):
            seen.add(c)
            uniq.append(c)

    results = []
    for i, delta in enumerate(uniq):
        _, res = train_v2(
            rounds=rounds, delta=delta,
            n_samples=tiny_budget_samples, batch_size=1024,
            epochs=tiny_budget_epochs, lr=2e-3, weight_decay=1e-5,
            seed=seed + i * 7919, depth=depth, width=width,
            kernel_size=kernel_size,
        )
        results.append({
            "delta": delta,
            "val_accuracy": res["final_val_accuracy"],
            "training_loss_final": res["final_loss"],
        })
    results.sort(key=lambda c: c["val_accuracy"], reverse=True)
    return results


# ---------- Main driver ----------


SIGNAL_THRESHOLD = 0.55  # if best tiny candidate exceeds this, invest in full training


def main() -> None:
    out_md = Path("docs/phase3b-results/v2_experiment.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Phase 3b v2 Spatial-Conv Experiment\n"]

    # Experiment 1: Δ search at depth 56 (control).
    print("[v2-exp] Δ search at depth 56 (control)...", flush=True)
    t0 = time.perf_counter()
    cands_56 = search_delta_v2(rounds=56, tiny_budget_samples=100_000, tiny_budget_epochs=2, seed=0)
    elapsed_56 = time.perf_counter() - t0
    best_56 = cands_56[0]
    lines.append(f"## Control: Δ search at depth 56 (v1 got best 0.688)\n")
    lines.append(f"Wall clock: {elapsed_56:.1f}s — top 5:\n")
    lines.append("| Δ | val_acc | loss |\n|---|---:|---:|")
    for c in cands_56[:5]:
        lines.append(f"| 0x{c['delta']:08x} | {c['val_accuracy']:.4f} | {c['training_loss_final']:.4f} |")
    print(json.dumps({"experiment": "control_56", "best": best_56, "wall_s": elapsed_56}), flush=True)

    # Experiment 2: Δ search at depth 88 (primary hypothesis).
    print("\n[v2-exp] Δ search at depth 88 (primary hypothesis)...", flush=True)
    t0 = time.perf_counter()
    cands_88 = search_delta_v2(rounds=88, tiny_budget_samples=100_000, tiny_budget_epochs=2, seed=0)
    elapsed_88 = time.perf_counter() - t0
    best_88 = cands_88[0]
    lines.append(f"\n## Primary: Δ search at depth 88 (v1 all < 0.517)\n")
    lines.append(f"Wall clock: {elapsed_88:.1f}s — top 10:\n")
    lines.append("| Δ | val_acc | loss |\n|---|---:|---:|")
    for c in cands_88[:10]:
        lines.append(f"| 0x{c['delta']:08x} | {c['val_accuracy']:.4f} | {c['training_loss_final']:.4f} |")
    print(json.dumps({"experiment": "primary_88", "best": best_88, "wall_s": elapsed_88}), flush=True)

    # Experiment 3 (conditional): Δ search at depth 120.
    print("\n[v2-exp] Δ search at depth 120 (stretch)...", flush=True)
    t0 = time.perf_counter()
    cands_120 = search_delta_v2(rounds=120, tiny_budget_samples=100_000, tiny_budget_epochs=2, seed=0)
    elapsed_120 = time.perf_counter() - t0
    best_120 = cands_120[0]
    lines.append(f"\n## Stretch: Δ search at depth 120 (v1 all < 0.515)\n")
    lines.append(f"Wall clock: {elapsed_120:.1f}s — top 10:\n")
    lines.append("| Δ | val_acc | loss |\n|---|---:|---:|")
    for c in cands_120[:10]:
        lines.append(f"| 0x{c['delta']:08x} | {c['val_accuracy']:.4f} | {c['training_loss_final']:.4f} |")
    print(json.dumps({"experiment": "stretch_120", "best": best_120, "wall_s": elapsed_120}), flush=True)

    # Experiment 4 (conditional): if depth 88 has signal, full train.
    verdict_lines: list[str] = []
    verdict_lines.append(f"\n## Verdict\n")
    if best_88["val_accuracy"] >= SIGNAL_THRESHOLD:
        verdict_lines.append(
            f"- Depth 88 best Δ=0x{best_88['delta']:08x} reached val-acc "
            f"{best_88['val_accuracy']:.4f} — **above the {SIGNAL_THRESHOLD} threshold**. "
            "Spatial conv architecture surfaces signal where v1's 1×1 version failed. "
            "Proceeding with a full-scale train at this Δ.\n"
        )
        print(f"\n[v2-exp] Depth 88 signal confirmed ({best_88['val_accuracy']:.4f}). "
              "Kicking off full train (10M samples × 20 epochs)...", flush=True)
        t0 = time.perf_counter()
        _, full_res = train_v2(
            rounds=88, delta=best_88["delta"],
            n_samples=10_000_000, batch_size=4096,
            epochs=20, lr=2e-3, weight_decay=1e-5,
            seed=1729, depth=5, width=256, kernel_size=3,
        )
        verdict_lines.append(
            f"- Full train: val_acc={full_res['final_val_accuracy']:.4f}, "
            f"loss={full_res['final_loss']:.4f}, "
            f"wall_time_s={full_res['wall_time_s']:.1f}.\n"
        )
        print(json.dumps({"experiment": "full_train_88", "result": full_res}), flush=True)
    else:
        verdict_lines.append(
            f"- Depth 88 best Δ=0x{best_88['delta']:08x} reached val-acc "
            f"{best_88['val_accuracy']:.4f} — **below the {SIGNAL_THRESHOLD} threshold**. "
            "Spatial conv architecture *also* fails to surface signal at depth 88. "
            "This tightens the negative result from 'v1 architecture fails' to "
            "'both 1×1 and spatial 3-tap architectures fail' — suggesting the "
            "signal horizon is a genuine property of KeeLoq's diffusion at these "
            "depths, not an artifact of any one architecture.\n"
        )
        print("\n[v2-exp] Depth 88 still below threshold. Negative result stands.", flush=True)

    lines.extend(verdict_lines)
    out_md.write_text("\n".join(lines) + "\n")
    print(f"\n[v2-exp] Wrote {out_md}", flush=True)


if __name__ == "__main__":
    main()

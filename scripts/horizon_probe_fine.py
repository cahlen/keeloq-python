"""Fine-grained horizon probe at depths 57, 58, 59 (between known-signal 56
and probe-collapsed 60).

If any cell at depths 57-59 crosses the viability threshold (0.55), full-scale
training at that (depth, Δ) would give us a distinguisher for attacks at
~(depth + 8) rounds, pushing the coverage table past d64.pt.

Tests 8 candidate Δs at each of 3 depths — 24 tiny trainings, ~2 min.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from keeloq.neural.distinguisher import TrainingConfig, train

# Top-8 Δs from the depth-56 search (val-acc ≥ 0.62).
FINE_DELTAS = [
    0x00000002,  # 0.688
    0x00010000,  # 0.670
    0x00000004,  # 0.649
    0x02000000,  # 0.642
    0x00020000,  # 0.637
    0x04000002,  # 0.634
    0x01000000,  # 0.634
    0x00800000,  # 0.633
]
FINE_DEPTHS = [57, 58, 59]
VIABILITY_THRESHOLD = 0.55


def probe() -> dict:
    print("=" * 70, flush=True)
    print(f"Fine horizon probe: depths {FINE_DEPTHS} x top-8 Δs from depth-56 search", flush=True)
    print("=" * 70, flush=True)
    results: dict[int, dict[int, float]] = {}
    t_all = time.perf_counter()

    for depth in FINE_DEPTHS:
        results[depth] = {}
        for delta in FINE_DELTAS:
            cfg = TrainingConfig(
                rounds=depth,
                delta=delta,
                n_samples=100_000,
                batch_size=1024,
                epochs=2,
                lr=2e-3,
                weight_decay=1e-5,
                seed=0,
                depth=2,
                width=16,
                val_samples=5000,
            )
            _, res = train(cfg)
            acc = res.final_val_accuracy
            results[depth][delta] = acc
            marker = "✅" if acc >= VIABILITY_THRESHOLD else " "
            print(f"  depth={depth:2d}  Δ=0x{delta:08x}  val_acc={acc:.4f}  {marker}", flush=True)

    print("=" * 70, flush=True)
    print(f"Probe complete in {time.perf_counter() - t_all:.1f}s", flush=True)
    return results


def main() -> None:
    results = probe()
    # Best cell overall:
    best_cell = None
    best_acc = 0.0
    for depth, cells in results.items():
        for delta, acc in cells.items():
            if acc > best_acc:
                best_acc = acc
                best_cell = (depth, delta)

    out = Path("docs/phase3b-results/horizon_probe_fine.md")
    lines = ["# Phase 3b Horizon Probe — Fine-Grained (57, 58, 59)\n"]
    lines.append(
        f"Top-8 Δs from depth-56 search, 100 k samples x 2 epochs. "
        f"Viability threshold: val_acc ≥ {VIABILITY_THRESHOLD}.\n"
    )
    lines.append("## Results\n")
    header = "| trained_depth | " + " | ".join(f"Δ=0x{d:08x}" for d in FINE_DELTAS) + " | best |"
    sep = "|---:|" + "|".join(["---:"] * (len(FINE_DELTAS) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for depth in FINE_DEPTHS:
        cells_fmt = [f"{results[depth][d]:.4f}" for d in FINE_DELTAS]
        best = max(results[depth].values())
        mark = "✅" if best >= VIABILITY_THRESHOLD else "❌"
        lines.append(f"| {depth} | " + " | ".join(cells_fmt) + f" | **{best:.4f}** {mark} |")
    lines.append("\n## Conclusion\n")
    if best_cell is not None and best_acc >= VIABILITY_THRESHOLD:
        depth, delta = best_cell
        lines.append(
            f"**Signal found at depth {depth}, Δ=0x{delta:08x} "
            f"(val_acc {best_acc:.4f}).** Full training at this cell is the "
            f"candidate for a new checkpoint (d{depth + 8}.pt — attacks at "
            f"depth {depth + 8} by peeling K=8 to the trained depth).\n"
        )
    else:
        lines.append(
            "No fine-grained depth crossed the viability threshold. The cliff "
            "is very sharp — at or just past depth 56. d64.pt's trained depth "
            "56 is the maximum viable point with this architecture + budget.\n"
        )
    out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}", flush=True)

    summary = {
        "best_cell": {
            "depth": best_cell[0] if best_cell else None,
            "delta": f"0x{best_cell[1]:08x}" if best_cell else None,
            "val_acc": best_acc,
        },
        "viable": best_cell is not None and best_acc >= VIABILITY_THRESHOLD,
    }
    print("\nSummary:\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

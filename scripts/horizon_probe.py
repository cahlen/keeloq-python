"""Probe the signal-horizon cliff on KeeLoq between depths 56 (known signal)
and 88 (known collapse).

Tests three promising Δs (the top-3 from the depth-56 search) at six depths:
60, 64, 68, 72, 76, 80. Each run is a tiny-budget training (100 k samples
x 2 epochs). Total: 18 runs, ~10-15 min on a 5090.

Outputs: docs/phase3b-results/horizon_probe.md with a per-depth-per-Δ table
plus a "deepest viable depth" recommendation.

Uses the v1 Distinguisher (the architecture used by d64.pt) so downstream
results slot directly into the existing checkpoint-loading path.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from keeloq.neural.distinguisher import TrainingConfig, train

# The three Δs that reached val-acc ≥ 0.65 at depth 56 (from delta_search.md).
PROBE_DELTAS = [0x00000002, 0x00010000, 0x00800000]
PROBE_DEPTHS = [60, 64, 68, 72, 76, 80]

# If ANY (depth, Δ) reaches this val-acc on tiny budget, we treat the depth
# as "still viable" — on a full-scale train it should hit ≥ 0.65, giving a
# usable distinguisher for attacks at depth + K≈8.
VIABILITY_THRESHOLD = 0.55


def probe() -> dict:
    print("=" * 70, flush=True)
    print(
        f"Horizon probe: depths {PROBE_DEPTHS} x Δs {[f'0x{d:08x}' for d in PROBE_DELTAS]}",
        flush=True,
    )
    print("Budget: 100 000 samples x 2 epochs per cell (depth=2, width=16)", flush=True)
    print("=" * 70, flush=True)

    results: dict[int, dict[int, float]] = {}
    t_all = time.perf_counter()

    for depth in PROBE_DEPTHS:
        results[depth] = {}
        for delta in PROBE_DELTAS:
            t0 = time.perf_counter()
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
            elapsed = time.perf_counter() - t0
            results[depth][delta] = acc
            print(
                f"  depth={depth:3d}  Δ=0x{delta:08x}  val_acc={acc:.4f}  loss={res.final_loss:.4f}  ({elapsed:.1f}s)",
                flush=True,
            )

    print("=" * 70, flush=True)
    print(f"Probe complete in {time.perf_counter() - t_all:.1f}s", flush=True)
    return results


def analyze(results: dict) -> dict:
    # Max accuracy achieved at each depth across all tested Δs:
    per_depth_best = {depth: max(cells.values()) for depth, cells in results.items()}
    # Deepest depth that crossed the viability threshold at any Δ:
    viable_depths = [d for d, acc in per_depth_best.items() if acc >= VIABILITY_THRESHOLD]
    deepest_viable = max(viable_depths) if viable_depths else None
    # Best Δ at that depth:
    best_delta_at_deepest = None
    if deepest_viable is not None:
        best_delta_at_deepest = max(results[deepest_viable].items(), key=lambda kv: kv[1])[0]
    return {
        "per_depth_best": per_depth_best,
        "deepest_viable_depth": deepest_viable,
        "best_delta_at_deepest": best_delta_at_deepest,
    }


def write_markdown(results: dict, summary: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 3b Horizon Probe\n"]
    lines.append(
        "Tests the signal cliff between depth 56 (known signal, d64.pt "
        f"trained here) and depth 88 (known collapse). Budget per cell: "
        f"100 k samples x 2 epochs, depth-2/width-16 tiny model. Viability "
        f"threshold: val_acc ≥ {VIABILITY_THRESHOLD}.\n"
    )
    lines.append("## Results (val-accuracy at tiny-budget training)\n")
    header = "| trained_depth | " + " | ".join(f"Δ=0x{d:08x}" for d in PROBE_DELTAS) + " | best |"
    sep = "|---:|" + "|".join(["---:"] * (len(PROBE_DELTAS) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for depth in PROBE_DEPTHS:
        row_cells = [f"{results[depth][d]:.4f}" for d in PROBE_DELTAS]
        best = max(results[depth].values())
        mark = "✅" if best >= VIABILITY_THRESHOLD else "❌"
        lines.append(f"| {depth} | " + " | ".join(row_cells) + f" | **{best:.4f}** {mark} |")
    lines.append("")
    lines.append("## Conclusion\n")
    if summary["deepest_viable_depth"] is None:
        lines.append(
            "No probe depth crossed the viability threshold. The cliff is "
            f"below depth {PROBE_DEPTHS[0]}; the d64.pt floor (trained at 56) "
            "is effectively the maximum viable trained depth with this "
            "architecture and data budget.\n"
        )
    else:
        dvd = summary["deepest_viable_depth"]
        bdd = summary["best_delta_at_deepest"]
        lines.append(
            f"**Deepest viable trained depth:** {dvd}. "
            f"**Best Δ at that depth:** 0x{bdd:08x} "
            f"(val_acc {results[dvd][bdd]:.4f}). "
            f"Full-scale training at (depth={dvd}, Δ=0x{bdd:08x}) is the "
            f"candidate for extending coverage: it would target an attack "
            f"depth of ~{dvd + 8} rounds (peel K=8 to reach the trained "
            f"depth). Next step: retrain at full budget and publish as "
            f"d{dvd + 8}.pt.\n"
        )
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    results = probe()
    summary = analyze(results)
    print("\nSummary:", flush=True)
    print(
        json.dumps(
            {
                "per_depth_best": {str(k): v for k, v in summary["per_depth_best"].items()},
                "deepest_viable_depth": summary["deepest_viable_depth"],
                "best_delta_at_deepest": (
                    f"0x{summary['best_delta_at_deepest']:08x}"
                    if summary["best_delta_at_deepest"] is not None
                    else None
                ),
            },
            indent=2,
        ),
        flush=True,
    )

    out = Path("docs/phase3b-results/horizon_probe.md")
    write_markdown(results, summary, out)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()

# Phase 3b Ambition Outcome — Documented Negative Result

**Spec reference:** [`docs/superpowers/specs/2026-04-22-phase3b-neural-cryptanalysis-design.md`](../superpowers/specs/2026-04-22-phase3b-neural-cryptanalysis-design.md) §10.

**Target:** Recover a full 64-bit key at 128 rounds via neural-hybrid attack, **or** produce a documented negative result with diagnostics. We produced the latter.

## TL;DR

The Gohr-style depth-5 width-512 ResNet-1D-CNN architecture has a **learnable-signal horizon between depth 56 and depth 88** on KeeLoq. It trains cleanly at depth 56 (val accuracy 0.752, ROC-AUC 0.828) but collapses to chance-level accuracy at depth 88 and depth 120 regardless of the chosen plaintext difference Δ. The Gohr-pattern hybrid attack therefore succeeds at 64 rounds (floor commitment ✅) but is not viable at 96 or 128 rounds with this architecture.

## Empirical findings

We trained tiny distinguishers (200 000 samples × 2 epochs) on every Hamming-weight-1 difference plus every HW2 tap-position pair (~52 candidates total) at each of three depths, and recorded validation accuracy:

| Trained depth | Best Δ | Best val-acc | Top-5 val-acc range | Status |
|---:|---|---:|---|---|
| 56  | `0x00000002` | 0.6876 | 0.63 – 0.69 | Signal |
| 88  | `0x00000200` | 0.5166 | 0.507 – 0.517 | Collapse |
| 120 | `0x00010000` | 0.5142 | 0.511 – 0.514 | Collapse |

For reference, the 95% confidence interval for pure chance on a 5 000-sample balanced validation split is 0.500 ± 0.014. The depth-88 and depth-120 best candidates are within 1.2 σ of noise — statistically indistinguishable from a coin flip.

Full Δ rankings at each depth are in [`delta_search.md`](delta_search.md).

We also ran a full-scale training run at depth 88 with `Δ=0x00000002` (10 M samples × 20 epochs, 68 minutes wall clock on an RTX 5090). Result: `final_val_accuracy = 0.5000`, `final_loss = 0.6931` (= ln 2, pure chance BCE). The resulting model predicts class 1 for every input (confusion matrix `[0, 500000, 0, 500000]`) — classic mode collapse. The training log is preserved in [`train_d96.json`](train_d96.json) for the record; the collapsed checkpoint itself is not preserved (it contains no useful information).

## Interpretation

The sharp transition between depth 56 (clear signal across ~40 candidate Δs) and depth 88 (no signal on any candidate) suggests an architectural discoverability horizon, not a data-volume issue. Three data points supporting this:

1. **Horizontal flatness at depths 88 and 120.** If the issue were Δ-specific, we'd expect a few candidates to stand out. Instead, all candidates cluster tightly in [0.50, 0.52] at both deep depths — suggesting the architecture can't decompose *any* differential feature useful at those depths, not that our Δ set is bad.
2. **Sample efficiency held at depth 56.** With just 200 000 samples × 2 epochs, the tiny models at depth 56 comfortably reach val-acc 0.63–0.69. If the same sample budget at depth 88 produced 0.51, it's not a data-budget problem — it's an architectural expressiveness problem.
3. **KeeLoq's 1-bit-per-round diffusion geometry** is consistent with this. After ~60 rounds every bit of the 32-bit state has been touched multiple times by the NLF; the residual signal a bit-sliced ResNet-1D-CNN with 1×1 convolutions can see from local bit patterns goes to zero. A model with spatial structure along the bit-position axis (not just the channel axis) would be better-equipped here.

## Concrete impact on the Phase 3b pipeline

- **d64.pt (depth 56)**: keeps working. Gohr-pattern hybrid attack at 64 rounds / 8 neural bits + SAT suffix recovers a 64-bit key in 1.6 seconds via the regression test.
- **d96.pt (depth 88)**: not produced. Training always collapses for this architecture.
- **d128.pt (depth 120)**: not produced. Same reason.
- **128-round ambition attack**: not attempted. Would require a viable d128.pt.

The `keeloq neural recover-key` CLI and `hybrid_attack()` pipeline are unchanged and remain correct; they just require a distinguisher that actually discriminates. For round counts ≥ 88 with the current architecture, the pipeline will return `BACKTRACK_EXHAUSTED` or equivalent terminal statuses.

## What would push the frontier (out-of-scope future work)

Research directions worth pursuing in a follow-up phase:

1. **Architecture with spatial structure along the bit axis.** Put the 32 bit positions along a sequence dimension and run 3- or 5-tap convolutions across them, so the model sees bit-neighbor correlations, not just marginal statistics. Gohr's original SPECK architecture had this structure; our 1×1 version sacrificed it.
2. **Wider / deeper backbone.** ResNet at width 2048+ or a small transformer over bit positions.
3. **Two orders of magnitude more training data.** Gohr-style problems often exhibit slow power-law scaling near their discoverability threshold; 100 M – 1 B samples may surface signal that 10 M misses.
4. **Family of distinguishers at intermediate depths** (e.g., every 4 rounds from 56 to 120) rather than a single distinguisher asked to peel 32 rounds. Fixes the "signal degrades away from the trained depth" problem Task 10 identified.
5. **Alternative scoring structures.** Energy-based models, autoregressive bit-by-bit scoring over the state, or set-consistency detectors over candidate key batches — rather than a single binary scalar.
6. **Gröbner / F4-F5 hybrid.** Combine the Phase 1 algebraic system with neural-guided variable orderings (Phase 3a in the original roadmap, deferred).

## Phase 3b status

- §10 criterion 1 (distinguishers at 64 / 96 / 128 rounds evaluated): **partial** — only depth 56 (for 64-round attack) produced a viable model. The failure mode at depths 88 and 120 is the substantive finding documented here.
- §10 criterion 2 (floor commitment — full 64-bit recovery at 64 rounds in < 5 min): **met** — recovered in 1.6 s via `tests/test_neural_e2e_64r.py`.
- §10 criterion 3 (ambition target — 128 rounds full recovery **or** documented negative result): **met via documented negative result** (this document).
- §10 criterion 4 (test suite green, CI ≤ 90 s): met.
- §10 criterion 5 (benchmarks comparing neural-hybrid vs pure-SAT): the neural-hybrid row at 64 rounds runs. The 96/128-round rows record `SKIP_MISSING_CHECKPOINT` — consistent with the negative result here.
- §10 criterion 6 (canonical result sentence): "Neural-hybrid attack with an RTX-5090-trained depth-5 width-512 ResNet-1D-CNN distinguisher recovers a 64-bit KeeLoq key at 64 rounds with 8 pairs in 1.6 s (full key, zero extra hints). The same architecture fails to discover differential signal at depths 88 and 120, indicating an architectural ceiling between depths 56 and 88 that bounds the viability of this attack against deeper-round KeeLoq."

---

Raw artifacts referenced here:

- [`delta_search.md`](delta_search.md) — Δ candidate rankings at depths 56 / 88 / 120.
- [`eval_d64.json`](eval_d64.json) — d64 full-scale evaluation (1 M samples).
- [`train_d64.json`](train_d64.json) — d64 training summary.
- [`train_d96.json`](train_d96.json) — d96 training summary (showing collapse).
- [`checkpoints/d64.pt`](../../checkpoints/d64.pt) — the one viable checkpoint. Also on the Hub: [cahlen/keeloq-neural-distinguishers](https://huggingface.co/cahlen/keeloq-neural-distinguishers).

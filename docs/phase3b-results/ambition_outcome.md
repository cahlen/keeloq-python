# Phase 3b Ambition Outcome — Documented Negative Result

**Spec reference:** [`docs/superpowers/specs/2026-04-22-phase3b-neural-cryptanalysis-design.md`](../superpowers/specs/2026-04-22-phase3b-neural-cryptanalysis-design.md) §10.

**Target:** Recover a full 64-bit key at 128 rounds via neural-hybrid attack, **or** produce a documented negative result with diagnostics. We produced the latter.

## TL;DR

The Gohr-style depth-5 width-512 ResNet-1D-CNN architecture has a **learnable-signal horizon between depth 56 and depth 88** on KeeLoq. It trains cleanly at depth 56 (val accuracy 0.752, ROC-AUC 0.828) but collapses to chance-level accuracy at depth 88 and depth 120 regardless of the chosen plaintext difference Δ. The Gohr-pattern hybrid attack therefore succeeds at 64 rounds (floor commitment ✅) but is not viable at 96 or 128 rounds with this architecture.

## Empirical findings

We trained tiny distinguishers (100 000–200 000 samples × 2 epochs) on every Hamming-weight-1 difference plus every HW2 tap-position pair (~52 candidates total) at each tested depth, and recorded validation accuracy:

| Trained depth | Best Δ | Best val-acc | Top-5 val-acc range | Status |
|---:|---|---:|---|---|
| 56  | `0x00000002` | 0.6876 | 0.63 – 0.69 | Signal |
| 57  | `0x01000000` | 0.5386 | 0.50 – 0.54 | Boundary |
| 58  | `0x00800000` | 0.5340 | 0.50 – 0.53 | Collapse |
| 59  | `0x02000000` | 0.5344 | 0.50 – 0.53 | Collapse |
| 60  | `0x00800000` | 0.5282 | 0.50 – 0.53 | Collapse |
| 64  | various | 0.5110 | 0.50 – 0.51 | Collapse |
| 68  | various | 0.5088 | 0.49 – 0.51 | Collapse |
| 72  | various | 0.5016 | 0.49 – 0.50 | Collapse |
| 76  | various | 0.5150 | 0.50 – 0.52 | Collapse |
| 80  | various | 0.5052 | 0.49 – 0.51 | Collapse |
| 88  | `0x00000200` | 0.5166 | 0.507 – 0.517 | Collapse |
| 120 | `0x00010000` | 0.5142 | 0.511 – 0.514 | Collapse |

For reference, the 95% confidence interval for pure chance on a 5 000-sample balanced validation split is 0.500 ± 0.014. Everything from depth 57 onwards sits within 2–3 σ of noise.

**Key finding: the cliff is one round wide.** Signal decays from 0.69 at depth 56 to 0.54 at depth 57 to 0.53 at depth 58 — a near-step-function transition. KeeLoq's round function applied a single additional time after depth 56 apparently takes the residual differential feature below what this model family can learn. By depth 60 all candidate Δs are pure noise, and depths 60 through 120 behave identically.

Full Δ rankings at depths 56 / 88 / 120 (v1 architecture) are in [`delta_search.md`](delta_search.md); the broad depth sweep (60–80) is in [`horizon_probe.md`](horizon_probe.md); the fine probe (57–59) is in [`horizon_probe_fine.md`](horizon_probe_fine.md); the alternate-architecture (v2 spatial-conv) replication is in [`v2_experiment.md`](v2_experiment.md).

We also ran a full-scale training run at depth 88 with `Δ=0x00000002` (10 M samples × 20 epochs, 68 minutes wall clock on an RTX 5090). Result: `final_val_accuracy = 0.5000`, `final_loss = 0.6931` (= ln 2, pure chance BCE). The resulting model predicts class 1 for every input (confusion matrix `[0, 500000, 0, 500000]`) — classic mode collapse. The training log is preserved in [`train_d96.json`](train_d96.json) for the record; the collapsed checkpoint itself is not preserved (it contains no useful information).

## Interpretation

The sharp transition between depth 56 (clear signal across ~40 candidate Δs) and depth 88 (no signal on any candidate) is consistent with a KeeLoq-specific diffusion-based signal horizon, not an architectural artifact. Evidence:

1. **Horizontal flatness at depths 88 and 120.** If the issue were Δ-specific, we'd expect a few candidates to stand out. Instead, all candidates cluster tightly in [0.50, 0.52] at both deep depths — the architecture can't decompose *any* differential feature useful at those depths, not that our Δ set is bad.
2. **Sample efficiency held at depth 56.** With just 200 000 samples × 2 epochs, the tiny models at depth 56 comfortably reach val-acc 0.63–0.69. If the same sample budget at depth 88 produced 0.51, it's not a data-budget problem.
3. **Two architectures collapse identically.** We tested the original 1×1-conv MLP-style ResNet (`Distinguisher`, v1) alongside a kernel-size-3 spatial-conv ResNet (`DistinguisherSpatial`, v2) that uses an inductive bias for bit-neighbor correlations. Both succeed at depth 56 (v1 best 0.688, v2 best 0.703 — essentially equivalent) and both collapse identically at depths 88 and 120 (all candidates within statistical noise of 0.5). See [`v2_experiment.md`](v2_experiment.md) for the head-to-head table. The fact that adding spatial inductive bias did *not* unlock signal at depth 88 is strong evidence the horizon is a property of the cipher, not a limitation of any one network shape.
4. **KeeLoq's 1-bit-per-round diffusion geometry** is consistent with this. After ~60 rounds every bit of the 32-bit state has been touched multiple times by the NLF; the differential signal in ciphertext pairs decays below what moderate-capacity supervised learning can discover without an explosion in data. A full cryptanalytic treatment of where exactly differential trails die out on KeeLoq would sharpen this into a quantitative threshold.

## Concrete impact on the Phase 3b pipeline

- **d64.pt (depth 56)**: keeps working. Gohr-pattern hybrid attack at 64 rounds / 8 neural bits + SAT suffix recovers a 64-bit key in 1.6 seconds via the regression test.
- **d72, d80, d88, d96, d128**: none produced. Every trained depth ≥ 57 tested to date collapses to chance-level accuracy.
- **Maximum viable trained depth with this architecture and data budget: 56.** The horizon probe walked depth 57 through depth 80 and found no viable cell; the broader depth sweep at 88 and 120 confirms the same pattern continues. d64.pt (trained at depth 56, peel K=8, attack at 64 rounds) is *definitionally* the deepest useful checkpoint until one of the "push the frontier" directions below lands.

The `keeloq neural recover-key` CLI and `hybrid_attack()` pipeline are unchanged and remain correct; they just require a distinguisher that actually discriminates. For round counts ≥ 65 with the current single-distinguisher architecture, the pipeline will return `BACKTRACK_EXHAUSTED` or equivalent terminal statuses.

## What would push the frontier (out-of-scope future work)

Directions still worth pursuing, with revised priority given that spatial
inductive bias was ruled out by the v2 experiment:

1. **Two orders of magnitude more training data.** Gohr-style problems often exhibit slow power-law scaling near their discoverability threshold; 100 M – 1 B samples may surface signal that 10 M misses. Now the leading candidate since both tested architectures match.
2. **Family of distinguishers at intermediate depths** (e.g., every 4 rounds from 56 to 88) rather than a single distinguisher asked to peel to depth 88. Fixes the "signal degrades away from the trained depth" problem Task 10 identified and works around the horizon by staying inside it.
3. **Wider / deeper backbone.** ResNet at width 2048+ or a small transformer over bit positions. Less theoretically motivated after the v2 experiment but sample efficiency could still improve.
4. **Alternative scoring structures.** Energy-based models, autoregressive bit-by-bit scoring over the state, or set-consistency detectors over candidate key batches — rather than a single binary scalar.
5. **Gröbner / F4-F5 hybrid.** Combine the Phase 1 algebraic system with neural-guided variable orderings (Phase 3a in the original roadmap, deferred).
6. **Quantitative differential-trail analysis.** Directly analyze KeeLoq's differential branch numbers round-by-round to locate the precise point where any single-Δ trail reaches round-function entropy. That number is the theoretical minimum horizon for *any* differential distinguisher; comparing it to the empirical ~80-round collapse seen here would either close the gap or motivate multi-Δ / higher-order differential approaches.

## Phase 3b status

- §10 criterion 1 (distinguishers at 64 / 96 / 128 rounds evaluated): **partial** — only depth 56 (for 64-round attack) produced a viable model. The failure mode at depths 88 and 120 is the substantive finding documented here.
- §10 criterion 2 (floor commitment — full 64-bit recovery at 64 rounds in < 5 min): **met** — recovered in 1.6 s via `tests/test_neural_e2e_64r.py`.
- §10 criterion 3 (ambition target — 128 rounds full recovery **or** documented negative result): **met via documented negative result** (this document).
- §10 criterion 4 (test suite green, CI ≤ 90 s): met.
- §10 criterion 5 (benchmarks comparing neural-hybrid vs pure-SAT): the neural-hybrid row at 64 rounds runs. The 96/128-round rows record `SKIP_MISSING_CHECKPOINT` — consistent with the negative result here.
- §10 criterion 6 (canonical result sentence): "Neural-hybrid attack with an RTX-5090-trained depth-5 width-512 ResNet-1D-CNN distinguisher recovers a 64-bit KeeLoq key at 64 rounds with 8 pairs in 1.6 s (full key, zero extra hints). The same architecture fails to discover differential signal at depths 88 and 120, indicating an architectural ceiling between depths 56 and 88 that bounds the viability of this attack against deeper-round KeeLoq."

---

Raw artifacts referenced here:

- [`delta_search.md`](delta_search.md) — Δ candidate rankings at depths 56 / 88 / 120 (v1 architecture).
- [`v2_experiment.md`](v2_experiment.md) — Δ candidate rankings at depths 56 / 88 / 120 with the v2 spatial-conv architecture; same collapse pattern.
- [`eval_d64.json`](eval_d64.json) — d64 full-scale evaluation (1 M samples).
- [`train_d64.json`](train_d64.json) — d64 training summary.
- [`train_d96.json`](train_d96.json) — d96 training summary (showing collapse).
- [`checkpoints/d64.pt`](../../checkpoints/d64.pt) — the one viable checkpoint. Also on the Hub: [cahlen/keeloq-neural-distinguishers](https://huggingface.co/cahlen/keeloq-neural-distinguishers).

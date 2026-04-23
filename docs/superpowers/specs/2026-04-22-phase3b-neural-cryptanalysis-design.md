# Phase 3b — Neural Differential Cryptanalysis: Design Spec

**Date:** 2026-04-22
**Author:** brainstorm session (Cahlen + Claude)
**Status:** Approved
**Target:** `keeloq-python` — neural-network-based cryptanalysis of reduced-round KeeLoq on RTX 5090, following the Gohr 2019 framework adapted for KeeLoq's 1-bit-per-round key schedule.

---

## 1. Context

Phase 1 (complete, merged to master) delivered a Python 3 modernized algebraic/SAT attack pipeline: pure-Python cipher, GPU bit-sliced cipher for property tests, ANF generator, dual CNF/XOR-aware encoders, CryptoMiniSat wrapper, and a full attack CLI. Baseline result on an RTX 5090 workstation: full-key recovery at 64 rounds / 4 plaintext-ciphertext pairs / 0 hints in **0.247 s** via XOR-aware encoder + CryptoMiniSat.

Phase 3b targets the next major capability class: **neural differential cryptanalysis**, in the Gohr 2019 lineage ("Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning", CRYPTO 2019). No prior published Gohr-style attack on KeeLoq exists; KeeLoq's NLFSR structure + cyclic 1-bit-per-round key schedule is structurally different from the SP-network ciphers that have dominated this research line.

**The research question:** Can a neural differential distinguisher plus Bayesian key recovery outperform a modern SAT attack on reduced-round KeeLoq, and if so, at what round count does the crossover happen?

This spec covers only Phase 3b. Phase 2 (research harness) and Phase 3a (SAT improvements) remain deferred; Phase 3b is independent of both.

## 2. Goals

1. Train binary neural distinguishers for reduced-round KeeLoq at multiple depths (target: 64, 96, 128 rounds), capable of separating chosen-plaintext-difference output pair distributions from random pairs with non-trivial accuracy.
2. Implement Bayesian key-bit recovery over the trained distinguisher, exploiting KeeLoq's 1-bit-per-round key schedule (each guess resolves 1 bit, not a multi-bit subkey).
3. Combine neural prefix recovery with Phase 1's SAT attack (neural peels outer rounds, SAT handles the remaining underdetermined suffix) to produce an **always-complete** hybrid attack pipeline.
4. Deliver a full key recovery at 64 rounds (floor commitment) and attempt 128 rounds (ambition; graceful degradation if the distinguisher margin collapses).
5. Publish a comparative benchmark: wall-clock and data complexity for neural-hybrid vs. pure-SAT attacks at 64 / 96 / 128 rounds.

## 3. Non-goals (deferred)

- **Full-round KeeLoq (528 rounds).** Out of scope. No published cryptanalysis breaks the full cipher via this approach; unrealistic ambition for Phase 3b.
- **Beating Phase 1 SAT at round counts where SAT already wins cleanly** (≤64 rounds). Neural is expected to parallel or underperform SAT at very low rounds; the interesting regime is ≥96.
- **Distributed / multi-GPU training.** Single RTX 5090 is the entire compute budget.
- **Hugging Face Hub publishing, W&B, lightning.** Minimize deps.
- **Conditional distinguishers** (one model that takes round number as input). Stick with one model per depth.
- **Transformers, diffusion, or exotic architectures.** Gohr's ResNet-1D-CNN is the starting point; deviation requires empirical justification (left to experimentation within the spec's architecture freedom).
- **Production inference API.** Research code; CLI invocation is the only interface.

## 4. Repository layout

```
src/keeloq/
├── ...existing Phase 1 modules...
└── neural/
    ├── __init__.py
    ├── data.py              # training-data streaming generator (uses gpu_cipher)
    ├── differences.py       # chosen-plaintext-difference candidate search
    ├── distinguisher.py     # ResNet-1D-CNN architecture + training loop + checkpoint I/O
    ├── evaluation.py        # accuracy / ROC-AUC / TPR@FPR metrics
    ├── key_recovery.py      # sequential binary key-bit Bayesian search with beam
    ├── hybrid.py            # neural-prefix + SAT-suffix orchestration
    └── cli_neural.py        # `keeloq neural {train, evaluate, recover-key, auto}`

tests/
├── ...existing Phase 1 tests...
├── test_neural_data.py
├── test_neural_differences.py
├── test_neural_distinguisher.py
├── test_neural_evaluation.py
├── test_neural_key_recovery.py
├── test_neural_hybrid.py
└── test_neural_cli.py

benchmarks/
├── ...existing Phase 1 benchmarks...
├── neural_matrix.toml
├── bench_neural.py
└── neural_summary.md       # populated after bench run

checkpoints/
├── README.md               # provenance + reproduction instructions
├── d64.pt                  # distinguisher for 64-round KeeLoq
├── d96.pt
└── d128.pt

docs/phase3b-results/
├── delta_search.md         # chosen Δ and its eval accuracy
├── training_curves.md      # loss/accuracy plots per depth
└── benchmark.md            # neural-hybrid vs. pure-SAT
```

### 4.1 Module dependency graph

```
gpu_cipher (Phase 1)    ◄──┐
cipher     (Phase 1)    ◄──┤
attack     (Phase 1)    ◄──┐│
                           ││
neural/data                │└─ differences
neural/distinguisher  ◄── data
neural/evaluation     ◄── data, distinguisher
neural/key_recovery   ◄── distinguisher, cipher
neural/hybrid         ◄── key_recovery, attack
neural/cli_neural     ◄── all of the above
```

Strict layering: `neural/data.py` imports only from `keeloq.gpu_cipher` and `keeloq.cipher`. No circular imports.

## 5. Component specifications

### 5.1 `data.py` — training data generator

**Interface:**

```python
@dataclass(frozen=True)
class TrainingBatch:
    pairs: torch.Tensor      # shape (N, 2), dtype=torch.uint32 — ciphertext pairs
    labels: torch.Tensor     # shape (N,), dtype=torch.float32 — 0.0 / 1.0

def generate_pairs(
    rounds: int,
    delta: int,              # 32-bit plaintext XOR difference
    n_samples: int,
    seed: int,
    batch_size: int = 65536,
) -> Iterator[TrainingBatch]: ...
```

Emits batches in an infinite generator; caller slices with `itertools.islice`. Labels are balanced 50/50 (real vs. random) per batch. "Real" pairs: random `p_0`, `p_1 = p_0 ⊕ Δ`, both encrypted under the same random key. "Random" pairs: two independent random (key, plaintext) combinations. All encryption uses `gpu_cipher.encrypt_batch`.

**Test contract:**
- Label balance within ±1% per batch.
- For every real-labeled pair, `p_0 ⊕ p_1 == Δ` (recovered by brute-force re-decryption with the seed-generated key — feasible since the test knows seeds).
- Cross-check: a single real-labeled pair encrypted via `cipher.encrypt` matches the tensor output element-wise.
- Seed determinism: two calls with the same seed produce identical batches.

### 5.2 `differences.py` — chosen plaintext difference search

**Interface:**

```python
@dataclass(frozen=True)
class DeltaCandidate:
    delta: int
    validation_accuracy: float
    training_loss_final: float

def search_delta(
    rounds: int,
    candidates: list[int] | None = None,
    tiny_budget_samples: int = 200_000,
    tiny_budget_epochs: int = 2,
    seed: int = 0,
) -> list[DeltaCandidate]: ...  # sorted by accuracy descending
```

Defaults: candidate set is all 32 Hamming-weight-1 differences plus ~8 theoretically-motivated differences at the NLF tap positions `{31, 26, 20, 9, 1}`. For each candidate, trains a small throwaway model (~100k params) for 2 epochs on 200k samples (≤1 min on 5090), records validation accuracy. Returns sorted list — caller picks head.

**Test contract:**
- Returns at least one candidate with `validation_accuracy > 0.5` at a trivially distinguishable depth (e.g., 4 rounds).
- Reproducible under fixed seed.
- Candidate set is deduplicated (no Δ appears twice).

### 5.3 `distinguisher.py` — architecture + training + checkpoints

**Interface:**

```python
@dataclass(frozen=True)
class TrainingConfig:
    rounds: int
    delta: int
    n_samples: int           # total seen during training, e.g., 10_000_000
    batch_size: int          # e.g., 4096
    epochs: int              # e.g., 20
    lr: float                # e.g., 2e-3 (AdamW + cosine)
    weight_decay: float
    seed: int

@dataclass(frozen=True)
class TrainingResult:
    final_loss: float
    final_val_accuracy: float
    wall_time_s: float
    config: TrainingConfig
    history: list[dict]      # per-epoch loss + accuracy for plotting

class Distinguisher(nn.Module):
    """ResNet-1D-CNN over bits of (c_0, c_1). Gohr-style.
    Input: (N, 64) of 0/1 floats (32 bits of c_0 ∥ 32 bits of c_1).
    Output: (N,) sigmoid probability of "real" label.
    """
    def __init__(self, depth: int = 5, width: int = 32): ...

def train(config: TrainingConfig) -> tuple[Distinguisher, TrainingResult]: ...

def save_checkpoint(model: Distinguisher, result: TrainingResult, path: Path) -> None: ...
def load_checkpoint(path: Path) -> tuple[Distinguisher, TrainingResult]: ...
```

Architecture is fixed Gohr-style: input embedding → residual block stack → global pool → MLP head → sigmoid. Roughly 10M params at default `depth=5, width=32`. No hyperparameter search in Phase 3b — we use documented defaults and only tune the training config (data volume, epochs).

**Test contract:**
- Forward pass runs on GPU at batch sizes up to 8192 without OOM.
- Backward pass produces non-zero gradients.
- Training on a deliberately-biased synthetic dataset (all-real vs. all-random identical tensors) reaches ≥95% validation accuracy within 3 epochs — sanity check that the architecture and training loop can learn.
- Checkpoint round-trip: `load(save(model, result))` yields a byte-identical model state_dict.
- `TrainingResult.history` is JSON-serializable (for the plot artifacts).

### 5.4 `evaluation.py` — metrics on trained distinguisher

**Interface:**

```python
@dataclass(frozen=True)
class EvalReport:
    rounds: int
    delta: int
    n_samples: int
    accuracy: float
    roc_auc: float
    tpr_at_fpr_01: float     # TPR when FPR = 1%
    confusion: tuple[int, int, int, int]   # (TN, FP, FN, TP)

def evaluate(
    model: Distinguisher,
    rounds: int,
    delta: int,
    n_samples: int = 1_000_000,
    seed: int = 42,
) -> EvalReport: ...
```

Uses fresh pairs generated via `data.generate_pairs(...)` with a held-out seed different from training. Computes metrics in a single pass.

**Test contract:**
- A deliberately-trivial classifier (sigmoid bias = +5, always predicts "real") produces accuracy ≈ 0.5 and TPR=1 / FPR=1 on balanced data.
- A randomly-initialized model produces accuracy ≈ 0.5 ± 0.05.
- Reproducible under fixed `seed`.

### 5.5 `key_recovery.py` — Bayesian key-bit search

**Interface:**

```python
@dataclass(frozen=True)
class RecoveryResult:
    recovered_bits: dict[int, int]           # {bit_index: value}
    terminated_at_depth: int                 # remaining rounds when recovery stopped
    beam_history: list[dict]                 # for diagnostics: per-step beam state
    distinguisher_margin_history: list[float]

def recover_prefix(
    pairs: list[tuple[int, int]],            # (c_0, c_1) tuples
    distinguisher: Distinguisher,
    starting_rounds: int,
    max_bits_to_recover: int,
    beam_width: int = 8,
    margin_floor: float = 0.02,              # min margin; below this, beam grows
    max_beam_width: int = 256,
    device: torch.device = ...,
) -> RecoveryResult: ...
```

Sequentially guesses key bits from `K_{(starting_rounds-1) mod 64}` down to `K_{(starting_rounds-max_bits_to_recover) mod 64}`. For each step: partially decrypt one round using each candidate (0 / 1) for every beam entry, score each resulting (c_0', c_1') batch via the distinguisher, keep top `beam_width` by aggregate log-evidence. If the top-2 margin drops below `margin_floor`, increase beam width (up to `max_beam_width`) rather than prune.

Uses `keeloq.cipher._state_bit`, `_key_bit`, `core`, and the algebraic-inverse decryption logic from Phase 1 for partial decryption.

**Test contract:**
- On synthetic pairs at 8 rounds (trivially distinguishable) with a trained tiny model, recovers all 8 key bits correctly.
- Under `margin_floor=0.0`, beam width never grows beyond initial.
- `RecoveryResult.recovered_bits` keys are unique and in descending round-order.
- Reproducible under fixed RNG state.

### 5.6 `hybrid.py` — neural-prefix + SAT-suffix attack

**Interface:**

```python
@dataclass(frozen=True)
class HybridAttackResult:
    recovered_key: int | None
    status: Literal["SUCCESS", "WRONG_KEY", "UNSAT", "TIMEOUT", "NEURAL_FAIL", "BACKTRACK_EXHAUSTED"]
    bits_recovered_neurally: int
    neural_wall_time_s: float
    sat_wall_time_s: float
    verify_result: bool

def hybrid_attack(
    rounds: int,
    pairs: list[tuple[int, int]],
    distinguisher: Distinguisher,
    beam_width: int = 8,
    neural_target_bits: int | None = None,  # default: min(64, rounds - 32)
    sat_timeout_s: float = 60.0,
    max_backtracks: int = 8,
) -> HybridAttackResult: ...
```

Orchestration:
1. Call `recover_prefix(..., max_bits_to_recover=neural_target_bits)` → partial key.
2. Call `keeloq.attack.attack(..., key_hints=recovered_bits, encoder=xor_aware, solver_fn=cryptominisat, timeout_s=sat_timeout_s)` → complete the key.
3. If SAT returns UNSAT or TIMEOUT, backtrack: flip the last neural bit, re-run SAT. Bounded by `max_backtracks`.
4. On success, verify via `cipher.encrypt(p, recovered_key, rounds) == c` for every pair. Mandatory.

**Test contract:**
- At 64 rounds with a pre-trained checkpoint committed under `checkpoints/d64.pt`, `hybrid_attack` recovers the full 64-bit key on synthetic pairs in <5 min and verifies correctly.
- Under simulated "always-wrong neural output" (mocked), the backtrack logic eventually reaches `status=BACKTRACK_EXHAUSTED` rather than returning a wrong key.

### 5.7 `cli_neural.py` — CLI subcommands

Mounts a sub-app under `keeloq neural`:

| Command | Purpose |
|---|---|
| `keeloq neural train --rounds N --delta 0xΔ --samples M --out <ckpt>` | Train a distinguisher, save checkpoint |
| `keeloq neural evaluate --checkpoint <ckpt> --rounds N --samples M` | Produce an `EvalReport`, print as JSON |
| `keeloq neural recover-key --checkpoint <ckpt> --rounds N --pair pt:ct ...` | Run `hybrid_attack`, print result |
| `keeloq neural auto --rounds N --checkpoint-out <path>` | End-to-end: search Δ → train → evaluate → attack on fresh synthetic pairs |

Exit codes mirror Phase 1's `keeloq attack`: 0=SUCCESS, 1=CRASH, 2=WRONG_KEY, 3=UNSAT, 4=TIMEOUT, 5=NEURAL_FAIL, 6=BACKTRACK_EXHAUSTED.

**Test contract:** each command exercised via Typer's `CliRunner` on a tiny round count (≤8 rounds) with a small model; exit codes and stdout format verified. The full-scale `auto --rounds 64` test is `@pytest.mark.slow + gpu` and excluded from fast CI.

## 6. Training & attack pipeline

### 6.1 Offline training workflow

Two ways — granular or one-shot.

**Granular (explicit control):**

```bash
# Step 1: find the best input difference for 128-round KeeLoq (~40 min)
# Run the Δ search directly via a Python driver (scripts/search_delta.py
# wraps differences.search_delta and writes JSON):
uv run python scripts/search_delta.py --rounds 128 --out delta_128.json

# Step 2: train the distinguisher (~30 min)
keeloq neural train --rounds 128 --delta $(jq -r .best_delta delta_128.json) \
  --samples 10000000 --out checkpoints/d128.pt

# Step 3: evaluate
keeloq neural evaluate --checkpoint checkpoints/d128.pt --rounds 128 \
  --samples 1000000 > docs/phase3b-results/eval_d128.json
```

**One-shot (convenience):**

```bash
# auto does Δ search + train + evaluate + attack in one call, emitting
# a JSON report with every artifact path.
keeloq neural auto --rounds 128 --samples 10000000 --pairs 512 \
  --checkpoint-out checkpoints/d128.pt
```

### 6.2 Attack workflow

```bash
# Given committed checkpoints, attack a target ciphertext set:
keeloq neural recover-key --checkpoint checkpoints/d128.pt --rounds 128 \
  --pair ... --pair ... --beam-width 32 --sat-timeout 60
```

Output: recovered 64-bit key, neural/SAT wall-clock split, backtracks used.

### 6.3 End-to-end auto mode

```bash
# One-shot: train fresh and attack with defaults
keeloq neural auto --rounds 128 --pairs 512
```

Runs Δ search, trains, evaluates, synthesizes target pairs, runs hybrid_attack. Reports final success/failure with full pipeline provenance.

## 7. TDD strategy

Five-layer pyramid matching Phase 1's discipline, with per-layer tagging for CI selectivity.

**Layer 1 — Unit (milliseconds, in CI).** One test module per source module. Each module's tests written before implementation. No mocking of cipher or gpu_cipher — real calls, synthetic tiny inputs.

**Layer 2 — Property-based (Hypothesis, seconds, in CI).**
- Label balance on generated batches.
- `p_0 ⊕ p_1 == Δ` invariant on real-labeled pairs.
- GPU ↔ CPU cipher agreement on training-batch-sized samples (already covered in Phase 1; one smoke test in this phase confirms it still holds with `neural/data.py`'s batching).

**Layer 3 — End-to-end integration (seconds, in CI).**
Tiny 8-round end-to-end: `test_neural_hybrid::test_toy_attack_recovers_key` trains a tiny distinguisher on a few thousand samples, runs `hybrid_attack`, confirms the full 8-bit key (first 8 K positions) is recovered and verified.

**Layer 4 — Slow end-to-end (minutes to hours, NOT in CI, gated `slow + gpu`).**
- `test_hybrid_attack_64_rounds_from_checkpoint`: loads committed `checkpoints/d64.pt`, attacks 64-round synthetic pairs, full key recovery in <5 min.
- (Optional, if `d128.pt` ships): equivalent at 128 rounds.

**Layer 5 — Benchmarks (manual, hours).** `keeloq benchmark --matrix benchmarks/neural_matrix.toml` runs the neural-hybrid attack at 64 / 96 / 128 rounds alongside pure-SAT baseline, outputs CSV + markdown.

**CI time budget:** Layers 1–3 complete in <90 s on the 5090 box. Layer 4 adds ~5–10 min on the 5090 (slow-tagged, explicitly invoked, not in PR gate).

**TDD invariant throughout:** `cipher.encrypt(plaintext, recovered_key, rounds) == ciphertext` is the truth-check at every stage. Distinguisher or key-recovery bugs surface as this assertion failing.

## 8. Error handling

**Distinct failure modes and exit codes:**

| Code | Status | Meaning |
|---|---|---|
| 0 | SUCCESS | Key recovered + cipher-verified |
| 1 | CRASH | Unhandled exception |
| 2 | WRONG_KEY | Solver returned SAT but verification failed (should be impossible with mandatory verify; indicates a bug) |
| 3 | UNSAT | SAT suffix proved inconsistent with neural prefix AND all backtracks exhausted |
| 4 | TIMEOUT | SAT timeout |
| 5 | NEURAL_FAIL | `DistinguisherTrainingError` (training didn't converge) |
| 6 | BACKTRACK_EXHAUSTED | Neural prefix consistently wrong, backtrack budget consumed |

**Invariant-violation policy:** loud failure, never silent. Same flat exception hierarchy as Phase 1 (`KeeloqError` base + `DistinguisherTrainingError`, `HybridAttackFailed`, reuse `SolverError`, `VerificationError`).

**CUDA OOM:** explicit exception propagation, no silent retry. Batch sizes configurable; defaults chosen conservatively for 32GB VRAM.

**Mandatory cipher verify:** every returned non-null key passes through `cipher.encrypt(p, key, rounds) == c` for every input pair. Reuse of Phase 1's verification step.

**Skip-with-reason policy:** CUDA-less machines skip `@pytest.mark.gpu`. Missing committed checkpoints skip `@pytest.mark.slow` tests that depend on them.

## 9. Packaging, platform & compute budget

**New pip deps:** `numpy` added to runtime deps (PyTorch's soft dep; training code needs it directly). Removes the `filterwarnings = ["ignore:Failed to initialize NumPy:UserWarning"]` Phase 1 workaround.

**No other new heavy deps.** No `lightning`, no `transformers`, no `wandb`. Training loop is hand-rolled.

**Optional dev extra:** `matplotlib` in a new `[project.optional-dependencies.plots]` group for `EvalReport.plot()`.

**Platform:** Linux x86_64, Python 3.12+, CUDA 12.8+ with RTX 5090 Blackwell sm_120. Same constraints as Phase 1.

**Compute budget (single 5090, 32 GB VRAM):**

| Task | Estimated wall-clock |
|---|---|
| Δ search (40 candidates × 1 min) | ~40 min |
| d64 training (10⁷ samples, 20 epochs) | ~30 min |
| d96 training | ~30 min |
| d128 training | ~30 min |
| Attack at 64 rounds, beam 8 | seconds to a few min |
| Attack at 128 rounds, beam 32 | minutes to ~1 hour (empirical) |
| Full benchmark matrix | ~2 hours |
| **Total** | **~5–8 GPU-hours** |

**Checkpoint commits:** `checkpoints/d64.pt`, `d96.pt`, `d128.pt` committed to the repo (each ~10–50 MB, total <150 MB). Experimental / superseded checkpoints in `.gitignored` `scratch/`. A `checkpoints/README.md` documents the training configs and reproduction commands.

## 10. Success criteria

Phase 3b is done when:

1. ✅ Distinguishers trained and evaluated at 64, 96, and 128 rounds. Each has a committed checkpoint + `EvalReport` artifact + training curves in `docs/phase3b-results/`. Reproducible from committed seed + config.
2. ✅ `keeloq neural recover-key --checkpoint checkpoints/d64.pt --rounds 64 ...` recovers the full 64-bit key on synthetic test pairs in <5 min wall-clock via `hybrid_attack`. **(Floor commitment.)**
3. ✅ `keeloq neural recover-key --checkpoint checkpoints/d128.pt --rounds 128 ...` either (a) recovers the full 64-bit key, or (b) produces a documented negative result with diagnostics (neural bits recovered before margin collapse, SAT handoff outcome, analysis of why it failed). **(Ambition target — either outcome is acceptable.)**
4. ✅ Full test suite green on the 5090 box, including the 64-round slow+gpu test. CI (fast lane) still under 90 s.
5. ✅ `benchmarks/neural_matrix.toml` + `bench_neural.py` run the neural-hybrid attack vs. Phase 1 pure-SAT at 64 / 96 / 128 rounds, writing CSV + markdown summary. Legacy README baseline (160 rounds / 25 hints / 2 pairs / 14 h) is NOT re-benchmarked here — this is already covered by Phase 1's `matrix.toml`.
6. ✅ One canonical result sentence in `benchmarks/neural_summary.md`, e.g.: "Neural-hybrid attack recovers a 64-bit KeeLoq key at N rounds in T seconds using D ciphertext pairs, compared to Phase 1 pure-SAT at (N, T_SAT, D_SAT)."

## 11. Open questions

None. All design-level questions were resolved during the brainstorming session. Implementation-level choices (exact ResNet block count / width, training hyperparameter tuning, beam-width defaults) are left to implementation-time empirical tuning, with the constraint that §5's test contracts and §10's success criteria must hold.

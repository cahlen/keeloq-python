# Phase 1 — Foundation: Design Spec

**Date:** 2026-04-22
**Author:** brainstorm session (Cahlen + Claude)
**Status:** Approved
**Target:** `keeloq-python` — 2026 modernization of the 2015 KeeLoq algebraic/SAT cryptanalysis pipeline.

---

## 1. Context

The repository (created 2015) implements an algebraic-cryptanalysis attack on reduced-round (160-round) KeeLoq: cipher equations are expressed in ANF, converted to DIMACS CNF via a SageMath text-file hack, and fed to miniSAT. Reported results in the original README: ~25 hint bits of key and two plaintext/ciphertext pairs required; ~14h solve time.

The 2026 modernization is conceptually split into four phases:

- **Phase 1 (this spec): Foundation.** Python 3 rewrite, modern tooling, dual encoder (pure CNF + XOR-aware), GPU bit-sliced reference cipher for property tests, single-binary Typer CLI, rigorous TDD.
- **Phase 2: Research harness.** Configurable benchmark matrix, solver-swap infrastructure, structured metrics.
- **Phase 3a: SAT improvement track.** Gröbner preprocessing, ML-guided branching, advanced XOR techniques.
- **Phase 3b: Neural cryptanalysis track.** Gohr-style neural differential distinguisher on the RTX 5090.

This spec covers only Phase 1. Deferred items are tracked in project memory so subsequent phases pick them up intact.

## 2. Goals

Phase 1 delivers a Python 3 codebase that:

1. Reproduces the 2015 attack's end-to-end behavior on the same test vectors, with a dramatically faster baseline thanks to a modern solver (CryptoMiniSat) and XOR-aware encoding.
2. Is round-count-parameterized throughout, so small-round configurations (e.g. 16- and 32-round) are first-class test targets solvable in seconds.
3. Has a dual encoder (pure CNF and XOR-aware) with a TDD invariant that both encoders must recover the same key on the same inputs — a strong cross-check catching variable-indexing bugs.
4. Uses the RTX 5090 for a GPU bit-sliced KeeLoq implementation that serves as an independent test oracle for the Python cipher via Hypothesis property tests.
5. Preserves the 2015 scripts in `legacy/` as a frozen external oracle for cipher output and ANF generation.
6. Ships a single `keeloq` binary with composable subcommands driving the attack pipeline.
7. Follows strict TDD discipline: tests written first, five-layer pyramid, CI completes in under 60 seconds including a 32-round end-to-end key recovery.

## 3. Non-goals (deferred to later phases)

See `~/.claude/projects/-home-cahlen-dev-keeloq-python/memory/phase1_deferred.md` for the authoritative list. Highlights:

- No Gröbner / F4 / F5 preprocessing (Phase 3a).
- No ML-guided SAT branching (Phase 3a).
- No neural differential cryptanalysis (Phase 3b).
- No TOML/YAML config-driven attacks — CLI args only (Phase 2).
- No distributed / remote solver execution (Phase 2).
- No Jupyter / REPL integration.
- No Docker / Nix / devcontainer — `uv sync` is the install story.
- No macOS / Windows — Linux x86_64 only.
- GPU usage is limited to the bit-sliced reference cipher; the solver path stays CPU.

## 4. Repository layout

```
keeloq-python/
├── legacy/                          # untouched 2015 scripts (Py2)
│   ├── keeloq-python.py
│   ├── keeloq160-python.py
│   ├── sage-equations.py
│   ├── polynomial-vars.py
│   ├── parse-miniSAT.py
│   └── sage-CNF-convert.txt
├── src/keeloq/
│   ├── __init__.py
│   ├── cipher.py                    # readable Python KeeLoq, rounds-parameterized
│   ├── gpu_cipher.py                # bit-sliced CUDA cipher (PyTorch)
│   ├── anf.py                       # ANF polynomial system generation
│   ├── encoders/
│   │   ├── __init__.py              # Encoder protocol
│   │   ├── cnf.py                   # pure CNF encoder
│   │   └── xor_aware.py             # CryptoMiniSat XOR-native hybrid encoder
│   ├── solvers/
│   │   ├── __init__.py              # Solver protocol
│   │   ├── cryptominisat.py         # pycryptosat wrapper
│   │   └── dimacs_subprocess.py     # kissat / minisat via subprocess
│   ├── attack.py                    # full pipeline composition
│   └── cli.py                       # Typer entry point
├── tests/
│   ├── test_cipher.py
│   ├── test_gpu_cipher.py
│   ├── test_anf.py
│   ├── test_encoders_cnf.py
│   ├── test_encoders_xor.py
│   ├── test_encoders_agree.py       # cross-validation (the TDD crown jewel)
│   ├── test_solvers.py
│   ├── test_attack.py               # 16- and 32-round end-to-end
│   ├── test_cli.py
│   ├── test_compat.py               # legacy Py2 parity
│   └── compat_helpers.py            # subprocess wrappers for legacy/ invocations
├── benchmarks/
│   ├── bench_attack.py
│   └── matrix.toml                  # (minimal; real config-matrix support is Phase 2)
├── docs/superpowers/specs/
│   └── 2026-04-22-phase1-foundation-design.md  # this file
├── pyproject.toml
├── uv.lock
├── CLAUDE.md
├── README.md
└── Makefile                         # convenience targets: check, test, ci-local, bench
```

### 4.1 Module dependency graph

```
cipher        ←  (pure Python, no deps)
gpu_cipher    ←  torch
anf           ←  cipher (shares round-function constants via a common module)
encoders/*    ←  anf
solvers/*     ←  encoders (via protocol; no circular imports)
attack        ←  anf, encoders, solvers, cipher (for verification)
cli           ←  attack
```

Strict layering is enforced — `anf` must not import `encoders`; `cipher` must not import anything project-specific. A `ruff` rule or a custom test can assert this.

## 5. Component specifications

Each component is specified as a **contract** (what it must make true) because Phase 1 is test-first. Tests are written before implementation.

### 5.1 `cipher.py` — Reference KeeLoq

**Interface:**

```python
def encrypt(plaintext: BitVec32, key: BitVec64, rounds: int) -> BitVec32
def decrypt(ciphertext: BitVec32, key: BitVec64, rounds: int) -> BitVec32
def core(a: int, b: int, c: int, d: int, e: int) -> int
def round_update(state: BitVec32, key_bit: int) -> tuple[BitVec32, int]  # returns new_state, new_bit
```

`BitVec32` / `BitVec64` are `int` at runtime with a `typing.NewType` wrapper for clarity; internal representation is little-endian bit index 0 = LSB. No hidden state.

**Test contract:**
- Round-trip: `decrypt(encrypt(pt, k, r), k, r) == pt` for `r ∈ {0, 1, 2, 16, 32, 64, 128, 160, 528}` and fixed KATs.
- `core` matches its algebraic definition on all 32 input combinations.
- At 160 rounds on the README's test vector, `encrypt` matches `legacy/keeloq160-python.py` output byte-for-byte (compat test — executes the legacy script in a `python2` subprocess).
- At 528 rounds on the README's test vector, matches `legacy/keeloq-python.py` (note: the 528-round and 160-round scripts differ in decrypt key index — `cipher.py` handles both correctly via its parameterization).
- Property test: `Hypothesis.integers()` for plaintext and key, round count drawn from `{1, ..., 200}`, round-trip holds.

### 5.2 `gpu_cipher.py` — Bit-sliced CUDA cipher

**Interface:**

```python
def encrypt_batch(
    plaintexts: torch.Tensor,  # shape (N,), dtype=torch.uint32
    keys: torch.Tensor,        # shape (N, 2), dtype=torch.uint32 (low, high halves of 64-bit key)
    rounds: int,
) -> torch.Tensor              # shape (N,), dtype=torch.uint32
```

Uses the bit-slicing trick: the whole batch advances through rounds in lockstep via bitwise ops on 32-bit lanes. Processes ≥10⁶ encryptions per second on the 5090. CUDA-only; raises `RuntimeError` with a clear message if CUDA is unavailable.

**Test contract:**
- For 10⁶ random (key, plaintext) pairs across `rounds ∈ {16, 32, 64, 128, 160}`, GPU output matches `cipher.py` output element-wise.
- Test skipped (not passed) with `reason="CUDA unavailable"` on machines without CUDA — surfaced in the pytest summary.
- Smoke test: 1024-batch at 160 rounds completes in <100ms on the 5090 box.

### 5.3 `anf.py` — ANF polynomial system

**Interface:**

```python
@dataclass(frozen=True)
class BoolPoly:
    """Sum of monomials over GF(2). Monomials are frozensets of variable names."""
    monomials: frozenset[frozenset[str]]  # {frozenset(), frozenset({"K0"}), frozenset({"K0","L3"}), ...}

def variables(rounds: int, num_pairs: int = 1) -> list[str]
    # K-variables are shared across pairs (the whole point — same key for all pairs).
    # L/A/B variables are per-pair: L{bit}_p{pair_index}, A{round}_p{pair_index}, B{round}_p{pair_index}.
    # Returns ordered list: K0..K63,
    #   then for each pair p in 0..num_pairs-1:
    #     A0_p..A{rounds-1}_p, B0_p..B{rounds-1}_p, L0_p..L{rounds+31}_p.

def round_equations(round_idx: int, pair_idx: int = 0) -> tuple[BoolPoly, BoolPoly, BoolPoly]
    # Returns (eq1, eq2, eq3) for round `round_idx` in pair `pair_idx`, matching
    # sage-equations.py:31-33 with L/A/B names namespaced by pair.

def system(
    rounds: int,
    pairs: list[tuple[BitVec32, BitVec32]],   # [(plaintext, ciphertext), ...]; at least 1
    key_hints: dict[int, int] | None = None,  # {bit_index: value} for hinted K_i bits
) -> list[BoolPoly]
    # Returns the full polynomial system:
    #   for each pair: plaintext bindings + ciphertext bindings + 3*rounds round equations
    #   + optional key hint bindings (shared across pairs).
    # Single-pair attacks pass a singleton list.

def substitute(poly: BoolPoly, assignment: dict[str, int]) -> int
    # Evaluate a polynomial under a total or partial assignment. Raises if partial.
```

`BoolPoly` is minimal by design — no Gröbner machinery, no reduction, no factoring. Just arithmetic over GF(2). The encoders consume it.

**Test contract:**
- Variable count with `num_pairs=1` matches `64 + (rounds + rounds + 32 + rounds)` = `3*rounds + 96`. With `num_pairs=P`, count is `64 + P * (3*rounds + 32)`.
- For random (key, plaintext) at round count `r`, derive the true `L`, `A`, `B` values from `cipher.py`, substitute into every equation in `system(...)`, assert all equal 0. (This is the "our equations are correct" test — arguably the single most important test in the repo.) Test covers both `num_pairs=1` and `num_pairs=2` cases.
- At 160 rounds on the README's test vector with `num_pairs=1`, the generated ANF string matches the output of `legacy/sage-equations.py`'s `anf.txt` byte-for-byte, **modulo the per-pair variable renaming** (the legacy script uses bare `L0`, `A0`, `B0`; our single-pair output uses `L0_p0`, `A0_p0`, `B0_p0`). A normalizer in the compat test strips `_p0` suffixes before the byte-for-byte comparison.
- `core` polynomial in `anf.py` is derivable from `cipher.py::core` via truth-table equivalence — unit test asserts the two agree on all 32 inputs.

### 5.4 `encoders/cnf.py` — Pure CNF encoder

**Interface:**

```python
@dataclass(frozen=True)
class CNFInstance:
    num_vars: int
    clauses: list[list[int]]
    var_names: list[str]    # index i → variable name; clauses use 1-indexed ints, negative = negation

def encode(system: list[BoolPoly]) -> CNFInstance
def to_dimacs(instance: CNFInstance) -> str
def from_dimacs(text: str) -> CNFInstance
```

Tseitin transformation for nonlinear monomials; naive clause expansion for XOR chains (the known-exponential case). For a 160-round instance we expect ~O(10⁵) clauses; this is the baseline our XOR encoder will beat.

**Test contract:**
- Trivial ANF systems (e.g., `x + y = 0`, `x*y + 1 = 0`) produce the minimal canonical CNF.
- For any `(rounds, pt, ct, hints)` instance where the true key is hinted sufficiently, `to_dimacs(encode(system))` piped through CryptoMiniSat yields a SAT assignment that satisfies the original ANF (round-trip correctness).
- `from_dimacs(to_dimacs(x)) == x` for random CNF instances.
- Clause count growth with rounds matches expected complexity (sanity check).

### 5.5 `encoders/xor_aware.py` — XOR-native hybrid encoder

**Interface:**

```python
@dataclass(frozen=True)
class HybridInstance:
    num_vars: int
    cnf_clauses: list[list[int]]
    xor_clauses: list[tuple[list[int], int]]  # (vars, rhs) where rhs is 0 or 1
    var_names: list[str]

def encode(system: list[BoolPoly]) -> HybridInstance
```

Each polynomial in the system is split into its linear and nonlinear parts. The linear part becomes a single XOR clause. The nonlinear part is Tseitin-transformed into CNF clauses. For KeeLoq this is the game-changer — round equations have one long linear XOR chain plus a handful of quadratic terms.

**Test contract:**
- XOR clause count after encoding equals the number of round equations with a non-empty linear part (one XOR per such equation).
- Round-trip: decode a satisfying assignment, substitute back into the original ANF, every equation is 0.
- Same fuzz harness as `encoders/cnf.py`: random small systems, both encoders fed to CryptoMiniSat, both return assignments satisfying the original ANF.

### 5.6 `encoders/__init__.py` — Encoder protocol

```python
class Encoder(Protocol):
    def encode(self, system: list[BoolPoly]) -> SolverInstance: ...

SolverInstance: TypeAlias = CNFInstance | HybridInstance
```

Phase 2 adds more encoders; this protocol is forward-compatible.

### 5.7 `solvers/cryptominisat.py` — pycryptosat wrapper

**Interface:**

```python
@dataclass(frozen=True)
class SolveResult:
    status: Literal["SAT", "UNSAT", "TIMEOUT"]
    assignment: dict[str, int] | None
    stats: SolverStats           # restarts, propagations, conflicts, wall_time_s, ...

def solve(instance: SolverInstance, timeout_s: float) -> SolveResult
```

Accepts both `CNFInstance` (via `add_clause`) and `HybridInstance` (via `add_clause` + `add_xor_clause`). Passes `timeout_s` to the solver's native timeout mechanism — no process-kill games.

**Test contract:**
- Trivial SAT instance round-trips correctly with stable variable assignment ordering.
- Trivial UNSAT instance returns `status="UNSAT"`, `assignment=None`.
- Timeout test: a constructed hard-ish instance with a 1-second timeout returns `status="TIMEOUT"` and non-zero stats.

### 5.8 `solvers/dimacs_subprocess.py` — External binary solvers

**Interface:**

```python
def solve(
    instance: CNFInstance,
    solver_binary: str,   # "kissat", "minisat", or a path
    timeout_s: float,
) -> SolveResult
```

Writes DIMACS to a tempfile, invokes `solver_binary` with appropriate flags, parses output. Only supports `CNFInstance` (external solvers don't speak XOR natively). On solver crash, captures stderr and raises `SolverError` with full output.

**Test contract:**
- Parses the exact `out.result` format from `legacy/parse-miniSAT.py` (compat test).
- Handles `s SATISFIABLE`, `s UNSATISFIABLE`, and empty-assignment lines robustly.
- Skip-with-reason if the requested binary isn't on PATH.

### 5.9 `attack.py` — Pipeline orchestration

**Interface:**

```python
@dataclass(frozen=True)
class AttackResult:
    recovered_key: BitVec64 | None
    status: Literal["SUCCESS", "WRONG_KEY", "UNSAT", "TIMEOUT", "CRASH"]
    solve_result: SolveResult
    verify_result: bool       # True iff for every pair, cipher.encrypt(pt, recovered_key, rounds) == ct
    encoder_used: str
    solver_used: str

def attack(
    rounds: int,
    pairs: list[tuple[BitVec32, BitVec32]],   # at least 1 (pt, ct) pair
    key_hints: dict[int, int] | None = None,
    encoder: Encoder = ...,
    solver_fn: Callable[[SolverInstance, float], SolveResult] = ...,
    timeout_s: float = 3600.0,
) -> AttackResult
```

Composes: `anf.system(rounds, pairs, key_hints)` → `encoder.encode(...)` → `solver_fn(...)` → key extraction → **mandatory verification** (the recovered key must correctly encrypt *every* pair's plaintext to its ciphertext). Verification is not optional; the `status` field distinguishes SAT-correct (`SUCCESS`) from SAT-wrong-key (`WRONG_KEY`).

**Test contract (solvability-aware):**

Underdetermination is real — a single (pt, ct) pair gives 32 bits of constraint against 64 unknown key bits, so a 0-hint single-pair attack is fundamentally ambiguous at any round count. Tests are designed around three regimes:

- **Heavily-hinted, single-pair (fast CI)**: 16-round attack, 1 pair, 48 hint bits (16 unknowns), recovers k in <1s — both encoders.
- **Moderately-hinted, single-pair (CI)**: 32-round attack, 1 pair, 32 hint bits (32 unknowns), recovers k in <10s — both encoders.
- **Multi-pair, no hints (CI)**: 32-round attack, 2 pairs, 0 hints, recovers k in <10s — both encoders. This exercises the multi-pair code path and mirrors the legacy README's "two pairs, fewer hints" strategy at a tractable round count.

Additional tests:
- `test_encoders_agree.py`: for each of the three regimes above, both encoders produce `SUCCESS` with identical recovered keys.
- Underdetermined case (negative test): 16-round attack, 1 pair, 0 hints — assert `status="WRONG_KEY"` (the solver finds *a* key but verification detects it's not the original). This proves our verification gate works.
- UNSAT case: 16-round attack with a hint deliberately set to the wrong bit — assert `status="UNSAT"` (solver proves inconsistency).

### 5.10 `cli.py` — Typer entry point

Subcommands:

| Command | Purpose |
|---|---|
| `encrypt` | Run `cipher.encrypt` |
| `decrypt` | Run `cipher.decrypt` |
| `generate-anf` | Emit the ANF polynomial system (JSON or text) |
| `encode` | Read ANF from stdin, emit DIMACS or hybrid JSON |
| `solve` | Read instance from stdin, emit `SolveResult` JSON |
| `verify` | Check a recovered key decrypts correctly |
| `attack` | Full in-process pipeline, printing recovered key + stats |
| `benchmark` | Run the benchmark matrix, writing CSV + markdown |

All subcommands accept keys/plaintexts/ciphertexts as bit strings, `0x...` hex, or `-` for stdin. Inputs are validated by pydantic at the CLI boundary; internal functions trust their inputs.

Exit codes:
- `0` — success
- `1` — unexpected error / crash
- `2` — SAT but wrong key (underdetermined)
- `3` — UNSAT
- `4` — timeout

**Test contract:** each subcommand invoked via Typer's `CliRunner`; exit codes, stdout format, stderr on error.

## 6. Data flow

The CLI subcommands compose via Unix pipes:

```bash
keeloq generate-anf --rounds 32 --plaintext 01100010... --ciphertext 01101000... --hint-bits 0 \
  | keeloq encode --encoder xor \
  | keeloq solve --solver cryptominisat \
  | keeloq verify --original-key 00110100...
```

`keeloq attack` is equivalent to that pipeline but runs in-process, avoiding JSON serialization overhead. The piped form exists for debugging and future Phase 3a experiments where we'll want to inspect intermediates.

Stage-boundary serialization is JSON validated by pydantic models. DIMACS is supported for interop with external tools.

## 7. TDD strategy

Five-layer pyramid. Tests written before implementation.

**Layer 1 — Unit (milliseconds).** One test module per source module. Parameterized over round counts where applicable.

**Layer 2 — Property-based (seconds, Hypothesis).**
- `test_gpu_cipher::test_matches_reference` — 10⁶ random (key, pt, rounds); GPU ↔ Python cipher agree.
- `test_anf::test_true_solution_satisfies_system` — derived truth values satisfy every equation.
- `test_encoders_agree::test_both_recover_same_key` — CNF and XOR encoders produce identical key recoveries at small rounds with moderate hints.

Hypothesis profile `ci` = 200 examples/property; `dev` = 10⁴.

**Layer 3 — End-to-end (seconds, in CI).**
- 16-round attack, 1 pair, 48 hints, both encoders.
- 32-round attack, 1 pair, 32 hints, both encoders.
- 32-round attack, 2 pairs, 0 hints, both encoders (multi-pair code path).
- Underdetermined negative test (status=WRONG_KEY detected by verification).
- UNSAT negative test (bad hint → solver proves inconsistency).
- CLI integration via `CliRunner`.

**Layer 4 — Compatibility (seconds, in CI).**
- Cipher output matches `python2 legacy/keeloq160-python.py` on the README test vector.
- ANF output matches `python2 legacy/sage-equations.py` byte-for-byte.
- DIMACS subprocess parser matches `legacy/parse-miniSAT.py` on canned `out.result`.

Skipped with reason (not silently passed) if `python2` is unavailable.

**Layer 5 — Benchmarks (minutes to hours, never in CI).** Matrix of rounds × hint-bits × encoder × solver. Output CSV + markdown. Headline metric: "160 rounds, 25 hints, 2 pt/ct pairs, XOR + CryptoMiniSat: X minutes" — directly comparable to the legacy README's 14 hours.

**CI time budget:** Layers 1–4 complete in <60s on a laptop; <30s on the 5090 box.

**Strict TDD discipline:**
- For each module, commit the failing test first (`test: ...` prefix), then the implementation (`impl: ...` prefix).
- Red-green-refactor pattern visible in `git log`.

**No-mock policy:** The cipher, the encoder, and the solver are never mocked in unit tests. Mocks appear only at the CLI layer for subcommands that would otherwise require a live solver.

## 8. Error handling

**Four non-success outcomes from a solve attempt**, each with a distinct exit code and log signature:

1. **UNSAT** (exit 3): instance has no solution. Usually means caller's hint bits are inconsistent with the (pt, ct) pair.
2. **SAT but wrong key** (exit 2): solver returned an assignment but re-encryption with the recovered key doesn't yield the original ciphertext. Classic underdetermined-system case from the legacy README.
3. **TIMEOUT** (exit 4): solver exceeded `--timeout`. Partial stats emitted.
4. **SOLVER_CRASH** (exit 1): external solver crashed or produced unparseable output. Full stderr captured.

**Verification is mandatory.** Every `SolveResult` with `status="SAT"` is re-checked via `cipher.encrypt(pt, recovered_key, rounds) == ct` before the outer result's `status` is set to `SUCCESS`. No opt-out.

**Invariant violations** raise `InvariantError` with detail — never silent. Examples: ANF references undeclared variable, encoder emits clause with unknown variable, assignment missing a key variable.

**Exception hierarchy (flat, four classes):** `InvariantError`, `SolverError`, `EncodingError`, `VerificationError`. All extend `KeeloqError` (base).

**Input validation** happens at the CLI boundary only (pydantic models). Internal functions trust inputs.

**External-dependency policy:** if an external dep (CUDA, `python2`, `kissat`, `minisat`) is missing, the relevant tests skip with a reason string. We do *not* substitute faked/mocked equivalents.

**Retry / resilience:** none. Determinism matters more than resilience. A solver crash is a bug report, not a retry candidate.

## 9. Packaging & platform

**Python:** 3.12+.
**Package manager:** `uv` (Astral). `pyproject.toml` + `uv.lock` committed. `uv sync --frozen` is the install command.

**Runtime dependencies:**
- `torch >= 2.5` (CUDA 12.8+ wheels for Blackwell `sm_120` / RTX 5090)
- `pycryptosat` (CryptoMiniSat with XOR)
- `typer[all]`
- `pydantic >= 2`
- `structlog`

**Dev/test:**
- `pytest`, `pytest-xdist`, `pytest-timeout`, `pytest-cov`
- `hypothesis`
- `ruff` (lint + format)
- `mypy` (strict on `src/`, lax on `tests/`)

**Optional system deps (skip-with-reason if absent):**
- `python2` — legacy compat tests
- `kissat` — external solver backend
- `minisat` — legacy parity
- CUDA 12.8+ + RTX 5090 — GPU cipher tests

**Platform:** Linux x86_64 only. Other platforms not broken but not tested.

**Reproducibility:** `uv.lock` is source of truth. No unpinned transitive deps. Wheels only — no source builds at install time.

**`legacy/` is frozen.** Never modified. Invocation wrappers live in `tests/compat_helpers.py`.

## 10. Success criteria

Phase 1 is done when:

1. ✅ All tests pass on the 5090 box, including GPU and legacy-compat layers.
2. ✅ CI (Linux x86_64, Python 3.12, solvers installed, `python2` installed, no GPU) passes Layers 1–4 in under 60 seconds.
3. ✅ `keeloq attack --rounds 32 --pairs pt1:ct1,pt2:ct2 --hint-bits 0 --encoder xor --solver cryptominisat` recovers the correct key in under 10 seconds. (Note: CLI flag syntax for multi-pair is indicative; exact form is implementation-time detail, but multi-pair must be expressible.)
4. ✅ `keeloq benchmark` produces a CSV + markdown table with at minimum the 160-round/25-hint/2-pair baseline number directly comparable to the 2015 README's 14h result.
5. ✅ `uv sync --frozen` + `make ci-local` works from a clean checkout on the 5090 box.
6. ✅ The `test_encoders_agree` TDD crown jewel is green.
7. ✅ The `test_compat::test_anf_matches_legacy_anf_txt` test is green (or the mismatch is investigated and one side corrected, with rationale committed).

## 11. Open questions

None. All design-level questions were resolved during the brainstorming session. Implementation-level decisions (e.g., exact bit-slicing kernel layout, exact Tseitin gadget choice) are left to the implementation phase with the constraint that the test contracts in §5 must hold.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Phase 3b status (2026)

Neural differential cryptanalysis pipeline in `src/keeloq/neural/`, following
Gohr 2019 adapted for KeeLoq's 1-bit-per-round key schedule. CLI:

- `keeloq neural train --rounds N --delta 0xΔ --samples M --out <ckpt>`
- `keeloq neural evaluate --checkpoint <ckpt> --rounds N`
- `keeloq neural recover-key --checkpoint <ckpt> --rounds N --diff-pair <c0>:<c1> --sat-pair <pt>:<ct>`
- `keeloq neural auto --rounds N --trained-depth D --samples M --checkpoint-out <path>`

**Gohr-pattern constraint.** A distinguisher trained at depth D gives strong signal only on
D-round pairs. Peeling K rounds with a single distinguisher works well only for small K. For
deep-round attacks, train a family of distinguishers at strategic depths (e.g. D=56 for a
64-round attack peeling K=8 rounds). The `auto` subcommand handles this automatically.
`neural-target-bits` controls how many prefix key-bits the neural phase covers before handing
off to SAT.

**Checkpoint policy.** Checkpoints are NOT committed to the repo by default (external GPU
contention prevented full training runs for d64.pt/d96.pt/d128.pt). Produce them via:

    uv run keeloq neural auto --rounds 64 --trained-depth 56 \
        --samples 10000000 --pairs 512 --checkpoint-out checkpoints/d64.pt

The regression test `tests/test_neural_e2e_64r.py` auto-skips when `checkpoints/d64.pt` is
absent. Benchmark runner (`benchmarks/bench_neural.py` + `benchmarks/neural_matrix.toml`) is
smoke-safe against missing checkpoints.

**Hybrid-attack CLI.** The `recover-key` subcommand takes two distinct argument streams:
`--diff-pair <c0>:<c1>` (differential ciphertext pairs for the neural distinguisher) and
`--sat-pair <pt>:<ct>` (plaintext:ciphertext pairs for the SAT suffix). These are separate
because a differential attack doesn't require known plaintexts; only the SAT phase does.

**CUDA XOR limitation.** `rshift_cuda` for uint32 is not implemented in PyTorch 2.11 + CUDA
13. All XOR / delta application is done CPU-side before tensors are moved to the GPU.

Checkpoints in `checkpoints/` with training metadata embedded. Results under
`docs/phase3b-results/`. Spec/plan in `docs/superpowers/`.

## Phase 1 status (2026)

The repo is mid-migration from the 2015 Python 2 scripts (now in `legacy/`, frozen) to a
Python 3 modernized pipeline in `src/keeloq/`. Driver is `keeloq` (a Typer CLI,
installed via `uv sync --all-extras`). Key entry points:

- `keeloq encrypt / decrypt` — the cipher, rounds-parameterized.
- `keeloq generate-anf | encode | solve | verify` — pipeline stages composable via Unix pipes (JSON between stages).
- `keeloq attack` — the pipeline in-process. Use `--pair pt:ct` (repeatable) for multi-pair attacks.
- `keeloq benchmark` — matrix runner driven by `benchmarks/matrix.toml`.

Strict TDD discipline. Commits are prefixed `test:`, `impl:`, `refactor:`, `feat:`, `docs:`, `ci:`, `chore:` — do NOT squash them.

**Cross-validation TDD invariant.** Two encoders (`encoders/cnf.py`, `encoders/xor_aware.py`)
must recover the same key on the same inputs; `tests/test_encoders_agree.py` enforces this.
Variable-indexing bugs are the biggest risk in this domain, and this cross-check is the
primary defense.

**Legacy is frozen and runs via Docker.** Never modify files in `legacy/`. The compat tests
run the 2015 Python 2 scripts inside an ephemeral `python:2.7` container (not host
python2, which is EOL). `tests/compat_helpers.py` handles the docker invocation. Tests
mark `@pytest.mark.legacy`; they auto-skip if docker or the python:2.7 image is absent.

**Key-schedule quirk.** KeeLoq's key cycles at 64 rounds. Attacks at `rounds < 64`
fundamentally cannot constrain `K{rounds}..K63` — those bits must be hinted, or the
attack must run at rounds ≥ 64 with enough plaintext/ciphertext pairs to over-determine
the system. A clean 0-hint key recovery needs 64 rounds + ~4 pairs. This is why the
benchmark `matrix.toml` rows with rounds < 64 always pin `hint_bits >= (64 - rounds)`.

**GPU bit-sliced cipher.** `src/keeloq/gpu_cipher.py` is a PyTorch bit-sliced KeeLoq
used as a property-test oracle. Requires CUDA; tests auto-skip on CUDA-less machines.
Works internally on int64 tensors because `rshift_cuda` isn't implemented for uint32 on
PyTorch 2.11 + CUDA 13; the public API uses uint32.

## What this repo is

Research code for an algebraic / SAT-based attack on a reduced-round (160-round) version of the KeeLoq block cipher, originally authored 2015. It is not a library or a product — it's a small pipeline of one-shot scripts that cooperate via files (`anf.txt`, `vars.txt`, the CNF output, `out.result`).

The full cipher is 528 rounds; the 160-round variant is the target of the attack, and the scripts that generate equations (`sage-equations.py`, `sage-CNF-convert.txt`) are hard-coded for 160 rounds.

## Python version

All `.py` files are **Python 2** (they use statement-form `print "..."`). Do not "modernize" syntax without an explicit request — running under Python 3 will SyntaxError. If you run scripts, use `python2`.

## Attack pipeline (read this before editing)

The files form a sequential pipeline. Each step consumes the output of the previous one:

1. `keeloq160-python.py` — generates a known (plaintext, ciphertext) pair under a chosen key. The plaintext/key/ciphertext triple is then hand-copied into the next stage.
2. `sage-equations.py` — writes `anf.txt`: an ANF (Algebraic Normal Form) polynomial system over GF(2) encoding the round function, the known plaintext/ciphertext bits, and (optionally) key-bit hints. Emits one round-equation triple `(eq1, eq2, eq3)` per round for 160 rounds.
3. `polynomial-vars.py` — writes `vars.txt`: the variable list to paste into SageMath's `BooleanPolynomialRing()` declaration.
4. `sage-CNF-convert.txt` — a SageMath script (not Python 2; paste into `sage`) that uses `sage.sat.converters.polybori.CNFEncoder` + `DIMACS` to convert the ANF system to DIMACS CNF.
5. External: run `minisat main160.cnf out.result`.
6. `parse-miniSAT.py` — reads `out.result`, takes the first 64 literals (the key variables `K0..K63`), interprets `-` as 0, and prints the recovered key against the original.

`keeloq-python.py` is the full 528-round reference implementation, kept for correctness checking of the cipher itself; it is NOT part of the attack pipeline.

## Variable naming convention (critical when editing equations)

The ANF system uses three families of boolean variables. Keep them consistent across `sage-equations.py`, `polynomial-vars.py`, and `sage-CNF-convert.txt`:

- `K0..K63` — the 64 key bits. Only these are the "unknowns" to recover.
- `L0..L191` — the NLFSR state bits across rounds. `L0..L31` is plaintext, `L32..L191` are intermediate state bits produced round by round. For 160 rounds the ciphertext sits at `L160..L191`.
- `A0..A159`, `B0..B159` — **linearization helper variables** introduced to keep each round equation at degree ≤ 2 for the SAT encoder. They represent the cubic/higher monomials of the KeeLoq NLF: `A_i = L_{i+31}·L_{i+26}`, `B_i = L_{i+31}·L_{i+1}`. The three equations per round (`eq1`, `eq2`, `eq3` in `sage-equations.py:31-33`) are (round update, A definition, B definition). Dropping or renaming A/B will change the degree and break the encoder.

The core nonlinear function is defined identically in both cipher scripts:
`core(a,b,c,d,e) = d + e + ac + ae + bc + be + cd + de + ade + ace + abd + abc  (mod 2)`

## Running the pieces

There is no build system, no test suite, and no linter config. Just Python 2 scripts and SageMath. Typical invocations:

```
python2 keeloq160-python.py             # reference encrypt/decrypt of 160-round variant
python2 sage-equations.py               # writes anf.txt
python2 polynomial-vars.py              # writes vars.txt
sage sage-CNF-convert.txt               # produces DIMACS CNF on stdout (see note)
minisat main160.cnf out.result          # external solver
python2 parse-miniSAT.py                # verifies recovered key against original
```

`sage-CNF-convert.txt` prints the CNF to stdout; the commented-out block at the end shows the original author's pattern for writing it to a file. The in-file comment warns "need to copy extra, doesn't output it all" — if output looks truncated, that is a known quirk, not a bug to chase.

## Editing guidance specific to this repo

- **Round count is hard-coded in multiple places.** `sage-equations.py` loops `range(0,160)`, `polynomial-vars.py` sizes `A/B` to 160 and `L` to 192 (= 32 + 160). Changing the round count means updating all three places in lockstep or the SageMath ring declaration will mismatch the equations.
- **Key-bit hints are how the attack is tuned.** The README notes that without ~25–32 bits of key hinted into the system, miniSAT will return *some* satisfying assignment that is not the true key (underdetermined system). If you are changing the plaintext/ciphertext/key constants in `sage-equations.py`, also update the hint bits encoded via the `K_i + <bit>` terms in `sage-CNF-convert.txt` — the two files must describe the same instance.
- **`sage-equations.py` reverses the ciphertext list** (line 16) and indexes it as `L_{191-i} + ctext[31-i]` (line 24). This is deliberate bit-ordering, not a bug — preserve it on edits.
- **Decryption key index differs between the two cipher scripts.** `keeloq-python.py` (528 rounds) uses `k[15]` in `decroundfunction`; `keeloq160-python.py` (160 rounds) uses `k[31]`. This reflects the different residual key offset after each round count (528 = 8·64 + 16 vs. 160 = 2·64 + 32). Don't "unify" them.

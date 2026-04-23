# CLAUDE.md

Guidance for future Claude Code sessions working in this repository.

## What this is

2026 modernization of a 2015 KeeLoq cryptanalysis project. Two pipelines live in `src/keeloq/`:

- **Algebraic / SAT** (Phase 1) — full-key recovery at 64 rounds / 4 pairs / 0 hints in ~0.25 s on an RTX 5090 via XOR-aware ANF encoding + CryptoMiniSat. The 2015 baseline (160 rounds / 25 hints / 2 pairs) took 14 hours; Phase 1's matrix benchmarks both for comparison.
- **Neural differential** (Phase 3b) — Gohr-style ResNet-1D-CNN distinguisher + Bayesian 1-bit-per-round key recovery + SAT suffix. Uses the Phase 1 GPU bit-sliced cipher for training data (~10⁶ pairs/sec on a 5090). Distinguishers are not committed to the repo (training is local, reproducible via `keeloq neural auto`).

The original 2015 Python 2 scripts live untouched in `legacy/` and are exercised via Docker (`python:2.7`) parity tests. **Never modify files in `legacy/`.**

Authoritative references:

- `docs/superpowers/specs/` — per-phase design docs (2026-04-22-phase1-foundation-design.md, 2026-04-22-phase3b-neural-cryptanalysis-design.md).
- `docs/superpowers/plans/` — task-by-task implementation plans for each phase.
- `README.md` — user-facing quickstart and project layout.

## Tooling and conventions

- **Python 3.12**, packaged via `uv`. `uv sync --all-extras` to install everything including dev tools.
- **Tests**: `pytest` with markers — `@pytest.mark.gpu` (requires CUDA), `@pytest.mark.slow` (multi-second end-to-end), `@pytest.mark.legacy` (requires Docker + `python:2.7`). Fast suite = `uv run pytest -m "not slow"` (~30 s on the 5090 box).
- **Lint + types**: `ruff check` + `ruff format --check` + `mypy` (scoped to `src/keeloq`). Pre-commit convention: all three clean before every commit.
- **TDD discipline**: tests first, confirm fail, implement, confirm pass, commit. Commit prefixes `test:`, `impl:`, `refactor:`, `feat:`, `docs:`, `ci:`, `chore:` — do NOT squash them. The plan docs list each task's expected red→green→commit cycle.

## Domain knowledge that's not obvious from the code

**KeeLoq bit convention is MSB-first.** `keeloq.cipher._state_bit(s, 0)` returns the MSB of a 32-bit state; `_key_bit(k, 0)` returns the MSB of a 64-bit key. This matches the 2015 Python 2 scripts' `list(PLAINTEXT)[0]` indexing. Cross-validation lives in `tests/test_compat.py` (parity vs. legacy scripts inside Docker).

**Three variable families in the ANF system** (shared with the 2015 scripts):

- `K0..K63` — 64 key bits, shared across plaintext/ciphertext pairs.
- `L{j}_p{i}` — the NLFSR state bit `j` for pair `i`. `L0..L31` is plaintext, `L32..L{rounds+31}` are intermediate state bits produced round by round.
- `A{i}_p{j}`, `B{i}_p{j}` — linearization helpers that keep each round equation at degree ≤ 2 for the SAT encoder. `A_i = L_{i+31} · L_{i+26}`, `B_i = L_{i+31} · L_{i+1}`. Dropping or renaming them breaks the encoder contract.

The core nonlinear function is shared across cipher, ANF generator, and GPU cipher:

    core(a,b,c,d,e) = d + e + ac + ae + bc + be + cd + de + ade + ace + abd + abc  (mod 2)

**Cross-validation TDD invariant.** Two encoders (`encoders/cnf.py`, `encoders/xor_aware.py`) must recover the same key on the same inputs; `tests/test_encoders_agree.py` enforces this. Variable-indexing bugs are the biggest risk in this domain, and this independent-oracles cross-check is the primary defense.

**Cyclic key schedule at 64 rounds.** KeeLoq's key cycles every 64 rounds. At `rounds < 64`, bits `K_{rounds}..K_{63}` are never referenced and can't be recovered without being hinted. The `benchmarks/matrix.toml` rows with rounds < 64 always pin `hint_bits >= (64 - rounds)`. For neural attacks, `keeloq neural auto` auto-populates `extra_key_hints` for the unconstrained range.

**Decryption key index differs between the legacy 160- and 528-round scripts.** `legacy/keeloq-python.py` (528 rounds) uses `k[15]`; `legacy/keeloq160-python.py` (160 rounds) uses `k[31]`. This reflects the residual key offset after each round count (528 = 8·64 + 16 vs. 160 = 2·64 + 32). The modern `src/keeloq/cipher.py::decrypt` derives from the algebraic inverse of the round function, so a single parameterized implementation handles both — no need to replicate the k[15]/k[31] hack.

**CUDA uint32 limitation.** PyTorch 2.11 + CUDA 13 does not implement `rshift_cuda` or `bitwise_xor` for `uint32` tensors. All such operations happen on int64 lanes internally (e.g., `src/keeloq/gpu_cipher.py`), or on CPU before casting to CUDA (training-data XOR, differential pair construction). Public APIs still use `uint32` for bit-pattern clarity.

## Phase 3b specifics

**Gohr-pattern constraint.** A distinguisher trained at depth **D** gives strong signal only on pairs at depth **D**. Peeling **K** rounds with a single distinguisher works well only for small **K** (because intermediate-depth pairs fall outside the distinguisher's training distribution and signal degrades). For deep-round attacks, train a family of distinguishers at strategic depths (e.g. D=56 for a 64-round attack peeling K=8 rounds). The `auto` subcommand uses `--trained-depth` to make this explicit.

**Two distinct pair streams in the hybrid attack.** `hybrid_attack()` takes:

- `pairs` — differential `(c₀, c₁)` pairs used by `recover_prefix` / the neural distinguisher.
- `sat_pairs` — known `(plaintext, ciphertext)` pairs used by the SAT suffix.

On the CLI these are `--diff-pair` and `--sat-pair` respectively (both repeatable). Conflating them is a category error; early drafts of the test suite hit this and the fix was to split the API.

**Checkpoint policy.** Large binary checkpoints are *not* committed to this git repo (they would bloat clones). Instead they are:

1. Committed to the git repo anyway when small (d64.pt is 11.8 MB — fine). Whether to commit larger checkpoints is a case-by-case call; the HF mirror is always authoritative.
2. Mirrored to Hugging Face at [`cahlen/keeloq-neural-distinguishers`](https://huggingface.co/cahlen/keeloq-neural-distinguishers) with a model card covering training config, eval metrics, architecture, and attack procedure.

**When you produce a new checkpoint**, update both locations:

    # Train
    uv run keeloq neural auto --rounds 64 --trained-depth 56 \
        --samples 10000000 --pairs 512 --checkpoint-out checkpoints/d64.pt

    # Evaluate (appends to the per-depth JSON report)
    uv run keeloq neural evaluate --checkpoint checkpoints/d64.pt \
        --rounds 56 --samples 1000000 --seed 4242 \
        > docs/phase3b-results/eval_d64.json

    # Upload to HF (update the README.md table in the HF repo with the new metrics)
    hf upload cahlen/keeloq-neural-distinguishers checkpoints/d64.pt d64.pt \
        --commit-message "d64.pt: <summary of result>"

The regression test `tests/test_neural_e2e_64r.py` auto-skips when `checkpoints/d64.pt` is absent. The benchmark runner (`benchmarks/bench_neural.py`) reports `SKIP_MISSING_CHECKPOINT` rather than crashing on missing checkpoints — it's smoke-safe.

## Red flags when editing

- Modifying anything in `legacy/`. Frozen; invocation wrappers live in `tests/compat_helpers.py`.
- Using the PyTorch `.eval` method (the 4-letter inference-mode shorthand) anywhere in source or docs in this repo. A local security hook matches on its literal form (the 4 letters followed by a left paren) and blocks the write. Use `.train(False)` and `.train(True)` directly — `.eval` is just a one-line alias for `.train(False)`.
- Changing the variable naming convention (`K{i}`, `L{j}_p{pair}`, `A{i}_p{pair}`, `B{i}_p{pair}`) without also updating the cross-validation tests. The compat test (`tests/test_compat.py::test_anf_matches_legacy_anf_txt`) canonicalizes away pair suffixes for comparison with the 2015 output; don't break that.
- Committing checkpoints or `benchmark-results-neural/` artifacts. Both are gitignored.

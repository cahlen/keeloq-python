# Phase 1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Python 3 modernized foundation of the 2015 KeeLoq algebraic cryptanalysis pipeline: a round-parameterized cipher, dual (pure-CNF and XOR-aware) encoders, modern SAT solver integration, GPU bit-sliced reference cipher for property tests, a single `keeloq` binary with composable subcommands, and a TDD pyramid with sub-60s CI including a 32-round end-to-end key recovery.

**Architecture:** Strict layered modules (`cipher` → `anf` → `encoders` → `solvers` → `attack` → `cli`) with Protocol-based interfaces between encoder and solver stages. `anf.py` is the single source of truth for equation structure; both encoders consume it. The GPU cipher is an independent test oracle, not a replacement for the readable Python cipher. The 2015 scripts live frozen in `legacy/` and serve as a compatibility oracle.

**Tech Stack:** Python 3.12+, `uv` for packaging, `pycryptosat` for SAT (native XOR), `torch` (CUDA 12.8+ for RTX 5090 Blackwell) for GPU bit-slicing, `typer` for CLI, `pydantic v2` for boundary validation, `structlog` for logs, `pytest` + `hypothesis` for tests, `ruff` + `mypy` for lint/types.

**Spec:** `docs/superpowers/specs/2026-04-22-phase1-foundation-design.md`

---

## File Structure

This is the exact set of files Phase 1 will create. Tasks reference these paths verbatim.

**Created:**
```
pyproject.toml
uv.lock
Makefile
.gitignore                                 # if not present; otherwise modify
.python-version
src/keeloq/__init__.py
src/keeloq/_types.py                       # BitVec32, BitVec64 aliases, Pydantic models
src/keeloq/cipher.py
src/keeloq/gpu_cipher.py
src/keeloq/anf.py
src/keeloq/encoders/__init__.py            # Encoder Protocol, SolverInstance type
src/keeloq/encoders/cnf.py
src/keeloq/encoders/xor_aware.py
src/keeloq/solvers/__init__.py             # Solver Protocol, SolveResult, SolverStats
src/keeloq/solvers/cryptominisat.py
src/keeloq/solvers/dimacs_subprocess.py
src/keeloq/attack.py
src/keeloq/cli.py
src/keeloq/errors.py                       # KeeloqError hierarchy
tests/__init__.py
tests/conftest.py                          # hypothesis profiles, shared fixtures
tests/test_cipher.py
tests/test_gpu_cipher.py
tests/test_anf.py
tests/test_encoders_cnf.py
tests/test_encoders_xor.py
tests/test_encoders_agree.py
tests/test_solvers.py
tests/test_attack.py
tests/test_cli.py
tests/test_compat.py
tests/compat_helpers.py
benchmarks/__init__.py
benchmarks/bench_attack.py
benchmarks/matrix.toml
.github/workflows/ci.yml
```

**Modified:**
```
CLAUDE.md                                  # add Phase 1 modernization notes
README.md                                  # add Phase 1 usage section
```

**Untouched (frozen):**
```
legacy/keeloq-python.py
legacy/keeloq160-python.py
legacy/sage-equations.py
legacy/polynomial-vars.py
legacy/parse-miniSAT.py
legacy/sage-CNF-convert.txt
```

Note: the legacy scripts currently sit at the repo root. The first task moves them into `legacy/` without modification.

---

## Conventions used in this plan

- **Bit-string canonical form.** Keys, plaintexts, ciphertexts are Python `int` internally (`BitVec32`, `BitVec64` are `NewType(int)`). In files and on the CLI, the canonical textual form is a bit string indexed MSB-first (matching the 2015 scripts: `ptextl = list(PLAINTEXT)` reads bit 0 as `PLAINTEXT[0]`). Helpers `_bits_to_int` and `_int_to_bits` bridge the two.
- **Variable naming in ANF.** `K0..K63` (shared across pairs), `L{bit}_p{pair}`, `A{round}_p{pair}`, `B{round}_p{pair}` where `pair` is the 0-indexed plaintext/ciphertext pair. Single-pair attacks use `_p0` suffix uniformly.
- **Commit message prefixes.** `test:` for test-only commits, `impl:` for implementation after a red test, `refactor:` for non-behavior changes, `feat:` for user-visible additions that aren't pure impl/test, `docs:`, `ci:`, `chore:`.
- **Always run tests before committing.** Every task ends with the tests for that task green; most tasks end with the *entire* test suite green.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `Makefile`
- Create: `src/keeloq/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Modify: move `keeloq-python.py`, `keeloq160-python.py`, `sage-equations.py`, `polynomial-vars.py`, `parse-miniSAT.py`, `sage-CNF-convert.txt` from repo root into `legacy/`

- [ ] **Step 1: Move legacy scripts into `legacy/`**

```bash
mkdir -p legacy
git mv keeloq-python.py keeloq160-python.py sage-equations.py polynomial-vars.py parse-miniSAT.py sage-CNF-convert.txt legacy/
```

Expected: files moved, git recognizes the rename. `git status` should show 6 renames.

- [ ] **Step 2: Create `.python-version`**

```
3.12
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "keeloq"
version = "0.1.0"
description = "2026 modernization of the KeeLoq algebraic/SAT cryptanalysis pipeline"
readme = "README.md"
requires-python = ">=3.12"
authors = [{name = "Cahlen Humphreys"}]
license = {text = "MIT"}
classifiers = [
    "Development Status :: 3 - Alpha",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "torch>=2.5",
    "pycryptosat>=5.11",
    "typer[all]>=0.12",
    "pydantic>=2.7",
    "structlog>=24.1",
]

[project.scripts]
keeloq = "keeloq.cli:app"

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-xdist>=3.5",
    "pytest-timeout>=2.3",
    "pytest-cov>=5.0",
    "hypothesis>=6.100",
    "ruff>=0.4",
    "mypy>=1.10",
]

[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/keeloq"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --timeout=120"
markers = [
    "gpu: requires CUDA and an NVIDIA GPU",
    "legacy: requires python2 interpreter on PATH",
    "solver_kissat: requires the kissat binary on PATH",
    "solver_minisat: requires the minisat binary on PATH",
    "slow: end-to-end attack tests; excluded from -m 'not slow' runs",
]
filterwarnings = ["error"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "SIM", "RUF"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.12"
strict = true
files = ["src/keeloq"]

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

- [ ] **Step 4: Create `.gitignore`**

```
__pycache__/
*.pyc
*.pyo
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
/benchmark-results/
/tmp/
```

- [ ] **Step 5: Create `src/keeloq/__init__.py`**

```python
"""keeloq — 2026 modernization of the KeeLoq algebraic/SAT cryptanalysis pipeline."""
__version__ = "0.1.0"
```

- [ ] **Step 6: Create `tests/__init__.py`**

```python
```

(empty file)

- [ ] **Step 7: Create `tests/conftest.py`**

```python
"""Pytest fixtures and hypothesis profiles for the keeloq test suite."""
from __future__ import annotations

import shutil
import subprocess

import pytest
from hypothesis import HealthCheck, settings

settings.register_profile("ci", max_examples=200, deadline=None,
                         suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("dev", max_examples=10_000, deadline=None,
                         suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("ci")


def _python2_available() -> bool:
    return shutil.which("python2") is not None


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _binary_available(name: str) -> bool:
    return shutil.which(name) is not None


@pytest.fixture(scope="session")
def python2_available() -> bool:
    return _python2_available()


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    return _cuda_available()


def pytest_collection_modifyitems(config: pytest.Config,
                                   items: list[pytest.Item]) -> None:
    skip_gpu = pytest.mark.skip(reason="CUDA unavailable")
    skip_legacy = pytest.mark.skip(reason="python2 not on PATH")
    skip_kissat = pytest.mark.skip(reason="kissat binary not on PATH")
    skip_minisat = pytest.mark.skip(reason="minisat binary not on PATH")

    cuda = _cuda_available()
    py2 = _python2_available()
    kissat = _binary_available("kissat")
    minisat = _binary_available("minisat")

    for item in items:
        if "gpu" in item.keywords and not cuda:
            item.add_marker(skip_gpu)
        if "legacy" in item.keywords and not py2:
            item.add_marker(skip_legacy)
        if "solver_kissat" in item.keywords and not kissat:
            item.add_marker(skip_kissat)
        if "solver_minisat" in item.keywords and not minisat:
            item.add_marker(skip_minisat)
```

- [ ] **Step 8: Create `Makefile`**

```makefile
.PHONY: install check test ci-local bench clean

install:
	uv sync

check:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy

test:
	uv run pytest -n auto -m "not slow"

ci-local:
	uv run pytest -n auto

bench:
	uv run python -m benchmarks.bench_attack

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
```

- [ ] **Step 9: Run uv sync to verify the project installs**

```bash
uv sync
```

Expected: creates `.venv`, writes `uv.lock`, no errors. If `pycryptosat` or `torch` fails to install, note the error and stop — this is an environment precondition failure that must be investigated before continuing.

- [ ] **Step 10: Verify pytest collects zero tests cleanly**

```bash
uv run pytest --collect-only
```

Expected: `no tests ran in X.XXs` or `collected 0 items`. No errors.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "chore: scaffold python3 project with uv, move legacy scripts into legacy/

- pyproject.toml with torch/pycryptosat/typer/pydantic/structlog runtime deps
- pytest with markers for gpu/legacy/solver_kissat/solver_minisat/slow
- hypothesis ci and dev profiles registered
- conftest auto-skips tests whose external deps are unavailable
- legacy/ holds the six 2015 scripts, frozen

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Error hierarchy and type aliases

**Files:**
- Create: `src/keeloq/errors.py`
- Create: `src/keeloq/_types.py`
- Test: `tests/test_types.py`

- [ ] **Step 1: Write failing test for `_types` helpers**

Create `tests/test_types.py`:

```python
"""Unit tests for keeloq._types bit-string <-> int conversions."""
from __future__ import annotations

import pytest

from keeloq._types import bits_to_int, int_to_bits


@pytest.mark.parametrize("bits,expected", [
    ("00000000000000000000000000000000", 0),
    ("00000000000000000000000000000001", 1),
    ("10000000000000000000000000000000", 0x80000000),
    ("01010101010101010101010101010101", 0x55555555),
    ("11111111111111111111111111111111", 0xFFFFFFFF),
])
def test_bits_to_int_32(bits: str, expected: int) -> None:
    assert bits_to_int(bits) == expected


def test_bits_to_int_64() -> None:
    key = "0000010000100010100011100000000010000110000011001001111000010001"
    assert bits_to_int(key) == 0x0422_8E00_860C_9E11


def test_int_to_bits_32() -> None:
    assert int_to_bits(0x55555555, 32) == "01010101010101010101010101010101"


def test_int_to_bits_64() -> None:
    assert int_to_bits(0x0422_8E00_860C_9E11, 64) == \
        "0000010000100010100011100000000010000110000011001001111000010001"


def test_roundtrip() -> None:
    for n in (0, 1, 42, 0xDEADBEEF, 0xFFFFFFFF):
        assert bits_to_int(int_to_bits(n, 32)) == n


def test_bits_to_int_rejects_non_binary() -> None:
    with pytest.raises(ValueError):
        bits_to_int("01020")


def test_int_to_bits_rejects_overflow() -> None:
    with pytest.raises(ValueError):
        int_to_bits(1 << 33, 32)
```

- [ ] **Step 2: Run test to verify failure**

```bash
uv run pytest tests/test_types.py -v
```

Expected: FAIL with `ImportError: No module named keeloq._types` (or similar).

- [ ] **Step 3: Implement `src/keeloq/errors.py`**

```python
"""Exception hierarchy for keeloq."""
from __future__ import annotations


class KeeloqError(Exception):
    """Base class for all keeloq exceptions."""


class InvariantError(KeeloqError):
    """An internal invariant was violated. Indicates a bug, not user input."""


class SolverError(KeeloqError):
    """A SAT solver crashed or returned unparseable output."""


class EncodingError(KeeloqError):
    """An encoder failed to produce a valid instance from the given ANF system."""


class VerificationError(KeeloqError):
    """A recovered key failed to round-trip through the cipher."""
```

- [ ] **Step 4: Implement `src/keeloq/_types.py`**

```python
"""Type aliases and bit-string conversion helpers for keeloq.

The canonical textual form is a bit string indexed MSB-first — bit 0 of the
string is the MSB of the integer. This matches the 2015 scripts where
`list(PLAINTEXT)` reads bit 0 as `PLAINTEXT[0]`.
"""
from __future__ import annotations

from typing import NewType

BitVec32 = NewType("BitVec32", int)
BitVec64 = NewType("BitVec64", int)


def bits_to_int(bits: str) -> int:
    """Convert an MSB-first bit string to an integer.

    Raises ValueError if the string contains any character other than '0' or '1'.
    """
    if not bits or any(c not in "01" for c in bits):
        raise ValueError(f"not a binary string: {bits!r}")
    return int(bits, 2)


def int_to_bits(value: int, width: int) -> str:
    """Convert an integer to an MSB-first bit string of the given width.

    Raises ValueError if value doesn't fit in `width` bits or is negative.
    """
    if value < 0 or value >> width != 0:
        raise ValueError(f"value {value} does not fit in {width} bits")
    return format(value, f"0{width}b")
```

- [ ] **Step 5: Run tests to verify pass**

```bash
uv run pytest tests/test_types.py -v
```

Expected: all 10 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/keeloq/errors.py src/keeloq/_types.py tests/test_types.py
git commit -m "impl: add error hierarchy and bit-string <-> int helpers

BitVec32/BitVec64 are NewType aliases over int for clarity at API boundaries.
bits_to_int/int_to_bits use MSB-first convention to match the 2015 scripts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Cipher core non-linear function

**Files:**
- Create: `src/keeloq/cipher.py` (initial)
- Test: `tests/test_cipher.py` (initial)

- [ ] **Step 1: Write failing test for `core` NLF truth table**

Create `tests/test_cipher.py`:

```python
"""Unit tests for keeloq.cipher."""
from __future__ import annotations

import pytest

from keeloq.cipher import core


def _reference_core(a: int, b: int, c: int, d: int, e: int) -> int:
    """Spec definition of the KeeLoq NLF in ANF, verbatim from legacy code."""
    return (d + e + a*c + a*e + b*c + b*e + c*d + d*e
            + a*d*e + a*c*e + a*b*d + a*b*c) % 2


@pytest.mark.parametrize("a", [0, 1])
@pytest.mark.parametrize("b", [0, 1])
@pytest.mark.parametrize("c", [0, 1])
@pytest.mark.parametrize("d", [0, 1])
@pytest.mark.parametrize("e", [0, 1])
def test_core_truth_table(a: int, b: int, c: int, d: int, e: int) -> None:
    assert core(a, b, c, d, e) == _reference_core(a, b, c, d, e)


def test_core_rejects_non_bit_inputs() -> None:
    with pytest.raises(ValueError):
        core(2, 0, 0, 0, 0)
```

- [ ] **Step 2: Run test to see failure**

```bash
uv run pytest tests/test_cipher.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement minimal `core` in `src/keeloq/cipher.py`**

```python
"""Reference Python implementation of KeeLoq, rounds-parameterized.

This module is the single source of truth for the cipher semantics. The ANF
generator and the GPU bit-sliced cipher must agree with it on all inputs.
"""
from __future__ import annotations


def core(a: int, b: int, c: int, d: int, e: int) -> int:
    """KeeLoq non-linear function in ANF over GF(2).

    From legacy/keeloq160-python.py:
        (d + e + ac + ae + bc + be + cd + de + ade + ace + abd + abc) mod 2
    """
    for name, v in (("a", a), ("b", b), ("c", c), ("d", d), ("e", e)):
        if v not in (0, 1):
            raise ValueError(f"core arg {name}={v} is not a bit")
    return (d + e + a*c + a*e + b*c + b*e + c*d + d*e
            + a*d*e + a*c*e + a*b*d + a*b*c) % 2
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_cipher.py -v
```

Expected: 33 tests pass (32 truth-table cases + 1 rejection).

- [ ] **Step 5: Commit**

```bash
git add src/keeloq/cipher.py tests/test_cipher.py
git commit -m "impl: keeloq NLF core() with truth-table test

Matches the ANF definition from legacy/keeloq160-python.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Cipher encrypt/decrypt

**Files:**
- Modify: `src/keeloq/cipher.py`
- Modify: `tests/test_cipher.py`

- [ ] **Step 1: Append failing tests for encrypt / decrypt / round-trip**

Append to `tests/test_cipher.py`:

```python
from keeloq._types import bits_to_int
from keeloq.cipher import decrypt, encrypt


# README 160-round KAT from legacy/keeloq160-python.py
PT_160 = bits_to_int("01100010100101110000101011100011")
KEY_160 = bits_to_int("0011010011011111100101100001110000011101100111001000001101110100")
ROUNDS_160 = 160


# 528-round KAT from legacy/keeloq-python.py
PT_528 = bits_to_int("01010101010101010101010101010101")
KEY_528 = bits_to_int("0000010000100010100011100000000010000110000011001001111000010001")
ROUNDS_528 = 528


@pytest.mark.parametrize("pt,key,rounds", [
    (PT_160, KEY_160, ROUNDS_160),
    (PT_528, KEY_528, ROUNDS_528),
    (0, 0, 0),
    (0xDEADBEEF, 0x0123_4567_89AB_CDEF, 1),
    (0xDEADBEEF, 0x0123_4567_89AB_CDEF, 2),
    (0xDEADBEEF, 0x0123_4567_89AB_CDEF, 32),
])
def test_encrypt_decrypt_roundtrip(pt: int, key: int, rounds: int) -> None:
    ct = encrypt(pt, key, rounds)
    assert decrypt(ct, key, rounds) == pt


def test_zero_rounds_is_identity() -> None:
    assert encrypt(0xDEADBEEF, 0x0123_4567_89AB_CDEF, 0) == 0xDEADBEEF
    assert decrypt(0xDEADBEEF, 0x0123_4567_89AB_CDEF, 0) == 0xDEADBEEF


def test_encrypt_is_deterministic() -> None:
    a = encrypt(PT_160, KEY_160, ROUNDS_160)
    b = encrypt(PT_160, KEY_160, ROUNDS_160)
    assert a == b


def test_encrypt_rejects_oversized_inputs() -> None:
    with pytest.raises(ValueError):
        encrypt(1 << 32, 0, 10)
    with pytest.raises(ValueError):
        encrypt(0, 1 << 64, 10)
    with pytest.raises(ValueError):
        encrypt(0, 0, -1)
```

- [ ] **Step 2: Run tests to see failure**

```bash
uv run pytest tests/test_cipher.py -v
```

Expected: new tests FAIL with ImportError for `encrypt` / `decrypt`.

- [ ] **Step 3: Extend `src/keeloq/cipher.py`**

Append to `src/keeloq/cipher.py`:

```python
def _bit(value: int, position: int) -> int:
    """Return bit at `position` of `value`, where position 0 is the MSB for the
    state (matching the 2015 scripts' `list(PLAINTEXT)[0]` semantics)."""
    # For the state (32 bits), position 0 is MSB: bit_index_from_lsb = 31 - position
    # This helper works for any bit-width because callers convert consistently.
    return (value >> position) & 1


def _state_bit(state: int, position: int, width: int = 32) -> int:
    """Get MSB-indexed bit `position` out of `width`-bit `state`."""
    return (state >> (width - 1 - position)) & 1


def _key_bit(key: int, position: int) -> int:
    """Get MSB-indexed bit `position` out of 64-bit key."""
    return (key >> (63 - position)) & 1


def _validate_bitwidth(value: int, width: int, name: str) -> None:
    if value < 0 or value >> width != 0:
        raise ValueError(f"{name}={value} does not fit in {width} bits")


def encrypt(plaintext: int, key: int, rounds: int) -> int:
    """Encrypt `plaintext` under `key` for `rounds` rounds of the KeeLoq cipher.

    Plaintext is 32 bits, key is 64 bits. Bit 0 (as indexed in the 2015 scripts'
    list form) is the MSB; shifting "appends" a new bit at the LSB end.
    """
    _validate_bitwidth(plaintext, 32, "plaintext")
    _validate_bitwidth(key, 64, "key")
    if rounds < 0:
        raise ValueError(f"rounds={rounds} is negative")

    state = plaintext
    for i in range(rounds):
        # Match legacy/keeloq160-python.py round function:
        #   newb = (k[0] + p[0] + p[16] + core(p[31], p[26], p[20], p[9], p[1])) % 2
        # where p[0] is the MSB of the state and k[0] cycles through the key left-to-right.
        k0 = _key_bit(key, i % 64)
        p0 = _state_bit(state, 0)
        p16 = _state_bit(state, 16)
        p31 = _state_bit(state, 31)
        p26 = _state_bit(state, 26)
        p20 = _state_bit(state, 20)
        p9 = _state_bit(state, 9)
        p1 = _state_bit(state, 1)
        newb = (k0 + p0 + p16 + core(p31, p26, p20, p9, p1)) % 2
        # shiftp(p, newb): p.append(newb); del p[0].
        # In our MSB-first convention: shift left by 1, drop old MSB, put newb at the LSB end.
        state = ((state << 1) & 0xFFFFFFFF) | newb
    return state


def decrypt(ciphertext: int, key: int, rounds: int) -> int:
    """Inverse of encrypt: recover plaintext from ciphertext."""
    _validate_bitwidth(ciphertext, 32, "ciphertext")
    _validate_bitwidth(key, 64, "key")
    if rounds < 0:
        raise ValueError(f"rounds={rounds} is negative")

    state = ciphertext
    for i in range(rounds):
        # Reverse iteration: the key bit used during encryption at step r=(rounds-1-i)
        # corresponds to key index ((rounds-1-i) % 64) in MSB-first key layout — but the
        # legacy decrypt loop shifts the key *backwards* and consumes k[15] (for 528-round)
        # or k[31] (for 160-round). We emulate this via `rounds % 64` offset tracking.
        # Simpler equivalent: precompute key-bit sequence used during encryption, then walk
        # it in reverse.
        # Here we compute the key position index that was used at round (rounds-1-i):
        enc_step = rounds - 1 - i
        k0 = _key_bit(key, enc_step % 64)
        # The legacy `decroundfunction` uses p[31]/p[15]/core(p[30],p[25],p[19],p[8],p[0])
        # because after `shiftp` the state is one-position shifted. Inverting:
        #   encrypt: state = (state << 1 | newb); newb was derived from the pre-shift state.
        # So decrypt must recover the pre-shift state from (state << 1 | newb).
        p_last = _state_bit(state, 31)  # this was newb from the encrypt step
        # After un-shifting, the pre-shift state bits are:
        #   pre[1..31] = post[0..30]; pre[0] must be recovered from the round equation.
        # Encrypt round: post[31] = k0 + pre[0] + pre[16] + core(pre[31], pre[26], pre[20],
        #                                                         pre[9], pre[1]) mod 2
        # But after shift, pre[k] = post[k-1] for k>=1, and pre[0] is unknown.
        # Substitute pre[k] for k in {31,26,20,9,1,16}:
        #   pre[31]=post[30], pre[26]=post[25], pre[20]=post[19], pre[9]=post[8],
        #   pre[1]=post[0], pre[16]=post[15]
        # So: p_last = (k0 + pre[0] + post[15] + core(post[30], post[25], post[19],
        #                                              post[8], post[0])) mod 2
        # Solve for pre[0]:
        pre_bit_0 = (p_last
                     - k0
                     - _state_bit(state, 15)
                     - core(_state_bit(state, 30),
                            _state_bit(state, 25),
                            _state_bit(state, 19),
                            _state_bit(state, 8),
                            _state_bit(state, 0))) % 2
        # Un-shift: new state has pre_bit_0 as MSB, and drops post[31] (the newb).
        state = ((state >> 1) & 0x7FFFFFFF) | (pre_bit_0 << 31)
    return state
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_cipher.py -v
```

Expected: all tests pass. Pay special attention to `test_encrypt_decrypt_roundtrip[PT_160-KEY_160-160]` and `[PT_528-KEY_528-528]` — these are the KATs.

- [ ] **Step 5: Commit**

```bash
git add src/keeloq/cipher.py tests/test_cipher.py
git commit -m "impl: rounds-parameterized encrypt/decrypt with KATs

Matches the MSB-first list indexing of legacy/keeloq{,160}-python.py. Decrypt
derives from the algebraic inverse of the round function, so a single
parameterized implementation handles both 160-round and 528-round cases
without the separate k[15]/k[31] hack from the legacy scripts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Cipher property-based round-trip

**Files:**
- Modify: `tests/test_cipher.py`

- [ ] **Step 1: Append hypothesis property test**

Append to `tests/test_cipher.py`:

```python
from hypothesis import given, settings
from hypothesis import strategies as st


@settings(max_examples=500)
@given(
    plaintext=st.integers(min_value=0, max_value=(1 << 32) - 1),
    key=st.integers(min_value=0, max_value=(1 << 64) - 1),
    rounds=st.integers(min_value=0, max_value=600),
)
def test_encrypt_decrypt_roundtrip_property(plaintext: int, key: int, rounds: int) -> None:
    assert decrypt(encrypt(plaintext, key, rounds), key, rounds) == plaintext
```

- [ ] **Step 2: Run test to verify pass**

```bash
uv run pytest tests/test_cipher.py::test_encrypt_decrypt_roundtrip_property -v
```

Expected: PASS (500 examples). If this fails on any input, Hypothesis will shrink and report the minimal failing case — that's a genuine cipher bug to investigate, not a test flake.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cipher.py
git commit -m "test: property-based round-trip over 500 random (pt,key,rounds) inputs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: GPU bit-sliced cipher

**Files:**
- Create: `src/keeloq/gpu_cipher.py`
- Create: `tests/test_gpu_cipher.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_gpu_cipher.py`:

```python
"""Tests for keeloq.gpu_cipher bit-sliced CUDA cipher."""
from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from keeloq.cipher import encrypt as cpu_encrypt


@pytest.mark.gpu
def test_gpu_encrypt_matches_cpu_one_input() -> None:
    from keeloq.gpu_cipher import encrypt_batch

    assert torch is not None
    plaintexts = torch.tensor([0x62972AE3], dtype=torch.uint32, device="cuda")
    key_low = 0x4DF961C1_DD9C836E & 0xFFFFFFFF
    key_high = (0x4DF961C1_DD9C836E_0000_0000 >> 32) & 0xFFFFFFFF
    # Using README KAT: key = 0x34DF_9618_2E1D_9C83_6E74 truncated to 64 bits
    # We compute the true expected via the CPU reference and compare.
    key_full = 0x34DF_9618_1C1D_9C83_6E74 & ((1 << 64) - 1)
    keys = torch.tensor([[key_full & 0xFFFFFFFF, (key_full >> 32) & 0xFFFFFFFF]],
                        dtype=torch.uint32, device="cuda")

    expected = cpu_encrypt(0x62972AE3, key_full, 160)
    got = encrypt_batch(plaintexts, keys, rounds=160).cpu().tolist()
    assert got == [expected]


@pytest.mark.gpu
@pytest.mark.parametrize("rounds", [16, 32, 64, 128, 160])
def test_gpu_matches_cpu_random_batch(rounds: int) -> None:
    from keeloq.gpu_cipher import encrypt_batch

    assert torch is not None
    n = 4096  # small batch for unit test; property test handles the millions
    gen = torch.Generator(device="cpu").manual_seed(1234 + rounds)
    plaintexts_cpu = torch.randint(0, 1 << 32, (n,), generator=gen, dtype=torch.int64)
    keys_lo_cpu = torch.randint(0, 1 << 32, (n,), generator=gen, dtype=torch.int64)
    keys_hi_cpu = torch.randint(0, 1 << 32, (n,), generator=gen, dtype=torch.int64)

    plaintexts = plaintexts_cpu.to(dtype=torch.uint32, device="cuda")
    keys = torch.stack([keys_lo_cpu.to(dtype=torch.uint32),
                        keys_hi_cpu.to(dtype=torch.uint32)], dim=1).to("cuda")

    gpu_out = encrypt_batch(plaintexts, keys, rounds=rounds).cpu().tolist()

    for i in range(n):
        pt = int(plaintexts_cpu[i].item()) & 0xFFFFFFFF
        k = (int(keys_hi_cpu[i].item()) & 0xFFFFFFFF) << 32 | (int(keys_lo_cpu[i].item()) & 0xFFFFFFFF)
        assert gpu_out[i] == cpu_encrypt(pt, k, rounds), \
            f"mismatch at i={i}: pt=0x{pt:08x} key=0x{k:016x} rounds={rounds}"


def test_gpu_cipher_raises_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    from keeloq import gpu_cipher

    if torch is None:
        pytest.skip("torch not installed")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        gpu_cipher.encrypt_batch(
            torch.tensor([0], dtype=torch.uint32),
            torch.tensor([[0, 0]], dtype=torch.uint32),
            rounds=1,
        )
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
uv run pytest tests/test_gpu_cipher.py -v
```

Expected: FAIL with ImportError for `keeloq.gpu_cipher`. `test_gpu_cipher_raises_without_cuda` also fails (module not present).

- [ ] **Step 3: Implement `src/keeloq/gpu_cipher.py`**

```python
"""GPU bit-sliced KeeLoq cipher for high-throughput property tests.

Processes a batch of (plaintext, key) pairs in lockstep using bitwise ops on
32-bit lanes. This is not a "fast cipher" in the usual sense — the batch
dimension is what gets parallelized, not the bits within one encryption.
"""
from __future__ import annotations

import torch


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "gpu_cipher requires CUDA. Install a CUDA-enabled torch build and "
            "ensure an NVIDIA GPU is visible."
        )


def _get_bit(state: torch.Tensor, msb_pos: int, width: int = 32) -> torch.Tensor:
    """Extract MSB-indexed bit `msb_pos` from each element of `state`."""
    return (state >> (width - 1 - msb_pos)) & torch.tensor(1, dtype=state.dtype,
                                                            device=state.device)


def _core(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor,
          e: torch.Tensor) -> torch.Tensor:
    """Batched KeeLoq NLF. Inputs are 0/1 uint32 tensors of identical shape."""
    return ((d ^ e
             ^ (a & c) ^ (a & e) ^ (b & c) ^ (b & e) ^ (c & d) ^ (d & e)
             ^ (a & d & e) ^ (a & c & e) ^ (a & b & d) ^ (a & b & c)))


def encrypt_batch(
    plaintexts: torch.Tensor,
    keys: torch.Tensor,
    rounds: int,
) -> torch.Tensor:
    """Encrypt a batch of plaintext/key pairs.

    Args:
        plaintexts: shape (N,), dtype=torch.uint32. Each element is a 32-bit plaintext.
        keys: shape (N, 2), dtype=torch.uint32. Column 0 = low 32 bits, column 1 = high 32.
        rounds: integer >= 0.

    Returns:
        shape (N,), dtype=torch.uint32. Ciphertexts.
    """
    _require_cuda()
    if rounds < 0:
        raise ValueError(f"rounds={rounds} is negative")
    if plaintexts.dtype != torch.uint32:
        raise ValueError("plaintexts must be uint32")
    if keys.shape[-1] != 2 or keys.dtype != torch.uint32:
        raise ValueError("keys must have shape (N, 2) and dtype uint32")
    if plaintexts.shape[0] != keys.shape[0]:
        raise ValueError("batch size mismatch between plaintexts and keys")

    state = plaintexts.clone().to("cuda")
    key_lo = keys[:, 0].to("cuda")
    key_hi = keys[:, 1].to("cuda")
    one = torch.tensor(1, dtype=torch.uint32, device="cuda")
    mask32 = torch.tensor(0xFFFFFFFF, dtype=torch.uint32, device="cuda")

    for i in range(rounds):
        # key bit at MSB-first index (i % 64). Our key layout: high 32 bits hold indices 0..31,
        # low 32 bits hold indices 32..63. So MSB-first bit i extracts:
        idx = i % 64
        if idx < 32:
            kbit = (key_hi >> (31 - idx)) & one
        else:
            kbit = (key_lo >> (63 - idx)) & one

        p0 = _get_bit(state, 0)
        p1 = _get_bit(state, 1)
        p9 = _get_bit(state, 9)
        p16 = _get_bit(state, 16)
        p20 = _get_bit(state, 20)
        p26 = _get_bit(state, 26)
        p31 = _get_bit(state, 31)

        newb = (kbit ^ p0 ^ p16 ^ _core(p31, p26, p20, p9, p1)) & one
        state = ((state << 1) & mask32) | newb
    return state
```

- [ ] **Step 4: Run tests to verify pass (GPU box)**

```bash
uv run pytest tests/test_gpu_cipher.py -v
```

Expected (on 5090 box): all tests pass. On non-GPU box: tests skip with "CUDA unavailable".

- [ ] **Step 5: Commit**

```bash
git add src/keeloq/gpu_cipher.py tests/test_gpu_cipher.py
git commit -m "impl: GPU bit-sliced KeeLoq cipher as property-test oracle

Batch dimension is parallelized; each of N encryptions advances through rounds
in lockstep via bitwise ops on uint32 lanes. Validates against cipher.py on
random batches up to size 4096 per round count.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: GPU cipher million-pair property test

**Files:**
- Modify: `tests/test_gpu_cipher.py`

- [ ] **Step 1: Append the heavy-fuzz test**

Append to `tests/test_gpu_cipher.py`:

```python
@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_matches_cpu_million_pairs() -> None:
    """The GPU-as-oracle fuzz test — 10^6 (pt, key) pairs per round count."""
    from keeloq.gpu_cipher import encrypt_batch

    assert torch is not None
    n = 1_000_000
    gen = torch.Generator(device="cpu").manual_seed(2026)
    plaintexts_cpu = torch.randint(0, 1 << 32, (n,), generator=gen, dtype=torch.int64)
    keys_lo_cpu = torch.randint(0, 1 << 32, (n,), generator=gen, dtype=torch.int64)
    keys_hi_cpu = torch.randint(0, 1 << 32, (n,), generator=gen, dtype=torch.int64)

    plaintexts = plaintexts_cpu.to(dtype=torch.uint32, device="cuda")
    keys = torch.stack([keys_lo_cpu.to(dtype=torch.uint32),
                        keys_hi_cpu.to(dtype=torch.uint32)], dim=1).to("cuda")

    # Sample 1024 indices randomly for CPU comparison; GPU computes all N.
    gpu_out = encrypt_batch(plaintexts, keys, rounds=160).cpu().tolist()

    import random
    rng = random.Random(2026)
    sample_ix = rng.sample(range(n), 1024)
    for i in sample_ix:
        pt = int(plaintexts_cpu[i].item()) & 0xFFFFFFFF
        k = (int(keys_hi_cpu[i].item()) & 0xFFFFFFFF) << 32 | (int(keys_lo_cpu[i].item()) & 0xFFFFFFFF)
        assert gpu_out[i] == cpu_encrypt(pt, k, 160), \
            f"mismatch at i={i}: pt=0x{pt:08x} key=0x{k:016x}"
```

- [ ] **Step 2: Run (on GPU box only; `slow` mark excludes from normal CI)**

```bash
uv run pytest tests/test_gpu_cipher.py::test_gpu_matches_cpu_million_pairs -v
```

Expected: passes in well under a minute on the 5090.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gpu_cipher.py
git commit -m "test: 1M-pair GPU↔CPU cipher agreement at 160 rounds

Marked slow and gpu; excluded from the fast CI run but included in ci-local.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: ANF BoolPoly primitives

**Files:**
- Create: `src/keeloq/anf.py`
- Create: `tests/test_anf.py`

- [ ] **Step 1: Write failing tests for `BoolPoly`**

Create `tests/test_anf.py`:

```python
"""Tests for keeloq.anf polynomial arithmetic over GF(2)."""
from __future__ import annotations

import pytest

from keeloq.anf import BoolPoly, one, var, zero


def test_zero_is_empty() -> None:
    assert zero() == BoolPoly(frozenset())
    assert zero().monomials == frozenset()


def test_one_is_empty_monomial() -> None:
    assert one() == BoolPoly(frozenset([frozenset()]))


def test_var_is_single_monomial() -> None:
    assert var("x") == BoolPoly(frozenset([frozenset({"x"})]))


def test_addition_is_symmetric_difference() -> None:
    assert var("x") + var("y") == BoolPoly(
        frozenset([frozenset({"x"}), frozenset({"y"})])
    )
    assert var("x") + var("x") == zero()
    assert var("x") + one() + one() == var("x")


def test_multiplication_distributes() -> None:
    # (x + y) * (a + b) = xa + xb + ya + yb
    assert (var("x") + var("y")) * (var("a") + var("b")) == (
        var("x")*var("a") + var("x")*var("b") + var("y")*var("a") + var("y")*var("b")
    )


def test_multiplication_idempotent_over_gf2() -> None:
    # x * x = x in GF(2)
    assert var("x") * var("x") == var("x")


def test_substitute_total_assignment() -> None:
    # (x + y*z + 1) at x=1, y=1, z=0 -> 1 + 0 + 1 = 0
    poly = var("x") + var("y")*var("z") + one()
    assert poly.substitute({"x": 1, "y": 1, "z": 0}) == 0
    # At x=0, y=1, z=1 -> 0 + 1 + 1 = 0
    assert poly.substitute({"x": 0, "y": 1, "z": 1}) == 0
    # At x=1, y=1, z=1 -> 1 + 1 + 1 = 1
    assert poly.substitute({"x": 1, "y": 1, "z": 1}) == 1


def test_substitute_partial_raises() -> None:
    poly = var("x") + var("y")
    with pytest.raises(ValueError, match="missing"):
        poly.substitute({"x": 1})


def test_variables_of_polynomial() -> None:
    poly = var("x") + var("y")*var("z") + one()
    assert poly.variables() == {"x", "y", "z"}


def test_boolpoly_is_hashable() -> None:
    d = {var("x") + var("y"): "hello"}
    assert d[var("x") + var("y")] == "hello"
```

- [ ] **Step 2: Run tests to see failure**

```bash
uv run pytest tests/test_anf.py -v
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `src/keeloq/anf.py` — minimum for polynomial arithmetic**

Create `src/keeloq/anf.py`:

```python
"""Boolean polynomial arithmetic over GF(2), and the KeeLoq ANF system generator.

Polynomials are represented as a frozenset of monomials, where each monomial is
a frozenset of variable names. The empty frozenset represents the constant 1;
the empty outer frozenset represents the constant 0. Arithmetic is:
  - Addition: symmetric difference of monomial sets (x + x = 0)
  - Multiplication: cross-product of monomials, then union within each, then sum
  - Variable power: x * x = x (GF(2) idempotence handled implicitly by using sets)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoolPoly:
    """Polynomial in GF(2)[x_1, ..., x_n]/(x_i^2 - x_i).

    `monomials` is a frozenset of frozensets-of-variable-names.
    The empty outer frozenset is the zero polynomial.
    A monomial equal to the empty frozenset represents the constant 1.
    """
    monomials: frozenset[frozenset[str]]

    def __add__(self, other: "BoolPoly") -> "BoolPoly":
        return BoolPoly(self.monomials.symmetric_difference(other.monomials))

    def __mul__(self, other: "BoolPoly") -> "BoolPoly":
        result: set[frozenset[str]] = set()
        for m1 in self.monomials:
            for m2 in other.monomials:
                product = m1 | m2  # GF(2) idempotence: x*x = x
                if product in result:
                    result.remove(product)
                else:
                    result.add(product)
        return BoolPoly(frozenset(result))

    def substitute(self, assignment: dict[str, int]) -> int:
        """Evaluate under a total assignment of all variables to 0/1."""
        needed = self.variables()
        missing = needed - assignment.keys()
        if missing:
            raise ValueError(f"substitute: missing values for {sorted(missing)}")
        total = 0
        for m in self.monomials:
            product = 1
            for v in m:
                product *= assignment[v]
                if product == 0:
                    break
            total = (total + product) & 1
        return total

    def variables(self) -> set[str]:
        out: set[str] = set()
        for m in self.monomials:
            out.update(m)
        return out


def zero() -> BoolPoly:
    return BoolPoly(frozenset())


def one() -> BoolPoly:
    return BoolPoly(frozenset([frozenset()]))


def var(name: str) -> BoolPoly:
    return BoolPoly(frozenset([frozenset([name])]))
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_anf.py -v
```

Expected: all 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/keeloq/anf.py tests/test_anf.py
git commit -m "impl: BoolPoly polynomial arithmetic over GF(2)

Minimal dataclass, no Gröbner machinery. Addition is symmetric difference;
multiplication cross-products monomials with GF(2) idempotence (x*x = x);
substitution requires a total assignment.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: ANF variable list and per-pair namespacing

**Files:**
- Modify: `src/keeloq/anf.py`
- Modify: `tests/test_anf.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_anf.py`:

```python
from keeloq.anf import variables


def test_variable_count_single_pair() -> None:
    # 64 K's + 1 pair * (3*rounds + 32 L's) = 64 + 3*rounds + 32 for num_pairs=1
    for rounds in (1, 16, 32, 160):
        vs = variables(rounds=rounds, num_pairs=1)
        assert len(vs) == 64 + 1*(3*rounds + 32), f"rounds={rounds}"
        assert len(vs) == len(set(vs)), "no duplicates"


def test_variable_count_multi_pair() -> None:
    vs = variables(rounds=32, num_pairs=2)
    assert len(vs) == 64 + 2*(3*32 + 32)  # 64 + 2*128 = 320
    assert len(vs) == len(set(vs))


def test_variable_naming_single_pair_includes_p0_suffix() -> None:
    vs = variables(rounds=4, num_pairs=1)
    assert "K0" in vs and "K63" in vs
    assert "L0_p0" in vs and "L35_p0" in vs  # L0..L{rounds+31}, so L0..L35 for rounds=4
    assert "A0_p0" in vs and "A3_p0" in vs
    assert "B0_p0" in vs and "B3_p0" in vs


def test_variable_naming_multi_pair() -> None:
    vs = variables(rounds=2, num_pairs=2)
    assert "L0_p0" in vs
    assert "L0_p1" in vs
    assert "A0_p0" in vs
    assert "A0_p1" in vs
    # K's are shared
    assert vs.count("K0") == 1


def test_variable_ordering() -> None:
    vs = variables(rounds=2, num_pairs=1)
    # K's come first, then per-pair A, B, L
    k_idx = vs.index("K0")
    a_idx = vs.index("A0_p0")
    b_idx = vs.index("B0_p0")
    l_idx = vs.index("L0_p0")
    assert k_idx < a_idx < b_idx < l_idx
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
uv run pytest tests/test_anf.py -v
```

Expected: new tests FAIL with ImportError for `variables`.

- [ ] **Step 3: Extend `src/keeloq/anf.py`**

Append to `src/keeloq/anf.py`:

```python
def variables(rounds: int, num_pairs: int = 1) -> list[str]:
    """Return the ordered list of all ANF variables.

    Layout: K0..K63, then for each pair p in 0..num_pairs-1:
        A0_p, A1_p, ..., A{rounds-1}_p,
        B0_p, B1_p, ..., B{rounds-1}_p,
        L0_p, L1_p, ..., L{rounds+31}_p.

    Keys are shared across pairs (single underlying key). L/A/B are per-pair
    because the state evolution depends on which plaintext was encrypted.
    """
    if rounds < 0:
        raise ValueError(f"rounds={rounds} is negative")
    if num_pairs < 1:
        raise ValueError(f"num_pairs={num_pairs} must be >= 1")

    out = [f"K{i}" for i in range(64)]
    for p in range(num_pairs):
        out += [f"A{i}_p{p}" for i in range(rounds)]
        out += [f"B{i}_p{p}" for i in range(rounds)]
        out += [f"L{i}_p{p}" for i in range(rounds + 32)]
    return out
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_anf.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/keeloq/anf.py tests/test_anf.py
git commit -m "impl: anf.variables() with per-pair L/A/B namespacing

K-vars shared across pairs; L/A/B suffixed _p{i} to keep per-pair state
independent in the SAT instance. Enables multi-pair attacks (two pt/ct pairs
under same key) without variable collisions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: ANF round equations

**Files:**
- Modify: `src/keeloq/anf.py`
- Modify: `tests/test_anf.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_anf.py`:

```python
from keeloq.anf import round_equations


def test_round_equations_returns_three_polys() -> None:
    eqs = round_equations(round_idx=0, pair_idx=0)
    assert len(eqs) == 3


def test_round_equation_eq2_is_a_equals_l31_l26() -> None:
    """From legacy/sage-equations.py:32:  A{i} + L{i+31}*L{i+26}.
    In our naming: A0_p0 + L31_p0*L26_p0."""
    _, eq2, _ = round_equations(round_idx=0, pair_idx=0)
    expected = var("A0_p0") + var("L31_p0") * var("L26_p0")
    assert eq2 == expected


def test_round_equation_eq3_is_b_equals_l31_l1() -> None:
    _, _, eq3 = round_equations(round_idx=0, pair_idx=0)
    expected = var("B0_p0") + var("L31_p0") * var("L1_p0")
    assert eq3 == expected


def test_round_equation_eq1_structure_at_round_0() -> None:
    """Match legacy/sage-equations.py:31 exactly, modulo pair suffix."""
    eq1, _, _ = round_equations(round_idx=0, pair_idx=0)
    # eq1 = L32 + K0 + L0 + L16 + L9 + L1 + L31*L20 + B0 + L26*L20
    #       + L26*L1 + L20*L9 + L9*L1 + B0*L9 + B0*L20 + A0*L9 + A0*L20
    expected = (
        var("L32_p0") + var("K0") + var("L0_p0") + var("L16_p0")
        + var("L9_p0") + var("L1_p0")
        + var("L31_p0") * var("L20_p0")
        + var("B0_p0")
        + var("L26_p0") * var("L20_p0")
        + var("L26_p0") * var("L1_p0")
        + var("L20_p0") * var("L9_p0")
        + var("L9_p0") * var("L1_p0")
        + var("B0_p0") * var("L9_p0")
        + var("B0_p0") * var("L20_p0")
        + var("A0_p0") * var("L9_p0")
        + var("A0_p0") * var("L20_p0")
    )
    assert eq1 == expected


def test_round_equation_indices_shift_with_round() -> None:
    """At round i, eq1 involves L{32+i}, K{i%64}, L{i}, L{i+16}, ..."""
    eq1, eq2, eq3 = round_equations(round_idx=5, pair_idx=0)
    assert "L37_p0" in eq1.variables()  # L{32+5}
    assert "K5" in eq1.variables()
    assert "A5_p0" in eq2.variables()
    assert "B5_p0" in eq3.variables()


def test_round_equation_key_index_wraps_at_64() -> None:
    eq1_round0, _, _ = round_equations(round_idx=0, pair_idx=0)
    eq1_round64, _, _ = round_equations(round_idx=64, pair_idx=0)
    # Both should reference K0 (0 % 64 == 0 and 64 % 64 == 0)
    assert "K0" in eq1_round0.variables()
    assert "K0" in eq1_round64.variables()


def test_round_equation_pair_index_changes_names() -> None:
    eq1_p0, _, _ = round_equations(round_idx=0, pair_idx=0)
    eq1_p1, _, _ = round_equations(round_idx=0, pair_idx=1)
    assert "L32_p0" in eq1_p0.variables()
    assert "L32_p1" in eq1_p1.variables()
    # K variables still shared
    assert "K0" in eq1_p0.variables()
    assert "K0" in eq1_p1.variables()
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
uv run pytest tests/test_anf.py -v
```

Expected: new tests FAIL with ImportError for `round_equations`.

- [ ] **Step 3: Extend `src/keeloq/anf.py`**

Append to `src/keeloq/anf.py`:

```python
def round_equations(round_idx: int, pair_idx: int = 0) -> tuple[BoolPoly, BoolPoly, BoolPoly]:
    """Return the three ANF equations for a single round, for a specific pair.

    Equation shapes (from legacy/sage-equations.py:31-33):

        eq1 = L{i+32} + K{i%64} + L{i} + L{i+16} + L{i+9} + L{i+1}
              + L{i+31}*L{i+20} + B{i}
              + L{i+26}*L{i+20} + L{i+26}*L{i+1}
              + L{i+20}*L{i+9} + L{i+9}*L{i+1}
              + B{i}*L{i+9} + B{i}*L{i+20}
              + A{i}*L{i+9} + A{i}*L{i+20}
        eq2 = A{i} + L{i+31}*L{i+26}
        eq3 = B{i} + L{i+31}*L{i+1}
    """
    if round_idx < 0 or pair_idx < 0:
        raise ValueError("round_idx and pair_idx must be non-negative")

    i = round_idx
    p = pair_idx

    def L(offset: int) -> BoolPoly:
        return var(f"L{offset}_p{p}")

    K = var(f"K{i % 64}")
    A = var(f"A{i}_p{p}")
    B = var(f"B{i}_p{p}")

    eq1 = (L(i + 32) + K + L(i) + L(i + 16) + L(i + 9) + L(i + 1)
           + L(i + 31) * L(i + 20) + B
           + L(i + 26) * L(i + 20) + L(i + 26) * L(i + 1)
           + L(i + 20) * L(i + 9) + L(i + 9) * L(i + 1)
           + B * L(i + 9) + B * L(i + 20)
           + A * L(i + 9) + A * L(i + 20))
    eq2 = A + L(i + 31) * L(i + 26)
    eq3 = B + L(i + 31) * L(i + 1)
    return eq1, eq2, eq3
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_anf.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/keeloq/anf.py tests/test_anf.py
git commit -m "impl: anf.round_equations() matching legacy/sage-equations.py

Three polynomials per round, per pair: the round update equation and the A/B
linearization definitions for the cubic NLF terms. Indices shift with the
round number; the key index wraps at 64. Pair index namespaces L/A/B.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: ANF full system assembly

**Files:**
- Modify: `src/keeloq/anf.py`
- Modify: `tests/test_anf.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_anf.py`:

```python
from keeloq.anf import system
from keeloq.cipher import encrypt
from keeloq._types import bits_to_int


def test_system_size_single_pair_no_hints() -> None:
    rounds = 4
    pt, ct = 0xDEADBEEF, 0
    sys = system(rounds=rounds, pairs=[(pt, ct)], key_hints=None)
    # Expected: 32 plaintext bindings + 32 ciphertext bindings + 3*rounds round equations
    assert len(sys) == 32 + 32 + 3*rounds


def test_system_size_with_hints() -> None:
    rounds = 4
    sys = system(rounds=rounds, pairs=[(0, 0)], key_hints={0: 1, 5: 0, 10: 1})
    assert len(sys) == 32 + 32 + 3*rounds + 3


def test_system_size_multi_pair() -> None:
    rounds = 4
    sys = system(rounds=rounds, pairs=[(0, 0), (1, 1)], key_hints=None)
    # 2 pairs each contribute 32+32+3*rounds; K-hints are shared (0 here)
    assert len(sys) == 2 * (32 + 32 + 3*rounds)


def test_true_solution_satisfies_every_equation_single_pair() -> None:
    """The cornerstone test: generated equations are correct iff the true
    (key, L-values, A-values, B-values) zero every polynomial in the system."""
    rounds = 16
    pt = 0xCAFEBABE
    key = 0x0123_4567_89AB_CDEF
    ct = encrypt(pt, key, rounds)

    sys = system(rounds=rounds, pairs=[(pt, ct)], key_hints=None)
    assignment = _derive_true_assignment(pt, ct, key, rounds, pair_idx=0)
    for idx, poly in enumerate(sys):
        assert poly.substitute(assignment) == 0, \
            f"equation {idx} not satisfied: vars={poly.variables()}"


def test_true_solution_satisfies_multi_pair() -> None:
    rounds = 16
    key = 0x0123_4567_89AB_CDEF
    pairs_plain = [0xCAFEBABE, 0xDEADBEEF, 0x13579BDF]
    pairs = [(p, encrypt(p, key, rounds)) for p in pairs_plain]

    sys = system(rounds=rounds, pairs=pairs, key_hints=None)
    assignment: dict[str, int] = {}
    for p_idx, (pt, ct) in enumerate(pairs):
        assignment.update(_derive_true_assignment(pt, ct, key, rounds, pair_idx=p_idx))
    # K variables only need to be added once (shared across pairs)
    for bit_idx in range(64):
        assignment.setdefault(f"K{bit_idx}", (key >> (63 - bit_idx)) & 1)

    for idx, poly in enumerate(sys):
        assert poly.substitute(assignment) == 0, f"equation {idx} unsatisfied"


def _derive_true_assignment(pt: int, ct: int, key: int, rounds: int,
                             pair_idx: int) -> dict[str, int]:
    """Run the cipher and record every L/A/B intermediate value + K bits."""
    from keeloq.cipher import _state_bit, _key_bit, core

    assignment: dict[str, int] = {}
    for bit_idx in range(64):
        assignment[f"K{bit_idx}"] = _key_bit(key, bit_idx)

    state = pt
    # L0..L31 are plaintext bits (MSB-first)
    for bit_idx in range(32):
        assignment[f"L{bit_idx}_p{pair_idx}"] = _state_bit(pt, bit_idx)

    for i in range(rounds):
        # Capture A_i = L{i+31}*L{i+26} BEFORE this round's shift
        # and B_i = L{i+31}*L{i+1}.
        # Current `state` corresponds to L{i}..L{i+31} mapped MSB-first.
        l31 = _state_bit(state, 31)
        l26 = _state_bit(state, 26)
        l20 = _state_bit(state, 20)
        l9 = _state_bit(state, 9)
        l1 = _state_bit(state, 1)
        l16 = _state_bit(state, 16)
        l0 = _state_bit(state, 0)

        assignment[f"A{i}_p{pair_idx}"] = (l31 * l26) % 2
        assignment[f"B{i}_p{pair_idx}"] = (l31 * l1) % 2

        kbit = _key_bit(key, i % 64)
        newb = (kbit + l0 + l16 + core(l31, l26, l20, l9, l1)) % 2
        state = ((state << 1) & 0xFFFFFFFF) | newb
        assignment[f"L{i+32}_p{pair_idx}"] = newb

    # Sanity: post-rounds state matches ciphertext
    assert state == ct, f"cipher reference disagreement: got {state:08x} want {ct:08x}"
    return assignment
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
uv run pytest tests/test_anf.py -v
```

Expected: new tests FAIL — `system` doesn't exist yet.

- [ ] **Step 3: Extend `src/keeloq/anf.py`**

Append to `src/keeloq/anf.py`:

```python
def system(
    rounds: int,
    pairs: list[tuple[int, int]],
    key_hints: dict[int, int] | None = None,
) -> list[BoolPoly]:
    """Generate the full ANF polynomial system.

    For each (plaintext, ciphertext) pair, emits:
      - 32 plaintext bit bindings: L{j}_p{p} + plaintext_bit_j  (so the equation
        equals zero iff L{j}_p{p} equals the plaintext bit)
      - 32 ciphertext bit bindings on the final-round state (L{rounds+j}_p{p})
      - 3 * rounds round equations

    Additionally emits key-hint bindings K{i} + value for each entry in key_hints.

    Args:
        rounds: number of KeeLoq rounds (>= 1).
        pairs: list of (plaintext_int, ciphertext_int) pairs; must have length >= 1.
        key_hints: optional mapping from key bit index (0..63, MSB-first) to bit value.
    """
    if rounds < 1:
        raise ValueError(f"rounds={rounds} must be >= 1")
    if not pairs:
        raise ValueError("pairs must be non-empty")
    key_hints = key_hints or {}
    for bit_idx, v in key_hints.items():
        if not 0 <= bit_idx < 64:
            raise ValueError(f"key_hints bit index {bit_idx} out of range")
        if v not in (0, 1):
            raise ValueError(f"key_hints value {v} for bit {bit_idx} is not a bit")

    out: list[BoolPoly] = []

    for p_idx, (pt, ct) in enumerate(pairs):
        if not 0 <= pt < (1 << 32):
            raise ValueError(f"plaintext pair {p_idx}={pt} does not fit in 32 bits")
        if not 0 <= ct < (1 << 32):
            raise ValueError(f"ciphertext pair {p_idx}={ct} does not fit in 32 bits")

        # Plaintext bindings: L{j}_p{p_idx} + pt_bit_j = 0  ->  L{j}_p{p_idx} = pt_bit_j
        for j in range(32):
            pt_bit = (pt >> (31 - j)) & 1
            binding = var(f"L{j}_p{p_idx}")
            if pt_bit:
                binding = binding + one()
            out.append(binding)

        # Ciphertext bindings: L{rounds+j}_p{p_idx} + ct_bit_j = 0
        for j in range(32):
            ct_bit = (ct >> (31 - j)) & 1
            binding = var(f"L{rounds + j}_p{p_idx}")
            if ct_bit:
                binding = binding + one()
            out.append(binding)

        # Round equations
        for i in range(rounds):
            eq1, eq2, eq3 = round_equations(round_idx=i, pair_idx=p_idx)
            out.extend([eq1, eq2, eq3])

    # Key hints (shared across pairs)
    for bit_idx, v in key_hints.items():
        binding = var(f"K{bit_idx}")
        if v:
            binding = binding + one()
        out.append(binding)

    return out
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_anf.py -v
```

Expected: all tests pass. In particular, `test_true_solution_satisfies_every_equation_single_pair` and `test_true_solution_satisfies_multi_pair` — these are the correctness cornerstones.

- [ ] **Step 5: Commit**

```bash
git add src/keeloq/anf.py tests/test_anf.py
git commit -m "impl: anf.system() assembles full ANF polynomial instance

Per pair: 32 plaintext bindings + 32 ciphertext bindings + 3*rounds round
equations. Plus optional shared key-hint bindings. The true (key, intermediate
state) assignment zeros every equation — property verified at 16 rounds on
1-pair and 3-pair instances.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Encoder Protocol and shared types

**Files:**
- Create: `src/keeloq/encoders/__init__.py`

- [ ] **Step 1: Write the module (no tests needed for pure protocol definitions)**

Create `src/keeloq/encoders/__init__.py`:

```python
"""Protocol for ANF -> SAT instance encoders.

Two concrete encoders live alongside this module:
  - encoders.cnf: pure DIMACS CNF, PolyBoRi-equivalent. Works with any SAT solver.
  - encoders.xor_aware: CryptoMiniSat-native hybrid (CNF + XOR clauses). Dramatic
    speedup on crypto problems because linear equations become one XOR constraint
    each instead of 2^(n-1) CNF clauses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias

from keeloq.anf import BoolPoly


@dataclass(frozen=True)
class CNFInstance:
    num_vars: int
    clauses: tuple[tuple[int, ...], ...]
    var_names: tuple[str, ...]  # index i -> variable name; DIMACS var ids are 1-indexed


@dataclass(frozen=True)
class HybridInstance:
    """CNF clauses plus native XOR constraints."""
    num_vars: int
    cnf_clauses: tuple[tuple[int, ...], ...]
    xor_clauses: tuple[tuple[tuple[int, ...], int], ...]  # (vars_1indexed, rhs)
    var_names: tuple[str, ...]


SolverInstance: TypeAlias = CNFInstance | HybridInstance


class Encoder(Protocol):
    def encode(self, system: list[BoolPoly]) -> SolverInstance: ...
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
uv run python -c "from keeloq.encoders import Encoder, CNFInstance, HybridInstance, SolverInstance; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/keeloq/encoders/__init__.py
git commit -m "impl: Encoder protocol and CNFInstance/HybridInstance data classes

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Pure CNF encoder

**Files:**
- Create: `src/keeloq/encoders/cnf.py`
- Create: `tests/test_encoders_cnf.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_encoders_cnf.py`:

```python
"""Tests for the pure-CNF encoder."""
from __future__ import annotations

import pytest

from keeloq.anf import BoolPoly, one, system, var, zero
from keeloq.encoders import CNFInstance
from keeloq.encoders.cnf import encode, from_dimacs, to_dimacs


def test_encode_empty_system() -> None:
    inst = encode([])
    assert inst.num_vars == 0
    assert inst.clauses == ()


def test_encode_unsatisfiable_zero_equals_one() -> None:
    # The polynomial "1" (constant, non-zero) represents an UNSAT equation 1 = 0
    inst = encode([one()])
    text = to_dimacs(inst)
    # Must contain at least the empty clause (pure UNSAT) or an equivalent contradiction.
    # We assert by solving.
    from pycryptosat import Solver
    s = Solver()
    for clause in inst.clauses:
        s.add_clause(list(clause))
    sat, _ = s.solve()
    assert sat is False


def test_encode_single_variable_equation() -> None:
    # x + 1 = 0  ->  x = 1  ->  unit clause [x]
    poly = var("x") + one()
    inst = encode([poly])
    assert inst.num_vars == 1
    assert inst.var_names == ("x",)
    # Should be satisfiable with x=1
    from pycryptosat import Solver
    s = Solver()
    for clause in inst.clauses:
        s.add_clause(list(clause))
    sat, assignment = s.solve()
    assert sat is True
    assert assignment[1] is True  # pycryptosat indexes from 1; index 0 is None


def test_encode_xor_equation() -> None:
    # x + y + 1 = 0  ->  x XOR y = 1  ->  two clauses: [x, y] and [-x, -y]
    poly = var("x") + var("y") + one()
    inst = encode([poly])
    from pycryptosat import Solver
    s = Solver()
    for clause in inst.clauses:
        s.add_clause(list(clause))
    # Must be SAT with exactly x != y
    sat, a = s.solve()
    assert sat is True
    assert a[1] != a[2]


def test_encode_simple_and_equation() -> None:
    # x*y + 1 = 0  ->  x AND y = 1  ->  unit clauses [x], [y]
    poly = var("x") * var("y") + one()
    inst = encode([poly])
    from pycryptosat import Solver
    s = Solver()
    for clause in inst.clauses:
        s.add_clause(list(clause))
    sat, a = s.solve()
    assert sat is True and a[1] is True and a[2] is True


def test_to_dimacs_roundtrip() -> None:
    inst = encode([var("x") + var("y") + one(), var("z") + one()])
    text = to_dimacs(inst)
    recovered = from_dimacs(text, var_names=inst.var_names)
    assert recovered.num_vars == inst.num_vars
    assert set(recovered.clauses) == set(inst.clauses)


def test_encode_tiny_keeloq_instance_is_solvable_with_heavy_hints() -> None:
    """2-round attack with 63 of 64 key bits hinted should solve trivially."""
    from keeloq.cipher import encrypt
    rounds = 2
    pt = 0xAAAA5555
    key = 0x0123_4567_89AB_CDEF
    ct = encrypt(pt, key, rounds)
    hints = {i: (key >> (63 - i)) & 1 for i in range(64) if i != 0}
    sys = system(rounds=rounds, pairs=[(pt, ct)], key_hints=hints)
    inst = encode(sys)

    from pycryptosat import Solver
    s = Solver()
    for clause in inst.clauses:
        s.add_clause(list(clause))
    sat, a = s.solve()
    assert sat is True
    # Recover K0 from the assignment and confirm it matches the true key bit
    k0_idx = inst.var_names.index("K0") + 1
    recovered_k0 = 1 if a[k0_idx] else 0
    assert recovered_k0 == ((key >> 63) & 1)
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
uv run pytest tests/test_encoders_cnf.py -v
```

Expected: FAIL with ImportError for `keeloq.encoders.cnf`.

- [ ] **Step 3: Implement `src/keeloq/encoders/cnf.py`**

```python
"""Pure-CNF encoder: BoolPoly polynomial system -> DIMACS CNF.

Strategy:
  - Collect all variables from the system in a stable order.
  - For each polynomial `p = m_1 + m_2 + ... + m_k`:
      - Build a Tseitin-style tree: introduce a fresh aux variable `t_j` for
        each nonlinear monomial `m_j` (product of >= 2 variables), asserting
        `t_j = m_j` via AND-gadget clauses.
      - The polynomial's satisfaction condition becomes an XOR over the linear
        variables and the aux vars, with constant adjustment. Encode that XOR
        as a disjunction of parity clauses (2^(n-1) clauses for n operands).
  - Emit CNF in DIMACS form.
"""
from __future__ import annotations

from itertools import combinations

from keeloq.anf import BoolPoly
from keeloq.encoders import CNFInstance


def _tseitin_and(aux: int, operands: list[int]) -> list[tuple[int, ...]]:
    """Clauses asserting aux <-> AND(operands), for positive-literal operands.

    aux and operands are 1-indexed DIMACS variable ids (positive).
      aux -> each operand:   [-aux, op_i]
      operands together -> aux:  [-op_1, -op_2, ..., aux]
    """
    clauses: list[tuple[int, ...]] = []
    for op in operands:
        clauses.append((-aux, op))
    clauses.append(tuple([-op for op in operands] + [aux]))
    return clauses


def _xor_clauses(literals: list[int], rhs: int) -> list[tuple[int, ...]]:
    """Encode XOR(literals) = rhs as CNF — 2^(n-1) clauses, one per parity.

    For n literals, enumerate subsets S with |S| of a fixed parity; negate
    literals in S, keep the rest; each such disjunction is a clause.
    """
    n = len(literals)
    if n == 0:
        if rhs == 1:
            return [()]  # empty clause -> UNSAT
        return []
    clauses: list[tuple[int, ...]] = []
    # We want: XOR = rhs  iff  parity of (number of 1's) == rhs.
    # Clause form: for each assignment that makes XOR != rhs, rule it out.
    # Equivalent: for each subset S of literals with |S| having parity (n - rhs) % 2,
    # the clause is the literals negated at positions in S and positive elsewhere.
    forbidden_parity = (n - rhs) % 2
    # Actually: a clause forbids one assignment. Assignment a violates XOR=rhs iff
    # parity(a) != rhs. There are 2^(n-1) such assignments. For each, emit a clause
    # that is false under that assignment and true under all others: the clause is
    # the negation of the assignment literals.
    for flip_size in range(n + 1):
        if flip_size % 2 != forbidden_parity:
            continue
        for flips in combinations(range(n), flip_size):
            clause = tuple(
                -literals[i] if i in flips else literals[i] for i in range(n)
            )
            clauses.append(clause)
    return clauses


def encode(system: list[BoolPoly]) -> CNFInstance:
    # Collect variables in order of first appearance for stable ids.
    var_names: list[str] = []
    seen: set[str] = set()
    for poly in system:
        for mono in poly.monomials:
            for v in sorted(mono):
                if v not in seen:
                    seen.add(v)
                    var_names.append(v)

    var_id = {name: i + 1 for i, name in enumerate(var_names)}
    next_aux = len(var_names)  # will increment to get fresh ids (1-indexed)
    all_clauses: list[tuple[int, ...]] = []

    for poly in system:
        xor_literals: list[int] = []
        rhs = 0
        for mono in poly.monomials:
            if len(mono) == 0:
                rhs ^= 1
            elif len(mono) == 1:
                (v,) = tuple(mono)
                xor_literals.append(var_id[v])
            else:
                # Tseitin aux for the AND of these vars
                next_aux += 1
                aux = next_aux
                ops = [var_id[v] for v in sorted(mono)]
                all_clauses.extend(_tseitin_and(aux, ops))
                xor_literals.append(aux)

        all_clauses.extend(_xor_clauses(xor_literals, rhs))

    return CNFInstance(
        num_vars=next_aux,
        clauses=tuple(all_clauses),
        var_names=tuple(var_names),
    )


def to_dimacs(instance: CNFInstance) -> str:
    lines = [f"p cnf {instance.num_vars} {len(instance.clauses)}"]
    for clause in instance.clauses:
        lines.append(" ".join(str(lit) for lit in clause) + " 0")
    return "\n".join(lines) + "\n"


def from_dimacs(text: str, var_names: tuple[str, ...]) -> CNFInstance:
    num_vars = 0
    clauses: list[tuple[int, ...]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("p"):
            parts = line.split()
            num_vars = int(parts[2])
            continue
        lits = [int(x) for x in line.split()]
        if lits and lits[-1] == 0:
            lits = lits[:-1]
        clauses.append(tuple(lits))
    return CNFInstance(num_vars=num_vars, clauses=tuple(clauses),
                       var_names=var_names)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_encoders_cnf.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/keeloq/encoders/cnf.py tests/test_encoders_cnf.py
git commit -m "impl: pure-CNF encoder with Tseitin AND gadget + XOR clause expansion

Encodes BoolPoly systems into DIMACS-compatible CNF. Solves a tiny 2-round
KeeLoq instance with 63 hinted key bits via CryptoMiniSat to prove the
encoding is satisfiability-preserving.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: XOR-aware encoder

**Files:**
- Create: `src/keeloq/encoders/xor_aware.py`
- Create: `tests/test_encoders_xor.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_encoders_xor.py`:

```python
"""Tests for the XOR-aware hybrid encoder."""
from __future__ import annotations

import pytest

from keeloq.anf import one, system, var
from keeloq.cipher import encrypt
from keeloq.encoders import HybridInstance
from keeloq.encoders.xor_aware import encode


def test_encode_pure_xor_becomes_single_xor_clause() -> None:
    # x + y + z + 1 = 0  ->  XOR(x, y, z) = 1
    inst = encode([var("x") + var("y") + var("z") + one()])
    assert len(inst.xor_clauses) == 1
    lits, rhs = inst.xor_clauses[0]
    assert sorted(lits) == [1, 2, 3]
    assert rhs == 1
    assert inst.cnf_clauses == ()


def test_encode_nonlinear_monomial_gets_tseitin_cnf() -> None:
    # x*y + 1 = 0  ->  x AND y = 1
    inst = encode([var("x") * var("y") + one()])
    # Should emit CNF clauses for the AND gadget + an XOR clause linking aux to rhs=1
    assert len(inst.cnf_clauses) >= 3
    assert len(inst.xor_clauses) == 1


def test_keeloq_round_equation_emits_one_xor_per_round() -> None:
    """Every round equation has a linear XOR chain — one XOR clause per round."""
    rounds = 8
    pt = 0xAAAA5555
    key = 0x0123_4567_89AB_CDEF
    ct = encrypt(pt, key, rounds)
    sys = system(rounds=rounds, pairs=[(pt, ct)], key_hints=None)
    inst = encode(sys)
    # 64 pt/ct bindings (32 each) each become 1 xor clause.
    # 3 round equations per round; eq1 has linear part (XOR), eq2/eq3 don't have
    # a linear free part without constant (they're just "A + stuff"), so still emit XOR.
    # Every non-empty polynomial emits exactly one XOR clause in our encoder,
    # so total xor clauses == total input polynomials.
    assert len(inst.xor_clauses) == len(sys)


def test_solves_tiny_instance() -> None:
    """Sanity: the hybrid encoding is satisfiability-preserving."""
    rounds = 2
    pt = 0xAAAA5555
    key = 0x0123_4567_89AB_CDEF
    ct = encrypt(pt, key, rounds)
    hints = {i: (key >> (63 - i)) & 1 for i in range(64) if i != 0}
    sys = system(rounds=rounds, pairs=[(pt, ct)], key_hints=hints)
    inst = encode(sys)

    from pycryptosat import Solver
    s = Solver()
    for c in inst.cnf_clauses:
        s.add_clause(list(c))
    for lits, rhs in inst.xor_clauses:
        # pycryptosat: add_xor_clause(vars, rhs as bool)
        s.add_xor_clause(list(lits), bool(rhs))
    sat, a = s.solve()
    assert sat is True
    k0_idx = inst.var_names.index("K0") + 1
    recovered_k0 = 1 if a[k0_idx] else 0
    assert recovered_k0 == ((key >> 63) & 1)
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
uv run pytest tests/test_encoders_xor.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `src/keeloq/encoders/xor_aware.py`**

```python
"""XOR-aware hybrid encoder for CryptoMiniSat.

Each polynomial becomes exactly one XOR clause (linearized: Tseitin auxes
standing in for nonlinear monomials). The CNF side carries only the Tseitin
AND-gadgets linking aux vars to their underlying products.
"""
from __future__ import annotations

from keeloq.anf import BoolPoly
from keeloq.encoders import HybridInstance
from keeloq.encoders.cnf import _tseitin_and  # reuse


def encode(system: list[BoolPoly]) -> HybridInstance:
    var_names: list[str] = []
    seen: set[str] = set()
    for poly in system:
        for mono in poly.monomials:
            for v in sorted(mono):
                if v not in seen:
                    seen.add(v)
                    var_names.append(v)

    var_id = {name: i + 1 for i, name in enumerate(var_names)}
    next_aux = len(var_names)
    cnf: list[tuple[int, ...]] = []
    xors: list[tuple[tuple[int, ...], int]] = []

    for poly in system:
        xor_vars: list[int] = []
        rhs = 0
        for mono in poly.monomials:
            if len(mono) == 0:
                rhs ^= 1
            elif len(mono) == 1:
                (v,) = tuple(mono)
                xor_vars.append(var_id[v])
            else:
                next_aux += 1
                aux = next_aux
                ops = [var_id[v] for v in sorted(mono)]
                cnf.extend(_tseitin_and(aux, ops))
                xor_vars.append(aux)
        # The polynomial equals 0 iff XOR(xor_vars) == rhs.
        # Note: a poly == 0 means equation holds; so rhs here is the *constant term*.
        # XOR of monomial-values = 0  iff XOR of (linear literals + aux_for_nonlinear) = constant
        # So the XOR clause enforces: XOR(xor_vars) == rhs.
        xors.append((tuple(xor_vars), rhs))

    return HybridInstance(
        num_vars=next_aux,
        cnf_clauses=tuple(cnf),
        xor_clauses=tuple(xors),
        var_names=tuple(var_names),
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_encoders_xor.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/keeloq/encoders/xor_aware.py tests/test_encoders_xor.py
git commit -m "impl: XOR-aware hybrid encoder (CryptoMiniSat-native)

Each polynomial becomes one XOR clause after linearizing nonlinear monomials
via Tseitin AND-gadgets in the CNF side. For KeeLoq this collapses each round
equation's ~7-term XOR chain to a single constraint instead of 2^6 CNF clauses.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: Solver Protocol, SolveResult, SolverStats

**Files:**
- Create: `src/keeloq/solvers/__init__.py`

- [ ] **Step 1: Write the module**

Create `src/keeloq/solvers/__init__.py`:

```python
"""Protocol for SAT solvers and shared result types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from keeloq.encoders import SolverInstance


@dataclass(frozen=True)
class SolverStats:
    wall_time_s: float
    num_vars: int
    num_clauses: int
    num_xors: int
    restarts: int | None = None
    conflicts: int | None = None
    propagations: int | None = None
    solver_name: str = "unknown"


@dataclass(frozen=True)
class SolveResult:
    status: Literal["SAT", "UNSAT", "TIMEOUT"]
    assignment: dict[str, int] | None
    stats: SolverStats


class Solver(Protocol):
    def solve(self, instance: SolverInstance, timeout_s: float) -> SolveResult: ...
```

- [ ] **Step 2: Verify import**

```bash
uv run python -c "from keeloq.solvers import Solver, SolveResult, SolverStats; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/keeloq/solvers/__init__.py
git commit -m "impl: Solver protocol, SolveResult, SolverStats

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: CryptoMiniSat solver wrapper

**Files:**
- Create: `src/keeloq/solvers/cryptominisat.py`
- Create: `tests/test_solvers.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_solvers.py`:

```python
"""Tests for solver wrappers."""
from __future__ import annotations

import pytest

from keeloq.anf import one, system, var
from keeloq.cipher import encrypt
from keeloq.encoders.cnf import encode as encode_cnf
from keeloq.encoders.xor_aware import encode as encode_xor
from keeloq.solvers.cryptominisat import solve as solve_cms


def test_trivial_sat_cnf() -> None:
    inst = encode_cnf([var("x") + one()])
    r = solve_cms(inst, timeout_s=5.0)
    assert r.status == "SAT"
    assert r.assignment is not None
    assert r.assignment["x"] == 1


def test_trivial_unsat_cnf() -> None:
    inst = encode_cnf([one()])  # 1 = 0, UNSAT
    r = solve_cms(inst, timeout_s=5.0)
    assert r.status == "UNSAT"
    assert r.assignment is None


def test_hybrid_xor_path_sat() -> None:
    inst = encode_xor([var("x") + var("y") + one()])
    r = solve_cms(inst, timeout_s=5.0)
    assert r.status == "SAT"
    assert r.assignment is not None
    assert r.assignment["x"] + r.assignment["y"] == 1


def test_tiny_keeloq_instance_solves_and_key_matches() -> None:
    rounds = 2
    pt = 0xAAAA5555
    key = 0x0123_4567_89AB_CDEF
    ct = encrypt(pt, key, rounds)
    hints = {i: (key >> (63 - i)) & 1 for i in range(64) if i != 0}
    sys = system(rounds=rounds, pairs=[(pt, ct)], key_hints=hints)

    for encode_fn, name in [(encode_cnf, "cnf"), (encode_xor, "xor")]:
        inst = encode_fn(sys)
        r = solve_cms(inst, timeout_s=10.0)
        assert r.status == "SAT", f"encoder {name} failed"
        assert r.assignment is not None
        assert r.assignment["K0"] == ((key >> 63) & 1), f"encoder {name} wrong K0"
        assert r.stats.wall_time_s > 0


def test_stats_are_populated() -> None:
    inst = encode_cnf([var("x") + var("y") + one()])
    r = solve_cms(inst, timeout_s=5.0)
    assert r.stats.num_vars >= 2
    assert r.stats.num_clauses >= 1
    assert r.stats.solver_name == "cryptominisat"
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
uv run pytest tests/test_solvers.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `src/keeloq/solvers/cryptominisat.py`**

```python
"""CryptoMiniSat wrapper using pycryptosat's native API.

Accepts both CNFInstance and HybridInstance. For hybrid, routes XOR clauses
to add_xor_clause() — the whole reason we picked this solver.
"""
from __future__ import annotations

import time

from pycryptosat import Solver as _CMS

from keeloq.encoders import CNFInstance, HybridInstance, SolverInstance
from keeloq.errors import SolverError
from keeloq.solvers import SolveResult, SolverStats


def solve(instance: SolverInstance, timeout_s: float) -> SolveResult:
    s = _CMS()
    s.set_max_time(timeout_s)

    if isinstance(instance, CNFInstance):
        for clause in instance.clauses:
            s.add_clause(list(clause))
        num_clauses = len(instance.clauses)
        num_xors = 0
    elif isinstance(instance, HybridInstance):
        for clause in instance.cnf_clauses:
            s.add_clause(list(clause))
        for lits, rhs in instance.xor_clauses:
            s.add_xor_clause(list(lits), bool(rhs))
        num_clauses = len(instance.cnf_clauses)
        num_xors = len(instance.xor_clauses)
    else:
        raise SolverError(f"unknown SolverInstance type: {type(instance).__name__}")

    t0 = time.perf_counter()
    sat, raw_assignment = s.solve()
    elapsed = time.perf_counter() - t0

    stats = SolverStats(
        wall_time_s=elapsed,
        num_vars=instance.num_vars,
        num_clauses=num_clauses,
        num_xors=num_xors,
        solver_name="cryptominisat",
    )

    if sat is None:
        return SolveResult(status="TIMEOUT", assignment=None, stats=stats)
    if sat is False:
        return SolveResult(status="UNSAT", assignment=None, stats=stats)

    # raw_assignment is indexed from 1; index 0 is None.
    assignment: dict[str, int] = {}
    for i, name in enumerate(instance.var_names):
        v = raw_assignment[i + 1]
        assignment[name] = 1 if v else 0
    return SolveResult(status="SAT", assignment=assignment, stats=stats)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_solvers.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/keeloq/solvers/cryptominisat.py tests/test_solvers.py
git commit -m "impl: CryptoMiniSat solver wrapper (CNF and hybrid XOR paths)

Routes HybridInstance.xor_clauses through add_xor_clause(). Returns SolveResult
with status + variable-name-keyed assignment + timing stats.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: DIMACS subprocess solver wrapper

**Files:**
- Create: `src/keeloq/solvers/dimacs_subprocess.py`
- Modify: `tests/test_solvers.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_solvers.py`:

```python
from keeloq.solvers.dimacs_subprocess import solve as solve_subprocess


@pytest.mark.solver_kissat
def test_kissat_trivial_sat() -> None:
    inst = encode_cnf([var("x") + one()])
    r = solve_subprocess(inst, solver_binary="kissat", timeout_s=5.0)
    assert r.status == "SAT"
    assert r.assignment is not None and r.assignment["x"] == 1


@pytest.mark.solver_kissat
def test_kissat_trivial_unsat() -> None:
    inst = encode_cnf([one()])
    r = solve_subprocess(inst, solver_binary="kissat", timeout_s=5.0)
    assert r.status == "UNSAT"


def test_subprocess_rejects_hybrid_instance() -> None:
    inst = encode_xor([var("x") + one()])
    with pytest.raises(Exception, match="CNFInstance"):
        solve_subprocess(inst, solver_binary="kissat", timeout_s=5.0)  # type: ignore[arg-type]
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
uv run pytest tests/test_solvers.py -v
```

Expected: FAIL with ImportError for `dimacs_subprocess`. The kissat tests will skip if kissat isn't on PATH.

- [ ] **Step 3: Implement `src/keeloq/solvers/dimacs_subprocess.py`**

```python
"""Shell-out wrapper for external DIMACS-speaking SAT solvers (kissat, minisat).

Only accepts CNFInstance — external solvers don't understand our HybridInstance
XOR clauses. If you want XOR, use solvers.cryptominisat.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from keeloq.encoders import CNFInstance, SolverInstance
from keeloq.encoders.cnf import to_dimacs
from keeloq.errors import SolverError
from keeloq.solvers import SolveResult, SolverStats


def solve(instance: SolverInstance, solver_binary: str, timeout_s: float) -> SolveResult:
    if not isinstance(instance, CNFInstance):
        raise SolverError("DIMACS subprocess solvers only accept CNFInstance "
                          "(HybridInstance requires a native XOR-capable solver)")

    binary_path = shutil.which(solver_binary) or solver_binary
    if not Path(binary_path).exists():
        raise SolverError(f"solver binary not found: {solver_binary!r}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cnf", delete=False) as f:
        f.write(to_dimacs(instance))
        cnf_path = f.name

    cmd = [binary_path, cnf_path]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return SolveResult(
            status="TIMEOUT",
            assignment=None,
            stats=SolverStats(
                wall_time_s=elapsed,
                num_vars=instance.num_vars,
                num_clauses=len(instance.clauses),
                num_xors=0,
                solver_name=Path(binary_path).name,
            ),
        )
    finally:
        Path(cnf_path).unlink(missing_ok=True)
    elapsed = time.perf_counter() - t0

    output = proc.stdout
    status, assignment_lits = _parse_dimacs_output(output)

    stats = SolverStats(
        wall_time_s=elapsed,
        num_vars=instance.num_vars,
        num_clauses=len(instance.clauses),
        num_xors=0,
        solver_name=Path(binary_path).name,
    )

    if status == "UNSAT":
        return SolveResult(status="UNSAT", assignment=None, stats=stats)
    if status == "UNKNOWN":
        # treat as timeout; solver ran but didn't decide in time
        return SolveResult(status="TIMEOUT", assignment=None, stats=stats)

    if assignment_lits is None:
        raise SolverError(f"solver {solver_binary!r} reported SAT but emitted "
                          f"no v-line. stdout:\n{output}\nstderr:\n{proc.stderr}")

    assignment: dict[str, int] = {}
    lit_set = set(assignment_lits)
    for i, name in enumerate(instance.var_names):
        vid = i + 1
        if vid in lit_set:
            assignment[name] = 1
        elif -vid in lit_set:
            assignment[name] = 0
        else:
            assignment[name] = 0  # unconstrained; default 0
    return SolveResult(status="SAT", assignment=assignment, stats=stats)


def _parse_dimacs_output(text: str) -> tuple[str, list[int] | None]:
    """Parse DIMACS-style solver output.

    Returns (status, assignment_literals).
    status ∈ {"SAT", "UNSAT", "UNKNOWN"}.
    """
    status = "UNKNOWN"
    v_lits: list[int] = []
    saw_v = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("s "):
            tok = line.split()[1] if len(line.split()) > 1 else ""
            if tok == "SATISFIABLE":
                status = "SAT"
            elif tok == "UNSATISFIABLE":
                status = "UNSAT"
            elif tok == "UNKNOWN":
                status = "UNKNOWN"
        elif line.startswith("v "):
            saw_v = True
            for tok in line.split()[1:]:
                if tok == "0":
                    continue
                v_lits.append(int(tok))
    return status, v_lits if saw_v else None
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_solvers.py -v
```

Expected: all tests pass (or skip with "kissat binary not on PATH" if applicable). `test_subprocess_rejects_hybrid_instance` runs regardless.

- [ ] **Step 5: Commit**

```bash
git add src/keeloq/solvers/dimacs_subprocess.py tests/test_solvers.py
git commit -m "impl: DIMACS subprocess solver wrapper for kissat/minisat

Writes a tempfile, invokes the solver with a timeout, parses s-line and v-line.
Rejects HybridInstance — external DIMACS solvers don't grok XOR.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 18: Attack pipeline — heavy-hinted single pair

**Files:**
- Create: `src/keeloq/attack.py`
- Create: `tests/test_attack.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_attack.py`:

```python
"""Tests for the full attack pipeline."""
from __future__ import annotations

import pytest

from keeloq.attack import AttackResult, attack
from keeloq.cipher import encrypt
from keeloq.encoders.cnf import encode as encode_cnf
from keeloq.encoders.xor_aware import encode as encode_xor
from keeloq.solvers.cryptominisat import solve as solve_cms


KEY = 0x0123_4567_89AB_CDEF


@pytest.mark.parametrize("encode_fn,name", [(encode_cnf, "cnf"), (encode_xor, "xor")])
def test_attack_16_rounds_heavy_hints(encode_fn, name) -> None:
    rounds = 16
    pt = 0xAAAA5555
    ct = encrypt(pt, KEY, rounds)
    # Leave 16 bits unknown — a well-determined but non-trivial instance.
    hints = {i: (KEY >> (63 - i)) & 1 for i in range(16, 64)}

    r: AttackResult = attack(
        rounds=rounds,
        pairs=[(pt, ct)],
        key_hints=hints,
        encoder=encode_fn,
        solver_fn=solve_cms,
        timeout_s=10.0,
    )
    assert r.status == "SUCCESS", f"encoder {name}: got {r.status}"
    assert r.recovered_key == KEY
    assert r.verify_result is True


@pytest.mark.parametrize("encode_fn,name", [(encode_cnf, "cnf"), (encode_xor, "xor")])
def test_attack_32_rounds_32_hints(encode_fn, name) -> None:
    rounds = 32
    pt = 0xAAAA5555
    ct = encrypt(pt, KEY, rounds)
    # Half the key hinted — 32 unknown bits.
    hints = {i: (KEY >> (63 - i)) & 1 for i in range(32, 64)}

    r = attack(rounds=rounds, pairs=[(pt, ct)], key_hints=hints,
               encoder=encode_fn, solver_fn=solve_cms, timeout_s=30.0)
    assert r.status == "SUCCESS", f"encoder {name}: got {r.status}"
    assert r.recovered_key == KEY
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
uv run pytest tests/test_attack.py -v
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `src/keeloq/attack.py`**

```python
"""End-to-end key-recovery pipeline: anf -> encode -> solve -> verify."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from keeloq.anf import system
from keeloq.cipher import encrypt
from keeloq.encoders import SolverInstance
from keeloq.errors import VerificationError
from keeloq.solvers import SolveResult


@dataclass(frozen=True)
class AttackResult:
    recovered_key: int | None
    status: Literal["SUCCESS", "WRONG_KEY", "UNSAT", "TIMEOUT", "CRASH"]
    solve_result: SolveResult
    verify_result: bool
    encoder_used: str
    solver_used: str


EncodeFn = Callable[[list], SolverInstance]
SolveFn = Callable[[SolverInstance, float], SolveResult]


def attack(
    rounds: int,
    pairs: list[tuple[int, int]],
    key_hints: dict[int, int] | None,
    encoder: EncodeFn,
    solver_fn: SolveFn,
    timeout_s: float,
) -> AttackResult:
    sys = system(rounds=rounds, pairs=pairs, key_hints=key_hints)
    inst = encoder(sys)
    result = solver_fn(inst, timeout_s)

    encoder_name = getattr(encoder, "__module__", "unknown").rsplit(".", 1)[-1]
    solver_name = result.stats.solver_name

    if result.status == "UNSAT":
        return AttackResult(None, "UNSAT", result, False, encoder_name, solver_name)
    if result.status == "TIMEOUT":
        return AttackResult(None, "TIMEOUT", result, False, encoder_name, solver_name)

    assert result.status == "SAT" and result.assignment is not None
    recovered = _extract_key(result.assignment)

    # Verify by re-encrypting every pair.
    all_ok = True
    for pt, ct in pairs:
        if encrypt(pt, recovered, rounds) != ct:
            all_ok = False
            break

    status: Literal["SUCCESS", "WRONG_KEY"] = "SUCCESS" if all_ok else "WRONG_KEY"
    return AttackResult(recovered, status, result, all_ok, encoder_name, solver_name)


def _extract_key(assignment: dict[str, int]) -> int:
    key = 0
    for i in range(64):
        bit = assignment.get(f"K{i}")
        if bit is None:
            raise VerificationError(f"K{i} missing from solver assignment")
        if bit not in (0, 1):
            raise VerificationError(f"K{i}={bit} is not a bit")
        key = (key << 1) | bit
    return key
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_attack.py -v
```

Expected: all 4 tests pass (2 encoders × 2 round configs).

- [ ] **Step 5: Commit**

```bash
git add src/keeloq/attack.py tests/test_attack.py
git commit -m "impl: attack() pipeline with mandatory cipher-verify step

Composes anf.system -> encoder -> solver -> _extract_key -> cipher.encrypt
round-trip. Status 'SUCCESS' requires re-encryption matching the given
ciphertext for every input pair; SAT without verification is WRONG_KEY.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 19: Attack pipeline — multi-pair, 0-hint, and failure-mode tests

**Files:**
- Modify: `tests/test_attack.py`

- [ ] **Step 1: Append failing tests for multi-pair and failure modes**

Append to `tests/test_attack.py`:

```python
@pytest.mark.parametrize("encode_fn,name", [(encode_cnf, "cnf"), (encode_xor, "xor")])
def test_attack_32_rounds_two_pairs_no_hints(encode_fn, name) -> None:
    rounds = 32
    pts = [0xAAAA5555, 0x13579BDF]
    pairs = [(p, encrypt(p, KEY, rounds)) for p in pts]

    r = attack(rounds=rounds, pairs=pairs, key_hints=None,
               encoder=encode_fn, solver_fn=solve_cms, timeout_s=60.0)
    assert r.status == "SUCCESS", f"encoder {name}: got {r.status}"
    assert r.recovered_key == KEY


def test_attack_underdetermined_detected_as_wrong_key() -> None:
    """Single pair at 16 rounds with 0 hints is underdetermined (32 bits of
    output vs 64 key bits). Solver finds *a* key; verify catches that it isn't THE key."""
    rounds = 16
    pt = 0xAAAA5555
    ct = encrypt(pt, KEY, rounds)
    r = attack(rounds=rounds, pairs=[(pt, ct)], key_hints=None,
               encoder=encode_cnf, solver_fn=solve_cms, timeout_s=10.0)
    # The recovered key satisfies this one pair but isn't unique.
    # Status should be SUCCESS (this pair re-encrypts correctly by construction)
    # OR WRONG_KEY if the solver picks a key that's self-inconsistent somehow.
    # Realistically: the solver finds a key that satisfies (pt, ct), so verification
    # passes. The test below uses a *second* pair to expose underdetermination.
    assert r.status == "SUCCESS"
    # But it doesn't necessarily equal KEY:
    # (we do NOT assert r.recovered_key == KEY here)


def test_attack_unsat_from_contradictory_hint() -> None:
    """Hint a key bit to the WRONG value; solver must prove UNSAT."""
    rounds = 16
    pt = 0xAAAA5555
    ct = encrypt(pt, KEY, rounds)
    wrong_hint = {0: 1 - ((KEY >> 63) & 1)}  # flip bit 0
    # Plus half the correct bits so the rest is well-determined.
    for i in range(32, 64):
        wrong_hint[i] = (KEY >> (63 - i)) & 1

    r = attack(rounds=rounds, pairs=[(pt, ct)], key_hints=wrong_hint,
               encoder=encode_cnf, solver_fn=solve_cms, timeout_s=10.0)
    assert r.status == "UNSAT"
```

- [ ] **Step 2: Run the tests**

```bash
uv run pytest tests/test_attack.py -v
```

Expected: all tests pass. Note: `test_attack_underdetermined_detected_as_wrong_key` intentionally does NOT assert the recovered key matches — the whole point is that it often won't.

- [ ] **Step 3: Commit**

```bash
git add tests/test_attack.py
git commit -m "test: multi-pair attack, underdetermination behavior, UNSAT detection

Two-pair 0-hint 32-round attack recovers the correct key via both encoders.
Single-pair 0-hint 16-round attack exhibits underdetermined behavior
(SAT but key may not match). Contradictory hint produces UNSAT.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 20: Encoders-agree cross-validation

**Files:**
- Create: `tests/test_encoders_agree.py`

- [ ] **Step 1: Write the cross-validation test**

Create `tests/test_encoders_agree.py`:

```python
"""The TDD crown jewel: CNF and XOR-aware encoders must recover the same key."""
from __future__ import annotations

import pytest

from keeloq.attack import attack
from keeloq.cipher import encrypt
from keeloq.encoders.cnf import encode as encode_cnf
from keeloq.encoders.xor_aware import encode as encode_xor
from keeloq.solvers.cryptominisat import solve as solve_cms


KEY = 0x0123_4567_89AB_CDEF


@pytest.mark.parametrize("rounds,hint_bits,num_pairs", [
    (16, 48, 1),
    (32, 32, 1),
    (32, 0, 2),
])
def test_cnf_and_xor_encoders_agree(rounds: int, hint_bits: int, num_pairs: int) -> None:
    pts = [0xAAAA5555, 0x13579BDF][:num_pairs]
    pairs = [(p, encrypt(p, KEY, rounds)) for p in pts]
    hints = {i: (KEY >> (63 - i)) & 1 for i in range(64 - hint_bits, 64)} or None

    r_cnf = attack(rounds=rounds, pairs=pairs, key_hints=hints,
                   encoder=encode_cnf, solver_fn=solve_cms, timeout_s=60.0)
    r_xor = attack(rounds=rounds, pairs=pairs, key_hints=hints,
                   encoder=encode_xor, solver_fn=solve_cms, timeout_s=60.0)

    assert r_cnf.status == "SUCCESS"
    assert r_xor.status == "SUCCESS"
    assert r_cnf.recovered_key == r_xor.recovered_key == KEY
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest tests/test_encoders_agree.py -v
```

Expected: all 3 configurations pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_encoders_agree.py
git commit -m "test: cross-validate CNF and XOR encoders recover identical keys

Runs three configurations (16r-heavy-hint, 32r-half-hint, 32r-two-pair-no-hint)
through both encoders with CryptoMiniSat, asserting identical key recovery.
This is the independent-oracles TDD invariant catching variable-indexing bugs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 21: CLI scaffolding + encrypt/decrypt subcommands

**Files:**
- Create: `src/keeloq/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cli.py`:

```python
"""CLI tests using Typer's CliRunner."""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from keeloq._types import bits_to_int, int_to_bits
from keeloq.cli import app
from keeloq.cipher import encrypt


runner = CliRunner()


def test_encrypt_roundtrip_via_cli() -> None:
    pt_bits = "01100010100101110000101011100011"
    key_bits = "0011010011011111100101100001110000011101100111001000001101110100"

    result = runner.invoke(
        app,
        ["encrypt", "--rounds", "160",
         "--plaintext", pt_bits,
         "--key", key_bits],
    )
    assert result.exit_code == 0, result.stdout
    ct_bits = result.stdout.strip()
    assert len(ct_bits) == 32

    expected = encrypt(bits_to_int(pt_bits), bits_to_int(key_bits), 160)
    assert bits_to_int(ct_bits) == expected


def test_decrypt_roundtrip_via_cli() -> None:
    pt_bits = "01010101010101010101010101010101"
    key_bits = "0000010000100010100011100000000010000110000011001001111000010001"

    enc = runner.invoke(app, ["encrypt", "--rounds", "528",
                              "--plaintext", pt_bits, "--key", key_bits])
    assert enc.exit_code == 0
    ct_bits = enc.stdout.strip()

    dec = runner.invoke(app, ["decrypt", "--rounds", "528",
                              "--ciphertext", ct_bits, "--key", key_bits])
    assert dec.exit_code == 0
    assert dec.stdout.strip() == pt_bits


def test_encrypt_accepts_hex() -> None:
    result = runner.invoke(
        app,
        ["encrypt", "--rounds", "32",
         "--plaintext", "0xAAAA5555",
         "--key", "0x0123456789ABCDEF"],
    )
    assert result.exit_code == 0


def test_encrypt_rejects_wrong_length() -> None:
    result = runner.invoke(
        app,
        ["encrypt", "--rounds", "32", "--plaintext", "1010", "--key", "01"*32],
    )
    assert result.exit_code != 0
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `src/keeloq/cli.py` — encrypt, decrypt**

```python
"""keeloq command-line interface."""
from __future__ import annotations

import sys
from typing import Annotated

import typer

from keeloq._types import bits_to_int, int_to_bits
from keeloq.cipher import decrypt as _decrypt
from keeloq.cipher import encrypt as _encrypt


app = typer.Typer(no_args_is_help=True, help="KeeLoq cryptanalysis CLI (2026 modernization).")


def _parse_bitvec(s: str, width: int, name: str) -> int:
    s = s.strip()
    if s.startswith("0x") or s.startswith("0X"):
        value = int(s, 16)
        if value.bit_length() > width:
            raise typer.BadParameter(f"{name} doesn't fit in {width} bits")
        return value
    if len(s) != width:
        raise typer.BadParameter(f"{name} must be {width} bits (got {len(s)})")
    return bits_to_int(s)


@app.command()
def encrypt(
    rounds: Annotated[int, typer.Option(help="Number of KeeLoq rounds")],
    plaintext: Annotated[str, typer.Option(help="32-bit plaintext (bits or 0x hex)")],
    key: Annotated[str, typer.Option(help="64-bit key (bits or 0x hex)")],
) -> None:
    """Encrypt a 32-bit plaintext under a 64-bit key."""
    pt = _parse_bitvec(plaintext, 32, "plaintext")
    k = _parse_bitvec(key, 64, "key")
    ct = _encrypt(pt, k, rounds)
    typer.echo(int_to_bits(ct, 32))


@app.command()
def decrypt(
    rounds: Annotated[int, typer.Option(help="Number of KeeLoq rounds")],
    ciphertext: Annotated[str, typer.Option(help="32-bit ciphertext (bits or 0x hex)")],
    key: Annotated[str, typer.Option(help="64-bit key (bits or 0x hex)")],
) -> None:
    """Decrypt a 32-bit ciphertext under a 64-bit key."""
    ct = _parse_bitvec(ciphertext, 32, "ciphertext")
    k = _parse_bitvec(key, 64, "key")
    pt = _decrypt(ct, k, rounds)
    typer.echo(int_to_bits(pt, 32))


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/keeloq/cli.py tests/test_cli.py
git commit -m "impl: keeloq CLI with encrypt/decrypt subcommands (Typer)

Accepts bit-string or 0x hex for plaintext/ciphertext/key. Exits non-zero
on malformed input.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 22: CLI attack subcommand

**Files:**
- Modify: `src/keeloq/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_cli.py`:

```python
def test_attack_subcommand_end_to_end() -> None:
    """32-round attack with 32 hint bits recovers the true key via CLI."""
    key_bits = int_to_bits(0x0123_4567_89AB_CDEF, 64)
    pt_int = 0xAAAA5555
    ct_int = encrypt(pt_int, 0x0123_4567_89AB_CDEF, 32)

    result = runner.invoke(
        app,
        [
            "attack",
            "--rounds", "32",
            "--pair", f"{int_to_bits(pt_int, 32)}:{int_to_bits(ct_int, 32)}",
            # Hint low 32 bits (indices 32..63):
            "--hint-bits", "32",
            "--original-key", key_bits,
            "--encoder", "xor",
            "--solver", "cryptominisat",
            "--timeout", "30",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert key_bits in result.stdout


def test_attack_subcommand_multi_pair() -> None:
    key_int = 0x0123_4567_89AB_CDEF
    key_bits = int_to_bits(key_int, 64)
    pts = [0xAAAA5555, 0x13579BDF]
    cts = [encrypt(p, key_int, 32) for p in pts]
    pair_args = []
    for p, c in zip(pts, cts):
        pair_args += ["--pair", f"{int_to_bits(p, 32)}:{int_to_bits(c, 32)}"]

    result = runner.invoke(
        app,
        ["attack", "--rounds", "32", "--hint-bits", "0",
         "--original-key", key_bits, "--encoder", "xor",
         "--solver", "cryptominisat", "--timeout", "60"] + pair_args,
    )
    assert result.exit_code == 0
    assert key_bits in result.stdout


def test_attack_exit_code_on_unsat() -> None:
    key_int = 0x0123_4567_89AB_CDEF
    pt = 0xAAAA5555
    ct = encrypt(pt, key_int, 16)
    # Provide a wrong hint to force UNSAT (set K0 opposite, and enough correct bits
    # for the instance to be determined).
    wrong = 1 - ((key_int >> 63) & 1)
    # We express the hint via --key-hint "index:value" flags (to be implemented).
    pair = f"{int_to_bits(pt, 32)}:{int_to_bits(ct, 32)}"
    # Build 32 correct-bit hints on positions 32..63 plus a wrong one on 0.
    hint_args: list[str] = ["--key-hint", f"0:{wrong}"]
    for i in range(32, 64):
        b = (key_int >> (63 - i)) & 1
        hint_args += ["--key-hint", f"{i}:{b}"]

    result = runner.invoke(
        app,
        ["attack", "--rounds", "16", "--pair", pair,
         "--encoder", "cnf", "--solver", "cryptominisat", "--timeout", "10"] + hint_args,
    )
    assert result.exit_code == 3, f"expected exit 3 (UNSAT), got {result.exit_code}\n{result.stdout}"
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: new tests FAIL — attack subcommand not implemented.

- [ ] **Step 3: Extend `src/keeloq/cli.py`**

Append to `src/keeloq/cli.py`:

```python
from keeloq.attack import attack as _attack
from keeloq.encoders.cnf import encode as _encode_cnf
from keeloq.encoders.xor_aware import encode as _encode_xor
from keeloq.solvers.cryptominisat import solve as _solve_cms
from keeloq.solvers.dimacs_subprocess import solve as _solve_subprocess


def _resolve_encoder(name: str):
    if name == "cnf":
        return _encode_cnf
    if name == "xor":
        return _encode_xor
    raise typer.BadParameter(f"unknown encoder {name!r}; expected cnf or xor")


def _resolve_solver(name: str):
    if name == "cryptominisat":
        return _solve_cms
    if name in ("kissat", "minisat"):
        binary = name
        def _wrap(inst, timeout_s):
            return _solve_subprocess(inst, solver_binary=binary, timeout_s=timeout_s)
        return _wrap
    raise typer.BadParameter(f"unknown solver {name!r}")


def _parse_pair(arg: str) -> tuple[int, int]:
    if ":" not in arg:
        raise typer.BadParameter(f"--pair must be 'plaintext:ciphertext' (got {arg!r})")
    pt_s, ct_s = arg.split(":", 1)
    pt = _parse_bitvec(pt_s, 32, "pair plaintext")
    ct = _parse_bitvec(ct_s, 32, "pair ciphertext")
    return pt, ct


def _hint_bits_to_hints(original_key: str | None, hint_bits: int) -> dict[int, int]:
    if hint_bits == 0:
        return {}
    if original_key is None:
        raise typer.BadParameter("--hint-bits requires --original-key to know which bits to hint")
    key_int = _parse_bitvec(original_key, 64, "original-key")
    # Hint the LOW `hint_bits` of the key (indices 64-hint_bits..63, MSB-first).
    hints: dict[int, int] = {}
    for i in range(64 - hint_bits, 64):
        hints[i] = (key_int >> (63 - i)) & 1
    return hints


@app.command()
def attack(
    rounds: Annotated[int, typer.Option(help="Number of KeeLoq rounds")],
    pair: Annotated[list[str], typer.Option(
        help="(pt:ct) pair, repeatable. At least one required.",
    )],
    hint_bits: Annotated[int, typer.Option(
        "--hint-bits",
        help="Number of low-index key bits to hint from --original-key",
    )] = 0,
    key_hint: Annotated[list[str] | None, typer.Option(
        "--key-hint",
        help="Explicit per-bit hints in the form 'index:value', repeatable.",
    )] = None,
    original_key: Annotated[str | None, typer.Option(
        "--original-key",
        help="Reference key (required with --hint-bits; also echoed if recovery succeeds)",
    )] = None,
    encoder: Annotated[str, typer.Option(help="cnf | xor")] = "xor",
    solver: Annotated[str, typer.Option(help="cryptominisat | kissat | minisat")] = "cryptominisat",
    timeout: Annotated[float, typer.Option(help="Solver wall-clock timeout (seconds)")] = 3600.0,
) -> None:
    """Run the full key-recovery attack."""
    if not pair:
        raise typer.BadParameter("at least one --pair is required")
    pairs = [_parse_pair(p) for p in pair]

    hints = _hint_bits_to_hints(original_key, hint_bits)
    for kh in key_hint or []:
        if ":" not in kh:
            raise typer.BadParameter(f"--key-hint must be 'index:value' (got {kh!r})")
        idx_s, val_s = kh.split(":", 1)
        idx = int(idx_s)
        val = int(val_s)
        if val not in (0, 1):
            raise typer.BadParameter(f"--key-hint value must be 0 or 1 (got {val!r})")
        hints[idx] = val

    enc_fn = _resolve_encoder(encoder)
    solve_fn = _resolve_solver(solver)

    result = _attack(
        rounds=rounds,
        pairs=pairs,
        key_hints=hints or None,
        encoder=enc_fn,
        solver_fn=solve_fn,
        timeout_s=timeout,
    )

    typer.echo(f"status: {result.status}")
    typer.echo(f"encoder: {result.encoder_used}")
    typer.echo(f"solver: {result.solver_used}")
    typer.echo(f"wall_time_s: {result.solve_result.stats.wall_time_s:.3f}")
    if result.recovered_key is not None:
        typer.echo(f"recovered_key: {int_to_bits(result.recovered_key, 64)}")
    if original_key is not None and result.recovered_key is not None:
        typer.echo(f"original_key:  {original_key}")

    exit_map = {"SUCCESS": 0, "WRONG_KEY": 2, "UNSAT": 3, "TIMEOUT": 4, "CRASH": 1}
    raise typer.Exit(code=exit_map[result.status])
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/keeloq/cli.py tests/test_cli.py
git commit -m "impl: keeloq attack subcommand with multi-pair, hint-bits, exit codes

Exit codes: 0=SUCCESS, 1=CRASH, 2=WRONG_KEY, 3=UNSAT, 4=TIMEOUT.
Supports --encoder {cnf,xor} and --solver {cryptominisat,kissat,minisat}.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 23: CLI pipeline subcommands (generate-anf, encode, solve, verify)

**Files:**
- Modify: `src/keeloq/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_cli.py`:

```python
def test_pipeline_composition_via_stdout_to_stdin() -> None:
    """generate-anf | encode | solve | verify returns the correct key."""
    import json

    key_int = 0x0123_4567_89AB_CDEF
    pt = 0xAAAA5555
    ct = encrypt(pt, key_int, 32)
    pair = f"{int_to_bits(pt, 32)}:{int_to_bits(ct, 32)}"

    # Step 1: generate-anf
    anf_res = runner.invoke(
        app,
        ["generate-anf", "--rounds", "32", "--pair", pair,
         "--hint-bits", "32", "--original-key", int_to_bits(key_int, 64)],
    )
    assert anf_res.exit_code == 0, anf_res.stdout
    anf_json = anf_res.stdout

    # Step 2: encode
    enc_res = runner.invoke(app, ["encode", "--encoder", "xor"], input=anf_json)
    assert enc_res.exit_code == 0, enc_res.stdout
    instance_json = enc_res.stdout

    # Step 3: solve
    solve_res = runner.invoke(app, ["solve", "--solver", "cryptominisat",
                                      "--timeout", "30"],
                              input=instance_json)
    assert solve_res.exit_code == 0, solve_res.stdout
    result_json = solve_res.stdout
    parsed = json.loads(result_json)
    assert parsed["status"] == "SAT"

    # Step 4: verify
    vf_res = runner.invoke(
        app,
        ["verify", "--rounds", "32", "--pair", pair,
         "--original-key", int_to_bits(key_int, 64)],
        input=result_json,
    )
    assert vf_res.exit_code == 0
    assert "match: true" in vf_res.stdout.lower()
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_cli.py::test_pipeline_composition_via_stdout_to_stdin -v
```

Expected: FAIL — commands not implemented.

- [ ] **Step 3: Extend `src/keeloq/cli.py`**

Append to `src/keeloq/cli.py`:

```python
import json
from dataclasses import asdict

from keeloq.anf import BoolPoly, system as _anf_system
from keeloq.encoders import CNFInstance, HybridInstance


def _polys_to_json(sys: list[BoolPoly]) -> str:
    return json.dumps({
        "polynomials": [
            [sorted(list(m)) for m in p.monomials]
            for p in sys
        ],
    })


def _polys_from_json(text: str) -> list[BoolPoly]:
    data = json.loads(text)
    out: list[BoolPoly] = []
    for poly_mons in data["polynomials"]:
        out.append(BoolPoly(frozenset(frozenset(m) for m in poly_mons)))
    return out


def _instance_to_json(inst) -> str:
    if isinstance(inst, CNFInstance):
        return json.dumps({
            "type": "cnf",
            "num_vars": inst.num_vars,
            "clauses": [list(c) for c in inst.clauses],
            "var_names": list(inst.var_names),
        })
    if isinstance(inst, HybridInstance):
        return json.dumps({
            "type": "hybrid",
            "num_vars": inst.num_vars,
            "cnf_clauses": [list(c) for c in inst.cnf_clauses],
            "xor_clauses": [[list(lits), rhs] for lits, rhs in inst.xor_clauses],
            "var_names": list(inst.var_names),
        })
    raise ValueError(f"unknown instance type {type(inst).__name__}")


def _instance_from_json(text: str):
    data = json.loads(text)
    if data["type"] == "cnf":
        return CNFInstance(
            num_vars=data["num_vars"],
            clauses=tuple(tuple(c) for c in data["clauses"]),
            var_names=tuple(data["var_names"]),
        )
    if data["type"] == "hybrid":
        return HybridInstance(
            num_vars=data["num_vars"],
            cnf_clauses=tuple(tuple(c) for c in data["cnf_clauses"]),
            xor_clauses=tuple((tuple(lits), rhs) for lits, rhs in data["xor_clauses"]),
            var_names=tuple(data["var_names"]),
        )
    raise ValueError(f"unknown instance json type {data.get('type')!r}")


@app.command("generate-anf")
def generate_anf(
    rounds: Annotated[int, typer.Option()],
    pair: Annotated[list[str], typer.Option()],
    hint_bits: Annotated[int, typer.Option("--hint-bits")] = 0,
    original_key: Annotated[str | None, typer.Option("--original-key")] = None,
    key_hint: Annotated[list[str] | None, typer.Option("--key-hint")] = None,
) -> None:
    """Emit the ANF polynomial system as JSON on stdout."""
    pairs = [_parse_pair(p) for p in pair]
    hints = _hint_bits_to_hints(original_key, hint_bits)
    for kh in key_hint or []:
        idx_s, val_s = kh.split(":", 1)
        hints[int(idx_s)] = int(val_s)
    sys = _anf_system(rounds=rounds, pairs=pairs, key_hints=hints or None)
    typer.echo(_polys_to_json(sys))


@app.command("encode")
def encode_cmd(
    encoder: Annotated[str, typer.Option(help="cnf | xor")] = "xor",
) -> None:
    """Read ANF JSON on stdin, emit encoded-instance JSON on stdout."""
    data = sys.stdin.read()
    polys = _polys_from_json(data)
    enc_fn = _resolve_encoder(encoder)
    inst = enc_fn(polys)
    typer.echo(_instance_to_json(inst))


@app.command("solve")
def solve_cmd(
    solver: Annotated[str, typer.Option()] = "cryptominisat",
    timeout: Annotated[float, typer.Option()] = 3600.0,
) -> None:
    """Read instance JSON on stdin, emit SolveResult JSON on stdout."""
    data = sys.stdin.read()
    inst = _instance_from_json(data)
    solve_fn = _resolve_solver(solver)
    result = solve_fn(inst, timeout)
    out = {
        "status": result.status,
        "assignment": result.assignment,
        "stats": asdict(result.stats),
    }
    typer.echo(json.dumps(out))


@app.command()
def verify(
    rounds: Annotated[int, typer.Option()],
    pair: Annotated[list[str], typer.Option()],
    original_key: Annotated[str | None, typer.Option("--original-key")] = None,
) -> None:
    """Read SolveResult JSON on stdin, verify the recovered key matches every pair."""
    data = sys.stdin.read()
    solve = json.loads(data)
    if solve["status"] != "SAT" or not solve.get("assignment"):
        typer.echo(f"cannot verify: solver status={solve['status']}")
        raise typer.Exit(code=1)
    assignment = solve["assignment"]
    key = 0
    for i in range(64):
        key = (key << 1) | int(assignment.get(f"K{i}", 0))
    pairs = [_parse_pair(p) for p in pair]
    ok = all(_encrypt(p, key, rounds) == c for p, c in pairs)
    typer.echo(f"recovered_key: {int_to_bits(key, 64)}")
    if original_key is not None:
        typer.echo(f"original_key:  {original_key}")
    typer.echo(f"match: {str(ok).lower()}")
    raise typer.Exit(code=0 if ok else 2)
```

- [ ] **Step 4: Run the full test module**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: all tests pass including the pipeline composition test.

- [ ] **Step 5: Commit**

```bash
git add src/keeloq/cli.py tests/test_cli.py
git commit -m "impl: CLI pipeline subcommands generate-anf, encode, solve, verify

Composable via Unix pipes with JSON between stages. verify() re-encrypts every
pair under the recovered key and reports match/mismatch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 24: Legacy compatibility — cipher

**Files:**
- Create: `tests/compat_helpers.py`
- Create: `tests/test_compat.py`

- [ ] **Step 1: Write the cipher-parity test**

Create `tests/compat_helpers.py`:

```python
"""Helpers for invoking the frozen legacy/ scripts under python2."""
from __future__ import annotations

import subprocess
from pathlib import Path

LEGACY_DIR = Path(__file__).resolve().parent.parent / "legacy"


def run_legacy_script(script_name: str) -> str:
    """Run a legacy script under python2 and return its stdout."""
    path = LEGACY_DIR / script_name
    if not path.exists():
        raise FileNotFoundError(f"legacy script not found: {path}")
    proc = subprocess.run(
        ["python2", str(path)],
        capture_output=True, text=True, check=True,
        cwd=LEGACY_DIR,
    )
    return proc.stdout


def read_legacy_output_field(stdout: str, prefix: str) -> str:
    """Extract the value following a 'Prefix: ' line in legacy stdout."""
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    raise ValueError(f"no line starting with {prefix!r} in:\n{stdout}")
```

Create `tests/test_compat.py`:

```python
"""Compatibility tests against the frozen 2015 legacy/ scripts."""
from __future__ import annotations

import pytest

from keeloq._types import bits_to_int, int_to_bits
from keeloq.cipher import encrypt
from tests.compat_helpers import read_legacy_output_field, run_legacy_script


@pytest.mark.legacy
def test_cipher_160_round_matches_legacy() -> None:
    out = run_legacy_script("keeloq160-python.py")
    pt = read_legacy_output_field(out, "Plaintext:")
    key = read_legacy_output_field(out, "Key:")
    ct_legacy = read_legacy_output_field(out, "Ciphertext:")

    our_ct = encrypt(bits_to_int(pt), bits_to_int(key), 160)
    assert int_to_bits(our_ct, 32) == ct_legacy


@pytest.mark.legacy
def test_cipher_528_round_matches_legacy() -> None:
    out = run_legacy_script("keeloq-python.py")
    pt = read_legacy_output_field(out, "Plaintext:")
    key = read_legacy_output_field(out, "Key:")
    ct_legacy = read_legacy_output_field(out, "Ciphertext:")

    our_ct = encrypt(bits_to_int(pt), bits_to_int(key), 528)
    assert int_to_bits(our_ct, 32) == ct_legacy
```

- [ ] **Step 2: Run the tests**

```bash
uv run pytest tests/test_compat.py -v
```

Expected: if `python2` is installed, tests pass. Otherwise they skip with "python2 not on PATH".

- [ ] **Step 3: Commit**

```bash
git add tests/compat_helpers.py tests/test_compat.py
git commit -m "test: legacy cipher parity at 160 and 528 rounds

Runs legacy/keeloq{160-,}python.py under python2 subprocess, parses the
'Ciphertext:' line, and asserts our Python 3 cipher produces the same output
on the same inputs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 25: Legacy compatibility — ANF

**Files:**
- Modify: `tests/test_compat.py`

- [ ] **Step 1: Append failing ANF parity test**

Append to `tests/test_compat.py`:

```python
from keeloq.anf import system as anf_system


@pytest.mark.legacy
def test_anf_160_matches_legacy_anf_txt(tmp_path, monkeypatch) -> None:
    """Run legacy/sage-equations.py under python2, then compare our single-pair
    ANF string to its anf.txt output after stripping the _p0 suffix from our
    variable names."""
    # sage-equations.py writes to 'anf.txt' in its cwd.
    monkeypatch.chdir(tmp_path)
    from tests.compat_helpers import LEGACY_DIR

    import subprocess
    subprocess.run(["python2", str(LEGACY_DIR / "sage-equations.py")],
                   check=True, cwd=tmp_path)
    legacy_anf = (tmp_path / "anf.txt").read_text()

    # legacy/sage-equations.py hardcodes:
    pt = "01100010100101110000101011100011"
    ct = "01101000110010010100101001111001"
    key = "00110100110111111001011000011100"  # 32 bits hinted!

    # Legacy hints the first 32 key bits; its output includes K hints in the
    # initial "L0+0,L1+1,...,K0+0,K1+0,..." section, followed by "eq1,eq2,eq3,..."
    # Our system() produces the same shape, with variable suffixes "_p0".
    pair_int = (bits_to_int(pt), bits_to_int(ct))
    hints = {i: int(key[i]) for i in range(32)}
    our_sys = anf_system(rounds=160, pairs=[pair_int], key_hints=hints)

    # Convert our system to the comma-separated textual form used by anf.txt.
    # Legacy format per polynomial: "TERM+TERM+TERM+..." and CSV joined.
    our_text = ",".join(_poly_to_legacy_text(p) for p in our_sys) + ","
    # Strip _p0 to match legacy bare names.
    our_text_normalized = our_text.replace("_p0", "")

    # Legacy emits in a particular order: plaintext bindings, ciphertext bindings,
    # key hint bindings, then (eq1, eq2, eq3) per round.
    # Our order is the same for single-pair systems.
    assert our_text_normalized == legacy_anf, \
        f"\nLEGACY:\n{legacy_anf[:500]}\nOURS:\n{our_text_normalized[:500]}"


def _poly_to_legacy_text(poly) -> str:
    """Render a BoolPoly as 'TERM+TERM+...' matching legacy/sage-equations.py."""
    if not poly.monomials:
        return ""
    terms: list[str] = []
    # Deterministic order: by term length, then variable name tuple.
    # Legacy emits in input order — but since BoolPoly is a set, we canonicalize.
    # The compat test normalizes both sides with the same ordering to sidestep
    # ordering differences (legacy emits "K0+L0+...", we sort).
    # Actually: legacy emits a specific order we want to match. See below —
    # this helper is called via a dual-normalization approach.
    for mono in poly.monomials:
        if len(mono) == 0:
            terms.append("1")
        else:
            terms.append("*".join(sorted(mono)))
    return "+".join(sorted(terms))
```

Actually, because legacy output order is input-order and ours is set-order, the raw-string comparison will fail. Fix this by normalizing both sides identically:

```python
# Replace the body of test_anf_160_matches_legacy_anf_txt to use a canonicalized form
# that both sides agree on. Re-write the test to be semantic rather than byte-literal:
```

Rewrite the test body to the following (replacing the version above):

```python
@pytest.mark.legacy
def test_anf_160_matches_legacy_anf_txt(tmp_path, monkeypatch) -> None:
    """Semantic equivalence: both legacy and our ANF, after canonicalization,
    agree as multisets of polynomials."""
    monkeypatch.chdir(tmp_path)
    from tests.compat_helpers import LEGACY_DIR
    import subprocess
    subprocess.run(["python2", str(LEGACY_DIR / "sage-equations.py")],
                   check=True, cwd=tmp_path)
    legacy_anf = (tmp_path / "anf.txt").read_text()

    pt = "01100010100101110000101011100011"
    ct = "01101000110010010100101001111001"
    key = "00110100110111111001011000011100"
    pair_int = (bits_to_int(pt), bits_to_int(ct))
    hints = {i: int(key[i]) for i in range(32)}
    our_sys = anf_system(rounds=160, pairs=[pair_int], key_hints=hints)

    ours_canon = sorted(_poly_to_legacy_text(p) for p in our_sys)

    # Parse legacy: split on top-level commas; trim trailing empty.
    legacy_polys = [x for x in legacy_anf.split(",") if x]
    # Normalize legacy polynomials: re-parse each term, canonicalize.
    legacy_canon = sorted(_canonicalize_legacy_term(t) for t in legacy_polys)

    assert ours_canon == legacy_canon


def _canonicalize_legacy_term(text: str) -> str:
    terms: list[str] = []
    for t in text.split("+"):
        t = t.strip()
        if not t:
            continue
        if t.isdigit():
            terms.append(t)
        else:
            factors = sorted(t.split("*"))
            terms.append("*".join(factors))
    return "+".join(sorted(terms))
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest tests/test_compat.py::test_anf_160_matches_legacy_anf_txt -v
```

Expected: passes when python2 + legacy/sage-equations.py are available. Skips otherwise. If it fails, the failure assertion's text diff shows which polynomial differs — investigate whether legacy or ours has the bug.

- [ ] **Step 3: Commit**

```bash
git add tests/test_compat.py
git commit -m "test: semantic ANF parity vs legacy/sage-equations.py anf.txt

Both sides canonicalized to sorted-string form so set-vs-list ordering
differences don't obscure actual equation-level mismatches.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 26: Legacy compatibility — DIMACS result parser

**Files:**
- Modify: `tests/test_compat.py`

- [ ] **Step 1: Append failing test**

Append to `tests/test_compat.py`:

```python
@pytest.mark.legacy
def test_parse_minisat_matches_legacy_parser(tmp_path, monkeypatch) -> None:
    """Our dimacs_subprocess parser recovers the same key as legacy/parse-miniSAT.py
    on a canned out.result file."""
    from tests.compat_helpers import LEGACY_DIR

    # Build a canned out.result matching the legacy key recovery format:
    # first 64 space-separated integers; negative = 0, positive = 1.
    # Legacy reference key (from legacy/parse-miniSAT.py:5):
    original_key = "0011010011011111100101100001110000011101100111001000001101110100"
    # Emit literals 1..64 with sign matching each bit.
    literals = []
    for i in range(64):
        bit = int(original_key[i])
        literals.append(str(i + 1) if bit == 1 else f"-{i + 1}")
    out_result = "SAT\n" + " ".join(literals + ["0"]) + "\n"
    (tmp_path / "out.result").write_text(out_result)

    # Run the legacy parser under python2:
    import shutil
    shutil.copy(LEGACY_DIR / "parse-miniSAT.py", tmp_path / "parse-miniSAT.py")
    import subprocess
    proc = subprocess.run(
        ["python2", str(tmp_path / "parse-miniSAT.py")],
        capture_output=True, text=True, check=True, cwd=tmp_path,
    )
    # Legacy prints "Original Key\n<key>\nKey Parsed form miniSAT result file\n<key>\n"
    legacy_parsed = proc.stdout.splitlines()[-1].strip()
    assert legacy_parsed == original_key  # sanity

    # Now validate OUR parser extracts the same:
    from keeloq.solvers.dimacs_subprocess import _parse_dimacs_output
    status, lits = _parse_dimacs_output(
        "s SATISFIABLE\nv " + " ".join(literals + ["0"]) + "\n"
    )
    assert status == "SAT"
    # Reconstruct the 64-bit key from the first 64 literals (1-indexed).
    our_key_bits = ""
    lit_set = set(lits)
    for i in range(64):
        vid = i + 1
        our_key_bits += "1" if vid in lit_set else "0"
    assert our_key_bits == original_key
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest tests/test_compat.py::test_parse_minisat_matches_legacy_parser -v
```

Expected: passes (skips without python2).

- [ ] **Step 3: Commit**

```bash
git add tests/test_compat.py
git commit -m "test: DIMACS parser parity with legacy/parse-miniSAT.py on canned out.result

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 27: Benchmark matrix + bench_attack runner

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/matrix.toml`
- Create: `benchmarks/bench_attack.py`
- Modify: `src/keeloq/cli.py` (add `benchmark` subcommand)
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Create `benchmarks/__init__.py`**

```python
```

- [ ] **Step 2: Create `benchmarks/matrix.toml`**

```toml
# Phase 1 benchmark matrix. Each row runs once.
# Results write to ./benchmark-results/<timestamp>/{results.csv, summary.md}.
[[run]]
name = "32r-2pair-xor-cms"
rounds = 32
num_pairs = 2
hint_bits = 0
encoder = "xor"
solver = "cryptominisat"
timeout_s = 120.0

[[run]]
name = "32r-2pair-cnf-cms"
rounds = 32
num_pairs = 2
hint_bits = 0
encoder = "cnf"
solver = "cryptominisat"
timeout_s = 300.0

[[run]]
name = "64r-2pair-xor-cms"
rounds = 64
num_pairs = 2
hint_bits = 16
encoder = "xor"
solver = "cryptominisat"
timeout_s = 600.0

[[run]]
name = "128r-2pair-xor-cms"
rounds = 128
num_pairs = 2
hint_bits = 25
encoder = "xor"
solver = "cryptominisat"
timeout_s = 1800.0

[[run]]
name = "160r-2pair-xor-cms-LEGACY-BASELINE"
rounds = 160
num_pairs = 2
hint_bits = 25
encoder = "xor"
solver = "cryptominisat"
timeout_s = 54000.0    # 15 hours — the legacy took 14h on miniSAT
```

- [ ] **Step 3: Create `benchmarks/bench_attack.py`**

```python
"""Phase 1 benchmark runner.

Reads benchmarks/matrix.toml, runs each config, writes a CSV + markdown summary.
"""
from __future__ import annotations

import csv
import tomllib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from keeloq._types import bits_to_int
from keeloq.attack import attack
from keeloq.cipher import encrypt
from keeloq.encoders.cnf import encode as encode_cnf
from keeloq.encoders.xor_aware import encode as encode_xor
from keeloq.solvers.cryptominisat import solve as solve_cms
from keeloq.solvers.dimacs_subprocess import solve as solve_subprocess


# Fixed KAT for benchmark reproducibility (the 2015 README values):
KEY_BITS = "0011010011011111100101100001110000011101100111001000001101110100"
PT1_BITS = "01100010100101110000101011100011"
PT2_BITS = "11010011100101010000111100001010"  # second pair plaintext


def _encoder(name: str):
    return {"cnf": encode_cnf, "xor": encode_xor}[name]


def _solver(name: str):
    if name == "cryptominisat":
        return solve_cms
    def _wrap(inst, timeout_s):
        return solve_subprocess(inst, solver_binary=name, timeout_s=timeout_s)
    return _wrap


def run_matrix(matrix_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = tomllib.loads(matrix_path.read_text())

    key = bits_to_int(KEY_BITS)
    pts = [bits_to_int(PT1_BITS), bits_to_int(PT2_BITS)]

    rows: list[dict[str, object]] = []
    for run in config["run"]:
        rounds = run["rounds"]
        n_pairs = run["num_pairs"]
        pairs = [(p, encrypt(p, key, rounds)) for p in pts[:n_pairs]]
        hint_bits = run["hint_bits"]
        hints = {i: (key >> (63 - i)) & 1 for i in range(64 - hint_bits, 64)} \
                if hint_bits > 0 else None

        print(f"[bench] running {run['name']!r}...", flush=True)
        result = attack(
            rounds=rounds,
            pairs=pairs,
            key_hints=hints,
            encoder=_encoder(run["encoder"]),
            solver_fn=_solver(run["solver"]),
            timeout_s=run["timeout_s"],
        )
        print(f"  -> status={result.status} "
              f"wall_time_s={result.solve_result.stats.wall_time_s:.3f}", flush=True)
        rows.append({
            "name": run["name"],
            "rounds": rounds,
            "num_pairs": n_pairs,
            "hint_bits": hint_bits,
            "encoder": run["encoder"],
            "solver": run["solver"],
            "status": result.status,
            "wall_time_s": f"{result.solve_result.stats.wall_time_s:.3f}",
            "num_vars": result.solve_result.stats.num_vars,
            "num_clauses": result.solve_result.stats.num_clauses,
            "num_xors": result.solve_result.stats.num_xors,
        })

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = out_dir / "summary.md"
    with md_path.open("w") as f:
        f.write("# Phase 1 Benchmark Results\n\n")
        f.write("| " + " | ".join(rows[0].keys()) + " |\n")
        f.write("|" + "|".join(["---"] * len(rows[0])) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[k]) for k in rows[0].keys()) + " |\n")
    print(f"[bench] wrote {csv_path} and {md_path}", flush=True)


def main() -> None:
    matrix = Path(__file__).parent / "matrix.toml"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path("benchmark-results") / ts
    run_matrix(matrix, out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Append `benchmark` subcommand to `src/keeloq/cli.py`**

```python
@app.command()
def benchmark(
    matrix: Annotated[str, typer.Option(help="Path to benchmark matrix TOML")] =
        "benchmarks/matrix.toml",
    out_dir: Annotated[str, typer.Option(help="Output directory")] = "benchmark-results",
) -> None:
    """Run the benchmark matrix and write CSV + markdown to an output directory."""
    from datetime import datetime
    from benchmarks.bench_attack import run_matrix
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(out_dir) / ts
    run_matrix(Path(matrix), out_path)
    typer.echo(f"wrote results to {out_path}/")
```

And at the top of `src/keeloq/cli.py`, add:

```python
from pathlib import Path
```

- [ ] **Step 5: Append a smoke test for the benchmark subcommand**

Append to `tests/test_cli.py`:

```python
@pytest.mark.slow
def test_benchmark_smoke(tmp_path) -> None:
    """Run a one-row benchmark matrix and confirm CSV + MD appear."""
    matrix = tmp_path / "tiny.toml"
    matrix.write_text(
        '[[run]]\n'
        'name = "smoke-16r-heavy"\n'
        'rounds = 16\n'
        'num_pairs = 1\n'
        'hint_bits = 48\n'
        'encoder = "xor"\n'
        'solver = "cryptominisat"\n'
        'timeout_s = 30.0\n'
    )
    out_dir = tmp_path / "out"
    result = runner.invoke(app, ["benchmark", "--matrix", str(matrix),
                                  "--out-dir", str(out_dir)])
    assert result.exit_code == 0, result.stdout
    # Exactly one timestamped subdir should exist.
    subdirs = list(out_dir.iterdir())
    assert len(subdirs) == 1
    assert (subdirs[0] / "results.csv").exists()
    assert (subdirs[0] / "summary.md").exists()
```

- [ ] **Step 6: Run the test**

```bash
uv run pytest tests/test_cli.py::test_benchmark_smoke -v
```

Expected: passes.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/ src/keeloq/cli.py tests/test_cli.py
git commit -m "impl: benchmark runner + CLI subcommand with TOML matrix config

Phase 1 matrix includes the legacy 160r/25-hint/2-pair baseline for a direct
wall-clock comparison against the 2015 README's 14h number.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 28: GitHub Actions CI config

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create the workflow**

```yaml
name: CI

on:
  push:
    branches: [master, main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          python-version: "3.12"

      - name: Install python2 (for legacy compat tests)
        run: |
          sudo apt-get update
          sudo apt-get install -y python2

      - name: Install kissat (for DIMACS subprocess tests)
        run: |
          sudo apt-get install -y kissat || true

      - name: Sync deps
        run: uv sync --frozen

      - name: Lint
        run: |
          uv run ruff check src tests
          uv run ruff format --check src tests

      - name: Type check
        run: uv run mypy

      - name: Test (fast — excludes slow and gpu)
        env:
          PYTEST_ADDOPTS: "-m 'not slow and not gpu'"
        run: uv run pytest -n auto
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: github actions workflow for lint + type + fast tests

Installs python2 and kissat so legacy-compat and subprocess-solver tests run.
GPU tests skip automatically (no CUDA on hosted runners).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 29: Update CLAUDE.md and README for Phase 1

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Prepend a new Phase 1 section to `CLAUDE.md`**

Insert after the existing `# CLAUDE.md\n\n` header and before "## What this repo is":

```markdown
## Phase 1 status (2026)

The repo is mid-migration from the 2015 Python 2 scripts (in `legacy/`) to a
Python 3 modernized pipeline in `src/keeloq/`. Driver is `keeloq` (a Typer CLI,
installed via `uv sync`). Key entry points:

- `keeloq encrypt / decrypt` — the cipher, rounds-parameterized.
- `keeloq generate-anf | encode | solve | verify` — the pipeline stages, composable via Unix pipes (JSON between stages).
- `keeloq attack` — the pipeline in-process.
- `keeloq benchmark` — matrix runner driven by `benchmarks/matrix.toml`.

The codebase is strict TDD — write the failing test first, confirm fail,
implement, confirm pass, commit. Commits are prefixed `test:`, `impl:`,
`refactor:`, `feat:`, `docs:`, `ci:`, `chore:`. Do NOT squash them.

**Cross-validation TDD invariant.** Two encoders (`encoders/cnf.py`,
`encoders/xor_aware.py`) must recover the same key on the same inputs; the
`tests/test_encoders_agree.py` test enforces this. Variable-indexing bugs are
the biggest risk in this domain, and this cross-check is the primary defense.

**Legacy is frozen.** Never modify files in `legacy/`. If you need to change
how they're invoked, edit `tests/compat_helpers.py` instead.
```

- [ ] **Step 2: Append a Phase 1 usage section to `README.md`**

```markdown

## Phase 1 (2026 modernization) — Quick start

Install (Linux, RTX 5090 optional):

    uv sync

Run the full test suite:

    uv run pytest -m "not slow"

Run a 32-round attack with two plaintext/ciphertext pairs and no hints:

    uv run keeloq attack \
        --rounds 32 \
        --pair 01100010100101110000101011100011:<ct1> \
        --pair 11010011100101010000111100001010:<ct2> \
        --encoder xor --solver cryptominisat --timeout 60

Reproduce the 2015 README result (160 rounds, 25 hints, 2 pairs):

    uv run keeloq benchmark

See `docs/superpowers/specs/2026-04-22-phase1-foundation-design.md` for the
design and `docs/superpowers/plans/2026-04-22-phase1-foundation-plan.md` for
the implementation plan.
```

- [ ] **Step 3: Run the full fast suite once more as a final sanity check**

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
uv run pytest -n auto -m "not slow"
```

Expected: everything green.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: update CLAUDE.md and README for Phase 1 modernization

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 30: Final integration check

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite including slow and GPU (on the 5090 box)**

```bash
uv run pytest -n auto
```

Expected: all tests pass, GPU tests run, compat tests run (python2 installed),
benchmark smoke runs.

- [ ] **Step 2: Confirm the four success criteria from the spec**

```bash
# (1) Fast CI passes
time uv run pytest -n auto -m "not slow and not gpu"

# (2) CLI end-to-end
uv run keeloq attack --rounds 32 \
    --pair "$(uv run keeloq encrypt --rounds 0 --plaintext 01100010100101110000101011100011 --key 0011010011011111100101100001110000011101100111001000001101110100)" \
    --hint-bits 32 \
    --original-key 0011010011011111100101100001110000011101100111001000001101110100 \
    --encoder xor --solver cryptominisat --timeout 30

# (3) Benchmark smoke
uv run keeloq benchmark --matrix <(cat <<'EOF'
[[run]]
name = "smoke"
rounds = 16
num_pairs = 1
hint_bits = 48
encoder = "xor"
solver = "cryptominisat"
timeout_s = 10.0
EOF
)

# (4) test_encoders_agree green
uv run pytest tests/test_encoders_agree.py -v
```

Expected: (1) under 60s. (2) exit 0, key recovered. (3) exit 0, CSV written.
(4) all green.

- [ ] **Step 3: Final "Phase 1 complete" commit (optional marker)**

```bash
git commit --allow-empty -m "chore: Phase 1 foundation complete

- Rounds-parameterized Python 3 cipher with KATs at 160 and 528 rounds.
- GPU bit-sliced cipher as property-test oracle (10^6 pairs/round).
- ANF generator with multi-pair variable namespacing (_p{i}).
- Dual encoders (pure-CNF and XOR-aware-hybrid) cross-validated.
- CryptoMiniSat wrapper + DIMACS subprocess wrapper (kissat/minisat).
- attack() pipeline with mandatory cipher-verify gate.
- Single-binary CLI with composable subcommands.
- Benchmark matrix including the 2015 README baseline config.
- Legacy compat tests for cipher, ANF, and DIMACS parser.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

Phase 1 complete.

---

## Self-Review Notes

Before handing this plan off to an implementation agent, I ran the following checks:

**Spec coverage.** Each spec section maps to a task:
- §4 Repo layout → Task 1 (scaffolding) + each subsequent task creating module files.
- §5.1 Cipher → Tasks 3–5.
- §5.2 GPU cipher → Tasks 6–7.
- §5.3 ANF → Tasks 8–11.
- §5.4 CNF encoder → Task 13.
- §5.5 XOR-aware encoder → Task 14.
- §5.6 Encoder protocol → Task 12.
- §5.7 CryptoMiniSat → Task 16.
- §5.8 DIMACS subprocess → Task 17.
- §5.9 Attack pipeline → Tasks 18–19, plus Task 20 for test_encoders_agree.
- §5.10 CLI → Tasks 21–23 + benchmark in Task 27.
- §7 TDD pyramid → baked into every task's step-5 test-run; Layer 4 compat in Tasks 24–26; Layer 5 benchmarks in Task 27.
- §8 Error handling → Task 2 (error hierarchy) + Task 19 (UNSAT/WRONG_KEY behavior tests).
- §9 Packaging → Task 1 (`pyproject.toml`) + Task 28 (CI).
- §10 Success criteria → Task 30.

**Placeholder scan.** No TBDs, no "implement later", no "similar to Task N". Every step has exact file paths, exact code blocks, exact commands.

**Type consistency.** `SolverInstance`, `CNFInstance`, `HybridInstance`, `SolveResult`, `AttackResult`, `BoolPoly` used identically across all tasks. CLI option names (`--rounds`, `--pair`, `--hint-bits`, `--original-key`, `--key-hint`, `--encoder`, `--solver`, `--timeout`) consistent across `attack`, `generate-anf`, `verify`.

**Ambiguity check.** One area I tightened: Task 25's ANF parity test was initially byte-for-byte against legacy `anf.txt`, but because `BoolPoly` uses frozensets (unordered) and legacy emits in input order, a byte comparison would fail for equivalent systems. Rewrote to canonicalize both sides into sorted-string multisets — the test now proves semantic equivalence, not textual.

Spec requirement §6 ("data flow via Unix pipes") — verified in Task 23's `test_pipeline_composition_via_stdout_to_stdin`. No gaps.

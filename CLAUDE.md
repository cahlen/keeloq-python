# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

# keeloq-python

Algebraic + neural cryptanalysis of the [KeeLoq](https://en.wikipedia.org/wiki/KeeLoq) block cipher — a 2026 modernization of the 2015 SAT-based attack originally in this repo.

## What's here

Two independent cryptanalysis pipelines against reduced-round KeeLoq, both drivable from a single `keeloq` CLI:

1. **Algebraic / SAT** — Python 3, CryptoMiniSat with native XOR clauses, both pure-CNF and XOR-aware ANF encoders. Recovers a 64-bit key at **64 rounds / 0 hints / 4 plaintext-ciphertext pairs in 0.247 s** on an RTX 5090. The original 2015 pipeline took 14 hours for a harder variant (160 rounds, 25 key-bit hints, 2 pairs).

2. **Neural differential (Gohr 2019 style)** — PyTorch ResNet-1D-CNN distinguisher trained on a GPU bit-sliced KeeLoq implementation (~10⁶ pairs/sec on a 5090), Bayesian beam-search key recovery, SAT-suffix handoff for the bits the distinguisher doesn't resolve. Unique angle: KeeLoq's 1-bit-per-round key schedule lets the Bayesian search guess one bit at a time rather than a 16-bit subkey chunk as in Gohr's original SPECK attack.

The 2015 pipeline (SageMath + miniSAT + hand-edited polynomial systems) is frozen in `legacy/` and verified against the modern pipeline via Docker-hosted `python:2.7` parity tests.

## Requirements

- Linux x86_64, Python 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- **Optional**: CUDA 12.8+ GPU (RTX 5090 or similar) — required for the neural pipeline and for property tests against the GPU bit-sliced cipher; the algebraic attack runs fine on CPU.
- **Optional**: Docker — runs the legacy 2015 Python 2 scripts inside an ephemeral `python:2.7` container for parity tests.

## Install

    uv sync --all-extras

## Quickstart — algebraic attack

Recover a 64-bit key at 64 rounds, 4 pairs, zero hints:

    KEY=0011010011011111100101100001110000011101100111001000001101110100
    PT1=01100010100101110000101011100011
    PT2=11010011100101010000111100001010
    CT1=$(uv run keeloq encrypt --rounds 64 --plaintext $PT1 --key $KEY)
    CT2=$(uv run keeloq encrypt --rounds 64 --plaintext $PT2 --key $KEY)
    uv run keeloq attack --rounds 64 \
        --pair "$PT1:$CT1" --pair "$PT2:$CT2" \
        --encoder xor --solver cryptominisat --timeout 120

Typical wall clock: **< 1 second**.

Reproduce the 2015 README baseline (160 rounds, 25 hints, 2 pairs, legacy comparison):

    uv run keeloq benchmark

## Quickstart — neural cryptanalysis

End-to-end, one command (Δ search + train + attack a synthetic target):

    uv run keeloq neural auto --rounds 64 --trained-depth 56 \
        --samples 10000000 --pairs 512 \
        --checkpoint-out checkpoints/d64.pt

~60 minutes training time on an RTX 5090. Or skip training and pull a published checkpoint from Hugging Face:

    mkdir -p checkpoints
    hf download cahlen/keeloq-neural-distinguishers d64.pt --local-dir checkpoints/

Then attack any new ciphertext under the same key:

    uv run keeloq neural recover-key --checkpoint checkpoints/d64.pt \
        --rounds 64 --diff-pair "<c0>:<c1>" --sat-pair "<pt>:<ct>" \
        --beam-width 16 --sat-timeout 120

### How it works — the Gohr pattern on KeeLoq

A ResNet-1D-CNN distinguisher is trained at a fixed depth **D** (e.g. 56) to separate chosen-input-difference ciphertext pairs from random pairs. An **M = D + K**-round attack peels `K` rounds off via Bayesian beam search (`recover_prefix`), recovering the outer `K` key bits. The remaining suffix bits are handed to the algebraic pipeline's XOR-aware encoder + CryptoMiniSat. Every recovered key is cipher-verified (`cipher.encrypt(pt, k, rounds) == ct`) before reporting `SUCCESS`.

**Two pair streams.** `--diff-pair` carries differential ciphertext pairs `(c₀, c₁)` — both ciphertexts, consumed by the neural distinguisher. `--sat-pair` carries known `plaintext:ciphertext` — consumed by the SAT solver. These are distinct because a differential attack doesn't need known plaintexts; only the SAT phase does.

**Key-schedule constraint.** KeeLoq's key cycles every 64 rounds; at fewer than 64 rounds, bits `K_rounds..K_63` are never referenced and can't be recovered without being hinted. Attacks below 64 rounds therefore auto-populate `extra_key_hints` for the unconstrained range — handled automatically by `keeloq neural auto`.

### Pre-trained distinguishers on Hugging Face

Checkpoints are published at [**cahlen/keeloq-neural-distinguishers**](https://huggingface.co/cahlen/keeloq-neural-distinguishers) with a full model card covering training config, eval metrics, architecture, and attack procedure. Current availability:

| File | Trained Depth | Attack Target | Val Accuracy | ROC-AUC | Status |
|---|---:|---:|---:|---:|---|
| `d64.pt` | 56 | 64 rounds (peel K=8) | 0.752 | 0.828 | ✅ viable; used by the 64-round regression test |

`d64.pt` is the only viable checkpoint — we tested every depth from 57 through 120 and found a **one-round-wide signal cliff between depth 56 and depth 57** (signal drops from 0.69 to 0.54 val-acc in a single round, then stays at noise through depth 120). Two architectures (1×1-conv MLP-style and kernel-3 spatial-conv) collapse identically at depths past 56, indicating the horizon is a property of KeeLoq's diffusion at these rounds, not a neural-architecture artifact. The full diagnostic trail (horizon probe tables, v1-vs-v2 comparison, proposed frontier directions) is in [`docs/phase3b-results/ambition_outcome.md`](docs/phase3b-results/ambition_outcome.md).

Each `.pt` file embeds its full `TrainingConfig`, so results are reproducible from seed alone.

## Pipeline composition via Unix pipes

The algebraic pipeline is also exposed as discrete stages with JSON on stdin/stdout:

    keeloq generate-anf ... | keeloq encode ... | keeloq solve ... | keeloq verify ...

Useful for inspecting intermediate artifacts (ANF polynomial systems, DIMACS CNF, SAT result JSON) or for swapping in alternative encoders/solvers.

## Running the tests

    uv run pytest -n auto -m "not slow"    # fast suite — ~30 s on the 5090 box
    uv run pytest -n auto                   # full suite, including GPU and slow

Test markers:

- `@pytest.mark.gpu` — requires a CUDA GPU; auto-skips on CPU-only machines.
- `@pytest.mark.slow` — multi-second tests; excluded from the default fast suite but runs end-to-end attacks and benchmark smoke tests.
- `@pytest.mark.legacy` — requires Docker + the `python:2.7` image; runs 2015 scripts and verifies parity.

## Project layout

    src/keeloq/          # modern Python 3 pipeline
      cipher.py          #   readable reference cipher (rounds-parameterized)
      gpu_cipher.py      #   bit-sliced CUDA cipher (property-test oracle + training-data generator)
      anf.py             #   ANF polynomial system generator
      encoders/          #   pure-CNF + XOR-aware encoders
      solvers/           #   CryptoMiniSat wrapper + DIMACS subprocess wrapper
      attack.py          #   SAT-only attack pipeline with mandatory cipher-verify
      neural/            #   Phase 3b neural cryptanalysis
        data.py          #     training pair generator (differential + random)
        differences.py   #     Δ candidate search
        distinguisher.py #     ResNet-1D-CNN + training loop + checkpoint I/O
        evaluation.py    #     accuracy / ROC-AUC / TPR@FPR metrics
        key_recovery.py  #     partial_decrypt_round + recover_prefix beam search
        hybrid.py        #     neural-prefix + SAT-suffix orchestration
        cli_neural.py    #     `keeloq neural {train, evaluate, recover-key, auto}`
      cli.py             #   main Typer entry point

    legacy/              # frozen 2015 Python 2 scripts (run via Docker)

    benchmarks/
      matrix.toml        # algebraic benchmark matrix (Phase 1)
      bench_attack.py    #   runner
      neural_matrix.toml # neural-hybrid vs pure-SAT matrix (Phase 3b)
      bench_neural.py    #   runner

    tests/               # pytest suite (unit + property + integration + compat)

    checkpoints/         # trained distinguisher checkpoints (not committed; reproducible)

    docs/
      superpowers/specs/ # design docs per modernization phase
      superpowers/plans/ # task-by-task implementation plans
      phase3b-results/   # Δ search tables, eval reports, benchmark results

## Historical context (2015 origin)

The original 2015 pipeline was the target of this modernization. Frozen untouched in `legacy/`:

- `keeloq-python.py` / `keeloq160-python.py` — KeeLoq reference implementations (528- and 160-round) in Python 2.
- `sage-equations.py` / `polynomial-vars.py` — emit ANF polynomial systems and variable lists for SageMath.
- `sage-CNF-convert.txt` — SageMath driver that converts ANF to DIMACS CNF via PolyBoRi's `CNFEncoder`.
- `parse-miniSAT.py` — recovers the 64-bit key from miniSAT output.

Original 2015 README text (preserved below for historical reference):

> A repository of code that I used to break 160 rounds of the KeeLoq cryptosystem. Everything is written in python with the exception of some sage code which is used to converge a system of ANF boolean polynomial equations into DIMACS CNF form. Once in this form we can feed the output file to miniSAT which determines if the system is satisfiable. If it is you can pipe the output to a file and parse it with the other python file and check to see if the correct key was recovered.
>
> Usually you're going to have to feed miniSAT at least 32 bits of your key as a hint. If you start taking away bits of the key then your system is going to be underdefined (especially if you're only giving it one pair of plaintext/ciphertext). In this case the system is still satisfiable except you'll get the wrong key back since miniSAT stops as soon as it has found the system is satisfiable, and if the system is underdefined it will have multiple solutions.
>
> The quick fix to this is to produce another of plaintext/ciphertext under the same key. Then produce more equations for the new plaintext/ciphertext, and all the intermediate variables. The only thing that should remain the same is the actual key variables. Of course this system takes longer to solve, but you get the correct key back each time. I experimented with this and was able to cut the key bit hints down to 25 bits with two pairs of plaintext/ciphertext — however it took 14 hours to solve this system.

See [`docs/superpowers/specs/`](docs/superpowers/specs/) for the design docs tracking the path from that 2015 state to the current code.

## License

MIT — see [LICENSE](LICENSE).

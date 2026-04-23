"""Phase 3b neural benchmark runner.

Reads benchmarks/neural_matrix.toml, runs each config, writes a CSV + markdown
summary to benchmark-results-neural/<timestamp>/.

kind="neural" rows use hybrid_attack (neural prefix peeling + SAT suffix).
kind="sat"    rows use the Phase 1 pure-SAT attack (no checkpoint needed).

If a neural checkpoint is absent the row is recorded with
status="SKIP_MISSING_CHECKPOINT" and the runner continues — this allows
infrastructure validation (CI) without trained weights.

Reproduce full matrix (requires GPU + trained checkpoints):
  uv run python -m benchmarks.bench_neural
"""

from __future__ import annotations

import csv
import time
import tomllib
from datetime import datetime
from pathlib import Path

from keeloq._types import bits_to_int
from keeloq.attack import EncodeFn, SolveFn
from keeloq.attack import attack as sat_attack
from keeloq.cipher import encrypt
from keeloq.encoders.cnf import encode as encode_cnf
from keeloq.encoders.xor_aware import encode as encode_xor
from keeloq.solvers.cryptominisat import solve as solve_cms
from keeloq.solvers.dimacs_subprocess import solve as solve_subprocess

# ---------------------------------------------------------------------------
# Fixed KAT (same key + plaintexts as bench_attack.py for comparability)
# ---------------------------------------------------------------------------
KEY_BITS = "0011010011011111100101100001110000011101100111001000001101110100"
PT1_BITS = "01100010100101110000101011100011"
PT2_BITS = "11010011100101010000111100001010"
PT3_BITS = "10101010101010101010101010101010"
PT4_BITS = "01010101010101010101010101010101"
PT5_BITS = "00000000000000000000000000000001"
PT6_BITS = "11111111111111111111111111111110"
PT7_BITS = "10110011001100110011001100110011"
PT8_BITS = "01001100110011001100110011001100"

_PTS = [
    PT1_BITS,
    PT2_BITS,
    PT3_BITS,
    PT4_BITS,
    PT5_BITS,
    PT6_BITS,
    PT7_BITS,
    PT8_BITS,
]


def _encoder(name: str) -> EncodeFn:
    return {"cnf": encode_cnf, "xor": encode_xor}[name]  # type: ignore[return-value]


def _solver(name: str) -> SolveFn:
    if name == "cryptominisat":
        return solve_cms

    def _wrap(inst: object, timeout_s: float) -> object:
        return solve_subprocess(inst, solver_binary=name, timeout_s=timeout_s)  # type: ignore[arg-type]

    return _wrap  # type: ignore[return-value]


def _build_pairs(
    pts: list[int],
    key: int,
    rounds: int,
    n_pairs: int,
) -> list[tuple[int, int]]:
    """Build genuine (plaintext, ciphertext) pairs from the fixed KAT key."""
    if n_pairs > len(pts):
        raise ValueError(f"num_pairs={n_pairs} exceeds available fixtures ({len(pts)})")
    return [(p, encrypt(p, key, rounds)) for p in pts[:n_pairs]]


def _build_diff_pairs(
    pts: list[int],
    key: int,
    rounds: int,
    n_pairs: int,
    delta: int,
) -> list[tuple[int, int]]:
    """Build (c0, c1) differential pairs: c0=E(pt,key), c1=E(pt^delta,key)."""
    if n_pairs > len(pts):
        raise ValueError(f"num_pairs={n_pairs} exceeds available fixtures ({len(pts)})")
    result = []
    for pt in pts[:n_pairs]:
        # CUDA uint32 XOR unsupported — apply delta on CPU side.
        pt1 = (pt ^ delta) & 0xFFFFFFFF
        c0 = encrypt(pt, key, rounds)
        c1 = encrypt(pt1, key, rounds)
        result.append((c0, c1))
    return result


def _run_sat(run: dict[str, object], key: int, pts: list[int]) -> dict[str, object]:
    """Execute a pure-SAT benchmark row."""
    rounds = int(run["rounds"])  # type: ignore[arg-type]
    n_pairs = int(run["num_pairs"])  # type: ignore[arg-type]
    hint_bits = int(run.get("hint_bits", 0))  # type: ignore[arg-type]
    timeout_s = float(run.get("timeout_s", 300.0))  # type: ignore[arg-type]

    pairs = _build_pairs(pts, key, rounds, n_pairs)
    hints: dict[int, int] | None = (
        {i: (key >> (63 - i)) & 1 for i in range(64 - hint_bits, 64)} if hint_bits > 0 else None
    )

    t0 = time.perf_counter()
    result = sat_attack(
        rounds=rounds,
        pairs=pairs,
        key_hints=hints,
        encoder=_encoder(str(run.get("encoder", "xor"))),
        solver_fn=_solver(str(run.get("solver", "cryptominisat"))),
        timeout_s=timeout_s,
    )
    wall = time.perf_counter() - t0

    return {
        "name": run["name"],
        "kind": "sat",
        "rounds": rounds,
        "num_pairs": n_pairs,
        "checkpoint": "",
        "status": result.status,
        "wall_time_s": f"{wall:.3f}",
        "bits_recovered_neurally": 0,
        "neural_wall_time_s": "0.000",
        "sat_wall_time_s": f"{result.solve_result.stats.wall_time_s:.3f}",
        "num_vars": result.solve_result.stats.num_vars,
        "num_clauses": result.solve_result.stats.num_clauses,
        "num_xors": result.solve_result.stats.num_xors,
    }


def _run_neural(run: dict[str, object], key: int, pts: list[int]) -> dict[str, object]:
    """Execute a neural-hybrid benchmark row.

    Deviations from plan (per task brief):
    - Uses two-arg hybrid_attack: pairs (differential c0:c1) + sat_pairs (pt:ct).
    - Applies extra_key_hints for cyclic-schedule bits when rounds < 64.
    - If the checkpoint doesn't exist, records SKIP_MISSING_CHECKPOINT instead of crashing.
    """
    from keeloq.neural.distinguisher import load_checkpoint
    from keeloq.neural.hybrid import hybrid_attack

    rounds = int(run["rounds"])  # type: ignore[arg-type]
    n_pairs = int(run["num_pairs"])  # type: ignore[arg-type]
    ckpt_path = Path(str(run["checkpoint"]))
    beam_width = int(run.get("beam_width", 8))  # type: ignore[arg-type]
    neural_target_bits = run.get("neural_target_bits")
    neural_target_bits_int: int | None = (
        int(neural_target_bits) if neural_target_bits is not None else None  # type: ignore[arg-type]
    )
    sat_timeout_s = float(run.get("sat_timeout_s", 60.0))  # type: ignore[arg-type]
    max_backtracks = int(run.get("max_backtracks", 8))  # type: ignore[arg-type]

    # Smoke-safe path: skip gracefully when checkpoint is absent.
    if not ckpt_path.exists():
        print(f"  [SKIP] checkpoint not found: {ckpt_path}", flush=True)
        return {
            "name": run["name"],
            "kind": "neural",
            "rounds": rounds,
            "num_pairs": n_pairs,
            "checkpoint": str(ckpt_path),
            "status": "SKIP_MISSING_CHECKPOINT",
            "wall_time_s": "0.000",
            "bits_recovered_neurally": 0,
            "neural_wall_time_s": "0.000",
            "sat_wall_time_s": "0.000",
            "num_vars": 0,
            "num_clauses": 0,
            "num_xors": 0,
        }

    # Load distinguisher checkpoint to discover the trained delta.
    model, train_result = load_checkpoint(ckpt_path)
    delta = train_result.config.delta

    # Build differential pairs (c0, c1) for the neural distinguisher.
    diff_pairs = _build_diff_pairs(pts, key, rounds, n_pairs, delta)

    # Build genuine (plaintext, ciphertext) pairs for SAT.
    sat_pairs = _build_pairs(pts, key, rounds, n_pairs)

    # Apply extra_key_hints for cyclic-schedule bits when rounds < 64.
    extra_hints: dict[int, int] | None = None
    if rounds < 64:
        extra_hints = {i: (key >> (63 - i)) & 1 for i in range(rounds, 64)}

    t0 = time.perf_counter()
    result = hybrid_attack(
        rounds=rounds,
        pairs=diff_pairs,
        sat_pairs=sat_pairs,
        distinguisher=model,
        beam_width=beam_width,
        neural_target_bits=neural_target_bits_int,
        sat_timeout_s=sat_timeout_s,
        max_backtracks=max_backtracks,
        extra_key_hints=extra_hints,
    )
    wall = time.perf_counter() - t0

    return {
        "name": run["name"],
        "kind": "neural",
        "rounds": rounds,
        "num_pairs": n_pairs,
        "checkpoint": str(ckpt_path),
        "status": result.status,
        "wall_time_s": f"{wall:.3f}",
        "bits_recovered_neurally": result.bits_recovered_neurally,
        "neural_wall_time_s": f"{result.neural_wall_time_s:.3f}",
        "sat_wall_time_s": f"{result.sat_wall_time_s:.3f}",
        "num_vars": 0,
        "num_clauses": 0,
        "num_xors": 0,
    }


_FIELDS = [
    "name",
    "kind",
    "rounds",
    "num_pairs",
    "checkpoint",
    "status",
    "wall_time_s",
    "bits_recovered_neurally",
    "neural_wall_time_s",
    "sat_wall_time_s",
    "num_vars",
    "num_clauses",
    "num_xors",
]


def run_matrix(matrix_path: Path, out_dir: Path) -> Path:
    """Run every row in the neural matrix, write CSV + markdown. Returns out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    config = tomllib.loads(matrix_path.read_text())

    key = bits_to_int(KEY_BITS)
    pts = [bits_to_int(p) for p in _PTS]

    rows: list[dict[str, object]] = []
    for run in config["run"]:
        kind = str(run.get("kind", "sat"))
        print(f"[bench_neural] running {run['name']!r} (kind={kind})...", flush=True)

        try:
            row = _run_neural(run, key, pts) if kind == "neural" else _run_sat(run, key, pts)
        except Exception as exc:
            print(f"  [ERROR] {exc}", flush=True)
            row = {
                "name": run["name"],
                "kind": kind,
                "rounds": run.get("rounds", 0),
                "num_pairs": run.get("num_pairs", 0),
                "checkpoint": str(run.get("checkpoint", "")),
                "status": f"CRASH: {exc}",
                "wall_time_s": "0.000",
                "bits_recovered_neurally": 0,
                "neural_wall_time_s": "0.000",
                "sat_wall_time_s": "0.000",
                "num_vars": 0,
                "num_clauses": 0,
                "num_xors": 0,
            }

        print(f"  -> status={row['status']} wall_time_s={row['wall_time_s']}", flush=True)
        rows.append(row)

    # Write CSV
    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)

    # Write markdown summary
    md_path = out_dir / "summary.md"
    with md_path.open("w") as f:
        f.write("# Phase 3b Neural Benchmark Results\n\n")
        f.write(
            "> Full matrix requires trained checkpoints. "
            "Reproduce: `uv run python -m benchmarks.bench_neural`\n\n"
        )
        f.write("| " + " | ".join(_FIELDS) + " |\n")
        f.write("|" + "|".join(["---"] * len(_FIELDS)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[k]) for k in _FIELDS) + " |\n")

    print(f"[bench_neural] wrote {csv_path} and {md_path}", flush=True)
    return out_dir


def main() -> None:
    matrix = Path(__file__).parent / "neural_matrix.toml"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path("benchmark-results-neural") / ts
    run_matrix(matrix, out)


if __name__ == "__main__":
    main()

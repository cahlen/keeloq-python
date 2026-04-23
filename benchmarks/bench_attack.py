"""Phase 1 benchmark runner.

Reads benchmarks/matrix.toml, runs each config, writes a CSV + markdown summary.
"""

from __future__ import annotations

import csv
import tomllib
from datetime import datetime
from pathlib import Path

from keeloq._types import bits_to_int
from keeloq.attack import EncodeFn, SolveFn, attack
from keeloq.cipher import encrypt
from keeloq.encoders.cnf import encode as encode_cnf
from keeloq.encoders.xor_aware import encode as encode_xor
from keeloq.solvers.cryptominisat import solve as solve_cms
from keeloq.solvers.dimacs_subprocess import solve as solve_subprocess

# Fixed KAT for benchmark reproducibility (the 2015 README values plus a second pair):
KEY_BITS = "0011010011011111100101100001110000011101100111001000001101110100"
PT1_BITS = "01100010100101110000101011100011"
PT2_BITS = "11010011100101010000111100001010"
PT3_BITS = "10101010101010101010101010101010"
PT4_BITS = "01010101010101010101010101010101"

_PTS = [PT1_BITS, PT2_BITS, PT3_BITS, PT4_BITS]


def _encoder(name: str) -> EncodeFn:
    return {"cnf": encode_cnf, "xor": encode_xor}[name]  # type: ignore[return-value]


def _solver(name: str) -> SolveFn:
    if name == "cryptominisat":
        return solve_cms

    def _wrap(inst: object, timeout_s: float) -> object:
        return solve_subprocess(inst, solver_binary=name, timeout_s=timeout_s)  # type: ignore[arg-type]

    return _wrap  # type: ignore[return-value]


def run_matrix(matrix_path: Path, out_dir: Path) -> Path:
    """Run every row in the matrix, write CSV + markdown. Returns the timestamped dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    config = tomllib.loads(matrix_path.read_text())

    key = bits_to_int(KEY_BITS)
    pts = [bits_to_int(p) for p in _PTS]

    rows: list[dict[str, object]] = []
    for run in config["run"]:
        rounds = run["rounds"]
        n_pairs = run["num_pairs"]
        if n_pairs > len(pts):
            raise ValueError(f"num_pairs={n_pairs} exceeds available fixtures ({len(pts)})")
        pairs = [(p, encrypt(p, key, rounds)) for p in pts[:n_pairs]]
        hint_bits = run["hint_bits"]
        hints = (
            {i: (key >> (63 - i)) & 1 for i in range(64 - hint_bits, 64)} if hint_bits > 0 else None
        )

        print(f"[bench] running {run['name']!r}...", flush=True)
        result = attack(
            rounds=rounds,
            pairs=pairs,
            key_hints=hints,
            encoder=_encoder(run["encoder"]),
            solver_fn=_solver(run["solver"]),
            timeout_s=run["timeout_s"],
        )
        print(
            f"  -> status={result.status} wall_time_s={result.solve_result.stats.wall_time_s:.3f}",
            flush=True,
        )
        rows.append(
            {
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
            }
        )

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = out_dir / "summary.md"
    with md_path.open("w") as f:
        f.write("# Phase 1 Benchmark Results\n\n")
        f.write("| " + " | ".join(rows[0]) + " |\n")
        f.write("|" + "|".join(["---"] * len(rows[0])) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[k]) for k in rows[0]) + " |\n")
    print(f"[bench] wrote {csv_path} and {md_path}", flush=True)
    return out_dir


def main() -> None:
    matrix = Path(__file__).parent / "matrix.toml"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path("benchmark-results") / ts
    run_matrix(matrix, out)


if __name__ == "__main__":
    main()

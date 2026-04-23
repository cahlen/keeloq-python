"""keeloq command-line interface."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from keeloq._types import bits_to_int, int_to_bits
from keeloq.anf import BoolPoly
from keeloq.anf import system as _anf_system
from keeloq.attack import EncodeFn, SolveFn
from keeloq.attack import attack as _attack
from keeloq.cipher import decrypt as _decrypt
from keeloq.cipher import encrypt as _encrypt
from keeloq.encoders import CNFInstance, HybridInstance
from keeloq.encoders.cnf import encode as _encode_cnf
from keeloq.encoders.xor_aware import encode as _encode_xor
from keeloq.solvers import SolveResult
from keeloq.solvers.cryptominisat import solve as _solve_cms
from keeloq.solvers.dimacs_subprocess import solve as _solve_subprocess

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


def _resolve_encoder(name: str) -> EncodeFn:
    if name == "cnf":
        return _encode_cnf
    if name == "xor":
        return _encode_xor
    raise typer.BadParameter(f"unknown encoder {name!r}; expected cnf or xor")


def _resolve_solver(name: str) -> SolveFn:
    if name == "cryptominisat":
        return _solve_cms
    if name in ("kissat", "minisat"):
        binary = name

        def _wrap(inst: object, timeout_s: float) -> SolveResult:
            return _solve_subprocess(inst, solver_binary=binary, timeout_s=timeout_s)  # type: ignore[arg-type]

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
    pair: Annotated[
        list[str],
        typer.Option(
            help="(pt:ct) pair, repeatable. At least one required.",
        ),
    ],
    hint_bits: Annotated[
        int,
        typer.Option(
            "--hint-bits",
            help="Number of low-index key bits to hint from --original-key",
        ),
    ] = 0,
    key_hint: Annotated[
        list[str] | None,
        typer.Option(
            "--key-hint",
            help="Explicit per-bit hints in the form 'index:value', repeatable.",
        ),
    ] = None,
    original_key: Annotated[
        str | None,
        typer.Option(
            "--original-key",
            help="Reference key (required with --hint-bits; also echoed if recovery succeeds)",
        ),
    ] = None,
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


def _polys_to_json(polys: list[BoolPoly]) -> str:
    return json.dumps(
        {
            "polynomials": [[sorted(list(m)) for m in p.monomials] for p in polys],
        }
    )


def _polys_from_json(text: str) -> list[BoolPoly]:
    data = json.loads(text)
    out: list[BoolPoly] = []
    for poly_mons in data["polynomials"]:
        out.append(BoolPoly(frozenset(frozenset(m) for m in poly_mons)))
    return out


def _instance_to_json(inst: CNFInstance | HybridInstance) -> str:
    if isinstance(inst, CNFInstance):
        return json.dumps(
            {
                "type": "cnf",
                "num_vars": inst.num_vars,
                "clauses": [list(c) for c in inst.clauses],
                "var_names": list(inst.var_names),
            }
        )
    if isinstance(inst, HybridInstance):
        return json.dumps(
            {
                "type": "hybrid",
                "num_vars": inst.num_vars,
                "cnf_clauses": [list(c) for c in inst.cnf_clauses],
                "xor_clauses": [[list(lits), rhs] for lits, rhs in inst.xor_clauses],
                "var_names": list(inst.var_names),
            }
        )
    raise ValueError(f"unknown instance type {type(inst).__name__}")


def _instance_from_json(text: str) -> CNFInstance | HybridInstance:
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
    polys = _anf_system(rounds=rounds, pairs=pairs, key_hints=hints or None)
    typer.echo(_polys_to_json(polys))


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


@app.command()
def benchmark(
    matrix: Annotated[
        str, typer.Option(help="Path to benchmark matrix TOML")
    ] = "benchmarks/matrix.toml",
    out_dir: Annotated[str, typer.Option(help="Output directory")] = "benchmark-results",
) -> None:
    """Run the benchmark matrix and write CSV + markdown to an output directory."""
    from datetime import datetime

    from benchmarks.bench_attack import run_matrix

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(out_dir) / ts
    run_matrix(Path(matrix), out_path)
    typer.echo(f"wrote results to {out_path}/")


if __name__ == "__main__":
    app()

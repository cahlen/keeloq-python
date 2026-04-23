"""CLI tests using Typer's CliRunner."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from keeloq._types import bits_to_int, int_to_bits
from keeloq.cipher import encrypt
from keeloq.cli import app

runner = CliRunner()


def test_encrypt_roundtrip_via_cli() -> None:
    pt_bits = "01100010100101110000101011100011"
    key_bits = "0011010011011111100101100001110000011101100111001000001101110100"

    result = runner.invoke(
        app,
        ["encrypt", "--rounds", "160", "--plaintext", pt_bits, "--key", key_bits],
    )
    assert result.exit_code == 0, result.stdout
    ct_bits = result.stdout.strip()
    assert len(ct_bits) == 32

    expected = encrypt(bits_to_int(pt_bits), bits_to_int(key_bits), 160)
    assert bits_to_int(ct_bits) == expected


def test_decrypt_roundtrip_via_cli() -> None:
    pt_bits = "01010101010101010101010101010101"
    key_bits = "0000010000100010100011100000000010000110000011001001111000010001"

    enc = runner.invoke(
        app, ["encrypt", "--rounds", "528", "--plaintext", pt_bits, "--key", key_bits]
    )
    assert enc.exit_code == 0
    ct_bits = enc.stdout.strip()

    dec = runner.invoke(
        app, ["decrypt", "--rounds", "528", "--ciphertext", ct_bits, "--key", key_bits]
    )
    assert dec.exit_code == 0
    assert dec.stdout.strip() == pt_bits


def test_encrypt_accepts_hex() -> None:
    result = runner.invoke(
        app,
        ["encrypt", "--rounds", "32", "--plaintext", "0xAAAA5555", "--key", "0x0123456789ABCDEF"],
    )
    assert result.exit_code == 0


def test_encrypt_rejects_wrong_length() -> None:
    result = runner.invoke(
        app,
        ["encrypt", "--rounds", "32", "--plaintext", "1010", "--key", "01" * 32],
    )
    assert result.exit_code != 0


def test_attack_subcommand_end_to_end() -> None:
    """32-round attack with 32 hint bits recovers the true key via CLI."""
    key_bits = int_to_bits(0x0123_4567_89AB_CDEF, 64)
    pt_int = 0xAAAA5555
    ct_int = encrypt(pt_int, 0x0123_4567_89AB_CDEF, 32)

    result = runner.invoke(
        app,
        [
            "attack",
            "--rounds",
            "32",
            "--pair",
            f"{int_to_bits(pt_int, 32)}:{int_to_bits(ct_int, 32)}",
            # Hint low 32 bits (indices 32..63):
            "--hint-bits",
            "32",
            "--original-key",
            key_bits,
            "--encoder",
            "xor",
            "--solver",
            "cryptominisat",
            "--timeout",
            "30",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert key_bits in result.stdout


def test_attack_subcommand_multi_pair() -> None:
    # Corrected from spec: 32r can't constrain K32-K63, so use 64 rounds + 4 pairs
    # (matches the Task 19 correction).
    key_int = 0x0123_4567_89AB_CDEF
    key_bits = int_to_bits(key_int, 64)
    pts = [0xAAAA5555, 0x13579BDF, 0xCAFEBABE, 0xFEEDFACE]
    cts = [encrypt(p, key_int, 64) for p in pts]
    pair_args = []
    for p, c in zip(pts, cts, strict=True):
        pair_args += ["--pair", f"{int_to_bits(p, 32)}:{int_to_bits(c, 32)}"]

    result = runner.invoke(
        app,
        [
            "attack",
            "--rounds",
            "64",
            "--hint-bits",
            "0",
            "--encoder",
            "xor",
            "--solver",
            "cryptominisat",
            "--timeout",
            "120",
            *pair_args,
        ],
    )
    assert result.exit_code == 0, result.stdout
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
        [
            "attack",
            "--rounds",
            "16",
            "--pair",
            pair,
            "--encoder",
            "cnf",
            "--solver",
            "cryptominisat",
            "--timeout",
            "10",
            *hint_args,
        ],
    )
    assert result.exit_code == 3, (
        f"expected exit 3 (UNSAT), got {result.exit_code}\n{result.stdout}"
    )


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
        [
            "generate-anf",
            "--rounds",
            "32",
            "--pair",
            pair,
            "--hint-bits",
            "32",
            "--original-key",
            int_to_bits(key_int, 64),
        ],
    )
    assert anf_res.exit_code == 0, anf_res.stdout
    anf_json = anf_res.stdout

    # Step 2: encode
    enc_res = runner.invoke(app, ["encode", "--encoder", "xor"], input=anf_json)
    assert enc_res.exit_code == 0, enc_res.stdout
    instance_json = enc_res.stdout

    # Step 3: solve
    solve_res = runner.invoke(
        app, ["solve", "--solver", "cryptominisat", "--timeout", "30"], input=instance_json
    )
    assert solve_res.exit_code == 0, solve_res.stdout
    result_json = solve_res.stdout
    parsed = json.loads(result_json)
    assert parsed["status"] == "SAT"

    # Step 4: verify
    vf_res = runner.invoke(
        app,
        ["verify", "--rounds", "32", "--pair", pair, "--original-key", int_to_bits(key_int, 64)],
        input=result_json,
    )
    assert vf_res.exit_code == 0
    assert "match: true" in vf_res.stdout.lower()


@pytest.mark.slow
def test_benchmark_smoke(tmp_path) -> None:
    """Run a one-row benchmark matrix and confirm CSV + MD appear."""
    matrix = tmp_path / "tiny.toml"
    matrix.write_text(
        "[[run]]\n"
        'name = "smoke-16r-heavy"\n'
        "rounds = 16\n"
        "num_pairs = 1\n"
        "hint_bits = 48\n"
        'encoder = "xor"\n'
        'solver = "cryptominisat"\n'
        "timeout_s = 30.0\n"
    )
    out_dir = tmp_path / "out"
    result = runner.invoke(app, ["benchmark", "--matrix", str(matrix), "--out-dir", str(out_dir)])
    assert result.exit_code == 0, result.stdout
    # Exactly one timestamped subdir should exist.
    subdirs = list(out_dir.iterdir())
    assert len(subdirs) == 1
    assert (subdirs[0] / "results.csv").exists()
    assert (subdirs[0] / "summary.md").exists()

"""CLI tests using Typer's CliRunner."""
from __future__ import annotations

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
        ["attack", "--rounds", "64", "--hint-bits", "0",
         "--encoder", "xor",
         "--solver", "cryptominisat", "--timeout", "120", *pair_args],
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
        ["attack", "--rounds", "16", "--pair", pair,
         "--encoder", "cnf", "--solver", "cryptominisat", "--timeout", "10", *hint_args],
    )
    assert result.exit_code == 3, f"expected exit 3 (UNSAT), got {result.exit_code}\n{result.stdout}"

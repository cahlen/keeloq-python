"""CLI tests using Typer's CliRunner."""
from __future__ import annotations

from typer.testing import CliRunner

from keeloq._types import bits_to_int
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

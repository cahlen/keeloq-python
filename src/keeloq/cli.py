"""keeloq command-line interface."""
from __future__ import annotations

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

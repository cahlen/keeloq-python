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

"""Reference Python implementation of KeeLoq, rounds-parameterized.

This module is the single source of truth for the cipher semantics. The ANF
generator and the GPU bit-sliced cipher must agree with it on all inputs.
"""

from __future__ import annotations


def core(a: int, b: int, c: int, d: int, e: int) -> int:
    """KeeLoq non-linear function in ANF over GF(2).

    From legacy/keeloq160-python.py:
        (d + e + ac + ae + bc + be + cd + de + ade + ace + abd + abc) mod 2
    """
    for name, v in (("a", a), ("b", b), ("c", c), ("d", d), ("e", e)):
        if v not in (0, 1):
            raise ValueError(f"core arg {name}={v} is not a bit")
    return (
        d
        + e
        + a * c
        + a * e
        + b * c
        + b * e
        + c * d
        + d * e
        + a * d * e
        + a * c * e
        + a * b * d
        + a * b * c
    ) % 2

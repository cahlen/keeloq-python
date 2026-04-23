"""Compatibility tests against the frozen 2015 legacy/ scripts."""

from __future__ import annotations

import pytest

from keeloq._types import bits_to_int, int_to_bits
from keeloq.anf import system as anf_system
from keeloq.cipher import encrypt
from tests.compat_helpers import (
    read_legacy_output_field,
    run_legacy_script,
    run_legacy_script_in_cwd,
)


@pytest.mark.legacy
def test_cipher_160_round_matches_legacy() -> None:
    out = run_legacy_script("keeloq160-python.py")
    pt = read_legacy_output_field(out, "Plaintext:")
    key = read_legacy_output_field(out, "Key:")
    ct_legacy = read_legacy_output_field(out, "Ciphertext:")

    our_ct = encrypt(bits_to_int(pt), bits_to_int(key), 160)
    assert int_to_bits(our_ct, 32) == ct_legacy


@pytest.mark.legacy
def test_cipher_528_round_matches_legacy() -> None:
    out = run_legacy_script("keeloq-python.py")
    pt = read_legacy_output_field(out, "Plaintext:")
    key = read_legacy_output_field(out, "Key:")
    ct_legacy = read_legacy_output_field(out, "Ciphertext:")

    our_ct = encrypt(bits_to_int(pt), bits_to_int(key), 528)
    assert int_to_bits(our_ct, 32) == ct_legacy


@pytest.mark.legacy
def test_anf_160_matches_legacy_anf_txt(tmp_path) -> None:
    """Semantic equivalence: both legacy and our ANF, after canonicalization,
    agree as multisets of polynomials."""
    run_legacy_script_in_cwd("sage-equations.py", tmp_path)
    legacy_anf = (tmp_path / "anf.txt").read_text()

    pt = "01100010100101110000101011100011"
    ct = "01101000110010010100101001111001"
    key = "00110100110111111001011000011100"
    pair_int = (bits_to_int(pt), bits_to_int(ct))
    hints = {i: int(key[i]) for i in range(32)}
    our_sys = anf_system(rounds=160, pairs=[pair_int], key_hints=hints)

    ours_canon = sorted(_poly_to_legacy_text(p) for p in our_sys)

    # Parse legacy: split on top-level commas; trim trailing empty.
    legacy_polys = [x for x in legacy_anf.split(",") if x]
    # Normalize legacy polynomials: re-parse each term, canonicalize.
    legacy_canon = sorted(_canonicalize_legacy_term(t) for t in legacy_polys)

    assert ours_canon == legacy_canon


def _poly_to_legacy_text(poly) -> str:
    """Render a BoolPoly in the canonical normalized form for comparison.

    Our BoolPoly uses sets, so we can't preserve the legacy input order.
    Instead we canonicalize: within a monomial, variables are sorted alphabetically
    and joined with '*'; within a polynomial, monomials are sorted alphabetically
    and joined with '+'. The legacy parser does the same normalization.

    Strip the _p0 suffix from our variable names so they match legacy's bare names.
    """
    if not poly.monomials:
        return ""
    terms: list[str] = []
    for mono in poly.monomials:
        if len(mono) == 0:
            terms.append("1")
        else:
            names = [v.replace("_p0", "") for v in mono]
            terms.append("*".join(sorted(names)))
    return "+".join(sorted(terms))


def _canonicalize_legacy_term(text: str) -> str:
    """Legacy polynomials are 'TERM+TERM+...' where each TERM is a '*'-joined
    monomial or a digit constant. Canonicalize: sort variables within each
    monomial, sort monomials within the polynomial.

    Drop any '0' constant tokens: in GF(2), x + 0 = x so they are noise in the
    legacy output (e.g. 'K3+0' should canonicalize to 'K3', matching our 'K3').
    """
    terms: list[str] = []
    for t in text.split("+"):
        t = t.strip()
        if not t:
            continue
        if t.isdigit():
            if t != "0":
                terms.append(t)
        else:
            factors = sorted(t.split("*"))
            terms.append("*".join(factors))
    return "+".join(sorted(terms))

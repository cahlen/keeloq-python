"""Compatibility tests against the frozen 2015 legacy/ scripts."""

from __future__ import annotations

import pytest

from keeloq._types import bits_to_int, int_to_bits
from keeloq.cipher import encrypt
from tests.compat_helpers import read_legacy_output_field, run_legacy_script


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

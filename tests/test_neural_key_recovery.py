"""Tests for keeloq.neural.key_recovery."""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from keeloq.cipher import encrypt


@pytest.mark.gpu
def test_peel_one_round_equals_shorter_encryption() -> None:
    """Peeling 1 round from N-round ct equals the (N-1)-round ct of the same pt."""
    from keeloq.neural.key_recovery import partial_decrypt_round

    key = 0x0123_4567_89AB_CDEF
    pt = 0xAAAA5555
    n_rounds = 16
    ct_full = encrypt(pt, key, n_rounds)
    ct_short = encrypt(pt, key, n_rounds - 1)

    true_kbit = (key >> (63 - ((n_rounds - 1) % 64))) & 1
    batch = torch.tensor([ct_full], dtype=torch.uint32, device="cuda")
    peeled = partial_decrypt_round(batch, key_bit=true_kbit, round_idx=n_rounds - 1)
    assert int(peeled[0].item()) & 0xFFFFFFFF == ct_short


@pytest.mark.gpu
def test_wrong_key_bit_disagrees() -> None:
    from keeloq.neural.key_recovery import partial_decrypt_round

    key = 0x0123_4567_89AB_CDEF
    pt = 0xAAAA5555
    n_rounds = 16
    ct = encrypt(pt, key, n_rounds)
    true_kbit = (key >> (63 - ((n_rounds - 1) % 64))) & 1
    wrong = 1 - true_kbit

    batch = torch.tensor([ct], dtype=torch.uint32, device="cuda")
    right = partial_decrypt_round(batch, key_bit=true_kbit, round_idx=n_rounds - 1)
    wrongr = partial_decrypt_round(batch, key_bit=wrong, round_idx=n_rounds - 1)
    assert int(right[0].item()) != int(wrongr[0].item())

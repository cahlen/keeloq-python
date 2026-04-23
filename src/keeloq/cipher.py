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


def _bit(value: int, position: int) -> int:
    """Return bit at `position` of `value`, where position 0 is the MSB for the
    state (matching the 2015 scripts' `list(PLAINTEXT)[0]` semantics)."""
    # For the state (32 bits), position 0 is MSB: bit_index_from_lsb = 31 - position
    # This helper works for any bit-width because callers convert consistently.
    return (value >> position) & 1


def _state_bit(state: int, position: int, width: int = 32) -> int:
    """Get MSB-indexed bit `position` out of `width`-bit `state`."""
    return (state >> (width - 1 - position)) & 1


def _key_bit(key: int, position: int) -> int:
    """Get MSB-indexed bit `position` out of 64-bit key."""
    return (key >> (63 - position)) & 1


def _validate_bitwidth(value: int, width: int, name: str) -> None:
    if value < 0 or value >> width != 0:
        raise ValueError(f"{name}={value} does not fit in {width} bits")


def encrypt(plaintext: int, key: int, rounds: int) -> int:
    """Encrypt `plaintext` under `key` for `rounds` rounds of the KeeLoq cipher.

    Plaintext is 32 bits, key is 64 bits. Bit 0 (as indexed in the 2015 scripts'
    list form) is the MSB; shifting "appends" a new bit at the LSB end.
    """
    _validate_bitwidth(plaintext, 32, "plaintext")
    _validate_bitwidth(key, 64, "key")
    if rounds < 0:
        raise ValueError(f"rounds={rounds} is negative")

    state = plaintext
    for i in range(rounds):
        # Match legacy/keeloq160-python.py round function:
        #   newb = (k[0] + p[0] + p[16] + core(p[31], p[26], p[20], p[9], p[1])) % 2
        # where p[0] is the MSB of the state and k[0] cycles through the key left-to-right.
        k0 = _key_bit(key, i % 64)
        p0 = _state_bit(state, 0)
        p16 = _state_bit(state, 16)
        p31 = _state_bit(state, 31)
        p26 = _state_bit(state, 26)
        p20 = _state_bit(state, 20)
        p9 = _state_bit(state, 9)
        p1 = _state_bit(state, 1)
        newb = (k0 + p0 + p16 + core(p31, p26, p20, p9, p1)) % 2
        # shiftp(p, newb): p.append(newb); del p[0].
        # In our MSB-first convention: shift left by 1, drop old MSB, put newb at the LSB end.
        state = ((state << 1) & 0xFFFFFFFF) | newb
    return state


def decrypt(ciphertext: int, key: int, rounds: int) -> int:
    """Inverse of encrypt: recover plaintext from ciphertext."""
    _validate_bitwidth(ciphertext, 32, "ciphertext")
    _validate_bitwidth(key, 64, "key")
    if rounds < 0:
        raise ValueError(f"rounds={rounds} is negative")

    state = ciphertext
    for i in range(rounds):
        # Reverse iteration: the key bit used during encryption at step r=(rounds-1-i)
        # corresponds to key index ((rounds-1-i) % 64) in MSB-first key layout — but the
        # legacy decrypt loop shifts the key *backwards* and consumes k[15] (for 528-round)
        # or k[31] (for 160-round). We emulate this via `rounds % 64` offset tracking.
        # Simpler equivalent: precompute key-bit sequence used during encryption, then walk
        # it in reverse.
        # Here we compute the key position index that was used at round (rounds-1-i):
        enc_step = rounds - 1 - i
        k0 = _key_bit(key, enc_step % 64)
        # The legacy `decroundfunction` uses p[31]/p[15]/core(p[30],p[25],p[19],p[8],p[0])
        # because after `shiftp` the state is one-position shifted. Inverting:
        #   encrypt: state = (state << 1 | newb); newb was derived from the pre-shift state.
        # So decrypt must recover the pre-shift state from (state << 1 | newb).
        p_last = _state_bit(state, 31)  # this was newb from the encrypt step
        # After un-shifting, the pre-shift state bits are:
        #   pre[1..31] = post[0..30]; pre[0] must be recovered from the round equation.
        # Encrypt round: post[31] = k0 + pre[0] + pre[16] + core(pre[31], pre[26], pre[20],
        #                                                         pre[9], pre[1]) mod 2
        # But after shift, pre[k] = post[k-1] for k>=1, and pre[0] is unknown.
        # Substitute pre[k] for k in {31,26,20,9,1,16}:
        #   pre[31]=post[30], pre[26]=post[25], pre[20]=post[19], pre[9]=post[8],
        #   pre[1]=post[0], pre[16]=post[15]
        # So: p_last = (k0 + pre[0] + post[15] + core(post[30], post[25], post[19],
        #                                              post[8], post[0])) mod 2
        # Solve for pre[0]:
        pre_bit_0 = (
            p_last
            - k0
            - _state_bit(state, 15)
            - core(
                _state_bit(state, 30),
                _state_bit(state, 25),
                _state_bit(state, 19),
                _state_bit(state, 8),
                _state_bit(state, 0),
            )
        ) % 2
        # Un-shift: new state has pre_bit_0 as MSB, and drops post[31] (the newb).
        state = ((state >> 1) & 0x7FFFFFFF) | (pre_bit_0 << 31)
    return state

"""Exception hierarchy for keeloq."""

from __future__ import annotations


class KeeloqError(Exception):
    """Base class for all keeloq exceptions."""


class InvariantError(KeeloqError):
    """An internal invariant was violated. Indicates a bug, not user input."""


class SolverError(KeeloqError):
    """A SAT solver crashed or returned unparseable output."""


class EncodingError(KeeloqError):
    """An encoder failed to produce a valid instance from the given ANF system."""


class VerificationError(KeeloqError):
    """A recovered key failed to round-trip through the cipher."""

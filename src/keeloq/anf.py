"""Boolean polynomial arithmetic over GF(2), and the KeeLoq ANF system generator.

Polynomials are represented as a frozenset of monomials, where each monomial is
a frozenset of variable names. The empty frozenset represents the constant 1;
the empty outer frozenset represents the constant 0. Arithmetic is:
  - Addition: symmetric difference of monomial sets (x + x = 0)
  - Multiplication: cross-product of monomials, then union within each, then sum
  - Variable power: x * x = x (GF(2) idempotence handled implicitly by using sets)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoolPoly:
    """Polynomial in GF(2)[x_1, ..., x_n]/(x_i^2 - x_i).

    `monomials` is a frozenset of frozensets-of-variable-names.
    The empty outer frozenset is the zero polynomial.
    A monomial equal to the empty frozenset represents the constant 1.
    """

    monomials: frozenset[frozenset[str]]

    def __add__(self, other: BoolPoly) -> BoolPoly:
        return BoolPoly(self.monomials.symmetric_difference(other.monomials))

    def __mul__(self, other: BoolPoly) -> BoolPoly:
        result: set[frozenset[str]] = set()
        for m1 in self.monomials:
            for m2 in other.monomials:
                product = m1 | m2  # GF(2) idempotence: x*x = x
                if product in result:
                    result.remove(product)
                else:
                    result.add(product)
        return BoolPoly(frozenset(result))

    def substitute(self, assignment: dict[str, int]) -> int:
        """Evaluate under a total assignment of all variables to 0/1."""
        needed = self.variables()
        missing = needed - assignment.keys()
        if missing:
            raise ValueError(f"substitute: missing values for {sorted(missing)}")
        total = 0
        for m in self.monomials:
            product = 1
            for v in m:
                product *= assignment[v]
                if product == 0:
                    break
            total = (total + product) & 1
        return total

    def variables(self) -> set[str]:
        out: set[str] = set()
        for m in self.monomials:
            out.update(m)
        return out


def zero() -> BoolPoly:
    return BoolPoly(frozenset())


def one() -> BoolPoly:
    return BoolPoly(frozenset([frozenset()]))


def var(name: str) -> BoolPoly:
    return BoolPoly(frozenset([frozenset([name])]))

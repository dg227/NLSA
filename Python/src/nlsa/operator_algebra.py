"""Provide functions and classes implementing operator algebras."""

import nlsa.abstract_algebra as alg
import nlsa.function_algebra as fun
from collections.abc import Callable
from dataclasses import dataclass
from typing import final


type L[V] = Callable[[V], V]


@final
@dataclass(frozen=True, slots=True)
class OperatorAlgebra[V, K](alg.ImplementsOperatorAlgebra[L[V], V, K]):
    """Implement operator algebra structure on a vector space."""

    domain: alg.ImplementsInnerProductSpace[V, K]
    _zero: Callable[[], L[V]]
    _add: Callable[[L[V], L[V]], L[V]]
    _sub: Callable[[L[V], L[V]], L[V]]
    _neg: Callable[[L[V]], L[V]]
    _smul: Callable[[K, L[V]], L[V]]
    _sdiv: Callable[[K, L[V]], L[V]]
    _mul: Callable[[L[V], L[V]], L[V]]
    _mpower: Callable[[L[V], int], L[V]]
    _unit: Callable[[], L[V]]
    _app: Callable[[L[V], V], V]

    @property
    def scl(self) -> alg.ImplementsRealScalarField[K]:
        """Scalar field associated with an OperatorAlgebra object."""
        return self.domain.scl

    @property
    def dom(self) -> alg.ImplementsInnerProductSpace[V, K]:
        """Vector space domain associated with an OperatorAlgebra object."""
        return self.domain

    @property
    def codom(self) -> alg.ImplementsInnerProductSpace[V, K]:
        """Vector space codomain associated with an OperatorAlgebra object."""
        return self.domain

    def zero(self, /) -> L[V]:
        """Return zero operator."""
        return self._zero()

    def add(self, a: L[V], b: L[V], /) -> L[V]:
        """Add two operators."""
        return self._add(a, b)

    def sub(self, a: L[V], b: L[V], /) -> L[V]:
        """Subtract two operators."""
        return self._sub(a, b)

    def neg(self, a: L[V], /) -> L[V]:
        """Compute additive inverse (negation) of an operator."""
        return self._neg(a)

    def smul(self, k: K, a: L[V], /) -> L[V]:
        """Multiply an operator by a scalar."""
        return self._smul(k, a)

    def sdiv(self, k: K, a: L[V], /) -> L[V]:
        """Divide an operator by a scalar."""
        return self._sdiv(k, a)

    def unit(self, /) -> L[V]:
        """Return identity operator."""
        return self._unit()

    def mul(self, a: L[V], b: L[V], /) -> L[V]:
        """Compose two operators."""
        return self._mul(a, b)

    def mpower(self, a: L[V], m: int, /) -> L[V]:
        """M-fold composition of an operator."""
        return self._mpower(a, m)

    def app(self, a: L[V], v: V) -> V:
        """Apply an operator to a vector."""
        return self._app(a, v)


def operator_algebra[V, K](
    domain: alg.ImplementsInnerProductSpace[V, K],
    zero: Callable[[], L[V]] | None = None,
    add: Callable[[L[V], L[V]], L[V]] | None = None,
    sub: Callable[[L[V], L[V]], L[V]] | None = None,
    neg: Callable[[L[V]], L[V]] | None = None,
    smul: Callable[[K, L[V]], L[V]] | None = None,
    sdiv: Callable[[K, L[V]], L[V]] | None = None,
    mul: Callable[[L[V], L[V]], L[V]] | None = None,
    mpower: Callable[[L[V], int], L[V]] | None = None,
    unit: Callable[[], L[V]] | None = None,
    app: Callable[[L[V], V], V] | None = None,
) -> OperatorAlgebra[V, K]:
    """Build OperatorAlgebraObject."""
    return OperatorAlgebra(
        domain=domain,
        _zero=(fun.lift_constant(domain.zero) if zero is None else zero),
        _add=(fun.lift_binary(domain.add) if add is None else add),
        _sub=(fun.lift_binary(domain.sub) if sub is None else sub),
        _neg=(fun.lift_unary(domain.neg) if neg is None else neg),
        _smul=(fun.lift_left(domain.smul) if smul is None else smul),
        _sdiv=(fun.lift_left(domain.sdiv) if sdiv is None else sdiv),
        _unit=(fun.make_constant(fun.identity) if unit is None else unit),
        _mul=(fun.compose if mul is None else mul),
        _mpower=(fun.make_mpower(fun.compose) if mpower is None else mpower),
        _app=(fun.apply if app is None else app),
    )

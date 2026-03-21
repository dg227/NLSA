"""Provide functions and classes implementing operator algebras."""

import nlsa.abstract_algebra as alg
import nlsa.function_algebra as fun
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, final


type L[V] = Callable[[V], V]


@final
@dataclass(frozen=True)
class OperatorAlgebra[V, K](alg.ImplementsOperatorAlgebra[L[V], V, K]):
    """Implement operator algebra structure on a vector space."""

    domain: alg.ImplementsInnerProductSpace[V, K]
    _scl: Optional[alg.ImplementsScalarField[K]] = None
    _codom: Optional[alg.ImplementsInnerProductSpace[V, K]] = None
    _zero: Optional[Callable[[], L[V]]] = None
    _add: Optional[Callable[[L[V], L[V]], L[V]]] = None
    _sub: Optional[Callable[[L[V], L[V]], L[V]]] = None
    _neg: Optional[Callable[[L[V]], L[V]]] = None
    _smul: Optional[Callable[[K, L[V]], L[V]]] = None
    _sdiv: Optional[Callable[[K, L[V]], L[V]]] = None
    _mul: Optional[Callable[[L[V], L[V]], L[V]]] = None
    _mpower: Optional[Callable[[L[V], int], L[V]]] = None
    _unit: Optional[Callable[[], L[V]]] = None
    _app: Optional[Callable[[L[V], V], V]] = None

    @property
    def dom(self) -> alg.ImplementsInnerProductSpace[V, K]:
        """Return dom property of L2FnAlgebra object."""
        return self.domain

    @property
    def scl(self) -> alg.ImplementsScalarField[K]:
        """Return scl property of OperatorAlgebra object."""
        return self.domain.scl if self._scl is None else self._scl

    @property
    def codom(self) -> alg.ImplementsInnerProductSpace[V, K]:
        """Return codom property of OperatorAlgebra object."""
        return self.domain if self._codom is None else self._codom

    @property
    def zero(self) -> Callable[[], L[V]]:
        """Return zero property of OperatorAlgebra object."""
        return (
            fun.lift_constant(self.domain.zero)
            if self._zero is None
            else self._zero
        )

    @property
    def add(self) -> Callable[[L[V], L[V]], L[V]]:
        """Return add property of OperatorAlgebra object."""
        return (
            fun.lift_binary(self.domain.add)
            if self._add is None
            else self._add
        )

    @property
    def sub(self) -> Callable[[L[V], L[V]], L[V]]:
        """Return sub property of OperatorAlgebra object."""
        return (
            fun.lift_binary(self.domain.sub)
            if self._sub is None
            else self._sub
        )

    @property
    def neg(self) -> Callable[[L[V]], L[V]]:
        """Return neg property of OperatorAlgebra object."""
        return (
            fun.lift_unary(self.domain.neg) if self._neg is None else self._neg
        )

    @property
    def smul(self) -> Callable[[K, L[V]], L[V]]:
        """Return smul property of OperatorAlgebra object."""
        return (
            fun.lift_left(self.domain.smul)
            if self._smul is None
            else self._smul
        )

    @property
    def sdiv(self) -> Callable[[K, L[V]], L[V]]:
        """Return sdiv property of OperatorAlgebra object."""
        return (
            fun.lift_left(self.domain.sdiv)
            if self._sdiv is None
            else self._sdiv
        )

    @property
    def mul(self) -> Callable[[L[V], L[V]], L[V]]:
        """Return mul property of OperatorAlgebra object."""
        return fun.compose if self._mul is None else self._mul

    @property
    def mpower(self) -> Callable[[L[V], int], L[V]]:
        """Return mpower property of OperatorAlgebra object."""
        return (
            fun.make_mpower(fun.compose)
            if self._mpower is None
            else self._mpower
        )

    @property
    def unit(self) -> Callable[[], L[V]]:
        """Return unit property of OperatorAlgebra object."""
        return (
            fun.make_constant(fun.identity)
            if self._unit is None
            else self._unit
        )

    @property
    def app(self) -> Callable[[L[V], V], V]:
        """Return app property of OperatorAlgebra object."""
        return fun.apply if self._app is None else self._app

# pyright: basic
"""Provide scalar field operations for JAX arrays."""

import jax.numpy as jnp
import nlsa.abstract_algebra as alg
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from jax import Array
from jax.typing import DTypeLike
from typing import Optional, final

type K = Array
type F[*Xs, Y] = Callable[[*Xs], Y]


def neg(s: K, /) -> K:
    """Negate a scalar."""
    return jnp.multiply(-1, s)


def make_zero(dtype: DTypeLike) -> Callable[[], K]:
    """Make constant function returning scalar zero."""

    def zero() -> K:
        return jnp.zeros((), dtype=dtype)

    return zero


def make_unit(dtype: DTypeLike) -> Callable[[], K]:
    """Make constant function returning scalar unit."""

    def unit() -> K:
        return jnp.ones((), dtype=dtype)

    return unit


def make_inv(dtype: DTypeLike) -> Callable[[K], K]:
    """Make inversion function."""

    def inv(s: K, /) -> K:
        return jnp.divide(jnp.ones((), dtype=dtype), s)

    return inv


def ldiv(s: K, t: K, /) -> K:
    """Left-divide two scalars."""
    return jnp.divide(t, s)


@final
@dataclass(frozen=True)
class ScalarField[D: DTypeLike](alg.ImplementsScalarField[K]):
    """Implement scalar field operations on JAX arrays."""

    dtype: D
    _zero: Optional[Callable[[], K]] = None
    _add: Optional[Callable[[K, K], K]] = None
    _sub: Optional[Callable[[K, K], K]] = None
    _neg: Optional[Callable[[K], K]] = None
    _unit: Optional[Callable[[], K]] = None
    _mul: Optional[Callable[[K, K], K]] = None
    _mpower: Optional[Callable[[K, int], K]] = None
    _power: Optional[Callable[[K, K], K]] = None
    _div: Optional[Callable[[K, K], K]] = None
    _inv: Optional[Callable[[K], K]] = None
    _adj: Optional[Callable[[K], K]] = None
    _sqrt: Optional[Callable[[K], K]] = None
    _mod: Optional[Callable[[K], K]] = None

    @cached_property
    def zero(self) -> Callable[[], K]:
        """Return zero property of ScalarField object."""
        return make_zero(self.dtype) if self._zero is None else self._zero

    @cached_property
    def add(self) -> Callable[[K, K], K]:
        """Return add property of ScalarField object."""
        return jnp.add if self._add is None else self._add

    @cached_property
    def sub(self) -> Callable[[K, K], K]:
        """Return sub property of ScalarField object."""
        return jnp.subtract if self._sub is None else self._sub

    @cached_property
    def neg(self) -> Callable[[K], K]:
        """Return neg property of ScalarField object."""
        return neg if self._neg is None else self._neg

    @cached_property
    def unit(self) -> Callable[[], K]:
        """Return unit property of ScalarField object."""
        return make_unit(self.dtype) if self._unit is None else self._unit

    @cached_property
    def mul(self) -> Callable[[K, K], K]:
        """Return mul property of ScalarField object."""
        return jnp.multiply if self._mul is None else self._mul

    @cached_property
    def mpower(self) -> Callable[[K, int], K]:
        """Return mpower property of ScalarField object."""
        return jnp.power if self._mpower is None else self._mpower

    @cached_property
    def power(self) -> Callable[[K, K], K]:
        """Return power property of ScalarField object."""
        return jnp.power if self._power is None else self._power

    @cached_property
    def div(self) -> Callable[[K, K], K]:
        """Return div property of ScalarField object."""
        return jnp.divide if self._div is None else self._div

    @cached_property
    def inv(self) -> Callable[[K], K]:
        """Return inv property of ScalarField object."""
        return make_inv(self.dtype) if self._inv is None else self._inv

    @cached_property
    def adj(self) -> Callable[[K], K]:
        """Return adj property of ScalarField object."""
        return jnp.conjugate if self._adj is None else self._adj

    @cached_property
    def sqrt(self) -> Callable[[K], K]:
        """Return sqrt property of ScalarField object."""
        return jnp.sqrt if self._sqrt is None else self._sqrt

    @cached_property
    def mod(self) -> Callable[[K], K]:
        """Return mod property of ScalarField object."""
        return jnp.abs if self._mod is None else self._mod

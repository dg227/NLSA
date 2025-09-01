# pyright: basic

"""Provide scalar field operations for JAX arrays"""

import jax.numpy as jnp
import nlsa.abstract_algebra as alg
from collections.abc import Callable
from jax import Array
from jax.typing import DTypeLike
from typing import final

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
    def inv(s: K, /) -> K:
        return jnp.divide(jnp.ones((), dtype=dtype), s)
    return inv


def ldiv(s: K, t: K, /) -> K:
    """Left-divide two scalars."""
    return jnp.divide(t, s)


@final
class ScalarField[D: DTypeLike](alg.ImplementsScalarField[K]):
    """Implement scalar field operations for JAX arrays."""
    def __init__(self, dtype: D):
        self.zero: Callable[[], K] = make_zero(dtype)
        self.add: Callable[[K, K], K] = jnp.add
        self.sub: Callable[[K, K], K] = jnp.subtract
        self.neg: Callable[[K], K] = neg
        self.unit: Callable[[], K] = make_unit(dtype)
        self.mul: Callable[[K, K], K] = jnp.multiply
        self.power: Callable[[K, K], K] = jnp.power
        self.div: Callable[[K, K], K] = jnp.divide
        self.inv: Callable[[K], K] = make_inv(dtype)
        self.adj: Callable[[K], K] = jnp.conjugate
        self.sqrt: Callable[[K], K] = jnp.sqrt
        self.mod: Callable[[K], K] = jnp.abs

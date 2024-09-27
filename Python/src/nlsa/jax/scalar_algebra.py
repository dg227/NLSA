# pyright: basic

import jax.numpy as jnp
from jax import Array
from typing import Callable, Generic, Type, TypeAlias, TypeVar

N = TypeVar('N', bound=int)
K = TypeVar('K', jnp.float32, jnp.float64)
R: TypeAlias = K
S = Array
V = Array
X = Array
Xs = Array
Y = Array
T = TypeVar('T')
T1 = TypeVar('T1')
T2 = TypeVar('T2')
F = Callable[[T1], T2]


def neg(s: S, /) -> S:
    """Negate a scalar."""
    return jnp.multiply(-1, s)


def make_zero(dtype: Type[K]) -> Callable[[], S]:
    """Make constant function returning scalar zero."""
    def zero() -> S:
        return jnp.float32(0)
    return zero


def make_unit(dtype: Type[K]) -> Callable[[], S]:
    """Make constant function returning scalar unit."""
    def unit() -> S:
        return jnp.float32(1)
    return unit


def make_inv(dtype: Type[K]) -> Callable[[S], S]:
    """Make inversion function for scalars."""
    def inv(s: S) -> S:
        return jnp.divide(1, s)
    return inv


def ldiv(s: S, t: S, /) -> S:
    """Left-divide two scalars."""
    return jnp.divide(t, s)


def exp10(s: S, /) -> S:
    """Exponentiate with base 10."""
    return jnp.power(10, s)


class ScalarField(Generic[K]):
    """Implement scalar field operations for JAX arrays.

    The type parameter K parameterizes the field of scalars.
    """

    def __init__(self, dtype: Type[K]):
        self.zero: Callable[[], S] = make_zero(dtype)
        self.add: Callable[[S, S], S] = jnp.add
        self.neg: Callable[[S], S] = neg
        self.sub: Callable[[S, S], S] = jnp.subtract
        self.mul: Callable[[S, S], S] = jnp.multiply
        self.unit: Callable[[], S] = make_unit(dtype)
        self.inv: Callable[[S], S] = make_inv(dtype)
        self.div: Callable[[S, S], S] = jnp.divide
        self.star: Callable[[S], S] = jnp.conjugate
        self.lmul: Callable[[S, S], S] = jnp.multiply
        self.ldiv: Callable[[S, S], S] = ldiv
        self.sqrt: Callable[[S], S] = jnp.sqrt
        self.exp: Callable[[S], S] = jnp.exp
        self.exp10: Callable[[S], S] = exp10
        self.log: Callable[[S], S] = jnp.log
        self.log10: Callable[[S], S] = jnp.log10
        self.power: Callable[[S, S], S] = jnp.power

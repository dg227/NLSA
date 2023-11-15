# pyright: basic

import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import expm, sqrtm, inv
from nlsa.jax.vector_algebra import ScalarField, VectorAlgebra
from typing import Callable, Generic, Optional, Type, TypeVar

M = TypeVar('M', bound=int)
N = TypeVar('N', bound=int)
K = TypeVar('K', jnp.float32, jnp.float64)
S = Array
A = Array
V = Array
W = Array


def neg(a: A, /) -> A:
    """Negate a matrix."""
    return jnp.multiply(-1, a)


def make_unit(dim: int, dtype: Type[K]) -> Callable[[], A]:
    """Make unit matrix."""
    def unit() -> V:
        return jnp.eye(dim, dtype=dtype)
    return unit


def divm(a: A, b: A, /) -> A:
    """Matrix division."""
    return jnp.matmul(a, inv(b))


def ldivm(a: A, b: A, /) -> A:
    """Perform left module division as matrix division."""
    return jnp.matmul(inv(a), b)


def star(a: A, /) -> A:
    """Compute complex conjugate transpose of matrix."""
    return jnp.conjugate(a.T)


def b2_innerp(a: A, b: A, /) -> S:
    """Compute Hilbert-Schmidt (B2) inner product of two vectors."""
    return jnp.trace(jnp.matmul(jnp.conjugate(a.T), b))


def make_weighted_b2_innerp(w: A, /) -> Callable[[A, A], S]:
    """Make weighted Hilbert-Schmidt inner procuct from weight matrix."""
    def innerp(a: A, b: A, /) -> S:
        return jnp.trace(jnp.matmul(jnp.conjugate(a.T), jnp.matmul(w, b)))
    return innerp


class MatrixAlgebra(Generic[N, K]):
    """Implement matrix algebra operations for N by N JAX arrays.

    The type variable N parameterizes the dimension of the matrices. The type
    variable K parameterizes the field of scalars.
    """

    def __init__(self, dim: N, dtype: Type[K],
                 hilb: Optional[VectorAlgebra[N, K]] = None,
                 weight: Optional[A] = None):

        if hilb is None:
            self.hilb = VectorAlgebra(dim, dtype)
        else:
            self.hilb = hilb

        self.scl = ScalarField(dtype)
        self.add: Callable[[A, A], A] = jnp.add
        self.neg: Callable[[A], A] = neg
        self.sub: Callable[[A, A], A] = jnp.subtract
        self.smul: Callable[[S, A], A] = jnp.multiply
        self.mul: Callable[[A, A], A] = jnp.matmul
        self.unit: Callable[[], A] = make_unit(dim, dtype)
        self.inv: Callable[[A], A] = inv
        self.div: Callable[[A, A], A] = divm
        self.star: Callable[[A], A] = star
        self.lmul: Callable[[A, A], A] = jnp.matmul
        self.ldiv: Callable[[A, A], A] = ldivm
        self.rmul: Callable[[A, A], A] = jnp.matmul
        self.rdiv: Callable[[A, A], A] = divm
        self.sqrt: Callable[[A], A] = sqrtm
        self.exp: Callable[[A], A] = expm
        self.power: Callable[[A, S], A] = jnp.power
        self.app: Callable[[A, V], V] = jnp.matmul

        if weight is None:
            self.innerp: Callable[[A, A], S] = b2_innerp
        else:
            self.innerp: Callable[[A, A], S] = make_weighted_b2_innerp(weight)


class MatrixSpace(Generic[M, N, K]):
    """Implement matrix vector space operations for M by N JAX arrays.

    The type variables M and N parameterize the dimension of the matrices. The
    type variable K parameterizes the field of scalars.
    """

    def __init__(self, dim: tuple[M, N], dtype: Type[K],
                 hilb1: Optional[VectorAlgebra[N, K]] = None,
                 hilb2: Optional[VectorAlgebra[M, K]] = None,
                 weight: Optional[A] = None):

        if hilb1 is None:
            self.hilb1 = VectorAlgebra(dim, dtype)
        else:
            self.hilb1 = hilb1

        if hilb2 is None:
            self.hilb2 = VectorAlgebra(dim, dtype)
        else:
            self.hilb2 = hilb2

        self.scl = ScalarField(dtype)
        self.add: Callable[[A, A], A] = jnp.add
        self.neg: Callable[[A], A] = neg
        self.sub: Callable[[A, A], A] = jnp.subtract
        self.smul: Callable[[S, A], A] = jnp.multiply
        self.star: Callable[[A], A] = star
        self.lmul: Callable[[A, A], A] = jnp.matmul
        self.ldiv: Callable[[A, A], A] = ldivm
        self.rmul: Callable[[A, A], A] = jnp.matmul
        self.rdiv: Callable[[A, A], A] = divm
        self.app: Callable[[A, V], W] = jnp.matmul

        if weight is None:
            self.innerp: Callable[[A, A], S] = b2_innerp
        else:
            self.innerp: Callable[[A, A], S] = make_weighted_b2_innerp(weight)

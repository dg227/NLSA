"""Implements algebraic structures for spaces of column vectors, viewed as
abelian algebras with respect to entrywise vector multiplication.

This module broadcasts algebraic operations over collections of vectors stored
in higher-rank arrays.

We use the following TypeVar declarations:
    :N: Array dimension representing vector space dimension.
    :L, M: Array dimensions representing collections of vectors.
    :K: Scalars and arrays of scalars.
    :V: Vectors and arrays of vectors.
    :A: Arbitrary-sized arrays.

"""
import numpy as np
from nlsa.abstract_algebra import ImplementsAlgmul
from nptyping import Complex, Double, Int, NDArray, Shape
from typing import Literal, TypeVar


L = TypeVar('L')
M = TypeVar('M')
N = TypeVar('N')
K = TypeVar('K', complex, float, int,
            NDArray[Shape['M'], Complex],
            NDArray[Shape['M'], Double],
            NDArray[Shape['M'], Int],
            NDArray[Shape['L, M'], Complex],
            NDArray[Shape['L, M'], Double],
            NDArray[Shape['L, M'], Int])
V = TypeVar('V',
            NDArray[Shape['N'], Complex],
            NDArray[Shape['N'], Double],
            NDArray[Shape['N'], Int],
            NDArray[Shape['M, N'], Complex],
            NDArray[Shape['M, N'], Double],
            NDArray[Shape['M, N'], Int],
            NDArray[Shape['L, M, N'], Complex],
            NDArray[Shape['L, M, N'], Double],
            NDArray[Shape['L, M, N'], Int])
A = TypeVar('A',
            NDArray[Shape['*, ...'], Complex],
            NDArray[Shape['*, ...'], Double],
            NDArray[Shape['*, ...'], Int])


def add(u: V, v: V) -> V:
    """Implements vector addition"""
    w = u + v
    return w


def sub(u: V, v: V) -> V:
    """Implements vector subtraction"""
    w = u - v
    return w


def mul(a: K, v: V) -> V:
    """Implements scalar/left module multiplication for vectors."""
    w = np.multiply(a, v)
    return w


def algmul(u: V, v: V) -> V:
    """Algebraic multiplication as elementwise vector multiplication.

    """
    w = u * v
    return w


compose = algmul


def star(u: V) -> V:
    """Algebraic adjunction as complex conjugation."""
    v: V = np.conjugate(u)
    return v


def algdiv(u: V, v: V) -> V:
    """Algebraic division as elementwise vector division.

    """
    w = u / v
    return w


def algldiv(u: V, v: V) -> V:
    """Left algebraic division as elementwise vector division.

    """
    match (u.ndim, v.ndim):
        case (1, 1):
            w = v / u
        case _:
            w = v / np.atleast_2d(u).T
    return w


rdiv = algdiv


def power(v: V, a: K) -> V:
    """Elementwise exponentiation of vectors."""
    w = np.power(v, a)
    return w


def sqrt(v: V) -> V:
    """Elementwise square root of vectors."""
    w: V = np.sqrt(v)
    return w


def condition_by(impl: ImplementsAlgmul[V], a: V, v: V) -> V:
    """Bayesian conditioning of state vector by quantum effect.

    :a: Matrix representation of quantum effect.
    :v: State vector.
    :returns: Conditioned state vector.

    """
    u: V = impl.algmul(v, a)
    return u / np.linalg.norm(u)


def innerp(u: V, v: V) -> K:
    """Inner product of vectors.

    :u: Vector or array of vectors.
    :v: Vector or array of vectors.
    :returns: Inner product between u and v.

    """
    w: K = np.einsum('...i,...i->...', np.conjugate(u), v)
    return w


def uniform(n: int) -> V:
    """Uniform probability vector.

    :n: Dimension.
    :returns: Probability vector with elements 1/n.

    """
    u = np.full(n, 1 / n)
    return u


def std_basis(n: int, j: int, shape2d: Literal['row', 'col'] | None = None)\
        -> NDArray[Shape['N'], Double]\
        | NDArray[Shape['1, N'], Double]\
        | NDArray[Shape['N, 1'], Double]:
    """Standard basis vectors of C^n.

    :n: Dimension
    :j: Basis vector index.
    :returns: j-th n-dimensional basis vector.

    """
    u = np.zeros(n)
    u[j] = 1
    match shape2d:
        case 'col':
            u = u[:, np.newaxis]
        case 'row':
            u = u[np.newaxis, :] 
    return u


# TODO: Implement linear map application

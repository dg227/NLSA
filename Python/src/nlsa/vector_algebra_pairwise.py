
"""Implements algebraic structures for spaces of column vectors, viewed as
abelian algebras with respect to entrywise vector multiplication.

This module differs from vector_algebra in the broadcasting rules of binary
operations. Instead of broadcasting over the same array dimensions, in
introduces additional dimensions to implement pairwise evaluation of binary
operations.

We use the following TypeVar declarations:
    :N: Array dimension representing vector space dimension.
    :L, M: Array dimensions representing collections of vectors.
    :K: Arrays of scalars.
    :V: Vectors and arrays of vectors.
    :A: Arbitrary-sized arrays.

"""
import nlsa.vector_algebra as vec
import numpy as np
from nptyping import Complex, Double, Int, NDArray, Shape
from typing import Callable, TypeVar

L = TypeVar('L')
M = TypeVar('M')
N = TypeVar('N')
K = TypeVar('K',
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


def make_pairwise(f: Callable[[A, A], A], keepdims: bool = False)\
        -> Callable[[A, A], A]:
    """Lift binary operation to pairwise binary operation."""
    def g(vi: A, vj: A) -> A:
        vi1 = vi[..., :, np.newaxis, :]
        v1j = vj[..., np.newaxis, :, :]
        wij = f(vi1, v1j)
        if keepdims is False:
            wij = np.squeeze(wij)
        return wij
    return g


add: Callable[[V, V], V] = make_pairwise(vec.add)
algmul: Callable[[V, V], V] = make_pairwise(vec.algmul)
star: Callable[[V], V] = vec.star


def innerp(u: V, v: V) -> K:
    """Pairwise inner product of vectors.

    :u: Vector or array of vectors.
    :v: Vector or array of vectors.
    :returns: Array of pairwise inner products between u and v.

    """
    w: K = np.einsum('...ji,...ki->...jk',
                     np.conjugate(np.atleast_2d(u)), np.atleast_2d(v))
    return w

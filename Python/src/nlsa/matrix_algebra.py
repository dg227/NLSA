"""Implements algebraic structures for spaces of matrices"""

import numpy as np
from nptyping import Double, NDArray, Shape
from typing import Callable, TypeVar

K = TypeVar("K", float, complex)
A = NDArray[Shape["N, N"], Double]
V = NDArray[Shape["N"], Double]
L = NDArray[Shape["M, N"], Double]
M = TypeVar('M', V, L)

M_by_N = NDArray[Shape["M, N"], Double]
N_by_P = NDArray[Shape["N, P"], Double]
M_by_P = NDArray[Shape["M, P"], Double]


def add(u: M, v: M) -> M:
    """Implements matrix addition."""
    h: M = u + v
    return h


def compose(a: M_by_N, b: N_by_P) -> M_by_P:
    """Implements matrix composition (multiplication)"""
    c: M_by_P = a @ b
    return c


def pure_state(v: V) -> Callable[[A], K]:
    """Pure state on matrix algebra.

    :v: Vector inducing state.
    :returns: State omega as a functional on the matrix algebra.

    The state construction is vectorized, meaning that v can be an array of
    vectors and the resulting omega will map matrices to arrays of scalars.
    """
    def omega(a: A) -> K:
        y: K = np.einsum('...i,...i->...', np.dot(np.conjugate(v), a), v)
        return y
    return omega

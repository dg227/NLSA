"""Implements algebraic structures for spaces of matrices"""

import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as sla
from nptyping import Complex, Double, NDArray, Shape
from typing import Callable, Literal, Optional, Tuple, TypeVar

N = TypeVar('N')
M = TypeVar('M')
P = TypeVar('P')
V = TypeVar('V',
            NDArray[Shape['M, N'], Double],
            NDArray[Shape['M, N'], Complex])
K = TypeVar('K',
            NDArray[Shape['M'], Double],
            NDArray[Shape['M'], Complex])
W = TypeVar('W',
            NDArray[Shape['N'], Double],
            NDArray[Shape['N'], Complex])
A = TypeVar('A',
            NDArray[Shape['N, N'], Double],
            NDArray[Shape['N, N'], Complex])
M_by_N = TypeVar('M_by_N',
                 NDArray[Shape['M, N'], Double],
                 NDArray[Shape['M, N'], Complex])
N_by_P = TypeVar('N_by_P',
                 NDArray[Shape['N, P'], Double],
                 NDArray[Shape['N, P'], Complex])
M_by_P = TypeVar('M_by_P',
                 NDArray[Shape['M, P'], Double],
                 NDArray[Shape['M, P'], Complex])


def add(a: M_by_N, b: M_by_N) -> M_by_N:
    """Implements matrix addition."""
    c: M_by_N = a + b
    return c


def compose(a: M_by_N, b: N_by_P) -> M_by_P:
    """Implements matrix composition (multiplication)."""
    c: M_by_P = a @ b
    return c


def algmul(a: A, b: A) -> A:
    """Implements matrix algebra multiplication."""
    c: A = a @ b
    return c


def pure_state(v: V) -> Callable[[A], K]:
    """Pure state on matrix algebra.

    :v: State vector.
    :returns: State omega as a functional on the matrix algebra.

    The state construction is vectorized, meaning that v can be an array of
    vectors and the resulting omega will map matrices to arrays of scalars.

    """
    def omega(a: A) -> K:
        y: K = np.einsum('...i,...i->...', np.dot(np.conjugate(v), a), v)
        return y
    return omega


def state_vector_bayes(v: V, a: A) -> V:
    """Bayesian conditioning of state vector by quantum effect.

    :v: State vector.
    :a: Matrix representation of quantum effect.
    :returns: Conditioned state vector.

    """
    u: V = v @ a
    return u / np.linalg.norm(u)


WhichEigs = Literal['LA', 'SA', 'LM', 'SM', 'LR', 'SR', 'LI', 'SI']


# TODO: Include conversion to full array if a is sparse.
def eig_sorted(a: A, n: Optional[int] = None,
               which: WhichEigs = 'LR') -> Tuple[W, A]:
    """Compute eigenvalues and eigenvectors of a matrix using NumPy's eig
    solver. The eigenvalues are sorted according to the option passed in the
    string :which:, which is as per SciPy's eigs solver. The default is 'LR'
    (in descending order of real part).

    :a: Input matrix.
    :n: Number of eigenvalues to be returned. n = None returns all eigenvalues.
    :which: Sort order.
    :return: Tuple (w, v) where w are the sorted eigenvalues of a and v are
    corresponding eigenvectors.

    """
    wv = la.eig(a)
    if which in ('LA', 'SA', 'LR', 'SR'):
        idx = wv[0].real.argsort()
    elif which in ('LI', 'SI'):
        idx = wv[0].imag.argsort()
    elif which in ('LM', 'SM'):
        idx = np.abs(wv[0]).argsort()
    if n is None:
        n = len(a)
    if which in ('LA', 'LM', 'LR', 'LI'):
        idx = idx[::-1]
    w: W = wv[0][idx[0:n]]
    v: A = wv[1][:, idx[0:n]]
    return w, v


def eigs_sorted(a: A, n: Optional[int] = None, which: WhichEigs = 'LR',
                hermitian: bool = False)\
                        -> Tuple[W, A]:
    """Compute eigenvalues and eigenvectors of a matrix using SciPy's eigs
    solver. The eigenvalues are sorted by real part, in either ascending or
    descending order.

    :a: Input matrix.
    :n: Number of eigenvalues to be returned. n = None returns m - 1
    eigenvalues, where m is the number of columns of the input matrix.
    :which: Sort order.
    :hermitian: Use eigsh if set to true.
    :return: Tuple (w, v) where w are the sorted eigenvalues of a and v are
    corresponding eigenvectors.

    """
    if n is None:
        n = len(a) - 1
    if hermitian is True:
        wv = sla.eigsh(a, n, which=which)
    else:
        wv = sla.eigs(a, n, which=which)
    idx = np.arange(0, n)
    if which in ('LA', 'LM', 'LR', 'LI'):
        idx = idx[::-1]
    w: W = wv[0][idx[0:n]]
    v: A = wv[1][:, idx[0:n]]
    return w, v

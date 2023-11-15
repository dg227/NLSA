import numpy as np
from functools import partial
from nlsa.function_algebra2 import compose, identity
from nlsa.kernels import make_integral_operator, make_covariance_kernel,\
        make_exponential_rbf, left_normalize, right_normalize
from nlsa.vector_algebra2 import NPVectorAlgebra, NPMeasureFnAlgebra,\
        NPScalarField,\
        eval_at, counting_measure
from numba import jit, guvectorize
from numpy.typing import NDArray
from numpy.testing import assert_almost_equal
from scipy.sparse.linalg import LinearOperator, eigs
from typing import Callable, Literal, TypeVar

S = TypeVar('S')
T = TypeVar('T')
F = Callable[[S], T]

N = Literal[3]
R = np.float64
X = NDArray[R]
Y = NDArray[R]

def fst(x: R, y: R):
    y[0] = x[0]

gfst: Callable[[X], Y]\
        = guvectorize(['f8[:], f8[:]'], '(d) -> ()', nopython=True)(fst)


def sqeuclidean_core(x1: X, x2: X, y: Y):
    n1 = x1.shape[-2]
    n2 = x2.shape[-2]
    d = x1.shape[-1]
    for i in range(n1):
        for j in range(n2):
            y[i, j] = 0
            for k in range(d):
                y[i,j] += (x1[i, k] - x2[j, k]) ** 2


sqeuclidean: Callable[[X, X], Y] = guvectorize(['f8[:, :], f8[:, :], f8[:, :]'],
                                               '(n1, d), (n2, d) -> (n1, n2)',
                                               nopython=True)(sqeuclidean_core)


def geval_at(x: X, f: F[X, R]) -> Y:
    """Make evaluation functional at points or collection of points using Numba
    generalized ufuncs.

    """
    def g_core(y: X, fy: R):
        fy = f(y)

    g: Callable[[X], Y]\
        = guvectorize(['f8[:], f8[:]'], '(d) -> ()')(g_core)
    y: Y = g(x)
    return y


def geval_at2(x: X) -> Callable[[F[X, R]], Y]:
    """Make evaluation functional at points or collection of points using Numba
    generalized ufuncs.

    """
    @jit
    def evalx(f: F[X, R]) -> Y:
        return f(x)
    return evalx


if __name__ == "__main__":
    n = 3
    xs: X = np.array([[-1], [0], [1]], dtype=np.float64)
    print(sqeuclidean(xs, xs))


    # incl = eval_at(xs)
    # print(incl(gfst))
    # jincl = geval_at2(xs)
    # print(jincl(gfst))
    # print(gfst(xs))
    # ev = geval_at2(xs)
    # print(ev(gfst))

    # ys = np.empty((3, 1))
    # idty(xs, ys)
    # print(idty(xs, ys))
    # print(gidty(xs))

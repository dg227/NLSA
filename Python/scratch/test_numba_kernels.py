import numpy as np
import numba
from functools import partial
from nlsa.function_algebra2 import compose, identity
from nlsa.kernels import make_integral_operator, make_covariance_kernel,\
        make_exponential_rbf, left_normalize, right_normalize
from nlsa.numba.vector_algebra import NPVectorAlgebra, NPMeasureFnAlgebra,\
        NPScalarField,\
        eval_at, counting_measure, sqeuclidean
from numpy.typing import NDArray
from numpy.testing import assert_almost_equal
from scipy.sparse.linalg import LinearOperator, eigs
from typing import Callable, Literal, TypeVar

S = TypeVar('S')
T = TypeVar('T')
F = Callable[[S], T]

if __name__ == "__main__":
    N = Literal[16]
    R = np.float64
    X = NDArray[R]
    Y = NDArray[R]

    n = 3
    xs: X = np.array([[-1], [0], [1]], dtype=np.float64)
    ell2: NPMeasureFnAlgebra[X, N, R] =\
        NPMeasureFnAlgebra(dim=n,
                           dtype=np.float64,
                           inclusion_map=eval_at(xs),
                           measure=counting_measure)
    rbf = make_exponential_rbf(NPScalarField(dtype=np.float64),
                               bandwidth=np.array([1], dtype=np.float64))
    k: Callable[[X, X], Y] = compose(rbf, sqeuclidean)
    # k_jit = numba.jit(k, nopython=True)
    # kn: Callable[[X, X], Y] = left_normalize(ell2, k)
    # # kn: Callable[[X, X], Y] = right_normalize(ell2, k)
    x1 = np.array([[0], [0], [0]], dtype=np.float64)
    x2 = np.array([[1], [1]], dtype=np.float64)
    # k_op = make_integral_operator(ell2, kn)
    # l_op = LinearOperator((n, n), matvec=compose(ell2.incl, k_op))
    # lu = l_op.matvec(ell2.unit().T)
    # print(lu)
    # [evals, evecs] = eigs(l_op, 1, which='LM')
    # ku = k_op(ell2.unit())
    # print(ku(x1[0]))
    # print(ell2.incl(ku))
    # print(ell2.incl(compose(np.transpose, identity)))

    n = 2**15
    thetas = np.linspace(0, 2*np.pi, n)
    xs: X = np.column_stack((np.cos(thetas), np.sin(thetas)))
    ell2: NPMeasureFnAlgebra[X, N, R] =\
        NPMeasureFnAlgebra(dim=n,
                           dtype=np.float64,
                           inclusion_map=eval_at(xs),
                           measure=counting_measure)
    rbf = make_exponential_rbf(NPScalarField(dtype=np.float64),
                               bandwidth=np.array([0.2], dtype=np.float64))
    k: Callable[[X, X], Y] = compose(rbf, sqeuclidean)
    # kl: Callable[[X, X], Y] = left_normalize(ell2, k)
    kl_op = make_integral_operator(ell2, k)
    gl_op = compose(ell2.incl, kl_op)
    a = LinearOperator(shape=(n, n), dtype=np.float64, matvec=gl_op)
    print(n)
    [evals, evecs] = eigs(a, 5, which='LM')
    print(evals)

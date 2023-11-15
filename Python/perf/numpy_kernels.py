import numpy as np
import time
from nlsa.abstract_algebra2 import ldivide_by
from nlsa.function_algebra2 import compose
from nlsa.kernels import dmsym_normalize, make_integral_operator,\
        make_exponential_rbf
from nlsa.numpy.vector_algebra import VectorAlgebra, MeasureFnAlgebra,\
        ScalarField,\
        eval_at, counting_measure, sqeuclidean
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, eigs, eigsh
from typing import Callable, Literal

if __name__ == "__main__":
    R = np.float32
    N = Literal[64]
    X = NDArray[R]
    Xs = NDArray[R]
    Y = NDArray[R]

    n = 2**4
    n_eigs = 5
    alpha = '0.5'
    dtype = np.float32
    thetas = np.linspace(0, 2 * np.pi, n)
    xs: Xs = np.column_stack((np.cos(thetas), np.sin(thetas)))

    r_n: VectorAlgebra[N, R] = VectorAlgebra(dim=n, dtype=dtype)
    ell2: MeasureFnAlgebra[X, N, R] =\
        MeasureFnAlgebra(dim=n,
                         dtype=dtype,
                         inclusion_map=eval_at(xs),
                         measure=counting_measure)
    rbf = make_exponential_rbf(ScalarField(dtype=dtype),
                               bandwidth=dtype(0.05))
    k: Callable[[X, X], Y] = compose(rbf, sqeuclidean)
    p: Callable[[X, X], Y] = dmsym_normalize(ell2, alpha, k)
    p_op = make_integral_operator(ell2, p)
    g_op = compose(ell2.incl, p_op)

    print('NumPy backend')
    print(f'n = {n}')
    print(f'n_eigs = {n_eigs}')
    a = LinearOperator(shape=(n, n), dtype=dtype, matvec=g_op)
    start_time = time.perf_counter()
    [unsorted_evals, unsorted_evecs] = eigsh(a, n_eigs, which='LM')
    end_time = time.perf_counter()
    print(f'Eigendecomposition took {end_time - start_time:.3E} s')

    start_time = time.perf_counter()
    isort = np.argsort(np.real(unsorted_evals))
    evals = np.real(unsorted_evals[isort[::-1]])
    laplacian_evals = (1/evals - 1) / (1/evals[1] - 1)
    symmetric_evecs = unsorted_evecs[:, isort[::-1]]
    from_sym = ldivide_by(r_n, symmetric_evecs[:, 0])
    markov_evecs = from_sym(symmetric_evecs)
    end_time = time.perf_counter()
    print(f'Sorting took {end_time - start_time:.3E} s')
    print('First 5 Laplacian eigenvalues:')
    print(evals[0 : 5])

import numpy as np
import nlsa.abstract_algebra2 as alg
from nlsa.abstract_algebra2 import ldivide_by
from nlsa.function_algebra2 import compose, identity
from nlsa.kernels import make_integral_operator, make_covariance_kernel,\
        make_exponential_rbf, left_normalize, right_normalize,\
        right_sqrt_normalize, dmsym_normalize
from nlsa.numpy.vector_algebra import MeasureFnAlgebra, ScalarField,\
        VectorAlgebra,\
        eval_at, counting_measure, sqeuclidean
from numpy.typing import NDArray
from numpy.testing import assert_almost_equal
from scipy.sparse.linalg import LinearOperator, eigs
from typing import Callable, Literal, TypeVar

S = TypeVar('S')
T = TypeVar('T')
F = Callable[[S], T]


def test_integral_operator_int():
    N = Literal[3]
    Z = np.int32
    X = NDArray[np.int32]
    Y = NDArray[np.int32]
    n = 3

    z_n: VectorAlgebra[N, Z] = VectorAlgebra(dim=n, dtype=np.int32)
    xs: X = np.array([[-1], [0], [1]], dtype=np.int32)
    ell2: MeasureFnAlgebra[X, N, Z] =\
        MeasureFnAlgebra(dim=n,
                         dtype=np.int32,
                         inclusion_map=eval_at(xs),
                         measure=counting_measure)
    k: Callable[[X, X], Y] = make_covariance_kernel(z_n)
    u = np.array([1, 1, 1], dtype=np.int32)
    v = np.array([0, 0, 0], dtype=np.int32)
    idty: Callable[[X], X] = compose(np.transpose, identity)
    assert np.all(ell2.mul(u, v) == np.array([0, 0, 0]))
    assert np.all(ell2.incl(idty) == np.array([[-1, 0, 1]]))
    assert ell2.integrate(u) == 3
    k_op = make_integral_operator(ell2, k)
    f = k_op(v)
    x = np.array([0], dtype=np.int32)
    assert np.all(f(x) == 0)


def test_integral_operator_float():
    N = Literal[3]
    R = np.float64
    X = NDArray[R]
    Y = NDArray[R]

    n = 3
    r: ScalarField[R] = ScalarField(dtype=np.float64)
    xs: X = np.array([[-1], [0], [1]], dtype=np.float64)
    ell2: MeasureFnAlgebra[X, N, R] =\
        MeasureFnAlgebra(dim=n,
                         dtype=np.float64,
                         inclusion_map=eval_at(xs),
                         measure=counting_measure)
    rbf = make_exponential_rbf(r, bandwidth=np.array([1], dtype=np.float64))
    k: Callable[[X, X], Y] = compose(rbf, sqeuclidean)
    u = np.array([1, 1, 1], dtype=np.float64)
    v = np.array([0, 0, 0], dtype=np.float64)
    idty: Callable[[X], X] = compose(np.transpose, identity)
    assert_almost_equal(ell2.mul(u, v), np.array([0, 0, 0]))
    assert_almost_equal(ell2.incl(idty), np.array([[-1, 0, 1]]))
    assert_almost_equal(ell2.integrate(u), 3)
    k_op = make_integral_operator(ell2, k)
    f = k_op(v)
    x = np.array([[0], [1]], dtype=np.float64)
    y = f(x)
    assert_almost_equal(y[0], 0)


def test_normalized_integral_operator_float():
    N = Literal[3]
    R = np.float64
    X = NDArray[R]
    Y = NDArray[R]

    n = 3
    r: ScalarField[R] = ScalarField(dtype=np.float64)
    xs: X = np.array([[-1], [0], [1]], dtype=np.float64)
    ell2: MeasureFnAlgebra[X, N, R] =\
        MeasureFnAlgebra(dim=n,
                         dtype=np.float64,
                         inclusion_map=eval_at(xs),
                         measure=counting_measure)
    assert isinstance(ell2, alg.ImplementsMeasureUnitalFnAlgebra)
    rbf = make_exponential_rbf(r, bandwidth=np.array([1], dtype=np.float64))
    k: Callable[[X, X], Y] = compose(rbf, sqeuclidean)
    kl: Callable[[X, X], Y] = left_normalize(ell2, k)
    kr: Callable[[X, X], Y] = right_normalize(ell2, k)
    v = np.array([0, 0, 0], dtype=np.float64)
    x1 = np.array([[0], [0], [0], [0]], dtype=np.float64)
    x2 = np.array([[1], [1]], dtype=np.float64)
    kl_op = make_integral_operator(ell2, kl)
    kr_op = make_integral_operator(ell2, kr)
    assert_almost_equal(ell2.incl(kl_op(ell2.unit())),
                        np.atleast_2d(ell2.unit()))
    assert_almost_equal(ell2.incl(kr_op(v)), np.atleast_2d(v))
    kl_mat = kl(x1, x2)
    kr_mat = kr(x1, x2)
    assert np.all(kl_mat.shape == (4, 2))
    assert np.all(kr_mat.shape == (4, 2))


def test_heat_operator_circle():
    N = Literal[16]
    R = np.float64
    X = NDArray[R]
    Y = NDArray[R]

    n = 16
    thetas = np.linspace(0, 2*np.pi, n)
    xs: X = np.column_stack((np.cos(thetas), np.sin(thetas)))
    ell2: MeasureFnAlgebra[X, N, R] =\
        MeasureFnAlgebra(dim=n,
                         dtype=np.float64,
                         inclusion_map=eval_at(xs),
                         measure=counting_measure)
    assert isinstance(ell2, alg.ImplementsMeasureUnitalFnAlgebra)
    rbf = make_exponential_rbf(ScalarField(dtype=np.float64),
                               bandwidth=np.array([0.2], dtype=np.float64))
    k: Callable[[X, X], Y] = compose(rbf, sqeuclidean)
    kl: Callable[[X, X], Y] = left_normalize(ell2, k)
    kl_op = make_integral_operator(ell2, kl)
    gl_op = LinearOperator(shape=(n, n),
                           dtype=np.float64,
                           matvec=compose(ell2.incl, kl_op))
    [evals, evecs] = eigs(gl_op, 5, which='LM')
    print(evals)


def test_diffusion_maps_alpha_one_circle():
    R = np.float64
    N = Literal[64]
    X = NDArray[R]
    Xs = NDArray[R]
    Y = NDArray[R]

    n = 2**6
    n_eigs = 5
    alpha = '1'
    dtype = np.float64
    thetas = np.linspace(0, 2*np.pi, n)
    xs: X = np.column_stack((np.cos(thetas), np.sin(thetas)))

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

    a = LinearOperator(shape=(n, n), dtype=dtype, matvec=g_op)
    [unsorted_evals, unsorted_evecs] = eigs(a, n_eigs, which='LM')
    isort = np.argsort(np.real(unsorted_evals))
    evals = np.real(unsorted_evals[isort[::-1]])
    laplacian_evals = (1/evals - 1) / (1/evals[1] - 1)
    symmetric_evecs = unsorted_evecs[:, isort[::-1]]
    from_sym = ldivide_by(r_n, symmetric_evecs[:, 0])
    markov_evecs = from_sym(symmetric_evecs)
    assert_almost_equal(markov_evecs[:, 0], 1)
    assert_almost_equal(laplacian_evals[0], 0)
    assert_almost_equal(laplacian_evals[1:3], 1, decimal=2)
    assert_almost_equal(laplacian_evals[3:5], 4, decimal=1)


def test_diffusion_maps_alpha_half_circle():
    R = np.float64
    N = Literal[64]
    X = NDArray[R]
    Xs = NDArray[R]
    Y = NDArray[R]

    n = 2**6
    n_eigs = 5
    alpha = '0.5'
    dtype = np.float64
    thetas = np.linspace(0, 2*np.pi, n)
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

    a = LinearOperator(shape=(n, n), dtype=dtype, matvec=g_op)
    [unsorted_evals, unsorted_evecs] = eigs(a, n_eigs, which='LM')
    isort = np.argsort(np.real(unsorted_evals))
    evals = np.real(unsorted_evals[isort[::-1]])
    laplacian_evals = (1/evals - 1) / (1/evals[1] - 1)
    symmetric_evecs = unsorted_evecs[:, isort[::-1]]
    from_sym = ldivide_by(r_n, symmetric_evecs[:, 0])
    markov_evecs = from_sym(symmetric_evecs)
    assert_almost_equal(markov_evecs[:, 0], 1)
    assert_almost_equal(laplacian_evals[0], 0)
    assert_almost_equal(laplacian_evals[1:3], 1, decimal=1)
    assert_almost_equal(laplacian_evals[3:5], 4, decimal=0)

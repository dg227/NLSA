import jax
import jax.numpy as jnp
import time
from functools import partial
from jax import Array, grad, jit, vmap
from nlsa.abstract_algebra2 import ldivide_by, multiply_by
from nlsa.function_algebra2 import compose
from nlsa.kernels import dm_normalize, dmsym_normalize, \
        make_integral_operator, make_exponential_rbf, make_tuning_objective
from nlsa.jax.stats import make_von_mises_density
from nlsa.jax.vector_algebra import VectorAlgebra, MeasureFnAlgebra, \
        ScalarField, counting_measure, sqeuclidean, veval_at, \
        make_vector_analysis_operator, make_fn_synthesis_operator
from scipy.sparse.linalg import LinearOperator, eigsh
from typing import Callable, Literal, TypeVar

N = Literal[32]
M = Literal[8]

R = Array
Rm = Array
T1 = Array
T1n = Array
X = Array
Xn = Array
V = Array
Vm = Array
W = Array
K = jnp.float32

S = TypeVar('S')
T = TypeVar('T')
F = Callable[[S], T]

Alpha = Literal['0', '0.5', '1']

n: N = 2**6
n_bandwidth: int = 16
log10_bandwidth_lims: tuple[float, float] = (-3., 3.)
m_eigs: int = 8
m: M = m_eigs
n_test = 128
alpha: Alpha = '0.5'
kappa: float = 1.
dtype = jnp.float32


@jit
@vmap
def embed_r2(theta: T1) -> X:
    """Embed points in the circle into R2."""
    return jnp.array([jnp.cos(theta), jnp.sin(theta)])


if __name__ == "__main__":

    print(f'JAX {jax.default_backend()} backend')
    print(jax.devices())
    print(dtype)
    print(f'n = {n}')
    print(f'm_eigs = {m_eigs}')

    scl = ScalarField(dtype=dtype)

    thetas: T1n = jnp.linspace(0, 2 * jnp.pi, n)
    xs: Xn = embed_r2(thetas)
    ell2: MeasureFnAlgebra[X, N, K] = \
        MeasureFnAlgebra(dim=n,
                         dtype=dtype,
                         inclusion_map=veval_at(xs),
                         measure=counting_measure)

    start_time = time.perf_counter()
    log10_bandwidths = jnp.linspace(log10_bandwidth_lims[0],
                                    log10_bandwidth_lims[1], n_bandwidth)
    shape_func = partial(make_exponential_rbf, scl)

    def k_func(epsilon: R) -> Callable[[X, X], R]:
        return  compose(shape_func(epsilon), sqeuclidean)

    k_tune = jit(make_tuning_objective(ell2, k_func, grad))
    est_dims = jnp.array([2.*k_tune(epsilon) for epsilon in log10_bandwidths])
    i_opt = jnp.argmax(est_dims)
    bandwidth = 10. ** log10_bandwidths[i_opt]
    end_time = time.perf_counter()
    print(f'Kernel tuning took {end_time - start_time:.3e} s')
    print(f'Optimal bandwidth index: {i_opt}')
    print(f'Optimal bandwidth: {bandwidth:.3e}')
    print(f'Estimated dimension: {est_dims[i_opt]:.3e}')

    rbf: Callable[[X, X], R] = make_exponential_rbf(scl, bandwidth)
    k: Callable[[X, X], R] = compose(rbf, sqeuclidean)
    psym: Callable[[X, X], R] = dmsym_normalize(ell2, alpha, k)
    psym_op: Callable[[V], F[X, R]] = make_integral_operator(ell2, psym)
    gsym_op: F[V, V] = compose(veval_at(xs), psym_op)
    a = LinearOperator(shape=(n, n), dtype=dtype, matvec=jit(gsym_op))

    start_time = time.perf_counter()
    unsorted_evals: Rm
    unsorted_evecs: Vm
    unsorted_evals, unsorted_evecs = eigsh(a, m_eigs, which='LA')
    end_time = time.perf_counter()
    print(f'Eigendecomposition took {end_time - start_time:.3e} s')

    start_time = time.perf_counter()
    isort = jnp.argsort(unsorted_evals)
    lambs = unsorted_evals[isort[::-1]]
    laplacian_evals = (1./lambs - 1.) / (1./lambs[1] - 1.)
    phi_syms = unsorted_evecs[:, isort[::-1]]
    from_sym: F[Vm, Vm] = jit(vmap(ldivide_by(ell2, phi_syms[:, 0]),
                                   in_axes=1, out_axes=1))

    dual_from_sym: F[Vm, Vm] = jit(vmap(multiply_by(ell2, phi_syms[:, 0]),
                                        in_axes=1, out_axes=1))
    phis = from_sym(phi_syms)
    phi_duals = dual_from_sym(phi_syms)
    end_time = time.perf_counter()
    print(f'Sorting took {end_time - start_time:.3e} s')
    print('First 5 Laplacian eigenvalues:')
    print(laplacian_evals[0:5])

    start_time = time.perf_counter()
    r_m: VectorAlgebra[N, K] = VectorAlgebra(dim=m, dtype=dtype)
    an = make_vector_analysis_operator(ell2, phi_duals[:, 0:m])
    p: Callable[[X, X], R] = dm_normalize(ell2, alpha, k)
    p_op: Callable[[F[X, R]], V] = make_integral_operator(ell2, p)
    vp_op = vmap(lambda v, x: p_op(v)(x), in_axes=(1, None))
    basis = partial(vp_op, phis[:, 0:m])
    synth = make_fn_synthesis_operator(basis)
    nystrom = compose(synth, ldivide_by(r_m, lambs[0:m]))
    f = jit(vmap(make_von_mises_density(kappa, jnp.pi)))
    f_ell2 = f(thetas)
    f_phi = an(f_ell2)
    f_pred = jit(vmap(nystrom(f_phi)))
    end_time = time.perf_counter()
    print(f'Out-of-sample extension took {end_time - start_time:.3e} s')

    start_time = time.perf_counter()
    theta_tests = jnp.linspace(0.1, 0.1 + 2 * jnp.pi, n_test)
    x_tests: Xn = embed_r2(theta_tests)
    y_trues = f(theta_tests)
    y_preds = f_pred(x_tests)
    rmse = jnp.sqrt(jnp.sum((y_trues - y_preds) ** 2) / n_test)
    end_time = time.perf_counter()
    print(f'Prediction took {end_time - start_time:.3e} s')
    print('First 5 true values:')
    print(y_trues[0:5])
    print('First 5 predicted values:')
    print(y_preds[0:5])
    print(f'Normalized RMSE = {rmse:.3e}')

import jax
import jax.numpy as jnp
import time
from jax import Array, jit, vmap
from nlsa.abstract_algebra2 import ldivide_by
from nlsa.function_algebra2 import compose
from nlsa.kernels import dmsym_normalize, make_integral_operator, \
        make_exponential_rbf
from nlsa.jax.vector_algebra import VectorAlgebra, MeasureFnAlgebra, \
        ScalarField, counting_measure, sqeuclidean, jeval_at
from scipy.sparse.linalg import LinearOperator, eigsh
from typing import Callable, Literal, TypeVar

S = TypeVar('S')
T = TypeVar('T')
F = Callable[[S], T]


if __name__ == "__main__":
    """Parallelization using jit.

    Type variables postfixed by s indicate vectorization. Type variables
    postfixed by _ indicate sharding across devices.
    """

    R = jnp.float32
    N = Literal[16]
    X = Array
    Xs = Array
    Xs_ = Array
    Y = Array
    V = Array
    V_ = Array

    n: N = 2**17
    n_eigs = 500
    alpha = '0.5'
    dtype = jnp.float32
    devices = jax.local_devices()
    device_cpu = jax.devices('cpu')[0]
    n_par = len(devices)
    n_chunk = n // n_par

    # shard_x and shard_v could potentially be replaced by a more elegant
    # common sharding implementation.

    def shard_x(xs: Xs) -> Xs_:
        xs_ = jax.device_put_sharded([jnp.reshape(xs, (n_par, n_chunk, 2))[i]
                                      for i in range(len(devices))], devices)
        return xs_

    def shard_v(v: V) -> V_:
        v_ = jax.device_put_sharded([jnp.reshape(v, (n_par, n_chunk))[i]
                                     for i in range(len(devices))], devices)
        return v_

    print(f'JAX {jax.default_backend()} backend')
    print(jax.devices())
    print(f'n = {n}')
    print(f'n_eigs = {n_eigs}')
    print(f'n_par = {n_par}')

    with jax.default_device(device_cpu):
        thetas = jnp.linspace(0, 2 * jnp.pi, n)
        xs: Xs = jnp.column_stack((jnp.cos(thetas), jnp.sin(thetas)))

    xs_: Xs_ = shard_x(xs)
    inclusion_map: Callable[[F[X, R]], V_] = jeval_at(xs_)
    measure: F[V_, R] = compose(counting_measure, vmap(counting_measure))
    r_n: VectorAlgebra[N, R] = VectorAlgebra(dim=n, dtype=dtype)
    ell2: MeasureFnAlgebra[X, N, R] =\
        MeasureFnAlgebra(dim=n,
                         dtype=dtype,
                         inclusion_map=inclusion_map,
                         measure=measure)
    rbf: Callable[[X, X], R] = make_exponential_rbf(ScalarField(dtype=dtype),
                                                    bandwidth=dtype(0.2))
    u: V = ell2.unit()
    k: Callable[[X, X], R] = compose(rbf, sqeuclidean)
    p: Callable[[X, X], R] = dmsym_normalize(ell2, alpha, k,
                                             unit=shard_v(ell2.unit()))
    p_op = make_integral_operator(ell2, p)
    g_op = compose(inclusion_map, p_op)
    j_op = jax.jit(g_op)

    def matvec(v: V) -> V:
        vs = jax.device_put_sharded([jnp.reshape(v, (n_par, n_chunk))[i]
                                     for i in range(len(devices))],
                                    devices)
        ws = j_op(vs)
        w = jax.device_put(jnp.reshape(ws, n), device_cpu)
        return w

    a = LinearOperator(shape=(n, n), dtype=dtype, matvec=matvec)

    start_time = time.perf_counter()
    [unsorted_evals, unsorted_evecs] = eigsh(a, n_eigs, which='LA')
    end_time = time.perf_counter()
    print(f'Eigendecomposition took {end_time - start_time:.3E} s')

    start_time = time.perf_counter()
    isort = jnp.argsort(unsorted_evals)
    evals = unsorted_evals[isort[::-1]]
    laplacian_evals = (1/evals - 1) / (1/evals[1] - 1)
    symmetric_evecs = unsorted_evecs[:, isort[::-1]]
    from_sym = jit(vmap(ldivide_by(r_n, symmetric_evecs[:, 0]),
                        in_axes=1, out_axes=1))
    markov_evecs = from_sym(symmetric_evecs)
    end_time = time.perf_counter()
    print(f'Sorting took {end_time - start_time:.3E} s')
    print('First 5 Laplacian eigenvalues:')
    print(evals[0:5])

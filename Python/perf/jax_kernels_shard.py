import jax
import jax.numpy as jnp
import time
from functools import partial
from jax import Array, jit, pmap, vmap
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map as shmap
from jax.sharding import Mesh, PositionalSharding, PartitionSpec as P
from nlsa.abstract_algebra2 import ldivide_by
from nlsa.function_algebra2 import compose
from nlsa.kernels import dmsym_normalize, make_integral_operator, \
        make_exponential_rbf
from nlsa.jax.vector_algebra import VectorAlgebra, MeasureFnAlgebra, \
        ScalarField, counting_measure, peval_at, sqeuclidean, veval_at, \
        v2eval_at, sheval_at, jeval_at
from scipy.sparse.linalg import LinearOperator, eigsh
from typing import Callable, Literal, TypeVar

S = TypeVar('S')
T = TypeVar('T')
F = Callable[[S], T]


def do_v1():
    """Parallelization using shard_map."""
    R = jnp.float32
    N = Literal[16]
    X = Array
    Xs = Array
    Y = Array
    V = Array

    # n = 16000
    n = 85000
    n_eigs = 50
    n_par = 4

    print(f'JAX {jax.default_backend()} backend')
    print(jax.devices())
    print(f'n = {n}')
    print(f'n_eigs = {n_eigs}')
    print(f'n_par = {n_par}')

    devices = jax.local_devices()
    device_cpu = jax.devices('cpu')[0]
    alpha = '0.5'
    dtype = jnp.float32

    with jax.default_device(device_cpu):
        thetas = jnp.linspace(0, 2 * jnp.pi, n)
        xs = jnp.column_stack((jnp.cos(thetas), jnp.sin(thetas)))

    r_n: VectorAlgebra[N, R] = VectorAlgebra(dim=n, dtype=dtype)
    ell2: MeasureFnAlgebra[X, N, R] =\
        MeasureFnAlgebra(dim=n,
                         dtype=dtype,
                         inclusion_map=veval_at(xs),
                         measure=counting_measure)
    rbf = make_exponential_rbf(ScalarField(dtype=dtype),
                               bandwidth=dtype(0.2))
    k: Callable[[X, X], Y] = compose(rbf, sqeuclidean)
    p: Callable[[X, X], Y] = dmsym_normalize(ell2, alpha, k)
    p_op = make_integral_operator(ell2, p)
    n_chunk = n // n_par
    shards = [jnp.reshape(xs, (n_par, n_chunk, 2))[i]
              for i in range(len(devices))]
    xs_ = jax.device_put_sharded(shards, devices)
    g_op = compose(peval_at(xs_), p_op)
    j_op = jit(g_op)

    def matvec(v: V) -> V:
        # vs = jax.device_put_replicated(v, devices)
        ws = j_op(v)
        w = jax.device_put(jnp.reshape(ws, n), device_cpu)
        return w

    # print(matvec(v).shape)
    # xs_ = jax.device_put_replicated(xs, devices)
    # print(xs_.shape)
    # v = jax.device_put(ell2.incl(partial(k, jnp.array([0, 0]))), device_cpu)
    # print(v.shape)
    # w = matvec(v)
    # print(w.shape)
    # w2 = matvec(w)
    # print(w2.shape)

    a = LinearOperator(shape=(n, n), dtype=dtype, matvec=matvec)

    start_time = time.perf_counter()
    [unsorted_evals, unsorted_evecs] = eigsh(a, n_eigs, which='LA')
    end_time = time.perf_counter()
    print(f'Eigendecomposition took {end_time - start_time:.3E} s')

    # start_time = time.perf_counter()
    # isort = jnp.argsort(unsorted_evals)
    # evals = unsorted_evals[isort[::-1]]
    # laplacian_evals = (1/evals - 1) / (1/evals[1] - 1)
    # symmetric_evecs = unsorted_evecs[:, isort[::-1]]
    # from_sym = jit(vmap(ldivide_by(r_n, symmetric_evecs[:, 0]),
    #                     in_axes=1, out_axes=1))
    # markov_evecs = from_sym(symmetric_evecs)
    # end_time = time.perf_counter()
    # print(f'Sorting took {end_time - start_time:.3E} s')
    # print('First 5 Laplacian eigenvalues:')
    # print(evals[0 : 5])


def do_v2():
    R = jnp.float32
    N = Literal[16]
    X = Array
    Xs = Array
    Y = Array
    V = Array

    # n = 16000
    n = 2**17
    n_eigs = 50
    n_par = 4

    print(f'JAX {jax.default_backend()} backend')
    print(jax.devices())
    print(f'n = {n}')
    print(f'n_eigs = {n_eigs}')
    print(f'n_par = {n_par}')

    device_cpu = jax.devices('cpu')[0]
    devices = jax.local_devices()
    mesh = Mesh(devices, axis_names=('i'))

    alpha = '0.5'
    dtype = jnp.float32

    with jax.default_device(device_cpu):
        thetas = jnp.linspace(0, 2 * jnp.pi, n)
        xs = jnp.column_stack((jnp.cos(thetas), jnp.sin(thetas)))

    n_chunk = n // n_par
    r_n: VectorAlgebra[N, R] = VectorAlgebra(dim=n, dtype=dtype)
    ell2: MeasureFnAlgebra[X, N, R] =\
        MeasureFnAlgebra(dim=n,
                         dtype=dtype,
                         inclusion_map=sheval_at(xs),
                         measure=counting_measure)
    rbf = make_exponential_rbf(ScalarField(dtype=dtype),
                               bandwidth=dtype(0.2))
    u = ell2.unit()
    k: Callable[[X, X], Y] = compose(rbf, sqeuclidean)
    p: Callable[[X, X], Y] = dmsym_normalize(ell2, alpha, k)
    p_op = make_integral_operator(ell2, k)

    # g_op = compose(sheval_at(xs, axis_name='j'), p_op)
    g_op = compose(veval_at(xs), p_op)
    j_op = jit(g_op)


    u = ell2.unit()
    print(j_op(u).shape)



    # f = p_op(u)
    # vf = jit(vmap(f))

    # print(j_op(u))
    # v = j_op(u)
    # print(v.shape)
    # v = jax.device_put_replicated(jnp.ones(n_chunk), devices)
    # print(xs.shape)
    # kx = partial(k, jnp.ones(2))
    # print(ell2.incl(kx).shape)
    # print(u.shape)
    # f = p_op(u)
    # print(f(jnp.ones(2)))
    # print(v.shape)
    # print(j_op(v).shape)

    def matvec(v: V) -> V:
        # vs = jax.device_put_sharded([jnp.reshape(v, (n_par, n_chunk))[i]\
        #                              for i in range(len(devices))],
        #                             devices)
        ws = j_op(vs)
        w = jax.device_put(jnp.reshape(ws, n), device_cpu)
        return w


    a = LinearOperator(shape=(n, n), dtype=dtype, matvec=j_op)

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
    print(evals[0 : 5])

def do_v3():
    """Parallelization using jit."""
    R = jnp.float32
    Rs = Array
    N = Literal[16]
    X = Array
    Xs = Array
    Xs_ = Array
    Y = Array
    V = Array
    V_ = Array
    Vs = Array

    n = 128000
    # n = 2**10
    n_eigs = 5
    alpha = '0.5'
    dtype = jnp.float32
    devices = jax.local_devices()
    device_cpu = jax.devices('cpu')[0]
    n_par = len(devices)
    n_chunk = n // n_par

    def shard_x(xs: Xs) -> Xs_:
        xs_: Xs_ = jax.device_put_sharded([jnp.reshape(xs, (n_par, n_chunk, 2))[i] \
                              for i in range(len(devices))],
                                     devices)
        return xs_

    def shard_v(v: V) -> V_:
        v_ = jax.device_put_sharded([jnp.reshape(v, (n_par, n_chunk))[i] \
                              for i in range(len(devices))],
                                     devices)
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
    inclusion_map: Callable[[F[X, R]], Vs_] = jeval_at(xs_)
    measure: F[Vs_, R] = compose(counting_measure, vmap(counting_measure))
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

    # print(xs_.shape)
    # f = p_op(us)
    # vs = inclusion_map(f)
    # print(vs.shape)
    # print(j_op(v).shape)

    def matvec(v: V) -> V:
        vs = jax.device_put_sharded([jnp.reshape(v, (n_par, n_chunk))[i]\
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
    print(evals[0 : 5])

if __name__ == "__main__":
    do_v3()

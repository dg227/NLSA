import jax
import jax.numpy as jnp
import time
from functools import partial
from jax import Array, jit, vmap
from nlsa.abstract_algebra2 import gelfand, make_qeval
from nlsa.function_algebra2 import compose
from nlsa.kernels import make_exponential_rbf
from nlsa.jax.matrix_algebra import MatrixAlgebra
from nlsa.jax.vector_algebra import MeasureFnAlgebra, ScalarField,\
        counting_measure, sqeuclidean, veval_at
from typing import Callable, Literal

if __name__ == "__main__":
    R = jnp.float32
    N = Literal[64]
    X = Array
    Xs = Array
    Y = Array

    n = 2**4
    dtype = jnp.float32
    thetas = jnp.linspace(0, 2*jnp.pi, n)
    xs: Xs = jnp.column_stack((jnp.cos(thetas), jnp.sin(thetas)))
    ell2: MeasureFnAlgebra[X, N, R] =\
        MeasureFnAlgebra(dim=n,
                         dtype=dtype,
                         inclusion_map=veval_at(xs),
                         measure=counting_measure)
    mat: MatrixAlgebra[N, R] = MatrixAlgebra(dim=n, dtype=dtype, hilb=ell2)
    rbf = make_exponential_rbf(ScalarField(dtype=dtype),
                               bandwidth=dtype(0.2))
    k: Callable[[X, X], Y] = compose(rbf, sqeuclidean)
    feat: Callable[[X], MeasureFnAlgebra[X, N, R]]\
        = compose(ell2.incl, lambda x: partial(k, x))
    a = jnp.eye(n)
    qeval_a = compose(gelfand(mat, a), make_qeval(mat, feat))
    vqeval_a = jit(vmap(qeval_a))

    print(f'JAX {jax.default_backend()} backend')
    print(f'n = {n}')
    start_time = time.perf_counter()
    y = vqeval_a(xs)
    end_time = time.perf_counter()
    print(f'Quantum evaluation took {end_time - start_time:.3E} s')
    print(f'Output array shape: {y.shape}')

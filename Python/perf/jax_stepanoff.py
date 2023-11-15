import jax.numpy as jnp
import jax.random as jrd
import time
from functools import partial
from jax import Array, jit
from nlsa.jax.dynamics import make_stepanoff_generator_fourier, \
        stepanoff_generator_matrix
from nlsa.jax.matrix_algebra import MatrixAlgebra
from typing import Callable

V = Array
M = Array
I2 = tuple[int, int]


def make_stepanoff_generator_fourier_list(alpha: float, k: I2,
                                          hermitian: bool = False)\
        -> Callable[[V], V]:
    """Make generator of Stepanoff flow in the Fourier basis of the 2-torus.

    This implementation computes the action of the generator using list
    comprehension. It appears to be significantly slower than
    make_stepanoff_generator_fourier that performs in-place mutation of JAX
    arrays.
    """

    k1 = k[0]
    k2 = k[1]
    n1 = 2*k1 + 1
    n2 = 2*k2 + 1

    if hermitian:
        const = 1.
    else:
        const = 1j

    def gen(u: V) -> V:
        a = jnp.reshape(u, (n1, n2))

        v10 = jnp.array([[const * i1 * a[i1, i2]
                          for i2 in range(-k2, k2 + 1)]
                         for i1 in range(-k1, k1 + 1)])
        v11_ = jnp.array([[-.5 * const * alpha * (i1 - 1) * a[i1 - 1, i2 + 1]
                           for i2 in range(-k2, k2)]
                          for i1 in range(-k1 + 1, k1 + 1)])
        v11 = jnp.pad(v11_, ((1, 0), (0, 1)), 'constant')
        v12_ = jnp.array([[-.5 * const * alpha * (i1 + 1) * a[i1 + 1, i2 - 1]
                           for i2 in range(-k2 + 1, k2 + 1)]
                          for i1 in range(-k1, k1)])
        v12 = jnp.pad(v12_, ((0, 1), (1, 0)), 'constant')
        v13_ = jnp.array([[-.5 * const * (1. - alpha) * i1 * a[i1, i2 - 1]
                           for i2 in range(-k2 + 1, k2 + 1)]
                          for i1 in range(-k1, k1 + 1)])
        v13 = jnp.pad(v13_, ((0, 0), (1, 0)), 'constant')
        v14_ = jnp.array([[-.5 * const * (1. - alpha) * i1 * a[i1, i2 + 1]
                           for i2 in range(-k2, k2)]
                          for i1 in range(-k1, k1 + 1)])
        v14 = jnp.pad(v14_, ((0, 0), (0, 1)), 'constant')

        v20 = jnp.array([[const * alpha * i2 * a[i1, i2]
                          for i2 in range(-k2, k2 + 1)]
                         for i1 in range(-k1, k1 + 1)])
        v21_ = jnp.array([[-.5 * const * alpha * (i2 + 1) * a[i1 - 1, i2 + 1]
                           for i2 in range(-k2, k2)]
                          for i1 in range(-k1 + 1, k1 + 1)])
        v21 = jnp.pad(v21_, ((1, 0), (0, 1)), 'constant')
        v22_ = jnp.array([[-.5 * const * alpha * (i2 - 1) * a[i1 + 1, i2 - 1]
                           for i2 in range(-k2 + 1, k2 + 1)]
                          for i1 in range(-k1, k1)])
        v22 = jnp.pad(v22_, ((0, 1), (1, 0)), 'constant')

        return jnp.reshape(v10 + v11 + v12 + v13 + v14 + v20 + v21 + v22,
                           n1 * n2)
    return gen


if __name__ == '__main__':

    alpha = jnp.sqrt(20.)
    k_max = (3, 3)
    n = (2*k_max[0] + 1) * (2*k_max[1] + 1)
    n_eval = 100
    if_print = False

    if n_eval > 0:
        seed = 1701
        key = jrd.PRNGKey(seed)

    f = jnp.ones(n)

    print(f'Max. wavenumber = {k_max}')
    print(f'Dimension = {n}')

    print('List comprehension:')
    v = jit(make_stepanoff_generator_fourier_list(alpha, k_max,
                                                  hermitian=True))
    start_time = time.perf_counter()
    g = v(f)
    g.block_until_ready()
    end_time = time.perf_counter()
    print(f'First Stepanoff evaluation took {end_time - start_time:.3E} s')
    if if_print:
        print(g)

    start_time = time.perf_counter()
    for _ in range(n_eval):
        key, subkey = jrd.split(key)
        key = subkey
        f = jrd.normal(key, (n,))
        v(f).block_until_ready()
    end_time = time.perf_counter()
    print(f'{n_eval} Stepanoff evaluations took {end_time - start_time:.3E} s')

    print('JAX Array operator:')
    v2 = jit(make_stepanoff_generator_fourier(alpha, k_max, hermitian=True))
    start_time = time.perf_counter()
    g2 = v2(f)
    g2.block_until_ready()
    end_time = time.perf_counter()
    print(f'First Stepanoff evaluation took {end_time - start_time:.3E} s')
    if if_print:
        print(g2)

    start_time = time.perf_counter()
    for _ in range(n_eval):
        key, subkey = jrd.split(key)
        key = subkey
        f = jrd.normal(key, (n,))
        v2(f).block_until_ready()
    end_time = time.perf_counter()
    print(f'{n_eval} Stepanoff evaluations took {end_time - start_time:.3E} s')

    print('Multiplication by JAX matrix:')
    mat = MatrixAlgebra(dim=n, dtype=jnp.float32)
    a = stepanoff_generator_matrix(alpha, k_max[0], hermitian=True)
    v3 = jit(partial(mat.app, a))

    start_time = time.perf_counter()
    g3 = v3(f)
    g3.block_until_ready()
    end_time = time.perf_counter()
    print(f'First Stepanoff evaluation took {end_time - start_time:.3E} s')
    if if_print:
        print(g3)

    start_time = time.perf_counter()
    for _ in range(n_eval):
        key, subkey = jrd.split(key)
        key = subkey
        f = jrd.normal(key, (n,))
        v3(f).block_until_ready()
    end_time = time.perf_counter()
    print(f'{n_eval} Stepanoff evaluations took {end_time - start_time:.3E} s')

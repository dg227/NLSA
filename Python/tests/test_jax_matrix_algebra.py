import jax.numpy as jnp
from jax import Array
from jax.numpy import allclose
from nlsa.abstract_algebra2 import ImplementsOperatorAlgebra,\
    ImplementsOperatorSpace
from nlsa.jax.matrix_algebra import MatrixAlgebra, MatrixSpace
from typing import Literal


K = jnp.float32
A = Array
M = Literal[2]
N = Literal[2]


def test_matrix_algebra():
    mat: MatrixAlgebra[N, K] = MatrixAlgebra(dim=2, dtype=jnp.float32)
    v1 = jnp.array([1, 2], dtype=jnp.float32)
    v2 = jnp.array([3, -4], dtype=jnp.float32)
    a1 = jnp.diag(v1)
    a2 = jnp.diag(v2)
    k = jnp.array(jnp.float32(5))
    assert isinstance(mat, ImplementsOperatorAlgebra)
    # assert isinstance(mat, ImplementsPower)
    assert allclose(mat.add(a1, a2), jnp.diag(jnp.array([4, -2],
                                                        dtype=jnp.float32)))
    assert allclose(mat.sub(a1, a2), jnp.diag(jnp.array([-2, 6],
                                                        dtype=jnp.float32)))
    assert allclose(mat.mul(a1, a2), jnp.diag(jnp.array([3, -8],
                                                        dtype=jnp.float32)))
    assert allclose(mat.smul(k, a1), jnp.diag(jnp.array([5, 10],
                                                        dtype=jnp.float32)))
    # assert allclose(mat.power(a1, k), jnp.diag(jnp.array([1, 32],
    # dtype=jnp.float32)))
    assert allclose(mat.lmul(a1, a2), jnp.diag(jnp.array([3, -8],
                                                         dtype=jnp.float32)))
    assert allclose(mat.rmul(a1, a2), jnp.diag(jnp.array([3, -8],
                                                         dtype=jnp.float32)))
    assert allclose(mat.rdiv(a2, a1), jnp.diag(jnp.array([3, -2],
                                                         dtype=jnp.float32)))
    assert allclose(mat.ldiv(a1, a2), jnp.diag(jnp.array([3, -2],
                                                         dtype=jnp.float32)))
    assert allclose(mat.unit(), jnp.diag(jnp.array([1, 1], dtype=jnp.float32)))
    assert allclose(mat.star(a1), jnp.diag(jnp.array([1, 2],
                                                     dtype=jnp.float32)))

def test_matrix_space():
    mat: MatrixSpace[M, N, K] = MatrixSpace(dim=(2, 2), dtype=jnp.float32)
    v1 = jnp.array([1, 2], dtype=jnp.float32)
    v2 = jnp.array([3, -4], dtype=jnp.float32)
    a1 = jnp.diag(v1)
    a2 = jnp.diag(v2)
    k = jnp.array(jnp.float32(5))
    assert isinstance(mat, ImplementsOperatorSpace)
    assert allclose(mat.add(a1, a2), jnp.diag(jnp.array([4, -2],
                                                        dtype=jnp.float32)))
    assert allclose(mat.sub(a1, a2), jnp.diag(jnp.array([-2, 6],
                                                        dtype=jnp.float32)))
    assert allclose(mat.smul(k, a1), jnp.diag(jnp.array([5, 10],
                                                        dtype=jnp.float32)))
    assert allclose(mat.lmul(a1, a2), jnp.diag(jnp.array([3, -8],
                                                         dtype=jnp.float32)))
    assert allclose(mat.rmul(a1, a2), jnp.diag(jnp.array([3, -8],
                                                         dtype=jnp.float32)))
    assert allclose(mat.rdiv(a2, a1), jnp.diag(jnp.array([3, -2],
                                                         dtype=jnp.float32)))
    assert allclose(mat.ldiv(a1, a2), jnp.diag(jnp.array([3, -2],
                                                         dtype=jnp.float32)))
    assert allclose(mat.star(a1), jnp.diag(jnp.array([1, 2],
                                                     dtype=jnp.float32)))

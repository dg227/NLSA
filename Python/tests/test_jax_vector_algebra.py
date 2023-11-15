import jax.numpy as jnp
from jax import Array
from jax.numpy import allclose
from nlsa.abstract_algebra2 import ImplementsHilbertSpace, ImplementsPower,\
        ImplementsUnitalStarAlgebraLRModule
from nlsa.jax.vector_algebra import VectorAlgebra, counting_measure
from typing import Literal


K = jnp.float32
V = Array
N = Literal[2]


def test_vector_algebra():
    vec: VectorAlgebra[N, K] = VectorAlgebra(dim=2, dtype=jnp.float32)
    v1 = jnp.array([1, 2], dtype=jnp.float32)
    v2 = jnp.array([3, -4], dtype=jnp.float32)
    k = jnp.array(jnp.float32(5))
    assert isinstance(vec, ImplementsUnitalStarAlgebraLRModule)
    assert isinstance(vec, ImplementsHilbertSpace)
    assert isinstance(vec, ImplementsPower)
    assert allclose(vec.add(v1, v2), jnp.array([4, -2]))
    assert allclose(vec.sub(v1, v2), jnp.array([-2, 6]))
    assert allclose(vec.mul(v1, v2), jnp.array([3, -8]))
    assert allclose(vec.smul(k, v1), jnp.array([5, 10]))
    assert allclose(vec.power(v1, k), jnp.array([1, 32]))
    assert allclose(vec.lmul(v1, v2), jnp.array([3, -8]))
    assert allclose(vec.rmul(v1, v2), jnp.array([3, -8]))
    assert allclose(vec.rdiv(v2, v1), jnp.array([3, -2]))
    assert allclose(vec.ldiv(v1, v2), jnp.array([3, -2]))
    assert allclose(vec.unit(), jnp.array([1, 1]))
    assert allclose(vec.star(v1), jnp.array([1, 2]))
    assert allclose(counting_measure(jnp.array([v1, v2])), jnp.array([3, -1]))
    assert allclose(vec.innerp(v1, v2), jnp.array(jnp.float32(-5)))

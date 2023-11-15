import jax.numpy as jnp
from nlsa.abstract_algebra2 import ImplementsPower,\
        ImplementsUnitalStarAlgebraLRModule, normalize
from nlsa.jax.vector_algebra import VectorAlgebra, counting_measure
from jax import Array
from typing import Literal


K = jnp.int32
V = Array
N = Literal[2]


if __name__ == '__main__':
    vec: VectorAlgebra[N, K] = VectorAlgebra(dim=2, dtype=jnp.int32)
    v1 = jnp.array([1, 2], dtype=jnp.int32)
    v2 = jnp.array([3, -4], dtype=jnp.int32)
    k = jnp.array(jnp.int32(5))
    k2 = jnp.array(jnp.int32(1))
    # assert isinstance(vec, ImplementsUnitalStarAlgebraLRModule)
    # assert isinstance(vec, ImplementsPower)
    print(normalize(vec, v1))
    assert jnp.all(vec.add(v1, v2) == jnp.array([4, -2]))
    assert jnp.all(vec.sub(v1, v2) == jnp.array([-2, 6]))
    assert jnp.all(vec.mul(v1, v2) == jnp.array([3, -8]))
    assert jnp.all(vec.smul(k, v1) == jnp.array([5, 10]))
    # assert jnp.all(vec.sdiv(k, v1) == jnp.array([1, 2]))
    assert jnp.all(vec.power(v1, k) == jnp.array([1, 32]))
    assert jnp.all(vec.lmul(v1, v2) == jnp.array([3, -8]))
    assert jnp.all(vec.rmul(v1, v2) == jnp.array([3, -8]))
    assert jnp.all(vec.rdiv(v2, v1) == jnp.array([3, -2]))
    assert jnp.all(vec.ldiv(v1, v2) == jnp.array([3, -2]))
    assert jnp.all(vec.unit() == jnp.array([1, 1]))
    assert jnp.all(vec.star(v1) == jnp.array([1, 2]))
    assert jnp.all(counting_measure(jnp.array([v1, v2])) == jnp.array([3, -1]))

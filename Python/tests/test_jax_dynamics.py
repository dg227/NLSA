import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.numpy import allclose
from nlsa.jax.dynamics import flow, make_stepanoff_vec
from typing import Callable, Optional

R2 = Array
T2 = Array
X = Array

def test_stepanoff():
    v = make_stepanoff_vec(jnp.sqrt(30))
    phi = flow(v, 1.0)
    vphi = jit(vmap(phi))
    x = jnp.array([[0., 0.], [0., 0.]])
    assert allclose(vphi(x), 0.0)


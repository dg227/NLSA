import jax
import jax.numpy as jnp
from jax import Array
from typing import Callable


def materialize_array(matvec: Callable[[Array], Array], shape: tuple[int],
                      dtype=None, holomorphic=False, jit=False):
    """Materialize the matrix A used in matvec(x) = Ax."""
    x = jnp.zeros(shape, dtype)
    if jit:
        fn = jax.jit(jax.jacfwd(matvec, holomorphic=holomorphic))
    else:
        fn = jax.jacfwd(matvec, holomorphic=holomorphic)

    return fn(x)

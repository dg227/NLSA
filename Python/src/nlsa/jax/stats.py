import jax.numpy as jnp
from jax import Array
from jax.scipy.special import i0
from typing import Callable

R = Array
S1 = Array


def make_von_mises_density(kappa: R, mu: S1) -> Callable[[S1], R]:
    """Make Von Mises probability density function on the circle."""
    def f(x: S1) -> R:
        y = jnp.exp(kappa * jnp.cos(x - mu)) / i0(kappa)
        return y
    return f

# pyright: basic

"""Implement distance functions in JAX."""

import jax.numpy as jnp
from collections.abc import Callable
from jax import Array

type X = Array  # Euclidean space
type TX = Array  # tangent bundle
type R = Array  # Real number
type F[*Xs, Y] = Callable[[*Xs], Y]


def sqeuclidean(x: X, y: X, /) -> R:
    """Compute pairwise square Euclidean distance."""
    return jnp.sum((x - y) ** 2)


def make_sqcone(zeta: float, threshold: float = 0) -> F[TX, TX, R]:
    """Make square cone distance."""
    sqrt_threshold = jnp.sqrt(threshold)

    def sqdist(xu: TX, yv: TX, /) -> F[TX, TX, R]:
        s = xu[0] - yv[0]
        s2 = jnp.sum(s ** 2)
        su2 = jnp.sum(s * xu[1]) ** 2
        sv2 = jnp.sum(s * yv[1]) ** 2
        u2 = jnp.sum(xu[1] ** 2)
        v2 = jnp.sum(yv[1] ** 2)
        a = s2 - zeta*su2/u2
        b = s2 - zeta*sv2/v2
        return jnp.sqrt(a * b + threshold) - sqrt_threshold
    return sqdist

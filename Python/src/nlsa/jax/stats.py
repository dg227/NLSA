# pyright: basic

"""Implement basic statistical functions using JAX arrays."""

import jax.numpy as jnp
from collections.abc import Callable
from jax import Array
from jax.scipy.special import i0
from typing import Optional

type R = Array
type S1 = Array
type Xs = Array


def make_von_mises_density(concentration: float,
                           location: Optional[float] = 0) \
        -> Callable[[S1], R]:
    """Make Von Mises probability density function on the circle."""
    def f(x: S1, /) -> R:
        y = jnp.exp(concentration * jnp.cos(x - location)) / i0(concentration)
        return y
    return f


def normalized_rmse(xs_true: Xs, xs_pred: Xs) -> R:
    """Compute normalized root mean square error."""
    nmse = jnp.sum((xs_true - xs_pred)**2) / jnp.sum(xs_true**2)
    return jnp.sqrt(nmse)


def anomaly_correlation(xs_true: Xs, xs_pred: Xs) -> R:
    """Compute anomaly correlation coefficient."""
    anom_true = xs_true - jnp.mean(xs_true)
    anom_pred = xs_pred - jnp.mean(xs_pred)
    sqnorm_true = jnp.sum(anom_true**2)
    sqnorm_pred = jnp.sum(anom_pred**2)
    return jnp.sum(anom_true * anom_pred) / jnp.sqrt(sqnorm_true * sqnorm_pred)

# pyright: basic
"""Provide observables on Euclidean spaces."""

import jax.numpy as jnp
from collections.abc import Callable
from functools import partial
from jax import Array
from jax.typing import DTypeLike
from nlsa.jax.utils import make_vectorvalued

type X = Array  # Point in state space (R^n)
type Y = Array  # Point in covariate space
type F[*Ss, T] = Callable[[*Ss], T]  # Shorthand for Callables


def make_observable_x(dtype: DTypeLike | None =  None,
                      asvector: bool = False) -> F[X, Y]:
    """Make R-valued observable giving the first state vector component."""
    @partial(make_vectorvalued, vectorvalued=asvector, dtype=dtype)
    def f(x: X, /) -> Y:
        return x[0]
    return f


def make_observable_y(dtype: DTypeLike | None =  None,
                      asvector: bool = False) -> F[X, Y]:
    """Make R-valued observable giving the second state vector component."""
    @partial(make_vectorvalued, vectorvalued=asvector, dtype=dtype)
    def f(x: X, /) -> Y:
        return x[1]
    return f


def make_observable_z(dtype: DTypeLike | None =  None,
                      asvector: bool = False) -> F[X, Y]:
    """Make R-valued observable giving the second state vector component."""
    @partial(make_vectorvalued, vectorvalued=asvector, dtype=dtype)
    def f(x: X, /) -> Y:
        return x[2]
    return f


def make_observable_xy(dtype: DTypeLike | None = None) -> F[X, Y]:
    """Make R2-valued observable giving the first two state vector compoments.
    """
    def f(x: X, /) -> Y:
        return jnp.astype(x[:2], dtype)
    return f


def make_observable_id(dtype: DTypeLike | None = None) -> F[X, Y]:
    """Make identity observable on Euclidean space."""
    def f(x: X, /) -> Y:
        return jnp.astype(x, dtype)
    return f

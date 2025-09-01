# pyright: basic
"""Provide observables on the 2-torus."""

import jax
import jax.numpy as jnp
from collections.abc import Callable
from functools import partial
from jax import Array
from jax.typing import DTypeLike
from nlsa.jax.stats import make_von_mises_density
from nlsa.jax.utils import make_vectorvalued
from typing import Optional

type X = Array  # Point in state space (2-torus)
type Y = Array  # Point in covariate space
type F[*Ss, T] = Callable[[*Ss], T]  # Shorthand for Callables


def make_observable_r3(r: float = 0.5,
                       dtype: Optional[DTypeLike] = None) -> F[X, Y]:
    """Make embedding function from the 2-torus into R3."""
    def f(x: X, /) -> Y:
        y = jnp.empty(3, dtype=dtype)
        a = 1 + r*jnp.cos(x[1])
        y = y.at[0].set(a*jnp.cos(x[0]))
        y = y.at[1].set(a*jnp.sin(x[0]))
        y = y.at[2].set(r*jnp.sin(x[1]))
        return y
    return f


def make_observable_r4(dtype: Optional[DTypeLike] = None) -> F[X, Y]:
    """Make observable based on flat embedding of the 2-torus into R4."""
    def f(x: X, /) -> Y:
        y = jnp.empty(4, dtype=dtype)
        y = y.at[0].set(jnp.cos(x[0]))
        y = y.at[1].set(jnp.sin(x[0]))
        y = y.at[2].set(jnp.cos(x[1]))
        y = y.at[3].set(jnp.sin(x[1]))
        return y
    return f


def make_observable_cos(dtype: Optional[DTypeLike] = None,
                        asvector: bool = False) -> F[X, Y]:
    """Make R-valued observable based on cosine of the angles on the torus."""
    @partial(make_vectorvalued, vectorvalued=asvector, dtype=dtype)
    def f(x: X, /) -> Y:
        return jnp.cos(x[0]) * jnp.cos(x[1])
    return f


def make_observable_von_mises(concentration: tuple[float, float],
                              location: tuple[float, float],
                              dtype: Optional[DTypeLike] = None,
                              asvector: bool = False) -> F[X, Y]:
    """Make covariate based on von Mises density."""
    g0 = make_von_mises_density(concentration=concentration[0],
                                location=location[0])
    g1 = make_von_mises_density(concentration=concentration[1],
                                location=location[1])

    @partial(make_vectorvalued, vectorvalued=asvector, dtype=dtype)
    def f(x: X, /) -> Y:
        return g0(x[0]) * g1(x[1])
    return f


def make_observable_von_mises_grad(concentration: tuple[float, float],
                                   location: tuple[float, float],
                                   dtype: Optional[DTypeLike] = None,
                                   asvector: bool = False) -> F[X, Y]:
    """Make covariate based on gradient of von Mises density."""
    g0 = jax.grad(make_von_mises_density(concentration=concentration[0],
                                         location=location[0]))
    g1 = jax.grad(make_von_mises_density(concentration=concentration[1],
                                         location=location[1]))

    @partial(make_vectorvalued, vectorvalued=asvector, dtype=dtype)
    def f(x: X, /) -> Y:
        return g0(x[0]) * g1(x[1])
    return f

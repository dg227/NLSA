# pyright: basic

import jax.numpy as jnp
from jax import Array, vmap
from typing import Callable, Optional, TypeVar

T1 = Array
V = Array
Is = Array
S = Array
X = TypeVar('X')
Y = TypeVar('Y')

F = Callable[[X], Y]


def make_fourier_basis(ks: Is, ws: Optional[V] = None) -> Callable[[T1], V]:
    """Make (scaled) Fourier basis functions from wavenumbers and weights."""

    if ws is None:
        def phi(x: T1) -> V:
            return jnp.exp(1j * ks * x)
    else:
        def phi(x: T1) -> V:
            return ws * jnp.exp(1j * ks * x)

    return phi


def make_fourier_synthesis(ks: Is, ws: Optional[V] = None) \
        -> Callable[[V], F[T1, S]]:
    """Make Fourier synthesis operator from wavenumbers."""

    phi = make_fourier_basis(ks, ws)
    vmul = vmap(jnp.multiply)

    def synth(v: V) -> F[T1, S]:
        def f(x: T1) -> S:
            return jnp.sum(vmul(v, phi(x)))
        return f
    return synth

# pyright: basic

import jax.numpy as jnp
import nlsa.jax.fourier_t1 as t1
from functools import partial
from jax import Array, vmap
from nlsa.function_algebra2 import compose
from nlsa.jax.scalar_algebra import ScalarField
from nlsa.jax.vector_algebra import neg
from typing import Callable, Generic, Literal, Optional, Type, TypeVar

K = TypeVar('K', jnp.float32, jnp.float64)
S = Array
V = Array
V2 = Array
Mode = TypeVar('Mode', bound=Literal['full', 'same', 'valid'])
T2 = Array
Is = Array
I2s = tuple[Is, Is]


def make_fourier_basis(ks: I2s, ws: tuple[Optional[V], [V]] = (None, None)) \
        -> Callable[[T2], V2]:
    """Make (scaled) Fourier basis functions from wavenumbers and weights."""
    phi1 = t1.make_fourier_basis(ks[0], ws[0])
    phi2 = t1.make_fourier_basis(ks[1], ws[1])

    def phi(x: T2) -> V2:
        return jnp.kron(phi1(x[0]), phi2(x[1]))

    return phi


def make_convolution(mode: Optional[Mode] = None) -> Callable[[V, V], V]:
    """Make convolution product between vectors."""
    if mode is None:
        mode = 'full'

    conv = partial(jnp.convolve, mode=mode)
    conv2 = compose(vmap(conv, in_axes=0, out_axes=0),  # along first dim.
                    vmap(conv, in_axes=1, out_axes=1))  # along second dim.

    def mul(u: V, v: V) -> V:
        return conv2(u, v)

    return mul


def flip_conj(v: V) -> V:
    """Perform involution operation (complex-conjugation and flip)."""
    return jnp.conjugate(v[::-1, ::-1])


class DualConvolutionAlgebra(Generic[Mode, K]):
    """Implement convolution algebra on dual group of the 2-torus.

    The type variable Mode parameterizes the numerical mode for performing
    convolution. The type parameter K parameterizes the field of scalars.

    TODO: Additive operations should depend on mode.
    """

    def __init__(self, dtype: Type[K], mode: Optional[Mode] = None):
        if mode is None:
            mode = 'full'

        self.scl = ScalarField(dtype)
        self.add: Callable[[V, V], V] = jnp.add
        self.neg: Callable[[V], V] = neg
        self.sub: Callable[[V, V], V] = jnp.subtract
        self.smul: Callable[[S, V], V] = jnp.multiply
        self.mul: Callable[[V, V], V] = make_convolution(mode)
        self.star: Callable[[V], V] = flip_conj

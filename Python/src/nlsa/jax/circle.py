# pyright: basic
"""Implement observables and function spaces on the 2-torus."""

import jax.numpy as jnp
import nlsa.function_algebra as fun
import nlsa.jax.vector_algebra as vec
from collections.abc import Callable
from functools import partial
from jax import Array
from jax.typing import DTypeLike
from nlsa.jax.kernels import KernelEigenbasis
from nlsa.jax.special import iv0_ratio
from nlsa.jax.stats import make_von_mises_density
from nlsa.jax.utils import make_vectorvalued
from typing import Optional

type X = Array  # Point in state space (circle S1)
type Y = Array  # Point in embedding space (Rd)
type Z = Array  # Point in dual group (Z)
type Zs = Array  # Collection of points in the dual group
type R = Array  # Real number
type C = Array  # Complex number
type Cs = Array  # Collection of complex numbers
type V = Array  # Lp vectors on S1
type Vhat = Array  # Lp vectors on the dual group
type F[*Ss, T] = Callable[[*Ss], T]  # Shorthand for Callables


def make_observable_r2(
    r: float | tuple[float, float], dtype: Optional[DTypeLike] = None
) -> F[X, Y]:
    """Make embedding function from the circle into R2."""
    match r:
        case (r1, r2) if isinstance(r1, float) and isinstance(r2, float):
            rs = r
        case float():
            rs = (r, r)

    def f(x: X, /) -> Y:
        y = jnp.empty(2, dtype=dtype)
        y = y.at[0].set(rs[0] * jnp.cos(x))
        y = y.at[1].set(rs[1] * jnp.sin(x))
        return y

    return f


def make_observable_von_mises(
    concentration: float,
    location: float,
    dtype: Optional[DTypeLike] = None,
    asvector: bool = False,
) -> F[X, Y]:
    """Make covariate based on von Mises density."""
    g = make_von_mises_density(concentration=concentration, location=location)

    @partial(make_vectorvalued, vectorvalued=asvector, dtype=dtype)
    def f(x: X, /) -> Y:
        return g(x)

    return f


def make_von_mises_density_fourier(
    concentration: float | Array,
    location: float | Array = 0,
    dtype: Optional[DTypeLike] = None,
) -> Callable[[Z], C]:
    """Make function returning Fourier coefficients of von Mises density."""
    conc = jnp.array(concentration, dtype=dtype)
    loc = jnp.array(location, dtype=dtype)

    def vm_fourier(k: Z) -> C:
        return iv0_ratio(jnp.array(jnp.abs(k), dtype=dtype), conc) * jnp.exp(
            -1j * k * loc
        )

    return vm_fourier


def make_fourier_basis(
    wavenums: int | Zs, weight: float | Cs = 1
) -> Callable[[X], Cs]:
    """Make (scaled) Fourier basis functions from wavenumbers and weights."""

    def phi(x: X, /) -> Cs:
        return weight * jnp.exp(1j * wavenums * x)

    return phi


def unit_fourier(max_wavenum: int, dtype: Optional[DTypeLike] = None) -> Vhat:
    """Compute Fourier representation of unit function on S1."""
    n = 2 * max_wavenum + 1
    u = jnp.zeros(n, dtype=dtype)
    u = u.at[max_wavenum].set(1)
    return u


def make_fourier_analysis_operator(
    weight: float | Vhat = 1,
) -> Callable[[V], Vhat]:
    """Make analysis operator for (scaled) Fourier basis on S1 using FFT."""

    def anal(f: V) -> V:
        return jnp.fft.fftshift(jnp.fft.fft(f, norm="forward")) / weight

    return anal


def make_fourier_fn_analysis_operator(
    max_wavenum: int,
    weight: float | Vhat = 1,
    dtype: Optional[DTypeLike] = None,
    jit: bool = False,
) -> Callable[[F[X, C]], Vhat]:
    """Make discrete Fourier analysis operator for functions on the circle."""
    ev = vec.veval_at(
        jnp.linspace(
            0,
            2 * jnp.pi,
            2 * max_wavenum + 1,
            endpoint=False,
            dtype=dtype,
        ),
        jit=jit,
    )
    anal = make_fourier_analysis_operator(weight)

    def fn_anal(f: F[X, C]) -> Vhat:
        return anal(ev(f))

    return fn_anal


def make_rkhs_inverse_weights(
    p: float, tau: float, dtype: Optional[DTypeLike] = None
) -> Callable[[Z], R]:
    """Make inverse (sub)exponential weight function on Z."""

    def lamb(k: Z) -> R:
        return jnp.exp(-tau * jnp.abs(jnp.array(k, dtype=dtype)) ** p)

    return lamb


def make_rkhs_eigenbasis(
    max_wavenum: int, p: float, tau: float, dtype: Optional[DTypeLike] = None
) -> KernelEigenbasis[X, C, V, Cs, int | Array]:
    """Make kernel eigenbasis for RKHS associated with exponential weights."""
    dim = 2 * max_wavenum + 1
    wavenums = jnp.arange(-max_wavenum, max_wavenum + 1)
    inv_weight_fn = make_rkhs_inverse_weights(p, tau / 2, dtype)
    sqrt_lambs = inv_weight_fn(wavenums)
    spec = sqrt_lambs**2
    lapl_spec = jnp.abs(wavenums) ** p
    vc = vec.make_one_hot_basis(dim, dtype)
    anal = make_fourier_analysis_operator(weight=sqrt_lambs)
    fn_anal = make_fourier_fn_analysis_operator(
        max_wavenum=max_wavenum, weight=sqrt_lambs
    )
    synth: Callable[[Cs], V] = fun.identity
    fn_synth = vec.make_fn_synthesis_operator(
        make_fourier_basis(wavenums=wavenums, weight=sqrt_lambs)
    )

    def evl(i: int | Array) -> R:
        return spec[i]

    def lapl_evl(i: int | Array) -> R:
        return lapl_spec[i]

    def fn(i: int | Array) -> Callable[[X], C]:
        return make_fourier_basis(wavenums=wavenums[i], weight=sqrt_lambs[i])

    basis = KernelEigenbasis(
        dim=dim,
        anal=anal,
        dual_anal=anal,
        synth=synth,
        dual_synth=synth,
        fn_anal=fn_anal,
        dual_fn_anal=fn_anal,
        fn_synth=fn_synth,
        dual_fn_synth=fn_synth,
        vec=vc,
        dual_vec=vc,
        fn=fn,
        dual_fn=fn,
        evl=evl,
        lapl_evl=lapl_evl,
        spec=spec,
        lapl_spec=lapl_spec,
    )
    return basis

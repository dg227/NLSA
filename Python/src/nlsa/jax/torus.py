# pyright: basic
"""Implement observables and function spaces on the 2-torus."""

import jax
import jax.numpy as jnp
import nlsa.function_algebra as fun
import nlsa.jax.circle as circle
import nlsa.jax.vector_algebra as vec
from collections.abc import Callable
from functools import partial
from jax import Array
from jax.sharding import Sharding
from jax.typing import DTypeLike
from nlsa.jax.kernels import KernelEigenbasis
from nlsa.jax.sharding import shardit
from nlsa.jax.stats import make_von_mises_density
from nlsa.jax.special import iv0_ratio
from nlsa.jax.utils import make_vectorvalued
from typing import Optional

type X = Array  # Point in state space (2-torus)
type Y = Array  # Point in embedding space (R^d)
type R = Array  # Real number
type C = Array  # Complex number
type Cs = Array  # Collection of complex numbers
type V = Array  # Lp vectors on T2
type Vhat = Array  # Lp vectors on the dual group
type Zs = Array  # Collection of points in the dual group Z of S1
type Z2 = Array  # Point in the dual group Z2 of T2
type Z2s = Array  # Collection of points in the dual group Z2
type F[*Ss, T] = Callable[[*Ss], T]  # Shorthand for Callables


def make_observable_r3(
    r: float = 0.5, dtype: Optional[DTypeLike] = None
) -> F[X, Y]:
    """Make embedding function from the 2-torus into R3."""

    def f(x: X, /) -> Y:
        y = jnp.empty(3, dtype=dtype)
        a = 1 + r * jnp.cos(x[1])
        y = y.at[0].set(a * jnp.cos(x[0]))
        y = y.at[1].set(a * jnp.sin(x[0]))
        y = y.at[2].set(r * jnp.sin(x[1]))
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


def make_observable_cos(
    dtype: Optional[DTypeLike] = None, asvector: bool = False
) -> F[X, Y]:
    """Make R-valued observable based on cosine of the angles on the torus."""

    @partial(make_vectorvalued, vectorvalued=asvector, dtype=dtype)
    def f(x: X, /) -> Y:
        return jnp.cos(x[0]) * jnp.cos(x[1])

    return f


def make_observable_von_mises(
    concentrations: float | tuple[float, float],
    locations: float | tuple[float, float],
    dtype: Optional[DTypeLike] = None,
    asvector: bool = False,
) -> F[X, Y]:
    """Make covariate based on von Mises density."""
    match concentrations:
        case float():
            concs = (concentrations, concentrations)
        case (_, _):
            concs = concentrations
    match locations:
        case float():
            locs = (locations, locations)
        case (_, _):
            locs = locations
    g0 = make_von_mises_density(concentration=concs[0], location=locs[0])
    g1 = make_von_mises_density(concentration=concs[1], location=locs[1])

    @partial(make_vectorvalued, vectorvalued=asvector, dtype=dtype)
    def f(x: X, /) -> Y:
        return g0(x[0]) * g1(x[1])

    return f


def make_observable_von_mises_grad(
    concentrations: float | tuple[float, float],
    locations: float | tuple[float, float],
    dtype: Optional[DTypeLike] = None,
    asvector: bool = False,
) -> F[X, Y]:
    """Make covariate based on gradient of von Mises density."""
    match concentrations:
        case float():
            concs = (concentrations, concentrations)
        case (_, _):
            concs = concentrations
    match locations:
        case float():
            locs = (locations, locations)
        case (_, _):
            locs = locations
    g0 = jax.grad(
        make_von_mises_density(concentration=concs[0], location=locs[0])
    )
    g1 = jax.grad(
        make_von_mises_density(concentration=concs[1], location=locs[1])
    )

    @partial(make_vectorvalued, vectorvalued=asvector, dtype=dtype)
    def f(x: X, /) -> Y:
        return g0(x[0]) * g1(x[1])

    return f


# TODO: Here and in other similar functions that perform structural pattern
# matching over floats, try matching against SupportsFloat.
def make_von_mises_density_fourier(
    concentrations: float | tuple[float, float] | Array,
    locations: tuple[float, float] | Array = (0, 0),
    dtype: Optional[DTypeLike] = None,
) -> Callable[[Z2], C]:
    """Make function returning Fourier coefficients of von Mises density."""
    match concentrations:
        case Array():
            concs = concentrations
        case float() | int():
            concs = jnp.array([concentrations, concentrations], dtype=dtype)
        case tuple():
            concs = jnp.array(concentrations, dtype=dtype)
    match locations:
        case Array():
            locs = locations
        case tuple():
            locs = jnp.array(locations)

    def vm_fourier(ks: Z2) -> C:
        vms = iv0_ratio(jnp.array(jnp.abs(ks), dtype=dtype), concs) * jnp.exp(
            -1j * ks * locs
        )
        return jnp.prod(vms)

    return vm_fourier


def make_fourier_basis(
    wavenums: Z2s, weight: float | Cs = 1
) -> Callable[[X], Cs]:
    """Make (scaled) Fourier basis functions from wavenumbers and weights."""

    def phi(x: X, /) -> Cs:
        return weight * jnp.exp(1j * jnp.sum(wavenums * x, axis=-1))

    return phi


def zero_idx_fourier(max_wavenums: int | tuple[int, int]) -> int:
    """Compute index of 0 wavenumber in Fourier basis."""
    match max_wavenums:
        case int():
            k_max = (max_wavenums, max_wavenums)
        case (_, _):
            k_max = max_wavenums
    return k_max[0] * (2 * k_max[1] + 1) + k_max[1]


def unit_fourier(
    max_wavenums: int | tuple[int, int], dtype: Optional[DTypeLike] = None
) -> Vhat:
    """Compute Fourier representation of unit function on T2."""
    i0 = zero_idx_fourier(max_wavenums)
    match max_wavenums:
        case int():
            k_max = (max_wavenums, max_wavenums)
        case (_, _):
            k_max = max_wavenums
    u1 = jnp.zeros(((2 * k_max[0] + 1) * (2 * k_max[1] + 1)), dtype=dtype)
    u1 = u1.at[i0].set(1)
    return u1


def make_fourier_analysis_operator(
    max_wavenums: int | tuple[int, int],
    weight: float | Vhat = 1,
) -> Callable[[V], Vhat]:
    """Make analysis operator for (scaled) Fourier basis on T2 using FFT."""
    match max_wavenums:
        case int():
            k_max = (max_wavenums, max_wavenums)
        case (_, _):
            k_max = max_wavenums
    n1 = 2 * k_max[0] + 1
    n2 = 2 * k_max[1] + 1

    def anal(f: V) -> V:
        f_hat = jnp.fft.fftshift(
            jnp.fft.fft2(f.reshape((n1, n2)), norm="forward")
        )
        return f_hat.ravel() / weight

    return anal


def make_fourier_fn_analysis_operator(
    max_wavenums: int | tuple[int, int],
    weight: float | Vhat = 1,
    dtype: Optional[DTypeLike] = None,
    jit: bool = False,
) -> Callable[[F[X, C]], Vhat]:
    """Make discrete Fourier analysis operator for functions on the 2-torus."""
    match max_wavenums:
        case int():
            k_max = (max_wavenums, max_wavenums)
        case (_, _):
            k_max = max_wavenums
    x1s = jnp.linspace(
        0, 2 * jnp.pi, 2 * k_max[0] + 1, endpoint=False, dtype=dtype
    )
    x2s = jnp.linspace(
        0, 2 * jnp.pi, 2 * k_max[1] + 1, endpoint=False, dtype=dtype
    )
    xs = jnp.stack(jnp.meshgrid(x1s, x2s, indexing="ij"), axis=-1).reshape(
        (-1, 2)
    )
    ev = vec.veval_at(xs, jit=jit)
    anal = make_fourier_analysis_operator(
        max_wavenums=max_wavenums, weight=weight
    )

    def fn_anal(f: F[X, C]) -> Vhat:
        return anal(ev(f))

    return fn_anal


def make_rkhs_eigenbasis(
    max_wavenums: int | tuple[int, int],
    p: float,
    tau: float,
    dtype: Optional[DTypeLike] = None,
) -> KernelEigenbasis[X, C, V, Cs, int | Array]:
    """Make kernel eigenbasis for RKHS associated with exponential weights."""
    match max_wavenums:
        case int():
            k_max = (max_wavenums, max_wavenums)
        case (_, _):
            k_max = max_wavenums
    dim = (2 * k_max[0] + 1) * (2 * k_max[1] + 1)
    inv_weight_fn = circle.make_rkhs_inverse_weights(p, tau / 2, dtype)
    k1s = jnp.arange(-k_max[0], k_max[0] + 1)
    k2s = jnp.arange(-k_max[1], k_max[1] + 1)
    ks = jnp.stack(jnp.meshgrid(k1s, k2s, indexing="ij"), axis=-1).reshape(
        (-1, 2)
    )
    sqrt_lambs = jnp.kron(inv_weight_fn(k1s), inv_weight_fn(k2s))
    spec = sqrt_lambs**2
    lapl_spec = jnp.sum(jnp.abs(ks) ** p, axis=-1)
    vc = vec.make_one_hot_basis(dim, dtype=dtype)
    anal = make_fourier_analysis_operator(
        max_wavenums=max_wavenums, weight=sqrt_lambs
    )
    fn_anal = make_fourier_fn_analysis_operator(
        max_wavenums=max_wavenums, weight=sqrt_lambs
    )
    synth: Callable[[Cs], V] = fun.identity
    fn_synth = vec.make_fn_synthesis_operator(
        make_fourier_basis(wavenums=ks, weight=sqrt_lambs)
    )

    def evl(i: int | Array) -> R:
        return spec[i]

    def lapl_evl(i: int | Array) -> R:
        return lapl_spec[i]

    def fn(i: int | Array) -> Callable[[X], C]:
        return make_fourier_basis(wavenums=ks[i], weight=sqrt_lambs[i])

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


def make_to_zero_mean_fourier(
    max_wavenums: int | tuple[int, int], sharding: Optional[Sharding] = None
) -> Callable[[Vhat], Vhat]:
    """Make projection onto zero-mean functions in the Fourier basis."""
    i0 = zero_idx_fourier(max_wavenums)

    @partial(shardit, sharding=sharding)
    def proj(v: Vhat) -> Vhat:
        return jnp.concatenate((v[:i0], v[i0 + 1 :]))

    return proj


def make_from_zero_mean_fourier(
    max_wavenums: int | tuple[int, int], sharding: Optional[Sharding] = None
) -> Callable[[Vhat], Vhat]:
    """Make inclusion map into L2 functions from zero-mean functions."""
    i0 = zero_idx_fourier(max_wavenums)

    @partial(shardit, sharding=sharding)
    def incl(v: Vhat) -> Vhat:
        return jnp.concatenate((v[:i0], jnp.zeros(1, dtype=v.dtype), v[i0:]))

    return incl

"""Implement specifal functions in JAX."""

import jax.numpy as jnp
from jax import Array
# from tensorflow_probability.substrates.jax.math import bessel_ive as ive


# def iv0_ratio(v: Array, z: Array) -> Array:
#     """Compute ratio of modified Bessel functions of the first kind.

#     Let I(v,z) be the modified Bessel function of the first kind of order v
#     evaluated at z. iv0_ratio returns the ratio I(v,z) / I(0,z) by computing
#     the ratio of the corresponding exponentially scaled Bessel functions Ive,
#     defined as Ive(v, z) = Iv(v, z) * exp(-abs(z)).

#     We use the implementaion of Ive provided by tensorflow_probability.

#     Warning: This implementation of ive0_ratio may not be numerically stable.
#     """
#     return jnp.divide(ive(v, z), ive(0, z))


def dawsn(x: Array) -> Array:
    """Compute Dawson's integral D(x) = exp(-x^2) * int_0^x exp(t^2) dt.

    Pure-JAX replacement for scipy.special.dawsn, which is not exposed in
    jax.scipy.special. Uses a 4-term Maclaurin series for |x| < 0.2 and
    Rybicki's sampling formula (NMAX=6, H=0.4) elsewhere; accurate to
    ~1e-7 across the real line. See Numerical Recipes 3rd ed., section 6.10.
    """
    h = 0.4
    nmax = 6
    a1, a2, a3 = 2.0 / 3.0, 0.4, 2.0 / 7.0

    abs_x = jnp.abs(x)
    sign_x = jnp.sign(x)

    x2 = x * x
    series = x * (1.0 - a1 * x2 * (1.0 - a2 * x2 * (1.0 - a3 * x2)))

    n0 = 2.0 * jnp.floor(0.5 * abs_x / h + 0.5)
    xp = abs_x - n0 * h
    e1 = jnp.exp(2.0 * xp * h)
    e2 = e1 * e1
    s = jnp.zeros_like(abs_x)
    en = e1
    for i in range(nmax):
        c_i = jnp.exp(-(((2 * i + 1) * h) ** 2))
        d1 = n0 + (2 * i + 1)
        d2 = n0 - (2 * i + 1)
        s = s + c_i * (en / d1 + 1.0 / (en * d2))
        en = en * e2
    rybicki = (sign_x / jnp.sqrt(jnp.pi)) * jnp.exp(-xp * xp) * s

    return jnp.where(abs_x < 0.2, series, rybicki)

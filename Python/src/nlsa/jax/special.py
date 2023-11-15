import jax.numpy as jnp
from jax import Array
from tensorflow_probability.substrates.jax.math import bessel_ive as ive


def iv0_ratio(v: Array, z: Array) -> Array:
    """Compute ratio of modified Bessel functions of the first kind.

    Let I(v,z) be the modified Bessel function of the first kind of order v
    evaluated at z. iv0_ratio returns the ratio I(v,z) / I(0,z) by computing
    the ratio of the corresponding exponentially scaled Bessel functions Ive,
    defined as Ive(v, z) = Iv(v, z) * exp(-abs(z)).

    We use the implementaion of Ive provided by tensorflow_probability.

    Warning: This implementation of ive0_ratio may not be numerically stable.
    """

    return jnp.divide(ive(v, z), ive(0., z))

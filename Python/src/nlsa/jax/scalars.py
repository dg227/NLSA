"""Provide scalar field operations for JAX arrays."""

import jax.numpy as jnp
import nlsa.abstract_algebra as alg
from collections.abc import Callable
from dataclasses import dataclass
from jax import Array
from jax.typing import DTypeLike
from nlsa.typing import DEFAULT
from typing import SupportsComplex, SupportsFloat, final

type K = Array


def make_zero(dtype: DTypeLike) -> Callable[[], K]:
    """Make constant function returning scalar zero."""

    def zero() -> K:
        return jnp.zeros((), dtype=dtype)

    return zero


def make_unit(dtype: DTypeLike) -> Callable[[], K]:
    """Make constant function returning scalar unit."""

    def unit() -> K:
        return jnp.ones((), dtype=dtype)

    return unit


# TODO: We should distinguish more carefully between real and complex dtypes.
# This may warrant having separate RealScalarField and ComplexScalarField.
@final
@dataclass(frozen=True, slots=True)
class ScalarField[D: DTypeLike](alg.ImplementsComplexScalarField[K]):
    """Implement scalar field operations on JAX arrays."""

    dtype: D
    _zero: Callable[[], K]
    _add: Callable[[K, K], K]
    _sub: Callable[[K, K], K]
    _neg: Callable[[K], K]
    _unit: Callable[[], K]
    _mul: Callable[[K, K], K]
    _mpower: Callable[[K, int], K]
    _power: Callable[[K, K], K]
    _div: Callable[[K, K], K]
    _inv: Callable[[K], K]
    _adj: Callable[[K], K]
    _sqrt: Callable[[K], K]
    _abs: Callable[[K], K]
    _log: Callable[[K], K]
    _log10: Callable[[K], K]
    _exp: Callable[[K], K]
    _exp10: Callable[[K], K]

    def zero(self, /) -> K:
        """Return a zero scalar array."""
        return self._zero()

    def add(self, w: K, z: K) -> K:
        """Add two scalar arrays."""
        return self._add(w, z)

    def sub(self, w: K, z: K) -> K:
        """Subtract two scalar arrays."""
        return self._sub(w, z)

    def neg(self, z: K) -> K:
        """Compute additive inverse (negation) of a scalar array."""
        return self._neg(z)

    def unit(self, /) -> K:
        """Return a unit scalar array."""
        return self._unit()

    def mul(self, w: K, z: K) -> K:
        """Multiply two scalar arrays."""
        return self._mul(w, z)

    def mpower(self, z: K, m: int) -> K:
        """Exponentiate a scalar array by an integer."""
        return self._mpower(z, m)

    def power(self, w: K, z: K) -> K:
        """Exponentiate a scalar array by another scalar array."""
        return self._power(w, z)

    def div(self, w: K, z: K) -> K:
        """Divide two scalar arrays."""
        return self._div(w, z)

    def inv(self, z: K) -> K:
        """Compute multiplicative inverse of a scalar array."""
        return self._inv(z)

    def adj(self, z: K) -> K:
        """Compute complex conjugate of a scalar array."""
        return self._adj(z)

    def sqrt(self, z: K) -> K:
        """Compute the square root of a scalar array."""
        return self._sqrt(z)

    def abs(self, z: K) -> K:
        """Compute the absolute value of a scalar array."""
        return self._abs(z)

    def exp(self, z: K) -> K:
        """Exponentiate a scalar array."""
        return self._exp(z)

    def exp10(self, z: K) -> K:
        """Compute base-10 exponentiation of a scalar array."""
        return self._exp10(z)

    def log(self, z: K) -> K:
        """Compute natural logarithm of a scalar array."""
        return self._log(z)

    def log10(self, z: K) -> K:
        """Compute base-10 logarithm of a scalar array."""
        return self._log10(z)

    def from_pyscalar(self, z: SupportsFloat | SupportsComplex, /) -> K:
        """Convert real or complex scalar to JAX array."""
        return jnp.asarray(z, dtype=self.dtype)


def scalar_field[D: DTypeLike](
    dtype: D,
    zero: Callable[[], K] | DEFAULT = DEFAULT,
    add: Callable[[K, K], K] | DEFAULT = DEFAULT,
    sub: Callable[[K, K], K] | DEFAULT = DEFAULT,
    neg: Callable[[K], K] | DEFAULT = DEFAULT,
    unit: Callable[[], K] | DEFAULT = DEFAULT,
    mul: Callable[[K, K], K] | DEFAULT = DEFAULT,
    mpower: Callable[[K, int], K] | DEFAULT = DEFAULT,
    power: Callable[[K, K], K] | DEFAULT = DEFAULT,
    div: Callable[[K, K], K] | DEFAULT = DEFAULT,
    inv: Callable[[K], K] | DEFAULT = DEFAULT,
    adj: Callable[[K], K] | DEFAULT = DEFAULT,
    sqrt: Callable[[K], K] | DEFAULT = DEFAULT,
    abs: Callable[[K], K] | DEFAULT = DEFAULT,
    exp: Callable[[K], K] | DEFAULT = DEFAULT,
    exp10: Callable[[K], K] | DEFAULT = DEFAULT,
    log: Callable[[K], K] | DEFAULT = DEFAULT,
    log10: Callable[[K], K] | DEFAULT = DEFAULT,
) -> ScalarField[D]:
    """Build ScalarField object."""
    return ScalarField(
        dtype=dtype,
        _zero=make_zero(dtype) if zero is DEFAULT else zero,
        _add=jnp.add if add is DEFAULT else add,
        _sub=jnp.subtract if sub is DEFAULT else sub,
        _neg=(lambda z: -z) if neg is DEFAULT else neg,
        _unit=make_unit(dtype) if unit is DEFAULT else unit,
        _mul=jnp.multiply if mul is DEFAULT else mul,
        _mpower=jnp.power if mpower is DEFAULT else mpower,
        _power=jnp.power if power is DEFAULT else power,
        _div=jnp.divide if div is DEFAULT else div,
        _inv=(lambda z: 1 / z) if inv is DEFAULT else inv,
        _adj=jnp.conjugate if adj is DEFAULT else adj,
        _sqrt=jnp.sqrt if sqrt is DEFAULT else sqrt,
        _abs=jnp.abs if abs is DEFAULT else abs,
        _exp=jnp.exp if exp is DEFAULT else exp,
        _exp10=(lambda z: 10**z) if exp10 is DEFAULT else exp10,
        _log=jnp.log if log is DEFAULT else log,
        _log10=jnp.log10 if log10 is DEFAULT else log10,
    )

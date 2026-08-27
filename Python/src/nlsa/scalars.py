"""Provide classes and functions implementing operations on scalar fields."""

import math
import cmath
import nlsa.abstract_algebra as alg
from typing import SupportsComplex, SupportsFloat, final


@final
class FloatScalarField(alg.ImplementsRealScalarField[float]):
    """Implement scalar field operations on float objects."""

    def zero(self, /) -> float:
        """Return 0 as a float."""
        return float(0)

    def add(self, x: float, y: float, /) -> float:
        """Add two floats."""
        return x + y

    def sub(self, x: float, y: float, /) -> float:
        """Subtract two floats."""
        return x - y

    def neg(self, x: float, /) -> float:
        """Compute additive inverse (negation) of a float."""
        return -x

    def unit(self, /) -> float:
        """Return 1 as a float."""
        return float(1)

    def mul(self, x: float, y: float, /) -> float:
        """Multiply two floats."""
        return x * y

    def mpower(self, x: float, k: int, /) -> float:
        """Exponentiate a float by an integer."""
        return x**k

    def power(self, x: float, y: float, /) -> float:
        """Exponentiate a float by another float."""
        return math.pow(x, y)

    def div(self, x: float, y: float, /) -> float:
        """Divide two floats."""
        return x / y

    def inv(self, x: float, /) -> float:
        """Compute multiplicative inverse of a float."""
        return 1 / x

    def sqrt(self, x: float, /) -> float:
        """Compute the square root of a float."""
        return math.sqrt(x)

    def abs(self, x: float, /) -> float:
        """Compute the absolute value of a float."""
        return abs(x)

    def exp(self, x: float, /) -> float:
        """Exponentiate a float."""
        return math.exp(x)

    def exp10(self, x: float, /) -> float:
        """Compute base-10 exponentiation of a float."""
        return 10**x

    def log(self, x: float, /) -> float:
        """Compute natural logarithm of a float."""
        return math.log(x)

    def log10(self, x: float, /) -> float:
        """Compute base-10 logarithm of a float."""
        return math.log10(x)

    def from_pyscalar(self, x: SupportsFloat, /) -> float:
        """Convert real scalar to float."""
        return float(x)


@final
class ComplexScalarField(alg.ImplementsComplexScalarField[complex]):
    """Implement scalar field operations on complex objects."""

    def zero(self, /) -> complex:
        """Return 0 as a complex number."""
        return complex(0)

    def add(self, w: complex, z: complex, /) -> complex:
        """Add two complex numbers."""
        return w + z

    def sub(self, w: complex, z: complex, /) -> complex:
        """Subtract two complex numbers."""
        return w - z

    def neg(self, z: complex, /) -> complex:
        """Compute additive inverse (negation) of a complex number."""
        return -z

    def unit(self, /) -> complex:
        """Return 1 as a complex number."""
        return complex(1)

    def mul(self, w: complex, z: complex, /) -> complex:
        """Multiply two complex numbers."""
        return w * z

    def mpower(self, z: complex, k: int, /) -> complex:
        """Exponentiate a complex number by an integer."""
        return z**k

    def power(self, w: complex, z: complex, /) -> complex:
        """Exponentiate a complex number by another complex number."""
        return w**z

    def div(self, w: complex, z: complex, /) -> complex:
        """Divide two complex numbers."""
        return w / z

    def inv(self, z: complex, /) -> complex:
        """Compute multiplicative inverse of a complex number."""
        return 1 / z

    def sqrt(self, z: complex, /) -> complex:
        """Compute the square root of a complex number."""
        return cmath.sqrt(z)

    def abs(self, z: complex, /) -> complex:
        """Compute the absolute value of a complex number."""
        return abs(z)

    def exp(self, z: complex, /) -> complex:
        """Exponentiate a complex number."""
        return cmath.exp(z)

    def exp10(self, z: complex, /) -> complex:
        """Compute base-10 exponentiation of a complex number."""
        return 10**z

    def log(self, z: complex, /) -> complex:
        """Compute natural logarithm of a complex number."""
        return cmath.log(z)

    def log10(self, z: complex, /) -> complex:
        """Compute base-10 logarithm of a complex number."""
        return cmath.log10(z)

    def adj(self, z: complex, /) -> complex:
        """Compute conjugate of a complex number."""
        return z.conjugate()

    def from_pyscalar(self, z: SupportsFloat | SupportsComplex, /) -> complex:
        """Convert real scalar to float."""
        return complex(z)


@final
class AsVectorSpace[K](alg.ImplementsVectorSpace[K, K]):
    """Implement scalar field as vector space over itself."""

    def __init__(self, scl: alg.ImplementsScalarField[K]):
        """Initialize AsVectorSpaceobjects."""
        self._scl = scl

    @property
    def scl(self) -> alg.ImplementsScalarField[K]:
        """Return scl property of AsVectorSpace object."""
        return self._scl

    def zero(self, /) -> K:
        """Return zero scalar as a vector."""
        return self._scl.zero()

    def add(self, x: K, y: K, /) -> K:
        """Add two scalars as vectors."""
        return self._scl.add(x, y)

    def sub(self, x: K, y: K, /) -> K:
        """Subtract two scalars as vectors."""
        return self._scl.sub(x, y)

    def neg(self, x: K, /) -> K:
        """Negate a scalar as a vector."""
        return self._scl.neg(x)

    def smul(self, x: K, y: K, /) -> K:
        """Perform multiplication of scalars as scalar multiplication."""
        return self._scl.mul(x, y)

    def sdiv(self, x: K, y: K, /) -> K:
        """Perform division of scalars as scalar division in vector space."""
        return self._scl.div(y, x)


@final
class AsRealVectorSpace[K](alg.ImplementsRealVectorSpace[K, K]):
    """Implement real scalar field as real vector space over itself."""

    def __init__(self, scl: alg.ImplementsRealScalarField[K]):
        """Initialize AsRealVectorSpaceobjects."""
        self._scl = scl

    @property
    def scl(self) -> alg.ImplementsRealScalarField[K]:
        """Return scl property of AsRealVectorSpace object."""
        return self._scl

    def zero(self, /) -> K:
        """Return zero real scalar as a vector."""
        return self._scl.zero()

    def add(self, x: K, y: K, /) -> K:
        """Add two real scalars as vectors."""
        return self._scl.add(x, y)

    def sub(self, x: K, y: K, /) -> K:
        """Subtract two real scalars as vectors."""
        return self._scl.sub(x, y)

    def neg(self, x: K, /) -> K:
        """Negate a scalar as a vector."""
        return self._scl.neg(x)

    def smul(self, x: K, y: K, /) -> K:
        """Perform multiplication of real scalars as scalar multiplication."""
        return self._scl.mul(x, y)

    def sdiv(self, x: K, y: K, /) -> K:
        """Perform scalar division of real scalars as scalar division."""
        return self._scl.div(y, x)


@final
class AsAlgebraWithCalculus[K](alg.ImplementsAlgebraWithCalculus[K, K]):
    """Implement scalar field as an algebra over itself."""

    def __init__(self, scl: alg.ImplementsRealScalarField[K]):
        """Initialize AsAlgebraWithCalculus objects."""
        self._scl = scl

    @property
    def scl(self) -> alg.ImplementsRealScalarField[K]:
        """Return scl property of AsAlgebra object."""
        return self._scl

    def zero(self, /) -> K:
        """Return zero real scalar as algebra element."""
        return self._scl.zero()

    def add(self, x: K, y: K, /) -> K:
        """Add two real scalars as algebra elements."""
        return self._scl.add(x, y)

    def sub(self, x: K, y: K, /) -> K:
        """Subtract two real scalars as algebra elements."""
        return self._scl.sub(x, y)

    def neg(self, x: K, /) -> K:
        """Negate a scalar as an algebra element."""
        return self._scl.neg(x)

    def smul(self, x: K, y: K, /) -> K:
        """Perform multiplication of real scalars as scalar multiplication."""
        return self._scl.mul(x, y)

    def sdiv(self, x: K, y: K, /) -> K:
        """Perform scalar division of real scalars as scalar division."""
        return self._scl.div(y, x)

    def unit(self, /) -> K:
        """Return unit real scalar as algebra element."""
        return self._scl.zero()

    def mul(self, x: K, y: K, /) -> K:
        """Multiply two real scalars as algebra elements."""
        return self._scl.mul(x, y)

    def div(self, x: K, y: K, /) -> K:
        """Divide two real scalars as algebra elements."""
        return self._scl.div(x, y)

    def inv(self, x: K, /) -> K:
        """Invert a real scalar as an algebra element."""
        return self._scl.inv(x)

    def mpower(self, x: K, k: int, /) -> K:
        """Compute monoidal power of a real scalar as an algebra element."""
        return self._scl.mpower(x, k)

    def power(self, x: K, y: K, /) -> K:
        """Compute power (exponentiation) for scalars as an algebra element."""
        return self._scl.power(x, y)

    def sqrt(self, x: K, /) -> K:
        """Compute square root of a scalar as an algebra element."""
        return self._scl.sqrt(x)

    def abs(self, x: K, /) -> K:
        """Compute absolute value of a real scalar as an algebra element."""
        return self._scl.abs(x)

    def exp(self, x: K, /) -> K:
        """Compute exponentiation of a real scalar as an algebra element."""
        return self._scl.exp(x)

    def log(self, x: K, /) -> K:
        """Compute natural logarithm of a real scalar as an algebra element."""
        return self._scl.log(x)


@final
class AsStarAlgebraWithCalculus[K](
    alg.ImplementsStarAlgebraWithCalculus[K, K]
):
    """Implement scalar field as an algebra over itself."""

    def __init__(self, scl: alg.ImplementsComplexScalarField[K]):
        """Initialize AsAlgebraWithCalculus objects."""
        self._scl = scl

    @property
    def scl(self) -> alg.ImplementsComplexScalarField[K]:
        """Return scl property of AsAlgebra object."""
        return self._scl

    def zero(self, /) -> K:
        """Return zero complex scalar as algebra element."""
        return self._scl.zero()

    def add(self, x: K, y: K, /) -> K:
        """Add two complex scalars as algebra elements."""
        return self._scl.add(x, y)

    def sub(self, x: K, y: K, /) -> K:
        """Subtract two complex scalars as algebra elements."""
        return self._scl.sub(x, y)

    def neg(self, x: K, /) -> K:
        """Negate a scalar as an algebra element."""
        return self._scl.neg(x)

    def smul(self, x: K, y: K, /) -> K:
        """Perform multiplication of complex scalars as scalar mult."""
        return self._scl.mul(x, y)

    def sdiv(self, x: K, y: K, /) -> K:
        """Perform scalar division of complex scalars as scalar division."""
        return self._scl.div(y, x)

    def unit(self, /) -> K:
        """Return unit complex scalar as algebra element."""
        return self._scl.zero()

    def mul(self, x: K, y: K, /) -> K:
        """Multiply two complex scalars as algebra elements."""
        return self._scl.mul(x, y)

    def div(self, x: K, y: K, /) -> K:
        """Divide two complex scalars as algebra elements."""
        return self._scl.div(x, y)

    def inv(self, x: K, /) -> K:
        """Invert a complex scalar as an algebra element."""
        return self._scl.inv(x)

    def mpower(self, x: K, k: int, /) -> K:
        """Compute monoidal power of a complex scalar as an algebra element."""
        return self._scl.mpower(x, k)

    def power(self, x: K, y: K, /) -> K:
        """Compute power (exponentiation) for scalars as an algebra element."""
        return self._scl.power(x, y)

    def sqrt(self, x: K, /) -> K:
        """Compute square root of a scalar as an algebra element."""
        return self._scl.sqrt(x)

    def adj(self, x: K, /) -> K:
        """Return adj property of AsAlgebra object."""
        return self._scl.adj(x)

    def abs(self, x: K, /) -> K:
        """Compute absolute value of a complex scalar as an algebra element."""
        return self._scl.abs(x)

    def exp(self, x: K, /) -> K:
        """Compute exponentiation of a complex scalar as an algebra element."""
        return self._scl.exp(x)

    def log(self, x: K, /) -> K:
        """Compute natural log of a complex scalar as an algebra element."""
        return self._scl.log(x)


@final
class AsDivBimodule[K](alg.ImplementsDivBimodule[K, K, K, K]):
    """Implement scalar field as bimodule over itself."""

    def __init__(self, scl: alg.ImplementsScalarField[K]):
        """Initialize bimodule implementation from scalar field."""
        self._scl = scl

    @property
    def scl(self) -> alg.ImplementsScalarField[K]:
        """Return scl property of AsDivBimodule object."""
        return self._scl

    def zero(self, /) -> K:
        """Return zero scalar as a vector."""
        return self._scl.zero()

    def add(self, x: K, y: K, /) -> K:
        """Add two scalars as vectors."""
        return self._scl.add(x, y)

    def sub(self, x: K, y: K, /) -> K:
        """Subtract two scalars as vectors."""
        return self._scl.sub(x, y)

    def neg(self, x: K, /) -> K:
        """Negate a scalar as a vector."""
        return self._scl.neg(x)

    def smul(self, x: K, y: K, /) -> K:
        """Perform multiplication of scalars as scalar multiplication."""
        return self._scl.mul(x, y)

    def sdiv(self, x: K, y: K, /) -> K:
        """Perform division of scalars as scalar division."""
        return self._scl.div(y, x)

    def lmul(self, x: K, y: K, /) -> K:
        """Return lmul property of AsBimodule object."""
        return self._scl.mul(x, y)

    def rmul(self, x: K, y: K, /) -> K:
        """Return rmul property of AsBimodule object."""
        return self._scl.mul(x, y)

    def ldiv(self, x: K, y: K, /) -> K:
        """Return ldiv property of AsBimodule object."""
        return self._scl.div(y, x)

    def rdiv(self, x: K, y: K, /) -> K:
        """Return rdiv property of AsBimodule object."""
        return self._scl.div(x, y)

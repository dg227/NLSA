"""Provide classes and functions implementing operations on scalar fields."""

import math
import cmath
import nlsa.abstract_algebra as alg
from collections.abc import Callable
from fractions import Fraction
from nlsa.utils import swap_args
from typing import SupportsComplex, SupportsFloat, Type, final


def make_zero[K: (int, Fraction, float, complex)](
    ty: Type[K],
) -> Callable[[], K]:
    """Make constant function that returns scalar zero."""

    def zero() -> K:
        return ty(0)

    return zero


def add[K: (int, Fraction, float, complex)](x: K, y: K, /) -> K:
    """Compute scalar addition."""
    return x + y


def neg[K: (int, Fraction, float, complex)](x: K, /) -> K:
    """Compute scalar negation."""
    return -1 * x


def sub[K: (int, Fraction, float, complex)](x: K, y: K, /) -> K:
    """Compute scalar subtraction."""
    return x - y


def make_unit[K: (int, Fraction, float, complex)](
    ty: Type[K],
) -> Callable[[], K]:
    """Make constant function that returns scalar one."""

    def unit() -> K:
        return ty(1)

    return unit


def mul[K: (int, Fraction, float, complex)](x: K, y: K, /) -> K:
    """Compute scalar multiplication."""
    return x * y


def complex_power(x: complex, n: int, /) -> complex:
    """Compute complex power."""
    return x**n


def floor_div(x: int, y: int, /) -> int:
    """Compute floor division."""
    return x // y


def div[K: (Fraction, float, complex)](x: K, y: K, /) -> K:
    """Compute scalar division."""
    return x / y


def floor_inv(x: int, /) -> int:
    """Compute floor inversion."""
    return 1 // x


def make_inv[K: (Fraction, float, complex)](ty: Type[K]) -> Callable[[K], K]:
    """Make scalar inversion function."""

    def inv(x: K, /) -> K:
        return ty(1) / x

    return inv


def complex_conj(x: complex, /) -> complex:
    """Compute complex conjugate."""
    return complex(x.conjugate())


@final
class FloatScalarField(alg.ImplementsRealScalarField[float]):
    """Implement scalar field operations on float objects."""

    @property
    def zero(self) -> Callable[[], float]:
        """Return zero property of FloatScalarField object."""
        return make_zero(float)

    @property
    def add(self) -> Callable[[float, float], float]:
        """Return add property of FloatScalarField object."""
        return add

    @property
    def sub(self) -> Callable[[float, float], float]:
        """Return sub property of FloatScalarField object."""
        return sub

    @property
    def neg(self) -> Callable[[float], float]:
        """Return neg property of FloatScalarField object."""
        return neg

    @property
    def unit(self) -> Callable[[], float]:
        """Return unit property of FloatScalarField object."""
        return make_unit(float)

    @property
    def mul(self) -> Callable[[float, float], float]:
        """Return mul property of FloatScalarField object."""
        return mul

    @property
    def mpower(self) -> Callable[[float, int], float]:
        """Return mpower property of FloatScalarField object."""
        return math.pow

    @property
    def power(self) -> Callable[[float, float], float]:
        """Return power property of FloatScalarField object."""
        return math.pow

    @property
    def div(self) -> Callable[[float, float], float]:
        """Return div property of FloatScalarField object."""
        return div

    @property
    def inv(self) -> Callable[[float], float]:
        """Return inv property of FloatScalarField object."""
        return make_inv(float)

    @property
    def sqrt(self) -> Callable[[float], float]:
        """Return sqrt property of FloatScalarField object."""
        return math.sqrt

    @property
    def abs(self) -> Callable[[float], float]:
        """Return abs property of FloatScalarField object."""
        return abs

    @property
    def exp(self) -> Callable[[float], float]:
        """Return exp property of FloatScalarField object."""
        return math.exp

    @property
    def exp10(self) -> Callable[[float], float]:
        """Return exp10 property of FloatScalarField object."""
        return lambda x: 10**x

    @property
    def log(self) -> Callable[[float], float]:
        """Return log property of FloatScalarField object."""
        return math.log

    @property
    def log10(self) -> Callable[[float], float]:
        """Return log10 property of FloatScalarField object."""
        return math.log10

    @property
    def from_pyscalar(self) -> Callable[[SupportsFloat], float]:
        """Return from_pyscalar property of FloatScalarField object."""
        return lambda x: float(x)


@final
class ComplexScalarField(alg.ImplementsComplexScalarField[complex]):
    """Implement scalar field operations on complex objects."""

    @property
    def zero(self) -> Callable[[], complex]:
        """Return zero property of ComplexScalarField object."""
        return make_zero(complex)

    @property
    def add(self) -> Callable[[complex, complex], complex]:
        """Return add property of ComplexScalarField object."""
        return add

    @property
    def sub(self) -> Callable[[complex, complex], complex]:
        """Return sub property of ComplexScalarField object."""
        return sub

    @property
    def neg(self) -> Callable[[complex], complex]:
        """Return neg property of ComplexScalarField object."""
        return neg

    @property
    def unit(self) -> Callable[[], complex]:
        """Return unit property of ComplexScalarField object."""
        return make_unit(complex)

    @property
    def mul(self) -> Callable[[complex, complex], complex]:
        """Return mul property of ComplexScalarField object."""
        return mul

    @property
    def mpower(self) -> Callable[[complex, int], complex]:
        """Return mpower property of ComplexScalarField object."""
        return lambda z, n: z**n

    @property
    def power(self) -> Callable[[complex, complex], complex]:
        """Return power property of ComplexScalarField object."""
        return lambda z, w: z**w

    @property
    def div(self) -> Callable[[complex, complex], complex]:
        """Return div property of ComplexScalarField object."""
        return div

    @property
    def inv(self) -> Callable[[complex], complex]:
        """Return inv property of ComplexScalarField object."""
        return make_inv(complex)

    @property
    def sqrt(self) -> Callable[[complex], complex]:
        """Return sqrt property of ComplexScalarField object."""
        return cmath.sqrt

    @property
    def abs(self) -> Callable[[complex], complex]:
        """Return abs property of ComplexScalarField object."""
        return abs

    @property
    def exp(self) -> Callable[[complex], complex]:
        """Return exp property of ComplexScalarField object."""
        return cmath.exp

    @property
    def exp10(self) -> Callable[[complex], complex]:
        """Return exp10 property of ComplexScalarField object."""
        return lambda x: 10**x

    @property
    def log(self) -> Callable[[complex], complex]:
        """Return log property of ComplexScalarField object."""
        return cmath.log

    @property
    def log10(self) -> Callable[[complex], complex]:
        """Return log10 property of ComplexScalarField object."""
        return cmath.log10

    @property
    def adj(self) -> Callable[[complex], complex]:
        """Return adj property of ComplexScalarField object."""
        return lambda z: z.conjugate()

    @property
    def from_pyscalar(
        self,
    ) -> Callable[[SupportsFloat | SupportsComplex], complex]:
        """Return from_pyscalar property of ComplexScalarField object."""
        return lambda x: complex(x)


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

    @property
    def zero(self) -> Callable[[], K]:
        """Return zero property of AsVectorSpace object."""
        return self._scl.zero

    @property
    def add(self) -> Callable[[K, K], K]:
        """Return add property of AsVectorSpace object."""
        return self._scl.add

    @property
    def sub(self) -> Callable[[K, K], K]:
        """Return sub property of AsVectorSpace object."""
        return self._scl.sub

    @property
    def neg(self) -> Callable[[K], K]:
        """Return neg property of AsVectorSpace object."""
        return self._scl.neg

    @property
    def smul(self) -> Callable[[K, K], K]:
        """Return smul property of AsVectorSpace object."""
        return self._scl.mul

    @property
    def sdiv(self) -> Callable[[K, K], K]:
        """Return sdiv property of AsVectorSpace object."""
        return swap_args(self._scl.div)


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

    @property
    def zero(self) -> Callable[[], K]:
        """Return zero property of AsRealVectorSpace object."""
        return self._scl.zero

    @property
    def add(self) -> Callable[[K, K], K]:
        """Return add property of AsRealVectorSpace object."""
        return self._scl.add

    @property
    def sub(self) -> Callable[[K, K], K]:
        """Return sub property of AsRealVectorSpace object."""
        return self._scl.sub

    @property
    def neg(self) -> Callable[[K], K]:
        """Return neg property of AsRealVectorSpace object."""
        return self._scl.neg

    @property
    def smul(self) -> Callable[[K, K], K]:
        """Return smul property of AsRealVectorSpace object."""
        return self._scl.mul

    @property
    def sdiv(self) -> Callable[[K, K], K]:
        """Return sdiv property of AsRealVectorSpace object."""
        return swap_args(self._scl.div)


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

    @property
    def zero(self) -> Callable[[], K]:
        """Return zero property of AsAlgebra object."""
        return self._scl.zero

    @property
    def add(self) -> Callable[[K, K], K]:
        """Return add property of AsAlgebra object."""
        return self._scl.add

    @property
    def sub(self) -> Callable[[K, K], K]:
        """Return sub property of AsAlgebra object."""
        return self._scl.sub

    @property
    def neg(self) -> Callable[[K], K]:
        """Return neg property of AsAlgebra object."""
        return self._scl.neg

    @property
    def smul(self) -> Callable[[K, K], K]:
        """Return smul property of AsAlgebra object."""
        return self._scl.mul

    @property
    def sdiv(self) -> Callable[[K, K], K]:
        """Return sdiv property of AsAlgebra object."""
        return swap_args(self._scl.div)

    @property
    def unit(self) -> Callable[[], K]:
        """Return unit property of AsAlgebra object."""
        return self._scl.zero

    @property
    def mul(self) -> Callable[[K, K], K]:
        """Return mul property of AsAlgebra object."""
        return self._scl.mul

    @property
    def div(self) -> Callable[[K, K], K]:
        """Return div property of AsAlgebra object."""
        return self._scl.div

    @property
    def inv(self) -> Callable[[K], K]:
        """Return inv property of AsAlgebra object."""
        return self._scl.inv

    @property
    def mpower(self) -> Callable[[K, int], K]:
        """Return mpower property of AsAlgebra object."""
        return self._scl.mpower

    @property
    def power(self) -> Callable[[K, K], K]:
        """Return power property of AsAlgebra object."""
        return self._scl.power

    @property
    def sqrt(self) -> Callable[[K], K]:
        """Return sqrt property of AsAlgebra object."""
        return self._scl.sqrt

    @property
    def abs(self) -> Callable[[K], K]:
        """Return mod property of AsAlgebra object."""
        return self._scl.abs


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

    @property
    def zero(self) -> Callable[[], K]:
        """Return zero property of AsAlgebra object."""
        return self._scl.zero

    @property
    def add(self) -> Callable[[K, K], K]:
        """Return add property of AsAlgebra object."""
        return self._scl.add

    @property
    def sub(self) -> Callable[[K, K], K]:
        """Return sub property of AsAlgebra object."""
        return self._scl.sub

    @property
    def neg(self) -> Callable[[K], K]:
        """Return neg property of AsAlgebra object."""
        return self._scl.neg

    @property
    def smul(self) -> Callable[[K, K], K]:
        """Return smul property of AsAlgebra object."""
        return self._scl.mul

    @property
    def sdiv(self) -> Callable[[K, K], K]:
        """Return sdiv property of AsAlgebra object."""
        return swap_args(self._scl.div)

    @property
    def unit(self) -> Callable[[], K]:
        """Return unit property of AsAlgebra object."""
        return self._scl.zero

    @property
    def mul(self) -> Callable[[K, K], K]:
        """Return mul property of AsAlgebra object."""
        return self._scl.mul

    @property
    def div(self) -> Callable[[K, K], K]:
        """Return div property of AsAlgebra object."""
        return self._scl.div

    @property
    def inv(self) -> Callable[[K], K]:
        """Return inv property of AsAlgebra object."""
        return self._scl.inv

    @property
    def mpower(self) -> Callable[[K, int], K]:
        """Return mpower property of AsAlgebra object."""
        return self._scl.mpower

    @property
    def power(self) -> Callable[[K, K], K]:
        """Return power property of AsAlgebra object."""
        return self._scl.power

    @property
    def sqrt(self) -> Callable[[K], K]:
        """Return sqrt property of AsAlgebra object."""
        return self._scl.sqrt

    @property
    def adj(self) -> Callable[[K], K]:
        """Return adj property of AsAlgebra object."""
        return self._scl.adj

    @property
    def abs(self) -> Callable[[K], K]:
        """Return mod property of AsAlgebra object."""
        return self._scl.abs


@final
class AsDivBimodule[K](alg.ImplementsDivBimodule[K, K, K, K]):
    """Implement scalar field as bimodule over itself."""

    def __init__(self, scl: alg.ImplementsScalarField[K]):
        """Initialize bimodule implementation from scalar field."""
        self._scl = scl

    @property
    def scl(self) -> alg.ImplementsScalarField[K]:
        """Return scl property of AsAlgebra object."""
        return self._scl

    @property
    def zero(self) -> Callable[[], K]:
        """Return zero property of AsBimodule object."""
        return self._scl.zero

    @property
    def add(self) -> Callable[[K, K], K]:
        """Return add property of AsBimodule object."""
        return self._scl.add

    @property
    def sub(self) -> Callable[[K, K], K]:
        """Return sub property of AsBimodule object."""
        return self._scl.sub

    @property
    def neg(self) -> Callable[[K], K]:
        """Return neg property of AsBimodule object."""
        return self._scl.neg

    @property
    def smul(self) -> Callable[[K, K], K]:
        """Return smul property of AsBimodule object."""
        return self._scl.mul

    @property
    def sdiv(self) -> Callable[[K, K], K]:
        """Return sdiv property of AsAlgebra object."""
        return swap_args(self._scl.div)

    @property
    def lmul(self) -> Callable[[K, K], K]:
        """Return lmul property of AsBimodule object."""
        return self._scl.mul

    @property
    def rmul(self) -> Callable[[K, K], K]:
        """Return rmul property of AsBimodule object."""
        return self._scl.mul

    @property
    def ldiv(self) -> Callable[[K, K], K]:
        """Return ldiv property of AsBimodule object."""
        return swap_args(self._scl.div)

    @property
    def rdiv(self) -> Callable[[K, K], K]:
        """Return rdiv property of AsBimodule object."""
        return self._scl.div

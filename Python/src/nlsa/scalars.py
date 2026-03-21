"""Provide classes and functions implementing operations on scalar fields."""

import math
import nlsa.abstract_algebra as alg
import nlsa.function_algebra as fun
from collections.abc import Callable
from fractions import Fraction
from nlsa.utils import swap_args
from typing import Type, final


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


# Pyright fails when attempting to generalize make_real_power to complex types.
def make_real_power[K: (int, Fraction, float)](
    ty: Type[K],
) -> Callable[[K, K], K]:
    """Make exponentiation function of reals by integers."""

    def power(x: K, y: K, /) -> K:
        return ty(x**y)

    return power


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
class FloatScalarField(alg.ImplementsScalarField[float]):
    """Implement scalar field operations on float objects."""

    def __init__(self):
        """Initialize scalar field on float objects."""
        self.zero: Callable[[], float] = make_zero(float)
        self.add: Callable[[float, float], float] = add
        self.sub: Callable[[float, float], float] = sub
        self.neg: Callable[[float], float] = neg
        self.unit: Callable[[], float] = make_unit(float)
        self.mul: Callable[[float, float], float] = mul
        self.mpower: Callable[[float, int], float] = math.pow
        self.power: Callable[[float, float], float] = make_real_power(float)
        self.div: Callable[[float, float], float] = div
        self.inv: Callable[[float], float] = make_inv(float)
        self.adj: Callable[[float], float] = fun.identity
        self.sqrt: Callable[[float], float] = math.sqrt
        self.mod: Callable[[float], float] = abs
        self.exp: Callable[[float], float] = math.exp


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
class AsAlgebraWithCalculus[K](alg.ImplementsAlgebraWithCalculus[K, K]):
    """Implement scalar field as an algebra over itself."""

    def __init__(self, scl: alg.ImplementsScalarField[K]):
        """Initialize AsAlgebraWithCalculus objects."""
        self._scl = scl

    @property
    def scl(self) -> alg.ImplementsScalarField[K]:
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
    def mod(self) -> Callable[[K], K]:
        """Return mod property of AsAlgebra object."""
        return self._scl.mod


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

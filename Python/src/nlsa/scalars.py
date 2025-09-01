"""Provide classes and functions implementing operations on scalar fields.
"""

import math
import nlsa.abstract_algebra as alg
import nlsa.function_algebra as fun
from collections.abc import Callable
from fractions import Fraction
from nlsa.utils import swap_args
from typing import Type, final


def make_zero[K: (int, Fraction, float, complex)](ty: Type[K]) \
        -> Callable[[], K]:
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


def make_unit[K: (int, Fraction, float, complex)](ty: Type[K]) \
        -> Callable[[], K]:
    """Make constant function that returns scalar one."""
    def unit() -> K:
        return ty(1)
    return unit


def mul[K: (int, Fraction, float, complex)](x: K, y: K, /) -> K:
    """Compute scalar multiplication."""
    return x * y


# Pyright fails when attempting to generalize make_real_power to complex types.
def make_real_power[K: (int, Fraction, float)](ty: Type[K]) \
        -> Callable[[K, K], K]:
    """Make exponentiation function of reals by integers."""
    def power(x: K, y: K, /) -> K:
        return ty(x ** y)
    return power


def complex_power(x: complex, n: int, /) -> complex:
    return x ** n


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
    """Implement scalar field operations."""
    def __init__(self):
        self.zero: Callable[[], float] = make_zero(float)
        self.add: Callable[[float, float], float] = add
        self.sub: Callable[[float, float], float] = sub
        self.neg: Callable[[float], float] = neg
        self.unit: Callable[[], float] = make_unit(float)
        self.mul: Callable[[float, float], float] = mul
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
        self.scl = scl
        self.zero: Callable[[], K] = scl.zero
        self.add: Callable[[K, K], K] = scl.add
        self.sub: Callable[[K, K], K] = scl.sub
        self.neg: Callable[[K], K] = scl.neg
        self.smul: Callable[[K, K], K] = scl.mul
        self.sdiv: Callable[[K, K], K] = swap_args(scl.div)


@final
class AsAlgebra[K](alg.ImplementsAlgebra[K, K]):
    """Implement scalar field as an algebra over itself."""
    def __init__(self, scl: alg.ImplementsScalarField[K]):
        self.scl = scl
        self.zero: Callable[[], K] = scl.zero
        self.add: Callable[[K, K], K] = scl.add
        self.sub: Callable[[K, K], K] = scl.sub
        self.neg: Callable[[K], K] = scl.neg
        self.smul: Callable[[K, K], K] = scl.mul
        self.sdiv: Callable[[K, K], K] = swap_args(scl.div)
        self.unit: Callable[[], K] = scl.zero
        self.mul: Callable[[K, K], K] = scl.mul
        self.div: Callable[[K, K], K] = scl.div
        self.inv: Callable[[K], K] = scl.inv
        self.power: Callable[[K, K], K] = scl.power
        self.sqrt: Callable[[K], K] = scl.sqrt
        self.adj: Callable[[K], K] = scl.adj
        self.mod: Callable[[K], K] = scl.mod


@final
class AsBimodule[K](alg.ImplementsBimodule[K, K, K, K]):
    """Implement scalar field as bimodule over itself."""
    def __init__(self, scl: alg.ImplementsScalarField[K]):
        self.scl = scl
        self.zero: Callable[[], K] = scl.zero
        self.add: Callable[[K, K], K] = scl.add
        self.sub: Callable[[K, K], K] = scl.sub
        self.neg: Callable[[K], K] = scl.neg
        self.smul: Callable[[K, K], K] = scl.mul
        self.sdiv: Callable[[K, K], K] = swap_args(scl.div)
        self.lmul: Callable[[K, K], K] = scl.mul
        self.rmul: Callable[[K, K], K] = scl.mul
        self.ldiv: Callable[[K, K], K] = swap_args(scl.div)
        self.rdiv: Callable[[K, K], K] = scl.div

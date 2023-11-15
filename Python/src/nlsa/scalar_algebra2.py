import math
import numpy as np
from fractions import Fraction
from typing import Callable, Generic, Type, TypeVar, overload

Z = TypeVar('Z', bound=np.integer)
Q = Fraction | int
R = TypeVar('R', bound=np.floating)
C = TypeVar('C', bound=np.complexfloating)


@overload
def make_unit(dtype: Type[Z]) -> Callable[[], Z]:
    pass


@overload
def make_unit(dtype: Type[R]) -> Callable[[], R]:
    pass


@overload
def make_unit(dtype: Type[C]) -> Callable[[], C]:
    pass


def make_unit(dtype: Type[Z] | Type[R] | Type[C]) -> Callable[[], Z | R | C]:
    def unit() -> Z | R | C:
        return dtype(1)
    return unit


class IntField:
    """Implement field operations for Python int type."""

    @staticmethod
    def add(a: int, b: int, /) -> int:
        return a + b

    @staticmethod
    def sub(a: int, b: int, /) -> int:
        return a - b

    @staticmethod
    def neg(a: int, /) -> int:
        return -a

    @staticmethod
    def mul(a: int, b: int, /) -> int:
        return a * b

    @staticmethod
    def unit() -> int:
        return 1

    @staticmethod
    def div(a: int, b: int, /) -> int:
        return a // b

    @staticmethod
    def inv(a: int, /) -> int:
        return 1 // a

    @staticmethod
    def power(a: int, b: int, /) -> int:
        return a ** b

    @staticmethod
    def star(a: int, /) -> int:
        return a.conjugate()


class IntegerField(Generic[Z]):
    """Implement field operations for NumPy integer types."""

    def __init__(self, dtype: Type[Z]):
        self.unit: Callable[[], Z] = make_unit(dtype)

    @staticmethod
    def add(a: Z, b: Z, /) -> Z:
        return np.add(a, b)

    @staticmethod
    def sub(a: Z, b: Z, /) -> Z:
        return np.subtract(a, b)

    @staticmethod
    def neg(a: Z, /) -> Z:
        return -a

    @staticmethod
    def mul(a: Z, b: Z, /) -> Z:
        return np.multiply(a, b)

    @staticmethod
    def smul(a: Z, b: Z, /) -> Z:
        return np.multiply(a, b)

    @staticmethod
    def div(a: Z, b: Z, /) -> Z:
        return np.floor_divide(a, b)

    @staticmethod
    def inv(a: Z, /) -> Z:
        return np.floor_divide(1, a)

    @staticmethod
    def power(a: Z, b: Z, /) -> Z:
        return np.power(a, b)

    @staticmethod
    def star(a: Z, /) -> Z:
        return a.conjugate()


class RationalField:
    """Implement field operations for rational numbers."""

    @staticmethod
    def add(a: Q, b: Q, /) -> Q:
        return a + b

    @staticmethod
    def sub(a: Q, b: Q, /) -> Q:
        return a - b

    @staticmethod
    def neg(a: Q, /) -> Q:
        return -a

    @staticmethod
    def mul(a: Q, b: Q, /) -> Q:
        return a * b

    @staticmethod
    def unit() -> Q:
        return 1

    @staticmethod
    def div(a: Q, b: Q, /) -> Q:
        return Fraction(a) / b

    @staticmethod
    def inv(a: Q, /) -> Q:
        return 1 / Fraction(a)

    @staticmethod
    def star(a: Q, /) -> Q:
        return a.conjugate()


class FloatField:
    """Implement field operations for Python float type."""

    @staticmethod
    def add(a: float, b: float, /) -> float:
        return a + b

    @staticmethod
    def sub(a: float, b: float, /) -> float:
        return a - b

    @staticmethod
    def neg(a: float, /) -> float:
        return -a

    @staticmethod
    def mul(a: float, b: float, /) -> float:
        return a * b

    @staticmethod
    def unit() -> float:
        return 1

    @staticmethod
    def div(a: float, b: float, /) -> float:
        return a / b

    @staticmethod
    def inv(a: float, /) -> float:
        return 1 / a

    @staticmethod
    def star(a: float, /) -> float:
        return a.conjugate()

    @staticmethod
    def sqrt(a: float, /) -> float:
        return math.sqrt(a)

    @staticmethod
    def exp(a: float, /) -> float:
        return math.exp(a)

    @staticmethod
    def power(a: float, b: float, /) -> float:
        return a ** b


class FloatingField(Generic[R]):
    """Implement field operations for NumPy floating types."""

    def __init__(self, dtype: Type[R]):
        self.unit: Callable[[], R] = make_unit(dtype)

    @staticmethod
    def add(a: R, b: R, /) -> R:
        return np.add(a, b)

    @staticmethod
    def sub(a: R, b: R, /) -> R:
        return np.subtract(a, b)

    @staticmethod
    def neg(a: R, /) -> R:
        return -a

    @staticmethod
    def mul(a: R, b: R, /) -> R:
        return np.multiply(a, b)

    @staticmethod
    def div(a: R, b: R, /) -> R:
        return np.divide(a, b)

    @staticmethod
    def inv(a: R, /) -> R:
        return np.divide(1, a)

    @staticmethod
    def star(a: R, /) -> R:
        return a.conjugate()

    @staticmethod
    def sqrt(a: R, /) -> R:
        return np.sqrt(a)

    @staticmethod
    def exp(a: R, /) -> R:
        return np.exp(a)

    @staticmethod
    def power(a: R, b: R, /) -> R:
        return np.power(a, b)


class ComplexField:
    """Implement complex field operations for Python complex type."""

    @staticmethod
    def add(a: complex, b: complex, /) -> complex:
        return a + b

    @staticmethod
    def sub(a: complex, b: complex, /) -> complex:
        return a - b

    @staticmethod
    def neg(a: complex, /) -> complex:
        return -a

    @staticmethod
    def mul(a: complex, b: complex, /) -> complex:
        return a * b

    @staticmethod
    def unit() -> complex:
        return 1

    @staticmethod
    def div(a: complex, b: complex, /) -> complex:
        return a / b

    @staticmethod
    def inv(a: complex, /) -> complex:
        return 1 / a

    @staticmethod
    def star(a: complex, /) -> complex:
        return a.conjugate()

    @staticmethod
    def sqrt(a: complex, /) -> complex:
        return np.sqrt(a)

    @staticmethod
    def exp(a: complex, /) -> complex:
        return np.exp(a)

    @staticmethod
    def power(a: complex, b: complex, /) -> complex:
        return a ** b


class ComplexfloatingField(Generic[C]):
    """Implement field operations for NumPy complexfloating types."""

    def __init__(self, dtype: Type[C]):
        self.unit: Callable[[], C] = make_unit(dtype)

    @staticmethod
    def add(a: C, b: C, /) -> C:
        return np.add(a, b)

    @staticmethod
    def sub(a: C, b: C, /) -> C:
        return np.subtract(a, b)

    @staticmethod
    def neg(a: C, /) -> C:
        return -a

    @staticmethod
    def mul(a: C, b: C, /) -> C:
        return np.multiply(a, b)

    @staticmethod
    def div(a: C, b: C, /) -> C:
        return np.divide(a, b)

    @staticmethod
    def inv(a: C, /) -> C:
        return np.divide(1, a)

    @staticmethod
    def star(a: C, /) -> C:
        return a.conjugate()

    @staticmethod
    def sqrt(a: C, /) -> C:
        return np.sqrt(a)

    @staticmethod
    def exp(a: complex, /) -> complex:
        return np.exp(a)

    @staticmethod
    def power(a: C, b: C, /) -> C:
        return np.power(a, b)

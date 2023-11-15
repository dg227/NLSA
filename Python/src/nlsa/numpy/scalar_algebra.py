import numpy as np
import nlsa.abstract_algebra2 as alg
from numpy.typing import NDArray
from typing import Callable, Generic, Type, TypeVar

N = TypeVar('N', bound=int)
K = TypeVar('K', bound=np.generic)
R = np.float64
S = NDArray[K]
V = NDArray[K]
X = TypeVar('X')
Y = TypeVar('Y')
F = Callable[[X], Y]

# TODO: When numpy.typing supports variadic generics for type hinting array
# dimensions, the functions make_unit and make_inv could be made generic with N
# as the type parameter for dim.


def neg(s: S[K], /) -> S[K]:
    """Negate a scalar."""
    return np.multiply(-1, s)


def make_unit(dtype: Type[K]) -> Callable[[], S[K]]:
    """Make constant function returning scalar unit."""
    def unit() -> S[K]:
        return np.ones(1, dtype=dtype)
    return unit


def make_inv(dtype: Type[K]) -> Callable[[S[K]], S[K]]:
    """Make inversion function for scalars."""
    def inv(s: S[K]) -> S[K]:
        return np.divide(np.ones(1, dtype=dtype), s)
    return inv


def sldiv(s: S[K], t: S[K], /) -> S[K]:
    """Left-divide two scalars.

    Modified version for broadcasting.
    """
    return np.divide(t, np.atleast_2d(s).T)


class ScalarField(alg.ImplementsAlgebraLRModule[S[K], S[K], S[K], S[K]],
                  alg.ImplementsUnitalStarAlgebraLRModule[S[K], S[K], S[K], S[K]],
                  Generic[K]):
    """Implement scalar field operations for NumPy arrays.

    The type parameter K parameterizes the field of scalars.
    """

    def __init__(self, dtype: Type[K]):
        self.add: Callable[[S[K], S[K]], S[K]] = np.add
        self.neg: Callable[[S[K]], S[K]] = neg
        self.sub: Callable[[S[K], S[K]], S[K]] = np.subtract
        self.mul: Callable[[S[K], S[K]], S[K]] = np.multiply
        self.unit: Callable[[], S[K]] = make_unit(dtype)
        self.inv: Callable[[S[K]], S[K]] = make_inv(dtype)
        self.div: Callable[[S[K], S[K]], S[K]] = np.divide
        self.star: Callable[[S[K]], S[K]] = np.conjugate
        self.lmul: Callable[[S[K], S[K]], S[K]] = np.multiply
        self.ldiv: Callable[[S[K], S[K]], S[K]] = sldiv
        self.sqrt: Callable[[S[K]], S[K]] = np.sqrt
        self.exp: Callable[[S[K]], S[K]] = np.exp
        self.power: Callable[[S[K], S[K]], S[K]] = np.power

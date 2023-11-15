import numpy as np
import nlsa.abstract_algebra2 as alg
from nlsa.numpy.scalar_algebra import ScalarField
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
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


def neg(v: V[K], /) -> V[K]:
    """Negate a vector."""
    return np.multiply(-1, v)


def make_unit(dim: int, dtype: Type[K]) -> Callable[[], V[K]]:
    """Make constant function returning vector of all 1s."""
    def unit() -> V[K]:
        return np.ones((1, dim), dtype=dtype)
    return unit


def make_inv(dim: int, dtype: Type[K]) -> Callable[[V[K]], V[K]]:
    """Make inversion function for specified dimension and dtype."""
    def inv(v: V[K]) -> V[K]:
        return np.divide(np.ones(dim, dtype=dtype), v)
    return inv


def ldiv(u: V[K], v: V[K], /) -> V[K]:
    """Perform left module division as elementwise vector division."""
    match (u.ndim, v.ndim):
        case (1, 1):
            w = np.divide(v, u)
        case _:
            w = np.divide(v, np.atleast_2d(u).T)
    return w


def counting_measure(v: V[K], /) -> S[K]:
    """Sum the elements of a vector."""
    return np.atleast_2d(np.sum(v, axis=-1))


def eval_at(x: X) -> Callable[[F[X, S[K]]], V[K]]:
    """Make evaluation functional at points or collection of points.

    If evaluating at a collection of points (i.e., x is an NDArray), eval_at(x)
    assumes that the functions that it acts on are generalized ufuncs.
    """

    def evalx(f: F[X, S[K]]) -> V[K]:
        y = f(x)
        return np.atleast_2d(y)
        # if y.shape[-1] == 1:
        #     return np.squeeze(y, axis=-1)
        # else:
        #     return y
    return evalx


def sqeuclidean(u: V[R], v: V[R]) -> S[R]:
    """Compute pairwise squared Euclidean distance."""
    s2: S[R] = cdist(np.atleast_2d(u), np.atleast_2d(v), 'sqeuclidean')
    return s2


class VectorAlgebra(Generic[N, K]):
    """Implement vector algebra operations for NumPy arrays.

    The type parameter N parameterizes the dimension of the algebra. The type
    parameter K parameterizes the field of scalars.
    """

    def __init__(self, dim: N, dtype: Type[K]):
        self.scl = ScalarField(dtype)
        self.add: Callable[[V[K], V[K]], V[K]] = np.add
        self.neg: Callable[[V[K]], V[K]] = neg
        self.sub: Callable[[V[K], V[K]], V[K]] = np.subtract
        self.smul: Callable[[S[K], V[K]], V[K]] = np.multiply
        self.mul: Callable[[V[K], V[K]], V[K]] = np.multiply
        self.unit: Callable[[], V[K]] = make_unit(dim, dtype)
        self.inv: Callable[[V[K]], V[K]] = make_inv(dim, dtype)
        self.div: Callable[[V[K], V[K]], V[K]] = np.divide
        self.star: Callable[[V[K]], V[K]] = np.conjugate
        self.lmul: Callable[[V[K], V[K]], V[K]] = np.multiply
        self.ldiv: Callable[[V[K], V[K]], V[K]] = ldiv
        self.rmul: Callable[[V[K], V[K]], V[K]] = np.multiply
        self.rdiv: Callable[[V[K], V[K]], V[K]] = np.divide
        self.sqrt: Callable[[V[K]], V[K]] = np.sqrt
        self.exp: Callable[[V[K]], V[K]] = np.exp
        self.power: Callable[[V[K], V[K]], V[K]] = np.power

    def innerp(self, u: V[K], v: V[K], /) -> S[K]:
        """Compute inner product of vectors."""
        w = np.sum(np.multiply(np.conjugate(u), v), axis=-1, keepdims=True)
        # w = np.einsum('...i,...i->...', np.conjugate(u), v)
        return w


class MeasurableFnAlgebra(VectorAlgebra[N, K],
                          alg.ImplementsMeasurableFnAlgebra[X, V[K], S[K]],
                          Generic[X, N, K]):
    """Implement operations on equivalence classes of functions using NumPy
    arrays as the representation type.

    """
    def __init__(self, dim: N, dtype: Type[K],
                 inclusion_map: Callable[[F[X, S[K]]], V[K]]):
        super().__init__(dim, dtype)
        self.incl: Callable[[F[X, S[K]]], V[K]] = inclusion_map


class MeasureFnAlgebra(MeasurableFnAlgebra[X, N, K],
                       alg.ImplementsMeasureFnAlgebra[X, V[K], S[K]]):
    """Implement MeasurableFunctionAlgebra equipped with measure."""
    def __init__(self, dim: N, dtype: Type[K],
                 inclusion_map: Callable[[F[X, S[K]]], V[K]],
                 measure: Callable[[V[K]], S[K]]):
        super().__init__(dim, dtype, inclusion_map)
        self.integrate: Callable[[V[K]], S[K]] = measure


# Old methods of ScalarField class:

# self.unit: Callable[[], S[K]] = make_sunit(dtype)

# def add(self, u: S[K], v: S[K], /) -> S[K]:
#     """Add two scalars."""
#     return np.add(u, v)

# def sub(self, u: S[K], v: S[K], /) -> S[K]:
#     """Subtract two scalars."""
#     return np.subtract(u, v)

# def neg(self, v: S[K], /) -> S[K]:
#     """Negate a scalar."""
#     return np.multiply(-1, v)

# def mul(self, u: S[K], v: S[K], /) -> S[K]:
#     """Multiply two scalars."""
#     return np.multiply(u, v)

# def inv(self, v: S[K], /) -> S[K]:
#     """Invert a scalar."""
#     return np.divide(self.unit(), v)

# def div(self, u: S[K], v: S[K], /) -> S[K]:
#     """Divide two scalars."""
#     return np.divide(u, v)

# def lmul(self, u: S[K], v: S[K], /) -> S[K]:
#     """Left-multiply two scalars. """
#     # TODO: Check whether this is the right definiction for broadcasting/
#     return np.multiply(u, v)

# def ldiv(self, u: S[K], v: S[K], /) -> S[K]:
#     """Left-divide two scalars.

#     Modified version for broadcasting.
#     """
#     return np.divide(v, np.atleast_2d(u).T)

# def sqrt(self, v: S[K], /) -> S[K]:
#     """Compute square root of scalar."""
#     return np.sqrt(v)

# def exp(self, v: S[K], /) -> S[K]:
#     """Compute exponential function on scalar."""
#     return np.exp(v)

# def power(self, v: S[K], k: S[K], /) -> S[K]:
#     """Compute exponentiation of scalar by scalar."""
#     return np.power(v, k)

# def star(self, v: S[K], /) -> S[K]:
#     """Compute complex conjugation of scalar."""
#     return np.conjugate(v)

# def unit(self) -> S[K]:
#     return self._unit()


# Methods from old VectorAlgebra class
# def add(self, u: V[K], v: V[K], /) -> V[K]:
#     """Add two vectors."""
#     return np.add(u, v)

# def sub(self, u: V[K], v: V[K], /) -> V[K]:
#     """Subtract two vectors."""
#     return np.subtract(u, v)

# def neg(self, v: V[K], /) -> V[K]:
#     """Negate a vector."""
#     return np.multiply(-1, v)

# def smul(self, k: S[K], v: V[K], /) -> V[K]:
#     """Multiply a scalar and a vector."""
#     return np.multiply(k, v)

# def mul(self, u: V[K], v: V[K], /) -> V[K]:
#     """Multiply two vectors elementwise."""
#     return np.multiply(u, v)

# def inv(self, v: V[K], /) -> V[K]:
#     """Invert a vector elementwise."""
#     return np.divide(self.unit(), v)

# def unit(self) -> V[K]:
#     return self._unit()

# def div(self, u: V[K], v: V[K], /) -> V[K]:
#     """Divide two vectors elementwise."""
#     return np.divide(u, v)

# def lmul(self, u: V[K], v: V[K], /) -> V[K]:
#     """Perform left module multiplication as elementwise vector
#     multiplication.

#     """
#     return np.multiply(u, v)

# def ldiv(self, u: V[K], v: V[K], /) -> V[K]:
#     """Perform left module division as elementwise vector division."""
#     match (u.ndim, v.ndim):
#         case (1, 1):
#             w = np.divide(v, u)
#         case _:
#             w = np.divide(v, np.atleast_2d(u).T)
#     return w

# def rmul(self, u: V[K], v: V[K], /) -> V[K]:
#     """Perform right module multiplication as elementwise vector
#     multiplication.

#     """
#     return np.multiply(u, v)

# def rdiv(self, u: V[K], v: V[K], /) -> V[K]:
#     """Perform right module division as elementwise vector division."""
#     return np.divide(u, v)

# def sqrt(self, v: V[K], /) -> V[K]:
#     """Compute elementwise square root of vector."""
#     return np.sqrt(v)

# def exp(self, v: V[K], /) -> V[K]:
#     """Compute elementwise exponential function of vector."""
#     return np.exp(v)

# def power(self, v: V[K], k: S[K], /) -> V[K]:
#     """Compute elementwise exponentiation of vector by scalar."""
#     return np.power(v, k)

# def star(self, v: V[K], /) -> V[K]:
#     """Compute elementwise complex conjugation of vector."""
#     return np.conjugate(v)

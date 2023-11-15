import numpy as np
import nlsa.abstract_algebra2 as alg
import nlsa.scalar_algebra2 as salg
from numba import guvectorize
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


def make_scl(dtype: Type[K]) -> alg.ImplementsScalarField[K]:
    """Make scalar field from dtype."""
    if issubclass(dtype, np.complexfloating):
        return salg.ComplexfloatingField(dtype)
    elif issubclass(dtype, np.floating):
        return salg.FloatingField(dtype)
    else:
        return salg.IntegerField(dtype)


def make_sunit(dtype: Type[K]) -> Callable[[], S[K]]:
    def unit() -> S[K]:
        return np.ones(1, dtype=dtype)
    return  unit


def make_unit(dim: int, dtype: Type[K]) -> Callable[[], V[K]]:
    """Make constant function returning vector of all 1s.

    The type variables N and K specify the dimension, and dtype, respectively.

    """
    def unit() -> V[K]:
        return np.ones((1, dim), dtype=dtype)
    return unit


def make_inv(dim: int, dtype: Type[K]) -> Callable[[V[K]], V[K]]:
    """Make inversion function for specified dimension and dtype."""
    def inv(v: V[K]) -> V[K]:
        return np.divide(np.ones(dim, dtype=dtype), v)
    return inv


def ldiv(u: V[K], v: V[K]) -> V[K]:
    """Left module division as elementwise vector division."""
    match (u.ndim, v.ndim):
        case (1, 1):
            w = np.divide(v, u)
        case _:
            w = np.divide(v, np.atleast_2d(u).T)
    return w


def counting_measure(v: V[K], /) -> S[K]:
    """Sum the elements of vector."""
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
    d2: S[R] = cdist(np.atleast_2d(u), np.atleast_2d(v), 'sqeuclidean')
    return d2


def sqeuclidean_core(u: V[R], v: V[R], s: S[R]):
    """Compute squared Euclidean distance."""
    nu = u.shape[-2]
    nv = v.shape[-2]
    d = u.shape[-1]
    for i in range(nu):
        for j in range(nv):
            s[i, j] = 0
            for k in range(d):
                s[i,j] += (u[i, k] - v[j, k]) ** 2


sqeuclidean: Callable[[X, X], Y] = guvectorize(['f8[:, :], f8[:, :], f8[:, :]'],
                                               '(n1, d), (n2, d) -> (n1, n2)',
                                               target='parallel',
                                               fastmath=True,
                                               nopython=True)(sqeuclidean_core)


class NPScalarField(Generic[K]):
    """Implement scalar field operations for NumPy arrays.

    The type parameter K parameterizes the field of scalars.

    """
    def __init__(self, dtype: Type[K]):
        self.unit: Callable[[], S[K]] = make_sunit(dtype)

    @staticmethod
    def add(u: S[K], v: S[K], /) -> S[K]:
        """Add two scalars."""
        return np.add(u, v)

    @staticmethod
    def sub(u: S[K], v: S[K], /) -> S[K]:
        """Subtract two scalars."""
        return np.subtract(u, v)

    @staticmethod
    def neg(v: S[K], /) -> S[K]:
        """Negate a scalar."""
        return np.multiply(-1, v)

    @staticmethod
    def mul(u: S[K], v: S[K], /) -> S[K]:
        """Multiply two scalars."""
        return np.multiply(u, v)

    def inv(self, v: S[K], /) -> S[K]:
        """Invert a scalar."""
        return np.divide(self.unit(), v)

    @staticmethod
    def div(u: S[K], v: S[K], /) -> S[K]:
        """Divide two scalars."""
        return np.divide(u, v)

    @staticmethod
    def lmul(u: S[K], v: S[K], /) -> S[K]:
        """Left-multiply two scalars. """
        # TODO: Check whether this is the right definiction for broadcasting/
        return np.multiply(u, v)

    @staticmethod
    def ldiv(u: S[K], v: S[K], /) -> S[K]:
        """Left-divide two scalars.

        Modified version for broadcasting.

        """
        return np.divide(v, np.atleast_2d(u).T)

    @staticmethod
    def sqrt(v: S[K], /) -> S[K]:
        """Compute square root of scalar."""
        return np.sqrt(v)

    @staticmethod
    def exp(v: S[K], /) -> S[K]:
        """Compute exponential function on scalar."""
        return np.exp(v)

    @staticmethod
    def power(v: S[K], k: S[K], /) -> S[K]:
        """Compute exponentiation of scalar by scalar."""
        return np.power(v, k)

    @staticmethod
    def star(v: S[K], /) -> S[K]:
        """Compute complex conjugation of scalar."""
        return np.conjugate(v)


class NPVectorAlgebra(Generic[N, K]):
    """Implement vector algebra operations for NumPy arrays.

    The type parameter N parameterizes the dimension of the algebra. The type
    parameter K parameterizes the field of scalars.

    """
    def __init__(self, dim: N, dtype: Type[K]):
        self.unit: Callable[[], V[K]] = make_unit(dim, dtype)
        self.scl: alg.ImplementsScalarField[S[K]] = NPScalarField(dtype)
        # self.scl: alg.ImplementsScalarField[K] = make_scl(dtype)

    @staticmethod
    def add(u: V[K], v: V[K], /) -> V[K]:
        """Add two vectors."""
        return np.add(u, v)

    @staticmethod
    def sub(u: V[K], v: V[K], /) -> V[K]:
        """Subtract two vectors."""
        return np.subtract(u, v)

    @staticmethod
    def neg(v: V[K], /) -> V[K]:
        """Negate a vector."""
        return np.multiply(-1, v)

    @staticmethod
    def smul(k: S[K], v: V[K], /) -> V[K]:
        """Multiply a scalar and a vector."""
        return np.multiply(k, v)

    @staticmethod
    def mul(u: V[K], v: V[K], /) -> V[K]:
        """Multiply two vectors elementwise."""
        return np.multiply(u, v)

    def inv(self, v: V[K], /) -> V[K]:
        """Invert a vector elementwise."""
        return np.divide(self.unit(), v)

    @staticmethod
    def div(u: V[K], v: V[K], /) -> V[K]:
        """Divide two vectors elementwise."""
        return np.divide(u, v)

    @staticmethod
    def lmul(u: V[K], v: V[K], /) -> V[K]:
        """Perform left module multiplication as elementwise vector
        multiplication.

        """
        return np.multiply(u, v)

    @staticmethod
    def ldiv(u: V[K], v: V[K], /) -> V[K]:
        """Perform left module division as elementwise vector division."""
        match (u.ndim, v.ndim):
            case (1, 1):
                w = np.divide(v, u)
            case _:
                w = np.divide(v, np.atleast_2d(u).T)
        return w

    @staticmethod
    def rmul(u: V[K], v: V[K], /) -> V[K]:
        """Perform right module multiplication as elementwise vector
        multiplication.

        """
        return np.multiply(u, v)

    @staticmethod
    def rdiv(u: V[K], v: V[K], /) -> V[K]:
        """Perform right module division as elementwise vector division."""
        return np.divide(u, v)

    @staticmethod
    def sqrt(v: V[K], /) -> V[K]:
        """Compute elementwise square root of vector."""
        return np.sqrt(v)

    @staticmethod
    def exp(v: V[K], /) -> V[K]:
        """Compute elementwise exponential function of vector."""
        return np.exp(v)

    @staticmethod
    def power(v: V[K], k: S[K], /) -> V[K]:
        """Compute elementwise exponentiation of vector by scalar."""
        return np.power(v, k)

    @staticmethod
    def star(v: V[K], /) -> V[K]:
        """Compute elementwise complex conjugation of vector."""
        return np.conjugate(v)

    @staticmethod
    def innerp(u: V[K], v: V[K], /) -> S[K]:
        """Compute inner product of vectors."""
        w = np.sum(np.multiply(np.conjugate(u), v), axis=-1, keepdims=True)
        # w = np.einsum('...i,...i->...', np.conjugate(u), v)
        return w


class NPMeasurableFnAlgebra(NPVectorAlgebra[N, K],
                            alg.ImplementsMeasurableFnAlgebra[X, V[K], S[K]],
                            Generic[X, N, K]):
    """Implement operations on equivalence classes of functions using NumPy
    arrays as the representation type.

    """
    def __init__(self, dim: N, dtype: Type[K],
                 inclusion_map: Callable[[F[X, S[K]]], V[K]]):
        super().__init__(dim, dtype)
        self.incl: Callable[[F[X, S[K]]], V[K]] = inclusion_map


class NPMeasureFnAlgebra(NPMeasurableFnAlgebra[X, N, K],
                         alg.ImplementsMeasureFnAlgebra[X, V[K], S[K]]):
    """Implement NPMeasurableFunctionAlgebra equipped with measure."""
    def __init__(self, dim: N, dtype: Type[K],
                 inclusion_map: Callable[[F[X, S[K]]], V[K]],
                 measure: Callable[[V[K]], S[K]]):
        super().__init__(dim, dtype, inclusion_map)
        self.integrate: Callable[[V[K]], S[K]] = measure

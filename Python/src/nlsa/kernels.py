import nlsa.abstract_algebra2 as alg
from functools import partial
from nlsa.abstract_algebra2 import FromScalarField
from nlsa.function_algebra2 import FunctionSpace, BivariateFunctionSpace
from typing import Callable, Literal, Optional, Protocol, TypeVar, TypeVarTuple

K_con = TypeVar('K_con', contravariant=True)
K_cov = TypeVar('K_cov', covariant=True)
K = TypeVar('K')
V = TypeVar('V')
Vs = TypeVar('Vs')
W = TypeVar('W')
X = TypeVar('X')
X_cov = TypeVar('X_cov', covariant=True)
X1 = TypeVar('X1')
X2 = TypeVar('X2')
Xs = TypeVarTuple('Xs')
Y = TypeVar('Y')
S = TypeVar('S')
Y_con = TypeVar('Y_con', contravariant=True)
F = Callable[[X], Y]


class ImplementsExponentialRBF(alg.ImplementsInv[S],
                               alg.ImplementsMul[S],
                               alg.ImplementsNeg[S],
                               alg.ImplementsExp[S],
                               Protocol[S]):
    pass


def make_exponential_rbf(impl: alg.ImplementsScalarField[S], bandwidth: S)\
        -> Callable[[S], S]:
    """Make exponential radial basis function."""

    neg_concentration = impl.neg(impl.inv(bandwidth))

    def rbf(s: S) -> S:
        return impl.exp(impl.mul(neg_concentration, s))
    return rbf


def make_covariance_kernel(impl: alg.ImplementsInnerp[V, K], /)\
        -> Callable[[V, V], K]:
    """Make covariance kernel on inner product space."""

    def k(u: V, v: V, /) -> K:
        return impl.innerp(u, v)
    return k


def make_integral_operator(impl: alg.ImplementsMeasureFnAlgebra[X, V, K],
                           k: Callable[[X, X], K], /) -> Callable[[V], F[X, K]]:
    """Make integral operator from kernel function."""

    def k_op(v: V, /) -> F[X, K]:
        def g(x: X, /) -> K:
            kx = partial(k, x)
            gx = impl.integrate(impl.mul(impl.incl(kx), v))
            return gx
        return g
    return k_op


def left_normalize(impl: alg.ImplementsMeasureUnitalFnAlgebra[X, V, K],
                   k: Callable[[X, X], K], /,
                   unit: Optional[V] = None) -> Callable[[X, X], K]:
    """Perform left normalization of kernel function."""

    func: BivariateFunctionSpace[X, X, K, K] =\
        BivariateFunctionSpace(codomain=FromScalarField(impl.scl))
    k_op: Callable[[V], F[X, K]] = make_integral_operator(impl, k)
    if unit is None:
        unit = impl.unit()
    lfun = k_op(unit)
    k_l = func.ldiv(lfun, k)
    return k_l


def right_normalize(impl: alg.ImplementsMeasureUnitalFnAlgebra[X, V, K],
                    k: Callable[[X, X], K], /,
                    unit: Optional[V] = None) -> Callable[[X, X], K]:
    """Perform right normalization of kernel function."""

    func: BivariateFunctionSpace[X, X, K, K] =\
        BivariateFunctionSpace(codomain=FromScalarField(impl.scl))
    k_op: Callable[[V], F[X, K]] = make_integral_operator(impl, k)
    if unit is None:
        unit = impl.unit()
    rfun = k_op(unit)
    k_r = func.rdiv(k, rfun)
    return k_r


def sym_normalize(impl: alg.ImplementsMeasureUnitalFnAlgebra[X, V, K],
                  k: Callable[[X, X], K], /,
                  unit: Optional[V] = None) -> Callable[[X, X], K]:
    """Perform symmetric normalization of kernel function."""

    func: BivariateFunctionSpace[X, X, K, K] =\
        BivariateFunctionSpace(codomain=FromScalarField(impl.scl))
    k_op: Callable[[V], F[X, K]] = make_integral_operator(impl, k)
    if unit is None:
        unit = impl.unit()
    sfun = k_op(unit)
    k_r = func.rdiv(k, sfun)
    k_s: Callable[[X, X], K] = func.ldiv(sfun, k_r)
    return k_s


class ImplementsSqrtNormalize(alg.ImplementsMeasureUnitalFnAlgebra[X_cov, V, K],
                              Protocol[X_cov, V, K]):
    """Implement operations required for kernel normalization by square roots
    of functions.
    """
    pass


def right_sqrt_normalize(impl: ImplementsSqrtNormalize[X, V, K],
                         k: Callable[[X, X], K], /,
                         unit: Optional[V] = None) -> Callable[[X, X], K]:
    """Perform right square root normalization of kernel function."""

    c = FromScalarField(impl.scl)
    func: FunctionSpace[X, K, K] = FunctionSpace(codomain=c)
    func2: BivariateFunctionSpace[X, X, K, K] =\
        BivariateFunctionSpace(codomain=c)
    k_op: Callable[[V], F[X, K]] = make_integral_operator(impl, k)
    if unit is None:
        unit = impl.unit()
    rfun: Callable[[X], K] = func.sqrt(k_op(unit))

    # pyright gives errors when passing rfun directly as a function.
    # k_r: Callable[[X, X], K] = func2.rdiv(k, rfun)
    k_r: Callable[[X, X], K] = func2.rdiv(k, lambda x: rfun(x))
    return k_r


def sym_sqrt_normalize(impl: ImplementsSqrtNormalize[X, V, K],
                       k: Callable[[X, X], K], /,
                       unit: Optional[V] = None) -> Callable[[X, X], K]:
    """Perform symmetric square root normalization of kernel function."""

    c = FromScalarField(impl.scl)
    func: FunctionSpace[X, K, K] = FunctionSpace(codomain=c)
    func2: BivariateFunctionSpace[X, X, K, K] =\
        BivariateFunctionSpace(codomain=c)
    k_op: Callable[[V], F[X, K]] = make_integral_operator(impl, k)
    if unit is None:
        unit = impl.unit()
    sfun: Callable[[X], K] = func.sqrt(k_op(unit))

    # pyright gives errors when passing sfun directly as a function
    # k_s = func2.ldiv(sfun, func2.rdiv(k, sfun))

    k_r: Callable[[X, X], K] = func2.rdiv(k, lambda x: sfun(x))
    k_s: Callable[[X, X], K] = func2.ldiv(lambda x: sfun(x), k_r)
    return k_s


def dm_normalize(impl: ImplementsSqrtNormalize[X, V, K],
                 alpha: Literal['0', '0.5', '1'],
                 k: Callable[[X, X], K], /,
                 unit: Optional[V] = None) -> Callable[[X, X], K]:
    """Perform Diffusion Maps kernel normalization."""

    match alpha:
        case '0':
            k_r: Callable[[X, X], K] = k
        case '0.5':
            k_r: Callable[[X, X], K] = sym_sqrt_normalize(impl, k, unit)
        case '1':
            k_r: Callable[[X, X], K] = sym_normalize(impl, k, unit)

    k_dm: Callable[[X, X], K] = left_normalize(impl, k_r, unit)
    return k_dm


def dmsym_normalize(impl: ImplementsSqrtNormalize[X, V, K],
                    alpha: Literal['0', '0.5', '1'],
                    k: Callable[[X, X], K], /,
                    unit: Optional[V] = None) -> Callable[[X, X], K]:
    """Perform Diffusion Maps symmetric kernel normalization."""

    match alpha:
        case '0':
            k_r: Callable[[X, X], K] = k
        case '0.5':
            k_r: Callable[[X, X], K] = sym_sqrt_normalize(impl, k, unit)
        case '1':
            k_r: Callable[[X, X], K] = sym_normalize(impl, k, unit)

    k_dm: Callable[[X, X], K] = sym_sqrt_normalize(impl, k_r, unit)
    return k_dm


def from_dmsym(impl: alg.ImplementsAlgebraLModule[Vs, K, V], v0: V, vs: Vs)\
        -> Vs:
    """Normalize eigenvectors from symmetric diffusion maps normalization to
    obtain eigenvectors with respect to Markov normalization.
    """

    return impl.ldiv(v0, vs)

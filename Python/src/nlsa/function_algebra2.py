import nlsa.abstract_algebra2 as alg
from itertools import repeat
from functools import reduce
from typing import Callable, Generic, TypeGuard, TypeVar, TypeVarTuple

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
K = TypeVar('K')
L = TypeVar('L')
R = TypeVar('R')
X = TypeVar('X')
X1 = TypeVar('X1')
X2 = TypeVar('X2')
Xs = TypeVarTuple('Xs')
Y = TypeVar('Y')
Y1 = TypeVar('Y1')
Y2 = TypeVar('Y2')
Z = TypeVar('Z')
F = Callable[[X], Y]
F2 = Callable[[X1, X2], Y]
FS = Callable[[*Xs], Y]
V = TypeVar('V')


def identity(x: X, /) -> X:
    return x


def apply(f: F[X, Y], x: X, /) -> Y:
    return f(x)


def compose(f: F[Y, Z], g: FS[*Xs, Y], /) -> FS[*Xs, Z]:
    """Compose two functions."""
    def h(*xs: *Xs) -> Z:
        z = f(g(*xs))
        return z
    return h


def compose2(f: FS[Y1, Y2, Z], gs: tuple[F[X1, Y1], F[X2, Y2]], /) \
        -> FS[X1, X2, Z]:
    """Compose a bivariate function with a tuple of univariate functions."""
    def h(x1: X1, x2: X2) -> Z:
        return f(gs[0](x1), gs[1](x2))
    return h


def eval_at(*xs: *Xs) -> Callable[[FS[*Xs, Y]], Y]:
    """Build pointwise evaluation functional."""
    def evalx(f: FS[*Xs, Y]) -> Y:
        return f(*xs)
    return evalx


def pair(x: X, y: Y, /) -> tuple[X, Y]:
    """Pack two objects into tuple."""
    z: tuple[X, Y] = (x, y)
    return z


def diag(f: F2[X, X, Y], /) -> F[X, Y]:
    """Make univariate function from bivariate function on diagonal."""
    def g(x: X) -> Y:
        return f(x, x)
    return g


def make_bivariate_tensor_product(impl: alg.ImplementsMul[A], /) \
        -> Callable[[F[X, A], F[Y, A]], F2[X, Y, A]]:
    """Make tensor product of functions as a bivariate function."""
    def tensorp(f: F[X, A], g: F[Y, A]) -> F2[X, Y, A]:
        def h(x: X, y: Y) -> A:
            return impl.mul(f(x), g(y))
        return h
    return tensorp


def make_mpower(f: Callable[[A, A], A], /) -> Callable[[A, int], A]:
    """Make monoidal power from binary operation."""
    def mpower(a: A, n: int) -> A:
        return reduce(f, repeat(a, n))
    return mpower


def make_constant(a: A) -> Callable[[], A]:
    """Make constant function."""
    def f() -> A:
        return a
    return f


class Lift(Generic[*Xs]):
    """Lift function to function on domain parameterized by Xs."""
    @staticmethod
    def constant(g: Callable[[], A], /) -> Callable[[], FS[*Xs, A]]:
        """Lift of constant function."""
        def lg() -> FS[*Xs, A]:
            def fxa(*_: *Xs) -> A:
                return g()
            return fxa
        return lg

    @staticmethod
    def unary(g: Callable[[A], B], /) -> Callable[[FS[*Xs, A]], FS[*Xs, B]]:
        """Lift of unary function."""
        def lg(fxa: FS[*Xs, A], /) -> FS[*Xs, B]:
            def fxb(*xs: *Xs) -> B:
                return g(fxa(*xs))
            return fxb
        return lg

    @staticmethod
    def binary(g: Callable[[A, B], C], /)\
            -> Callable[[FS[*Xs, A], FS[*Xs, B]], FS[*Xs, C]]:
        """Lift of binary function."""
        def lg(fxa: FS[*Xs, A], fxb: FS[*Xs, B], /) -> FS[*Xs, C]:
            def fxc(*xs: *Xs) -> C:
                return g(fxa(*xs), fxb(*xs))
            return fxc
        return lg

    @staticmethod
    def left(g: Callable[[K, A], B], /)\
            -> Callable[[K, FS[*Xs, A]], FS[*Xs, B]]:
        """Lift of scalar/left module operation."""
        def lg(k: K, fxa: FS[*Xs, A], /) -> FS[*Xs, B]:
            def fxb(*xs: *Xs) -> B:
                return g(k, fxa(*xs))
            return fxb
        return lg

    @staticmethod
    def right(g: Callable[[A, K], B], /)\
            -> Callable[[FS[*Xs, A], K], FS[*Xs, B]]:
        """Lift of right module operations."""
        def lg(fxa: FS[*Xs, A], k: K, /) -> FS[*Xs, B]:
            def fxb(*xs: *Xs) -> B:
                return g(fxa(*xs), k)
            return fxb
        return lg


class LiftBivariate(Generic[X1, X2]):
    """Lift bivariate function to left and right module operations on bivariate
    functions.

    """
    @staticmethod
    def left(g: Callable[[A, B], C], /) -> Callable[[F[X1, A], F2[X1, X2, B]],
                                                    F2[X1, X2, C]]:
        """Lift bivariate function to left module operation."""
        def lg(fx1a: F[X1, A], fx12b: F2[X1, X2, B], /) -> F2[X1, X2, C]:
            def fx12c(x1: X1, x2: X2, /) -> C:
                return g(fx1a(x1), fx12b(x1, x2))
            return fx12c
        return lg

    @staticmethod
    def right(g: Callable[[A, B], C], /) -> Callable[[F2[X1, X2, A], F[X2, B]],
                                                     F2[X1, X2, C]]:
        """Lift bivariate function to right module operation."""
        def lg(fx12a: F2[X1, X2, A], fx2b: F[X2, B], /) -> F2[X1, X2, C]:
            def fx12c(x1: X1, x2: X2, /) -> C:
                return g(fx12a(x1, x2), fx2b(x2))
            return fx12c
        return lg


Codomain = alg.ImplementsVectorSpace[Y, K]\
        | alg.ImplementsAlgebra[Y, K]\
        | alg.ImplementsRootAlgebra[Y, K]\
        | alg.ImplementsStarAlgebra[Y, K]\
        | alg.ImplementsUnitalAlgebra[Y, K]


def implements_algebra(codomain: Codomain[Y, K])\
        -> TypeGuard[alg.ImplementsAlgebra[Y, K]]:
    return isinstance(codomain, alg.ImplementsAlgebra)


def implements_root_algebra(codomain: Codomain[Y, K])\
        -> TypeGuard[alg.ImplementsRootAlgebra[Y, K]]:
    return isinstance(codomain, alg.ImplementsRootAlgebra)


def implements_power_algebra(codomain: Codomain[Y, K])\
        -> TypeGuard[alg.ImplementsPowerAlgebra[Y, K]]:
    return isinstance(codomain, alg.ImplementsPowerAlgebra)


def implements_star_algebra(codomain: Codomain[Y, K])\
        -> TypeGuard[alg.ImplementsStarAlgebra[Y, K]]:
    return isinstance(codomain, alg.ImplementsStarAlgebra)


def implements_unital_algebra(codomain: Codomain[Y, K])\
        -> TypeGuard[alg.ImplementsUnitalAlgebra[Y, K]]:
    return isinstance(codomain, alg.ImplementsUnitalAlgebra)


class FunctionSpace(Generic[*Xs, Y, K]):
    """Implement function space structure."""
    def __init__(self, codomain: Codomain[Y, K]):
        self.scl: alg.ImplementsScalarField[K] = codomain.scl
        self.zero: Callable[[], FS[*Xs, Y]] = Lift.constant(codomain.zero)
        self.add: Callable[[FS[*Xs, Y], FS[*Xs, Y]], FS[*Xs, Y]]\
            = Lift.binary(codomain.add)
        self.sub: Callable[[FS[*Xs, Y], FS[*Xs, Y]], FS[*Xs, Y]]\
            = Lift.binary(codomain.sub)
        self.neg: Callable[[FS[*Xs, Y]], FS[*Xs, Y]]\
            = Lift.unary(codomain.neg)
        self.smul: Callable[[K, FS[*Xs, Y]], FS[*Xs, Y]]\
            = Lift.left(codomain.smul)

        if implements_algebra(codomain):
            self.mul: Callable[[FS[*Xs, Y], FS[*Xs, Y]], FS[*Xs, Y]]\
                = Lift.binary(codomain.mul)

        if implements_root_algebra(codomain):
            self.sqrt: Callable[[FS[*Xs, Y]], FS[*Xs, Y]]\
                = Lift.unary(codomain.sqrt)

        if implements_power_algebra(codomain):
            self.power: Callable[[FS[*Xs, Y], K], FS[*Xs, Y]]\
                = Lift.right(codomain.power)

        if implements_star_algebra(codomain):
            self.star: Callable[[FS[*Xs, Y]], FS[*Xs, Y]]\
                = Lift.unary(codomain.star)

        if implements_unital_algebra(codomain):
            self.unit: Callable[[], FS[*Xs, Y]] = Lift.constant(codomain.unit)
            self.inv: Callable[[FS[*Xs, Y]], FS[*Xs, Y]]\
                = Lift.unary(codomain.inv)
            self.div: Callable[[FS[*Xs, Y], FS[*Xs, Y]], FS[*Xs, Y]]\
                = Lift.binary(codomain.div)


CodomainL = alg.ImplementsLModule[Y, K, L]\
        | alg.ImplementsAlgebraLModule[Y, K, L]\
        | alg.ImplementsRootAlgebraLModule[Y, K, L]\
        | alg.ImplementsStarAlgebraLModule[Y, K, L]\
        | alg.ImplementsUnitalAlgebraLModule[Y, K, L]


class FunctionLModule(FunctionSpace[*Xs, Y, K], Generic[*Xs, Y, K, L]):
    """Implement function algebra and left module structure."""
    def __init__(self, codomain: CodomainL[Y, K, L]):
        super().__init__(codomain)
        self.lmul: Callable[[L, FS[*Xs, Y]], FS[*Xs, Y]]\
            = Lift.left(codomain.lmul)
        self.ldiv: Callable[[L, FS[*Xs, Y]], FS[*Xs, Y]]\
            = Lift.left(codomain.ldiv)


CodomainR = alg.ImplementsRModule[Y, K, R]\
        | alg.ImplementsAlgebraRModule[Y, K, R]\
        | alg.ImplementsRootAlgebraRModule[Y, K, R]\
        | alg.ImplementsStarAlgebraRModule[Y, K, R]\
        | alg.ImplementsUnitalAlgebraRModule[Y, K, R]


class FunctionRModule(FunctionSpace[*Xs, Y, K], Generic[*Xs, Y, K, R]):
    """Implement function space and right module structure."""
    def __init__(self, codomain: CodomainR[Y, K, R]):
        super().__init__(codomain)
        self.rmul: Callable[[FS[*Xs, Y], R], FS[*Xs, Y]]\
            = Lift.right(codomain.rmul)
        self.rdiv: Callable[[FS[*Xs, Y], R], FS[*Xs, Y]]\
            = Lift.right(codomain.rdiv)


CodomainLR = alg.ImplementsLRModule[Y, K, L, R]\
        | alg.ImplementsAlgebraLRModule[Y, K, L, R]\
        | alg.ImplementsRootAlgebraLRModule[Y, K, L, R]\
        | alg.ImplementsStarAlgebraLRModule[Y, K, L, R]\
        | alg.ImplementsUnitalAlgebraLRModule[Y, K, L, R]


class FunctionLRModule(FunctionLModule[*Xs, Y, K, L],
                       FunctionRModule[*Xs, Y, K, R],
                       Generic[*Xs, Y, K, L, R]):
    """Implement function space and left-right module structure."""
    def __init__(self, codomain: CodomainLR[Y, K, L, R]):
        super().__init__(codomain)


CodomainB = alg.ImplementsAlgebraLRModule[Y, K, L, R]\
        | alg.ImplementsUnitalAlgebraLRModule[Y, K, L, R]


def implements_unital_algebra_lrmodule(codomain: CodomainB[Y, K, L, R])\
        -> TypeGuard[alg.ImplementsUnitalAlgebraLRModule[Y, K, L, R]]:
    return isinstance(codomain, alg.ImplementsUnitalAlgebraLRModule)


class BivariateFunctionSpace(FunctionSpace[X1, X2, Y, K],
                             Generic[X1, X2, Y, K]):
    """Implement bivariate function structure mapping into an algebra."""
    def __init__(self, codomain: CodomainB[Y, K, Y, Y]):
        super().__init__(codomain)
        self.lmul: Callable[[F[X1, Y], F2[X1, X2, Y]], F2[X1, X2, Y]]\
            = LiftBivariate.left(codomain.lmul)
        self.rmul: Callable[[F2[X1, X2, Y], F[X2, Y]], F2[X1, X2, Y]]\
            = LiftBivariate.right(codomain.rmul)

        if implements_unital_algebra_lrmodule(codomain):
            self.ldiv: Callable[[F[X1, Y], F2[X1, X2, Y]], F2[X1, X2, Y]]\
                = LiftBivariate.left(codomain.ldiv)
            self.rdiv: Callable[[F2[X1, X2, Y], F[X2, Y]], F2[X1, X2, Y]]\
                = LiftBivariate.right(codomain.rdiv)

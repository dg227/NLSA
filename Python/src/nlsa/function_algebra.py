"""Provide classes and functions implementing operations on function spaces.
"""

import nlsa.abstract_algebra as alg
from collections.abc import Callable
from itertools import repeat
from functools import reduce

type F[*Xs, Y] = Callable[[*Xs], Y]


def identity[X](x: X, /) -> X:
    """Evaluate identity map."""
    return x


def apply[X, Y](f: F[X, Y], x: X, /) -> Y:
    """Apply function to argument."""
    return f(x)


def compose[*Xs, Y, Z](f: F[Y, Z], g: F[*Xs, Y], /) -> F[*Xs, Z]:
    """Compose two functions."""
    def h(*xs: *Xs) -> Z:
        z = f(g(*xs))
        return z
    return h


def compose2[X1, X2, Y1, Y2, Z](
        f: F[Y1, Y2, Z], gs: tuple[F[X1, Y1], F[X2, Y2]], /) -> F[X1, X2, Z]:
    """Compose a bivariate function with a tuple of univariate functions."""
    def h(x1: X1, x2: X2) -> Z:
        return f(gs[0](x1), gs[1](x2))
    return h


def eval_at[*Xs, Y](*xs: *Xs) -> Callable[[F[*Xs, Y]], Y]:
    """Make pointwise evaluation functional."""
    def evalx(f: F[*Xs, Y]) -> Y:
        return f(*xs)
    return evalx


def diag[X, Y](f: F[X, X, Y], /) -> F[X, Y]:
    """Make univariate function from bivariate function on diagonal."""
    def g(x: X) -> Y:
        return f(x, x)
    return g


def uncurry[X, Y, Z](f: Callable[[X], F[Y, Z]], /) -> F[X, Y, Z]:
    """Uncurry high-order function."""
    def g(x: X, y: Y) -> Z:
        return f(x)(y)
    return g


def uncurry2[X1, X2, Y, Z](f: Callable[[X1, X2], F[Y, Z]], /) \
        -> F[X1, X2, Y, Z]:
    """Uncurry high-order function."""
    def g(x1: X1, x2: X2, y: Y) -> Z:
        return f(x1, x2)(y)
    return g


def make_bivariate_tensor_product[X, Y, A](impl: alg.ImplementsMul[A], /) \
        -> Callable[[F[X, A], F[Y, A]], F[X, Y, A]]:
    """Make tensor product of functions as a bivariate function."""
    def tensorp(f: F[X, A], g: F[Y, A]) -> F[X, Y, A]:
        def h(x: X, y: Y) -> A:
            return impl.mul(f(x), g(y))
        return h
    return tensorp


def make_mpower[A](f: Callable[[A, A], A], /) -> Callable[[A, int], A]:
    """Make monoidal power from binary operation."""
    def mpower(a: A, n: int) -> A:
        return reduce(f, repeat(a, n))
    return mpower


def make_constant[A](a: A, /) -> Callable[[], A]:
    """Make constant function."""
    def f() -> A:
        return a
    return f


def lift_constant[*Xs, A](g: Callable[[], A], /) -> Callable[[], F[*Xs, A]]:
    """Lift constant function."""
    def lg() -> F[*Xs, A]:
        def fxa(*_: *Xs) -> A:
            return g()
        return fxa
    return lg


def lift_unary[*Xs, A, B](g: Callable[[A], B], /) -> Callable[[F[*Xs, A]],
                                                              F[*Xs, B]]:
    """Lift unary function."""
    def lg(fxa: F[*Xs, A], /) -> F[*Xs, B]:
        def fxb(*xs: *Xs) -> B:
            return g(fxa(*xs))
        return fxb
    return lg


def lift_binary[*Xs, A, B, C](g: Callable[[A, B], C], /)\
        -> Callable[[F[*Xs, A], F[*Xs, B]], F[*Xs, C]]:
    """Lift binary function."""
    def lg(fxa: F[*Xs, A], fxb: F[*Xs, B], /) -> F[*Xs, C]:
        def fxc(*xs: *Xs) -> C:
            return g(fxa(*xs), fxb(*xs))
        return fxc
    return lg


def lift_left[*Xs, K, A, B](g: Callable[[K, A], B], /)\
        -> Callable[[K, F[*Xs, A]], F[*Xs, B]]:
    """Lift scalar/left module operation."""
    def lg(k: K, fxa: F[*Xs, A], /) -> F[*Xs, B]:
        def fxb(*xs: *Xs) -> B:
            return g(k, fxa(*xs))
        return fxb
    return lg


def lift_right[*Xs, K, A, B](g: Callable[[A, K], B], /)\
        -> Callable[[F[*Xs, A], K], F[*Xs, B]]:
    """Lift right module operations."""
    def lg(fxa: F[*Xs, A], k: K, /) -> F[*Xs, B]:
        def fxb(*xs: *Xs) -> B:
            return g(fxa(*xs), k)
        return fxb
    return lg


def lift_left_bivariate[X1, X2, A, B, C](g: Callable[[A, B], C], /) \
        -> Callable[[F[X1, A], F[X1, X2, B]], F[X1, X2, C]]:
    """Lift bivariate function to left module operation."""
    def lg(fx1a: F[X1, A], fx12b: F[X1, X2, B], /) -> F[X1, X2, C]:
        def fx12c(x1: X1, x2: X2, /) -> C:
            return g(fx1a(x1), fx12b(x1, x2))
        return fx12c
    return lg


def lift_right_bivariate[X1, X2, A, B, C](g: Callable[[A, B], C], /) \
        -> Callable[[F[X1, X2, A], F[X2, B]], F[X1, X2, C]]:
    """Lift bivariate function to right module operation."""
    def lg(fx12a: F[X1, X2, A], fx2b: F[X2, B], /) -> F[X1, X2, C]:
        def fx12c(x1: X1, x2: X2, /) -> C:
            return g(fx12a(x1, x2), fx2b(x2))
        return fx12c
    return lg


class FunctionSpace[*Xs, Y, K](alg.ImplementsVectorSpace[F[*Xs, Y], K]):
    """Implement function space operations."""
    def __init__(self, codomain: alg.ImplementsVectorSpace[Y, K]):
        self.scl: alg.ImplementsScalarField[K] = codomain.scl
        self.zero: Callable[[], F[*Xs, Y]] = lift_constant(codomain.zero)
        self.add: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]\
            = lift_binary(codomain.add)
        self.sub: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]\
            = lift_binary(codomain.sub)
        self.neg: Callable[[F[*Xs, Y]], F[*Xs, Y]] = lift_unary(codomain.neg)
        self.smul: Callable[[K, F[*Xs, Y]], F[*Xs, Y]]\
            = lift_left(codomain.smul)
        self.sdiv: Callable[[K, F[*Xs, Y]], F[*Xs, Y]]\
            = lift_left(codomain.sdiv)


class FunctionAlgebra[*Xs, Y, K](alg.ImplementsAlgebra[F[*Xs, Y], K]):
    """Implement function algebra operations."""
    def __init__(self, codomain: alg.ImplementsAlgebra[Y, K]):
        self.scl: alg.ImplementsScalarField[K] = codomain.scl
        self.zero: Callable[[], F[*Xs, Y]] = lift_constant(codomain.zero)
        self.add: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]] \
            = lift_binary(codomain.add)
        self.sub: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]] \
            = lift_binary(codomain.sub)
        self.neg: Callable[[F[*Xs, Y]], F[*Xs, Y]] = lift_unary(codomain.neg)
        self.smul: Callable[[K, F[*Xs, Y]], F[*Xs, Y]] \
            = lift_left(codomain.smul)
        self.sdiv: Callable[[K, F[*Xs, Y]], F[*Xs, Y]] \
            = lift_left(codomain.sdiv)
        self.unit: Callable[[], F[*Xs, Y]] = lift_constant(codomain.unit)
        self.mul: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]] \
            = lift_binary(codomain.mul)
        self.div: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]] \
            = lift_binary(codomain.div)
        self.inv: Callable[[F[*Xs, Y]], F[*Xs, Y]] = lift_unary(codomain.inv)
        self.sqrt: Callable[[F[*Xs, Y]], F[*Xs, Y]] = lift_unary(codomain.sqrt)
        self.adj: Callable[[F[*Xs, Y]], F[*Xs, Y]] = lift_unary(codomain.adj)
        self.mod: Callable[[F[*Xs, Y]], F[*Xs, Y]] = lift_unary(codomain.mod)
        self.power: Callable[[F[*Xs, Y], K], F[*Xs, Y]] \
            = lift_right(codomain.power)


class BivariateFunctionModule[X1, X2, Y, K](
        alg.ImplementsBimodule[F[X1, X2, Y], K, F[X1, Y], F[X2, Y]]):
    """Implement bivariate function space as a univariate function bimodule.
    """
    def __init__(self, codomain: alg.ImplementsBimodule[Y, K, Y, Y]):
        self.scl: alg.ImplementsScalarField[K] = codomain.scl
        self.zero: Callable[[], F[X1, X2, Y]] = lift_constant(codomain.zero)
        self.add: Callable[[F[X1, X2, Y], F[X1, X2, Y]], F[X1, X2, Y]] \
            = lift_binary(codomain.add)
        self.sub: Callable[[F[X1, X2, Y], F[X1, X2, Y]], F[X1, X2, Y]] \
            = lift_binary(codomain.sub)
        self.neg: Callable[[F[X1, X2, Y]], F[X1, X2, Y]] \
            = lift_unary(codomain.neg)
        self.smul: Callable[[K, F[X1, X2, Y]], F[X1, X2, Y]] \
            = lift_left(codomain.smul)
        self.sdiv: Callable[[K, F[X1, X2, Y]], F[X1, X2, Y]] \
            = lift_left(codomain.sdiv)
        self.lmul: Callable[[F[X1, Y], F[X1, X2, Y]], F[X1, X2, Y]] \
            = lift_left_bivariate(codomain.lmul)
        self.ldiv: Callable[[F[X1, Y], F[X1, X2, Y]], F[X1, X2, Y]] \
            = lift_left_bivariate(codomain.ldiv)
        self.rmul: Callable[[F[X1, X2, Y], F[X2, Y]], F[X1, X2, Y]] \
            = lift_right_bivariate(codomain.rmul)
        self.rdiv: Callable[[F[X1, X2, Y], F[X2, Y]], F[X1, X2, Y]] \
            = lift_right_bivariate(codomain.rdiv)


# class FunctionSpace(Generic[*Xs, Y, K]):
#     """Implement function space structure."""
#     def __init__(self, codomain: Codomain[Y, K]):
#         self.scl: alg.ImplementsScalarField[K] = codomain.scl
#         self.zero: Callable[[], FS[*Xs, Y]] = Lift.constant(codomain.zero)
#         self.add: Callable[[FS[*Xs, Y], FS[*Xs, Y]], FS[*Xs, Y]]\
#             = Lift.binary(codomain.add)
#         self.sub: Callable[[FS[*Xs, Y], FS[*Xs, Y]], FS[*Xs, Y]]\
#             = Lift.binary(codomain.sub)
#         self.neg: Callable[[FS[*Xs, Y]], FS[*Xs, Y]]\
#             = Lift.unary(codomain.neg)
#         self.smul: Callable[[K, FS[*Xs, Y]], FS[*Xs, Y]]\
#             = Lift.left(codomain.smul)

#         if implements_algebra(codomain):
#             self.mul: Callable[[FS[*Xs, Y], FS[*Xs, Y]], FS[*Xs, Y]]\
#                 = Lift.binary(codomain.mul)

#         if implements_root_algebra(codomain):
#             self.sqrt: Callable[[FS[*Xs, Y]], FS[*Xs, Y]]\
#                 = Lift.unary(codomain.sqrt)

#         if implements_power_algebra(codomain):
#             self.power: Callable[[FS[*Xs, Y], K], FS[*Xs, Y]]\
#                 = Lift.right(codomain.power)

#         if implements_star_algebra(codomain):
#             self.adj: Callable[[FS[*Xs, Y]], FS[*Xs, Y]]\
#                 = Lift.unary(codomain.adj)

#         if implements_unital_algebra(codomain):
#             self.unit: Callable[[], FS[*Xs, Y]] = Lift.constant(codomain.unit)
#             self.inv: Callable[[FS[*Xs, Y]], FS[*Xs, Y]]\
#                 = Lift.unary(codomain.inv)
#             self.div: Callable[[FS[*Xs, Y], FS[*Xs, Y]], FS[*Xs, Y]]\
#                 = Lift.binary(codomain.div)


# CodomainL = alg.ImplementsLModule[Y, K, L]\
#         | alg.ImplementsLModule[Y, K, L]\
#         | alg.ImplementsLModule[Y, K, L]\
#         | alg.ImplementsLModule[Y, K, L]\
#         | alg.ImplementsLModule[Y, K, L]


# class FunctionLModule(FunctionSpace[*Xs, Y, K], Generic[*Xs, Y, K, L]):
#     """Implement function algebra and left module structure."""
#     def __init__(self, codomain: CodomainL[Y, K, L]):
#         super().__init__(codomain)
#         self.lmul: Callable[[L, FS[*Xs, Y]], FS[*Xs, Y]]\
#             = Lift.left(codomain.lmul)
#         self.ldiv: Callable[[L, FS[*Xs, Y]], FS[*Xs, Y]]\
#             = Lift.left(codomain.ldiv)


# CodomainR = alg.ImplementsRModule[Y, K, R]\
#         | alg.ImplementsRModule[Y, K, R]\
#         | alg.ImplementsRModule[Y, K, R]\
#         | alg.ImplementsRModule[Y, K, R]\
#         | alg.ImplementsRModule[Y, K, R]


# class FunctionRModule(FunctionSpace[*Xs, Y, K], Generic[*Xs, Y, K, R]):
#     """Implement function space and right module structure."""
#     def __init__(self, codomain: CodomainR[Y, K, R]):
#         super().__init__(codomain)
#         self.rmul: Callable[[FS[*Xs, Y], R], FS[*Xs, Y]]\
#             = Lift.right(codomain.rmul)
#         self.rdiv: Callable[[FS[*Xs, Y], R], FS[*Xs, Y]]\
#             = Lift.right(codomain.rdiv)


# CodomainLR = alg.ImplementsBimodule[Y, K, L, R]\
#         | alg.ImplementsBimodule[Y, K, L, R]\
#         | alg.ImplementsBimodule[Y, K, L, R]\
#         | alg.ImplementsBimodule[Y, K, L, R]\
#         | alg.ImplementsBimodule[Y, K, L, R]


# class FunctionLRModule(FunctionLModule[*Xs, Y, K, L],
#                        FunctionRModule[*Xs, Y, K, R],
#                        Generic[*Xs, Y, K, L, R]):
#     """Implement function space and left-right module structure."""
#     def __init__(self, codomain: CodomainLR[Y, K, L, R]):
#         super().__init__(codomain)


# CodomainB = alg.ImplementsBimodule[Y, K, L, R]\
#         | alg.ImplementsBimodule[Y, K, L, R]


# def implements_unital_algebra_lrmodule(codomain: CodomainB[Y, K, L, R])\
#         -> TypeGuard[alg.ImplementsBimodule[Y, K, L, R]]:
#     return isinstance(codomain, alg.ImplementsBimodule)


# class BivariateFunctionSpace(FunctionSpace[X1, X2, Y, K],
#                              Generic[X1, X2, Y, K]):
#     """Implement bivariate function structure mapping into an algebra."""
#     def __init__(self, codomain: CodomainB[Y, K, Y, Y]):
#         super().__init__(codomain)
#         self.lmul: Callable[[F[X1, Y], F2[X1, X2, Y]], F2[X1, X2, Y]]\
#             = LiftBivariate.left(codomain.lmul)
#         self.rmul: Callable[[F2[X1, X2, Y], F[X2, Y]], F2[X1, X2, Y]]\
#             = LiftBivariate.right(codomain.rmul)

#         if implements_unital_algebra_lrmodule(codomain):
#             self.ldiv: Callable[[F[X1, Y], F2[X1, X2, Y]], F2[X1, X2, Y]]\
#                 = LiftBivariate.left(codomain.ldiv)
#             self.rdiv: Callable[[F2[X1, X2, Y], F[X2, Y]], F2[X1, X2, Y]]\
#                 = LiftBivariate.right(codomain.rdiv)

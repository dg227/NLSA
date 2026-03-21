"""Provide classes and functions implementing function space operations."""

import nlsa.abstract_algebra as alg
from collections.abc import Callable
from dataclasses import dataclass
from itertools import repeat
from functools import reduce
from typing import Optional, final

type F[*Xs, Y] = Callable[[*Xs], Y]


def identity[X](x: X) -> X:
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
    f: F[Y1, Y2, Z], gs: tuple[F[X1, Y1], F[X2, Y2]], /
) -> F[X1, X2, Z]:
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


def uncurry2[X1, X2, Y, Z](
    f: Callable[[X1, X2], F[Y, Z]], /
) -> F[X1, X2, Y, Z]:
    """Uncurry high-order function."""

    def g(x1: X1, x2: X2, y: Y) -> Z:
        return f(x1, x2)(y)

    return g


def make_bivariate_tensor_product[X, Y, A](
    impl: alg.ImplementsMul[A], /
) -> Callable[[F[X, A], F[Y, A]], F[X, Y, A]]:
    """Make tensor product of functions as a bivariate function."""

    def tensorp(f: F[X, A], g: F[Y, A]) -> F[X, Y, A]:
        def h(x: X, y: Y) -> A:
            return impl.mul(f(x), g(y))

        return h

    return tensorp


def mpower[A](f: F[A, A], n: int, /) -> F[A, A]:
    """Form monoidal power of endomorphism."""
    if n == 0:
        fn = identity
    else:
        fn = reduce(compose, repeat(f, n))
    return fn


def make_mpower[A](f: Callable[[A, A], A], /) -> Callable[[A, int], A]:
    """Make monoidal power flom binary operation."""

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


def lift_unary[*Xs, A, B](
    g: Callable[[A], B], /
) -> Callable[[F[*Xs, A]], F[*Xs, B]]:
    """Lift unary function."""

    def lg(fxa: F[*Xs, A], /) -> F[*Xs, B]:
        def fxb(*xs: *Xs) -> B:
            return g(fxa(*xs))

        return fxb

    return lg


def lift_binary[*Xs, A, B, C](
    g: Callable[[A, B], C], /
) -> Callable[[F[*Xs, A], F[*Xs, B]], F[*Xs, C]]:
    """Lift binary function."""

    def lg(fxa: F[*Xs, A], fxb: F[*Xs, B], /) -> F[*Xs, C]:
        def fxc(*xs: *Xs) -> C:
            return g(fxa(*xs), fxb(*xs))

        return fxc

    return lg


def lift_left[*Xs, K, A, B](
    g: Callable[[K, A], B], /
) -> Callable[[K, F[*Xs, A]], F[*Xs, B]]:
    """Lift scalar/left module operation."""

    def lg(k: K, fxa: F[*Xs, A], /) -> F[*Xs, B]:
        def fxb(*xs: *Xs) -> B:
            return g(k, fxa(*xs))

        return fxb

    return lg


def lift_right[*Xs, K, A, B](
    g: Callable[[A, K], B], /
) -> Callable[[F[*Xs, A], K], F[*Xs, B]]:
    """Lift right module operations."""

    def lg(fxa: F[*Xs, A], k: K, /) -> F[*Xs, B]:
        def fxb(*xs: *Xs) -> B:
            return g(fxa(*xs), k)

        return fxb

    return lg


def lift_left_bivariate[X1, X2, A, B, C](
    g: Callable[[A, B], C], /
) -> Callable[[F[X1, A], F[X1, X2, B]], F[X1, X2, C]]:
    """Lift bivariate function to left module operation."""

    def lg(fx1a: F[X1, A], fx12b: F[X1, X2, B], /) -> F[X1, X2, C]:
        def fx12c(x1: X1, x2: X2, /) -> C:
            return g(fx1a(x1), fx12b(x1, x2))

        return fx12c

    return lg


def lift_right_bivariate[X1, X2, A, B, C](
    g: Callable[[A, B], C], /
) -> Callable[[F[X1, X2, A], F[X2, B]], F[X1, X2, C]]:
    """Lift bivariate function to right module operation."""

    def lg(fx12a: F[X1, X2, A], fx2b: F[X2, B], /) -> F[X1, X2, C]:
        def fx12c(x1: X1, x2: X2, /) -> C:
            return g(fx12a(x1, x2), fx2b(x2))

        return fx12c

    return lg


@final
@dataclass(frozen=True)
class FunctionSpace[*Xs, Y, K](alg.ImplementsVectorSpace[F[*Xs, Y], K]):
    """Implement function space."""

    codomain: alg.ImplementsVectorSpace[Y, K]
    _scl: Optional[alg.ImplementsScalarField[K]] = None
    _zero: Optional[Callable[[], F[*Xs, Y]]] = None
    _add: Optional[Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]] = None
    _sub: Optional[Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]] = None
    _neg: Optional[Callable[[F[*Xs, Y]], F[*Xs, Y]]] = None
    _smul: Optional[Callable[[K, F[*Xs, Y]], F[*Xs, Y]]] = None
    _sdiv: Optional[Callable[[K, F[*Xs, Y]], F[*Xs, Y]]] = None

    @property
    def scl(self) -> alg.ImplementsScalarField[K]:
        """Return scl property of FunctionSpace object."""
        return self.codomain.scl if self._scl is None else self._scl

    @property
    def zero(self) -> Callable[[], F[*Xs, Y]]:
        """Return zero property of FunctionSpace object."""
        return (
            lift_constant(self.codomain.zero)
            if self._zero is None
            else self._zero
        )

    @property
    def add(self) -> Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]:
        """Return add property of FunctionSpace object."""
        return (
            lift_binary(self.codomain.add) if self._add is None else self._add
        )

    @property
    def sub(self) -> Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]:
        """Return sub property of FunctionSpace object."""
        return (
            lift_binary(self.codomain.sub) if self._sub is None else self._sub
        )

    @property
    def neg(self) -> Callable[[F[*Xs, Y]], F[*Xs, Y]]:
        """Return neg property of FunctionSpace object."""
        return (
            lift_unary(self.codomain.neg) if self._neg is None else self._neg
        )

    @property
    def smul(self) -> Callable[[K, F[*Xs, Y]], F[*Xs, Y]]:
        """Return smul property of FunctionSpace object."""
        return (
            lift_left(self.codomain.smul) if self._smul is None else self._smul
        )

    @property
    def sdiv(self) -> Callable[[K, F[*Xs, Y]], F[*Xs, Y]]:
        """Return sdiv property of FunctionSpace object."""
        return (
            lift_left(self.codomain.sdiv) if self._sdiv is None else self._sdiv
        )


@final
@dataclass(frozen=True)
class FunctionAlgebra[*Xs, Y, K](alg.ImplementsAlgebra[F[*Xs, Y], K]):
    """Implement algebra of algebra-valued functions."""

    codomain: alg.ImplementsAlgebra[Y, K]
    _scl: Optional[alg.ImplementsScalarField[K]] = None
    _zero: Optional[Callable[[], F[*Xs, Y]]] = None
    _add: Optional[Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]] = None
    _sub: Optional[Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]] = None
    _neg: Optional[Callable[[F[*Xs, Y]], F[*Xs, Y]]] = None
    _smul: Optional[Callable[[K, F[*Xs, Y]], F[*Xs, Y]]] = None
    _sdiv: Optional[Callable[[K, F[*Xs, Y]], F[*Xs, Y]]] = None
    _unit: Optional[Callable[[], F[*Xs, Y]]] = None
    _mul: Optional[Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]] = None
    _mpower: Optional[Callable[[F[*Xs, Y], int], F[*Xs, Y]]] = None

    @property
    def scl(self) -> alg.ImplementsScalarField[K]:
        """Return scl property of FunctionAlgebra object."""
        return self.codomain.scl if self._scl is None else self._scl

    @property
    def zero(self) -> Callable[[], F[*Xs, Y]]:
        """Return zero property of FunctionAlgebra object."""
        return (
            lift_constant(self.codomain.zero)
            if self._zero is None
            else self._zero
        )

    @property
    def add(self) -> Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]:
        """Return add property of FunctionAlgebra object."""
        return (
            lift_binary(self.codomain.add) if self._add is None else self._add
        )

    @property
    def sub(self) -> Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]:
        """Return sub property of FunctionAlgebra object."""
        return (
            lift_binary(self.codomain.sub) if self._sub is None else self._sub
        )

    @property
    def neg(self) -> Callable[[F[*Xs, Y]], F[*Xs, Y]]:
        """Return neg property of FunctionAlgebra object."""
        return (
            lift_unary(self.codomain.neg) if self._neg is None else self._neg
        )

    @property
    def smul(self) -> Callable[[K, F[*Xs, Y]], F[*Xs, Y]]:
        """Return smul property of FunctionAlgebra object."""
        return (
            lift_left(self.codomain.smul) if self._smul is None else self._smul
        )

    @property
    def sdiv(self) -> Callable[[K, F[*Xs, Y]], F[*Xs, Y]]:
        """Return sdiv property of FunctionAlgebra object."""
        return (
            lift_left(self.codomain.sdiv) if self._sdiv is None else self._sdiv
        )

    @property
    def unit(self) -> Callable[[], F[*Xs, Y]]:
        """Return unit property of FunctionAlgebra object."""
        return (
            lift_constant(self.codomain.unit)
            if self._unit is None
            else self._unit
        )

    @property
    def mul(self) -> Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]:
        """Return mul property of FunctionAlgebra object."""
        return (
            lift_binary(self.codomain.mul) if self._mul is None else self._mul
        )

    @property
    def mpower(self) -> Callable[[F[*Xs, Y], int], F[*Xs, Y]]:
        """Return mpower property of FunctionAlgebra object."""
        return (
            lift_right(self.codomain.mpower)
            if self._mpower is None
            else self._mpower
        )


@final
@dataclass(frozen=True)
class FunctionAlgebraWithCalculus[*Xs, Y, K](
    alg.ImplementsAlgebraWithCalculus[F[*Xs, Y], K]
):
    """Implement function algebra with functional calculus."""

    codomain: alg.ImplementsAlgebraWithCalculus[Y, K]
    _scl: Optional[alg.ImplementsScalarField[K]] = None
    _zero: Optional[Callable[[], F[*Xs, Y]]] = None
    _add: Optional[Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]] = None
    _sub: Optional[Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]] = None
    _neg: Optional[Callable[[F[*Xs, Y]], F[*Xs, Y]]] = None
    _smul: Optional[Callable[[K, F[*Xs, Y]], F[*Xs, Y]]] = None
    _sdiv: Optional[Callable[[K, F[*Xs, Y]], F[*Xs, Y]]] = None
    _unit: Optional[Callable[[], F[*Xs, Y]]] = None
    _mul: Optional[Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]] = None
    _div: Optional[Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]] = None
    _inv: Optional[Callable[[F[*Xs, Y]], F[*Xs, Y]]] = None
    _sqrt: Optional[Callable[[F[*Xs, Y]], F[*Xs, Y]]] = None
    _adj: Optional[Callable[[F[*Xs, Y]], F[*Xs, Y]]] = None
    _mod: Optional[Callable[[F[*Xs, Y]], F[*Xs, Y]]] = None
    _power: Optional[Callable[[F[*Xs, Y], K], F[*Xs, Y]]] = None
    _mpower: Optional[Callable[[F[*Xs, Y], int], F[*Xs, Y]]] = None

    @property
    def scl(self) -> alg.ImplementsScalarField[K]:
        """Return scl property of FunctionAlgebraWithCalculus object."""
        return self.codomain.scl if self._scl is None else self._scl

    @property
    def zero(self) -> Callable[[], F[*Xs, Y]]:
        """Return zero property of FunctionAlgebraWithCalculus object."""
        return (
            lift_constant(self.codomain.zero)
            if self._zero is None
            else self._zero
        )

    @property
    def add(self) -> Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]:
        """Return add property of FunctionAlgebraWithCalculus object."""
        return (
            lift_binary(self.codomain.add) if self._add is None else self._add
        )

    @property
    def sub(self) -> Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]:
        """Return sub property of FunctionAlgebraWithCalculus object."""
        return (
            lift_binary(self.codomain.sub) if self._sub is None else self._sub
        )

    @property
    def neg(self) -> Callable[[F[*Xs, Y]], F[*Xs, Y]]:
        """Return neg property of FunctionAlgebraWithCalculus object."""
        return (
            lift_unary(self.codomain.neg) if self._neg is None else self._neg
        )

    @property
    def smul(self) -> Callable[[K, F[*Xs, Y]], F[*Xs, Y]]:
        """Return smul property of FunctionAlgebraWithCalculus object."""
        return (
            lift_left(self.codomain.smul) if self._smul is None else self._smul
        )

    @property
    def sdiv(self) -> Callable[[K, F[*Xs, Y]], F[*Xs, Y]]:
        """Return sdiv property of FunctionAlgebraWithCalculus object."""
        return (
            lift_left(self.codomain.sdiv) if self._sdiv is None else self._sdiv
        )

    @property
    def unit(self) -> Callable[[], F[*Xs, Y]]:
        """Return unit property of FunctionAlgebraWithCalculus object."""
        return (
            lift_constant(self.codomain.unit)
            if self._unit is None
            else self._unit
        )

    @property
    def mul(self) -> Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]:
        """Return mul property of FunctionAlgebraWithCalculus object."""
        return (
            lift_binary(self.codomain.mul) if self._mul is None else self._mul
        )

    @property
    def div(self) -> Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]:
        """Return div property of FunctionAlgebraWithCalculus object."""
        return (
            lift_binary(self.codomain.div) if self._div is None else self._div
        )

    @property
    def inv(self) -> Callable[[F[*Xs, Y]], F[*Xs, Y]]:
        """Return inv property of FunctionAlgebraWithCalculus object."""
        return (
            lift_unary(self.codomain.inv) if self._inv is None else self._inv
        )

    @property
    def sqrt(self) -> Callable[[F[*Xs, Y]], F[*Xs, Y]]:
        """Return sqrt property of FunctionAlgebraWithCalculus object."""
        return (
            lift_unary(self.codomain.sqrt)
            if self._sqrt is None
            else self._sqrt
        )

    @property
    def adj(self) -> Callable[[F[*Xs, Y]], F[*Xs, Y]]:
        """Return adj property of FunctionAlgebraWithCalculus object."""
        return (
            lift_unary(self.codomain.adj) if self._adj is None else self._adj
        )

    @property
    def mod(self) -> Callable[[F[*Xs, Y]], F[*Xs, Y]]:
        """Return mod property of FunctionAlgebraWithCalculus object."""
        return (
            lift_unary(self.codomain.mod) if self._mod is None else self._mod
        )

    @property
    def power(self) -> Callable[[F[*Xs, Y], K], F[*Xs, Y]]:
        """Return power property of FunctionAlgebraWithCalculus object."""
        return (
            lift_right(self.codomain.power)
            if self._power is None
            else self._power
        )

    @property
    def mpower(self) -> Callable[[F[*Xs, Y], int], F[*Xs, Y]]:
        """Return mpower property of FunctionAlgebraWithCalculus object."""
        return (
            lift_right(self.codomain.mpower)
            if self._mpower is None
            else self._mpower
        )


@final
@dataclass(frozen=True)
class FunctionBimodule[*Xs, Y, K, L, R](
    alg.ImplementsBimodule[F[*Xs, Y], K, L, R]
):
    """Implement bimodule of bimodule-valued functions."""

    codomain: alg.ImplementsBimodule[Y, K, L, R]
    _scl: Optional[alg.ImplementsScalarField[K]] = None
    _zero: Optional[Callable[[], F[*Xs, Y]]] = None
    _add: Optional[Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]] = None
    _sub: Optional[Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]] = None
    _neg: Optional[Callable[[F[*Xs, Y]], F[*Xs, Y]]] = None
    _smul: Optional[Callable[[K, F[*Xs, Y]], F[*Xs, Y]]] = None
    _sdiv: Optional[Callable[[K, F[*Xs, Y]], F[*Xs, Y]]] = None
    _lmul: Optional[Callable[[L, F[*Xs, Y]], F[*Xs, Y]]] = None
    _rmul: Optional[Callable[[F[*Xs, Y], R], F[*Xs, Y]]] = None

    @property
    def scl(self) -> alg.ImplementsScalarField[K]:
        """Return scl property of FunctionBimodule object."""
        return self.codomain.scl if self._scl is None else self._scl

    @property
    def zero(self) -> Callable[[], F[*Xs, Y]]:
        """Return zero property of FunctionBimodule object."""
        return (
            lift_constant(self.codomain.zero)
            if self._zero is None
            else self._zero
        )

    @property
    def add(self) -> Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]:
        """Return add property of FunctionBimodule object."""
        return (
            lift_binary(self.codomain.add) if self._add is None else self._add
        )

    @property
    def sub(self) -> Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]:
        """Return sub property of FunctionBimodule object."""
        return (
            lift_binary(self.codomain.sub) if self._sub is None else self._sub
        )

    @property
    def neg(self) -> Callable[[F[*Xs, Y]], F[*Xs, Y]]:
        """Return neg property of FunctionBimodule object."""
        return (
            lift_unary(self.codomain.neg) if self._neg is None else self._neg
        )

    @property
    def smul(self) -> Callable[[K, F[*Xs, Y]], F[*Xs, Y]]:
        """Return smul property of FunctionBimodule object."""
        return (
            lift_left(self.codomain.smul) if self._smul is None else self._smul
        )

    @property
    def sdiv(self) -> Callable[[K, F[*Xs, Y]], F[*Xs, Y]]:
        """Return sdiv property of FunctionBimodule object."""
        return (
            lift_left(self.codomain.sdiv) if self._sdiv is None else self._sdiv
        )

    @property
    def lmul(self) -> Callable[[L, F[*Xs, Y]], F[*Xs, Y]]:
        """Return lmul property of FunctionBimodule object."""
        return (
            lift_left(self.codomain.lmul) if self._lmul is None else self._lmul
        )

    @property
    def rmul(self) -> Callable[[F[*Xs, Y], R], F[*Xs, Y]]:
        """Return rmul property of FunctionBimodule object."""
        return (
            lift_right(self.codomain.rmul)
            if self._rmul is None
            else self._rmul
        )


@final
@dataclass(frozen=True)
class BivariateFunctionBimodule[X1, X2, Y, K](
    alg.ImplementsBimodule[F[X1, X2, Y], K, F[X1, Y], F[X2, Y]]
):
    """Implement bivariate function space as a bivariate function bimodule."""

    codomain: alg.ImplementsBimodule[Y, K, Y, Y]
    _scl: Optional[alg.ImplementsScalarField[K]] = None
    _zero: Optional[Callable[[], F[X1, X2, Y]]] = None
    _add: Optional[Callable[[F[X1, X2, Y], F[X1, X2, Y]], F[X1, X2, Y]]] = None
    _sub: Optional[Callable[[F[X1, X2, Y], F[X1, X2, Y]], F[X1, X2, Y]]] = None
    _neg: Optional[Callable[[F[X1, X2, Y]], F[X1, X2, Y]]] = None
    _smul: Optional[Callable[[K, F[X1, X2, Y]], F[X1, X2, Y]]] = None
    _sdiv: Optional[Callable[[K, F[X1, X2, Y]], F[X1, X2, Y]]] = None
    _lmul: Optional[Callable[[F[X1, Y], F[X1, X2, Y]], F[X1, X2, Y]]] = None
    _rmul: Optional[Callable[[F[X1, X2, Y], F[X2, Y]], F[X1, X2, Y]]] = None

    @property
    def scl(self) -> alg.ImplementsScalarField[K]:
        """Return scl property of BivariateFunctionBimodule object."""
        return self.codomain.scl if self._scl is None else self._scl

    @property
    def zero(self) -> Callable[[], F[X1, X2, Y]]:
        """Return zero property of BivariateFunctionBimodule object."""
        return (
            lift_constant(self.codomain.zero)
            if self._zero is None
            else self._zero
        )

    @property
    def add(self) -> Callable[[F[X1, X2, Y], F[X1, X2, Y]], F[X1, X2, Y]]:
        """Return add property of BivariateFunctionBimodule object."""
        return (
            (lift_binary(self.codomain.add))
            if self._add is None
            else self._add
        )

    @property
    def sub(self) -> Callable[[F[X1, X2, Y], F[X1, X2, Y]], F[X1, X2, Y]]:
        """Return sub property of BivariateFunctionBimodule object."""
        return (
            (lift_binary(self.codomain.sub))
            if self._sub is None
            else self._sub
        )

    @property
    def neg(self) -> Callable[[F[X1, X2, Y]], F[X1, X2, Y]]:
        """Return neg property of BivariateFunctionBimodule object."""
        return (
            lift_unary(self.codomain.neg) if self._neg is None else self._neg
        )

    @property
    def smul(self) -> Callable[[K, F[X1, X2, Y]], F[X1, X2, Y]]:
        """Return smul property of BivariateFunctionBimodule object."""
        return (
            lift_left(self.codomain.smul) if self._smul is None else self._smul
        )

    @property
    def sdiv(self) -> Callable[[K, F[X1, X2, Y]], F[X1, X2, Y]]:
        """Return sdiv property of BivariateFunctionBimodule object."""
        return (
            lift_left(self.codomain.sdiv) if self._sdiv is None else self._sdiv
        )

    @property
    def lmul(self) -> Callable[[F[X1, Y], F[X1, X2, Y]], F[X1, X2, Y]]:
        """Return lmul property of BivariateFunctionBimodule object."""
        return (
            (lift_left_bivariate(self.codomain.lmul))
            if self._lmul is None
            else self._lmul
        )

    @property
    def rmul(self) -> Callable[[F[X1, X2, Y], F[X2, Y]], F[X1, X2, Y]]:
        """Return rmul property of BivariateFunctionBimodule object."""
        return (
            (lift_right_bivariate(self.codomain.rmul))
            if self._rmul is None
            else self._rmul
        )


@final
@dataclass(frozen=True)
class BivariateFunctionDivBimodule[X1, X2, Y, K](
    alg.ImplementsDivBimodule[F[X1, X2, Y], K, F[X1, Y], F[X2, Y]]
):
    """Implement bivariate function space as a bivariate function bimodule."""

    codomain: alg.ImplementsDivBimodule[Y, K, Y, Y]
    _scl: Optional[alg.ImplementsScalarField[K]] = None
    _zero: Optional[Callable[[], F[X1, X2, Y]]] = None
    _add: Optional[Callable[[F[X1, X2, Y], F[X1, X2, Y]], F[X1, X2, Y]]] = None
    _sub: Optional[Callable[[F[X1, X2, Y], F[X1, X2, Y]], F[X1, X2, Y]]] = None
    _neg: Optional[Callable[[F[X1, X2, Y]], F[X1, X2, Y]]] = None
    _smul: Optional[Callable[[K, F[X1, X2, Y]], F[X1, X2, Y]]] = None
    _sdiv: Optional[Callable[[K, F[X1, X2, Y]], F[X1, X2, Y]]] = None
    _lmul: Optional[Callable[[F[X1, Y], F[X1, X2, Y]], F[X1, X2, Y]]] = None
    _ldiv: Optional[Callable[[F[X1, Y], F[X1, X2, Y]], F[X1, X2, Y]]] = None
    _rmul: Optional[Callable[[F[X1, X2, Y], F[X2, Y]], F[X1, X2, Y]]] = None
    _rdiv: Optional[Callable[[F[X1, X2, Y], F[X2, Y]], F[X1, X2, Y]]] = None

    @property
    def scl(self) -> alg.ImplementsScalarField[K]:
        """Return scl property of BivariateFunctionDivBimodule object."""
        return self.codomain.scl if self._scl is None else self._scl

    @property
    def zero(self) -> Callable[[], F[X1, X2, Y]]:
        """Return zero property of BivariateFunctionDivBimodule object."""
        return (
            lift_constant(self.codomain.zero)
            if self._zero is None
            else self._zero
        )

    @property
    def add(self) -> Callable[[F[X1, X2, Y], F[X1, X2, Y]], F[X1, X2, Y]]:
        """Return add property of BivariateFunctionDivBimodule object."""
        return (
            (lift_binary(self.codomain.add))
            if self._add is None
            else self._add
        )

    @property
    def sub(self) -> Callable[[F[X1, X2, Y], F[X1, X2, Y]], F[X1, X2, Y]]:
        """Return sub property of BivariateFunctionDivBimodule object."""
        return (
            (lift_binary(self.codomain.sub))
            if self._sub is None
            else self._sub
        )

    @property
    def neg(self) -> Callable[[F[X1, X2, Y]], F[X1, X2, Y]]:
        """Return neg property of BivariateFunctionDivBimodule object."""
        return (
            lift_unary(self.codomain.neg) if self._neg is None else self._neg
        )

    @property
    def smul(self) -> Callable[[K, F[X1, X2, Y]], F[X1, X2, Y]]:
        """Return smul property of BivariateFunctionDivBimodule object."""
        return (
            lift_left(self.codomain.smul) if self._smul is None else self._smul
        )

    @property
    def sdiv(self) -> Callable[[K, F[X1, X2, Y]], F[X1, X2, Y]]:
        """Return sdiv property of BivariateFunctionDivBimodule object."""
        return (
            lift_left(self.codomain.sdiv) if self._sdiv is None else self._sdiv
        )

    @property
    def lmul(self) -> Callable[[F[X1, Y], F[X1, X2, Y]], F[X1, X2, Y]]:
        """Return lmul property of BivariateFunctionDivBimodule object."""
        return (
            (lift_left_bivariate(self.codomain.lmul))
            if self._lmul is None
            else self._lmul
        )

    @property
    def ldiv(self) -> Callable[[F[X1, Y], F[X1, X2, Y]], F[X1, X2, Y]]:
        """Return ldiv property of BivariateFunctionDivBimodule object."""
        return (
            (lift_left_bivariate(self.codomain.ldiv))
            if self._ldiv is None
            else self._ldiv
        )

    @property
    def rmul(self) -> Callable[[F[X1, X2, Y], F[X2, Y]], F[X1, X2, Y]]:
        """Return rmul property of BivariateFunctionDivBimodule object."""
        return (
            (lift_right_bivariate(self.codomain.rmul))
            if self._rmul is None
            else self._rmul
        )

    @property
    def rdiv(self) -> Callable[[F[X1, X2, Y], F[X2, Y]], F[X1, X2, Y]]:
        """Return rdiv property of BivariateFunctionDivBimodule object."""
        return (
            (lift_right_bivariate(self.codomain.rdiv))
            if self._rdiv is None
            else self._rdiv
        )

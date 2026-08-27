"""Provide classes and functions implementing function space operations."""

import nlsa.abstract_algebra as alg
from collections.abc import Callable
from dataclasses import dataclass
from itertools import repeat
from functools import reduce
from nlsa.typing import DEFAULT
from typing import final

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


def mpower[A](f: F[A, A], m: int, /) -> F[A, A]:
    """Form monoidal power of endomorphism."""
    if m == 0:
        fn = identity
    else:
        fn = reduce(compose, repeat(f, m))
    return fn


def make_mpower[A](f: Callable[[A, A], A], /) -> Callable[[A, int], A]:
    """Make monoidal power flom binary operation."""

    def mpower(a: A, m: int) -> A:
        return reduce(f, repeat(a, m))

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
@dataclass(frozen=True, slots=True)
class FunctionSpace[*Xs, Y, K](alg.ImplementsVectorSpace[F[*Xs, Y], K]):
    """Implement function space."""

    codomain: alg.ImplementsVectorSpace[Y, K]
    _zero: Callable[[], F[*Xs, Y]]
    _add: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]
    _sub: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]
    _neg: Callable[[F[*Xs, Y]], F[*Xs, Y]]
    _smul: Callable[[K, F[*Xs, Y]], F[*Xs, Y]]
    _sdiv: Callable[[K, F[*Xs, Y]], F[*Xs, Y]]

    @property
    def scl(self) -> alg.ImplementsScalarField[K]:
        """Scalar field associated with FunctionSpace object."""
        return self.codomain.scl

    def zero(self, /) -> F[*Xs, Y]:
        """Return zero function."""
        return self._zero()

    def add(self, f: F[*Xs, Y], g: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Add two functions."""
        return self._add(f, g)

    def sub(self, f: F[*Xs, Y], g: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Subtract two functions."""
        return self._sub(f, g)

    def neg(self, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Compute additive inverse (negation) of a function."""
        return self._neg(f)

    def smul(self, k: K, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Multiply a function by a scalar."""
        return self._smul(k, f)

    def sdiv(self, k: K, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Divide a function by a scalar."""
        return self._sdiv(k, f)


def function_space[*Xs, Y, K](
    codomain: alg.ImplementsVectorSpace[Y, K],
    zero: Callable[[], F[*Xs, Y]] | DEFAULT = DEFAULT,
    add: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    sub: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    neg: Callable[[F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    smul: Callable[[K, F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    sdiv: Callable[[K, F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
) -> FunctionSpace[*Xs, Y, K]:
    """Build FunctionAlgebraWithCalculus object."""
    return FunctionSpace(
        codomain=codomain,
        _zero=(lift_constant(codomain.zero) if zero is DEFAULT else zero),
        _add=(lift_binary(codomain.add) if add is DEFAULT else add),
        _sub=(lift_binary(codomain.sub) if sub is DEFAULT else sub),
        _neg=(lift_unary(codomain.neg) if neg is DEFAULT else neg),
        _smul=(lift_left(codomain.smul) if smul is DEFAULT else smul),
        _sdiv=(lift_left(codomain.sdiv) if sdiv is DEFAULT else sdiv),
    )


@final
@dataclass(frozen=True, slots=True)
class FunctionAlgebra[*Xs, Y, K](alg.ImplementsAlgebra[F[*Xs, Y], K]):
    """Implement algebra of algebra-valued functions."""

    codomain: alg.ImplementsAlgebra[Y, K]
    _zero: Callable[[], F[*Xs, Y]]
    _add: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]
    _sub: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]
    _neg: Callable[[F[*Xs, Y]], F[*Xs, Y]]
    _smul: Callable[[K, F[*Xs, Y]], F[*Xs, Y]]
    _sdiv: Callable[[K, F[*Xs, Y]], F[*Xs, Y]]
    _unit: Callable[[], F[*Xs, Y]]
    _mul: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]
    _mpower: Callable[[F[*Xs, Y], int], F[*Xs, Y]]

    @property
    def scl(self) -> alg.ImplementsScalarField[K]:
        """Scalar field associated with FunctionAlgebra object."""
        return self.codomain.scl

    def zero(self, /) -> F[*Xs, Y]:
        """Return zero function."""
        return self._zero()

    def add(self, f: F[*Xs, Y], g: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Add two functions."""
        return self._add(f, g)

    def sub(self, f: F[*Xs, Y], g: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Subtract two functions."""
        return self._sub(f, g)

    def neg(self, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Compute additive inverse (negation) of a function."""
        return self._neg(f)

    def smul(self, k: K, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Multiply a function by a scalar."""
        return self._smul(k, f)

    def sdiv(self, k: K, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Divide a function by a scalar."""
        return self._sdiv(k, f)

    def unit(self, /) -> F[*Xs, Y]:
        """Return multiplicative unit function."""
        return self._unit()

    def mul(self, f: F[*Xs, Y], g: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Multiply two functions."""
        return self._mul(f, g)

    def mpower(self, f: F[*Xs, Y], m: int, /) -> F[*Xs, Y]:
        """Exponentiate a function by an integer."""
        return self._mpower(f, m)


def function_algebra[*Xs, Y, K](
    codomain: alg.ImplementsAlgebra[Y, K],
    zero: Callable[[], F[*Xs, Y]] | DEFAULT = DEFAULT,
    add: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    sub: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    neg: Callable[[F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    smul: Callable[[K, F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    sdiv: Callable[[K, F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    unit: Callable[[], F[*Xs, Y]] | DEFAULT = DEFAULT,
    mul: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    mpower: Callable[[F[*Xs, Y], int], F[*Xs, Y]] | DEFAULT = DEFAULT,
) -> FunctionAlgebra[*Xs, Y, K]:
    """Build FunctionAlgebraWithCalculus object."""
    return FunctionAlgebra(
        codomain=codomain,
        _zero=(lift_constant(codomain.zero) if zero is DEFAULT else zero),
        _add=(lift_binary(codomain.add) if add is DEFAULT else add),
        _sub=(lift_binary(codomain.sub) if sub is DEFAULT else sub),
        _neg=(lift_unary(codomain.neg) if neg is DEFAULT else neg),
        _smul=(lift_left(codomain.smul) if smul is DEFAULT else smul),
        _sdiv=(lift_left(codomain.sdiv) if sdiv is DEFAULT else sdiv),
        _unit=(lift_constant(codomain.unit) if unit is DEFAULT else unit),
        _mul=(lift_binary(codomain.mul) if mul is DEFAULT else mul),
        _mpower=(lift_right(codomain.mpower) if mpower is DEFAULT else mpower),
    )


@final
@dataclass(frozen=True)
class FunctionAlgebraWithCalculus[*Xs, Y, K](
    alg.ImplementsAlgebraWithCalculus[F[*Xs, Y], K]
):
    """Implement function algebra with functional calculus."""

    codomain: alg.ImplementsAlgebraWithCalculus[Y, K]
    _zero: Callable[[], F[*Xs, Y]]
    _add: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]
    _sub: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]
    _neg: Callable[[F[*Xs, Y]], F[*Xs, Y]]
    _smul: Callable[[K, F[*Xs, Y]], F[*Xs, Y]]
    _sdiv: Callable[[K, F[*Xs, Y]], F[*Xs, Y]]
    _unit: Callable[[], F[*Xs, Y]]
    _mul: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]
    _div: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]
    _inv: Callable[[F[*Xs, Y]], F[*Xs, Y]]
    _sqrt: Callable[[F[*Xs, Y]], F[*Xs, Y]]
    _abs: Callable[[F[*Xs, Y]], F[*Xs, Y]]
    _exp: Callable[[F[*Xs, Y]], F[*Xs, Y]]
    _log: Callable[[F[*Xs, Y]], F[*Xs, Y]]
    _power: Callable[[F[*Xs, Y], K], F[*Xs, Y]]
    _mpower: Callable[[F[*Xs, Y], int], F[*Xs, Y]]

    @property
    def scl(self) -> alg.ImplementsRealScalarField[K]:
        """Scalar field associated with FunctionAlgebra object."""
        return self.codomain.scl

    def zero(self, /) -> F[*Xs, Y]:
        """Return zero function."""
        return self._zero()

    def add(self, f: F[*Xs, Y], g: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Add two functions."""
        return self._add(f, g)

    def sub(self, f: F[*Xs, Y], g: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Subtract two functions."""
        return self._sub(f, g)

    def neg(self, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Compute additive inverse (negation) of a function."""
        return self._neg(f)

    def smul(self, k: K, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Multiply a function by a scalar."""
        return self._smul(k, f)

    def sdiv(self, k: K, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Divide a function by a scalar."""
        return self._sdiv(k, f)

    def unit(self, /) -> F[*Xs, Y]:
        """Return multiplicative unit function."""
        return self._unit()

    def mul(self, f: F[*Xs, Y], g: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Multiply two functions."""
        return self._mul(f, g)

    def mpower(self, f: F[*Xs, Y], m: int, /) -> F[*Xs, Y]:
        """Exponentiate a function by an integer."""
        return self._mpower(f, m)

    def div(self, f: F[*Xs, Y], g: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Divide two functions."""
        return self._div(f, g)

    def inv(self, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Compute multiplicative inverse of a function."""
        return self._inv(f)

    def sqrt(self, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Compute square root of a function."""
        return self._sqrt(f)

    def abs(self, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Compute modulus of a function."""
        return self._abs(f)

    def exp(self, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Compute exponential of a function."""
        return self._exp(f)

    def log(self, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Compute natural logarithm of a function."""
        return self._log(f)

    def power(self, f: F[*Xs, Y], k: K, /) -> F[*Xs, Y]:
        """Exponentiate a function by a scalar."""
        return self._power(f, k)


def function_algebra_with_calculus[*Xs, Y, K](
    codomain: alg.ImplementsAlgebraWithCalculus[Y, K],
    zero: Callable[[], F[*Xs, Y]] | DEFAULT = DEFAULT,
    add: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    sub: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    neg: Callable[[F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    smul: Callable[[K, F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    sdiv: Callable[[K, F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    unit: Callable[[], F[*Xs, Y]] | DEFAULT = DEFAULT,
    mul: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    div: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    inv: Callable[[F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    sqrt: Callable[[F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    abs: Callable[[F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    exp: Callable[[F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    log: Callable[[F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    power: Callable[[F[*Xs, Y], K], F[*Xs, Y]] | DEFAULT = DEFAULT,
    mpower: Callable[[F[*Xs, Y], int], F[*Xs, Y]] | DEFAULT = DEFAULT,
) -> FunctionAlgebraWithCalculus[*Xs, Y, K]:
    """Build FunctionAlgebraWithCalculus object."""
    return FunctionAlgebraWithCalculus(
        codomain=codomain,
        _zero=(lift_constant(codomain.zero) if zero is DEFAULT else zero),
        _add=(lift_binary(codomain.add) if add is DEFAULT else add),
        _sub=(lift_binary(codomain.sub) if sub is DEFAULT else sub),
        _neg=(lift_unary(codomain.neg) if neg is DEFAULT else neg),
        _smul=(lift_left(codomain.smul) if smul is DEFAULT else smul),
        _sdiv=(lift_left(codomain.sdiv) if sdiv is DEFAULT else sdiv),
        _unit=(lift_constant(codomain.unit) if unit is DEFAULT else unit),
        _mul=(lift_binary(codomain.mul) if mul is DEFAULT else mul),
        _div=(lift_binary(codomain.div) if div is DEFAULT else div),
        _inv=(lift_unary(codomain.inv) if inv is DEFAULT else inv),
        _sqrt=(lift_unary(codomain.sqrt) if sqrt is DEFAULT else sqrt),
        _abs=(lift_unary(codomain.abs) if abs is DEFAULT else abs),
        _exp=(lift_unary(codomain.exp) if exp is DEFAULT else exp),
        _log=(lift_unary(codomain.log) if log is DEFAULT else log),
        _power=(lift_right(codomain.power) if power is DEFAULT else power),
        _mpower=(lift_right(codomain.mpower) if mpower is DEFAULT else mpower),
    )


@final
@dataclass(frozen=True, slots=True)
class FunctionStarAlgebraWithCalculus[*Xs, Y, K](
    alg.ImplementsStarAlgebraWithCalculus[F[*Xs, Y], K]
):
    """Implement function star algebra with functional calculus."""

    codomain: alg.ImplementsStarAlgebraWithCalculus[Y, K]
    _zero: Callable[[], F[*Xs, Y]]
    _add: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]
    _sub: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]
    _neg: Callable[[F[*Xs, Y]], F[*Xs, Y]]
    _smul: Callable[[K, F[*Xs, Y]], F[*Xs, Y]]
    _sdiv: Callable[[K, F[*Xs, Y]], F[*Xs, Y]]
    _unit: Callable[[], F[*Xs, Y]]
    _mul: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]
    _div: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]
    _inv: Callable[[F[*Xs, Y]], F[*Xs, Y]]
    _sqrt: Callable[[F[*Xs, Y]], F[*Xs, Y]]
    _abs: Callable[[F[*Xs, Y]], F[*Xs, Y]]
    _exp: Callable[[F[*Xs, Y]], F[*Xs, Y]]
    _log: Callable[[F[*Xs, Y]], F[*Xs, Y]]
    _power: Callable[[F[*Xs, Y], K], F[*Xs, Y]]
    _mpower: Callable[[F[*Xs, Y], int], F[*Xs, Y]]
    _adj: Callable[[F[*Xs, Y]], F[*Xs, Y]]

    @property
    def scl(self) -> alg.ImplementsComplexScalarField[K]:
        """Scalar field associated with FunctionStarAlgebra object."""
        return self.codomain.scl

    def zero(self, /) -> F[*Xs, Y]:
        """Return zero function."""
        return self._zero()

    def add(self, f: F[*Xs, Y], g: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Add two functions."""
        return self._add(f, g)

    def sub(self, f: F[*Xs, Y], g: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Subtract two functions."""
        return self._sub(f, g)

    def neg(self, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Compute additive inverse (negation) of a function."""
        return self._neg(f)

    def smul(self, k: K, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Multiply a function by a scalar."""
        return self._smul(k, f)

    def sdiv(self, k: K, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Divide a function by a scalar."""
        return self._sdiv(k, f)

    def unit(self, /) -> F[*Xs, Y]:
        """Return multiplicative unit function."""
        return self._unit()

    def mul(self, f: F[*Xs, Y], g: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Multiply two functions."""
        return self._mul(f, g)

    def div(self, f: F[*Xs, Y], g: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Divide two functions."""
        return self._div(f, g)

    def mpower(self, f: F[*Xs, Y], m: int, /) -> F[*Xs, Y]:
        """Exponentiate a function by an integer."""
        return self._mpower(f, m)

    def inv(self, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Compute the multiplicative inverse of a function."""
        return self._inv(f)

    def sqrt(self, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Compute the square root of a function."""
        return self._sqrt(f)

    def abs(self, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Compute the modulus of a function."""
        return self._abs(f)

    def exp(self, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Compute exponential of a function."""
        return self._exp(f)

    def log(self, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Compute natural logarithm of a function."""
        return self._log(f)

    def power(self, f: F[*Xs, Y], k: K, /) -> F[*Xs, Y]:
        """Exponentiate a function by a scalar."""
        return self._power(f, k)

    def adj(self, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Compute algebraic adjoint of a function."""
        return self._adj(f)


def function_star_algebra_with_calculus[*Xs, Y, K](
    codomain: alg.ImplementsStarAlgebraWithCalculus[Y, K],
    zero: Callable[[], F[*Xs, Y]] | DEFAULT = DEFAULT,
    add: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    sub: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    neg: Callable[[F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    smul: Callable[[K, F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    sdiv: Callable[[K, F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    unit: Callable[[], F[*Xs, Y]] | DEFAULT = DEFAULT,
    mul: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    div: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    inv: Callable[[F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    sqrt: Callable[[F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    abs: Callable[[F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    exp: Callable[[F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    log: Callable[[F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    power: Callable[[F[*Xs, Y], K], F[*Xs, Y]] | DEFAULT = DEFAULT,
    mpower: Callable[[F[*Xs, Y], int], F[*Xs, Y]] | DEFAULT = DEFAULT,
    adj: Callable[[F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
) -> FunctionStarAlgebraWithCalculus[*Xs, Y, K]:
    """Build FunctionStarAlgebraWithCalculus object."""
    return FunctionStarAlgebraWithCalculus(
        codomain=codomain,
        _zero=(lift_constant(codomain.zero) if zero is DEFAULT else zero),
        _add=(lift_binary(codomain.add) if add is DEFAULT else add),
        _sub=(lift_binary(codomain.sub) if sub is DEFAULT else sub),
        _neg=(lift_unary(codomain.neg) if neg is DEFAULT else neg),
        _smul=(lift_left(codomain.smul) if smul is DEFAULT else smul),
        _sdiv=(lift_left(codomain.sdiv) if sdiv is DEFAULT else sdiv),
        _unit=(lift_constant(codomain.unit) if unit is DEFAULT else unit),
        _mul=(lift_binary(codomain.mul) if mul is DEFAULT else mul),
        _div=(lift_binary(codomain.div) if div is DEFAULT else div),
        _inv=(lift_unary(codomain.inv) if inv is DEFAULT else inv),
        _sqrt=(lift_unary(codomain.sqrt) if sqrt is DEFAULT else sqrt),
        _abs=(lift_unary(codomain.abs) if abs is DEFAULT else abs),
        _power=(lift_right(codomain.power) if power is DEFAULT else power),
        _exp=(lift_unary(codomain.exp) if exp is DEFAULT else exp),
        _log=(lift_unary(codomain.log) if log is DEFAULT else log),
        _mpower=(lift_right(codomain.mpower) if mpower is DEFAULT else mpower),
        _adj=(lift_unary(codomain.adj) if adj is DEFAULT else adj),
    )


@final
@dataclass(frozen=True, slots=True)
class FunctionBimodule[*Xs, Y, K, L, R](
    alg.ImplementsBimodule[F[*Xs, Y], K, L, R]
):
    """Implement bimodule of bimodule-valued functions."""

    codomain: alg.ImplementsBimodule[Y, K, L, R]
    _zero: Callable[[], F[*Xs, Y]]
    _add: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]
    _sub: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]]
    _neg: Callable[[F[*Xs, Y]], F[*Xs, Y]]
    _smul: Callable[[K, F[*Xs, Y]], F[*Xs, Y]]
    _sdiv: Callable[[K, F[*Xs, Y]], F[*Xs, Y]]
    _lmul: Callable[[L, F[*Xs, Y]], F[*Xs, Y]]
    _rmul: Callable[[F[*Xs, Y], R], F[*Xs, Y]]

    @property
    def scl(self) -> alg.ImplementsScalarField[K]:
        """Return scl property of FunctionAlgebra object."""
        return self.codomain.scl

    def zero(self, /) -> F[*Xs, Y]:
        """Return zero property of FunctionAlgebraWithCalculus object."""
        return self._zero()

    def add(self, f: F[*Xs, Y], g: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Return add property of FunctionAlgebraWithCalculus object."""
        return self._add(f, g)

    def sub(self, f: F[*Xs, Y], g: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Return sub property of FunctionAlgebraWithCalculus object."""
        return self._sub(f, g)

    def neg(self, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Return neg property of FunctionAlgebraWithCalculus object."""
        return self._neg(f)

    def smul(self, k: K, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Return smul property of FunctionAlgebraWithCalculus object."""
        return self._smul(k, f)

    def sdiv(self, k: K, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Return sdiv property of FunctionAlgebraWithCalculus object."""
        return self._sdiv(k, f)

    def lmul(self, k: L, f: F[*Xs, Y], /) -> F[*Xs, Y]:
        """Return lmul property of FunctionBimodule object."""
        return self._lmul(k, f)

    def rmul(self, f: F[*Xs, Y], k: R, /) -> F[*Xs, Y]:
        """Return rmul property of FunctionBimodule object."""
        return self._rmul(f, k)


def function_bimodule[*Xs, Y, K, L, R](
    codomain: alg.ImplementsBimodule[Y, K, L, R],
    zero: Callable[[], F[*Xs, Y]] | DEFAULT = DEFAULT,
    add: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    sub: Callable[[F[*Xs, Y], F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    neg: Callable[[F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    smul: Callable[[K, F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    sdiv: Callable[[K, F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    lmul: Callable[[L, F[*Xs, Y]], F[*Xs, Y]] | DEFAULT = DEFAULT,
    rmul: Callable[[F[*Xs, Y], R], F[*Xs, Y]] | DEFAULT = DEFAULT,
) -> FunctionBimodule[*Xs, Y, K, L, R]:
    """Build FunctionAlgebraWithCalculus object."""
    return FunctionBimodule(
        codomain=codomain,
        _zero=(lift_constant(codomain.zero) if zero is DEFAULT else zero),
        _add=(lift_binary(codomain.add) if add is DEFAULT else add),
        _sub=(lift_binary(codomain.sub) if sub is DEFAULT else sub),
        _neg=(lift_unary(codomain.neg) if neg is DEFAULT else neg),
        _smul=(lift_left(codomain.smul) if smul is DEFAULT else smul),
        _sdiv=(lift_left(codomain.sdiv) if sdiv is DEFAULT else sdiv),
        _lmul=(lift_left(codomain.lmul) if lmul is DEFAULT else lmul),
        _rmul=(lift_right(codomain.rmul) if rmul is DEFAULT else rmul),
    )


@final
@dataclass(frozen=True, slots=True)
class BivariateFunctionBimodule[X1, X2, Y, K](
    alg.ImplementsBimodule[F[X1, X2, Y], K, F[X1, Y], F[X2, Y]]
):
    """Implement space of bivariate functions as a bimodule."""

    codomain: alg.ImplementsBimodule[Y, K, Y, Y]
    _zero: Callable[[], F[X1, X2, Y]]
    _add: Callable[[F[X1, X2, Y], F[X1, X2, Y]], F[X1, X2, Y]]
    _sub: Callable[[F[X1, X2, Y], F[X1, X2, Y]], F[X1, X2, Y]]
    _neg: Callable[[F[X1, X2, Y]], F[X1, X2, Y]]
    _smul: Callable[[K, F[X1, X2, Y]], F[X1, X2, Y]]
    _sdiv: Callable[[K, F[X1, X2, Y]], F[X1, X2, Y]]
    _lmul: Callable[[F[X1, Y], F[X1, X2, Y]], F[X1, X2, Y]]
    _rmul: Callable[[F[X1, X2, Y], F[X2, Y]], F[X1, X2, Y]]

    @property
    def scl(self) -> alg.ImplementsScalarField[K]:
        """Scalar field associated with BivariateFunctionBimodule object."""
        return self.codomain.scl

    def zero(self, /) -> F[X1, X2, Y]:
        """Return zero bivariate function."""
        return self._zero()

    def add(self, f: F[X1, X2, Y], g: F[X1, X2, Y], /) -> F[X1, X2, Y]:
        """Add two bivariate functions."""
        return self._add(f, g)

    def sub(self, f: F[X1, X2, Y], g: F[X1, X2, Y], /) -> F[X1, X2, Y]:
        """Subtract two bivariate functions."""
        return self._sub(f, g)

    def neg(self, f: F[X1, X2, Y], /) -> F[X1, X2, Y]:
        """Compute additive inverse (negation) of a bivariate function."""
        return self._neg(f)

    def smul(self, k: K, f: F[X1, X2, Y], /) -> F[X1, X2, Y]:
        """Multiply a bivariate function by a scalar."""
        return self._smul(k, f)

    def sdiv(self, k: K, f: F[X1, X2, Y], /) -> F[X1, X2, Y]:
        """Divide a bivariate function by a scalar."""
        return self._sdiv(k, f)

    def lmul(self, f: F[X1, Y], g: F[X1, X2, Y], /) -> F[X1, X2, Y]:
        """Left-multiply a bivariate function by a univariate function."""
        return self._lmul(f, g)

    def rmul(self, f: F[X1, X2, Y], g: F[X2, Y], /) -> F[X1, X2, Y]:
        """Right-divide a bivariate function by a univariate function."""
        return self._rmul(f, g)


def bivariate_function_bimodule[X1, X2, Y, K](
    codomain: alg.ImplementsBimodule[Y, K, Y, Y],
    zero: Callable[[], F[X1, X2, Y]] | DEFAULT = DEFAULT,
    add: Callable[[F[X1, X2, Y], F[X1, X2, Y]], F[X1, X2, Y]]
    | DEFAULT = DEFAULT,
    sub: Callable[[F[X1, X2, Y], F[X1, X2, Y]], F[X1, X2, Y]]
    | DEFAULT = DEFAULT,
    neg: Callable[[F[X1, X2, Y]], F[X1, X2, Y]] | DEFAULT = DEFAULT,
    smul: Callable[[K, F[X1, X2, Y]], F[X1, X2, Y]] | DEFAULT = DEFAULT,
    sdiv: Callable[[K, F[X1, X2, Y]], F[X1, X2, Y]] | DEFAULT = DEFAULT,
    lmul: Callable[[F[X1, Y], F[X1, X2, Y]], F[X1, X2, Y]] | DEFAULT = DEFAULT,
    rmul: Callable[[F[X1, X2, Y], F[X2, Y]], F[X1, X2, Y]] | DEFAULT = DEFAULT,
) -> BivariateFunctionBimodule[X1, X2, Y, K]:
    """Build BivariateFunctionBimodule object."""
    return BivariateFunctionBimodule(
        codomain=codomain,
        _zero=(lift_constant(codomain.zero) if zero is DEFAULT else zero),
        _add=(lift_binary(codomain.add) if add is DEFAULT else add),
        _sub=(lift_binary(codomain.sub) if sub is DEFAULT else sub),
        _neg=(lift_unary(codomain.neg) if neg is DEFAULT else neg),
        _smul=(lift_left(codomain.smul) if smul is DEFAULT else smul),
        _sdiv=(lift_left(codomain.sdiv) if sdiv is DEFAULT else sdiv),
        _lmul=(
            lift_left_bivariate(codomain.lmul) if lmul is DEFAULT else lmul
        ),
        _rmul=(
            lift_right_bivariate(codomain.rmul) if rmul is DEFAULT else rmul
        ),
    )


@final
@dataclass(frozen=True, slots=True)
class BivariateFunctionDivBimodule[X1, X2, Y, K](
    alg.ImplementsDivBimodule[F[X1, X2, Y], K, F[X1, Y], F[X2, Y]]
):
    """Implement space of bivariate functions as a bimodule with division."""

    codomain: alg.ImplementsDivBimodule[Y, K, Y, Y]
    _zero: Callable[[], F[X1, X2, Y]]
    _add: Callable[[F[X1, X2, Y], F[X1, X2, Y]], F[X1, X2, Y]]
    _sub: Callable[[F[X1, X2, Y], F[X1, X2, Y]], F[X1, X2, Y]]
    _neg: Callable[[F[X1, X2, Y]], F[X1, X2, Y]]
    _smul: Callable[[K, F[X1, X2, Y]], F[X1, X2, Y]]
    _sdiv: Callable[[K, F[X1, X2, Y]], F[X1, X2, Y]]
    _lmul: Callable[[F[X1, Y], F[X1, X2, Y]], F[X1, X2, Y]]
    _ldiv: Callable[[F[X1, Y], F[X1, X2, Y]], F[X1, X2, Y]]
    _rmul: Callable[[F[X1, X2, Y], F[X2, Y]], F[X1, X2, Y]]
    _rdiv: Callable[[F[X1, X2, Y], F[X2, Y]], F[X1, X2, Y]]

    @property
    def scl(self) -> alg.ImplementsScalarField[K]:
        """Scalar field associated with BivariateFunctionDivBimodule object."""
        return self.codomain.scl

    def zero(self, /) -> F[X1, X2, Y]:
        """Return zero bivariate function."""
        return self._zero()

    def add(self, f: F[X1, X2, Y], g: F[X1, X2, Y], /) -> F[X1, X2, Y]:
        """Add two bivariate functions."""
        return self._add(f, g)

    def sub(self, f: F[X1, X2, Y], g: F[X1, X2, Y], /) -> F[X1, X2, Y]:
        """Subtract two bivariate functions."""
        return self._sub(f, g)

    def neg(self, f: F[X1, X2, Y], /) -> F[X1, X2, Y]:
        """Compute additive inverse (negation) of a bivariate function."""
        return self._neg(f)

    def smul(self, k: K, f: F[X1, X2, Y], /) -> F[X1, X2, Y]:
        """Multiply a bivariate function by a scalar."""
        return self._smul(k, f)

    def sdiv(self, k: K, f: F[X1, X2, Y], /) -> F[X1, X2, Y]:
        """Divide a bivariate function by a scalar."""
        return self._sdiv(k, f)

    def lmul(self, f: F[X1, Y], g: F[X1, X2, Y], /) -> F[X1, X2, Y]:
        """Left-multiply a bivariate function by a univariate function."""
        return self._lmul(f, g)

    def ldiv(self, f: F[X1, Y], g: F[X1, X2, Y], /) -> F[X1, X2, Y]:
        """Left-divide a bivariate function by a univariate function."""
        return self._ldiv(f, g)

    def rmul(self, f: F[X1, X2, Y], g: F[X2, Y], /) -> F[X1, X2, Y]:
        """Right-multiply a bivariate function by a univariate function."""
        return self._rmul(f, g)

    def rdiv(self, f: F[X1, X2, Y], g: F[X2, Y], /) -> F[X1, X2, Y]:
        """Right-divide a bivariate function by a univariate function."""
        return self._rdiv(f, g)


def bivariate_function_div_bimodule[X1, X2, Y, K](
    codomain: alg.ImplementsDivBimodule[Y, K, Y, Y],
    zero: Callable[[], F[X1, X2, Y]] | DEFAULT = DEFAULT,
    add: Callable[[F[X1, X2, Y], F[X1, X2, Y]], F[X1, X2, Y]]
    | DEFAULT = DEFAULT,
    sub: Callable[[F[X1, X2, Y], F[X1, X2, Y]], F[X1, X2, Y]]
    | DEFAULT = DEFAULT,
    neg: Callable[[F[X1, X2, Y]], F[X1, X2, Y]] | DEFAULT = DEFAULT,
    smul: Callable[[K, F[X1, X2, Y]], F[X1, X2, Y]] | DEFAULT = DEFAULT,
    sdiv: Callable[[K, F[X1, X2, Y]], F[X1, X2, Y]] | DEFAULT = DEFAULT,
    lmul: Callable[[F[X1, Y], F[X1, X2, Y]], F[X1, X2, Y]] | DEFAULT = DEFAULT,
    ldiv: Callable[[F[X1, Y], F[X1, X2, Y]], F[X1, X2, Y]] | DEFAULT = DEFAULT,
    rmul: Callable[[F[X1, X2, Y], F[X2, Y]], F[X1, X2, Y]] | DEFAULT = DEFAULT,
    rdiv: Callable[[F[X1, X2, Y], F[X2, Y]], F[X1, X2, Y]] | DEFAULT = DEFAULT,
) -> BivariateFunctionDivBimodule[X1, X2, Y, K]:
    """Build BivariateFunctionDivBimodule object."""
    return BivariateFunctionDivBimodule(
        codomain=codomain,
        _zero=(lift_constant(codomain.zero) if zero is DEFAULT else zero),
        _add=(lift_binary(codomain.add) if add is DEFAULT else add),
        _sub=(lift_binary(codomain.sub) if sub is DEFAULT else sub),
        _neg=(lift_unary(codomain.neg) if neg is DEFAULT else neg),
        _smul=(lift_left(codomain.smul) if smul is DEFAULT else smul),
        _sdiv=(lift_left(codomain.sdiv) if sdiv is DEFAULT else sdiv),
        _lmul=(
            lift_left_bivariate(codomain.lmul) if lmul is DEFAULT else lmul
        ),
        _ldiv=(
            lift_left_bivariate(codomain.ldiv) if ldiv is DEFAULT else ldiv
        ),
        _rmul=(
            lift_right_bivariate(codomain.rmul) if rmul is DEFAULT else rmul
        ),
        _rdiv=(
            lift_right_bivariate(codomain.rdiv) if rdiv is DEFAULT else rdiv
        ),
    )

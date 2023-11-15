"""Provides functions and protocols for abstract algebraic structures."""

from functools import partial
from itertools import accumulate, repeat
from typing import Callable, Iterator, Optional, Protocol, Self, TypeVar, \
        runtime_checkable

B = TypeVar('B')
B_contra = TypeVar('B_contra', contravariant=True)
C = TypeVar('C')
C_contra = TypeVar('C_contra', contravariant=True)
T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
T_contra = TypeVar('T_contra', contravariant=True)
S = TypeVar('S')
S_co = TypeVar('S_co', covariant=True)
S_contra = TypeVar('S_contra', contravariant=True)

F_co = TypeVar('F_co', covariant=True)
H_co = TypeVar('H_co', covariant=True)
F_contra = TypeVar('F_contra', contravariant=True)
G_contra = TypeVar('G_contra', contravariant=True)
H_contra = TypeVar('H_contra', contravariant=True)
F = TypeVar('F')
G = TypeVar('G')
H = TypeVar('H')
I = TypeVar('I')
U = TypeVar('U')
V = TypeVar('V')
V_co = TypeVar('V_co', covariant=True)

J_co = TypeVar('J_co', covariant=True)
K_contra = TypeVar('K_contra', contravariant=True)
K_co = TypeVar('K_co', covariant=True)
K = TypeVar('K', float, complex)

X = TypeVar('X')
Y = TypeVar('Y')
Z = TypeVar('Z')
Z_co = TypeVar('Z_co', covariant=True)
X_co = TypeVar('X_co', covariant=True)
Y_co = TypeVar('Y_co', covariant=True)
Y_contra = TypeVar('Y_contra', contravariant=True)


@runtime_checkable
class SupportsAdd(Protocol):
    """Represents types supporting addition."""
    def __add__(a: Self, b: Self) -> Self:
        ...


@runtime_checkable
class ImplementsAdd(Protocol[T]):
    """Represents types implementing addition."""
    def add(self, a: T, b: T) -> T:
        ...


@runtime_checkable
class Implements__Add__(Protocol[T]):
    """Represents types implementing magic method for addition."""
    def __add__(self, a: T, b: T) -> T:
        ...


class SupportsSub(Protocol):
    """Represents types supporting addition."""
    def __sub__(a: T, b: T) -> T:
        ...


@runtime_checkable
class ImplementsSub(Protocol[T]):
    """Represents types implementing subtraction."""
    def sub(self, a: T, b: T) -> T:
        ...


class SupportsMul(Protocol[K_co, T]):
    """Represents types supporting scalar/module multiplication."""
    def __mul__(a: K_co, b: T) -> T:
        ...


@runtime_checkable
class ImplementsMul(Protocol[K_contra, T]):
    """Represents types implementing scalar/module multiplication."""
    def mul(self, a: K_contra, b: T) -> T:
        ...


class SupportsMatmul(Protocol):
    """Represents types supporting algebraic multiplication.

    The intent of the SupportsMatMul protocol is to generalize matrix
    multiplication, which is the algebraic product for square matrices, to
    other algebras (e.g., function algebras). We use the matmul method in the
    protocol specification to allow types such as NumPy arrays to automatically
    implement it. In mathematical terms, a more descriptive name for this class
    could be SupportsAlgmul, but we use SupportsMatmul for consistency with
    with other Supports* classes defined in this module.

    """
    def __matmul__(a: T, b: T) -> T:
        ...


@runtime_checkable
class ImplementsAlgmul(Protocol[T]):
    """Represents types implementing algebraic multiplication."""
    def algmul(self, a: T, b: T) -> T:
        ...


@runtime_checkable
class ImplementsPower(Protocol[T, S_contra]):
    """Represents types implementing algebraic power (exponentiation)."""
    def power(self, a: T, b: S_contra) -> T:
        ...


@runtime_checkable
class ImplementsSqrt(Protocol[T]):
    """Represents types implementing square root."""
    def sqrt(self, a: T) -> T:
        ...


@runtime_checkable
class ImplementsStar(Protocol[T]):
    """Represents types implementing algebraic adjunction."""
    def star(self, a: T) -> T:
        ...


@runtime_checkable
class ImplementsUnit(Protocol[T_co]):
    """Represents types implementing algebraic unit."""
    def unit(self) -> T_co:
        ...


@runtime_checkable
class ImplementsInv(Protocol[T]):
    """Represents types implementing algebraic inversion."""
    def inv(self, a: T) -> T:
        ...


@runtime_checkable
class ImplementsAlgdiv(Protocol[T]):
    """Represents types implementing algebraic division.

    algdiv(a, b) = algmul(a, inv(b)).

    """
    def algdiv(self, a: T, b: T) -> T:
        ...


@runtime_checkable
class ImplementsAlgldiv(Protocol[T]):
    """Represents types implementing algebraic left division.

    algldiv(a, b) = algmul(inv(a), b).

    """
    def algldiv(self, a: T, b: T) -> T:
        ...


@runtime_checkable
class ImplementsRdiv(Protocol[T, K_contra]):
    """Represents types implementing right module division."""
    def algdiv(self, a: T, b: K_contra) -> T:
        ...


@runtime_checkable
class ImplementsVectorSpace(ImplementsAdd[T], ImplementsSub[T],
                            ImplementsMul[K_contra, T], Protocol[T, K_contra]):
    """Represents types implementing vector space operations."""
    ...


@runtime_checkable
class ImplementsAlgebra(ImplementsVectorSpace[T, K_contra], ImplementsAlgmul[T],
                        ImplementsSqrt[T], ImplementsPower[T, K_contra],
                        Protocol[T, K_contra]):
    """Represents types implementing algebra operations."""


@runtime_checkable
class ImplementsStarAlgebra(ImplementsAlgebra[T, K_contra], ImplementsStar[T],
                            Protocol[T, K_contra]):
    """Reperesents types implementing star algebra operations."""
    ...


@runtime_checkable
class ImplementsUnitalAlgebra(ImplementsAlgebra[T, K_contra],
                              ImplementsAlgdiv[T], ImplementsAlgldiv[T],
                              ImplementsInv[T], ImplementsUnit[T],
                              Protocol[T, K_contra]):
    """Reperesents types implementing unital algebra operations."""
    ...


@runtime_checkable
class ImplementsUnitalStarAlgebra(ImplementsUnitalAlgebra[T, K_contra],
                                  ImplementsStar[T], Protocol[T, K_contra]):
    """Reperesents types implementing unital star algebra operations."""
    ...


@runtime_checkable
class ImplementsPureState(Protocol[T_contra, S_contra]):
    """Represents types implementing pure algebra states."""
    def pure_state(self, a: T_contra) -> Callable[[S_contra], K]:
        ...


@runtime_checkable
class ImplementsInnerp(Protocol[T_contra, S_co]):
    """Represents types implementing inner product"""
    def innerp(self, a: T_contra, b: T_contra) -> S_co:
        ...


class SupportsCompose(Protocol[G_contra, F_co, H_co]):
    """Represents types supporting composition."""
    def cmp(f: F_co, g: G_contra) -> H_co:
        ...


@runtime_checkable
class ImplementsCompose(Protocol[G_contra, F_contra, H_co]):
    """Represents types implementing composition."""
    def compose(t, f: F_contra, g: G_contra) -> H_co:
        ...


@runtime_checkable
class ImplementsSynthesis(Protocol[B_contra, C_contra, V_co]):
    """Represents types implementing synthesis (linear combination) of vectors
    from a dictionary.

    B: Dictionary/basis type.
    C: Expansion coefficients type.
    V: Returned vector type.
    """

    def synthesis(self, b: B_contra, c: C_contra) -> V_co:
        ...


class Lift:
    pass


class SupportsLift(Protocol):
    """Represents types supporting a functorial lift operation."""
    def lift(self, base: object) -> Lift:
        ...


def compose_by(impl: ImplementsCompose[G, F, H], g: G) -> Callable[[F], H]:
    """Composition map."""
    def u(f: F) -> H:
        h = impl.compose(f, g)
        return h
    return u


def precompose_by(impl: ImplementsCompose[G, F, H], f: F) -> Callable[[G], H]:
    """Precomposition map."""
    def u(g: G) -> H:
        h = impl.compose(f, g)
        return h
    return u


def conjugate_by(t1: ImplementsCompose[H, U, F], u: U,
                 t2: ImplementsCompose[V, G, H], v: V) -> Callable[[G], F]:
    """Conjugation map."""
    def c(g: G) -> F:
        h = t2.compose(g, v)
        f = t1.compose(u, h)
        return f
    return c


def multiply_by(impl: ImplementsAlgmul[T], a: T) -> Callable[[T], T]:
    """Multiplication operator.

    :impl: Object that implements algebraic multiplication for a type T.
    :a: Object of type T.
    :returns: Multiplication function by a.

    """
    def m(b: T) -> T:
        c = impl.algmul(a, b)
        return c
    return m


def algdiv_by(impl: ImplementsAlgdiv[T], a: T) -> Callable[[T], T]:
    """Algebraic division operator.

    :impl: Object that implements algebraic division for a type T.
    :a: Object of type T.
    :returns: Division function by a.

    """
    def m(b: T) -> T:
        c = impl.algdiv(b, a)
        return c
    return m


def exponentiate_by(impl: ImplementsPower[T, S], a: S)\
        -> Callable[[T], T]:
    """Exponentiation map by a fixed exponent.

    :impl: Object that implements exponentiation (power) for types T and K.
    :a: Exponent.
    :returns: Exponentiation map.

    """
    def exp_a(b: T) -> T:
        c = impl.power(b, a)
        return c
    return exp_a


def synthesis_operator(impl: ImplementsSynthesis[B, C, V], b: B)\
        -> Callable[[C], V]:
    synth = partial(impl.synthesis, b)
    return synth


def id_map(x: T) -> T:
    """Identity map."""
    return x


def l2_innerp(impl: ImplementsStarAlgebra[T, S], mu: Callable[[T], S],
              a: T, b: T)\
                      -> S:
    """L2 inner product."""
    y = mu(impl.algmul(impl.star(a), b))
    return y


def make_l2_innerp(impl: ImplementsStarAlgebra[T, S], mu: Callable[[T], S])\
        -> Callable[[T, T], S]:
    """Make L2 inner product function."""
    ip = partial(l2_innerp, impl, mu)
    return ip


def riesz_dual(w: Callable[[U, U], S], u: U) -> Callable[[U], S]:
    """Riesz dual vector associated with bilinear form.

    :w: Bilinear form
    :u: Input vector.
    :returns: Riesz dual.

    """
    u_star = partial(w, u)
    return u_star


# TODO: Consider replacing with implementation from more_itertools
def iterate(impl: ImplementsCompose[F, F, F], f: F, initial: Optional[F] = None)\
        -> Iterator[F]:
    """Iterated function."""
    if initial is None:
        initial = f
    itf = accumulate(repeat(f), impl.compose, initial=initial)
    return itf


# @runtime_checkable
# class ImplementsPairwiseAlgmul(Protocol[T]):
#     """Represents types implementing pairwise algebraic multiplication."""
#     def pairwise_algmul(self, a: Indexable[T], b:Indexable[T])\
#             -> Indexable[T]:
#         ...


# @runtime_checkable
# class ImplementsPairwiseStarAlgebra(ImplementsPairwiseAdd[T],
#                                     ImplementsPairwiseAlgmul[T],
#                                     ImplementsStar[T], Protocol[T]):
#     """Reperesents types implementing pairwise star algebra operations."""
#     pass

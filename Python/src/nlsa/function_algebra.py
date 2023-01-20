"""Implements algebraic structures for function spaces."""

import numpy as np
from nlsa.abstract_algebra import \
        ImplementsAdd, SupportsAdd, \
        ImplementsSub, SupportsSub,\
        SupportsMul, \
        ImplementsAlgmul, SupportsMatmul, \
        Lift
from nlsa.utils import bind
from nptyping import Complex, Double, NDArray, Shape
from typing import Callable, Protocol, TypeVar

X = TypeVar('X')
X_co = TypeVar('X_co', covariant=True)
X2 = TypeVar('X2')
Y = TypeVar('Y')
Z = TypeVar('Z')
A = TypeVar('A', bound=SupportsAdd)
S = TypeVar('S', bound=SupportsSub)
K = TypeVar('K', bound=SupportsMul[object, object])
M = TypeVar('M', bound=SupportsMatmul)
M_co = TypeVar('M_co', bound=SupportsMatmul, covariant=True)
T = TypeVar('T')
T_contra = TypeVar('T_contra', contravariant=True)
U = TypeVar('U')
V = TypeVar('V')
V_co = TypeVar('V_co', covariant=True)
FT = Callable[[X], T]
FU = Callable[[X], U]
FV = Callable[[X], V]

# TypeVars for nptyping shape expressions
L = TypeVar('L')
N = TypeVar('N')

B_mat = TypeVar('B_mat', NDArray[Shape['*, L'], Complex],
                NDArray[Shape['*, L'], Double])
C_mat = TypeVar('C_mat', NDArray[Shape['L, N'], Double],
                NDArray[Shape['L, N'], Complex])
S_mat = TypeVar('S_mat', NDArray[Shape['*, N'], Double],
                NDArray[Shape['*, N'], Complex])

Ys = TypeVar('Ys', NDArray[Shape['*, N'], Double],
                   NDArray[Shape['*, N'], Complex])


# TODO: Consider renaming to lift_call?
def fmap(g: Callable[[T], U]) -> Callable[[FT[X, T]], FU[X, U]]:
    """Functorial map of functions from codomain to function space."""
    def h(f: FT[X, T]) -> FU[X, U]:
        return compose(g, f)
    return h


# # TODO: Consider calling this lifta2 as in Haskell?
def fmap2(a: Callable[[T, U], V]) -> Callable[[FT[X, T], FU[X, U]], FV[X, V]]:
    """Functorial map of functions of two arguments."""
    def b(f: FT[X, T], g: FU[X, U]) -> FV[X, V]:
        def h(x: X) -> V:
            y = a(f(x), g(x))
            return y
        return h
    return b


# TODO: This class should be deprecated in favor of an overloaded lift_from
# method using @overload decorators.
class C(ImplementsAdd[T], ImplementsSub[T], ImplementsAlgmul[T]):
    pass


def lift_from(base: C[T]) -> Lift:
    """Lift algebraic operations from codomain to function space."""
    lft = Lift()
    if isinstance(base, ImplementsAdd):
        def add(self, f: Callable[[X], T], g: Callable[[X], T]) \
                -> Callable[[X], T]:
            def h(x):
                y = base.add(f(x), g(x))
                return y
            return h
        bind(lft, add)
    if isinstance(base, ImplementsSub):
        def sub(self, f: Callable[[X], T], g: Callable[[X], T]) \
                -> Callable[[X], T]:
            def h(x):
                y = base.sub(f(x), g(x))
                return y
            return h
        bind(lft, sub)
    if isinstance(base, ImplementsAlgmul):
        def algmul(self, f: Callable[[X], T], g: Callable[[X], T]) \
                -> Callable[[X], T]:
            def h(x):
                y = base.algmul(f(x), g(x))
                return y
            return h
        bind(lft, algmul)

    return lft


class ImplementsIncl(Protocol[X_co, T_contra, V_co]):
    """Represents types implementing function inclusion."""
    def incl(self, f: Callable[[X_co], T_contra]) -> V_co:
        ...


def add(f: Callable[[X], A], g: Callable[[X], A]) -> Callable[[X], A]:
    """Implements function addition."""
    def h(x: X) -> A:
        y = f(x) + g(x)
        return y
    return h


def sub(f: Callable[[X], S], g: Callable[[X], S]) -> Callable[[X], S]:
    """Implements function subtraction."""
    def h(x: X) -> S:
        y = f(x) - g(x)
        return y
    return h


def mul(a: K, f: Callable[[X], object]) -> Callable[[X], object]:
    """Lifts scalar/left module multiplication to functions.

    The signature of this function should have been:

    mul(a: K, f: Callable[[X], M]) -> Callable[[X], M],

    where M is bound by SupportsMul[K, M]. However, this would require
    application of generic bounds which is currently not supported by the
    Python type system. Using Callable[[X], object] as the return type may
    prevent type checkers from inferring that functions returning
    Callable[[X], M] objects implement the SupportsMul protocol.
    """

    def h(x: X) -> object:
        y = a * f(x)
        return y

    return h


def algmul(f: Callable[[X], M], g: Callable[[X], M]) -> Callable[[X], M]:
    """Lifts algebraic multiplication to functions."""
    def h(x: X) -> M:
        y = f(x) @ g(x)
        return y
    return h


def compose(f: Callable[[Y], Z], g: Callable[[X], Y]) -> Callable[[X], Z]:
    """Implements function composition."""
    h: Callable[[X], Z] = lambda x: f(g(x))
    return h


def compose2(f: Callable[[Y], Z], g: Callable[[X, X2], Y]) \
        -> Callable[[X, X2], Z]:
    """Implements composition of function with two arguments.

    :f: Function of a single argument.
    :g: Function of two arguments.
    :returns: The composition of f with g.

    """
    h: Callable[[X, X2], Z] = lambda x, x2: f(g(x, x2))
    return h


def evaluate_at(x: X, f: Callable[[X], Y]) -> Y:
    """Pointwise evaluation functional."""
    return f(x)


def evaluate_at_vectorized(x: X, f: Callable[[X], Ys]) -> Ys:
    """Pointwise evaluation functional, vectorized over NumPy arrays."""
    y = f(x)
    if y.ndim == 2:
        y = y[:, np.newaxis, :]
    return y 


def synthesis(phi: Callable[[X], B_mat], c: C_mat) -> Callable[[X], S_mat]:
    """Performs synthesis of a linear combination of basis functions

    :phi: Array-valued function that returns the values of the basis functions.
    :c: Array of expansion coefficients.
    :returns: Synthesized function

    """
    def f(x: X) -> S_mat:
        y: S_mat = phi(x) @ np.atleast_2d(c)
        # y = np.einsum('...i,i', phi(x), c)
        return y
    return f

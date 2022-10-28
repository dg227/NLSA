"""Implements algebraic structures for function spaces."""

from nlsa.abstract_algebra import \
        ImplementsAdd, SupportsAdd, \
        ImplementsSub, SupportsSub,\
        ImplementsMul, SupportsMul, \
        ImplementsMatmul, SupportsMatmul, \
        Lift
from nlsa.utils import bind
from nptyping import Complex, Double, NDArray, Shape
from typing import Callable, TypeVar
import numpy as np

X = TypeVar("X")
Y = TypeVar("Y")
Z = TypeVar("Z")
A = TypeVar("A", bound=SupportsAdd)
S = TypeVar("S", bound=SupportsSub)
K = TypeVar("K", bound=SupportsMul[object, object])
M = TypeVar("M", bound=SupportsMatmul)
M_co = TypeVar("M_co", bound=SupportsMatmul, covariant=True)
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

# TypeVars for nptyping shape expressions
L = TypeVar("L")
N = TypeVar("N")

B_mat = TypeVar("B_mat", NDArray[Shape["*, L"], Complex],
                NDArray[Shape["*, L"], Double])
C_mat = TypeVar("C_mat", NDArray[Shape["L, N"], Double],
                NDArray[Shape["L, N"], Complex])
S_mat = TypeVar("S_mat", NDArray[Shape["*, N"], Double],
                NDArray[Shape["*, N"], Complex])

# TODO: This class should be deprecated in favor of an overloaded lift_from
# method using @overload decorators.
class C(ImplementsAdd[T], ImplementsSub[T], ImplementsMatmul[T]):
    pass


def lift_from(base: C[T]) -> Lift:
    """Lift algebraic operations from codomain to function space"""
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
    if isinstance(base, ImplementsMatmul):
        def matmul(self, f: Callable[[X], T], g: Callable[[X], T]) \
                -> Callable[[X], T]:
            def h(x):
                y = base.matmul(f(x), g(x))
                return y
            return h
        bind(lft, matmul)

    return lft


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


def matmul(f: Callable[[X], M], g: Callable[[X], M]) -> Callable[[X], M]:
    """Lifts algebraic multiplication to functions."""
    def h(x: X) -> M:
        y = f(x) @ g(x)
        return y
    return h


def compose(f: Callable[[Y], Z], g: Callable[[X], Y]) -> Callable[[X], Z]:
    """Implements function composition."""
    h: Callable[[X], Z] = lambda x: f(g(x))
    return h


# def synthesis(phi: Callable[[X], M], c: M) -> Callable[[X], M]:
def synthesis(phi: Callable[[X], B_mat], c: C_mat) -> Callable[[X], S_mat]:
    """Performs synthesis of a linear combination of basis functions

    :phi: Array-valued function that returns the values of the basis functions.
    :c: Array of expansion coefficients.
    :returns: Synthesized function

    """
    def f(x: X):
        y = phi(x) @ c
        # y = np.einsum('...i,i', phi(x), c)
        return y
    return f

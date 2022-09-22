"""Implements algebraic structures for function spaces"""

from nlsa.abstract_algebra import \
        ImplementsAdd, SupportsAdd, \
        ImplementsSub, SupportsSub,\
        ImplementsMul, SupportsMul, \
        ImplementsMatmul, SupportsMatmul, \
        Lift
from nlsa.utils import bind
from typing import Callable, TypeVar

X = TypeVar('X')
Y = TypeVar('Y')
Z = TypeVar('Z')
A = TypeVar('A', bound=SupportsAdd[object])
S = TypeVar('S', bound=SupportsSub[object])
K = TypeVar('K', bound=SupportsMul[object, object])
M = TypeVar('M', bound=SupportsMatmul[object])
T = TypeVar('T')

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


def add(f: Callable[[X], A], g: Callable[[X], A]) -> Callable[[X], object]:
    """Implements function addition.

    The return type of this function should have been Callable[[X], A] instead
    of Callable[[X], object]. However, this would require applying the generic
    bound SupportsAdd[A] to the type variable A, which is currently not
    supported by the Python type system. Using Callable[[X], object] as the
    return type may prevent type checkers from inferring that functions
    returning Callable[[X], A] objects implement the SupportsAdd protocol.
    """

    def h(x: X) -> object:
        y = f(x) + g(x)
        return y

    return h


def sub(f: Callable[[X], S], g: Callable[[X], S]) -> Callable[[X], object]:
    """Implements function subtraction.

    The return type of this function should have been Callable[[X], S] instead
    of Callable[[X], object]. However, this would require applying the generic
    bound SupportsSub[S] to the type variable S, which is currently not
    supported by the Python type system. Using Callable[[X], object] as the
    return type may prevent type checkers from inferring that functions
    returning Callable[[X], S] objects implement the SupportsSub protocol.
    """

    def h(x: X) -> object:
        y = f(x) - g(x)
        return y

    return h


def mul(a: K, f: Callable[[X], object]) -> Callable[[X], object]:
    """Lifts scalar/module multiplication to functions.

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


def matmul(f: Callable[[X], M], g: Callable[[X], M]) -> Callable[[X], object]:
    """Lifts algebraic multiplication to functions.

    The return type of this function should have been Callable[[X], M] instead
    of Callable[[X], object]. However, this would require applying the generic
    bound SupportsMatmul[M] to the type variable M, which is currently not
    supported by the Python type system. Using Callable[[X], object] as the
    return type may prevent type checkers from inferring that functions
    returning Callable[[X], M] objects implement the SupportsMatmul protocol.
    """

    def h(x: X) -> object:
        y = f(x) @ g(x)
        return y

    return h


def compose(f: Callable[[Y], Z], g: Callable[[X], Y]) -> Callable[[X], Z]:
    """Implements function composition"""
    h: Callable[[X], Z] = lambda x: f(g(x))
    return h

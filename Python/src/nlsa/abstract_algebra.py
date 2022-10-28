"""Provides functions and protocols for abstract algebraic structures."""

from typing import Callable, Protocol, runtime_checkable, TypeVar

T = TypeVar('T')
T_contra = TypeVar("T_contra", contravariant=True)
S = TypeVar('S')
S_contra = TypeVar("S_contra", contravariant=True)
F_contra = TypeVar('F_contra', contravariant=True)
G_contra = TypeVar('G_contra', contravariant=True)
H_co = TypeVar('H_co', covariant=True)
H_contra = TypeVar('H_contra', contravariant=True)
I = TypeVar("I")
J_co = TypeVar('J_co', covariant=True)
K_contra = TypeVar('K_contra', contravariant=True)
K_co = TypeVar('K_co', covariant=True)
K = TypeVar("K", float, complex)

X = TypeVar("X")
Y = TypeVar("Y")
Z = TypeVar("Z")
X_co = TypeVar("X_co", covariant=True)
Y_co = TypeVar("Y_co", covariant=True)


class SupportsAdd(Protocol):
    """Represents types supporting addition."""
    def __add__(a: T, b: T) -> T:
        ...


@runtime_checkable
class ImplementsAdd(Protocol[T]):
    """Represents types implementing addition."""
    def add(self, a: T, b: T) -> T:
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
class ImplementsMatmul(Protocol[T]):
    """Represents types implementing algebraic multiplication."""
    def matmul(self, a: T, b: T) -> T:
        ...


@runtime_checkable
class ImplementsPureState(Protocol[T_contra, S_contra]):
    """Represents types implementing pure algebra states."""
    def pure_state(self, a: T_contra, b: S_contra) -> Callable[[S_contra], K]:
        ...


class Arrow(Protocol[X_co, Y_co]):
    ...


class SupportsCompose(Protocol):
    """Represents types supporting composition."""
    def cmp(f: Arrow[Y, Z], g: Arrow[X, Y]) -> Arrow[X, Z]:
        ...


@runtime_checkable
class ImplementsCompose(Protocol):
    """Represents types implementing composition."""
    def compose(t, f: Arrow[Y, Z], g: Arrow[X, Y]) -> Arrow[X, Z]:
        ...


class Lift:
    pass


class SupportsLift(Protocol):
    """Represents types supporting a functorial lift operation."""
    def lift(self, base: object) -> Lift:
        ...


def compose_by(t: ImplementsCompose, g: Arrow[X, Y]) \
        -> Callable[[Arrow[Y, Z]], Arrow[X, Z]]:
    """Composition map."""
    def u(f: Arrow[Y, Z]) -> Arrow[X, Z]:
        h = t.compose(f, g)
        return h
    return u


def conjugate_by(t: ImplementsCompose, h: Arrow[Z, X], g: Arrow[X, Y]) \
                         -> Callable[[Arrow[Y, Z]], Arrow[X, X]]:
    """Conjugation map."""
    def u(f: Arrow[Y, Z]) -> Arrow[X, X]:
        i = t.compose(f, g)
        j = t.compose(h, i)
        return j
    return u


def multiply_by(t: ImplementsMatmul[T], a: T) -> Callable[[T], T]:
    """Multiplication operator.

    :t: Object that implements multiplication for a type T.
    :a: Object of type T.
    :returns: Multiplication function by a.

    """
    def m(b: T) -> T:
        c = t.matmul(a, b)
        return c
    return m


def id_map(x: T) -> T:
    """Identity map."""
    return x

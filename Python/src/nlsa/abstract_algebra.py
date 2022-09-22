"""Provides functions and protocols for abstract algebraic structures."""

from typing import Callable, Protocol, runtime_checkable, TypeVar

T = TypeVar('T')
F_contra = TypeVar('F_contra', contravariant=True)
G_contra = TypeVar('G_contra', contravariant=True)
K_contra = TypeVar('K_contra', contravariant=True)
K_co = TypeVar('K_co', covariant=True)
H_co = TypeVar('H_co', covariant=True)


class SupportsAdd(Protocol[T]):
    """Represents types supporting addition."""
    def __add__(a: T, b: T) -> T:
        ...


@runtime_checkable
class ImplementsAdd(Protocol[T]):
    """Represents types implementing addition."""
    def add(self, a: T, b: T) -> T:
        ...


class SupportsSub(Protocol[T]):
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


class SupportsMatmul(Protocol[T]):
    """Represents types supporting algebraic multiplication.

    The intent of the Supports MatMul protocol is to generalize matrix
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


class SupportsCompose(Protocol[G_contra, H_co]):
    """Represents types supporting composition."""
    def compose(self, other: G_contra) -> H_co:
        ...


@runtime_checkable
class ImplementsCompose(Protocol[F_contra, G_contra, H_co]):
    """Represents types implementing composition."""
    def compose(self, a: F_contra, b: G_contra) -> H_co:
        ...


class Lift:
    pass


class SupportsLift(Protocol):
    """Represents types supporting a functorial lift operation"""
    def lift(self, base: object) -> Lift:
        ...


def compose_by(t: ImplementsCompose[F_contra, G_contra, H_co], g: G_contra) \
        -> Callable[[F_contra], H_co]:
    """Composition map"""
    def u(f: F_contra) -> H_co:
        h = t.compose(f, g)
        return h

    return u

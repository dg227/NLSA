"""Provide definitions of types and protocols."""

import numpy as np
import jax
from collections.abc import Iterable, Sized
from numpy.typing import ArrayLike
from types import EllipsisType
from typing import (
    Optional,
    Protocol,
    Self,
    Sequence,
    SupportsIndex,
    TypeIs,
    overload,
    runtime_checkable,
)


@runtime_checkable
class SupportsRealComplex[R](Protocol):
    """Protocol for objects supporting complex arithmetic."""

    @property
    def real(self) -> R:
        """Real part of ArrayLike object."""
        ...

    @property
    def imag(self) -> R:
        """Imaginary part of ArrayLike object."""
        ...

    def conj(self: Self) -> Self:
        """Take conjugate of ArrayLike object."""
        ...


def is_array_like(obj: object) -> TypeIs[ArrayLike]:
    """Check if object is ArrayLike."""
    return isinstance(obj, (np.ndarray, jax.Array))


def is_real_complex_array_like(
    obj: object,
) -> TypeIs[SupportsRealComplex[ArrayLike]]:
    """Check if object is array-like with real/complex arithmetic support."""
    return is_array_like(obj) and isinstance(obj, SupportsRealComplex)


@runtime_checkable
class Subscriptable[KT, VT](Protocol):
    """Protocol for objects implementing subscription."""

    def __getitem__(self, key: KT) -> VT:
        """Implement subscription."""
        ...


type SliceItem = (
    slice[
        Optional[SupportsIndex],
        Optional[SupportsIndex],
        Optional[SupportsIndex],
    ]
    | list[SupportsIndex]
    | Sequence[SupportsIndex]
    | None
    | EllipsisType
)


@runtime_checkable
class Sliceable[VT](Protocol):
    """Protocol for objects implementing slicing."""

    @overload
    def __getitem__(self, i: SupportsIndex, /) -> VT: ...

    @overload
    def __getitem__(self, s: SliceItem, /) -> Self: ...

    @overload
    def __getitem__(
        self, s: tuple[SupportsIndex | SliceItem, ...], /
    ) -> Self: ...


def is_sliceable(obj: object) -> TypeIs[Sliceable[object]]:
    """Check that input is Sliceable."""
    return isinstance(obj, Sliceable)


@runtime_checkable
class IterableAndSubscriptable[KT, VT](
    Iterable[VT], Subscriptable[KT, VT], Protocol
):
    """Protocol for iterable and subscriptable objects."""

    pass


@runtime_checkable
class IterableAndSliceable[VT](Iterable[VT], Sliceable[VT], Protocol):
    """Protocol for iterable and slicable objects."""

    pass


def is_iterable_and_sliceable(
    obj: object,
) -> TypeIs[IterableAndSliceable[object]]:
    """Check that input is IterableAndSliceable."""
    return isinstance(obj, IterableAndSliceable)


@runtime_checkable
class SizedIterable[VT](Iterable[VT], Sized, Protocol):
    """Protocol for sized iterables."""

    pass


@runtime_checkable
class SizedIterableAndSubscriptable[KT, VT](
    SizedIterable[VT], Subscriptable[KT, VT], Protocol
):
    """Protocol for sized iterable and subscriptable objects."""

    pass


@runtime_checkable
class SizedIterableAndSliceable[VT](
    SizedIterable[VT], Sliceable[VT], Protocol
):
    """Protocol for sized iterable and slicable objects."""

    pass

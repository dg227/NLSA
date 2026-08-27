"""Provide common type definitions in the nlsa.jax module."""

import jax
import numpy.typing as npt
from collections.abc import Callable
from nlsa.typing import cast_like, is_array_like
from typing import Any, TypeIs, cast, overload


type Idx = int | jax.Array
type PyTree = (
    None
    | int
    | float
    | jax.Array
    | npt.NDArray[Any]
    | list[PyTree]
    | tuple[PyTree, ...]
    | dict[Any, PyTree]
)


def is_py_tree(obj: object) -> TypeIs[PyTree]:
    """Check if an object is a PyTree."""
    match obj:
        case list() | tuple():
            xs = cast(list[PyTree] | tuple[PyTree, ...], obj)
            return all(map(is_py_tree, xs))
        case dict():
            dct = cast(dict[Any, PyTree], obj)
            return all(map(is_py_tree, dct))
        case _:
            return is_array_like(obj)


@overload
def typestable_jit[S: PyTree](f: Callable[[], S]) -> Callable[[], S]: ...


@overload
def typestable_jit[S: PyTree, T1: PyTree](
    f: Callable[[T1], S],
) -> Callable[[T1], S]: ...


@overload
def typestable_jit[S: PyTree, T1: PyTree, T2: PyTree](
    f: Callable[[T1, T2], S],
) -> Callable[[T1, T2], S]: ...


@overload
def typestable_jit[S: PyTree, T1: PyTree, T2: PyTree, T3: PyTree](
    f: Callable[[T1, T2, T3], S],
) -> Callable[[T1, T2, T3], S]: ...


@overload
def typestable_jit[S: PyTree, T1: PyTree, T2: PyTree, T3: PyTree, T4: PyTree](
    f: Callable[[T1, T2, T3, T4], S],
) -> Callable[[T1, T2, T3, T4], S]: ...


@overload
def typestable_jit[
    S: PyTree,
    T1: PyTree,
    T2: PyTree,
    T3: PyTree,
    T4: PyTree,
    T5: PyTree,
](
    f: Callable[[T1, T2, T3, T4, T5], S],
) -> Callable[[T1, T2, T3, T4, T5], S]: ...


@overload
def typestable_jit[
    S: PyTree,
    T1: PyTree,
    T2: PyTree,
    T3: PyTree,
    T4: PyTree,
    T5: PyTree,
    T6: PyTree,
](
    f: Callable[[T1, T2, T3, T4, T5, T6], S],
) -> Callable[[T1, T2, T3, T4, T5, T6], S]: ...


def typestable_jit[
    S: PyTree,
    T1: PyTree,
    T2: PyTree,
    T3: PyTree,
    T4: PyTree,
    T5: PyTree,
    T6: PyTree,
](
    f: Callable[[], S]
    | Callable[[T1], S]
    | Callable[[T1, T2], S]
    | Callable[[T1, T2, T3], S]
    | Callable[[T1, T2, T3, T4], S]
    | Callable[[T1, T2, T3, T4, T5], S]
    | Callable[[T1, T2, T3, T4, T5, T6], S],
) -> (
    Callable[[], S]
    | Callable[[T1], S]
    | Callable[[T1, T2], S]
    | Callable[[T1, T2, T3], S]
    | Callable[[T1, T2, T3, T4], S]
    | Callable[[T1, T2, T3, T4, T5], S]
    | Callable[[T1, T2, T3, T4, T5, T6], S]
):
    """Typestable version of jax.jit."""
    return cast_like(f, jax.jit(f))

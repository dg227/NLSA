"""Provide common type definitions in the nlsa.jax module."""

from jax.typing import ArrayLike
from typing import Any


type PyTree = (
    ArrayLike
    | None
    | float
    | int
    | bool
    | list[PyTree]
    | tuple[PyTree, ...]
    | dict[Any, PyTree]
)

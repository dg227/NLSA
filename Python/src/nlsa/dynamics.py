"""Provides functions for dynamical systems computations."""
from typing import Callable, Iterator, TypeVar, Generator

T = TypeVar('T')

def orbit(x0: T, f: Callable[[T], T]) -> Generator[T, None, None]:
    x = x0
    while True:
        yield x
        x = f(x)

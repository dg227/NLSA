"""This module provides overloaded versions of certain algebraic operations,
such as algebraic multiplication for scalars.
"""

from typing import TypeVar

K = TypeVar('K', int, float, complex)

def algmul(a: K, b: K) -> K:
    """Algebraic multiplication for scalars."""
    c = a * b
    return c

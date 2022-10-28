"""This module provides overloaded versions of certain algebraic operations,
such as matrix multiplication for scalars.
"""

from typing import TypeVar

K = TypeVar('K', int, float, complex)

def matmul(a: K, b: K) -> K:
    """Algebraic multiplication for scalars."""
    c = a * b
    return c

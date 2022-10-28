"""Implements algebraic structures for spaces of column vectors, viewed as
abelian algebras with respect to entrywise vector multiplication."""

import numpy as np
from nptyping import Complex, Double, Int, NDArray, Shape
from typing import TypeVar

N = TypeVar("N")
K = TypeVar("K", complex, float, int)
V = TypeVar("V",
            NDArray[Shape["*, N"], Complex],
            NDArray[Shape["*, N"], Double],
            NDArray[Shape["*, N"], Int])


def add(u: V, v: V) -> V:
    """Implements vector addition"""
    w = u + v
    return w


def sub(u: V, v: V) -> V:
    """Implements vector subtraction"""
    w = u - v
    return w


def mul(a: K, v: V) -> V:
    """Implements scalar/left module multiplication for vectors"""
    w = np.multiply(a, v)
    return w


def matmul(u: V, v: V) -> V:
    """Implements algebraic multiplication as elementwise vector 
    multiplication.

    """
    w = u * v
    return w
 

compose = matmul

# TODO: Implement dual, linear map application

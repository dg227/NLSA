from typing import Any, TypeVar
from nptyping import NDArray, Shape

V = NDArray[Shape["N"], Any]
L = NDArray[Shape["M, N"], Any]
W = NDArray[Shape["M, 1"], Any]
M = TypeVar('M', V, L, W)

M_by_N = NDArray[Shape["M, N"], Any]
N_by_P = NDArray[Shape["N, P"], Any]
M_by_P = NDArray[Shape["M, P"], Any]


def add(u: M, v: M) -> M:
    """Implements matrix addition"""
    h: M = u + v
    return h


def compose(a: M_by_N, b: N_by_P) -> M_by_P:
    """Implements matrix composition (multiplication)"""
    c: M_by_P = a @ b
    return c

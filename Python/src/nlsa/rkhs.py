"""Implements various aspects of reproducing kernel Hilbert spaces (RKHSs)."""

import numpy as np
from nptyping import Complex, Double, NDArray, Shape
from typing import Callable, TypeVar

N = TypeVar("N")
V = NDArray[Shape["N"], Double]
R = NDArray[Shape["1, ..."], Double]
M = TypeVar("M",
            NDArray[Shape["N, ..."], Double],
            NDArray[Shape["N, ..."], Complex])


def energy(w: V) -> Callable[[M], V]:
    """Energy function associated with weights.

    :w: Vector of weights.
    :returns: Energy function

    """
    def engy(c: M) -> R:
        e: R = np.sum(np.abs(c) ** 2 * w[:, np.newaxis], axis=0)
        return e
    return engy

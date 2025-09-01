"""Provide coefficients for finite-difference_schemes."""
from typing import Literal


def central_1d(order: Literal[2, 4, 6, 8]) -> list[float]:
    """Return coefficients of 1D central finite-difference schemes."""
    match order:
        case 2:
            return [-1/2, 0.0, 1/2]
        case 4:
            return [1/12, -2/3, 0.0, 2/3, -1/12]
        case 6:
            return [-1/60, 3/20, -3/4, 0.0, 3/4, -3/20, 1/60]
        case 8:
            return [1/280, -4/105, 1/5, -4/5, 0.0, 4/5, -1/5, 4/105, -1/280]


def forward_1d(order: Literal[1, 2, 3, 4]) -> list[float]:
    """Return coefficients of 1D forward finite-difference schemes."""
    match order:
        case 1:
            return [-1.0, 1.0]
        case 2:
            return [-3/2, 2.0, -1/2]
        case 3:
            return [-11/6, 3.0, -3/2, 1/3]
        case 4:
            return [-25/12, 4.0, -3.0, 4/3, -1/4]


def backward_1d(order: Literal[1, 2, 3, 4]) -> list[float]:
    """Return coefficients of 1D backward finite-difference schemes."""
    match order:
        case 1:
            return [-1.0, 1.0]
        case 2:
            return [1/2, -2.0, 3/2]
        case 3:
            return [-1/3, 3/2, -3.0, 11/6]
        case 4:
            return [1/4, -4/3, 3.0, -4.0, 25/12]

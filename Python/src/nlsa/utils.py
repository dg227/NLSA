"""Provide various utility functions."""

import time
from collections.abc import Callable, Generator, Sequence
from functools import wraps
from typing import Literal


def fst[X](x: Sequence[X]) -> X:
    """Return the first element of a sequence."""
    return x[0]


def snd[X](x: Sequence[X]) -> X:
    """Return the second element of a sequence."""
    return x[1]


def pair[X, Y](x: X, y: Y, /) -> tuple[X, Y]:
    """Pack two objects into tuple."""
    z: tuple[X, Y] = (x, y)
    return z


def swap_args[X, Y, Z](f: Callable[[X, Y], Z], /) -> Callable[[Y, X], Z]:
    """Swap arguments of bivariate function."""
    def g(y: Y, x: X, /) -> Z:
        z: Z = f(x, y)
        return z
    return g


def decomp1d(n: int, n_batch: int, i_batch: int) -> tuple[int, int]:
    """Decompose a 1D domain of n samples into n_batch batches for near-uniform
    load. Modelled after MPI function MPE_DECOMP1D."""
    m = n // n_batch
    deficit = n % m
    j = i_batch*m + min(i_batch, deficit)
    if i_batch < deficit:
        m = m + 1
    k = j + m
    if k > n or i_batch == (n_batch - 1):
        k = n
    return j, k


def batched[X](seq: Sequence[X], n: int,
               mode: Literal['batch_size', 'batch_number'] = 'batch_size') \
        -> Generator[Sequence[X], None, None]:
    """Yield successive chunks from seq.

    Adopted from https://stackoverflow.com/questions/312443
    /how-do-i-split-a-list-into-equally-sized-chunks
    """
    match mode:
        case 'batch_size':
            for i in range(0, len(seq), n):
                yield seq[i:i + n]
        case 'batch_number':
            for j in range(0, n):
                i0, i1 = decomp1d(len(seq), n, j)
                yield seq[i0:i1]


def timeit[*Args, T](func: Callable[[*Args], T]) -> Callable[[*Args], T]:
    """Add basic timing wrapper to function."""
    @wraps(func)
    def timeit_wrapper(*args: *Args):
        start_time = time.perf_counter()
        result = func(*args)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"{func.__name__}: Took {total_time:.4f} seconds")
        return result
    return timeit_wrapper

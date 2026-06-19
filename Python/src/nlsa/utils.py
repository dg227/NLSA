"""Provide various utility functions."""

import inspect
import math
import time
from collections.abc import Callable, Generator, Sequence
from functools import wraps
from typing import Literal, TypeGuard


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
    """Decompose a 1D domain into batches for near-uniform load.

    Modelled after MPI function MPE_DECOMP1D.
    """
    m = n // n_batch
    deficit = n % m
    j = i_batch * m + min(i_batch, deficit)
    if i_batch < deficit:
        m = m + 1
    k = j + m
    if k > n or i_batch == (n_batch - 1):
        k = n
    return j, k


def get_closest_factors(n: int) -> tuple[int, int]:
    """Calculate factors of a positive integer with minimal difference."""
    factor1 = math.isqrt(n)
    while n % factor1 != 0:
        factor1 -= 1
    factor2 = n // factor1
    return factor1, factor2


def batched[X](
    seq: Sequence[X],
    n: int,
    mode: Literal["batch_size", "batch_number"] = "batch_size",
) -> Generator[Sequence[X], None, None]:
    """Yield successive chunks from seq.

    Adopted from https://stackoverflow.com/questions/312443
    /how-do-i-split-a-list-into-equally-sized-chunks
    """
    match mode:
        case "batch_size":
            for i in range(0, len(seq), n):
                yield seq[i : i + n]
        case "batch_number":
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


def get_integer_in_range(prompt: str, min_val: int, max_val: int) -> int:
    """Prompt and validate integer user input within a specified range.

    Args:
        prompt (str): The message displayed to the user.
        min_val (int): The minimum allowable integer value.
        max_val (int): The maximum allowable integer value.

    Returns:
        int: The valid integer entered by the user.

    """
    while True:
        try:
            user_input = int(input(prompt))
            if min_val <= user_input <= max_val:
                return user_input
            else:
                print(
                    "Error: Please enter a value "
                    f"between {min_val} and {max_val}."
                )
        except ValueError:
            print("Error: Invalid input. Please enter a whole number.")


def has_one_arg[D, X, Y](
    f: Callable[[D, X], Y] | Callable[[X], Y],
) -> TypeGuard[Callable[[X], Y]]:
    """Return True if the function takes exactly one argument."""
    sig = inspect.signature(f)
    return len(sig.parameters) == 1


def has_two_args[D, X, Y](
    f: Callable[[D, X, X], Y] | Callable[[X, X], Y],
) -> TypeGuard[Callable[[X, X], Y]]:
    """Return True if the function takes exactly 2 arguments."""
    sig = inspect.signature(f)
    return len(sig.parameters) == 2

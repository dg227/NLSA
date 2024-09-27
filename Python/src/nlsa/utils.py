import math
from typing import Callable, Generator, Literal, Sequence, TypeVar

X = TypeVar('X')
Y = TypeVar('Y')
Z = TypeVar('Z')


def bind(instance, func, as_name=None):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the
    instance as the first argument, i.e. "self".

    Copied from https://stackoverflow.com/questions/1015307
    /python-bind-an-unbound-method#comment8431145_1015405
    """

    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method


def swap_args(f: Callable[[X, Y], Z], /) -> Callable[[Y, X], Z]:
    """Swap arguments of bivariate function."""
    def g(y: Y, x: X, /) -> Z:
        z: Z = f(x, y)
        return z
    return g


def decomp1d(n: int, n_batch: int, i_batch) -> tuple[int, int]:
    """Decompose a 1D domain of n samples into n_batch batches for near-uniform
    load. Modelled after MPI function MPE_DECOMP1D."""
    m = n // n_batch
    deficit = n % m
    j = i_batch*m + min(i_batch, deficit)
    if i_batch < deficit:
        m = m + 1
    k = j + m
    if k > n or i_batch == (n_batch -1):
        k = n
    return j, k


def batched(seq: Sequence[X], n: int,
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
                yield seq[i0: i1]

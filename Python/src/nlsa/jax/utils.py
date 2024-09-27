import jax
import jax.numpy as jnp
from functools import partial
from jax import Array, vmap
from jax.lax import concatenate
from jax.typing import DTypeLike
from nlsa.utils import batched
from typing import Callable, Literal, Optional, TypeVar

V = Array  # vector
Vs = Array  # collection of vectors
K = DTypeLike  # scalar
Ks = Array  # collection of scalars
S = TypeVar('S')
T = TypeVar('T')
F = Callable[[S], T]  # alias for univariate function


def materialize_array(matvec: Callable[[Array], Array], shape: tuple[int],
                      dtype=None, holomorphic=False, jit=False):
    """Materialize the matrix A used in matvec(x) = Ax."""
    x = jnp.zeros(shape, dtype)
    if jit:
        fn = jax.jit(jax.jacfwd(matvec, holomorphic=holomorphic))
    else:
        fn = jax.jacfwd(matvec, holomorphic=holomorphic)

    return fn(x)


def make_batched(f: F[Vs, Vs], max_batch_size: int,
                 in_axis: Optional[Literal[0, 1]] = None,
                 out_axis: Optional[Literal[0, 1]] = None) \
        -> F[Vs, Vs]:
    """Map to a function that operates over batches."""
    if in_axis is None:
        in_axis = 0
    if out_axis is None:
        out_axis = in_axis

    def g(vs: Vs) -> Vs:
        m = vs.shape[in_axis]
        if m > max_batch_size:
            n_batch = -(m // -max_batch_size)  # ceiling division
            match in_axis:
                case 0:
                    vss = batched(vs, n_batch, mode='batch_number')
                case 1:
                    vss = map(jnp.transpose, batched(vs.T, n_batch,
                                                     mode='batch_number'))
            return concatenate([f(v_batch) for v_batch in vss],
                               dimension=out_axis)
        else:
            return f(vs)
    return g


def make_bbatched(f: F[Vs, Vs], max_batch_sizes: tuple[int, int]) -> F[Vs, Vs]:
    """Map to a function that operates over a 2D tile of batches."""
    g = make_batched(f, max_batch_size=max_batch_sizes[1], in_axis=0)

    def h(vs: Vs) -> Vs:
        m = vs.shape[0]
        n_batch = -(m // -max_batch_sizes[0])
        vss = batched(vs, n_batch, mode='batch_number')
        return concatenate([g(v_batch) for v_batch in vss], dimension=0)

    return h


def make_batched2(f: Callable[[Vs, Vs], Vs], max_batch_sizes: tuple[int, int],
                  in_axes: Optional[tuple[Literal[0, 1],
                                          Literal[0, 1]]] = None) \
        -> Callable[[Vs, Vs], Vs]:
    """Map to a bivariate function that operates over batches."""
    if in_axes is None:
        in_axes = (0, 0)

    def g(vs: Vs, ws: Vs) -> Vs:
        m = vs.shape[in_axes[0]]
        if m > max_batch_sizes[0]:
            n_batch = -(m // -max_batch_sizes[0])  # ceiling division
            match in_axes[0]:
                case 0:
                    vss = batched(vs, n_batch, mode='batch_number')
                case 1:
                    vss = map(jnp.transpose, batched(vs.T, n_batch,
                                                     mode='batch_number'))
            return concatenate([make_batched(partial(f, v_batch),
                                             max_batch_size=max_batch_sizes[1],
                                             in_axis=in_axes[1],
                                             out_axis=1)(ws)
                                for v_batch in vss], dimension=0)
        else:
            return make_batched(partial(f, vs),
                                max_batch_size=max_batch_sizes[1],
                                in_axis=in_axes[1],
                                out_axis=1)(ws)
    return g


def vmap2(f: Callable[[V, V], K]) -> Callable[[Vs, Vs], Ks]:
    """Vectorize bivariate function."""
    g = vmap(f, in_axes=(-1, None), out_axes=0)
    h = vmap(g, in_axes=(None, -1), out_axes=1)
    return h

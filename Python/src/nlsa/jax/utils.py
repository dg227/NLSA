# pyright: basic
"""Provide miscellaneous utility functions in JAX."""

import jax
import jax.numpy as jnp
import nlsa.abstract_algebra as alg
import nlsa.function_algebra as fun
import numpy as np
from functools import partial, wraps
from jax import Array, vmap
from jax.lax import concatenate
from jax.typing import DTypeLike
from nlsa.jax.typing import PyTree
from nlsa.utils import batched, snd
from typing import Callable, Literal, Optional

type V = Array  # vector
type Vs = Array  # collection of vectors
type K = Array  # scalar
type Ks = Array  # collection of scalars
type F[*Xs, Y] = Callable[[*Xs], Y]


def fst(x: V) -> V:
    """Return the first element of an array."""
    return x[0]


def make_vectorvalued[X](
    f: F[X, K], vectorvalued: bool = True, dtype: Optional[DTypeLike] = None
) -> F[X, K] | F[X, V]:
    """Make vector-valued function."""
    if vectorvalued:

        def g(x: X, /) -> V:
            y = jnp.empty(1, dtype=dtype)
            y = y.at[0].set(f(x))
            return y
    else:

        def g(x: X, /) -> K:
            return jnp.array(f(x), dtype=dtype)

    return g


def materialize_array(
    matvec: Callable[[Array], Array],
    shape: int | tuple[int],
    dtype=None,
    holomorphic=False,
    jit=False,
):
    """Materialize the matrix A used in matvec(x) = Ax."""
    x = jnp.zeros(shape, dtype)
    if jit:
        fn = jax.jit(jax.jacfwd(matvec, holomorphic=holomorphic))
    else:
        fn = jax.jacfwd(matvec, holomorphic=holomorphic)
    return fn(x)


def make_batched(
    f: F[Vs, Vs],
    max_batch_size: int,
    in_axis: Optional[Literal[0, 1]] = None,
    out_axis: Optional[Literal[0, 1]] = None,
) -> F[Vs, Vs]:
    """Map to a function that operates over batches."""
    if in_axis is None:
        in_axis = 0
    if out_axis is None:
        out_axis = in_axis

    def g(vs: Vs) -> Vs:
        m = vs.shape[in_axis]
        if m > max_batch_size:
            num_batches = -(m // -max_batch_size)  # ceiling division
            match in_axis:
                case 0:
                    vss = batched(vs, num_batches, mode="batch_number")
                case 1:
                    vss = map(
                        jnp.transpose,
                        batched(vs.T, num_batches, mode="batch_number"),
                    )
            return concatenate(
                [f(v_batch) for v_batch in vss], dimension=out_axis
            )
        else:
            return f(vs)

    return g


def make_bbatched(f: F[Vs, Vs], max_batch_sizes: tuple[int, int]) -> F[Vs, Vs]:
    """Map to a function that operates over a 2D tile of batches."""
    g = make_batched(f, max_batch_size=max_batch_sizes[1], in_axis=0)

    def h(vs: Vs) -> Vs:
        m = vs.shape[0]
        num_batches = -(m // -max_batch_sizes[0])
        vss = batched(vs, num_batches, mode="batch_number")
        return concatenate([g(v_batch) for v_batch in vss], dimension=0)

    return h


def make_batched2(
    f: Callable[[Vs, Vs], Vs],
    max_batch_sizes: tuple[int, int],
    in_axes: Optional[tuple[Literal[0, 1], Literal[0, 1]]] = None,
) -> Callable[[Vs, Vs], Vs]:
    """Map to a bivariate function that operates over batches."""
    if in_axes is None:
        in_axes = (0, 0)

    def g(vs: Vs, ws: Vs) -> Vs:
        m = vs.shape[in_axes[0]]
        if m > max_batch_sizes[0]:
            num_batches = -(m // -max_batch_sizes[0])  # ceiling division
            match in_axes[0]:
                case 0:
                    vss = batched(vs, num_batches, mode="batch_number")
                case 1:
                    vss = map(
                        jnp.transpose,
                        batched(vs.T, num_batches, mode="batch_number"),
                    )
            return concatenate(
                [
                    make_batched(
                        partial(f, v_batch),
                        max_batch_size=max_batch_sizes[1],
                        in_axis=in_axes[1],
                        out_axis=1,
                    )(ws)
                    for v_batch in vss
                ],
                dimension=0,
            )
        else:
            return make_batched(
                partial(f, vs),
                max_batch_size=max_batch_sizes[1],
                in_axis=in_axes[1],
                out_axis=1,
            )(ws)

    return g


def scan_map(f: Callable[[Array], Array]) -> Callable[[Array], Array]:
    """Transform function for sequential execution using jax.lax.scan."""

    def scan_body(i: int, x: Array) -> tuple[int, Array]:
        return i, f(x)

    def g(xs: Array) -> Array:
        _, ys = jax.lax.scan(scan_body, 1, xs)
        return ys

    return g


def batch_map(
    f: Callable[[Array], Array],
    in_axis: int = 0,
    out_axis: int = 0,
    batch_size: Optional[int] = None,
) -> Callable[[Array], Array]:
    """Transform function for batched execution."""
    if batch_size is not None:
        if False:
            g = scan_map(f)
        else:
            g = partial(jax.lax.map, jax.checkpoint(f), batch_size=batch_size)
        if in_axis != 0:
            move_in_axis = partial(jnp.moveaxis, source=in_axis, destination=0)
            g: Callable[[Array], Array] = fun.compose(g, move_in_axis)
        if out_axis != 0:
            move_out_axis = partial(
                jnp.moveaxis, source=0, destination=out_axis
            )
            g = fun.compose(move_out_axis, g)
    else:
        g = vmap(f, in_axes=in_axis, out_axes=out_axis)
    return g


def curried_batch_map[P: PyTree](
    f: Callable[[P, Array], Array],
    in_axis: int = 0,
    out_axis: int = 0,
    batch_size: Optional[int] = None,
) -> Callable[[PyTree, Array], Array]:
    """Transform function for batched execution -- curried version."""

    def g(p: P, xs: Array) -> Array:
        fp = batch_map(
            partial(f, p),
            in_axis=in_axis,
            out_axis=out_axis,
            batch_size=batch_size,
        )
        return fp(xs)

    return g


def batch_map_bivariate(
    f: Callable[[Array, Array], Array],
    in_axis: int = 0,
    out_axis: int = 0,
    batch_size: Optional[int] = None,
) -> Callable[[Array, Array], Array]:
    """Transform bivariate function for batched execution."""
    if batch_size is not None:

        def f_tuple(xss: tuple[Array, Array], /) -> Array:
            return f(*xss)

        g_tuple = partial(jax.lax.map, f_tuple, batch_size=batch_size)
        if in_axis != 0:
            move_in_axis = partial(jnp.moveaxis, source=in_axis, destination=0)
            move_in_axes = partial(map, move_in_axis)
            g_tuple_in: Callable[[tuple[Array, Array]], Array] = fun.compose(
                g_tuple, move_in_axes
            )
        else:
            g_tuple_in = g_tuple

        def g_in(x: Array, y: Array) -> Array:
            return g_tuple_in((x, y))

        if out_axis != 0:
            move_out_axis = partial(
                jnp.moveaxis, source=0, destination=out_axis
            )
            g_in_out = fun.compose(move_out_axis, g_in)
        else:
            g_in_out = g_in
    else:
        g_in_out = vmap(f, in_axes=in_axis, out_axes=out_axis)
    return g_in_out


def vmap_2d(
    f: Callable[[Array, Array], Array],
    in_axes: tuple[int, int] = (0, 0),
    out_axes: tuple[int, int] = (0, 1),
) -> Callable[[Array, Array], Array]:
    """Vectorize bivariate function along two axes."""
    vmap_x = partial(
        vmap,
        in_axes=(in_axes[0], None),
        out_axes=0,
    )
    vmap_y = partial(
        vmap,
        in_axes=(None, in_axes[1]),
        out_axes=0,
    )
    g = vmap_x(vmap_y(f))
    if out_axes[0] != 0 or out_axes[1] != 1:
        move_out_axis = partial(
            jnp.moveaxis, source=(0, 1), destination=out_axes
        )
        g_out = fun.compose(move_out_axis, g)
    else:
        g_out = g
    return g_out


def _batch_map_2d(
    f: Callable[[Array, Array], Array],
    in_axes: tuple[int, int] = (0, 0),
    out_axes: tuple[int, int] = (0, 1),
    batch_sizes: Optional[int] | tuple[Optional[int], Optional[int]] = None,
) -> Callable[[Array, Array], Array]:
    """Transform bivariate function for batched execution along two axes."""
    match batch_sizes:
        case int():
            _batch_sizes = (batch_sizes, batch_sizes)
        case (_, _):
            _batch_sizes = batch_sizes
    batch_map_x = partial(
        batch_map,
        in_axis=in_axes[0],
        out_axis=0,
        batch_size=_batch_sizes[0],
    )
    batch_map_y = partial(
        batch_map,
        in_axis=in_axes[1],
        out_axis=0,
        batch_size=_batch_sizes[1],
    )

    def g(x: Array, y: Array) -> Array:
        def fx(_x: Array) -> Array:
            return batch_map_y(partial(f, _x))(y)

        return batch_map_x(fx)(x)

    if out_axes[0] != 0 or out_axes[1] != 1:
        moveaxis = partial(jnp.moveaxis, source=(0, 1), destination=out_axes)
        g_out = fun.compose(moveaxis, g)
    else:
        g_out = g
    return g_out


def batch_map_2d(
    f: Callable[[Array, Array], Array],
    in_axes: tuple[int, int] = (0, 0),
    out_axes: tuple[int, int] = (0, 1),
    batch_sizes: Optional[int] | tuple[Optional[int], Optional[int]] = None,
) -> Callable[[Array, Array], Array]:
    """Transform bivariate function for vectorized or batched execution."""
    match batch_sizes:
        case None | (None, None):
            return vmap_2d(f, in_axes=in_axes, out_axes=out_axes)
        case _:
            return _batch_map_2d(
                f, in_axes=in_axes, out_axes=out_axes, batch_sizes=batch_sizes
            )


def numpyit[D: DTypeLike](
    f: Callable[[Array], Array],
    to_jax: Optional[Callable[[np.ndarray], Array]] = None,
    to_numpy: Optional[Callable[[Array], np.ndarray]] = None,
    dtype: Optional[D] = None,
    copy: Optional[bool] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Map JAX computation to NumPy."""

    def _to_jax(a: np.ndarray) -> Array:
        return jnp.asarray(a, dtype=dtype)

    def _to_numpy(a: Array) -> np.ndarray:
        return np.asarray(a, copy=copy)

    if to_jax is None:
        to_jax = _to_jax
    if to_numpy is None:
        to_numpy = _to_numpy

    conj: Callable[[F[Array, Array]], F[np.ndarray, np.ndarray]] = (
        alg.conjugate_by(fun, to_numpy, fun, to_jax)
    )

    @wraps(f)
    def f_wrapped(a: np.ndarray) -> np.ndarray:
        return conj(f)(a)

    return f_wrapped

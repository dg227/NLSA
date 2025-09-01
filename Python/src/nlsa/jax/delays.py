# pyright: basic

"""Implement delay-coordinate maps in JAX."""

import jax
import jax.numpy as jnp
import nlsa.finite_differences as fd
import nlsa.jax.dynamics as dyn
from collections.abc import Callable
from functools import partial
from jax import Array
from jax.scipy.signal import convolve
from jax.typing import ArrayLike
from typing import Literal, TypeGuard, overload

type A = Array
type V = Array
type Vs = Array
type Vd = Array
type X = Array
type Xd = Array
type Xs = Array
type Y = Array
type F[*Xs, Y] = Callable[[*Xs], Y]


def roll_delay_eval_at(xs: Xs, /, num_delays: int = 0, delay_step: int = 1) \
        -> Callable[[F[Xd, Y]], A]:
    """Make vectorized evaluation functional with delays."""
    num_delay_samples = xs.shape[0] - num_delays*delay_step

    def ev(f: F[Xd, Y]) -> A:
        @jax.jit
        @jax.vmap
        def evf(i: int) -> Y:
            xs_shift = jnp.roll(xs, -i, axis=0)
            xd = xs_shift[:(num_delays*delay_step + 1):delay_step]
            return f(jnp.hstack(xd))
            # return f(xd.ravel())
        return evf(jnp.arange(num_delay_samples))
    return ev


def _delay_eval_at(xs: Xs, /, num_delays: int = 0, delay_step: int = 1,
                   jit: bool = False) -> Callable[[F[X, Y]], A]:
    """Make vectorized evaluation functional with delays."""
    num_delay_samples = xs.shape[0] - num_delays*delay_step
    inds = jnp.arange(0, num_delays*delay_step + 1, delay_step)

    def ev(f: F[X, Y]) -> A:
        @jax.vmap
        def evf(i: ArrayLike) -> Y:
            return f(jnp.hstack(xs[inds + i]))

        if jit:
            evf = jax.jit(evf)
        return evf(jnp.arange(num_delay_samples))
    return ev


def _delay_eval_at_tuple(xss: tuple[Xs, Xs], /, num_delays: int = 0,
                         delay_step: int = 1, jit: bool = False) \
        -> Callable[[F[X, X, Y]], A]:
    """Make evaluation functional with delays for Array tuples."""
    num_delay_samples = xss[0].shape[0] - num_delays*delay_step
    assert all(xs.shape[0] - num_delays*delay_step == num_delay_samples
               for xs in xss)
    inds = jnp.arange(0, num_delays*delay_step + 1, delay_step)

    def ev(f: F[X, X, Y]) -> A:
        @jax.vmap
        def evf(i: ArrayLike) -> Y:
            def go(xs: Xs) -> Xd:
                return jnp.hstack(xs[inds + i])
            xds = map(go, xss)
            return f(*xds)

        if jit:
            evf = jax.jit(evf)
        return evf(jnp.arange(num_delay_samples))
    return ev


@overload
def delay_eval_at(xs: Xs, /, num_delays: int = 0, delay_step: int = 1,
                  jit: bool = False) -> Callable[[F[X, Y]], A]:
    ...


@overload
def delay_eval_at(xss: tuple[Xs, Xs], /, num_delays: int = 0,
                  delay_step: int = 1, jit: bool = False) \
        -> Callable[[F[X, X, Y]], A]:
    ...


def delay_eval_at(xss: Xs | tuple[Xs, Xs], /, num_delays: int = 0,
                  delay_step: int = 1, jit: bool = False) \
            -> Callable[[F[X, Y]], A] | Callable[[F[X, X, Y]], A]:
    """Make evaluation functional with delays for Arrays or Array tuples."""
    if isinstance(xss, Array):
        ev = _delay_eval_at(xss, num_delays=num_delays, delay_step=delay_step,
                            jit=jit)
    else:
        ev = _delay_eval_at_tuple(xss, num_delays=num_delays,
                                  delay_step=delay_step, jit=jit)
    return ev


def _batch_delay_eval_at(xs: Xs, /, batch_size: int, num_delays: int = 0,
                         delay_step: int = 1) -> Callable[[F[X, Y]], A]:
    """Make batched evaluation functional with delays for Arrays."""
    num_delay_samples = xs.shape[0] - num_delays*delay_step
    assert num_delay_samples % batch_size == 0, \
        "Number of delay samples must be divisible by the batch size."
    delay_batch_size = batch_size + num_delays*delay_step
    num_batches = num_delay_samples // batch_size
    eval_at = partial(_delay_eval_at, num_delays=num_delays,
                      delay_step=delay_step)
    inds = jnp.arange(delay_batch_size)

    def ev(f: F[X, Y]) -> A:
        def evf(i: ArrayLike) -> Y:
            xs_i = jnp.take(xs, inds + i*batch_size, axis=0)
            return eval_at(xs_i)(f)
        ys = jax.lax.map(evf, jnp.arange(num_batches))
        return ys.reshape((num_delay_samples,) + ys.shape[2:])
    return ev


def _batch_delay_eval_at_tuple(xss: tuple[Xs, Xs], /, batch_size: int,
                               num_delays: int = 0, delay_step: int = 1) \
            -> Callable[[F[X, X, Y]], A]:
    """Make batched evaluation functional with delays for Array tuples."""
    num_delay_samples = xss[0].shape[0] - num_delays*delay_step
    assert num_delay_samples % batch_size == 0, \
        "Number of delay samples must be divisible by the batch size."
    delay_batch_size = batch_size + num_delays*delay_step
    num_batches = num_delay_samples // batch_size
    eval_at = partial(_delay_eval_at_tuple, num_delays=num_delays,
                      delay_step=delay_step)
    inds = jnp.arange(delay_batch_size)

    def ev(f: F[X, X, Y]) -> A:
        def evf(i: ArrayLike) -> Y:
            def go(xs: Xs) -> Xs:
                return jnp.take(xs, inds + i*batch_size, axis=0)
            xss_i = (go(xss[0]), go(xss[1]))
            return eval_at(xss_i)(f)
        ys = jax.lax.map(evf, jnp.arange(num_batches))
        return ys.reshape((num_delay_samples,) + ys.shape[2:])
    return ev


@overload
def batch_delay_eval_at(xs: Xs, /, batch_size: int, num_delays: int = 0,
                        delay_step: int = 1) -> Callable[[F[X, Y]], A]:
    ...


@overload
def batch_delay_eval_at(xss: tuple[Xs, Xs], /, batch_size: int,
                        num_delays: int = 0, delay_step: int = 1) \
        -> Callable[[F[X, X, Y]], A]:
    ...


def batch_delay_eval_at(xss: Xs | tuple[Xs, Xs], /, batch_size: int,
                        num_delays: int = 0, delay_step: int = 1) \
            -> Callable[[F[X, Y]], A] | Callable[[F[X, X, Y]], A]:
    """Make batched evaluation functional with delays for Arrays or Array
    tuples.
    """
    if isinstance(xss, Array):
        ev = _batch_delay_eval_at(xss, batch_size, num_delays=num_delays,
                                  delay_step=delay_step)
    else:
        ev = _batch_delay_eval_at_tuple(xss, batch_size, num_delays=num_delays,
                                        delay_step=delay_step)
    return ev


# def delay_eval2_at(xs: Xs, vs: Vs, /, num_delays: int = 0,
#                    delay_step: int = 1) -> Callable[[F[Xd, Vd, Y]], A]:
#     """Make vectorized bivariate evaluation functional with delays."""
#     num_delay_samples = xs.shape[0] - num_delays*delay_step
#     inds = jnp.arange(0, num_delays*delay_step + 1, delay_step)

#     def ev(f: F[Xd, Vd, Y]) -> A:
#         @jax.vmap
#         def evf(i: ArrayLike) -> Y:
#             xd = jnp.take(xs, inds + i, axis=0)
#             vd = jnp.take(vs, inds + i, axis=0)
#             # xs_shift = jnp.roll(xs, -i, axis=0)
#             # xd = xs_shift[:(num_delays*delay_step + 1):delay_step]
#             # vs_shift = jnp.roll(vs, -i, axis=0)
#             # vd = vs_shift[:(num_delays*delay_step + 1):delay_step]
#             # return f(xd.ravel(), vd.ravel())
#             return f(jnp.hstack(xd), jnp.hstack(vd))
#         return evf(jnp.arange(num_delay_samples))
#     return ev


# def batch_delay_eval2_at(xs: Xs, vs: Vs, /, batch_size: int,
#                          num_delays: int = 0, delay_step: int = 1) \
#         -> Callable[[F[Xd, Vd, Y]], A]:
#     """Make batched bivariate evaluation functional with delays."""
#     num_delay_samples = xs.shape[0] - num_delays*delay_step
#     assert num_delay_samples % batch_size == 0, \
#         "Number of delay samples must be divisible by the batch size."
#     delay_batch_size = batch_size + num_delays*delay_step
#     num_batches = num_delay_samples // batch_size
#     eval_at = partial(delay_eval2_at, num_delays=num_delays,
#                       delay_step=delay_step)
#     inds = jnp.arange(delay_batch_size)

#     def ev(f: F[Xd, Vd, Y]) -> A:
#         def evf(i: int) -> Y:
#             xs_i = jnp.take(xs, inds + i*batch_size, axis=0)
#             vs_i = jnp.take(vs, inds + i*batch_size, axis=0)
#             return eval_at(xs_i, vs_i)(f)

#         ys = jax.lax.map(evf, jnp.arange(num_batches))
#         return ys.reshape((num_delay_samples, -1))
#     return ev


def make_central_fd_operator(order: Literal[2, 4, 6, 8] = 2, dt: float = 1.0,
                             extrap: bool = True) -> F[V, V]:
    """Make central finite-difference operator on vectors."""
    w = jnp.flip(jnp.array(fd.central_1d(order)) / dt)
    if extrap is True:
        query_before = jnp.arange(order // 2)
        test_before = jnp.arange(order // 2, order + 1)
        test_after = jnp.arange(order//2 + 1)
        query_after = jnp.arange(order//2 + 1, order + 1)

        def fd_op(v: V) -> V:
            dv = convolve(v, w, mode="same")
            dv_before = jnp.interp(query_before, test_before,
                                   dv[(order // 2):(order + 1)],
                                   left="extrapolate")
            dv_after = jnp.interp(query_after, test_after,
                                  dv[-(order + 1):-(order // 2)],
                                  right="extrapolate")
            dv = dv.at[:(order // 2)].set(dv_before)
            dv = dv.at[-(order // 2):].set(dv_after)
            return dv
    else:
        def fd_op(v: V) -> V:
            return convolve(v, w, mode="same")
    return fd_op


def make_forward_fd_operator(order: Literal[1, 2, 3, 4] = 1, dt: float = 1.0,
                             extrap: bool = True) -> F[V, V]:
    """Make forward finite-difference operator on vectors."""
    w = jnp.flip(jnp.array(fd.backward_1d(order)) / dt)
    if extrap is True:
        test_after = jnp.arange(order + 1)
        query_after = jnp.arange(order + 1, 2*order + 1)

        def fd_op(v: V) -> V:
            dv = convolve(v, w, mode="same")
            dv_after = jnp.interp(query_after, test_after,
                                  dv[-2*(order + 1):-(order + 1)],
                                  right="extrapolate")
            dv = dv.at[-order:].set(dv_after)
            return dv
    else:
        def fd_op(v: V) -> V:
            return convolve(v, w, mode="same")
    return fd_op


def make_backward_fd_operator(order: Literal[1, 2, 3, 4] = 1, dt: float = 1.0,
                              extrap: bool = True) -> F[V, V]:
    """Make backward finite-difference operator on vectors."""
    w = jnp.flip(jnp.array(fd.backward_1d(order)) / dt)
    if extrap is True:
        query_before = jnp.arange(order)
        test_before = jnp.arange(order, 2*order + 1)

        def fd_op(v: V) -> V:
            dv = convolve(v, w, mode="same")
            dv_before = jnp.interp(query_before, test_before,
                                   dv[order:(2*order + 1)], left="extrapolate")
            dv = dv.at[:order].set(dv_before)
            return dv
    else:
        def fd_op(v: V) -> V:
            return convolve(v, w, mode="same")
    return fd_op


def is_valid_central_fd_order(order: int) -> TypeGuard[Literal[2, 4, 6, 8]]:
    """Check for valid order of central finite-difference approximation."""
    return order in [2, 4, 6, 8]


def is_valid_forward_fd_order(order: int) -> TypeGuard[Literal[1, 2, 3, 4]]:
    """Check for valid order of forward finite-difference approximation."""
    return order in [1, 2, 3, 4]


def is_valid_backward_fd_order(order: int) -> TypeGuard[Literal[1, 2, 3, 4]]:
    """Check for valid order of backward finite-difference approximation."""
    return order in [1, 2, 3, 4]


def make_fd_operator(order: int, dt: float = 1.0, extrap: bool = True,
                     mode: Literal["central", "forward",
                                   "backward"] = "central") -> F[V, V]:
    """Make finite-difference operator on vectors."""
    match mode:
        case "central":
            if is_valid_central_fd_order(order):
                fd_op = make_central_fd_operator(order=order, dt=dt,
                                                 extrap=extrap)
        case "forward":
            if is_valid_forward_fd_order(order):
                fd_op = make_forward_fd_operator(order=order, dt=dt,
                                                 extrap=extrap)
        case "backward":
            if is_valid_backward_fd_order(order):
                fd_op = make_backward_fd_operator(order=order, dt=dt,
                                                  extrap=extrap)
    return fd_op


def make_circ_shift_operator(step: int) -> F[V, V]:
    """Make circular shift operator on vectors."""
    return partial(jnp.roll, shift=step, axis=0)


def make_left_shift_operator(step: int, pad: ArrayLike = 0) -> F[V, V]:
    """Make left shift operator on vectors."""
    def shift_op(v: V) -> V:
        w = jnp.roll(v, shift=-step, axis=0)
        w = w.at[-step:].set(pad)
        return w
    return shift_op


def make_right_shift_operator(step: int, pad: ArrayLike = 0) -> F[V, V]:
    """Make left shift operator on vectors."""
    def shift_op(v: V) -> V:
        w = jnp.roll(v, shift=step, axis=0)
        w = w.at[:step].set(pad)
        return w
    return shift_op


def hankel(xs: Xs, /, num_delays: int = 0, delay_step: int = 1,
           flatten: bool = False) -> Xd:
    """Compute delay matrix (Hankel matrix)."""
    num_delay_samples = xs.shape[0] - num_delays*delay_step
    shift_op = make_left_shift_operator(step=delay_step)
    delay_embed = dyn.make_fin_orbit(shift_op, num_steps=num_delays + 1)
    xds = jnp.swapaxes(delay_embed(xs), 0, 1)[:num_delay_samples, :]
    if flatten:
        xds = xds.reshape((num_delay_samples, -1))
    return xds


def make_laplace_transform(num_quad: int, dt: float = 1, z: float = 1) \
        -> F[V, V]:
    """Make Laplace transform operator based on trapezoidal rule"""
    w = jnp.flip(jnp.array(jnp.exp(-z * dt * jnp.arange(num_quad))))

    def lapl(v: V) -> V:
        vw = convolve(v, w, mode="valid")
        return (vw[:-1] + vw[1:]) * dt / 2

    return lapl

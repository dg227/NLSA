"""Implement vector algebra operations for JAX arrays."""

import jax
import jax.numpy as jnp
import math
import nlsa.abstract_algebra as alg
import nlsa.function_algebra as fun
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from jax import Array, vmap
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding
from jax.scipy.signal import convolve
from jax.typing import DTypeLike
from nlsa.jax.scalars import ScalarField
from nlsa.jax.sharding import shardit
from nlsa.jax.utils import batch_map, batch_map_bivariate
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Literal,
    NamedTuple,
    Optional,
    final,
    overload,
)

if TYPE_CHECKING:
    type Device = Any
else:
    from jax import Device

type K = Array
type Ks = Array
type V = Array
type Vs = Array
type X = Array
type Y = Array
type Xs = Array
type PyTree = (
    Array
    | float
    | int
    | bool
    | list[PyTree]
    | tuple[PyTree, ...]
    | dict[Any, PyTree]
)
type ConvMode = Literal["full", "same", "valid"]
type Shape = tuple[int, ...]
type F[*Xs, Y] = Callable[[*Xs], Y]


def neg(v: V, /) -> V:
    """Perform vector negation."""
    return -v


def make_zero(
    shape: Shape,
    dtype: DTypeLike,
    sharding: Sharding | None = None,
) -> Callable[[], V]:
    """Make constant function returning vector of all 0s."""

    @partial(shardit, sharding=sharding)
    def zero() -> V:
        return jnp.zeros(shape, dtype=dtype)

    return zero


def make_unit(
    shape: Shape,
    dtype: DTypeLike,
    sharding: Sharding | None = None,
) -> Callable[[], V]:
    """Make constant function returning vector of all 1s."""

    @partial(shardit, sharding=sharding)
    def unit() -> V:
        return jnp.ones(shape, dtype=dtype)

    return unit


def inv(v: V, /) -> V:
    """Perform vector inversion."""
    return jnp.divide(1, v)


def sdiv(s: K, v: V, /) -> V:
    """Perform sclar division of a vector."""
    return jnp.divide(v, s)


def ldiv(u: V, v: V, /) -> V:
    """Perform left module division as elementwise vector division."""
    return jnp.divide(v, u)


def euclidean_innerp(u: V, v: V, /) -> K:
    """Compute Euclidean product of two vectors."""
    return jnp.sum(jnp.conjugate(u) * v)


def linf_norm(u: V, /):
    """Compute L-infinity norm of a vector."""
    return jnp.max(jnp.abs(u))


def make_weighted_innerp(weight: V) -> Callable[[V, V], K]:
    """Make inner procuct from weight vector."""

    def innerp(u: V, v: V, /) -> K:
        return jnp.sum(jnp.conjugate(u) * weight * v)

    return innerp


def make_l2_innerp(measure: Callable[[V], K]) -> Callable[[V, V], K]:
    """Make L2 inner product from measure."""

    def innerp(u: V, v: V, /) -> K:
        return measure(jnp.conjugate(u) * v)

    return innerp


def to_norm(innerp: Callable[[V, V], K], /) -> Callable[[V], K]:
    """Make norm from inner product."""

    def norm(v: V, /) -> K:
        return jnp.sqrt(innerp(v, v))

    return norm


def to_sqnorm(innerp: Callable[[V, V], K], /) -> Callable[[V], K]:
    """Make square norm from inner product."""

    def sqnorm(v: V, /) -> K:
        return innerp(v, v)

    return sqnorm


make_weighted_sqnorm: Callable[[V], F[V, K]] = fun.compose(
    to_sqnorm, make_weighted_innerp
)


def make_convolution(mode: ConvMode = "same") -> Callable[[V, V], V]:
    """Make convolution product between vectors."""

    def cnv(u: V, v: V, /) -> V:
        return convolve(u, v, mode=mode)

    return cnv


def make_weighted_convolution(
    weight: V, mode: ConvMode = "same"
) -> Callable[[V, V], V]:
    """Make weighted convolution product between vectors."""

    def cnv(u: V, v: V, /) -> V:
        return convolve(weight * u, weight * v, mode=mode) / weight

    return cnv


def counting_measure(v: V, /) -> Y:
    """Sum the elements of a vector."""
    return jnp.sum(v)


def make_normalized_counting_measure(n: int) -> Callable[[V], Y]:
    """Make normalized counting measure from dimension parameter."""

    def mu(v: V, /) -> Y:
        return counting_measure(v) / n

    return mu


def eval_at(xs: Xs, /) -> Callable[[F[Xs, V]], V]:
    """Make evaluation functional."""

    def ev(f: F[Xs, V], /) -> V:
        return f(xs)

    return ev


def _veval_at(
    xs: Xs,
    /,
    in_axis: int = 0,
    out_sharding: Optional[Sharding] = None,
    jit: bool = False,
) -> Callable[[F[X, Y]], V]:
    """Make vectorized evaluation functional."""

    def ev(f: F[Xs, Y], /) -> V:
        g = vmap(f, in_axes=in_axis)
        if out_sharding is not None:
            out_shard = partial(
                jax.lax.with_sharding_constraint, shardings=out_sharding
            )
            g = fun.compose(out_shard, g)
        if jit:
            g = jax.jit(g)
        return g(xs)

    return ev


def _veval_at_tuple(
    xss: tuple[Xs, Xs],
    /,
    in_axis: int = 0,
    out_sharding: Optional[Sharding] = None,
    jit: bool = False,
) -> Callable[[F[X, X, Y]], V]:
    """Make vectorized evaluation functional for bivariate functions."""

    def ev(f: F[X, X, Y], /) -> V:
        g = vmap(f, in_axes=in_axis)
        if out_sharding is not None:
            out_shard = partial(
                jax.lax.with_sharding_constraint, shardings=out_sharding
            )
            g = fun.compose(out_shard, g)
        if jit:
            g = jax.jit(g)
        return g(*xss)

    return ev


@overload
def veval_at(
    xs: Xs,
    /,
    in_axis: int = 0,
    out_sharding: Optional[Sharding] = None,
    jit: bool = False,
) -> Callable[[F[X, Y]], V]: ...


@overload
def veval_at(
    xss: tuple[Xs, Xs],
    /,
    in_axis: int = 0,
    out_sharding: Optional[Sharding] = None,
    jit: bool = False,
) -> Callable[[F[X, X, Y]], V]: ...


def veval_at(
    xss: Xs | tuple[Xs, Xs],
    /,
    in_axis: int = 0,
    out_sharding: Optional[Sharding] = None,
    jit: bool = False,
) -> Callable[[F[Xs, Y]], V] | Callable[[F[Xs, Xs, Y]], V]:
    """Make vectorized evaluation functional."""
    match xss:
        case Array():
            ev = _veval_at(
                xss, in_axis=in_axis, out_sharding=out_sharding, jit=jit
            )
        case tuple():
            ev = _veval_at_tuple(
                xss, in_axis=in_axis, out_sharding=out_sharding, jit=jit
            )
    return ev


def _batch_eval_at(
    xs: Xs,
    /,
    in_axis: int = 0,
    batch_size: Optional[int] = None,
    out_sharding: Optional[Sharding] = None,
    jit: bool = False,
) -> Callable[[F[X, Y]], V]:
    """Make batched evaluation functional."""

    def ev(f: F[X, Y], /) -> V:
        g = batch_map(f, in_axis=in_axis, batch_size=batch_size)
        if out_sharding is not None:
            out_shard = partial(
                jax.lax.with_sharding_constraint, shardings=out_sharding
            )
            g = fun.compose(out_shard, g)
        if jit:
            g = jax.jit(g)
        return g(xs)

    return ev


def _batch_eval_at_tuple(
    xss: tuple[Xs, Xs],
    /,
    in_axis: int = 0,
    batch_size: Optional[int] = None,
    out_sharding: Optional[Sharding] = None,
    jit: bool = False,
) -> Callable[[F[X, X, Y]], V]:
    """Make batched evaluation functional for bivariate functions."""

    def ev(f: F[X, X, Y], /) -> V:
        g = batch_map_bivariate(f, in_axis=in_axis, batch_size=batch_size)
        if out_sharding is not None:
            out_shard = partial(
                jax.lax.with_sharding_constraint, shardings=out_sharding
            )
            g = fun.compose(out_shard, g)
        if jit:
            g = jax.jit(g)
        return g(*xss)

    return ev


@overload
def batch_eval_at(
    xs: Xs,
    /,
    in_axis: int = 0,
    batch_size: Optional[int] = None,
    out_sharding: Optional[Sharding] = None,
    jit: bool = False,
) -> Callable[[F[X, Y]], V]: ...


@overload
def batch_eval_at(
    xss: tuple[Xs, Xs],
    /,
    in_axis: int = 0,
    batch_size: Optional[int] = None,
    out_sharding: Optional[Sharding] = None,
    jit: bool = False,
) -> Callable[[F[X, X, Y]], V]: ...


def batch_eval_at(
    xss: Xs | tuple[Xs, Xs],
    /,
    in_axis: int = 0,
    batch_size: Optional[int] = None,
    out_sharding: Optional[Sharding] = None,
    jit: bool = False,
) -> Callable[[F[X, Y]], V] | Callable[[F[X, X, Y]], V]:
    """Make vectorized and batched evaluation functional."""
    if isinstance(xss, Array):
        ev = _batch_eval_at(
            xss,
            in_axis=in_axis,
            batch_size=batch_size,
            out_sharding=out_sharding,
            jit=jit,
        )
    else:
        ev = _batch_eval_at_tuple(
            xss,
            in_axis=in_axis,
            batch_size=batch_size,
            out_sharding=out_sharding,
            jit=jit,
        )
    return ev


def shardeval_at(
    xs: Xs, /, devices: Optional[Sequence[Device]] = None
) -> Callable[[F[X, Y]], V]:
    """Make doubly-vectorized and sharded evaluation functional."""
    if devices is None:
        devices = jax.local_devices()
    ys_sharding = NamedSharding(
        Mesh(devices, axis_names="i"), PartitionSpec("i", None)
    )

    def ev(f: F[X, Y], /) -> V:
        g: Callable[[Xs], V] = vmap(vmap(f), axis_name="i")

        # @partial(jit, out_shardings=ys_sharding)
        @jax.jit
        def evg(xss: Xs, /) -> V:
            # ys = g(xss)
            ys = jax.lax.with_sharding_constraint(g(xss), ys_sharding)
            return ys

        return evg(xs)

    return ev


def flip_conj(v: V, /) -> V:
    """Perform involution (complex-conjugation and flip) on convolution alg."""
    return jnp.conjugate(jnp.flip(v))


def make_synthesis_operator_cols(
    basis: Vs, idxs: Optional[Array] = None
) -> Callable[[Ks], V]:
    """Make synthesis operator for vectors from basis.

    This function assumes that the basis elements are stored in the columns of
    the input array basis.

    """
    if idxs is not None:
        _basis = jnp.take(basis, idxs, axis=-1)
    else:
        _basis = basis

    def synth(coeffs: Ks, /) -> V:
        return _basis @ coeffs

    return synth


def make_synthesis_operator_rows(
    basis: Vs, idxs: Optional[Array] = None
) -> Callable[[Ks], V]:
    """Make synthesis operator for vectors from basis.

    This function assumes that the basis elements are stored in the rows of
    the input array basis.

    """
    if idxs is not None:
        _basis = jnp.take(basis, idxs, axis=0)
    else:
        _basis = basis

    def synth(coeffs: Ks, /) -> V:
        return coeffs @ _basis

    return synth


def make_synthesis_operator(
    basis: Vs, idxs: Optional[Array] = None, axis: Literal[0, 1] = 0
) -> Callable[[Ks], V]:
    """Make synthesis operator for vectors from basis or a subset thereof."""
    match axis:
        case 0:
            synth = make_synthesis_operator_rows(basis, idxs)
        case 1:
            synth = make_synthesis_operator_cols(basis, idxs)
    return synth


def make_fn_synthesis_operator[X: Array](
    basis: F[X, Ks],
) -> Callable[[Ks], F[X, K]]:
    """Make synthesis operator for functions from basis."""

    def synth(coeffs: Ks, /) -> F[X, K]:
        def f(x: X, /) -> K:
            b = basis(x)
            return jnp.sum(coeffs * b)

        return f

    return synth


def fn_synthesis[X: Array](coeffs: Ks, /, basis: F[X, Ks]) -> F[X, K]:
    """Perform function synthesis from basis."""

    def f(x: X, /) -> K:
        b = basis(x)
        return jnp.sum(coeffs * b)

    return f


def make_one_hot_basis(
    dim: int,
    value: float | Array = 1,
    dtype: Optional[DTypeLike] = None,
    sharding: Optional[NamedSharding] = None,
) -> Callable[[int | V], V]:
    """Make standard basis of real or complex Euclidean space."""

    @partial(shardit, sharding=sharding)
    def vc(i: int | V) -> V:
        e = jnp.zeros(dim, dtype=dtype)
        e = e.at[i].set(value)
        return e

    return vc


# TODO: Consider creating separate LpVectorAlgebra classes implementing the
# other Lp norms
@final
@dataclass(frozen=True)
class L2VectorAlgebra[N: Shape, D: DTypeLike](
    alg.ImplementsInnerProductStarAlgebraWithCalculus[V, K]
):
    """Implement vector algebra operations for JAX arrays."""

    shape: N
    dtype: D
    weight: Optional[V] = None
    sharding: Optional[Sharding] = None
    _scl: Optional[ScalarField[D]] = None
    _zero: Optional[Optional[Callable[[], V]]] = None
    _unit: Optional[Optional[Callable[[], V]]] = None
    _add: Optional[Callable[[V, V], V]] = None
    _neg: Optional[Callable[[V], V]] = None
    _sub: Optional[Callable[[V, V], V]] = None
    _sdiv: Optional[Callable[[K, V], V]] = None
    _smul: Optional[Callable[[K, V], V]] = None
    _mul: Optional[Callable[[V, V], V]] = None
    _div: Optional[Callable[[V, V], V]] = None
    _inv: Optional[Callable[[V], V]] = None
    _adj: Optional[Callable[[V], V]] = None
    _sqrt: Optional[Callable[[V], V]] = None
    _exp: Optional[Callable[[V], V]] = None
    _abs: Optional[Callable[[V], V]] = None
    _mpower: Optional[Callable[[V, int], V]] = None
    _power: Optional[Callable[[V, K], V]] = None
    _innerp: Optional[Callable[[V, V], K]] = None
    _norm: Optional[Callable[[V], K]] = None

    @property
    def dim(self) -> int:
        """Return dimension property of L2VectorAlgebra object."""
        return math.prod(self.shape)

    @property
    def zero(self) -> Callable[[], V]:
        """Return zero property of L2VectorAlgebra object."""
        return (
            make_zero(self.shape, self.dtype, self.sharding)
            if self._zero is None
            else self._zero
        )

    @property
    def unit(self) -> Callable[[], V]:
        """Return unit property of L2VectorAlgebra object."""
        return (
            make_unit(self.shape, self.dtype, self.sharding)
            if self._unit is None
            else self._unit
        )

    @property
    def scl(self) -> ScalarField[D]:
        """Return scl property of L2VectorAlgebra object."""
        return ScalarField(self.dtype) if self._scl is None else self._scl

    @property
    def add(self) -> Callable[[V, V], V]:
        """Return add property of L2VectorAlgebra object."""
        return jnp.add if self._add is None else self._add

    @property
    def neg(self) -> Callable[[V], V]:
        """Return neg property of L2VectorAlgebra object."""
        return neg if self._neg is None else self._neg

    @property
    def sub(self) -> Callable[[V, V], V]:
        """Return sub property of L2VectorAlgebra object."""
        return jnp.subtract if self._sub is None else self._sub

    @property
    def sdiv(self) -> Callable[[K, V], V]:
        """Return sdiv property of L2VectorAlgebra object."""
        return sdiv if self._sdiv is None else self._sdiv

    @property
    def smul(self) -> Callable[[K, V], V]:
        """Return smul property of L2VectorAlgebra object."""
        return jnp.multiply if self._smul is None else self._smul

    @property
    def mul(self) -> Callable[[V, V], V]:
        """Return mul property of L2VectorAlgebra object."""
        return jnp.multiply if self._mul is None else self._mul

    @property
    def div(self) -> Callable[[V, V], V]:
        """Return div property of L2VectorAlgebra object."""
        return jnp.divide if self._div is None else self._div

    @property
    def inv(self) -> Callable[[V], V]:
        """Return inv property of L2VectorAlgebra object."""
        return inv if self._inv is None else self._inv

    @property
    def adj(self) -> Callable[[V], V]:
        """Return adj property of L2VectorAlgebra object."""
        return jnp.conjugate if self._adj is None else self._adj

    @property
    def sqrt(self) -> Callable[[V], V]:
        """Return sqrt property of L2VectorAlgebra object."""
        return jnp.sqrt if self._sqrt is None else self._sqrt

    @property
    def exp(self) -> Callable[[V], V]:
        """Return exp property of L2VectorAlgebra object."""
        return jnp.exp if self._exp is None else self._exp

    @property
    def abs(self) -> Callable[[V], V]:
        """Return abs property of L2VectorAlgebra object."""
        return jnp.abs if self._abs is None else self._abs

    @property
    def mpower(self) -> Callable[[V, int], V]:
        """Return mpower property of L2VectorAlgebra object."""
        return jnp.power if self._mpower is None else self._mpower

    @property
    def power(self) -> Callable[[V, K], V]:
        """Return power property of L2VectorAlgebra object."""
        return jnp.power if self._power is None else self._power

    @property
    def innerp(self) -> Callable[[V, V], K]:
        """Return inner product property of L2VectorAlgebra object."""
        if self._innerp is None:
            return (
                euclidean_innerp
                if self.weight is None
                else make_weighted_innerp(self.weight)
            )
        else:
            return self._innerp

    @property
    def norm(self) -> Callable[[V], K]:
        """Return norm property of L2VectorAlgebra object."""
        return to_norm(self.innerp) if self._norm is None else self._norm


@final
@dataclass(frozen=True)
class L2FnAlgebra[N: Shape, D: DTypeLike, X: Array, Y: Array](
    alg.ImplementsL2FnAlgebra[X, Y, V, K]
):
    """Implement L2 function algebra operations for JAX arrays."""

    shape: N
    dtype: D
    measure: Callable[[V], Y]
    inclusion_map: Callable[[F[X, Y]], V]
    sharding: Optional[Sharding] = None
    _scl: Optional[alg.ImplementsComplexScalarField[K]] = None
    _zero: Optional[Optional[Callable[[], V]]] = None
    _unit: Optional[Optional[Callable[[], V]]] = None
    _add: Optional[Callable[[V, V], V]] = None
    _neg: Optional[Callable[[V], V]] = None
    _sub: Optional[Callable[[V, V], V]] = None
    _sdiv: Optional[Callable[[K, V], V]] = None
    _smul: Optional[Callable[[K, V], V]] = None
    _mul: Optional[Callable[[V, V], V]] = None
    _div: Optional[Callable[[V, V], V]] = None
    _inv: Optional[Callable[[V], V]] = None
    _adj: Optional[Callable[[V], V]] = None
    _sqrt: Optional[Callable[[V], V]] = None
    _exp: Optional[Callable[[V], V]] = None
    _abs: Optional[Callable[[V], V]] = None
    _mpower: Optional[Callable[[V, int], V]] = None
    _power: Optional[Callable[[V, K], V]] = None
    _innerp: Optional[Callable[[V, V], K]] = None
    _norm: Optional[Callable[[V], K]] = None

    @property
    def dim(self) -> int:
        """Return dimension property of L2FnAlgebra object."""
        return math.prod(self.shape)

    @property
    def zero(self) -> Callable[[], V]:
        """Return zero property of L2FnAlgebra object."""
        return (
            make_zero(self.shape, self.dtype, self.sharding)
            if self._zero is None
            else self._zero
        )

    @property
    def unit(self) -> Callable[[], V]:
        """Return unit property of L2FnAlgebra object."""
        return (
            make_unit(self.shape, self.dtype, self.sharding)
            if self._unit is None
            else self._unit
        )

    @property
    def scl(self) -> alg.ImplementsComplexScalarField[K]:
        """Return scl property of L2FnAlgebra object."""
        return ScalarField(self.dtype) if self._scl is None else self._scl

    @property
    def add(self) -> Callable[[V, V], V]:
        """Return add property of L2FnAlgebra object."""
        return jnp.add if self._add is None else self._add

    @property
    def neg(self) -> Callable[[V], V]:
        """Return neg property of L2FnAlgebra object."""
        return neg if self._neg is None else self._neg

    @property
    def sub(self) -> Callable[[V, V], V]:
        """Return sub property of L2FnAlgebra object."""
        return jnp.subtract if self._sub is None else self._sub

    @property
    def sdiv(self) -> Callable[[K, V], V]:
        """Return sdiv property of L2FnAlgebra object."""
        return sdiv if self._sdiv is None else self._sdiv

    @property
    def smul(self) -> Callable[[K, V], V]:
        """Return smul property of L2FnAlgebra object."""
        return jnp.multiply if self._smul is None else self._smul

    @property
    def mul(self) -> Callable[[V, V], V]:
        """Return mul property of L2FnAlgebra object."""
        return jnp.multiply if self._mul is None else self._mul

    @property
    def div(self) -> Callable[[V, V], V]:
        """Return div property of L2FnAlgebra object."""
        return jnp.divide if self._div is None else self._div

    @property
    def inv(self) -> Callable[[V], V]:
        """Return inv property of L2FnAlgebra object."""
        return inv if self._inv is None else self._inv

    @property
    def adj(self) -> Callable[[V], V]:
        """Return adj property of L2FnAlgebra object."""
        return jnp.conjugate if self._adj is None else self._adj

    @property
    def sqrt(self) -> Callable[[V], V]:
        """Return sqrt property of L2FnAlgebra object."""
        return jnp.sqrt if self._sqrt is None else self._sqrt

    @property
    def exp(self) -> Callable[[V], V]:
        """Return exp property of L2FnAlgebra object."""
        return jnp.exp if self._exp is None else self._exp

    @property
    def abs(self) -> Callable[[V], V]:
        """Return abs property of L2FnAlgebra object."""
        return jnp.abs if self._abs is None else self._abs

    @property
    def mpower(self) -> Callable[[V, int], V]:
        """Return mpower property of L2FnAlgebra object."""
        return jnp.power if self._mpower is None else self._mpower

    @property
    def power(self) -> Callable[[V, K], V]:
        """Return power property of L2FnAlgebra object."""
        return jnp.power if self._power is None else self._power

    @property
    def innerp(self) -> Callable[[V, V], K]:
        """Return inner product property of L2FnAlgebra object."""
        return (
            make_l2_innerp(self.measure)
            if self._innerp is None
            else self._innerp
        )

    @property
    def norm(self) -> Callable[[V], K]:
        """Return norm property of L2FnAlgebra object."""
        return to_norm(self.innerp) if self._norm is None else self._norm

    @property
    def integrate(self) -> Callable[[V], Y]:
        """Return integrate property of L2FnAlgebra object."""
        return self.measure

    @property
    def incl(self) -> Callable[[F[X, Y]], V]:
        """Return inclusion map property of L2FnAlgebra object."""
        return self.inclusion_map


class L2FnAlgebraShardings(NamedTuple):
    """NamedTuple holding data and vector shardings for L2FnAlgebra objects."""

    data: Optional[NamedSharding] = None
    """Data sharding"""

    vectors: Optional[NamedSharding] = None
    """L2 vector sharding."""


def make_l2_analysis_operator[N: Shape, D: DTypeLike, X: Array, Y: Array](
    impl: L2VectorAlgebra[N, D] | L2FnAlgebra[N, D, X, Y],
    basis: Iterable[V],
    axis: Optional[int] = None,
) -> Callable[[V], Ks]:
    """Make analysis operator from an array of vectors."""
    vinnerp = vmap(impl.innerp, in_axes=(axis, None))

    def an(v: V) -> Ks:
        return vinnerp(jnp.asarray(basis), v)

    return an


# # # TODO: Consider renaming this L1ConvolutionAlgebra and equip with L1 norm.
# # class ConvolutionAlgebra(alg.ImplementsInnerProductSpace[V, K],
# Generic[N, K]):
# #     """Implement convolution algebra operations for JAX arrays.

# #     The type variable N parameterizes the dimension of the algebra. The
# type
# #     variable K parameterizes the field of scalars.

# #     The class constructor takes in the zero element of the algebra as an
# #     optional argument. This is to allow the use of sharded arrays.
# #     """

# #     def __init__(self, dim: N, dtype: Type[K],
# #                  zero: Optional[Callable[[], V]] = None,
# #                  weight: Optional[V] = None,
# #                  conv_mode: Optional[ConvMode] = 'same',
# #                  conv_weight: Optional[V] = None):
# #         self.dim = dim
# #         self.scl = ScalarField(dtype)
# #         self.add: Callable[[V, V], V] = jnp.add
# #         self.neg: Callable[[V], V] = neg
# #         self.sub: Callable[[V, V], V] = jnp.subtract
# #         self.smul: Callable[[S, V], V] = jnp.multiply

# #         if zero is None:
# #             self.zero: Callable[[], V] = make_zero(dim, dtype)
# #         else:
# #             self.unit = zero

# #         if conv_weight is None:
# #             self.mul = make_convolution(mode=conv_mode)
# #         else:
# #             self.mul = make_weighted_convolution(mode=conv_mode,
# #                                                  weight=conv_weight)

# #         self.adj: Callable[[V], V] = flip_conj
# #         self.sqrt: Callable[[V], V] = jnp.sqrt
# #         self.exp: Callable[[V], V] = jnp.exp
# #         self.power: Callable[[V, V], V] = make_mpower(self.mul)

# #         if weight is None:
# #             self.innerp: Callable[[V, V], S] = euclidean_innerp
# #         else:
# #             self.innerp: Callable[[V, V], S] =
# make_weighted_euclidean_innerp(weight)

# #         self.norm: Callable[[V], S] = to_norm(self.innerp)


# # # TODO: Inheritance from VectorAlgebra can lead to inconsistency between
# inner
# # # product and integration.
# # class MeasureFnAlgebra(MeasurableFnAlgebra[T, N, K],
# #                        alg.ImplementsMeasureFnAlgebra[T, V, S]):
# #     """Implement MeasurableFunctionAlgebra equipped with measure."""
# #     def __init__(self, dim: N, dtype: Type[K],
# #                  inclusion_map: Callable[[F[T, S]], V],
# #                  measure: Callable[[V], S],
# #                  zero: Optional[Callable[[], V]] = None,
# #                  unit: Optional[Callable[[], V]] = None,
# #                  weight: Optional[V] = None):
# #         super().__init__(dim, dtype, inclusion_map, zero=zero, unit=unit,
# #                          weight=weight)
# #         self.integrate: Callable[[V], S] = measure


# # class LInfFnAlgebra(LInfVectorAlgebra[N, K],
# #                     alg.ImplementsMeasureFnAlgebra[T, V, S],
# #                     Generic[T, N, K]):
# #     """Implement operations on equivalence classes of functions using
# JAX arrays
# #     as the representation type.
# #     """

# #     def __init__(self, dim: N, dtype: Type[K],
# #                  inclusion_map: Callable[[F[T, S]], V],
# #                  measure: Callable[[V], S],
# #                  zero: Optional[Callable[[], V]] = None,
# #                  unit: Optional[Callable[[], V]] = None):
# #         super().__init__(dim, dtype, zero=zero, unit=unit)
# #         self.incl: Callable[[F[T, S]], V] = inclusion_map
# #         self.integrate: Callable[[V], S] = measure


# # class FnSynthesis(FunctionSpace[T, VectorAlgebra[N, K]], Generic[T, N, K]):
# #     """Implement function synthesis from coefficients in JAX array."""
# #     def __init__(self, dim: N, dtype: Type[K]):
# #         self.dom: VectorAlgebra[N, K] = VectorAlgebra(dim=dim, dtype=dtype)
# #         self.codom: FunctionSpace[T, K] = \
# #             FunctionSpace(codomain=ScalarField(dtype))
# #         super().__init__(codomain=FunctionSpace(codomain=self.dom))
# #         self.app = fn_synthesis()
# # # def sheval_at(xs: Xs, /, axis_name: str = 'i') -> Callable[[F[X, Y]], V]:
# # #     """Make sharded evaluation functional."""
# # #     devices = jax.local_devices()
# # #     mesh = Mesh(devices, axis_names=(axis_name))

# # #     def eval(f: F[X, Y]) -> V:
# # #         g: Callable[[Xs], V] = shmap(vmap(f), mesh=mesh,
# # #                                      in_specs=P(axis_name, None),
# # #                                      out_specs=P(axis_name))
# # #         return g(xs)
# # #     return eval

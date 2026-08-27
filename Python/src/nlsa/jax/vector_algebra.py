"""Implement vector algebra operations for JAX arrays."""

import jax
import jax.numpy as jnp
import math
import nlsa.abstract_algebra as alg
import nlsa.function_algebra as fun
import nlsa.jax.scalars as scls
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from jax import Array, vmap
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding
from jax.scipy.signal import convolve
from jax.typing import DTypeLike
from nlsa.jax.sharding import shardit
from nlsa.jax.typing import Idx, PyTree, typestable_jit
from nlsa.jax.utils import batch_map, batch_map_bivariate
from nlsa.typing import DEFAULT
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    final,
    overload,
)
from collections.abc import Iterable

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


def eval_at[Xs: PyTree](xs: Xs, /) -> Callable[[F[Xs, V]], V]:
    """Make evaluation functional."""

    def ev(f: F[Xs, V], /) -> V:
        return f(xs)

    return ev


def _veval_at[X: PyTree, Y: Array](
    xs: X,
    /,
    in_axis: int = 0,
    out_sharding: Sharding | None = None,
    jit: bool = False,
) -> Callable[[F[X, Y]], V]:
    """Make vectorized evaluation functional."""

    def ev(f: F[X, Y], /) -> V:
        g = vmap(f, in_axes=in_axis)
        if out_sharding is not None:
            out_shard = partial(
                jax.lax.with_sharding_constraint, shardings=out_sharding
            )
            g = fun.compose(out_shard, g)
        if jit:
            g = typestable_jit(g)
        return g(xs)

    return ev


def _veval_at_tuple[X: PyTree](
    xss: tuple[X, X],
    /,
    in_axis: int = 0,
    out_sharding: Sharding | None = None,
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
            g = typestable_jit(g)
        return g(*xss)

    return ev


@overload
def veval_at(
    xs: Xs,
    /,
    in_axis: int = 0,
    out_sharding: Sharding | None = None,
    jit: bool = False,
) -> Callable[[F[X, Y]], V]: ...


@overload
def veval_at(
    xss: tuple[Xs, Xs],
    /,
    in_axis: int = 0,
    out_sharding: Sharding | None = None,
    jit: bool = False,
) -> Callable[[F[X, X, Y]], V]: ...


def veval_at(
    xss: Xs | tuple[Xs, Xs],
    /,
    in_axis: int = 0,
    out_sharding: Sharding | None = None,
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
    batch_size: int | None = None,
    out_sharding: Sharding | None = None,
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
            g = typestable_jit(g)
        return g(xs)

    return ev


def _batch_eval_at_tuple(
    xss: tuple[Xs, Xs],
    /,
    in_axis: int = 0,
    batch_size: int | None = None,
    out_sharding: Sharding | None = None,
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
            g = typestable_jit(g)
        return g(*xss)

    return ev


@overload
def batch_eval_at(
    xs: Xs,
    /,
    in_axis: int = 0,
    batch_size: int | None = None,
    out_sharding: Sharding | None = None,
    jit: bool = False,
) -> Callable[[F[X, Y]], V]: ...


@overload
def batch_eval_at(
    xss: tuple[Xs, Xs],
    /,
    in_axis: int = 0,
    batch_size: int | None = None,
    out_sharding: Sharding | None = None,
    jit: bool = False,
) -> Callable[[F[X, X, Y]], V]: ...


def batch_eval_at(
    xss: Xs | tuple[Xs, Xs],
    /,
    in_axis: int = 0,
    batch_size: int | None = None,
    out_sharding: Sharding | None = None,
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
    xs: Xs, /, devices: Sequence[Device] | None = None
) -> Callable[[F[X, Y]], V]:
    """Make doubly-vectorized and sharded evaluation functional."""
    if devices is None:
        devices = jax.local_devices()
    ys_sharding = NamedSharding(
        Mesh(devices, axis_names="i"), PartitionSpec("i", None)
    )

    def ev(f: F[X, Y], /) -> V:
        g: Callable[[Xs], V] = vmap(vmap(f), axis_name="i")

        @typestable_jit
        def evg(xss: Xs, /) -> V:
            ys = jax.lax.with_sharding_constraint(g(xss), ys_sharding)
            return ys

        return evg(xs)

    return ev


def flip_conj(v: V, /) -> V:
    """Perform involution (complex-conjugation and flip) on convolution alg."""
    return jnp.conjugate(jnp.flip(v))


def make_synthesis_operator_cols(
    basis: Vs, idxs: Array | None = None
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
    basis: Vs, idxs: Array | None = None
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
    basis: Vs, idxs: Array | None = None, axis: Literal[0, 1] = 0
) -> Callable[[Ks], V]:
    """Make synthesis operator for vectors from basis or a subset thereof."""
    match axis:
        case 0:
            synth = make_synthesis_operator_rows(basis, idxs)
        case 1:
            synth = make_synthesis_operator_cols(basis, idxs)
    return synth


def make_fn_synthesis_operator[X: PyTree, Ks: Array](
    basis: F[X, Ks],
) -> Callable[[Ks], F[X, K]]:
    """Make synthesis operator for functions from basis."""

    def synth(coeffs: Ks, /) -> F[X, K]:
        def f(x: X, /) -> K:
            b = basis(x)
            return jnp.sum(coeffs * b)

        return f

    return synth


def fn_synthesis[X: PyTree, Ks: Array](
    coeffs: Ks, /, basis: F[X, Ks]
) -> F[X, K]:
    """Perform function synthesis from basis."""

    def f(x: X, /) -> K:
        b = basis(x)
        return jnp.sum(coeffs * b)

    return f


def make_one_hot_basis(
    dim: int,
    value: float | Array = 1,
    dtype: DTypeLike | None = None,
    sharding: NamedSharding | None = None,
) -> Callable[[Idx], V]:
    """Make standard basis of real or complex Euclidean space."""

    @partial(shardit, sharding=sharding)
    def vc(i: Idx) -> V:
        e = jnp.zeros(dim, dtype=dtype)
        e = e.at[i].set(value)
        return e

    return vc


# TODO: Consider creating separate LpVectorAlgebra classes implementing the
# other Lp norms
@final
@dataclass(frozen=True, slots=True)
class L2VectorAlgebra[N: Shape, D: DTypeLike](
    alg.ImplementsInnerProductStarAlgebraWithCalculus[V, K]
):
    """Implement L2 vector algebra operations for JAX arrays."""

    shape: N
    dtype: D
    weight: V | None
    sharding: Sharding | None
    _scl: alg.ImplementsComplexScalarField[K]
    _zero: Callable[[], V]
    _unit: Callable[[], V]
    _add: Callable[[V, V], V]
    _neg: Callable[[V], V]
    _sub: Callable[[V, V], V]
    _sdiv: Callable[[K, V], V]
    _smul: Callable[[K, V], V]
    _mul: Callable[[V, V], V]
    _div: Callable[[V, V], V]
    _inv: Callable[[V], V]
    _adj: Callable[[V], V]
    _sqrt: Callable[[V], V]
    _exp: Callable[[V], V]
    _log: Callable[[V], V]
    _abs: Callable[[V], V]
    _mpower: Callable[[V, int], V]
    _power: Callable[[V, K], V]
    _innerp: Callable[[V, V], K]
    _norm: Callable[[V], K]

    @property
    def dim(self) -> int:
        """Return dimension property of L2VectorAlgebra object."""
        return math.prod(self.shape)

    @property
    def scl(self) -> alg.ImplementsComplexScalarField[K]:
        """Scalar field associated with L2VectorAlgebra object."""
        return self._scl

    def zero(self, /) -> V:
        """Return zero vector."""
        return self._zero()

    def add(self, u: V, v: V, /) -> V:
        """Add two vectors."""
        return self._add(u, v)

    def neg(self, v: V, /) -> V:
        """Compute additive inverse (negation) of a vector."""
        return self._neg(v)

    def sub(self, u: V, v: V, /) -> V:
        """Subtract two vectors."""
        return self._sub(u, v)

    def smul(self, k: K, v: V, /) -> V:
        """Multiply a vector by a scalar."""
        return self._smul(k, v)

    def sdiv(self, k: K, v: V, /) -> V:
        """Divide a vector by a scalar."""
        return self._sdiv(k, v)

    def unit(self, /) -> V:
        """Return vector with elements equal to 1 (multiplicative unit)."""
        return self._unit()

    def mul(self, u: V, v: V, /) -> V:
        """Compute elementwise multiplication of two vectors."""
        return self._mul(u, v)

    def div(self, u: V, v: V, /) -> V:
        """Compute elementwise division of two vectors."""
        return self._div(u, v)

    def inv(self, v: V, /) -> V:
        """Compute elementwise multiplicative inverse of a vector."""
        return self._inv(v)

    def adj(self, v: V, /) -> V:
        """Compute elementwise complex conjugate of a vector."""
        return self._adj(v)

    def sqrt(self, v: V, /) -> V:
        """Compute elementwise square root of a vector."""
        return self._sqrt(v)

    def abs(self, v: V, /) -> V:
        """Compute elementwise modulus of a vector."""
        return self._abs(v)

    def exp(self, v: V, /) -> V:
        """Compute elementwise exponential of a vector."""
        return self._exp(v)

    def log(self, a: V, /) -> V:
        """Compute elementwise natural logarithm of a vector."""
        return self._log(a)

    def mpower(self, v: V, k: int, /) -> V:
        """Compute elementwise exponentiation of a vector by an integer."""
        return self._mpower(v, k)

    def power(self, v: V, k: K, /) -> V:
        """Compute elementwise exponentiation of a vector by a scalar."""
        return self._power(v, k)

    def innerp(self, u: V, v: V, /) -> K:
        """Compute the inner product between two vectors."""
        return self._innerp(u, v)

    def norm(self, v: V, /) -> K:
        """Compute the norm of a vector."""
        return self._norm(v)


def l2_vector_algebra[N: Shape, D: DTypeLike](
    shape: N,
    dtype: D,
    weight: V | None = None,
    sharding: Sharding | None = None,
    scl: alg.ImplementsComplexScalarField[K] | DEFAULT = DEFAULT,
    zero: Callable[[], V] | DEFAULT = DEFAULT,
    unit: Callable[[], V] | DEFAULT = DEFAULT,
    add: Callable[[V, V], V] | DEFAULT = DEFAULT,
    neg: Callable[[V], V] | DEFAULT = DEFAULT,
    sub: Callable[[V, V], V] | DEFAULT = DEFAULT,
    sdiv: Callable[[K, V], V] | DEFAULT = DEFAULT,
    smul: Callable[[K, V], V] | DEFAULT = DEFAULT,
    mul: Callable[[V, V], V] | DEFAULT = DEFAULT,
    div: Callable[[V, V], V] | DEFAULT = DEFAULT,
    inv: Callable[[V], V] | DEFAULT = DEFAULT,
    adj: Callable[[V], V] | DEFAULT = DEFAULT,
    sqrt: Callable[[V], V] | DEFAULT = DEFAULT,
    abs: Callable[[V], V] | DEFAULT = DEFAULT,
    exp: Callable[[V], V] | DEFAULT = DEFAULT,
    log: Callable[[V], V] | DEFAULT = DEFAULT,
    mpower: Callable[[V, int], V] | DEFAULT = DEFAULT,
    power: Callable[[V, K], V] | DEFAULT = DEFAULT,
    innerp: Callable[[V, V], K] | DEFAULT = DEFAULT,
    norm: Callable[[V], K] | DEFAULT = DEFAULT,
) -> L2VectorAlgebra[N, D]:
    """Build L2VectorAlgebra object."""
    if innerp is not DEFAULT:
        _innerp = innerp
    elif weight is not None:
        _innerp = make_weighted_innerp(weight)
    else:
        _innerp = euclidean_innerp
    return L2VectorAlgebra(
        shape=shape,
        dtype=dtype,
        weight=weight,
        sharding=sharding,
        _scl=scls.scalar_field(dtype) if scl is DEFAULT else scl,
        _zero=(make_zero(shape, dtype, sharding) if zero is DEFAULT else zero),
        _unit=(make_unit(shape, dtype, sharding) if unit is DEFAULT else unit),
        _add=jnp.add if add is DEFAULT else add,
        _neg=(lambda v: -v) if neg is DEFAULT else neg,
        _sub=jnp.subtract if sub is DEFAULT else sub,
        _sdiv=(lambda k, v: v / k) if sdiv is DEFAULT else sdiv,
        _smul=jnp.multiply if smul is DEFAULT else smul,
        _mul=jnp.multiply if mul is DEFAULT else mul,
        _div=jnp.divide if div is DEFAULT else div,
        _inv=(lambda v: 1 / v) if inv is DEFAULT else inv,
        _adj=jnp.conjugate if adj is DEFAULT else adj,
        _sqrt=jnp.sqrt if sqrt is DEFAULT else sqrt,
        _abs=jnp.abs if abs is DEFAULT else abs,
        _exp=jnp.exp if exp is DEFAULT else exp,
        _log=jnp.log if log is DEFAULT else log,
        _mpower=jnp.power if mpower is DEFAULT else mpower,
        _power=jnp.power if power is DEFAULT else power,
        _innerp=_innerp,
        _norm=to_norm(_innerp) if norm is DEFAULT else norm,
    )


@final
@dataclass(frozen=True, slots=True)
class L2FnAlgebra[N: Shape, D: DTypeLike, X: PyTree, Y: Array](
    alg.ImplementsL2FnAlgebra[X, Y, V, K]
):
    """Implement L2 function algebra operations for JAX arrays."""

    shape: N
    dtype: D
    measure: Callable[[V], Y]
    inclusion_map: Callable[[F[X, Y]], V]
    sharding: Sharding | None
    _scl: alg.ImplementsComplexScalarField[K]
    _zero: Callable[[], V]
    _unit: Callable[[], V]
    _add: Callable[[V, V], V]
    _neg: Callable[[V], V]
    _sub: Callable[[V, V], V]
    _sdiv: Callable[[K, V], V]
    _smul: Callable[[K, V], V]
    _mul: Callable[[V, V], V]
    _div: Callable[[V, V], V]
    _inv: Callable[[V], V]
    _adj: Callable[[V], V]
    _sqrt: Callable[[V], V]
    _exp: Callable[[V], V]
    _log: Callable[[V], V]
    _exp: Callable[[V], V]
    _abs: Callable[[V], V]
    _mpower: Callable[[V, int], V]
    _power: Callable[[V, K], V]
    _innerp: Callable[[V, V], K]
    _norm: Callable[[V], K]

    @property
    def dim(self) -> int:
        """Return dimension property of L2FunctionAlgebra object."""
        return math.prod(self.shape)

    @property
    def scl(self) -> alg.ImplementsComplexScalarField[K]:
        """Scalar field associated with L2FnAlgebra object."""
        return self._scl

    def zero(self, /) -> V:
        """Return zero vector."""
        return self._zero()

    def add(self, u: V, v: V, /) -> V:
        """Add two vectors."""
        return self._add(u, v)

    def neg(self, v: V, /) -> V:
        """Compute additive inverse (negation) of a vector."""
        return self._neg(v)

    def sub(self, u: V, v: V, /) -> V:
        """Subtract two vectors."""
        return self._sub(u, v)

    def smul(self, k: K, v: V, /) -> V:
        """Multiply a vector by a scalar."""
        return self._smul(k, v)

    def sdiv(self, k: K, v: V, /) -> V:
        """Divide a vector by a scalar."""
        return self._sdiv(k, v)

    def unit(self, /) -> V:
        """Return vector with elements equal to 1 (multiplicative unit)."""
        return self._unit()

    def mul(self, u: V, v: V, /) -> V:
        """Compute elementwise multiplication of two vectors."""
        return self._mul(u, v)

    def div(self, u: V, v: V, /) -> V:
        """Compute elementwise division of two vectors."""
        return self._div(u, v)

    def inv(self, v: V, /) -> V:
        """Compute elementwise multiplicative inverse of a vector."""
        return self._inv(v)

    def adj(self, v: V, /) -> V:
        """Compute elementwise complex conjugate of a vector."""
        return self._adj(v)

    def sqrt(self, v: V, /) -> V:
        """Compute elementwise square root of a vector."""
        return self._sqrt(v)

    def abs(self, v: V, /) -> V:
        """Compute elementwise modulus of a vector."""
        return self._abs(v)

    def exp(self, v: V, /) -> V:
        """Compute elementwise exponential of a vector."""
        return self._exp(v)

    def log(self, v: V, /) -> V:
        """Compute elementwise natural logarithm of a vector."""
        return self._log(v)

    def mpower(self, v: V, m: int, /) -> V:
        """Compute elementwise exponentiation of a vector by an integer."""
        return self._mpower(v, m)

    def power(self, v: V, k: K, /) -> V:
        """Compute elementwise exponentiation of a vector by a scalar."""
        return self._power(v, k)

    def innerp(self, u: V, v: V, /) -> K:
        """Compute the inner product between two vectors."""
        return self._innerp(u, v)

    def norm(self, v: V, /) -> K:
        """Compute the norm of a vector."""
        return self._norm(v)

    def integrate(self, v: V, /) -> Y:
        """Integrate a vector with respect to a measure."""
        return self.measure(v)

    def incl(self, f: F[X, Y]) -> V:
        """Apply the inclusion map to a function to obtain a vector."""
        return self.inclusion_map(f)


def l2_fn_algebra[N: Shape, D: DTypeLike, X: PyTree, Y: Array](
    shape: N,
    dtype: D,
    measure: Callable[[V], Y],
    inclusion_map: Callable[[F[X, Y]], V],
    sharding: Sharding | None = None,
    scl: alg.ImplementsComplexScalarField[K] | DEFAULT = DEFAULT,
    zero: Callable[[], V] | DEFAULT = DEFAULT,
    unit: Callable[[], V] | DEFAULT = DEFAULT,
    add: Callable[[V, V], V] | DEFAULT = DEFAULT,
    neg: Callable[[V], V] | DEFAULT = DEFAULT,
    sub: Callable[[V, V], V] | DEFAULT = DEFAULT,
    sdiv: Callable[[K, V], V] | DEFAULT = DEFAULT,
    smul: Callable[[K, V], V] | DEFAULT = DEFAULT,
    mul: Callable[[V, V], V] | DEFAULT = DEFAULT,
    div: Callable[[V, V], V] | DEFAULT = DEFAULT,
    inv: Callable[[V], V] | DEFAULT = DEFAULT,
    adj: Callable[[V], V] | DEFAULT = DEFAULT,
    sqrt: Callable[[V], V] | DEFAULT = DEFAULT,
    exp: Callable[[V], V] | DEFAULT = DEFAULT,
    log: Callable[[V], V] | DEFAULT = DEFAULT,
    abs: Callable[[V], V] | DEFAULT = DEFAULT,
    mpower: Callable[[V, int], V] | DEFAULT = DEFAULT,
    power: Callable[[V, K], V] | DEFAULT = DEFAULT,
    innerp: Callable[[V, V], K] | DEFAULT = DEFAULT,
    norm: Callable[[V], K] | DEFAULT = DEFAULT,
) -> L2FnAlgebra[N, D, X, Y]:
    """Build L2FnAlgebra object."""
    _innerp = make_l2_innerp(measure) if innerp is DEFAULT else innerp
    return L2FnAlgebra(
        shape=shape,
        dtype=dtype,
        measure=measure,
        inclusion_map=inclusion_map,
        sharding=sharding,
        _scl=scls.scalar_field(dtype) if scl is DEFAULT else scl,
        _zero=(make_zero(shape, dtype, sharding) if zero is DEFAULT else zero),
        _unit=(make_unit(shape, dtype, sharding) if unit is DEFAULT else unit),
        _add=jnp.add if add is DEFAULT else add,
        _neg=(lambda v: -v) if neg is DEFAULT else neg,
        _sub=jnp.subtract if sub is DEFAULT else sub,
        _sdiv=(lambda k, v: v / k) if sdiv is DEFAULT else sdiv,
        _smul=jnp.multiply if smul is DEFAULT else smul,
        _mul=jnp.multiply if mul is DEFAULT else mul,
        _div=jnp.divide if div is DEFAULT else div,
        _inv=(lambda v: 1 / v) if inv is DEFAULT else inv,
        _adj=jnp.conjugate if adj is DEFAULT else adj,
        _sqrt=jnp.sqrt if sqrt is DEFAULT else sqrt,
        _exp=jnp.exp if exp is DEFAULT else exp,
        _log=jnp.log if log is DEFAULT else log,
        _abs=jnp.abs if abs is DEFAULT else abs,
        _mpower=jnp.power if mpower is DEFAULT else mpower,
        _power=jnp.power if power is DEFAULT else power,
        _innerp=_innerp,
        _norm=to_norm(_innerp) if norm is DEFAULT else norm,
    )


class L2FnAlgebraShardings(NamedTuple):
    """NamedTuple holding data and vector shardings for L2FnAlgebra objects."""

    data: NamedSharding | None = None
    """Data sharding"""

    vectors: NamedSharding | None = None
    """L2 vector sharding."""

    @property
    def shard_pytree[Data: PyTree](
        self,
    ) -> Callable[[Data], Data]:
        """Shard PyTree objects."""

        def shard(
            data: Data,
        ) -> Data:
            return jax.tree.map(
                partial(jax.device_put, device=self.data), data
            )

        return shard


def make_l2_analysis_operator[N: Shape, D: DTypeLike, X: PyTree, Y: Array](
    impl: L2VectorAlgebra[N, D] | L2FnAlgebra[N, D, X, Y],
    basis: Iterable[V],
    axis: int | None = None,
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

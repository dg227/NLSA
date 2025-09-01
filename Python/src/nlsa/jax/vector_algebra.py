# pyright: basic

"""Implement vector algebra operations for JAX arrays."""

import jax
import jax.numpy as jnp
import nlsa.abstract_algebra as alg
from collections.abc import Callable, Sequence
from jax import Array, Device, vmap
from jax.sharding import Mesh, NamedSharding, PartitionSpec as ParSpec
from jax.scipy.signal import convolve
from jax.typing import DTypeLike
from nlsa.jax.scalars import ScalarField
from nlsa.function_algebra import compose
from typing import Literal, NamedTuple, Optional, final, overload

type K = Array
type Ks = Array
type V = Array
type Vs = Array
type X = Array
type Y = Array
type Xs = Array
type ConvMode = Literal['full', 'same', 'valid']
type Shape = tuple[int, ...]
type F[*Xs, Y] = Callable[[*Xs], Y]


def neg(v: V, /) -> V:
    """Perform vector negation."""
    return -v


def make_zero(shape: Shape, dtype: DTypeLike) -> Callable[[], V]:
    """Make constant function returning vector of all 0s."""
    def zero() -> V:
        return jnp.zeros(shape, dtype=dtype)
    return zero


def make_unit(shape: Shape, dtype: DTypeLike) -> Callable[[], V]:
    """Make constant function returning vector of all 1s."""
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


make_weighted_sqnorm: Callable[[V], F[V, K]] \
    = compose(to_sqnorm, make_weighted_innerp)


def make_convolution(mode: ConvMode = 'same') -> Callable[[V, V], V]:
    """Make convolution product between vectors."""
    def cnv(u: V, v: V, /) -> V:
        return convolve(u, v, mode=mode)
    return cnv


def make_weighted_convolution(weight: V, mode: ConvMode = 'same') \
        -> Callable[[V, V], V]:
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


def _veval_at(xs: Xs, /, in_axis: int = 0,
              jit: bool = False) -> Callable[[F[X, Y]], V]:
    """Make vectorized evaluation functional."""
    def ev(f: F[Xs, Y], /) -> V:
        g = vmap(f, in_axes=in_axis)
        if jit:
            g = jax.jit(g)
        return g(xs)
    return ev


def _veval_at_tuple(xss: tuple[Xs, Xs], /, in_axis: int = 0,
                    jit: bool = False) \
        -> Callable[[F[X, X, Y]], V]:
    """Make vectorized evaluation functional for bivariate functions."""
    def ev(f: F[X, X, Y], /) -> V:
        g = vmap(f, in_axes=in_axis)
        if jit:
            g = jax.jit(g)
        return g(*xss)
    return ev


@overload
def veval_at(xs: Xs, /, in_axis: int = 0,
             jit: bool = False) -> Callable[[F[X, Y]], V]:
    ...


@overload
def veval_at(xss: tuple[Xs, Xs], /, in_axis: int = 0,
             jit: bool = False) -> Callable[[F[X, X, Y]], V]:
    ...


def veval_at(xss: Xs | tuple[Xs, Xs], /, in_axis: int = 0,
             jit: bool = False) \
        -> Callable[[F[Xs, Y]], V] | Callable[[F[Xs, Xs, Y]], V]:
    """Make vectorized evaluation functional for univariate or bivariate
    functions.
    """
    if isinstance(xss, Array):
        ev = _veval_at(xss, in_axis=in_axis, jit=jit)
    else:
        ev = _veval_at_tuple(xss, in_axis=in_axis, jit=jit)
    return ev


def _batch_eval_at(xs: Xs, /, batch_size: Optional[int] = None) \
        -> Callable[[F[X, Y]], V]:
    """Make vectorized and batched evaluation functional."""
    def ev(f: F[X, Y], /) -> V:
        return jax.lax.map(f, xs, batch_size=batch_size)
    return ev


def _batch_eval_at_tuple(xss: tuple[Xs, Xs], /,
                         batch_size: Optional[int] = None) \
        -> Callable[[F[X, X, Y]], V]:
    """Make vectorized and batched evaluation functional for bivariate
    functions.
    """
    def ev(f: F[X, X, Y], /) -> V:
        def g(args: tuple[Xs, Xs], /) -> Y:
            return f(*args)
        return jax.lax.map(g, xss, batch_size=batch_size)
    return ev


@overload
def batch_eval_at(xs: Xs, /, batch_size: Optional[int] = None) \
        -> Callable[[F[X, Y]], V]:
    ...


@overload
def batch_eval_at(xss: tuple[Xs, Xs], /, batch_size: Optional[int] = None) \
        -> Callable[[F[X, X, Y]], V]:
    ...


def batch_eval_at(xss: Xs | tuple[Xs, Xs], batch_size: Optional[int] = None) \
        -> Callable[[F[X, Y]], V] | Callable[[F[X, X, Y]], V]:
    """Make vectorized and batched evaluation functional."""
    if isinstance(xss, Array):
        ev = _batch_eval_at(xss, batch_size=batch_size)
    else:
        ev = _batch_eval_at_tuple(xss, batch_size=batch_size)
    return ev


def shardeval_at(xs: Xs, /, devices: Optional[Sequence[Device]] = None) \
        -> Callable[[F[X, Y]], V]:
    """Make doubly-vectorized and sharded evaluation functional."""
    if devices is None:
        devices = jax.local_devices()
    ys_sharding = NamedSharding(Mesh(devices, axis_names='i'),
                                ParSpec('i', None))

    def ev(f: F[X, Y], /) -> V:
        g: Callable[[Xs], V] = vmap(vmap(f), axis_name='i')

        # @partial(jit, out_shardings=ys_sharding)
        @jax.jit
        def evg(xss: Xs, /) -> V:
            # ys = g(xss)
            ys = jax.lax.with_sharding_constraint(g(xss), ys_sharding)
            return ys
        return evg(xs)
    return ev


# # # TODO: Add deprecation warning in favor of shardeval_at
# # def jeval_at(xs: Xs, /, axis_name: str = 'i', devices=None) \
# #         -> Callable[[F[X, Y]], V]:
# #     """Make doubly-vectorized and jitted evaluation functional."""
# #     if devices is None:
# #         devices = jax.local_devices()

# #     mesh = Mesh(devices, axis_names=(axis_name))
# #     # xs_sharding = NamedSharding(mesh, P(axis_name, None, None))
# #     ys_sharding = NamedSharding(mesh, P(axis_name, None))

# #     def eval(f: F[X, Y], /.) -> V:
# #         g: Callable[[Xs], V] = vmap(vmap(f), axis_name=axis_name)

# #         @jit
# #         def evalg(xss: Xs) -> V:
# #             ys = jax.lax.with_sharding_constraint(g(xss), ys_sharding)
# #             return ys
# #         return evalg(xs)
# #     return eval


def flip_conj(v: V, /) -> V:
    """Perform involution (complex-conjugation and flip) on convolution alg."""
    return jnp.conjugate(jnp.flip(v))


def make_synthesis_operator_full(basis: Vs) -> Callable[[Ks], V]:
    """Make synthesis operator for vectors from basis."""
    def synth(coeffs: Ks, /) -> V:
        return basis @ coeffs
    return synth


def make_synthesis_operator_sub(basis: Vs, idxs: Array) -> Callable[[Ks], V]:
    """Make synthesis operator for vectors from subset of basis.

    This attempts to emulate "lazy" slicing of basis array.

    """
    def synth(coeffs: Ks, /) -> V:
        return jnp.take(basis, idxs, axis=-1) @ coeffs
    return synth


def make_synthesis_operator(basis: Vs, idxs: Optional[Array] = None) \
        -> Callable[[Ks], V]:
    """Make synthesis operator for vectors from basis or a subset thereof."""
    if idxs is not None:
        synth = make_synthesis_operator_sub(basis, idxs)
    else:
        synth = make_synthesis_operator_full(basis)
    return synth


def make_fn_synthesis_operator[X: Array](basis: F[X, Ks]) \
        -> Callable[[Ks], F[X, K]]:
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


@final
class VectorAlgebra[N: Shape,
                    D: DTypeLike](alg.ImplementsInnerProductAlgebra[V, K]):
    """Implement vector algebra operations for JAX arrays."""
    def __init__(self, shape: N, dtype: D,
                 zero: Optional[Callable[[], V]] = None,
                 unit: Optional[Callable[[], V]] = None,
                 weight: Optional[V] = None):
        self.shape = shape
        self.dtype = dtype
        self.scl: ScalarField[D] = ScalarField(dtype)
        self.add: Callable[[V, V], V] = jnp.add
        self.neg: Callable[[V], V] = neg
        self.sub: Callable[[V, V], V] = jnp.subtract
        self.sdiv: Callable[[K, V], V] = sdiv
        self.smul: Callable[[K, V], V] = jnp.multiply
        self.mul: Callable[[V, V], V] = jnp.multiply
        self.div: Callable[[V, V], V] = jnp.divide
        self.inv: Callable[[V], V] = inv
        self.adj: Callable[[V], V] = jnp.conjugate
        self.sqrt: Callable[[V], V] = jnp.sqrt
        self.exp: Callable[[V], V] = jnp.exp
        self.mod: Callable[[V], V] = jnp.abs
        self.power: Callable[[V, int], V] = jnp.power

        if zero is None:
            self.zero: Callable[[], V] = make_zero(shape, dtype)
        else:
            self.zero = zero

        if unit is None:
            self.unit: Callable[[], V] = make_unit(shape, dtype)
        else:
            self.unit = unit

        if weight is None:
            self.innerp: Callable[[V, V], K] = euclidean_innerp
        else:
            # TODO: Add check that weight has the right shape.
            self.innerp: Callable[[V, V], K] = make_weighted_innerp(weight)

        self.norm: Callable[[V], K] = to_norm(self.innerp)


@final
class L2VectorAlgebra[N: Shape, D: DTypeLike, X: Array, Y: Array](
        alg.ImplementsL2FnAlgebra[X, Y, V, K]):
    """Implement L2 vector algebra operations for JAX arrays. """
    def __init__(self, shape: N, dtype: D,
                 measure: Callable[[V], Y],
                 inclusion_map: Callable[[F[X, Y]], V],
                 zero: Optional[Callable[[], V]] = None,
                 unit: Optional[Callable[[], V]] = None):
        self.shape = shape
        self.dtype = dtype
        self.scl: ScalarField[D] = ScalarField(dtype)
        self.add: Callable[[V, V], V] = jnp.add
        self.neg: Callable[[V], V] = neg
        self.sub: Callable[[V, V], V] = jnp.subtract
        self.sdiv: Callable[[K, V], V] = sdiv
        self.smul: Callable[[K, V], V] = jnp.multiply
        self.mul: Callable[[V, V], V] = jnp.multiply
        self.div: Callable[[V, V], V] = jnp.divide
        self.inv: Callable[[V], V] = inv
        self.adj: Callable[[V], V] = jnp.conjugate
        self.sqrt: Callable[[V], V] = jnp.sqrt
        self.exp: Callable[[V], V] = jnp.exp
        self.mod: Callable[[V], V] = jnp.abs
        self.power: Callable[[V, int], V] = jnp.power
        self.integrate: Callable[[V], Y] = measure
        self.incl: Callable[[F[X, Y]], V] = inclusion_map

        if zero is None:
            self.zero: Callable[[], V] = make_zero(shape, dtype)
        else:
            self.unit = zero

        if unit is None:
            self.unit: Callable[[], V] = make_unit(shape, dtype)
        else:
            self.unit = unit

        self.innerp: Callable[[V, V], K] = make_l2_innerp(measure)
        self.norm: Callable[[V], K] = to_norm(self.innerp)


def make_l2_analysis_operator[N: Shape, D: DTypeLike, X: Array, Y: Array](
        impl: VectorAlgebra[N, D] | L2VectorAlgebra[N, D, X, Y], basis: Vs,
        axis: Optional[int] = None) -> Callable[[V], Ks]:
    """Make analysis operator from an array of vectors"""
    if axis is None:
        axis = -1
    vinnerp = vmap(impl.innerp, in_axes=(axis, None))

    def an(v: V) -> Ks:
        return vinnerp(basis, v)
    return an


# # # TODO: Consider renaming this L1ConvolutionAlgebra and equip with L1 norm.
# # class ConvolutionAlgebra(alg.ImplementsInnerProductSpace[V, K], Generic[N, K]):
# #     """Implement convolution algebra operations for JAX arrays.

# #     The type variable N parameterizes the dimension of the algebra. The type
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
# #             self.innerp: Callable[[V, V], S] = make_weighted_euclidean_innerp(weight)

# #         self.norm: Callable[[V], S] = to_norm(self.innerp)


# # class MeasurableFnAlgebra(VectorAlgebra[N, K],
# #                           alg.ImplementsMeasurableFnAlgebra[T, V, S],
# #                           Generic[T, N, K]):
# #     """Implement operations on equivalence classes of functions using JAX arrays
# #     as the representation type.
# #     """
# #     def __init__(self, dim: N, dtype: Type[K],
# #                  inclusion_map: Callable[[F[T, S]], V],
# #                  zero: Optional[Callable[[], V]] = None,
# #                  unit: Optional[Callable[[], V]] = None,
# #                  weight: Optional[V] = None):
# #         super().__init__(dim, dtype, zero=zero, unit=unit, weight=weight)
# #         self.incl: Callable[[F[T, S]], V] = inclusion_map


# # # TODO: Inheritance from VectorAlgebra can lead to inconsistency between inner
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


# # class LInfVectorAlgebra(alg.ImplementsNormedAlgebra[V, K], Generic[N, K]):
# #     """Implement L infinity algebra operations for JAX arrays.

# #     The type variable N parameterizes the dimension of the algebra. The type
# #     variable K parameterizes the field of scalars.
# #     """

# #     def __init__(self, dim: N, dtype: Type[K],
# #                  zero: Optional[Callable[[], V]] = None,
# #                  unit: Optional[Callable[[], V]] = None):
# #         self.dim = dim
# #         self.scl = ScalarField(dtype)
# #         self.add: Callable[[V, V], V] = jnp.add
# #         self.neg: Callable[[V], V] = neg
# #         self.sub: Callable[[V, V], V] = jnp.subtract
# #         self.smul: Callable[[S, V], V] = jnp.multiply
# #         self.mul: Callable[[V, V], V] = jnp.multiply
# #         self.inv: Callable[[V], V] = make_inv(dim, dtype)
# #         self.div: Callable[[V, V], V] = jnp.divide
# #         self.adj: Callable[[V], V] = jnp.conjugate
# #         self.lmul: Callable[[V, V], V] = jnp.multiply
# #         self.ldiv: Callable[[V, V], V] = ldiv
# #         self.rmul: Callable[[V, V], V] = jnp.multiply
# #         self.rdiv: Callable[[V, V], V] = jnp.divide
# #         self.sqrt: Callable[[V], V] = jnp.sqrt
# #         self.exp: Callable[[V], V] = jnp.exp
# #         self.power: Callable[[V, V], V] = jnp.power
# #         self.norm: Callable[[V], S] = linf_norm

# #         if zero is None:
# #             self.zero: Callable[[], V] = make_zero(dim, dtype)
# #         else:
# #             self.unit = zero

# #         if unit is None:
# #             self.unit: Callable[[], V] = make_unit(dim, dtype)
# #         else:
# #             self.unit = unit


# # class LInfFnAlgebra(LInfVectorAlgebra[N, K],
# #                     alg.ImplementsMeasureFnAlgebra[T, V, S],
# #                     Generic[T, N, K]):
# #     """Implement operations on equivalence classes of functions using JAX arrays
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

# pyright: basic

# TODO: Make power have a consistent signature.

import jax
import jax.numpy as jnp
import nlsa.abstract_algebra2 as alg
from jax import Array, jit, pmap, vmap
from jax.experimental.shard_map import shard_map as shmap
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.scipy.signal import convolve
from nlsa.jax.scalar_algebra import ScalarField
from nlsa.function_algebra2 import FunctionSpace, compose, make_mpower
from typing import Callable, Generic, Literal, Optional, Type, TypeAlias, \
        TypeVar

N = TypeVar('N', bound=int)
K = TypeVar('K', jnp.float32, jnp.float64)
R: TypeAlias = K
S = Array
Ss = Array
V = Array
Vs = Array
X = Array
Xs = Array
Y = Array
T = TypeVar('T')
T1 = TypeVar('T1')
T2 = TypeVar('T2')
F = Callable[[T1], T2]
ConvMode = TypeVar('ConvMode', bound=Literal['full', 'same', 'valid'])


def neg(v: V, /) -> V:
    """Negate a vector."""
    return jnp.multiply(-1, v)


def make_zero(dim: int | tuple[int], dtype: Type[K]) -> Callable[[], V]:
    """Make constant function returning vector of all 0s."""
    def zero() -> V:
        return jnp.zeros(dim, dtype=dtype)
    return zero


def make_unit(dim: int | tuple[int], dtype: Type[K]) -> Callable[[], V]:
    """Make constant function returning vector of all 1s."""
    def unit() -> V:
        return jnp.ones(dim, dtype=dtype)
    return unit


def make_inv(unit: V) -> Callable[[V], V]:
    """Make inversion function."""
    def inv(v: V) -> V:
        return jnp.divide(unit, v)
    return inv


def ldiv(u: V, v: V, /) -> V:
    """Perform left module division as elementwise vector division."""
    return jnp.divide(v, u)


def l2_innerp(u: V, v: V, /) -> S:
    """Compute L2 inner product of two vectors."""
    return jnp.sum(jnp.conjugate(u) * v)


def linf_norm(u: V, /):
    """Compute L-infinity norm of a vector."""
    return jnp.max(jnp.abs(u))


def make_weighted_l2_innerp(w: V, /) -> Callable[[V, V], S]:
    """Make L2 inner procuct from weight vector."""
    def innerp(u: V, v: V, /) -> S:
        return jnp.sum(jnp.conjugate(u) * w * v)
    return innerp


def to_norm(inner: Callable[[V, V], S], /) -> Callable[[V], S]:
    """Make norm from inner product."""
    def norm(v: V, /) -> S:
        return jnp.sqrt(inner(v, v))
    return norm


def to_sqnorm(inner: Callable[[V, V], S], /) -> Callable[[V], S]:
    """Make square norm from inner product."""
    def sqnorm(v: V, /) -> S:
        return inner(v, v)
    return sqnorm


make_weighted_l2_sqnorm = compose(to_sqnorm, make_weighted_l2_innerp)


def make_convolution(mode: ConvMode = 'same') -> Callable[[V, V], V]:
    """Make convolution product between vectors."""
    def cnv(u: V, v: V) -> V:
        return convolve(u, v, mode=mode)
    return cnv


def make_weighted_convolution(weight: V, mode: ConvMode = 'same') \
        -> Callable[[V, V], V]:
    """Make weighted convolution product between vectors."""
    def cnv(u: V, v: V) -> V:
        return convolve(weight * u, weight * v, mode=mode) / weight
    return cnv


def counting_measure(v: V, /) -> S:
    """Sum the elements of a vector."""
    return jnp.sum(v)
    # return jnp.sum(v, axis=-1)


def make_normalized_counting_measure(n: int) -> Callable[[V], S]:
    """Make normalized counting measure from dimension parameter."""
    def mu(v: V, /) -> S:
        return counting_measure(v) / float(n)
    return mu


def eval_at(xs: Xs, /) -> Callable[[F[Xs, V]], V]:
    """Make evaluation functional."""
    def eval(f: F[Xs, V]) -> V:
        return f(xs)
    return eval


def veval_at(xs: Xs, /) -> Callable[[F[X, Y]], V]:
    """Make vectorized evaluation functional."""
    def eval(f: F[X, Y]) -> V:
        g: Callable[[Xs], V] = vmap(f, in_axes=0)
        return g(xs)
    return eval


def sheval_at(xs: Xs, /, axis_name: str = 'i') -> Callable[[F[X, Y]], V]:
    """Make sharded evaluation functional."""
    devices = jax.local_devices()
    mesh = Mesh(devices, axis_names=(axis_name))

    def eval(f: F[X, Y]) -> V:
        g: Callable[[Xs], V] = shmap(vmap(f), mesh=mesh,
                                     in_specs=P(axis_name, None),
                                     out_specs=P(axis_name))
        return g(xs)
    return eval


def jeval_at(xs: Xs, /, axis_name: str = 'i', devices=None) \
        -> Callable[[F[X, Y]], V]:
    """Make doubly-vectorized and jitted evaluation functional."""
    if devices is None:
        devices = jax.local_devices()

    mesh = Mesh(devices, axis_names=(axis_name))
    # xs_sharding = NamedSharding(mesh, P(axis_name, None, None))
    ys_sharding = NamedSharding(mesh, P(axis_name, None))

    def eval(f: F[X, Y]) -> V:
        g: Callable[[Xs], V] = vmap(vmap(f), axis_name=axis_name)

        @jit
        def evalg(xss: Xs) -> V:
            # xss = jax.lax.with_sharding_constraint(xss, xs_sharding)
            ys = jax.lax.with_sharding_constraint(g(xss), ys_sharding)
            return ys

        return evalg(xs)
    return eval


def v2eval_at(xs: Xs, /, in_axes=0, axis_name=None) -> Callable[[F[X, Y]], V]:
    """Make doubly-vectorized evaluation functional."""
    def eval(f: F[X, Y]) -> V:
        g: Callable[[Xs], V] = vmap(vmap(f),
                                    in_axes=in_axes,
                                    axis_name=axis_name)
        return g(xs)
    return eval


def peval_at(xs: Xs, /, in_axes=0, axis_name=None) -> Callable[[F[X, Y]], V]:
    """Make parallelized evaluation functional."""
    def eval(f: F[X, Y]) -> V:
        g: Callable[[Xs], V] = pmap(vmap(f))
        return g(xs)
    return eval


def flip_conj(v: V) -> V:
    """Perform involution (complex-conjugation and flip) convolution algebra."""
    return jnp.conjugate(jnp.flip(v))


def sqeuclidean(u: V, v: V, /) -> S:
    """Compute pairwise squared Euclidean distance."""
    s2 = jnp.sum((u - v) ** 2)
    return s2


def make_fn_synthesis_operator(basis: F[X, V]) -> Callable[[V], F[X, S]]:
    """Make synthesis operator for functions from basis."""
    def synth(v: V) -> F[X, S]:
        def f(x: X) -> S:
            b = basis(x)
            return jnp.sum(v * b)
        return f
    return synth


def make_vector_synthesis_operator(basis: Vs) -> Callable[[Ss], V]:
    """Make synthesis operator for vectors from basis."""
    def synth(c: Ss) -> V:
        return basis @ c
    return synth


def fn_synthesis(basis: F[X, V], v: V) -> F[X, S]:
    """Perform function synthesis from basis."""
    def f(x: X) -> S:
        b = basis(x)
        return jnp.sum(v * b)
    return f


class VectorAlgebra(alg.ImplementsHilbertSpace[V, K], Generic[N, K]):
    """Implement vector algebra operations for JAX arrays.

    The type variable N parameterizes the dimension of the algebra. The type
    variable K parameterizes the field of scalars.

    The class constructor takes in the zero and unit elements of the algebra as
    optional arguments. This is to allow the use of sharded arrays.
    """

    # TODO: This class seems to obfuscate L2 Hilbert space and L\infty algebra.
    # It might be better to split into two classes, one that implements L2
    # without algebra operations, and one that implements L\infty without inner
    # product. The L\infty class could then act on L2 as a module.

    def __init__(self, dim: N, dtype: Type[K],
                 zero: Optional[Callable[[], V]] = None,
                 unit: Optional[Callable[[], V]] = None,
                 weight: Optional[V] = None):
        self.dim = dim
        self.scl = ScalarField(dtype)
        self.add: Callable[[V, V], V] = jnp.add
        self.neg: Callable[[V], V] = neg
        self.sub: Callable[[V, V], V] = jnp.subtract
        self.smul: Callable[[S, V], V] = jnp.multiply
        self.mul: Callable[[V, V], V] = jnp.multiply
        self.div: Callable[[V, V], V] = jnp.divide
        self.star: Callable[[V], V] = jnp.conjugate
        self.lmul: Callable[[V, V], V] = jnp.multiply
        self.ldiv: Callable[[V, V], V] = ldiv
        self.rmul: Callable[[V, V], V] = jnp.multiply
        self.rdiv: Callable[[V, V], V] = jnp.divide
        self.sqrt: Callable[[V], V] = jnp.sqrt
        self.exp: Callable[[V], V] = jnp.exp
        self.power: Callable[[V, V], V] = jnp.power

        if zero is None:
            self.zero: Callable[[], V] = make_zero(dim, dtype)
        else:
            self.unit = zero

        if unit is None:
            self.unit: Callable[[], V] = make_unit(dim, dtype)
        else:
            self.unit = unit

        self.inv: Callable[[V], V] = make_inv(self.unit)

        if weight is None:
            self.innerp: Callable[[V, V], S] = l2_innerp
        else:
            # TODO: Add check that weight has the right shape.
            self.innerp: Callable[[V, V], S] = make_weighted_l2_innerp(weight)

        self.norm: Callable[[V], S] = to_norm(self.innerp)


# TODO: Consider renaming this L1ConvolutionAlgebra and equip with L1 norm.
class ConvolutionAlgebra(alg.ImplementsHilbertSpace[V, K], Generic[N, K]):
    """Implement convolution algebra operations for JAX arrays.

    The type variable N parameterizes the dimension of the algebra. The type
    variable K parameterizes the field of scalars.

    The class constructor takes in the zero element of the algebra as an
    optional argument. This is to allow the use of sharded arrays.
    """

    def __init__(self, dim: N, dtype: Type[K],
                 zero: Optional[Callable[[], V]] = None,
                 weight: Optional[V] = None,
                 conv_mode: Optional[ConvMode] = 'same',
                 conv_weight: Optional[V] = None):
        self.dim = dim
        self.scl = ScalarField(dtype)
        self.add: Callable[[V, V], V] = jnp.add
        self.neg: Callable[[V], V] = neg
        self.sub: Callable[[V, V], V] = jnp.subtract
        self.smul: Callable[[S, V], V] = jnp.multiply

        if zero is None:
            self.zero: Callable[[], V] = make_zero(dim, dtype)
        else:
            self.unit = zero

        if conv_weight is None:
            self.mul = make_convolution(mode=conv_mode)
        else:
            self.mul = make_weighted_convolution(mode=conv_mode,
                                                 weight=conv_weight)

        self.star: Callable[[V], V] = flip_conj
        self.sqrt: Callable[[V], V] = jnp.sqrt
        self.exp: Callable[[V], V] = jnp.exp
        self.power: Callable[[V, V], V] = make_mpower(self.mul)

        if weight is None:
            self.innerp: Callable[[V, V], S] = l2_innerp
        else:
            self.innerp: Callable[[V, V], S] = make_weighted_l2_innerp(weight)

        self.norm: Callable[[V], S] = to_norm(self.innerp)


class MeasurableFnAlgebra(VectorAlgebra[N, K],
                          alg.ImplementsMeasurableFnAlgebra[T, V, S],
                          Generic[T, N, K]):
    """Implement operations on equivalence classes of functions using JAX arrays
    as the representation type.
    """
    def __init__(self, dim: N, dtype: Type[K],
                 inclusion_map: Callable[[F[T, S]], V],
                 zero: Optional[Callable[[], V]] = None,
                 unit: Optional[Callable[[], V]] = None,
                 weight: Optional[V] = None):
        super().__init__(dim, dtype, zero=zero, unit=unit, weight=weight)
        self.incl: Callable[[F[T, S]], V] = inclusion_map


# TODO: Inheritance from VectorAlgebra can lead to inconsistency between inner
# product and integration.
class MeasureFnAlgebra(MeasurableFnAlgebra[T, N, K],
                       alg.ImplementsMeasureFnAlgebra[T, V, S]):
    """Implement NPMeasurableFunctionAlgebra equipped with measure."""
    def __init__(self, dim: N, dtype: Type[K],
                 inclusion_map: Callable[[F[T, S]], V],
                 measure: Callable[[V], S],
                 zero: Optional[Callable[[], V]] = None,
                 unit: Optional[Callable[[], V]] = None,
                 weight: Optional[V] = None):
        super().__init__(dim, dtype, inclusion_map, zero=zero, unit=unit,
                         weight=weight)
        self.integrate: Callable[[V], S] = measure


class LInfVectorAlgebra(alg.ImplementsBanachAlgebra[V, K], Generic[N, K]):
    """Implement L infinity algebra operations for JAX arrays.

    The type variable N parameterizes the dimension of the algebra. The type
    variable K parameterizes the field of scalars.
    """

    def __init__(self, dim: N, dtype: Type[K],
                 zero: Optional[Callable[[], V]] = None,
                 unit: Optional[Callable[[], V]] = None):
        self.dim = dim
        self.scl = ScalarField(dtype)
        self.add: Callable[[V, V], V] = jnp.add
        self.neg: Callable[[V], V] = neg
        self.sub: Callable[[V, V], V] = jnp.subtract
        self.smul: Callable[[S, V], V] = jnp.multiply
        self.mul: Callable[[V, V], V] = jnp.multiply
        self.inv: Callable[[V], V] = make_inv(dim, dtype)
        self.div: Callable[[V, V], V] = jnp.divide
        self.star: Callable[[V], V] = jnp.conjugate
        self.lmul: Callable[[V, V], V] = jnp.multiply
        self.ldiv: Callable[[V, V], V] = ldiv
        self.rmul: Callable[[V, V], V] = jnp.multiply
        self.rdiv: Callable[[V, V], V] = jnp.divide
        self.sqrt: Callable[[V], V] = jnp.sqrt
        self.exp: Callable[[V], V] = jnp.exp
        self.power: Callable[[V, V], V] = jnp.power
        self.norm: Callable[[V], S] = linf_norm

        if zero is None:
            self.zero: Callable[[], V] = make_zero(dim, dtype)
        else:
            self.unit = zero

        if unit is None:
            self.unit: Callable[[], V] = make_unit(dim, dtype)
        else:
            self.unit = unit


class LInfFnAlgebra(LInfVectorAlgebra[N, K],
                    alg.ImplementsMeasureFnAlgebra[T, V, S],
                    Generic[T, N, K]):
    """Implement operations on equivalence classes of functions using JAX arrays
    as the representation type.
    """

    def __init__(self, dim: N, dtype: Type[K],
                 inclusion_map: Callable[[F[T, S]], V],
                 measure: Callable[[V], S],
                 zero: Optional[Callable[[], V]] = None,
                 unit: Optional[Callable[[], V]] = None):
        super().__init__(dim, dtype, zero=zero, unit=unit)
        self.incl: Callable[[F[T, S]], V] = inclusion_map
        self.integrate: Callable[[V], S] = measure


class FnSynthesis(FunctionSpace[T, VectorAlgebra[N, K]], Generic[T, N, K]):
    """Implement function synthesis from coefficients in JAX array."""
    def __init__(self, dim: N, dtype: Type[K]):
        self.hilb1: VectorAlgebra[N, K] = VectorAlgebra(dim=dim, dtype=dtype)
        self.hilb2: FunctionSpace[T, K] = \
            FunctionSpace(codomain=ScalarField(dtype))
        super().__init__(codomain=FunctionSpace(codomain=self.hilb1))
        self.app = fn_synthesis()


def  make_vector_analysis_operator(vec: VectorAlgebra[N, K], basis: Vs,
                                  axis: Optional[int] = None) \
        -> Callable[[V], V]:
    """Make analysis operator from an array of vectors"""
    if axis is None:
        axis = -1

    vinnerp = vmap(vec.innerp, in_axes=(axis, None))

    def an(v: V) -> Ss:
        return vinnerp(basis, v)
    return an


# Old methods of ScalarField class

# self._unit: Callable[[], S] = make_sunit(dtype)

# def add(self, u: S, v: S, /) -> S:
    # """Add two scalars."""
    # return jnp.add(u, v)

# def sub(self, u: S, v: S, /) -> S:
    # """Subtract two scalars."""
    # return jnp.subtract(u, v)

# def neg(self, v: S, /) -> S:
    # """Negate a scalar."""
    # return jnp.multiply(-1, v)

# def mul(self, u: S, v: S, /) -> S:
    # """Multiply two scalars."""
    # return jnp.multiply(u, v)

# def inv(self, v: S, /) -> S:
    # """Invert a scalar."""
    # return jnp.divide(self.unit(), v)

# def div(self, u: S, v: S, /) -> S:
    # """Divide two scalars."""
    # return jnp.divide(u, v)

# def sqrt(self, v: S, /) -> S:
    # """Compute square root of scalar."""
    # return jnp.sqrt(v)

# def exp(self, v: S, /) -> S:
    # """Compute exponential function on scalar."""
    # return jnp.exp(v)

# def power(self, v: S, k: S, /) -> S:
    # """Compute exponentiation of scalar by scalar."""
    # return jnp.power(v, k)

# def star(self, v: S, /) -> S:
    # """Compute complex conjugation of scalar."""
    # return jnp.conjugate(v)

# def unit(self) -> S:
    # return self._unit()


# Old methods of VectorAlgebra class

#    def __init__(self, dim: N, dtype: Type[K]):
#        self._unit: Callable[[], V] = make_unit(dim, dtype)
#        self.scl: alg.ImplementsScalarField[S] = ScalarField(dtype)

#    def add(self, u: V, v: V, /) -> V:
#        """Add two vectors."""
#        return jnp.add(u, v)

#    def sub(self, u: V, v: V, /) -> V:
#        """Subtract two vectors."""
#        return jnp.subtract(u, v)

#    def neg(self, v: V, /) -> V:
#        """Negate a vector."""
#        return jnp.multiply(-1, v)

#    def smul(self, k: S, v: V, /) -> V:
#        """Multiply a scalar and a vector."""
#        return jnp.multiply(k, v)

#    def mul(self, u: V, v: V, /) -> V:
#        """Multiply two vectors elementwise."""
#        return jnp.multiply(u, v)

#    def inv(self, v: V, /) -> V:
#        """Invert a vector elementwise."""
#        return jnp.divide(self.unit(), v)

#    def unit(self) -> V:
#        return self._unit()

#    def div(self, u: V, v: V, /) -> V:
#        """Divide two vectors elementwise."""
#        return jnp.divide(u, v)

#    def lmul(self, u: V, v: V, /) -> V:
#        """Perform left module multiplication as elementwise vector
#        multiplication.

#        """
#        return jnp.multiply(u, v)

#    def ldiv(self, u: V, v: V, /) -> V:
#        """Perform left module division as elementwise vector division."""
#        return jnp.divide(v, u)

#    def rmul(self, u: V, v: V, /) -> V:
#        """Perform right module multiplication as elementwise vector
#        multiplication.

#        """
#        return jnp.multiply(u, v)

#    def rdiv(self, u: V, v: V, /) -> V:
#        """Perform right module division as elementwise vector division."""
#        return jnp.divide(u, v)

#    def sqrt(self, v: V, /) -> V:
#        """Compute elementwise square root of vector."""
#        return jnp.sqrt(v)

#    def exp(self, v: V, /) -> V:
#        """Compute elementwise exponential function of vector."""
#        return jnp.exp(v)

#    def power(self, v: V, k: S, /) -> V:
#        """Compute elementwise exponentiation of vector by scalar."""
#        return jnp.power(v, k)

#    def star(self, v: V, /) -> V:
#        """Compute elementwise complex conjugation of vector."""
#        return jnp.conjugate(v)

#    def innerp(self, u: V, v: V, /) -> S:
#        """Compute inner product of vectors."""
#        w = jnp.sum(jnp.multiply(jnp.conjugate(u), v), axis=-1)
#        return w

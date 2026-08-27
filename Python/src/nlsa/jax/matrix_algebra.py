# pyright: basic
"""Implement matrix algebra operations for JAX arrays."""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import nlsa.abstract_algebra as alg
import nlsa.jax.scalars as scls
import nlsa.jax.vector_algebra as vec
import numpy as np
import scipy.linalg as la
from dataclasses import dataclass
from functools import partial
from itertools import chain
from jax import Array
from jax.sharding import NamedSharding, Sharding
from jax.typing import DTypeLike
from nlsa.jax.sharding import shardit
from nlsa.jax.vector_algebra import L2VectorAlgebra
from nlsa.jax.utils import batch_map
from nlsa.typing import DEFAULT
from typing import final
from collections.abc import Callable

type K = Array
type A = Array
type Av = Array
type Aw = Array
type V = Array
type W = Array
type Shape = tuple[int, ...]


def neg(a: A, /) -> A:
    """Negate a matrix."""
    return -a


def make_zero(
    shape: tuple[Shape, Shape],
    dtype: DTypeLike,
    sharding: Sharding | None = None,
) -> Callable[[], A]:
    """Make zero matrix."""

    def zero() -> A:
        return jnp.zeros(
            tuple(chain.from_iterable(shape)), dtype=dtype, device=sharding
        )

    return zero


def make_unit(
    shape: tuple[Shape, Shape],
    dtype: DTypeLike,
    sharding: Sharding | None = None,
) -> Callable[[], A]:
    """Make unit matrix."""

    def unit() -> A:
        return jnp.eye(
            tuple(chain.from_iterable(shape)), dtype=dtype, device=sharding
        )

    return unit


def divm(a: A, b: A, /) -> A:
    """Compute matrix division."""
    return jnp.matmul(a, jla.inv(b))


def sdiv(s: K, a: A, /) -> A:
    """Perform scalar division of a matrix."""
    return jnp.divide(a, s)


def ldivm(a: A, b: A, /) -> A:
    """Perform left module division as matrix division."""
    return jnp.matmul(jla.inv(a), b)


def adjm(a: A, /) -> A:
    """Compute complex conjugate transpose of matrix."""
    return jnp.conjugate(a.T)


def absm(a: A, /) -> A:
    """Compute modulus of matrix."""
    return jla.sqrtm(adjm(a) @ a)


def logm(a: A, /) -> A:
    """Compute matrix logarithm.

    WARNING: This function uses scipy to compute the matrix logarithm as there
    is no corresponding JAX implementation.
    """
    return jnp.array(la.logm(np.asarray(a)))


def matrix_scalar_power(a: A, k: K, /) -> A:
    """Exponentiate a matrix by a scalar."""
    return jla.expm(k * logm(a))


def b2_innerp(a: A, b: A, /) -> K:
    """Compute Hilbert-Schmidt (B2) inner product of two matrices."""
    return jnp.sum(jnp.conjugate(a) * b)


def _make_weighted_b2_innerp_scalar(w: V, /) -> Callable[[A, A], K]:
    """Make weighted Hilbert-Schmidt inner procuct from scalar weight."""

    def innerp(a: A, b: A, /) -> K:
        return jnp.sum(jnp.conjugate(a) * w * b)

    return innerp


def _make_weighted_b2_innerp_vector(w: V, /) -> Callable[[A, A], K]:
    """Make weighted Hilbert-Schmidt inner procuct from weight vector."""

    def innerp(a: A, b: A, /) -> K:
        return jnp.sum(jnp.conjugate(a) * w[:, jnp.newaxis] * b)

    return innerp


def _make_weighted_b2_innerp_matrix(w: A, /) -> Callable[[A, A], K]:
    """Make weighted Hilbert-Schmidt inner procuct from weight matrix."""

    def innerp(a: A, b: A, /) -> K:
        return jnp.sum(jnp.conjugate(a) * jnp.matmul(w[:, jnp.newaxis], b))

    return innerp


def make_weighted_b2_innerp(w: A, /) -> Callable[[A, A], K]:
    """Make weighted Hilbert-Schmidt inner product."""
    match len(w.shape):
        case 0:
            innerp = _make_weighted_b2_innerp_scalar(w)
        case 1:
            innerp = _make_weighted_b2_innerp_vector(w)
        case _:
            innerp = _make_weighted_b2_innerp_matrix(w)
    return innerp


def to_norm(innerp: Callable[[A, A], K], /) -> Callable[[A], K]:
    """Make norm from inner product."""

    def norm(v: V, /) -> K:
        return jnp.sqrt(innerp(v, v))

    return norm


def materialize_in_std_basis(
    f: Callable[[V], V],
    in_dim: int,
    basis_vec_value: float | Array = 1,
    dtype: DTypeLike | None = None,
    batch_size: int | None = None,
    in_sharding: NamedSharding | None = None,
    out_sharding: NamedSharding | None = None,
    jit: bool = False,
) -> A:
    """Compute matrix representation of linear map in standard basis of C^n."""
    basis = vec.make_one_hot_basis(
        dim=in_dim, value=basis_vec_value, dtype=dtype, sharding=in_sharding
    )

    @partial(shardit, sharding=out_sharding)
    @partial(batch_map, out_axis=1, batch_size=batch_size)
    def cols(j: Array) -> V:
        return f(basis(j))

    # if out_sharding is not None:
    #     out_shard = partial(
    #         jax.lax.with_sharding_constraint, shardings=out_sharding
    #     )
    #     cols = fun.compose(out_shard, cols)
    if jit:
        cols = jax.jit(cols)

    return cols(jnp.arange(in_dim, out_sharding=in_sharding))


@final
@dataclass(frozen=True, slots=True)
class HilbertSchmidtMatrixAlgebra[N: int, D: DTypeLike](
    alg.ImplementsInnerProductOperatorStarAlgebraWithCalculus[A, V, K],
    alg.ImplementsDivBimodule[A, K, A, A],
):
    """Implement Hilbert-Schmidt matrix algebra operations for JAX arrays."""

    dtype: D
    domain: L2VectorAlgebra[tuple[N], D]
    weight: A | None
    sharding: Sharding | None
    _scl: alg.ImplementsComplexScalarField[K]
    _zero: Callable[[], A]
    _unit: Callable[[], A]
    _add: Callable[[A, A], A]
    _neg: Callable[[A], A]
    _sub: Callable[[A, A], A]
    _sdiv: Callable[[K, A], A]
    _smul: Callable[[K, A], A]
    _mul: Callable[[A, A], A]
    _inv: Callable[[A], A]
    _div: Callable[[A, A], A]
    _adj: Callable[[A], A]
    _lmul: Callable[[A, A], A]
    _ldiv: Callable[[A, A], A]
    _rmul: Callable[[A, A], A]
    _rdiv: Callable[[A, A], A]
    _sqrt: Callable[[A], A]
    _exp: Callable[[A], A]
    _log: Callable[[A], A]
    _mpower: Callable[[A, int], A]
    _power: Callable[[A, K], A]
    _innerp: Callable[[A, A], K]
    _norm: Callable[[A], K]
    _abs: Callable[[A], A]
    _app: Callable[[A, V], V]

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of HilbertSchmidtMatrixAlgebra object."""
        return (self.domain.dim, self.domain.dim)

    @property
    def dim(self) -> int:
        """Vector space dimension of HilbertSchmidtMatrixAlgebra object."""
        return self.domain.dim**2

    @property
    def scl(self) -> alg.ImplementsComplexScalarField[K]:
        """Scalar field associated with HilbertSchmidtMatrixAlgebra object."""
        return self._scl

    @property
    def dom(self) -> L2VectorAlgebra[tuple[N], D]:
        """Vector space domain of HilbertSchmidtMatrixAlgebra object."""
        return self.domain

    @property
    def codom(self) -> L2VectorAlgebra[tuple[N], D]:
        """Vector space codomain of HilbertSchmidtMatrixAlgebra object."""
        return self.domain

    def zero(self, /) -> A:
        """Return zero matrix."""
        return self._zero()

    def add(self, a: A, b: A, /) -> A:
        """Add two matrices."""
        return self._add(a, b)

    def neg(self, a: A, /) -> A:
        """Compute additive inverse (negation) of a matrix."""
        return self._neg(a)

    def sub(self, a: A, b: A, /) -> A:
        """Subtract two matrices."""
        return self._sub(a, b)

    def smul(self, k: K, a: A, /) -> A:
        """Multiply a matrix by a scalar."""
        return self._smul(k, a)

    def sdiv(self, k: K, a: A, /) -> A:
        """Divide a matrix by a scalar."""
        return self._sdiv(k, a)

    def unit(self, /) -> A:
        """Return vector with elements equal to 1 (multiplicative unit)."""
        return self._unit()

    def mul(self, a: A, b: A, /) -> A:
        """Compute elementwise multiplication of two vectors."""
        return self._mul(a, b)

    def div(self, a: A, b: A, /) -> A:
        """Compute elementwise division of two vectors."""
        return self._div(a, b)

    def inv(self, a: A, /) -> A:
        """Compute elementwise multiplicative inverse of a matrix."""
        return self._inv(a)

    def adj(self, a: A, /) -> A:
        """Compute elementwise complex conjugate of a matrix."""
        return self._adj(a)

    def sqrt(self, a: A, /) -> A:
        """Compute elementwise square root of a matrix."""
        return self._sqrt(a)

    def abs(self, a: A, /) -> A:
        """Compute elementwise modulus of a matrix."""
        return self._abs(a)

    def exp(self, a: A, /) -> A:
        """Compute elementwise exponential of a matrix."""
        return self._exp(a)

    def log(self, a: A, /) -> A:
        """Compute elementwise natural logarithm of a matrix."""
        return self._log(a)

    def mpower(self, a: A, m: int, /) -> A:
        """Compute elementwise exponentiation of a matrix by an integer."""
        return self._mpower(a, m)

    def power(self, a: A, k: K, /) -> A:
        """Compute elementwise exponentiation of a matrix by a scalar."""
        return self._power(a, k)

    def innerp(self, a: A, b: A, /) -> K:
        """Compute the inner product between two matrices."""
        return self._innerp(a, b)

    def norm(self, v: A, /) -> K:
        """Compute the norm of a matrix."""
        return self._norm(v)

    def app(self, a: A, v: V, /) -> V:
        """Compute application of matrix to vector as a linear operator."""
        return self._app(a, v)

    def lmul(self, a: A, b: A, /) -> A:
        """Compute left module multiplication of two matrices."""
        return self._lmul(a, b)

    def ldiv(self, a: A, b: A, /) -> A:
        """Compute left module division of two matrices."""
        return self._ldiv(a, b)

    def rmul(self, a: A, b: A, /) -> A:
        """Compute right module multiplication of two matrices."""
        return self._rmul(a, b)

    def rdiv(self, a: A, b: A, /) -> A:
        """Compute right module division of two matrices."""
        return self._rdiv(a, b)


def hilbert_schmidt_matrix_algebra[N: int, D: DTypeLike](
    dtype: D,
    domain: L2VectorAlgebra[tuple[N], D],
    weight: A | None = None,
    sharding: Sharding | None = None,
    scl: alg.ImplementsComplexScalarField[K] | DEFAULT = DEFAULT,
    zero: Callable[[], V] | DEFAULT = DEFAULT,
    unit: Callable[[], V] | DEFAULT = DEFAULT,
    add: Callable[[A, A], A] | DEFAULT = DEFAULT,
    neg: Callable[[A], A] | DEFAULT = DEFAULT,
    sub: Callable[[A, A], A] | DEFAULT = DEFAULT,
    sdiv: Callable[[K, A], A] | DEFAULT = DEFAULT,
    smul: Callable[[K, A], A] | DEFAULT = DEFAULT,
    mul: Callable[[A, A], A] | DEFAULT = DEFAULT,
    inv: Callable[[A], A] | DEFAULT = DEFAULT,
    div: Callable[[A, A], A] | DEFAULT = DEFAULT,
    adj: Callable[[A], A] | DEFAULT = DEFAULT,
    lmul: Callable[[A, A], A] | DEFAULT = DEFAULT,
    ldiv: Callable[[A, A], A] | DEFAULT = DEFAULT,
    rmul: Callable[[A, A], A] | DEFAULT = DEFAULT,
    rdiv: Callable[[A, A], A] | DEFAULT = DEFAULT,
    sqrt: Callable[[A], A] | DEFAULT = DEFAULT,
    exp: Callable[[A], A] | DEFAULT = DEFAULT,
    log: Callable[[A], A] | DEFAULT = DEFAULT,
    mpower: Callable[[A, int], A] | DEFAULT = DEFAULT,
    power: Callable[[A, K], A] | DEFAULT = DEFAULT,
    innerp: Callable[[A, A], K] | DEFAULT = DEFAULT,
    norm: Callable[[A], K] | DEFAULT = DEFAULT,
    abs: Callable[[A], A] | DEFAULT = DEFAULT,
    app: Callable[[A, V], V] | DEFAULT = DEFAULT,
) -> HilbertSchmidtMatrixAlgebra[N, D]:
    """Build HilbertSchmidtMatrixAlgebra object."""
    shape = (domain.shape, domain.shape)
    if innerp is not DEFAULT:
        _innerp = innerp
    elif weight is not None:
        _innerp = make_weighted_b2_innerp(weight)
    else:
        _innerp = b2_innerp
    return HilbertSchmidtMatrixAlgebra(
        dtype=dtype,
        domain=domain,
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
        _mul=jnp.matmul if mul is DEFAULT else mul,
        _div=divm if div is DEFAULT else div,
        _inv=jla.inv if inv is DEFAULT else inv,
        _adj=adjm if adj is DEFAULT else adj,
        _sqrt=jla.expm if sqrt is DEFAULT else sqrt,
        _abs=absm if abs is DEFAULT else abs,
        _exp=jla.expm if exp is DEFAULT else exp,
        _log=logm if log is DEFAULT else log,
        _mpower=jnp.linalg.matrix_power if mpower is DEFAULT else mpower,
        _power=matrix_scalar_power if power is DEFAULT else power,
        _app=jnp.matmul if app is DEFAULT else app,
        _innerp=_innerp,
        _norm=to_norm(_innerp) if norm is DEFAULT else norm,
        _lmul=jnp.matmul if lmul is DEFAULT else lmul,
        _ldiv=ldivm if ldiv is DEFAULT else ldiv,
        _rmul=jnp.matmul if rmul is DEFAULT else rmul,
        _rdiv=divm if rdiv is DEFAULT else rdiv,
    )


@final
@dataclass(frozen=True, slots=True)
class HilbertSchmidtMatrixSpace[M: int, N: int, D: DTypeLike](
    alg.ImplementsInnerProductOperatorSystem[A, V, W, K],
    alg.ImplementsDivBimodule[A, K, Aw, Av],
):
    """Implement matrix space operations for JAX arrays."""

    dtype: D
    domain: L2VectorAlgebra[tuple[N], D]
    codomain: L2VectorAlgebra[tuple[M], D]
    weight: A | None
    sharding: Sharding | None
    _scl: alg.ImplementsComplexScalarField[K]
    _zero: Callable[[], A]
    _unit: Callable[[], A]
    _add: Callable[[A, A], A]
    _neg: Callable[[A], A]
    _sub: Callable[[A, A], A]
    _sdiv: Callable[[K, A], A]
    _smul: Callable[[K, A], A]
    _adj: Callable[[A], A]
    _lmul: Callable[[A, A], A]
    _ldiv: Callable[[A, A], A]
    _rmul: Callable[[A, A], A]
    _rdiv: Callable[[A, A], A]
    _innerp: Callable[[A, A], K]
    _norm: Callable[[A], K]
    _app: Callable[[A, V], W]

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of HilbertSchmidtMatrixAlgebra object."""
        return (self.domain.dim, self.domain.dim)

    @property
    def dim(self) -> int:
        """Vector space dimension of HilbertSchmidtMatrixAlgebra object."""
        return self.domain.dim**2

    @property
    def scl(self) -> alg.ImplementsComplexScalarField[K]:
        """Scalar field associated with HilbertSchmidtMatrixAlgebra object."""
        return self._scl

    @property
    def dom(self) -> L2VectorAlgebra[tuple[N], D]:
        """Vector space domain of HilbertSchmidtMatrixAlgebra object."""
        return self.domain

    @property
    def codom(self) -> L2VectorAlgebra[tuple[N], D]:
        """Vector space codomain of HilbertSchmidtMatrixAlgebra object."""
        return self.domain

    def zero(self, /) -> A:
        """Return zero matrix."""
        return self._zero()

    def add(self, a: A, b: A, /) -> A:
        """Add two matrices."""
        return self._add(a, b)

    def neg(self, a: A, /) -> A:
        """Compute additive inverse (negation) of a matrix."""
        return self._neg(a)

    def sub(self, a: A, b: A, /) -> A:
        """Subtract two matrices."""
        return self._sub(a, b)

    def smul(self, k: K, a: A, /) -> A:
        """Multiply a matrix by a scalar."""
        return self._smul(k, a)

    def sdiv(self, k: K, a: A, /) -> A:
        """Divide a matrix by a scalar."""
        return self._sdiv(k, a)

    def adj(self, a: A, /) -> A:
        """Compute elementwise complex conjugate of a matrix."""
        return self._adj(a)

    def innerp(self, a: A, b: A, /) -> K:
        """Compute the inner product between two matrices."""
        return self._innerp(a, b)

    def norm(self, v: A, /) -> K:
        """Compute the norm of a matrix."""
        return self._norm(v)

    def app(self, a: A, v: V, /) -> V:
        """Compute application of matrix to vector as a linear operator."""
        return self._app(a, v)

    def lmul(self, a: A, b: A, /) -> A:
        """Compute left module multiplication of two matrices."""
        return self._lmul(a, b)

    def ldiv(self, a: A, b: A, /) -> A:
        """Compute left module division of two matrices."""
        return self._ldiv(a, b)

    def rmul(self, a: A, b: A, /) -> A:
        """Compute right module multiplication of two matrices."""
        return self._rmul(a, b)

    def rdiv(self, a: A, b: A, /) -> A:
        """Compute right module division of two matrices."""
        return self._rdiv(a, b)


def hilbert_schmidt_matrix_space[M: int, N: int, D: DTypeLike](
    dtype: D,
    domain: L2VectorAlgebra[tuple[N], D],
    codomain: L2VectorAlgebra[tuple[M], D],
    weight: A | None = None,
    sharding: Sharding | None = None,
    scl: alg.ImplementsComplexScalarField[K] | DEFAULT = DEFAULT,
    zero: Callable[[], V] | DEFAULT = DEFAULT,
    unit: Callable[[], V] | DEFAULT = DEFAULT,
    add: Callable[[A, A], A] | DEFAULT = DEFAULT,
    neg: Callable[[A], A] | DEFAULT = DEFAULT,
    sub: Callable[[A, A], A] | DEFAULT = DEFAULT,
    sdiv: Callable[[K, A], A] | DEFAULT = DEFAULT,
    smul: Callable[[K, A], A] | DEFAULT = DEFAULT,
    adj: Callable[[A], A] | DEFAULT = DEFAULT,
    lmul: Callable[[A, A], A] | DEFAULT = DEFAULT,
    ldiv: Callable[[A, A], A] | DEFAULT = DEFAULT,
    rmul: Callable[[A, A], A] | DEFAULT = DEFAULT,
    rdiv: Callable[[A, A], A] | DEFAULT = DEFAULT,
    innerp: Callable[[A, A], K] | DEFAULT = DEFAULT,
    norm: Callable[[A], K] | DEFAULT = DEFAULT,
    app: Callable[[A, V], V] | DEFAULT = DEFAULT,
) -> HilbertSchmidtMatrixSpace[M, N, D]:
    """Build HilbertSchmidtMatrixSpace object."""
    shape = (codomain.shape, domain.shape)
    if innerp is not DEFAULT:
        _innerp = innerp
    elif weight is not None:
        _innerp = make_weighted_b2_innerp(weight)
    else:
        _innerp = b2_innerp
    return HilbertSchmidtMatrixSpace(
        dtype=dtype,
        domain=domain,
        codomain=codomain,
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
        _adj=adjm if adj is DEFAULT else adj,
        _app=jnp.matmul if app is DEFAULT else app,
        _innerp=_innerp,
        _norm=to_norm(_innerp) if norm is DEFAULT else norm,
        _lmul=jnp.matmul if lmul is DEFAULT else lmul,
        _ldiv=ldivm if ldiv is DEFAULT else ldiv,
        _rmul=jnp.matmul if rmul is DEFAULT else rmul,
        _rdiv=divm if rdiv is DEFAULT else rdiv,
    )

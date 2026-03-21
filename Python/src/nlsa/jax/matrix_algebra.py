# pyright: basic
"""Implement matrix algebra operations for JAX arrays."""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import nlsa.abstract_algebra as alg
import nlsa.jax.vector_algebra as vec
import numpy as np
import scipy.linalg as la
from dataclasses import InitVar, dataclass, field
from functools import partial
from itertools import chain
from jax import Array
from jax.sharding import NamedSharding, Sharding
from jax.typing import DTypeLike
from nlsa.jax.sharding import shardit
from nlsa.jax.vector_algebra import ScalarField, L2VectorAlgebra
from nlsa.jax.utils import batch_map
from typing import Callable, Optional, final

type K = Array
type A = Array
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


def div(a: A, b: A, /) -> A:
    """Compute matrix division."""
    return jnp.matmul(a, jla.inv(b))


def sdiv(s: K, a: A, /) -> A:
    """Perform scalar division of a matrix."""
    return jnp.divide(a, s)


def ldiv(a: A, b: A, /) -> A:
    """Perform left module division as matrix division."""
    return jnp.matmul(jla.inv(a), b)


def adj(a: A, /) -> A:
    """Compute complex conjugate transpose of matrix."""
    return jnp.conjugate(a.T)


def mod(a: A, /) -> A:
    """Compute modulus of matrix."""
    return jla.sqrtm(adj(a) @ a)


def power(a: A, k: K) -> A:
    """Compute (fractional) matrix power.

    WARNING: This function uses scipy to compute the matrix logarithm as there
    is no corresponding JAX implementation.
    """
    return jla.expm(k * jnp.array(la.logm(np.asarray(a))))


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
    dtype: Optional[DTypeLike] = None,
    batch_size: Optional[int] = None,
    in_sharding: Optional[NamedSharding] = None,
    out_sharding: Optional[NamedSharding] = None,
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


# TODO: Include ord parameter to specify operator norm.
@final
@dataclass(frozen=True)
class MatrixAlgebra[N: Shape, D: DTypeLike](
    alg.ImplementsOperatorAlgebraWithCalculus[A, V, K]
):
    """Implement matrix algebra operations for JAX arrays."""

    domain: L2VectorAlgebra[N, D]
    sharding: Optional[Sharding | None] = None
    _scl: Optional[ScalarField[D]] = None
    _zero: Optional[Callable[[], V]] = None
    _unit: Optional[Callable[[], V]] = None
    _codom: Optional[L2VectorAlgebra[N, D]] = None
    _add: Optional[Callable[[A, A], A]] = None
    _neg: Optional[Callable[[A], A]] = None
    _sub: Optional[Callable[[A, A], A]] = None
    _sdiv: Optional[Callable[[K, A], A]] = None
    _smul: Optional[Callable[[K, A], A]] = None
    _mul: Optional[Callable[[A, A], A]] = None
    _inv: Optional[Callable[[A], A]] = None
    _div: Optional[Callable[[A, A], A]] = None
    _adj: Optional[Callable[[A], A]] = None
    _lmul: Optional[Callable[[A, A], A]] = None
    _ldiv: Optional[Callable[[A, A], A]] = None
    _rmul: Optional[Callable[[A, A], A]] = None
    _rdiv: Optional[Callable[[A, A], A]] = None
    _sqrt: Optional[Callable[[A], A]] = None
    _exp: Optional[Callable[[A], A]] = None
    _mpower: Optional[Callable[[A, int], A]] = None
    _power: Optional[Callable[[A, K], A]] = None
    _norm: Optional[Callable[[A], K]] = None
    _mod: Optional[Callable[[A], A]] = None
    _app: Optional[Callable[[A, V], V]] = None

    @property
    def dom(self) -> L2VectorAlgebra[N, D]:
        """Return dom property of MatrixAlgebra object."""
        return self.domain

    @property
    def dtype(self) -> D:
        """Return dom property of MatrixAlgebra object."""
        return self.domain.dtype

    @property
    def shape(self) -> tuple[N, N]:
        """Return shape property of MatrixAlgebra object."""
        return (self.dom.shape, self.dom.shape)

    @property
    def scl(self) -> ScalarField[D]:
        """Return scl property of OperatorAlgebra object."""
        return self.domain.scl if self._scl is None else self._scl

    @property
    def codom(self) -> L2VectorAlgebra[N, D]:
        """Return codom property of OperatorAlgebra object."""
        return self.domain if self._codom is None else self._codom

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
    def add(self) -> Callable[[A, A], A]:
        """Return add property of MatrixAlgebra object."""
        return jnp.add if self._add is None else self._add

    @property
    def neg(self) -> Callable[[A], A]:
        """Return neg property of MatrixAlgebra object."""
        return neg if self._neg is None else self._neg

    @property
    def sub(self) -> Callable[[A, A], A]:
        """Return sub property of MatrixAlgebra object."""
        return jnp.subtract if self._sub is None else self._sub

    @property
    def sdiv(self) -> Callable[[K, A], A]:
        """Return sdiv property of MatrixAlgebra object."""
        return sdiv if self._sdiv is None else self._sdiv

    @property
    def smul(self) -> Callable[[K, A], A]:
        """Return smul property of MatrixAlgebra object."""
        return jnp.multiply if self._smul is None else self._smul

    @property
    def mul(self) -> Callable[[A, A], A]:
        """Return mul property of MatrixAlgebra object."""
        return jnp.matmul if self._mul is None else self._mul

    @property
    def inv(self) -> Callable[[A], A]:
        """Return inv property of MatrixAlgebra object."""
        return jla.inv if self._inv is None else self._inv

    @property
    def div(self) -> Callable[[A, A], A]:
        """Return div property of MatrixAlgebra object."""
        return div if self._div is None else self._div

    @property
    def adj(self) -> Callable[[A], A]:
        """Return adj property of MatrixAlgebra object."""
        return adj if self._adj is None else self._adj

    @property
    def lmul(self) -> Callable[[A, A], A]:
        """Return lmul property of MatrixAlgebra object."""
        return jnp.matmul if self._lmul is None else self._lmul

    @property
    def ldiv(self) -> Callable[[A, A], A]:
        """Return ldiv property of MatrixAlgebra object."""
        return ldiv if self._ldiv is None else self._ldiv

    @property
    def rmul(self) -> Callable[[A, A], A]:
        """Return rmul property of MatrixAlgebra object."""
        return jnp.matmul if self._rmul is None else self._rmul

    @property
    def rdiv(self) -> Callable[[A, A], A]:
        """Return rdiv property of MatrixAlgebra object."""
        return div if self._rdiv is None else self._rdiv

    @property
    def sqrt(self) -> Callable[[A], A]:
        """Return sqrt property of MatrixAlgebra object."""
        return jla.sqrtm if self._sqrt is None else self._sqrt

    @property
    def exp(self) -> Callable[[A], A]:
        """Return exp property of MatrixAlgebra object."""
        return jla.expm if self._exp is None else self._exp

    @property
    def mpower(self) -> Callable[[A, int], A]:
        """Return mpower property of MatrixAlgebra object."""
        return (
            jnp.linalg.matrix_power if self._mpower is None else self._mpower
        )

    @property
    def power(self) -> Callable[[A, K], A]:
        """Return power property of MatrixAlgebra object."""
        return power if self._power is None else self._power

    @property
    def norm(self) -> Callable[[A], K]:
        """Return norm property of MatrixAlgebra object."""
        return (
            partial(jnp.linalg.norm, ord=2)
            if self._norm is None
            else self._norm
        )

    @property
    def mod(self) -> Callable[[A], A]:
        """Return mod property of MatrixAlgebra object."""
        return mod if self._mod is None else self._mod

    @property
    def app(self) -> Callable[[A, V], V]:
        """Return app property of MatrixAlgebra object."""
        return jnp.matmul if self._app is None else self._app


@final
@dataclass(frozen=True)
class HilbertSchmidtMatrixAlgebra[N: Shape, D: DTypeLike](
    alg.ImplementsInnerProductOperatorAlgebraWithCalculus[A, V, K]
):
    """Implement Hilbert-Schmidt matrix algebra operations for JAX arrays."""

    dom: L2VectorAlgebra[N, D]
    shape: tuple[N, N] = field(init=False)
    dtype: D = field(init=False)
    scl: ScalarField[D] = field(init=False)
    zero: Callable[[], V] = field(init=False)
    unit: Callable[[], V] = field(init=False)
    codom: L2VectorAlgebra[N, D] = field(init=False)
    add: Callable[[A, A], A] = field(default=jnp.add)
    neg: Callable[[A], A] = field(default=neg)
    sub: Callable[[A, A], A] = field(default=jnp.subtract)
    sdiv: Callable[[K, A], A] = field(default=sdiv)
    smul: Callable[[K, A], A] = field(default=jnp.multiply)
    mul: Callable[[A, A], A] = field(default=jnp.matmul)
    inv: Callable[[A], A] = field(default=jla.inv)
    div: Callable[[A, A], A] = field(default=div)
    adj: Callable[[A], A] = field(default=adj)
    lmul: Callable[[A, A], A] = field(default=jnp.matmul)
    ldiv: Callable[[A, A], A] = field(default=ldiv)
    rmul: Callable[[A, A], A] = field(default=jnp.matmul)
    rdiv: Callable[[A, A], A] = field(default=div)
    sqrt: Callable[[A], A] = field(default=jla.sqrtm)
    exp: Callable[[A], A] = field(default=jla.expm)
    mpower: Callable[[A, int], A] = field(default=jnp.linalg.matrix_power)
    power: Callable[[A, K], A] = field(default=power)
    innerp: Callable[[A, A], K] = field(init=False)
    norm: Callable[[A], K] = field(init=False)
    mod: Callable[[A], A] = field(default=mod)
    app: Callable[[A, V], V] = field(default=jnp.matmul)
    sharding: InitVar[Sharding | None] = None
    _zero: InitVar[Optional[Callable[[], A]]] = None
    _unit: InitVar[Optional[Callable[[], A]]] = None
    weight: InitVar[Optional[A]] = None

    def __post_init__(self, sharding, _zero, _unit, weight) -> None:
        """Post-initialization of matrix algebra objects."""
        object.__setattr__(self, "dtype", self.dom.dtype)
        object.__setattr__(self, "shape", (self.dom.shape, self.dom.shape))
        object.__setattr__(self, "scl", self.dom.scl)
        object.__setattr__(self, "codom", self.dom)
        if _zero is not None:
            object.__setattr__(self, "zero", _zero)
        else:
            object.__setattr__(
                self, "zero", make_zero(self.shape, self.dtype, sharding)
            )
        if _unit is not None:
            object.__setattr__(self, "unit", _unit)
        else:
            object.__setattr__(
                self, "unit", make_unit(self.shape, self.dtype, sharding)
            )
        if weight is not None:
            # TODO: Add check that weight has the right shape.
            object.__setattr__(self, "innerp", make_weighted_b2_innerp(weight))
        else:
            object.__setattr__(self, "innerp", b2_innerp)
        object.__setattr__(self, "norm", to_norm(self.innerp))


@final
@dataclass(frozen=True)
class MatrixSpace[M: Shape, N: Shape, D: DTypeLike](
    alg.ImplementsOperatorSpace[A, V, W, K]
):
    """Implement matrix space operations for JAX arrays."""

    dom: L2VectorAlgebra[N, D]
    codom: L2VectorAlgebra[M, D]
    shape: tuple[N, N] = field(init=False)
    dtype: D = field(init=False)
    scl: ScalarField[D] = field(init=False)
    zero: Callable[[], V] = field(init=False)
    add: Callable[[A, A], A] = field(default=jnp.add)
    neg: Callable[[A], A] = field(default=neg)
    sub: Callable[[A, A], A] = field(default=jnp.subtract)
    sdiv: Callable[[K, A], A] = field(default=sdiv)
    smul: Callable[[K, A], A] = field(default=jnp.multiply)
    norm: Callable[[A], K] = field(default=partial(jnp.linalg.norm, ord=2))
    app: Callable[[A, V], V] = field(default=jnp.matmul)
    sharding: InitVar[Sharding | None] = None
    _zero: InitVar[Optional[Callable[[], A]]] = None

    def __post_init__(self, sharding, _zero) -> None:
        """Post-initialization of HS matrix space objects."""
        object.__setattr__(self, "dtype", self.dom.dtype)
        object.__setattr__(self, "shape", (self.codom.shape, self.dom.shape))
        object.__setattr__(self, "scl", self.dom.scl)
        if _zero is not None:
            object.__setattr__(self, "zero", _zero)
        else:
            object.__setattr__(
                self, "zero", make_zero(self.shape, self.dtype, sharding)
            )


@final
@dataclass(frozen=True)
class HilbertSchmidtMatrixSpace[M: Shape, N: Shape, D: DTypeLike](
    alg.ImplementsInnerProductOperatorSpace[A, V, W, K]
):
    """Implement Hilbert-Schmidt matrix space operations for JAX arrays."""

    dom: L2VectorAlgebra[N, D]
    codom: L2VectorAlgebra[M, D]
    shape: tuple[N, N] = field(init=False)
    dtype: D = field(init=False)
    scl: ScalarField[D] = field(init=False)
    zero: Callable[[], V] = field(init=False)
    add: Callable[[A, A], A] = field(default=jnp.add)
    neg: Callable[[A], A] = field(default=neg)
    sub: Callable[[A, A], A] = field(default=jnp.subtract)
    sdiv: Callable[[K, A], A] = field(default=sdiv)
    smul: Callable[[K, A], A] = field(default=jnp.multiply)
    innerp: Callable[[A, A], K] = field(init=False)
    norm: Callable[[A], K] = field(init=False)
    app: Callable[[A, V], V] = field(default=jnp.matmul)
    sharding: InitVar[Sharding | None] = None
    _zero: InitVar[Optional[Callable[[], A]]] = None
    weight: InitVar[Optional[A]] = None

    def __post_init__(self, sharding, _zero, weight) -> None:
        """Post-initialization of HS matrix space objects."""
        object.__setattr__(self, "dtype", self.dom.dtype)
        object.__setattr__(self, "shape", (self.codom.shape, self.dom.shape))
        object.__setattr__(self, "scl", self.dom.scl)
        if _zero is not None:
            object.__setattr__(self, "zero", _zero)
        else:
            object.__setattr__(
                self, "zero", make_zero(self.shape, self.dtype, sharding)
            )
        if weight is not None:
            # TODO: Add check that weight has the right shape.
            object.__setattr__(self, "innerp", make_weighted_b2_innerp(weight))
        else:
            object.__setattr__(self, "innerp", b2_innerp)
        object.__setattr__(self, "norm", to_norm(self.innerp))

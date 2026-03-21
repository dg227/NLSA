# pyright: basic
"""Provide utilities for sharding JAX arrays."""

import jax
import math
import nlsa.function_algebra as fun
from collections.abc import Callable, Sequence
from functools import partial, wraps
from dataclasses import InitVar, dataclass, field
from jax import Array, Device
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec, Sharding
from jax.tree_util import tree_map
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, final

if TYPE_CHECKING:
    type Device = Any
else:
    from jax import Device

type F[*Xs, Y] = Callable[[*Xs], Y]


@final
@dataclass(frozen=True)
class NamedSharder[N: tuple[int, ...], AxisNames: str]:
    """Provide named sharding for a set of devices with specific shape."""

    devices: InitVar[Sequence[Device]]
    shape: N
    axis_names: Sequence[AxisNames]
    mesh: Mesh = field(init=False)
    axis_types: AxisType | tuple[AxisType, ...] = AxisType.Auto

    def __post_init__(self, devices: Sequence[Device]) -> None:
        """Post-initialization for NamedSharder objects."""
        match self.axis_types:
            case AxisType():
                _axis_types = (self.axis_types,) * len(self.axis_names)
            case tuple():
                _axis_types = self.axis_types
        mesh = jax.make_mesh(
            axis_shapes=self.shape,
            axis_names=self.axis_names,
            axis_types=_axis_types,
            devices=devices,
        )
        object.__setattr__(self, "mesh", mesh)

    @property
    def ndim(self) -> int:
        """Number of sharding axes created by NamedSharder object."""
        return math.prod(self.shape)

    def sharding(self, *axis_names: Optional[AxisNames]) -> NamedSharding:
        """Create sharding from axes names."""
        return NamedSharding(self.mesh, PartitionSpec(*axis_names))


def with_sharding_constraints[A: Sequence[Array]](
    xs: A, shardings: Sequence[Sharding]
) -> A:
    """Apply sharding constraints to sequence of arrays."""
    return tree_map(jax.lax.with_sharding_constraint, xs, shardings)


def shardit[*Xs](
    f: F[*Xs, Array], sharding: Optional[Sharding] = None
) -> F[*Xs, Array]:
    """Map to a function returning an array or a sequence of sharded arrays."""
    if sharding is not None:
        g = fun.compose(
            partial(jax.lax.with_sharding_constraint, shardings=sharding), f
        )
        f_wrapped = wraps(f)(g)
    else:
        f_wrapped = f
    return f_wrapped


def shardem[*Xs, A: Sequence[Array]](
    f: F[*Xs, A],
    shardings: Optional[Sequence[Sharding]] = None,
) -> F[*Xs, A]:
    """Map to a function returning a sequence of sharded arrays."""
    if shardings is not None:
        g = fun.compose(
            partial(with_sharding_constraints, shardings=shardings), f
        )
        return wraps(f)(g)
    else:
        return f


class EigShardings(NamedTuple):
    """NamedTuple holding shardings of jax.linalg.eig output."""

    eigenvalues: Optional[NamedSharding] = None
    """Sharding of eigenvalue array."""

    eigenvectors: Optional[NamedSharding] = None
    """Sharding of eigenvector array."""


def make_eig_with_sharding_constraints(
    shardings: EigShardings = EigShardings(),
) -> Callable[[Array], tuple[Array, Array]]:
    """Make jax.numpy.linalg.eig solver with sharding constraint."""

    def eig(a: Array) -> tuple[Array, Array]:
        eigenvalues, eigenvectors = jax.numpy.linalg.eig(a)
        if shardings.eigenvalues is not None:
            eigenvalues = jax.lax.with_sharding_constraint(
                eigenvalues, shardings.eigenvalues
            )
        if shardings.eigenvectors is not None:
            eigenvectors = jax.lax.with_sharding_constraint(
                eigenvectors, shardings.eigenvectors
            )
        return eigenvalues, eigenvectors

    return eig


def make_eigh_with_sharding_constraints(
    shardings: EigShardings = EigShardings(),
) -> Callable[[Array], tuple[Array, Array]]:
    """Make jax.numpy.linalg.eigh solver with sharding constraint."""

    def eig(a: Array) -> tuple[Array, Array]:
        eigenvalues, eigenvectors = jax.numpy.linalg.eigh(a)
        if shardings.eigenvalues is not None:
            eigenvalues = jax.lax.with_sharding_constraint(
                eigenvalues, shardings.eigenvalues
            )
        if shardings.eigenvectors is not None:
            eigenvectors = jax.lax.with_sharding_constraint(
                eigenvectors, shardings.eigenvectors
            )
        return eigenvalues, eigenvectors

    return eig


class SvdShardings(NamedTuple):
    """NamedTuple holding shardings of jax.numpy.linalg.svd output."""

    sing_values: Optional[NamedSharding] = None
    """Singular values array."""

    left_sing_vectors: Optional[NamedSharding] = None
    """Left singular vectors array."""

    right_sing_vectors: Optional[NamedSharding] = None
    """Right singular vectors array."""


def make_svd_with_sharding_constraints(
    shardings: SvdShardings = SvdShardings(),
) -> Callable[[Array], tuple[Array, Array, Array]]:
    """Make jax.numpy.linalg.svd solver with sharding constraint."""

    def eig(a: Array) -> tuple[Array, Array, Array]:
        left_sing_vectors, sing_values, right_sing_vectors = (
            jax.numpy.linalg.svd(a)
        )
        if shardings.left_sing_vectors is not None:
            left_sing_vectors = jax.lax.with_sharding_constraint(
                left_sing_vectors, shardings.left_sing_vectors
            )
        if shardings.sing_values is not None:
            sing_values = jax.lax.with_sharding_constraint(
                sing_values, shardings.sing_values
            )
        if shardings.right_sing_vectors is not None:
            right_sing_vectors = jax.lax.with_sharding_constraint(
                right_sing_vectors, shardings.right_sing_vectors
            )
        return left_sing_vectors, sing_values, right_sing_vectors

    return eig

"""Provide core functionality for modules in the nlsa_models subpackage."""

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from jax import Array
from jax.typing import DTypeLike
from numpy.typing import ArrayLike
from pathlib import Path
from tabulate import tabulate
from typing import TYPE_CHECKING, Any, Literal, TypedDict
from collections.abc import Sequence

if TYPE_CHECKING:
    type Device = Any
else:
    from jax import Device


type NPVector[N: int, D: np.dtype[np.floating[Any]]] = np.ndarray[tuple[N], D]
type Vector[N: int, D: np.dtype[np.floating[Any]]] = jax.Array | NPVector[N, D]
type NPMatrix[M: int, N: int, D: np.dtype[np.floating[Any]]] = np.ndarray[
    tuple[M, N], D
]
type Matrix[M: int, N: int, D: np.dtype[np.floating[Any]]] = (
    jax.Array | NPMatrix[M, N, D]
)


@dataclass(frozen=True, slots=True)
class JaxEnv:
    """Dataclass holding attributes of JAX environment."""

    device_cpu: Device
    """CPU device."""

    devices: list[Device] = field(default_factory=list)
    """GPU/TPU devices."""

    xla_mem_fraction: str | None = None
    """Preallocated memory fraction on accelerators."""

    real_dtype: DTypeLike = jnp.float32
    """DType for real numbers."""

    complex_dtype: DTypeLike = jnp.complex64
    """DType for complex numbers."""

    cache_dir: Path | None = None
    """Cache directory for JAX compilation."""

    def __post_init__(self) -> None:
        """Post-initialization of JaxEnv objects."""
        if len(self.devices) > 1:
            jax.config.update("jax_default_device", self.devices[0])
        if (
            self.real_dtype == jnp.float64
            or self.complex_dtype == jnp.complex128
        ):
            jax.config.update("jax_enable_x64", True)

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            jax.config.update("jax_compilation_cache_dir", str(self.cache_dir))
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 2)

    def tabulate(self, show: bool = True) -> str:
        """Create tabulated summary of the attributes of a JaxEnv object."""
        headers = ["JaxEnv attribute", "Value"]
        data = {
            "CPU device": self.device_cpu,
            "GPU/TPU devices": self.devices,
            "XLA_MEM_FRACTION": self.xla_mem_fraction,
            "Real dtype": self.real_dtype,
            "Complex dtype": self.complex_dtype,
            "Cache dir": self.cache_dir,
        }
        table = tabulate(data.items(), headers=headers)
        if show:
            print(table)
        return table


def initialize_jax(
    idx_cpu: int | None = None,
    idx_gpu: int | None | Sequence[int] = None,
    xla_mem_fraction: str | None = None,
    fp: Literal["f32", "f64"] = "f32",
    cache_dir: str | Path | None = None,
) -> JaxEnv:
    """Initialize JAX environment."""
    if xla_mem_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = xla_mem_fraction
    if idx_cpu is not None:
        device_cpu = jax.devices("cpu")[idx_cpu]
    else:
        device_cpu = jax.devices("cpu")[0]
    match idx_gpu:
        case None | []:
            devices = []
        case int():
            devices = [jax.devices("gpu")[idx_gpu]]
        case [_, *_]:
            devices = [jax.devices("gpu")[idx] for idx in idx_gpu]
        case _:
            devices = []  # Needed to satisfy pyright
    match fp:
        case "f32":
            real_dtype = jnp.float32
            complex_dtype = jnp.complex64
        case "f64":
            real_dtype = jnp.float64
            complex_dtype = jnp.complex128
    match cache_dir:
        case None:
            _cache_dir = None
        case str():
            _cache_dir = Path.cwd() / Path(cache_dir)
        case Path():
            _cache_dir = cache_dir
    return JaxEnv(
        device_cpu=device_cpu,
        devices=devices,
        xla_mem_fraction=xla_mem_fraction,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        cache_dir=_cache_dir,
    )


# TODO: Change this to NamedTuple and make generic over the array type.
class SkillScores(TypedDict):
    """TypedDict containing prediction skill scores."""

    nrmses: Array
    """Normalized RMSE scores."""

    accs: Array
    """Anomaly correlation scores."""


def to_skill_scores(
    dict_in: Mapping[str, ArrayLike],
    dtype: DTypeLike | None = None,
) -> SkillScores:
    """Convert dict of numpy ArrayLike objects to SkillScores TypedDict."""
    try:
        skill_scores: SkillScores = {
            "nrmses": jnp.asarray(dict_in["nrmses"], dtype),
            "accs": jnp.asarray(dict_in["accs"], dtype),
        }
        return skill_scores
    except ValueError as exc:
        raise ValueError("Incompatible keys/values") from exc


def initialize_matplotlib(backend: Literal["Agg"] | None = None) -> None:
    """Initialize matplolib library."""
    if backend is not None:
        matplotlib.use(backend)

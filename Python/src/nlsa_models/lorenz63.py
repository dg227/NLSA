"""Computation and plotting functions for analysis of the Lorenz 63 system."""

# pyright: basic

import diffrax as dfx
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.figure as mpf
import matplotlib.pyplot as plt
import nlsa.abstract_algebra as alg
import nlsa.function_algebra as fun
import nlsa.jax.delays as dl
import nlsa.jax.dynamics as dyn
import nlsa.jax.euclidean as r3
import nlsa.jax.koopman as koop
from nlsa.jax.koopman._koopman import GeneratorShardings
import nlsa.jax.stats as stats
import nlsa.jax.vector_algebra as vec
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from diffrax import Dopri5, ODETerm, PIDController, SaveAt
from functools import partial
from jax import Array, jit, vmap
from jax.typing import DTypeLike
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from nlsa.jax.kernels import KernelEigenbasis
from nlsa.jax.koopman import KoopmanEigenbasis, QzShardings
from nlsa.jax.stats import (
    MultivariateTimeseriesStats,
    anomaly_correlation_coefficient,
    normalized_rmse,
)
from nlsa.jax.vector_algebra import L2FnAlgebra, L2FnAlgebraShardings
from numpy.typing import ArrayLike
from pathlib import Path
from tabulate import tabulate
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    Sequence,
    TypedDict,
)

if TYPE_CHECKING:
    type Device = Any
else:
    from jax import Device


type Xs = Array  # Collection of points in state space
type Ys = Array  # Collection of points in covariate space
type Y = Array  # Point in covariate space
type Yd = Array  # Point in delay-coordinate space
type TYd = Array  # Tangent vector in delay coordinate space
type R = Array  # Real number
type Rs = Array  # Collection of real numbers
type C = Array  # Complex number
type Cs = Array  # Collection of complex numbers
type V = Array  # Vector in L2
type Vs = Array  # Collection of vectors in L2
type Vtsts = Array  # Collection of vectors in L2 with respect to test dataset
type Mat = Array  # Matrix
type F[*Ss, T] = Callable[[*Ss], T]  # Shorthand for Callables


@dataclass(frozen=True, slots=True)
class JaxEnv:
    """NamedTuple holding attributes of JAX environment."""

    device_cpu: Device
    """CPU device."""

    devices: list[Device] = field(default_factory=list)
    """GPU/TPU devices."""

    xla_mem_fraction: Optional[str] = None
    """Preallocated memory fraction on accelerators."""

    real_dtype: DTypeLike = jnp.float32
    """DType for real numbers."""

    complex_dtype: DTypeLike = jnp.complex64
    """DType for complex numbers."""

    cache_dir: Optional[Path] = None
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
    idx_cpu: Optional[int] = None,
    idx_gpu: Optional[int] | Sequence[int] = None,
    xla_mem_fraction: Optional[str] = None,
    fp: Literal["f32", "f64"] = "f32",
    cache_dir: Optional[str | Path] = None,
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


@dataclass(frozen=True)
class DataPars[N: int]:
    """Dataclass containing training and test data parameter values."""

    covariate: Literal["x", "y", "z", "xy", "xyz"]
    """Covariate function."""

    response: Literal["x", "y", "z"]
    """Response function."""

    dt: float
    """Sampling interval."""

    x0: tuple[float, float, float]
    """Initial condition."""

    num_spinup: int
    """Number of spinup samples."""

    num_samples: N
    """Number of samples (after spinup, delays, and finite difference)."""

    num_half_delays: int = 0
    """Half number of delays (to ensure even two-sided embedding window)."""

    num_before: int = 0
    """Number of extra samples before delay embedding."""

    num_after: int = 0
    """Number of extra after delay embedding."""

    velocity_covariate: bool = False
    """Include time tendencies (velocities) in covariate data."""

    velocity_fd_order: Optional[Literal[2, 4, 6, 8]] = None
    """Finite-difference order for velocity data."""

    eval_batch_size: Optional[int] = None
    """Number of batches for batchwise evaluation."""

    @property
    def num_velocity_fd(self) -> int:
        """Number of extra samples for velocity finite differencing."""
        if self.velocity_fd_order is not None:
            num_vel_fd = self.velocity_fd_order
        else:
            num_vel_fd = 0
        return num_vel_fd

    @property
    def num_delays(self) -> int:
        """Number of delays."""
        return 2 * self.num_half_delays

    @property
    def num_delay_samples(self) -> int:
        """Number of samples used for delay embedding."""
        return 2 * self.num_half_delays + self.num_samples

    @property
    def delay_embedding_origin(self) -> int:
        """Index of delay embedding origin."""
        return self.num_velocity_fd // 2 + self.num_before

    @property
    def delay_embedding_center(self) -> int:
        """Index of delay embedding center."""
        return self.delay_embedding_origin + self.num_half_delays

    @property
    def delay_embedding_end(self) -> int:
        """Index of delay embedding end."""
        return self.delay_embedding_origin + self.num_delays

    @property
    def num_total_samples(self) -> int:
        """Total number of samples (excluding spinup)."""
        return (
            self.num_samples
            + self.num_velocity_fd
            + self.num_delays
            + self.num_before
            + self.num_after
        )

    def __str__(self) -> str:
        """Create string representation of data parameters."""
        x0_str = f"x0_{self.x0[0]:.2f}_{self.x0[1]:.2f}_{self.x0[2]:.2f}"
        if self.velocity_covariate:
            assert self.velocity_fd_order is not None
            vel_str = f"vfd{self.velocity_fd_order}"
        else:
            vel_str = ""
        return "_".join(
            filter(
                None,
                (
                    f"dt{self.dt:.2f}",
                    x0_str,
                    self.covariate,
                    self.response,
                    f"nspin{self.num_spinup}",
                    f"ns{self.num_samples}",
                    f"nd{self.num_delays}",
                    f"nb{self.num_before}",
                    f"na{self.num_after}",
                    vel_str,
                ),
            )
        )

    def tabulate(self, name: str = "DataPars", show: bool = True) -> str:
        """Create tabulated summary of the properties of a DataPars object."""
        headers = [name, "Property Value"]
        data = {
            "Total number of samples": self.num_total_samples,
            "Number of analysis samples": self.num_samples,
            "Number of delays": self.num_delays,
            "Number of samples before": self.num_before,
            "Number of samples after": self.num_after,
            "Number of finite-difference samples": self.num_velocity_fd,
        }
        table = tabulate(data.items(), headers=headers)
        if show:
            print(table)
        return table


# TODO: Classes such as Data and SkillScores could be made
# NamedTuples or Dataclasses so we can implement validation, to/from methods,
# etc. These new classes could also be made generic over the array type (e.g.,
# supporting the array protocol) so we can use them with computational backends
# other than JAX.
class Data(TypedDict):
    """TypedDict containing training data."""

    states: Xs
    """Dynamical states."""

    covariates: Ys
    """Covariates."""

    responses: Rs
    """Responses."""


class SkillScores(TypedDict):
    """TypedDict containing prediction skill scores."""

    nrmses: Rs
    """Normalized RMSE scores."""

    accs: Rs
    """Anomaly correlation scores."""


def to_data(
    dict_in: dict[str, ArrayLike], dtype: Optional[DTypeLike] = None
) -> Data:
    """Convert dictionary of numpy ArrayLike objects to Data TypedDict."""
    try:
        data: Data = {
            "states": jnp.asarray(dict_in["states"], dtype),
            "covariates": jnp.asarray(dict_in["covariates"], dtype),
            "responses": jnp.asarray(dict_in["responses"], dtype),
        }
        return data
    except ValueError as exc:
        raise ValueError("Incompatible keys/values") from exc


def to_skill_scores(
    dict_in: dict[str, ArrayLike], dtype: Optional[DTypeLike] = None
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


def generate_data[N: int](
    pars: DataPars[N],
    dtype: DTypeLike,
    fp: Literal["F32", "F64"] = "F32",
    device: Optional[Device] = None,
) -> Data:
    """Generate L63 data."""
    if device is None:
        _device = jax.devices("cpu")[0]
    else:
        _device = device

    match pars.covariate:
        case "xyz":
            cov = r3.make_observable_id(dtype)
        case "xy":
            cov = r3.make_observable_xy(dtype)
        case "x":
            cov = r3.make_observable_x(dtype, asvector=True)
        case "y":
            cov = r3.make_observable_y(dtype, asvector=True)
        case "z":
            cov = r3.make_observable_z(dtype, asvector=True)
    match pars.response:
        case "x":
            rsp = r3.make_observable_x(dtype)
        case "y":
            rsp = r3.make_observable_y(dtype)
        case "z":
            rsp = r3.make_observable_z(dtype)
    covariate = jit(vmap(cov))
    response = jit(vmap(rsp))
    v = dyn.make_l63_vector_field()
    num_ode_samples = pars.num_total_samples + pars.num_spinup

    @jax.jit
    def diffeqsolve(y0: Array) -> dfx.Solution:
        solution = dfx.diffeqsolve(
            terms=ODETerm(dyn.from_autonomous(v)),
            solver=Dopri5(),
            t0=0,
            t1=ts_tot[-1],
            dt0=pars.dt,
            y0=y0,
            saveat=SaveAt(ts=ts_tot[pars.num_spinup :]),
            stepsize_controller=PIDController(rtol=1e-8, atol=1e-8),
            max_steps=200_000_000,
        )
        return solution

    with jax.default_device(_device):
        if fp == "F32":
            jax.config.update("jax_enable_x64", True)
        ts_tot = jnp.arange(num_ode_samples) * pars.dt
        solution = diffeqsolve(jnp.array(pars.x0))
        if fp == "F32":
            jax.config.update("jax_enable_x64", False)
    xs = jnp.array(solution.ys, dtype=dtype)
    ys = covariate(xs)
    zs = response(xs)
    if pars.velocity_covariate:
        assert pars.velocity_fd_order is not None
        fd_op = jit(
            vmap(
                dl.make_fd_operator(
                    order=pars.velocity_fd_order, mode="central", dt=pars.dt
                ),
                in_axes=-1,
                out_axes=-1,
            )
        )
        vs = fd_op(ys)
        data: Data = {
            "states": xs,
            "covariates": jnp.stack((ys, vs), axis=1),
            "responses": zs,
        }
    else:
        data: Data = {
            "states": xs,
            "covariates": ys,
            "responses": zs,
        }
    return data


def make_l2_space[N: int, D: DTypeLike](
    pars: DataPars[N],
    dtype: D,
    data: Data,
    delay_embedding_mode: Literal["explicit", "on_the_fly"] = "on_the_fly",
    shardings: L2FnAlgebraShardings = L2FnAlgebraShardings(),
    jit: bool = False,
) -> L2FnAlgebra[tuple[N], D, Yd, R]:
    """Make L2 space over covariate data space."""
    i0 = pars.delay_embedding_origin
    i1 = i0 + pars.num_delay_samples
    incl: Callable[[F[Array, Array]], V]
    if pars.num_half_delays > 0:
        match delay_embedding_mode:
            case "on_the_fly":
                incl = dl.delay_eval_at(
                    jnp.asarray(
                        data["covariates"][i0:i1],
                        dtype=dtype,
                        device=shardings.data,
                    ),
                    num_delays=pars.num_delays,
                    batch_size=pars.eval_batch_size,
                    out_sharding=shardings.vectors,
                    jit=jit,
                )
            case "explicit":
                if pars.velocity_covariate:
                    hankel = jax.jit(
                        vmap(
                            partial(
                                dl.hankel,
                                num_delays=pars.num_delays,
                                flatten=True,
                            ),
                            in_axes=1,
                            out_axes=1,
                        )
                    )
                else:
                    hankel = jax.jit(
                        partial(
                            dl.hankel, num_delays=pars.num_delays, flatten=True
                        )
                    )
                incl = vec.batch_eval_at(
                    jnp.asarray(
                        hankel(data["covariates"][i0:i1]),
                        dtype=dtype,
                        device=shardings.data,
                    ),
                    batch_size=pars.eval_batch_size,
                    out_sharding=shardings.vectors,
                    jit=jit,
                )
    else:
        incl = vec.batch_eval_at(
            jnp.asarray(
                data["covariates"][i0:i1], dtype=dtype, device=shardings.data
            ),
            batch_size=pars.eval_batch_size,
            out_sharding=shardings.vectors,
            jit=jit,
        )
    mu = vec.make_normalized_counting_measure(pars.num_samples)
    return L2FnAlgebra(
        shape=(pars.num_samples,),
        dtype=dtype,
        measure=mu,
        inclusion_map=incl,
        sharding=shardings.vectors,
    )


def make_tangent_evaluation_functional_fd[N: int](
    pars: DataPars[N],
    dtype: DTypeLike,
    data: Data,
    fd_order: Literal[2, 4, 6, 8],
    delay_embedding_mode: Literal["explicit", "on_the_fly"] = "on_the_fly",
    shardings: L2FnAlgebraShardings = L2FnAlgebraShardings(),
    jit: bool = False,
) -> Callable[[F[Yd, TYd, R]], V]:
    """Make evaluation using finite-difference approx. of the generator."""
    if pars.velocity_covariate:
        fd_op = jax.jit(
            vmap(
                vmap(
                    dl.make_fd_operator(
                        order=fd_order,
                        mode="central",
                        dt=pars.dt,
                        extrap=False,
                    ),
                    in_axes=-1,
                    out_axes=-1,
                ),
                in_axes=-1,
                out_axes=-1,
            )
        )
    else:
        fd_op = jax.jit(
            vmap(
                dl.make_fd_operator(
                    order=fd_order,
                    mode="central",
                    dt=pars.dt,
                    extrap=False,
                ),
                in_axes=-1,
                out_axes=-1,
            )
        )
    num_half_fd = fd_order // 2
    i0 = pars.delay_embedding_origin
    i1 = i0 + pars.num_delay_samples
    i0_fd = i0 - num_half_fd
    i1_fd = i1 + num_half_fd
    eval_tx: Callable[[F[Yd, TYd, R]], V]
    if pars.num_half_delays > 0:
        match delay_embedding_mode:
            case "on_the_fly":
                eval_tx = dl.delay_eval_at(
                    (
                        jnp.asarray(
                            data["covariates"][i0:i1],
                            dtype=dtype,
                            device=shardings.data,
                        ),
                        jnp.asarray(
                            fd_op(data["covariates"][i0_fd:i1_fd]),
                            dtype=dtype,
                            device=shardings.data,
                        ),
                    ),
                    num_delays=pars.num_delays,
                    batch_size=pars.eval_batch_size,
                    out_sharding=shardings.vectors,
                    jit=jit,
                )
            case "explicit":
                if pars.velocity_covariate:
                    hankel = jax.jit(
                        vmap(
                            partial(
                                dl.hankel,
                                num_delays=pars.num_delays,
                                flatten=True,
                            ),
                            in_axes=1,
                            out_axes=1,
                        )
                    )
                else:
                    hankel = jax.jit(
                        partial(
                            dl.hankel,
                            num_delays=pars.num_delays,
                            flatten=True,
                        )
                    )
                eval_tx = vec.batch_eval_at(
                    (
                        jnp.asarray(
                            hankel(data["covariates"][i0:i1]),
                            dtype=dtype,
                            device=shardings.data,
                        ),
                        jnp.asarray(
                            hankel(fd_op(data["covariates"][i0_fd:i1_fd])),
                            dtype=dtype,
                            device=shardings.data,
                        ),
                    ),
                    batch_size=pars.eval_batch_size,
                    out_sharding=shardings.vectors,
                    jit=jit,
                )
    else:
        eval_tx = vec.batch_eval_at(
            (
                jnp.asarray(
                    data["covariates"][i0:i1],
                    dtype=dtype,
                    device=shardings.data,
                ),
                jnp.asarray(
                    fd_op(data["covariates"][i0_fd:i1_fd]),
                    dtype=dtype,
                    device=shardings.data,
                ),
            ),
            batch_size=pars.eval_batch_size,
            out_sharding=shardings.vectors,
            jit=jit,
        )
    return eval_tx


def compute_generator_matrix[N: int, D: DTypeLike](
    pars: DataPars[N],
    fd_order: Literal[2, 4, 6, 8],
    l2_space: L2FnAlgebra[tuple[N], D, Yd, R],
    train_data: Data,
    basis: alg.ImplementsDimensionedL2FnFrame[Yd, R, V, Rs, int | Array],
    delay_embedding_mode: Literal["explicit", "on_the_fly"] = "on_the_fly",
    grad_batch_size: Optional[int] = None,
    gram_batch_size: Optional[int] = None,
    shardings: GeneratorShardings = GeneratorShardings(),
    jit: bool = True,
) -> Mat:
    """Compute generator matrix representation using finite differences."""
    eval_tx = make_tangent_evaluation_functional_fd(
        pars,
        dtype=l2_space.dtype,
        data=train_data,
        fd_order=fd_order,
        delay_embedding_mode=delay_embedding_mode,
        shardings=shardings.tangents,
        jit=jit,
    )
    basis_idxs = jnp.arange(1, basis.dim)
    gen_mat = koop.compute_generator_matrix(
        l2_space,
        eval_tangents=eval_tx,
        basis=basis,
        basis_idxs=basis_idxs,
        grad_batch_size=grad_batch_size,
        gram_batch_size=gram_batch_size,
    )
    return gen_mat


def make_quadrature_evaluation_functional[N: int](
    pars: DataPars[N],
    dtype: DTypeLike,
    data: Data,
    num_quad: int,
    delay_embedding_mode: Literal["explicit", "on_the_fly"] = "on_the_fly",
    eval_batch_size: Optional[int] = None,
    shardings: L2FnAlgebraShardings = L2FnAlgebraShardings(),
    jit: bool = False,
) -> Callable[[F[Yd, R]], V]:
    """Make quadrature evaluation for resolvent approximation."""
    i0 = pars.delay_embedding_origin
    i1 = i0 + pars.num_samples + pars.num_delays + num_quad
    eval_quad: Callable[[F[Yd, R]], V]
    if pars.num_half_delays > 0:
        match delay_embedding_mode:
            case "on_the_fly":
                eval_quad = dl.delay_eval_at(
                    jnp.asarray(
                        data["covariates"][i0:i1],
                        dtype=dtype,
                        device=shardings.data,
                    ),
                    num_delays=pars.num_delays,
                    batch_size=eval_batch_size,
                    out_sharding=shardings.vectors,
                    jit=jit,
                )
            case "explicit":
                if pars.velocity_covariate:
                    hankel = jax.jit(
                        vmap(
                            partial(
                                dl.hankel,
                                num_delays=pars.num_delays,
                                flatten=True,
                            ),
                            in_axes=1,
                            out_axes=1,
                        )
                    )
                else:
                    hankel = jax.jit(
                        partial(
                            dl.hankel,
                            num_delays=pars.num_delays,
                            flatten=True,
                        )
                    )
                eval_quad = vec.batch_eval_at(
                    jnp.asarray(
                        hankel(data["covariates"][i0:i1]),
                        dtype=dtype,
                        device=shardings.data,
                    ),
                    batch_size=eval_batch_size,
                    out_sharding=shardings.vectors,
                    jit=jit,
                )
    else:
        eval_quad = vec.batch_eval_at(
            jnp.asarray(
                data["covariates"][i0:i1],
                dtype=dtype,
                device=shardings.data,
            ),
            batch_size=eval_batch_size,
            out_sharding=shardings.vectors,
            jit=jit,
        )
    return eval_quad


def compute_qz_matrix[N: int, D: DTypeLike](
    pars: DataPars[N],
    res_z: float,
    dt: float,
    num_quad: int,
    l2_space: L2FnAlgebra[tuple[N], D, Yd, R],
    train_data: Data,
    basis: alg.ImplementsDimensionedL2FnFrame[Yd, R, V, Rs, int | Array],
    delay_embedding_mode: Literal["explicit", "on_the_fly"] = "on_the_fly",
    quad_batch_size: Optional[int] = None,
    gram_batch_size: Optional[int] = None,
    shardings: QzShardings = QzShardings(),
) -> Mat:
    """Compute matrix representation of Qz operator using Laplace transform."""
    eval_quad = make_quadrature_evaluation_functional(
        pars,
        dtype=l2_space.dtype,
        data=train_data,
        num_quad=num_quad,
        delay_embedding_mode=delay_embedding_mode,
        shardings=shardings.quadrature,
    )
    basis_idxs = jnp.arange(1, basis.dim)
    qz_mat = koop.compute_qz_matrix(
        res_z=res_z,
        dt=dt,
        num_quad=num_quad,
        l2_space=l2_space,
        eval_quad=eval_quad,
        basis=basis,
        basis_idxs=basis_idxs,
        quad_batch_size=quad_batch_size,
        gram_batch_size=gram_batch_size,
        shardings=shardings,
    )
    return qz_mat


def make_kaf_prediction_function[N: int](
    pars: DataPars[N],
    train_data: Data,
    nyst: Callable[[V], F[Yd, R]],
    num_steps: int,
) -> F[Yd, Rs]:
    """Make time-series-valued kernel analog prediction function."""
    i0 = pars.delay_embedding_end
    i1 = i0 + num_steps + pars.num_samples
    fxs_ts = dl.hankel(train_data["responses"][i0:i1], num_delays=num_steps)

    @partial(vmap, in_axes=(1, None))
    def predict(v: V, y: Yd) -> R:
        return nyst(v)(y)

    return partial(predict, fxs_ts)


def make_iterative_kaf_prediction_function[N: int](
    pars: DataPars[N], train_data: Data, nyst: Callable[[V], F[Yd, R]]
) -> F[Yd, Rs]:
    """Make single-step prediction function."""
    i0 = pars.delay_embedding_end + 1
    i1 = i0 + pars.num_samples

    @partial(vmap, in_axes=(1, None))
    def nystrom_predict(v: V, y: Yd) -> Y:
        """Predict next snapshot."""
        return nyst(v)(y)

    return partial(nystrom_predict, train_data["covariates"][i0:i1])


def make_iterative_kaf_prediction_function_with_delays[N: int](
    pars: DataPars[N], train_data: Data, nyst: Callable[[V], F[Yd, R]]
) -> F[Yd, Rs]:
    """Make single-step prediction function for delay-embedded data."""

    @partial(vmap, in_axes=(1, None))
    def nystrom_predict(v: V, y: Yd) -> Y:
        """Predict next snapshot from delay-embedded data using Nystrom."""
        return nyst(v)(y)

    i0 = pars.delay_embedding_end + 1
    i1 = i0 + pars.num_samples
    predict = partial(nystrom_predict, train_data["covariates"][i0:i1])

    def predict_window(y: Yd) -> Yd:
        """Predict next delay embedding window."""
        y_next = predict(y)
        y_prev_unrolled = y.reshape((pars.num_delays + 1, -1))[1:]
        y_pred_unrolled = jnp.concatenate(
            (y_prev_unrolled, y_next[jnp.newaxis, :])
        )
        return jnp.hstack(y_pred_unrolled)

    return predict_window


def make_koopman_prediction_function[N: int](
    pars: DataPars[N],
    train_data: Data,
    koopman_basis: KoopmanEigenbasis[Yd, C, V, Cs, int | Array],
) -> F[R, Yd, Cs]:
    """Make prediction function based on Koopman eigendecomposition."""
    i0 = pars.delay_embedding_end
    i1 = i0 + pars.num_samples
    f_coeffs = koopman_basis.anal(train_data["responses"][i0:i1])

    def predict(t: R, y: Yd) -> C:
        phases = jnp.exp(koopman_basis.gen_spec * t)
        return koopman_basis.fn_synth(phases * f_coeffs)(y)

    return predict


def compute_response_skill_scores[Ntst: int](
    pars: DataPars[Ntst],
    test_data: Data,
    fys_pred: Vtsts,
    dropna: bool = False,
) -> SkillScores:
    """Compute NRMSE and ACC skill scores over the prediction ensemble."""
    num_pred_steps = fys_pred.shape[1] - 1
    i0 = pars.delay_embedding_end
    i1 = i0 + num_pred_steps + pars.num_samples
    hankel = jax.jit(partial(dl.hankel, num_delays=num_pred_steps))
    fxs_true = hankel(test_data["responses"][i0:i1])
    if dropna:
        mask = ~jnp.isnan(fys_pred).any(axis=1)
        fys_pred = fys_pred[mask]
        fxs_true = fxs_true[mask]
    nrmses = jax.jit(vmap(normalized_rmse, in_axes=1))(fxs_true, fys_pred)
    accs = jax.jit(vmap(anomaly_correlation_coefficient, in_axes=1))(
        fxs_true, fys_pred
    )
    scores: SkillScores = {"nrmses": nrmses, "accs": accs}
    return scores


def compute_covariate_skill_scores[Ntst: int](
    pars: DataPars[Ntst], test_data: Data, ys_pred: Vtsts, dropna: bool = False
) -> SkillScores:
    """Compute NRMSE and ACC skill scores over the prediction ensemble."""
    num_pred_steps = len(ys_pred) - 1
    i0 = pars.delay_embedding_end
    i1 = i0 + num_pred_steps + pars.num_samples
    hankel = jax.jit(
        fun.compose(
            partial(jnp.swapaxes, axis1=0, axis2=1),
            vmap(
                partial(dl.hankel, num_delays=num_pred_steps),
                in_axes=-1,
                out_axes=-1,
            ),
        )
    )
    normalized_rmses = jax.jit(
        vmap(vmap(stats.normalized_rmse, in_axes=1), in_axes=2)
    )
    anomaly_correlation_coefficients = jax.jit(
        vmap(vmap(stats.anomaly_correlation_coefficient, in_axes=1), in_axes=2)
    )
    ys_true = hankel(test_data["covariates"][i0:i1])
    if dropna:
        mask = ~jnp.isnan(ys_pred).any(axis=(1, 2))
        ys_pred = ys_pred[mask]
        ys_true = ys_true[mask]
    nrmses = normalized_rmses(ys_true, ys_pred)
    accs = anomaly_correlation_coefficients(ys_true, ys_pred)
    scores: SkillScores = {"nrmses": nrmses, "accs": accs}
    return scores


def compute_skill_scores[Ntst: int](
    pars: DataPars[Ntst],
    test_data: Data,
    preds: Vtsts,
    what: Literal["covariates", "responses"] = "responses",
    dropna: bool = False,
) -> SkillScores:
    """Compute NRMSE and ACC skill scores over the prediction ensemble."""
    match what:
        case "covariates":
            scores = compute_covariate_skill_scores(
                pars, test_data, preds, dropna=dropna
            )
        case "responses":
            scores = compute_response_skill_scores(
                pars, test_data, preds, dropna=dropna
            )
    return scores


def compute_trajectory_stats[N: int](
    pars: DataPars[N],
    train_data: Data,
    ys_traj: Vtsts,
    num_pred_steps: int,
    num_stat_steps: int,
    dropna: bool = False,
) -> tuple[MultivariateTimeseriesStats, MultivariateTimeseriesStats]:
    """Compute statistics of training and reconstructed covariate data."""
    i0 = pars.delay_embedding_end
    i1 = i0 + pars.num_samples + num_pred_steps
    train_stats = stats.multivariate_timeseries_stats(
        train_data["covariates"][i0:i1].T,
        num_lags=num_pred_steps,
        autocorrelation_mode="exact",
        autocorrelation_num_samples=pars.num_samples,
        dropna=dropna,
    )
    i0_tst = pars.delay_embedding_end
    i1_tst = i0_tst + num_stat_steps + num_pred_steps
    traj_stats = stats.multivariate_timeseries_stats(
        ys_traj[i0_tst:i1_tst].T,
        num_lags=num_pred_steps,
        autocorrelation_mode="exact",
        autocorrelation_num_samples=num_stat_steps,
        dropna=dropna,
    )
    return train_stats, traj_stats


def initialize_matplotlib(backend: Optional[Literal["Agg"]] = None) -> None:
    """Initialize matplolib library."""
    if backend is not None:
        matplotlib.use(backend)


def plot_bandwidth_function[N: int, Ntst: int, D: DTypeLike](
    pars: DataPars[N],
    l2y: L2FnAlgebra[tuple[N], D, Yd, R],
    bandwidth_func: F[Yd, R],
    train_data: Data,
    test_pars: Optional[DataPars[Ntst]] = None,
    l2y_tst: Optional[L2FnAlgebra[tuple[Ntst], D, Yd, R]] = None,
    test_data: Optional[Data] = None,
    delay_plot_mode: Literal["backward", "central"] = "central",
    num_plt: Optional[int] = None,
    num_plt_tst: Optional[int] = None,
    plt_step: int = 1,
    plt_step_tst: int = 1,
    i_fig: int = 1,
) -> Figure:
    """Plot bandwidth function on training and, optionally, test data."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    if test_pars is not None:
        fig = plt.figure(num=i_fig, figsize=tuple(mpf.figaspect(0.5)))
        axs = (
            fig.add_subplot(1, 2, 1, projection="3d"),
            fig.add_subplot(1, 2, 2, projection="3d"),
        )
        ax, ax_tst = axs
        assert isinstance(ax, Axes3D)
        assert isinstance(ax_tst, Axes3D)
    else:
        fig = plt.figure(num=i_fig)

        ax = fig.add_subplot(projection="3d")
        assert isinstance(ax, Axes3D)
    fig.set_layout_engine("constrained")
    match delay_plot_mode:
        case "backward":
            i0 = pars.delay_embedding_end
        case "central":
            i0 = pars.delay_embedding_center
    if num_plt is None:
        num_plt = pars.num_samples
    i1 = i0 + num_plt
    bw_vals = l2y.incl(bandwidth_func)
    vmin = float(jnp.min(bw_vals))
    vmax = float(jnp.max(bw_vals))
    if l2y_tst is not None:
        bw_vals_tst = l2y_tst.incl(bandwidth_func)
        vmin = min(vmin, float(jnp.max(bw_vals_tst)))
        vmax = max(vmax, float(jnp.max(bw_vals_tst)))
    plt.rcParams["grid.color"] = "yellow"
    sc = ax.scatter(
        train_data["states"][i0:i1:plt_step, 0],
        train_data["states"][i0:i1:plt_step, 1],
        train_data["states"][i0:i1:plt_step, 2],
        c=bw_vals[:num_plt:plt_step],
        s=1,
        vmin=vmin,
        vmax=vmax,
        cmap="binary",
    )
    ax.set_xlabel("$x^1$")
    ax.set_ylabel("$x^2$")
    ax.set_zlabel("$x^3$")
    ax.set_title("Kernel bandwidth function (training)")
    ax.xaxis.pane.set_facecolor("orange")
    ax.yaxis.pane.set_facecolor("orange")
    ax.zaxis.pane.set_facecolor("orange")

    if test_pars is not None and l2y_tst is not None and test_data is not None:
        match delay_plot_mode:
            case "backward":
                i0_tst = test_pars.delay_embedding_end
            case "central":
                i0_tst = test_pars.delay_embedding_center
        if num_plt_tst is None:
            num_plt_tst = test_pars.num_samples
        i1_tst = i0_tst + num_plt_tst
        sc_tst = ax_tst.scatter(
            test_data["states"][i0_tst:i1_tst:plt_step_tst, 0],
            test_data["states"][i0_tst:i1_tst:plt_step_tst, 1],
            test_data["states"][i0_tst:i1_tst:plt_step_tst, 2],
            c=bw_vals_tst[:num_plt_tst:plt_step_tst],
            s=1,
            vmin=vmin,
            vmax=vmax,
            cmap="binary",
        )
        ax_tst.set_xlabel("$x^1$")
        ax_tst.set_ylabel("$x^2$")
        ax_tst.set_zlabel("$x^3$")
        ax_tst.set_title("Kernel bandwidth function (test)")
        ax_tst.xaxis.pane.set_facecolor("orange")
        ax_tst.yaxis.pane.set_facecolor("orange")
        ax_tst.zaxis.pane.set_facecolor("orange")
        fig.colorbar(sc_tst, ax=ax_tst)
    else:
        fig.colorbar(sc, ax=ax)

    plt.rcdefaults()
    return fig


def make_kernel_evecs_plotter[N: int, Ntst: int, D: DTypeLike](
    pars: DataPars[N],
    train_data: Data,
    kernel_basis: KernelEigenbasis[Yd, R, V, Rs, int | Array],
    test_pars: Optional[DataPars[Ntst]] = None,
    l2y_tst: Optional[L2FnAlgebra[tuple[Ntst], D, Yd, R]] = None,
    test_data: Optional[Data] = None,
    delay_plot_mode: Literal["backward", "central"] = "backward",
    num_plt: Optional[int] = None,
    num_plt_tst: Optional[int] = None,
    plt_step: int = 1,
    plt_step_tst: int = 1,
    i_fig: int = 1,
) -> tuple[Figure, F[int, None]]:
    """Make plotting function for kernel eigenfunctions."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    if test_pars is not None:
        fig = plt.figure(num=i_fig, figsize=tuple(mpf.figaspect(0.5)))
        axs = (
            fig.add_subplot(1, 2, 1, projection="3d"),
            fig.add_subplot(1, 2, 2, projection="3d"),
        )
        ax, ax_tst = axs
        assert isinstance(ax, Axes3D)
        assert isinstance(ax_tst, Axes3D)
    else:
        fig = plt.figure(num=i_fig)
        ax = fig.add_subplot(projection="3d")
        assert isinstance(ax, Axes3D)
    fig.set_layout_engine("constrained")
    match delay_plot_mode:
        case "backward":
            i0 = pars.delay_embedding_end
        case "central":
            i0 = pars.delay_embedding_center
    if num_plt is None:
        num_plt = pars.num_samples
    i1 = i0 + num_plt
    if test_pars is not None:
        match delay_plot_mode:
            case "backward":
                i0_tst = test_pars.delay_embedding_end
            case "central":
                i0_tst = test_pars.delay_embedding_center
        if num_plt_tst is None:
            num_plt_tst = test_pars.num_samples
        i1_tst = i0_tst + num_plt_tst

    def plot_eig(j: int):
        evec = kernel_basis.vec(j)
        amax = float(jnp.max(jnp.abs(evec)))
        if (
            test_pars is not None
            and l2y_tst is not None
            and test_data is not None
        ):
            evec_tst = l2y_tst.incl(kernel_basis.fn(j))
            amax = max(amax, float(jnp.abs(jnp.max(evec_tst))))
        for figax in fig.axes:
            figax.cla()

        sc = ax.scatter(
            train_data["states"][i0:i1:plt_step, 0],
            train_data["states"][i0:i1:plt_step, 1],
            train_data["states"][i0:i1:plt_step, 2],
            c=evec[:num_plt:plt_step],
            s=1,
            vmin=-amax,
            vmax=amax,
            cmap="seismic",
        )
        eta = kernel_basis.lapl_evl(j)
        ax.set_xlabel("$x^1$")
        ax.set_ylabel("$x^2$")
        ax.set_zlabel("$x^3$")
        ax.set_title(f"Eigenvector {j}: $\\eta_{{{j}}} = {eta: .3f}$")

        if (
            test_pars is not None
            and l2y_tst is not None
            and test_data is not None
            and kernel_basis is not None
        ):
            sc_tst = ax_tst.scatter(
                test_data["states"][i0_tst:i1_tst:plt_step_tst, 0],
                test_data["states"][i0_tst:i1_tst:plt_step_tst, 1],
                test_data["states"][i0_tst:i1_tst:plt_step_tst, 2],
                c=evec_tst[:num_plt_tst:plt_step_tst],
                s=1,
                vmin=-amax,
                vmax=amax,
                cmap="seismic",
            )
            ax_tst.set_xlabel("$x^1$")
            ax_tst.set_ylabel("$x^2$")
            ax_tst.set_zlabel("$x^3$")
            ax_tst.set_title("Nystrom")
            if len(fig.axes) > 2:
                fig.colorbar(sc_tst, ax=ax_tst, cax=fig.axes[2])
            else:
                fig.colorbar(sc_tst, ax=ax_tst)
        else:
            if len(fig.axes) > 1:
                fig.colorbar(sc, ax=ax, cax=fig.axes[1])
            else:
                fig.colorbar(sc, ax=ax)

    return fig, plot_eig


def make_koopman_evecs_plotter[N: int, Ntst: int, D: DTypeLike](
    pars: DataPars[N],
    train_data: Data,
    koopman_basis: KoopmanEigenbasis[Yd, C, V, Cs, Array | int],
    test_pars: Optional[DataPars[Ntst]] = None,
    l2y_tst: Optional[L2FnAlgebra[tuple[Ntst], D, Yd, R]] = None,
    test_data: Optional[Data] = None,
    delay_plot_mode: Literal["backward", "central"] = "backward",
    num_plt: Optional[int] = None,
    num_plt_tst: Optional[int] = None,
    plt_step: int = 1,
    plt_step_tst: int = 1,
    i_fig: int = 1,
) -> tuple[Figure, F[int, None]]:
    """Make plotting function for Koopman eigenfunctions."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    if test_pars is not None:
        fig = plt.figure(num=i_fig, figsize=tuple(1.5 * mpf.figaspect(0.5)))
        axs = (
            fig.add_subplot(2, 4, 1, projection="3d"),
            fig.add_subplot(2, 4, 2, projection="3d"),
            fig.add_subplot(2, 4, 3),
            fig.add_subplot(2, 4, 4),
        )
        axs_tst = (
            fig.add_subplot(2, 4, 5, projection="3d"),
            fig.add_subplot(2, 4, 6, projection="3d"),
            fig.add_subplot(2, 4, 7),
            fig.add_subplot(2, 4, 8),
        )
    else:
        fig = plt.figure(num=i_fig, figsize=tuple(1.5 * mpf.figaspect(0.5)))
        axs = (
            fig.add_subplot(1, 4, 1, projection="3d"),
            fig.add_subplot(1, 4, 2, projection="3d"),
            fig.add_subplot(1, 4, 3),
            fig.add_subplot(1, 4, 4),
        )
    fig.set_layout_engine("constrained")
    match delay_plot_mode:
        case "backward":
            i0 = pars.delay_embedding_end
        case "central":
            i0 = pars.delay_embedding_center
    if num_plt is None:
        num_plt = pars.num_samples
    i1 = i0 + num_plt
    ts = jnp.arange(num_plt) * pars.dt
    if test_pars is not None:
        match delay_plot_mode:
            case "backward":
                i0_tst = test_pars.delay_embedding_end
            case "central":
                i0_tst = test_pars.delay_embedding_center
        if num_plt_tst is None:
            num_plt_tst = test_pars.num_samples
        i1_tst = i0_tst + num_plt_tst
        ts_tst = jnp.arange(num_plt_tst) * test_pars.dt

    def plot_eig(j: int):
        evec = koopman_basis.vec(j)
        evl = koopman_basis.gen_evl(j)
        amax = max(
            float(jnp.max(jnp.abs(evec.real))),
            float(jnp.max(jnp.abs(evec.imag))),
        )
        if (
            test_pars is not None
            and l2y_tst is not None
            and test_data is not None
        ):
            evec_tst = l2y_tst.incl(koopman_basis.fn(j))
            amax = max(
                amax,
                float(jnp.max(jnp.abs(evec_tst.real))),
                float(jnp.max(jnp.abs(evec_tst.imag))),
            )
        for ax in fig.axes:
            ax.cla()

        ax = axs[0]
        assert isinstance(ax, Axes3D)
        sc = ax.scatter(
            train_data["states"][i0:i1:plt_step, 0],
            train_data["states"][i0:i1:plt_step, 1],
            train_data["states"][i0:i1:plt_step, 2],
            c=evec.real[:num_plt:plt_step],
            s=1,
            vmin=-amax,
            vmax=amax,
            cmap="seismic",
        )
        ax.set_xlabel("$x^1$")
        ax.set_ylabel("$x^2$")
        ax.set_zlabel("$x^3$")
        ax.set_title(f"$\\mathrm{{Re}}\\zeta_{{{j}}}$ - training")

        ax = axs[1]
        assert isinstance(ax, Axes3D)
        ax.scatter(
            train_data["states"][i0:i1:plt_step, 0],
            train_data["states"][i0:i1:plt_step, 1],
            train_data["states"][i0:i1:plt_step, 2],
            c=evec.imag[:num_plt:plt_step],
            s=1,
            vmin=-amax,
            vmax=amax,
            cmap="seismic",
        )
        ax.set_xlabel("$x^1$")
        ax.set_ylabel("$x^2$")
        ax.set_zlabel("$x^3$")
        ax.set_title(f"$\\mathrm{{Im}}\\zeta_{{{j}}}$")

        ax = axs[2]
        ax.plot(
            evec.real[:num_plt:plt_step], evec.imag[:num_plt:plt_step], "-"
        )
        ax.set_xlabel(f"$\\mathrm{{Re}}\\zeta_{{{j}}}$")
        ax.set_ylabel(f"$\\mathrm{{Im}}\\zeta_{{{j}}}$")
        ax.set_title(f"Frequency $\\omega_{{{j}}} = {evl.imag: .3f}$")
        ax.grid()

        ax = axs[3]
        ax.plot(
            ts[:num_plt:plt_step],
            evec.real[:num_plt:plt_step],
            "-",
            label=f"$\\mathrm{{Re}}\\zeta_{{{j}}}$",
        )
        ax.plot(
            ts[:num_plt:plt_step],
            evec.imag[:num_plt:plt_step],
            "-",
            label=f"$\\mathrm{{Im}}\\zeta_{{{j}}}$",
        )
        ax.set_xlabel("$t$")
        ax.grid()
        ax.legend()

        if (
            test_pars is not None
            and l2y_tst is not None
            and test_data is not None
        ):
            ax = axs_tst[0]
            assert isinstance(ax, Axes3D)
            sc_tst = ax.scatter(
                test_data["states"][i0_tst:i1_tst:plt_step_tst, 0],
                test_data["states"][i0_tst:i1_tst:plt_step_tst, 1],
                test_data["states"][i0_tst:i1_tst:plt_step_tst, 2],
                c=evec_tst.real[:num_plt_tst:plt_step_tst],
                s=1,
                vmin=-amax,
                vmax=amax,
                cmap="seismic",
            )
            ax.set_xlabel("$x^1$")
            ax.set_ylabel("$x^2$")
            ax.set_zlabel("$x^3$")
            ax.set_title(f"$\\mathrm{{Re}}\\zeta_{{{j}}}$ - test")

            ax = axs_tst[1]
            assert isinstance(ax, Axes3D)
            ax.scatter(
                test_data["states"][i0_tst:i1_tst:plt_step_tst, 0],
                test_data["states"][i0_tst:i1_tst:plt_step_tst, 1],
                test_data["states"][i0_tst:i1_tst:plt_step_tst, 2],
                c=evec_tst.imag[:num_plt_tst:plt_step_tst],
                s=1,
                vmin=-amax,
                vmax=amax,
                cmap="seismic",
            )
            ax.set_xlabel("$x^1$")
            ax.set_ylabel("$x^2$")
            ax.set_zlabel("$x^3$")
            ax.set_title(f"$\\mathrm{{Im}}\\zeta_{{{j}}}$")
            ax.grid()

            ax = axs_tst[2]
            ax.plot(
                evec_tst.real[:num_plt_tst:plt_step_tst],
                evec_tst.imag[:num_plt_tst:plt_step_tst],
                "-",
            )
            ax.set_xlabel(f"$\\mathrm{{Re}}\\zeta_{{{j}}}$")
            ax.set_ylabel(f"$\\mathrm{{Im}}\\zeta_{{{j}}}$")
            ax.grid()

            ax = axs_tst[3]
            ax.plot(
                ts_tst[:num_plt_tst:plt_step_tst],
                evec_tst.real[:num_plt_tst:plt_step_tst],
                "-",
            )
            ax.plot(
                ts_tst[:num_plt_tst:plt_step_tst],
                evec_tst.imag[:num_plt_tst:plt_step_tst],
                "-",
            )
            ax.set_xlabel("$t$")
            ax.grid()

        if test_pars is None:
            if len(fig.axes) > 4:
                fig.colorbar(
                    sc,
                    ax=axs[:2],
                    cax=fig.axes[4],
                    location="bottom",
                    shrink=0.75,
                    aspect=60,
                    pad=0,
                )
            else:
                fig.colorbar(
                    sc,
                    ax=axs[:2],
                    location="bottom",
                    shrink=0.75,
                    aspect=60,
                    pad=0,
                )
        else:
            if len(fig.axes) > 8:
                fig.colorbar(
                    sc_tst,
                    ax=axs_tst[:2],
                    cax=fig.axes[8],
                    location="bottom",
                    shrink=0.75,
                    aspect=30,
                    pad=0,
                )
            else:
                fig.colorbar(
                    sc_tst,
                    ax=axs_tst[:2],
                    location="bottom",
                    shrink=0.75,
                    aspect=30,
                    pad=0,
                )

    return fig, plot_eig


def make_running_pred_plotter[Ntst: int](
    test_pars: DataPars[Ntst],
    test_data: Data,
    preds: Vtsts,
    what: Literal["covariates", "responses"] = "responses",
    num_plt_tst: Optional[int] = None,
    plt_step_tst: int = 1,
    i_fig: int = 1,
) -> tuple[Figure, F[int, None]]:
    """Make plotting function for prediction over different lead times."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig = plt.figure(num=i_fig, figsize=tuple(mpf.figaspect(0.3)))
    axs = (
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d"),
    )
    fig.set_layout_engine("constrained")

    def plot_pred(i_step: int):
        i0_tst = test_pars.delay_embedding_end
        i1_tst = i0_tst + test_pars.num_samples
        i0_pred = i0_tst + i_step
        i1_pred = i1_tst + i_step
        match what:
            case "covariates":
                err = (
                    preds[:num_plt_tst, i_step, 0]
                    - test_data["covariates"][i0_pred:i1_pred, 0]
                )
                amax = max(
                    float(
                        jnp.max(
                            jnp.abs(
                                test_data["covariates"][i0_pred:i1_pred, 0]
                            )
                        )
                    ),
                    float(jnp.max(jnp.abs(preds[:, i_step, 0]))),
                )
            case "responses":
                err = (
                    preds[:num_plt_tst, i_step]
                    - test_data["responses"][i0_pred:i1_pred]
                )
                amax = max(
                    float(
                        jnp.max(
                            jnp.abs(test_data["responses"][i0_pred:i1_pred])
                        )
                    ),
                    float(jnp.max(jnp.abs(preds[:, i_step]))),
                )
        emax = float(jnp.max(jnp.abs(err)))
        for ax in axs:
            ax.cla()

        ax = axs[0]
        assert isinstance(ax, Axes3D)
        match what:
            case "covariates":
                c = test_data["covariates"][i0_pred:i1_pred:plt_step_tst, 0]
            case "responses":
                c = test_data["responses"][i0_pred:i1_pred:plt_step_tst]
        sc_tst = ax.scatter(
            test_data["states"][i0_tst:i1_tst:plt_step_tst, 0],
            test_data["states"][i0_tst:i1_tst:plt_step_tst, 1],
            test_data["states"][i0_tst:i1_tst:plt_step_tst, 2],
            c=c,
            s=1,
            vmin=-amax,
            vmax=amax,
            cmap="seismic",
        )
        ax.set_title("True")
        ax.set_xlabel("$x^1$")
        ax.set_ylabel("$x^2$")
        ax.set_zlabel("$x^3$")

        ax = axs[1]
        assert isinstance(ax, Axes3D)
        match what:
            case "covariates":
                c = preds[:num_plt_tst:plt_step_tst, i_step, 0]
            case "responses":
                c = preds[:num_plt_tst:plt_step_tst, i_step]
        ax.scatter(
            test_data["states"][i0_tst:i1_tst:plt_step_tst, 0],
            test_data["states"][i0_tst:i1_tst:plt_step_tst, 1],
            test_data["states"][i0_tst:i1_tst:plt_step_tst, 2],
            c=c,
            s=1,
            vmin=-amax,
            vmax=amax,
            cmap="seismic",
        )
        ax.set_xlabel("$x^1$")
        ax.set_ylabel("$x^2$")
        ax.set_zlabel("$x^3$")
        ax.set_title(f"Prediction; lead time = {i_step * test_pars.dt}")

        ax = axs[2]
        assert isinstance(ax, Axes3D)
        sc_err = ax.scatter(
            test_data["states"][i0_tst:i1_tst:plt_step_tst, 0],
            test_data["states"][i0_tst:i1_tst:plt_step_tst, 1],
            test_data["states"][i0_tst:i1_tst:plt_step_tst, 2],
            c=err[::plt_step_tst],
            s=1,
            vmin=-emax,
            vmax=emax,
            cmap="seismic",
        )
        ax.set_xlabel("$x^1$")
        ax.set_ylabel("$x^2$")
        ax.set_zlabel("$x^3$")
        ax.set_title("Error")

        if len(fig.axes) > 3:
            fig.colorbar(
                sc_tst,
                ax=axs[:2],
                cax=fig.axes[3],
                location="bottom",
                shrink=0.75,
                aspect=60,
                pad=0,
            )
            fig.colorbar(
                sc_err,
                ax=axs[2],
                cax=fig.axes[4],
                location="bottom",
                shrink=0.75,
                aspect=30,
                pad=0,
            )
        else:
            fig.colorbar(
                sc_tst,
                ax=axs[:2],
                location="bottom",
                shrink=0.75,
                aspect=60,
                pad=0,
            )
            fig.colorbar(
                sc_err,
                ax=axs[2],
                location="bottom",
                shrink=0.75,
                aspect=30,
                pad=0,
            )

    return fig, plot_pred


def make_pred_timeseries_plotter[Ntst: int](
    pars: DataPars[Ntst],
    test_data: Data,
    preds: Vtsts,
    what: Literal["covariates", "responses"] = "responses",
    i_fig: int = 1,
) -> tuple[Figure, F[int, None]]:
    """Make plotting function over different initial conditions."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    num_pred_steps = preds.shape[1] - 1
    ts = jnp.arange(num_pred_steps + 1) * pars.dt

    def plot_pred(i_init: int):
        i0_tst = pars.delay_embedding_end + i_init
        i1_tst = i0_tst + num_pred_steps + 1
        ax.cla()
        match what:
            case "covariates":
                ax.plot(
                    ts,
                    test_data["covariates"][i0_tst:i1_tst, 0],
                    "o-",
                    label="test",
                )
                ax.plot(ts, preds[i_init, :, 0], "o-", label="prediction")
            case "responses":
                ax.plot(
                    ts,
                    test_data["responses"][i0_tst:i1_tst],
                    "o-",
                    label="test",
                )
                ax.plot(ts, preds[i_init, :], "o-", label="prediction")
        ax.grid()
        ax.legend()
        ax.set_xlabel("Forecast lead time (model time units)")
        ax.set_title(f"Initial condition = {i_init}")

    return fig, plot_pred


def plot_response_forecast_skill_scores[Ntst: int](
    pars: DataPars[Ntst], scores: SkillScores, i_fig: int = 1
) -> Figure:
    """Plot NRMSE and ACC versus forecast lead time for responses."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, axs = plt.subplots(
        2, 1, num=i_fig, constrained_layout=True, sharex=True
    )
    labels = ("NRMSE", "Anomaly correlation")
    num_pred_steps = len(scores["nrmses"]) - 1
    ts = jnp.arange(num_pred_steps + 1) * pars.dt
    for ax, score, label in zip(
        axs, (scores["nrmses"], scores["accs"]), labels
    ):
        ax.plot(ts, score, "o-")
        ax.grid()
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("Forecast lead time (model time units)")
        ax.set_ylabel(label)
    return fig


def plot_covariate_forecast_skill_scores[Ntst: int](
    pars: DataPars[Ntst], scores: SkillScores, i_fig: int = 1
) -> Figure:
    """Plot NRMSE and ACC versus forecast lead time for covariates."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    match pars.covariate:
        case "x":
            figsize = None
            component_labels = (r"$x$",)
        case "y":
            figsize = None
            component_labels = (r"$y$",)
        case "z":
            figsize = None
            component_labels = (r"$z$",)
        case "xy":
            figsize = tuple(mpf.figaspect(0.6))
            component_labels = (
                r"$x$",
                r"$y$",
            )
        case "xyz":
            figsize = tuple(mpf.figaspect(0.4))
            component_labels = (r"$x$", r"$y$", r"$z$")
    fig, axss = plt.subplots(
        2,
        len(component_labels),
        num=i_fig,
        constrained_layout=True,
        sharex=True,
        sharey="row",
        figsize=figsize,
    )
    titles = [lbl + " component" for lbl in component_labels]
    score_labels = ("NRMSE", "Anomaly correlation")
    num_pred_steps = scores["nrmses"].shape[1] - 1
    ts = jnp.arange(num_pred_steps + 1) * pars.dt
    for axs, title, i in zip(axss.T, titles, range(len(component_labels))):
        for ax, score, label in zip(
            axs, (scores["nrmses"][i], scores["accs"][i]), score_labels
        ):
            ax.plot(ts, score, "o-")
            ax.grid()
            if ax.get_subplotspec().is_first_row():
                ax.set_title(title)
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel("Forecast time")
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel(label)
    return fig


def plot_forecast_skill_scores[Ntst: int](
    pars: DataPars[Ntst],
    scores: SkillScores,
    what: Literal["covariates", "responses"] = "responses",
    i_fig: int = 1,
) -> Figure:
    """Plot NRMSE and ACC versus forecast lead time."""
    match what:
        case "covariates":
            fig = plot_covariate_forecast_skill_scores(pars, scores, i_fig)
        case "responses":
            fig = plot_response_forecast_skill_scores(pars, scores, i_fig)
    return fig


def plot_reconstructed_trajectory[N: int](
    pars: DataPars[N],
    train_data: Data,
    recon_data: Ys,
    num_plt: Optional[int] = None,
    num_plt_tst: Optional[int] = None,
    plt_step: int = 1,
    plt_step_tst: int = 1,
    i_fig: int = 1,
) -> Figure:
    """Plot training and reconstructed dynamical trajectories."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig = plt.figure(num=i_fig, figsize=tuple(mpf.figaspect(0.5)))
    axs = (
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    )
    ax, ax_tst = axs
    assert isinstance(ax, Axes3D)
    assert isinstance(ax_tst, Axes3D)
    fig.set_layout_engine("constrained")
    if num_plt is None:
        num_plt = pars.num_samples
    i0 = pars.delay_embedding_end
    i1 = i0 + num_plt
    if num_plt_tst is None:
        num_plt_tst = len(recon_data)

    ax.plot(
        train_data["states"][i0:i1:plt_step, 0],
        train_data["states"][i0:i1:plt_step, 1],
        train_data["states"][i0:i1:plt_step, 2],
        "-",
    )
    ax.set_xlabel("$x^1$")
    ax.set_ylabel("$x^2$")
    ax.set_zlabel("$x^3$")
    ax.set_title("Training")

    ax_tst.plot(
        recon_data[:num_plt_tst:plt_step_tst, 0],
        recon_data[:num_plt_tst:plt_step_tst, 1],
        recon_data[:num_plt_tst:plt_step_tst, 2],
        "-",
    )
    ax_tst.set_xlabel("$x^1$")
    ax_tst.set_ylabel("$x^2$")
    ax_tst.set_zlabel("$x^3$")
    ax_tst.set_title("Reconstruction")

    return fig


def plot_trajectory_stats[Ntst: int](
    pars: DataPars[Ntst],
    train_stats: MultivariateTimeseriesStats,
    traj_stats: MultivariateTimeseriesStats,
    num_pred_steps: int,
    i_fig: int = 1,
) -> Figure:
    """Plot covariate PDFs and autocorrelation functions."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    match pars.covariate:
        case "x":
            figsize = None
            component_labels = (r"$x$",)
        case "y":
            figsize = None
            component_labels = (r"$y$",)
        case "z":
            figsize = None
            component_labels = (r"$z$",)
        case "xy":
            figsize = tuple(mpf.figaspect(0.6))
            component_labels = (
                r"$x$",
                r"$y$",
            )
        case "xyz":
            figsize = tuple(mpf.figaspect(0.4))
            component_labels = (r"$x$", r"$y$", r"$z$")
    num_vars = len(component_labels)
    fig, axss = plt.subplots(
        2,
        num_vars,
        num=i_fig,
        constrained_layout=True,
        figsize=figsize,
    )
    titles = [lbl + " component" for lbl in component_labels]
    data_labels = ("true", "recon.")
    y_labels = ("prob. density", "correlation")
    ts = jnp.arange(num_pred_steps + 1) * pars.dt
    for axs, title, i in zip(axss.T, titles, range(len(component_labels))):
        for stat, data_label in zip((train_stats, traj_stats), data_labels):
            bin_edges = stat["pdfs"][i]["bin_edges"]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            densities = stat["pdfs"][i]["densities"]
            autocorrs = stat["autocorrs"][i]
            axs[0].plot(bin_centers, densities, "-", label=data_label)
            axs[1].plot(ts, autocorrs, "-")
        for ax, y_label in zip(axs, y_labels):
            ax.grid()
            if ax.get_subplotspec().is_first_row():
                ax.set_title(title)
                if ax.get_subplotspec().is_first_col():
                    ax.legend()
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel("Lead time")
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel(y_label)
    return fig

# pyright: basic
"""Koopman spectral analysis of torus rotation by diffusion regularization."""

import jax
import jax.numpy as jnp
import matplotlib.figure as mpf
import matplotlib.pyplot as plt
import math
import nlsa.abstract_algebra as alg
import nlsa.jax.delays as dl
import nlsa.jax.distance as dst
import nlsa.jax.dynamics as dyn
import nlsa.jax.kernels as knl
import nlsa.jax.koopman as koop
import nlsa.jax.vector_algebra as vec
import nlsa.jax.torus as torus
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from jax import Array, jit, vmap
from jax.typing import DTypeLike
from matplotlib.figure import Figure
from nlsa.function_algebra import compose, compose2
from nlsa.io_actions import IO, h5it, npyit, pickleit, plotit, plotem, timeit
from nlsa.jax.kernels import (
    ConePars,
    KernelEigen,
    KernelEigenbasis,
    KernelPars,
    TuneInfo,
    TunePars,
)
from nlsa.jax.koopman import KoopmanEigen, KoopmanEigenbasis, KoopmanParsDiff
from nlsa.jax.stats import anomaly_correlation, normalized_rmse
from nlsa.jax.vector_algebra import L2VectorAlgebra, VectorAlgebra
from nlsa.jax.utils import fst, make_batched2
from pathlib import Path
from numpy.typing import ArrayLike
from tabulate import tabulate
from typing import Literal, Optional, TypedDict

IDX_CPU: Optional[int] = None
IDX_GPU: Optional[int] = None
XLA_MEM_FRACTION: Optional[str] = "0.9"
FP: Literal["F32", "F64"] = "F32"
OUTPUT_DATA_DIR = "examples/torus_rotation/data"
NUM_TABULATE = 40
DELAY_EMBEDDING_MODE: Optional[Literal["explicit", "on_the_fly"]] = (
    "on_the_fly"
)
GENERATE_DATA_MODE: Literal["calc", "calcsave", "read"] = "calcsave"
TUNE_KERNEL_MODE: Literal["calc", "calcsave", "read"] = "calcsave"
KERNEL_EIGEN_MODE: Literal["calc", "calcsave", "read"] = "calcsave"
GENERATOR_MATRIX_MODE: Literal["calc", "calcsave", "read"] = "calcsave"
KOOPMAN_EIGEN_MODE: Literal["calc", "calcsave", "read"] = "calcsave"
SKILL_SCORES_MODE: Literal["calc", "calcsave", "read"] = "calcsave"
PLOT_MODE: Optional[Literal["save", "show", "saveshow"]] = "show"
DELAY_PLOT_MODE: Literal["backward", "central"] = "backward"
KERNEL_EIGS_PLT: Optional[Sequence[int] | Literal["interactive"]] = (
    "interactive"
)
KOOPMAN_EIGS_PLT: Optional[Sequence[int] | Literal["interactive"]] = (
    "interactive"
)
LEAD_TIMES_PLT: Optional[Sequence[int] | Literal["interactive"]] = (
    "interactive"
)
INITIALIZATION_TIMES_PLT: Optional[Sequence[int] | Literal["interactive"]] = (
    "interactive"
)

if IDX_GPU is not None and XLA_MEM_FRACTION is not None:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = XLA_MEM_FRACTION

if IDX_CPU is not None:
    jax.config.update("jax_default_device", jax.devices("cpu")[IDX_CPU])

if IDX_GPU is not None:
    jax.config.update("jax_default_device", jax.devices("gpu")[IDX_GPU])

match FP:
    case "F32":
        r_dtype = jnp.float32
        c_dtype = jnp.complex64
    case "F64":
        jax.config.update("jax_enable_x64", True)
        r_dtype = jnp.float64
        c_dtype = jnp.complex128

type Xs = Array  # Collection of points in state space
type Ys = Array  # Collection of points in covariate space
type Yd = Array  # Point in delay-coordinate space
type R = Array  # Real number
type Rs = Array  # Collection of real numbers
type C = Array  # Complex number
type Cs = Array  # Collection of complex numbers
type M = Array  # Matrix
type V = Array  # Vector in L2
type Vs = Array  # Collection of vectors in L2
type Vtsts = Array  # Collection of vectors in L2 with respect to test dataset
type F[*Ss, T] = Callable[[*Ss], T]  # Shorthand for Callables

jvmap: Callable[[F[Array, Array]], F[Array, Array]] = compose(jit, vmap)


@dataclass(frozen=True)
class DataPars[N: int]:
    """Dataclass containing training and test data parameter values."""

    covariate: Literal["cos", "r3", "r4", "von_mises", "von_mises_grad"]
    """Covariate function."""

    response: Literal["cos", "von_mises", "von_mises_grad"]
    """Response function."""

    rot_freqs: tuple[float, float]
    """Rotation frequencies of the dynamics."""

    dt: float
    """Sampling interval."""

    x0: tuple[float, float]
    """Initial condition."""

    num_samples: N
    """Number of analysis samples."""

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

    covariate_von_mises_locs: Optional[tuple[float, float]] = None
    """von Mises location parameters for covariate."""

    covariate_von_mises_concs: Optional[tuple[float, float]] = None
    """von Mises concentration parameters for covariate."""

    response_von_mises_locs: Optional[tuple[float, float]] = None
    """von Mises location parameters for response."""

    response_von_mises_concs: Optional[tuple[float, float]] = None
    """von Mises concentration parameters for response."""

    batch_size: Optional[int] = None
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
        """Total number of samples."""
        return (
            self.num_samples
            + self.num_velocity_fd
            + self.num_delays
            + self.num_before
            + self.num_after
        )

    def __str__(self) -> str:
        """Create string representation of data parameters."""
        rot_freqs_str = f"freq{self.rot_freqs[0]:.2f}_{self.rot_freqs[1]:.2f}"
        x0_str = f"x0{self.x0[0]:.2f}_{self.x0[1]:.2f}"
        if self.velocity_covariate:
            assert self.velocity_fd_order is not None
            vel_str = f"vfd{self.velocity_fd_order}"
        else:
            vel_str = ""
        if "von_mises" in self.covariate:
            assert self.covariate_von_mises_locs is not None
            assert self.covariate_von_mises_concs is not None
            cov_vm_str = "_".join(
                (
                    "mu"
                    + "_".join(
                        (
                            f"{self.covariate_von_mises_locs[0]:.3g}",
                            f"{self.covariate_von_mises_locs[1]:.3g}",
                        )
                    ),
                    "kappa"
                    + "_".join(
                        (
                            f"{self.covariate_von_mises_concs[0]:.3g}",
                            f"{self.covariate_von_mises_concs[1]:.3g}",
                        )
                    ),
                )
            )
        else:
            cov_vm_str = ""
        if "von_mises" in self.response:
            assert self.response_von_mises_locs is not None
            assert self.response_von_mises_concs is not None
            resp_vm_str = "_".join(
                (
                    "mu"
                    + "_".join(
                        (
                            f"{self.response_von_mises_locs[0]:.3g}",
                            f"{self.response_von_mises_locs[1]:.3g}",
                        )
                    ),
                    "kappa_"
                    + "_".join(
                        (
                            f"{self.response_von_mises_concs[0]:.3g}",
                            f"{self.response_von_mises_concs[1]:.3g}",
                        )
                    ),
                )
            )
        else:
            resp_vm_str = ""
        return "_".join(
            filter(
                None,
                (
                    rot_freqs_str,
                    f"dt{self.dt:.2f}",
                    x0_str,
                    self.covariate,
                    cov_vm_str,
                    self.response,
                    resp_vm_str,
                    f"ns{self.num_samples}",
                    f"nd{self.num_delays}",
                    f"nb{self.num_before}",
                    f"na{self.num_after}",
                    vel_str,
                ),
            )
        )


@dataclass(frozen=True)
class PredPars:
    """Dataclass containing prediction parameters."""

    dt: float
    """Prediction timestep."""

    num_steps: int
    """Number of timesteps for prediction."""

    which_eigs: int | tuple[int, int] | list[int]
    """Kernel eigenfunctions used in the prediction function."""

    def __str__(self) -> str:
        """Create string representation of prediction parameters."""
        match self.which_eigs:
            case int():
                eigs_str = "-".join(map(str, (0, self.which_eigs)))
            case (_, _):
                eigs_str = "-".join(map(str, self.which_eigs))
            case list():
                eigs_str = "_".join(map(str, self.which_eigs))
        return "_".join(
            (
                "pred",
                f"dt{self.dt:.2g}",
                f"nsteps{self.num_steps}",
                "neigs" + eigs_str,
            )
        )


@dataclass(frozen=True)
class TrainPars[N: int]:
    """Dataclass containing the training parameter values."""

    data: DataPars[N]
    """Training data parameters."""

    tune: TunePars
    """Kernel tuning parameters."""

    kernel: KernelPars
    """Kernel eigendecomposition parameters."""

    koopman: KoopmanParsDiff
    """Koopman operator approximation parameters."""

    pred: PredPars
    """Prediction parameters."""

    cone: Optional[ConePars] = None
    """Cone kernel parameters."""

    bw_tune: Optional[TunePars] = None
    """Tuning parameters for kernel bandwidth function."""

    def __str__(self) -> str:
        """Create string representation of training parameters."""
        if self.cone is not None:
            cone_str = str(self.cone)
        else:
            cone_str = ""
        if self.bw_tune is None:
            bw_tune_str = ""
        else:
            bw_tune_str = str(self.bw_tune)
        return "_".join(
            filter(
                None,
                (
                    str(self.data),
                    str(self.tune),
                    bw_tune_str,
                    cone_str,
                    str(self.kernel),
                    str(self.koopman),
                    str(self.pred),
                ),
            )
        )


@dataclass(frozen=True)
class TestPars[Ntst: int]:
    """Dataclass containing test parameter values."""

    num_pred_steps: int
    """Number of prediction steps."""

    data: DataPars[Ntst]
    """Test data parameters."""

    max_batch_size: Optional[int] = None
    """Max batch size for evalation of prediction function."""


@dataclass(frozen=True)
class Pars[N: int, Ntst: int]:
    """Dataclass containing the parameter values used in this example."""

    train: TrainPars[N]
    test: TestPars[Ntst]


class Data(TypedDict):
    """TypedDict containing training data."""

    states: Xs
    """Dynamical states."""

    covariates: Ys
    """Covariate variables."""

    responses: Rs
    """Response variables."""


class SkillScores(TypedDict):
    """TypedDict containing prediction skill scores."""

    nrmses: Rs
    """Normalized RMSE scores."""

    accs: Rs
    """Anomaly correlation scores."""


class CommonPars(TypedDict):
    """Helper TypedDict to check common training/test parameter values."""

    covariate: Literal["cos", "r3", "r4", "von_mises", "von_mises_grad"]
    response: Literal["cos", "von_mises", "von_mises_grad"]
    covariate_von_mises_locs: Optional[tuple[float, float]]
    covariate_von_mises_concs: Optional[tuple[float, float]]
    response_von_mises_locs: Optional[tuple[float, float]]
    response_von_mises_concs: Optional[tuple[float, float]]
    dt: float
    rot_freqs: tuple[float, float]
    num_half_delays: int
    velocity_covariate: bool
    velocity_fd_order: Optional[Literal[2, 4, 6, 8]]


fd_order = 4
cone_pars = None
# cone_pars = ConePars(zeta=0.99)
common_pars: CommonPars = {
    "covariate": "r3",
    "covariate_von_mises_concs": (3, 3),
    "covariate_von_mises_locs": (jnp.pi, jnp.pi),
    "response": "von_mises",
    "response_von_mises_concs": (3, 3),
    "response_von_mises_locs": (jnp.pi, jnp.pi),
    "rot_freqs": (1, math.sqrt(30)),
    "dt": jnp.pi / math.sqrt(1313),
    "num_half_delays": 0,
    "velocity_covariate": True if cone_pars is not None else False,
    "velocity_fd_order": fd_order if cone_pars is not None else None,
}
num_pred_steps = 50
train_data_pars = DataPars(
    **common_pars,
    x0=(jnp.pi / math.sqrt(13), jnp.pi / math.sqrt(13)),
    num_samples=4096,
    num_before=fd_order // 2,
    num_after=fd_order // 2,
)
test_data_pars = DataPars(
    **common_pars,
    x0=(jnp.pi / math.sqrt(7), jnp.pi / math.sqrt(7)),
    num_samples=1024,
    num_before=0,
    num_after=num_pred_steps,
)
bw_tune_pars = TunePars(
    manifold_dim=2,
    num_bandwidths=128,
    log10_bandwidth_lims=(-3, 3),
    bandwidth_scl=1,
)
if cone_pars is not None:
    tune_pars = TunePars(
        manifold_dim=1.52,
        num_bandwidths=128,
        log10_bandwidth_lims=(-3, 3),
        bandwidth_scl=1,
    )
else:
    tune_pars = bw_tune_pars
kernel_pars = KernelPars(
    normalization="laplace", eigensolver="eigsh", num_eigs=256
)
koopman_pars = KoopmanParsDiff(
    fd_order=fd_order,
    dt=common_pars["dt"],
    antisym=True,
    tau=0.002,
    which_eigs_galerkin=250,
    sort_by="energy",
    gram_batch_size=None,
)
pred_pars = PredPars(
    dt=test_data_pars.dt, num_steps=num_pred_steps, which_eigs=129
)
train_pars = TrainPars(
    data=train_data_pars,
    bw_tune=bw_tune_pars,
    cone=cone_pars,
    tune=tune_pars,
    kernel=kernel_pars,
    koopman=koopman_pars,
    pred=pred_pars,
)
test_pars = TestPars(data=test_data_pars, num_pred_steps=pred_pars.num_steps)
pars = Pars(train=train_pars, test=test_pars)
io = IO(root=Path.cwd() / OUTPUT_DATA_DIR)


def to_data(
    dict_in: dict[str, ArrayLike], dtype: Optional[DTypeLike] = None
) -> Data:
    """Convert dictionary of numpy ArrayLike objects to Data TypedDict."""
    try:
        data: Data = {
            "states": jnp.array(dict_in["states"], dtype),
            "covariates": jnp.array(dict_in["covariates"], dtype),
            "responses": jnp.array(dict_in["responses"], dtype),
        }
        return data
    except ValueError as exc:
        raise ValueError("Incompatible keys/values") from exc


def to_kernel_eigen(
    dict_in: dict[str, ArrayLike], dtype: Optional[DTypeLike] = None
) -> KernelEigen:
    """Convert dict of numpy ArrayLike objects to KernelEigen TypedDict."""
    try:
        kernel_eigen: KernelEigen = {
            "evals": jnp.array(dict_in["evals"], dtype),
            "evecs": jnp.array(dict_in["evecs"], dtype),
            "dual_evecs": jnp.array(dict_in["dual_evecs"], dtype),
            "weights": jnp.array(dict_in["weights"], dtype),
            "bandwidth": jnp.array(dict_in["bandwidth"], dtype),
        }
        return kernel_eigen
    except ValueError as exc:
        raise ValueError("Incompatible keys/values") from exc


def to_koopman_eigen(
    dict_in: dict[str, ArrayLike], dtype: Optional[DTypeLike] = None
) -> KoopmanEigen:
    """Convert dict of numpy ArrayLike objects to KoopmanEigen TypedDict."""
    try:
        koopman_eigen: KoopmanEigen = {
            "evals": jnp.array(dict_in["evals"], dtype),
            "gen_evals": jnp.array(dict_in["gen_evals"], dtype),
            "efreqs": jnp.array(dict_in["efreqs"], dtype),
            "eperiods": jnp.array(dict_in["eperiods"], dtype),
            "evec_coeffs": jnp.array(dict_in["evec_coeffs"], dtype),
            "dual_evec_coeffs": jnp.array(dict_in["dual_evec_coeffs"], dtype),
            "engys": jnp.array(dict_in["engys"], dtype),
        }
        return koopman_eigen
    except ValueError as exc:
        raise ValueError("Incompatible keys/values") from exc


def to_skill_scores(
    dict_in: dict[str, ArrayLike], dtype: Optional[DTypeLike] = None
) -> SkillScores:
    """Convert dict of numpy ArrayLike objects to SkillScores TypedDict."""
    try:
        skill_scores: SkillScores = {
            "nrmses": jnp.array(dict_in["nrmses"], dtype),
            "accs": jnp.array(dict_in["accs"], dtype),
        }
        return skill_scores
    except ValueError as exc:
        raise ValueError("Incompatible keys/values") from exc


tune_kernel_bandwidth = timeit(
    pickleit(
        knl.tune_bandwidth,
        io=io,
        mode=TUNE_KERNEL_MODE,
        fname="tune_kernel",
        cls=tuple[Array, TuneInfo],
    )
)
compute_kernel_eigen = timeit(
    h5it(
        knl.compute_eigen,
        io=io,
        mode=KERNEL_EIGEN_MODE,
        fname="kernel_eigen",
        cls=KernelEigen,
        callback=partial(to_kernel_eigen, dtype=r_dtype),
    )
)


@timeit
@partial(
    h5it,
    io=io,
    mode=GENERATE_DATA_MODE,
    fname="data",
    cls=Data,
    callback=partial(to_data, dtype=r_dtype),
)
def generate_data[N: int](pars: DataPars[N], dtype: DTypeLike) -> Data:
    """Generate rotation data on the 2-torus."""
    match pars.covariate:
        case "r3":
            cov = torus.make_observable_r3(dtype=dtype)
        case "r4":
            cov = torus.make_observable_r4(dtype)
        case "cos":
            cov = torus.make_observable_cos(dtype, asvector=True)
        case "von_mises":
            assert pars.covariate_von_mises_concs is not None
            assert pars.covariate_von_mises_locs is not None
            cov = torus.make_observable_von_mises(
                pars.covariate_von_mises_concs,
                pars.covariate_von_mises_locs,
                dtype,
                asvector=True,
            )
        case "von_mises_grad":
            assert pars.covariate_von_mises_concs is not None
            assert pars.covariate_von_mises_locs is not None
            cov = torus.make_observable_von_mises_grad(
                pars.covariate_von_mises_concs,
                pars.covariate_von_mises_locs,
                dtype,
                asvector=True,
            )
    match pars.response:
        case "cos":
            rsp = torus.make_observable_cos(dtype)
        case "von_mises":
            assert pars.response_von_mises_concs is not None
            assert pars.response_von_mises_locs is not None
            rsp = torus.make_observable_von_mises(
                pars.response_von_mises_concs,
                pars.response_von_mises_locs,
                dtype,
            )
        case "von_mises_grad":
            assert pars.response_von_mises_concs is not None
            assert pars.response_von_mises_locs is not None
            rsp = torus.make_observable_von_mises_grad(
                pars.response_von_mises_concs,
                pars.response_von_mises_locs,
                dtype,
            )
    covariate = jvmap(cov)
    response = jvmap(rsp)
    rot_angles = [rot_freq * pars.dt for rot_freq in pars.rot_freqs]
    dyn_map = dyn.make_rotation_map(rot_angles)
    orb = dyn.make_fin_orbit(dyn_map, pars.num_total_samples)
    xs = orb(jnp.array(pars.x0, dtype=dtype))
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
    delay_embedding_mode: Optional[
        Literal["explicit", "on_the_fly"]
    ] = DELAY_EMBEDDING_MODE,
    jit: bool = False,
) -> L2VectorAlgebra[tuple[N], D, Yd, R]:
    """Make L2 space over covariate data space."""
    i0 = pars.delay_embedding_origin
    i1 = i0 + pars.num_delay_samples
    if pars.velocity_covariate:
        hankel = jax.jit(
            vmap(
                partial(dl.hankel, num_delays=pars.num_delays, flatten=True),
                in_axes=1,
                out_axes=1,
            )
        )
    else:
        hankel = jax.jit(
            partial(dl.hankel, num_delays=pars.num_delays, flatten=True)
        )
    incl: Callable[[F[Array, Array]], V]
    match pars.num_half_delays, pars.batch_size:
        case 0, None:
            incl = vec.veval_at(data["covariates"][i0:i1], jit=jit)
        case 0, _:
            incl = vec.batch_eval_at(
                data["covariates"][i0:i1], batch_size=pars.batch_size
            )
        case _, None:
            assert delay_embedding_mode is not None
            if delay_embedding_mode == "explicit":
                incl = vec.veval_at(hankel(data["covariates"][i0:i1]))
            elif delay_embedding_mode == "on_the_fly":
                incl = dl.delay_eval_at(
                    data["covariates"][i0:i1],
                    num_delays=pars.num_delays,
                    jit=jit,
                )
        case _, _:
            assert delay_embedding_mode is not None
            if delay_embedding_mode == "explicit":
                incl = vec.batch_eval_at(
                    hankel(data["covariates"][i0:i1]),
                    batch_size=pars.batch_size,
                )
            elif delay_embedding_mode == "on_the_fly":
                incl = dl.batch_delay_eval_at(
                    data["covariates"][i0:i1],
                    batch_size=pars.batch_size,
                    num_delays=pars.num_delays,
                )
    mu = vec.make_normalized_counting_measure(pars.num_samples)
    return L2VectorAlgebra(
        shape=(pars.num_samples,), dtype=dtype, measure=mu, inclusion_map=incl
    )


@timeit
@partial(
    npyit,
    io=io,
    mode=GENERATOR_MATRIX_MODE,
    fname="gen_mat",
    cls=Array,
    callback=partial(jnp.array, dtype=r_dtype),
)
def compute_generator_matrix[N: int, D: DTypeLike](
    pars: TrainPars[N],
    l2y: L2VectorAlgebra[tuple[N], D, Yd, R],
    train_data: Data,
    basis: alg.ImplementsDimensionedL2FnFrame[Yd, R, V, Rs, int | Array],
) -> M:
    """Compute matrix representation of the generator."""
    if pars.data.velocity_covariate:
        fd_op = jit(
            vmap(
                vmap(
                    dl.make_fd_operator(
                        order=pars.koopman.fd_order,
                        mode="central",
                        dt=pars.koopman.dt,
                    ),
                    in_axes=-1,
                    out_axes=-1,
                ),
                in_axes=-1,
                out_axes=-1,
            )
        )
    else:
        fd_op = jit(
            vmap(
                dl.make_fd_operator(
                    order=pars.koopman.fd_order,
                    mode="central",
                    dt=pars.koopman.dt,
                ),
                in_axes=-1,
                out_axes=-1,
            )
        )
    i0 = pars.data.delay_embedding_origin
    i1 = i0 + pars.data.num_samples + pars.data.num_delays
    vs = fd_op(train_data["covariates"])[i0:i1]
    match pars.data.num_half_delays:
        case 0:
            eval_at_ys = vec.veval_at(train_data["covariates"][i0:i1])
            eval_at_yvs = vec.veval_at((train_data["covariates"][i0:i1], vs))
        case _:
            eval_at_ys = dl.delay_eval_at(
                train_data["covariates"][i0:i1],
                num_delays=pars.data.num_delays,
            )
            eval_at_yvs = dl.delay_eval_at(
                (train_data["covariates"][i0:i1], vs),
                num_delays=pars.data.num_delays,
            )
    vgrad_phi = compose(eval_at_yvs, compose(dyn.vgrad, basis.fn))
    phi = compose(eval_at_ys, basis.dual_fn)

    @jit
    @partial(vmap, in_axes=(None, 0), out_axes=1)
    @partial(vmap, in_axes=(0, None), out_axes=0)
    def compute_generator(i: int, j: int) -> R:
        return l2y.innerp(phi(i), vgrad_phi(j))

    idxs = jnp.arange(1, basis.dim)
    if pars.koopman.gram_batch_size is not None:
        compute_generator_batched = make_batched2(
            compute_generator,
            max_batch_sizes=(
                pars.koopman.gram_batch_size,
                pars.koopman.gram_batch_size,
            ),
            in_axes=(0, 0),
        )
        gen_mat = compute_generator_batched(idxs, idxs)
    else:
        gen_mat = compute_generator(idxs, idxs)
    return gen_mat


compute_koopman_eigen = timeit(
    h5it(
        koop.compute_eigen_diff,
        io=io,
        mode=KOOPMAN_EIGEN_MODE,
        fname="koopman_eigen_diff",
        cls=KoopmanEigen,
        callback=to_koopman_eigen,
    )
)


def make_timeseries_prediction_function[N: int](
    pars: TrainPars[N],
    train_data: Data,
    koopman_basis: KoopmanEigenbasis[Yd, C, V, Cs, int | Array],
) -> F[Yd, Cs]:
    """Make vector-valued prediction function for time series prediction."""
    i0 = pars.data.delay_embedding_end
    i1 = i0 + pars.data.num_samples
    f_coeffs = koopman_basis.anal(train_data["responses"][i0:i1])
    ts = pars.pred.dt * jnp.arange(pars.pred.num_steps + 1)

    @partial(vmap, in_axes=(0, None))
    def predict(t: R, y: Yd) -> C:
        phases = jnp.exp(1j * koopman_basis.efreqs * t)
        return koopman_basis.fn_synth(phases * f_coeffs)(y)

    return partial(predict, ts)


@timeit
@partial(
    h5it,
    io=io,
    mode=SKILL_SCORES_MODE,
    fname="pred_scores",
    cls=SkillScores,
    callback=to_skill_scores,
)
def compute_skill_scores[Ntst: int](
    pars: TestPars[Ntst], test_data: Data, fys_pred: Vtsts
) -> SkillScores:
    """Compute NRMSE and ACC skill scores over the prediction ensemble."""
    i0 = pars.data.delay_embedding_end
    i1 = i0 + pars.num_pred_steps + pars.data.num_samples
    fxs_true = dl.hankel(
        test_data["responses"][i0:i1], num_delays=pars.num_pred_steps
    )
    nrmses = jax.jit(vmap(normalized_rmse, in_axes=1))(fxs_true, fys_pred)
    accs = jax.jit(vmap(anomaly_correlation, in_axes=1))(fxs_true, fys_pred)
    scores: SkillScores = {"nrmses": nrmses, "accs": accs}
    return scores


plot_kernel_tuning = plotit(
    knl.plot_kernel_tuning,
    io=io,
    mode=PLOT_MODE,
    fname="bandwidth_tuning_func",
)
plot_laplace_spectrum = plotit(
    knl.plot_laplace_spectrum, io=io, mode=PLOT_MODE, fname="lapl_spec"
)


@partial(plotit, io=io, mode=PLOT_MODE, fname="bandwidth_func")
def plot_bandwidth_func[N: int, Ntst: int, D: DTypeLike](
    pars: DataPars[N],
    l2y: L2VectorAlgebra[tuple[N], D, Yd, R],
    bandwidth_func: F[Yd, R],
    train_data: Data,
    test_pars: Optional[DataPars[Ntst]] = None,
    l2y_tst: Optional[L2VectorAlgebra[tuple[Ntst], D, Yd, R]] = None,
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
        fig, axs = plt.subplots(
            1,
            2,
            num=i_fig,
            figsize=tuple(mpf.figaspect(0.5)),
            constrained_layout=True,
            sharey=True,
            subplot_kw={"box_aspect": 1},
        )
        ax, ax_tst = axs
    else:
        fig, ax = plt.subplots(
            num=i_fig, constrained_layout=True, subplot_kw={"box_aspect": 1}
        )
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
        train_data["states"][i0:i1:plt_step, 0] / jnp.pi,
        train_data["states"][i0:i1:plt_step, 1] / jnp.pi,
        c=bw_vals[::plt_step],
        s=1,
        vmin=vmin,
        vmax=vmax,
        cmap="binary",
    )
    ax.set_xlabel(r"$\theta_1/\pi$")
    ax.set_ylabel(r"$\theta_2/\pi$")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_facecolor("orange")
    ax.set_title("Kernel bandwidth function (training)")

    if test_pars is not None and l2y_tst is not None and test_data is not None:
        if num_plt_tst is None:
            num_plt_tst = test_pars.num_samples
        i1_tst = i0 + num_plt_tst
        sc_tst = ax_tst.scatter(
            test_data["states"][i0:i1_tst:plt_step_tst, 0] / jnp.pi,
            test_data["states"][i0:i1_tst:plt_step_tst, 1] / jnp.pi,
            c=bw_vals_tst[::plt_step_tst],
            s=1,
            vmin=vmin,
            vmax=vmax,
            cmap="binary",
        )
        ax_tst.set_xlabel(r"$\theta_1/\pi$")
        ax_tst.set_xlim(0, 2)
        ax_tst.set_ylim(0, 2)
        ax_tst.set_title("Kernel bandwidth function (test)")
        ax_tst.set_facecolor("orange")
        fig.colorbar(sc_tst, ax=ax_tst)
    else:
        fig.colorbar(sc, ax=ax)

    plt.rcdefaults()
    return fig


@partial(plotem, io=io, mode=PLOT_MODE, fname="kernel_eigen")
def make_kernel_evecs_plotter[N: int, Ntst: int, D: DTypeLike](
    pars: DataPars[N],
    train_data: Data,
    kernel_basis: KernelEigenbasis[Yd, R, V, Rs, int | Array],
    test_pars: Optional[DataPars[Ntst]] = None,
    l2y_tst: Optional[L2VectorAlgebra[tuple[Ntst], D, Yd, R]] = None,
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
        fig, axs = plt.subplots(
            1,
            2,
            num=i_fig,
            figsize=tuple(mpf.figaspect(0.5)),
            constrained_layout=True,
            sharey=True,
            subplot_kw={"box_aspect": 1},
        )
        ax, ax_tst = axs
    else:
        fig, ax = plt.subplots(
            num=i_fig, constrained_layout=True, subplot_kw={"box_aspect": 1}
        )
    match delay_plot_mode:
        case "backward":
            i0 = pars.delay_embedding_end
        case "central":
            i0 = pars.delay_embedding_center
    if num_plt is None:
        num_plt = pars.num_samples
    i1 = i0 + num_plt
    if test_pars is not None:
        if num_plt_tst is None:
            num_plt_tst = test_pars.num_samples
        i1_tst = i0 + num_plt_tst

    def plot_eigs(j: int):
        evec = kernel_basis.vec(j)
        amax = float(jnp.max(jnp.abs(evec)))
        if (
            test_pars is not None
            and l2y_tst is not None
            and test_data is not None
        ):
            evec_tst = l2y_tst.incl(kernel_basis.fn(j))
            amax = max(amax, float(jnp.max(jnp.abs(evec_tst))))
        for figax in fig.axes:
            figax.cla()

        sc = ax.scatter(
            train_data["states"][i0:i1:plt_step, 0] / jnp.pi,
            train_data["states"][i0:i1:plt_step, 1] / jnp.pi,
            c=evec[:num_plt:plt_step],
            s=1,
            vmin=-amax,
            vmax=amax,
            cmap="seismic",
        )
        ax.set_xlabel(r"$\theta_1/\pi$")
        ax.set_ylabel(r"$\theta_2/\pi$")
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        eta = kernel_basis.lapl_evl(j)
        ax.set_title(f"Eigenvector {j}: $\\eta_{{{j}}} = {eta: .3f}$")

        if (
            test_pars is not None
            and l2y_tst is not None
            and test_data is not None
        ):
            sc_tst = ax_tst.scatter(
                test_data["states"][i0:i1_tst:plt_step_tst, 0] / jnp.pi,
                test_data["states"][i0:i1_tst:plt_step_tst, 1] / jnp.pi,
                c=evec_tst[:num_plt_tst:plt_step_tst],
                s=1,
                vmin=-amax,
                vmax=amax,
                cmap="seismic",
            )
            ax_tst.set_xlabel(r"$\theta_1/\pi$")
            ax_tst.set_xlim(0, 2)
            ax_tst.set_ylim(0, 2)
            ax_tst.set_title("Nystrom")

        if test_pars is not None:
            if len(fig.axes) > 2:
                fig.colorbar(sc_tst, ax=ax_tst, cax=fig.axes[2])
            else:
                fig.colorbar(sc_tst, ax=ax_tst)
        else:
            if len(fig.axes) > 1:
                fig.colorbar(sc, ax=ax, cax=fig.axes[2])
            else:
                fig.colorbar(sc, ax=ax)

    return fig, plot_eigs


plot_generator_matrix = plotit(
    koop.plot_operator_matrix, io=io, mode=PLOT_MODE, fname="gen_mat"
)
plot_generator_spectrum = plotit(
    koop.plot_generator_spectrum, io=io, mode=PLOT_MODE, fname="gen_spec"
)


@partial(plotem, io=io, mode=PLOT_MODE, fname="zeta")
def make_koopman_evecs_plotter[N: int, Ntst: int, D: DTypeLike](
    pars: DataPars[N],
    train_data: Data,
    koopman_basis: KoopmanEigenbasis[Yd, C, V, Cs, int | Array],
    test_pars: Optional[DataPars[Ntst]] = None,
    l2y_tst: Optional[L2VectorAlgebra[tuple[Ntst], D, Yd, R]] = None,
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
        fig, axss = plt.subplots(
            2,
            4,
            num=i_fig,
            figsize=tuple(1.5 * mpf.figaspect(0.6)),
            constrained_layout=True,
            subplot_kw={"box_aspect": 1},
        )
        axs, axs_tst = axss
    else:
        fig, axs = plt.subplots(
            1,
            4,
            num=i_fig,
            figsize=tuple(mpf.figaspect(0.5)),
            constrained_layout=True,
            subplot_kw={"box_aspect": 1},
        )
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
        if num_plt_tst is None:
            num_plt_tst = test_pars.num_samples
        i1_tst = i0 + num_plt_tst
        ts_tst = jnp.arange(num_plt_tst) * test_pars.dt

    def plot_eigs(j: int):
        evec = koopman_basis.vec(j)
        evl = koopman_basis.evl(j)
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
        sc = ax.scatter(
            train_data["states"][i0:i1:plt_step, 0] / jnp.pi,
            train_data["states"][i0:i1:plt_step, 1] / jnp.pi,
            c=evec.real[:num_plt:plt_step],
            s=1,
            vmin=-amax,
            vmax=amax,
            cmap="seismic",
        )
        ax.set_xlabel(r"$\theta_1/\pi$")
        ax.set_ylabel(r"$\theta_2/\pi$")
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_title(f"$\\mathrm{{Re}}\\zeta_{{{j}}}$ - training")

        ax = axs[1]
        ax.scatter(
            train_data["states"][i0:i1:plt_step, 0] / jnp.pi,
            train_data["states"][i0:i1:plt_step, 1] / jnp.pi,
            c=evec.imag[:num_plt:plt_step],
            s=1,
            vmin=-amax,
            vmax=amax,
            cmap="seismic",
        )
        ax.set_xlabel(r"$\theta_1/\pi$")
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
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
        ax.set_title(f"Growth rate $\\gamma_{{{j}}} = {evl.real: .3f}$")
        ax.set_xlabel("$t$")
        ax.grid()
        ax.legend()

        if (
            test_pars is not None
            and l2y_tst is not None
            and test_data is not None
        ):
            ax = axs_tst[0]
            sc_tst = ax.scatter(
                test_data["states"][i0:i1_tst:plt_step_tst, 0] / jnp.pi,
                test_data["states"][i0:i1_tst:plt_step_tst, 1] / jnp.pi,
                c=evec_tst.real[:num_plt_tst:plt_step_tst],
                s=1,
                vmin=-amax,
                vmax=amax,
                cmap="seismic",
            )
            ax.set_xlabel(r"$\theta_1/\pi$")
            ax.set_ylabel(r"$\theta_2/\pi$")
            ax.set_title(f"$\\mathrm{{Re}}\\zeta_{{{j}}}$ - test")
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)

            ax = axs_tst[1]
            ax.scatter(
                test_data["states"][i0:i1_tst:plt_step_tst, 0] / jnp.pi,
                test_data["states"][i0:i1_tst:plt_step_tst, 1] / jnp.pi,
                c=evec_tst.imag[:num_plt_tst:plt_step_tst],
                s=1,
                vmin=-amax,
                vmax=amax,
                cmap="seismic",
            )
            ax.set_xlabel(r"$\theta_1/\pi$")
            ax.set_title(f"$\\mathrm{{Im}}\\zeta_{{{j}}}$")
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)

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

    return fig, plot_eigs


@partial(plotem, io=io, mode=PLOT_MODE, fname="pred_running")
def make_pred_plotter[Ntst: int](
    test_pars: DataPars[Ntst],
    test_data: Data,
    preds: Vtsts,
    num_plt_tst: Optional[int] = None,
    plt_step_tst: int = 1,
    i_fig: int = 1,
) -> tuple[Figure, F[int, None]]:
    """Make plotting function for prediction over different lead times."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, axs = plt.subplots(
        1,
        3,
        num=i_fig,
        figsize=tuple(mpf.figaspect(0.3)),
        constrained_layout=True,
        sharey=True,
        subplot_kw={"box_aspect": 1},
    )

    def plot_pred(i_step: int):
        i0_tst = test_pars.delay_embedding_end
        if num_plt_tst is not None:
            i1_tst = i0_tst + num_plt_tst
        else:
            i1_tst = i0_tst + test_pars.num_samples
        i0_pred = i0_tst + i_step
        i1_pred = i1_tst + i_step
        err = (
            preds[:num_plt_tst, i_step]
            - test_data["responses"][i0_pred:i1_pred]
        )
        amax = max(
            float(jnp.max(jnp.abs(test_data["responses"][i0_pred:i1_pred]))),
            float(jnp.max(jnp.abs(preds[:, i_step]))),
        )
        emax = float(jnp.max(jnp.abs(err)))
        for figax in fig.axes:
            figax.cla()

        ax = axs[0]
        sc_tst = ax.scatter(
            test_data["states"][i0_tst:i1_tst:plt_step_tst, 0] / jnp.pi,
            test_data["states"][i0_tst:i1_tst:plt_step_tst, 1] / jnp.pi,
            c=test_data["responses"][i0_pred:i1_pred:plt_step_tst],
            s=1,
            vmin=-amax,
            vmax=amax,
            cmap="seismic",
        )
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_xlabel(r"$\theta_2$")
        ax.set_title("True")

        ax = axs[1]
        ax.scatter(
            test_data["states"][i0_tst:i1_tst:plt_step_tst, 0] / jnp.pi,
            test_data["states"][i0_tst:i1_tst:plt_step_tst, 1] / jnp.pi,
            c=preds[:num_plt_tst:plt_step_tst, i_step],
            s=1,
            vmin=-amax,
            vmax=amax,
            cmap="seismic",
        )
        ax.set_xlabel(r"$\theta$")
        ax.set_title(f"Prediction; lead time = {i_step * test_pars.dt}")

        ax = axs[2]
        sc_err = ax.scatter(
            test_data["states"][i0_tst:i1_tst:plt_step_tst, 0] / jnp.pi,
            test_data["states"][i0_tst:i1_tst:plt_step_tst, 1] / jnp.pi,
            c=err[::plt_step_tst],
            s=1,
            vmin=-emax,
            vmax=emax,
            cmap="seismic",
        )
        ax.set_xlabel(r"$\theta$")
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


@partial(plotem, io=io, mode=PLOT_MODE, fname="pred_timeseries")
def make_pred_timeseries_plotter[Ntst: int](
    pars: TestPars[Ntst], test_data: Data, preds: Vtsts, i_fig: int = 1
) -> tuple[Figure, F[int, None]]:
    """Make plotting function over different initial conditions."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    ts = jnp.arange(pars.num_pred_steps + 1) * pars.data.dt

    def plot_pred(i_init: int):
        i0_tst = pars.data.delay_embedding_end + i_init
        i1_tst = i0_tst + pars.num_pred_steps + 1
        ax.cla()
        ax.plot(ts, test_data["responses"][i0_tst:i1_tst], "o-", label="test")
        ax.plot(ts, preds[i_init, :], "o-", label="prediction")
        ax.grid()
        ax.legend()
        ax.set_xlabel("Forecast timesteps")
        ax.set_title(f"Initial condition = {i_init}")

    return fig, plot_pred


@partial(plotit, io=io, mode=PLOT_MODE, fname="pred_scores")
def plot_forecast_skill_scores[Ntst: int](
    pars: TestPars[Ntst], scores: SkillScores, i_fig: int = 1
) -> Figure:
    """Plot NRMSE and ACC versus forecast lead time."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, axs = plt.subplots(
        2, 1, num=i_fig, constrained_layout=True, sharex=True
    )
    labels = ("NRMSE", "Anomaly correlation")
    ts = jnp.arange(pars.num_pred_steps + 1) * pars.data.dt
    for ax, score, label in zip(
        axs, (scores["nrmses"], scores["accs"]), labels
    ):
        ax.plot(ts, score, "o-")
        ax.grid()
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("Forecast time")
        ax.set_ylabel(label)
    return fig


def main():
    """Diffusion-regularized Koopman spectral analysis for torus rotation."""
    global io

    # Generate training and test data
    io @= str(pars.test.data)
    test_data = generate_data(pars.test.data, r_dtype)
    l2y_tst = make_l2_space(pars.test.data, r_dtype, test_data, jit=True)
    io @= str(pars.train.data)
    train_data = generate_data(pars.train.data, r_dtype)
    l2y = make_l2_space(pars.train.data, r_dtype, train_data)

    # Set kernel shape function
    shape_func = jnp.exp

    # Create and tune bandwidth function
    if pars.train.bw_tune is not None:
        io /= str(pars.train.bw_tune)
        if pars.train.data.velocity_covariate:
            bw_sqdist = compose2(dst.sqeuclidean, (fst, fst))
        else:
            bw_sqdist = dst.sqeuclidean
        bw_kernel_family = knl.make_kernel_family(
            l2y.scl, shape_func, bw_sqdist
        )
        bw_bandwidth, bw_tune_info = tune_kernel_bandwidth(
            pars.train.bw_tune, l2y, bw_kernel_family
        )
        bw_kernel = bw_kernel_family(bw_bandwidth)
        bandwidth_func = knl.make_bandwidth_function(
            l2y,
            bw_kernel,
            dim=jnp.asarray(bw_tune_info["dim"], r_dtype),
            vol=jnp.asarray(bw_tune_info["vol"], r_dtype),
        )
        print("Bandwidth function tuning:")
        print(f"Optimal bandwidth index: {bw_tune_info['i_opt']}")
        print(f"Optimal bandwidth: {bw_tune_info['opt_bandwidth']:.3e}")
        print(f"Optimal dimension: {bw_tune_info['opt_dim']:.3e}")
        print(
            "Bandwidth used for diffusion maps: "
            f"{bw_tune_info['bandwidth']:.3e}"
        )
        print(
            "Dimension based on diffusion maps bandwidth: "
            f"{bw_tune_info['dim']:.3e}"
        )
        print(f"Manifold volume: {bw_tune_info['vol']:.3e}")

        # Plot bandwidth function
        if PLOT_MODE is not None:
            plot_kernel_tuning(bw_tune_info, title="Bandwidth function tuning")
            plot_bandwidth_func(
                pars.train.data,
                l2y,
                bandwidth_func,
                train_data,
                pars.test.data,
                l2y_tst,
                test_data,
            )

    # Create and tune kernel
    io /= str(pars.train.bw_tune)
    if pars.train.cone is not None:
        sqdist = dst.make_sqcone(
            pars.train.cone.zeta, pars.train.cone.threshold
        )
    else:
        sqdist = dst.sqeuclidean
    if pars.train.bw_tune is not None:
        scaled_sqdist = knl.make_scaled_sqdist(l2y.scl, sqdist, bandwidth_func)
        kernel_family = knl.make_kernel_family(
            l2y.scl, shape_func, scaled_sqdist
        )
    else:
        kernel_family = knl.make_kernel_family(l2y.scl, shape_func, sqdist)
    bandwidth, tune_info = tune_kernel_bandwidth(
        pars.train.tune, l2y, kernel_family
    )
    kernel = kernel_family(bandwidth)
    print("Kernel tuning:")
    print(f"Bandwidth index: {tune_info['i_opt']}")
    print(f"Optimal bandwidth: {tune_info['opt_bandwidth']:.3e}")
    print(f"Optimal dimension: {tune_info['opt_dim']:.3e}")
    print(f"Bandwidth used for diffusion maps: {tune_info['bandwidth']:.3e}")
    print(
        f"Dimension based on diffusion maps bandwidth: {tune_info['dim']:.3e}"
    )
    print(f"Manifold volume: {tune_info['vol']:.3e}")

    # Plot kernel tuning function
    if PLOT_MODE is not None:
        plot_kernel_tuning(tune_info, title="Kernel tuning")

    # Solve kernel eigenvalue problem
    io /= str(pars.train.kernel)
    kernel_eigen = compute_kernel_eigen(
        pars.train.kernel, l2y, kernel, bandwidth
    )
    print(
        tabulate(
            jnp.vstack(
                (
                    kernel_eigen["evals"][:NUM_TABULATE],
                    knl.to_laplace_eigenvalues(
                        kernel_eigen["evals"][:NUM_TABULATE],
                        kernel_eigen["bandwidth"],
                    ),
                )
            ).T,
            headers=["Kernel eigenvalues", "Laplace eigenvalues"],
            floatfmt=".4f",
            showindex=True,
        )
    )

    # Plot spectrum of Laplace eigenvalues
    if PLOT_MODE is not None:
        plot_laplace_spectrum(kernel_eigen)

    # Build analysis, synthesis, and Nystrom operators for the kernel
    # eigenbasis
    if isinstance(pars.train.koopman.which_eigs_galerkin, int):
        which_eigs = pars.train.koopman.which_eigs_galerkin + 1
    else:
        which_eigs = pars.train.koopman.which_eigs_galerkin
    kernel_basis = knl.make_eigenbasis(
        pars.train.kernel,
        l2y,
        kernel,
        kernel_eigen,
        laplace_method=pars.train.koopman.laplace_method,
        which_eigs=which_eigs,
    )

    # Plot representative kernel eigenfunctions
    if PLOT_MODE is not None and KERNEL_EIGS_PLT is not None:
        _, plot_kernel_eig = make_kernel_evecs_plotter(
            pars.train.data,
            train_data,
            kernel_basis,
            pars.test.data,
            l2y_tst,
            test_data,
            delay_plot_mode=DELAY_PLOT_MODE,
        )
        if KERNEL_EIGS_PLT == "interactive":
            while True:
                i = input(
                    "Select kernel eigenfunction "
                    f"0-{pars.train.kernel.num_eigs - 1} to plot, "
                    "or press Enter to continue... "
                )
                if i == "":
                    break
                else:
                    try:
                        plot_kernel_eig(int(i))
                    except ValueError:
                        print("Invalid input.")
        else:
            for i in KERNEL_EIGS_PLT:
                plot_kernel_eig(i)

    # Compute regularized generator matrix
    io /= str(pars.train.koopman)
    gen_mat = compute_generator_matrix(
        pars.train, l2y, train_data, kernel_basis
    )

    # Plot generator matrix
    if PLOT_MODE is not None:
        plot_generator_matrix(gen_mat, title="Generator matrix")

    # Compute Koopan eigendecomposition
    koopman_eigen = compute_koopman_eigen(
        pars.train.koopman, gen_mat, kernel_basis
    )
    print(
        tabulate(
            jnp.vstack(
                (
                    koopman_eigen["evals"][:NUM_TABULATE].real,
                    koopman_eigen["engys"][:NUM_TABULATE],
                    koopman_eigen["efreqs"][:NUM_TABULATE],
                    koopman_eigen["eperiods"][:NUM_TABULATE],
                )
            ).T,
            headers=[
                "Growth rate",
                "Dirichlet energies",
                "Eigenfreqs.",
                "Eigenperiods",
            ],
            floatfmt=".4f",
            showindex=True,
        )
    )

    # Plot generator spectrum
    if PLOT_MODE is not None:
        plot_generator_spectrum(koopman_eigen)

    # Build analysis and synthesis operators for the Koopman eigenbasis
    c_k = VectorAlgebra(
        shape=(pars.train.koopman.dim_galerkin,), dtype=c_dtype
    )
    koopman_basis = koop.make_eigenbasis(
        c_k, kernel_basis, koopman_eigen, which_eigs=pars.train.pred.which_eigs
    )

    # Plot representative Koopman eigenfunctions
    if PLOT_MODE is not None and KOOPMAN_EIGS_PLT is not None:
        _, plot_koopman_eig = make_koopman_evecs_plotter(
            pars.train.data,
            train_data,
            koopman_basis,
            pars.test.data,
            l2y_tst,
            test_data,
        )
        if KOOPMAN_EIGS_PLT == "interactive":
            while True:
                i = input(
                    "Select Koopman eigenfunction "
                    f"0-{koopman_basis.dim - 1} to plot, "
                    "or press Enter to continue... "
                )
                if i == "":
                    break
                else:
                    try:
                        plot_koopman_eig(int(i))
                    except ValueError:
                        print("Invalid input.")
        else:
            for i in KOOPMAN_EIGS_PLT:
                plot_koopman_eig(i)

    # Perform time series prediction
    io /= str(pars.train.pred)
    io /= str(pars.test.data)
    predict = jit(
        make_timeseries_prediction_function(
            pars.train, train_data, koopman_basis
        )
    )
    fys_pred = timeit(l2y_tst.incl)(predict).real

    # Plot running forecast
    if PLOT_MODE is not None and LEAD_TIMES_PLT is not None:
        _, plot_pred = make_pred_plotter(pars.test.data, test_data, fys_pred)
        if LEAD_TIMES_PLT == "interactive":
            while True:
                i = input(
                    "Select lead time "
                    f"0-{pars.test.num_pred_steps} to plot, "
                    "or press Enter to continue... "
                )
                if i == "":
                    break
                else:
                    try:
                        plot_pred(int(i))
                    except ValueError:
                        print("Invalid input.")
        else:
            for i in LEAD_TIMES_PLT:
                plot_pred(i)

    # Plot time series forecast
    if PLOT_MODE is not None and INITIALIZATION_TIMES_PLT is not None:
        _, plot_pred_ts = make_pred_timeseries_plotter(
            pars.test, test_data, fys_pred
        )
        if INITIALIZATION_TIMES_PLT == "interactive":
            while True:
                i = input(
                    "Select initialization time "
                    f"0-{pars.test.data.num_samples} to plot, "
                    "or press Enter to continue... "
                )
                if i == "":
                    break
                else:
                    try:
                        plot_pred_ts(int(i))
                    except ValueError:
                        print("Invalid input.")
        else:
            for i in INITIALIZATION_TIMES_PLT:
                plot_pred_ts(i)

    # Compute normalized RMSE
    skill_scores = compute_skill_scores(pars.test, test_data, fys_pred)
    ts = pars.train.koopman.dt * jnp.arange(pars.test.num_pred_steps + 1)
    print(
        tabulate(
            jnp.vstack((ts, skill_scores["nrmses"], skill_scores["accs"])).T,
            headers=[
                "Lead time (month)",
                "Normalized RMSE",
                "Anomaly Correlation",
            ],
            floatfmt=".4f",
        )
    )

    # Plot forecast skill scores
    if PLOT_MODE is not None:
        plot_forecast_skill_scores(pars.test, skill_scores)


if __name__ == "__main__":
    if len(sys.argv) == 2 and (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
        print(__doc__)
    else:
        main()

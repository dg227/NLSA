"""Koopman spectral analysis of L63 system: Diffusion regularization."""

import jax
import jax.numpy as jnp
import nlsa.jax.distance as dst
import nlsa.jax.kernels as knl
import nlsa.jax.koopman as koop
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from functools import partial
from jax import Array
from nlsa.function_algebra import compose2
from nlsa.io_actions import IO, h5it, npyit, pickleit, plotit, plotem, timeit
from nlsa.jax.kernels import (
    BsKernelPars,
    ConePars,
    DmKernelPars,
    KernelEigen,
    KernelEigenShardings,
    KernelPars,
    TuneInfo,
    TunePars,
)
from nlsa.jax.koopman import (
    GeneratorShardings,
    KoopmanEigen,
    KoopmanEigenShardings,
    KoopmanParsDiff,
)
from nlsa.jax.scalars import ScalarField
from nlsa.jax.sharding import NamedSharder
from nlsa.jax.utils import fst
from nlsa.jax.vector_algebra import L2FnAlgebraShardings, L2VectorAlgebra
from nlsa_models import lorenz63 as l63
from nlsa_models.lorenz63 import Data, DataPars, SkillScores
from pathlib import Path
from tabulate import tabulate
from typing import Literal, Optional, TypedDict


class Experiment(StrEnum):
    """Experiments provided in this script."""

    A100_EIGH = auto()
    """Runs on 40GB A100 GPU using eigh direct kernel eigenvalue solver."""

    A100_EIGSH = auto()
    """Runs on 40GB A100 GPU using eigsh iterative kernel eigenvalue solver."""

    A100_EIGSH_2GPU = auto()
    """Multi-GPU case using eigsh iterative solver on 2 40GB A100s."""

    TEST = auto()
    """Test case."""


EXPERIMENT: Experiment = Experiment.TEST
IDX_GPU: Optional[int | Sequence[int]] = None  # 0
XLA_MEM_FRACTION: Optional[str] = "0.95"
JAX_CACHE_DIR: Optional[str] = "jax_cache"
FP: Literal["f32", "f64"] = "f32"
CONE_KERNEL: bool = False
KERNEL_TUNING_GRAD_METHOD: Literal["explicit", "automatic"] = "automatic"
KERNEL_NORMALIZATION: Literal["diffusion_maps", "bistochastic"] = (
    "diffusion_maps"
)
MATPLOTLIB_BACKEND: Optional[Literal["Agg"]] = None
OUTPUT_DATA_DIR = "examples/lorenz63/data"
NUM_TABULATE = 40
NUM_PLT_TST: Optional[int] = None
GENERATE_DATA_MODE: Literal["calc", "calcsave", "read"] = "calc"
TUNE_KERNEL_MODE: Literal["calc", "calcsave", "read"] = "calc"
KERNEL_EIGEN_MODE: Literal["calc", "calcsave", "read"] = "calc"
GENERATOR_MATRIX_MODE: Literal["calc", "calcsave", "read"] = "calc"
KOOPMAN_EIGEN_MODE: Literal["calc", "calcsave", "read"] = "calc"
KOOPMAN_RESPONSE_COEFFS_MODE: Literal["calc", "calcsave", "read"] = "calc"
KOOPMAN_PREDS_MODE: Literal["calc", "calcsave", "read"] = "calc"
SKILL_SCORES_MODE: Literal["calc", "calcsave", "read"] = "calc"
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

jax_env = l63.initialize_jax(
    idx_gpu=IDX_GPU,
    xla_mem_fraction=XLA_MEM_FRACTION,
    fp=FP,
    cache_dir=JAX_CACHE_DIR,
)
l63.initialize_matplotlib(backend=MATPLOTLIB_BACKEND)


@dataclass(frozen=True, slots=True)
class PredPars:
    """Dataclass containing prediction parameters."""

    dt: float
    """Prediction timestep."""

    num_steps: int
    """Number of timesteps for prediction."""

    which_eigs: int | tuple[int, int] | list[int]
    """Koopman eigenfunctions used in the prediction function."""

    def __str__(self) -> str:
        """Create string representation of prediction parameters."""
        match self.which_eigs:
            case int():
                eigs_str = "-".join(map(str, (0, self.which_eigs)))
            case tuple():
                eigs_str = "-".join(map(str, self.which_eigs))
            case list():
                eigs_str = "_".join(map(str, self.which_eigs))
        return "_".join(
            (
                "koop_pred",
                f"dt{self.dt:.2g}",
                f"nsteps{self.num_steps}",
                "neigs" + eigs_str,
            )
        )


@dataclass(frozen=True, slots=True)
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

    # TODO: Complete this
    def tabulate(self, show: bool = True) -> str:
        """Create tabulated summary of the properties of a TrainPars object."""
        tables = [self.data.tabulate(name="Training data", show=show)]
        return "".join(tables)


@dataclass(frozen=True, slots=True)
class TestPars[Ntst: int]:
    """Dataclass containing test parameter values."""

    num_pred_steps: int
    """Number of prediction steps."""

    data: DataPars[Ntst]
    """Test data parameters."""

    max_batch_size: Optional[int] = None
    """Max batch size for evaluation of prediction function."""

    # TODO: Complete this
    def tabulate(self, show: bool = True) -> str:
        """Create tabulated summary of the properties of a TestPars object."""
        tables = [self.data.tabulate(name="Test data", show=show)]
        return "".join(tables)


@dataclass(frozen=True, slots=True)
class Pars[N: int, Ntst: int]:
    """Dataclass containing the parameter values used in this example."""

    train: TrainPars[N]
    """Training parameters."""

    test: TestPars[Ntst]
    """Test parameters."""

    def tabulate(self, show: bool = True) -> str:
        """Create tabulated summary of the properties of a Pars object."""
        tables = [
            self.train.tabulate(show=show),
            self.test.tabulate(show=show),
        ]
        return "".join(tables)


class CommonPars(TypedDict):
    """Helper TypedDict to check common training/test parameter values."""

    covariate: Literal["x", "y", "z", "xy", "xyz"]
    response: Literal["x", "y", "z"]
    num_half_delays: int
    velocity_covariate: bool
    velocity_fd_order: Optional[Literal[2, 4, 6, 8]]


@dataclass(frozen=True, slots=True)
class TrainShardings:
    """Dataclass containing shardings related to the training data."""

    l2: L2FnAlgebraShardings = L2FnAlgebraShardings()
    """Shardings for the L2 space associated with the training data."""

    kernel_eigen: KernelEigenShardings = KernelEigenShardings()
    """Shardings for the kernel eigenvalue problem."""

    generator: GeneratorShardings = GeneratorShardings()
    """Sharding of the Koopman generator matrix."""

    koopman_eigen: KoopmanEigenShardings = KoopmanEigenShardings()
    """Shardings for the Koopman eigenvalue problem."""


@dataclass(frozen=True, slots=True)
class TestShardings:
    """Dataclass containing shardings related to the test data."""

    l2: L2FnAlgebraShardings = L2FnAlgebraShardings()
    """Shardings for the L2 space associated with the test data."""


@dataclass(frozen=True, slots=True)
class Shardings:
    """Dataclass containing shardings used in this example."""

    train: TrainShardings = TrainShardings()
    """Training shardings."""

    test: TestShardings = TestShardings()
    """Test shardings."""


def initialize(
    experiment: Experiment,
    cone_kernel: bool,
    kernel_normalization: Literal["diffusion_maps", "bistochastic"],
) -> tuple[Pars[int, int], Shardings]:
    """Prepare parameters and shardings for the numerical experiment."""
    match experiment:
        case Experiment.TEST:
            fd_order = 4
            cone_pars = ConePars(zeta=0.99) if cone_kernel else None
            common_pars: CommonPars = {
                "covariate": "xyz",
                "response": "x",
                "num_half_delays": 0,
                "velocity_covariate": True if cone_pars is not None else False,
                "velocity_fd_order": fd_order
                if cone_pars is not None
                else None,
            }
            num_pred_steps = 100
            train_data_pars = DataPars(
                **common_pars,
                x0=(1, 1, 1.1),
                dt=0.01,
                num_spinup=10_000,
                num_samples=4096,
                num_before=fd_order // 2,
                num_after=fd_order // 2,
            )
            test_data_pars = DataPars(
                **common_pars,
                x0=(1, 1, 0.9),
                dt=0.1,
                num_spinup=10_000,
                num_samples=2048,
                num_before=0,
                num_after=num_pred_steps,
            )
            bw_tune_pars = TunePars(
                manifold_dim=None,
                num_bandwidths=128,
                log10_bandwidth_lims=(-3, 3),
                bandwidth_scl=1,
            )
            if cone_pars is not None:
                tune_pars = TunePars(
                    manifold_dim=None,
                    num_bandwidths=128,
                    log10_bandwidth_lims=(-3, 3),
                    bandwidth_scl=1,
                )
            else:
                tune_pars = bw_tune_pars
            match kernel_normalization:
                case "diffusion_maps":
                    kernel_pars = DmKernelPars(
                        normalization="fokkerplanck",
                        eigensolver="eigh",
                        num_eigs=512,
                        batch_size=32,
                    )
                case "bistochastic":
                    kernel_pars = BsKernelPars(
                        eigensolver="svd",
                        num_eigs=512,
                        batch_size=32,
                    )
            koopman_pars = KoopmanParsDiff(
                fd_order=fd_order,
                dt=train_data_pars.dt,
                antisym=True,
                tau=0.005,
                laplacian_method="log",
                which_eigs_galerkin=256,
                num_eigs=129,
                sort_by="energy",
            )
            pred_pars = PredPars(
                dt=test_data_pars.dt, num_steps=num_pred_steps, which_eigs=129
            )
            shardings = Shardings()
        case Experiment.A100_EIGH:
            fd_order = 4
            cone_pars = ConePars(zeta=0.99) if cone_kernel else None
            common_pars: CommonPars = {
                "covariate": "xyz",
                "response": "x",
                "num_half_delays": 0,
                "velocity_covariate": True if cone_pars is not None else False,
                "velocity_fd_order": fd_order
                if cone_pars is not None
                else None,
            }
            num_pred_steps = 50
            train_data_pars = DataPars(
                **common_pars,
                x0=(1, 1, 1.1),
                dt=0.01,
                num_spinup=10_000,
                num_samples=16_384,
                num_before=fd_order // 2,
                num_after=fd_order // 2,
            )
            test_data_pars = DataPars(
                **common_pars,
                x0=(1, 1, 0.9),
                dt=0.01,
                num_spinup=10_000,
                num_samples=2048,
                num_before=0,
                num_after=num_pred_steps,
            )
            bw_tune_pars = TunePars(
                manifold_dim=None,
                num_bandwidths=128,
                log10_bandwidth_lims=(-3, 3),
                bandwidth_scl=1,
            )
            if cone_pars is not None:
                tune_pars = TunePars(
                    manifold_dim=None,
                    num_bandwidths=128,
                    log10_bandwidth_lims=(-3, 3),
                    bandwidth_scl=1,
                )
            else:
                tune_pars = bw_tune_pars
            match kernel_normalization:
                case "diffusion_maps":
                    kernel_pars = DmKernelPars(
                        normalization="fokkerplanck",
                        eigensolver="eigh",
                        num_eigs=2048,
                    )
                case "bistochastic":
                    kernel_pars = BsKernelPars(
                        eigensolver="svd",
                        num_eigs=2048,
                    )
            koopman_pars = KoopmanParsDiff(
                fd_order=fd_order,
                dt=train_data_pars.dt,
                antisym=True,
                tau=0.003,
                laplacian_method="log",
                which_eigs_galerkin=1024,
                num_eigs=1025,
                sort_by="energy",
            )
            pred_pars = PredPars(
                dt=test_data_pars.dt, num_steps=num_pred_steps, which_eigs=1025
            )
            shardings = Shardings()
        case Experiment.A100_EIGSH:
            fd_order = 4
            cone_pars = ConePars(zeta=0.99) if cone_kernel else None
            common_pars: CommonPars = {
                "covariate": "xyz",
                "response": "x",
                "num_half_delays": 0,
                "velocity_covariate": True if cone_pars is not None else False,
                "velocity_fd_order": fd_order
                if cone_pars is not None
                else None,
            }
            num_pred_steps = 500
            train_data_pars = DataPars(
                **common_pars,
                x0=(1, 1, 1.1),
                dt=0.01,
                num_spinup=10_000,
                num_samples=65_536,
                num_before=fd_order // 2,
                num_after=fd_order // 2,
            )
            test_data_pars = DataPars(
                **common_pars,
                x0=(1, 1, 0.9),
                dt=0.01,
                num_spinup=10_000,
                num_samples=2048,
                num_before=0,
                num_after=num_pred_steps,
            )
            bw_tune_pars = TunePars(
                manifold_dim=None,
                num_bandwidths=128,
                log10_bandwidth_lims=(-3, 3),
                bandwidth_scl=1,
            )
            if cone_pars is not None:
                tune_pars = TunePars(
                    manifold_dim=None,
                    num_bandwidths=128,
                    log10_bandwidth_lims=(-3, 3),
                    bandwidth_scl=1,
                )
            else:
                tune_pars = bw_tune_pars
            match kernel_normalization:
                case "diffusion_maps":
                    kernel_pars = DmKernelPars(
                        normalization="fokkerplanck",
                        eigensolver="eigsh",
                        num_eigs=2048,
                    )
                case "bistochastic":
                    kernel_pars = BsKernelPars(
                        eigensolver="svds",
                        num_eigs=2048,
                    )
            koopman_pars = KoopmanParsDiff(
                fd_order=fd_order,
                dt=train_data_pars.dt,
                antisym=True,
                tau=0.003,
                laplacian_method="log",
                which_eigs_galerkin=1024,
                num_eigs=1025,
                sort_by="energy",
                grad_batch_size=128,
                gram_batch_size=512,
            )
            pred_pars = PredPars(
                dt=test_data_pars.dt, num_steps=num_pred_steps, which_eigs=1025
            )
            shardings = Shardings()
        case Experiment.A100_EIGSH_2GPU:
            fd_order = 4
            cone_pars = ConePars(zeta=0.99) if cone_kernel else None
            common_pars: CommonPars = {
                "covariate": "xyz",
                "response": "x",
                "num_half_delays": 0,
                "velocity_covariate": True if cone_pars is not None else False,
                "velocity_fd_order": fd_order
                if cone_pars is not None
                else None,
            }
            num_pred_steps = 500
            train_data_pars = DataPars(
                **common_pars,
                x0=(1, 1, 1.1),
                dt=0.01,
                num_spinup=10_000,
                num_samples=131_072,
                num_before=fd_order // 2,
                num_after=fd_order // 2,
            )
            test_data_pars = DataPars(
                **common_pars,
                x0=(1, 1, 0.9),
                dt=0.01,
                num_spinup=10_000,
                num_samples=2048,
                num_before=0,
                num_after=num_pred_steps,
            )
            bw_tune_pars = TunePars(
                manifold_dim=None,
                num_bandwidths=128,
                log10_bandwidth_lims=(-3, 3),
                bandwidth_scl=1,
                batch_size=16,
            )
            if cone_pars is not None:
                tune_pars = TunePars(
                    manifold_dim=None,
                    num_bandwidths=128,
                    log10_bandwidth_lims=(-3, 3),
                    bandwidth_scl=1,
                )
            else:
                tune_pars = bw_tune_pars
            match kernel_normalization:
                case "diffusion_maps":
                    kernel_pars = DmKernelPars(
                        normalization="fokkerplanck",
                        eigensolver="eigsh",
                        num_eigs=2048,
                    )
                case "bistochastic":
                    kernel_pars = BsKernelPars(
                        eigensolver="svds",
                        num_eigs=2048,
                    )
            koopman_pars = KoopmanParsDiff(
                fd_order=fd_order,
                dt=train_data_pars.dt,
                antisym=True,
                tau=0.002,
                laplacian_method="log",
                which_eigs_galerkin=2000,
                num_eigs=513,
                sort_by="energy",
                eval_tx_batch_size=8192,
                gram_batch_size=1000,
            )
            pred_pars = PredPars(
                dt=test_data_pars.dt, num_steps=num_pred_steps, which_eigs=129
            )
            if len(jax_env.devices) > 1:
                sharder = NamedSharder(
                    devices=jax_env.devices,
                    shape=(len(jax_env.devices),),
                    axis_names=("x"),
                )
                i_sharding = sharder.sharding("x")
                j_sharding = sharder.sharding(None, "x")
                replicating = sharder.sharding(None)
                l2_shardings = L2FnAlgebraShardings(
                    data=replicating, vectors=i_sharding
                )
                l2_tst_shardings = L2FnAlgebraShardings(
                    data=replicating, vectors=replicating
                )
                l2_tx_shardings = L2FnAlgebraShardings(
                    data=replicating, vectors=i_sharding
                )
                kernel_eigen_shardings = KernelEigenShardings(
                    eigenvalues=replicating,
                    eigenvectors=j_sharding,
                    weights=i_sharding,
                )
                gen_shardings = GeneratorShardings(
                    tangents=l2_tx_shardings,
                    basis_grads=j_sharding,
                    matrix=i_sharding,
                )
                koopman_eigen_shardings = KoopmanEigenShardings(
                    eigenvalues=replicating, eigenvectors=replicating
                )
                train_shardings = TrainShardings(
                    l2=l2_shardings,
                    kernel_eigen=kernel_eigen_shardings,
                    generator=gen_shardings,
                    koopman_eigen=koopman_eigen_shardings,
                )
                test_shardings = TestShardings(l2=l2_tst_shardings)
            else:
                train_shardings = TrainShardings()
                test_shardings = TestShardings()
            shardings = Shardings(train=train_shardings, test=test_shardings)

    train_pars = TrainPars(
        data=train_data_pars,
        bw_tune=bw_tune_pars,
        cone=cone_pars,
        tune=tune_pars,
        kernel=kernel_pars,
        koopman=koopman_pars,
        pred=pred_pars,
    )
    test_pars = TestPars(data=test_data_pars, num_pred_steps=num_pred_steps)
    pars = Pars(train=train_pars, test=test_pars)
    return pars, shardings


pars, shardings = initialize(EXPERIMENT, CONE_KERNEL, KERNEL_NORMALIZATION)
io = IO(root=Path.cwd() / OUTPUT_DATA_DIR)

generate_data = timeit(
    pickleit(
        l63.generate_data,
        io=io,
        mode=GENERATE_DATA_MODE,
        fname="data",
        cls=Data,
    )
)
compute_kernel_bandwidth = timeit(
    pickleit(
        knl.tune_bandwidth,
        io=io,
        mode=TUNE_KERNEL_MODE,
        fname="tune_info",
        cls=TuneInfo[Array, Array, Array],
    )
)
compute_kernel_eigen = timeit(
    pickleit(
        knl.compute_eigen,
        io=io,
        mode=KERNEL_EIGEN_MODE,
        fname="kernel_eigen",
        cls=KernelEigen[Array, Array, Array, Array],
        callback=shardings.train.kernel_eigen.shard_kernel_eigen,
    )
)
compute_generator_matrix = timeit(
    npyit(
        koop.compute_generator_matrix,
        io=io,
        mode=GENERATOR_MATRIX_MODE,
        fname="gen_mat",
        cls=Array,
        callback=partial(
            jnp.asarray,
            dtype=jax_env.real_dtype,
            device=shardings.train.generator.matrix,
        ),
    )
)
compute_generator_eigen_diff = timeit(
    pickleit(
        koop.compute_generator_eigen_diff,
        io=io,
        mode=KOOPMAN_EIGEN_MODE,
        fname="generator_eigen_diff",
        cls=KoopmanEigen[Array, Array, Array],
    )
)
compute_koopman_response_coeffs = timeit(
    pickleit(
        l63.compute_koopman_response_coeffs,
        io=io,
        mode=KOOPMAN_RESPONSE_COEFFS_MODE,
        fname="koopman_response_coeffs",
        cls=Array,
    )
)
compute_koopman_preds = timeit(
    pickleit(
        koop.compute_koopman_preds,
        io=io,
        mode=KOOPMAN_PREDS_MODE,
        fname="koopman_preds",
        cls=Array,
    )
)
compute_skill_scores = timeit(
    h5it(
        l63.compute_skill_scores,
        io=io,
        mode=SKILL_SCORES_MODE,
        fname="pred_scores",
        cls=SkillScores,
        callback=l63.to_skill_scores,
    )
)
plot_kernel_tuning = plotit(
    knl.plot_kernel_tuning,
    io=io,
    mode=PLOT_MODE,
    fname="bandwidth_tuning_func",
)
plot_bandwidth_function = plotit(
    l63.plot_bandwidth_function,
    io=io,
    mode=PLOT_MODE,
    fname="bandwidth_func",
)
plot_laplace_spectrum = plotit(
    knl.plot_laplace_spectrum, io=io, mode=PLOT_MODE, fname="lapl_spec"
)
make_kernel_evecs_plotter = plotem(
    l63.make_kernel_evecs_plotter, io=io, mode=PLOT_MODE, fname="kernel_eigen"
)
plot_generator_matrix = plotit(
    koop.plot_operator_matrix, io=io, mode=PLOT_MODE, fname="gen_mat"
)
plot_generator_spectrum = plotit(
    koop.plot_generator_spectrum, io=io, mode=PLOT_MODE, fname="gen_spec"
)
make_koopman_evecs_plotter = plotem(
    l63.make_koopman_evecs_plotter,
    io=io,
    mode=PLOT_MODE,
    fname="koopman_eigen",
)
make_running_pred_plotter = plotem(
    l63.make_running_pred_plotter, io=io, mode=PLOT_MODE, fname="pred_running"
)
make_pred_timeseries_plotter = plotem(
    l63.make_pred_timeseries_plotter,
    io=io,
    mode=PLOT_MODE,
    fname="pred_timeseries",
)
plot_forecast_skill_scores = plotit(
    l63.plot_forecast_skill_scores, io=io, mode=PLOT_MODE, fname="pred_scores"
)


def main():
    """Perform Koopman analysis L63 using diffusion regularization."""
    global io

    # Display information about the computation to be performed
    jax_env.tabulate()
    pars.tabulate()

    # Generate training and test data
    io @= str(pars.test.data)
    test_data = generate_data(
        pars.test.data, dtype=jax_env.real_dtype, device=jax_env.device_cpu
    )
    io @= str(pars.train.data)
    train_data = generate_data(
        pars.train.data, dtype=jax_env.real_dtype, device=jax_env.device_cpu
    )

    # Make scalar field and L2 space builders
    scl_r = ScalarField(jax_env.real_dtype)
    impl_l2 = l63.make_data_driven_l2_space(
        pars=pars.train.data,
        dtype=jax_env.real_dtype,
        shardings=shardings.train.l2,
    )
    impl_l2_tst = l63.make_data_driven_l2_space(
        pars=pars.test.data,
        dtype=jax_env.real_dtype,
        shardings=shardings.test.l2,
    )

    # Set kernel shape function
    shape_func = jnp.exp
    match KERNEL_TUNING_GRAD_METHOD:
        case "explicit":
            neg_grad_shape_func = jnp.exp
        case "automatic":
            neg_grad_shape_func = None

    # Create and tune bandwidth function
    if pars.train.bw_tune is not None:
        io /= str(pars.train.bw_tune)
        if pars.train.data.velocity_covariate:
            bw_sqdist = compose2(dst.sqeuclidean, (fst, fst))
        else:
            bw_sqdist = dst.sqeuclidean
        bw_tune_info = compute_kernel_bandwidth(
            pars.train.bw_tune,
            impl_l2,
            shape_func,
            bw_sqdist,
            train_data,
            neg_grad_shape_func,
        )
        bandwidth_func = knl.make_data_driven_bandwidth_function(
            impl_l2, shape_func, bw_sqdist, bw_tune_info
        )
        bw_tune_info.tabulate(name="Bandwidth function tuning")

        # Plot bandwidth function
        if PLOT_MODE is not None:
            plot_kernel_tuning(bw_tune_info, title="Bandwidth function tuning")
            plot_bandwidth_function(
                pars.train.data,
                impl_l2,
                bandwidth_func,
                train_data,
                pars.test.data,
                impl_l2_tst,
                test_data,
                num_plt_tst=NUM_PLT_TST,
            )
    else:
        bandwidth_func = None

    # Create and tune kernel
    if pars.train.cone is not None:
        io /= str(pars.train.cone)
        sqdist = dst.make_sqcone(
            pars.train.cone.zeta, pars.train.cone.threshold
        )
    else:
        sqdist = dst.sqeuclidean
    if bandwidth_func is not None:
        sqdist = knl.make_data_driven_scaled_sqdist(
            scl_r, sqdist, bandwidth_func
        )
    else:
        sqdist = sqdist
    io /= str(pars.train.tune)
    tune_info = compute_kernel_bandwidth(
        pars.train.tune,
        impl_l2,
        shape_func,
        sqdist,
        train_data,
        neg_grad_shape_func,
    )
    tune_info.tabulate(name="Kernel tuning")

    # Plot kernel tuning function
    if PLOT_MODE is not None:
        plot_kernel_tuning(tune_info, title="Kernel tuning")

    # Solve kernel eigenvalue problem
    io /= str(pars.train.kernel)
    kernel = knl.make_data_driven_rbf_kernel(
        scl_r, shape_func, sqdist, tune_info.bandwidth
    )
    kernel_eigen = compute_kernel_eigen(
        pars.train.kernel,
        impl_l2,
        kernel,
        train_data,
        tune_info.bandwidth,
        pars.train.data.num_samples,
        jax_env.real_dtype,
        shardings=shardings.train.kernel_eigen,
    )
    if len(jax_env.devices) > 1:
        jax.debug.inspect_array_sharding(kernel_eigen.evecs, callback=print)
        jax.debug.inspect_array_sharding(
            kernel_eigen.dual_evecs, callback=print
        )
        jax.debug.inspect_array_sharding(kernel_eigen.evals, callback=print)
    kernel_eigen.tabulate(num_tabulate=NUM_TABULATE)

    # Plot spectrum of Laplace eigenvalues
    if PLOT_MODE is not None:
        plot_laplace_spectrum(kernel_eigen)

    # Plot representative kernel eigenfunctions
    if PLOT_MODE is not None and KERNEL_EIGS_PLT is not None:
        _, plot_kernel_eig = make_kernel_evecs_plotter(
            (pars.train.data, pars.train.kernel),
            impl_l2,
            train_data,
            kernel_eigen,
            pars.test.data,
            impl_l2_tst,
            test_data,
            kernel,
            delay_plot_mode=DELAY_PLOT_MODE,
            num_plt_tst=NUM_PLT_TST,
        )
        if KERNEL_EIGS_PLT == "interactive":
            while True:
                i = input(
                    "Select kernel eigenfunction "
                    f"0-{pars.train.kernel.num_eigs - 1} to plot, "
                    "or press Enter to continue. "
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
                if "show" in PLOT_MODE:
                    input("Press any key to continue...")

    # Compute generator matrix
    io /= str(pars.train.koopman)
    impl_eval_tx = l63.make_data_driven_tangent_evaluation_functional_fd(
        pars=pars.train.data,
        dtype=jax_env.real_dtype,
        fd_order=pars.train.koopman.fd_order,
        batch_size=pars.train.koopman.eval_tx_batch_size,
        shardings=shardings.train.l2,
    )
    gen_mat = compute_generator_matrix(
        (pars.train.kernel, pars.train.koopman),
        impl_l2,
        impl_eval_tx,
        kernel,
        train_data,
        kernel_eigen,
        shardings=shardings.train.generator,
    )

    if len(jax_env.devices) > 1:
        jax.debug.inspect_array_sharding(gen_mat, callback=print)

    # Plot generator matrix
    if PLOT_MODE is not None:
        plot_generator_matrix(gen_mat, title="Generator matrix")

    # Compute Koopman eigendecomposition
    c_k = L2VectorAlgebra(
        shape=(pars.train.koopman.dim_galerkin + 1,),
        dtype=jax_env.complex_dtype,
    )
    koopman_eigen = compute_generator_eigen_diff(
        (pars.train.kernel, pars.train.koopman),
        impl_l2,
        kernel,
        train_data,
        kernel_eigen,
        gen_mat,
        out_shardings=shardings.train.koopman_eigen,
    )
    print(
        tabulate(
            jnp.vstack(
                (
                    koopman_eigen.evals[:NUM_TABULATE].real,
                    koopman_eigen.engys[:NUM_TABULATE],
                    koopman_eigen.efreqs[:NUM_TABULATE],
                    koopman_eigen.eperiods[:NUM_TABULATE],
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

    # Plot representative Koopman eigenfunctions
    if PLOT_MODE is not None and KOOPMAN_EIGS_PLT is not None:
        _, plot_koopman_eig = make_koopman_evecs_plotter(
            (pars.train.data, pars.train.kernel, pars.train.koopman),
            c_k,
            impl_l2,
            train_data,
            kernel_eigen,
            koopman_eigen,
            pars.test.data,
            impl_l2_tst,
            test_data,
            kernel,
            delay_plot_mode=DELAY_PLOT_MODE,
            num_plt_tst=NUM_PLT_TST,
        )
        if KOOPMAN_EIGS_PLT == "interactive":
            while True:
                i = input(
                    "Select Koopman eigenfunction "
                    f"0-{koopman_eigen.num_eigs - 1} to plot, "
                    "or press Enter to continue. "
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

    # Compute expansion coefficients of the response function
    io /= str(pars.train.pred)
    io /= str(pars.train.data.response)
    coeffs = compute_koopman_response_coeffs(
        (pars.train.data, pars.train.kernel, pars.train.koopman),
        c_k,
        impl_l2,
        train_data,
        kernel,
        kernel_eigen,
        koopman_eigen,
        pars.train.pred.which_eigs,
    )

    # Perform time series prediction
    io /= str(pars.test.data)
    preds = compute_koopman_preds(
        (pars.train.kernel, pars.train.koopman),
        c_k,
        impl_l2,
        train_data,
        kernel,
        kernel_eigen,
        koopman_eigen,
        coeffs,
        impl_l2_tst,
        test_data,
        num_steps=pars.train.pred.num_steps,
        dt=pars.train.pred.dt,
        which_eigs=pars.train.pred.which_eigs,
    ).real

    # Plot running forecast
    if PLOT_MODE is not None and LEAD_TIMES_PLT is not None:
        _, plot_pred = make_running_pred_plotter(
            pars.test.data, test_data, preds
        )
        if LEAD_TIMES_PLT == "interactive":
            while True:
                i = input(
                    "Select lead time "
                    f"0-{pars.test.num_pred_steps} to plot, "
                    "or press Enter to continue. "
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
            pars.test.data, test_data, preds
        )
        if INITIALIZATION_TIMES_PLT == "interactive":
            while True:
                i = input(
                    "Select initialization time "
                    f"0-{pars.test.data.num_samples} to plot, "
                    "or press Enter to continue. "
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

    # Compute forecast skill scores
    skill_scores = compute_skill_scores(pars.test.data, test_data, preds)
    ts = jnp.arange(pars.test.num_pred_steps + 1) * pars.test.data.dt
    print(
        tabulate(
            jnp.vstack((ts, skill_scores["nrmses"], skill_scores["accs"])).T,
            headers=["Lead time", "Normalized RMSE", "Anomaly Correlation"],
            floatfmt=".4f",
        )
    )

    # Plot forecast skill scores
    if PLOT_MODE is not None:
        plot_forecast_skill_scores(pars.test.data, skill_scores)


if __name__ == "__main__":
    if len(sys.argv) == 2 and (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
        print(__doc__)
    else:
        main()

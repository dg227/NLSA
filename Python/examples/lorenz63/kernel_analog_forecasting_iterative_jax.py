"""Kernel analog forecasting of the Lorenz 63 system -- iterative approach."""

import jax
import jax.numpy as jnp
import nlsa.jax.delays as dl
import nlsa.jax.distance as dst
import nlsa.jax.dynamics as dyn
import nlsa.jax.kernels as knl
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from functools import partial
from jax import Array, jit
from nlsa.function_algebra import compose, compose2
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
from nlsa.jax.sharding import NamedSharder
from nlsa.jax.stats import MultivariateTimeseriesStats
from nlsa.jax.utils import fst
from nlsa.jax.vector_algebra import L2FnAlgebraShardings
from nlsa.utils import get_closest_factors
from nlsa.io_actions import IO, h5it, pickleit, plotit, plotem, timeit
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

    TEST = auto()
    """Test case."""


EXPERIMENT: Experiment = Experiment.TEST
IDX_GPU: Optional[int] = 0
XLA_MEM_FRACTION: Optional[str] = "0.95"
JAX_CACHE_DIR: Optional[str] = "jax_cache"
CONE_KERNEL: bool = False
FP: Literal["f32", "f64"] = "f32"
KERNEL_NORMALIZATION: Literal["diffusion_maps", "bistochastic"] = (
    "diffusion_maps"
)
MATPLOTLIB_BACKEND: Optional[Literal["Agg"]] = None
OUTPUT_DATA_DIR = "examples/l63/data"
NUM_TABULATE = 40
NUM_PLT_TST: Optional[int] = None
DELAY_EMBEDDING_MODE: Optional[Literal["explicit", "on_the_fly"]] = (
    "on_the_fly"
)
GENERATE_DATA_MODE: Literal["calc", "calcsave", "read"] = "calc"
TUNE_KERNEL_MODE: Literal["calc", "calcsave", "read"] = "calc"
KERNEL_EIGEN_MODE: Literal["calc", "calcsave", "read"] = "calc"
SKILL_SCORES_MODE: Literal["calc", "calcsave", "read"] = "calc"
TRAJECTORY_STATS_MODE: Literal["calc", "calcsave", "read"] = "calc"
PLOT_MODE: Optional[Literal["save", "show", "saveshow"]] = "show"
DELAY_PLOT_MODE: Literal["backward", "central"] = "backward"
KERNEL_EIGS_PLT: Optional[Sequence[int] | Literal["interactive"]] = (
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
                "iterative_kaf_pred",
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

    data: DataPars[Ntst]
    """Test data parameters."""

    num_pred_steps: int
    """Number of prediction steps."""

    num_stat_steps: int
    """Number of samples for long-term trajectory reconstruction."""

    stat_ic: int
    """Initial condition in test dataset for statistics reconstruction."""

    max_batch_size: Optional[int] = None
    """Max batch size for evaluation of prediction function."""

    def __str__(self) -> str:
        """Create string representation of test parameters."""
        return "_".join(
            (
                str(self.data),
                f"nsteps{self.num_pred_steps}",
                f"nstatsteps{self.num_stat_steps}",
                f"ic{self.stat_ic}",
            )
        )

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
    dt: float
    num_spinup: int
    num_half_delays: int
    velocity_covariate: bool
    velocity_fd_order: Optional[Literal[2, 4, 6, 8]]
    num_before: int
    num_after: int


@dataclass(frozen=True, slots=True)
class TrainShardings:
    """Dataclass containing shardings related to the training data."""

    l2: L2FnAlgebraShardings = L2FnAlgebraShardings()
    """Shardings for the L2 space associated with the training data."""

    kernel_eigen: KernelEigenShardings = KernelEigenShardings()
    """Shardings for the kernel eigenvalue problem."""


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


def from_experiment(
    experiment: Experiment,
) -> tuple[Pars[int, int], Shardings]:
    """Prepare parameters and shardings for the numerical experiment."""
    match experiment:
        case Experiment.TEST:
            cone_pars = ConePars(zeta=0.99) if CONE_KERNEL else None
            num_pred_steps = 50
            common_pars: CommonPars = {
                "covariate": "xyz",
                "response": "x",
                "dt": 0.1,
                "num_spinup": 1000,
                "num_half_delays": 0,
                "velocity_covariate": True if cone_pars is not None else False,
                "velocity_fd_order": 4 if cone_pars is not None else None,
                "num_before": 0,
                "num_after": num_pred_steps,
            }
            train_data_pars = DataPars(
                **common_pars, x0=(1, 1, 1.1), num_samples=4096
            )
            test_data_pars = DataPars(
                **common_pars,
                x0=(1, 1, 0.9),
                num_samples=2048,
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
            match KERNEL_NORMALIZATION:
                case "diffusion_maps":
                    kernel_pars = DmKernelPars(
                        normalization="fokkerplanck",
                        eigensolver="eigh",
                        num_eigs=512,
                    )
                case "bistochastic":
                    kernel_pars = BsKernelPars(
                        eigensolver="svds",
                        num_eigs=512,
                    )
            pred_pars = PredPars(num_steps=num_pred_steps, which_eigs=256)
            shardings = Shardings()
        case Experiment.A100_EIGH:
            cone_pars = ConePars(zeta=0.99) if CONE_KERNEL else None
            num_pred_steps = 50
            common_pars: CommonPars = {
                "covariate": "xyz",
                "response": "x",
                "dt": 0.1,
                "num_spinup": 1000,
                "num_half_delays": 0,
                "velocity_covariate": True if cone_pars is not None else False,
                "velocity_fd_order": 4 if cone_pars is not None else None,
                "num_before": 0,
                "num_after": num_pred_steps,
            }
            train_data_pars = DataPars(
                **common_pars,
                x0=(1, 1, 1.1),
                num_samples=16_384,
            )
            test_data_pars = DataPars(
                **common_pars,
                x0=(1, 1, 0.9),
                num_samples=8192,
                eval_batch_size=None,
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
            match KERNEL_NORMALIZATION:
                case "diffusion_maps":
                    kernel_pars = DmKernelPars(
                        normalization="fokkerplanck",
                        eigensolver="eigh",
                        num_eigs=1024,
                    )
                case "bistochastic":
                    kernel_pars = BsKernelPars(
                        eigensolver="svd",
                        num_eigs=1024,
                    )
            pred_pars = PredPars(num_steps=num_pred_steps, which_eigs=512)
            shardings = Shardings()
        case Experiment.A100_EIGSH:
            cone_pars = ConePars(zeta=0.99) if CONE_KERNEL else None
            num_pred_steps = 50
            common_pars: CommonPars = {
                "covariate": "xyz",
                "response": "x",
                "dt": 0.1,
                "num_spinup": 1000,
                "num_half_delays": 0,
                "velocity_covariate": True if cone_pars is not None else False,
                "velocity_fd_order": 4 if cone_pars is not None else None,
                "num_before": 0,
                "num_after": num_pred_steps,
            }
            train_data_pars = DataPars(
                **common_pars,
                x0=(1, 1, 1.1),
                num_samples=65_536,
            )
            test_data_pars = DataPars(
                **common_pars,
                x0=(1, 1, 0.9),
                num_samples=2048,
                eval_batch_size=None,
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
            match KERNEL_NORMALIZATION:
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
            pred_pars = PredPars(num_steps=num_pred_steps, which_eigs=1024)
            if len(jax_env.devices) > 1:
                sharder_1d = NamedSharder(
                    devices=jax_env.devices,
                    shape=(len(jax_env.devices),),
                    axis_names=("x"),
                )
                sharder_2d = NamedSharder(
                    devices=jax_env.devices,
                    shape=get_closest_factors(len(jax_env.devices)),
                    axis_names=("x", "y"),
                )
                x_sharding = sharder_1d.sharding("x")
                replicating = sharder_1d.sharding(None)
                xy_sharding = sharder_2d.sharding("x", "y")
                l2_shardings = L2FnAlgebraShardings(
                    data=replicating, vectors=x_sharding
                )
                l2_tst_shardings = L2FnAlgebraShardings(
                    data=replicating, vectors=replicating
                )
                kernel_eigen_shardings = KernelEigenShardings(
                    eigenvalues=replicating,
                    eigenvectors=x_sharding,
                    matrix=xy_sharding,
                )
                train_shardings = TrainShardings(
                    l2=l2_shardings,
                    kernel_eigen=kernel_eigen_shardings,
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
        pred=pred_pars,
    )
    test_pars = TestPars(
        data=test_data_pars,
        num_pred_steps=num_pred_steps,
        num_stat_steps=test_data_pars.num_samples,
        stat_ic=0,
    )
    pars = Pars(train=train_pars, test=test_pars)
    return pars, shardings


pars, shardings = from_experiment(EXPERIMENT)
io = IO(root=Path.cwd() / OUTPUT_DATA_DIR)

generate_data = timeit(
    h5it(
        l63.generate_data,
        io=io,
        mode=GENERATE_DATA_MODE,
        fname="data",
        cls=Data,
        callback=partial(l63.to_data, dtype=jax_env.real_dtype),
    )
)
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
        callback=partial(
            knl.to_kernel_eigen,
            dtype=jax_env.real_dtype,
            shardings=shardings.train.kernel_eigen,
        ),
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
compute_trajectory_stats = pickleit(
    l63.compute_trajectory_stats,
    io=io,
    mode=TRAJECTORY_STATS_MODE,
    fname="trajectory_stats",
    cls=tuple[MultivariateTimeseriesStats, MultivariateTimeseriesStats],
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
plot_reconstructed_trajectory = plotit(
    l63.plot_reconstructed_trajectory,
    io=io,
    mode=PLOT_MODE,
    fname="trajectory",
)
plot_trajectory_stats = plotit(
    l63.plot_trajectory_stats, io=io, mode=PLOT_MODE, fname="trajectory_stats"
)


def main():
    """Perform iterative kernel analog forecasting of the Lorenz 63 system."""
    global io

    # Display information about the computation to be performed.
    jax_env.tabulate()
    pars.tabulate()

    # Generate training and test data
    io @= str(pars.test.data)
    test_data = generate_data(
        pars.test.data, dtype=jax_env.real_dtype, device=jax_env.device_cpu
    )
    l2y_tst = l63.make_l2_space(
        pars.test.data,
        jax_env.real_dtype,
        test_data,
        shardings=shardings.test.l2,
        jit=True,
    )
    io @= str(pars.train.data)
    train_data = generate_data(
        pars.train.data, dtype=jax_env.real_dtype, device=jax_env.device_cpu
    )
    l2y = l63.make_l2_space(
        pars.train.data,
        jax_env.real_dtype,
        train_data,
        shardings=shardings.train.l2,
        jit=True,
    )

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
        bandwidth_normalization = partial(
            knl.bandwidth_normalization, l2y, bw_kernel
        )
        bandwidth_func = knl.make_bandwidth_function(
            l2y,
            bw_kernel,
            dim=jnp.asarray(bw_tune_info["dim"], jax_env.real_dtype),
            vol=jnp.asarray(bw_tune_info["vol"], jax_env.real_dtype),
            normalization=jit(bandwidth_normalization)(),
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
            plot_bandwidth_function(
                pars.train.data,
                l2y,
                bandwidth_func,
                train_data,
                pars.test.data,
                l2y_tst,
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
    io /= str(pars.train.tune)
    if bandwidth_func is not None:
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
        pars.train.kernel,
        l2y,
        kernel,
        bandwidth,
        shardings=shardings.train.kernel_eigen,
    )
    if len(jax_env.devices) > 1:
        jax.debug.inspect_array_sharding(kernel_eigen["evecs"], callback=print)
        jax.debug.inspect_array_sharding(
            kernel_eigen["dual_evecs"], callback=print
        )
        jax.debug.inspect_array_sharding(kernel_eigen["evals"], callback=print)
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
    if isinstance(pars.train.pred.which_eigs, int):
        which_eigs = pars.train.pred.which_eigs
    else:
        which_eigs = pars.train.pred.which_eigs
    kernel_basis = knl.make_eigenbasis(
        pars.train.kernel,
        l2y,
        kernel,
        kernel_eigen,
        laplace_method="log",
        which_eigs=which_eigs,
    )
    nyst = compose(kernel_basis.fn_synth, kernel_basis.anal)

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

    # Perform time series prediction
    io /= str(pars.train.pred)
    io /= str(pars.test)
    if pars.train.data.num_half_delays == 0:
        predict = l63.make_iterative_kaf_prediction_function(
            pars.train.data, train_data, nyst
        )
    else:
        predict = l63.make_iterative_kaf_prediction_function_with_delays(
            pars.train.data, train_data, nyst
        )
    predict_ts = dyn.make_fin_orbit(predict, pars.test.num_pred_steps + 1)
    ys_pred = timeit(l2y_tst.incl)(predict_ts)
    if pars.train.data.num_half_delays > 0:
        ys_pred = ys_pred.reshape(
            (
                pars.test.data.num_samples,
                pars.test.num_pred_steps + 1,
                pars.test.data.num_delays + 1,
                -1,
            )
        )[:, :, -1, :]

    # Plot running forecast
    if PLOT_MODE is not None and LEAD_TIMES_PLT is not None:
        _, plot_pred = make_running_pred_plotter(
            pars.test.data, test_data, ys_pred, what="covariates"
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
                if "show" in PLOT_MODE:
                    input("Press any key to continue...")

    # Plot time series forecast
    if PLOT_MODE is not None and INITIALIZATION_TIMES_PLT is not None:
        _, plot_pred_ts = make_pred_timeseries_plotter(
            pars.test.data, test_data, ys_pred, what="covariates"
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
                if "show" in PLOT_MODE:
                    input("Press any key to continue...")

    # Compute forecast skill scores
    skill_scores = compute_skill_scores(
        pars.test.data, test_data, ys_pred, what="covariates", dropna=True
    )
    ts = jnp.arange(pars.test.num_pred_steps + 1) * pars.test.data.dt
    print(
        tabulate(
            jnp.vstack(
                (ts, skill_scores["nrmses"][0], skill_scores["accs"][0])
            ).T,
            headers=[
                "Lead time",
                "Normalized RMSE (x)",
                "Anomaly Correlation (x)",
            ],
            floatfmt=".4f",
        )
    )

    # Plot forecast skill scores
    if PLOT_MODE is not None:
        plot_forecast_skill_scores(
            pars.test.data, skill_scores, what="covariates"
        )

    # Compute long-time statistical trajectory
    predict_traj = jit(
        dyn.make_fin_orbit(
            predict, pars.test.num_stat_steps + pars.test.num_pred_steps
        )
    )
    if pars.test.data.num_delays == 0:
        ys_traj = predict_traj(test_data["covariates"][pars.test.stat_ic])
        assert isinstance(ys_traj, Array)
    else:
        i0 = pars.test.stat_ic
        i1 = i0 + pars.test.data.num_delays + 1
        ys_traj = dl.delay_eval_at(
            test_data["covariates"][i0:i1],
            num_delays=pars.test.data.num_delays,
            jit=True,
        )(predict_traj)
        ys_traj = ys_traj.reshape(-1, 3, 3)[:, -1, :]

    if jnp.isnan(ys_traj).any():
        warnings.warn("NaN values occurred in trajectory.", RuntimeWarning)

    # Plot trajectory
    if PLOT_MODE is not None:
        plot_reconstructed_trajectory(pars.train.data, train_data, ys_traj)

    # Compute PDFs and lagged autocorrelations of covariate variables
    train_stats, traj_stats = compute_trajectory_stats(
        pars.train.data,
        train_data,
        ys_traj,
        num_pred_steps=pars.test.num_pred_steps,
        num_stat_steps=pars.test.num_stat_steps,
        dropna=True,
    )

    # Plot PDFs and lagged autocorrelations
    if PLOT_MODE is not None:
        plot_trajectory_stats(
            pars.test.data,
            train_stats,
            traj_stats,
            num_pred_steps=pars.test.num_pred_steps,
        )


if __name__ == "__main__":
    if len(sys.argv) == 2 and (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
        print(__doc__)
    else:
        main()

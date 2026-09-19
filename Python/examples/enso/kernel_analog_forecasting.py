"""Kernel analog forecasting of the El Nino Southern Oscillation.

The training and test data used for experiment ENSO_FROM_ERA5_NINO34SST
are included under /examples/enso/data.
"""

import jax.numpy as jnp
import nc_time_axis as nc_time_axis
import nlsa.jax.distance as dst
import nlsa.jax.kernels as knl
import nlsa.jax.scalars as scls
import nlsa_models.climate as clim
import nlsa_models.era5 as era5
import numpy as np
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from jax import Array
from nlsa.function_algebra import compose2
from nlsa.io_actions import (
    IO,
    h5it,
    netcdfit,
    pickleit,
    plotit,
    plotem,
    timeit,
)
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
from nlsa.jax.vector_algebra import L2FnAlgebraShardings
from nlsa.jax.utils import fst
from nlsa_models.climate import (
    Covariate,
    Climatology,
    DataPars,
    NPData,
    ProjectionPars,
    Response,
    SkillScores,
    Time,
    TimeSampling,
)
from pathlib import Path
from tabulate import tabulate
from typing import Literal, TypedDict


type Plots = Literal[
    "all",
    "bandwidth_tuning",
    "bandwidth_func",
    "kernel_tuning",
    "laplacian_spec",
    "kernel_eigen",
    "kernel_modes",
    "expansion_coeffs",
    "running_pred",
    "pred_timeseries",
    "skill_scores",
]


class Experiment(StrEnum):
    """Experiments provided in this script."""

    NINO34_FROM_ERA5_NINO34SST = auto()
    """Prediction of the Nino 3.4 index from ERA5 Nino 3.4 SST."""


EXPERIMENT: Experiment = Experiment.NINO34_FROM_ERA5_NINO34SST
IDX_GPU: int | Sequence[int] | None = 0  # 0
XLA_MEM_FRACTION: str | None = "0.95"
JAX_CACHE_DIR: str | None = "jax_cache"
FP: Literal["f32", "f64"] = "f32"
CONE_KERNEL: bool = True
KERNEL_TUNING_GRAD_METHOD: Literal["explicit", "automatic"] = "automatic"
KERNEL_NORMALIZATION: Literal["diffusion_maps", "bistochastic"] = (
    "diffusion_maps"
)
MATPLOTLIB_BACKEND: Literal["Agg"] | None = None
ERA5_DAILY_DATA_DIR = "/storage/data/era5/daily"
ERA5_MONTHLY_DATA_DIR = "/storage/data/era5/month_nc_1940_2025"
OUTPUT_DATA_DIR = "examples/enso/data"
NUM_TABULATE = 40
EXTRACT_DATA_MODE: Literal["calc", "calcsave", "read"] = "calcsave"
TUNE_KERNEL_MODE: Literal["calc", "calcsave", "read"] = "calc"
KERNEL_EIGEN_MODE: Literal["calc", "calcsave", "read"] = "calc"
KERNEL_PROJECTION_MODE: Literal["calc", "calcsave", "read"] = "calc"
KAF_EXPANSION_COEFFS_MODE: Literal["calc", "calcsave", "read"] = "calc"
KAF_PREDS_MODE: Literal["calc", "calcsave", "read"] = "calc"
SKILL_SCORES_MODE: Literal["calc", "calcsave", "read"] = "calc"
PLOT_MODE: Literal["save", "show", "saveshow"] | None = "show"
WHICH_PLOTS: set[Plots] = {"kernel_modes"}
DELAY_PLOT_MODE: Literal["backward", "central"] = "central"
PLT_DATE_RANGE: tuple[str, str] | None = None
KERNEL_EIGS_PLT: Sequence[int] | Literal["interactive"] | None = "interactive"
KERNEL_MODES_PLT: Sequence[int] | Literal["interactive"] | None = "interactive"
LEAD_TIMES_PLT: Sequence[int] | Literal["interactive"] | None = "interactive"
INITIALIZATION_TIMES_PLT: Sequence[int] | Literal["interactive"] | None = (
    "interactive"
)


jax_env = clim.initialize_jax(
    idx_gpu=IDX_GPU,
    xla_mem_fraction=XLA_MEM_FRACTION,
    fp=FP,
)
clim.initialize_matplotlib(backend=MATPLOTLIB_BACKEND)


@dataclass(frozen=True, slots=True)
class PredPars:
    """Dataclass containing prediction parameters."""

    dt: int
    """Prediction timestep."""
    # WARNING: The current KAF implementation ignores this parameter and
    # assumes dt=1.

    num_steps: int
    """Number of timesteps for prediction."""

    # TODO: Replace tuple[int, int] by slice[int, int, int].
    which_eigs: int | tuple[int, int] | list[int]
    """Kernel eigenfunctions eigenfunctions used in the prediction function."""

    def __str__(self) -> str:
        """Create string representing prediction parameters."""
        match self.which_eigs:
            case int():
                eigs_str = "-".join(map(str, (0, self.which_eigs)))
            case (_, _):
                eigs_str = "-".join(map(str, self.which_eigs))
            case list():
                eigs_str = "_".join(map(str, self.which_eigs))
        return "_".join((f"pred{self.num_steps}", "eigs" + eigs_str))


@dataclass(frozen=True, slots=True)
class TrainPars[T: TimeSampling]:
    """Dataclass containing the training parameter values."""

    data: DataPars[T]
    """Training data parameters."""

    tune: TunePars
    """Kernel tuning parameters."""

    kernel: KernelPars
    """Kernel eigendecomposition parameters."""

    pred: PredPars
    """Prediction parameters."""

    cone: ConePars | None = None
    """Cone kernel parameters."""

    bw_tune: TunePars | None = None
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
class TestPars[T: TimeSampling]:
    """Dataclass containing test parameter values."""

    num_pred_steps: int
    """Number of prediction steps."""

    data: DataPars[T]
    """Test data parameters."""

    # TODO: Complete this
    def tabulate(self, show: bool = True) -> str:
        """Create tabulated summary of the properties of a TestPars object."""
        tables = (
            self.data.tabulate(name="Test data", show=show),
            tabulate([{"Number of prediction steps": self.num_pred_steps}]),
        )

        return "".join(tables)


@dataclass(frozen=True, slots=True)
class Pars[T: TimeSampling]:
    """Dataclass containing the parameter values used in this example."""

    train: TrainPars[T]
    """Training parameters."""

    test: TestPars[T]
    """Test parameters."""

    kernel_projection: ProjectionPars[T] | None = None
    """Projection parameters."""

    def tabulate(self, show: bool = True) -> str:
        """Create tabulated summary of the properties of a Pars object."""
        if self.kernel_projection is not None:
            kernel_proj_tab = self.kernel_projection.tabulate(show=show)
        else:
            kernel_proj_tab = None
        tables = (
            self.train.tabulate(show=show),
            self.test.tabulate(show=show),
            kernel_proj_tab,
        )
        return "".join(filter(None, tables))


class CommonPars(TypedDict):
    """Helper TypedDict to check common training/test parameter values."""

    num_half_delays: int
    velocity_covariate: bool
    velocity_fd_order: Literal[2, 4, 6, 8] | None
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
    cone_kernel: bool,
    kernel_normalization: Literal["diffusion_maps", "bistochastic"],
) -> tuple[Pars[Literal["monthly"]], Shardings]:
    """Prepare parameters and shardings for the numerical experiment."""
    match experiment:
        case Experiment.NINO34_FROM_ERA5_NINO34SST:
            cone_pars = ConePars(zeta=0.99) if cone_kernel else None
            era5_io = era5.IO(
                input_path=ERA5_MONTHLY_DATA_DIR, file_format="nc"
            )
            time_sampling = "monthly"
            fd_order = 4
            num_pred_steps = 24
            train_date_range = ("1940-01-01", "2019-12-31")
            test_date_range = ("2018-01-01", "2025-12-31")
            common_pars: CommonPars = {
                "num_half_delays": 6,
                "velocity_covariate": True if cone_pars is not None else False,
                "velocity_fd_order": fd_order
                if cone_pars is not None
                else None,
                "num_before": 0,
                "num_after": num_pred_steps,
            }
            climatology_date_range = train_date_range
            covariate_rolling_window = None
            covariate_rolling_mode = "center"
            train_covariate_time = Time(
                date_range=train_date_range,
                sampling=time_sampling,
                custom_climatology_date_range=climatology_date_range,
                rolling_window=covariate_rolling_window,
                rolling_mode=covariate_rolling_mode,
            )
            test_covariate_time = Time(
                date_range=test_date_range,
                sampling=time_sampling,
                custom_climatology_date_range=climatology_date_range,
                rolling_window=covariate_rolling_window,
                rolling_mode=covariate_rolling_mode,
            )
            covariate_climatology = Climatology(remove=False, standardize=True)
            covariate_vars = [era5.Var.SST]
            covariate_domain = era5.nino34_domain(step_lon=4, step_lat=4)
            projection_rolling_window = None
            projection_rolling_mode = "center"
            projection_time = Time(
                date_range=train_date_range,
                sampling=time_sampling,
                custom_climatology_date_range=climatology_date_range,
                rolling_window=projection_rolling_window,
                rolling_mode=projection_rolling_mode,
            )
            projection_climatology = Climatology(
                remove=False, standardize=False
            )
            projection_vars = [era5.Var.SST]
            projection_domain = era5.global_domain(step_lon=4, step_lat=4)
            response_rolling_window = None
            response_rolling_mode = "center"
            train_response_time = Time(
                date_range=train_date_range,
                sampling=time_sampling,
                rolling_window=response_rolling_window,
                rolling_mode=response_rolling_mode,
                custom_climatology_date_range=climatology_date_range,
            )
            test_response_time = Time(
                date_range=test_date_range,
                sampling=time_sampling,
                rolling_window=response_rolling_window,
                rolling_mode=response_rolling_mode,
                custom_climatology_date_range=climatology_date_range,
            )
            response_climatology = Climatology(remove=True, standardize=False)
            response_var = era5.Var.SST
            response_domain = era5.nino34_domain(sampling="area_averaged")
            train_covariate_specs = (
                era5.DataSpecs(
                    vars=covariate_vars,
                    domain=covariate_domain,
                    time=train_covariate_time,
                    io=era5_io,
                    climatology=covariate_climatology,
                ),
            )
            projection_specs = era5.DataSpecs(
                vars=projection_vars,
                domain=projection_domain,
                time=projection_time,
                io=era5_io,
                climatology=projection_climatology,
            )
            train_response_specs = era5.DataSpecs(
                vars=[response_var],
                domain=response_domain,
                time=train_response_time,
                io=era5_io,
                climatology=response_climatology,
            )
            test_covariate_specs = (
                era5.DataSpecs(
                    vars=covariate_vars,
                    domain=covariate_domain,
                    time=test_covariate_time,
                    io=era5_io,
                    climatology=covariate_climatology,
                ),
            )
            test_response_specs = era5.DataSpecs(
                vars=[response_var],
                domain=response_domain,
                time=test_response_time,
                io=era5_io,
                climatology=response_climatology,
            )
            train_data_pars = DataPars(
                covariate=Covariate(specs=train_covariate_specs),
                response=Response(specs=train_response_specs),
                **common_pars,
            )
            test_data_pars = DataPars(
                covariate=Covariate(specs=test_covariate_specs),
                response=Response(specs=test_response_specs),
                **common_pars,
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
                    bandwidth_scl=2,
                )
            else:
                tune_pars = TunePars(
                    manifold_dim=None,
                    num_bandwidths=128,
                    log10_bandwidth_lims=(-3, 3),
                    bandwidth_scl=1,
                )
            match kernel_normalization:
                case "diffusion_maps":
                    kernel_pars = DmKernelPars(
                        normalization="fokkerplanck",
                        eigensolver="eigh",
                        num_eigs=512,
                    )
                case "bistochastic":
                    kernel_pars = BsKernelPars(
                        eigensolver="svd",
                        num_eigs=512,
                    )
            pred_pars = PredPars(
                dt=1, which_eigs=512, num_steps=num_pred_steps
            )
            kernel_projection_pars = ProjectionPars(
                data=projection_specs,
                time_origin=train_data_pars.delay_embedding_center,
                which_pcs=[0, 1, 2, 3],
                lead_steps=0,
            )
            if len(jax_env.devices) > 1:
                sharder = NamedSharder(
                    devices=jax_env.devices,
                    shape=(len(jax_env.devices),),
                    axis_names=("x"),
                )
                x_sharding = sharder.sharding("x")
                replicating = sharder.sharding(None)
                l2_shardings = L2FnAlgebraShardings(
                    data=x_sharding, vectors=x_sharding
                )
                l2_tst_shardings = L2FnAlgebraShardings(
                    data=replicating, vectors=replicating
                )
                kernel_eigen_shardings = KernelEigenShardings(
                    eigenvalues=replicating,
                    eigenvectors=x_sharding,
                    weights=x_sharding,
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
        data=test_data_pars, num_pred_steps=pred_pars.num_steps
    )
    pars = Pars(
        train=train_pars,
        test=test_pars,
        kernel_projection=kernel_projection_pars,
    )
    return pars, shardings


pars, shardings = from_experiment(
    EXPERIMENT, CONE_KERNEL, KERNEL_NORMALIZATION
)
io = IO(root=Path.cwd() / OUTPUT_DATA_DIR)


extract_data_arrays = timeit(
    pickleit(
        clim.extract_data_arrays,
        io=io,
        mode=EXTRACT_DATA_MODE,
        fname="data",
        cls=NPData,
    )
)
compute_kernel_bandwidth = timeit(
    pickleit(
        knl.tune_bandwidth,
        io=io,
        mode=TUNE_KERNEL_MODE,
        fname="tune_info",
        cls=TuneInfo,
    )
)
compute_kernel_eigen = timeit(
    pickleit(
        knl.compute_eigen,
        io=io,
        mode=KERNEL_EIGEN_MODE,
        fname="kernel_eigen",
        cls=KernelEigen,
    )
)
compute_kernel_eigen_projections = timeit(
    netcdfit(
        clim.compute_eigenfunction_projections,
        io=io,
        mode=KERNEL_PROJECTION_MODE,
        fname="kernel_eigen_proj",
    )
)
compute_kaf_expansion_coeffs = timeit(
    pickleit(
        clim.compute_kaf_expansion_coeffs,
        io=io,
        mode=KAF_EXPANSION_COEFFS_MODE,
        fname="kaf_coeffs",
        cls=Array,
    )
)
compute_kaf_preds = timeit(
    pickleit(
        knl.compute_kaf_preds,
        io=io,
        mode=KAF_PREDS_MODE,
        fname="kaf_preds",
        cls=Array,
    )
)
compute_skill_scores = timeit(
    h5it(
        clim.compute_skill_scores,
        io=io,
        mode=SKILL_SCORES_MODE,
        fname="pred_scores",
        cls=SkillScores,
        callback=clim.to_skill_scores,
    )
)
plot_kernel_tuning = plotit(
    knl.plot_kernel_tuning,
    io=io,
    mode=PLOT_MODE,
    fname="bandwidth_tuning_func",
)
plot_bandwidth_function = plotit(
    clim.plot_bandwidth_function,
    io=io,
    mode=PLOT_MODE,
    fname="bandwidth_func",
)
plot_laplace_spectrum = plotit(
    knl.plot_laplacian_spectrum, io=io, mode=PLOT_MODE, fname="lapl_spec"
)
make_kernel_evecs_plotter = plotem(
    clim.make_kernel_evecs_plotter,
    io=io,
    mode=PLOT_MODE,
    fname="kernel_eigen",
)
make_kernel_mode_plotter = plotem(
    clim.make_2d_mode_plotter_with_leads,
    io=io,
    mode=PLOT_MODE,
    fname="kernel_modes",
)
plot_kaf_expansion_coeffs = plotit(
    knl.plot_kaf_expansion_coeffs,
    io=io,
    mode=PLOT_MODE,
    fname="kaf_expansion_coeffs",
)
make_running_pred_plotter = plotem(
    clim.make_running_pred_plotter,
    io=io,
    mode=PLOT_MODE,
    fname="pred_running",
)
make_pred_timeseries_plotter = plotem(
    clim.make_pred_timeseries_plotter,
    io=io,
    mode=PLOT_MODE,
    fname="pred_timeseries",
)
plot_forecast_skill_scores = plotit(
    clim.plot_forecast_skill_scores,
    io=io,
    mode=PLOT_MODE,
    fname="pred_scores",
)


def main():
    """Kernel analog forecasting of ESTCP data."""
    global io

    # Display information about the computation to be performed
    jax_env.tabulate()
    pars.tabulate()

    # Generate training and test data
    io @= str(pars.train.data.covariate)
    io /= str(pars.train.data.response)
    io /= str(pars.train.data)
    train_data = extract_data_arrays(
        pars.train.data,
        dtype=np.dtype(np.float64),
    ).to_device(dtype=jax_env.real_dtype, shardings=shardings.train.l2.data)
    io @= str(pars.test.data.covariate)
    io /= str(pars.test.data.response)
    io /= str(pars.test.data)
    test_data = extract_data_arrays(
        pars.test.data, dtype=np.dtype(np.float64)
    ).to_device(dtype=jax_env.real_dtype, shardings=shardings.test.l2.data)

    # Make scalar field and L2 space builders
    scl_r = scls.scalar_field(jax_env.real_dtype)
    impl_l2 = clim.make_data_driven_l2_space(
        data_pars=pars.train.data,
        dtype=jax_env.real_dtype,
        shardings=shardings.train.l2,
    )
    impl_l2_tst = clim.make_data_driven_l2_space(
        data_pars=pars.test.data,
        dtype=jax_env.real_dtype,
        shardings=shardings.test.l2,
    )

    # Set kernel shape function
    io @= str(pars.train.data.covariate)
    io /= str(pars.train.data)
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

        # Plot bandwidth function tuning and bandwidth function
        if PLOT_MODE is not None and not {
            "all",
            "bandwidth_tuning",
        }.isdisjoint(WHICH_PLOTS):
            plot_kernel_tuning(bw_tune_info, title="Bandwidth function tuning")
        if PLOT_MODE is not None and not {"all", "bandwidth_func"}.isdisjoint(
            WHICH_PLOTS
        ):
            plot_bandwidth_function(
                pars.train.data,
                impl_l2,
                bandwidth_func,
                train_data,
                shardings.train.l2.data,
                pars.test.data,
                impl_l2_tst,
                test_data,
                shardings.test.l2.data,
                delay_plot_mode=DELAY_PLOT_MODE,
                plt_date_range=PLT_DATE_RANGE,
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
    if PLOT_MODE is not None and not {
        "all",
        "kernel_tuning",
    }.isdisjoint(WHICH_PLOTS):
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
        kernel_eigen.inspect_array_shardings()
    kernel_eigen.tabulate(num_tabulate=NUM_TABULATE)

    # Plot spectrum of Laplacian eigenvalues
    if PLOT_MODE is not None and not {"all", "laplacian_spec"}.isdisjoint(
        WHICH_PLOTS
    ):
        plot_laplace_spectrum(kernel_eigen)

    # Plot representative kernel eigenfunctions
    if (
        PLOT_MODE is not None
        and not {"all", "kernel_eigen"}.isdisjoint(WHICH_PLOTS)
        and KERNEL_EIGS_PLT is not None
    ):
        _, plot_kernel_eig = make_kernel_evecs_plotter(
            (pars.train.data, pars.train.kernel),
            impl_l2,
            train_data,
            kernel_eigen,
            shardings.train.l2.data,
            pars.test.data,
            impl_l2_tst,
            test_data,
            shardings.test.l2.data,
            kernel,
            delay_plot_mode=DELAY_PLOT_MODE,
            plt_date_range=PLT_DATE_RANGE,
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

    # Compute projections onto kernel eigenfunctions
    if pars.kernel_projection is not None:
        io /= str(pars.kernel_projection.data)
        io /= str(pars.kernel_projection)
        ds_kernel_proj = compute_kernel_eigen_projections(
            pars.kernel_projection,
            kernel_eigen.dual_evecs,
            weights=1 / pars.train.data.num_delay_samples,
        )

        # Plot representative kernel modes
        if (
            PLOT_MODE is not None
            and not {"all", "kernel_modes"}.isdisjoint(WHICH_PLOTS)
            and KERNEL_MODES_PLT is not None
        ):
            _, plot_kernel_mode = make_kernel_mode_plotter(
                ds_kernel_proj,
                vars=pars.kernel_projection.data.varnames,
                ocean_only_vars=pars.kernel_projection.data.varnames,
            )
            if KERNEL_MODES_PLT == "interactive":
                while True:
                    i = input(
                        "Select kernel mode "
                        f"0-{ds_kernel_proj.sizes['mode'] - 1} to plot, "
                        "or press Enter to continue. "
                    )
                    if i == "":
                        break
                    else:
                        try:
                            plot_kernel_mode(int(i))
                        except ValueError:
                            print("Invalid input.")
            else:
                for i in KERNEL_MODES_PLT:
                    plot_kernel_mode(i)
                    if "show" in PLOT_MODE:
                        input("Press any key to continue...")
        io -= 2

    # Compute expansion coefficients of the response function
    io /= str(pars.train.data.response)
    io /= str(pars.train.pred)
    coeffs = compute_kaf_expansion_coeffs(
        (pars.train.data, pars.train.kernel),
        impl_l2,
        train_data,
        kernel,
        kernel_eigen,
        pars.train.pred.num_steps,
        pars.train.pred.which_eigs,
    )

    # Make heatmap of KAF response coefficients
    if PLOT_MODE is not None and not {"all", "expansion_coeffs"}.isdisjoint(
        WHICH_PLOTS
    ):
        plot_kaf_expansion_coeffs(
            coeffs,
            title=f"Response: {pars.train.data.response}",
        )

    # Perform time series prediction
    io /= str(pars.test.data)
    preds = compute_kaf_preds(
        pars.train.kernel,
        impl_l2,
        train_data,
        kernel,
        kernel_eigen,
        coeffs,
        impl_l2_tst,
        test_data,
        pars.train.pred.which_eigs,
    )

    # Plot running forecast
    if (
        PLOT_MODE is not None
        and not {"all", "running_pred"}.isdisjoint(WHICH_PLOTS)
        and LEAD_TIMES_PLT is not None
    ):
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
                if "show" in PLOT_MODE:
                    input("Press any key to continue...")

    # Plot time series forecast
    if (
        PLOT_MODE is not None
        and not {"all", "pred_timeseries"}.isdisjoint(WHICH_PLOTS)
        and INITIALIZATION_TIMES_PLT is not None
    ):
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
                if "show" in PLOT_MODE:
                    input("Press any key to continue...")

    # Compute forecast skill scores
    skill_scores = compute_skill_scores(
        pars.test.data,
        test_data,
        preds,
    )
    ts = jnp.arange(pars.test.num_pred_steps + 1)
    match pars.train.data.time_sampling:
        case "daily":
            timestep_str = "days"
        case "monthly":
            timestep_str = "months"
    print(
        tabulate(
            jnp.vstack((ts, skill_scores["nrmses"], skill_scores["accs"])).T,
            headers=[
                f"Lead time ({timestep_str})",
                "Normalized RMSE",
                "Anomaly Correlation",
            ],
            floatfmt=".4f",
        )
    )

    # Plot forecast skill scores
    if PLOT_MODE is not None and not {"all", "skill_scores"}.isdisjoint(
        WHICH_PLOTS
    ):
        plot_forecast_skill_scores(pars.test.data, skill_scores)


if __name__ == "__main__":
    if len(sys.argv) == 2 and (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
        print(__doc__)
    else:
        main()

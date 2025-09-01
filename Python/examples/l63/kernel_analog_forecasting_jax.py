# pyright: basic
"""Kernel analog forecasting of the Lorenz 63 system."""

import diffrax as dfx
import jax
import jax.numpy as jnp
import matplotlib.figure as mpf
import matplotlib.pyplot as plt
import nlsa.jax.delays as dl
import nlsa.jax.distance as dst
import nlsa.jax.dynamics as dyn
import nlsa.jax.kernels as knl
import nlsa.jax.vector_algebra as vec
import nlsa.jax.euclidean as r3
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from diffrax import Dopri5, ODETerm, PIDController, SaveAt
from functools import partial
from jax import Array, vmap
from jax.typing import DTypeLike
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from nlsa.function_algebra import compose
from nlsa.jax.kernels import KernelEigen, KernelEigenbasis, KernelPars, \
    TuneInfo, TunePars
from nlsa.jax.stats import anomaly_correlation, normalized_rmse
from nlsa.jax.vector_algebra import L2VectorAlgebra
from nlsa.io_actions import IO, h5it, pickleit, plotit, plotem, timeit
from numpy.typing import ArrayLike
from pathlib import Path
from tabulate import tabulate
from typing import Literal, Optional, TypedDict

IDX_CPU: Optional[int] = None
IDX_GPU: Optional[int] = None
XLA_MEM_FRACTION: Optional[str] = None
FP: Literal['F32', 'F64'] = "F32"
OUTPUT_DATA_DIR = "examples/l63/data"
NUM_TABULATE = 40
DELAY_EMBEDDING_MODE: Optional[Literal['explicit',
                                       'on_the_fly']] = "on_the_fly"
GENERATE_DATA_MODE: Literal['calc', 'calcsave', 'read'] = "calcsave"
TUNE_KERNEL_MODE: Literal['calc', 'calcsave', 'read'] = "calcsave"
KERNEL_EIGEN_MODE: Literal['calc', 'calcsave', 'read'] = "calcsave"
SKILL_SCORES_MODE: Literal['calc', 'calcsave', 'read'] = "calcsave"
PLOT_MODE: Optional[Literal['save', 'show', 'saveshow']] = "save"
DELAY_PLOT_MODE: Literal['backward', 'central'] = "backward"
KERNEL_EIGS_PLT: Optional[Sequence[int] | Literal['interactive']] \
    = "interactive"
LEAD_TIMES_PLT: Optional[Sequence[int] | Literal['interactive']] \
    = "interactive"
INITIALIZATION_TIMES_PLT: Optional[Sequence[int] | Literal['interactive']] \
    = "interactive"

if IDX_GPU is not None and XLA_MEM_FRACTION is not None:
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = XLA_MEM_FRACTION

if IDX_CPU is not None:
    jax.config.update("jax_default_device", jax.devices("cpu")[IDX_CPU])
    device_cpu = jax.devices("cpu")[IDX_CPU]
else:
    device_cpu = jax.devices("cpu")[0]

if IDX_GPU is not None:
    jax.config.update("jax_default_device", jax.devices("gpu")[IDX_GPU])

if IDX_GPU is not None:
    device = jax.devices("gpu")[IDX_GPU]
else:
    device = device_cpu
jax.config.update("jax_default_device", device)

match FP:
    case "F32":
        r_dtype = jnp.float32
        c_dtype = jnp.complex64
    case "F64":
        jax.config.update("jax_enable_x64", True)
        r_dtype = jnp.float64
        c_dtype = jnp.complex128

type X = Array  # Point in state space
type Xs = Array  # Collection of points in state space
type Y = Array  # Point in covariate space
type Ys = Array  # Collection of points in covariate space
type Yd = Array  # Point in delay-coordinate space
type R = Array  # Real number
type Rs = Array  # Collection of real numbers
type V = Array  # Vector in L2
type Vs = Array  # Collection of vectors in L2
type Vtsts = Array  # Collection of vectors in L2 with respect to test dataset
type F[*Ss, T] = Callable[[*Ss], T]  # Shorthand for Callables

jvmap: Callable[[F[X, Y]], F[Xs, Ys]] = compose(jax.jit, vmap)


@dataclass(frozen=True)
class DataPars[N: int]():
    """Dataclass containing training and test data parameter values."""

    covariate: Literal['x', 'y', 'z', 'xy', 'xyz']
    """Covariate function."""

    response: Literal['x', 'y', 'z']
    """Response function."""

    dt: float
    """Sampling interval."""

    num_spinup: int
    """Number of spinup samples."""

    num_samples: N
    """Number of samples (after spinup, delays, and finite difference)."""

    x0: tuple[float, float, float]
    """Initial condition."""

    num_fd: int = 0
    """Number of extra samples for finite differencing."""

    num_half_delays: int = 0
    """Half number of delays (to ensure even two-sided embedding window)."""

    num_quad: int = 0
    """Number of extra samples for resolvent quadrature."""

    num_pred_steps: int = 0
    """Number of extra samples for prediction."""

    batch_size: Optional[int] = None
    """Number of batches for batchwise evaluation."""

    @property
    def num_delays(self) -> int:
        """Number of delays."""
        return 2 * self.num_half_delays

    @property
    def num_delay_samples(self) -> int:
        """Index of delay embedding origin."""
        return 2*self.num_half_delays + self.num_samples

    @property
    def delay_embedding_origin(self) -> int:
        """Index of delay embedding origin."""
        return self.num_fd // 2

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
        """Total number of analysis samples (excluding spinup)."""
        return self.num_samples + self.num_delays + self.num_fd \
            + self.num_pred_steps + self.num_quad

    def __str__(self) -> str:
        """Create string representing prediction parameters."""
        x0str = "x0_{self.x0[0]:.2f}_{self.x0[1]:.2f}_{self.x0[2]:.2f}"
        return '_'.join((self.covariate,
                         self.response,
                         "dt{self.dt:.2f}",
                         "n{self.num_samples}",
                         "nspinup{self.num_spinup}",
                         "nfd{self.num_fd}",
                         "ndl{self.num_half_delays}",
                         "nquad{self.num_quad}",
                         "npred{self.num_pred_steps}",
                         x0str))


@dataclass(frozen=True)
class PredPars:
    """Dataclass containing prediction parameters."""

    which_eigs: int | tuple[int, int] | list[int]
    """Kernel eigenfunctions used in the prediction function."""

    def __str__(self) -> str:
        """Create string representing prediction parameters."""
        match self.which_eigs:
            case int():
                eigs_str = "-".join(map(str, (0, self.which_eigs)))
            case tuple(ints) if all(isinstance(i, int) for i in ints) \
                    and len(ints) == 2:
                eigs_str = "-".join(map(str, self.which_eigs))
            case _:
                eigs_str = "_".join(map(str, self.which_eigs))
        return '_'.join(("pred",
                         "neigs" + eigs_str))


@dataclass(frozen=True)
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

    bw_tune: Optional[TunePars] = None
    """Tuning parameters for kernel bandwidth function."""

    def __str__(self) -> str:
        """Create string training parameters."""
        if self.bw_tune is None:
            bw_tune_str = ""
        else:
            bw_tune_str = str(self.bw_tune)
        return '_'.join((str(self.data),
                         str(self.tune),
                         bw_tune_str,
                         str(self.kernel),
                         str(self.pred)))


@dataclass(frozen=True)
class TestPars[Ntst: int]:
    """Dataclass containing test parameter values."""

    data: DataPars[Ntst]
    """Test data parameters."""


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

    covariate: Literal['x', 'y', 'z', 'xy', 'xyz']
    response: Literal['x', 'y', 'z']
    dt: float
    num_spinup: int
    num_half_delays: int
    num_pred_steps: int


common_pars: CommonPars = {'covariate': "xyz",
                           'response': "x",
                           'dt': 0.1,
                           'num_spinup': 1000,
                           'num_half_delays': 5,
                           'num_pred_steps': 50}
train_data_pars = DataPars(**common_pars, x0=(1, 1, 1.1),
                           num_samples=1024)
test_data_pars = DataPars(**common_pars, x0=(1, 1, 0.9),
                          num_samples=1024,
                          batch_size=None)
tune_pars = TunePars(manifold_dim=2.1,
                     num_bandwidths=128,
                     log10_bandwidth_lims=(-3, 3),
                     bandwidth_scl=1)
kernel_pars = KernelPars(normalization="bistochastic",
                         eigensolver="eigsh",
                         num_eigs=128)
pred_pars = PredPars(which_eigs=kernel_pars.num_eigs)
train_pars = TrainPars(data=train_data_pars,
                       bw_tune=tune_pars,
                       tune=tune_pars,
                       kernel=kernel_pars,
                       pred=pred_pars)
test_pars = TestPars(data=test_data_pars)
pars = Pars(train=train_pars, test=test_pars)
io = IO(root=Path.cwd() / OUTPUT_DATA_DIR)


def to_data(dict_in: dict[str, ArrayLike],
            dtype: Optional[DTypeLike] = None) -> Data:
    """Convert dictionary of numpy ArrayLike objects to Data TypedDict."""
    try:
        data: Data = {'states': jnp.array(dict_in['states'], dtype),
                      'covariates': jnp.array(dict_in['covariates'], dtype),
                      'responses': jnp.array(dict_in['responses'], dtype)}
        return data
    except ValueError as exc:
        raise ValueError("Incompatible keys/values") from exc


def to_kernel_eigen(dict_in: dict[str, ArrayLike],
                    dtype: Optional[DTypeLike] = None) -> KernelEigen:
    """Convert dict of numpy ArrayLike objects to KernelEigen TypedDict."""
    try:
        kernel_eigen: KernelEigen = {
            'evals': jnp.array(dict_in['evals'], dtype),
            'evecs': jnp.array(dict_in['evecs'], dtype),
            'dual_evecs': jnp.array(dict_in['dual_evecs'], dtype),
            'weights': jnp.array(dict_in['weights'], dtype),
            'bandwidth': jnp.array(dict_in['bandwidth'], dtype)}
        return kernel_eigen
    except ValueError as exc:
        raise ValueError("Incompatible keys/values") from exc


def to_skill_scores(dict_in: dict[str, ArrayLike],
                    dtype: Optional[DTypeLike] = None) -> SkillScores:
    """Convert dict of numpy ArrayLike objects to SkillScores TypedDict."""
    try:
        skill_scores: SkillScores = {
            'nrmses': jnp.array(dict_in['nrmses'], dtype),
            'accs': jnp.array(dict_in['accs'], dtype)}
        return skill_scores
    except ValueError as exc:
        raise ValueError("Incompatible keys/values") from exc


tune_kernel_bandwidth = timeit(pickleit(knl.tune_bandwidth, io=io,
                                        mode=TUNE_KERNEL_MODE,
                                        fname="tune_kernel",
                                        cls=tuple[Array, TuneInfo]))
compute_kernel_eigen = timeit(h5it(knl.compute_eigen, io=io,
                                   mode=KERNEL_EIGEN_MODE,
                                   fname="kernel_eigen", cls=KernelEigen,
                                   callback=partial(to_kernel_eigen,
                                                    dtype=r_dtype)))


@timeit
@partial(h5it, io=io, mode=GENERATE_DATA_MODE, fname="data", cls=Data,
         callback=partial(to_data, dtype=r_dtype))
def generate_data[N: int](pars: DataPars[N], dtype: DTypeLike) -> Data:
    """Generate L63 data."""
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
    covariate = jvmap(cov)
    response = jvmap(rsp)
    v = dyn.make_l63_vector_field()
    num_ode_samples = pars.num_total_samples + pars.num_spinup
    with jax.default_device(device_cpu):
        if FP == "F32":
            jax.config.update("jax_enable_x64", True)
        ts_tot = jnp.arange(num_ode_samples) * pars.dt
        solution = dfx.diffeqsolve(
            terms=ODETerm(jax.jit(dyn.from_autonomous(v))), solver=Dopri5(),
            t0=0, t1=ts_tot[-1], dt0=pars.dt,
            y0=jnp.array(pars.x0, dtype=dtype),
            saveat=SaveAt(ts=ts_tot[pars.num_spinup:]),
            stepsize_controller=PIDController(rtol=1e-8, atol=1e-8),
            max_steps=200_000_000)
        if FP == "F32":
            jax.config.update("jax_enable_x64", False)
    xs = jnp.array(solution.ys, dtype=r_dtype)
    ys = covariate(xs)
    zs = response(xs)
    return {'states': xs, 'covariates': ys, 'responses': zs}


def make_l2_space[N: int, D: DTypeLike](
        pars: DataPars[N], dtype: D, data: Data,
        delay_embedding_mode: Optional[Literal['explicit', 'on_the_fly']]
        = DELAY_EMBEDDING_MODE,
        jit: bool = False) -> L2VectorAlgebra[tuple[N], D, Yd, R]:
    """Make L2 space over covariate data space."""
    i0 = pars.delay_embedding_origin
    i1 = i0 + pars.num_delay_samples
    hankel = jax.jit(partial(dl.hankel, num_delays=pars.num_delays,
                             flatten=True))
    incl: Callable[[F[Array, Array]], V]
    match pars.num_half_delays, pars.batch_size:
        case 0, None:
            incl = vec.veval_at(data['covariates'][i0:i1], jit=jit)
        case 0, _:
            incl = vec.batch_eval_at(data['covariates'][i0:i1],
                                     batch_size=pars.batch_size)
        case _, None:
            assert delay_embedding_mode is not None
            if delay_embedding_mode == "explicit":
                incl = vec.veval_at(hankel(data['covariates'][i0:i1]))
            elif delay_embedding_mode == "on_the_fly":
                incl = dl.delay_eval_at(data['covariates'][i0:i1],
                                        num_delays=pars.num_delays, jit=jit)
        case _, _:
            assert delay_embedding_mode is not None
            if delay_embedding_mode == "explicit":
                incl = vec.batch_eval_at(hankel(data['covariates'][i0:i1]),
                                         batch_size=pars.batch_size)
            elif delay_embedding_mode == "on_the_fly":
                incl = dl.batch_delay_eval_at(data['covariates'][i0:i1],
                                              batch_size=pars.batch_size,
                                              num_delays=pars.num_delays)
    mu = vec.make_normalized_counting_measure(pars.num_samples)
    return L2VectorAlgebra(shape=(pars.num_samples,), dtype=dtype, measure=mu,
                           inclusion_map=incl)


def make_timeseries_prediction_function[N: int](
        pars: DataPars[N], train_data: Data, nyst: Callable[[V], F[Yd, R]]) \
            -> F[Yd, Rs]:
    """Make vector-valued prediction function for time series prediction."""
    i0 = pars.delay_embedding_end
    i1 = i0 + pars.num_pred_steps + pars.num_samples
    fxs_ts = dl.hankel(train_data['responses'][i0:i1],
                       num_delays=pars.num_pred_steps)

    @partial(vmap, in_axes=(1, None))
    def predict(v: V, y: Yd) -> R:
        return nyst(v)(y)

    return partial(predict, fxs_ts)


plot_kernel_tuning = plotit(knl.plot_kernel_tuning, io=io, mode=PLOT_MODE,
                            fname="bandwidth_tuning_func")
plot_laplace_spectrum = plotit(knl.plot_laplace_spectrum, io=io,
                               mode=PLOT_MODE, fname="lapl_spec")


@timeit
@partial(h5it, io=io, mode=SKILL_SCORES_MODE, fname="pred_scores",
         cls=SkillScores, callback=to_skill_scores)
def compute_skill_scores[Ntst: int](pars: DataPars[Ntst], test_data: Data,
                                    fys_pred: Vtsts) -> SkillScores:
    """Compute NRMSE and ACC skill scores over the prediction ensemble."""
    i0 = pars.delay_embedding_end
    i1 = i0 + pars.num_pred_steps + pars.num_samples
    fxs_true = dl.hankel(test_data['responses'][i0:i1],
                         num_delays=pars.num_pred_steps)
    nrmses = jax.jit(vmap(normalized_rmse, in_axes=1))(fxs_true, fys_pred)
    accs = jax.jit(vmap(anomaly_correlation, in_axes=1))(fxs_true, fys_pred)
    scores: SkillScores = {'nrmses': nrmses, 'accs': accs}
    return scores


@partial(plotit, io=io, mode=PLOT_MODE, fname="bandwidth_func")
def plot_bandwidth_func[N: int, Ntst: int, D: DTypeLike](
        pars: DataPars[N], l2y: L2VectorAlgebra[tuple[N], D, Yd, R],
        bandwidth_func: F[Yd, R], train_data: Data,
        test_pars: Optional[DataPars[Ntst]] = None,
        l2y_tst: Optional[L2VectorAlgebra[tuple[Ntst], D, Yd, R]] = None,
        test_data: Optional[Data] = None,
        delay_plot_mode: Literal['backward', 'central'] = "central",
        num_plt: Optional[int] = None, num_plt_tst: Optional[int] = None,
        plt_step: int = 1, plt_step_tst: int = 1, i_fig: int = 1) -> Figure:
    """Plot bandwidth function on training and, optionally, test data."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    if test_pars is not None:
        fig = plt.figure(num=i_fig, figsize=tuple(mpf.figaspect(0.5)))
        axs = (fig.add_subplot(1, 2, 1, projection="3d"),
               fig.add_subplot(1, 2, 2, projection="3d"))
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
    plt.rcParams['grid.color'] = "yellow"
    sc = ax.scatter(train_data['states'][i0:i1:plt_step, 0],
                    train_data['states'][i0:i1:plt_step, 1],
                    train_data['states'][i0:i1:plt_step, 2],
                    c=bw_vals[:num_plt:plt_step], s=1, vmin=vmin, vmax=vmax,
                    cmap="binary")
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
            test_data['states'][i0_tst:i1_tst:plt_step_tst, 0],
            test_data['states'][i0_tst:i1_tst:plt_step_tst, 1],
            test_data['states'][i0_tst:i1_tst:plt_step_tst, 2],
            c=bw_vals_tst[:num_plt_tst:plt_step_tst], s=1, vmin=vmin,
            vmax=vmax, cmap="binary")
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


@partial(plotem, io=io, mode=PLOT_MODE, fname="kernel_eigen", block=False)
def make_kernel_evecs_plotter[N: int, Ntst: int, D: DTypeLike](
        pars: DataPars[N], train_data: Data,
        kernel_basis: KernelEigenbasis[Yd, R, V, Rs, int | Array],
        test_pars: Optional[DataPars[Ntst]] = None,
        l2y_tst: Optional[L2VectorAlgebra[tuple[Ntst], D, Yd, R]] = None,
        test_data: Optional[Data] = None,
        delay_plot_mode: Literal['backward', 'central'] = 'backward',
        num_plt: Optional[int] = None, num_plt_tst: Optional[int] = None,
        plt_step: int = 1, plt_step_tst: int = 1, i_fig: int = 1) \
            -> tuple[Figure, F[int, None]]:
    """Make plotting function for kernel eigenfunctions."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    if test_pars is not None:
        fig = plt.figure(num=i_fig, figsize=tuple(mpf.figaspect(0.5)))
        axs = (fig.add_subplot(1, 2, 1, projection="3d"),
               fig.add_subplot(1, 2, 2, projection="3d"))
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
        if test_pars is not None and l2y_tst is not None \
                and test_data is not None:
            evec_tst = l2y_tst.incl(kernel_basis.fn(j))
            amax = max(amax, float(jnp.abs(jnp.max(evec_tst))))

        for figax in fig.axes:
            figax.cla()
        sc = ax.scatter(train_data['states'][i0:i1:plt_step, 0],
                        train_data['states'][i0:i1:plt_step, 1],
                        train_data['states'][i0:i1:plt_step, 2],
                        c=evec[:num_plt:plt_step], s=1, vmin=-amax, vmax=amax,
                        cmap="seismic")
        eta = kernel_basis.lapl_evl(j)
        ax.set_xlabel("$x^1$")
        ax.set_ylabel("$x^2$")
        ax.set_zlabel("$x^3$")
        ax.set_title(f"Eigenvector {j}: $\\eta_{{{j}}} = {eta: .3f}$")

        if test_pars is not None and l2y_tst is not None \
                and test_data is not None and kernel_basis is not None:
            sc_tst = ax_tst.scatter(
                test_data['states'][i0_tst:i1_tst:plt_step_tst, 0],
                test_data['states'][i0_tst:i1_tst:plt_step_tst, 1],
                test_data['states'][i0_tst:i1_tst:plt_step_tst, 2],
                c=evec_tst[:num_plt_tst:plt_step_tst], s=1, vmin=-amax,
                vmax=amax, cmap="seismic")
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


@partial(plotem, io=io, mode=PLOT_MODE, fname="pred_running")
def make_pred_plotter[Ntst: int](
        test_pars: DataPars[Ntst], test_data: Data, preds: Vtsts,
        num_plt_tst: Optional[int] = None, plt_step_tst: int = 1,
        i_fig: int = 1) -> tuple[Figure, F[int, None]]:
    """Make plotting function for prediction over different lead times."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig = plt.figure(num=i_fig, figsize=tuple(mpf.figaspect(0.3)))
    axs = (fig.add_subplot(1, 3, 1, projection="3d"),
           fig.add_subplot(1, 3, 2, projection="3d"),
           fig.add_subplot(1, 3, 3, projection="3d"))
    fig.set_layout_engine("constrained")

    def plot_pred(i_step: int):
        i0_tst = test_pars.delay_embedding_end
        if num_plt_tst is not None:
            i1_tst = i0_tst + num_plt_tst
        else:
            i1_tst = i0_tst + test_pars.num_samples
        i0_pred = i0_tst + i_step
        i1_pred = i1_tst + i_step
        err = preds[:num_plt_tst, i_step] \
            - test_data['responses'][i0_pred:i1_pred]
        amax = max(
            float(jnp.max(jnp.abs(test_data['responses'][i0_pred:i1_pred]))),
            float(jnp.max(jnp.abs(preds[:, i_step]))))
        emax = float(jnp.max(jnp.abs(err)))
        for ax in axs:
            ax.cla()

        ax = axs[0]
        assert isinstance(ax, Axes3D)
        sc_tst = ax.scatter(
            test_data['states'][i0_tst:i1_tst:plt_step_tst, 0],
            test_data['states'][i0_tst:i1_tst:plt_step_tst, 1],
            test_data['states'][i0_tst:i1_tst:plt_step_tst, 2],
            c=test_data['responses'][i0_pred:i1_pred:plt_step_tst], s=1,
            vmin=-amax, vmax=amax, cmap="seismic")
        ax.set_title("True")
        ax.set_xlabel("$x^1$")
        ax.set_ylabel("$x^2$")
        ax.set_zlabel("$x^3$")

        ax = axs[1]
        assert isinstance(ax, Axes3D)
        ax.scatter(test_data['states'][i0_tst:i1_tst:plt_step_tst, 0],
                   test_data['states'][i0_tst:i1_tst:plt_step_tst, 1],
                   test_data['states'][i0_tst:i1_tst:plt_step_tst, 2],
                   c=preds[:num_plt_tst:plt_step_tst, i_step], s=1, vmin=-amax,
                   vmax=amax, cmap="seismic")
        ax.set_xlabel("$x^1$")
        ax.set_ylabel("$x^2$")
        ax.set_zlabel("$x^3$")
        ax.set_title(f"Prediction; lead time = {i_step * test_pars.dt}")

        ax = axs[2]
        assert isinstance(ax, Axes3D)
        sc_err = ax.scatter(test_data['states'][i0_tst:i1_tst:plt_step_tst, 0],
                            test_data['states'][i0_tst:i1_tst:plt_step_tst, 1],
                            test_data['states'][i0_tst:i1_tst:plt_step_tst, 2],
                            c=err[::plt_step_tst], s=1, vmin=-emax, vmax=emax,
                            cmap="seismic")
        ax.set_xlabel("$x^1$")
        ax.set_ylabel("$x^2$")
        ax.set_zlabel("$x^3$")
        ax.set_title("Error")

        if len(axs) > 4:
            fig.colorbar(sc_tst, ax=axs[:2], cax=axs[3], location="bottom",
                         shrink=0.75, aspect=60, pad=0)
        else:
            fig.colorbar(sc_tst, ax=axs[:2], location="bottom", shrink=0.75,
                         aspect=60, pad=0)
        if len(axs) > 5:
            fig.colorbar(sc_err, ax=axs[2], cax=axs[4], location="bottom",
                         shrink=0.75, aspect=30, pad=0)
        else:
            fig.colorbar(sc_err, ax=axs[2], location="bottom", shrink=0.75,
                         aspect=30, pad=0)
    return fig, plot_pred


@partial(plotem, io=io, mode=PLOT_MODE, fname="pred_timeseries")
def make_pred_timeseries_plotter[Ntst: int](
        pars: DataPars[Ntst], test_data: Data, preds: Vtsts,
        i_fig: int = 1) -> tuple[Figure, F[int, None]]:
    """Make plotting function over different initial conditions."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    ts = jnp.arange(pars.num_pred_steps + 1) * pars.dt

    def plot_pred(i_init: int):
        i0_tst = pars.delay_embedding_end + i_init
        i1_tst = i0_tst + pars.num_pred_steps + 1
        ax.cla()
        ax.plot(ts, test_data['responses'][i0_tst:i1_tst], "o-", label="test")
        ax.plot(ts, preds[i_init, :], "o-", label="prediction")
        ax.grid()
        ax.legend()
        ax.set_xlabel("Forecast time")
        ax.set_title(f"Initial condition = {i_init}")
    return fig, plot_pred


@partial(plotit, io=io, mode=PLOT_MODE, fname="pred_scores")
def plot_forecast_skill_scores[Ntst: int](pars: DataPars[Ntst],
                                          scores: SkillScores,
                                          i_fig: int = 1) -> Figure:
    """Plot NRMSE and ACC versus forecast lead time."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, axs = plt.subplots(2, 1, num=i_fig, constrained_layout=True,
                            sharex=True)
    labels = ("NRMSE", "Anomaly correlation")
    ts = jnp.arange(pars.num_pred_steps + 1) * pars.dt
    for ax, score, label in zip(axs, (scores['nrmses'], scores['accs']),
                                labels):
        ax.plot(ts, score, "o-")
        ax.grid()
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("Forecast time")
        ax.set_ylabel(label)
    return fig


def main():
    """Kernel analog forecasting of the Lorenz 63 system."""
    global io

    # Generate training and test data
    io @= str(pars.test.data)
    test_data = generate_data(pars.test.data, r_dtype)
    l2y_tst = make_l2_space(pars.test.data, r_dtype, test_data, jit=True)
    io @= str(pars.train.data)
    train_data = generate_data(pars.train.data, r_dtype)
    l2y = make_l2_space(pars.train.data, r_dtype, train_data)

    # Set distance function and kernel shape function
    sqdist = dst.sqeuclidean
    shape_func = jnp.exp

    # Create and tune bandwidth function
    if pars.train.bw_tune is not None:
        io /= str(pars.train.bw_tune)
        bw_kernel_family = knl.make_kernel_family(l2y.scl, shape_func, sqdist)
        bw_bandwidth, bw_tune_info = tune_kernel_bandwidth(
            pars.train.bw_tune, l2y, bw_kernel_family)
        bw_kernel = bw_kernel_family(bw_bandwidth)
        bandwidth_func = knl.make_bandwidth_function(
            l2y, bw_kernel, dim=jnp.asarray(bw_tune_info['dim'], r_dtype),
            vol=jnp.asarray(bw_tune_info['vol'], r_dtype))
        print("Bandwidth function tuning:")
        print(f"Optimal bandwidth index: {bw_tune_info['i_opt']}")
        print(f"Optimal bandwidth: {bw_tune_info['opt_bandwidth']:.3e}")
        print(f"Optimal dimension: {bw_tune_info['opt_dim']:.3e}")
        print("Bandwidth used for diffusion maps: "
              f"{bw_tune_info['bandwidth']:.3e}")
        print("Dimension based on diffusion maps bandwidth: "
              f"{bw_tune_info['dim']:.3e}")
        print(f"Manifold volume: {bw_tune_info['vol']:.3e}")

        # Plot bandwidth function
        if PLOT_MODE is not None:
            plot_kernel_tuning(bw_tune_info, title="Bandwidth function tuning")
            plot_bandwidth_func(pars.train.data, l2y, bandwidth_func,
                                train_data, pars.test.data, l2y_tst, test_data)

    # Create and tune kernel
    io /= str(pars.train.bw_tune)
    if pars.train.bw_tune is not None:
        scaled_sqdist = knl.make_scaled_sqdist(l2y.scl, sqdist, bandwidth_func)
        kernel_family = knl.make_kernel_family(l2y.scl, shape_func,
                                               scaled_sqdist)
    else:
        kernel_family = knl.make_kernel_family(l2y.scl, shape_func, sqdist)
    bandwidth, tune_info = tune_kernel_bandwidth(pars.train.tune, l2y,
                                                 kernel_family)
    kernel = kernel_family(bandwidth)
    print("Kernel tuning:")
    print(f"Bandwidth index: {tune_info['i_opt']}")
    print(f"Optimal bandwidth: {tune_info['opt_bandwidth']:.3e}")
    print(f"Optimal dimension: {tune_info['opt_dim']:.3e}")
    print(f"Bandwidth used for diffusion maps: {tune_info['bandwidth']:.3e}")
    print("Dimension based on diffusion maps bandwidth: "
          f"{tune_info['dim']:.3e}")
    print(f"Manifold volume: {tune_info['vol']:.3e}")

    # Plot kernel tuning function
    if PLOT_MODE is not None:
        plot_kernel_tuning(tune_info, title="Kernel tuning")

    # Solve kernel eigenvalue problem
    io /= str(pars.train.kernel)
    kernel_eigen = compute_kernel_eigen(pars.train.kernel, l2y, kernel,
                                        bandwidth)
    print(tabulate(jnp.vstack((
        kernel_eigen['evals'][:NUM_TABULATE],
        knl.to_laplace_eigenvalues(
            kernel_eigen['evals'][:NUM_TABULATE],
            kernel_eigen['bandwidth']))).T,
        headers=["Kernel eigenvalues", "Laplace eigenvalues"],
        floatfmt=".4f", showindex=True))

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
        pars.train.kernel, l2y, kernel, kernel_eigen, laplace_method="log",
        which_eigs=which_eigs)
    nyst = compose(kernel_basis.fn_synth, kernel_basis.anal)

    # Plot representative kernel eigenfunctions
    if PLOT_MODE is not None and KERNEL_EIGS_PLT is not None:
        _, plot_kernel_eig = make_kernel_evecs_plotter(
            pars.train.data, train_data, kernel_basis, pars.test.data, l2y_tst,
            test_data, delay_plot_mode=DELAY_PLOT_MODE)
        if KERNEL_EIGS_PLT == "interactive":
            while True:
                i = input(
                    "Select kernel eigenfunction "
                    f"0-{pars.train.kernel.num_eigs-1} to plot, "
                    "or press Enter to continue. ")
                if i == '':
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
    io /= str(pars.test.data)
    predict = make_timeseries_prediction_function(pars.train.data,
                                                  train_data, nyst)
    fys_pred = timeit(l2y_tst.incl)(predict)

    # Plot running forecast    # Plot running forecast
    if PLOT_MODE is not None and LEAD_TIMES_PLT is not None:
        _, plot_pred = make_pred_plotter(pars.test.data, test_data, fys_pred)
        if LEAD_TIMES_PLT == "interactive":
            while True:
                i = input(
                    "Select lead time "
                    f"0-{pars.test.data.num_pred_steps} to plot, "
                    "or press Enter to continue. ")
                if i == '':
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
        _, plot_pred_ts = make_pred_timeseries_plotter(pars.test.data,
                                                       test_data, fys_pred)
        if INITIALIZATION_TIMES_PLT == "interactive":
            while True:
                i = input(
                    "Select initialization time "
                    f"0-{pars.test.data.num_samples} to plot, "
                    "or press Enter to continue. ")
                if i == '':
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

    # Compute normalized RMSE
    skill_scores = compute_skill_scores(pars.test.data, test_data, fys_pred)
    ts = jnp.arange(pars.test.data.num_pred_steps + 1) * pars.test.data.dt
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


if __name__ == '__main__':
    if len(sys.argv) == 2 and (sys.argv[1] == '--help' or sys.argv[1] == '-h'):
        print(__doc__)
    else:
        main()

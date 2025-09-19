# pyright: basic
"""Koopman spectral analysis of torus rotation using resolvent
compactification.
"""
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import matplotlib.figure as mpf
import matplotlib.pyplot as plt
import math
import nlsa.jax.delays as dl
import nlsa.jax.distance as dst
import nlsa.jax.dynamics as dyn
import nlsa.jax.kernels as knl
import nlsa.jax.vector_algebra as vec
import nlsa.jax.torus as torus
import os
import seaborn as sns
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from jax import Array, jit, vmap
from jax.typing import DTypeLike
from matplotlib.figure import Figure
from nlsa.function_algebra import compose, compose2
from nlsa.jax.kernels import KernelEigen, KernelPars, TunePars
from nlsa.jax.stats import anomaly_correlation, normalized_rmse
from nlsa.jax.vector_algebra import L2VectorAlgebra, VectorAlgebra
from nlsa.utils import timeit
from typing import Literal, Optional, TypedDict

IDX_CPU: Optional[int] = None
IDX_GPU: Optional[int] = None
XLA_MEM_FRACTION: Optional[str] = None
F64: bool = True
PLOT_FIGS = False

if IDX_GPU is not None and XLA_MEM_FRACTION is not None:
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = XLA_MEM_FRACTION

if IDX_CPU is not None:
    jax.config.update("jax_default_device", jax.devices("cpu")[IDX_CPU])

if IDX_GPU is not None:
    jax.config.update("jax_default_device", jax.devices("gpu")[IDX_GPU])

if F64:
    jax.config.update("jax_enable_x64", True)
    r_dtype = jnp.float64
    c_dtype = jnp.complex128
else:
    r_dtype = jnp.float32
    c_dtype = jnp.complex64

type X = Array  # Point in state space
type TX = Array  # Point in tangent space
type Xs = Array  # Collection of points in state space
type Y = Array  # Point in covariate space
type Ys = Array  # Collection of points in covariate space
type R = Array  # Real number
type Rl = Array  # l-dimensional real vectors
type Rs = Array  # Collection of real numbers
type C = Array  # Complex number
type Cs = Array  # Collection of complex numbers
type Cl = Array  # l-dimensional complex vectors
type Cls = Array  # Collection of l-dimensional complex vectors
type Clk = Array  # lk-dimensional complex vectors
type M = Array  # Matrix
type V = Array  # Vector in L2
type Vs = Array  # Collection of vectors in L2
type Vtsts = Array  # Collection of vectors in L2 with respect to test dataset
type F[*Ss, T] = Callable[[*Ss], T]  # Shorthand for Callables

jvmap: Callable[[F[X, Y]], F[Xs, Ys]] = compose(jit, vmap)


@dataclass(frozen=True)
class DataPars[N: int]():
    """Dataclass containing training and test data parameter values."""
    covariate: Literal['r3', 'r4']
    """Covariate function."""

    response: Literal['cos', 'von_mises', 'von_mises_grad']
    """Response function."""

    von_mises_locs: tuple[float, float]
    """von Mises location parameters for prediction observable."""

    von_mises_concs: tuple[float, float]
    """von Mises concentration parameters for prediction observable."""

    rot_freqs: tuple[float, float]
    """Rotation frequency of the dynamics."""

    dt: float
    """Sampling interval."""

    num_pred_steps: int
    """Number of timesteps to predict."""

    num_samples: N
    """Number of training samples (after delays and finite difference)."""

    x0: tuple[float, float]
    """Initial condition."""


@dataclass(frozen=True)
class KoopmanPars[Lk: int]:
    """Dataclass containing Koopman eigendecomposition parameters."""
    res_z: float
    """Resolvent parameter"""

    res_kernel_normalization: Optional[Literal['laplace', 'fokkerplanck',
                                               'bistochastic']]
    """Kernel normalization for resolvent compactification."""

    num_eigs_galerkin: int
    """Number of kernel eigenvalue/eigenvector pairs used for Galerkin
    approximation of the generator."""

    num_eigs: Lk
    """Number of Koopman eigenfunctions to use for prediction."""

    sort_by: Literal['energy', 'frequency'] = 'frequency'
    """Koopman eigenvalue/eigenvector sorting."""


@dataclass(frozen=True)
class TrainPars[N: int, Lk: int]:
    """Dataclass containing the training parameter values."""
    data: DataPars[N]
    """Training data parameters."""

    tune: TunePars
    """Kernel tuning parameters."""

    kernel: KernelPars
    """Kernel eigendecomposition parameters."""

    koopman: KoopmanPars[Lk]
    """Koopman operator approximation parameters."""

    bw_tune: Optional[TunePars] = None
    """Tuning parameters for kernel bandwidth function."""


@dataclass(frozen=True)
class TestPars[Ntst: int]:
    """Dataclass containing test parameter values."""
    data: DataPars[Ntst]
    """Test data parameters."""

    max_batch_size: Optional[int] = None
    """Max batch size for evalation of prediction function."""


@dataclass(frozen=True)
class Pars[N: int, Lk: int, Ntst: int]:
    """Dataclass containing the parameter values used in this example."""
    train: TrainPars[N, Lk]
    test: TestPars[Ntst]


class TrainData(TypedDict):
    """TypedDict containing training data."""
    xs: Xs
    """Dynamical states."""

    ys: Ys
    """Covariates."""

    zs: Rs
    """Responses."""


class TestData(TypedDict):
    """TypedDict containing test data."""
    xs: Xs
    """Dynamical states."""

    ys: Ys
    """Covariates."""

    zs: Rs
    """Responses."""


class KoopmanEigen(TypedDict):
    """TypedDict containing spectral data of the Koopmman generator."""

    qz_evals: Cs
    """Eigenvalues of Qz operator (bounded transformation of generator)"""

    gen_evals: Cs
    """Generator eigenvalues"""

    evec_coeffs: Cls
    """Expansion coefficients of Koopman eigenvectors in kernel eigenbasis."""

    dual_evec_coeffs: Cls
    """Dual (left) generator eigenvectors."""

    engys: Rs
    """Dirichlet energies."""


class CommonPars(TypedDict):
    """Helper TypedDict to check common training/test parameter values."""
    covariate: Literal['r3', 'r4']
    response: Literal['cos', 'von_mises', 'von_mises_grad']
    von_mises_locs: tuple[float, float]
    von_mises_concs: tuple[float, float]
    rot_freqs: tuple[float, float]
    num_pred_steps: int


common_pars: CommonPars = {'covariate': "r4",
                           'response': "von_mises",
                           'von_mises_concs': (20, 20),
                           'von_mises_locs': (jnp.pi, jnp.pi),
                           'rot_freqs': (1, math.sqrt(30)),
                           'num_pred_steps': 20}
train_data_pars = DataPars(**common_pars,
                           dt=jnp.pi / math.sqrt(13),
                           x0=(jnp.pi / math.sqrt(13), jnp.pi / math.sqrt(13)),
                           num_samples=2048)
test_data_pars = DataPars(**common_pars,
                          dt=jnp.pi / math.sqrt(3113),
                          x0=(jnp.pi / math.sqrt(7), jnp.pi / math.sqrt(7)),
                          num_samples=128)
bw_tune_pars = TunePars(manifold_dim=2,
                        num_bandwidths=128,
                        log10_bandwidth_lims=(-3, 3),
                        bandwidth_scl=1)
tune_pars = TunePars(manifold_dim=2,
                     num_bandwidths=128,
                     log10_bandwidth_lims=(-3, 3),
                     bandwidth_scl=3)
kernel_pars = KernelPars(normalization=None,
                         eigensolver="eigsh",
                         num_eigs=32)
koopman_pars = KoopmanPars(res_z=0.1,
                           res_kernel_normalization='bistochastic',
                           num_eigs_galerkin=32,
                           num_eigs=17)
train_pars = TrainPars(data=train_data_pars,
                       bw_tune=bw_tune_pars,
                       tune=tune_pars,
                       kernel=kernel_pars,
                       koopman=koopman_pars)
test_pars = TestPars(data=test_data_pars)
pars = Pars(train=train_pars, test=test_pars)


@timeit
def generate_data[N: int, D: DTypeLike](pars: DataPars[N], dtype: D) \
            -> tuple[TrainData, L2VectorAlgebra[tuple[N], D, X, R], F[X, Y]]:
    """Generate rotation data on the unit circle for training."""
    match pars.covariate:
        case "r3":
            cov = torus.make_observable_r3(dtype=dtype)
        case "r4":
            cov = torus.make_observable_r4(dtype)
    match pars.response:
        case "cos":
            rsp = torus.make_observable_cos(dtype)
        case "von_mises":
            rsp = torus.make_observable_von_mises(pars.von_mises_concs,
                                                  pars.von_mises_locs, dtype)
        case "von_mises_grad":
            rsp = torus.make_observable_von_mises_grad(pars.von_mises_concs,
                                                       pars.von_mises_locs,
                                                       dtype)
    covariate = jvmap(cov)
    response = jvmap(rsp)
    rot_angles = [rot_freq * pars.dt for rot_freq in pars.rot_freqs]
    dyn_map = dyn.make_rotation_map(rot_angles)
    num_total_samples = pars.num_samples + pars.num_pred_steps
    orb = dyn.make_fin_orbit(dyn_map, num_total_samples)
    xs = orb(jnp.array(pars.x0, dtype=dtype))
    ys = covariate(xs)
    zs = response(xs)
    mu = vec.make_normalized_counting_measure(pars.num_samples)
    l2x = L2VectorAlgebra(shape=(pars.num_samples,), dtype=dtype, measure=mu,
                          inclusion_map=vec.veval_at(xs[:pars.num_samples]))
    return {'xs': xs, 'ys': ys, 'zs': zs}, l2x, cov


def jvp(f: Callable[[X], R], x: X, v: TX) -> R:
    """Convenience wrapper of jax.jvp"""
    _, df = jax.jvp(f, (x,), (v,))
    return df


@timeit
def compute_qz_matrix[N: int, Lk: int, D: DTypeLike](
        pars: KoopmanPars[Lk], l2x: L2VectorAlgebra[tuple[N], D, X, R],
        qz_i: Callable[[X, X], R], qz_j: Callable[[X, X], R],
        kernel_eigen: KernelEigen) -> M:
    """Compute matrix representation of Qz operator."""

    i_op = compose(l2x.incl, knl.make_integral_operator(l2x, qz_i))
    j_op = compose(l2x.incl, knl.make_integral_operator(l2x, qz_j))

    @jit
    @partial(vmap, in_axes=(None, None, 1, 0), out_axes=1)
    @partial(vmap, in_axes=(1, 0, None, None), out_axes=0)
    def compute_generator(phi_i: V, lamb_i: R, phi_j: V, lamb_j: R) -> R:
        qz_ij = l2x.innerp(i_op(phi_i), j_op(phi_j)) / (lamb_i * lamb_j)
        return qz_ij

    m = pars.num_eigs_galerkin
    qz_mat = compute_generator(kernel_eigen['dual_evecs'][:, :m],
                               kernel_eigen['evals'][:m],
                               kernel_eigen['evecs'][:, :m],
                               kernel_eigen['evals'][:m])
    return qz_mat


@timeit
def compute_koopman_eigen[Lk: int](pars: KoopmanPars[Lk], qz_mat: M,
                                   kernel_eigen: KernelEigen) -> KoopmanEigen:
    """Solve eigenvalue problem for regularized Koopman generator."""
    qz_evals, evec_coeffs = jla.eig((qz_mat - qz_mat.T) / 2)
    gen_evals = \
        1j * (1 + jnp.sqrt(1 - 4 * pars.res_z**2 * qz_evals.imag**2)) \
        / (2 * qz_evals.imag)
    engys = jnp.sum(jnp.abs(evec_coeffs)**2
                    / kernel_eigen['evals'][:pars.num_eigs_galerkin,
                                            jnp.newaxis], axis=0) - 1
    match pars.sort_by:
        case 'frequency':
            isort = jnp.argsort(jnp.abs(gen_evals.imag))[::-1]
        case 'energy':
            isort = jnp.argsort(engys)
    eigen: KoopmanEigen = {'qz_evals': qz_evals[isort],
                           'gen_evals': gen_evals[isort],
                           'engys': engys[isort],
                           'evec_coeffs': evec_coeffs[:, isort],
                           'dual_evec_coeffs': evec_coeffs[:, isort]}
    return eigen


def make_koopman_eigenbasis_operators[Lk: int, D: DTypeLike](
        pars: KoopmanPars[Lk], c_lk: VectorAlgebra[tuple[Lk], D],
        koopman_eigen: KoopmanEigen) -> tuple[F[V, Cl], F[Cl, V]]:
    """Make analysis and synthesis operators for Koopman eigenbasis."""
    m = pars.num_eigs
    anal = vec.make_l2_analysis_operator(
        c_lk, koopman_eigen['dual_evec_coeffs'][:, :m])
    synth = vec.make_synthesis_operator(koopman_eigen['evec_coeffs'][:, :m])
    return anal, synth


def make_timeseries_prediction_function[N: int, Lk: int](
        pars: TrainPars[N, Lk], train_data: TrainData,
        koopman_eigen: KoopmanEigen, koopman_anal: F[V, Clk],
        koopman_fn_synth: Callable[[Clk], F[X, C]]) -> F[X, Cs]:
    """Make vector-valued prediction function for time series prediction."""
    f_coeffs = koopman_anal(train_data['zs'][:pars.data.num_samples])
    m = pars.koopman.num_eigs
    iomegas = koopman_eigen['gen_evals'][:m]
    ts = pars.data.dt * jnp.arange(pars.data.num_pred_steps + 1)

    @partial(vmap, in_axes=(0, None))
    def predict(t: R, x: X) -> C:
        phases = jnp.exp(iomegas * t)
        return koopman_fn_synth(phases * f_coeffs)(x)

    return partial(predict, ts)


@timeit
def compute_skill_scores[Ntst: int](pars: DataPars[Ntst], test_data: TestData,
                                    fys_pred: Vtsts) -> tuple[Array, Array]:
    """Compute NRMSE and ACC skill scores over the prediction ensemble."""
    i1 = pars.num_pred_steps + pars.num_samples
    fxs_true = dl.hankel(test_data['zs'][:i1],
                         num_delays=pars.num_pred_steps)
    nrmse = jit(vmap(normalized_rmse, in_axes=1))(fxs_true, fys_pred)
    acc = jit(vmap(anomaly_correlation, in_axes=1))(fxs_true, fys_pred)
    return nrmse, acc


def plot_bandwidth_func[N: int, Ntst: int, D: DTypeLike](
        pars: DataPars[N], l2x: L2VectorAlgebra[tuple[N], D, X, R],
        bandwidth_func: F[X, R], train_data: TrainData,
        test_pars: Optional[DataPars[Ntst]] = None,
        l2x_tst: Optional[L2VectorAlgebra[tuple[Ntst], D, X, R]] = None,
        test_data: Optional[TestData] = None,
        num_plt: Optional[int] = None, num_plt_tst: Optional[int] = None,
        num_skip: int = 1, num_skip_tst: int = 1, i_fig: int = 1) -> Figure:
    """Plot bandwidth function on training and, optionally, test data."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    if test_pars is not None:
        fig, axs = plt.subplots(1, 2, num=i_fig,
                                figsize=tuple(mpf.figaspect(0.5)),
                                constrained_layout=True, sharey=True,
                                subplot_kw={'box_aspect': 1})
        ax, ax_tst = axs
    else:
        fig, ax = plt.subplots(num=i_fig, constrained_layout=True,
                               subplot_kw={'box_aspect': 1})
    if num_plt is None:
        num_plt = pars.num_samples
    bw_vals = l2x.incl(bandwidth_func)
    vmin = float(jnp.min(bw_vals))
    vmax = float(jnp.max(bw_vals))
    if l2x_tst is not None:
        bw_vals_tst = l2x_tst.incl(bandwidth_func)
        vmin = min(vmin, float(jnp.max(bw_vals_tst)))
        vmax = max(vmax, float(jnp.max(bw_vals_tst)))
    plt.rcParams['grid.color'] = "yellow"
    sc = ax.scatter(train_data['xs'][:num_plt:num_skip, 0] / jnp.pi,
                    train_data['xs'][:num_plt:num_skip, 1] / jnp.pi,
                    c=bw_vals[::num_skip], s=1, vmin=vmin, vmax=vmax,
                    cmap="binary")
    ax.set_xlabel(r"$\theta_1/\pi$")
    ax.set_ylabel(r"$\theta_2/\pi$")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_facecolor('orange')
    ax.set_title("Kernel bandwidth function (training)")

    if test_pars is not None and l2x_tst is not None and test_data is not None:
        if num_plt_tst is None:
            num_plt_tst = test_pars.num_samples
        sc_tst = ax_tst.scatter(
            test_data['xs'][:num_plt_tst:num_skip_tst, 0] / jnp.pi,
            test_data['xs'][:num_plt_tst:num_skip_tst, 1] / jnp.pi,
            c=bw_vals_tst[::num_skip_tst], s=1, vmin=vmin, vmax=vmax,
            cmap="binary")
        ax_tst.set_xlabel(r"$\theta_1/\pi$")
        ax_tst.set_xlim(0, 2)
        ax_tst.set_ylim(0, 2)
        ax_tst.set_title("Kernel bandwidth function (test)")
        ax_tst.set_facecolor('orange')
        fig.colorbar(sc_tst, ax=ax_tst)
    else:
        fig.colorbar(sc, ax=ax)

    plt.rcdefaults()
    return fig


def make_kernel_evecs_plotter[N: int, Ntst: int, D: DTypeLike](
        pars: DataPars[N], train_data: TrainData, kernel_eigen: KernelEigen,
        test_pars: Optional[DataPars[Ntst]] = None,
        l2x_tst: Optional[L2VectorAlgebra[tuple[Ntst], D, X, R]] = None,
        test_data: Optional[TestData] = None,
        extend: Optional[Callable[[V, R], F[X, R]]] = None,
        num_plt: Optional[int] = None, num_plt_tst: Optional[int] = None,
        num_skip: int = 1, num_skip_tst: int = 1, i_fig: int = 1) \
            -> tuple[Figure, F[int, None]]:
    """Make plotting function for kernel eigenfunctions."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    if test_pars is not None:
        fig, axs = plt.subplots(1, 2, num=i_fig,
                                figsize=tuple(mpf.figaspect(0.5)),
                                constrained_layout=True, sharey=True,
                                subplot_kw={'box_aspect': 1})
        ax, ax_tst = axs
    else:
        fig, ax = plt.subplots(num=i_fig, constrained_layout=True,
                               subplot_kw={'box_aspect': 1})
    if num_plt is None:
        num_plt = pars.num_samples
    if test_pars is not None:
        if num_plt_tst is None:
            num_plt_tst = test_pars.num_samples

    def plot_eigs(j: int):
        amax = float(jnp.max(jnp.abs(kernel_eigen['evecs'][:, j])))
        if test_pars is not None and l2x_tst is not None \
                and test_data is not None and extend is not None:
            kernel_efun = extend(kernel_eigen['evecs'][:, j],
                                 kernel_eigen['evals'][j])
            evec_tst = l2x_tst.incl(kernel_efun)
            amax = max(amax, float(jnp.abs(jnp.max(evec_tst))))

        ax.cla()
        sc = ax.scatter(train_data['xs'][:num_plt:num_skip, 0] / jnp.pi,
                        train_data['xs'][:num_plt:num_skip, 1] / jnp.pi,
                        c=kernel_eigen['evecs'][::num_skip, j], s=1,
                        vmin=-amax, vmax=amax, cmap="seismic")
        ax.set_xlabel(r"$\theta_1/\pi$")
        ax.set_ylabel(r"$\theta_2/\pi$")
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        eta = kernel_eigen['lapl_evals'][j]
        ax.set_title(f"Eigenvector {j}: $\\eta_{{{j}}} = {eta: .3f}$")

        if test_pars is not None and l2x_tst is not None \
                and test_data is not None and extend is not None:
            ax_tst.cla()
            sc_tst = ax_tst.scatter(
                test_data['xs'][:num_plt_tst:num_skip_tst, 0] / jnp.pi,
                test_data['xs'][:num_plt_tst:num_skip_tst, 1] / jnp.pi,
                c=evec_tst[::num_skip_tst], s=1, vmin=-amax, vmax=amax,
                cmap="seismic")
            ax_tst.set_xlabel(r"$\theta_1/\pi$")
            ax_tst.set_xlim(0, 2)
            ax_tst.set_ylim(0, 2)
            ax_tst.set_title("Nystrom")
            fig.colorbar(sc_tst, ax=ax_tst)
        else:
            fig.colorbar(sc, ax=ax)
    return fig, plot_eigs


def plot_generator_matrix(gen_mat: M, i_fig: int = 1) -> Figure:
    """Plot heatmap of generator matrix."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    sns.heatmap(gen_mat, ax=ax, cmap='seismic', center=0, robust=False)
    ax.set_title("Koopman generator matrix")
    return fig


def plot_generator_spectrum(koopman_eigen: KoopmanEigen,
                            num_eigs_plt: Optional[int] = None,
                            i_fig: int = 1) -> Figure:
    """Plot spectrum of Koopman generator."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    if num_eigs_plt is None:
        num_eigs_plt = len(koopman_eigen['gen_evals'])
    im = ax.scatter(koopman_eigen['engys'][:num_eigs_plt],
                    koopman_eigen['gen_evals'][:num_eigs_plt].imag, s=10,
                    c=jnp.arange(num_eigs_plt))
    cb = fig.colorbar(im, ax=ax)
    ax.set_xlabel("Dirichlet energy $E_j$")
    ax.set_ylabel(r"Eigenfrequency $\omega_j$")
    cb.set_label("$j$")
    ax.grid(True)
    return fig


def make_koopman_evecs_plotter[N: int, Ntst: int, D: DTypeLike](
        pars: DataPars[N], train_data: TrainData, koopman_eigen: KoopmanEigen,
        synth: F[Cl, V], test_pars: Optional[DataPars[Ntst]] = None,
        l2x_tst: Optional[L2VectorAlgebra[tuple[Ntst], D, X, R]] = None,
        test_data: Optional[TestData] = None,
        fn_synth: Optional[Callable[[Cl], F[X, C]]] = None,
        num_plt: Optional[int] = None, num_plt_tst: Optional[int] = None,
        num_skip: int = 1, num_skip_tst: int = 1, i_fig: int = 1) \
            -> tuple[Figure, F[int, None]]:
    """Make plotting function for Koopman eigenfunctions."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)

    if test_pars is not None:
        fig, axss = plt.subplots(2, 4, num=i_fig,
                                 figsize=tuple(1.5 * mpf.figaspect(0.6)),
                                 constrained_layout=True,
                                 subplot_kw={'box_aspect': 1})
        axs, axs_tst = axss
    else:
        fig, axs = plt.subplots(1, 4, num=i_fig,
                                figsize=tuple(mpf.figaspect(0.5)),
                                constrained_layout=True,
                                subplot_kw={'box_aspect': 1})

    if num_plt is None:
        num_plt = pars.num_samples
    ts = jnp.arange(num_plt) * pars.dt

    if test_pars is not None:
        if num_plt_tst is None:
            num_plt_tst = test_pars.num_samples
        ts_tst = jnp.arange(num_plt_tst) * test_pars.dt

    def plot_eigs(j: int):
        evec = synth(koopman_eigen['evec_coeffs'][:, j])
        evl = koopman_eigen['gen_evals'][j]
        amax = max(float(jnp.max(jnp.abs(evec.real))),
                   float(jnp.max(jnp.abs(evec.imag))))
        if test_pars is not None and l2x_tst is not None \
                and test_data is not None and fn_synth is not None:
            evec_tst \
                = l2x_tst.incl(fn_synth(koopman_eigen['evec_coeffs'][:, j]))
            amax = max(amax, float(jnp.max(jnp.abs(evec_tst.real))),
                       float(jnp.max(jnp.abs(evec_tst.imag))))

        for ax in axs:
            ax.cla()

        ax = axs[0]
        sc = ax.scatter(train_data['xs'][:num_plt:num_skip, 0] / jnp.pi,
                        train_data['xs'][:num_plt:num_skip, 1] / jnp.pi,
                        c=evec.real[::num_skip], s=1, vmin=-amax, vmax=amax,
                        cmap="seismic")
        ax.set_xlabel(r"$\theta_1/\pi$")
        ax.set_ylabel(r"$\theta_2/\pi$")
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_title(f"$\\mathrm{{Re}}\\zeta_{{{j}}}$ - training")

        ax = axs[1]
        ax.scatter(train_data['xs'][:num_plt:num_skip, 0] / jnp.pi,
                   train_data['xs'][:num_plt:num_skip, 1] / jnp.pi,
                   c=evec.imag[::num_skip], s=1, vmin=-amax, vmax=amax,
                   cmap="seismic")
        ax.set_xlabel(r"$\theta_1/\pi$")
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_title(f"$\\mathrm{{Im}}\\zeta_{{{j}}}$")

        ax = axs[2]
        ax.plot(evec.real[::num_skip], evec.imag[::num_skip], "-")
        ax.set_xlabel(f"$\\mathrm{{Re}}\\zeta_{{{j}}}$")
        ax.set_ylabel(f"$\\mathrm{{Im}}\\zeta_{{{j}}}$")
        ax.set_title(f"Frequency $\\omega_{{{j}}} = {evl.imag: .3f}$")
        ax.grid()

        ax = axs[3]
        ax.plot(ts[::num_skip], evec.real[::num_skip],
                "-", label=f"$\\mathrm{{Re}}\\zeta_{{{j}}}$")
        ax.plot(ts[::num_skip], evec.imag[::num_skip],
                "-", label=f"$\\mathrm{{Im}}\\zeta_{{{j}}}$")
        ax.set_title(f"Growth rate $\\gamma_{{{j}}} = {evl.real: .3f}$")
        ax.set_xlabel("$t$")
        ax.grid()
        ax.legend()

        if test_pars is not None and l2x_tst is not None \
                and test_data is not None and fn_synth is not None:
            evec_tst \
                = l2x_tst.incl(fn_synth(koopman_eigen['evec_coeffs'][:, j]))

            ax = axs_tst[0]
            sc_tst = ax.scatter(
                test_data['xs'][:num_plt_tst:num_skip_tst, 0] / jnp.pi,
                test_data['xs'][:num_plt_tst:num_skip_tst, 1] / jnp.pi,
                c=evec_tst.real[::num_skip_tst], s=1, vmin=-amax, vmax=amax,
                cmap="seismic")
            ax.set_xlabel(r"$\theta_1/\pi$")
            ax.set_ylabel(r"$\theta_2/\pi$")
            ax.set_title(f"$\\mathrm{{Re}}\\zeta_{{{j}}}$ - test")
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)

            ax = axs_tst[1]
            ax.scatter(test_data['xs'][:num_plt_tst:num_skip_tst, 0] / jnp.pi,
                       test_data['xs'][:num_plt_tst:num_skip_tst, 1] / jnp.pi,
                       c=evec_tst.imag[::num_skip_tst], s=1, vmin=-amax,
                       vmax=amax, cmap="seismic")
            ax.set_xlabel(r"$\theta_1/\pi$")
            ax.set_title(f"$\\mathrm{{Im}}\\zeta_{{{j}}}$")
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)

            ax = axs_tst[2]
            ax.plot(evec_tst.real[::num_skip_tst],
                    evec_tst.imag[::num_skip_tst], "-")
            ax.set_xlabel(f"$\\mathrm{{Re}}\\zeta_{{{j}}}$")
            ax.set_ylabel(f"$\\mathrm{{Im}}\\zeta_{{{j}}}$")
            ax.grid()

            ax = axs_tst[3]
            ax.plot(ts_tst[::num_skip_tst], evec_tst.real[::num_skip_tst], "-")
            ax.plot(ts_tst[::num_skip_tst], evec_tst.imag[::num_skip_tst], "-")
            ax.set_xlabel("$t$")
            ax.grid()

        if test_pars is None:
            if len(fig.axes) > 4:
                fig.colorbar(sc, ax=axs[:2], cax=fig.axes[4],
                             location='bottom', shrink=0.75, aspect=60, pad=0)
            else:
                fig.colorbar(sc, ax=axs[:2], location='bottom', shrink=0.75,
                             aspect=60, pad=0)
        else:
            if len(fig.axes) > 8:
                fig.colorbar(sc_tst, ax=axs_tst[:2], cax=fig.axes[8],
                             location='bottom', shrink=0.75, aspect=30, pad=0)
            else:
                fig.colorbar(sc_tst, ax=axs_tst[:2], location='bottom',
                             shrink=0.75, aspect=30, pad=0)
    return fig, plot_eigs


def make_pred_plotter[N: int, Ntst: int, D: DTypeLike](
        pars: DataPars[N], train_data: TrainData, pars_tst: DataPars[Ntst],
        test_data: TestData, preds: Vtsts, num_plt_tst: Optional[int] = None,
        num_skip_tst: int = 1, i_fig: int = 1) -> tuple[Figure, F[int, None]]:
    """Make plotting function for prediction over different lead times."""
    fig, axs = plt.subplots(1, 3, num=i_fig,
                            figsize=tuple(mpf.figaspect(0.3)),
                            constrained_layout=True, sharey=True,
                            subplot_kw={'box_aspect': 1})

    def plot_pred(i_step: int):
        i0_tst = i_step
        i1_tst = i0_tst + pars_tst.num_samples
        err = preds[:, i_step] - test_data['zs'][i0_tst:i1_tst]
        vmin = min(float(jnp.min(test_data['zs'][i0_tst:i1_tst])),
                   float(jnp.min(preds[:, i_step])))
        vmax = max(float(jnp.max(test_data['zs'][i0_tst:i1_tst])),
                   float(jnp.max(preds[:, i_step])))
        eabs = float(jnp.max(jnp.abs(err)))
        for ax in axs:
            ax.cla()

        ax = axs[0]
        sc_tst = ax.scatter(
            test_data['xs'][i0_tst:i1_tst:num_skip_tst, 0] / jnp.pi,
            test_data['xs'][i0_tst:i1_tst:num_skip_tst, 1] / jnp.pi,
            c=test_data['zs'][i0_tst:i1_tst:num_skip_tst], s=1,
            vmin=vmin, vmax=vmax, cmap="seismic")
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_xlabel(r"$\theta_2$")
        ax.set_title("True")

        ax = axs[1]
        ax.scatter(test_data['xs'][i0_tst:i1_tst:num_skip_tst, 0] / jnp.pi,
                   test_data['xs'][i0_tst:i1_tst:num_skip_tst, 1] / jnp.pi,
                   c=preds[::num_skip_tst, i_step], s=1, vmin=vmin, vmax=vmax,
                   cmap="seismic")
        ax.set_xlabel(r"$\theta$")
        ax.set_title(f"Prediction; lead time = {i_step} timesteps")

        ax = axs[2]
        sc_err = ax.scatter(
            test_data['xs'][i0_tst:i1_tst:num_skip_tst, 0] / jnp.pi,
            test_data['xs'][i0_tst:i1_tst:num_skip_tst, 1] / jnp.pi,
            c=err, s=1, vmin=-eabs, vmax=eabs, cmap="seismic")
        ax.set_xlabel(r"$\theta$")
        ax.set_title("Error")

        if len(axs) > 4:
            fig.colorbar(sc_tst, ax=axs[:2], cax=axs[3], location='bottom',
                         shrink=0.75, aspect=60, pad=0)
        else:
            fig.colorbar(sc_tst, ax=axs[:2], location='bottom', shrink=0.75,
                         aspect=60, pad=0)

        if len(axs) > 5:
            fig.colorbar(sc_err, ax=axs[2], cax=axs[4], location='bottom',
                         shrink=0.75, aspect=30, pad=0)
        else:
            fig.colorbar(sc_err, ax=axs[2], location='bottom', shrink=0.75,
                         aspect=30, pad=0)
    return fig, plot_pred


def make_pred_timeseries_plotter[Ntst: int](
        pars: DataPars[Ntst], test_data: TestData, preds: Vtsts,
        i_fig: int = 1) -> tuple[Figure, F[int, None]]:
    """Make plotting function over different initial conditions."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    ts = jnp.arange(pars.num_pred_steps + 1) * pars.dt

    def plot_pred(i_init: int):
        i0_tst = i_init
        i1_tst = i0_tst + pars.num_pred_steps + 1
        ax.cla()
        ax.plot(ts, test_data['zs'][i0_tst:i1_tst], "o-", label="test")
        ax.plot(ts, preds[i_init, :], "o-", label="prediction")
        ax.grid()
        ax.legend()
        ax.set_xlabel("Forecast time")
        ax.set_title(f"Initial condition = {i_init}")
    return fig, plot_pred


def plot_forecast_skill_scores(nrmse: Array, acc: Array, i_fig: int = 1) \
        -> Figure:
    """Plot NRMSE and ACC versus forecast lead time."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, axs = plt.subplots(2, 1, num=i_fig, constrained_layout=True,
                            sharex=True)
    labels = ("NRMSE", "Anomaly correlation")
    scores = (nrmse, acc)
    for ax, score, label in zip(axs, scores, labels):
        ax.plot(score, "o-")
        ax.grid()
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("Forecast timesteps")
        ax.set_ylabel(label)
    return fig


def main():
    """Diffusion-regularized Koopman spectral analysis for torus rotation."""

    # Generate training and test data
    train_data, l2x, cov = generate_data(pars.train.data, r_dtype)
    test_data, l2x_tst, _ = generate_data(pars.test.data, r_dtype)

    # Set distance function and kernel shape function
    sqdist = compose2(dst.sqeuclidean, (cov, cov))
    shape_func = jnp.exp

    # Create and tune bandwidth function
    if pars.train.bw_tune is not None:
        bw_kernel_family = knl.make_kernel_family(l2x.scl, shape_func, sqdist)
        bw_kernel, bw_bandwidth, bw_tune_info = timeit(knl.tune_bandwidth)(
            pars.train.bw_tune, l2x, bw_kernel_family)
        bandwidth_func = knl.make_bandwidth_function(
            l2x, bw_kernel, dim=jnp.asarray(bw_tune_info['dim'], r_dtype),
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

        # Plot tuning function
        if PLOT_FIGS:
            knl.plot_kernel_tuning(bw_tune_info,
                                   title="Bandwidth function tuning")
            plt.show()
            plot_bandwidth_func(pars.train.data, l2x, bandwidth_func,
                                train_data, pars.test.data, l2x_tst,
                                test_data)
            plt.show()

    # Create and tune kernel
    if pars.train.bw_tune is not None:
        scaled_sqdist = knl.make_scaled_sqdist(l2x.scl, sqdist, bandwidth_func)
        kernel_family = knl.make_kernel_family(l2x.scl, shape_func,
                                               scaled_sqdist)
    else:
        kernel_family = knl.make_kernel_family(l2x.scl, shape_func, sqdist)
    kernel, bandwidth, tune_info = timeit(knl.tune_bandwidth)(
        pars.train.tune, l2x, kernel_family)
    print("Kernel tuning:")
    print(f"Bandwidth index: {tune_info['i_opt']}")
    print(f"Optimal bandwidth: {tune_info['opt_bandwidth']:.3e}")
    print(f"Optimal dimension: {tune_info['opt_dim']:.3e}")
    print(f"Bandwidth used for diffusion maps: {tune_info['bandwidth']:.3e}")
    print("Dimension based on diffusion maps bandwidth: "
          f"{tune_info['dim']:.3e}")
    print(f"Manifold volume: {tune_info['vol']:.3e}")

    # Plot kernel tuning function
    if PLOT_FIGS:
        knl.plot_kernel_tuning(tune_info, title="Kernel tuning")
        plt.show()

    # Build dynamical vector field and kernels for resolvent compactification
    v = dyn.make_rotation_vector_field(pars.train.data.rot_freqs, r_dtype)
    match pars.train.kernel.normalization:
        case "laplace":
            normalized_kernel = knl.dm_normalize(l2x, kernel, alpha="1")
        case "fokkerplanck":
            normalized_kernel = knl.dm_normalize(l2x, kernel, alpha="0.5")
        case "bistochastic":
            normalized_kernel = knl.bs_normalize(l2x, kernel)
        case None:
            normalized_kernel = kernel
    qz_i, qz_j, zv_kernel = knl.make_resolvent_compactification_kernels(
        l2x, v, jnp.array(pars.train.koopman.res_z), normalized_kernel, jvp)

    # Solve kernel eigenvalue problem
    kernel_eigen = timeit(knl.compute_eigen)(pars.train.kernel, l2x, zv_kernel,
                                             bandwidth)
    print("First 5 heat kernel eigenvalues:")
    print(kernel_eigen['evals'][0:5])
    print("First 5 Laplacian eigenvalues:")
    print(kernel_eigen['lapl_evals'][0:5])

    # Plot spectrum of Laplace eigenvalues
    if PLOT_FIGS:
        knl.plot_laplace_spectrum(kernel_eigen)
        plt.show()

    # Build analysis, synthesis, and Nystrom operators for the kernel
    # eigenbasis
    extend = knl.make_eigenvector_extension(l2x, zv_kernel, normalization=None)
    kernel_anal, kernel_synth, kernel_fn_synth \
        = knl.make_eigenbasis_operators(l2x, kernel_eigen, extend,
                                        pars.train.koopman.num_eigs_galerkin)

    # Plot representative kernel eigenfunctions
    # if PLOT_FIGS:
    if True:
        _, plot_eigs = make_kernel_evecs_plotter(
            pars.train.data, train_data, kernel_eigen, pars.test.data, l2x_tst,
            test_data, extend)
        plot_eigs(3)
        plt.show()

    # Compute regularized generator matrix
    gen_mat = compute_qz_matrix(pars.train.koopman, l2x, qz_i, qz_j,
                                kernel_eigen)

    # Plot generator matrix
    if PLOT_FIGS:
        plot_generator_matrix(gen_mat)
        plt.show()

    # Compute generator eigendecomposition
    koopman_eigen = compute_koopman_eigen(pars.train.koopman, gen_mat,
                                          kernel_eigen)
    print("First 5 eigenvalues of Qz operator:")
    print(koopman_eigen['qz_evals'][0:5])
    print("First 5 regularized generator eigenvalues:")
    print(koopman_eigen['gen_evals'][0:5])

    # Plot generator spectrum
    if PLOT_FIGS:
        plot_generator_spectrum(koopman_eigen)
        plt.show()

    # Build analysis, synthesis, and Nystrom operators for the Koopman
    # eigenbasis
    c_lk = VectorAlgebra(shape=(pars.train.koopman.num_eigs,), dtype=c_dtype)
    koopman_c_anal, koopman_c_synth = make_koopman_eigenbasis_operators(
        pars.train.koopman, c_lk, koopman_eigen)
    koopman_anal = compose(koopman_c_anal, kernel_anal)
    koopman_fn_synth = compose(kernel_fn_synth, koopman_c_synth)

    # Plot representative Koopman eigenfunctions
    if PLOT_FIGS:
        _, plot_eigs = make_koopman_evecs_plotter(
            pars.train.data, train_data, koopman_eigen, kernel_synth,
            pars.test.data, l2x_tst, test_data, kernel_fn_synth)
        plot_eigs(1)
        plt.show()

    # Perform time series prediction
    predict = jit(make_timeseries_prediction_function(
        pars.train, train_data, koopman_eigen, koopman_anal, koopman_fn_synth))
    fys_pred = timeit(l2x_tst.incl)(predict).real
    print("Time series prediction function on first initial condition:")
    print(fys_pred[0])

    # Plot running forecast
    if PLOT_FIGS:
        _, plot_pred = make_pred_plotter(pars.train.data, train_data,
                                         pars.test.data, test_data, fys_pred)
        plot_pred(15)
        plt.show()

    # Plot time series forecast
    if PLOT_FIGS:
        _, plot_pred_ts = make_pred_timeseries_plotter(pars.test.data,
                                                       test_data, fys_pred)
        plot_pred_ts(5)
        plt.show()

    # Compute normalized RMSE
    nrmse, acc = compute_skill_scores(pars.test.data, test_data, fys_pred)
    print("Normalized RMSE:")
    print(nrmse)
    print("Anomaly correlation coefficient:")
    print(acc)

    # Plot forecast skill scores
    if PLOT_FIGS:
        plot_forecast_skill_scores(nrmse, acc)
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 2 and (sys.argv[1] == '--help' or sys.argv[1] == '-h'):
        print(__doc__)
    else:
        main()

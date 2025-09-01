# pyright: basic
"""Diffusion maps on the circle using the JAX backend of the NLSA package."""

import jax
import jax.numpy as jnp
import jax.random as jrn
import jax.experimental.sparse.linalg as jsla
import matplotlib.pyplot as plt
import nlsa.jax.distance as dst
import nlsa.jax.vector_algebra as vec
import nlsa.kernels as knl
import numpy as np
import os
import scipy.sparse.linalg as sla
import sys
from collections.abc import Callable
from functools import partial
from jax import Array, jit, vmap
from jax.typing import DTypeLike
from matplotlib.figure import Figure
from nlsa.abstract_algebra import ImplementsScalarField
from nlsa.function_algebra import compose
from nlsa.jax.stats import make_von_mises_density, normalized_rmse
from nlsa.jax.utils import make_batched, materialize_array
from nlsa.jax.vector_algebra import L2VectorAlgebra
from nlsa.utils import swap_args, timeit
from scipy.sparse.linalg import LinearOperator
from typing import Final, Literal, Optional, TypedDict

IDX_CPU: Optional[int] = None
IDX_GPU: Optional[int] = None
XLA_MEM_FRACTION: Optional[str] = None
F64 = True
VARIABLE_BANDWIDTH = True
NORMALIZATION: Literal['laplace', 'fokkerplanck', 'bistochastic'] \
    = "bistochastic"
EIGENSOLVER: Literal['eigsh', 'eigsh_mat', 'lobpcg'] = "eigsh"
PLOT_FIGS = True

if IDX_GPU is not None and XLA_MEM_FRACTION is not None:
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = XLA_MEM_FRACTION

if IDX_CPU is not None:
    jax.config.update("jax_default_device", jax.devices("cpu")[IDX_CPU])

if IDX_GPU is not None:
    jax.config.update("jax_default_device", jax.devices("gpu")[IDX_GPU])

if F64:
    jax.config.update("jax_enable_x64", True)
    r_dtype = jnp.float64
else:
    r_dtype = jnp.float32

type X = Array  # Point in state space
type Xs = Array  # Collection of points in state space
type Y = Array  # Point in data space
type Ys = Array  # Collection of points in data space
type K = Array  # Scalar
type Ks = Array  # Collection of scalars
type V = Array  # Vector in L2
type Vs = Array  # Collection of vectors in L2
type F[*Ss, T] = Callable[[*Ss], T]

jvmap: Callable[[F[X, Y]], F[Xs, Ys]] = compose(jit, vmap)


class Pars[N: int](TypedDict):
    """TypedDict containing the parameter values used in this example."""

    num_samples: N
    """Number of training samples."""

    x0_tst: float
    """Initial angle in test data."""

    num_tst_samples: int
    """Number of test samples."""

    max_tst_batch_size: Optional[int]
    """Max batch size for evalation of prediction function."""

    manifold_dim: Optional[float]
    """Manifold dimension."""

    num_bandwidths: int
    """Number of trial kernel bandwidth parameters."""

    log10_bandwidth_lims: tuple[int, int]
    """Log upper and lower limits of trial kernel bandwidth range."""

    bandwidth_scl: float
    """Scaling factor to multiply estimated optimal kernel bandwidth."""

    vb_bandwidth_scl: Optional[float]
    """Scaling factor to multiply estimated optimal bandwidth for
    variable-bandwifth kernel.
    """

    num_eigs: int
    """Number of kernel eigenvalue/eigenvector pairs to compute. Must be no
    greater than num_samples."""

    num_eigs_nyst: int
    """Number of kernel eigenvalue/eigenvector pairs used for Nystrom
    out-of-sample extension. Must be no greater than num_eigs."""

    von_mises_loc: float
    """von Mises location parameter for prediction observable."""

    von_mises_conc: float
    """von Mises concentration parameter for prediction observable."""


class TuneInfo(TypedDict):
    """TypedDict containing kernel tuning information."""

    log10_bandwidths: Array
    """Array of trial kernel bandwidths."""

    est_dims: Array
    """Array of estimated dimensions based on trial bandwidths."""

    opt_bandwidth: float
    """Optimal bandwidth from autotuning procedure."""

    opt_dim: float
    """Optimal (maximum) dimension from autotuning procedure."""

    i_opt: int
    """Index of optimal bandwidth in array of trial bandwidths."""

    bandwidth: float
    """Selected bandwidth after scaling by user-defined factor."""

    dim: float
    """Estimated dimension based on selected bandwidth."""


pars: Final[Pars[Literal[512]]] = {'num_samples': 512,
                                    'x0_tst': 0.0,
                                    'num_tst_samples': 256,
                                    'max_tst_batch_size': None,
                                    'manifold_dim': 1.0,
                                    'num_bandwidths': 128,
                                    'log10_bandwidth_lims': (-3, 3),
                                    'bandwidth_scl': 1.0,
                                    'vb_bandwidth_scl': 1.0,
                                    'num_eigs': 33,
                                    'num_eigs_nyst': 33,
                                    'von_mises_conc': 20,
                                    'von_mises_loc': jnp.pi}


def make_embedding_r2[D: DTypeLike](dtype: Optional[D]) -> Callable[[X], Y]:
    """Make embedding of the circle into R2."""
    def embed(x: X) -> Y:
        y = jnp.empty(2, dtype=dtype)
        y = y.at[0].set(jnp.cos(x))
        y = y.at[1].set(jnp.sin(x))
        return y
    return embed


@timeit
def generate_training_data[N: int, D: DTypeLike](pars: Pars[N], dtype: D)\
        -> tuple[Xs, Ys, L2VectorAlgebra[tuple[N], D, Y, K]]:
    """Generate uniformly sampled data on the unit circle for training."""
    embed = jvmap(make_embedding_r2(dtype))
    xs = jnp.linspace(0, 2*jnp.pi, pars['num_samples'], dtype=dtype,
                      endpoint=False)
    ys = embed(xs)
    mu = vec.make_normalized_counting_measure(pars['num_samples'])
    incl = vec.veval_at(ys)
    l2y = L2VectorAlgebra(shape=(pars['num_samples'],), dtype=dtype,
                          measure=mu, inclusion_map=incl)
    return xs, ys, l2y


@timeit
def generate_test_data[N: int](pars: Pars[N], dtype: DTypeLike) \
        -> tuple[Xs, Ys]:
    """Generate uniformly sampled data on the unit circle for prediction."""
    embed = jvmap(make_embedding_r2(dtype))
    xs = jnp.linspace(pars['x0_tst'], 2*jnp.pi + pars['x0_tst'],
                      pars['num_tst_samples'], dtype=dtype)
    ys = embed(xs)
    return xs, ys


def make_kernel_family(scl: ImplementsScalarField[K],
                       shape_func: Callable[[K], K],
                       sqdist: Callable[[Y, Y], K]) \
        -> Callable[[K], F[Y, Y, K]]:
    """Make bandwdith-parameterized kernel family."""
    make_shape_func: Callable[[K], F[K, K]] = partial(knl.make_bandwidth_rbf,
                                                      scl,
                                                      shape_func=shape_func)

    def kernel_family(epsilon: K) -> Callable[[Y, Y], K]:
        return compose(make_shape_func(epsilon), sqdist)
    return kernel_family


@timeit
def tune_kernel[N: int, D: DTypeLike, Y: Array](pars: Pars[N],
                                      l2y: L2VectorAlgebra[tuple[N], D, Y, K],
                                      kernel_family: Callable[[K],
                                                              F[Y, Y, K]]) \
        -> tuple[F[Y, Y, K], K, TuneInfo]:
    """Compute optimal bandwidth for Gaussian RBF kernel."""
    log10_bandwidths = jnp.linspace(pars['log10_bandwidth_lims'][0],
                                    pars['log10_bandwidth_lims'][1],
                                    pars['num_bandwidths'])
    kernel_dim = jit(knl.make_tuning_objective(l2y, kernel_family,
                                               grad=jax.grad,
                                               exp=partial(jnp.power, 10),
                                               log=jnp.log10))

    def body_fun(i: int, a: Array) -> Array:
        est_dim = kernel_dim(log10_bandwidths[i])
        return a.at[i].set(est_dim)

    est_dims = jax.lax.fori_loop(0, pars['num_bandwidths'], body_fun,
                                 jnp.empty(pars['num_bandwidths']))
    if pars['manifold_dim'] is None:
        i_opt = jnp.argmax(est_dims)
    else:
        i_opt = jnp.argmin(jnp.abs(est_dims - pars['manifold_dim']))
    log10_opt_bandwidth = log10_bandwidths[i_opt]
    opt_bandwidth = 10.0 ** log10_opt_bandwidth
    bandwidth = pars['bandwidth_scl'] * opt_bandwidth
    kernel = kernel_family(bandwidth)
    tune_info: TuneInfo = {'log10_bandwidths': log10_bandwidths,
                           'est_dims': est_dims,
                           'opt_bandwidth': float(opt_bandwidth),
                           'opt_dim': float(kernel_dim(log10_opt_bandwidth)),
                           'i_opt': int(i_opt),
                           'bandwidth': float(bandwidth),
                           'dim': float(kernel_dim(jnp.log10(bandwidth)))}
    return kernel, bandwidth, tune_info


def compute_kernel_eigs_dm[N: int, D: DTypeLike, Y: Array]\
    (pars: Pars[N], dtype: D, l2y: L2VectorAlgebra[tuple[N], D, Y, K],
     kernel: Callable[[Y, Y], K],
     normalization: Literal['laplace', 'fokkerplanck'],
     eigensolver: Literal['eigsh', 'eigsh_mat', 'lobpcg']) \
        -> tuple[Ks, Vs]:
    """Solve kernel eigenvalue problem for diffusion maps normalization."""
    match NORMALIZATION:
        case "laplace":
            s = knl.dmsym_normalize(l2y, kernel, alpha="1")
        case "fokkerplanck":
            s = knl.dmsym_normalize(l2y, kernel, alpha="0.5")
        case "bistochastic":
            s = knl.bssym_normalize(l2y, kernel)

    s_op: Callable[[V], V] = compose(l2y.incl,
                                     knl.make_integral_operator(l2y, s))
    match EIGENSOLVER:
        case "eigsh":
            a = LinearOperator(shape=(pars['num_samples'],
                                      pars['num_samples']),
                               dtype=dtype, matvec=jit(s_op))
            unsorted_evals, unsorted_evecs = sla.eigsh(a, pars['num_eigs'],
                                                       which="LA")
        case "eigsh_mat":
            a = np.asarray(materialize_array(s_op,
                                             shape=pars['num_samples']))
            unsorted_evals, unsorted_evecs = sla.eigsh(a, pars['num_eigs'],
                                                       which="LA")
        case "lobpcg":
            s_ops = vmap(s_op, in_axes=1, out_axes=1)
            v_const = jnp.ones((pars['num_samples'], 1), dtype=dtype)
            key = jrn.PRNGKey(758493)
            v_rnd = jrn.uniform(key, shape=(pars['num_samples'],
                                            pars['num_eigs'] - 1))
            v0 = jnp.concatenate((v_const, v_rnd), axis=1) \
                / jnp.sqrt(pars['num_samples'])
            unsorted_evals, unsorted_evecs, num_iters \
                = jsla.lobpcg_standard(jit(s_ops), v0, m=200)
            print(f"Number of eigensolver iterations: {num_iters}")
    return jnp.array(unsorted_evals, dtype=dtype), \
        jnp.array(unsorted_evecs, dtype=dtype)


def compute_kernel_eigs_bs[N: int, D: DTypeLike, Y: Array]\
    (pars: Pars[N], dtype: D, l2y: L2VectorAlgebra[tuple[N], D, Y, K],
     kernel: Callable[[Y, Y], K],
     eigensolver: Literal['eigsh', 'eigsh_mat', 'lobpcg']) \
        -> tuple[Ks, Vs]:
    """Solve kernel eigenvalue problem for bistochastic normalization."""
    match NORMALIZATION:
        case "laplace":
            s = knl.dmsym_normalize(l2y, kernel, alpha="1")
        case "fokkerplanck":
            s = knl.dmsym_normalize(l2y, kernel, alpha="0.5")
    match EIGENSOLVER:
        case "eigsh":
            s = knl.bs_normalize(l2y, kernel)
            s_op: Callable[[V], V] = compose(l2y.incl,
                                             knl.make_integral_operator(l2y,
                                                                        s))
            s_adj_op = compose(l2y.incl,
                               knl.make_integral_operator(l2y, swap_args(s)))
            s_ops = vmap(s_op, in_axes=1, out_axes=1)
            s_adj_ops = vmap(s_adj_op, in_axes=1, out_axes=1)
            a = LinearOperator(shape=(pars['num_samples'],
                                      pars['num_samples']),
                               dtype=dtype,
                               matvec=jit(s_op),
                               matmat=jit(s_ops),
                               rmatvec=jit(s_adj_op),
                               rmatmat=jit(s_adj_ops))
            unsorted_evecs, unsorted_singvals, _ = sla.svds(a,
                                                            pars['num_eigs'])
            unsorted_evals = unsorted_singvals**2
        case "eigsh_mat":
            s = knl.bs_normalize(l2y, kernel)
            s_op: Callable[[V], V] = compose(l2y.incl,
                                             knl.make_integral_operator(l2y,
                                                                        s))
            a = materialize_array(s_op, shape=pars['num_samples'])
            unsorted_evecs, unsorted_singvals, _ = sla.svds(a,
                                                            pars['num_eigs'])
            unsorted_evals = unsorted_singvals**2
        case "lobpcg":
            s = knl.bssym_normalize(l2y, kernel)
            s_op: Callable[[V], V] = compose(l2y.incl,
                                             knl.make_integral_operator(l2y,
                                                                        s))
            s_ops = vmap(s_op, in_axes=1, out_axes=1)
            v_const = jnp.ones((pars['num_samples'], 1), dtype=dtype)
            key = jrn.PRNGKey(758493)
            v_rnd = jrn.uniform(key, shape=(pars['num_samples'],
                                            pars['num_eigs'] - 1))
            v0 = jnp.concatenate((v_const, v_rnd), axis=1) \
                / jnp.sqrt(pars['num_samples'])
            unsorted_evals, unsorted_evecs, num_iters \
                = jsla.lobpcg_standard(jit(s_ops), v0, m=200)
            print(f"Number of eigensolver iterations: {num_iters}")
    return jnp.array(unsorted_evals, dtype=dtype), \
        jnp.array(unsorted_evecs, dtype=dtype)


@timeit
def compute_kernel_eigs[N: int, D: DTypeLike, Y: Array]\
    (pars: Pars[N], dtype: D, l2y: L2VectorAlgebra[tuple[N], D, Y, K],
     kernel: Callable[[Y, Y], K],
     normalization: Literal['laplace', 'fokkerplanck', 'bistochastic'],
     eigensolver: Literal['lobpcg', 'eigsh', 'eigsh_mat']) \
        -> tuple[Ks, Vs, Vs, V]:
    """Solve kernel eigenvalue problem for alpha=1 DM normalization."""
    match normalization:
        case "laplace" | "fokkerplanck":
            unsorted_evals, unsorted_evecs \
                = compute_kernel_eigs_dm(pars, dtype, l2y, kernel,
                                         normalization, eigensolver)
        case "bistochastic":
            unsorted_evals, unsorted_evecs \
                = compute_kernel_eigs_bs(pars, dtype, l2y, kernel, eigensolver)

    isort = jnp.argsort(unsorted_evals)[::-1]
    lambs = jnp.array(unsorted_evals[isort])
    sqrt_mus = jnp.abs(unsorted_evecs[:, isort[0]])
    if unsorted_evecs[0, isort[0]] > 0:
        phis = jnp.array(unsorted_evecs[:, isort] / sqrt_mus.reshape((-1, 1)))
        phi_duals = jnp.array(unsorted_evecs[:, isort] * pars['num_samples']
                              * sqrt_mus.reshape((-1, 1)))
    else:
        phis = jnp.array(unsorted_evecs[:, isort]
                         / (-sqrt_mus.reshape((-1, 1))))
        phi_duals = jnp.array(unsorted_evecs[:, isort] * pars['num_samples']
                              * (-sqrt_mus.reshape((-1, 1))))
    return lambs, phis, phi_duals, sqrt_mus**2


@timeit
def generate_labeled_data[N: int](pars: Pars[N], xs: Xs) -> tuple[V, F[X, K]]:
    """Generate von Mises data for supervised learning."""
    f = jax.grad(make_von_mises_density(concentration=pars['von_mises_conc'],
                                        location=pars['von_mises_loc']))
    v = jvmap(f)(xs)
    return v, f


def make_nystrom_extension[N: int, D: DTypeLike, Y: Array]\
    (pars: Pars[N], dtype: D, l2y: L2VectorAlgebra[tuple[N], D, Y, K],
     kernel: Callable[[Y, Y], K],
     normalization: Literal['laplace', 'fokkerplanck', 'bistochastic']) \
        -> Callable[[V, K], F[Y, K]]:
    """Make Nystrom extension from kernel eigenvalue/eigenvector pairs."""
    match normalization:
        case "laplace":
            p = knl.dm_normalize(l2y, kernel, alpha="1")
        case "fokkerplanck":
            p = knl.dm_normalize(l2y, kernel, alpha="0.5")
        case "bistochastic":
            p = knl.bssym_normalize(l2y, kernel)
    p_op: Callable[[V], F[Y, K]] = knl.make_integral_operator(l2y, p)

    def nyst(phi: V, lamb: K) -> F[Y, K]:
        return p_op(phi / lamb)
    return nyst


def make_nystrom_operator[N: int, D: DTypeLike, Y: Array]\
    (pars: Pars[N], dtype: D, l2y: L2VectorAlgebra[tuple[N], D, Y, K],
     kernel: Callable[[Y, Y], K],
     normalization: Literal['laplace', 'fokkerplanck', 'bistochastic'],
     lambs: Ks, phis: Vs, phi_duals: Vs) \
        -> Callable[[V], F[Y, K]]:
    """Make Nystrom out-of-sample extension operator."""
    anal = vec.make_l2_analysis_operator(l2y,
                                         phi_duals[:, :pars['num_eigs_nyst']])
    match normalization:
        case "laplace":
            p = knl.dm_normalize(l2y, kernel, alpha="1")
        case "fokkerplanck":
            p = knl.dm_normalize(l2y, kernel, alpha="0.5")
        case "bistochastic":
            p = knl.bssym_normalize(l2y, kernel)
    p_op: Callable[[V], F[Y, K]] = knl.make_integral_operator(l2y, p)

    @partial(vmap, in_axes=(-1, None))
    def vp_op(v: V, y: Y) -> K:
        return p_op(v)(y)

    basis = partial(vp_op, phis[:, :pars['num_eigs_nyst']]
                    / lambs[:pars['num_eigs_nyst']])
    synth = vec.make_fn_synthesis_operator(basis)
    nyst = compose(synth, anal)
    return nyst


def plot_kernel_tuning(tune_info: TuneInfo, title: Optional[str] = None,
                       i_fig: int = 1) -> Figure:
    """Plot kernel tuning function."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    ax.plot(tune_info['log10_bandwidths'], tune_info['est_dims'], '.-')
    eps_label = f"$\\epsilon_{{opt}} = {tune_info['opt_bandwidth']: .3e}$"
    ax.axvline(float(jnp.log10(tune_info['opt_bandwidth'])),
               color=u'#ff7f0e', label=eps_label)
    ax.grid()
    ax.legend()
    ax.set_xlabel(r"$\log_{10}(\epsilon)$")
    ax.set_ylabel("Estimated manifold dimension")
    if title is not None:
        ax.set_title(title)
    return fig


def plot_bandwidth_func[N: int, D: DTypeLike, Y: Array]\
        (l2y: L2VectorAlgebra[tuple[N], D, Y, K], xs: Xs, xs_tst: Xs,
         ys_tst: Ys, bandwidth_func: F[Ys, K], i_fig: int = 1) -> Figure:
    """Plot kernel bandwidth function on training and test data."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    ax.plot(xs / jnp.pi, l2y.incl(bandwidth_func), "o", label="training")
    ax.plot(xs_tst / jnp.pi, jvmap(bandwidth_func)(ys_tst), "-", label="test")
    ax.grid()
    ax.legend()
    ax.set_xlabel(r"$\theta$")
    ax.set_title("Kernel bandwidth function")
    return fig


def plot_laplace_spectrum(etas: Ks, num_eigs_plt: Optional[int] = None,
                          i_fig: int = 1) -> Figure:
    """Plot spectrum of Laplacian eigenvalues"""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    if num_eigs_plt is None:
        num_eigs_plt = len(etas)
    ax.plot(jnp.arange(num_eigs_plt), etas, '.')
    ax.grid()
    ax.set_xlabel("$j$")
    ax.set_ylabel(r"$\eta_j$")
    ax.set_title("Laplacian eigenvalues")
    return fig


def make_eigs_plotter(xs: Xs, phis: Vs, etas: Ks, xs_tst: Xs,
                      varphis: list[F[Xs, Ks]], i_fig: int = 1) \
        -> tuple[Figure, Callable[[int], None]]:
    """Make plotting function for a list of kernel eigenfunctions."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)

    def plot_eigs(j: int):
        ax.cla()
        ax.plot(xs / jnp.pi, phis[:, j], "o", label=f"$\\phi_{{{j}}}$")
        ax.plot(xs_tst / jnp.pi, varphis[j](xs_tst), "-",
                label="Nystrom")
        ax.grid()
        ax.legend()
        ax.set_xlabel(r"$\theta$")
        ax.set_title(f"Eigenvector {j}: $\\eta_{{{j}}} = {etas[j]: .3f}$")
        ax.set_xlim(0.0, 2.0)
    return fig, plot_eigs


def plot_pred(xs: Xs, fxs: V, xs_tst: Xs, fxs_true: Ks, fys_pred: Ks,
              i_fig: int = 1) -> Figure:
    """Plot training data, true function values, and predictions."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    ax.plot(xs / jnp.pi, fxs, "o", label="training")
    ax.plot(xs_tst / jnp.pi, fxs_true, "-", label="true")
    ax.plot(xs_tst / jnp.pi, fys_pred, "-", label="prediction")
    ax.grid()
    ax.legend()
    ax.set_xlabel(r"$\theta$")
    return fig


def main():
    """Perform diffusion maps on uniformly sampled data on circle."""
    # Generate training data
    xs, ys, l2y = generate_training_data(pars, r_dtype)

    # Generate test data
    xs_tst, ys_tst = generate_test_data(pars, r_dtype)

    # Tune fixed-bandwidth kernel
    sqdist = dst.sqeuclidean
    shape_func = jnp.exp
    kernel_family = make_kernel_family(l2y.scl, shape_func, sqdist)
    kernel, bandwidth, tune_info = tune_kernel(pars, l2y, kernel_family)
    dim = jnp.asarray(tune_info['opt_dim'], dtype=r_dtype)
    vol = knl.riemannian_vol(l2y,
                             kernel=knl.dm_normalize(l2y, kernel, alpha="1"),
                             dim=dim,
                             t_heat=jnp.asarray(bandwidth**2/4, dtype=r_dtype),
                             fourpi=jnp.asarray(4*jnp.pi, dtype=r_dtype))
    print(f"Optimal bandwidth index: {tune_info['i_opt']}")
    print(f"Optimal bandwidth: {tune_info['opt_bandwidth']:.3e}")
    print(f"Optimal dimension: {dim:.3e}")
    print(f"Bandwidth used for diffusion maps: {tune_info['bandwidth']:.3e}")
    print("Dimension based on diffusion maps bandwidth: "
          f"{tune_info['dim']:.3e}")
    print(f"Manifold volume: {vol:.3e}")

    # Plot tuning function
    if PLOT_FIGS:
        plot_kernel_tuning(tune_info, title="Kernel tuning")
        plt.show()

    # Tune variable-bandwidth kernel
    if VARIABLE_BANDWIDTH:
        bandwidth_func = knl.make_bandwidth_function(l2y, kernel, dim, vol)
        scaled_sqdist = knl.make_scaled_sqdist(l2y.scl, sqdist, bandwidth_func)
        vb_kernel_family = make_kernel_family(l2y.scl, shape_func,
                                              scaled_sqdist)
        vb_kernel, vb_bandwidth, vb_tune_info = tune_kernel(pars, l2y,
                                                            vb_kernel_family)
        vb_dim = vb_tune_info['opt_dim']
        vb_vol = knl.riemannian_vol(l2y,
                                    kernel=knl.dm_normalize(l2y, vb_kernel,
                                                            alpha="1"),
                                    dim=dim,
                                    t_heat=jnp.asarray(vb_bandwidth**2/4,
                                                       dtype=r_dtype),
                                    fourpi=jnp.asarray(4*jnp.pi,
                                                       dtype=r_dtype))
        print(f"VB optimal bandwidth index: {vb_tune_info['i_opt']}")
        print(f"VB optimal bandwidth: {vb_tune_info['opt_bandwidth']:.3e}")
        print(f"VB optimal dimension: {vb_dim:.3e}")
        print("VB bandwidth used for "
              f"diffusion maps: {vb_tune_info['bandwidth']:.3e}")
        print("Dimension based on VB diffusion maps bandwidth: "
              f"{vb_tune_info['dim']:.3e}")
        print(f"VB manifold volume: {vb_vol:.3e}")

        # Plot variable-bandwidth tuning function
        if PLOT_FIGS:
            plot_kernel_tuning(vb_tune_info,
                               title="Variable-bandwidth kernel tuning")
            plt.show()
            plot_bandwidth_func(l2y, xs, xs_tst, ys_tst, bandwidth_func)
            plt.show()

    # Solve kernel eigenvalue problem
    if VARIABLE_BANDWIDTH:
        eigs_kernel = vb_kernel
        eigs_bandwidth = vb_bandwidth
    else:
        eigs_kernel = kernel
        eigs_bandwidth = bandwidth
    lambs, phis, phi_duals, _ \
        = compute_kernel_eigs(pars, r_dtype, l2y, eigs_kernel, NORMALIZATION,
                              EIGENSOLVER)
    etas = -4 * jnp.log(lambs) / eigs_bandwidth**2
    # etas = jnp.array(4.0 * (1.0 - lambs) / eigs_bandwidth**2)
    print("First 5 heat kernel eigenvalues:")
    print(lambs[0:5])
    print("First 5 Laplacian eigenvalues:")
    print(etas[0:5])

    # Plot spectrum of Laplace eigenvalues
    if PLOT_FIGS:
        plot_laplace_spectrum(etas)
        plt.show()

    # Perform Nystrom extension of the kernel eigenfunctions
    extend = make_nystrom_extension(pars, r_dtype, l2y, kernel, NORMALIZATION)
    if pars['max_tst_batch_size'] is not None:
        varphis = [make_batched(jvmap(compose(extend(phi, lamb),
                                              make_embedding_r2(r_dtype))),
                                max_batch_size=pars['max_tst_batch_size'])
                   for phi, lamb in zip(phis.T, lambs)]
    else:
        varphis = [jvmap(compose(extend(phi, lamb),
                                 make_embedding_r2(r_dtype)))
                   for phi, lamb in zip(phis.T, lambs)]
    print("Eigenfunction 1 on first 5 test datapoints:")
    print(varphis[1](xs_tst[:5]))

    # Plot representative eigenfunctions
    if PLOT_FIGS:
        _, plot_eigs = make_eigs_plotter(xs, phis, etas, xs_tst, varphis)
        plot_eigs(3)
        plt.show()

    # Generate verification data
    fxs, f_true = generate_labeled_data(pars, xs)
    f_trues = jvmap(f_true)
    fxs_true = timeit(f_trues)(xs_tst)
    print("True target function on first 5 test datapoints:")
    print(fxs_true[:5])

    # Perform prediction
    nyst = make_nystrom_operator(pars, r_dtype, l2y, kernel, NORMALIZATION,
                                 lambs, phis, phi_duals)
    f_pred = nyst(fxs)
    f_preds = jvmap(f_pred)
    fys_pred = timeit(f_preds)(ys_tst)
    print("Prediction function on first 5 training datapoints:")
    print(fys_pred[:5])

    # Compute normalized RMSE
    nrmse = timeit(normalized_rmse)(fxs_true, fys_pred)
    print(f"Normalized RMSE: {nrmse: .3f}")

    # Plot true and predicted functions
    if PLOT_FIGS:
        _ = plot_pred(xs, fxs, xs_tst, fxs_true, fys_pred)
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 2 and (sys.argv[1] == '--help' or sys.argv[1] == '-h'):
        print(__doc__)
    else:
        main()

# pyright: basic
"""Provide functions for kernel computations in JAX."""

import jax
import jax.experimental.sparse.linalg as jsla
import jax.numpy as jnp
import jax.random as jrn
import matplotlib.pyplot as plt
import nlsa.kernels as knl
import nlsa.jax.vector_algebra as vec
import nlsa.function_algebra as fun
import numpy as np
import scipy.sparse.linalg as sla
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from jax import Array, jit, vmap
from jax.typing import ArrayLike, DTypeLike
from matplotlib.figure import Figure
from nlsa.jax.utils import make_batched, materialize_array
from nlsa.jax.vector_algebra import L2VectorAlgebra
from nlsa.kernels import KernelEigenbasis
from nlsa.utils import swap_args
from scipy.sparse.linalg import LinearOperator
from typing import Literal, Optional, TypedDict

type R = Array  # Real number
type Rl = Array  # l-dimensional real vectors
type Rs = Array  # Collection of real numbers
type K = Array  # Scalar
type Ks = Array  # Collection of scalars
type V = Array  # Vector
type Vs = Array  # Collection of vectors
type Shape = tuple[int, ...]
type F[*Ss, T] = Callable[[*Ss], T]  # Shorthand for Callables


@dataclass(frozen=True)
class TunePars:
    """Dataclass containing kernel parameter tuning values."""

    num_bandwidths: int = 128
    """Number of trial kernel bandwidth parameters."""

    log10_bandwidth_lims: tuple[int, int] = (-3, 3)
    """Log upper and lower limits of trial kernel bandwidth range."""

    manifold_dim: Optional[float] = None
    """Manifold dimension."""

    bandwidth_scl: float = 1
    """Scaling factor to multiply estimated optimal kernel bandwidth."""

    def __str__(self) -> str:
        loglimstr = '_'.join((f'loglim{self.log10_bandwidth_lims[0]}',
                              f'loglim{self.log10_bandwidth_lims[1]}'))
        return '_'.join((f'nb{self.num_bandwidths}',
                         loglimstr,
                         f'dim{self.manifold_dim}',
                         f'scl{self.bandwidth_scl}'))


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

    vol: float
    """Estimated manifold volume based on selected bandwidth."""


@dataclass(frozen=True)
class KernelPars:
    """Dataclass containing kernel eigendecomposition parameters."""

    normalization: Optional[Literal['laplace', 'fokkerplanck', 'bistochastic']]
    """Kernel normalization method."""

    eigensolver: Literal['eigsh', 'eigsh_mat', 'lobpcg']
    """Eigensolver used for kernel eigendecomposition."""

    num_eigs: int
    """Number of kernel eigenvalue/eigenvector pairs to compute."""

    batch_size: Optional[int] = None
    """Maximum batch size for matrix-matrix products."""

    def __str__(self) -> str:
        if self.normalization is None:
            normstr = 'nonorm'
        else:
            normstr = self.normalization
        return '_'.join((normstr,
                         self.eigensolver,
                         f'neigs{self.num_eigs}'))


class KernelEigen(TypedDict):
    """TypedDict containing kernel spectral data."""

    evals: Rs
    """Kernel eigenvalues."""

    evecs: Vs
    """Kernel eigenvectors."""

    dual_evecs: Vs
    """Dual (left) kernel eigenvectors."""

    weights: V
    """Inner product weights that orthonormalize the eigenvectors."""

    bandwidth: R
    """Bandwidth parameter."""


def tune_bandwidth[Ns: Shape, D: DTypeLike, X: Array](
        pars: TunePars, l2x: L2VectorAlgebra[Ns, D, X, R],
        kernel_family: Callable[[R], F[X, X, R]]) -> tuple[R, TuneInfo]:
    """Compute optimal bandwidth for RBF kernel family."""
    log10_bandwidths = jnp.linspace(pars.log10_bandwidth_lims[0],
                                    pars.log10_bandwidth_lims[1],
                                    pars.num_bandwidths)
    kernel_dim = jit(knl.make_tuning_objective(l2x, kernel_family,
                                               grad=jax.grad,
                                               exp=partial(jnp.power, 10),
                                               log=jnp.log10))

    def body_fun(i: int, a: Array) -> Array:
        est_dim = kernel_dim(log10_bandwidths[i])
        return a.at[i].set(est_dim)

    est_dims = jax.lax.fori_loop(0, pars.num_bandwidths, body_fun,
                                 jnp.empty(pars.num_bandwidths))
    if pars.manifold_dim is None:
        i_opt = jnp.argmax(est_dims)
    else:
        i_opt = jnp.argmin(jnp.abs(est_dims - pars.manifold_dim))
    log10_opt_bandwidth = log10_bandwidths[i_opt]
    opt_bandwidth = 10 ** log10_opt_bandwidth
    opt_dim = kernel_dim(log10_opt_bandwidth)
    bandwidth = pars.bandwidth_scl * opt_bandwidth
    dim = kernel_dim(jnp.log10(bandwidth))
    kernel = kernel_family(bandwidth)
    riemannian_vol = partial(knl.riemannian_vol, l2x,
                             kernel=knl.dm_normalize(l2x, kernel, alpha="1"),
                             dim=dim,
                             fourpi=jnp.asarray(4*jnp.pi, dtype=l2x.dtype))
    vol = jit(riemannian_vol)(t_heat=jnp.asarray(bandwidth**2/4,
                                                 dtype=l2x.dtype))
    # vol = knl.riemannian_vol(l2x,
    #                          kernel=knl.dm_normalize(l2x, kernel, alpha="1"),
    #                          dim=dim, t_heat=jnp.asarray(bandwidth**2/4,
    #                                                      dtype=l2x.dtype),
    #                          fourpi=jnp.asarray(4*jnp.pi, dtype=l2x.dtype))
    tune_info: TuneInfo = {'log10_bandwidths': log10_bandwidths,
                           'est_dims': est_dims,
                           'opt_bandwidth': float(opt_bandwidth),
                           'opt_dim': float(opt_dim),
                           'i_opt': int(i_opt),
                           'bandwidth': float(bandwidth),
                           'dim': float(dim),
                           'vol': float(vol)}
    return bandwidth, tune_info


def compute_eigen_dm[Ns: Shape, D: DTypeLike, X: Array](
        l2x: L2VectorAlgebra[Ns, D, X, R], kernel: Callable[[X, X], R],
        normalization: Optional[Literal['laplace', 'fokkerplanck']],
        num_eigs: int, eigensolver: Literal['eigsh', 'eigsh_mat', 'lobpcg']) \
            -> tuple[Rs, Vs, Vs, V]:
    """Solve kernel eigenvalue problem for diffusion maps normalization."""
    num_samples = l2x.shape[0]
    match normalization:
        case "laplace":
            s = knl.dmsym_normalize(l2x, kernel, alpha="1")
        case "fokkerplanck":
            s = knl.dmsym_normalize(l2x, kernel, alpha="0.5")
        case None:
            s = kernel
    s_op = fun.compose(l2x.incl, knl.make_integral_operator(l2x, s))
    match eigensolver:
        case "eigsh":
            a = LinearOperator(shape=(num_samples, num_samples),
                               dtype=l2x.dtype, matvec=jit(s_op))
            unsorted_evals, unsorted_evecs = sla.eigsh(a, num_eigs, which="LA")
        case "eigsh_mat":
            a = np.asarray(materialize_array(s_op, shape=num_samples))
            unsorted_evals, unsorted_evecs = sla.eigsh(a, num_eigs, which="LA")
        case "lobpcg":
            s_ops = vmap(s_op, in_axes=1, out_axes=1)
            v_const = jnp.ones((num_samples, 1), dtype=l2x.dtype)
            key = jrn.PRNGKey(758493)
            v_rnd = jrn.uniform(key, shape=(num_samples, num_eigs - 1))
            v0 = jnp.concatenate((v_const, v_rnd), axis=1) \
                / jnp.sqrt(num_samples)
            unsorted_evals, unsorted_evecs, num_iters \
                = jsla.lobpcg_standard(jit(s_ops), v0, m=200)
            print(f"Number of eigensolver iterations: {num_iters}")
    isort = jnp.argsort(unsorted_evals)[::-1]
    lambs = jnp.array(unsorted_evals[isort])
    sqrt_mus = jnp.abs(unsorted_evecs[:, isort[0]])
    norm = jit(vmap(l2x.norm, in_axes=1))
    if unsorted_evecs[0, isort[0]] > 0:
        phis = jnp.array(unsorted_evecs[:, isort] / sqrt_mus.reshape((-1, 1)))
        phi_duals = jnp.array(unsorted_evecs[:, isort] * num_samples
                              * sqrt_mus.reshape((-1, 1)))
    else:
        phis = jnp.array(unsorted_evecs[:, isort]
                         / (-sqrt_mus.reshape((-1, 1))))
        phi_duals = jnp.array(unsorted_evecs[:, isort] * num_samples
                              * (-sqrt_mus.reshape((-1, 1))))
    phi_norms = norm(phis)
    return jnp.array(lambs, dtype=l2x.dtype), \
        jnp.array(phis / phi_norms, dtype=l2x.dtype), \
        jnp.array(phi_duals * phi_norms, dtype=l2x.dtype), \
        jnp.array(sqrt_mus ** 2, dtype=l2x.dtype)


def compute_eigen_bs[Ns: Shape, D: DTypeLike, X: Array](
        l2x: L2VectorAlgebra[Ns, D, X, R], kernel: Callable[[X, X], R],
        num_eigs: int, eigensolver: Literal['eigsh', 'eigsh_mat', 'lobpcg'],
        batch_size: Optional[int] = None) -> tuple[Rs, Vs, Vs, V]:
    """Solve kernel eigenvalue problem for bistochastic normalization."""
    num_samples = l2x.shape[0]
    match eigensolver:
        case "eigsh":
            s = knl.bs_normalize(l2x, kernel)
            s_op = fun.compose(l2x.incl, knl.make_integral_operator(l2x, s))
            s_adj_op = fun.compose(l2x.incl,
                                   knl.make_integral_operator(l2x,
                                                              swap_args(s)))
            s_ops = vmap(s_op, in_axes=1, out_axes=1)
            s_adj_ops = vmap(s_adj_op, in_axes=1, out_axes=1)
            if batch_size is not None:
                make_matmat = fun.compose(partial(make_batched,
                                                  max_batch_size=batch_size,
                                                  in_axis=1),
                                          jit)
            else:
                make_matmat = jit
            a = LinearOperator(shape=(num_samples, num_samples),
                               dtype=l2x.dtype,
                               matvec=jit(s_op),
                               matmat=make_matmat(s_ops),
                               rmatvec=jit(s_adj_op),
                               rmatmat=make_matmat(s_adj_ops))
            unsorted_evecs, unsorted_singvals, unsorted_dual_evecs \
                = sla.svds(a, num_eigs)
            assert unsorted_evecs is not None
            assert unsorted_dual_evecs is not None
            unsorted_evals = unsorted_singvals**2
            unsorted_dual_evecs = unsorted_dual_evecs.T
        case "eigsh_mat":
            s = knl.bs_normalize(l2x, kernel)
            s_op = fun.compose(l2x.incl, knl.make_integral_operator(l2x, s))
            a = np.asarray(materialize_array(s_op, shape=num_samples))
            unsorted_evecs, unsorted_singvals, unsorted_dual_evecs \
                = sla.svds(a, num_eigs)
            assert unsorted_evecs is not None
            assert unsorted_dual_evecs is not None
            unsorted_evals = unsorted_singvals**2
            unsorted_dual_evecs = unsorted_dual_evecs.T
        case "lobpcg":
            s = knl.bssym_normalize(l2x, kernel)
            s_op = fun.compose(l2x.incl, knl.make_integral_operator(l2x, s))
            s_ops = vmap(s_op, in_axes=1, out_axes=1)
            v_const = jnp.ones((num_samples, 1), dtype=l2x.dtype)
            key = jrn.PRNGKey(758493)
            v_rnd = jrn.uniform(key, shape=(num_samples, num_eigs - 1))
            v0 = jnp.concatenate((v_const, v_rnd), axis=1) \
                / jnp.sqrt(num_samples)
            unsorted_evals, unsorted_evecs, num_iters \
                = jsla.lobpcg_standard(jit(s_ops), v0, m=200)
            assert unsorted_evecs is not None
            print(f"Number of eigensolver iterations: {num_iters}")
            k = knl.bs_normalize(l2x, kernel)
            k_adj_op = fun.compose(l2x.incl,
                                   knl.make_integral_operator(l2x,
                                                              swap_args(k)))
            k_adj_ops = vmap(k_adj_op, in_axes=1, out_axes=1)
            unsorted_dual_evecs = k_adj_ops(unsorted_evecs) \
                / jnp.sqrt(unsorted_evals)
    isort = jnp.argsort(unsorted_evals)[::-1]
    lambs = jnp.array(unsorted_evals[isort])
    sqrt_mus = jnp.abs(unsorted_evecs[:, isort[0]])
    norm = jit(vmap(l2x.norm, in_axes=1))
    if unsorted_evecs[0, isort[0]] > 0:
        phis = jnp.array(unsorted_evecs[:, isort] / sqrt_mus.reshape((-1, 1)))
        phi_duals = jnp.array(unsorted_dual_evecs[:, isort] * num_samples
                              * sqrt_mus.reshape((-1, 1)))
    else:
        phis = jnp.array(unsorted_evecs[:, isort]
                         / (-sqrt_mus.reshape((-1, 1))))
        phi_duals = jnp.array(unsorted_dual_evecs[:, isort] * num_samples
                              * (-sqrt_mus.reshape((-1, 1))))
    phi_norms = norm(phis)
    return jnp.array(lambs, dtype=l2x.dtype), \
        jnp.array(phis / phi_norms, dtype=l2x.dtype), \
        jnp.array(phi_duals * phi_norms, dtype=l2x.dtype), \
        jnp.array(sqrt_mus ** 2, dtype=l2x.dtype)


def to_laplace_eigenvalues(lambs: Array, bandwidth: ArrayLike,
                           method: Literal['lin', 'log', 'inv'] = "log") \
         -> Array:
    """Compute Laplace eigenvalues from kernel eigenvalues."""
    match method:
        case "lin":
            etas = 4 * (1 - lambs) / bandwidth**2
        case "log":
            etas = -4 * jnp.log(lambs) / bandwidth**2
        case "inv":
            inv_lambs = 1 / lambs
            etas = (inv_lambs - 1) / (inv_lambs[1] - 1)
    return etas


def compute_eigen[D: DTypeLike, X: Array](
        pars: KernelPars, l2x: L2VectorAlgebra[Shape, D, X, R],
        kernel: Callable[[X, X], R], bandwidth: R) -> KernelEigen:
    """Solve kernel eigenvalue problem."""
    match pars.normalization:
        case "laplace" | "fokkerplanck" | None:
            lambs, phis, phi_duals, weights \
                = compute_eigen_dm(l2x, kernel, pars.normalization,
                                   pars.num_eigs, pars.eigensolver)
        case "bistochastic":
            lambs, phis, phi_duals, weights \
                = compute_eigen_bs(l2x, kernel, pars.num_eigs,
                                   pars.eigensolver, pars.batch_size)
    eigen: KernelEigen = {"evals": lambs,
                          "evecs": phis,
                          "dual_evecs": phi_duals,
                          "weights": weights,
                          "bandwidth": bandwidth}
    return eigen


def make_eigenvector_extension_dm[Ns: tuple[int, ...], D: DTypeLike, X: Array](
        l2x: L2VectorAlgebra[Ns, D, X, R], kernel: Callable[[X, X], R],
        normalization: Optional[Literal['laplace', 'fokkerplanck']]) \
            -> tuple[Callable[[V, R], F[X, R]], Callable[[X, X], R]]:
    """Make Nystrom extension for diffusion maps kernels."""
    match normalization:
        case "laplace":
            extension_kernel = knl.dm_normalize(l2x, kernel, alpha="1")
        case "fokkerplanck":
            extension_kernel = knl.dm_normalize(l2x, kernel, alpha="0.5")
        case None:
            extension_kernel = kernel
    extension_kernel_op: Callable[[V], F[X, R]] \
        = knl.make_integral_operator(l2x, extension_kernel)

    def nyst(phi: V, lamb: R) -> F[X, R]:
        return extension_kernel_op(phi / lamb)
    return nyst, extension_kernel


def make_eigenvector_extension_bs[Ns: tuple[int, ...], D: DTypeLike, X: Array](
        l2x: L2VectorAlgebra[Ns, D, X, R], kernel: Callable[[X, X], R]) \
            -> tuple[Callable[[V, R], F[X, R]], Callable[[X, X], R]]:
    """Make Nystrom extension for bistochastic kernels."""
    extension_kernel = knl.bs_normalize(l2x, kernel)
    extension_kernel_op: Callable[[V], F[X, R]] \
        = knl.make_integral_operator(l2x, extension_kernel)

    def nyst(phi: V, lamb: R) -> F[X, R]:
        return extension_kernel_op(phi / jnp.sqrt(lamb))
    return nyst, extension_kernel


def make_eigenvector_extension[Ns: tuple[int, ...], D: DTypeLike, X: Array](
        pars: KernelPars, l2x: L2VectorAlgebra[Ns, D, X, R],
        kernel: Callable[[X, X], R]) -> tuple[Callable[[V, R], F[X, R]],
                                              Callable[[X, X], R]]:
    """Make Nystrom extension for diffusion maps and bistochastic kernels."""
    match pars.normalization:
        case "laplace" | "fokkerplanck" | None:
            nyst, extension_kernel \
                    = make_eigenvector_extension_dm(l2x, kernel,
                                                    pars.normalization)
        case "bistochastic":
            nyst, extension_kernel \
                    = make_eigenvector_extension_bs(l2x, kernel)
    return nyst, extension_kernel


def make_eigenbasis_operators_dm[N: int, D: DTypeLike, X: Array](
        l2x: L2VectorAlgebra[tuple[N], D, X, R],
        extend: Callable[[V, R], F[X, R]],
        kernel_eigen: KernelEigen, num_eigs: int) \
            -> tuple[F[V, Rl], F[Rl, V], Callable[[V], F[X, R]]]:
    """Make analysis and synthesis operators for diffusion maps eigenbasis."""
    anal = vec.make_l2_analysis_operator(
        l2x, kernel_eigen['dual_evecs'][:, :num_eigs])
    synth = vec.make_synthesis_operator(kernel_eigen['evecs'][:, :num_eigs])

    @partial(vmap, in_axes=(-1, 0, None))
    def extend_eval(v: V, lamb: R, x: X) -> R:
        return extend(v, lamb)(x)

    basis = partial(extend_eval, kernel_eigen['evecs'][:, :num_eigs],
                    kernel_eigen['evals'][:num_eigs])
    fn_synth = vec.make_fn_synthesis_operator(basis)
    return anal, synth, fn_synth


def make_eigenbasis_operators_bs[N: int, D: DTypeLike, X: Array](
        l2x: L2VectorAlgebra[tuple[N], D, X, R],
        extend: Callable[[V, R], F[X, R]],
        kernel_eigen: KernelEigen, num_eigs: int) \
            -> tuple[F[V, Rl], F[Rl, V], Callable[[V], F[X, R]]]:
    """Make analysis and synthesis operators for bistochastic eigenbasis."""
    anal = vec.make_l2_analysis_operator(
        l2x, kernel_eigen['evecs'][:, :num_eigs])
    synth = vec.make_synthesis_operator(kernel_eigen['evecs'][:, :num_eigs])

    @partial(vmap, in_axes=(-1, 0, None))
    def extend_eval(v: V, lamb: R, x: X) -> R:
        return extend(v, lamb)(x)

    basis = partial(extend_eval, kernel_eigen['dual_evecs'][:, :num_eigs],
                    kernel_eigen['evals'][:num_eigs])
    fn_synth = vec.make_fn_synthesis_operator(basis)
    return anal, synth, fn_synth


def make_eigenbasis_operators[N: int, D: DTypeLike, X: Array](
        pars: KernelPars, l2x: L2VectorAlgebra[tuple[N], D, X, R],
        kernel: Callable[[X, X], R], kernel_eigen: KernelEigen,
        num_eigs: int) -> tuple[F[V, Rl], F[Rl, V], Callable[[V], F[X, R]]]:
    """Make analysis and synthesis operators."""
    match pars.normalization:
        case "laplace" | "fokkerplanck" | None:
            extend, _ = make_eigenvector_extension_dm(l2x, kernel,
                                                      pars.normalization)
            anal, synth, fn_synth \
                = make_eigenbasis_operators_dm(l2x, extend, kernel_eigen,
                                               num_eigs)
        case "bistochastic":
            extend, _ = make_eigenvector_extension_bs(l2x, kernel)
            anal, synth, fn_synth \
                = make_eigenbasis_operators_bs(l2x, extend, kernel_eigen,
                                               num_eigs)
    return anal, synth, fn_synth


def make_eigenbasis_dm[Ns: Shape, D: DTypeLike, X: Array](
        l2x: L2VectorAlgebra[Ns, D, X, R], kernel: Callable[[X, X], R],
        normalization: Optional[Literal['laplace', 'fokkerplanck']],
        laplace_method: Literal['lin', 'log', 'inv'],
        kernel_eigen: KernelEigen,
        which_eigs: int | tuple[int, int] | list[int]) \
            -> KernelEigenbasis[X, R, V, Rs, int | Array]:
    """Make kernel eigenbasis for diffusion maps kernels."""
    match normalization:
        case "laplace":
            extension_kernel = knl.dm_normalize(l2x, kernel, alpha="1")
        case "fokkerplanck":
            extension_kernel = knl.dm_normalize(l2x, kernel, alpha="0.5")
        case None:
            extension_kernel = kernel
    match which_eigs:
        case int():
            idxs = jnp.arange(which_eigs)
        case tuple(ints) if all(isinstance(i, int) for i in ints) \
                and len(ints) == 2:
            idxs = jnp.arange(which_eigs[0], which_eigs[1])
        case _:
            idxs = jnp.array(which_eigs)
    lapl_spec = to_laplace_eigenvalues(kernel_eigen['evals'][idxs],
                                       kernel_eigen['bandwidth'],
                                       method=laplace_method)
    extension_op = knl.make_integral_operator(l2x, extension_kernel)
    dual_extension_op = knl.make_integral_operator(l2x,
                                                   swap_args(extension_kernel))

    def vc(i: int | Array) -> V:
        return kernel_eigen['evecs'][:, idxs[i]]

    def dual_vc(i: int | Array) -> V:
        return kernel_eigen['dual_evecs'][:, idxs[i]]

    def evl(i: int | Array) -> K:
        return kernel_eigen['evals'][idxs[i]]

    def lapl_evl(i: int | Array) -> K:
        return lapl_spec[idxs[i]]

    def fn(i: int | Array) -> F[X, K]:
        return extension_op(kernel_eigen['evecs'][:, idxs[i]]
                            / kernel_eigen['evals'][idxs[i]])

    def dual_fn(i: int | Array) -> Callable[[X], K]:
        return dual_extension_op(kernel_eigen['dual_evecs'][:, idxs[i]]
                                 / kernel_eigen['evals'][idxs[i]])

    @partial(vmap, in_axes=(0, None))
    def anal_eval(i: int | Array, v: V) -> K:
        return l2x.innerp(kernel_eigen['dual_evecs'][:, idxs[i]], v)

    @partial(vmap, in_axes=(0, None))
    def dual_anal_eval(i: int | Array, v: V) -> K:
        return l2x.innerp(kernel_eigen['evecs'][:, idxs[i]], v)

    @partial(vmap, in_axes=(0, None))
    def fn_eval(i: int | Array, x: X) -> R:
        return fn(i)(x)

    @partial(vmap, in_axes=(0, None))
    def dual_fn_eval(i: int | Array, x: X) -> R:
        return dual_fn(i)(x)

    anal = partial(anal_eval, idxs)
    dual_anal = partial(dual_anal_eval, idxs)
    fn_anal = fun.compose(anal, l2x.incl)
    dual_fn_anal = fun.compose(dual_anal, l2x.incl)
    synth = vec.make_synthesis_operator(kernel_eigen['evecs'], idxs)
    dual_synth = vec.make_synthesis_operator(kernel_eigen['dual_evecs'], idxs)
    fn_synth = vec.make_fn_synthesis_operator(partial(fn_eval, idxs))
    dual_fn_synth = vec.make_fn_synthesis_operator(partial(dual_fn_eval, idxs))
    spec = kernel_eigen['evals'][idxs]
    basis = KernelEigenbasis(dim=len(idxs), anal=anal, dual_anal=dual_anal,
                             synth=synth, dual_synth=dual_synth,
                             fn_anal=fn_anal, dual_fn_anal=dual_fn_anal,
                             fn_synth=fn_synth, dual_fn_synth=dual_fn_synth,
                             vec=vc, dual_vec=dual_vc, fn=fn, dual_fn=dual_fn,
                             evl=evl, lapl_evl=lapl_evl, spec=spec,
                             lapl_spec=lapl_spec)
    return basis


def make_eigenbasis_bs[Ns: Shape, D: DTypeLike, X: Array](
        l2x: L2VectorAlgebra[Ns, D, X, K], kernel: Callable[[X, X], K],
        laplace_method: Literal['lin', 'log', 'inv'],
        kernel_eigen: KernelEigen,
        which_eigs: int | tuple[int, int] | list[int]) \
            -> KernelEigenbasis[X, R, V, Rs, int | Array]:
    """Make kernel eigenbasis for bistochastic kernels."""
    match which_eigs:
        case int():
            idxs = jnp.arange(which_eigs)
        case tuple(ints) if all(isinstance(i, int) for i in ints) \
                and len(ints) == 2:
            idxs = jnp.arange(which_eigs[0], which_eigs[1])
        case _:
            idxs = jnp.array(which_eigs)
    lapl_spec = to_laplace_eigenvalues(kernel_eigen['evals'][idxs],
                                       kernel_eigen['bandwidth'],
                                       method=laplace_method)
    extension_kernel = knl.bs_normalize(l2x, kernel)
    extension_op = knl.make_integral_operator(l2x, extension_kernel)

    def vc(i: int | Array) -> V:
        return kernel_eigen['evecs'][:, idxs[i]]

    def evl(i: int | Array) -> K:
        return kernel_eigen['evals'][idxs[i]]

    def lapl_evl(i: int | Array) -> K:
        return lapl_spec[idxs[i]]

    def fn(i: int | Array) -> F[X, K]:
        return extension_op(kernel_eigen['dual_evecs'][:, idxs[i]]
                            / jnp.sqrt(kernel_eigen['evals'][idxs[i]]))

    @partial(vmap, in_axes=(0, None))
    def anal_eval(i: int | Array, v: V) -> K:
        return l2x.innerp(kernel_eigen['evecs'][:, idxs[i]], v)

    @partial(vmap, in_axes=(0, None))
    def fn_eval(i: int | Array, x: X) -> R:
        return fn(i)(x)

    anal = partial(anal_eval, idxs)
    fn_anal = fun.compose(anal, l2x.incl)
    synth = vec.make_synthesis_operator(kernel_eigen['evecs'], idxs)
    fn_synth = vec.make_fn_synthesis_operator(partial(fn_eval, idxs))
    spec = kernel_eigen['evals'][idxs]
    basis = KernelEigenbasis(dim=len(idxs), anal=anal, dual_anal=anal,
                             synth=synth, dual_synth=synth,
                             fn_anal=fn_anal, dual_fn_anal=fn_anal,
                             fn_synth=fn_synth, dual_fn_synth=fn_synth,
                             vec=vc, dual_vec=vc, fn=fn, dual_fn=fn,
                             evl=evl, lapl_evl=lapl_evl, spec=spec,
                             lapl_spec=lapl_spec)
    return basis


def make_eigenbasis[D: DTypeLike, X: Array](
        pars: KernelPars, l2x: L2VectorAlgebra[Shape, D, X, K],
        kernel: Callable[[X, X], K],
        kernel_eigen: KernelEigen,
        laplace_method: Literal['lin', 'log', 'inv'],
        which_eigs: int | tuple[int, int] | list[int]) \
            -> KernelEigenbasis[X, R, V, Rs, int | Array]:
    """Make kernel eigenbasis."""
    match pars.normalization:
        case "laplace" | "fokkerplanck" | None:
            basis = make_eigenbasis_dm(l2x, kernel, pars.normalization,
                                       laplace_method, kernel_eigen,
                                       which_eigs)
        case "bistochastic":
            basis = make_eigenbasis_bs(l2x, kernel, laplace_method,
                                       kernel_eigen, which_eigs)
    return basis


def plot_kernel_tuning(tune_info: TuneInfo, title: Optional[str] = None,
                       i_fig: int = 1) -> Figure:
    """Plot kernel tuning function."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    ax.plot(tune_info['log10_bandwidths'], tune_info['est_dims'], '.-')
    eps_label = f"$\\epsilon_{{opt}} = {tune_info['opt_bandwidth']: .3e}$"
    ax.axvline(float(np.log10(tune_info['opt_bandwidth'])), color=u'#ff7f0e',
               label=eps_label)
    ax.grid()
    ax.legend()
    ax.set_xlabel(r"$\log_{10}(\epsilon)$")
    ax.set_ylabel("Estimated manifold dimension")
    if title is not None:
        ax.set_title(title)
    return fig


def plot_laplace_spectrum(
        kernel_eigen: KernelEigen,
        num_eigs_plt: Optional[int] = None, i_fig: int = 1) -> Figure:
    """Plot spectrum of Laplacian eigenvalues."""
    if num_eigs_plt is None:
        num_eigs_plt = len(kernel_eigen['evals'])
    kernel_evals = kernel_eigen['evals'][:num_eigs_plt]
    lapl_evals = partial(to_laplace_eigenvalues, kernel_evals,
                         kernel_eigen['bandwidth'])
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    ax.plot(jnp.arange(1, num_eigs_plt),
            jnp.log10(lapl_evals('lin')[1:]), '.',
            label=r"$4(1-\lambda_j)/\epsilon^2$")
    ax.plot(jnp.arange(1, num_eigs_plt),
            jnp.log10(lapl_evals('log')[1:]), '.',
            label=r"$-4\log\lambda_j/\epsilon^2$")
    ax.plot(jnp.arange(1, num_eigs_plt),
            jnp.log10(lapl_evals('inv')[1:]), '.',
            label=r"$(\lambda_j^{-1}-1)/(\lambda_1-1)$")
    ax.grid()
    ax.legend()
    ax.set_xlabel("$j$")
    ax.set_ylabel(r"$\eta_j$")
    ax.set_title("Laplacian eigenvalues")
    return fig

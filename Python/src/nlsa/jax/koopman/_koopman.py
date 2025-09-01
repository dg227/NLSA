# pyright: basic
"""Provide functions for Koopmman operator computations in JAX."""

import jax.numpy as jnp
import jax.numpy.linalg as jla
import jax.scipy as jsp
import matplotlib.pyplot as plt
import nlsa.jax.vector_algebra as vec
from collections.abc import Callable
from functools import partial
from jax import Array, vmap
from jax.typing import DTypeLike
from matplotlib.figure import Figure
from nlsa.function_algebra import compose
from nlsa.jax.vector_algebra import VectorAlgebra
from nlsa.kernels import KernelEigenbasis
from nlsa.koopman import KoopmanEigenbasis, KoopmanParsDiff, KoopmanParsQz
from typing import Optional, TypedDict

type Css = Array  # Collection of basis expansion coefficient vectors
type K = Array  # Scalar
type Ks = Array  # Collection of scalars
type V = Array  # L2 observable vector


class KoopmanEigen(TypedDict):
    """TypedDict containing spectral data of the Koopmman generator."""

    evals: Ks
    """Operator eigenvalues."""

    gen_evals: Ks
    """Generator eigenvalues."""

    efreqs: Ks
    """Generator eigenfrequecies"""

    eperiods: Ks
    """Generator eigenperiods."""

    engys: Ks
    """Dirichlet energies."""

    evec_coeffs: Css
    """Basis expansion coefficients of Koopman eigenvectors."""

    dual_evec_coeffs: Css
    """Basis expansion coefficients of dual (left) Koopman eigenvectors."""


def compute_eigen_diff[X: Array](
            pars: KoopmanParsDiff, gen_mat: Array,
            kernel_basis: KernelEigenbasis[X, K, V, Ks, int | Array]) \
        -> KoopmanEigen:
    """Solve eigenvalue problem for diffusion-regularized generator."""
    match pars.which_eigs_galerkin:
        case int():
            idxs = jnp.arange(1, pars.which_eigs_galerkin + 1)
        case tuple(ints) if all(isinstance(i, int) for i in ints) \
                and len(ints) == 2:
            idxs = jnp.arange(pars.which_eigs_galerkin[0],
                              pars.which_eigs_galerkin[1])
        case _:
            idxs = jnp.array(pars.which_eigs_galerkin)
    assert jnp.all(idxs)  # All indices should be nonzero
    diff_mat = pars.tau * jnp.diag(kernel_basis.lapl_spec[idxs])
    if pars.antisym:
        reg_gen_mat = (gen_mat - gen_mat.T)/2 - diff_mat
    else:
        reg_gen_mat = gen_mat - diff_mat
    gen_evals, evec_coeffs = jla.eig(reg_gen_mat)
    engys = jnp.sum(jnp.abs(evec_coeffs)**2
                    / kernel_basis.spec[idxs, jnp.newaxis], axis=0) - 1
    gram = evec_coeffs @ evec_coeffs.conj().T
    dual_evec_coeffs = jsp.linalg.solve(gram, evec_coeffs, assume_a="her")
    match pars.sort_by:
        case "frequency":
            isort = jnp.argsort(jnp.abs(gen_evals.real))
        case "energy":
            isort = jnp.argsort(engys)

    eigen: KoopmanEigen = {
        'evals': jnp.concatenate((jnp.atleast_1d(0), gen_evals[isort])),
        'gen_evals': jnp.concatenate((jnp.atleast_1d(0), gen_evals[isort])),
        'efreqs': jnp.concatenate((jnp.atleast_1d(0),
                                   gen_evals[isort].imag)),
        'eperiods': jnp.concatenate((jnp.atleast_1d(jnp.inf),
                                     2 * jnp.pi / gen_evals[isort].imag)),
        'engys': jnp.concatenate((jnp.atleast_1d(0), engys[isort])),
        'evec_coeffs': jsp.linalg.block_diag(1, evec_coeffs[:, isort]),
        'dual_evec_coeffs': jsp.linalg.block_diag(1,
                                                  dual_evec_coeffs[:, isort])}
    return eigen


def compute_eigen_qz[X: Array](
            pars: KoopmanParsQz, qz_mat: Array,
            kernel_basis: KernelEigenbasis[X, K, V, Ks, int | Array]) \
        -> KoopmanEigen:
    """Solve eigenvalue problem for regularized Qz operator."""
    match pars.which_eigs_galerkin:
        case int():
            idxs = jnp.arange(1, pars.which_eigs_galerkin + 1)
        case tuple(ints) if all(isinstance(i, int) for i in ints) \
                and len(ints) == 2:
            idxs = jnp.arange(pars.which_eigs_galerkin[0],
                              pars.which_eigs_galerkin[1])
        case _:
            idxs = jnp.array(pars.which_eigs_galerkin)
    assert jnp.all(idxs)  # All indices should be nonzero
    lambs = kernel_basis.spec[idxs] ** pars.tau
    reg_qz_mat = lambs * (qz_mat - qz_mat.T)/2 * lambs[:, jnp.newaxis]
    qz_evals, evec_coeffs = jla.eig(reg_qz_mat)
    gen_evals = \
        1j * (1 + jnp.sqrt(1 - 4 * pars.res_z**2 * qz_evals.imag**2)) \
        / (2 * qz_evals.imag)
    engys = jnp.sum(jnp.abs(evec_coeffs)**2
                    / kernel_basis.spec[idxs, jnp.newaxis], axis=0) - 1
    match pars.sort_by:
        case "frequency":
            isort = jnp.argsort(jnp.abs(gen_evals.real))
        case "energy":
            isort = jnp.argsort(engys)

    eigen: KoopmanEigen = {
        'evals': jnp.concatenate((jnp.atleast_1d(0), qz_evals[isort])),
        'gen_evals': jnp.concatenate((jnp.atleast_1d(0), gen_evals[isort])),
        'efreqs': jnp.concatenate((jnp.atleast_1d(0),
                                   gen_evals[isort].imag)),
        'eperiods': jnp.concatenate((jnp.atleast_1d(jnp.inf),
                                     2 * jnp.pi / gen_evals[isort].imag)),
        'engys': jnp.concatenate((jnp.atleast_1d(0), engys[isort])),
        'evec_coeffs': jsp.linalg.block_diag(1, evec_coeffs[:, isort]),
        'dual_evec_coeffs': jsp.linalg.block_diag(1, evec_coeffs[:, isort])}
    return eigen


def make_eigenbasis[X: Array, L: int, D: DTypeLike](
        c_l: VectorAlgebra[tuple[L], D],
        kernel_basis: KernelEigenbasis[X, K, V, Ks, int | Array],
        koopman_eigen: KoopmanEigen,
        which_eigs: int | tuple[int, int] | list[int]) \
            -> KoopmanEigenbasis[X, K, V, Ks, int | Array]:
    """Make Koopman eigenbasis from eigendecomposition of asymmetric op."""
    match which_eigs:
        case int():
            idxs = jnp.arange(which_eigs)
        case tuple(ints) if all(isinstance(i, int) for i in ints) \
                and len(ints) == 2:
            idxs = jnp.arange(which_eigs[0], which_eigs[1])
        case _:
            idxs = jnp.array(which_eigs)

    def vc(i: int | Array) -> V:
        return kernel_basis.synth(koopman_eigen['evec_coeffs'][:, idxs[i]])

    def dual_vc(i: int | Array) -> V:
        return kernel_basis.dual_synth(
            koopman_eigen['dual_evec_coeffs'][:, idxs[i]])

    def evl(i: int | Array) -> K:
        return koopman_eigen['evals'][idxs[i]]

    def gen_evl(i: int | Array) -> K:
        return koopman_eigen['gen_evals'][idxs[i]]

    def efreq(i: int | Array) -> K:
        return koopman_eigen['efreqs'][idxs[i]]

    def eperiod(i: int | Array) -> K:
        return koopman_eigen['eperiods'][idxs[i]]

    def engy(i: int | Array) -> K:
        return koopman_eigen['engys'][idxs[i]]

    def fn(i: int | Array) -> Callable[[X], K]:
        return kernel_basis.fn_synth(koopman_eigen['evec_coeffs'][:, idxs[i]])

    def dual_fn(i: int | Array) -> Callable[[X], K]:
        return kernel_basis.dual_fn_synth(
            koopman_eigen['dual_evec_coeffs'][:, idxs[i]])

    @partial(vmap, in_axes=(0, None))
    def anal_eval_c(i: int | Array, v: V) -> K:
        return c_l.innerp(koopman_eigen['dual_evec_coeffs'][:, idxs[i]], v)

    @partial(vmap, in_axes=(0, None))
    def dual_anal_eval_c(i: int | Array, v: V) -> K:
        return c_l.innerp(koopman_eigen['evec_coeffs'][:, idxs[i]], v)

    anal_c = partial(anal_eval_c, idxs)
    anal = compose(anal_c, kernel_basis.anal)
    dual_anal_c = partial(dual_anal_eval_c, idxs)
    dual_anal = compose(dual_anal_c, kernel_basis.dual_anal)
    fn_anal = compose(anal_c, kernel_basis.fn_anal)
    dual_fn_anal = compose(dual_anal_c, kernel_basis.dual_fn_anal)
    synth_c = vec.make_synthesis_operator(koopman_eigen['evec_coeffs'], idxs)
    synth = compose(kernel_basis.synth, synth_c)
    dual_synth_c = vec.make_synthesis_operator(
        koopman_eigen['dual_evec_coeffs'], idxs)
    dual_synth = compose(kernel_basis.dual_synth, dual_synth_c)
    fn_synth = compose(kernel_basis.fn_synth, synth_c)
    dual_fn_synth = compose(kernel_basis.dual_fn_synth, synth_c)
    spec = koopman_eigen['evals'][idxs]
    gen_spec = koopman_eigen['gen_evals'][idxs]
    efreqs = koopman_eigen['efreqs'][idxs]
    eperiods = koopman_eigen['eperiods'][idxs]
    engys = koopman_eigen['engys'][idxs]
    basis = KoopmanEigenbasis(dim=len(idxs), anal=anal, dual_anal=dual_anal,
                              synth=synth, dual_synth=dual_synth,
                              fn_anal=fn_anal, dual_fn_anal=dual_fn_anal,
                              fn_synth=fn_synth,
                              dual_fn_synth=dual_fn_synth, vec=vc,
                              dual_vec=dual_vc, fn=fn, dual_fn=dual_fn,
                              evl=evl, gen_evl=gen_evl, efreq=efreq,
                              eperiod=eperiod, engy=engy, spec=spec,
                              gen_spec=gen_spec, efreqs=efreqs,
                              eperiods=eperiods, engys=engys)
    return basis


def make_eigenbasis_antisym[X: Array, L: int, D: DTypeLike](
        c_l: VectorAlgebra[tuple[L], D],
        kernel_basis: KernelEigenbasis[X, K, V, Ks, int | Array],
        koopman_eigen: KoopmanEigen,
        which_eigs: int | tuple[int, int] | list[int]) \
            -> KoopmanEigenbasis[X, K, V, Ks, int | Array]:
    """Make Koopman eigenbasis from eigendecomposition of antisymmetric op."""
    match which_eigs:
        case int():
            idxs = jnp.arange(which_eigs)
        case tuple(ints) if all(isinstance(i, int) for i in ints) \
                and len(ints) == 2:
            idxs = jnp.arange(which_eigs[0], which_eigs[1])
        case _:
            idxs = jnp.array(which_eigs)

    def vc(i: int | Array) -> V:
        return kernel_basis.synth(koopman_eigen['evec_coeffs'][:, idxs[i]])

    def evl(i: int | Array) -> K:
        return koopman_eigen['evals'][idxs[i]]

    def gen_evl(i: int | Array) -> K:
        return koopman_eigen['gen_evals'][idxs[i]]

    def efreq(i: int | Array) -> K:
        return koopman_eigen['efreqs'][idxs[i]]

    def eperiod(i: int | Array) -> K:
        return koopman_eigen['eperiods'][idxs[i]]

    def engy(i: int | Array) -> K:
        return koopman_eigen['engys'][idxs[i]]

    def fn(i: int | Array) -> Callable[[X], K]:
        return kernel_basis.fn_synth(koopman_eigen['evec_coeffs'][:, idxs[i]])

    @partial(vmap, in_axes=(0, None))
    def anal_eval_c(i: int | Array, v: V) -> K:
        return c_l.innerp(koopman_eigen['evec_coeffs'][:, idxs[i]], v)

    anal_c = partial(anal_eval_c, idxs)
    anal = compose(anal_c, kernel_basis.anal)
    fn_anal = compose(anal_c, kernel_basis.fn_anal)
    synth_c = vec.make_synthesis_operator(koopman_eigen['evec_coeffs'], idxs)
    synth = compose(kernel_basis.synth, synth_c)
    fn_synth = compose(kernel_basis.fn_synth, synth_c)
    spec = koopman_eigen['evals'][idxs]
    gen_spec = koopman_eigen['gen_evals'][idxs]
    efreqs = koopman_eigen['efreqs'][idxs]
    eperiods = koopman_eigen['eperiods'][idxs]
    engys = koopman_eigen['engys'][idxs]
    basis = KoopmanEigenbasis(dim=len(idxs), anal=anal, dual_anal=anal,
                              synth=synth, dual_synth=synth, fn_anal=fn_anal,
                              dual_fn_anal=fn_anal, fn_synth=fn_synth,
                              dual_fn_synth=fn_synth, vec=vc, dual_vec=vc,
                              fn=fn, dual_fn=fn, evl=evl, gen_evl=gen_evl,
                              efreq=efreq, eperiod=eperiod, engy=engy,
                              spec=spec, gen_spec=gen_spec,
                              efreqs=efreqs, eperiods=eperiods, engys=engys)
    return basis


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

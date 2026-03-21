# pyright: basic
"""Provide classes and functions for kernel computations in JAX."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nlsa.function_algebra as fun
import nlsa.kernels as knl
import nlsa.jax.matrix_algebra as mat
import nlsa.jax.vector_algebra as vec
import numpy as np
import numpy.typing as npt
import scipy.sparse.linalg as sla
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from jax import Array, jit, vmap
from jax.sharding import NamedSharding
from jax.typing import ArrayLike, DTypeLike
from matplotlib.figure import Figure
from nlsa.jax.sharding import (
    EigShardings,
    NamedSharder,
    SvdShardings,
    make_eigh_with_sharding_constraints,
    make_svd_with_sharding_constraints,
    shardit,
)
from nlsa.jax.utils import batch_map
from nlsa.jax.vector_algebra import L2FnAlgebra
from nlsa.kernels import KernelEigenbasis
from nlsa.utils import swap_args
from scipy.sparse.linalg import LinearOperator
from typing import Literal, NamedTuple, Optional, Self, TypedDict

type R = Array  # Real number
type Rl = Array  # l-dimensional real vectors
type Rs = Array  # Collection of real numbers
type K = Array  # Scalar
type Ks = Array  # Collection of scalars
type V = Array  # Vector
type Vs = Array  # Collection of vectors
type Shape = tuple[int, ...]
type F[*Ss, T] = Callable[[*Ss], T]  # Shorthand for Callables
type NamedSharder1D = NamedSharder[tuple[int], Literal["x"]]
type NamedSharder2D = NamedSharder[tuple[int, int], Literal["x", "y"]]


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

    batch_size: Optional[int] = None
    """Batch size in tuning loop."""

    def __str__(self) -> str:
        loglimstr = "_".join(
            (
                f"loglim{self.log10_bandwidth_lims[0]}",
                f"loglim{self.log10_bandwidth_lims[1]}",
            )
        )
        return "_".join(
            (
                f"nb{self.num_bandwidths}",
                loglimstr,
                f"dim{self.manifold_dim}",
                f"scl{self.bandwidth_scl}",
            )
        )


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
class DmKernelPars:
    """Dataclass containing diffusion maps eigendecomposition parameters."""

    normalization: Optional[Literal["laplace", "fokkerplanck"]]
    """Kernel normalization method."""

    eigensolver: Literal["eigh", "eigsh", "lobpcg"]
    """Eigensolver used for kernel eigendecomposition."""

    num_eigs: int
    """Number of kernel eigenvalue/eigenvector pairs to compute."""

    batch_size: Optional[int] = None
    """Maximum batch size for matrix-matrix products."""

    def __str__(self) -> str:
        """Create string representation of diffusion maps kernel parameters."""
        if self.normalization is None:
            normstr = "nonorm"
        else:
            normstr = self.normalization
        return "_".join((normstr, self.eigensolver, f"neigs{self.num_eigs}"))


@dataclass(frozen=True)
class BsKernelPars:
    """Dataclass containing bistochastic kernel eigendecomp parameters."""

    eigensolver: Literal["svd", "svds"]
    """Eigensolver used for kernel eigendecomposition."""

    num_eigs: int
    """Number of kernel eigenvalue/eigenvector pairs to compute."""

    batch_size: Optional[int] = None
    """Maximum batch size for matrix-matrix products."""

    def __str__(self) -> str:
        """Create string representation of diffusion maps kernel parameters."""
        return "_".join(
            ("bistochastic", self.eigensolver, f"neigs{self.num_eigs}")
        )


type KernelPars = DmKernelPars | BsKernelPars


# TODO: Consider changing this to a namedtuple in a future implementation. The
# namedtuple could be generic and put into nlsa.kernels so we inherit from it
# here to define custom methods such as sharding.
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


class KernelEigenShardings(NamedTuple):
    """NamedTuple holding shardings for computation KernelEigen objects."""

    matrix: Optional[NamedSharding] = None
    """Sharding of kernel matrix."""

    eigenvalues: Optional[NamedSharding] = None
    """Sharding of eigenvalue array."""

    eigenvectors: Optional[NamedSharding] = None
    """Sharding of eigenvector array."""

    @classmethod
    def from_named_sharder[Shape: tuple[int, ...], AxisNames: str](
        cls, sharder: Optional[NamedSharder[Shape, AxisNames]]
    ) -> Self:
        """Create KernelEigenSharding object from NamedSharder."""
        if sharder is not None:
            x_sharding = sharder.sharding(sharder.axis_names[0])
            replicating = sharder.sharding(None)
            return cls(eigenvalues=replicating, eigenvectors=x_sharding)
        else:
            return cls()


def tune_bandwidth[Ns: Shape, D: DTypeLike, X: Array](
    pars: TunePars,
    l2x: L2FnAlgebra[Ns, D, X, R],
    kernel_family: Callable[[R], F[X, X, R]],
) -> tuple[R, TuneInfo]:
    """Compute optimal bandwidth for RBF kernel family."""
    log10_bandwidths = jnp.linspace(
        pars.log10_bandwidth_lims[0],
        pars.log10_bandwidth_lims[1],
        pars.num_bandwidths,
    )
    kernel_dim = knl.make_tuning_objective(
        l2x,
        kernel_family,
        grad=jax.grad,
        exp=partial(jnp.power, 10),
        log=jnp.log10,
    )

    def tune() -> dict[str, Array]:
        # est_dims = jax.lax.fori_loop(
        #     0, pars.num_bandwidths, body_fun, jnp.empty(pars.num_bandwidths)
        # )
        est_dims = batch_map(kernel_dim, batch_size=pars.batch_size)(
            log10_bandwidths
        )
        if pars.manifold_dim is None:
            i_opt = jnp.argmax(est_dims)
        else:
            i_opt = jnp.argmin(jnp.abs(est_dims - pars.manifold_dim))
        log10_opt_bandwidth = log10_bandwidths[i_opt]
        opt_bandwidth = 10**log10_opt_bandwidth
        opt_dim = kernel_dim(log10_opt_bandwidth)
        bandwidth = pars.bandwidth_scl * opt_bandwidth
        dim = kernel_dim(jnp.log10(bandwidth))
        kernel = kernel_family(bandwidth)
        vol = knl.riemannian_vol(
            l2x,
            kernel=knl.dm_normalize(l2x, kernel, alpha="1"),
            dim=dim,
            t_heat=jnp.asarray(bandwidth**2 / 4, dtype=l2x.dtype),
            fourpi=jnp.asarray(4 * jnp.pi, dtype=l2x.dtype),
        )
        return {
            "log10_bandwidths": log10_bandwidths,
            "est_dims": est_dims,
            "opt_bandwidth": opt_bandwidth,
            "opt_dim": opt_dim,
            "i_opt": i_opt,
            "bandwidth": bandwidth,
            "dim": dim,
            "vol": vol,
        }

    _tune_info = jit(tune)()
    tune_info: TuneInfo = {
        "log10_bandwidths": _tune_info["log10_bandwidths"],
        "est_dims": _tune_info["est_dims"],
        "opt_bandwidth": float(_tune_info["opt_bandwidth"]),
        "opt_dim": float(_tune_info["opt_dim"]),
        "i_opt": int(_tune_info["i_opt"]),
        "bandwidth": float(_tune_info["bandwidth"]),
        "dim": float(_tune_info["dim"]),
        "vol": float(_tune_info["vol"]),
    }
    return _tune_info["bandwidth"], tune_info


def tune_bandwidth_from_shape_function[Ns: Shape, D: DTypeLike, X: Array](
    pars: TunePars,
    l2x: L2FnAlgebra[Ns, D, X, R],
    shape_func: Callable[[K], K],
    neg_grad_shape_func: Callable[[K], K],
    sqdist: F[X, X, K],
) -> tuple[R, TuneInfo]:
    """Compute optimal bandwidth for RBF kernel family."""
    log10_bandwidths = jnp.linspace(
        pars.log10_bandwidth_lims[0],
        pars.log10_bandwidth_lims[1],
        pars.num_bandwidths,
    )
    kernel_family = knl.make_kernel_family(l2x, shape_func, sqdist)
    kernel_dim = knl.make_tuning_objective_from_shape_function(
        l2x,
        shape_func,
        neg_grad_shape_func,
        sqdist,
        two=jnp.asarray(2, dtype=l2x.dtype),
        exp=partial(jnp.power, 10),
    )

    def tune() -> dict[str, Array]:
        est_dims = batch_map(kernel_dim, batch_size=pars.batch_size)(
            log10_bandwidths
        )
        if pars.manifold_dim is None:
            i_opt = jnp.argmax(est_dims)
        else:
            i_opt = jnp.argmin(jnp.abs(est_dims - pars.manifold_dim))
        log10_opt_bandwidth = log10_bandwidths[i_opt]
        opt_bandwidth = 10**log10_opt_bandwidth
        opt_dim = kernel_dim(log10_opt_bandwidth)
        bandwidth = pars.bandwidth_scl * opt_bandwidth
        dim = kernel_dim(jnp.log10(bandwidth))
        kernel = kernel_family(bandwidth)
        vol = knl.riemannian_vol(
            l2x,
            kernel=knl.dm_normalize(l2x, kernel, alpha="1"),
            dim=dim,
            t_heat=jnp.asarray(bandwidth**2 / 4, dtype=l2x.dtype),
            fourpi=jnp.asarray(4 * jnp.pi, dtype=l2x.dtype),
        )
        return {
            "log10_bandwidths": log10_bandwidths,
            "est_dims": est_dims,
            "opt_bandwidth": opt_bandwidth,
            "opt_dim": opt_dim,
            "i_opt": i_opt,
            "bandwidth": bandwidth,
            "dim": dim,
            "vol": vol,
        }

    _tune_info = jit(tune)()
    tune_info: TuneInfo = {
        "log10_bandwidths": _tune_info["log10_bandwidths"],
        "est_dims": _tune_info["est_dims"],
        "opt_bandwidth": float(_tune_info["opt_bandwidth"]),
        "opt_dim": float(_tune_info["opt_dim"]),
        "i_opt": int(_tune_info["i_opt"]),
        "bandwidth": float(_tune_info["bandwidth"]),
        "dim": float(_tune_info["dim"]),
        "vol": float(_tune_info["vol"]),
    }
    return _tune_info["bandwidth"], tune_info


class _DmSymOperatorSpectrum(NamedTuple):
    """NamedTuple holding symmetric diffusion maps operator spectral data."""

    evals: Array
    """Kernel eigenvalues."""

    evecs: Array
    """Kernel eigenvectors."""


def _make_eigh_dm_eigensolver[Ns: Shape, D: DTypeLike, X: Array](
    l2x: L2FnAlgebra[Ns, D, X, R],
    kernel: Callable[[X, X], R],
    batch_size: Optional[int] = None,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> Callable[[], _DmSymOperatorSpectrum]:
    """Make eigensolver for diffusion-maps normalized operaror using eigh."""
    kernel_op = fun.compose(l2x.incl, knl.make_integral_operator(l2x, kernel))
    eig_shardings = EigShardings(
        eigenvalues=shardings.eigenvalues, eigenvectors=shardings.matrix
    )
    eig = make_eigh_with_sharding_constraints(shardings=eig_shardings)

    def eigensolve() -> _DmSymOperatorSpectrum:
        a = mat.materialize_in_std_basis(
            kernel_op,
            in_dim=l2x.dim,
            dtype=l2x.dtype,
            batch_size=batch_size,
            out_sharding=shardings.matrix,
        )
        return _DmSymOperatorSpectrum(*eig(a))

    return eigensolve


def _make_eigsh_dm_eigensolver[Ns: Shape, D: DTypeLike, X: Array](
    l2x: L2FnAlgebra[Ns, D, X, R],
    kernel: Callable[[X, X], R],
    num_eigs: int,
) -> Callable[[], _DmSymOperatorSpectrum]:
    """Make eigensolver for diffusion-maps normalized operaror using eigsh."""
    num_samples = l2x.shape[0]
    kernel_op = fun.compose(l2x.incl, knl.make_integral_operator(l2x, kernel))
    to_device = partial(jnp.asarray, device=l2x.sharding)
    matvec: Callable[[ArrayLike], Array] = fun.compose(
        jit(kernel_op), to_device
    )
    a = LinearOperator(
        shape=(num_samples, num_samples),
        dtype=np.dtype(l2x.dtype),
        matvec=matvec,
    )

    def eigensolve() -> _DmSymOperatorSpectrum:
        evals, evecs = sla.eigsh(a, num_eigs, which="LA")
        evals = jnp.asarray(evals, dtype=l2x.dtype)
        evecs = jnp.asarray(evecs, dtype=l2x.dtype, device=l2x.sharding)
        return _DmSymOperatorSpectrum(evals=evals, evecs=evecs)

    return eigensolve


def _make_from_dm_sym_operator_spectrum[Ns: Shape, D: DTypeLike, X: Array](
    l2x: L2FnAlgebra[Ns, D, X, R],
    num_eigs: int,
    bandwidth: R,
    out_shardings: KernelEigenShardings = KernelEigenShardings(),
) -> Callable[[_DmSymOperatorSpectrum], KernelEigen]:
    """Create conversion function from DmSymOperatorSpectrum to KerneEigen."""
    num_samples = l2x.shape[0]
    norm = vmap(l2x.norm, in_axes=1)

    def from_dm_sym_operator_spectrum(
        spec: _DmSymOperatorSpectrum,
    ) -> KernelEigen:
        unsorted_evals, unsorted_evecs = spec
        isort = jnp.argsort(unsorted_evals)[::-1][:num_eigs]
        lambs = jnp.array(unsorted_evals[isort])
        sqrt_mus = jnp.abs(unsorted_evecs[:, isort[0]])
        scl = jnp.sign(unsorted_evecs[0, isort[0]])
        phis = jnp.array(
            unsorted_evecs[:, isort] / (scl * sqrt_mus.reshape((-1, 1)))
        )
        phi_duals = jnp.array(
            unsorted_evecs[:, isort]
            * num_samples
            * (scl * sqrt_mus.reshape((-1, 1)))
        )
        phi_norms = norm(phis)
        if out_shardings.eigenvalues is not None:
            lambs = jax.lax.with_sharding_constraint(
                lambs, shardings=out_shardings.eigenvalues
            )
        if out_shardings.eigenvectors is not None:
            sqrt_mus = jax.lax.with_sharding_constraint(
                sqrt_mus, shardings=out_shardings.eigenvectors
            )
            phis = jax.lax.with_sharding_constraint(
                phis, shardings=out_shardings.eigenvectors
            )
            phi_duals = jax.lax.with_sharding_constraint(
                phi_duals, shardings=out_shardings.eigenvectors
            )
        eigen: KernelEigen = {
            "evals": lambs,
            "evecs": phis / phi_norms,
            "dual_evecs": phi_duals * phi_norms,
            "weights": sqrt_mus**2,
            "bandwidth": bandwidth,
        }
        return eigen

    return from_dm_sym_operator_spectrum


def compute_eigen_dm[Ns: Shape, D: DTypeLike, X: Array](
    l2x: L2FnAlgebra[Ns, D, X, R],
    kernel: Callable[[X, X], R],
    normalization: Optional[Literal["laplace", "fokkerplanck"]],
    num_eigs: int,
    eigensolver: Literal["eigh", "eigsh", "lobpcg"],
    bandwidth: float | R = 1,
    batch_size: Optional[int] = None,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> KernelEigen:
    """Solve kernel eigenvalue problem for diffusion maps normalization."""
    if not isinstance(bandwidth, Array):
        _bandwidth = jnp.asarray(bandwidth)
    else:
        _bandwidth = bandwidth
    match normalization:
        case "laplace":
            dm_kernel = knl.dmsym_normalize(l2x, kernel, alpha="1")
        case "fokkerplanck":
            dm_kernel = knl.dmsym_normalize(l2x, kernel, alpha="0.5")
        case None:
            dm_kernel = kernel
    from_dm_sym_operator_spectrum = _make_from_dm_sym_operator_spectrum(
        l2x,
        bandwidth=_bandwidth,
        num_eigs=num_eigs,
        out_shardings=shardings,
    )
    match eigensolver:
        case "eigh":
            eigensolve = _make_eigh_dm_eigensolver(
                l2x,
                kernel=dm_kernel,
                batch_size=batch_size,
                shardings=shardings,
            )
            run = jax.jit(
                fun.compose(from_dm_sym_operator_spectrum, eigensolve)
            )
        case "eigsh":
            eigensolve = _make_eigsh_dm_eigensolver(
                l2x,
                kernel=dm_kernel,
                num_eigs=num_eigs,
            )
            run = fun.compose(
                jax.jit(from_dm_sym_operator_spectrum), eigensolve
            )
    return run()


class _BsOperatorSpectrum(NamedTuple):
    """NamedTuple holding bistochastic kernel operator spectral data."""

    left_sing_vecs: Array
    """Left singular vectors"""

    sing_vals: Array
    """Singular values of asymmetric kernel operator."""

    right_sing_vecs: Array
    """Right singular vectors."""


def _make_svd_bs_eigensolver[Ns: Shape, D: DTypeLike, X: Array](
    l2x: L2FnAlgebra[Ns, D, X, R],
    kernel: Callable[[X, X], R],
    batch_size: Optional[int] = None,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> Callable[[], _BsOperatorSpectrum]:
    """Make SVD solver for bistochastic kernel operator using svd."""
    kernel_op = fun.compose(l2x.incl, knl.make_integral_operator(l2x, kernel))
    svd_shardings = SvdShardings(
        left_sing_vectors=shardings.matrix,
        sing_values=shardings.eigenvalues,
        right_sing_vectors=shardings.matrix,
    )
    svd = make_svd_with_sharding_constraints(shardings=svd_shardings)

    def svdsolve() -> _BsOperatorSpectrum:
        a = mat.materialize_in_std_basis(
            kernel_op,
            in_dim=l2x.dim,
            dtype=l2x.dtype,
            batch_size=batch_size,
            out_sharding=shardings.matrix,
        )
        left_sing_vecs, sing_vals, right_sing_vecs = svd(a)
        return _BsOperatorSpectrum(
            left_sing_vecs=left_sing_vecs,
            sing_vals=sing_vals,
            right_sing_vecs=right_sing_vecs.T,
        )

    return svdsolve


def _make_svds_bs_eigensolver[Ns: Shape, D: DTypeLike, X: Array](
    l2x: L2FnAlgebra[Ns, D, X, R],
    kernel: Callable[[X, X], R],
    num_eigs: int,
    batch_size: Optional[int] = None,
) -> Callable[[], _BsOperatorSpectrum]:
    """Make SVD solver for bistochastic kernel operator using svds."""
    num_samples = l2x.shape[0]
    kernel_op = fun.compose(l2x.incl, knl.make_integral_operator(l2x, kernel))
    adj_kernel_op = fun.compose(
        l2x.incl, knl.make_integral_operator(l2x, swap_args(kernel))
    )
    to_device = partial(jnp.asarray, device=l2x.sharding)
    matvec: Callable[[ArrayLike], Array] = fun.compose(
        jit(kernel_op), to_device
    )
    rmatvec: Callable[[ArrayLike], Array] = fun.compose(
        jit(adj_kernel_op), to_device
    )
    matmat: Callable[[ArrayLike], Array] = fun.compose(
        jit(
            shardit(
                batch_map(
                    kernel_op, in_axis=1, out_axis=1, batch_size=batch_size
                ),
                sharding=l2x.sharding,
            )
        ),
        to_device,
    )
    rmatmat: Callable[[ArrayLike], Array] = fun.compose(
        jit(
            shardit(
                batch_map(
                    adj_kernel_op, in_axis=1, out_axis=1, batch_size=batch_size
                ),
                sharding=l2x.sharding,
            )
        ),
        to_device,
    )
    a = LinearOperator(
        shape=(num_samples, num_samples),
        dtype=np.dtype(l2x.dtype),
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
    )

    def svdsolve() -> _BsOperatorSpectrum:
        left_sing_vecs, sing_vals, right_sing_vecs = sla.svds(a, num_eigs)
        left_sing_vecs = jnp.asarray(left_sing_vecs, dtype=l2x.dtype)
        sing_vals = jnp.asarray(
            sing_vals, dtype=l2x.dtype, device=l2x.sharding
        )
        right_sing_vecs = jnp.asarray(
            right_sing_vecs, dtype=l2x.dtype, device=l2x.sharding
        ).T
        spec = _BsOperatorSpectrum(
            left_sing_vecs=left_sing_vecs,
            sing_vals=sing_vals,
            right_sing_vecs=right_sing_vecs,
        )
        return spec

    return svdsolve


def _make_from_bs_operator_spectrum[Ns: Shape, D: DTypeLike, X: Array](
    l2x: L2FnAlgebra[Ns, D, X, R],
    num_eigs: int,
    bandwidth: R,
    out_shardings: KernelEigenShardings = KernelEigenShardings(),
) -> Callable[[_BsOperatorSpectrum], KernelEigen]:
    """Create conversion function from BsOperatorSpectrum to KerneEigen."""
    num_samples = l2x.shape[0]
    norm = vmap(l2x.norm, in_axes=1)

    def from_bs_operator_spectrum(
        spec: _BsOperatorSpectrum,
    ) -> KernelEigen:
        unsorted_evecs, unsorted_singvals, unsorted_dual_evecs = spec
        unsorted_evals = unsorted_singvals**2
        isort = jnp.argsort(unsorted_evals)[::-1][:num_eigs]
        lambs = jnp.array(unsorted_evals[isort])
        sqrt_mus = jnp.abs(unsorted_evecs[:, isort[0]])
        scl = jnp.sign(unsorted_evecs[0, isort[0]])
        phis = jnp.array(
            unsorted_evecs[:, isort] / (scl * sqrt_mus.reshape((-1, 1)))
        )
        phi_duals = jnp.array(
            unsorted_dual_evecs[:, isort]
            * num_samples
            * (scl * sqrt_mus.reshape((-1, 1)))
        )
        phi_norms = norm(phis)
        if out_shardings.eigenvalues is not None:
            lambs = jax.lax.with_sharding_constraint(
                lambs, shardings=out_shardings.eigenvalues
            )
        if out_shardings.eigenvectors is not None:
            sqrt_mus = jax.lax.with_sharding_constraint(
                sqrt_mus, shardings=out_shardings.eigenvectors
            )
            phis = jax.lax.with_sharding_constraint(
                phis, shardings=out_shardings.eigenvectors
            )
            phi_duals = jax.lax.with_sharding_constraint(
                phi_duals, shardings=out_shardings.eigenvectors
            )
        eigen: KernelEigen = {
            "evals": lambs,
            "evecs": phis / phi_norms,
            "dual_evecs": phi_duals * phi_norms,
            "weights": sqrt_mus**2,
            "bandwidth": jnp.asarray(bandwidth),
        }
        return eigen

    return from_bs_operator_spectrum


def compute_eigen_bs[Ns: Shape, D: DTypeLike, X: Array](
    l2x: L2FnAlgebra[Ns, D, X, R],
    kernel: Callable[[X, X], R],
    num_eigs: int,
    eigensolver: Literal["svd", "svds"],
    bandwidth: float | R = 1,
    batch_size: Optional[int] = None,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> KernelEigen:
    """Solve kernel eigenvalue problem for bistochastic normalization."""
    if not isinstance(bandwidth, Array):
        _bandwidth = jnp.asarray(bandwidth)
    else:
        _bandwidth = bandwidth
    bs_kernel = knl.bs_normalize(l2x, kernel)
    from_bs_operator_spectrum = _make_from_bs_operator_spectrum(
        l2x,
        bandwidth=_bandwidth,
        num_eigs=num_eigs,
        out_shardings=shardings,
    )
    match eigensolver:
        case "svd":
            eigensolve = _make_svd_bs_eigensolver(
                l2x,
                kernel=bs_kernel,
                batch_size=batch_size,
                shardings=shardings,
            )
            run = jax.jit(fun.compose(from_bs_operator_spectrum, eigensolve))
        case "svds":
            eigensolve = _make_svds_bs_eigensolver(
                l2x,
                kernel=bs_kernel,
                num_eigs=num_eigs,
                batch_size=batch_size,
            )
            run = fun.compose(jax.jit(from_bs_operator_spectrum), eigensolve)
    return run()


def compute_eigen[D: DTypeLike, X: Array](
    pars: KernelPars,
    l2x: L2FnAlgebra[Shape, D, X, R],
    kernel: Callable[[X, X], R],
    bandwidth: float | R,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> KernelEigen:
    """Solve eigenvalue problem for DM or BS kernels."""
    match pars:
        case DmKernelPars():
            eigen = compute_eigen_dm(
                l2x,
                kernel=kernel,
                normalization=pars.normalization,
                num_eigs=pars.num_eigs,
                eigensolver=pars.eigensolver,
                bandwidth=bandwidth,
                batch_size=pars.batch_size,
                shardings=shardings,
            )
        case BsKernelPars():
            eigen = compute_eigen_bs(
                l2x,
                kernel=kernel,
                num_eigs=pars.num_eigs,
                eigensolver=pars.eigensolver,
                bandwidth=bandwidth,
                batch_size=pars.batch_size,
                shardings=shardings,
            )
    return eigen


def to_kernel_eigen(
    dict_in: dict[str, npt.ArrayLike],
    dtype: Optional[DTypeLike] = None,
    shardings: Optional[KernelEigenShardings] = None,
) -> KernelEigen:
    """Convert dict of numpy ArrayLike objects to KernelEigen TypedDict."""
    try:
        if shardings is not None:
            kernel_eigen: KernelEigen = {
                "evals": jnp.asarray(
                    dict_in["evals"], dtype, device=shardings.eigenvalues
                ),
                "evecs": jnp.asarray(
                    dict_in["evecs"], dtype, device=shardings.eigenvectors
                ),
                "dual_evecs": jnp.asarray(
                    dict_in["dual_evecs"], dtype, device=shardings.eigenvectors
                ),
                "weights": jnp.asarray(
                    dict_in["weights"], dtype, device=shardings.eigenvectors
                ),
                "bandwidth": jnp.asarray(dict_in["bandwidth"], dtype),
            }
        else:
            kernel_eigen: KernelEigen = {
                "evals": jnp.asarray(dict_in["evals"], dtype),
                "evecs": jnp.asarray(dict_in["evecs"], dtype),
                "dual_evecs": jnp.asarray(dict_in["dual_evecs"], dtype),
                "weights": jnp.asarray(dict_in["weights"], dtype),
                "bandwidth": jnp.asarray(dict_in["bandwidth"], dtype),
            }
        return kernel_eigen
    except ValueError as exc:
        raise ValueError("Incompatible keys/values") from exc


def to_laplace_eigenvalues(
    lambs: Array,
    bandwidth: ArrayLike,
    method: Literal["lin", "log", "inv"] = "log",
) -> Array:
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


def make_eigenvector_extension_dm[Ns: tuple[int, ...], D: DTypeLike, X: Array](
    l2x: L2FnAlgebra[Ns, D, X, R],
    kernel: Callable[[X, X], R],
    normalization: Optional[Literal["laplace", "fokkerplanck"]],
) -> tuple[Callable[[V, R], F[X, R]], Callable[[X, X], R]]:
    """Make Nystrom extension for diffusion maps kernels."""
    match normalization:
        case "laplace":
            extension_kernel = knl.dm_normalize(l2x, kernel, alpha="1")
        case "fokkerplanck":
            extension_kernel = knl.dm_normalize(l2x, kernel, alpha="0.5")
        case None:
            extension_kernel = kernel
    extension_kernel_op: Callable[[V], F[X, R]] = knl.make_integral_operator(
        l2x, extension_kernel
    )

    def nyst(phi: V, lamb: R) -> F[X, R]:
        return extension_kernel_op(phi / lamb)

    return nyst, extension_kernel


def make_eigenvector_extension_bs[Ns: tuple[int, ...], D: DTypeLike, X: Array](
    l2x: L2FnAlgebra[Ns, D, X, R], kernel: Callable[[X, X], R]
) -> tuple[Callable[[V, R], F[X, R]], Callable[[X, X], R]]:
    """Make Nystrom extension for bistochastic kernels."""
    extension_kernel = knl.bs_normalize(l2x, kernel)
    extension_kernel_op: Callable[[V], F[X, R]] = knl.make_integral_operator(
        l2x, extension_kernel
    )

    def nyst(phi: V, lamb: R) -> F[X, R]:
        return extension_kernel_op(phi / jnp.sqrt(lamb))

    return nyst, extension_kernel


def make_eigenvector_extension[Ns: tuple[int, ...], D: DTypeLike, X: Array](
    pars: KernelPars,
    l2x: L2FnAlgebra[Ns, D, X, R],
    kernel: Callable[[X, X], R],
) -> tuple[Callable[[V, R], F[X, R]], Callable[[X, X], R]]:
    """Make Nystrom extension for diffusion maps and bistochastic kernels."""
    match pars:
        case DmKernelPars():
            nyst, extension_kernel = make_eigenvector_extension_dm(
                l2x, kernel, pars.normalization
            )
        case BsKernelPars():
            nyst, extension_kernel = make_eigenvector_extension_bs(l2x, kernel)
    return nyst, extension_kernel


def make_eigenbasis_operators_dm[N: int, D: DTypeLike, X: Array](
    l2x: L2FnAlgebra[tuple[N], D, X, R],
    extend: Callable[[V, R], F[X, R]],
    kernel_eigen: KernelEigen,
    num_eigs: int,
) -> tuple[F[V, Rl], F[Rl, V], Callable[[V], F[X, R]]]:
    """Make analysis and synthesis operators for diffusion maps eigenbasis."""
    anal = vec.make_l2_analysis_operator(
        l2x, kernel_eigen["dual_evecs"][:, :num_eigs]
    )
    synth = vec.make_synthesis_operator(kernel_eigen["evecs"][:, :num_eigs])

    @partial(vmap, in_axes=(-1, 0, None))
    def extend_eval(v: V, lamb: R, x: X) -> R:
        return extend(v, lamb)(x)

    basis = partial(
        extend_eval,
        kernel_eigen["evecs"][:, :num_eigs],
        kernel_eigen["evals"][:num_eigs],
    )
    fn_synth = vec.make_fn_synthesis_operator(basis)
    return anal, synth, fn_synth


def make_eigenbasis_operators_bs[N: int, D: DTypeLike, X: Array](
    l2x: L2FnAlgebra[tuple[N], D, X, R],
    extend: Callable[[V, R], F[X, R]],
    kernel_eigen: KernelEigen,
    num_eigs: int,
) -> tuple[F[V, Rl], F[Rl, V], Callable[[V], F[X, R]]]:
    """Make analysis and synthesis operators for bistochastic eigenbasis."""
    anal = vec.make_l2_analysis_operator(
        l2x, kernel_eigen["evecs"][:, :num_eigs]
    )
    synth = vec.make_synthesis_operator(kernel_eigen["evecs"][:, :num_eigs])

    @partial(vmap, in_axes=(-1, 0, None))
    def extend_eval(v: V, lamb: R, x: X) -> R:
        return extend(v, lamb)(x)

    basis = partial(
        extend_eval,
        kernel_eigen["dual_evecs"][:, :num_eigs],
        kernel_eigen["evals"][:num_eigs],
    )
    fn_synth = vec.make_fn_synthesis_operator(basis)
    return anal, synth, fn_synth


def make_eigenbasis_operators[N: int, D: DTypeLike, X: Array](
    pars: KernelPars,
    l2x: L2FnAlgebra[tuple[N], D, X, R],
    kernel: Callable[[X, X], R],
    kernel_eigen: KernelEigen,
    num_eigs: int,
) -> tuple[F[V, Rl], F[Rl, V], Callable[[V], F[X, R]]]:
    """Make analysis and synthesis operators."""
    match pars:
        case DmKernelPars():
            extend, _ = make_eigenvector_extension_dm(
                l2x, kernel, pars.normalization
            )
            anal, synth, fn_synth = make_eigenbasis_operators_dm(
                l2x, extend, kernel_eigen, num_eigs
            )
        case BsKernelPars():
            extend, _ = make_eigenvector_extension_bs(l2x, kernel)
            anal, synth, fn_synth = make_eigenbasis_operators_bs(
                l2x, extend, kernel_eigen, num_eigs
            )
    return anal, synth, fn_synth


def make_eigenbasis_dm[Ns: Shape, D: DTypeLike, X: Array](
    l2x: L2FnAlgebra[Ns, D, X, R],
    kernel: Callable[[X, X], R],
    normalization: Optional[Literal["laplace", "fokkerplanck"]],
    laplace_method: Literal["lin", "log", "inv"],
    kernel_eigen: KernelEigen,
    which_eigs: int | tuple[int, int] | list[int],
) -> KernelEigenbasis[X, R, V, Rs, int | Array]:
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
        case (_, _):
            idxs = jnp.arange(which_eigs[0], which_eigs[1])
        case list():
            idxs = jnp.array(which_eigs)
    lapl_spec = to_laplace_eigenvalues(
        kernel_eigen["evals"][idxs],
        kernel_eigen["bandwidth"],
        method=laplace_method,
    )
    extension_op = knl.make_integral_operator(l2x, extension_kernel)
    dual_extension_op = knl.make_integral_operator(
        l2x, swap_args(extension_kernel)
    )

    def vc(i: int | Array) -> V:
        return kernel_eigen["evecs"][:, idxs[i]]

    def dual_vc(i: int | Array) -> V:
        return kernel_eigen["dual_evecs"][:, idxs[i]]

    def evl(i: int | Array) -> K:
        return kernel_eigen["evals"][idxs[i]]

    def lapl_evl(i: int | Array) -> K:
        return lapl_spec[idxs[i]]

    def fn(i: int | Array) -> Callable[[X], K]:
        return extension_op(
            kernel_eigen["evecs"][:, idxs[i]] / kernel_eigen["evals"][idxs[i]]
        )

    def dual_fn(i: int | Array) -> Callable[[X], K]:
        return dual_extension_op(
            kernel_eigen["dual_evecs"][:, idxs[i]]
            / kernel_eigen["evals"][idxs[i]]
        )

    @partial(vmap, in_axes=(0, None))
    def anal_eval(i: int | Array, v: V) -> K:
        return l2x.innerp(kernel_eigen["dual_evecs"][:, idxs[i]], v)

    @partial(vmap, in_axes=(0, None))
    def dual_anal_eval(i: int | Array, v: V) -> K:
        return l2x.innerp(kernel_eigen["evecs"][:, idxs[i]], v)

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
    synth = vec.make_synthesis_operator(kernel_eigen["evecs"], idxs)
    dual_synth = vec.make_synthesis_operator(kernel_eigen["dual_evecs"], idxs)
    fn_synth = vec.make_fn_synthesis_operator(partial(fn_eval, idxs))
    dual_fn_synth = vec.make_fn_synthesis_operator(partial(dual_fn_eval, idxs))
    spec = kernel_eigen["evals"][idxs]
    basis = KernelEigenbasis(
        dim=len(idxs),
        anal=anal,
        dual_anal=dual_anal,
        synth=synth,
        dual_synth=dual_synth,
        fn_anal=fn_anal,
        dual_fn_anal=dual_fn_anal,
        fn_synth=fn_synth,
        dual_fn_synth=dual_fn_synth,
        vec=vc,
        dual_vec=dual_vc,
        fn=fn,
        dual_fn=dual_fn,
        evl=evl,
        lapl_evl=lapl_evl,
        spec=spec,
        lapl_spec=lapl_spec,
    )
    return basis


def make_eigenbasis_bs[Ns: Shape, D: DTypeLike, X: Array](
    l2x: L2FnAlgebra[Ns, D, X, K],
    kernel: Callable[[X, X], K],
    laplace_method: Literal["lin", "log", "inv"],
    kernel_eigen: KernelEigen,
    which_eigs: int | tuple[int, int] | list[int],
) -> KernelEigenbasis[X, R, V, Rs, int | Array]:
    """Make kernel eigenbasis for bistochastic kernels."""
    match which_eigs:
        case int():
            idxs = jnp.arange(which_eigs)
        case (_, _):
            idxs = jnp.arange(which_eigs[0], which_eigs[1])
        case list():
            idxs = jnp.array(which_eigs)
    lapl_spec = to_laplace_eigenvalues(
        kernel_eigen["evals"][idxs],
        kernel_eigen["bandwidth"],
        method=laplace_method,
    )
    extension_kernel = knl.bs_normalize(l2x, kernel)
    extension_op = knl.make_integral_operator(l2x, extension_kernel)

    def vc(i: int | Array) -> V:
        return kernel_eigen["evecs"][:, idxs[i]]

    def evl(i: int | Array) -> K:
        return kernel_eigen["evals"][idxs[i]]

    def lapl_evl(i: int | Array) -> K:
        return lapl_spec[idxs[i]]

    def fn(i: int | Array) -> F[X, K]:
        return extension_op(
            kernel_eigen["dual_evecs"][:, idxs[i]]
            / jnp.sqrt(kernel_eigen["evals"][idxs[i]])
        )

    @partial(vmap, in_axes=(0, None))
    def anal_eval(i: int | Array, v: V) -> K:
        return l2x.innerp(kernel_eigen["evecs"][:, idxs[i]], v)

    @partial(vmap, in_axes=(0, None))
    def fn_eval(i: int | Array, x: X) -> R:
        return fn(i)(x)

    anal = partial(anal_eval, idxs)
    fn_anal = fun.compose(anal, l2x.incl)
    synth = vec.make_synthesis_operator(kernel_eigen["evecs"], idxs)
    fn_synth = vec.make_fn_synthesis_operator(partial(fn_eval, idxs))
    spec = kernel_eigen["evals"][idxs]
    basis = KernelEigenbasis(
        dim=len(idxs),
        anal=anal,
        dual_anal=anal,
        synth=synth,
        dual_synth=synth,
        fn_anal=fn_anal,
        dual_fn_anal=fn_anal,
        fn_synth=fn_synth,
        dual_fn_synth=fn_synth,
        vec=vc,
        dual_vec=vc,
        fn=fn,
        dual_fn=fn,
        evl=evl,
        lapl_evl=lapl_evl,
        spec=spec,
        lapl_spec=lapl_spec,
    )
    return basis


def make_eigenbasis[D: DTypeLike, X: Array](
    pars: KernelPars,
    l2x: L2FnAlgebra[Shape, D, X, K],
    kernel: Callable[[X, X], K],
    kernel_eigen: KernelEigen,
    laplace_method: Literal["lin", "log", "inv"],
    which_eigs: int | tuple[int, int] | list[int],
) -> KernelEigenbasis[X, R, V, Rs, int | Array]:
    """Make kernel eigenbasis."""
    match pars:
        case DmKernelPars():
            basis = make_eigenbasis_dm(
                l2x,
                kernel,
                pars.normalization,
                laplace_method,
                kernel_eigen,
                which_eigs,
            )
        case BsKernelPars():
            basis = make_eigenbasis_bs(
                l2x, kernel, laplace_method, kernel_eigen, which_eigs
            )
    return basis


def plot_kernel_tuning(
    tune_info: TuneInfo, title: Optional[str] = None, i_fig: int = 1
) -> Figure:
    """Plot kernel tuning function."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    ax.plot(tune_info["log10_bandwidths"], tune_info["est_dims"], ".-")
    eps_label = f"$\\epsilon_{{opt}} = {tune_info['opt_bandwidth']: .3e}$"
    ax.axvline(
        float(np.log10(tune_info["opt_bandwidth"])),
        color="#ff7f0e",
        label=eps_label,
    )
    ax.grid()
    ax.legend()
    ax.set_xlabel(r"$\log_{10}(\epsilon)$")
    ax.set_ylabel("Estimated manifold dimension")
    if title is not None:
        ax.set_title(title)
    return fig


def plot_laplace_spectrum(
    kernel_eigen: KernelEigen,
    num_eigs_plt: Optional[int] = None,
    i_fig: int = 1,
) -> Figure:
    """Plot spectrum of Laplacian eigenvalues."""
    if num_eigs_plt is None:
        num_eigs_plt = len(kernel_eigen["evals"])
    kernel_evals = kernel_eigen["evals"][:num_eigs_plt]
    lapl_evals = partial(
        to_laplace_eigenvalues, kernel_evals, kernel_eigen["bandwidth"]
    )
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    ax.plot(
        jnp.arange(1, num_eigs_plt),
        jnp.log10(lapl_evals("lin")[1:]),
        ".",
        label=r"$4(1-\lambda_j)/\epsilon^2$",
    )
    ax.plot(
        jnp.arange(1, num_eigs_plt),
        jnp.log10(lapl_evals("log")[1:]),
        ".",
        label=r"$-4\log\lambda_j/\epsilon^2$",
    )
    ax.plot(
        jnp.arange(1, num_eigs_plt),
        jnp.log10(lapl_evals("inv")[1:]),
        ".",
        label=r"$(\lambda_j^{-1}-1)/(\lambda_1-1)$",
    )
    ax.grid()
    ax.legend()
    ax.set_xlabel("$j$")
    ax.set_ylabel(r"$\log_{10}\eta_j$")
    ax.set_title("Laplacian eigenvalues")
    return fig

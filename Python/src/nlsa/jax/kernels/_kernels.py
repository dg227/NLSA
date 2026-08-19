"""Provide classes and functions for kernel computations in JAX."""

# TODO: With the exception of the eigsh eigensolvers, we could consider
# dropping jit from the input arguments of many of the make... functions.
# The eigsh eigensolvers need jit as an input argument to perform internal
# compilation of matrix-vector and matrix-matrix products passed to
# eigsh. However, other make... functions are JAX-native and can be jitted
# at the call site as needed (e.g., in the various compute... functions).
import jax
import jax.numpy as jnp

# import matplotlib.pyplot as plt
import nlsa.abstract_algebra as alg
import nlsa.function_algebra as fun
import nlsa.kernels as knl
import nlsa.jax.delays as dl
import nlsa.jax.dynamics as dyn
import nlsa.jax.matrix_algebra as mat
import nlsa.jax.vector_algebra as vec
import numpy as np
import scipy.sparse.linalg as sla
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from jax import Array, vmap
from jax.sharding import NamedSharding, Sharding
from jax.typing import ArrayLike, DTypeLike
from nlsa.jax.sharding import (
    EigShardings,
    NamedSharder,
    SvdShardings,
    make_eigh_with_sharding_constraints,
    make_svd_with_sharding_constraints,
    shardit,
)
from nlsa.jax.typing import PyTree, typestable_jit
from nlsa.jax.utils import batch_map, curried_batch_map
from nlsa.kernels import DmKernelPars, BsKernelPars, KernelPars, TunePars
from nlsa.typing import SliceItem
from nlsa.utils import has_two_args, swap_args
from scipy.sparse.linalg import LinearOperator
from typing import Literal, NamedTuple, Self, final
from collections.abc import Sequence

type R = Array  # Real number
type Rl = Array  # l-dimensional real vectors
type Rs = Array  # Collection of real numbers
type K = Array  # Scalar
type Ks = Array  # Collection of scalars
type V = Array  # Vector in L2
type Vtst = Array  # Vector in L2 with respect to the test data
type Vs = Array  # Collection of vectors
type X = Array  # Covariate
type Xs = Array  # Collection of covariates
type Xs_tst = Array  # Test covariate data
type Idx = int | Array  # Basis index
type Shape = tuple[int, ...]
type F[*Ss, T] = Callable[[*Ss], T]  # Shorthand for Callables


class TuneInfo(NamedTuple):
    """NamedTuple holding kernel tuning information in JAX arrays."""

    log10_bandwidths: Array
    """Array of trial kernel bandwidths."""

    est_dims: Array
    """Array of estimated dimensions based on trial bandwidths."""

    opt_bandwidth: Array
    """Optimal bandwidth from autotuning procedure."""

    opt_dim: Array
    """Optimal (maximum) dimension from autotuning procedure."""

    i_opt: Array
    """Index of optimal bandwidth in array of trial bandwidths."""

    bandwidth: Array
    """Selected bandwidth after scaling by user-defined factor."""

    dim: Array
    """Estimated dimension based on selected bandwidth."""

    vol: Array
    """Estimated manifold volume based on selected bandwidth."""

    kernel_vol: Array
    """Volume based on kernel integral."""

    def tabulate(
        self, name: str = "Kernel Tuning Info", show: bool = True
    ) -> str:
        """Create tabulated summary of the elements of a TuneInfo object."""
        return knl.tabulate_tune_info(self, name, show)


class KernelEigen(NamedTuple):
    """NamedTuple containing kernel spectral data."""

    evals: Ks
    """Kernel eigenvalues."""

    evecs: Vs
    """Kernel eigenvectors."""

    dual_evecs: Vs
    """Dual (left) kernel eigenvectors."""

    weights: V
    """Inner product weights that orthonormalize the eigenvectors."""

    bandwidth: K
    """Bandwidth parameter."""

    @property
    def num_eigs(
        self,
    ) -> int:
        """Return number of eigenvalues/eigenvectors in KernelEigenObject."""
        return knl.num_eigs_in_eigen(self)

    def isel(self, s: SliceItem) -> Self:
        """Slice a KernelEigen object."""
        return type(self)(
            evals=self.evals[s],
            evecs=self.evecs[s],
            dual_evecs=self.dual_evecs[s],
            weights=self.weights,
            bandwidth=self.bandwidth,
        )

    def tabulate(
        self,
        num_tabulate: int | None = None,
        headers: Sequence[str] | None = None,
        show: bool = True,
    ) -> str:
        """Tabulate the eigenvalues in a KernelEigen object."""
        return knl.tabulate_eigen(self, num_tabulate, headers, show)

    def inspect_array_shardings(
        self, callback: Callable[[Sharding], None] = print
    ) -> None:
        """Inspect array shardings in JIT-ted functions."""
        jax.debug.inspect_array_sharding(self.evecs, callback=callback)
        jax.debug.inspect_array_sharding(self.dual_evecs, callback=callback)
        jax.debug.inspect_array_sharding(self.evals, callback=callback)


class KernelEigenShardings(NamedTuple):
    """NamedTuple holding shardings for computation KernelEigen objects."""

    matrix: NamedSharding | None = None
    """Sharding of kernel matrix."""

    eigenvalues: NamedSharding | None = None
    """Sharding of eigenvalue array."""

    eigenvectors: NamedSharding | None = None
    """Sharding of eigenvector array."""

    weights: NamedSharding | None = None
    """Sharding of inner product weights array."""

    @classmethod
    def from_named_sharder[
        Shape: tuple[int, int, *tuple[int, ...]],
        AxisNames: str,
    ](cls, sharder: NamedSharder[Shape, AxisNames] | None) -> Self:
        """Create KernelEigenSharding object from NamedSharder."""
        if sharder is not None:
            y_sharding = sharder.sharding(None, sharder.axis_names[1])
            replicating = sharder.sharding(None)
            return cls(eigenvalues=replicating, eigenvectors=y_sharding)
        else:
            return cls()

    @property
    def shard_kernel_eigen(
        self,
    ) -> Callable[[KernelEigen], KernelEigen]:
        """Shard KernelEigen objects."""

        def shard(
            kernel_eigen: KernelEigen,
        ) -> KernelEigen:
            return KernelEigen(
                evals=jax.device_put(
                    kernel_eigen.evals, device=self.eigenvalues
                ),
                evecs=jax.device_put(
                    kernel_eigen.evecs, device=self.eigenvectors
                ),
                dual_evecs=jax.device_put(
                    kernel_eigen.dual_evecs, device=self.eigenvectors
                ),
                weights=jax.device_put(
                    kernel_eigen.weights, device=self.weights
                ),
                bandwidth=kernel_eigen.bandwidth,
            )

        return shard


@final
@dataclass(frozen=True, slots=True)
class KernelEigenbasis(knl.ImplementsKernelEigenbasis[X, K, K, V, Ks, Idx]):
    """Dataclass implementing frame operators for kernel eigenbasis."""

    dim: int
    """Number of eigenfunctions."""

    anal: Callable[[V], Ks]
    """Analysis operator."""

    dual_anal: Callable[[V], Ks]
    """Dual analysis operator."""

    synth: Callable[[Ks], V]
    """Synthesis operator."""

    dual_synth: Callable[[Ks], V]
    """Dual synthesis operator."""

    fn_anal: Callable[[F[X, K]], Ks]
    """Function analysis operator."""

    dual_fn_anal: Callable[[F[X, K]], Ks]
    """Dual function analysis operator."""

    fn_synth: Callable[[Ks], F[X, K]]
    """Function synthesis operator."""

    dual_fn_synth: Callable[[Ks], F[X, K]]
    """Dual function synthesis operator."""

    vec: Callable[[Idx], V]
    """Basis vectors."""

    dual_vec: Callable[[Idx], V]
    """Dual basis vectors."""

    fn: Callable[[Idx], F[X, K]]
    """Function representatives of basis vectors."""

    dual_fn: Callable[[Idx], F[X, K]]
    """Function representatives of dual basis vectors."""

    spec: Ks
    """Kernel operator spectrum (set of eigenvalues)."""

    lapl_spec: Ks
    """Laplace spectrum."""

    evl: Callable[[Idx], K]
    """Kernel eigenvalues."""

    lapl_evl: Callable[[Idx], K]
    """Laplacian eigenvalues."""


def _tune_bandwidth_from_kernel_family(
    pars: TunePars,
    l2x: alg.ImplementsL2FnAlgebra[X, R, V, R],
    kernel_family: Callable[[R], F[X, X, R]],
) -> TuneInfo:
    """Compute optimal bandwidth for bandwidth-parameterized kernel family."""
    log10_bandwidths = jnp.linspace(
        pars.log10_bandwidth_lims[0],
        pars.log10_bandwidth_lims[1],
        pars.num_bandwidths,
    )
    kernel_dim = knl.make_tuning_objective_from_kernel_family(
        l2x,
        kernel_family,
        grad=jax.grad,
    )
    est_dims = batch_map(kernel_dim, batch_size=pars.bandwidth_batch_size)(
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
        t_heat=bandwidth**2 / 4,
    )
    kernel_vol = knl.kernel_vol(l2x, kernel)
    return TuneInfo(
        log10_bandwidths=log10_bandwidths,
        est_dims=est_dims,
        opt_bandwidth=opt_bandwidth,
        opt_dim=opt_dim,
        i_opt=i_opt,
        bandwidth=bandwidth,
        dim=dim,
        vol=vol,
        kernel_vol=kernel_vol,
    )


def _tune_bandwidth_from_shape_function(
    pars: TunePars,
    l2x: alg.ImplementsL2FnAlgebra[X, K, V, K],
    shape_func: Callable[[K], K],
    neg_grad_shape_func: Callable[[K], K],
    sqdist: Callable[[X, X], K],
) -> TuneInfo:
    """Compute optimal bandwidth for RBF kernel family."""
    log10_bandwidths = jnp.linspace(
        pars.log10_bandwidth_lims[0],
        pars.log10_bandwidth_lims[1],
        pars.num_bandwidths,
    )
    kernel_family = knl.make_rbf_kernel_family(l2x, shape_func, sqdist)
    kernel_dim = knl.make_tuning_objective_from_shape_function(
        l2x,
        shape_func,
        neg_grad_shape_func,
        sqdist,
    )
    est_dims = batch_map(kernel_dim, batch_size=pars.bandwidth_batch_size)(
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
        t_heat=bandwidth**2 / 4,
    )
    kernel_vol = knl.kernel_vol(l2x, kernel)
    return TuneInfo(
        log10_bandwidths=log10_bandwidths,
        est_dims=est_dims,
        opt_bandwidth=opt_bandwidth,
        opt_dim=opt_dim,
        i_opt=i_opt,
        bandwidth=bandwidth,
        dim=dim,
        vol=vol,
        kernel_vol=kernel_vol,
    )


# TODO: Generalize X to PyTree
def make_bandwidth_tuner[Data: PyTree](
    pars: TunePars,
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[X, K, V, K]],
    shape_func: Callable[[K], K],
    sqdist: Callable[[X, X], K] | Callable[[Data, X, X], K],
    neg_grad_shape_func: Callable[[K], K] | None = None,
) -> Callable[[Data], TuneInfo]:
    """Make kernel tuning function from shape function and square distance."""

    def tune(data: Data) -> TuneInfo:
        l2x = impl_l2(data)
        if has_two_args(sqdist):
            _sqdist = sqdist
        else:
            _sqdist = partial(sqdist, data)
        if neg_grad_shape_func is not None:
            tune_info = _tune_bandwidth_from_shape_function(
                pars, l2x, shape_func, neg_grad_shape_func, _sqdist
            )
        else:
            kernel_family = knl.make_rbf_kernel_family(
                l2x.scl, shape_func, _sqdist
            )
            tune_info = _tune_bandwidth_from_kernel_family(
                pars, l2x, kernel_family
            )

        return tune_info

    return tune


def tune_bandwidth[Data: PyTree](
    pars: TunePars,
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[X, K, V, K]],
    shape_func: Callable[[K], K],
    sqdist: Callable[[X, X], K] | Callable[[Data, X, X], K],
    data: Data,
    neg_grad_shape_func: Callable[[K], K] | None = None,
    jit: bool = True,
) -> TuneInfo:
    """Tune kernel bandwidth."""
    tune = make_bandwidth_tuner(
        pars, impl_l2, shape_func, sqdist, neg_grad_shape_func
    )
    if jit:
        return typestable_jit(tune)(data)
    return tune(data)


class _DmSymOperatorSpectrum(NamedTuple):
    """NamedTuple holding symmetric diffusion maps operator spectral data."""

    evals: Array
    """Kernel eigenvalues."""

    evecs: Array
    """Kernel eigenvectors."""


def _from_dm_sym_operator_spectrum(
    l2x: alg.ImplementsDimensionedL2FnAlgebra[X, R, V, R],
    spec: _DmSymOperatorSpectrum,
    bandwidth: R,
    num_eigs: int | None = None,
    out_shardings: KernelEigenShardings = KernelEigenShardings(),
) -> KernelEigen:
    """Convert _DmSymOperatorSpectrum to KerneEigen."""
    num_samples = l2x.dim
    norm = vmap(l2x.norm)
    unsorted_evals, unsorted_evecs = spec
    if num_eigs is None:
        num_eigs = len(unsorted_evals)
    isort = jnp.argsort(unsorted_evals)[::-1][:num_eigs]
    lambs = unsorted_evals[isort]
    sqrt_mus = jnp.abs(unsorted_evecs[:, isort[0]])
    scl = jnp.sign(unsorted_evecs[0, isort[0]])
    phis = unsorted_evecs[:, isort].T / (scl * sqrt_mus)
    phi_duals = unsorted_evecs[:, isort].T * num_samples * (scl * sqrt_mus)

    phi_norms = norm(phis)[:, jnp.newaxis]
    if out_shardings.eigenvalues is not None:
        lambs = jax.lax.with_sharding_constraint(
            lambs, shardings=out_shardings.eigenvalues
        )
    if out_shardings.eigenvectors is not None:
        sqrt_mus = jax.lax.with_sharding_constraint(
            sqrt_mus, shardings=out_shardings.weights
        )
        phis = jax.lax.with_sharding_constraint(
            phis, shardings=out_shardings.eigenvectors
        )
        phi_duals = jax.lax.with_sharding_constraint(
            phi_duals, shardings=out_shardings.eigenvectors
        )
    eigen = KernelEigen(
        evals=lambs,
        evecs=phis / phi_norms,
        dual_evecs=phi_duals * phi_norms,
        weights=sqrt_mus**2,
        bandwidth=bandwidth,
    )
    return eigen


def make_eigh_dm_eigensolver[Data: PyTree](
    impl_l2: Callable[
        [Data], alg.ImplementsDimensionedL2FnAlgebra[X, R, V, R]
    ],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    bandwidth: R,
    normalization: Literal["laplace", "fokkerplanck"] | None,
    num_eigs: int | None = None,
    dtype: DTypeLike | None = None,
    batch_size: int | None = None,
    jit: bool = False,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> Callable[[Data], KernelEigen]:
    """Make eigensolver for diffusion-maps normalized operaror using eigh."""
    # WARNING: In recent versions of the code we are sharding the eigenvectors
    # in KernelEigen along rows. Assigning eig_shardings based on
    # sharding likely leads to sharding inconsistencies. A possible solution
    # would be to rename shardings to out_shardings and create a separate
    # eig_shardings input argument to pass the correct shardings to
    # make_eig_with_sharding_constraints.
    eig_shardings = EigShardings(
        eigenvalues=shardings.eigenvalues, eigenvectors=shardings.matrix
    )
    eig = make_eigh_with_sharding_constraints(shardings=eig_shardings)

    def eigensolve(data: Data) -> KernelEigen:
        l2x = impl_l2(data)
        if has_two_args(kernel):
            _kernel = kernel
        else:
            _kernel = partial(kernel, data)
        match normalization:
            case "laplace":
                dm_kernel = knl.dmsym_normalize(l2x, _kernel, alpha="1")
            case "fokkerplanck":
                dm_kernel = knl.dmsym_normalize(l2x, _kernel, alpha="0.5")
            case None:
                dm_kernel = _kernel
        kernel_op = fun.compose(
            l2x.incl, knl.make_integral_operator(l2x, dm_kernel)
        )
        a = mat.materialize_in_std_basis(
            kernel_op,
            in_dim=l2x.dim,
            dtype=dtype,
            batch_size=batch_size,
            out_sharding=shardings.matrix,
        )
        spec = _DmSymOperatorSpectrum(*eig(a))
        eigen = _from_dm_sym_operator_spectrum(
            l2x, spec, bandwidth, num_eigs, shardings
        )
        return eigen

    if jit:
        return typestable_jit(eigensolve)
    return eigensolve


def make_eigsh_dm_eigensolver[Data: PyTree](
    impl_l2: Callable[
        [Data], alg.ImplementsDimensionedL2FnAlgebra[X, R, V, R]
    ],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    bandwidth: R,
    normalization: Literal["laplace", "fokkerplanck"] | None,
    num_samples: int,
    num_eigs: int,
    dtype: DTypeLike | None = None,
    jit: bool = True,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> Callable[[Data], KernelEigen]:
    """Make eigensolver for diffusion maps normalized operaror using eigsh."""
    kernel_op = knl.make_data_driven_dmsym_kernel_op(
        impl_l2, kernel, normalization
    )
    if jit:
        kernel_op = typestable_jit(kernel_op)
    to_device = partial(jnp.asarray, device=shardings.weights)
    # NOTE: We are using shardings.weights as opposed to shardings.eigenvectors
    # since KernelEigen stores singular vectors in row-major format, but
    # eigs stores singular vectors in column-major format.

    def eigensolve(data: Data) -> KernelEigen:

        matvec: Callable[[ArrayLike], Array] = fun.compose(
            partial(kernel_op, data),
            to_device,
        )
        a = LinearOperator(
            shape=(num_samples, num_samples),
            dtype=np.dtype(dtype),
            matvec=matvec,
        )

        def from_dm_sym_operator_spectrum(
            data: Data, spec: _DmSymOperatorSpectrum
        ) -> KernelEigen:
            l2x = impl_l2(data)
            eigen = _from_dm_sym_operator_spectrum(
                l2x, spec, bandwidth, num_eigs, out_shardings=shardings
            )
            return eigen

        evals, evecs = sla.eigsh(a, num_eigs, which="LA")
        evals = jnp.asarray(evals, dtype=dtype)
        evecs = jnp.asarray(evecs, dtype=dtype, device=shardings.weights)
        spec = _DmSymOperatorSpectrum(evals=evals, evecs=evecs)

        if jit:
            return typestable_jit(from_dm_sym_operator_spectrum)(data, spec)
        return from_dm_sym_operator_spectrum(data, spec)

    return eigensolve


def make_dm_eigensolver[Data: PyTree](
    impl_l2: Callable[
        [Data], alg.ImplementsDimensionedL2FnAlgebra[X, R, V, R]
    ],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    bandwidth: R,
    normalization: Literal["laplace", "fokkerplanck"] | None,
    solver: Literal["eigh", "eigsh"],
    num_samples: int | None = None,
    num_eigs: int | None = None,
    dtype: DTypeLike | None = None,
    batch_size: int | None = None,
    jit: bool = True,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> Callable[[Data], KernelEigen]:
    """Make eigensolver for diffusion-maps normalized kernel operator."""
    match solver:
        case "eigh":
            eigensolve = make_eigh_dm_eigensolver(
                impl_l2,
                kernel,
                bandwidth,
                normalization,
                num_eigs,
                dtype,
                batch_size,
                jit,
                shardings,
            )
        case "eigsh":
            assert num_samples is not None
            assert num_eigs is not None
            eigensolve = make_eigsh_dm_eigensolver(
                impl_l2,
                kernel,
                bandwidth,
                normalization,
                num_samples,
                num_eigs,
                dtype,
                jit,
                shardings,
            )
    return eigensolve


class _BsOperatorSpectrum(NamedTuple):
    """NamedTuple holding bistochastic kernel operator spectral data."""

    left_sing_vecs: Array
    """Left singular vectors"""

    sing_vals: Array
    """Singular values of asymmetric kernel operator."""

    right_sing_vecs: Array
    """Right singular vectors."""


def _from_bs_operator_spectrum(
    l2x: alg.ImplementsDimensionedL2FnAlgebra[X, R, V, R],
    spec: _BsOperatorSpectrum,
    num_eigs: int | None,
    bandwidth: R,
    out_shardings: KernelEigenShardings = KernelEigenShardings(),
) -> KernelEigen:
    """Convert _BsOperatorSpectrum to KerneEigen."""
    num_samples = l2x.dim
    norm = vmap(l2x.norm)
    unsorted_evecs = spec.left_sing_vecs
    unsorted_evals = spec.sing_vals**2
    unsorted_dual_evecs = spec.right_sing_vecs
    if num_eigs is None:
        num_eigs = len(unsorted_evals)
    isort = jnp.argsort(unsorted_evals)[::-1][:num_eigs]
    lambs = unsorted_evals[isort]
    sqrt_mus = jnp.abs(unsorted_evecs[:, isort[0]])
    scl = jnp.sign(unsorted_evecs[0, isort[0]])
    phis = unsorted_evecs[:, isort].T / (scl * sqrt_mus)
    phi_duals = unsorted_dual_evecs[isort] * num_samples * scl * sqrt_mus

    phi_norms = norm(phis)[:, jnp.newaxis]
    if out_shardings.eigenvalues is not None:
        lambs = jax.lax.with_sharding_constraint(
            lambs, shardings=out_shardings.eigenvalues
        )
    if out_shardings.eigenvectors is not None:
        sqrt_mus = jax.lax.with_sharding_constraint(
            sqrt_mus, shardings=out_shardings.weights
        )
        phis = jax.lax.with_sharding_constraint(
            phis, shardings=out_shardings.eigenvectors
        )
        phi_duals = jax.lax.with_sharding_constraint(
            phi_duals, shardings=out_shardings.eigenvectors
        )
    eigen = KernelEigen(
        evals=lambs,
        evecs=phis / phi_norms,
        dual_evecs=phi_duals * phi_norms,
        weights=sqrt_mus**2,
        bandwidth=bandwidth,
    )
    return eigen


def make_svd_bs_eigensolver[Data: PyTree](
    impl_l2: Callable[
        [Data], alg.ImplementsDimensionedL2FnAlgebra[X, R, V, R]
    ],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    bandwidth: R,
    num_eigs: int | None = None,
    dtype: DTypeLike | None = None,
    batch_size: int | None = None,
    jit: bool = True,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> Callable[[Data], KernelEigen]:
    """Make SVD solver for bistochastic kernel operator using svd."""
    # WARNING: In recent versions of the code we are sharding the singular
    # vectors in KernelEigen along rows. Assigning svd_shardings based on
    # sharding likely leads to sharding inconsistencies. A possible solution
    # would be to rename shardings to out_shardings and create a separate
    # svd_shardings input argument to pass the correct shardings to
    # make_svd_with_sharding_constraints.
    svd_shardings = SvdShardings(
        left_sing_vectors=shardings.matrix,
        sing_values=shardings.eigenvalues,
        right_sing_vectors=shardings.matrix,
    )
    svd = make_svd_with_sharding_constraints(shardings=svd_shardings)

    def svdsolve(data: Data) -> KernelEigen:
        l2x = impl_l2(data)
        if has_two_args(kernel):
            _kernel = kernel
        else:
            _kernel = partial(kernel, data)
        bs_kernel = knl.bs_normalize(l2x, _kernel)
        kernel_op = fun.compose(
            l2x.incl, knl.make_integral_operator(l2x, bs_kernel)
        )
        a = mat.materialize_in_std_basis(
            kernel_op,
            in_dim=l2x.dim,
            dtype=dtype,
            batch_size=batch_size,
            out_sharding=shardings.matrix,
        )
        spec = _BsOperatorSpectrum(*svd(a))
        eigen = _from_bs_operator_spectrum(
            l2x, spec, num_eigs, bandwidth, shardings
        )
        return eigen

    if jit:
        return typestable_jit(svdsolve)
    return svdsolve


def make_svds_bs_eigensolver[Data: PyTree](
    impl_l2: Callable[
        [Data], alg.ImplementsDimensionedL2FnAlgebra[X, R, V, R]
    ],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    bandwidth: R,
    num_samples: int,
    num_eigs: int,
    dtype: DTypeLike | None = None,
    batch_size: int | None = None,
    jit: bool = True,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> Callable[[Data], KernelEigen]:
    """Make SVD solver for bistochastic kernel operator using svds."""
    kernel_op = knl.make_data_driven_bs_kernel_op(impl_l2, kernel)
    adj_kernel_op = knl.make_data_driven_bs_kernel_op(
        impl_l2, kernel, adj=True
    )
    if jit:
        kernel_op = typestable_jit(kernel_op)
        adj_kernel_op = typestable_jit(adj_kernel_op)
    to_device = partial(jnp.asarray, device=shardings.weights)
    # NOTE: We are using shardings.weights as opposed to shardings.eigenvectors
    # since KernelEigen stores singular vectors in row-major format, but
    # svds stores singular vectors in column-major format.

    def svdsolve(data: Data) -> KernelEigen:
        matvec: Callable[[ArrayLike], Array] = fun.compose(
            partial(typestable_jit(kernel_op), data),
            to_device,
        )
        rmatvec: Callable[[ArrayLike], Array] = fun.compose(
            partial(typestable_jit(adj_kernel_op), data),
            to_device,
        )
        matmat: Callable[[ArrayLike], Array] = fun.compose(
            partial(
                typestable_jit(
                    shardit(
                        curried_batch_map(
                            kernel_op,
                            in_axis=1,
                            out_axis=1,
                            batch_size=batch_size,
                        ),
                        sharding=shardings.eigenvectors,
                    )
                ),
                data,
            ),
            to_device,
        )
        rmatmat: Callable[[ArrayLike], Array] = fun.compose(
            partial(
                typestable_jit(
                    shardit(
                        curried_batch_map(
                            adj_kernel_op,
                            in_axis=1,
                            out_axis=1,
                            batch_size=batch_size,
                        ),
                        sharding=shardings.eigenvectors,
                    )
                ),
                data,
            ),
            to_device,
        )
        a = LinearOperator(
            shape=(num_samples, num_samples),
            dtype=np.dtype(dtype),
            matvec=matvec,
            rmatvec=rmatvec,
            matmat=matmat,
            rmatmat=rmatmat,
        )

        def from_bs_operator_spectrum(
            data: Data, spec: _BsOperatorSpectrum
        ) -> KernelEigen:
            l2x = impl_l2(data)
            eigen = _from_bs_operator_spectrum(
                l2x, spec, num_eigs, bandwidth, out_shardings=shardings
            )
            return eigen

        left_sing_vecs, sing_vals, right_sing_vecs = sla.svds(a, num_eigs)
        left_sing_vecs = jnp.asarray(left_sing_vecs, dtype=dtype)
        sing_vals = jnp.asarray(
            sing_vals, dtype=dtype, device=shardings.eigenvectors
        )
        right_sing_vecs = jnp.asarray(
            right_sing_vecs, dtype=dtype, device=shardings.eigenvectors
        )
        spec = _BsOperatorSpectrum(
            left_sing_vecs=left_sing_vecs,
            sing_vals=sing_vals,
            right_sing_vecs=right_sing_vecs,
        )

        if jit:
            return typestable_jit(from_bs_operator_spectrum)(data, spec)
        return from_bs_operator_spectrum(data, spec)

    return svdsolve


def make_bs_eigensolver[Data: PyTree](
    impl_l2: Callable[
        [Data], alg.ImplementsDimensionedL2FnAlgebra[X, R, V, R]
    ],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    bandwidth: R,
    solver: Literal["svd", "svds"],
    num_samples: int | None = None,
    num_eigs: int | None = None,
    dtype: DTypeLike | None = None,
    batch_size: int | None = None,
    jit: bool = True,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> Callable[[Data], KernelEigen]:
    """Solve kernel eigenvalue problem for bistochastic normalization."""
    match solver:
        case "svd":
            eigensolve = make_svd_bs_eigensolver(
                impl_l2,
                kernel,
                bandwidth,
                num_eigs,
                dtype,
                batch_size,
                jit,
                shardings,
            )
        case "svds":
            assert num_samples is not None
            assert num_eigs is not None
            eigensolve = make_svds_bs_eigensolver(
                impl_l2,
                kernel,
                bandwidth,
                num_samples,
                num_eigs,
                dtype,
                batch_size,
                jit,
                shardings,
            )
    return eigensolve


def make_eigensolver[Data: PyTree](
    pars: KernelPars,
    impl_l2: Callable[
        [Data], alg.ImplementsDimensionedL2FnAlgebra[X, R, V, R]
    ],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    bandwidth: R,
    num_samples: int | None = None,
    dtype: DTypeLike | None = None,
    jit: bool = True,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> Callable[[Data], KernelEigen]:
    """Make eigensolver for DM or BS kernels."""
    match pars:
        case DmKernelPars():
            eigensolve = make_dm_eigensolver(
                impl_l2,
                kernel,
                bandwidth,
                pars.normalization,
                pars.eigensolver,
                num_samples,
                pars.num_eigs,
                dtype,
                pars.batch_size,
                jit,
                shardings,
            )
        case BsKernelPars():
            eigensolve = make_bs_eigensolver(
                impl_l2,
                kernel,
                bandwidth,
                pars.eigensolver,
                num_samples,
                pars.num_eigs,
                dtype,
                pars.batch_size,
                jit,
                shardings,
            )
    return eigensolve


def compute_eigen[Data: PyTree](
    kernel_pars: KernelPars,
    impl_l2: Callable[
        [Data], alg.ImplementsDimensionedL2FnAlgebra[X, R, V, R]
    ],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    data: Data,
    bandwidth: R,
    num_samples: int | None = None,
    dtype: DTypeLike | None = None,
    jit: bool = True,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> KernelEigen:
    """Solve eigenvalue prblem for DM or BS kernels."""
    eigensolve = make_eigensolver(
        kernel_pars,
        impl_l2,
        kernel,
        bandwidth,
        num_samples,
        dtype,
        jit,
        shardings,
    )
    return eigensolve(data)


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


def make_eigenbasis_dm(
    l2x: alg.ImplementsL2FnAlgebra[X, R, V, R],
    kernel: Callable[[X, X], R],
    normalization: Literal["laplace", "fokkerplanck"] | None,
    laplacian_method: Literal["lin", "log", "inv"],
    kernel_eigen: knl.ImplementsKernelEigen[R, Rs, V, Vs],
) -> KernelEigenbasis:
    """Make kernel eigenbasis for diffusion maps kernels."""
    match normalization:
        case "laplace":
            extension_kernel = knl.dm_normalize(l2x, kernel, alpha="1")
        case "fokkerplanck":
            extension_kernel = knl.dm_normalize(l2x, kernel, alpha="0.5")
        case None:
            extension_kernel = kernel
    lapl_spec = to_laplace_eigenvalues(
        kernel_eigen.evals,
        kernel_eigen.bandwidth,
        method=laplacian_method,
    )
    extension_op = knl.make_integral_operator(l2x, extension_kernel)
    dual_extension_op = knl.make_integral_operator(
        l2x, swap_args(extension_kernel)
    )

    def vc(i: Idx) -> V:
        return kernel_eigen.evecs[i]

    def dual_vc(i: Idx) -> V:
        return kernel_eigen.dual_evecs[i]

    def evl(i: Idx) -> R:
        return kernel_eigen.evals[i]

    def lapl_evl(i: Idx) -> R:
        return lapl_spec[i]

    def fn(i: Idx) -> Callable[[X], R]:
        return extension_op(kernel_eigen.evecs[i] / kernel_eigen.evals[i])

    def dual_fn(i: Idx) -> Callable[[X], R]:
        return dual_extension_op(
            kernel_eigen.dual_evecs[i] / kernel_eigen.evals[i]
        )

    @partial(vmap, in_axes=(0, None))
    def anal_eval(i: Idx, v: V) -> R:
        return l2x.innerp(kernel_eigen.dual_evecs[i], v)

    @partial(vmap, in_axes=(0, None))
    def dual_anal_eval(i: Idx, v: V) -> R:
        return l2x.innerp(kernel_eigen.evecs[i], v)

    @partial(vmap, in_axes=(0, None))
    def fn_eval(i: Idx, x: X) -> R:
        return fn(i)(x)

    @partial(vmap, in_axes=(0, None))
    def dual_fn_eval(i: Idx, x: X) -> R:
        return dual_fn(i)(x)

    num_eigs = knl.num_eigs_in_eigen(kernel_eigen)
    idxs = jnp.arange(num_eigs)
    anal = partial(anal_eval, idxs)
    dual_anal = partial(dual_anal_eval, idxs)
    fn_anal = fun.compose(anal, l2x.incl)
    dual_fn_anal = fun.compose(dual_anal, l2x.incl)
    synth = vec.make_synthesis_operator(kernel_eigen.evecs, idxs)
    dual_synth = vec.make_synthesis_operator(kernel_eigen.dual_evecs, idxs)
    fn_synth = vec.make_fn_synthesis_operator(partial(fn_eval, idxs))
    dual_fn_synth = vec.make_fn_synthesis_operator(partial(dual_fn_eval, idxs))
    spec = kernel_eigen.evals[idxs]
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


def make_eigenbasis_bs(
    l2x: alg.ImplementsL2FnAlgebra[X, R, V, R],
    kernel: Callable[[X, X], R],
    laplacian_method: Literal["lin", "log", "inv"],
    kernel_eigen: knl.ImplementsKernelEigen[R, Rs, V, Vs],
) -> KernelEigenbasis:
    """Make kernel eigenbasis for bistochastic kernels."""
    lapl_spec = to_laplace_eigenvalues(
        kernel_eigen.evals,
        kernel_eigen.bandwidth,
        method=laplacian_method,
    )
    extension_kernel = knl.bs_normalize(l2x, kernel)
    extension_op = knl.make_integral_operator(l2x, extension_kernel)

    def vc(i: Idx) -> V:
        return kernel_eigen.evecs[i]

    def evl(i: Idx) -> R:
        return kernel_eigen.evals[i]

    def lapl_evl(i: Idx) -> R:
        return lapl_spec[i]

    def fn(i: Idx) -> F[X, R]:
        return extension_op(
            kernel_eigen.dual_evecs[i] / jnp.sqrt(kernel_eigen.evals[i])
        )

    @partial(vmap, in_axes=(0, None))
    def anal_eval(i: Idx, v: V) -> R:
        return l2x.innerp(kernel_eigen.evecs[i], v)

    @partial(vmap, in_axes=(0, None))
    def fn_eval(i: Idx, x: X) -> R:
        return fn(i)(x)

    num_eigs = knl.num_eigs_in_eigen(kernel_eigen)
    idxs = jnp.arange(num_eigs)
    anal = partial(anal_eval, idxs)
    fn_anal = fun.compose(anal, l2x.incl)
    synth = vec.make_synthesis_operator(kernel_eigen.evecs, idxs)
    fn_synth = vec.make_fn_synthesis_operator(partial(fn_eval, idxs))
    spec = kernel_eigen.evals[idxs]
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


def make_eigenbasis(
    kernel_pars: KernelPars,
    l2x: alg.ImplementsL2FnAlgebra[X, R, V, R],
    kernel: Callable[[X, X], R],
    kernel_eigen: knl.ImplementsKernelEigen[R, Rs, V, Vs],
    laplacian_method: Literal["lin", "log", "inv"] = "log",
) -> KernelEigenbasis:
    """Make kernel eigenbasis."""
    match kernel_pars:
        case DmKernelPars():
            basis = make_eigenbasis_dm(
                l2x,
                kernel,
                kernel_pars.normalization,
                laplacian_method,
                kernel_eigen,
            )
        case BsKernelPars():
            basis = make_eigenbasis_bs(
                l2x, kernel, laplacian_method, kernel_eigen
            )
    return basis


def make_data_driven_eigenbasis[Data: PyTree](
    kernel_pars: KernelPars,
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[X, R, V, R]],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    which_eigs: int | tuple[int, int] | list[int] | None = None,
) -> Callable[
    [Data, knl.ImplementsSliceableKernelEigen[R, Rs, V, Vs]], KernelEigenbasis
]:
    """Make data-driven kernel eigenbasis builder."""

    def _make_eigenbasis(
        data: Data,
        kernel_eigen: knl.ImplementsSliceableKernelEigen[R, Rs, V, Vs],
    ) -> KernelEigenbasis:
        l2x = impl_l2(data)
        if has_two_args(kernel):
            _kernel = kernel
        else:
            _kernel = partial(kernel, data)
        _kernel_eigen = knl.slice_eigen(kernel_eigen, which_eigs)
        return make_eigenbasis(kernel_pars, l2x, _kernel, _kernel_eigen)

    return _make_eigenbasis


def make_kaf_analysis_operator[
    Data: PyTree,
    Eigen: knl.ImplementsKernelEigen[R, Rs, V, Vs],
](
    impl_basis: Callable[
        [Data, Eigen],
        knl.ImplementsKernelEigenbasis[X, R, V, R, Rs, Idx],
    ],
    num_steps: int,
    which_samples: tuple[int, int] | None = None,
) -> Callable[[Data, V, Eigen], Rs]:
    """Make analysis operator for kernel analog forecast."""

    def anal(
        data: Data,
        response: V,
        kernel_eigen: Eigen,
    ) -> Rs:
        if which_samples is not None:
            i0 = which_samples[0]
            i1 = which_samples[1]
        else:
            i0 = 0
            i1 = len(response)
        basis = impl_basis(data, kernel_eigen)
        anal = vmap(basis.anal)
        time_shifted_response = dl.hankel(
            response[i0:i1], num_delays=num_steps, delay_axis=0
        )
        return anal(time_shifted_response)

    return anal


def make_kaf_prediction_function[
    Data: PyTree,
    Eigen: knl.ImplementsKernelEigen[R, Rs, V, Vs],
    TestData: PyTree,
](
    impl_basis: Callable[
        [Data, Eigen],
        knl.ImplementsKernelEigenbasis[X, R, V, R, Rs, Idx],
    ],
    impl_l2_tst: Callable[
        [TestData], alg.ImplementsL2FnAlgebra[X, R, Vtst, R]
    ],
) -> Callable[[Data, Eigen, Rs, TestData], Vtst]:
    """Make prediction function for kernel analog forecast."""

    def predict(
        data: Data, kernel_eigen: Eigen, coeffs: Rs, test_data: TestData
    ) -> Vtst:
        basis = impl_basis(data, kernel_eigen)
        l2x_tst = impl_l2_tst(test_data)

        @partial(vmap, in_axes=(0, None))
        def _predict(cs: Rs, x: X) -> R:
            return basis.fn_synth(cs)(x)

        return l2x_tst.incl(partial(_predict, coeffs))

    return predict


def compute_kaf_preds[Data: PyTree, TestData: PyTree](
    kernel_pars: KernelPars,
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[X, R, V, R]],
    train_data: Data,
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    kernel_eigen: KernelEigen,
    coeffs: Rs,
    impl_l2_tst: Callable[
        [TestData], alg.ImplementsL2FnAlgebra[X, R, Vtst, R]
    ],
    test_data: TestData,
    which_eigs: int | tuple[int, int] | list[int] | None = None,
    jit: bool = True,
) -> Rs:
    """Compute KAF predictions."""
    impl_basis = make_data_driven_eigenbasis(
        kernel_pars, impl_l2, kernel, which_eigs
    )
    predict: Callable[[Data, KernelEigen, Rs, TestData], Vtst] = (
        make_kaf_prediction_function(impl_basis, impl_l2_tst)
    )
    if jit:
        predict = typestable_jit(predict)
    return predict(train_data, kernel_eigen, coeffs, test_data)


def make_iterative_kaf_analysis_operator[
    Data: PyTree,
    Eigen: knl.ImplementsKernelEigen[R, Rs, V, Vs],
](
    impl_basis: Callable[
        [Data, Eigen],
        knl.ImplementsKernelEigenbasis[X, R, V, R, Rs, Idx],
    ],
    which_samples: tuple[int, int] | None = None,
) -> Callable[[Data, Rs, Eigen], Rs]:
    """Make analysis operator for kernel analog forecast."""

    def anal(data: Data, covariates: Xs, kernel_eigen: Eigen) -> Rs:
        if which_samples is not None:
            i0 = which_samples[0]
            i1 = which_samples[1]
        else:
            i0 = 1
            i1 = len(covariates)
        basis = impl_basis(data, kernel_eigen)
        anal = vmap(basis.anal, in_axes=1)
        return anal(covariates[i0:i1])

    return anal


def make_iterative_kaf_prediction_function[
    Data: PyTree,
    Eigen: knl.ImplementsKernelEigen[R, Rs, V, Vs],
    TestData: PyTree,
](
    impl_basis: Callable[
        [Data, Eigen],
        knl.ImplementsKernelEigenbasis[X, R, V, R, Rs, Idx],
    ],
    impl_l2_tst: Callable[
        [TestData], alg.ImplementsL2FnAlgebra[X, X, Vtst, R]
    ],
    num_steps: int,
) -> Callable[[Data, Eigen, Rs, TestData], Vtst]:
    """Make prediction function for iterative KAF."""

    def predict(
        data: Data, kernel_eigen: Eigen, coeffs: Rs, test_data: TestData
    ) -> Xs:
        basis = impl_basis(data, kernel_eigen)
        l2x_tst = impl_l2_tst(test_data)

        @partial(vmap, in_axes=(0, None))
        def predict_snapshot(cs: Rs, x: X) -> X:
            """Predict next snapshot."""
            return basis.fn_synth(cs)(x)

        _predict = dyn.make_fin_orbit(
            partial(predict_snapshot, coeffs), num_steps + 1
        )
        return l2x_tst.incl(_predict)

    return predict


def make_iterative_kaf_prediction_function_with_delays[
    Data: PyTree,
    Eigen: knl.ImplementsKernelEigen[R, Rs, V, Vs],
    TestData: PyTree,
](
    impl_basis: Callable[
        [Data, Eigen],
        knl.ImplementsKernelEigenbasis[X, R, V, R, Rs, Idx],
    ],
    impl_l2_tst: Callable[
        [TestData], alg.ImplementsL2FnAlgebra[X, X, Vtst, R]
    ],
    num_delays: int,
    num_steps: int,
    project: bool = True,
) -> Callable[[Data, Eigen, Rs, TestData], Vtst]:
    """Make single-step prediction function for iterative KAF."""

    def predict(
        data: Data,
        kernel_eigen: Eigen,
        coeffs: Rs,
        test_data: TestData,
    ) -> Vtst:
        basis = impl_basis(data, kernel_eigen)
        l2x_tst = impl_l2_tst(test_data)

        @partial(vmap, in_axes=(0, None))
        def predict_snapshot(cs: Rs, xs: Xs) -> X:
            """Predict next snapshot from delay-embedded data."""
            return basis.fn_synth(cs)(xs)

        def predict_window(cs: Rs, xs: Xs) -> Xs:
            """Predict next delay embedding window."""
            x_next = predict_snapshot(cs, xs)
            x_prev_unrolled = xs.reshape((num_delays + 1, -1))[1:]
            x_pred_unrolled = jnp.concatenate(
                (x_prev_unrolled, x_next[jnp.newaxis, :])
            )
            return jnp.hstack(x_pred_unrolled)

        _predict = dyn.make_fin_orbit(
            partial(predict_window, coeffs), num_steps + 1
        )
        preds = l2x_tst.incl(_predict)
        if project:
            num_samples = preds.shape[0]
            preds = preds.reshape(
                (
                    num_samples,
                    num_steps + 1,
                    num_delays + 1,
                    -1,
                )
            )[:, :, -1, :]

        return preds

    return predict


def compute_iterative_kaf_preds[Data: PyTree, TestData: PyTree](
    kernel_pars: KernelPars,
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[X, R, V, R]],
    train_data: Data,
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    kernel_eigen: KernelEigen,
    coeffs: Rs,
    impl_l2_tst: Callable[
        [TestData], alg.ImplementsL2FnAlgebra[X, R, Vtst, R]
    ],
    test_data: TestData,
    num_steps: int,
    num_delays: int | None,
    which_eigs: int | tuple[int, int] | list[int] | None = None,
    project: bool = True,
    jit: bool = True,
) -> Vtst:
    """Compute iterative KAF predictions of the covariate variables."""
    impl_basis = make_data_driven_eigenbasis(
        kernel_pars, impl_l2, kernel, which_eigs
    )
    predict: Callable[[Data, KernelEigen, Rs, TestData], Vtst]
    if num_delays is None or num_delays == 0:
        predict = make_iterative_kaf_prediction_function(
            impl_basis,
            impl_l2_tst,
            num_steps,
        )
    else:
        predict = make_iterative_kaf_prediction_function_with_delays(
            impl_basis, impl_l2_tst, num_delays, num_steps, project
        )
    if jit:
        predict = typestable_jit(predict)
    return predict(train_data, kernel_eigen, coeffs, test_data)

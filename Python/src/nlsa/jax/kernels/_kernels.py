"""Provide classes and functions for kernel computations in JAX."""

# TODO: With the exception of the eigsh eigensolvers, we could consider
# dropping jit from the input arguments of many of the make... functions.
# The eigsh eigensolvers need jit as an input argument to perform internal
# compilation of matrix-vector and matrix-matrix products passed to
# eigsh. However, other make... functions are JAX-native and can be jitted
# at the call site as needed (e.g., in the various compute... functions).
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
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
from functools import partial
from jax import Array, vmap
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
from nlsa.jax.typing import PyTree
from nlsa.jax.utils import batch_map, curried_batch_map
from nlsa.jax.vector_algebra import L2FnAlgebra
from nlsa.kernels import (
    DmKernelPars,
    BsKernelPars,
    KernelPars,
    KernelEigen,
    KernelEigenbasis,
    TuneInfo,
    TunePars,
)
from nlsa.utils import has_one_arg, has_two_args, swap_args
from scipy.sparse.linalg import LinearOperator
from typing import Literal, NamedTuple, Optional, Self

type R = Array  # Real number
type Rl = Array  # l-dimensional real vectors
type Rs = Array  # Collection of real numbers
type K = Array  # Scalar
type Ks = Array  # Collection of scalars
type V = Array  # Vector
type Vs = Array  # Collection of vectors
type Xs = Array  # Covariate data
type Xs_tst = Array  # Test covariate data
type Shape = tuple[int, ...]
type F[*Ss, T] = Callable[[*Ss], T]  # Shorthand for Callables


class KernelEigenShardings(NamedTuple):
    """NamedTuple holding shardings for computation KernelEigen objects."""

    matrix: Optional[NamedSharding] = None
    """Sharding of kernel matrix."""

    eigenvalues: Optional[NamedSharding] = None
    """Sharding of eigenvalue array."""

    eigenvectors: Optional[NamedSharding] = None
    """Sharding of eigenvector array."""

    weights: Optional[NamedSharding] = None
    """Sharding of inner product weights array."""

    @classmethod
    def from_named_sharder[
        Shape: tuple[int, int, *tuple[int, ...]],
        AxisNames: str,
    ](cls, sharder: Optional[NamedSharder[Shape, AxisNames]]) -> Self:
        """Create KernelEigenSharding object from NamedSharder."""
        if sharder is not None:
            y_sharding = sharder.sharding(sharder.axis_names[1])
            replicating = sharder.sharding(None)
            return cls(eigenvalues=replicating, eigenvectors=y_sharding)
        else:
            return cls()

    @property
    def shard_kernel_eigen(
        self,
    ) -> Callable[[KernelEigen[R, Rs, Vs, Vs]], KernelEigen[R, Rs, Vs, Vs]]:
        """Shard KernelEigen objects."""

        def shard(
            kernel_eigen: KernelEigen[R, Rs, Vs, Vs],
        ) -> KernelEigen[R, Rs, Vs, Vs]:
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


def _tune_bandwidth_from_kernel_family[X: Array](
    pars: TunePars,
    l2x: alg.ImplementsL2FnAlgebra[X, K, V, K],
    kernel_family: Callable[[R], F[X, X, R]],
) -> TuneInfo[Array, Array, Array]:
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


def _tune_bandwidth_from_shape_function[X: Array](
    pars: TunePars,
    l2x: alg.ImplementsL2FnAlgebra[X, K, V, K],
    shape_func: Callable[[K], K],
    neg_grad_shape_func: Callable[[K], K],
    sqdist: Callable[[X, X], K],
) -> TuneInfo[Array, Array, Array]:
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


# NOTE: In a posssible Mojo implementation, impl_l2, shape_func, etc. would
# be passed in as compile-time parameters, and the arguments of the
# resulting Callable would be regular run-time arguments.
def make_data_driven_bandwidth_function[X: Array, Data: PyTree](
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[X, K, V, K]],
    shape_func: Callable[[K], K],
    sqdist: F[X, X, K],
    tune_info: TuneInfo[Array, Array, Array],
) -> Callable[[Data, X], K]:
    """Make data-driven kernel bandwidth function."""

    def bandwidth_func(data: Data, x: X) -> K:
        l2x = impl_l2(data)
        kernel_family = knl.make_rbf_kernel_family(l2x.scl, shape_func, sqdist)
        f = knl.make_bandwidth_function(
            l2x,
            kernel_family(tune_info.bandwidth),
            dim=tune_info.dim,
            vol=tune_info.vol,
            normalization=tune_info.kernel_vol,
        )
        return f(x)

    return bandwidth_func


def make_data_driven_scaled_sqdist[X: Array, Data: PyTree](
    impl: alg.ImplementsRealScalarField[K],
    sqdist: Callable[[X, X], K] | Callable[[Data, X, X], K],
    bandwidth_func: Callable[[X], K] | Callable[[Data, X], K],
) -> Callable[[Data, X, X], K]:
    """Make data-driven scaled square distance from bandwidth function."""

    def scaled_sqdist(data: Data, x: X, y: X) -> K:
        if has_two_args(sqdist):
            _sqdist = sqdist
        else:
            _sqdist = partial(sqdist, data)
        if has_one_arg(bandwidth_func):
            _bandwidth_func = bandwidth_func
        else:
            _bandwidth_func = partial(bandwidth_func, data)
        _scaled_sqdist = knl.make_scaled_sqdist(impl, _sqdist, _bandwidth_func)
        return _scaled_sqdist(x, y)

    return scaled_sqdist


def make_data_driven_rbf_kernel[X: Array, Data: PyTree](
    impl: alg.ImplementsRealScalarField[K],
    shape_func: Callable[[K], K],
    sqdist: Callable[[X, X], K] | Callable[[Data, X, X], K],
    bandwidth: K,
) -> Callable[[Data, X, X], K]:
    """Make data-driven, bandwidth-paramterized RBF kernel."""

    def kernel(data: Data, x: X, y: X) -> K:
        if has_two_args(sqdist):
            _sqdist = sqdist
        else:
            _sqdist = partial(sqdist, data)
        _kernel = knl.make_rbf_kernel(impl, shape_func, _sqdist, bandwidth)
        return _kernel(x, y)

    return kernel


# TODO: Generalize X to PyTree
def make_bandwidth_tuner[Data: PyTree, X: Array](
    pars: TunePars,
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[X, K, V, K]],
    shape_func: Callable[[K], K],
    sqdist: Callable[[X, X], K] | Callable[[Data, X, X], K],
    neg_grad_shape_func: Optional[Callable[[K], K]] = None,
    jit: bool = False,
) -> Callable[[Data], TuneInfo[Array, Array, Array]]:
    """Make kernel tuning function from shape function and square distance."""

    def tune(data: Data) -> TuneInfo[Array, Array, Array]:
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

    if jit:
        tune = jax.jit(tune)

    return tune


def tune_bandwidth[Data: PyTree, X: Array](
    pars: TunePars,
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[X, K, V, K]],
    shape_func: Callable[[K], K],
    sqdist: Callable[[X, X], K] | Callable[[Data, X, X], K],
    data: Data,
    neg_grad_shape_func: Optional[Callable[[K], K]] = None,
    jit: bool = True,
) -> TuneInfo[Array, Array, Array]:
    """Tune kernel bandwidth."""
    tune = make_bandwidth_tuner(
        pars, impl_l2, shape_func, sqdist, neg_grad_shape_func, jit
    )
    return tune(data)


class _DmSymOperatorSpectrum(NamedTuple):
    """NamedTuple holding symmetric diffusion maps operator spectral data."""

    evals: Array
    """Kernel eigenvalues."""

    evecs: Array
    """Kernel eigenvectors."""


def _from_dm_sym_operator_spectrum[Ns: Shape, D: DTypeLike, X: Array](
    l2x: L2FnAlgebra[Ns, D, X, R],
    spec: _DmSymOperatorSpectrum,
    bandwidth: R,
    num_eigs: Optional[int] = None,
    out_shardings: KernelEigenShardings = KernelEigenShardings(),
) -> KernelEigen[R, Rs, V, Vs]:
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


def make_data_driven_dm_kernel_op[
    Ns: Shape,
    D: DTypeLike,
    X: Array,
    Data: PyTree,
](
    impl_l2: Callable[[Data], L2FnAlgebra[Ns, D, X, R]],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    normalization: Optional[Literal["laplace", "fokkerplanck"]],
) -> Callable[[Data, V], V]:
    """Make data-driven kernel integral op with bistochastic normalization."""

    def kernel_op(data: Data, v: V) -> V:
        l2x = impl_l2(data)
        if has_two_args(kernel):
            _kernel = kernel
        else:
            _kernel = partial(kernel, data)
        match normalization:
            case "laplace":
                normalized_kernel = knl.dmsym_normalize(
                    l2x, _kernel, alpha="1"
                )
            case "fokkerplanck":
                normalized_kernel = knl.dmsym_normalize(
                    l2x, _kernel, alpha="0.5"
                )
            case None:
                normalized_kernel = _kernel
        _kernel_op = fun.compose(
            l2x.incl, knl.make_integral_operator(l2x, normalized_kernel)
        )
        return _kernel_op(v)

    return kernel_op


def make_eigh_dm_eigensolver[Ns: Shape, D: DTypeLike, X: Array, Data: PyTree](
    impl_l2: Callable[[Data], L2FnAlgebra[Ns, D, X, R]],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    bandwidth: R,
    normalization: Optional[Literal["laplace", "fokkerplanck"]],
    num_eigs: Optional[int] = None,
    batch_size: Optional[int] = None,
    jit: bool = False,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> Callable[[Data], KernelEigen[R, Rs, V, Vs]]:
    """Make eigensolver for diffusion-maps normalized operaror using eigh."""
    eig_shardings = EigShardings(
        eigenvalues=shardings.eigenvalues, eigenvectors=shardings.matrix
    )
    eig = make_eigh_with_sharding_constraints(shardings=eig_shardings)

    def eigensolve(data: Data) -> KernelEigen[R, Rs, V, Vs]:
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
            dtype=l2x.dtype,
            batch_size=batch_size,
            out_sharding=shardings.matrix,
        )
        spec = _DmSymOperatorSpectrum(*eig(a))
        eigen = _from_dm_sym_operator_spectrum(
            l2x, spec, bandwidth, num_eigs, shardings
        )
        return eigen

    if jit:
        return jax.jit(eigensolve)
    return eigensolve


def make_eigsh_dm_eigensolver[Ns: Shape, D: DTypeLike, X: Array, Data: PyTree](
    impl_l2: Callable[[Data], L2FnAlgebra[Ns, D, X, R]],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    bandwidth: R,
    normalization: Optional[Literal["laplace", "fokkerplanck"]],
    num_samples: int,
    num_eigs: int,
    dtype: Optional[D] = None,
    jit: bool = True,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> Callable[[Data], KernelEigen[R, Rs, V, Vs]]:
    """Make eigensolver for diffusion maps normalized operaror using eigsh."""
    kernel_op = make_data_driven_dm_kernel_op(impl_l2, kernel, normalization)
    if jit:
        kernel_op = jax.jit(kernel_op)
    to_device = partial(jnp.asarray, device=shardings.weights)
    # NOTE: We are using shardings.weights as opposed to shardings.eigenvectors
    # since KernelEigen stores singular vectors in row-major format, but
    # eigs stores singular vectors in column-major format.

    def eigensolve(data: Data) -> KernelEigen[R, Rs, V, Vs]:

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
        ) -> KernelEigen[R, Rs, V, Vs]:
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
            return jax.jit(from_dm_sym_operator_spectrum)(data, spec)
        return from_dm_sym_operator_spectrum(data, spec)

    return eigensolve


def make_dm_eigensolver[Ns: Shape, D: DTypeLike, X: Array, Data: PyTree](
    impl_l2: Callable[[Data], L2FnAlgebra[Ns, D, X, R]],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    bandwidth: R,
    normalization: Optional[Literal["laplace", "fokkerplanck"]],
    solver: Literal["eigh", "eigsh"],
    num_samples: Optional[int] = None,
    num_eigs: Optional[int] = None,
    dtype: Optional[D] = None,
    batch_size: Optional[int] = None,
    jit: bool = True,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> Callable[[Data], KernelEigen[R, Rs, V, Vs]]:
    """Make eigensolver for diffusion-maps normalized kernel operator."""
    match solver:
        case "eigh":
            eigensolve = make_eigh_dm_eigensolver(
                impl_l2,
                kernel,
                bandwidth,
                normalization,
                num_eigs,
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


def _from_bs_operator_spectrum[Ns: Shape, D: DTypeLike, X: Array](
    l2x: L2FnAlgebra[Ns, D, X, R],
    spec: _BsOperatorSpectrum,
    num_eigs: Optional[int],
    bandwidth: R,
    out_shardings: KernelEigenShardings = KernelEigenShardings(),
) -> KernelEigen[R, Rs, V, Vs]:
    """Convert _BsOperatorSpectrum to KerneEigen."""
    num_samples = l2x.shape[0]
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


def make_data_driven_bs_kernel_op[
    Ns: Shape,
    D: DTypeLike,
    X: Array,
    Data: PyTree,
](
    impl_l2: Callable[[Data], L2FnAlgebra[Ns, D, X, R]],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    adj: bool = False,
) -> Callable[[Data, V], V]:
    """Make data-driven kernel integral op with bistochastic normalization."""

    def kernel_op(data: Data, v: V) -> V:
        l2x = impl_l2(data)
        if has_two_args(kernel):
            _kernel = kernel
        else:
            _kernel = partial(kernel, data)
        bs_kernel = knl.bs_normalize(l2x, _kernel)
        if adj:
            bs_kernel = swap_args(bs_kernel)
        _kernel_op = fun.compose(
            l2x.incl, knl.make_integral_operator(l2x, bs_kernel)
        )
        return _kernel_op(v)

    return kernel_op


def make_svd_bs_eigensolver[Ns: Shape, D: DTypeLike, X: Array, Data: PyTree](
    impl_l2: Callable[[Data], L2FnAlgebra[Ns, D, X, R]],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    bandwidth: R,
    num_eigs: Optional[int] = None,
    batch_size: Optional[int] = None,
    jit: bool = True,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> Callable[[Data], KernelEigen[R, Rs, V, Vs]]:
    """Make SVD solver for bistochastic kernel operator using svd."""
    svd_shardings = SvdShardings(
        left_sing_vectors=shardings.matrix,
        sing_values=shardings.eigenvalues,
        right_sing_vectors=shardings.matrix,
    )
    svd = make_svd_with_sharding_constraints(shardings=svd_shardings)

    def svdsolve(data: Data) -> KernelEigen[R, Rs, V, Vs]:
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
            dtype=l2x.dtype,
            batch_size=batch_size,
            out_sharding=shardings.matrix,
        )
        spec = _BsOperatorSpectrum(*svd(a))
        eigen = _from_bs_operator_spectrum(
            l2x, spec, num_eigs, bandwidth, shardings
        )
        return eigen

    if jit:
        return jax.jit(svdsolve)
    return svdsolve


def make_svds_bs_eigensolver[Ns: Shape, D: DTypeLike, X: Array, Data: PyTree](
    impl_l2: Callable[[Data], L2FnAlgebra[Ns, D, X, R]],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    bandwidth: R,
    num_samples: int,
    num_eigs: int,
    dtype: Optional[D] = None,
    batch_size: Optional[int] = None,
    jit: bool = True,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> Callable[[Data], KernelEigen[R, Rs, V, Vs]]:
    """Make SVD solver for bistochastic kernel operator using svds."""
    kernel_op = make_data_driven_bs_kernel_op(impl_l2, kernel)
    adj_kernel_op = make_data_driven_bs_kernel_op(impl_l2, kernel, adj=True)
    if jit:
        kernel_op = jax.jit(kernel_op)
        adj_kernel_op = jax.jit(adj_kernel_op)
    to_device = partial(jnp.asarray, device=shardings.weights)
    # NOTE: We are using shardings.weights as opposed to shardings.eigenvectors
    # since KernelEigen stores singular vectors in row-major format, but
    # svds stores singular vectors in column-major format.

    def svdsolve(data: Data) -> KernelEigen[R, Rs, V, Vs]:
        matvec: Callable[[ArrayLike], Array] = fun.compose(
            partial(jax.jit(kernel_op), data),
            to_device,
        )
        rmatvec: Callable[[ArrayLike], Array] = fun.compose(
            partial(jax.jit(adj_kernel_op), data),
            to_device,
        )
        matmat: Callable[[ArrayLike], Array] = fun.compose(
            partial(
                jax.jit(
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
                jax.jit(
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
        ) -> KernelEigen[R, Rs, V, Vs]:
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
            return jax.jit(from_bs_operator_spectrum)(data, spec)
        return from_bs_operator_spectrum(data, spec)

    return svdsolve


def make_bs_eigensolver[Ns: Shape, D: DTypeLike, X: Array, Data: PyTree](
    impl_l2: Callable[[Data], L2FnAlgebra[Ns, D, X, R]],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    bandwidth: R,
    solver: Literal["svd", "svds"],
    num_samples: Optional[int] = None,
    num_eigs: Optional[int] = None,
    dtype: Optional[D] = None,
    batch_size: Optional[int] = None,
    jit: bool = True,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> Callable[[Data], KernelEigen[R, Rs, V, Vs]]:
    """Solve kernel eigenvalue problem for bistochastic normalization."""
    match solver:
        case "svd":
            eigensolve = make_svd_bs_eigensolver(
                impl_l2,
                kernel,
                bandwidth,
                num_eigs,
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


def make_eigensolver[Ns: Shape, D: DTypeLike, X: Array, Data: PyTree](
    pars: KernelPars,
    impl_l2: Callable[[Data], L2FnAlgebra[Ns, D, X, R]],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    bandwidth: R,
    num_samples: Optional[int] = None,
    dtype: Optional[D] = None,
    jit: bool = True,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> Callable[[Data], KernelEigen[R, Rs, V, Vs]]:
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


def compute_eigen[Ns: Shape, D: DTypeLike, X: Array, Data: PyTree](
    pars: KernelPars,
    impl_l2: Callable[[Data], L2FnAlgebra[Ns, D, X, R]],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    data: Data,
    bandwidth: R,
    num_samples: Optional[int] = None,
    dtype: Optional[D] = None,
    jit: bool = True,
    shardings: KernelEigenShardings = KernelEigenShardings(),
) -> KernelEigen[R, Rs, V, Vs]:
    """Solve eigenvalue prblem for DM or BS kernels."""
    eigensolve = make_eigensolver(
        pars, impl_l2, kernel, bandwidth, num_samples, dtype, jit, shardings
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
    kernel_eigen: KernelEigen[R, Rs, V, Vs],
) -> tuple[F[V, Rl], F[Rl, V], Callable[[V], F[X, R]]]:
    """Make analysis and synthesis operators for diffusion maps eigenbasis."""
    anal = vec.make_l2_analysis_operator(l2x, kernel_eigen.dual_evecs)
    synth = vec.make_synthesis_operator(kernel_eigen.evecs)

    @partial(vmap, in_axes=(0, 0, None))
    def extend_eval(v: V, lamb: R, x: X) -> R:
        return extend(v, lamb)(x)

    basis = partial(
        extend_eval,
        kernel_eigen.evecs,
        kernel_eigen.evals,
    )
    fn_synth = vec.make_fn_synthesis_operator(basis)
    return anal, synth, fn_synth


def make_eigenbasis_operators_bs[N: int, D: DTypeLike, X: Array](
    l2x: L2FnAlgebra[tuple[N], D, X, R],
    extend: Callable[[V, R], F[X, R]],
    kernel_eigen: KernelEigen[R, Rs, V, Vs],
) -> tuple[F[V, Rl], F[Rl, V], Callable[[V], F[X, R]]]:
    """Make analysis and synthesis operators for bistochastic eigenbasis."""
    anal = vec.make_l2_analysis_operator(l2x, kernel_eigen.evecs)
    synth = vec.make_synthesis_operator(kernel_eigen.evecs)

    @partial(vmap, in_axes=(0, 0, None))
    def extend_eval(v: V, lamb: R, x: X) -> R:
        return extend(v, lamb)(x)

    basis = partial(
        extend_eval,
        kernel_eigen.dual_evecs,
        kernel_eigen.evals,
    )
    fn_synth = vec.make_fn_synthesis_operator(basis)
    return anal, synth, fn_synth


def make_eigenbasis_operators[N: int, D: DTypeLike, X: Array](
    pars: KernelPars,
    l2x: L2FnAlgebra[tuple[N], D, X, R],
    kernel: Callable[[X, X], R],
    kernel_eigen: KernelEigen[R, Rs, V, Vs],
) -> tuple[F[V, Rl], F[Rl, V], Callable[[V], F[X, R]]]:
    """Make analysis and synthesis operators."""
    match pars:
        case DmKernelPars():
            extend, _ = make_eigenvector_extension_dm(
                l2x, kernel, pars.normalization
            )
            anal, synth, fn_synth = make_eigenbasis_operators_dm(
                l2x, extend, kernel_eigen
            )
        case BsKernelPars():
            extend, _ = make_eigenvector_extension_bs(l2x, kernel)
            anal, synth, fn_synth = make_eigenbasis_operators_bs(
                l2x, extend, kernel_eigen
            )
    return anal, synth, fn_synth


def make_eigenbasis_dm[Ns: Shape, D: DTypeLike, X: Array](
    l2x: L2FnAlgebra[Ns, D, X, R],
    kernel: Callable[[X, X], R],
    normalization: Optional[Literal["laplace", "fokkerplanck"]],
    laplacian_method: Literal["lin", "log", "inv"],
    kernel_eigen: KernelEigen[R, Rs, V, Vs],
) -> KernelEigenbasis[X, R, V, Rs, int | Array]:
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

    def vc(i: int | Array) -> V:
        return kernel_eigen.evecs[i]

    def dual_vc(i: int | Array) -> V:
        return kernel_eigen.dual_evecs[i]

    def evl(i: int | Array) -> K:
        return kernel_eigen.evals[i]

    def lapl_evl(i: int | Array) -> K:
        return lapl_spec[i]

    def fn(i: int | Array) -> Callable[[X], K]:
        return extension_op(kernel_eigen.evecs[i] / kernel_eigen.evals[i])

    def dual_fn(i: int | Array) -> Callable[[X], K]:
        return dual_extension_op(
            kernel_eigen.dual_evecs[i] / kernel_eigen.evals[i]
        )

    @partial(vmap, in_axes=(0, None))
    def anal_eval(i: int | Array, v: V) -> K:
        return l2x.innerp(kernel_eigen.dual_evecs[i], v)

    @partial(vmap, in_axes=(0, None))
    def dual_anal_eval(i: int | Array, v: V) -> K:
        return l2x.innerp(kernel_eigen.evecs[i], v)

    @partial(vmap, in_axes=(0, None))
    def fn_eval(i: int | Array, x: X) -> R:
        return fn(i)(x)

    @partial(vmap, in_axes=(0, None))
    def dual_fn_eval(i: int | Array, x: X) -> R:
        return dual_fn(i)(x)

    idxs = jnp.arange(kernel_eigen.num_eigs)
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


def make_eigenbasis_bs[Ns: Shape, D: DTypeLike, X: Array](
    l2x: L2FnAlgebra[Ns, D, X, K],
    kernel: Callable[[X, X], K],
    laplacian_method: Literal["lin", "log", "inv"],
    kernel_eigen: KernelEigen[R, Rs, V, Vs],
) -> KernelEigenbasis[X, R, V, Rs, int | Array]:
    """Make kernel eigenbasis for bistochastic kernels."""
    lapl_spec = to_laplace_eigenvalues(
        kernel_eigen.evals,
        kernel_eigen.bandwidth,
        method=laplacian_method,
    )
    extension_kernel = knl.bs_normalize(l2x, kernel)
    extension_op = knl.make_integral_operator(l2x, extension_kernel)

    def vc(i: int | Array) -> V:
        return kernel_eigen.evecs[i]

    def evl(i: int | Array) -> K:
        return kernel_eigen.evals[i]

    def lapl_evl(i: int | Array) -> K:
        return lapl_spec[i]

    def fn(i: int | Array) -> F[X, K]:
        return extension_op(
            kernel_eigen.dual_evecs[i] / jnp.sqrt(kernel_eigen.evals[i])
        )

    @partial(vmap, in_axes=(0, None))
    def anal_eval(i: int | Array, v: V) -> K:
        return l2x.innerp(kernel_eigen.evecs[i], v)

    @partial(vmap, in_axes=(0, None))
    def fn_eval(i: int | Array, x: X) -> R:
        return fn(i)(x)

    idxs = jnp.arange(kernel_eigen.num_eigs)
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


def make_eigenbasis[D: DTypeLike, X: Array, N: Shape](
    pars: KernelPars,
    l2x: L2FnAlgebra[N, D, X, K],
    kernel: Callable[[X, X], K],
    kernel_eigen: KernelEigen[R, Rs, V, Vs],
    laplacian_method: Literal["lin", "log", "inv"] = "log",
) -> KernelEigenbasis[X, R, V, Rs, int | Array]:
    """Make kernel eigenbasis."""
    match pars:
        case DmKernelPars():
            basis = make_eigenbasis_dm(
                l2x,
                kernel,
                pars.normalization,
                laplacian_method,
                kernel_eigen,
            )
        case BsKernelPars():
            basis = make_eigenbasis_bs(
                l2x, kernel, laplacian_method, kernel_eigen
            )
    return basis


def slice_eigen(
    eigen: KernelEigen[K, Ks, V, Vs],
    which_eigs: int | tuple[int, int] | list[int] | None = None,
) -> KernelEigen[K, Ks, V, Vs]:
    """Slice KernelEigen object using `which_eigs` convention."""
    match which_eigs:
        case None:
            sliced_eigen = eigen
        case int() as num_eigs:
            sliced_eigen = eigen.isel(slice(0, num_eigs))
        case tuple() as idx:
            sliced_eigen = eigen.isel(slice(idx[0], idx[1] + 1))
        case list() as idxs:
            sliced_eigen = eigen.isel(idxs)
    return sliced_eigen


# TODO: Consider automating the process of building these data-driven wrappers
# using a decorator.
def make_data_driven_eigenbasis[
    Data: PyTree,
    D: DTypeLike,
    X: Array,
    N: Shape,
](
    pars: KernelPars,
    impl_l2: Callable[[Data], L2FnAlgebra[N, D, X, K]],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    which_eigs: int | tuple[int, int] | list[int] | None = None,
) -> Callable[
    [Data, KernelEigen[K, Ks, V, Vs]],
    KernelEigenbasis[X, K, V, Ks, int | Array],
]:
    """Make data-driven kernel eigenbasis builder."""

    def _make_eigenbasis(
        data: Data,
        kernel_eigen: KernelEigen[K, Ks, V, Vs],
    ) -> KernelEigenbasis[X, K, V, Ks, int | Array]:
        l2x = impl_l2(data)
        if has_two_args(kernel):
            _kernel = kernel
        else:
            _kernel = partial(kernel, data)
        _kernel_eigen = slice_eigen(kernel_eigen, which_eigs)
        return make_eigenbasis(pars, l2x, _kernel, _kernel_eigen)

    return _make_eigenbasis


# TODO: Generalize X to PyTree
def make_kaf_analysis_operator[
    Data: PyTree,
    X: Array,
](
    impl_basis: Callable[
        [Data, KernelEigen[K, Ks, V, Vs]],
        KernelEigenbasis[X, K, V, Ks, int | Array],
    ],
    num_steps: int,
    which_samples: Optional[tuple[int, int]] = None,
    jit: bool = False,
) -> Callable[[Data, Rs, KernelEigen[K, Ks, V, Vs]], Rs]:
    """Make analysis operator for kernel analog forecast."""

    def anal(
        data: Data,
        response: Rs,
        kernel_eigen: KernelEigen[K, Ks, V, Vs],
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

    if jit:
        return jax.jit(anal)
    return anal


def make_kaf_prediction_function[
    Data: PyTree,
    D: DTypeLike,
    X: Array,
    Ntst: Shape,
](
    impl_basis: Callable[
        [Data, KernelEigen[K, Ks, V, Vs]],
        KernelEigenbasis[X, K, V, Ks, int | Array],
    ],
    impl_l2_tst: Callable[[Data], L2FnAlgebra[Ntst, D, X, K]],
    jit: bool = False,
) -> Callable[[Data, KernelEigen[K, Ks, V, Vs], Rs, Data], R]:
    """Make prediction function for kernel analog forecast."""

    def predict(
        data: Data,
        kernel_eigen: KernelEigen[K, Ks, V, Vs],
        coeffs: Rs,
        test_data: Data,
    ) -> Rs:
        basis = impl_basis(data, kernel_eigen)
        l2x_tst = impl_l2_tst(test_data)

        @partial(vmap, in_axes=(0, None))
        def _predict(cs: Rs, x: X) -> R:
            return basis.fn_synth(cs)(x)

        return l2x_tst.incl(partial(_predict, coeffs))

    if jit:
        return jax.jit(predict)
    return predict


# TODO: Try moving this to nlsa.kernels by abstracting over L2FnAlgebra (using
# the already defined protocol from alg, and making KernelEigen a protocol.
def compute_kaf_preds[
    Data: PyTree,
    D: DTypeLike,
    X: Array,
    N: Shape,
    Ntst: Shape,
](
    kernel_pars: KernelPars,
    impl_l2: Callable[[Data], L2FnAlgebra[N, D, X, R]],
    train_data: Data,
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    kernel_eigen: KernelEigen[R, Rs, V, Vs],
    coeffs: Rs,
    impl_l2_tst: Callable[[Data], L2FnAlgebra[Ntst, D, X, R]],
    test_data: Data,
    which_eigs: int | tuple[int, int] | list[int] | None = None,
    jit: bool = True,
) -> Array:
    """Compute KAF predictions."""
    impl_basis = make_data_driven_eigenbasis(
        kernel_pars, impl_l2, kernel, which_eigs
    )
    predict = make_kaf_prediction_function(impl_basis, impl_l2_tst, jit)
    return predict(train_data, kernel_eigen, coeffs, test_data)


def make_iterative_kaf_analysis_operator[
    Data: PyTree,
    X: Array,
](
    impl_basis: Callable[
        [Data, KernelEigen[K, Ks, V, Vs]],
        KernelEigenbasis[X, K, V, Ks, int | Array],
    ],
    which_samples: Optional[tuple[int, int]] = None,
    jit: bool = False,
) -> Callable[[Data, Rs, KernelEigen[K, Ks, V, Vs]], Rs]:
    """Make analysis operator for kernel analog forecast."""

    def anal(
        data: Data,
        covariates: Xs,
        kernel_eigen: KernelEigen[K, Ks, V, Vs],
    ) -> Rs:
        if which_samples is not None:
            i0 = which_samples[0]
            i1 = which_samples[1]
        else:
            i0 = 1
            i1 = len(covariates)
        basis = impl_basis(data, kernel_eigen)
        anal = vmap(basis.anal, in_axes=1)
        return anal(covariates[i0:i1])

    if jit:
        return jax.jit(anal)
    return anal


def make_iterative_kaf_prediction_function[
    Data: PyTree,
    D: DTypeLike,
    Ntst: Shape,
](
    impl_basis: Callable[
        [Data, KernelEigen[K, Ks, V, Vs]],
        KernelEigenbasis[Array, K, V, Ks, int | Array],
    ],
    impl_l2_tst: Callable[[Data], L2FnAlgebra[Ntst, D, Array, K]],
    num_steps: int,
    jit: bool = False,
) -> Callable[[Data, KernelEigen[K, Ks, V, Vs], Rs, Data], Array]:
    """Make prediction function for iterative KAF."""

    def predict(
        data: Data,
        kernel_eigen: KernelEigen[K, Ks, V, Vs],
        coeffs: Xs,
        test_data: Data,
    ) -> Rs:
        basis = impl_basis(data, kernel_eigen)
        l2x_tst = impl_l2_tst(test_data)

        @partial(vmap, in_axes=(0, None))
        def predict_snapshot(cs: Rs, x: Array) -> Array:
            """Predict next snapshot."""
            return basis.fn_synth(cs)(x)

        _predict = dyn.make_fin_orbit(
            partial(predict_snapshot, coeffs), num_steps + 1
        )
        return l2x_tst.incl(_predict)

    if jit:
        return jax.jit(predict)
    return predict


def make_iterative_kaf_prediction_function_with_delays[
    Data: PyTree,
    D: DTypeLike,
    Ntst: Shape,
](
    impl_basis: Callable[
        [Data, KernelEigen[K, Ks, V, Vs]],
        KernelEigenbasis[Array, K, V, Ks, int | Array],
    ],
    impl_l2_tst: Callable[[Data], L2FnAlgebra[Ntst, D, Array, K]],
    num_delays: int,
    num_steps: int,
    project: bool = True,
    jit: bool = False,
) -> Callable[[Data, KernelEigen[K, Ks, V, Vs], Rs, Data], R]:
    """Make single-step prediction function for iterative KAF."""

    def predict(
        data: Data,
        kernel_eigen: KernelEigen[K, Ks, V, Vs],
        coeffs: Rs,
        test_data: Data,
    ) -> Rs:
        basis = impl_basis(data, kernel_eigen)
        l2x_tst = impl_l2_tst(test_data)

        @partial(vmap, in_axes=(0, None))
        def predict_snapshot(cs: Rs, xs: Xs) -> Array:
            """Predict next snapshot from delay-embedded data."""
            return basis.fn_synth(cs)(xs)

        def predict_window(cs: Rs, xs: Array) -> Array:
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

    if jit:
        return jax.jit(predict)
    return predict


def compute_iterative_kaf_preds[
    Data: PyTree,
    D: DTypeLike,
    N: Shape,
    Ntst: Shape,
](
    kernel_pars: KernelPars,
    impl_l2: Callable[[Data], L2FnAlgebra[N, D, Array, R]],
    train_data: Data,
    kernel: Callable[[Array, Array], R] | Callable[[Data, Array, Array], R],
    kernel_eigen: KernelEigen[R, Rs, V, Vs],
    coeffs: Rs,
    impl_l2_tst: Callable[[Data], L2FnAlgebra[Ntst, D, Array, R]],
    test_data: Data,
    num_steps: int,
    num_delays: Optional[int],
    which_eigs: int | tuple[int, int] | list[int] | None = None,
    project: bool = True,
    jit: bool = True,
) -> Array:
    """Compute iterative KAF predictions of the covariate variables."""
    impl_basis = make_data_driven_eigenbasis(
        kernel_pars, impl_l2, kernel, which_eigs
    )
    if num_delays is None or num_delays == 0:
        predict = make_iterative_kaf_prediction_function(
            impl_basis, impl_l2_tst, num_steps, jit
        )
    else:
        predict = make_iterative_kaf_prediction_function_with_delays(
            impl_basis, impl_l2_tst, num_delays, num_steps, project, jit
        )
    return predict(train_data, kernel_eigen, coeffs, test_data)


def plot_laplace_spectrum(
    kernel_eigen: KernelEigen[R, Rs, V, Vs],
    num_eigs_plt: Optional[int] = None,
    i_fig: int = 1,
) -> Figure:
    """Plot spectrum of Laplacian eigenvalues."""
    if num_eigs_plt is None:
        num_eigs_plt = len(kernel_eigen.evals)
    kernel_evals = kernel_eigen.evals[:num_eigs_plt]
    lapl_evals = partial(
        to_laplace_eigenvalues, kernel_evals, kernel_eigen.bandwidth
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

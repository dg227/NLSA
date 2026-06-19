"""Provide generic functions and classes for kernel computations."""

import math
import matplotlib.pyplot as plt
import nlsa.abstract_algebra as alg
import nlsa.function_algebra as fun
import nlsa.scalars as scl
import numpy as np
import seaborn as sns
from collections.abc import Callable, Sized
from dataclasses import dataclass
from functools import partial
from matplotlib.figure import Figure
from nlsa.function_algebra import (
    FunctionAlgebra,
    FunctionAlgebraWithCalculus,
    BivariateFunctionDivBimodule,
)
from nlsa.typing import (
    SliceItem,
    is_array_like,
    is_sliceable,
)
from nlsa.utils import swap_args
from numpy.typing import ArrayLike
from tabulate import tabulate
from typing import Literal, NamedTuple, Optional, Sequence, final

type F[*Xs, Y] = Callable[[*Xs], Y]


@dataclass(frozen=True, slots=True)
class ConePars:
    """Dataclass containing cone kernel parameters."""

    zeta: float
    """Cone kernel anisotropy parameter."""

    threshold: float = 1e-12
    """Cone distance threshold parameter (for stable autodiff at zero)."""

    def __str__(self) -> str:
        """Create string representation of cone kernel parameters."""
        return "_".join(
            (
                f"zeta{self.zeta:.4f}",
                f"thresh{self.threshold}",
            )
        )


@dataclass(frozen=True, slots=True)
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
        """Create string representation of kernel tuning parameters."""
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


class TuneInfo[K, Ks, I](NamedTuple):
    """NamedTuple holding kernel tuning information."""

    log10_bandwidths: Ks
    """Array of trial kernel bandwidths."""

    est_dims: Ks
    """Array of estimated dimensions based on trial bandwidths."""

    opt_bandwidth: K
    """Optimal bandwidth from autotuning procedure."""

    opt_dim: K
    """Optimal (maximum) dimension from autotuning procedure."""

    i_opt: I
    """Index of optimal bandwidth in array of trial bandwidths."""

    bandwidth: K
    """Selected bandwidth after scaling by user-defined factor."""

    dim: K
    """Estimated dimension based on selected bandwidth."""

    vol: K
    """Estimated manifold volume based on selected bandwidth."""

    kernel_vol: K
    """Volume based on kernel integral."""

    def tabulate(
        self, name: str = "Kernel Tuning Info", show: bool = True
    ) -> str:
        """Create tabulated summary of the elements of a TuneInfo object."""
        headers = [name, "Value"]
        data = {
            "Optimal bandwidth index": f"{self.i_opt}",
            "Optimal bandwidth": f"{self.opt_bandwidth:.3e}",
            "Optimal dimension": f"{self.opt_dim:.3e}",
            "Bandwidth used for diffusion maps": f"{self.bandwidth:.3e}",
            "Dimension based on diffusion maps bandwidth": f"{self.dim:.3e}",
            "Manifold volume": f"{self.vol:.3e}",
            "Kernel volume": f"{self.kernel_vol: .3e}",
        }
        table = tabulate(data.items(), headers=headers)
        if show:
            print(table)
        return table


# TODO: The batch_size parameter in this and other related classes feels
# somewhat out-of-place as it is not a parameter affecting the result of the
# computation -- it only affects how the computation is performed. Consider
# moving this to a different class, e.g., a class representing both batching
# and sharding in JAX computations. Similar considerations apply to TunePars.
@dataclass(frozen=True, slots=True)
class DmKernelPars:
    """Dataclass containing diffusion maps eigendecomposition parameters."""

    normalization: Optional[Literal["laplace", "fokkerplanck"]]
    """Kernel normalization method."""

    eigensolver: Literal["eigh", "eigsh"]
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


@dataclass(frozen=True, slots=True)
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


class KernelEigen[Scalar, Scalars, Vector, Vectors](NamedTuple):
    """NamedTuple containing kernel spectral data."""

    evals: Scalars
    """Kernel eigenvalues."""

    evecs: Vectors
    """Kernel eigenvectors."""

    dual_evecs: Vectors
    """Dual (left) kernel eigenvectors."""

    weights: Vector
    """Inner product weights that orthonormalize the eigenvectors."""

    bandwidth: Scalar
    """Bandwidth parameter."""

    @property
    def num_eigs(
        self,
    ) -> int:
        """Return number of eigenvalues/eigenvectors in KernelEigenObject."""
        assert isinstance(self.evals, Sized)
        return len(self.evals)

    def isel(
        self,
        s: SliceItem,
    ) -> "KernelEigen[Scalar, Scalars, Vector, Vectors]":
        """Slice a KernelEigen object."""
        assert is_sliceable(self.evals)
        assert is_sliceable(self.evecs)
        assert is_sliceable(self.dual_evecs)
        return KernelEigen(
            evals=self.evals[s],
            evecs=self.evecs[s],
            dual_evecs=self.dual_evecs[s],
            weights=self.weights,
            bandwidth=self.bandwidth,
        )

    def tabulate(
        self,
        num_tabulate: Optional[int] = None,
        headers: Sequence[str] = ["Kernel eigenvalues"],
        show: bool = True,
    ) -> str:
        """Tabulate the eigenvalues in a KernelEigen object."""
        assert is_array_like(self.evals)
        data = np.vstack(((self.evals),))[:, :num_tabulate].T
        table = tabulate(data, headers=headers, floatfmt=".4f", showindex=True)
        if show:
            print(table)
        return table


# TODO: Introduce a type parameter for dim
@final
@dataclass(frozen=True, slots=True)
class KernelEigenbasis[X, K, V, Ks, I](
    alg.ImplementsDimensionedL2FnFrame[X, K, V, Ks, I]
):
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

    vec: Callable[[I], V]
    """Basis vectors."""

    dual_vec: Callable[[I], V]
    """Dual basis vectors."""

    fn: Callable[[I], F[X, K]]
    """Function representatives of basis vectors."""

    dual_fn: Callable[[I], F[X, K]]
    """Function representatives of dual basis vectors."""

    spec: Ks
    """Kernel operator spectrum (set of eigenvalues)."""

    lapl_spec: Ks
    """Laplace spectrum."""

    evl: Callable[[I], K]
    """Kernel eigenvalues."""

    lapl_evl: Callable[[I], K]
    """Laplacian eigenvalues."""


def make_scaled_sqdist[X, K](
    impl: alg.ImplementsRealScalarField[K],
    sqdist: Callable[[X, X], K],
    bandwdith_func: Callable[[X], K],
) -> Callable[[X, X], K]:
    """Make scaled square distance function from bandwidth function."""
    func: FunctionAlgebraWithCalculus[X, X, K, K] = (
        FunctionAlgebraWithCalculus(codomain=scl.AsAlgebraWithCalculus(impl))
    )
    tensorp = fun.make_bivariate_tensor_product(impl)
    d2_scl = func.div(sqdist, tensorp(bandwdith_func, bandwdith_func))
    return d2_scl


def make_rbf[K](
    impl: alg.ImplementsScalarField[K],
    shape_func: F[K, K],
    bandwidth: K,
) -> F[K, K]:
    """Make bandwidth-parameterized radial basis function."""
    neg_concentration = impl.neg(impl.inv(impl.mul(bandwidth, bandwidth)))

    def rbf(s: K) -> K:
        return shape_func(impl.mul(neg_concentration, s))

    return rbf


def make_rbf_kernel[X, K](
    impl: alg.ImplementsScalarField[K],
    shape_func: F[K, K],
    sqdist: F[X, X, K],
    bandwidth: K,
) -> F[X, X, K]:
    """Make bandwidth-parameterized radial basis function kernel."""
    rbf = make_rbf(impl, shape_func, bandwidth)
    return fun.compose(rbf, sqdist)


def make_rbf_kernel_family[X, K](
    impl: alg.ImplementsScalarField[K], shape_func: F[K, K], sqdist: F[X, X, K]
) -> Callable[[K], F[X, X, K]]:
    """Make bandwdith-parameterized kernel family."""
    return partial(make_rbf_kernel, impl, shape_func, sqdist)

    # make_shape_func: Callable[[K], F[K, K]] = partial(
    #     make_rbf, impl, shape_func=shape_func
    # )

    # def kernel_family(epsilon: K, /) -> Callable[[X, X], K]:
    #     return fun.compose(make_shape_func(epsilon), sqdist)

    # return kernel_family


def make_integral_operator[X, V, K](
    impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K],
    k: Callable[[X, X], K],
    /,
) -> Callable[[V], F[X, K]]:
    """Make integral operator from kernel function."""

    def k_op(v: V, /) -> F[X, K]:
        def g(x: X, /) -> K:
            kx = partial(k, x)
            gx = impl.integrate(impl.mul(impl.incl(kx), v))
            return gx

        return g

    return k_op


def left_normalize[X, V, K](
    impl: alg.ImplementsMeasureFnStarAlgebra[X, K, V, K],
    k: Callable[[X, X], K],
) -> Callable[[X, X], K]:
    """Perform left normalization of kernel function."""
    func: BivariateFunctionDivBimodule[X, X, K, K] = (
        BivariateFunctionDivBimodule(codomain=scl.AsDivBimodule(impl.scl))
    )
    k_op = make_integral_operator(impl, k)
    lfun = k_op(impl.unit())
    k_l = func.ldiv(lfun, k)
    return k_l


def right_normalize[X, V, K](
    impl: alg.ImplementsMeasureFnStarAlgebra[X, K, V, K],
    k: Callable[[X, X], K],
) -> Callable[[X, X], K]:
    """Perform right normalization of kernel function."""
    func: BivariateFunctionDivBimodule[X, X, K, K] = (
        BivariateFunctionDivBimodule(codomain=scl.AsDivBimodule(impl.scl))
    )
    k_op = make_integral_operator(impl, k)
    rfun = k_op(impl.unit())
    k_r = func.rdiv(k, rfun)
    return k_r


def sym_normalize[X, V, K](
    impl: alg.ImplementsMeasureFnStarAlgebra[X, K, V, K],
    k: Callable[[X, X], K],
) -> Callable[[X, X], K]:
    """Perform symmetric normalization of kernel function."""
    func: BivariateFunctionDivBimodule[X, X, K, K] = (
        BivariateFunctionDivBimodule(codomain=scl.AsDivBimodule(impl.scl))
    )
    k_op = make_integral_operator(impl, k)
    sfun = k_op(impl.unit())
    k_r = func.rdiv(k, sfun)
    k_s = func.ldiv(sfun, k_r)
    return k_s


def right_sqrt_normalize[X, V, K](
    impl: alg.ImplementsMeasureFnStarAlgebra[X, K, V, K],
    k: Callable[[X, X], K],
) -> Callable[[X, X], K]:
    """Perform right square root normalization of kernel function."""
    func: FunctionAlgebraWithCalculus[X, K, K] = FunctionAlgebraWithCalculus(
        codomain=scl.AsAlgebraWithCalculus(impl.scl)
    )
    func2: BivariateFunctionDivBimodule[X, X, K, K] = (
        BivariateFunctionDivBimodule(codomain=scl.AsDivBimodule(impl.scl))
    )
    k_op = make_integral_operator(impl, k)
    rfun = func.sqrt(k_op(impl.unit()))
    k_r = func2.rdiv(k, rfun)
    return k_r


def sym_sqrt_normalize[X, V, K](
    impl: alg.ImplementsMeasureFnStarAlgebra[X, K, V, K],
    k: Callable[[X, X], K],
) -> Callable[[X, X], K]:
    """Perform symmetric square root normalization of kernel function."""
    func: FunctionAlgebraWithCalculus[X, K, K] = FunctionAlgebraWithCalculus(
        codomain=scl.AsAlgebraWithCalculus(impl.scl)
    )
    func2: BivariateFunctionDivBimodule[X, X, K, K] = (
        BivariateFunctionDivBimodule(codomain=scl.AsDivBimodule(impl.scl))
    )
    k_op = make_integral_operator(impl, k)
    sfun = func.sqrt(k_op(impl.unit()))
    k_r = func2.rdiv(k, sfun)
    k_s = func2.ldiv(sfun, k_r)
    return k_s


def dm_normalize[X, V, K](
    impl: alg.ImplementsMeasureFnStarAlgebra[X, K, V, K],
    k: Callable[[X, X], K],
    /,
    alpha: Literal["0", "0.5", "1"],
) -> Callable[[X, X], K]:
    """Perform Diffusion Maps kernel normalization."""
    match alpha:
        case "0":
            k_r = k
        case "0.5":
            k_r = sym_sqrt_normalize(impl, k)
        case "1":
            k_r = sym_normalize(impl, k)
    k_dm = left_normalize(impl, k_r)
    return k_dm


def dmsym_normalize[X, V, K](
    impl: alg.ImplementsMeasureFnStarAlgebra[X, K, V, K],
    k: Callable[[X, X], K],
    /,
    alpha: Literal["0", "0.5", "1"],
) -> Callable[[X, X], K]:
    """Perform Diffusion Maps symmetric kernel normalization."""
    match alpha:
        case "0":
            k_r = k
        case "0.5":
            k_r = sym_sqrt_normalize(impl, k)
        case "1":
            k_r = sym_normalize(impl, k)
    k_dm = sym_sqrt_normalize(impl, k_r)
    return k_dm


def from_dmsym[Vs, V, K](
    impl: alg.ImplementsLDivModule[Vs, K, V], v0: V, vs: Vs, /
) -> Vs:
    """Normalize eigenvectors from symmetric diffusion maps.

    Resulting eigenvectors are normalized with respect to Markov normalization.
    """
    return impl.ldiv(v0, vs)


def bs_normalize[X, V, K](
    impl: alg.ImplementsMeasureFnStarAlgebra[X, K, V, K],
    k: Callable[[X, X], K],
) -> Callable[[X, X], K]:
    """Perform bistochastic kernel normalization (left part)."""
    func: FunctionAlgebraWithCalculus[X, K, K] = FunctionAlgebraWithCalculus(
        codomain=scl.AsAlgebraWithCalculus(impl.scl)
    )
    func2: BivariateFunctionDivBimodule[X, X, K, K] = (
        BivariateFunctionDivBimodule(codomain=scl.AsDivBimodule(impl.scl))
    )
    k_op = make_integral_operator(impl, k)
    k_op = make_integral_operator(impl, k)
    d = k_op(impl.unit())
    k_r = func2.rdiv(k, d)
    k_r_op = make_integral_operator(impl, k_r)
    q: F[X, K] = k_r_op(impl.unit())
    k_q = func2.rdiv(k, func.sqrt(q))
    k_bs = func2.ldiv(d, k_q)
    return k_bs


def bssym_normalize[X, V, K](
    impl: alg.ImplementsMeasureFnStarAlgebra[X, K, V, K],
    k: Callable[[X, X], K],
) -> Callable[[X, X], K]:
    """Perform bistochastic kernel normalization (symmetrized)."""
    k_bs = bs_normalize(impl, k)

    def k_sym(x: X, y: X, /) -> K:
        u = impl.incl(partial(k_bs, x))
        v = impl.incl(partial(k_bs, y))
        return impl.integrate(impl.mul(u, v))

    return k_sym


def compose[X, V, K](
    impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K],
    k1: Callable[[X, X], K],
    k2: Callable[[X, X], K],
    /,
) -> Callable[[X, X], K]:
    """Compose two kernels."""
    k2_transp = swap_args(k2)

    def k3(x: X, y: X, /) -> K:
        v1 = impl.incl(partial(k1, x))
        v2 = impl.incl(partial(k2_transp, y))
        return impl.integrate(impl.mul(v1, v2))

    return k3


def make_mercer_kernel[X, V, K](
    impl: alg.ImplementsInnerProductAlgebra[V, K],
    psi_l: F[X, V],
    psi_r: F[X, V],
    /,
) -> Callable[[X, X], K]:
    """Make Mercer kernel from 'left' and 'right' feature vectors."""

    def k(x: X, y: X, /) -> K:
        return impl.innerp(psi_l(x), psi_r(y))

    return k


def riemannian_vol[X, V, K](
    impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K],
    kernel: Callable[[X, X], K],
    dim: K,
    t_heat: K,
) -> K:
    """Compute Riemannian volume using heat trace formula."""
    h: F[X, K] = fun.diag(kernel)
    a = impl.scl.power(
        impl.scl.sqrt(
            impl.scl.mul(impl.scl.from_pyscalar(4 * math.pi), t_heat)
        ),
        dim,
    )
    vol = impl.scl.mul(a, impl.integrate(impl.incl(h)))
    return vol


def kernel_vol[X, V, K](
    impl: alg.ImplementsMeasureFnStarAlgebra[X, K, V, K],
    kernel: Callable[[X, X], K],
) -> K:
    """Compute normalization of kernel bandwidth function."""
    w = sym_normalize(impl, kernel)
    w_op = make_integral_operator(impl, w)
    d = w_op(impl.unit())
    d_bar = impl.integrate(impl.incl(d))
    return d_bar


def make_bandwidth_function[X, V, K](
    impl: alg.ImplementsMeasureFnStarAlgebra[X, K, V, K],
    k: Callable[[X, X], K],
    /,
    dim: K,
    vol: K,
    normalization: Optional[K] = None,
) -> Callable[[X], K]:
    """Make bandwidth function for variable-bandwidth kernel."""
    func: FunctionAlgebraWithCalculus[X, K, K] = FunctionAlgebraWithCalculus(
        codomain=scl.AsAlgebraWithCalculus(impl.scl)
    )
    w = sym_normalize(impl, k)
    w_op = make_integral_operator(impl, w)
    d = w_op(impl.unit())
    if normalization is not None:
        c = impl.scl.div(vol, normalization)
    else:
        c = vol
    b = func.power(func.smul(c, d), impl.scl.inv(dim))
    return b


def make_tuning_objective_from_kernel_family[X, V, K](
    impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K],
    k_func: Callable[[K], Callable[[X, X], K]],
    /,
    grad: Callable[[F[K, K]], F[K, K]],
) -> Callable[[K], K]:
    """Make objective function for kernel tuning."""

    def log_k_sum(log_eps: K) -> K:
        epsilon = impl.scl.exp10(log_eps)
        k_op = make_integral_operator(impl, k_func(epsilon))
        s = impl.integrate(impl.incl(k_op(impl.unit())))
        return impl.scl.log10(s)

    return grad(log_k_sum)


def make_tuning_objective_from_shape_function[X, V, K](
    impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K],
    shape_func: Callable[[K], K],
    neg_grad_shape_func: Callable[[K], K],
    sqdist: F[X, X, K],
) -> Callable[[K], K]:
    """Make objective function for kernel tuning."""
    func: FunctionAlgebra[X, X, K, K] = FunctionAlgebra(
        codomain=scl.AsAlgebraWithCalculus(impl.scl)
    )
    kernel_family = make_rbf_kernel_family(impl.scl, shape_func, sqdist)
    grad_kernel_family = make_rbf_kernel_family(
        impl.scl, neg_grad_shape_func, sqdist
    )

    def grad_log10_k_sum(log10_eps: K) -> K:
        epsilon = impl.scl.exp10(log10_eps)
        c = impl.scl.div(
            impl.scl.from_pyscalar(2), impl.scl.mul(epsilon, epsilon)
        )
        k_op = make_integral_operator(impl, kernel_family(epsilon))
        s = impl.integrate(impl.incl(k_op(impl.unit())))
        k2_op = make_integral_operator(
            impl, func.mul(grad_kernel_family(epsilon), sqdist)
        )
        s2 = impl.integrate(impl.incl(k2_op(impl.unit())))
        return impl.scl.mul(c, impl.scl.div(s2, s))

    return grad_log10_k_sum


def make_resolvent_compactification_kernels[X, TX, V, K](
    impl: alg.ImplementsMeasureFnStarAlgebra[X, K, V, K],
    v: Callable[[X], TX],
    z: K,
    k: Callable[[X, X], K],
    /,
    jvp: Callable[[F[X, K], X, TX], K],
) -> tuple[F[X, X, K], F[X, X, K], F[X, X, K]]:
    """Make kernels for resolvent compactification scheme."""
    func2: BivariateFunctionDivBimodule[X, X, K, K] = (
        BivariateFunctionDivBimodule(codomain=scl.AsDivBimodule(impl.scl))
    )

    @swap_args
    def v_grad_k(x: X, y: X, /) -> K:
        return jvp(partial(swap_args(k), x), y, v(y))

    zk_vk = func2.sub(func2.smul(z, k), v_grad_k)
    kvk = compose(impl, swap_args(k), v_grad_k)
    qz_i = swap_args(zk_vk)
    qz_j = compose(impl, kvk, swap_args(zk_vk))
    gz = compose(impl, zk_vk, swap_args(zk_vk))
    return qz_i, qz_j, gz


def plot_kernel_tuning[A: ArrayLike](
    tune_info: TuneInfo[A, A, A],
    title: Optional[str] = None,
    i_fig: int = 1,
) -> Figure:
    """Plot kernel tuning function."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    ax.plot(tune_info.log10_bandwidths, tune_info.est_dims, ".-")
    eps_label = f"$\\epsilon_{{opt}} = {tune_info.opt_bandwidth: .3e}$"
    ax.axvline(
        float(np.log10(tune_info.opt_bandwidth)),
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


def plot_kaf_response_coeffs(
    coeffs: ArrayLike,
    i_fig: int = 1,
    dt: Optional[float] = None,
    title: Optional[str] = None,
) -> Figure:
    """Plot heatmap of KAF response coefficients."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    sns.heatmap(
        np.asarray(coeffs),
        ax=ax,
        cmap="seismic",
        center=0,
        robust=False,
        square=False,
        cbar_kws={
            "label": r"Projection coeff. $\langle\phi_i, U^{j} f\rangle$"
        },
    )
    ax.invert_yaxis()
    ax.set_xlabel(r"Eigenfunction index $i$")
    if dt is not None:
        ax.set_ylabel(f"Timestep $j$ ($dt = {dt:.3f})$")
    else:
        ax.set_ylabel("Timestep $j$")
    if title is not None:
        ax.set_title(title)
    return fig

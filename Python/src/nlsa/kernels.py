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
from nlsa.typing import SliceItem
from nlsa.utils import has_one_arg, has_two_args, swap_args
from numpy.typing import ArrayLike
from tabulate import tabulate
from typing import (
    Literal,
    Protocol,
    Self,
    runtime_checkable,
)
from collections.abc import Sequence

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

    manifold_dim: float | None = None
    """Manifold dimension."""

    bandwidth_scl: float = 1
    """Scaling factor to multiply estimated optimal kernel bandwidth."""

    bandwidth_batch_size: int | None = None
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


@runtime_checkable
class ImplementsTuneInfo[K, Ks, I](Protocol):
    """Represents objects holding kernel tuning information.

    The type parameters K, Ks, I represent scalars, collections of scalars, and
    integer indices, respectively.
    """

    @property
    def log10_bandwidths(self) -> Ks:
        """Trial kernel bandwidths."""
        ...

    @property
    def est_dims(self) -> Ks:
        """Estimated dimensions based on trial bandwidths."""
        ...

    @property
    def opt_bandwidth(self) -> K:
        """Optimal bandwidth from autotuning procedure."""
        ...

    @property
    def opt_dim(self) -> K:
        """Optimal (maximum) dimension from autotuning procedure."""
        ...

    @property
    def i_opt(self) -> I:
        """Index of optimal bandwidth in array of trial bandwidths."""
        ...

    @property
    def bandwidth(self) -> K:
        """Selected bandwidth after scaling by user-defined factor."""
        ...

    @property
    def dim(self) -> K:
        """Estimated dimension based on selected bandwidth."""
        ...

    @property
    def vol(self) -> K:
        """Estimated manifold volume based on selected bandwidth."""
        ...

    @property
    def kernel_vol(self) -> K:
        """Volume based on kernel integral."""
        ...


def tabulate_tune_info[K, Ks, I](
    impl: ImplementsTuneInfo[K, Ks, I],
    name: str = "Kernel Tuning Info",
    show: bool = True,
) -> str:
    """Create tabulated summary of an ImplementsTuneInfo instance."""
    headers = [name, "Value"]
    data = {
        "Optimal bandwidth index": f"{impl.i_opt}",
        "Optimal bandwidth": f"{impl.opt_bandwidth:.3e}",
        "Optimal dimension": f"{impl.opt_dim:.3e}",
        "Bandwidth used for diffusion maps": f"{impl.bandwidth:.3e}",
        "Dimension based on diffusion maps bandwidth": f"{impl.dim:.3e}",
        "Manifold volume": f"{impl.vol:.3e}",
        "Kernel volume": f"{impl.kernel_vol: .3e}",
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

    normalization: Literal["laplace", "fokkerplanck"] | None
    """Kernel normalization method."""

    eigensolver: Literal["eigh", "eigsh"]
    """Eigensolver used for kernel eigendecomposition."""

    num_eigs: int
    """Number of kernel eigenvalue/eigenvector pairs to compute."""

    batch_size: int | None = None
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

    batch_size: int | None = None
    """Maximum batch size for matrix-matrix products."""

    def __str__(self) -> str:
        """Create string representation of bistochastic kernel parameters."""
        return "_".join(
            ("bistochastic", self.eigensolver, f"neigs{self.num_eigs}")
        )


type KernelPars = DmKernelPars | BsKernelPars


@runtime_checkable
class ImplementsKernelEigen[K, Ks, V, Vs](Protocol):
    """Represents objects holding kernel spectral data."""

    @property
    def evals(self) -> Ks:
        """Kernel eigenvalues."""
        ...

    @property
    def evecs(self) -> Vs:
        """Kernel eigenvectors."""
        ...

    @property
    def dual_evecs(self) -> Vs:
        """Dual (left) kernel eigenvectors."""
        ...

    @property
    def weights(self) -> V:
        """Inner product weights that orthonormalize the eigenvectors."""
        ...

    @property
    def bandwidth(self) -> K:
        """Bandwidth parameter."""
        ...


@runtime_checkable
class ImplementsSliceableKernelEigen[K, Ks, V, Vs](
    ImplementsKernelEigen[K, Ks, V, Vs], Protocol
):
    """Represents objects holding sliceable kernel spectral data."""

    def isel(self, s: SliceItem) -> Self:
        """Slice an ImplementsKernelEigen object."""
        ...


def tabulate_eigen[K: ArrayLike, Ks: ArrayLike, V: ArrayLike, Vs: ArrayLike](
    impl: ImplementsKernelEigen[K, Ks, V, Vs],
    num_tabulate: int | None = None,
    headers: Sequence[str] | None = None,
    show: bool = True,
) -> str:
    """Tabulate the eigenvalues in an ImplementsKernelEigen object."""
    data = np.vstack((impl.evals,))[:, :num_tabulate].T
    if headers is None:
        headers = ["Kernel eigenvalues"]
    table = tabulate(data, headers=headers, floatfmt=".4f", showindex=True)
    if show:
        print(table)
    return table


def num_eigs_in_eigen[K, Ks: Sized, V, Vs: Sized](
    impl: ImplementsKernelEigen[K, Ks, V, Vs],
) -> int:
    """Return number of eigenvalues in ImplementsKernelEigenObject."""
    return len(impl.evals)


def slice_eigen[K, Ks, V, Vs](
    eigen: ImplementsSliceableKernelEigen[K, Ks, V, Vs],
    which_eigs: int | tuple[int, int] | list[int] | None = None,
) -> ImplementsSliceableKernelEigen[K, Ks, V, Vs]:
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


class ImplementsKernelEigenbasis[X, Y, V, K, Ks, I](
    alg.ImplementsL2FnEigenbasis[X, Y, V, K, Ks, I], Protocol
):
    """Implement kernel eigenbasis."""

    @property
    def lapl_spec(self) -> Ks:
        """Laplace spectrum."""
        ...

    def lapl_evl(self, i: I, /) -> K:
        """Return Laplacian eigenvalues."""
        ...


def make_scaled_sqdist[X, K](
    impl: alg.ImplementsRealScalarField[K],
    sqdist: Callable[[X, X], K],
    bandwdith_func: Callable[[X], K],
) -> Callable[[X, X], K]:
    """Make scaled square distance function from bandwidth function."""
    func: FunctionAlgebraWithCalculus[X, X, K, K] = (
        fun.function_algebra_with_calculus(
            codomain=scl.AsAlgebraWithCalculus(impl)
        )
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


def make_data_driven_scaled_sqdist[K, X, Data](
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
        _scaled_sqdist = make_scaled_sqdist(impl, _sqdist, _bandwidth_func)
        return _scaled_sqdist(x, y)

    return scaled_sqdist


def make_data_driven_rbf_kernel[K, X, Data](
    impl: alg.ImplementsRealScalarField[K],
    shape_func: Callable[[K], K],
    sqdist: Callable[[X, X], K] | Callable[[Data, X, X], K],
    bandwidth: K,
) -> Callable[[Data, X, X], K]:
    """Make data-driven, bandwidth-parameterized RBF kernel."""

    def kernel(data: Data, x: X, y: X) -> K:
        if has_two_args(sqdist):
            _sqdist = sqdist
        else:
            _sqdist = partial(sqdist, data)
        kernel = make_rbf_kernel(impl, shape_func, _sqdist, bandwidth)
        return kernel(x, y)

    return kernel


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
        fun.bivariate_function_div_bimodule(
            codomain=scl.AsDivBimodule(impl.scl)
        )
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
        fun.bivariate_function_div_bimodule(
            codomain=scl.AsDivBimodule(impl.scl)
        )
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
        fun.bivariate_function_div_bimodule(
            codomain=scl.AsDivBimodule(impl.scl)
        )
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
    func: FunctionAlgebraWithCalculus[X, K, K] = (
        fun.function_algebra_with_calculus(
            codomain=scl.AsAlgebraWithCalculus(impl.scl)
        )
    )
    func2: BivariateFunctionDivBimodule[X, X, K, K] = (
        fun.bivariate_function_div_bimodule(
            codomain=scl.AsDivBimodule(impl.scl)
        )
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
    func: FunctionAlgebraWithCalculus[X, K, K] = (
        fun.function_algebra_with_calculus(
            codomain=scl.AsAlgebraWithCalculus(impl.scl)
        )
    )
    func2: BivariateFunctionDivBimodule[X, X, K, K] = (
        fun.bivariate_function_div_bimodule(
            codomain=scl.AsDivBimodule(impl.scl)
        )
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


def make_data_driven_dmsym_kernel_op[Data, X, K, V](
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[X, K, V, K]],
    kernel: Callable[[X, X], K] | Callable[[Data, X, X], K],
    normalization: Literal["laplace", "fokkerplanck"] | None,
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
                normalized_kernel = dmsym_normalize(l2x, _kernel, alpha="1")
            case "fokkerplanck":
                normalized_kernel = dmsym_normalize(l2x, _kernel, alpha="0.5")
            case None:
                normalized_kernel = _kernel
        _kernel_op = fun.compose(
            l2x.incl, make_integral_operator(l2x, normalized_kernel)
        )
        return _kernel_op(v)

    return kernel_op


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
    func: FunctionAlgebraWithCalculus[X, K, K] = (
        fun.function_algebra_with_calculus(
            codomain=scl.AsAlgebraWithCalculus(impl.scl)
        )
    )
    func2: BivariateFunctionDivBimodule[X, X, K, K] = (
        fun.bivariate_function_div_bimodule(
            codomain=scl.AsDivBimodule(impl.scl)
        )
    )
    k_op = make_integral_operator(impl, k)
    k_op = make_integral_operator(impl, k)
    d = k_op(impl.unit())
    k_r = func2.rdiv(k, d)
    k_r_op = make_integral_operator(impl, k_r)
    q = k_r_op(impl.unit())
    k_q = func2.rdiv(k, func.sqrt(q))
    k_bs = func2.ldiv(d, k_q)
    return k_bs


def make_data_driven_bs_kernel_op[Data, X, K, V](
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[X, K, V, K]],
    kernel: Callable[[X, X], K] | Callable[[Data, X, X], K],
    adj: bool = False,
) -> Callable[[Data, V], V]:
    """Make data-driven kernel integral op with bistochastic normalization."""

    def kernel_op(data: Data, v: V) -> V:
        l2x = impl_l2(data)
        if has_two_args(kernel):
            _kernel = kernel
        else:
            _kernel = partial(kernel, data)
        bs_kernel = bs_normalize(l2x, _kernel)
        if adj:
            bs_kernel = swap_args(bs_kernel)
        kernel_op = fun.compose(
            l2x.incl, make_integral_operator(l2x, bs_kernel)
        )
        return kernel_op(v)

    return kernel_op


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
    normalization: K | None = None,
) -> Callable[[X], K]:
    """Make bandwidth function for variable-bandwidth kernel."""
    func: FunctionAlgebraWithCalculus[X, K, K] = (
        fun.function_algebra_with_calculus(
            codomain=scl.AsAlgebraWithCalculus(impl.scl)
        )
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


# NOTE: In a posssible Mojo implementation, impl_l2, shape_func, etc. would
# be passed in as compile-time parameters, and the arguments of the
# resulting Callable would be regular run-time arguments.
def make_data_driven_bandwidth_function[Data, X, K, V, Ks, I](
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[X, K, V, K]],
    shape_func: Callable[[K], K],
    sqdist: Callable[[X, X], K],
    tune_info: ImplementsTuneInfo[K, Ks, I],
) -> Callable[[Data, X], K]:
    """Make data-driven kernel bandwidth function."""

    def bandwidth_func(data: Data, x: X) -> K:
        l2x = impl_l2(data)
        kernel_family = make_rbf_kernel_family(l2x.scl, shape_func, sqdist)
        f = make_bandwidth_function(
            l2x,
            kernel_family(tune_info.bandwidth),
            dim=tune_info.dim,
            vol=tune_info.vol,
            normalization=tune_info.kernel_vol,
        )
        return f(x)

    return bandwidth_func


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
    func: FunctionAlgebra[X, X, K, K] = fun.function_algebra(
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


def make_eigenvector_extension_dm[X, K, V](
    l2x: alg.ImplementsL2FnAlgebra[X, K, V, K],
    kernel: Callable[[X, X], K],
    normalization: Literal["laplace", "fokkerplanck"] | None,
) -> tuple[Callable[[V, K], F[X, K]], Callable[[X, X], K]]:
    """Make Nystrom extension for diffusion maps kernels."""
    match normalization:
        case "laplace":
            extension_kernel = dm_normalize(l2x, kernel, alpha="1")
        case "fokkerplanck":
            extension_kernel = dm_normalize(l2x, kernel, alpha="0.5")
        case None:
            extension_kernel = kernel
    extension_kernel_op: Callable[[V], F[X, K]] = make_integral_operator(
        l2x, extension_kernel
    )

    def nyst(phi: V, lamb: K) -> F[X, K]:
        return extension_kernel_op(l2x.sdiv(lamb, phi))

    return nyst, extension_kernel


def make_eigenvector_extension_bs[X, K, V](
    l2x: alg.ImplementsL2FnAlgebra[X, K, V, K], kernel: Callable[[X, X], K]
) -> tuple[Callable[[V, K], F[X, K]], Callable[[X, X], K]]:
    """Make Nystrom extension for bistochastic kernels."""
    extension_kernel = bs_normalize(l2x, kernel)
    extension_kernel_op: Callable[[V], F[X, K]] = make_integral_operator(
        l2x, extension_kernel
    )

    def nyst(phi: V, lamb: K) -> F[X, K]:
        return extension_kernel_op(l2x.sdiv(l2x.scl.sqrt(lamb), phi))

    return nyst, extension_kernel


def make_eigenvector_extension[X, K, V](
    pars: KernelPars,
    l2x: alg.ImplementsL2FnAlgebra[X, K, V, K],
    kernel: Callable[[X, X], K],
) -> tuple[Callable[[V, K], F[X, K]], Callable[[X, X], K]]:
    """Make Nystrom extension for diffusion maps and bistochastic kernels."""
    match pars:
        case DmKernelPars():
            nyst, extension_kernel = make_eigenvector_extension_dm(
                l2x, kernel, pars.normalization
            )
        case BsKernelPars():
            nyst, extension_kernel = make_eigenvector_extension_bs(l2x, kernel)
    return nyst, extension_kernel


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
        fun.bivariate_function_div_bimodule(
            codomain=scl.AsDivBimodule(impl.scl)
        )
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


def to_laplace_eigenvalues(
    lambs: ArrayLike,
    bandwidth: ArrayLike,
    method: Literal["lin", "log", "inv"] = "log",
) -> ArrayLike:
    """Compute Laplace eigenvalues from kernel eigenvalues."""
    lambs = np.asarray(lambs)
    bandwidth = np.asarray(bandwidth)
    match method:
        case "lin":
            etas = 4 * (1 - lambs) / bandwidth**2
        case "log":
            etas = -4 * np.log(lambs) / bandwidth**2
        case "inv":
            inv_lambs = 1 / lambs
            etas = (inv_lambs - 1) / (inv_lambs[1] - 1)
    return etas


def plot_kernel_tuning[A: ArrayLike](
    tune_info: ImplementsTuneInfo[A, A, A],
    title: str | None = None,
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


def plot_laplacian_spectrum(
    kernel_eigen: ImplementsKernelEigen[
        ArrayLike, ArrayLike, ArrayLike, ArrayLike
    ],
    num_eigs_plt: int | None = None,
    i_fig: int = 1,
) -> Figure:
    """Plot spectrum of Laplacian eigenvalues."""
    kernel_evals = np.asarray(kernel_eigen.evals)[:num_eigs_plt]
    lapl_evals = partial(
        to_laplace_eigenvalues, kernel_evals, kernel_eigen.bandwidth
    )
    idx_evals = np.arange(1, len(kernel_evals))
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    ax.plot(
        idx_evals,
        np.log10(np.asarray(lapl_evals("lin"))[1:]),
        ".",
        label=r"$4(1-\lambda_j)/\epsilon^2$",
    )
    ax.plot(
        idx_evals,
        np.log10(np.asarray(lapl_evals("log"))[1:]),
        ".",
        label=r"$-4\log\lambda_j/\epsilon^2$",
    )
    ax.plot(
        idx_evals,
        np.log10(np.asarray(lapl_evals("inv"))[1:]),
        ".",
        label=r"$(\lambda_j^{-1}-1)/(\lambda_1-1)$",
    )
    ax.grid()
    ax.legend()
    ax.set_xlabel("$j$")
    ax.set_ylabel(r"$\log_{10}\eta_j$")
    ax.set_title("Laplacian eigenvalues")
    return fig


def plot_kaf_expansion_coeffs(
    coeffs: ArrayLike,
    i_fig: int = 1,
    dt: float | None = None,
    title: str | None = None,
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

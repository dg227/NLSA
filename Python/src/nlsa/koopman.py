"""Provide generic functions and classes for Koopman operator computations."""

import matplotlib.pyplot as plt
import nlsa.abstract_algebra as alg
import numpy as np
import seaborn as sns
from collections.abc import Callable, Sequence, Sized
from dataclasses import dataclass
from matplotlib.figure import Figure
from nlsa.typing import (
    SliceItem,
)
from numpy.typing import ArrayLike
from tabulate import tabulate
from typing import (
    Literal,
    Protocol,
    Self,
    final,
    runtime_checkable,
)

type F[*Xs, Y] = Callable[[*Xs], Y]


class MeanZeroEigsGalerkin(Protocol):
    """Represent objects using zero-mean Galerkin approximation spaces."""

    which_eigs_galerkin: int | tuple[int, int] | list[int]
    """Kernel eigenvectors used for Galerkin approximation."""

    @property
    def dim_galerkin(self) -> int:
        """Determine dimension of Galerkin approximation space."""
        match self.which_eigs_galerkin:
            case int():
                dim = self.which_eigs_galerkin
            case tuple():
                dim = (
                    self.which_eigs_galerkin[1]
                    - self.which_eigs_galerkin[0]
                    + 1
                )
            case list():
                dim = len(self.which_eigs_galerkin)
        return dim

    @property
    def which_kernel_eigs(self) -> int | list[int]:
        """Include the constant kernel eigenvector."""
        match self.which_eigs_galerkin:
            case int():
                which_kernel_eigs = self.which_eigs_galerkin + 1
            case tuple():
                which_kernel_eigs = [0] + list(
                    range(
                        self.which_eigs_galerkin[0],
                        self.which_eigs_galerkin[1] + 1,
                    )
                )
            case list():
                which_kernel_eigs = [0] + self.which_eigs_galerkin
        return which_kernel_eigs


# TODO: Consider moving batching parameters to a different class (see
# kernels module).
# NOTE: The batching parameter grad_batch_size may not play an actual role in
# the current implementation. Consider removing it.
@final
@dataclass(frozen=True, slots=True)
class KoopmanParsDiff(MeanZeroEigsGalerkin):
    """Eigendecomposition parameters for diffusion-regularized generator."""

    fd_order: Literal[2, 4, 6, 8]
    """Finite-difference order."""

    dt: float
    """Finite difference interval."""

    tau: float
    """Regularization parameter."""

    antisym: bool
    """Perform antisymmetrization."""

    which_eigs_galerkin: int | tuple[int, int] | list[int]
    """Kernel eigenvectors used for Galerkin approximation of the generator."""

    num_eigs: int | None = None
    """Number of Koopman eigenfunctions to compute."""

    laplacian_method: Literal["log", "lin", "inv"] = "log"
    """Method for computing Laplacian eigenvalues."""

    sort_by: Literal["energy", "frequency"] = "frequency"
    """Koopman eigenvalue/eigenvector sorting."""

    eval_tx_batch_size: int | None = None
    """Batch size for tangent evaluation functional."""

    grad_batch_size: int | None = None
    """Batch size for gradient computation."""

    gram_batch_size: int | None = None
    """Batch size of inner product computation for generator."""

    def __str__(self) -> str:
        """Create string representation of eigendecommposition parameters."""
        antisym_str = "antisym" if self.antisym else ""
        match self.which_eigs_galerkin:
            case int():
                eigs_galerkin_str = "-".join(
                    map(str, (0, self.which_eigs_galerkin))
                )
            case tuple():
                eigs_galerkin_str = "-".join(
                    map(str, self.which_eigs_galerkin)
                )
            case list():
                eigs_galerkin_str = "_".join(
                    map(str, self.which_eigs_galerkin)
                )
        num_eigs_str = (
            f"neigs{self.num_eigs}" if self.num_eigs is not None else ""
        )
        return "_".join(
            filter(
                None,
                (
                    "gen_diff",
                    f"dt{self.dt:.2g}",
                    f"fdord{self.fd_order}",
                    self.laplacian_method,
                    f"tau{self.tau:.2g}",
                    antisym_str,
                    eigs_galerkin_str,
                    num_eigs_str,
                    self.sort_by,
                ),
            )
        )


@final
@dataclass(frozen=True, slots=True)
class KoopmanParsLapl(MeanZeroEigsGalerkin):
    """Eigendecomposition parameters for Qz operator (Laplace transform)."""

    num_quad: int
    """Number of quadrature points"""

    bandwidth: float
    """Resolvent parameter."""

    dt: float
    """Transform timestep."""

    tau: float
    """Regularization parameter."""

    which_eigs_galerkin: int | tuple[int, int] | list[int]
    """Kernel eigenvectors used for Galerkin approximation of Qz operator."""

    antisym: bool = True
    """Perform antisymmetrization."""

    num_eigs: int | None = None
    """Number of Koopman eigenfunctions to compute."""

    laplacian_method: Literal["log", "lin", "inv"] = "log"
    """Method for computing Laplacian eigenvalues."""

    smoothing_kernel: Literal["exponential", "fejer"] = "exponential"
    """Smoothing kernel used for operator compactification."""

    sort_by: Literal["energy", "frequency"] = "frequency"
    """Koopman eigenvalue/eigenvector sorting."""

    eval_quad_batch_size: int | None = None
    """Evaluation batch size for quadrature in resolvent computation."""

    quad_batch_size: int | None = None
    """Batch size for quadrature in resolvent computation."""

    gram_batch_size: int | None = None
    """Batch size of inner product computation for Qz operator."""

    def __str__(self) -> str:
        """Create string representation of eigendecommposition parameters."""
        match self.which_eigs_galerkin:
            case int():
                eigs_galerkin_str = "-".join(
                    map(str, (0, self.which_eigs_galerkin))
                )
            case tuple():
                eigs_galerkin_str = "-".join(
                    map(str, self.which_eigs_galerkin)
                )
            case list():
                eigs_galerkin_str = "_".join(
                    map(str, self.which_eigs_galerkin)
                )
        num_eigs_str = (
            f"neigs{self.num_eigs}" if self.num_eigs is not None else ""
        )
        return "_".join(
            filter(
                None,
                (
                    "lapl",
                    f"z{self.bandwidth:.2g}",
                    f"dt{self.dt:.2g}",
                    f"nq{self.num_quad}",
                    self.laplacian_method,
                    self.smoothing_kernel,
                    f"tau{self.tau:.2g}",
                    num_eigs_str,
                    eigs_galerkin_str,
                    self.sort_by,
                ),
            )
        )


@final
@dataclass(frozen=True, slots=True)
class KoopmanParsGauss(MeanZeroEigsGalerkin):
    """Eigendecomposition parameters for Iz operator (Gauss transform)."""

    num_quad: int
    """Number of quadrature points"""

    bandwidth: float
    """Resolvent parameter."""

    dt: float
    """Transform timestep."""

    tau: float
    """Regularization parameter."""

    which_eigs_galerkin: int | tuple[int, int] | list[int]
    """Kernel eigenvectors used for Galerkin approximation of Qz operator."""

    antisym: bool = True
    """Perform antisymmetrization."""

    num_eigs: int | None = None
    """Number of Koopman eigenfunctions to compute."""

    laplacian_method: Literal["log", "lin", "inv"] = "log"
    """Method for computing Laplacian eigenvalues."""

    smoothing_kernel: Literal["exponential", "fejer"] = "exponential"
    """Smoothing kernel used for operator compactification."""

    sort_by: Literal["energy", "frequency"] = "frequency"
    """Koopman eigenvalue/eigenvector sorting."""

    eval_quad_batch_size: int | None = None
    """Evaluation batch size for quadrature in resolvent computation."""

    quad_batch_size: int | None = None
    """Batch size for quadrature in resolvent computation."""

    gram_batch_size: int | None = None
    """Batch size of inner product computation for Qz operator."""

    def __str__(self) -> str:
        """Create string representation of eigendecommposition parameters."""
        match self.which_eigs_galerkin:
            case int():
                eigs_galerkin_str = "-".join(
                    map(str, (0, self.which_eigs_galerkin))
                )
            case tuple():
                eigs_galerkin_str = "-".join(
                    map(str, self.which_eigs_galerkin)
                )
            case list():
                eigs_galerkin_str = "_".join(
                    map(str, self.which_eigs_galerkin)
                )
        num_eigs_str = (
            f"neigs{self.num_eigs}" if self.num_eigs is not None else ""
        )
        return "_".join(
            filter(
                None,
                (
                    "gauss",
                    f"z{self.bandwidth:.2g}",
                    f"dt{self.dt:.2g}",
                    f"nq{self.num_quad}",
                    self.laplacian_method,
                    self.smoothing_kernel,
                    f"tau{self.tau:.2g}",
                    num_eigs_str,
                    eigs_galerkin_str,
                    self.sort_by,
                ),
            )
        )


@final
@dataclass(frozen=True, slots=True)
class KoopmanParsTransf(MeanZeroEigsGalerkin):
    """Eigendecomposition parameters for Iz operator (Gauss transform)."""

    transform: Literal["laplace", "gauss"]
    """Transform method."""

    quadrature: Literal["trapezoidal", "simpson"]
    """Quadrature method."""

    num_quad: int
    """Number of quadrature points"""

    bandwidth: float
    """Resolvent parameter."""

    dt: float
    """Transform timestep."""

    tau: float
    """Regularization parameter."""

    which_eigs_galerkin: int | tuple[int, int] | list[int]
    """Kernel eigenvectors used for Galerkin approximation of Qz operator."""

    antisym: bool = True
    """Perform antisymmetrization."""

    num_eigs: int | None = None
    """Number of Koopman eigenfunctions to compute."""

    laplacian_method: Literal["log", "lin", "inv"] = "log"
    """Method for computing Laplacian eigenvalues."""

    smoothing_kernel: Literal["exponential", "fejer"] = "exponential"
    """Smoothing kernel used for operator compactification."""

    sort_by: Literal["energy", "frequency"] = "frequency"
    """Koopman eigenvalue/eigenvector sorting."""

    eval_quad_batch_size: int | None = None
    """Evaluation batch size for quadrature in resolvent computation."""

    quad_batch_size: int | None = None
    """Batch size for quadrature in resolvent computation."""

    gram_batch_size: int | None = None
    """Batch size of inner product computation for Qz operator."""

    def __str__(self) -> str:
        """Create string representation of eigendecommposition parameters."""
        match self.which_eigs_galerkin:
            case int():
                eigs_galerkin_str = "-".join(
                    map(str, (0, self.which_eigs_galerkin))
                )
            case tuple():
                eigs_galerkin_str = "-".join(
                    map(str, self.which_eigs_galerkin)
                )
            case list():
                eigs_galerkin_str = "_".join(
                    map(str, self.which_eigs_galerkin)
                )
        num_eigs_str = (
            f"neigs{self.num_eigs}" if self.num_eigs is not None else ""
        )
        match self.quadrature:
            case "trapezoidal":
                quad_str = "trap"
            case "simpson":
                quad_str = "simpson"
        return "_".join(
            filter(
                None,
                (
                    self.transform,
                    f"z{self.bandwidth:.2g}",
                    f"dt{self.dt:.2g}",
                    quad_str,
                    f"nq{self.num_quad}",
                    self.laplacian_method,
                    self.smoothing_kernel,
                    f"tau{self.tau:.2g}",
                    num_eigs_str,
                    eigs_galerkin_str,
                    self.sort_by,
                ),
            )
        )


# type KoopmanParsTransf = KoopmanParsGauss | KoopmanParsLapl
type KoopmanPars = KoopmanParsDiff | KoopmanParsTransf


@runtime_checkable
class ImplementsKoopmanEigen[Rs, Cs, Vs](Protocol):
    """Represents objects holding Koopman spectral data."""

    @property
    def evals(self) -> Cs:
        """Operator eigenvalues."""
        ...

    @property
    def gen_evals(self) -> Cs:
        """Generator eigenvalues."""
        ...

    @property
    def engys(self) -> Rs:
        """Dirichlet energies."""
        ...

    @property
    def efreqs(self) -> Rs:
        """Koopman eigenfrequencies."""
        ...

    @property
    def eperiods(self) -> Rs:
        """Return Koopman eigenperiods."""
        ...

    @property
    def evec_coeffs(self) -> Vs:
        """Basis expansion coefficients of Koopman eigenvectors."""
        ...

    @property
    def dual_evec_coeffs(self) -> Vs:
        """Basis expansion coefficients of dual (left) Koopman eigenvectors."""
        ...


@runtime_checkable
class ImplementsSliceableKoopmanEigen[Rs, Cs, Vs](
    ImplementsKoopmanEigen[Rs, Cs, Vs], Protocol
):
    """Represents objects holding sliceable Koopman spectral data."""

    def isel(self, s: SliceItem) -> Self:
        """Slice an ImplementsKoopmanEigen object."""
        ...


def tabulate_eigen[Rs: ArrayLike, Cs: ArrayLike, Vs: ArrayLike](
    impl: ImplementsKoopmanEigen[Rs, Cs, Vs],
    num_tabulate: int | None = None,
    headers: Sequence[str] | None = None,
    frequency_scaling: float | None = None,
    period_scaling: float | None = None,
    show: bool = True,
) -> str:
    """Tabulate the eigenvalues in an ImplementsKernelEigen object."""
    if frequency_scaling is None:
        frequency_scaling = 1
    efreqs = np.asarray(impl.efreqs) * frequency_scaling
    evals = np.asarray(impl.evals)
    if period_scaling is None:
        period_scaling = 1
    eperiods = np.asarray(impl.eperiods) * period_scaling
    data = np.vstack(
        (
            np.real(evals),
            np.imag(evals),
            impl.engys,
            efreqs,
            eperiods,
        )
    )[:, :num_tabulate].T
    if headers is None:
        headers = [
            "Koopman evals (Re)",
            "(Im)",
            "Dirichlet engys.",
            "Eigenfreqs.",
            "Eigenperiods",
        ]
    table = tabulate(data, headers=headers, floatfmt=".4f", showindex=True)
    if show:
        print(table)
    return table


def num_eigs_in_eigen[Rs, Cs: Sized, Vs](
    impl: ImplementsKoopmanEigen[Rs, Cs, Vs],
) -> int:
    """Return number of eigenvalues in ImplementsKernelEigenObject."""
    return len(impl.evals)


def slice_eigen[Rs, Cs, Vs](
    eigen: ImplementsSliceableKoopmanEigen[Rs, Cs, Vs],
    which_eigs: int | tuple[int, int] | list[int] | None = None,
) -> ImplementsSliceableKoopmanEigen[Rs, Cs, Vs]:
    """Slice KoopmanEigen object using `which_eigs` convention."""
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


class ImplementsKoopmanEigenbasis[X, Y, V, K, Ks, I](
    alg.ImplementsL2FnEigenbasis[X, Y, V, K, Ks, I], Protocol
):
    """Implement Koopman eigenbasis."""

    @property
    def gen_spec(self) -> Ks:
        """Generator spectrum."""
        ...

    @property
    def efreqs(self) -> Ks:
        """Eigenfrequencies."""
        ...

    @property
    def eperiods(self) -> Ks:
        """Eigenperiods."""
        ...

    @property
    def engys(self) -> Ks:
        """Dirichlet energies."""
        ...

    def gen_evl(self, i: I, /) -> K:
        """Return generator eigenvalues."""
        ...

    def efreq(self, i: I, /) -> K:
        """Return generator eigenfrequencies."""
        ...

    def eperiod(self, i: I, /) -> K:
        """Return generator eigenperiods."""
        ...

    def engy(self, i: I, /) -> K:
        """Return Dirichlet energies."""
        ...


def plot_operator_matrix(
    op_mat: ArrayLike, i_fig: int = 1, title: str | None = None
) -> Figure:
    """Plot heatmap of matrices used in Koopman operator problems."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    sns.heatmap(
        np.asarray(op_mat), ax=ax, cmap="seismic", center=0, robust=False
    )
    if title is not None:
        ax.set_title(title)
    return fig


def plot_generator_spectrum(
    koopman_eigen: ImplementsKoopmanEigen[ArrayLike, ArrayLike, ArrayLike],
    num_eigs_plt: int | None = None,
    frequency_symbol: str = "$\\omega_j$",
    frequency_scaling: float = 1,
    frequency_units: str | None = None,
    i_fig: int = 1,
) -> Figure:
    """Plot spectrum of Koopman generator."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    gen_evals = np.asarray(koopman_eigen.gen_evals)
    engys = np.asarray(koopman_eigen.engys)
    if num_eigs_plt is None:
        num_eigs_plt = len(gen_evals)
    im = ax.scatter(
        engys[:num_eigs_plt],
        np.imag(gen_evals[:num_eigs_plt]) * frequency_scaling,
        s=10,
        c=np.arange(num_eigs_plt),
    )
    cb = fig.colorbar(im, ax=ax)
    ax.set_xlabel("Dirichlet energy $E_j$")
    if frequency_units is not None:
        units_str = f" ({frequency_units})"
    else:
        units_str = ""
    ax.set_ylabel(f"Eigenfrequency {frequency_symbol}{units_str}")
    cb.set_label("$j$")
    ax.grid(True)
    return fig

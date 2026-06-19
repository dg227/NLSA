"""Provide generic functions and classes for Koopman operator computations."""

import matplotlib.pyplot as plt
import nlsa.abstract_algebra as alg
import numpy as np
import seaborn as sns
from collections.abc import Callable
from dataclasses import dataclass
from matplotlib.figure import Figure
from nlsa.typing import (
    SliceItem,
    is_sliceable,
)
from numpy.typing import ArrayLike
from typing import Literal, NamedTuple, Optional, Sized, final

type F[*Xs, Y] = Callable[[*Xs], Y]


# TODO: Consider moving batching parameters to a different class (see
# kernels module).
# NOTE: The batching parameter grad_batch_size may not play an actual role in
# the current implementation. Consider removing it.
@dataclass(frozen=True, slots=True)
class KoopmanParsDiff:
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

    num_eigs: Optional[int] = None
    """Number of Koopman eigenfunctions to compute."""

    laplacian_method: Literal["log", "lin", "inv"] = "log"
    """Method for computing Laplacian eigenvalues."""

    sort_by: Literal["energy", "frequency"] = "frequency"
    """Koopman eigenvalue/eigenvector sorting."""

    eval_tx_batch_size: Optional[int] = None
    """Batch size for tangent evaluation functional."""

    grad_batch_size: Optional[int] = None
    """Batch size for gradient computation."""

    gram_batch_size: Optional[int] = None
    """Batch size of inner product computation for generator."""

    @property
    def dim_galerkin(self) -> int:
        """Determine dimension of Galerkin approximation space."""
        match self.which_eigs_galerkin:
            case int():
                dim = self.which_eigs_galerkin
            case tuple():
                dim = self.which_eigs_galerkin[1] - self.which_eigs_galerkin[0]
            case list():
                dim = len(self.which_eigs_galerkin)
        return dim

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


@dataclass(frozen=True, slots=True)
class KoopmanParsLapl:
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

    num_eigs: Optional[int] = None
    """Number of Koopman eigenfunctions to compute."""

    laplacian_method: Literal["log", "lin", "inv"] = "log"
    """Method for computing Laplacian eigenvalues."""

    smoothing_kernel: Literal["exponential", "fejer"] = "exponential"
    """Smoothing kernel used for operator compactification."""

    sort_by: Literal["energy", "frequency"] = "frequency"
    """Koopman eigenvalue/eigenvector sorting."""

    eval_quad_batch_size: Optional[int] = None
    """Evaluation batch size for quadrature in resolvent computation."""

    quad_batch_size: Optional[int] = None
    """Batch size for quadrature in resolvent computation."""

    gram_batch_size: Optional[int] = None
    """Batch size of inner product computation for Qz operator."""

    @property
    def dim_galerkin(self) -> int:
        """Determine dimension of Galerkin approximation space."""
        match self.which_eigs_galerkin:
            case int():
                dim = self.which_eigs_galerkin
            case tuple():
                dim = self.which_eigs_galerkin[1] - self.which_eigs_galerkin[0]
            case list():
                dim = len(self.which_eigs_galerkin)
        return dim

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


@dataclass(frozen=True, slots=True)
class KoopmanParsGauss:
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

    num_eigs: Optional[int] = None
    """Number of Koopman eigenfunctions to compute."""

    laplacian_method: Literal["log", "lin", "inv"] = "log"
    """Method for computing Laplacian eigenvalues."""

    smoothing_kernel: Literal["exponential", "fejer"] = "exponential"
    """Smoothing kernel used for operator compactification."""

    sort_by: Literal["energy", "frequency"] = "frequency"
    """Koopman eigenvalue/eigenvector sorting."""

    eval_quad_batch_size: Optional[int] = None
    """Evaluation batch size for quadrature in resolvent computation."""

    quad_batch_size: Optional[int] = None
    """Batch size for quadrature in resolvent computation."""

    gram_batch_size: Optional[int] = None
    """Batch size of inner product computation for Qz operator."""

    @property
    def dim_galerkin(self) -> int:
        """Determine dimension of Galerkin approximation space."""
        match self.which_eigs_galerkin:
            case int():
                dim = self.which_eigs_galerkin
            case tuple():
                dim = self.which_eigs_galerkin[1] - self.which_eigs_galerkin[0]
            case list():
                dim = len(self.which_eigs_galerkin)
        return dim

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


type KoopmanParsTransf = KoopmanParsGauss | KoopmanParsLapl
type KoopmanPars = KoopmanParsDiff | KoopmanParsTransf


class KoopmanEigen[Rs, Cs, Css](NamedTuple):
    """NamedTuple containing Koopman spectral data."""

    evals: Cs
    """Operator eigenvalues."""

    gen_evals: Cs
    """Generator eigenvalues."""

    engys: Rs
    """Dirichlet energies."""

    efreqs: Rs
    """Koopman eigenfrequencies."""

    eperiods: Rs
    """Return Koopman eigenperiods."""

    evec_coeffs: Css
    """Basis expansion coefficients of Koopman eigenvectors."""

    dual_evec_coeffs: Css
    """Basis expansion coefficients of dual (left) Koopman eigenvectors."""

    @property
    def num_eigs(
        self,
    ) -> int:
        """Return number of eigenvalues/eigenvectors in KoopmanEigenObject."""
        assert isinstance(self.evals, Sized)
        return len(self.evals)

    def isel(
        self,
        s: SliceItem,
    ) -> "KoopmanEigen[Rs, Cs, Css]":
        """Slice a KoopmanEigen object."""
        assert is_sliceable(self.evals)
        assert is_sliceable(self.gen_evals)
        assert is_sliceable(self.engys)
        assert is_sliceable(self.efreqs)
        assert is_sliceable(self.eperiods)
        assert is_sliceable(self.evec_coeffs)
        assert is_sliceable(self.dual_evec_coeffs)
        return KoopmanEigen(
            evals=self.evals[s],
            gen_evals=self.gen_evals[s],
            efreqs=self.efreqs[s],
            engys=self.engys[s],
            eperiods=self.eperiods[s],
            evec_coeffs=self.evec_coeffs[s],
            dual_evec_coeffs=self.dual_evec_coeffs[s],
        )


@final
@dataclass(frozen=True, slots=True)
class KoopmanEigenbasis[X, K, V, Ks, I](
    alg.ImplementsDimensionedL2FnFrame[X, K, V, Ks, I]
):
    """Dataclass implementing frame operators for Koopman eigenbasis."""

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
    """Operator spectrum."""

    gen_spec: Ks
    """Generator spectrum."""

    efreqs: Ks
    """Eigenfrequencies."""

    eperiods: Ks
    """Eigenperiods."""

    engys: Ks
    """Dirichlet energies."""

    evl: Callable[[I], K]
    """Operator eigenvalues."""

    gen_evl: Callable[[I], K]
    """Generator eigenvalues."""

    efreq: Callable[[I], K]
    """Function indexing eigenfrequencies."""

    eperiod: Callable[[I], K]
    """Function indexing eigenperiods."""

    engy: Callable[[I], K]
    """Function indexing Dirichlet energies."""


def plot_operator_matrix(
    op_mat: ArrayLike, i_fig: int = 1, title: Optional[str] = None
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

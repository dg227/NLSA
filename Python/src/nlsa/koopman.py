"""Provide generic functions and classes for Koopman operator computations."""
import matplotlib.pyplot as plt
import nlsa.abstract_algebra as alg
import numpy as np
import seaborn as sns
from collections.abc import Callable
from dataclasses import dataclass
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from typing import Literal, Optional, final

type F[*Xs, Y] = Callable[[*Xs], Y]


@dataclass(frozen=True)
class KoopmanParsDiff:
    """Eigendecomposition parameters for diffusion-regularized generator."""

    fd_order: Literal[2, 4, 6, 8]
    """Finite-difference order."""

    dt: float
    """Finite difference interval."""

    tau: float
    """Regularization parameter."""

    antisym: bool
    """Perform antisymmetrization"""

    which_eigs_galerkin: int | tuple[int, int] | list[int]
    """Kernel eigenvectors used for Galerkin approximation of the generator."""

    num_eigs: Optional[int] = None
    """Number of Koopman eigenfunctions to compute."""

    laplace_method: Literal['log', 'lin', 'inv'] = "log"
    """Method for computing Laplacian eigenvalues."""

    sort_by: Literal['energy', 'frequency'] = "frequency"
    """Koopman eigenvalue/eigenvector sorting."""

    gram_batch_size: Optional[int] = None
    """Batch size of inner product computation for generator."""

    @property
    def dim_galerkin(self) -> int:
        """Determine dimension of Galerkin approximation space."""
        match self.which_eigs_galerkin:
            case int():
                dim = self.which_eigs_galerkin
            case (_, _):
                dim = self.which_eigs_galerkin[1] - self.which_eigs_galerkin[0]
            case list():
                dim = len(self.which_eigs_galerkin)
        return dim

    def __str__(self) -> str:
        """Create string representation of eigendecommposition parameters."""
        antisym_str = "antisym" if self.antisym else ""
        match self.which_eigs_galerkin:
            case int():
                eigs_galerkin_str = "-".join(map(str,
                                                 (0,
                                                  self.which_eigs_galerkin)))
            case tuple(ints) if all(isinstance(i, int) for i in ints) \
                    and len(ints) == 2:
                eigs_galerkin_str = "-".join(map(str,
                                                 self.which_eigs_galerkin))
            case _:
                eigs_galerkin_str = "_".join(map(str,
                                                 self.which_eigs_galerkin))
        num_eigs_str = f"neigs{self.num_eigs}" if self.num_eigs is not None \
            else ""
        return '_'.join(filter(None, ("gen_diff",
                                      f"dt{self.dt:.2g}",
                                      f"fdord{self.fd_order}",
                                      self.laplace_method,
                                      f"tau{self.tau:.2g}",
                                      antisym_str,
                                      eigs_galerkin_str,
                                      num_eigs_str,
                                      self.sort_by)))


@dataclass(frozen=True)
class KoopmanParsQz:
    """Eigendecomposition parameters for Qz operator."""

    fd_order: Literal[2, 4, 6, 8]
    """Finite-difference order."""

    dt: float
    """Finite difference interval."""

    res_z: float
    """Resolvent parameter."""

    tau: float
    """Regularization parameter."""

    which_eigs_galerkin: int | tuple[int, int] | list[int]
    """Kernel eigenvectors used for Galerkin approximation of Qz operator."""

    num_eigs: Optional[int] = None
    """Number of Koopman eigenfunctions to compute."""

    laplace_method: Literal['log', 'lin', 'inv'] = "log"
    """Method for computing Laplacian eigenvalues."""

    sort_by: Literal['energy', 'frequency'] = "frequency"
    """Koopman eigenvalue/eigenvector sorting."""

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
            case tuple(ints) if all(isinstance(i, int) for i in ints) \
                    and len(ints) == 2:
                dim = self.which_eigs_galerkin[1] - self.which_eigs_galerkin[0]
            case _:
                dim = len(self.which_eigs_galerkin)
        return dim

    def __str__(self) -> str:
        """Create string representation of eigendecommposition parameters."""
        match self.which_eigs_galerkin:
            case int():
                eigs_galerkin_str = "-".join(map(str,
                                                 (0,
                                                  self.which_eigs_galerkin)))
            case tuple(ints) if all(isinstance(i, int) for i in ints) \
                    and len(ints) == 2:
                eigs_galerkin_str = "-".join(map(str,
                                                 self.which_eigs_galerkin))
            case list():
                eigs_galerkin_str = "_".join(map(str,
                                                 self.which_eigs_galerkin))
        num_eigs_str = f"neigs{self.num_eigs}" if self.num_eigs is not None \
            else ""
        return "_".join(filter(None, ("qz",
                                      f"resz{self.res_z:.2g}",
                                      f"dt{self.dt:.2g}",
                                      self.laplace_method,
                                      f"tau{self.tau:.2g}",
                                      f"fdord{self.fd_order}",
                                      num_eigs_str,
                                      eigs_galerkin_str,
                                      self.sort_by)))


@final
@dataclass(frozen=True)
class KoopmanEigenbasis[X, K, V, Ks, I](
        alg.ImplementsDimensionedL2FnFrame[X, K, V, Ks, I]):
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


def plot_operator_matrix(op_mat: ArrayLike, i_fig: int = 1,
                         title: Optional[str] = None) -> Figure:
    """Plot heatmap of matrices used in Koopman operator problems."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    sns.heatmap(np.asarray(op_mat), ax=ax, cmap='seismic', center=0,
                robust=False)
    if title is not None:
        ax.set_title(title)
    return fig

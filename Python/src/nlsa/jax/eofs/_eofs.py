"""Provide classes and functions for EOF computations in JAX."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nlsa.abstract_algebra as alg
import nlsa.function_algebra as fun
import nlsa.jax.delays as dl
import nlsa.jax.vector_algebra as vec
import numpy as np
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from jax import Array, vmap
from jax.sharding import NamedSharding
from matplotlib.figure import Figure
from nlsa.eofs import EEOFPars
from nlsa.jax.sharding import (
    NamedSharder,
    SvdShardings,
    make_svd_with_sharding_constraints,
)
from nlsa.jax.typing import PyTree
from nlsa.typing import SliceItem, is_array_like
from tabulate import tabulate
from typing import NamedTuple, Self, final
from collections.abc import Sequence

type V = Array
type Vs = Array
type Vd = Array
type Xd = Array
type Xs = Array
type Y = Array
type R = Array
type Rs = Array
type Shape = tuple[int, ...]
type Idx = int | Array
type F[*Xs, Y] = Callable[[*Xs], Y]


class EEOFEigen(NamedTuple):
    """NamedTuple containing kernel extended EOF data."""

    sing_vals: Array
    """Singular values."""

    eeofs: Array
    """Extended EOFs."""

    pcs: Array
    """Principal components."""

    @property
    def num_eigs(
        self,
    ) -> int:
        """Return number of eigenvalues/eigenvectors in EEOFEigen object."""
        return len(self.sing_vals)

    @property
    def evals(self) -> Array:
        """Return eigenvalues of EEOFEigen object."""
        return self.sing_vals**2

    def isel(
        self,
        s: SliceItem,
    ) -> "EEOFEigen":
        """Slice a EEOFEigen object."""
        return EEOFEigen(
            sing_vals=self.sing_vals[s],
            eeofs=self.eeofs[s],
            pcs=self.pcs[s],
        )

    def tabulate(
        self,
        num_tabulate: int | None = None,
        headers: Sequence[str] = ["Singular values"],
        show: bool = True,
    ) -> str:
        """Tabulate the eigenvalues in a KernelEigen object."""
        assert is_array_like(self.sing_vals)
        data = np.vstack(((self.sing_vals),))[:, :num_tabulate].T
        table = tabulate(data, headers=headers, floatfmt=".4f", showindex=True)
        if show:
            print(table)
        return table


# TODO: This would be an excellent candidate for an ImplementsEigen protocol
# method.
def slice_eigen(
    eigen: EEOFEigen,
    which_eigs: int | tuple[int, int] | list[int] | None = None,
) -> EEOFEigen:
    """Slice EEOFEigen oject using `which_eigs` convention."""
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


class EEOFEigenShardings(NamedTuple):
    """NamedTuple holding shardings for computation EEOFEigen objects."""

    matrix: NamedSharding | None = None
    """Sharding of Hankel matrix."""

    sing_vals: NamedSharding | None = None
    """Sharding of singluar value array."""

    eeofs: NamedSharding | None = None
    """Sharding of extended EOF array."""

    pcs: NamedSharding | None = None
    """Sharding of principal components array."""

    @classmethod
    def from_named_sharder[
        Shape: tuple[int, int, *tuple[int, ...]],
        AxisNames: str,
    ](cls, sharder: NamedSharder[Shape, AxisNames] | None) -> Self:
        """Create EEOFEigenSharding object from NamedSharder."""
        if sharder is not None:
            x_sharding = sharder.sharding(None, sharder.axis_names[0])
            y_sharding = sharder.sharding(None, sharder.axis_names[1])
            replicating = sharder.sharding(None)
            return cls(sing_vals=replicating, pcs=y_sharding, eeofs=x_sharding)
        else:
            return cls()

    @property
    def shard_eeof_eigen(
        self,
    ) -> Callable[[EEOFEigen], EEOFEigen]:
        """Shard EEOFEigen objects."""

        def shard(
            eeof_eigen: EEOFEigen,
        ) -> EEOFEigen:
            return EEOFEigen(
                sing_vals=jax.device_put(
                    eeof_eigen.sing_vals, device=self.sing_vals
                ),
                eeofs=jax.device_put(eeof_eigen.eeofs, device=self.eeofs),
                pcs=jax.device_put(eeof_eigen.pcs, device=self.pcs),
            )

        return shard


@final
@dataclass(frozen=True, slots=True)
class EEOFEigenbasis(alg.ImplementsL2FnEigenbasis[Xd, R, V, R, Rs, Idx]):
    """Dataclass implementing frame operators for kernel eigenbasis."""

    dim: int
    """Number of eigenfunctions."""

    anal: Callable[[V], Rs]
    """Analysis operator."""

    dual_anal: Callable[[V], Rs]
    """Dual analysis operator."""

    synth: Callable[[Rs], V]
    """Synthesis operator."""

    dual_synth: Callable[[Rs], V]
    """Dual synthesis operator."""

    fn_anal: Callable[[F[Xd, R]], Rs]
    """Function analysis operator."""

    dual_fn_anal: Callable[[F[Xd, R]], Rs]
    """Dual function analysis operator."""

    fn_synth: Callable[[Rs], F[Xd, R]]
    """Function synthesis operator."""

    dual_fn_synth: Callable[[Rs], F[Xd, R]]
    """Dual function synthesis operator."""

    vec: Callable[[Idx], V]
    """Basis vectors."""

    dual_vec: Callable[[Idx], V]
    """Dual basis vectors."""

    fn: Callable[[Idx], F[Xd, R]]
    """Function representatives of basis vectors."""

    dual_fn: Callable[[Idx], F[Xd, R]]
    """Function representatives of dual basis vectors."""

    spec: Rs
    """Rernel operator spectrum (set of eigenvalues)."""

    evl: Callable[[Idx], R]
    """Rernel eigenvalues."""


def make_svd_eeof_eigensolver(
    pars: EEOFPars,
    jit: bool = True,
    shardings: EEOFEigenShardings = EEOFEigenShardings(),
) -> Callable[[Array], EEOFEigen]:
    """Make SVD solver for extended EOF analysis using svd."""
    # WARNING: In recent versions of the code we are sharding the singular
    # vectors in KernelEigen along rows. Assigning svd_shardings based on
    # sharding likely leads to sharding inconsistencies. A possible solution
    # would be to rename shardings to out_shardings and create a separate
    # svd_shardings input argument to pass the correct shardings to
    # make_svd_with_sharding_constraints.
    svd_shardings = SvdShardings(
        left_sing_vectors=shardings.eeofs,
        sing_values=shardings.sing_vals,
        right_sing_vectors=shardings.pcs,
    )
    svd = make_svd_with_sharding_constraints(shardings=svd_shardings)

    def svdsolve(data: Array) -> EEOFEigen:
        if pars.num_delays > 0:
            a = dl.hankel(
                data,
                num_delays=pars.num_delays,
                delay_step=pars.delay_step,
                flatten=True,
            ).T
        else:
            a = data.T
        left_sing_vecs, sing_vals, right_sing_vecs = svd(a)
        if pars.num_eigs is None:
            _num_eigs = len(sing_vals)
        else:
            _num_eigs = pars.num_eigs
        return shardings.shard_eeof_eigen(
            EEOFEigen(
                eeofs=left_sing_vecs[:, :_num_eigs].T,
                sing_vals=sing_vals[:_num_eigs],
                pcs=right_sing_vecs[:_num_eigs],
            )
        )

    if jit:
        return jax.jit(svdsolve)

    return svdsolve


def make_eigenbasis(
    l2x: alg.ImplementsL2FnAlgebra[Xd, R, V, R],
    eeof_eigen: EEOFEigen,
) -> EEOFEigenbasis:
    """Make eigenbasis for EEOF covariance kernel."""

    def vc(i: int | Array) -> V:
        return eeof_eigen.pcs[i]

    def evl(i: int | Array) -> R:
        return eeof_eigen.sing_vals[i] ** 2

    def fn(i: int | Array) -> F[Xd, R]:
        def f(xd: Xd) -> R:
            return jnp.dot(eeof_eigen.eeofs[i], xd) / eeof_eigen.sing_vals[i]

        return f

    @partial(vmap, in_axes=(0, None))
    def anal_eval(i: int | Array, v: V) -> R:
        return l2x.innerp(eeof_eigen.pcs[i], v)

    @partial(vmap, in_axes=(0, None))
    def fn_eval(i: int | Array, x: Xd) -> R:
        return fn(i)(x)

    idxs = jnp.arange(eeof_eigen.num_eigs)
    anal = partial(anal_eval, idxs)
    fn_anal = fun.compose(anal, l2x.incl)
    synth = vec.make_synthesis_operator(eeof_eigen.pcs, idxs)
    fn_synth = vec.make_fn_synthesis_operator(partial(fn_eval, idxs))
    spec = eeof_eigen.sing_vals[idxs] ** 2
    basis = EEOFEigenbasis(
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
        spec=spec,
    )
    return basis


def make_data_driven_eigenbasis[Data: PyTree](
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[Xd, R, V, R]],
    which_eigs: int | tuple[int, int] | list[int] | None = None,
) -> Callable[[Data, EEOFEigen], EEOFEigenbasis]:
    """Make data-driven EEOF eigenbasis builder."""

    def _make_eigenbasis(
        data: Data,
        eeof_eigen: EEOFEigen,
    ) -> EEOFEigenbasis:
        l2x = impl_l2(data)
        _eeof_eigen = slice_eigen(eeof_eigen, which_eigs)
        return make_eigenbasis(l2x, _eeof_eigen)

    return _make_eigenbasis


def compute_eigen(
    pars: EEOFPars,
    data: Array,
    jit: bool = True,
    shardings: EEOFEigenShardings = EEOFEigenShardings(),
) -> EEOFEigen:
    """Solve eigenvalue problem for EEOF kernels."""
    svdsolve = make_svd_eeof_eigensolver(pars, jit=jit, shardings=shardings)
    return svdsolve(data)


def plot_eeof_spectrum(
    eeof_eigen: EEOFEigen,
    num_eigs_plt: int | None = None,
    i_fig: int = 1,
) -> Figure:
    """Plot spectrum of EEOF eigenvalues."""
    if num_eigs_plt is None:
        num_eigs_plt = len(eeof_eigen.evals)
    eeof_evals = eeof_eigen.evals[:num_eigs_plt]
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    ax.plot(
        jnp.arange(num_eigs_plt),
        jnp.log10(eeof_evals),
        ".",
    )
    ax.grid()
    ax.set_xlabel("$j$")
    ax.set_ylabel(r"$\log_{10}\lambda_j$")
    ax.set_title("EEOF eigenvalues (variances)")
    return fig

"""Provide functions for Koopmman operator computations in JAX."""
# pyright: basic
# TODO: Consider making the functions implemented in this module generic over
# the L2FnAlgebra implementation.

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import jax.scipy as jsp
import matplotlib.pyplot as plt
import nlsa.abstract_algebra as alg
import nlsa.function_algebra as fun
import nlsa.jax.delays as dl
import nlsa.jax.dynamics as dyn
import nlsa.jax.vector_algebra as vec
import numpy.typing as npt
from collections.abc import Callable
from functools import partial
from jax import Array, vmap
from jax.sharding import NamedSharding
from jax.typing import DTypeLike
from matplotlib.figure import Figure
from nlsa.jax.sharding import NamedSharder
from nlsa.jax.vector_algebra import (
    L2FnAlgebra,
    L2FnAlgebraShardings,
    L2VectorAlgebra,
)
from nlsa.kernels import KernelEigenbasis
from nlsa.koopman import (
    KoopmanEigenbasis,
    KoopmanParsDiff,
    KoopmanParsQz,
)
from nlsa.jax.utils import batch_map, batch_map_2d
from typing import Literal, NamedTuple, Optional, Self, TypedDict

type Css = Array  # Collection of basis expansion coefficient vectors
type K = Array  # Scalar
type Ks = Array  # Collection of scalars
type V = Array  # L2 observable vector
type Vs = Array  # Collection of L2 vectors
type Idxs = Array  # basis vector indices
type Mat = Array  # Matrix acting as linear operator on L2 vectors
type F[*Ss, T] = Callable[[*Ss], T]  # Shorthand for Callables


class GeneratorShardings(NamedTuple):
    """NamedTuple holding array shardings for generator matrix computation."""

    tangents: L2FnAlgebraShardings = L2FnAlgebraShardings()
    """Shardings for the L2 space used in tangent vector evaluation."""

    matrix: Optional[NamedSharding] = None
    """Sharding of generator matrix."""


def _make_vgrad_basis[X: Array, TX: Array](
    eval_tangents: Callable[[F[X, TX, K]], V],
    basis: alg.ImplementsDimensionedL2FnFrame[X, K, V, Ks, int | Idxs],
    batch_size: Optional[int] = None,
) -> Callable[[Idxs], Vs]:
    """Make function that computes directional derivatives of basis vectors."""

    @partial(batch_map, out_axis=1, batch_size=batch_size)
    def vgrad_basis(idx: int | Array) -> V:
        return eval_tangents(dyn.vgrad(basis.fn(idx)))

    return vgrad_basis


def compute_generator_matrix[
    Shape: tuple[int, ...],
    D: DTypeLike,
    X: Array,
    TX: Array,
](
    l2x: L2FnAlgebra[Shape, D, X, K],
    eval_tangents: Callable[[F[X, TX, K]], V],
    basis: alg.ImplementsDimensionedL2FnFrame[X, K, V, Ks, int | Idxs],
    basis_idxs: Optional[Idxs] = None,
    grad_batch_size: Optional[int] = None,
    gram_batch_size: Optional[int] = None,
    shardings: GeneratorShardings = GeneratorShardings(),
) -> Mat:
    """Compute matrix representation of generator in a basis."""
    if basis_idxs is not None:
        _basis_idxs = basis_idxs
    else:
        _basis_idxs = jnp.arange(basis.dim)
    sharding = l2x.sharding
    if sharding is not None:
        assert isinstance(sharding, NamedSharding)
    vgrad_basis = _make_vgrad_basis(
        eval_tangents, basis, batch_size=grad_batch_size
    )
    innerp = batch_map_2d(
        l2x.innerp, in_axes=(1, 1), batch_sizes=gram_batch_size
    )

    @partial(jax.jit, out_shardings=shardings.matrix)
    def run() -> Mat:
        phi_duals = basis.dual_vec(_basis_idxs)
        vgrad_phis = vgrad_basis(_basis_idxs)
        gen_mat = innerp(phi_duals, vgrad_phis)
        return gen_mat

    return run()


class QzShardings(NamedTuple):
    """NamedTuple holding array shardings for Qz matrix computation."""

    quadrature: L2FnAlgebraShardings = L2FnAlgebraShardings()
    """Shardings for the L2 space used in resolvent quadrature."""

    res_weights: Optional[NamedSharding] = None
    """Sharding of resolvent weight vector."""

    matrix: Optional[NamedSharding] = None
    """Sharding of Qz matrix."""


class _LaplaceTransformBasis(NamedTuple):
    """NamedTuple holding Laplace transform of basis and dual basis vectors."""

    vec: Callable[[Idxs], Vs]
    """Laplace transforms of basis vectors."""

    dual_vec: Callable[[Idxs], Vs]
    """Laplace transforms of dual basis vectors."""


def _make_laplace_transform_basis[X](
    z: float,
    dt: float,
    eval_quad: Callable[[F[X, K]], V],
    num_quad: int,
    basis: alg.ImplementsDimensionedL2FnFrame[X, K, V, Ks, int | Idxs],
    batch_size: Optional[int] = None,
    shardings: QzShardings = QzShardings(),
) -> _LaplaceTransformBasis:
    """Make function that computes Laplace transforms of basis vectors."""
    lapl = dl.make_laplace_transform(
        z=z,
        dt=dt,
        num_quad=num_quad,
        weight_sharding=shardings.res_weights,
        out_sharding=shardings.quadrature.vectors,
    )

    @partial(batch_map, out_axis=1, batch_size=batch_size)
    def transf_vec(idx: int | Array) -> V:
        return lapl(eval_quad(basis.fn(idx)))

    @partial(batch_map, out_axis=1, batch_size=batch_size)
    def transf_dual_vec(idx: int | Array) -> V:
        return lapl(eval_quad(basis.dual_fn(idx)))

    return _LaplaceTransformBasis(vec=transf_vec, dual_vec=transf_dual_vec)


def compute_qz_matrix[
    Shape: tuple[int, ...],
    D: DTypeLike,
    X: Array,
](
    res_z: float,
    dt: float,
    num_quad: int,
    l2_space: L2FnAlgebra[Shape, D, X, K],
    eval_quad: Callable[[F[X, K]], V],
    basis: alg.ImplementsDimensionedL2FnFrame[X, K, V, Ks, int | Idxs],
    basis_idxs: Optional[Idxs] = None,
    quad_batch_size: Optional[int] = None,
    gram_batch_size: Optional[int] = None,
    shardings: QzShardings = QzShardings(),
) -> Mat:
    """Compute matrix representation of Qz operator in a basis."""
    if basis_idxs is not None:
        _basis_idxs = basis_idxs
    else:
        _basis_idxs = jnp.arange(basis.dim)
    laplace_transform_basis = _make_laplace_transform_basis(
        z=res_z,
        dt=dt,
        num_quad=num_quad,
        eval_quad=eval_quad,
        basis=basis,
        batch_size=quad_batch_size,
        shardings=shardings,
    )

    innerp = batch_map_2d(
        l2_space.innerp, in_axes=(1, 1), batch_sizes=gram_batch_size
    )

    # TODO: This could be made more efficient if we know in advance that the
    # basis is orhtonormal so that phis == phi_duals.
    @partial(jax.jit, out_shardings=shardings.matrix)
    def run() -> Mat:
        phis = basis.vec(_basis_idxs)
        phi_duals = basis.dual_vec(_basis_idxs)
        lapl_phis = laplace_transform_basis.vec(_basis_idxs)
        lapl_phi_duals = laplace_transform_basis.dual_vec(_basis_idxs)
        rz = innerp(phi_duals, lapl_phis)
        rz_adj = innerp(lapl_phi_duals, phis)
        qz_mat = (rz - rz_adj) / 2
        return qz_mat

    return run()


# TODO: Make this a dataclass or named tuple.
# Introduce fields dim and dim_galerkin with associated type parameters.
# The dimensions of various arrays such as evals,
# gen_evals, etc., could be checked in __post_init__. The same approach could
# be used for KernelEigen.
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


class KoopmanEigenShardings(NamedTuple):
    """NamedTuple holding shardings of jax.linalg.eig output."""

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
            replicating = sharder.sharding(None)
            return cls(eigenvalues=replicating, eigenvectors=replicating)
        else:
            return cls()


def slice_koopman_eigen(
    eigen: KoopmanEigen, which_eigs: int | tuple[int, int] | list[int]
) -> KoopmanEigen:
    """Slice a KoopmanEigen object to produce a "sub eigenbasis"."""
    match which_eigs:
        case int():
            idxs = slice(which_eigs)
        case (_, _):
            idxs = slice(which_eigs[0], which_eigs[1])
        case list():
            idxs = which_eigs
    sliced_eigen: KoopmanEigen = {
        "evals": eigen["evals"][idxs],
        "gen_evals": eigen["gen_evals"][idxs],
        "efreqs": eigen["efreqs"][idxs],
        "eperiods": eigen["eperiods"][idxs],
        "evec_coeffs": eigen["evec_coeffs"][:, idxs],
        "dual_evec_coeffs": eigen["dual_evec_coeffs"][:, idxs],
        "engys": eigen["engys"][idxs],
    }
    return sliced_eigen


def to_koopman_eigen(
    dict_in: dict[str, npt.ArrayLike],
    dtype: Optional[DTypeLike] = None,
    shardings: Optional[KoopmanEigenShardings] = None,
) -> KoopmanEigen:
    """Convert dict of numpy ArrayLike objects to KoopmanEigen TypedDict."""
    try:
        if shardings is not None:
            koopman_eigen: KoopmanEigen = {
                "evals": jnp.array(
                    dict_in["evals"], dtype, device=shardings.eigenvalues
                ),
                "gen_evals": jnp.array(
                    dict_in["gen_evals"], dtype, device=shardings.eigenvalues
                ),
                "efreqs": jnp.array(
                    dict_in["efreqs"], dtype, device=shardings.eigenvalues
                ),
                "eperiods": jnp.array(
                    dict_in["eperiods"], dtype, device=shardings.eigenvalues
                ),
                "evec_coeffs": jnp.array(
                    dict_in["evec_coeffs"],
                    dtype,
                    device=shardings.eigenvectors,
                ),
                "dual_evec_coeffs": jnp.array(
                    dict_in["dual_evec_coeffs"],
                    dtype,
                    device=shardings.eigenvectors,
                ),
                "engys": jnp.array(
                    dict_in["engys"], dtype, device=shardings.eigenvalues
                ),
            }
        else:
            koopman_eigen: KoopmanEigen = {
                "evals": jnp.array(dict_in["evals"], dtype),
                "gen_evals": jnp.array(dict_in["gen_evals"], dtype),
                "efreqs": jnp.array(dict_in["efreqs"], dtype),
                "eperiods": jnp.array(dict_in["eperiods"], dtype),
                "evec_coeffs": jnp.array(dict_in["evec_coeffs"], dtype),
                "dual_evec_coeffs": jnp.array(
                    dict_in["dual_evec_coeffs"], dtype
                ),
                "engys": jnp.array(dict_in["engys"], dtype),
            }
        return koopman_eigen
    except ValueError as exc:
        raise ValueError("Incompatible keys/values") from exc


class _GeneratorSpectrum(NamedTuple):
    """NamedTuple holding spectral data of the Koopman generator."""

    evals: Array
    """Eigenvalues of generator or auxiliary operator."""

    gen_evals: Array
    """Generator eigenvalues."""

    evec_coeffs: Array
    """Basis expansion coefficients of the eigenvectors."""

    dual_evec_coeffs: Array
    """Basis expansion coefficients of the dual (left) eigenvectors."""


def _make_generator_diff_eigensolver[X: Array](
    gen_mat: Array,
    kernel_basis: KernelEigenbasis[X, K, V, Ks, int | Array],
    diffusion_strength: float,
    basis_idxs: Array,
    antisym: bool,
) -> Callable[[], _GeneratorSpectrum]:
    """Make eigensolver for diffusion-regularized Koopman generator."""

    def eigensolve() -> _GeneratorSpectrum:
        diff_mat = diffusion_strength * jnp.diag(
            kernel_basis.lapl_spec[basis_idxs]
        )
        if antisym:
            reg_gen_mat = (gen_mat - gen_mat.T) / 2 - diff_mat
        else:
            reg_gen_mat = gen_mat - diff_mat
        gen_evals, evec_coeffs = jla.eig(reg_gen_mat)
        gram = evec_coeffs @ evec_coeffs.conj().T
        dual_evec_coeffs = jsp.linalg.solve(gram, evec_coeffs, assume_a="her")
        return _GeneratorSpectrum(
            evals=gen_evals,
            gen_evals=gen_evals,
            evec_coeffs=evec_coeffs,
            dual_evec_coeffs=dual_evec_coeffs,
        )

    return eigensolve


def _make_qz_eigensolver[X: Array](
    qz_mat: Array,
    kernel_basis: KernelEigenbasis[X, K, V, Ks, int | Array],
    resolvent_z: float,
    smoothing_kernel: Literal["exponential", "fejer"],
    regularization_strength: float,
    basis_idxs: Array,
) -> Callable[[], _GeneratorSpectrum]:
    """Make eigensolver for compactified Qz operator."""

    def eigensolve() -> _GeneratorSpectrum:
        assert jnp.all(basis_idxs)  # All indices should be nonzero
        match smoothing_kernel:
            case "exponential":
                lambs = (
                    kernel_basis.spec[basis_idxs] ** regularization_strength
                )
            case "fejer":
                idx_max = jnp.max(basis_idxs)
                lambs = (
                    1
                    - jnp.sqrt(
                        kernel_basis.lapl_spec[basis_idxs]
                        / kernel_basis.lapl_spec[idx_max + 1]
                    )
                ) ** regularization_strength
        reg_qz_mat = lambs * qz_mat * lambs[:, jnp.newaxis]
        qz_evals, evec_coeffs = jla.eig(reg_qz_mat)
        gen_evals = to_gen_evals(qz_evals, resolvent_z)
        return _GeneratorSpectrum(
            evals=qz_evals,
            gen_evals=gen_evals,
            evec_coeffs=evec_coeffs,
            dual_evec_coeffs=evec_coeffs,
        )

    return eigensolve


def _make_from_generator_spectrum(
    kernel_basis: KernelEigenbasis,
    basis_idxs: Array,
    num_eigs: int,
    sort_by: Literal["frequency", "energy"],
    out_shardings: KoopmanEigenShardings = KoopmanEigenShardings(),
) -> Callable[[_GeneratorSpectrum], KoopmanEigen]:
    """Create conversion function from GeneratorSpectrum to KoopmanEigen."""

    def from_generator_spectrum(spec: _GeneratorSpectrum) -> KoopmanEigen:
        engys = (
            jnp.sum(
                jnp.abs(spec.evec_coeffs) ** 2
                / kernel_basis.spec[basis_idxs, jnp.newaxis],
                axis=0,
            )
            - 1
        )
        match sort_by:
            case "frequency":
                isort = jnp.argsort(jnp.abs(spec.gen_evals.real))[
                    : num_eigs - 1
                ]
            case "energy":
                isort = jnp.argsort(engys)[: num_eigs - 1]
        _evals = jnp.concatenate((jnp.atleast_1d(0), spec.gen_evals[isort]))
        _gen_evals = jnp.concatenate(
            (jnp.atleast_1d(0), spec.gen_evals[isort])
        )
        _efreqs = jnp.concatenate(
            (jnp.atleast_1d(0), spec.gen_evals[isort].imag)
        )
        _eperiods = jnp.concatenate(
            (
                jnp.atleast_1d(jnp.inf),
                2 * jnp.pi / spec.gen_evals[isort].imag,
            )
        )
        _engys = jnp.concatenate((jnp.atleast_1d(0), engys[isort]))
        _evec_coeffs = jsp.linalg.block_diag(1, spec.evec_coeffs[:, isort])
        _dual_evec_coeffs = jsp.linalg.block_diag(
            1, spec.dual_evec_coeffs[:, isort]
        )

        if out_shardings.eigenvalues is not None:
            _evals = jax.lax.with_sharding_constraint(
                _evals, shardings=out_shardings.eigenvalues
            )
            _gen_evals = jax.lax.with_sharding_constraint(
                _gen_evals, shardings=out_shardings.eigenvalues
            )
            _efreqs = jax.lax.with_sharding_constraint(
                _efreqs, shardings=out_shardings.eigenvalues
            )
            _eperiods = jax.lax.with_sharding_constraint(
                _eperiods, shardings=out_shardings.eigenvalues
            )
        if out_shardings.eigenvectors is not None:
            _engys = jax.lax.with_sharding_constraint(
                _engys, shardings=out_shardings.eigenvectors
            )

            _evec_coeffs = jax.lax.with_sharding_constraint(
                _evec_coeffs, shardings=out_shardings.eigenvectors
            )

            _dual_evec_coeffs = jax.lax.with_sharding_constraint(
                _dual_evec_coeffs, shardings=out_shardings.eigenvectors
            )

        eigen: KoopmanEigen = {
            "evals": _evals,
            "gen_evals": _gen_evals,
            "efreqs": _efreqs,
            "eperiods": _eperiods,
            "engys": _engys,
            "evec_coeffs": _evec_coeffs,
            "dual_evec_coeffs": _dual_evec_coeffs,
        }
        return eigen

    return from_generator_spectrum


def compute_eigen_diff[X: Array](
    pars: KoopmanParsDiff,
    gen_mat: Array,
    kernel_basis: KernelEigenbasis[X, K, V, Ks, int | Array],
    out_shardings: KoopmanEigenShardings = KoopmanEigenShardings(),
) -> KoopmanEigen:
    """Solve eigenvalue problem for diffusion-regularized generator."""
    match pars.which_eigs_galerkin:
        case int():
            idxs = jnp.arange(1, pars.which_eigs_galerkin + 1)
        case tuple() if len(pars.which_eigs_galerkin) == 2:
            idxs = jnp.arange(
                pars.which_eigs_galerkin[0], pars.which_eigs_galerkin[1]
            )
        case list():
            idxs = jnp.array(pars.which_eigs_galerkin)
    assert jnp.all(idxs)  # All indices should be nonzero
    if pars.num_eigs is not None:
        num_eigs = pars.num_eigs
    else:
        num_eigs = len(idxs)
    eigensolve = _make_generator_diff_eigensolver(
        gen_mat,
        kernel_basis,
        basis_idxs=idxs,
        antisym=pars.antisym,
        diffusion_strength=pars.tau,
    )
    from_generator_spectrum = _make_from_generator_spectrum(
        kernel_basis,
        basis_idxs=idxs,
        num_eigs=num_eigs,
        sort_by=pars.sort_by,
        out_shardings=out_shardings,
    )
    run = fun.compose(from_generator_spectrum, eigensolve)
    return run()


def to_gen_evals(qz_evals: Ks, /, res_z: float | Array) -> Ks:
    """Compute generator eigenvalues from Qz operator eigenvalues."""
    gen_evals = (
        1j
        * (1 + jnp.sqrt(1 - 4 * res_z**2 * qz_evals.imag**2))
        / (2 * qz_evals.imag)
    )
    return gen_evals


def compute_eigen_qz[X: Array](
    pars: KoopmanParsQz,
    qz_mat: Array,
    kernel_basis: KernelEigenbasis[X, K, V, Ks, int | Array],
    out_shardings: KoopmanEigenShardings = KoopmanEigenShardings(),
) -> KoopmanEigen:
    """Solve eigenvalue problem for regularized Qz operator."""
    match pars.which_eigs_galerkin:
        case int():
            idxs = jnp.arange(1, pars.which_eigs_galerkin + 1)
        case (_, _):
            idxs = jnp.arange(
                pars.which_eigs_galerkin[0], pars.which_eigs_galerkin[1]
            )
        case list():
            idxs = jnp.array(pars.which_eigs_galerkin)
    assert jnp.all(idxs)  # All indices should be nonzero
    if pars.num_eigs is not None:
        num_eigs = pars.num_eigs
    else:
        num_eigs = len(idxs)

    eigensolve = _make_qz_eigensolver(
        qz_mat,
        kernel_basis,
        basis_idxs=idxs,
        resolvent_z=pars.res_z,
        regularization_strength=pars.tau,
        smoothing_kernel=pars.smoothing_kernel,
    )
    from_generator_spectrum = _make_from_generator_spectrum(
        kernel_basis,
        basis_idxs=idxs,
        num_eigs=num_eigs,
        sort_by=pars.sort_by,
        out_shardings=out_shardings,
    )
    run = fun.compose(from_generator_spectrum, eigensolve)
    return run()


def make_eigenbasis[X: Array, L: int, D: DTypeLike](
    c_l: L2VectorAlgebra[tuple[L], D],
    kernel_basis: KernelEigenbasis[X, K, V, Ks, int | Array],
    koopman_eigen: KoopmanEigen,
    which_eigs: int | tuple[int, int] | list[int],
) -> KoopmanEigenbasis[X, K, V, Ks, int | Array]:
    """Make Koopman eigenbasis from eigendecomposition of asymmetric op."""
    match which_eigs:
        case int():
            idxs = jnp.arange(which_eigs)
        case (_, _):
            idxs = jnp.arange(which_eigs[0], which_eigs[1])
        case list():
            idxs = jnp.array(which_eigs)

    def vc(i: int | Array) -> V:
        return kernel_basis.synth(koopman_eigen["evec_coeffs"][:, idxs[i]])

    def dual_vc(i: int | Array) -> V:
        return kernel_basis.dual_synth(
            koopman_eigen["dual_evec_coeffs"][:, idxs[i]]
        )

    def evl(i: int | Array) -> K:
        return koopman_eigen["evals"][idxs[i]]

    def gen_evl(i: int | Array) -> K:
        return koopman_eigen["gen_evals"][idxs[i]]

    def efreq(i: int | Array) -> K:
        return koopman_eigen["efreqs"][idxs[i]]

    def eperiod(i: int | Array) -> K:
        return koopman_eigen["eperiods"][idxs[i]]

    def engy(i: int | Array) -> K:
        return koopman_eigen["engys"][idxs[i]]

    def fn(i: int | Array) -> Callable[[X], K]:
        return kernel_basis.fn_synth(koopman_eigen["evec_coeffs"][:, idxs[i]])

    def dual_fn(i: int | Array) -> Callable[[X], K]:
        return kernel_basis.dual_fn_synth(
            koopman_eigen["dual_evec_coeffs"][:, idxs[i]]
        )

    @partial(vmap, in_axes=(0, None))
    def anal_eval_c(i: int | Array, v: V) -> K:
        return c_l.innerp(koopman_eigen["dual_evec_coeffs"][:, idxs[i]], v)

    @partial(vmap, in_axes=(0, None))
    def dual_anal_eval_c(i: int | Array, v: V) -> K:
        return c_l.innerp(koopman_eigen["evec_coeffs"][:, idxs[i]], v)

    anal_c = partial(anal_eval_c, idxs)
    anal = fun.compose(anal_c, kernel_basis.anal)
    dual_anal_c = partial(dual_anal_eval_c, idxs)
    dual_anal = fun.compose(dual_anal_c, kernel_basis.dual_anal)
    fn_anal = fun.compose(anal_c, kernel_basis.fn_anal)
    dual_fn_anal = fun.compose(dual_anal_c, kernel_basis.dual_fn_anal)
    synth_c = vec.make_synthesis_operator(koopman_eigen["evec_coeffs"], idxs)
    synth = fun.compose(kernel_basis.synth, synth_c)
    dual_synth_c = vec.make_synthesis_operator(
        koopman_eigen["dual_evec_coeffs"], idxs
    )
    dual_synth = fun.compose(kernel_basis.dual_synth, dual_synth_c)
    fn_synth = fun.compose(kernel_basis.fn_synth, synth_c)
    dual_fn_synth = fun.compose(kernel_basis.dual_fn_synth, synth_c)
    spec = koopman_eigen["evals"][idxs]
    gen_spec = koopman_eigen["gen_evals"][idxs]
    efreqs = koopman_eigen["efreqs"][idxs]
    eperiods = koopman_eigen["eperiods"][idxs]
    engys = koopman_eigen["engys"][idxs]
    basis = KoopmanEigenbasis(
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
        gen_evl=gen_evl,
        efreq=efreq,
        eperiod=eperiod,
        engy=engy,
        spec=spec,
        gen_spec=gen_spec,
        efreqs=efreqs,
        eperiods=eperiods,
        engys=engys,
    )
    return basis


def make_eigenbasis_antisym[X: Array, L: int, D: DTypeLike](
    c_l: L2VectorAlgebra[tuple[L], D],
    kernel_basis: KernelEigenbasis[X, K, V, Ks, int | Array],
    koopman_eigen: KoopmanEigen,
    which_eigs: int | tuple[int, int] | list[int],
) -> KoopmanEigenbasis[X, K, V, Ks, int | Array]:
    """Make Koopman eigenbasis from eigendecomposition of antisymmetric op."""
    match which_eigs:
        case int():
            idxs = jnp.arange(which_eigs)
        case (_, _):
            idxs = jnp.arange(which_eigs[0], which_eigs[1])
        case list():
            idxs = jnp.array(which_eigs)

    def vc(i: int | Array) -> V:
        return kernel_basis.synth(koopman_eigen["evec_coeffs"][:, idxs[i]])

    def evl(i: int | Array) -> K:
        return koopman_eigen["evals"][idxs[i]]

    def gen_evl(i: int | Array) -> K:
        return koopman_eigen["gen_evals"][idxs[i]]

    def efreq(i: int | Array) -> K:
        return koopman_eigen["efreqs"][idxs[i]]

    def eperiod(i: int | Array) -> K:
        return koopman_eigen["eperiods"][idxs[i]]

    def engy(i: int | Array) -> K:
        return koopman_eigen["engys"][idxs[i]]

    def fn(i: int | Array) -> Callable[[X], K]:
        return kernel_basis.fn_synth(koopman_eigen["evec_coeffs"][:, idxs[i]])

    @partial(vmap, in_axes=(0, None))
    def anal_eval_c(i: int | Array, v: V) -> K:
        return c_l.innerp(koopman_eigen["evec_coeffs"][:, idxs[i]], v)

    anal_c = partial(anal_eval_c, idxs)
    anal = fun.compose(anal_c, kernel_basis.anal)
    fn_anal = fun.compose(anal_c, kernel_basis.fn_anal)
    synth_c = vec.make_synthesis_operator(koopman_eigen["evec_coeffs"], idxs)
    synth = fun.compose(kernel_basis.synth, synth_c)
    fn_synth = fun.compose(kernel_basis.fn_synth, synth_c)
    spec = koopman_eigen["evals"][idxs]
    gen_spec = koopman_eigen["gen_evals"][idxs]
    efreqs = koopman_eigen["efreqs"][idxs]
    eperiods = koopman_eigen["eperiods"][idxs]
    engys = koopman_eigen["engys"][idxs]
    basis = KoopmanEigenbasis(
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
        gen_evl=gen_evl,
        efreq=efreq,
        eperiod=eperiod,
        engy=engy,
        spec=spec,
        gen_spec=gen_spec,
        efreqs=efreqs,
        eperiods=eperiods,
        engys=engys,
    )
    return basis


def plot_generator_spectrum(
    koopman_eigen: KoopmanEigen,
    num_eigs_plt: Optional[int] = None,
    frequency_symbol: str = "$\\omega_j$",
    frequency_scaling: float = 1,
    frequency_units: Optional[str] = None,
    i_fig: int = 1,
) -> Figure:
    """Plot spectrum of Koopman generator."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    if num_eigs_plt is None:
        num_eigs_plt = len(koopman_eigen["gen_evals"])
    im = ax.scatter(
        koopman_eigen["engys"][:num_eigs_plt],
        koopman_eigen["gen_evals"][:num_eigs_plt].imag * frequency_scaling,
        s=10,
        c=jnp.arange(num_eigs_plt),
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

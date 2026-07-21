"""Provide classes and functions for Koopmman operator computations in JAX."""

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import jax.scipy as jsp
import nlsa.abstract_algebra as alg
import nlsa.function_algebra as fun
import nlsa.jax.delays as dl
import nlsa.jax.dynamics as dyn
import nlsa.jax.kernels as knl
import nlsa.jax.vector_algebra as vec
import nlsa.koopman as koop
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from jax import Array, vmap
from jax.sharding import NamedSharding
from jax.typing import DTypeLike
from nlsa.jax.kernels import KernelEigen, KernelPars
from nlsa.jax.sharding import NamedSharder, shardit
from nlsa.jax.typing import PyTree, typestable_jit
from nlsa.jax.vector_algebra import (
    L2FnAlgebraShardings,
    L2VectorAlgebra,
)
from nlsa.jax.special import dawsn
from nlsa.jax.utils import batch_map, batch_map_2d
from nlsa.koopman import (
    KoopmanPars,
    KoopmanParsDiff,
    KoopmanParsTransf,
)
from nlsa.typing import SliceItem
from typing import Literal, NamedTuple, Self, final
from collections.abc import Sequence

type Css = Array  # Collection of basis expansion coefficient vectors
type C = Array  # Complex scalar
type Cs = Array  # Collection of complex scalars
type R = Array  # Real scalar
type Rs = Array  # Collection of real scalars
type K = Array  # Scalar
type Ks = Array  # Collection of scalars
type V = Array  # L2 observable vector
type Vtst = Array  # Vector in L2 with respect to the test data
type Vs = Array  # Collection of L2 vectors
type X = Array  # Covariate
type TX = Array  # Tangent covariate
type Idx = int | Array  # basis vector index
type Idxs = Array  # basis vector indices
type Mat = Array  # Matrix acting as linear operator on L2 vectors
type Shape = tuple[int, ...]
type F[*Ss, T] = Callable[[*Ss], T]  # Shorthand for Callables


class KoopmanEigen(NamedTuple):
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
        return len(self.evals)

    def isel(self, s: SliceItem) -> Self:
        """Slice a KoopmanEigen object."""
        return type(self)(
            evals=self.evals[s],
            gen_evals=self.gen_evals[s],
            efreqs=self.efreqs[s],
            engys=self.engys[s],
            eperiods=self.eperiods[s],
            evec_coeffs=self.evec_coeffs[s],
            dual_evec_coeffs=self.dual_evec_coeffs[s],
        )

    def tabulate(
        self,
        num_tabulate: int | None = None,
        headers: Sequence[str] | None = None,
        show: bool = True,
    ) -> str:
        """Tabulate the eigenvalues in a KoopmanEigen object."""
        return koop.tabulate_eigen(self, num_tabulate, headers, show)


@final
@dataclass(frozen=True, slots=True)
class KoopmanEigenbasis(koop.ImplementsKoopmanEigenbasis[X, K, K, V, Ks, Idx]):
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

    vec: Callable[[Idx], V]
    """Basis vectors."""

    dual_vec: Callable[[Idx], V]
    """Dual basis vectors."""

    fn: Callable[[Idx], F[X, K]]
    """Function representatives of basis vectors."""

    dual_fn: Callable[[Idx], F[X, K]]
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

    evl: Callable[[Idx], K]
    """Operator eigenvalues."""

    gen_evl: Callable[[Idx], K]
    """Generator eigenvalues."""

    efreq: Callable[[Idx], K]
    """Function indexing eigenfrequencies."""

    eperiod: Callable[[Idx], K]
    """Function indexing eigenperiods."""

    engy: Callable[[Idx], K]
    """Function indexing Dirichlet energies."""


class GeneratorShardings(NamedTuple):
    """NamedTuple holding array shardings for generator matrix computation."""

    tangents: L2FnAlgebraShardings = L2FnAlgebraShardings()
    """Shardings for the L2 space used in tangent vector evaluation."""

    basis_grads: NamedSharding | None = None
    """Sharding of basis vector gradients."""

    matrix: NamedSharding | None = None
    """Sharding of generator matrix."""


def make_vgrad_basis(
    eval_tangents: Callable[[F[X, TX, K]], V],
    basis: alg.ImplementsDimensionedL2FnFrame[X, K, V, Ks, Idx],
    batch_size: int | None = None,
    out_shardings: NamedSharding | None = None,
) -> Callable[[Idxs], V]:
    """Make function that computes directional derivatives of basis vectors."""

    @partial(shardit, sharding=out_shardings)
    @partial(batch_map, batch_size=batch_size)
    def vgrad_basis(idx: Idx) -> V:
        return eval_tangents(dyn.vgrad(basis.fn(idx)))

    return vgrad_basis


def make_generator_builder[
    Data: PyTree,
    Eigen: knl.ImplementsSliceableKernelEigen[R, Rs, V, Vs],
](
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[X, R, V, R]],
    impl_basis: Callable[
        [Data, Eigen],
        knl.ImplementsKernelEigenbasis[X, R, V, R, Rs, Idx],
    ],
    impl_eval_tangents: Callable[[Data], Callable[[F[X, TX, R]], V]],
    grad_batch_size: int | None = None,
    gram_batch_size: int | None = None,
    shardings: GeneratorShardings = GeneratorShardings(),
) -> Callable[[Data, Eigen], Mat]:
    """Make function that computes Koopman generator in a kernel eigenbasis."""

    def build_generator(
        data: Data,
        kernel_eigen: Eigen,
    ) -> Mat:
        """Compute matrix representation of Koopman generator."""
        l2x = impl_l2(data)
        eval_tangents = impl_eval_tangents(data)
        basis = impl_basis(data, kernel_eigen)
        basis_idxs = jnp.arange(basis.dim)
        vgrad_basis = make_vgrad_basis(
            eval_tangents,
            basis,
            batch_size=grad_batch_size,
            out_shardings=shardings.basis_grads,
        )
        innerp = batch_map_2d(l2x.innerp, batch_sizes=gram_batch_size)
        phi_duals = basis.dual_vec(basis_idxs)
        vgrad_phis = vgrad_basis(basis_idxs)
        gen_mat = innerp(phi_duals, vgrad_phis)
        if shardings.matrix is not None:
            return jax.lax.with_sharding_constraint(
                gen_mat, shardings=shardings.matrix
            )
        return gen_mat

    return build_generator


def compute_generator_matrix[Data: PyTree](
    pars: tuple[KernelPars, KoopmanParsDiff],
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[X, K, V, K]],
    impl_eval_tangents: Callable[[Data], Callable[[F[X, TX, K]], V]],
    kernel: Callable[[X, X], K] | Callable[[Data, X, X], K],
    train_data: Data,
    kernel_eigen: KernelEigen,
    shardings: GeneratorShardings = GeneratorShardings(),
    jit: bool = True,
) -> Mat:
    """Compute generator matrix representation in kernel eigenbasis."""
    kernel_pars, koopman_pars = pars
    match koopman_pars.which_eigs_galerkin:
        case int():
            which_eigs = (1, koopman_pars.which_eigs_galerkin)
        case _:
            which_eigs = koopman_pars.which_eigs_galerkin
    impl_basis = knl.make_data_driven_eigenbasis(
        kernel_pars, impl_l2, kernel, which_eigs=which_eigs
    )
    op_build: Callable[[Data, KernelEigen], Mat] = make_generator_builder(
        impl_l2,
        impl_basis,
        impl_eval_tangents,
        grad_batch_size=koopman_pars.grad_batch_size,
        gram_batch_size=koopman_pars.gram_batch_size,
        shardings=shardings,
    )

    if jit:
        op_build = typestable_jit(op_build)
    return op_build(train_data, kernel_eigen)


class IntegralTransformShardings(NamedTuple):
    """NamedTuple holding array shardings for Qz matrix computation."""

    quadrature: L2FnAlgebraShardings = L2FnAlgebraShardings()
    """Shardings for the L2 space used in resolvent quadrature."""

    weights: NamedSharding | None = None
    """Sharding of integral transform weight vector."""

    matrix: NamedSharding | None = None
    """Sharding of Qz matrix."""


def make_integral_transform_basis(
    bandwidth: float,
    dt: float,
    transform: Literal["gauss", "laplace"],
    quadrature: Literal["trapezoidal", "simpson"],
    eval_quad: Callable[[F[X, K]], V],
    num_quad: int,
    basis: alg.ImplementsDimensionedL2FnFrame[X, K, V, Ks, Idx],
    batch_size: int | None = None,
    shardings: IntegralTransformShardings = IntegralTransformShardings(),
) -> Callable[[Idxs], Vs]:
    """Make function that computes integral transforms of basis vectors."""
    match transform:
        case "gauss":
            transf_weights = partial(dl.gauss_transform_weights, bandwidth)
        case "laplace":
            transf_weights = partial(dl.laplace_transform_weights, bandwidth)
    match quadrature:
        case "trapezoidal":
            quad_weights = dl.trapezoidal_quadrature_weights
        case "simpson":
            quad_weights = dl.simpson_quadrature_weights
    transf = dl.make_integral_transform(
        dt=dt,
        num_quad=num_quad,
        transform_weights=transf_weights,
        quadrature_weights=quad_weights,
        weight_sharding=shardings.weights,
        out_sharding=shardings.quadrature.vectors,
    )

    @partial(shardit, sharding=shardings.quadrature.vectors)
    @partial(batch_map, batch_size=batch_size)
    def transf_vec(idx: int | Array) -> V:
        return transf(eval_quad(basis.fn(idx)))

    return transf_vec


def make_integral_transform_builder[
    Data: PyTree,
    Eigen: knl.ImplementsSliceableKernelEigen[R, Rs, V, Vs],
](
    bandwidth: float,
    dt: float,
    transform: Literal["gauss", "laplace"],
    quadrature: Literal["trapezoidal", "simpson"],
    num_quad: int,
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[X, K, V, K]],
    impl_basis: Callable[
        [Data, Eigen],
        knl.ImplementsKernelEigenbasis[X, R, V, R, Rs, Idx],
    ],
    impl_eval_quad: Callable[[Data], Callable[[F[X, K]], V]],
    quad_batch_size: int | None = None,
    gram_batch_size: int | None = None,
    shardings: IntegralTransformShardings = IntegralTransformShardings(),
) -> Callable[[Data, Eigen], Mat]:
    """Make function that computes integral transform in a kernel basis."""

    def build_integral_transform(data: Data, kernel_eigen: Eigen) -> Mat:
        """Compute matrix representation of Koopman integral transform."""
        l2x = impl_l2(data)
        eval_quad = impl_eval_quad(data)
        basis = impl_basis(data, kernel_eigen)
        basis_idxs = jnp.arange(basis.dim)
        transf_basis = make_integral_transform_basis(
            bandwidth=bandwidth,
            dt=dt,
            transform=transform,
            quadrature=quadrature,
            num_quad=num_quad,
            eval_quad=eval_quad,
            basis=basis,
            batch_size=quad_batch_size,
            shardings=shardings,
        )
        phi_duals = basis.dual_vec(basis_idxs)
        transf_phis = transf_basis(basis_idxs)
        innerp = batch_map_2d(l2x.innerp, batch_sizes=gram_batch_size)
        transf_mat_asym = innerp(phi_duals, transf_phis)
        transf_mat = (transf_mat_asym - transf_mat_asym.T) / 2
        if shardings.matrix is not None:
            return jax.lax.with_sharding_constraint(
                transf_mat, shardings=shardings.matrix
            )
        return transf_mat

    return build_integral_transform


def compute_integral_transform_matrix[Data: PyTree](
    pars: tuple[KernelPars, KoopmanParsTransf],
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[X, K, V, K]],
    impl_eval_quad: Callable[[Data], Callable[[F[X, K]], V]],
    kernel: Callable[[X, X], K] | Callable[[Data, X, X], K],
    train_data: Data,
    kernel_eigen: KernelEigen,
    shardings: IntegralTransformShardings = IntegralTransformShardings(),
    jit: bool = True,
) -> Mat:
    """Compute generator matrix representation in kernel eigenbasis."""
    kernel_pars, koopman_pars = pars
    match koopman_pars.which_eigs_galerkin:
        case int():
            which_eigs = (1, koopman_pars.which_eigs_galerkin)
        case _:
            which_eigs = koopman_pars.which_eigs_galerkin
    impl_basis = knl.make_data_driven_eigenbasis(
        kernel_pars, impl_l2, kernel, which_eigs
    )
    op_build: Callable[[Data, KernelEigen], Mat] = (
        make_integral_transform_builder(
            koopman_pars.bandwidth,
            koopman_pars.dt,
            koopman_pars.transform,
            koopman_pars.quadrature,
            koopman_pars.num_quad,
            impl_l2,
            impl_basis,
            impl_eval_quad,
            quad_batch_size=koopman_pars.quad_batch_size,
            gram_batch_size=koopman_pars.gram_batch_size,
            shardings=shardings,
        )
    )
    if jit:
        op_build = typestable_jit(op_build)
    return op_build(train_data, kernel_eigen)


class KoopmanEigenShardings(NamedTuple):
    """NamedTuple holding shardings of KoopmanEigen objects."""

    eigenvalues: NamedSharding | None = None
    """Sharding of eigenvalue array."""

    eigenvectors: NamedSharding | None = None
    """Sharding of eigenvector array."""

    @classmethod
    def from_named_sharder[Shape: tuple[int, ...], AxisNames: str](
        cls, sharder: NamedSharder[Shape, AxisNames] | None
    ) -> Self:
        """Create KernelEigenSharding object from NamedSharder."""
        if sharder is not None:
            replicating = sharder.sharding(None)
            return cls(eigenvalues=replicating, eigenvectors=replicating)
        else:
            return cls()


class _GeneratorSpectrum(NamedTuple):
    """NamedTuple holding generator eigendecomposition results."""

    evals: Cs
    """Generator eigenvalues."""

    evec_coeffs: Css
    """Basis expansion coefficients of the eigenvectors."""

    dual_evec_coeffs: Css
    """Basis expansion coefficients of the dual (left) eigenvectors."""


def _from_generator_spectrum(
    kernel_basis: knl.ImplementsKernelEigenbasis[X, R, V, R, Rs, Idx],
    spec: _GeneratorSpectrum,
    sort_by: Literal["frequency", "energy"],
    num_eigs: int | None = None,
    out_shardings: KoopmanEigenShardings = KoopmanEigenShardings(),
) -> KoopmanEigen:
    """Convert _GeneratorSpectrum to KoopmanEigen."""
    if num_eigs is None:
        _num_eigs = kernel_basis.dim - 1
    else:
        _num_eigs = num_eigs
    engys = (
        jnp.sum(
            jnp.abs(spec.evec_coeffs) ** 2 / kernel_basis.spec[:, jnp.newaxis],
            axis=0,
        )
        - 1
    )
    match sort_by:
        case "frequency":
            isort = jnp.argsort(jnp.abs(spec.evals.imag))[: _num_eigs - 1]
        case "energy":
            isort = jnp.argsort(engys)[: _num_eigs - 1]
    evals = jnp.concatenate((jnp.atleast_1d(0), spec.evals[isort]))
    efreqs = jnp.concatenate((jnp.atleast_1d(0), evals[1:].imag))
    eperiods = jnp.concatenate(
        (
            jnp.atleast_1d(jnp.inf),
            2 * jnp.pi / evals[1:].imag,
        )
    )
    engys = jnp.concatenate((jnp.atleast_1d(0), engys[isort]))
    evec_coeffs = jsp.linalg.block_diag(1, spec.evec_coeffs[:, isort].T)
    dual_evec_coeffs = jsp.linalg.block_diag(
        1, spec.dual_evec_coeffs[:, isort].T
    )
    assert isinstance(evec_coeffs, Array)
    assert isinstance(dual_evec_coeffs, Array)
    if out_shardings.eigenvalues is not None:
        evals = jax.lax.with_sharding_constraint(
            evals, shardings=out_shardings.eigenvalues
        )
        efreqs = jax.lax.with_sharding_constraint(
            efreqs, shardings=out_shardings.eigenvalues
        )
        eperiods = jax.lax.with_sharding_constraint(
            eperiods, shardings=out_shardings.eigenvalues
        )
    if out_shardings.eigenvectors is not None:
        engys = jax.lax.with_sharding_constraint(
            engys, shardings=out_shardings.eigenvectors
        )
        evec_coeffs = jax.lax.with_sharding_constraint(
            evec_coeffs, shardings=out_shardings.eigenvectors
        )
        dual_evec_coeffs = jax.lax.with_sharding_constraint(
            dual_evec_coeffs, shardings=out_shardings.eigenvectors
        )
    eigen = KoopmanEigen(
        evals=evals,
        gen_evals=evals,
        efreqs=efreqs,
        eperiods=eperiods,
        engys=engys,
        evec_coeffs=evec_coeffs,
        dual_evec_coeffs=dual_evec_coeffs,
    )
    return eigen


def make_diffusion_regularized_generator_eigensolver[
    Data: PyTree,
    Eigen: knl.ImplementsKernelEigen[R, Rs, V, Vs],
](
    koopman_pars: KoopmanParsDiff,
    impl_basis: Callable[
        [Data, Eigen],
        knl.ImplementsKernelEigenbasis[X, R, V, R, Rs, Idx],
    ],
    out_shardings: KoopmanEigenShardings = KoopmanEigenShardings(),
) -> Callable[[Data, Eigen, Mat], KoopmanEigen]:
    """Make eigensolver for diffusion-regularized generator."""

    def eigensolve(
        data: Data, kernel_eigen: Eigen, gen_mat: Mat
    ) -> KoopmanEigen:
        basis = impl_basis(data, kernel_eigen)
        diff_mat = koopman_pars.tau * jnp.diag(basis.lapl_spec)
        if koopman_pars.antisym:
            reg_gen_mat = (gen_mat - gen_mat.T) / 2 - diff_mat
        else:
            reg_gen_mat = gen_mat - diff_mat
        evals, evec_coeffs = jla.eig(reg_gen_mat)
        anal_op = evec_coeffs @ evec_coeffs.conj().T
        dual_evec_coeffs = jsp.linalg.solve(
            anal_op, evec_coeffs, assume_a="her"
        )
        spec = _GeneratorSpectrum(
            evals=evals,
            evec_coeffs=evec_coeffs,
            dual_evec_coeffs=dual_evec_coeffs,
        )
        eigen = _from_generator_spectrum(
            basis,
            spec,
            num_eigs=koopman_pars.num_eigs,
            sort_by=koopman_pars.sort_by,
            out_shardings=out_shardings,
        )
        return eigen

    return eigensolve


def compute_diffusion_regularized_generator_eigen[Data: PyTree](
    pars: tuple[KernelPars, KoopmanParsDiff],
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[X, X, V, R]],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    data: Data,
    kernel_eigen: KernelEigen,
    gen_mat: Mat,
    out_shardings: KoopmanEigenShardings = KoopmanEigenShardings(),
    jit: bool = True,
) -> KoopmanEigen:
    """Compute eigendecomposition of diffusion-regularized generator."""
    kernel_pars, koopman_pars = pars
    match koopman_pars.which_eigs_galerkin:
        case int():
            which_eigs = (1, koopman_pars.which_eigs_galerkin)
        case _:
            which_eigs = koopman_pars.which_eigs_galerkin
    impl_basis = knl.make_data_driven_eigenbasis(
        kernel_pars, impl_l2, kernel, which_eigs=which_eigs
    )
    eigensolve: Callable[[Data, KernelEigen, Mat], KoopmanEigen] = (
        make_diffusion_regularized_generator_eigensolver(
            koopman_pars,
            impl_basis,
            out_shardings,
        )
    )
    if jit:
        eigensolve = typestable_jit(eigensolve)
    return eigensolve(data, kernel_eigen, gen_mat)


def invert_dawson(iz_evals: Ks, /, bandwidth: float | Array) -> Ks:
    """Compute generator eigenvalues from Gaussian transform eigenvalues.

    Inverts K_D(x, z) = dawsn(x / (2*z)) / z = a via Newton's method,
    taking the large root. Matches the forward transform's Gaussian
    weight exp(-(z*s)**2) assembled in make_gauss_transform.
    """
    a = iz_evals.imag
    s = 2 * bandwidth
    target = a * bandwidth

    def newton(x0: Array) -> Array:
        x = x0
        for _ in range(50):
            u = x / s
            d = dawsn(u)
            fx = d - target
            dfx = (1 - 2 * u * d) / s
            x = x - fx / dfx
        return x

    large_root = newton(s / (2 * target))
    return 1j * large_root


def invert_qz(qz_evals: Ks, /, bandwidth: float | Array) -> Ks:
    """Compute generator eigenvalues from Laplace transform eigenvalues."""
    gen_evals = (
        1j
        * (1 + jnp.sqrt(1 - 4 * bandwidth**2 * qz_evals.imag**2))
        / (2 * qz_evals.imag)
    )
    return gen_evals


class _IntegralTransformSpectrum(NamedTuple):
    """NamedTuple holding integral transform eigendecomposition results."""

    evals: Array
    """Eigenvalues of integral transform operator."""

    evec_coeffs: Array
    """Basis expansion coefficients of the eigenvectors."""

    dual_evec_coeffs: Array
    """Basis expansion coefficients of the dual (left) eigenvectors."""


def _from_integral_transform_spectrum(
    bandwidth: float,
    transform: Literal["gauss", "laplace"],
    kernel_basis: knl.ImplementsKernelEigenbasis[X, R, V, R, Rs, Idx],
    spec: _IntegralTransformSpectrum,
    sort_by: Literal["frequency", "energy"],
    num_eigs: int | None = None,
    out_shardings: KoopmanEigenShardings = KoopmanEigenShardings(),
) -> KoopmanEigen:
    """Convert _IntegralTransformSpectrum to KoopmanEigen."""
    if num_eigs is None:
        _num_eigs = kernel_basis.dim - 1
    else:
        _num_eigs = num_eigs
    match transform:
        case "gauss":
            gen_evals = invert_dawson(spec.evals, bandwidth)
        case "laplace":
            gen_evals = invert_qz(spec.evals, bandwidth)
    engys = (
        jnp.sum(
            jnp.abs(spec.evec_coeffs) ** 2 / kernel_basis.spec[:, jnp.newaxis],
            axis=0,
        )
        - 1
    )
    match sort_by:
        case "frequency":
            isort = jnp.argsort(jnp.abs(gen_evals.imag))[: _num_eigs - 1]
        case "energy":
            isort = jnp.argsort(engys)[: _num_eigs - 1]
    evals = jnp.concatenate((jnp.atleast_1d(0), spec.evals[isort]))
    gen_evals = jnp.concatenate((jnp.atleast_1d(0), gen_evals[isort]))
    efreqs = jnp.concatenate((jnp.atleast_1d(0), gen_evals[1:].imag))
    eperiods = jnp.concatenate(
        (
            jnp.atleast_1d(jnp.inf),
            2 * jnp.pi / gen_evals[1:].imag,
        )
    )
    engys = jnp.concatenate((jnp.atleast_1d(0), engys[isort]))
    evec_coeffs = jsp.linalg.block_diag(1, spec.evec_coeffs[:, isort].T)
    dual_evec_coeffs = jsp.linalg.block_diag(
        1, spec.dual_evec_coeffs[:, isort].T
    )
    assert isinstance(evec_coeffs, Array)
    assert isinstance(dual_evec_coeffs, Array)
    if out_shardings.eigenvalues is not None:
        evals = jax.lax.with_sharding_constraint(
            evals, shardings=out_shardings.eigenvalues
        )
        gen_evals = jax.lax.with_sharding_constraint(
            gen_evals, shardings=out_shardings.eigenvalues
        )
        efreqs = jax.lax.with_sharding_constraint(
            efreqs, shardings=out_shardings.eigenvalues
        )
        eperiods = jax.lax.with_sharding_constraint(
            eperiods, shardings=out_shardings.eigenvalues
        )
    if out_shardings.eigenvectors is not None:
        engys = jax.lax.with_sharding_constraint(
            engys, shardings=out_shardings.eigenvectors
        )
        evec_coeffs = jax.lax.with_sharding_constraint(
            evec_coeffs, shardings=out_shardings.eigenvectors
        )
        dual_evec_coeffs = jax.lax.with_sharding_constraint(
            dual_evec_coeffs, shardings=out_shardings.eigenvectors
        )
    eigen = KoopmanEigen(
        evals=evals,
        gen_evals=gen_evals,
        efreqs=efreqs,
        eperiods=eperiods,
        engys=engys,
        evec_coeffs=evec_coeffs,
        dual_evec_coeffs=dual_evec_coeffs,
    )
    return eigen


def make_compactified_integral_transform_eigensolver[
    Data: PyTree,
    Eigen: knl.ImplementsKernelEigen[R, Rs, V, Vs],
](
    koopman_pars: KoopmanParsTransf,
    impl_basis: Callable[
        [Data, Eigen],
        knl.ImplementsKernelEigenbasis[X, R, V, R, Rs, Idx],
    ],
    out_shardings: KoopmanEigenShardings = KoopmanEigenShardings(),
) -> Callable[[Data, Eigen, Mat], KoopmanEigen]:
    """Make eigensolver for compactified integral transform operator."""

    def eigensolve(
        data: Data, kernel_eigen: Eigen, transf_mat: Mat
    ) -> KoopmanEigen:
        basis = impl_basis(data, kernel_eigen)
        match koopman_pars.smoothing_kernel:
            case "exponential":
                lambs = basis.spec**koopman_pars.tau
            case "fejer":
                lambs = (
                    1 - jnp.sqrt(basis.lapl_spec / basis.lapl_spec[-1])
                ) ** koopman_pars.tau
        reg_transf_mat = lambs * transf_mat * lambs[:, jnp.newaxis]
        if koopman_pars.antisym:
            _evals, evec_coeffs = jla.eigh(-1j * reg_transf_mat)
            assert isinstance(_evals, Array)
            assert isinstance(evec_coeffs, Array)
            evals = 1j * _evals
            dual_evec_coeffs = evec_coeffs
        else:
            evals, evec_coeffs = jla.eig(reg_transf_mat)
            anal_op = evec_coeffs @ evec_coeffs.conj().T
            dual_evec_coeffs = jsp.linalg.solve(
                anal_op, evec_coeffs, assume_a="her"
            )
        spec = _IntegralTransformSpectrum(
            evals=evals,
            evec_coeffs=evec_coeffs,
            dual_evec_coeffs=dual_evec_coeffs,
        )
        eigen = _from_integral_transform_spectrum(
            koopman_pars.bandwidth,
            koopman_pars.transform,
            basis,
            spec,
            num_eigs=koopman_pars.num_eigs,
            sort_by=koopman_pars.sort_by,
            out_shardings=out_shardings,
        )
        return eigen

    return eigensolve


def compute_integral_transform_eigen_comp[Data: PyTree](
    pars: tuple[KernelPars, KoopmanParsTransf],
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[X, X, V, R]],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    data: Data,
    kernel_eigen: KernelEigen,
    transf_mat: Mat,
    out_shardings: KoopmanEigenShardings = KoopmanEigenShardings(),
    jit: bool = True,
) -> KoopmanEigen:
    """Compute eigendecomposition of compactified integral transform."""
    kernel_pars, koopman_pars = pars
    match koopman_pars.which_eigs_galerkin:
        case int():
            which_eigs = (1, koopman_pars.which_eigs_galerkin)
        case _:
            which_eigs = koopman_pars.which_eigs_galerkin
    impl_basis = knl.make_data_driven_eigenbasis(
        kernel_pars, impl_l2, kernel, which_eigs=which_eigs
    )
    eigensolve: Callable[[Data, KernelEigen, Mat], KoopmanEigen] = (
        make_compactified_integral_transform_eigensolver(
            koopman_pars, impl_basis, out_shardings
        )
    )
    if jit:
        eigensolve = typestable_jit(eigensolve)
    return eigensolve(data, kernel_eigen, transf_mat)


def make_eigenbasis_asym[L: int, D: DTypeLike](
    c_l: L2VectorAlgebra[tuple[L], D],
    kernel_basis: knl.ImplementsKernelEigenbasis[X, R, V, R, Rs, Idx],
    koopman_eigen: koop.ImplementsSliceableKoopmanEigen[C, Cs, Css],
) -> KoopmanEigenbasis:
    """Make Koopman eigenbasis from eigendecomposition of asymmetric op."""

    def vc(i: int | Array) -> V:
        return kernel_basis.synth(koopman_eigen.evec_coeffs[i])

    def dual_vc(i: int | Array) -> V:
        return kernel_basis.dual_synth(koopman_eigen.dual_evec_coeffs[i])

    def evl(i: int | Array) -> K:
        return koopman_eigen.evals[i]

    def gen_evl(i: int | Array) -> K:
        return koopman_eigen.gen_evals[i]

    def efreq(i: int | Array) -> K:
        return koopman_eigen.efreqs[i]

    def eperiod(i: int | Array) -> K:
        return koopman_eigen.eperiods[i]

    def engy(i: int | Array) -> K:
        return koopman_eigen.engys[i]

    def fn(i: int | Array) -> Callable[[X], K]:
        return kernel_basis.fn_synth(koopman_eigen.evec_coeffs[i])

    def dual_fn(i: int | Array) -> Callable[[X], K]:
        return kernel_basis.dual_fn_synth(koopman_eigen.dual_evec_coeffs[i])

    @partial(vmap, in_axes=(0, None))
    def anal_eval_c(i: int | Array, v: V) -> K:
        return c_l.innerp(koopman_eigen.dual_evec_coeffs[i], v)

    @partial(vmap, in_axes=(0, None))
    def dual_anal_eval_c(i: int | Array, v: V) -> K:
        return c_l.innerp(koopman_eigen.evec_coeffs[i], v)

    num_eigs = koop.num_eigs_in_eigen(koopman_eigen)
    idxs = jnp.arange(num_eigs)
    anal_c = partial(anal_eval_c, idxs)
    anal = fun.compose(anal_c, kernel_basis.anal)
    dual_anal_c = partial(dual_anal_eval_c, idxs)
    dual_anal = fun.compose(dual_anal_c, kernel_basis.dual_anal)
    fn_anal = fun.compose(anal_c, kernel_basis.fn_anal)
    dual_fn_anal = fun.compose(dual_anal_c, kernel_basis.dual_fn_anal)
    synth_c = vec.make_synthesis_operator(koopman_eigen.evec_coeffs, idxs)
    synth = fun.compose(kernel_basis.synth, synth_c)
    dual_synth_c = vec.make_synthesis_operator(
        koopman_eigen.dual_evec_coeffs, idxs
    )
    dual_synth = fun.compose(kernel_basis.dual_synth, dual_synth_c)
    fn_synth = fun.compose(kernel_basis.fn_synth, synth_c)
    dual_fn_synth = fun.compose(kernel_basis.dual_fn_synth, synth_c)
    spec = koopman_eigen.evals[idxs]
    gen_spec = koopman_eigen.gen_evals[idxs]
    efreqs = koopman_eigen.efreqs[idxs]
    eperiods = koopman_eigen.eperiods[idxs]
    engys = koopman_eigen.engys[idxs]
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


def make_eigenbasis_antisym[L: int, D: DTypeLike](
    c_l: L2VectorAlgebra[tuple[L], D],
    kernel_basis: knl.ImplementsKernelEigenbasis[X, R, V, R, Rs, Idx],
    koopman_eigen: koop.ImplementsSliceableKoopmanEigen[C, Cs, Css],
) -> KoopmanEigenbasis:
    """Make Koopman eigenbasis from eigendecomposition of antisymmetric op."""

    def vc(i: int | Array) -> V:
        return kernel_basis.synth(koopman_eigen.evec_coeffs[i])

    def evl(i: int | Array) -> K:
        return koopman_eigen.evals[i]

    def gen_evl(i: int | Array) -> K:
        return koopman_eigen.gen_evals[i]

    def efreq(i: int | Array) -> K:
        return koopman_eigen.efreqs[i]

    def eperiod(i: int | Array) -> K:
        return koopman_eigen.eperiods[i]

    def engy(i: int | Array) -> K:
        return koopman_eigen.engys[i]

    def fn(i: int | Array) -> Callable[[X], K]:
        return kernel_basis.fn_synth(koopman_eigen.evec_coeffs[i])

    @partial(vmap, in_axes=(0, None))
    def anal_eval_c(i: int | Array, v: V) -> K:
        return c_l.innerp(koopman_eigen.evec_coeffs[i], v)

    num_eigs = koop.num_eigs_in_eigen(koopman_eigen)
    idxs = jnp.arange(num_eigs)
    anal_c = partial(anal_eval_c, idxs)
    anal = fun.compose(anal_c, kernel_basis.anal)
    fn_anal = fun.compose(anal_c, kernel_basis.fn_anal)
    synth_c = vec.make_synthesis_operator(koopman_eigen.evec_coeffs, idxs)
    synth = fun.compose(kernel_basis.synth, synth_c)
    fn_synth = fun.compose(kernel_basis.fn_synth, synth_c)
    spec = koopman_eigen.evals[idxs]
    gen_spec = koopman_eigen.gen_evals[idxs]
    efreqs = koopman_eigen.efreqs[idxs]
    eperiods = koopman_eigen.eperiods[idxs]
    engys = koopman_eigen.engys[idxs]
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


def make_eigenbasis[L: int, D: DTypeLike](
    pars: KoopmanPars,
    c_l: L2VectorAlgebra[tuple[L], D],
    kernel_basis: knl.ImplementsKernelEigenbasis[X, R, V, R, Rs, Idx],
    koopman_eigen: koop.ImplementsSliceableKoopmanEigen[C, Cs, Css],
) -> KoopmanEigenbasis:
    """Make Koopman eigenbasis."""
    match pars, pars.antisym:
        case KoopmanParsTransf(), True:
            basis = make_eigenbasis_antisym(c_l, kernel_basis, koopman_eigen)
        case _, _:
            basis = make_eigenbasis_asym(c_l, kernel_basis, koopman_eigen)
    return basis


# def slice_eigen(
#     eigen: KoopmanEigen[C, Cs, Css],
#     which_eigs: int | tuple[int, int] | list[int] | None = None,
# ) -> KoopmanEigen[C, Cs, Css]:
#     """Slice KoopmanEigen object using `which_eigs` convention."""
#     match which_eigs:
#         case None:
#             sliced_eigen = eigen
#         case int() as num_eigs:
#             sliced_eigen = eigen.isel(slice(0, num_eigs))
#         case tuple() as idx:
#             sliced_eigen = eigen.isel(slice(idx[0], idx[1] + 1))
#         case list() as idxs:
#             sliced_eigen = eigen.isel(idxs)
#     return sliced_eigen


# TODO: Consider automating the process of building these data-driven wrappers
# using a decorator.
def make_data_driven_eigenbasis[
    Data: PyTree,
    KnlEigen: knl.ImplementsKernelEigen[R, Rs, V, Vs],
    D: DTypeLike,
    L: int,
](
    koopman_pars: KoopmanPars,
    c_l: L2VectorAlgebra[tuple[L], D],
    impl_kernel_basis: Callable[
        [Data, KnlEigen],
        knl.ImplementsKernelEigenbasis[X, R, V, R, Rs, Idx],
    ],
    which_eigs: int | tuple[int, int] | list[int] | None = None,
) -> Callable[
    [Data, KnlEigen, koop.ImplementsSliceableKoopmanEigen[C, Cs, Css]],
    KoopmanEigenbasis,
]:
    """Make data-driven Koopman eigenbasis builder."""

    def _make_eigenbasis(
        data: Data,
        kernel_eigen: KnlEigen,
        koopman_eigen: koop.ImplementsSliceableKoopmanEigen[C, Cs, Css],
    ) -> KoopmanEigenbasis:
        kernel_basis = impl_kernel_basis(data, kernel_eigen)
        _koopman_eigen = koop.slice_eigen(koopman_eigen, which_eigs)
        return make_eigenbasis(koopman_pars, c_l, kernel_basis, _koopman_eigen)

    return _make_eigenbasis


def make_koopman_analysis_operator[
    Data: PyTree,
    KnlEigen: knl.ImplementsKernelEigen[R, Rs, V, Vs],
    KoopEigen: koop.ImplementsKoopmanEigen[C, Cs, Css],
](
    impl_basis: Callable[
        [Data, KnlEigen, KoopEigen],
        KoopmanEigenbasis,
    ],
    which_samples: tuple[int, int] | None = None,
) -> Callable[[Data, V, KnlEigen, KoopEigen], Cs]:
    """Make analysis operator for Koopman forecast."""

    def anal(
        data: Data,
        response: V,
        kernel_eigen: KnlEigen,
        koopman_eigen: KoopEigen,
    ) -> Cs:
        if which_samples is not None:
            i0 = which_samples[0]
            i1 = which_samples[1]
        else:
            i0 = 0
            i1 = len(response)
        basis = impl_basis(data, kernel_eigen, koopman_eigen)
        return basis.anal(response[i0:i1])

    return anal


def make_koopman_prediction_function[
    Data: PyTree,
    KnlEigen: knl.ImplementsKernelEigen[R, Rs, V, Vs],
    KoopEigen: koop.ImplementsKoopmanEigen[C, Cs, Css],
    TestData: PyTree,
](
    impl_basis: Callable[
        [Data, KnlEigen, KoopEigen],
        KoopmanEigenbasis,
    ],
    impl_l2_tst: Callable[
        [TestData], alg.ImplementsL2FnAlgebra[X, R, Vtst, R]
    ],
) -> Callable[
    [Data, KnlEigen, KoopEigen, Cs, Rs, TestData],
    Vtst,
]:
    """Make prediction function for Koopman forecast."""

    def predict(
        data: Data,
        kernel_eigen: KnlEigen,
        koopman_eigen: KoopEigen,
        coeffs: Cs,
        ts: Rs,
        test_data: TestData,
    ) -> Rs:
        basis = impl_basis(data, kernel_eigen, koopman_eigen)
        l2x_tst = impl_l2_tst(test_data)

        @partial(vmap, in_axes=(None, 0, None))
        def _predict(cs: Cs, t: R, x: X) -> R:
            phases = jnp.exp(basis.gen_spec * t)
            return basis.fn_synth(phases * cs)(x)

        return l2x_tst.incl(partial(_predict, coeffs, ts))

    return predict


def compute_koopman_preds[
    Data: PyTree,
    D: DTypeLike,
    L: int,
    TestData: PyTree,
](
    pars: tuple[KernelPars, KoopmanPars],
    c_l: L2VectorAlgebra[tuple[L], D],
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[X, R, V, R]],
    train_data: Data,
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    kernel_eigen: KernelEigen,
    koopman_eigen: KoopmanEigen,
    coeffs: Cs,
    impl_l2_tst: Callable[
        [TestData], alg.ImplementsL2FnAlgebra[X, R, Vtst, R]
    ],
    test_data: TestData,
    num_steps: int,
    dt: float,
    which_eigs: int | tuple[int, int] | list[int] | None = None,
    jit: bool = True,
) -> Array:
    """Compute Koopman predictions."""
    kernel_pars, koopman_pars = pars
    match koopman_pars.which_eigs_galerkin:
        case int():
            which_kernel_eigs = koopman_pars.which_eigs_galerkin + 1
        case tuple():
            which_kernel_eigs = [0] + list(
                range(
                    koopman_pars.which_eigs_galerkin[0],
                    koopman_pars.which_eigs_galerkin[1] + 1,
                )
            )
        case list():
            which_kernel_eigs = [0] + koopman_pars.which_eigs_galerkin
    impl_kernel_basis = knl.make_data_driven_eigenbasis(
        kernel_pars, impl_l2, kernel, which_kernel_eigs
    )
    impl_koopman_basis = make_data_driven_eigenbasis(
        koopman_pars, c_l, impl_kernel_basis, which_eigs
    )
    predict: Callable[
        [Data, KernelEigen, KoopmanEigen, Cs, Rs, TestData], Vtst
    ] = make_koopman_prediction_function(impl_koopman_basis, impl_l2_tst)
    ts = jnp.arange(num_steps + 1) * dt
    if jit:
        predict = typestable_jit(predict)
    return predict(
        train_data, kernel_eigen, koopman_eigen, coeffs, ts, test_data
    )


def make_eigenfunction_evaluation_functional[
    Data: PyTree,
    TestData: PyTree,
    KnlEigen: knl.ImplementsKernelEigen[R, Rs, V, Vs],
    KoopEigen: koop.ImplementsKoopmanEigen[C, Cs, Css],
](
    impl_eval: Callable[[TestData], Callable[[F[X, R]], Vtst]],
    impl_koopman_basis: Callable[
        [Data, KnlEigen, KoopEigen],
        koop.ImplementsKoopmanEigenbasis[X, R, V, R, Rs, Idx],
    ],
    idx_eig: int,
) -> Callable[[Data, KnlEigen, KoopEigen, TestData], Vtst]:
    """Make evaluation functional for a Koopman eigenfunction on dataset."""

    def eval_koop(
        train_data: Data,
        kernel_eigen: KnlEigen,
        koopman_eigen: KoopEigen,
        test_data: TestData,
    ) -> Vtst:
        koopman_basis = impl_koopman_basis(
            train_data, kernel_eigen, koopman_eigen
        )
        eval_response = impl_eval(test_data)
        return eval_response(koopman_basis.fn(idx_eig))

    return eval_koop


def evaluate_eigenfunction[
    Data: PyTree,
    TestData: PyTree,
    D: DTypeLike,
    L: int,
](
    pars: tuple[KernelPars, KoopmanPars],
    c_l: L2VectorAlgebra[tuple[L], D],
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[X, R, V, R]],
    impl_eval: Callable[[TestData], Callable[[F[X, R]], Vtst]],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    idx_eig: int,
    train_data: Data,
    kernel_eigen: KernelEigen,
    koopman_eigen: KoopmanEigen,
    test_data: TestData,
    jit: bool = True,
) -> Array:
    """Compute values of Koopman eigenfunction on a test dataset."""
    kernel_pars, koopman_pars = pars
    impl_kernel_basis = knl.make_data_driven_eigenbasis(
        kernel_pars, impl_l2, kernel, koopman_pars.which_kernel_eigs
    )
    impl_koopman_basis = make_data_driven_eigenbasis(
        koopman_pars, c_l, impl_kernel_basis
    )
    eval_koop: Callable[[Data, KernelEigen, KoopmanEigen, TestData], Vtst] = (
        make_eigenfunction_evaluation_functional(
            impl_eval, impl_koopman_basis, idx_eig
        )
    )
    if jit:
        eval_koop = typestable_jit(eval_koop)
    return eval_koop(train_data, kernel_eigen, koopman_eigen, test_data)

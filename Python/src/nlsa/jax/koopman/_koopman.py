"""Provide functions for Koopmman operator computations in JAX."""
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
import nlsa.jax.kernels as knl
import nlsa.jax.vector_algebra as vec
from collections.abc import Callable
from functools import partial
from jax import Array, vmap
from jax.sharding import NamedSharding
from jax.typing import DTypeLike
from matplotlib.figure import Figure
from nlsa.kernels import KernelEigen, KernelEigenbasis, KernelPars
from nlsa.koopman import (
    KoopmanEigen,
    KoopmanEigenbasis,
    KoopmanPars,
    KoopmanParsDiff,
    KoopmanParsLapl,
    KoopmanParsGauss,
    KoopmanParsTransf,
)
from nlsa.jax.sharding import NamedSharder, shardit
from nlsa.jax.typing import PyTree
from nlsa.jax.vector_algebra import (
    L2FnAlgebra,
    L2FnAlgebraShardings,
    L2VectorAlgebra,
)
from nlsa.jax.special import dawsn
from nlsa.jax.utils import batch_map, batch_map_2d
from typing import Literal, NamedTuple, Optional, Self

type Css = Array  # Collection of basis expansion coefficient vectors
type C = Array  # Complex scalar
type Cs = Array  # Collection of complex scalars
type R = Array  # Real scalar
type Rs = Array  # Collection of real scalars
type K = Array  # Scalar
type Ks = Array  # Collection of scalars
type V = Array  # L2 observable vector
type Vs = Array  # Collection of L2 vectors
type Idxs = Array  # basis vector indices
type Mat = Array  # Matrix acting as linear operator on L2 vectors
type Shape = tuple[int, ...]
type F[*Ss, T] = Callable[[*Ss], T]  # Shorthand for Callables


class GeneratorShardings(NamedTuple):
    """NamedTuple holding array shardings for generator matrix computation."""

    tangents: L2FnAlgebraShardings = L2FnAlgebraShardings()
    """Shardings for the L2 space used in tangent vector evaluation."""

    basis_grads: Optional[NamedSharding] = None
    """Sharding of basis vector gradients."""

    matrix: Optional[NamedSharding] = None
    """Sharding of generator matrix."""


def make_vgrad_basis[X: Array, TX: Array](
    eval_tangents: Callable[[F[X, TX, K]], V],
    basis: alg.ImplementsDimensionedL2FnFrame[X, K, V, Ks, int | Idxs],
    batch_size: Optional[int] = None,
    out_shardings: Optional[NamedSharding] = None,
) -> Callable[[Idxs], Vs]:
    """Make function that computes directional derivatives of basis vectors."""

    @partial(shardit, sharding=out_shardings)
    @partial(batch_map, batch_size=batch_size)
    def vgrad_basis(idx: int | Array) -> V:
        return eval_tangents(dyn.vgrad(basis.fn(idx)))

    return vgrad_basis


def make_generator_builder[
    Data: PyTree,
    Ns: Shape,
    D: DTypeLike,
    X: Array,
    TX: Array,
](
    pars: KernelPars,
    impl_l2: Callable[[Data], L2FnAlgebra[Ns, D, X, K]],
    impl_eval_tangents: Callable[[Data], Callable[[F[X, TX, K]], V]],
    kernel: Callable[[X, X], K] | Callable[[Data, X, X], K],
    which_eigs_galerkin: int | tuple[int, int] | list[int],
    grad_batch_size: Optional[int] = None,
    gram_batch_size: Optional[int] = None,
    shardings: GeneratorShardings = GeneratorShardings(),
    jit: bool = True,
) -> Callable[[Data, KernelEigen[K, Ks, V, Vs]], Mat]:
    """Make function that computes Koopman generator in a kernel eigenbasis."""
    match which_eigs_galerkin:
        case int():
            which_eigs = (1, which_eigs_galerkin)
        case _:
            which_eigs = which_eigs_galerkin
    impl_basis = knl.make_data_driven_eigenbasis(
        pars, impl_l2, kernel, which_eigs
    )

    def build_generator(
        data: Data, kernel_eigen: KernelEigen[R, Rs, V, Vs]
    ) -> Mat:
        """Compute matrix representation of Koopman generator."""
        l2x = impl_l2(data)
        eval_tangents = impl_eval_tangents(data)
        basis = impl_basis(data, kernel_eigen)
        basis_idxs = jnp.arange(basis.dim)
        sharding = l2x.sharding
        if sharding is not None:
            assert isinstance(sharding, NamedSharding)
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

    # def build_generator(
    #     data: Data, kernel_eigen: KernelEigen[R, Rs, V, Vs]
    # ) -> Mat:
    #     """Compute matrix representation of Koopman generator."""
    #     l2x = impl_l2(data)
    #     basis = impl_basis(data, kernel_eigen)
    #     eval_tangents = impl_eval_tangents(data)

    #     def vgrad_basis(idx: int | Array) -> V:
    #         return eval_tangents(dyn.vgrad(basis.fn(idx)))

    #     @partial(batch_map_2d, batch_sizes=gram_batch_size)
    #     def generator_elements(i: int | Array, j: int | Array) -> R:
    #         return l2x.innerp(basis.dual_vec(i), vgrad_basis(j))

    #     basis_idxs = jnp.arange(basis.dim)
    #     gen_mat = generator_elements(basis_idxs, basis_idxs)

    #     if shardings.matrix is not None:
    #         return jax.lax.with_sharding_constraint(
    #             gen_mat, shardings=shardings.matrix
    #         )
    #     return gen_mat

    if jit:
        return jax.jit(build_generator)
    return build_generator


def compute_generator_matrix[
    Data: PyTree,
    Ns: Shape,
    D: DTypeLike,
    X: Array,
    TX: Array,
](
    pars: tuple[KernelPars, KoopmanParsDiff],
    impl_l2: Callable[[Data], L2FnAlgebra[Ns, D, X, K]],
    impl_eval_tangents: Callable[[Data], Callable[[F[X, TX, K]], V]],
    kernel: Callable[[X, X], K] | Callable[[Data, X, X], K],
    train_data: Data,
    kernel_eigen: KernelEigen[R, Rs, V, Vs],
    shardings: GeneratorShardings = GeneratorShardings(),
    jit: bool = True,
) -> Mat:
    """Compute generator matrix representation in kernel eigenbasis."""
    kernel_pars, koopman_pars = pars
    op_build = make_generator_builder(
        kernel_pars,
        impl_l2,
        impl_eval_tangents,
        kernel,
        which_eigs_galerkin=koopman_pars.which_eigs_galerkin,
        grad_batch_size=koopman_pars.grad_batch_size,
        gram_batch_size=koopman_pars.gram_batch_size,
        shardings=shardings,
        jit=jit,
    )
    return op_build(train_data, kernel_eigen)


class IntegralTransformShardings(NamedTuple):
    """NamedTuple holding array shardings for Qz matrix computation."""

    quadrature: L2FnAlgebraShardings = L2FnAlgebraShardings()
    """Shardings for the L2 space used in resolvent quadrature."""

    weights: Optional[NamedSharding] = None
    """Sharding of integral transform weight vector."""

    matrix: Optional[NamedSharding] = None
    """Sharding of Qz matrix."""


def make_integral_transform_basis[X: Array](
    bandwidth: float,
    dt: float,
    transform: Literal["gauss", "laplace"],
    eval_quad: Callable[[F[X, K]], V],
    num_quad: int,
    basis: alg.ImplementsDimensionedL2FnFrame[X, K, V, Ks, int | Idxs],
    batch_size: Optional[int] = None,
    shardings: IntegralTransformShardings = IntegralTransformShardings(),
) -> Callable[[Idxs], Vs]:
    """Make function that computes integral transforms of basis vectors."""
    match transform:
        case "gauss":
            transf = dl.make_gauss_transform(
                z=bandwidth,
                dt=dt,
                num_quad=num_quad,
                weight_sharding=shardings.weights,
                out_sharding=shardings.quadrature.vectors,
            )
        case "laplace":
            transf = dl.make_laplace_transform(
                z=bandwidth,
                dt=dt,
                num_quad=num_quad,
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
    Ns: Shape,
    D: DTypeLike,
    X: Array,
](
    bandwidth: float,
    dt: float,
    transform: Literal["gauss", "laplace"],
    num_quad: int,
    pars: KernelPars,
    impl_l2: Callable[[Data], L2FnAlgebra[Ns, D, X, K]],
    impl_eval_quad: Callable[[Data], Callable[[F[X, K]], V]],
    kernel: Callable[[X, X], K] | Callable[[Data, X, X], K],
    which_eigs_galerkin: int | tuple[int, int] | list[int],
    quad_batch_size: Optional[int] = None,
    gram_batch_size: Optional[int] = None,
    shardings: IntegralTransformShardings = IntegralTransformShardings(),
    jit: bool = False,
) -> Callable[[Data, KernelEigen[K, Ks, V, Vs]], Mat]:
    """Make function that computes integral transform in a kernel basis."""
    match which_eigs_galerkin:
        case int():
            which_eigs = (1, which_eigs_galerkin)
        case _:
            which_eigs = which_eigs_galerkin
    impl_basis = knl.make_data_driven_eigenbasis(
        pars, impl_l2, kernel, which_eigs
    )

    def build_integral_transform(
        data: Data, kernel_eigen: KernelEigen[R, Rs, V, Vs]
    ) -> Mat:
        """Compute matrix representation of Koopman integral transform."""
        l2x = impl_l2(data)
        eval_quad = impl_eval_quad(data)
        basis = impl_basis(data, kernel_eigen)
        basis_idxs = jnp.arange(basis.dim)
        sharding = l2x.sharding
        if sharding is not None:
            assert isinstance(sharding, NamedSharding)
        transf_basis = make_integral_transform_basis(
            bandwidth=bandwidth,
            dt=dt,
            transform=transform,
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

    if jit:
        return jax.jit(build_integral_transform)
    return build_integral_transform


def compute_integral_transform_matrix[
    Data: PyTree,
    Ns: Shape,
    D: DTypeLike,
    X: Array,
](
    pars: tuple[KernelPars, KoopmanParsTransf],
    impl_l2: Callable[[Data], L2FnAlgebra[Ns, D, X, K]],
    impl_eval_quad: Callable[[Data], Callable[[F[X, K]], V]],
    kernel: Callable[[X, X], K] | Callable[[Data, X, X], K],
    train_data: Data,
    kernel_eigen: KernelEigen[R, Rs, V, Vs],
    shardings: IntegralTransformShardings = IntegralTransformShardings(),
    jit: bool = True,
) -> Mat:
    """Compute generator matrix representation in kernel eigenbasis."""
    kernel_pars, koopman_pars = pars
    match koopman_pars:
        case KoopmanParsGauss():
            transform = "gauss"
        case KoopmanParsLapl():
            transform = "laplace"
    op_build = make_integral_transform_builder(
        koopman_pars.bandwidth,
        koopman_pars.dt,
        transform,
        koopman_pars.num_quad,
        kernel_pars,
        impl_l2,
        impl_eval_quad,
        kernel,
        which_eigs_galerkin=koopman_pars.which_eigs_galerkin,
        quad_batch_size=koopman_pars.quad_batch_size,
        gram_batch_size=koopman_pars.gram_batch_size,
        shardings=shardings,
        jit=jit,
    )
    return op_build(train_data, kernel_eigen)


class KoopmanEigenShardings(NamedTuple):
    """NamedTuple holding shardings of KoopmanEigen objects."""

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


class _GeneratorSpectrum(NamedTuple):
    """NamedTuple holding generator eigendecomposition results."""

    evals: Array
    """Generator eigenvalues."""

    evec_coeffs: Array
    """Basis expansion coefficients of the eigenvectors."""

    dual_evec_coeffs: Array
    """Basis expansion coefficients of the dual (left) eigenvectors."""


def _from_generator_spectrum[X: Array](
    kernel_basis: KernelEigenbasis[X, K, V, Ks, int | Array],
    spec: _GeneratorSpectrum,
    sort_by: Literal["frequency", "energy"],
    num_eigs: Optional[int] = None,
    out_shardings: KoopmanEigenShardings = KoopmanEigenShardings(),
) -> KoopmanEigen[Ks, Ks, Css]:
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


def make_generator_eigensolver_diff[
    Ns: Shape,
    D: DTypeLike,
    X: Array,
    Data: PyTree,
](
    pars: tuple[KernelPars, KoopmanParsDiff],
    impl_l2: Callable[[Data], L2FnAlgebra[Ns, D, X, K]],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    out_shardings: KoopmanEigenShardings = KoopmanEigenShardings(),
    jit: bool = True,
) -> Callable[
    [Data, KernelEigen[R, Rs, V, Vs], Mat], KoopmanEigen[C, Cs, Css]
]:
    """Make eigensolver for diffusion-regularized generator."""
    kernel_pars, koopman_pars = pars
    match koopman_pars.which_eigs_galerkin:
        case int():
            which_eigs = (1, koopman_pars.which_eigs_galerkin)
        case _:
            which_eigs = koopman_pars.which_eigs_galerkin
    impl_basis = knl.make_data_driven_eigenbasis(
        kernel_pars, impl_l2, kernel, which_eigs=which_eigs
    )

    def eigensolve(
        data: Data, kernel_eigen: KernelEigen[R, Rs, V, Vs], gen_mat: Array
    ) -> KoopmanEigen[C, Cs, Css]:
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

    if jit:
        return jax.jit(eigensolve)
    return eigensolve


def compute_generator_eigen_diff[
    Ns: Shape,
    D: DTypeLike,
    X: Array,
    Data: PyTree,
](
    pars: tuple[KernelPars, KoopmanParsDiff],
    impl_l2: Callable[[Data], L2FnAlgebra[Ns, D, X, K]],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    data: Data,
    kernel_eigen: KernelEigen[R, Rs, V, Vs],
    gen_mat: Mat,
    out_shardings: KoopmanEigenShardings = KoopmanEigenShardings(),
    jit: bool = True,
) -> KoopmanEigen[C, Cs, Css]:
    """Compute eigendecomposition of diffusion-regularized generator."""
    eigensolve = make_generator_eigensolver_diff(
        pars, impl_l2, kernel, out_shardings, jit
    )
    return eigensolve(data, kernel_eigen, gen_mat)


def invert_dawson(iz_evals: Ks, /, bandwidth: float | Array) -> Ks:
    """Compute generator eigenvalues from Gaussian transform eigenvalues.

    Inverts K_D(x, z) = dawsn(x / (2*sqrt(z))) / sqrt(z) = a via
    Newton's method, taking the large root.
    """
    a = iz_evals.imag
    s = 2 * jnp.sqrt(bandwidth)
    target = a * jnp.sqrt(bandwidth)

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


def _from_integral_transform_spectrum[X: Array](
    bandwidth: float,
    transform: Literal["gauss", "laplace"],
    kernel_basis: KernelEigenbasis[X, K, V, Ks, int | Array],
    spec: _IntegralTransformSpectrum,
    sort_by: Literal["frequency", "energy"],
    num_eigs: Optional[int] = None,
    out_shardings: KoopmanEigenShardings = KoopmanEigenShardings(),
) -> KoopmanEigen[Ks, Ks, Css]:
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


def make_integral_transform_eigensolver_comp[
    Ns: Shape,
    D: DTypeLike,
    X: Array,
    Data: PyTree,
](
    pars: tuple[KernelPars, KoopmanParsTransf],
    impl_l2: Callable[[Data], L2FnAlgebra[Ns, D, X, K]],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    out_shardings: KoopmanEigenShardings = KoopmanEigenShardings(),
    jit: bool = True,
) -> Callable[
    [Data, KernelEigen[R, Rs, V, Vs], Mat], KoopmanEigen[C, Cs, Css]
]:
    """Make eigensolver for compactified integral transform operator."""
    kernel_pars, koopman_pars = pars
    match koopman_pars:
        case KoopmanParsGauss():
            transform = "gauss"
        case KoopmanParsLapl():
            transform = "laplace"
    match koopman_pars.which_eigs_galerkin:
        case int():
            which_eigs = (1, koopman_pars.which_eigs_galerkin)
        case _:
            which_eigs = koopman_pars.which_eigs_galerkin
    impl_basis = knl.make_data_driven_eigenbasis(
        kernel_pars, impl_l2, kernel, which_eigs=which_eigs
    )

    def eigensolve(
        data: Data, kernel_eigen: KernelEigen[R, Rs, V, Vs], transf_mat: Array
    ) -> KoopmanEigen[C, Cs, Css]:
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
            transform,
            basis,
            spec,
            num_eigs=koopman_pars.num_eigs,
            sort_by=koopman_pars.sort_by,
            out_shardings=out_shardings,
        )
        return eigen

    if jit:
        return jax.jit(eigensolve)
    return eigensolve


def compute_integral_transform_eigen_comp[
    Ns: Shape,
    D: DTypeLike,
    X: Array,
    Data: PyTree,
](
    pars: tuple[KernelPars, KoopmanParsTransf],
    impl_l2: Callable[[Data], L2FnAlgebra[Ns, D, X, K]],
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    data: Data,
    kernel_eigen: KernelEigen[R, Rs, V, Vs],
    transf_mat: Mat,
    out_shardings: KoopmanEigenShardings = KoopmanEigenShardings(),
    jit: bool = True,
) -> KoopmanEigen[C, Cs, Css]:
    """Compute eigendecomposition of compactified integral transform."""
    eigensolve = make_integral_transform_eigensolver_comp(
        pars, impl_l2, kernel, out_shardings, jit
    )
    return eigensolve(data, kernel_eigen, transf_mat)


def make_eigenbasis_asym[X: Array, L: int, D: DTypeLike](
    c_l: L2VectorAlgebra[tuple[L], D],
    kernel_basis: KernelEigenbasis[X, K, V, Ks, int | Array],
    koopman_eigen: KoopmanEigen[C, Cs, Css],
) -> KoopmanEigenbasis[X, K, V, Ks, int | Array]:
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

    idxs = jnp.arange(koopman_eigen.num_eigs)
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


def make_eigenbasis_antisym[X: Array, L: int, D: DTypeLike](
    c_l: L2VectorAlgebra[tuple[L], D],
    kernel_basis: KernelEigenbasis[X, K, V, Ks, int | Array],
    koopman_eigen: KoopmanEigen[C, Cs, Css],
) -> KoopmanEigenbasis[X, K, V, Ks, int | Array]:
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

    idxs = jnp.arange(koopman_eigen.num_eigs)
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


def make_eigenbasis[X: Array, L: int, D: DTypeLike](
    pars: KoopmanPars,
    c_l: L2VectorAlgebra[tuple[L], D],
    kernel_basis: KernelEigenbasis[X, K, V, Ks, int | Array],
    koopman_eigen: KoopmanEigen[C, Cs, Css],
) -> KoopmanEigenbasis[X, K, V, Ks, int | Array]:
    """Make Koopman eigenbasis from eigendecomposition of antisymmetric op."""
    match pars, pars.antisym:
        case KoopmanParsGauss() | KoopmanParsLapl(), True:
            basis = make_eigenbasis_antisym(c_l, kernel_basis, koopman_eigen)
        case _, _:
            basis = make_eigenbasis_asym(c_l, kernel_basis, koopman_eigen)
    return basis


def slice_eigen(
    eigen: KoopmanEigen[C, Cs, Css],
    which_eigs: int | tuple[int, int] | list[int] | None = None,
) -> KoopmanEigen[C, Cs, Css]:
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


# TODO: Consider automating the process of building these data-driven wrappers
# using a decorator.
def make_data_driven_eigenbasis[
    Data: PyTree,
    D: DTypeLike,
    X: Array,
    L: int,
](
    pars: KoopmanPars,
    c_l: L2VectorAlgebra[tuple[L], D],
    impl_kernel_basis: Callable[
        [Data, KernelEigen[K, Ks, V, Vs]],
        KernelEigenbasis[X, K, V, Ks, int | Array],
    ],
    which_eigs: int | tuple[int, int] | list[int] | None = None,
) -> Callable[
    [Data, KernelEigen[R, Rs, V, Vs], KoopmanEigen[C, Cs, Css]],
    KoopmanEigenbasis[X, K, V, Ks, int | Array],
]:
    """Make data-driven Koopman eigenbasis builder."""

    def _make_eigenbasis(
        data: Data,
        kernel_eigen: KernelEigen[R, Rs, V, Vs],
        koopman_eigen: KoopmanEigen[C, Cs, Css],
    ) -> KoopmanEigenbasis[X, K, V, Ks, int | Array]:
        kernel_basis = impl_kernel_basis(data, kernel_eigen)
        _koopman_eigen = slice_eigen(koopman_eigen, which_eigs)
        return make_eigenbasis(pars, c_l, kernel_basis, _koopman_eigen)

    return _make_eigenbasis


def make_koopman_analysis_operator[
    Data: PyTree,
    X: Array,
](
    impl_basis: Callable[
        [Data, KernelEigen[R, Rs, V, Vs], KoopmanEigen[C, Cs, Css]],
        KoopmanEigenbasis[X, K, V, Ks, int | Array],
    ],
    which_samples: Optional[tuple[int, int]] = None,
    jit: bool = True,
) -> Callable[
    [Data, Rs, KernelEigen[R, Rs, V, Vs], KoopmanEigen[C, Cs, Css]], Cs
]:
    """Make analysis operator for Koopman forecast."""

    def anal(
        data: Data,
        response: Rs,
        kernel_eigen: KernelEigen[K, Ks, V, Vs],
        koopman_eigen: KoopmanEigen[C, Cs, Css],
    ) -> Cs:
        if which_samples is not None:
            i0 = which_samples[0]
            i1 = which_samples[1]
        else:
            i0 = 0
            i1 = len(response)
        basis = impl_basis(data, kernel_eigen, koopman_eigen)
        return basis.anal(response[i0:i1])

    if jit:
        return jax.jit(anal)
    return anal


def make_koopman_prediction_function[
    Data: PyTree,
    D: DTypeLike,
    X: Array,
    Ntst: Shape,
](
    impl_basis: Callable[
        [Data, KernelEigen[R, Rs, V, Vs], KoopmanEigen[C, Cs, Css]],
        KoopmanEigenbasis[X, K, V, Ks, int | Array],
    ],
    impl_l2_tst: Callable[[Data], L2FnAlgebra[Ntst, D, X, K]],
    jit: bool = True,
) -> Callable[
    [Data, KernelEigen[K, Ks, V, Vs], KoopmanEigen[C, Cs, Css], Cs, Rs, Data],
    R,
]:
    """Make prediction function for Koopman forecast."""

    def predict(
        data: Data,
        kernel_eigen: KernelEigen[K, Ks, V, Vs],
        koopman_eigen: KoopmanEigen[C, Cs, Css],
        coeffs: Cs,
        ts: Rs,
        test_data: Data,
    ) -> Rs:
        basis = impl_basis(data, kernel_eigen, koopman_eigen)
        l2x_tst = impl_l2_tst(test_data)

        @partial(vmap, in_axes=(None, 0, None))
        def _predict(cs: Cs, t: R, x: X) -> R:
            phases = jnp.exp(basis.gen_spec * t)
            return basis.fn_synth(phases * cs)(x)

        return l2x_tst.incl(partial(_predict, coeffs, ts))

    if jit:
        return jax.jit(predict)
    return predict


# TODO: Try moving this to nlsa.koopman by abstracting over L2FnAlgebra (using
# the already defined protocol from alg, and making KernelEigen, KoopmanEigen
# protocols.
def compute_koopman_preds[
    Data: PyTree,
    D: DTypeLike,
    X: Array,
    N: Shape,
    L: int,
    Ntst: Shape,
](
    pars: tuple[KernelPars, KoopmanPars],
    c_l: L2VectorAlgebra[tuple[L], D],
    impl_l2: Callable[[Data], L2FnAlgebra[N, D, X, R]],
    train_data: Data,
    kernel: Callable[[X, X], R] | Callable[[Data, X, X], R],
    kernel_eigen: KernelEigen[R, Rs, V, Vs],
    koopman_eigen: KoopmanEigen[C, Cs, Css],
    coeffs: Rs,
    impl_l2_tst: Callable[[Data], L2FnAlgebra[Ntst, D, X, R]],
    test_data: Data,
    num_steps: int,
    dt: float,
    which_eigs: int | tuple[int, int] | list[int] | None = None,
    jit: bool = True,
) -> Array:
    """Compute KAF predictions."""
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
    predict = make_koopman_prediction_function(
        impl_koopman_basis, impl_l2_tst, jit
    )
    ts = jnp.arange(num_steps + 1) * dt
    return predict(
        train_data, kernel_eigen, koopman_eigen, coeffs, ts, test_data
    )


def plot_generator_spectrum(
    koopman_eigen: KoopmanEigen[C, Cs, Css],
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
        num_eigs_plt = len(koopman_eigen.gen_evals)
    im = ax.scatter(
        koopman_eigen.engys[:num_eigs_plt],
        koopman_eigen.gen_evals[:num_eigs_plt].imag * frequency_scaling,
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

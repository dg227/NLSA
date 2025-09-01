"""Provide generic functions and classes for kernel computations."""
import nlsa.abstract_algebra as alg
import nlsa.function_algebra as fun
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from nlsa.scalars import AsAlgebra, AsBimodule
from nlsa.function_algebra import FunctionAlgebra, BivariateFunctionModule
from nlsa.utils import swap_args
from typing import Literal, Optional, final

type F[*Xs, Y] = Callable[[*Xs], Y]


@final
@dataclass(frozen=True)
class KernelEigenbasis[X, K, V, Ks, I](
        alg.ImplementsDimensionedL2FnFrame[X, K, V, Ks, I]):
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


def make_bandwidth_rbf[K](impl: alg.ImplementsScalarField[K],
                          bandwidth: K, shape_func: F[K, K]) -> F[K, K]:
    """Make bandwidth-parameterized radial basis function."""
    neg_concentration = impl.neg(impl.inv(impl.mul(bandwidth, bandwidth)))

    def rbf(s: K) -> K:
        return shape_func(impl.mul(neg_concentration, s))

    return rbf


def make_kernel_family[X, K](impl: alg.ImplementsScalarField[K],
                             shape_func: F[K, K], sqdist: F[X, X, K]) \
        -> Callable[[K], F[X, X, K]]:
    """Make bandwdith-parameterized kernel family."""
    make_shape_func: Callable[[K], F[K, K]] = partial(
        make_bandwidth_rbf, impl, shape_func=shape_func)

    def kernel_family(epsilon: K, /) -> Callable[[X, X], K]:
        return fun.compose(make_shape_func(epsilon), sqdist)

    return kernel_family


def make_integral_operator[X, V, K](
        impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K],
        k: Callable[[X, X], K], /) -> Callable[[V], F[X, K]]:
    """Make integral operator from kernel function."""
    def k_op(v: V, /) -> F[X, K]:
        def g(x: X, /) -> K:
            kx = partial(k, x)
            gx = impl.integrate(impl.mul(impl.incl(kx), v))
            return gx
        return g
    return k_op


def left_normalize[X, V, K](impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K],
                            k: Callable[[X, X], K], /,
                            unit: Optional[V] = None) -> Callable[[X, X], K]:
    """Perform left normalization of kernel function."""
    func: BivariateFunctionModule[X, X, K, K] \
        = BivariateFunctionModule(codomain=AsBimodule(impl.scl))
    k_op = make_integral_operator(impl, k)
    if unit is None:
        unit = impl.unit()
    lfun = k_op(unit)
    k_l = func.ldiv(lfun, k)
    return k_l


def right_normalize[X, V, K](impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K],
                             k: Callable[[X, X], K], /,
                             unit: Optional[V] = None) -> Callable[[X, X], K]:
    """Perform right normalization of kernel function."""
    func: BivariateFunctionModule[X, X, K, K] \
        = BivariateFunctionModule(codomain=AsBimodule(impl.scl))
    k_op = make_integral_operator(impl, k)
    if unit is None:
        unit = impl.unit()
    rfun = k_op(unit)
    k_r = func.rdiv(k, rfun)
    return k_r


def sym_normalize[X, V, K](impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K],
                           k: Callable[[X, X], K], /,
                           unit: Optional[V] = None) -> Callable[[X, X], K]:
    """Perform symmetric normalization of kernel function."""
    func: BivariateFunctionModule[X, X, K, K] \
        = BivariateFunctionModule(codomain=AsBimodule(impl.scl))
    k_op = make_integral_operator(impl, k)
    if unit is None:
        unit = impl.unit()
    sfun = k_op(unit)
    k_r = func.rdiv(k, sfun)
    k_s = func.ldiv(sfun, k_r)
    return k_s


def right_sqrt_normalize[X, V, K](
        impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K],
        k: Callable[[X, X], K], /,
        unit: Optional[V] = None) -> Callable[[X, X], K]:
    """Perform right square root normalization of kernel function."""
    func: FunctionAlgebra[X, K, K] \
        = FunctionAlgebra(codomain=AsAlgebra(impl.scl))
    func2: BivariateFunctionModule[X, X, K, K] \
        = BivariateFunctionModule(codomain=AsBimodule(impl.scl))
    k_op = make_integral_operator(impl, k)
    if unit is None:
        unit = impl.unit()
    rfun = func.sqrt(k_op(unit))
    k_r = func2.rdiv(k, rfun)
    return k_r


def sym_sqrt_normalize[X, V, K](
        impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K],
        k: Callable[[X, X], K], /,
        unit: Optional[V] = None) -> Callable[[X, X], K]:
    """Perform symmetric square root normalization of kernel function."""
    func: FunctionAlgebra[X, K, K] \
        = FunctionAlgebra(codomain=AsAlgebra(impl.scl))
    func2: BivariateFunctionModule[X, X, K, K] \
        = BivariateFunctionModule(codomain=AsBimodule(impl.scl))
    k_op = make_integral_operator(impl, k)
    if unit is None:
        unit = impl.unit()
    sfun = func.sqrt(k_op(unit))
    k_r = func2.rdiv(k, sfun)
    k_s = func2.ldiv(sfun, k_r)
    return k_s


def dm_normalize[X, V, K](impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K],
                          k: Callable[[X, X], K], /,
                          alpha: Literal['0', '0.5', '1'],
                          unit: Optional[V] = None) -> Callable[[X, X], K]:
    """Perform Diffusion Maps kernel normalization."""
    if unit is None:
        unit = impl.unit()

    match alpha:
        case '0':
            k_r = k
        case '0.5':
            k_r = sym_sqrt_normalize(impl, k, unit)
        case '1':
            k_r = sym_normalize(impl, k, unit)
    k_dm = left_normalize(impl, k_r, unit)
    return k_dm


def dmsym_normalize[X, V, K](impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K],
                             k: Callable[[X, X], K], /,
                             alpha: Literal['0', '0.5', '1'],
                             unit: Optional[V] = None) -> Callable[[X, X], K]:
    """Perform Diffusion Maps symmetric kernel normalization."""
    match alpha:
        case '0':
            k_r = k
        case '0.5':
            k_r = sym_sqrt_normalize(impl, k, unit)
        case '1':
            k_r = sym_normalize(impl, k, unit)
    k_dm = sym_sqrt_normalize(impl, k_r, unit)
    return k_dm


def from_dmsym[Vs, V, K](impl: alg.ImplementsLModule[Vs, K, V], v0: V,
                         vs: Vs, /) -> Vs:
    """Normalize eigenvectors from symmetric diffusion maps normalization to
    obtain eigenvectors with respect to Markov normalization.
    """
    return impl.ldiv(v0, vs)


def bs_normalize[X, V, K](impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K],
                          k: Callable[[X, X], K], /,
                          unit: Optional[V] = None) -> Callable[[X, X], K]:
    """Perform bistochastic kernel normalization (left part)."""
    func: FunctionAlgebra[X, K, K] \
        = FunctionAlgebra(codomain=AsAlgebra(impl.scl))
    func2: BivariateFunctionModule[X, X, K, K] \
        = BivariateFunctionModule(codomain=AsBimodule(impl.scl))
    k_op = make_integral_operator(impl, k)
    if unit is None:
        unit = impl.unit()
    k_op = make_integral_operator(impl, k)
    d = k_op(unit)
    k_r = func2.rdiv(k, d)
    k_r_op = make_integral_operator(impl, k_r)
    q: F[X, K] = k_r_op(unit)
    k_q = func2.rdiv(k, func.sqrt(q))
    k_bs = func2.ldiv(d, k_q)
    return k_bs


def bssym_normalize[X, V, K](impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K],
                             k: Callable[[X, X], K], /,
                             unit: Optional[V] = None) \
        -> Callable[[X, X], K]:
    """Perform bistochastic kernel normalization (symmetrized)."""
    k_bs = bs_normalize(impl, k, unit)

    def k_sym(x: X, y: X, /) -> K:
        u = impl.incl(partial(k_bs, x))
        v = impl.incl(partial(k_bs, y))
        return impl.integrate(impl.mul(u, v))

    return k_sym


def compose[X, V, K](impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K],
                     k1: Callable[[X, X], K], k2: Callable[[X, X], K], /) \
        -> Callable[[X, X], K]:
    """Compose two kernels."""
    k2_transp = swap_args(k2)

    def k3(x: X, y: X, /) -> K:
        v1 = impl.incl(partial(k1, x))
        v2 = impl.incl(partial(k2_transp, y))
        return impl.integrate(impl.mul(v1, v2))

    return k3


def make_mercer_kernel[X, V, K](impl: alg.ImplementsInnerProductAlgebra[V, K],
                                psi_l: F[X, V], psi_r: F[X, V], /) \
       -> Callable[[X, X], K]:
    """Make Mercer kernel from 'left' and 'right' feature vectors"""
    def k(x: X, y: X, /) -> K:
        return impl.innerp(psi_l(x), psi_r(y))
    return k


def riemannian_vol[X, V, K](impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K],
                            kernel: Callable[[X, X], K],
                            dim: K, t_heat: K, fourpi: K) -> K:
    """Compute Riemannian volume using heat trace formula."""
    scl = impl.scl
    h: F[X, K] = fun.diag(kernel)
    a = scl.power(scl.sqrt(scl.mul(fourpi, t_heat)), dim)
    vol = scl.mul(a, impl.integrate(impl.incl(h)))
    return vol


def make_bandwidth_function[X, V, K](
        impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K],
        k: Callable[[X, X], K], /, dim: K, vol: K,
        unit: Optional[V] = None) -> Callable[[X], K]:
    """Make bandwidth function for variable-bandwidth kernel."""
    if unit is None:
        unit = impl.unit()
    scl = impl.scl
    func: FunctionAlgebra[X, K, K] = FunctionAlgebra(codomain=AsAlgebra(scl))
    w = sym_normalize(impl, k, unit)
    w_op = make_integral_operator(impl, w)
    d = w_op(unit)
    d_bar = impl.integrate(impl.incl(d))
    c = scl.div(vol, d_bar)
    b = func.power(func.smul(c, d), scl.inv(dim))
    return b


def make_scaled_sqdist[X, K](impl: alg.ImplementsScalarField[K],
                             d2: Callable[[X, X], K], b: Callable[[X], K], /) \
        -> Callable[[X, X], K]:
    """Make scaled square distance function from bandwidth function."""
    func: FunctionAlgebra[X, X, K, K] \
        = FunctionAlgebra(codomain=AsAlgebra(impl))
    tensorp = fun.make_bivariate_tensor_product(impl)
    d2_scl = func.div(d2, tensorp(b, b))
    return d2_scl


def make_tuning_objective[X, V, K](
        impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K],
        k_func: Callable[[K], Callable[[X, X], K]], /,
        grad: Callable[[F[K, K]], F[K, K]], exp: Callable[[K], K],
        log: Callable[[K], K], unit: Optional[V] = None) -> Callable[[K], K]:
    """Make objective function for kernel tuning."""
    if unit is None:
        unit = impl.unit()

    def log_k_sum(log_eps: K) -> K:
        epsilon = exp(log_eps)
        k_op = make_integral_operator(impl, k_func(epsilon))
        s = impl.integrate(impl.incl(k_op(unit)))
        return log(s)

    return grad(log_k_sum)


def make_resolvent_compactification_kernels[X, TX, V, K](
        impl: alg.ImplementsMeasureFnAlgebra[X, K, V, K], v: Callable[[X], TX],
        z: K, k: Callable[[X, X], K], /,
        jvp: Callable[[F[X, K], X, TX], K]) -> tuple[F[X, X, K], F[X, X, K],
                                                     F[X, X, K]]:
    """Make kernels for resolvent compactification scheme."""
    func2: BivariateFunctionModule[X, X, K, K] \
        = BivariateFunctionModule(codomain=AsBimodule(impl.scl))

    @swap_args
    def v_grad_k(x: X, y: X, /) -> K:
        return jvp(partial(swap_args(k), x), y, v(y))

    zk_vk = func2.sub(func2.smul(z, k), v_grad_k)
    kvk = compose(impl, swap_args(k), v_grad_k)
    qz_i = swap_args(zk_vk)
    qz_j = compose(impl, kvk, swap_args(zk_vk))
    gz = compose(impl, zk_vk, swap_args(zk_vk))
    return qz_i, qz_j, gz

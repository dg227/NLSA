"""Implements various aspects of reproducing kernel Hilbert spaces (RKHSs)."""

# mypy: ignore-errors

import nlsa.function_algebra as fun
import nlsa.matrix_algebra as mat
import numpy as np
from functools import partial
from nptyping import Complex, Double, NDArray, Shape
from nlsa.abstract_algebra import ImplementsAlgdiv, ImplementsAlgldiv,\
        ImplementsPower, ImplementsSqrt, ImplementsStarAlgebra,\
        algdiv_by, exponentiate_by, l2_innerp
from scipy.spatial.distance import cdist
from typing import Callable, Literal, Optional, Protocol, Tuple, TypeVar,\
        runtime_checkable

Rs = NDArray[Shape['1, ...'], Double]
RN = NDArray[Literal['N'], Double]
CRN = TypeVar('CRN',
              NDArray[Shape['N'], Double], NDArray[Shape['N'], Complex])
CRNs = TypeVar('CRNs',
               NDArray[Shape['N, ...'], Double],
               NDArray[Shape['N, ...'], Complex])
K = TypeVar('K', Double, Complex)
X = TypeVar('X')
Y = TypeVar('Y')
Z = TypeVar('Z')
Z_contra = TypeVar('Z_contra', contravariant=True)
X_co = TypeVar('X_co', covariant=True)
V = TypeVar('V')
W = TypeVar('W')
W_contra = TypeVar('W_contra', contravariant=True)
M = NDArray[Shape['*, ...'], Double]
LN = NDArray[Shape['L, N'], Double]
MN = NDArray[Shape['M, N'], Double]
NN = NDArray[Shape['N, N'], Double]


def dist2(x: LN, y: LN) -> NN:
    """Squared Euclidean pairwise distance"""
    d2: NN = cdist(np.atleast_2d(x), np.atleast_2d(y), 'sqeuclidean')
    return d2


def energy(w: RN) -> Callable[[CRNs], Rs]:
    """Energy function associated with weights.

    :w: Vector of weights.
    :returns: Energy function

    """
    def engy(c: CRNs) -> Rs:
        e: Rs = np.sum(np.abs(c) ** 2 * w[:, np.newaxis], axis=0)
        return e
    return engy


def eval_sampling_measure(x: X, f: Callable[[X], CRNs]) -> CRN:
    """Evaluate sampling measure associated with dataset on function.

    :x: Dataset.
    :f: Function to integrate.
    :returns: Integral of function against sampling measure.

    """
    fx = f(x)
    y: CRN = np.average(fx, axis=tuple(range(1, fx.ndim)))
    return y


# TODO: Implementation of this function requires indexable objects and objects
# that support transpose (T). Consider moving kernel_opereator to the vec
# module or overloading this function.
def kernel_operator(impl: ImplementsStarAlgebra[V, W],
                    incl: Callable[[Callable[[Y], V]], V],
                    mu: Callable[[V], W],
                    k: Callable[[X, Y], V]) \
                            -> Callable[[V], Callable[[X], W]]:
    """Build kernel integral operator from kernel and measure.

    :impl: Object that implements star algebra operations.
    :incl: Inclusion map on functions.
    :mu: Dual vector (measure).
    :k: Kernel function.
    :returns: Operator from vectors to functions.

    """
    def k_op(f: V) -> Callable[[X], W]:
        def g(x: X) -> W:
            kx = incl(partial(k, x))
            gx = l2_innerp(impl, mu, kx, f)
            return gx
        return g
    return k_op


def rnormalize(impl: ImplementsAlgdiv[V], k: Callable[[X, Y], V],
               r: Callable[[Y], V]) \
        -> Callable[[X, Y], V]:
    """Right kernel normalization.

    :impl: Object that implements algebraic division.
    :k: Kernel function.
    :r: Normalization function.
    :returns: Right-normalized kernel.

    """
    def k_normalized(x: X, y: Y) -> V:
        v = impl.algdiv(k(x, y), r(y))
        return v
    return k_normalized


def lnormalize(impl: ImplementsAlgldiv[V], k: Callable[[X, Y], V],
               l: Callable[[X], V])\
                       -> Callable[[X, Y], V]:
    """Left kernel normalization.

    :impl: Object that implements left algebraic division.
    :k: Kernel function.
    :l: Normalization function.
    :returns: Left-normalized kernel.

    """
    def k_normalized(x: X, y: Y) -> V:
        v = impl.algldiv(l(x), k(x, y))
        return v
    return k_normalized


class ImplementsAlgdivAndAlgldiv(ImplementsAlgdiv[V], ImplementsAlgldiv[V],
                                 Protocol[V]):
    ...


def snormalize(impl: ImplementsAlgdivAndAlgldiv[V], k: Callable[[X, X], V],
               s: Callable[[X], V])\
                       -> Callable[[X, X], V]:
    """Symmetric kernel normalization.

    :impl: Object that implements algebraic division and left algebraic
    division.
    :k: Kernel function.
    :s: Normalization function.
    :returns: Symmetrically normalized kernel.

    """
    ks = rnormalize(impl, k, s)
    kss = lnormalize(impl, ks, s)
    return kss


@runtime_checkable
class ImplementsDMProtocols(ImplementsAlgdiv[V], ImplementsAlgldiv[V],
                            ImplementsStarAlgebra[V, W_contra],
                            Protocol[V, W_contra]):
    ...


def dm_normalize(impl: ImplementsDMProtocols[V, Z],
                 alpha: Z,
                 unit: V,
                 incl: Callable[[Callable[[X], V]], V],
                 mu: Callable[[V], V],
                 k: Callable[[X, X], V]) \
        -> Callable[[X, X], V]:
    """Diffusion maps kernel normalization.

    :impl: Object that implements the required methods to perfom diffusion maps
    normalization.
    :alpha: Diffusion maps normalization parameter.
    :unit: Unit element of type V.
    :mu: Dual vector (measure).
    :k: Kernel.
    :returns: Diffusion maps normalized kernel.

    """
    exp_alpha = fun.fmap(exponentiate_by(impl, alpha))
    k_op = kernel_operator(impl, incl, mu, k)
    rfun = k_op(unit)
    kr = rnormalize(impl, k, exp_alpha(rfun))
    kr_op = kernel_operator(impl, incl, mu, kr)
    lfun = kr_op(unit)
    kdm = lnormalize(impl, kr, lfun)
    return kdm


def dmsym_normalize(impl: ImplementsDMProtocols[V, Z],
                    alpha: Z,
                    unit: V,
                    incl: Callable[[Callable[[X], V]], V],
                    mu: Callable[[V], V],
                    k: Callable[[X, X], V])\
                            -> Callable[[X, X], V]:
    """Diffusion maps symmetric kernel normalization.

    :impl: Object that implements the required methods to perfom diffusion maps
    normalization.
    :alpha: Diffusion maps normalization parameter.
    :unit: Unit element of type V.
    :mu: Dual vector (measure).
    :k: Kernel.
    :returns: Symmetric diffusion maps normalized kernel.

    """
    sqrt = fun.fmap(impl.sqrt)
    k_op = kernel_operator(impl, incl, mu, k)
    rfun = k_op(unit)
    match alpha:
        case 0:
            krr_op = k_op
        case _:
            exp_alpha = fun.fmap(exponentiate_by(impl, alpha))
            rfun_a = exp_alpha(rfun)
            krr = snormalize(impl, k, rfun_a)
            krr_op = kernel_operator(impl, incl, mu, krr)
    lfun = sqrt(krr_op(unit))
    kdm = snormalize(impl, krr, lfun)
    return kdm


def nystrom_basis(impl_v: ImplementsAlgdiv[W],
                  k_op: Callable[[W], Callable[[X], W]], lam: W, phi: W)\
                          -> Callable[[X], W]:

    varphi = fun.compose(algdiv_by(impl_v, lam), k_op(phi))
    return varphi


def gaussian(epsilon: float, d2: M) -> M:
    """Gaussian shape function.

    :epsilon: Bandwidth parameter.
    :d2: Squared pairwise distances
    :returns: Values of Gaussian shape function.

    """
    y: M = np.exp(-d2 / epsilon ** 2)
    return y


def dm_eigen(s: NN, n: Optional[int] = None, weighted: bool = True,
             solver: Literal['eig', 'eigs'] = 'eigs')\
                     -> Tuple[RN, NN, RN]:
    """Sorted eigenvalues and eigenvectors from symmetric diffusion maps kernel
    matrix.

    :s: Symmetric kernel matrix.
    :n: Number of eigenvalues/eigenvectors to compute.
    :weighted: Weigh the eigenvectors by the inner product weights.
    :solver: Eigenvalue solver.
    :returns: Tuple of eigenvalues sorted in decreasing order, the
    corresponding eigenvectors, and inner product weights.

    """
    match solver:
        case 'eig':
            wv = mat.eig_sorted(s, n=n, which='LR')
        case 'eigs':
            wv = mat.eigs_sorted(s, n=n, which='LA', hermitian=True)
    w: RN = wv[0]
    v: NN = wv[1]
    mu: RN = v[:, 0] ** 2
    if v[0, 0] < 0:
        v[:, 0] = -v[:, 0]
    if weighted is True:
        v = v / np.sqrt(mu[:, np.newaxis])
    return w, v, mu

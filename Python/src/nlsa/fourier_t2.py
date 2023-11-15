"""
This module implements analysis/synthesis operators and related methods for L2
spaces and reproducing kernel Hilbert algebras (RKHAs) on the 2-torus T2.

"""
import nlsa.fourier_s1 as f1
import numpy as np
from nptyping import Complex, Double, Int, NDArray, Shape
from scipy.special import iv
from typing import Callable, Literal, Tuple, TypeVar

K = NDArray[Shape["2, N"], Int]
X = NDArray[Shape["*, 2"], Double]
Y = NDArray[Shape["*"], Double]
XK = NDArray[Shape["*, N"], Complex]
V = TypeVar("V", NDArray[Shape["N"], Double], NDArray[Shape["N"], Complex])
VR = NDArray[Literal['N'], Double]
VC = NDArray[Literal['N'], Complex]
M = TypeVar("M",
            NDArray[Shape["N, N"], Double],
            NDArray[Shape["N, N"], Complex])


def dual_group(k_max: Tuple[int, int]) -> K:
    """Elements of the dual group Z2 (Fourier wavenumbers) of T2 up to maximal
    wavenumbers.

    :k_max: 2-tuple of ints specifying maximal wavenumbers.
    :returns: Array k of shape [2, n], where
    n = (2 * k_max[0] + 1) * (2 * k_max[1] + 1) such that each column [k1, k2]
    is an element of Z2 with abs(k1) <= k_max[0] and abs(k2) <= k_max[1]. The
    "fast" index in the ordering of the columns of k is k1.

    """
    k1 = f1.dual_group(k_max[0])
    k2 = f1.dual_group(k_max[1])
    kk1, kk2 = np.meshgrid(k1, k2)
    k = np.vstack((kk1.ravel(), kk2.ravel()))
    return k


def dual_size(k_max: Tuple[int, int]):
    """Number of elements in the dual group returned by the dual_group
    function.

    :k_max: 2-tuple of ints specifying maximal wavenumbers.
    :returns: Number of elements in dual_group(k_max).

    """
    n = f1.dual_size(k_max[0]) * f1.dual_size(k_max[1])
    return n


def fourier_basis(k: K) -> Callable[[X], XK]:
    """Fourier basis functions on the 2-torus up to maximal wavenumbers.

    :k: Array of Fourier wavenumbers.
    :returns: Function phi that takes as inputs two-dimensional vectors or
    arrays of shape [..., 2], representing points on the circle, and returns an
    array of complex numbers that contains the values of the corresponding
    Fourier functions.

    """
    def phi(x: X) -> XK:
        y = np.exp(1j * x @ k)
        # y = np.exp(1j * np.einsum('...i,ij->...j', x, k))
        return y
    return phi


def rkha_weights(p: float, tau: float) -> Callable[[K], VR]:
    """Weight function for RKHA on the circle.

    :p: RKHA exponent parameter. p should lie in the interval (0, 1).
    :tau: RKHA "heat flow" parameter. tau should be strictly positive
    :returns: Function w that computes the RKHA weights on the dual group Z2 of
    T2.

    """
    w1 = f1.rkha_weights(p, tau)

    def w(k: K) -> VR:
        y: VR = np.prod(w1(k), axis=0)
        return y
    return w


def make_mult_op(k: Tuple[int, int]) -> Callable[[V], M]:
    """Make multiplication operator in the Fourier basis of T2.

    :k: Maximal Fourier wavenumber.
    :returns: Function that constructs a multiplication operator from a vector
    of Fourier expansion coefficients.

    """
    n = dual_size(k)
    js = dual_group(k)
    ks = js[:, :, np.newaxis] - js[:, np.newaxis, :]
    idxs = np.ravel_multi_index((ks[1, ...] + 2 * k[1], ks[0, ...] + 2 * k[0]),
                                (4 * k[1] + 1, 4 * k[0] + 1))

    def op(v: V) -> M:
        """Multiplication operator function.

        :v: Input vector of shape [(2 * k[0] + 1 ) * (2 * k[1] + 1),]
        :returns: Multiplication operator matrix m of shape [n + 1, n + 1],
        where n = (k[0] + 1) * k[1] + k[0] * (k[1] + 1).

        """
        m: M = np.reshape(v[idxs], (n, n))
        return m
    return op


def von_mises(kappa: Tuple[float, float], loc: Tuple[float, float] = (0, 0)) \
        -> Callable[[X], Y]:
    """Von Mises probability density on T2.

    :kappa: 2-tuple of Von Mises concentration parameters.
    :loc: 2-tuple of Von Mises location (mean) parameters.
    :returns: Function f_hat that computes the Fourier coefficients of the von
    Mises density.

    """
    vm1 = f1.von_mises(kappa[0], loc[0])
    vm2 = f1.von_mises(kappa[1], loc[1])

    def f_hat(x: X) -> Y:
        y = vm1(x[..., 0]) * vm2(x[..., 1])
        return y
    return f_hat


def von_mises_fourier(kappa: Tuple[float, float],
                      loc: Tuple[float, float] = (0, 0)) -> Callable[[K], V]:
    """Returns a function that computes the expansion coefficients of the von
    Mises density function with respect to the Fourier basis on T2.

    :kappa: 2-tuple of Von Mises concentration parameters.
    :loc: 2-tuple of Von Mises location (mean) parameters.
    :returns: Function f_hat that computes the Fourier coefficients of the von
    Mises density.

    """
    vm1 = f1.von_mises_fourier(kappa[0], loc[0])
    vm2 = f1.von_mises_fourier(kappa[1], loc[1])

    def f_hat(k: K) -> V:
        y = vm1(k[0, :]) * vm2(k[1, :])
        return y
    return f_hat


def von_mises_feature_map(kappa: Tuple[float, float], k_max: Tuple[int, int]) \
        -> Callable[[X], XK]:
    """Von Mises feature map on T2.

    :kappa: 2-tuple of Von Mises concentration parameters.
    :k_max: 2-tuple of maximal wavenumbers.
    :returns: Vector-valued function xi (feature map).

    """
    k = dual_group(k_max)
    phi = fourier_basis(k)

    def xi(x: X) -> XK:
        y: XK = iv(np.abs(k[0, :]), kappa[0] / 2) / iv(0, kappa[0]) \
                * iv(np.abs(k[1, :]), kappa[1] / 2) / iv(0, kappa[1]) \
                * np.conj(phi(x))
        return y
    return xi


def rotation_koopman_eigs(a: Tuple[float, float]) -> Callable[[K], V]:
    """Returns a function that computes the eigenvalues of the Koopman operator
    associated with a discrete-time rotation on the 2-torus.

    :kappa: 2-tuple of rotation angles.
    :returns: Function g that computes the eigenvalues of the Koopman operator.

    """
    e1 = f1.rotation_koopman_eigs(a[0])
    e2 = f1.rotation_koopman_eigs(a[1])

    def g(k: K) -> V:
        y = e1(k[0, :]) * e2(k[1, :])
        return y
    return g


def rotation_generator_eigs(a: Tuple[float, float]) -> Callable[[K], VC]:
    """Returns a function that computes the eigenvalues of the Koopman
    generator associated with a continuous-time rotation on the 2-torus.

    :kappa: 2-tuple of rotation angles.
    :returns: Function g that computes the eigenvalues of the generator.

    """
    e1 = f1.rotation_generator_eigs(a[0])
    e2 = f1.rotation_generator_eigs(a[1])

    def g(k: K) -> VC:
        y = e1(k[0, :]) + e2(k[1, :])
        return y
    return g


def rotation_generator(alpha: Tuple[float, float], k: int) -> M:
    """Matrix representation of the generator of a rotation on the 2-torus with
    respect to the Fourier basis.

    alpha: Frequency parameters.
    k: Spectral resolution parameter (max. Fourier wavenumber).

    return: n^2 by n^2 generator matrix v with n = 2 * k + 1.
    """

    spec = rotation_generator_eigs(alpha)
    v: M = np.diag(spec(dual_group((k, k))))
    return v

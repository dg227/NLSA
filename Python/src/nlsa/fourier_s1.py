"""
This module implements analysis/synthesis operators and related methods for L2
spaces and reproducing kernel Hilbert algebras (RKHAs) on the circle S1.
"""

import nlsa.function_algebra as fun
import numpy as np
from nptyping import Complex, Double, Int, NDArray, Shape
from scipy.linalg import toeplitz
from scipy.special import iv
from scipy.stats import vonmises
from typing import Callable, Literal, TypeVar

# Typevar declarations
# K represents vectors of elements of the dual group (Fourier wavenumbers)
K = NDArray[Literal["N"], Int]
X = NDArray[Shape["*, ..."], Double]
XK = NDArray[Shape["*, N"], Complex]
V = TypeVar("V", NDArray[Shape["N"], Double], NDArray[Shape["N"], Complex])
VR = NDArray[Literal["N"], Double]
VC = NDArray[Literal["N"], Complex]
M = TypeVar("M", NDArray[Shape["L, L"], Double],
            NDArray[Shape["L, L"], Complex])


def dual_group(k_max: int) -> K:
    """Compute elements of the dual group Z (Fourier wavenumbers) of S1 up to
    a maximal wavenumber.

    :k_max: Maximal wavenumber.
    :returns: Integer array of shape [2 * k_max + 1] containing the wavenumbers
    in the range -k to k.

    """
    k = np.arange(-k_max, k_max + 1)
    return k


def dual_group0(k_max: int) -> K:
    """Compute elements of the dual group Z (Fourier wavenumbers) of S1 up to
    a maximal wavenumber, excluding the zero wavenumber.

    :k_max: Maximal wavenumber.
    :returns: Integer array of shape [2 * k_max,] containing the wavenumbers
    in the range -k to k excluding 0.

    """
    k: K = np.concatenate((np.arange(-k_max, 0), np.arange(1, k_max + 1)))
    return k


def dual_size(k_max: int):
    """Number of elements in the dual group returned by the dual_group
    function.

    :k_max: Maximal wavenumber.
    :returns: Number of elements in dual_group(k_max).

    """
    n = 2 * k_max + 1
    return n


def rkha_weights(p: float, tau: float) -> Callable[[K], VR]:
    """Weight function for RKHA on the circle.

    :p: RKHA exponent parameter. p should lie in the interval (0, 1).
    :tau: RKHA "heat flow" parameter. tau should be strictly positive
    :returns: Function w that computes the RKHA weights on the dual group Z of
    S1.

    """
    def w(k: K) -> VR:
        y: VR = np.exp(tau * np.abs(k) ** p)
        return y
    return w


def fourier_basis(k: K) -> Callable[[X], XK]:
    """Returns Fourier basis functions on the circle up to a maximal
    wavenumber.

    :k: Vector of wavenumbers.
    :returns: A function phi that takes as inputs floats or arrays of floats,
    representing points on the circle, and returns an array of complex numbers
    that contains the values of the corresponding Fourier functions.

    """
    def phi(x: X) -> XK:
        y = np.exp(1j * k * x[..., np.newaxis])
        return y
    return phi


def rkha_basis(p: float, tau: float, k: K) -> Callable[[X], object]:
    """Returns RKHA basis functions on the circle up to a maximal wavenumber.

    :p: RKHA exponent parameter. p should lie in the interval (0, 1).
    :tau: RKHA "heat flow" parameter. tau should be strictly positive
    :k: Vector of wavenumbers.
    :returns: Function psi that takes as inputs floats or arrays of floats,
    representing points on the circle, and returns an array of floats that
    contains the values of the corresponding RKHA basis functions.

    """

    # TODO: Improve typing of this function. The nlsa.function_algebra module
    # does not know that we can pass arrays of points in X as function
    # arguments, which is what we do throughout this module for efficiency.
    w = rkha_weights(p, tau)
    lam = w(k)
    phi = fourier_basis(k)
    psi = fun.mul(lam, phi)
    return psi


# TODO: In addition to (or instead of) representing multiplication operators by
# matrices in the Fourier basis, represent them as convolution operators. The
# matrix representation could then be computed automatically by acting by the
# convolution operators on basis vectors.

def make_mult_op(k: int) -> Callable[[V], M]:
    """Make multiplication operator in the Fourier basis of S1.

    :k: Maximal Fourier wavenumber.
    :returns: Function that constructs a multiplication operator from a vector
    of Fourier expansion coefficients.

    """
    def op(v: V) -> M:
        """Multiplication operator function.

        :v: Vector of shape (2 * 4 + 1,).
        :returns: Toeplitz matrix m of shape (2 * k + 1, 2 * k + 1).

        """
        c = v[2 * k:]
        r = np.flip(v[:2 * k + 1])
        m: M = toeplitz(c, r)
        return m
    return op


def make_mult_op0(k: int):
    """Make multiplication operator in the Fourier basis of S1, excluding the
    excluding the constant Fourier function (zero wavenumber).

    :k: Maximal Fourier wavenumber.
    :returns: Function that constructs a multiplication operator from a vector
    of Fourier expansion coefficients.

    """
    op = make_mult_op(k)
    i = np.concatenate((np.arange(0, k), np.arange(k + 1, 2 * k + 1)))
    j = i[:, np.newaxis]

    def op0(v: V) -> M:
        """Multiplication operator function.

        :v: Vector of shape (2 * 4 + 1,).
        :returns: Toeplitz matrix m of shape (2 * k, 2 * k).

        """
        m = op(v)
        m0: M = m[i, j]
        return m0
    return op0


def von_mises(kappa: float, loc: float = 0) -> Callable[[X], X]:
    """Von Mises probability density on the circle.

    :kappa: Von Mises concentration parameter.
    :loc: Von Mises location (mean) parameter.
    :returns: Von Mises probability density.

    """
    def f(x: X) -> X:
        y: X = 2 * np.pi * vonmises.pdf(x, kappa, loc=loc)
        return y
    return f


def von_mises_fourier(kappa: float, loc: float = 0) -> Callable[[K], V]:
    """Returns a function that computes the expansion coefficients of the von
    Mises density function with respect to the Fourier basis on S1.

    :kappa: Von Mises concentration parameter.
    :loc: Von Mises location (mean) parameter.
    :returns: Function f_hat that computes the Fourier coefficients of the von
    Mises density.

    """
    def f_hat(k: K) -> V:
        y: V = iv(np.abs(k), kappa) * np.exp(-1j * k * loc) / iv(0, kappa)
        return y
    return f_hat


def von_mises_feature_map(kappa: float, k: K) -> Callable[[X], XK]:
    """Von Mises feature map on S1.

    :kappa: Von Mises concentration parameter.
    :k: Fourier wavenumbers.
    :returns: Vector-valued function xi (feature map).

    """
    phi = fourier_basis(k)

    def xi(x: X) -> XK:
        y: XK = iv(np.abs(k), kappa / 2) * np.conj(phi(x)) / iv(0, kappa)
        return y
    return xi


def sqrt_von_mises_fourier(kappa: float, loc: float = 0) -> Callable[[K], V]:
    """Returns a function that computes the expansion coefficients of the
    square root of the Mises density function with respect to the Fourier basis
    on S1.

    :kappa: Von Mises concentration parameter.
    :loc: Von Mises location (mean) parameter.
    :returns: Function f_hat that computes the Fourier coefficients of the
    square root of the von Mises density.

    """
    def f_hat(k: K) -> V:
        y: V = iv(np.abs(k), kappa / 2) * np.exp(-1j * k * loc) / iv(0, kappa)
        return y
    return f_hat


def rotation_koopman_eigs(a: float) -> Callable[[K], V]:
    """Returns a function that computes the eigenvalues of the Koopman operator
    associated with a discrete-time circle rotation.

    :a: Rotation angle.
    :returns: Function g that computes the eigenvalues of the Koopman operator.

    """
    z = np.exp(1j * a)

    def g(k: K) -> V:
        y: V = z ** k
        return y
    return g


def rotation_generator_eigs(a: float) -> Callable[[K], VC]:
    """Returns a function that computes the eigenvalues of the Koopman
    generator associated with a continuous-time circle rotation.

    :a: Rotation frequency.
    :returns: Function g that computes the eigenvalues of the Koopman operator.

    """
    def g(k: K) -> VC:
        y: VC = 1j * a * k
        return y
    return g


def doubling_map_fourier(k_max: int) -> Callable[[V], V]:
    """Returns the representation of the projected Koopman operator associated
    with doubling map in the Fourier basis.

    :k_max: Maximal wavenumber.
    :returns: Projected Koopman operator on Fourier coefficient vectors.

    """
    k = dual_group(k_max)

    def g(y: V) -> V:
        z = np.zeros_like(y)
        z[k % 2 == 0] = y[np.abs(k) <= k_max // 2]
        return z
    return g

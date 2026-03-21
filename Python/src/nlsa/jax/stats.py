# pyright: basic
"""Implement basic statistical functions using JAX arrays."""

import jax.numpy as jnp
import numpy as np
from collections.abc import Callable
from functools import partial
from jax import Array, jit, vmap
from jax.scipy.special import i0
from jax.typing import ArrayLike
from typing import Literal, Optional, TypedDict

type R = Array
type C = Array
type S1 = Array
type Xs = Array
type Xss = Array


class ProbDensity(TypedDict):
    """TypedDict containing probability density function data."""

    densities: Array
    """Array containing probability density values."""

    bin_edges: Array
    """Array containing bin edges used in density estimation."""


class TimeseriesStats(TypedDict):
    """TypedDict containing scalar timeseries statistics."""

    pdf: ProbDensity
    """Probability density function (histogram) data."""

    autocorr: Array
    """Autocorrelation function."""


class MultivariateTimeseriesStats(TypedDict):
    """TypedDict containing multivariate timeseries statistics."""

    pdfs: list[ProbDensity]
    """Probability density function (histogram) data."""

    autocorrs: Array
    """Autocorrelation functions."""


def make_von_mises_density(
    concentration: float, location: float = 0
) -> Callable[[S1], R]:
    """Make Von Mises probability density function on the circle."""

    def f(x: S1, /) -> R:
        y = jnp.exp(concentration * jnp.cos(x - location)) / i0(concentration)
        return y

    return f


def normalized_rmse(xs_true: Xs, xs_pred: Xs) -> R:
    """Compute normalized root mean square error."""
    nmse = jnp.sum((xs_true - xs_pred) ** 2) / jnp.sum(xs_true**2)
    return jnp.sqrt(nmse)


def anomaly_correlation_coefficient(xs_true: Xs, xs_pred: Xs) -> R:
    """Compute anomaly correlation coefficient."""
    anom_true = xs_true - jnp.mean(xs_true)
    anom_pred = xs_pred - jnp.mean(xs_pred)
    sqnorm_true = jnp.sum(anom_true**2)
    sqnorm_pred = jnp.sum(anom_pred**2)
    return jnp.sum(anom_true * anom_pred) / jnp.sqrt(sqnorm_true * sqnorm_pred)


def lagged_cross_correlation(
    xs: Xs,
    ys: Xs,
    lag: int | Array,
    mode: Literal["full", "exact"] = "full",
    num_samples: Optional[int] = None,
) -> C:
    """Compute lagged cross-correlation of two time series."""
    anom_xs = xs - jnp.mean(xs)
    anom_ys = ys - jnp.mean(ys)
    match mode:
        case "full":
            corr = jnp.mean(jnp.conj(anom_xs[:-lag]) * anom_ys[lag:0])
        case "exact":
            assert num_samples is not None
            # corr = jnp.mean(
            #     jnp.conj(anom_xs[:num_samples])
            #     * anom_ys[lag : (lag + num_samples)]
            # )
            anom_ys_lagged = jnp.roll(anom_ys, -lag)
            corr = jnp.mean(
                jnp.conj(anom_xs[:num_samples]) * anom_ys_lagged[:num_samples]
            )
    return corr


def lagged_autocorrelation(
    xs: Xs,
    lag: int | Array,
    mode: Literal["full", "exact"] = "full",
    num_samples: Optional[int] = None,
) -> C:
    """Compute lagged autocorrelation of two time series."""
    return lagged_cross_correlation(
        xs, xs, lag, mode=mode, num_samples=num_samples
    )


def histogram(
    xs: Xs, bins: ArrayLike | Literal["auto"] = "auto", density: bool = True
) -> tuple[Array, Array]:
    """Compute histogram from array data."""
    if bins == "auto":
        np_hist, np_bin_edges = np.histogram(xs, bins=bins, density=density)
        hist = jnp.array(np_hist)
        bin_edges = jnp.array(np_bin_edges)
    else:
        hist, bin_edges = jnp.histogram(xs, bins=bins, density=density)
    return hist, bin_edges


def timeseries_stats(
    xs: Xs,
    num_lags: int,
    bins: int | Literal["auto"] = "auto",
    density: bool = True,
    autocorrelation_mode: Literal["full", "exact"] = "full",
    autocorrelation_num_samples: Optional[int] = None,
) -> TimeseriesStats:
    """Compute probability density and autocorrelation functions."""
    lagged_autocorr = partial(
        lagged_autocorrelation,
        mode=autocorrelation_mode,
        num_samples=autocorrelation_num_samples,
    )
    compute_autocorr = jit(vmap(lagged_autocorr, in_axes=(None, 0)))
    autocorr = compute_autocorr(xs, jnp.arange(num_lags + 1))
    densities, bin_edges = histogram(xs, bins, density)
    pdf: ProbDensity = {"densities": densities, "bin_edges": bin_edges}
    stats: TimeseriesStats = {"pdf": pdf, "autocorr": autocorr}
    return stats


def multivariate_timeseries_stats(
    xss: Xss,
    num_lags: int,
    bins: int | Literal["auto"] = "auto",
    density: bool = True,
    autocorrelation_mode: Literal["full", "exact"] = "full",
    autocorrelation_num_samples: Optional[int] = None,
    dropna: bool = False,
) -> MultivariateTimeseriesStats:
    """Compute probability density and autocorrelation functions.

    This function assumes that xss is either a vector of shape (num_samples,)
    or a 2D array of shape (num_vars, num_samples) where num_samples is the
    number of samples and num_vars the number of variables.
    """
    lagged_autocorr = partial(
        lagged_autocorrelation,
        mode=autocorrelation_mode,
        num_samples=autocorrelation_num_samples,
    )
    lagged_autocorrs = vmap(
        vmap(lagged_autocorr, in_axes=(None, 0)),
        in_axes=(0, None),
    )

    hist = partial(histogram, bins=bins, density=density)

    def _dropna(xss: Xs) -> Xs:
        mask = ~jnp.isnan(xss).any(axis=1)
        return xss[mask]

    autocorrs = jit(lagged_autocorrs)(xss, jnp.arange(num_lags + 1))
    if dropna:
        pdfs: list[ProbDensity] = [
            {"densities": densities, "bin_edges": bin_edges}
            for (densities, bin_edges) in map(hist, _dropna(xss))
        ]
    else:
        pdfs: list[ProbDensity] = [
            {"densities": densities, "bin_edges": bin_edges}
            for (densities, bin_edges) in map(hist, xss)
        ]
    stats: MultivariateTimeseriesStats = {"pdfs": pdfs, "autocorrs": autocorrs}
    return stats

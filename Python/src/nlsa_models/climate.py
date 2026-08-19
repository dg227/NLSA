"""Computation and plotting functions for general climate data."""

import jax
import nlsa.jax.delays as dl
import jax.numpy as jnp
import matplotlib.figure as mpf
import matplotlib.pyplot as plt
import nlsa.abstract_algebra as alg
import nlsa.function_algebra as fun
import nlsa.jax.eofs as eof
import nlsa.jax.kernels as knl
import nlsa.jax.koopman as koop
import nlsa.jax.stats as stats
import nlsa.jax.vector_algebra as vec
import numpy as np
import os
import pandas as pd
import xarray as xr
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from jax import Array, NamedSharding, vmap
from jax.typing import DTypeLike
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from nlsa.jax.eofs import EEOFEigen
from nlsa.jax.kernels import KernelEigen, KernelPars
from nlsa.jax.koopman import KoopmanEigen, KoopmanPars
from nlsa.jax.stats import anomaly_correlation_coefficient, normalized_rmse
from nlsa.jax.typing import typestable_jit
from nlsa.jax.vector_algebra import (
    L2FnAlgebra,
    L2FnAlgebraShardings,
    L2VectorAlgebra,
)
from nlsa.koopman import ImplementsKoopmanEigenbasis
from nlsa.typing import cast_like
from nlsa_models.core import (
    JaxEnv as JaxEnv,
    Matrix,
    NPMatrix,
    NPVector,
    SkillScores as SkillScores,
    Vector,
    initialize_jax as initialize_jax,
    initialize_matplotlib as initialize_matplotlib,
    to_skill_scores as to_skill_scores,
)
from pandas import DataFrame, DatetimeIndex, Series
from pathlib import Path
from tabulate import tabulate
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Protocol,
    assert_never,
    runtime_checkable,
)
from xarray import Dataset

if TYPE_CHECKING:
    type Device = Any
else:
    from jax import Device


type Y = Array  # Point in covariate space
type Ys = Array  # Collection of points in covariate space
type Yd = Array  # Point in delay-coordinate space
type TYd = Array  # Tangent vector in delay coordinate space
type R = Array  # Real number
type Rs = Array  # Collection of real numbers
type C = Array  # Complex number
type Cs = Array  # Collection of complex numbers
type Css = Array  # 2D array of complex numbers
type Mat = Array  # Matrix
type V = Array  # Vector in L2
type Vs = Array  # Collection of vectors in L2
type Vtst = Array  #  Vector in L2 with respect to the test dataset
type Vtsts = Array  # Collection of vectors in L2 with respect to test dataset
type Idx = int | Array  # basis vector index
type F[*Ss, T] = Callable[[*Ss], T]  # Shorthand for Callables

type RollingMode = Literal["forward", "backward", "center"]
type SpaceSampling = Literal[
    "pointwise",
    "coarsened",
    "area_averaged",
    "meridionally_averaged",
    "zonally_averaged",
]
type TimeSampling = Literal["monthly", "daily"]


@runtime_checkable
class ImplementsGriddedDataSpecs[T: TimeSampling, S: SpaceSampling, Dat](
    Protocol
):
    """Gridded variable protocol."""

    @property
    def varnames(self) -> Sequence[str]:
        """Return varnames property of GriddedDataSpecs protocol."""
        ...

    @property
    def input_varnames(self) -> Sequence[str]:
        """Return input_varnames property of GriddedDataSpecs protocol."""
        ...

    @property
    def min_lon(self) -> float | None:
        """Return min_lon property of GriddedDataSpecs protocol."""
        ...

    @property
    def max_lon(self) -> float | None:
        """Return max_lon property of GriddedDataSpecs protocol."""
        ...

    @property
    def step_lon(self) -> int | None:
        """Return step_lon property of GriddedDataSpecs protocol."""
        ...

    @property
    def min_lat(self) -> float | None:
        """Return min_lat property of GriddedDataSpecs protocol."""
        ...

    @property
    def max_lat(self) -> float | None:
        """Return max_lat property of GriddedDataSpecs protocol."""
        ...

    @property
    def step_lat(self) -> int | None:
        """Return step_lat property of GriddedDataSpecs protocol."""
        ...

    @property
    def space_sampling(self) -> S:
        """Return space_sampling property of GriddedDataSpecs protocol."""
        ...

    @property
    def time_sampling(self) -> T:
        """Return sampling frequency of GriddedDataSpecs protocol."""
        ...

    @property
    def rolling_window(self) -> int | None:
        """Return rolling window length of GriddedDataSpecs protocol."""
        ...

    @property
    def rolling_mode(self) -> RollingMode:
        """Return rolling averaging mode of GriddedDataSpecs protocol."""
        ...

    @property
    def standardize(self) -> bool:
        """Return standardize property of GriddedDataSpecs protocol."""
        ...

    @property
    def remove_climatology(self) -> bool:
        """Return standardize property of GriddedDataSpecs protocol."""
        ...

    @property
    def date_range(self) -> tuple[str, str]:
        """Return date_range property of GriddedDataSpecs."""
        ...

    @property
    def climatology_date_range(self) -> tuple[str, str]:
        """Return climatology_date_range property of GriddedDataSpecs."""
        ...

    @property
    def input_path(self) -> str | Path | None:
        """Return input_path property of GriddedDataSpecs protocol."""
        ...

    @property
    def file_format(self) -> Literal["grib", "nc"] | None:
        """Return file format property of GriddedDataSpecs protocol."""
        ...

    def read_data(self) -> Dat:
        """Read data associated with a GriddedDataSpecs object."""
        ...


@runtime_checkable
class ImplementsStationDataSpecs[T: TimeSampling, Dat](Protocol):
    """Station variable protocol."""

    @property
    def varnames(self) -> Sequence[str]:
        """Return varnames property of StationDataSpecs protocol."""
        ...

    @property
    def input_varnames(self) -> Sequence[str]:
        """Return input_varnames property of StationDataSpecs protocol."""
        ...

    @property
    def station_names(self) -> Sequence[str]:
        """Return station_names property of StationDataSpecs protocol."""
        ...

    @property
    def time_sampling(self) -> T:
        """Return sampling frequency of StationDataSpecs protocol."""
        ...

    @property
    def rolling_window(self) -> int | None:
        """Return rolling window length of StationDataSpecs protocol."""
        ...

    @property
    def rolling_mode(self) -> RollingMode:
        """Return rolling averaging mode of StationDataSpecs protocol."""
        ...

    @property
    def standardize(self) -> bool:
        """Return standardize property of StationDataSpecs protocol."""
        ...

    @property
    def remove_climatology(self) -> bool:
        """Return standardize property of StationDataSpecs protocol."""
        ...

    @property
    def date_range(self) -> tuple[str, str]:
        """Return date_range property of StationDataSpecs."""
        ...

    @property
    def climatology_date_range(self) -> tuple[str, str]:
        """Return climatology_date_range property of StationDataSpecs."""
        ...

    @property
    def input_path(self) -> str | Path | None:
        """Return input_path property of StationDataSpecs protocol."""
        ...

    @property
    def file_format(self) -> Literal["csv"] | None:
        """Return file format property of StationDataSpecs protocol."""
        ...

    @property
    def input_year(self) -> str:
        """Return input_year property of StationDataSpecs protocol."""
        ...

    @property
    def input_month(self) -> str:
        """Return input_month property of StationDataSpecs protocol."""
        ...

    @property
    def input_day(self) -> str:
        """Return input_day property of StationDataSpecs protocol."""
        ...

    def read_data(self) -> Dat:
        """Read data associated with a StationDataSpacs object."""
        ...


@runtime_checkable
class ImplementsCovariate(Protocol):
    """Represent objects that contain covariate data."""

    @property
    def covariate(
        self,
    ) -> Matrix[int, int, np.dtype[np.floating[Any]]]:
        """Covariate variables."""
        ...


@runtime_checkable
class ImplementsTimedCovariate(ImplementsCovariate, Protocol):
    """Represent objects that contain covariate and time data."""

    @property
    def time(self) -> DatetimeIndex:
        """Timestamps of the response data."""
        ...


@runtime_checkable
class ImplementsResponse(Protocol):
    """Represent objects that contain response data."""

    @property
    def response(
        self,
    ) -> (
        Vector[int, np.dtype[np.floating[Any]]]
        | Matrix[int, int, np.dtype[np.floating[Any]]]
    ):
        """Response variable."""
        ...


@runtime_checkable
class ImplementsTimedResponse(ImplementsResponse, Protocol):
    """Represent objects that contain response and time data."""

    @property
    def time(self) -> DatetimeIndex:
        """Timestamps of the response data."""
        ...


@dataclass(frozen=True, slots=True)
class Time[T: TimeSampling]:
    """Temporal and spatial sampling of climate data."""

    date_range: tuple[str, str]
    sampling: T
    rolling_window: int | None = None
    rolling_mode: RollingMode = "center"
    custom_climatology_date_range: tuple[str, str] | None = None

    @property
    def climatology_date_range(self) -> tuple[str, str]:
        """Return climatology_date_range property of Time object."""
        if self.custom_climatology_date_range is not None:
            clim = self.custom_climatology_date_range
        else:
            clim = self.date_range
        return clim

    def __str__(self) -> str:
        """Create string representation of Time object."""
        time = "-".join(self.date_range)
        if self.rolling_window is not None:
            roll = "_".join((f"roll{self.rolling_window}", self.rolling_mode))
        else:
            roll = None
        if self.custom_climatology_date_range is not None:
            clim = "clim" + "-".join(self.custom_climatology_date_range)
        else:
            clim = None
        return "_".join(filter(None, (time, self.sampling, roll, clim)))


@dataclass(frozen=True, slots=True)
class Climatology:
    """Climatology and standardization specs."""

    remove: bool = False
    standardize: bool = False

    def __str__(self) -> str:
        """Create string representation of Climatology object."""
        anom = "anom" if self.remove else None
        std = "std" if self.standardize else None
        match anom, std:
            case None, None:
                clim = ""
            case _:
                clim = "_".join(filter(None, (anom, std)))
        return clim


class Covariate[T: TimeSampling](NamedTuple):
    """NamedTuple for covariate specification."""

    specs: Sequence[
        ImplementsGriddedDataSpecs[T, SpaceSampling, Dataset]
        | ImplementsStationDataSpecs[T, Iterator[DataFrame]]
    ]

    def __str__(self) -> str:
        """Create string representation of covariate variables."""
        return "_".join(map(str, self.specs))


class Response[T: TimeSampling](NamedTuple):
    """NamedTuple for response specification."""

    specs: (
        ImplementsGriddedDataSpecs[T, Literal["area_averaged"], Dataset]
        | ImplementsStationDataSpecs[T, Iterator[DataFrame]]
    )

    def __str__(self) -> str:
        """Create string representation of response variable."""
        return str(self.specs)


@dataclass(frozen=True, slots=True)
class DataPars[T: TimeSampling]:
    """Dataclass containing training and test data parameter values."""

    covariate: Covariate[T]
    """Covariate function."""

    response: Response[T]
    """Response function."""

    num_half_delays: int = 0
    """Half number of delays (to ensure even two-sided embedding window)."""

    delay_embedding_mode: Literal["explicit", "on_the_fly"] = "on_the_fly"
    """Delay embedding mmode."""

    num_before: int = 0
    """Number of extra samples before delay embedding."""

    num_after: int = 0
    """Number of extra samples after delay embedding."""

    velocity_covariate: bool = False
    """Include time tendencies (velocities) in covariate data."""

    velocity_fd_order: Literal[2, 4, 6, 8] | None = None
    """Finite-difference order for velocity data."""

    eval_batch_size: int | None = None
    """Number of batches for batchwise evaluation."""

    @property
    def date_range(self) -> tuple[str, str]:
        """Analysis time interval (in YYYY-MM-DD format)."""
        return self.covariate.specs[0].date_range

    @property
    def time_sampling(self) -> TimeSampling:
        """Sampling time interval."""
        return self.covariate.specs[0].time_sampling

    @property
    def num_total_samples(self) -> int:
        """Total number of samples in the analysis interval."""
        match self.time_sampling:
            case "daily":
                freq = "D"
            case "monthly":
                freq = "M"
            case _ as unreachable:
                assert_never(unreachable)
        periods = pd.period_range(
            start=self.date_range[0], end=self.date_range[1], freq=freq
        )
        return len(periods)

    @property
    def num_velocity_fd(self) -> int:
        """Number of extra samples for velocity finite differencing."""
        if self.velocity_fd_order is not None:
            num_vel_fd = self.velocity_fd_order
        else:
            num_vel_fd = 0
        return num_vel_fd

    @property
    def num_delays(self) -> int:
        """Number of delays."""
        return 2 * self.num_half_delays

    @property
    def num_samples(self) -> int:
        """Number of samples after embedding, fd, quadrature, prediction."""
        num_samples = (
            self.num_total_samples
            - self.num_delays
            - self.num_before
            - self.num_after
            - self.num_velocity_fd
        )
        return num_samples

    @property
    def num_delay_samples(self) -> int:
        """Number of samples required for delay embedding."""
        return 2 * self.num_half_delays + self.num_samples

    @property
    def delay_embedding_origin(self) -> int:
        """Index of delay embedding origin."""
        return self.num_velocity_fd // 2 + self.num_before

    @property
    def delay_embedding_center(self) -> int:
        """Index of delay embedding center."""
        return self.delay_embedding_origin + self.num_half_delays

    @property
    def delay_embedding_end(self) -> int:
        """Index of delay embedding end."""
        return self.delay_embedding_origin + self.num_delays

    def __str__(self) -> str:
        """Create string representation of data parameters."""
        if self.velocity_covariate:
            assert self.velocity_fd_order is not None
            vel_str = f"vfd{self.velocity_fd_order}"
        else:
            vel_str = ""
        return "_".join(
            filter(
                None,
                (
                    f"ns{self.num_samples}",
                    f"nd{self.num_delays}",
                    f"nb{self.num_before}",
                    f"na{self.num_after}",
                    vel_str,
                ),
            )
        )

    def tabulate(self, name: str = "DataPars", show: bool = True) -> str:
        """Create tabulated summary of the properties of a DataPars object."""
        headers = [name, "Property Value"]
        data = {
            "Total number of samples": self.num_total_samples,
            "Number of analysis samples": self.num_samples,
            "Number of delays": self.num_delays,
            "Number of samples before": self.num_before,
            "Number of samples after": self.num_after,
            "Number of finite-difference samples": self.num_velocity_fd,
        }
        table = tabulate(data.items(), headers=headers)
        if show:
            print(table)
        return table

    def to_datetime_index(self) -> DatetimeIndex:
        """Get DatetimeIndex associated with DataPars object."""
        match self.covariate.specs[0].time_sampling:
            case "daily":
                freq = "D"
            case "monthly":
                freq = "MS"
        return pd.date_range(
            start=self.covariate.specs[0].date_range[0],
            end=self.covariate.specs[0].date_range[1],
            freq=freq,
        )


class Data(NamedTuple):
    """NamedTuple containing JAX arrays fof the covariate/response vars."""

    covariate: Array
    """Covariate variables."""

    response: Array
    """Response variables."""


class TimedResponse(NamedTuple):
    """NamedTuple containing time and response data."""

    time: DatetimeIndex
    """Timestamps of the response data."""

    response: Array | NPVector[int, np.dtype[np.floating[Any]]]
    """Response variables."""


class NPData(NamedTuple):
    """NamedTuple containing time, covariate, and response data."""

    time: DatetimeIndex
    """Timestamps of the covariate/response data."""

    covariate: NPMatrix[int, int, np.dtype[np.floating[Any]]]
    """Covariate variables."""

    response: (
        NPVector[int, np.dtype[np.floating[Any]]]
        | NPMatrix[int, int, np.dtype[np.floating[Any]]]
    )
    """Response variables."""

    def to_device(
        self,
        dtype: DTypeLike | None = None,
        shardings: NamedSharding | Device | None = None,
    ) -> Data:
        """Put NDArray data to on-device JAX arrays."""
        return Data(
            covariate=jnp.asarray(
                self.covariate, dtype=dtype, device=shardings
            ),
            response=jnp.asarray(self.response, dtype=dtype, device=shardings),
        )


def read_gridded_dataset[T: TimeSampling](
    specs: ImplementsGriddedDataSpecs[T, SpaceSampling, Dataset],
) -> Dataset:
    """Read gridded data into Xarray dataset."""
    # Open the dataset
    match specs.input_path, specs.file_format:
        case None, None:
            pth = "*"
        case None, str():
            pth = "*." + specs.file_format
        case str() | Path(), None:
            pth = Path(specs.input_path)
        case str() | Path(), str():
            pth = Path(specs.input_path) / ("*." + specs.file_format)

    ds_in = (
        xr.open_mfdataset(os.fspath(pth), parallel=True, chunks="auto")
        .unify_chunks()
        .sortby("latitude")
        .sel(
            longitude=slice(specs.min_lon, specs.max_lon),
            latitude=slice(specs.min_lat, specs.max_lat),
        )
    )

    # Extract and process input variables from input dataset
    match specs.space_sampling, specs.step_lon, specs.step_lat:
        case "pointwise", None, None:
            ds = ds_in[specs.input_varnames]
        case "pointwise", int(), None:
            ds = ds_in[specs.input_varnames].thin(longitude=specs.step_lon)
        case "pointwise", None, int():
            ds = ds_in[specs.input_varnames].thin(latitude=specs.step_lat)
        case "pointwise", int(), int():
            ds = ds_in[specs.input_varnames].thin(
                longitude=specs.step_lon, latitude=specs.step_lat
            )
        case "coarsened", None, None:
            raise ValueError("Longitude/latitude steps cannot be both None.")
        case "coarsened", int(), None:
            ds = (
                ds_in[specs.input_varnames]
                .coarsen(longitude=specs.step_lon, boundary="trim")
                .mean()
            )
        case "coarsened", None, int():
            ds = (
                ds_in[specs.input_varnames]
                .coarsen(latitude=specs.step_lat, boundary="trim")
                .mean()
            )
        case "coarsened", int(), int():
            ds = (
                ds_in[specs.input_varnames]
                .coarsen(
                    longitude=specs.step_lon,
                    latitude=specs.step_lat,
                    boundary="trim",
                )
                .mean()
            )
        case "area_averaged", _, _:
            ds = ds_in[specs.input_varnames].mean(
                dim=("longitude", "latitude")
            )
        case "meridionally_averaged", int(), _:
            ds = (
                ds_in[specs.input_varnames]
                .coarsen(longitude=specs.step_lon, boundary="trim")
                .mean()
                .mean(dim=("latitude"))
            )
        case "meridionally_averaged", _, _:
            ds = ds_in[specs.input_varnames].mean(dim=("latitude"))
        case "zonally_averaged", _, int():
            ds = (
                ds_in[specs.input_varnames]
                .coarsen(longitude=specs.step_lat, boundary="trim")
                .mean()
                .mean(dim=("longitude"))
            )
        case "zonally_averaged", _, _:
            ds = ds_in[specs.input_varnames].mean(dim=("longitude"))

    assert isinstance(ds, Dataset)
    ds = ds.rename(dict(zip(specs.input_varnames, specs.varnames)))

    # Remove climatology if requested
    if specs.remove_climatology:
        match specs.time_sampling:
            case "daily":
                groupby = "time.dayofyear"
            case "monthly":
                groupby = "time.month"
            case _ as unreachable:
                assert_never(unreachable)
        climatology_means = (
            ds.sel(
                time=slice(
                    specs.climatology_date_range[0],
                    specs.climatology_date_range[1],
                )
            )
            .groupby(groupby)
            .mean(dim="time")
        )
        ds = ds.groupby(groupby) - climatology_means

    # Perform rolling average if requested
    if specs.rolling_window is not None:
        match specs.rolling_mode:
            case "backward":
                ds = ds.rolling(
                    time=specs.rolling_window, min_periods=1
                ).mean()
            case "center":
                ds = ds.rolling(
                    time=specs.rolling_window, min_periods=1, center=True
                ).mean()
            case "forward":
                ds = (
                    ds.rolling(time=specs.rolling_window, min_periods=1)
                    .mean()
                    .shift(time=-(specs.rolling_window - 1))
                )

    # Standardize if requested
    if specs.standardize:
        std_means = ds.sel(
            time=slice(
                specs.climatology_date_range[0],
                specs.climatology_date_range[1],
            )
        ).mean(dim="time")
        anomalies = ds - std_means
        energies = (
            anomalies.sel(
                time=slice(
                    specs.climatology_date_range[0],
                    specs.climatology_date_range[1],
                )
            )
            ** 2
        )
        if specs.space_sampling == "pointwise":
            space = [dim for dim in energies.dims if dim != "time"]
            energies = energies.sum(dim=space)
        ds = anomalies / np.sqrt(energies.mean(dim="time"))

    # Extract requested date range
    ds = ds.sel(time=slice(specs.date_range[0], specs.date_range[1]))

    return ds


def extract_data_arrays[T: TimeSampling](
    data_pars: DataPars[T],
    dtype: np.dtype[np.floating[Any]] | None = None,
) -> NPData:
    """Extract gridded and station data."""

    def from_dataset(
        ds: Dataset,
    ) -> NPMatrix[int, int, np.dtype[np.floating[Any]]]:
        """Extract numpy array from xarray dataset."""
        a = (
            ds.to_stacked_array(new_dim="stacked_dim", sample_dims=["time"])
            .dropna(dim="stacked_dim")
            .astype(dtype)
            .to_numpy()
        )
        return a

    def from_dataframe(
        df: DataFrame,
        dtype: np.dtype[np.floating[Any]] | None = None,
    ) -> NPMatrix[int, int, np.dtype[np.floating[Any]]]:
        """Extract numpy array from Pandas dataframe."""
        a = df.to_numpy().astype(dtype)
        return a

    def to_2darray(
        specs: ImplementsGriddedDataSpecs[T, SpaceSampling, Dataset]
        | ImplementsStationDataSpecs[T, Iterator[DataFrame]],
    ) -> NPMatrix[int, int, np.dtype[np.floating[Any]]]:
        """Extract gridded or station data from specs to numpy array."""
        match specs:
            case ImplementsGriddedDataSpecs():
                a = from_dataset(specs.read_data())
            case ImplementsStationDataSpecs():
                a = np.hstack([from_dataframe(df) for df in specs.read_data()])
        return a

    print("Reading covariates:")
    print(*data_pars.covariate.specs, sep="\n")
    covariates = np.hstack(
        [to_2darray(specs) for specs in data_pars.covariate.specs]
    )
    print(f"Covariates array shape: {covariates.shape}")

    print("Reading response:")
    print(data_pars.response.specs)
    response = to_2darray(data_pars.response.specs).ravel()
    print(f"Response array shape: {response.shape}")
    match data_pars.covariate.specs[0].time_sampling:
        case "daily":
            freq = "D"
        case "monthly":
            freq = "MS"

    time = pd.date_range(
        start=data_pars.covariate.specs[0].date_range[0],
        end=data_pars.covariate.specs[0].date_range[1],
        freq=freq,
    )
    assert isinstance(time, DatetimeIndex)
    if data_pars.velocity_covariate:
        assert data_pars.velocity_fd_order is not None
        fd_op = typestable_jit(
            vmap(
                dl.make_fd_operator(
                    order=data_pars.velocity_fd_order, mode="central"
                ),
                in_axes=-1,
                out_axes=-1,
            )
        )
        vs = np.asarray(fd_op(jnp.asarray(covariates)), dtype=dtype)
        data = NPData(
            time=time,
            covariate=np.stack((covariates, vs), axis=1).astype(dtype),
            response=response,
        )
    else:
        data = NPData(
            time=time,
            covariate=covariates,
            response=response,
        )
    return data


def koopman_efuncs_to_data_frame[T: TimeSampling](
    data_pars: DataPars[T],
    koopman_basis: ImplementsKoopmanEigenbasis[Yd, C, C, V, Cs, Idx],
    timestamps: Series,
    which_eigs: int | tuple[int, int] | list[int],
    delay_timestamp_method: Literal["backward", "center"],
) -> DataFrame:
    """Extract Koopman eigenvectors to DataFrame."""
    match which_eigs:
        case int():
            idxs = jnp.arange(which_eigs)
        case tuple():
            idxs = jnp.arange(which_eigs[0], which_eigs[1])
        case list():
            idxs = jnp.array(which_eigs)
    match delay_timestamp_method:
        case "backward":
            i0 = data_pars.delay_embedding_end
        case "center":
            i0 = data_pars.delay_embedding_center
    i1 = i0 + data_pars.num_samples
    evecs = vmap(koopman_basis.vec, out_axes=1)(idxs)
    dual_evecs = vmap(koopman_basis.dual_vec, out_axes=1)(idxs)
    column_names = [
        f"Evec {idx}, efreq={koopman_basis.efreqs[idx]:.4g}" for idx in idxs
    ] + [
        f"Dual evec {idx}, efreq={koopman_basis.efreqs[idx]:.4g}"
        for idx in idxs
    ]
    df = pd.DataFrame(jnp.hstack((evecs, dual_evecs)), columns=column_names)
    df["Date"] = timestamps.iloc[i0:i1].values
    df.set_index("Date", inplace=True)
    return df


def make_data_driven_evaluation_functional[T: TimeSampling](
    data_pars: DataPars[T],
    dtype: DTypeLike,
    num_before: int = 0,
    num_after: int = 0,
    delay_embedding_mode: Literal["explicit", "on_the_fly"] = "on_the_fly",
    delay_window_pad: float | None = None,
    shardings: L2FnAlgebraShardings = L2FnAlgebraShardings(),
    jit: bool = False,
) -> Callable[[Data], Callable[[F[Yd, R]], V]]:
    """Make evaluation functional covariate data space."""

    def prepend_window(val: float, a: V) -> Array:
        return jnp.concatenate((jnp.full((data_pars.num_delays,), val), a))

    def impl_eval(data: Data) -> Callable[[F[Yd, R]], V]:
        i0 = data_pars.delay_embedding_origin - num_before
        i1 = i0 + data_pars.num_delay_samples + num_after
        if data_pars.num_half_delays > 0:
            match delay_embedding_mode:
                case "on_the_fly":
                    incl = dl.delay_eval_at(
                        jnp.asarray(
                            data.covariate[i0:i1],
                            dtype=dtype,
                            device=shardings.data,
                        ),
                        num_delays=data_pars.num_delays,
                        batch_size=data_pars.eval_batch_size,
                        out_sharding=shardings.vectors,
                        jit=jit,
                    )
                case "explicit":
                    if data_pars.velocity_covariate:
                        hankel = vmap(
                            partial(
                                dl.hankel,
                                num_delays=data_pars.num_delays,
                                flatten=True,
                            ),
                            in_axes=1,
                            out_axes=1,
                        )
                    else:
                        hankel = partial(
                            dl.hankel,
                            num_delays=data_pars.num_delays,
                            flatten=True,
                        )
                    if jit:
                        hankel = typestable_jit(hankel)
                    incl = vec.batch_eval_at(
                        jnp.asarray(
                            hankel(data.covariate[i0:i1]),
                            dtype=dtype,
                            device=shardings.data,
                        ),
                        batch_size=data_pars.eval_batch_size,
                        out_sharding=shardings.vectors,
                        jit=jit,
                    )
        else:
            incl = vec.batch_eval_at(
                jnp.asarray(
                    data.covariate[i0:i1],
                    dtype=dtype,
                    device=shardings.data,
                ),
                batch_size=data_pars.eval_batch_size,
                out_sharding=shardings.vectors,
                jit=jit,
            )
        if delay_window_pad is not None:
            return fun.compose(partial(prepend_window, delay_window_pad), incl)
        return incl

    return impl_eval


def make_data_driven_l2_space[T: TimeSampling, D: DTypeLike](
    data_pars: DataPars[T],
    dtype: D,
    delay_embedding_mode: Literal["explicit", "on_the_fly"] = "on_the_fly",
    shardings: L2FnAlgebraShardings = L2FnAlgebraShardings(),
    jit: bool = False,
) -> Callable[[Data], L2FnAlgebra[tuple[int], D, Yd, R]]:
    """Make implementation function for L2 space over covariate data space."""

    def impl_l2(data: Data) -> L2FnAlgebra[tuple[int], D, Yd, R]:
        i0 = data_pars.delay_embedding_origin
        i1 = i0 + data_pars.num_delay_samples
        incl: Callable[[F[Yd, R]], V]
        if data_pars.num_half_delays > 0:
            match delay_embedding_mode:
                case "on_the_fly":
                    incl = dl.delay_eval_at(
                        jnp.asarray(
                            data.covariate[i0:i1],
                            dtype=dtype,
                            device=shardings.data,
                        ),
                        num_delays=data_pars.num_delays,
                        batch_size=data_pars.eval_batch_size,
                        out_sharding=shardings.vectors,
                        jit=jit,
                    )
                case "explicit":
                    if data_pars.velocity_covariate:
                        hankel = vmap(
                            partial(
                                dl.hankel,
                                num_delays=data_pars.num_delays,
                                flatten=True,
                            ),
                            in_axes=1,
                            out_axes=1,
                        )
                    else:
                        hankel = partial(
                            dl.hankel,
                            num_delays=data_pars.num_delays,
                            flatten=True,
                        )
                    if jit:
                        hankel = typestable_jit(hankel)
                    incl = vec.batch_eval_at(
                        jnp.asarray(
                            hankel(data.covariate[i0:i1]),
                            dtype=dtype,
                            device=shardings.data,
                        ),
                        batch_size=data_pars.eval_batch_size,
                        out_sharding=shardings.vectors,
                        jit=jit,
                    )
        else:
            incl = vec.batch_eval_at(
                jnp.asarray(
                    data.covariate[i0:i1],
                    dtype=dtype,
                    device=shardings.data,
                ),
                batch_size=data_pars.eval_batch_size,
                out_sharding=shardings.vectors,
                jit=jit,
            )
        mu = vec.make_normalized_counting_measure(data_pars.num_samples)
        return L2FnAlgebra(
            shape=(data_pars.num_samples,),
            dtype=dtype,
            measure=mu,
            inclusion_map=incl,
            sharding=shardings.vectors,
        )

    return impl_l2


def make_data_driven_tangent_evaluation_functional_fd[T: TimeSampling](
    data_pars: DataPars[T],
    dtype: DTypeLike,
    fd_order: Literal[2, 4, 6, 8],
    delay_embedding_mode: Literal["explicit", "on_the_fly"] = "on_the_fly",
    shardings: L2FnAlgebraShardings = L2FnAlgebraShardings(),
    jit: bool = False,
) -> Callable[[Data], Callable[[F[Yd, TYd, R]], V]]:
    """Make evaluation functional for finite-difference approximation."""

    def impl_eval_tx(data: Data) -> Callable[[F[Yd, TYd, R]], V]:
        if data_pars.velocity_covariate:
            fd_op = vmap(
                vmap(
                    dl.make_fd_operator(
                        order=fd_order,
                        mode="central",
                        dt=1,
                        extrap=False,
                    ),
                    in_axes=-1,
                    out_axes=-1,
                ),
                in_axes=-1,
                out_axes=-1,
            )
        else:
            fd_op = vmap(
                dl.make_fd_operator(
                    order=fd_order,
                    mode="central",
                    dt=1,
                    extrap=False,
                ),
                in_axes=-1,
                out_axes=-1,
            )
        if jit:
            fd_op = typestable_jit(fd_op)

        num_half_fd = fd_order // 2
        i0 = data_pars.delay_embedding_origin
        i1 = i0 + data_pars.num_delay_samples
        i0_fd = i0 - num_half_fd
        i1_fd = i1 + num_half_fd
        eval_tx: Callable[[F[Yd, TYd, R]], V]
        if data_pars.num_half_delays > 0:
            match delay_embedding_mode:
                case "on_the_fly":
                    eval_tx = dl.delay_eval_at(
                        (
                            jnp.asarray(
                                data.covariate[i0:i1],
                                dtype=dtype,
                                device=shardings.data,
                            ),
                            jnp.asarray(
                                fd_op(data.covariate[i0_fd:i1_fd]),
                                dtype=dtype,
                                device=shardings.data,
                            ),
                        ),
                        num_delays=data_pars.num_delays,
                        batch_size=data_pars.eval_batch_size,
                        out_sharding=shardings.vectors,
                        jit=jit,
                    )
                case "explicit":
                    if data_pars.velocity_covariate:
                        hankel = vmap(
                            partial(
                                dl.hankel,
                                num_delays=data_pars.num_delays,
                                flatten=True,
                            ),
                            in_axes=1,
                            out_axes=1,
                        )
                    else:
                        hankel = partial(
                            dl.hankel,
                            num_delays=data_pars.num_delays,
                            flatten=True,
                        )
                    if jit:
                        hankel = typestable_jit(hankel)
                    eval_tx = vec.batch_eval_at(
                        (
                            jnp.asarray(
                                hankel(jnp.asarray(data.covariate[i0:i1])),
                                dtype=dtype,
                                device=shardings.data,
                            ),
                            jnp.asarray(
                                hankel(
                                    fd_op(
                                        jnp.asarray(
                                            data.covariate[i0_fd:i1_fd]
                                        )
                                    )
                                ),
                                dtype=dtype,
                                device=shardings.data,
                            ),
                        ),
                        batch_size=data_pars.eval_batch_size,
                        out_sharding=shardings.vectors,
                        jit=jit,
                    )
        else:
            eval_tx = vec.batch_eval_at(
                (
                    jnp.asarray(
                        data.covariate[i0:i1],
                        dtype=dtype,
                        device=shardings.data,
                    ),
                    jnp.asarray(
                        fd_op(jnp.asarray(data.covariate[i0_fd:i1_fd])),
                        dtype=dtype,
                        device=shardings.data,
                    ),
                ),
                batch_size=data_pars.eval_batch_size,
                out_sharding=shardings.vectors,
                jit=jit,
            )
        return eval_tx

    return impl_eval_tx


# TODO: There is alot of repetition between this function and the
# corresponding one in lorenz63.py. If we make DataPars a protocol it
# might be possible to move this to kernels.py as a generic function.
# In this particular function, the only things that we need from DataPars are
# delay_embedding end and num_samples. These could be easily defined as a
# protocol.
def compute_kaf_expansion_coeffs[T: TimeSampling](
    pars: tuple[DataPars[T], KernelPars],
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[Yd, R, V, R]],
    train_data: Data,
    kernel: Callable[[Yd, Yd], R] | Callable[[Data, Yd, Yd], R],
    kernel_eigen: KernelEigen,
    num_steps: int,
    which_eigs: int | tuple[int, int] | list[int] | None = None,
    responses: Array | None = None,
    jit: bool = True,
) -> Array:
    """Compute basis expansion coefficients for kernel analog forecast."""
    data_pars, kernel_pars = pars
    impl_basis = knl.make_data_driven_eigenbasis(
        kernel_pars, impl_l2, kernel, which_eigs
    )
    anal: Callable[[Data, Vs, KernelEigen], Rs]
    if responses is not None:
        anal = knl.make_kaf_analysis_operator(
            impl_basis,
            num_steps,
        )
        coeffs = anal(train_data, responses, kernel_eigen)
        if jit:
            anal = typestable_jit(anal)
    else:
        i0 = data_pars.delay_embedding_end
        i1 = i0 + num_steps + data_pars.num_samples
        anal = knl.make_kaf_analysis_operator(
            impl_basis,
            num_steps,
            which_samples=(i0, i1),
        )
        if jit:
            anal = typestable_jit(anal)
        coeffs = anal(train_data, train_data.response, kernel_eigen)
    return coeffs


def compute_koopman_response_coeffs[D: DTypeLike, L: int, T: TimeSampling](
    pars: tuple[DataPars[T], KernelPars, KoopmanPars],
    c_l: L2VectorAlgebra[tuple[L], D],
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[Yd, R, V, R]],
    train_data: Data,
    kernel: Callable[[Yd, Yd], R] | Callable[[Data, Yd, Yd], R],
    kernel_eigen: KernelEigen,
    koopman_eigen: KoopmanEigen,
    which_eigs: int | tuple[int, int] | list[int] | None = None,
    jit: bool = True,
) -> Cs:
    """Compute basis expansion coefficients for Koopman forecast."""
    data_pars, kernel_pars, koopman_pars = pars
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
    impl_koopman_basis = koop.make_data_driven_eigenbasis(
        koopman_pars, c_l, impl_kernel_basis, which_eigs
    )
    i0 = data_pars.delay_embedding_end
    i1 = i0 + data_pars.num_samples
    anal: Callable[[Data, Vs, KernelEigen, KoopmanEigen], Cs] = (
        koop.make_koopman_analysis_operator(
            impl_koopman_basis,
            which_samples=(i0, i1),
        )
    )
    if jit:
        anal = typestable_jit(anal)
    return anal(train_data, train_data.response, kernel_eigen, koopman_eigen)


def compute_response_skill_scores[T: TimeSampling](
    data_pars: DataPars[T],
    test_data: ImplementsResponse,
    preds: Vtsts,
    dropna: bool = False,
) -> SkillScores:
    """Compute NRMSE and ACC skill scores over the prediction ensemble."""
    num_pred_steps = preds.shape[1] - 1
    i0 = data_pars.delay_embedding_end
    i1 = i0 + num_pred_steps + data_pars.num_samples
    hankel = typestable_jit(partial(dl.hankel, num_delays=num_pred_steps))
    fxs_true = hankel(jnp.asarray(test_data.response[i0:i1]))
    if dropna:
        mask = ~jnp.isnan(preds).any(axis=1)
        preds = preds[mask]
        fxs_true = fxs_true[mask]
        assert isinstance(fxs_true, Array)
    normalized_rmses = typestable_jit(vmap(normalized_rmse, in_axes=1))
    nrmses = normalized_rmses(fxs_true, preds)
    anomaly_correlation_coefficients = typestable_jit(
        vmap(anomaly_correlation_coefficient, in_axes=1)
    )
    accs = anomaly_correlation_coefficients(fxs_true, preds)
    scores: SkillScores = {"nrmses": nrmses, "accs": accs}
    return scores


def compute_covariate_skill_scores[T: TimeSampling](
    data_pars: DataPars[T],
    test_data: ImplementsCovariate,
    ys_pred: Vtsts,
    dropna: bool = False,
) -> SkillScores:
    """Compute NRMSE and ACC skill scores over the prediction ensemble."""
    num_pred_steps = len(ys_pred) - 1
    i0 = data_pars.delay_embedding_end
    i1 = i0 + num_pred_steps + data_pars.num_samples
    hankel = fun.compose(
        partial(jnp.swapaxes, axis1=0, axis2=1),
        vmap(
            partial(dl.hankel, num_delays=num_pred_steps),
            in_axes=-1,
            out_axes=-1,
        ),
    )
    hankel = cast_like(hankel, jax.jit(hankel))
    normalized_rmses = typestable_jit(
        vmap(vmap(stats.normalized_rmse, in_axes=1), in_axes=2)
    )
    anomaly_correlation_coefficients = typestable_jit(
        vmap(vmap(stats.anomaly_correlation_coefficient, in_axes=1), in_axes=2)
    )
    ys_true = hankel(jnp.asarray(test_data.covariate[i0:i1]))
    if dropna:
        mask = ~jnp.isnan(ys_pred).any(axis=(1, 2))
        ys_pred = ys_pred[mask]
        ys_true = ys_true[mask]
        assert isinstance(ys_true, Array)
    nrmses = normalized_rmses(ys_true, ys_pred)
    accs = anomaly_correlation_coefficients(ys_true, ys_pred)
    scores: SkillScores = {"nrmses": nrmses, "accs": accs}
    return scores


def compute_skill_scores[T: TimeSampling](
    data_pars: DataPars[T],
    test_data: ImplementsCovariate | ImplementsResponse,
    preds: Vtsts,
    what: Literal["covariates", "responses"] = "responses",
    dropna: bool = False,
) -> SkillScores:
    """Compute NRMSE and ACC skill scores over the prediction ensemble."""
    match what:
        case "covariates":
            assert isinstance(test_data, ImplementsCovariate)
            scores = compute_covariate_skill_scores(
                data_pars, test_data, preds, dropna=dropna
            )
        case "responses":
            assert isinstance(test_data, ImplementsResponse)
            scores = compute_response_skill_scores(
                data_pars, test_data, preds, dropna=dropna
            )
    return scores


def plot_bandwidth_function[T: TimeSampling](
    data_pars: DataPars[T],
    impl_l2y: Callable[[Data], alg.ImplementsL2FnAlgebra[Yd, R, V, R]],
    bandwidth_func: Callable[[Data, Yd], R],
    train_data: Data | NPData,
    train_shardings: NamedSharding | Device | None = None,
    test_data_pars: DataPars[T] | None = None,
    impl_l2y_tst: Callable[[Data], alg.ImplementsL2FnAlgebra[Yd, R, V, R]]
    | None = None,
    test_data: Data | NPData | None = None,
    test_shardings: NamedSharding | Device | None = None,
    delay_plot_mode: Literal["backward", "central"] = "central",
    plt_date_range: tuple[str, str] | None = None,
    plt_date_range_tst: tuple[str, str] | None = None,
    plt_step: int = 1,
    plt_step_tst: int = 1,
    i_fig: int = 1,
) -> Figure:
    """Plot bandwidth function on training and, optionally, test data."""

    @typestable_jit
    def bandwidths(xs_train: Data) -> V:
        l2y = impl_l2y(xs_train)
        return l2y.incl(partial(bandwidth_func, xs_train))

    @typestable_jit
    def bandwidths_tst(xs_train: Data, xs_tst: Data) -> Vtst:
        if impl_l2y_tst is not None:
            l2y_tst = impl_l2y(xs_tst)
            return l2y_tst.incl(partial(bandwidth_func, xs_train))
        else:
            return jnp.zeros(shape=())

    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    ax: Axes
    ax_tst: Axes | None
    if test_data_pars is not None:
        fig, (ax, ax_tst) = plt.subplots(
            1,
            2,
            num=i_fig,
            figsize=tuple(mpf.figaspect(0.5)),
            constrained_layout=True,
            sharey=True,
        )
    else:
        fig, ax = plt.subplots(
            num=i_fig,
            figsize=tuple(mpf.figaspect(0.5)),
            constrained_layout=True,
        )
        ax_tst = None
    match delay_plot_mode:
        case "backward":
            i0_dl = data_pars.delay_embedding_end
        case "central":
            i0_dl = data_pars.delay_embedding_center
    if plt_date_range is not None:
        i0 = data_pars.to_datetime_index().get_loc(plt_date_range[0])
        i1 = data_pars.to_datetime_index().get_loc(plt_date_range[1])
        assert isinstance(i0, int)
        assert isinstance(i1, int)
        i1 += 1
    else:
        i0 = i0_dl
        i1 = i0 + data_pars.num_samples
    j0 = i0 - i0_dl
    j1 = i1 - i0_dl
    match train_data:
        case Data():
            bw_vals = bandwidths(train_data)
        case NPData():
            bw_vals = bandwidths(
                train_data.to_device(shardings=train_shardings)
            )
    ax.plot(
        data_pars.to_datetime_index()[i0:i1:plt_step].values,
        bw_vals[j0:j1:plt_step],
        "-",
    )
    ax.grid(True)
    ax.set_title("Kernel bandwidth function (training)")

    if (
        impl_l2y_tst is not None
        and test_data is not None
        and test_data_pars is not None
        and ax_tst is not None
    ):
        match delay_plot_mode:
            case "backward":
                i0_dl_tst = test_data_pars.delay_embedding_end
            case "central":
                i0_dl_tst = test_data_pars.delay_embedding_center
        if plt_date_range_tst is not None:
            i0_tst = test_data_pars.to_datetime_index().get_loc(
                plt_date_range_tst[0]
            )
            i1_tst = test_data_pars.to_datetime_index().get_loc(
                plt_date_range_tst[1]
            )
            assert isinstance(i0_tst, int)
            assert isinstance(i1_tst, int)
            i1_tst += 1
        else:
            i0_tst = i0_dl_tst
            i1_tst = i0_tst + test_data_pars.num_samples
        j0_tst = i0_tst - i0_dl_tst
        j1_tst = i1_tst - i0_dl_tst
        match train_data, test_data:
            case Data(), Data():
                bw_vals_tst = bandwidths_tst(
                    train_data,
                    test_data,
                )
            case NPData(), Data():
                bw_vals_tst = bandwidths_tst(
                    train_data.to_device(shardings=train_shardings),
                    test_data,
                )
            case Data(), NPData():
                bw_vals_tst = bandwidths_tst(
                    train_data,
                    test_data.to_device(shardings=test_shardings),
                )
            case NPData(), NPData():
                bw_vals_tst = bandwidths_tst(
                    train_data.to_device(shardings=train_shardings),
                    test_data.to_device(shardings=test_shardings),
                )
        ax_tst.plot(
            test_data_pars.to_datetime_index()[
                i0_tst:i1_tst:plt_step_tst
            ].values,
            bw_vals_tst[j0_tst:j1_tst:plt_step_tst],
            "-",
        )
        ax_tst.grid(True)
        ax_tst.set_title("Kernel bandwidth function (test)")
    return fig


def make_kernel_evecs_plotter[T: TimeSampling](
    pars: tuple[DataPars[T], KernelPars],
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[Yd, R, V, R]],
    train_data: Data | NPData,
    kernel_eigen: KernelEigen,
    train_shardings: NamedSharding | Device | None = None,
    test_data_pars: DataPars[T] | None = None,
    impl_l2_tst: Callable[[Data], alg.ImplementsL2FnAlgebra[Yd, R, Vtst, R]]
    | None = None,
    test_data: Data | NPData | None = None,
    test_shardings: NamedSharding | Device | None = None,
    kernel: Callable[[Data, Yd, Yd], R] | None = None,
    delay_plot_mode: Literal["backward", "central"] = "backward",
    plt_date_range: tuple[str, str] | None = None,
    plt_date_range_tst: tuple[str, str] | None = None,
    plt_step: int = 1,
    plt_step_tst: int = 1,
    i_fig: int = 1,
) -> tuple[Figure, Callable[[int], None]]:
    """Make plotting function for kernel eigenfunctions."""
    data_pars, kernel_pars = pars
    lapl_evals = knl.to_laplace_eigenvalues(
        kernel_eigen.evals, kernel_eigen.bandwidth
    )
    if kernel is not None:
        impl_kernel_basis = knl.make_data_driven_eigenbasis(
            kernel_pars, impl_l2, kernel
        )
    else:
        impl_kernel_basis = None

    @typestable_jit
    def efunc(
        _train_data: Data,
        _kernel_eigen: KernelEigen,
        _test_data: Data,
        j: Idx,
    ) -> Vtst:
        assert impl_kernel_basis is not None
        assert impl_l2_tst is not None
        l2y_tst = impl_l2_tst(_test_data)
        eigenbasis = impl_kernel_basis(_train_data, _kernel_eigen)
        return l2y_tst.incl(eigenbasis.fn(j))

    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    ax: Axes
    ax_tst: Axes | None
    if test_data_pars is not None:
        fig, (ax, ax_tst) = plt.subplots(
            1,
            2,
            num=i_fig,
            figsize=tuple(mpf.figaspect(0.5)),
            constrained_layout=True,
            sharey=True,
        )
    else:
        fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
        ax_tst = None
    match delay_plot_mode:
        case "backward":
            i0_dl = data_pars.delay_embedding_end
        case "central":
            i0_dl = data_pars.delay_embedding_center
    if plt_date_range is not None:
        i0 = data_pars.to_datetime_index().get_loc(plt_date_range[0])
        i1 = data_pars.to_datetime_index().get_loc(plt_date_range[1])
        assert isinstance(i0, int)
        assert isinstance(i1, int)
        i1 += 1
    else:
        i0 = i0_dl
        i1 = i0 + data_pars.num_samples
    j0 = i0 - i0_dl
    j1 = i1 - i0_dl
    if test_data_pars is not None and test_data is not None:
        match delay_plot_mode:
            case "backward":
                i0_dl_tst = test_data_pars.delay_embedding_end
            case "central":
                i0_dl_tst = test_data_pars.delay_embedding_center
        if plt_date_range_tst is not None:
            i0_tst = test_data_pars.to_datetime_index().get_loc(
                plt_date_range_tst[0]
            )
            i1_tst = test_data_pars.to_datetime_index().get_loc(
                plt_date_range_tst[1]
            )
            assert isinstance(i0_tst, int)
            assert isinstance(i1_tst, int)
            i1_tst += 1
        else:
            i0_tst = i0_dl_tst
            i1_tst = i0_tst + test_data_pars.num_samples
        j0_tst = i0_tst - i0_dl_tst
        j1_tst = i1_tst - i0_dl_tst
    else:
        i0_tst, i1_tst, j0_tst, j1_tst = None, None, None, None

    def plot_eig(k: int):
        evec = kernel_eigen.evecs[k]
        if test_data is not None:
            match train_data, test_data:
                case Data(), Data():
                    evec_tst = efunc(train_data, kernel_eigen, test_data, k)
                case Data(), NPData():
                    evec_tst = efunc(
                        train_data,
                        kernel_eigen,
                        test_data.to_device(shardings=test_shardings),
                        k,
                    )
                case NPData(), Data():
                    evec_tst = efunc(
                        train_data.to_device(shardings=train_shardings),
                        kernel_eigen,
                        test_data,
                        k,
                    )
                case NPData(), NPData():
                    evec_tst = efunc(
                        train_data.to_device(shardings=train_shardings),
                        kernel_eigen,
                        test_data.to_device(shardings=test_shardings),
                        k,
                    )
        else:
            evec_tst = None
        for figax in fig.axes:
            figax.cla()
        ax.plot(
            data_pars.to_datetime_index()[i0:i1:plt_step],
            evec[j0:j1:plt_step],
            "-",
        )
        eta = lapl_evals[k]
        ax.grid()
        ax.set_title(f"Eigenvector {k}: $\\eta_{{{k}}} = {eta: .3f}$")
        if (
            ax_tst is not None
            and test_data_pars is not None
            and evec_tst is not None
        ):
            ax_tst.plot(
                test_data_pars.to_datetime_index()[i0_tst:i1_tst:plt_step_tst],
                evec_tst[j0_tst:j1_tst:plt_step_tst],
                "-",
            )
            ax_tst.grid()
            ax_tst.set_title("Nystrom")

    return fig, plot_eig


def make_koopman_evecs_plotter[T: TimeSampling, D: DTypeLike, L: int](
    pars: tuple[DataPars[T], KernelPars, KoopmanPars],
    c_l: L2VectorAlgebra[tuple[L], D],
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[Yd, R, V, R]],
    train_data: Data | NPData,
    kernel_eigen: KernelEigen,
    koopman_eigen: KoopmanEigen,
    train_shardings: NamedSharding | Device | None = None,
    test_data_pars: DataPars[T] | None = None,
    impl_l2_tst: Callable[[Data], alg.ImplementsL2FnAlgebra[Yd, R, Vtst, R]]
    | None = None,
    test_data: Data | NPData | None = None,
    test_shardings: NamedSharding | Device | None = None,
    kernel: Callable[[Yd, Yd], R] | Callable[[Data, Yd, Yd], R] | None = None,
    delay_plot_mode: Literal["backward", "central"] = "backward",
    plt_date_range: tuple[str, str] | None = None,
    plt_date_range_tst: tuple[str, str] | None = None,
    plt_step: int = 1,
    plt_step_tst: int = 1,
    i_fig: int = 1,
) -> tuple[Figure, Callable[[int], None]]:
    """Make plotting function for Koopman eigenfunctions."""
    data_pars, kernel_pars, koopman_pars = pars
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
    if kernel is not None:
        impl_kernel_basis = knl.make_data_driven_eigenbasis(
            kernel_pars, impl_l2, kernel, which_kernel_eigs
        )
        impl_koopman_basis = koop.make_data_driven_eigenbasis(
            koopman_pars, c_l, impl_kernel_basis
        )
    else:
        impl_koopman_basis = None

    @typestable_jit
    def efunc(
        _train_data: Data,
        _kernel_eigen: KernelEigen,
        _koopman_eigen: KoopmanEigen,
        _test_data: Data,
        j: Idx,
    ) -> Vtst:
        assert impl_koopman_basis is not None
        assert impl_l2_tst is not None
        eigenbasis = impl_koopman_basis(
            _train_data, _kernel_eigen, _koopman_eigen
        )
        l2y_tst = impl_l2_tst(_test_data)
        return l2y_tst.incl(eigenbasis.fn(j))

    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    axs: Sequence[Axes]
    axs_tst: Sequence[Axes] | None
    if test_data_pars is not None:
        figsize = plt.rcParams["figure.figsize"]
        fig, (axs, axs_tst) = plt.subplots(
            2,
            2,
            num=i_fig,
            figsize=(1.75 * figsize[0], 1.75 * figsize[1]),
            constrained_layout=True,
        )
    else:
        fig, axs = plt.subplots(
            1,
            2,
            num=i_fig,
            figsize=tuple(mpf.figaspect(0.5)),
            constrained_layout=True,
        )
        axs_tst = None
    match delay_plot_mode:
        case "backward":
            i0_dl = data_pars.delay_embedding_end
        case "central":
            i0_dl = data_pars.delay_embedding_center
    if plt_date_range is not None:
        i0 = data_pars.to_datetime_index().get_loc(plt_date_range[0])
        i1 = data_pars.to_datetime_index().get_loc(plt_date_range[1])
        assert isinstance(i0, int)
        assert isinstance(i1, int)
        i1 += 1
    else:
        i0 = i0_dl
        i1 = i0 + data_pars.num_samples
    j0 = i0 - i0_dl
    j1 = i1 - i0_dl
    if test_data_pars is not None and test_data is not None:
        match delay_plot_mode:
            case "backward":
                i0_dl_tst = test_data_pars.delay_embedding_end
            case "central":
                i0_dl_tst = data_pars.delay_embedding_center
        if plt_date_range_tst is not None:
            i0_tst = test_data_pars.to_datetime_index().get_loc(
                plt_date_range_tst[0]
            )
            i1_tst = test_data_pars.to_datetime_index().get_loc(
                plt_date_range_tst[1]
            )
            assert isinstance(i0_tst, int)
            assert isinstance(i1_tst, int)
            i1_tst += 1
        else:
            i0_tst = i0_dl_tst
            i1_tst = i0_tst + test_data_pars.num_samples
        j0_tst = i0_tst - i0_dl_tst
        j1_tst = i1_tst - i0_dl_tst
    else:
        i0_tst, i1_tst, j0_tst, j1_tst = None, None, None, None

    def plot_eig(k: int):
        for ax in fig.axes:
            ax.cla()
        evec = (
            koopman_eigen.evec_coeffs[k]
            @ knl.slice_eigen(kernel_eigen, which_kernel_eigs).evecs
        )
        if test_data is not None:
            match train_data, test_data:
                case Data(), Data():
                    evec_tst = efunc(
                        train_data, kernel_eigen, koopman_eigen, test_data, k
                    )
                case Data(), NPData():
                    evec_tst = efunc(
                        train_data,
                        kernel_eigen,
                        koopman_eigen,
                        test_data.to_device(shardings=test_shardings),
                        k,
                    )
                case NPData(), Data():
                    evec_tst = efunc(
                        train_data.to_device(shardings=train_shardings),
                        kernel_eigen,
                        koopman_eigen,
                        test_data,
                        k,
                    )
                case NPData(), NPData():
                    evec_tst = efunc(
                        train_data.to_device(shardings=train_shardings),
                        kernel_eigen,
                        koopman_eigen,
                        test_data.to_device(shardings=test_shardings),
                        k,
                    )
        else:
            evec_tst = None
        match data_pars.time_sampling:
            case "daily":
                efreq = koopman_eigen.efreqs[k] / (2 * jnp.pi) * 365
                eperiod = koopman_eigen.eperiods[k]
                efreq_str = "cycles/year"
                eperiod_str = "days"
            case "monthly":
                efreq = koopman_eigen.efreqs[k] / (2 * jnp.pi) * 12
                eperiod = koopman_eigen.eperiods[k] / 12
                efreq_str = "cycles/year"
                eperiod_str = "years"

        ax = axs[0]
        ax.plot(evec.real[j0:j1:plt_step], evec.imag[j0:j1:plt_step], "-")
        ax.set_xlabel(f"$\\mathrm{{Re}}\\zeta_{{{k}}}$")
        ax.set_ylabel(f"$\\mathrm{{Im}}\\zeta_{{{k}}}$")
        ax.set_title(
            f"Eigenfrequency $\\nu_{{{k}}} = {efreq: .3f}$ {efreq_str}"
        )
        ax.grid()

        ax = axs[1]
        ax.plot(
            data_pars.to_datetime_index()[i0:i1:plt_step],
            evec.real[j0:j1:plt_step],
            "-",
            label=f"$\\mathrm{{Re}}\\zeta_{{{k}}}$",
        )
        ax.plot(
            data_pars.to_datetime_index()[i0:i1:plt_step],
            evec.imag[j0:j1:plt_step],
            "-",
            label=f"$\\mathrm{{Im}}\\zeta_{{{k}}}$",
        )
        ax.set_title(f"Eigenperiod $T_{{{k}}} = {eperiod: .3f}$ {eperiod_str}")
        ax.grid()
        ax.legend()

        if (
            axs_tst is not None
            and test_data_pars is not None
            and evec_tst is not None
        ):
            ax = axs_tst[0]
            ax.plot(
                evec_tst.real[j0_tst:j1_tst:plt_step_tst],
                evec_tst.imag[j0_tst:j1_tst:plt_step_tst],
                "-",
            )
            ax.set_xlabel(f"$\\mathrm{{Re}}\\zeta_{{{k}}}$")
            ax.set_ylabel(f"$\\mathrm{{Im}}\\zeta_{{{k}}}$")
            ax.grid()

            ax = axs_tst[1]
            ax.plot(
                test_data_pars.to_datetime_index()[i0_tst:i1_tst:plt_step_tst],
                evec_tst.real[j0_tst:j1_tst:plt_step_tst],
                "-",
                label=f"$\\mathrm{{Re}}\\zeta_{{{k}}}$",
            )
            ax.plot(
                test_data_pars.to_datetime_index()[i0_tst:i1_tst:plt_step_tst],
                evec_tst.imag[j0_tst:j1_tst:plt_step_tst],
                "-",
                label=f"$\\mathrm{{Im}}\\zeta_{{{k}}}$",
            )
            ax.grid()
            ax.legend()

    return fig, plot_eig


def make_eeof_evecs_plotter[T: TimeSampling](
    data_pars: DataPars[T],
    impl_l2: Callable[[Data], alg.ImplementsL2FnAlgebra[Yd, R, V, R]],
    train_data: Data | NPData,
    eeof_eigen: EEOFEigen,
    train_shardings: NamedSharding | Device | None = None,
    test_data_pars: DataPars[T] | None = None,
    impl_l2_tst: Callable[[Data], alg.ImplementsL2FnAlgebra[Yd, R, V, R]]
    | None = None,
    test_data: Data | NPData | None = None,
    test_shardings: NamedSharding | Device | None = None,
    delay_plot_mode: Literal["backward", "central"] = "backward",
    plt_date_range: tuple[str, str] | None = None,
    plt_date_range_tst: tuple[str, str] | None = None,
    plt_step: int = 1,
    plt_step_tst: int = 1,
    i_fig: int = 1,
) -> tuple[Figure, F[int, None]]:
    """Make plotting function for Koopman eigenfunctions."""
    impl_eeof_basis = eof.make_data_driven_eigenbasis(impl_l2)

    @typestable_jit
    def efunc(
        _train_data: Data,
        _eeof_eigen: EEOFEigen,
        _test_data: Data,
        j: Idx,
    ) -> Vtst:
        assert impl_l2_tst is not None
        eigenbasis = impl_eeof_basis(_train_data, _eeof_eigen)
        l2y_tst = impl_l2_tst(_test_data)
        return l2y_tst.incl(eigenbasis.fn(j)) + 1j * l2y_tst.incl(
            eigenbasis.fn(j + 1)
        )

    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    axs: Sequence[Axes]
    axs_tst: Sequence[Axes] | None
    if test_data_pars is not None:
        figsize = plt.rcParams["figure.figsize"]
        fig, (axs, axs_tst) = plt.subplots(
            2,
            2,
            num=i_fig,
            figsize=(1.75 * figsize[0], 1.75 * figsize[1]),
            constrained_layout=True,
        )
    else:
        fig, axs = plt.subplots(
            1,
            2,
            num=i_fig,
            figsize=tuple(mpf.figaspect(0.5)),
            constrained_layout=True,
        )
        axs_tst = None
    match delay_plot_mode:
        case "backward":
            i0_dl = data_pars.delay_embedding_end
        case "central":
            i0_dl = data_pars.delay_embedding_center
    if plt_date_range is not None:
        i0 = data_pars.to_datetime_index().get_loc(plt_date_range[0])
        i1 = data_pars.to_datetime_index().get_loc(plt_date_range[1])
        assert isinstance(i0, int)
        assert isinstance(i1, int)
        i1 += 1
    else:
        i0 = i0_dl
        i1 = i0 + data_pars.num_samples
    j0 = i0 - i0_dl
    j1 = i1 - i0_dl
    if test_data_pars is not None and test_data is not None:
        match delay_plot_mode:
            case "backward":
                i0_dl_tst = test_data_pars.delay_embedding_end
            case "central":
                i0_dl_tst = data_pars.delay_embedding_center
        if plt_date_range_tst is not None:
            i0_tst = test_data_pars.to_datetime_index().get_loc(
                plt_date_range_tst[0]
            )
            i1_tst = test_data_pars.to_datetime_index().get_loc(
                plt_date_range_tst[1]
            )
            assert isinstance(i0_tst, int)
            assert isinstance(i1_tst, int)
            i1_tst += 1
        else:
            i0_tst = i0_dl_tst
            i1_tst = i0_tst + test_data_pars.num_samples
        j0_tst = i0_tst - i0_dl_tst
        j1_tst = i1_tst - i0_dl_tst
    else:
        i0_tst, i1_tst, j0_tst, j1_tst = None, None, None, None

    def plot_eig(k: int):
        for ax in fig.axes:
            ax.cla()
        evec = eeof_eigen.pcs[k] + 1j * eeof_eigen.pcs[k + 1]
        evals = (
            eeof_eigen.sing_vals[k] ** 2,
            eeof_eigen.sing_vals[k + 1] ** 2,
        )
        if test_data is not None:
            match train_data, test_data:
                case Data(), Data():
                    evec_tst = efunc(train_data, eeof_eigen, test_data, k)
                case Data(), NPData():
                    evec_tst = efunc(
                        train_data,
                        eeof_eigen,
                        test_data.to_device(shardings=test_shardings),
                        k,
                    )
                case NPData(), Data():
                    evec_tst = efunc(
                        train_data.to_device(shardings=train_shardings),
                        eeof_eigen,
                        test_data,
                        k,
                    )
                case NPData(), NPData():
                    evec_tst = efunc(
                        train_data.to_device(shardings=train_shardings),
                        eeof_eigen,
                        test_data.to_device(shardings=test_shardings),
                        k,
                    )
        else:
            evec_tst = None

        ax = axs[0]
        ax.plot(evec.real[j0:j1:plt_step], evec.imag[j0:j1:plt_step], "-")
        ax.set_xlabel(f"$\\mathrm{{PC}}_{{{k}}}$")
        ax.set_ylabel(f"$\\mathrm{{PC}}_{{{k + 1}}}$")
        ax.set_title(
            f"Eigenvalues $(\\lambda_{{{k}}}, \\lambda_{{{k + 1}}}) = ({evals[0]: .3g}, {evals[1]: .3g})$"
        )
        ax.grid()

        ax = axs[1]
        ax.plot(
            data_pars.to_datetime_index()[i0:i1:plt_step],
            evec.real[j0:j1:plt_step],
            "-",
            label=f"$\\mathrm{{PC}}_{{{k}}}$",
        )
        ax.plot(
            data_pars.to_datetime_index()[i0:i1:plt_step],
            evec.imag[j0:j1:plt_step],
            "-",
            label=f"$\\mathrm{{PC}}_{{{k + 1}}}$",
        )
        ax.grid()
        ax.legend()

        if (
            axs_tst is not None
            and test_data_pars is not None
            and evec_tst is not None
        ):
            ax = axs_tst[0]
            ax.plot(
                evec_tst.real[j0_tst:j1_tst:plt_step_tst],
                evec_tst.imag[j0_tst:j1_tst:plt_step_tst],
                "-",
            )
            ax.set_xlabel(f"$\\mathrm{{PC}}_{{{k}}}$")
            ax.set_ylabel(f"$\\mathrm{{PC}}_{{{k + 1}}}$")
            ax.grid()

            ax = axs_tst[1]
            ax.plot(
                test_data_pars.to_datetime_index()[i0_tst:i1_tst:plt_step_tst],
                evec_tst.real[j0_tst:j1_tst:plt_step_tst],
                "-",
                label=f"$\\mathrm{{PC}}_{{{k}}}$",
            )
            ax.plot(
                test_data_pars.to_datetime_index()[i0_tst:i1_tst:plt_step_tst],
                evec_tst.imag[j0_tst:j1_tst:plt_step_tst],
                "-",
                label=f"$\\mathrm{{PC}}_{{{k + 1}}}$",
            )
            ax.grid()
            ax.legend()

    return fig, plot_eig


def make_running_pred_plotter[T: TimeSampling](
    test_data_pars: DataPars[T],
    test_data: ImplementsResponse,
    preds: Vtsts,
    plt_date_range_tst: tuple[str, str] | None = None,
    plt_step_tst: int = 1,
    i_fig: int = 1,
) -> tuple[Figure, F[int, None]]:
    """Make plotting function for prediction over different lead times."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    axs: Sequence[Axes]
    fig, axs = plt.subplots(
        1,
        2,
        num=i_fig,
        figsize=tuple(mpf.figaspect(0.5)),
        constrained_layout=True,
    )

    def plot_pred(i_step: int):
        i0_dl_tst = test_data_pars.delay_embedding_end
        if plt_date_range_tst is not None:
            match test_data_pars.time_sampling:
                case "daily":
                    freq_str = "D"
                case "monthly":
                    freq_str = "M"
            plt_periods_tst = pd.period_range(
                start=plt_date_range_tst[0],
                end=plt_date_range_tst[1],
                freq=freq_str,
            )
            i0_periods_tst = pd.period_range(
                start=test_data_pars.to_datetime_index()[0],
                end=plt_date_range_tst[0],
                freq=freq_str,
            )
            num_plt_tst = len(plt_periods_tst)
            i0_tst = i0_dl_tst + len(i0_periods_tst)
        else:
            num_plt_tst = test_data_pars.num_samples
            i0_tst = i0_dl_tst
        i1_tst = i0_tst + num_plt_tst
        i0_pred = i0_tst + i_step
        i1_pred = i1_tst + i_step
        j0_tst = i0_tst - test_data_pars.delay_embedding_end
        j1_tst = i1_tst - test_data_pars.delay_embedding_end
        err = (
            preds[j0_tst:j1_tst, i_step] - test_data.response[i0_pred:i1_pred]
        )
        for ax in axs:
            ax.cla()

        ax = axs[0]
        match test_data_pars.time_sampling:
            case "daily":
                timestep_str = "days"
            case "monthly":
                timestep_str = "months"
        ax.plot(
            test_data_pars.to_datetime_index()[i0_tst:i1_tst:plt_step_tst],
            test_data.response[i0_pred:i1_pred:plt_step_tst],
            "-",
            label="True",
        )
        ax.plot(
            test_data_pars.to_datetime_index()[i0_tst:i1_tst:plt_step_tst],
            preds[j0_tst:j1_tst:plt_step_tst, i_step],
            "-",
            label="Prediction",
        )
        ax.set_xlabel("Verification time")
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(test_data_pars.response.specs)
        ax.set_title(f"Prediction; lead time = {i_step} {timestep_str}")

        ax = axs[1]
        ax.plot(
            test_data_pars.to_datetime_index()[i0_tst:i1_tst:plt_step_tst],
            err[::plt_step_tst],
            "-",
        )
        ax.set_xlabel("Verification time")
        ax.set_title("Error")
        ax.grid(True)

    return fig, plot_pred


def make_pred_timeseries_plotter[T: TimeSampling](
    test_data_pars: DataPars[T],
    test_data: ImplementsResponse,
    preds: Vtsts,
    i_fig: int = 1,
) -> tuple[Figure, F[int, None]]:
    """Make plotting function over different initial conditions."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    num_pred_steps = preds.shape[1] - 1
    ts = jnp.arange(num_pred_steps + 1)
    match test_data_pars.time_sampling:
        case "daily":
            timestep_str = "days"
        case "monthly":
            timestep_str = "months"

    def plot_pred(i_init: int):
        i0_tst = test_data_pars.delay_embedding_end + i_init
        i1_tst = i0_tst + num_pred_steps + 1
        init_timestamp = test_data_pars.to_datetime_index()[i0_tst]
        ax.cla()
        ax.plot(ts, test_data.response[i0_tst:i1_tst], "o-", label="True")
        ax.plot(ts, preds[i_init, :], "o-", label="Prediction")
        ax.grid()
        ax.legend()
        ax.set_xlabel(f"Lead time ({timestep_str})")
        ax.set_title(f"Initialization time = {init_timestamp}")

    return fig, plot_pred


def plot_forecast_skill_scores[T: TimeSampling](
    data_pars: DataPars[T], scores: SkillScores, i_fig: int = 1
) -> Figure:
    """Plot NRMSE and ACC versus forecast lead time."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, axs = plt.subplots(
        2, 1, num=i_fig, constrained_layout=True, sharex=True
    )
    labels = ("NRMSE", "Anomaly correlation")
    num_pred_steps = len(scores["nrmses"]) - 1
    ts = jnp.arange(num_pred_steps + 1)
    match data_pars.time_sampling:
        case "daily":
            timestep_str = "days"
        case "monthly":
            timestep_str = "months"
    for ax, score, label in zip(
        axs, (scores["nrmses"], scores["accs"]), labels
    ):
        ax.plot(ts, score, "o-")
        ax.grid()
        if ax.get_subplotspec().is_first_row():
            ax.set_title(data_pars.response.specs)
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel(f"Lead time ({timestep_str})")
        ax.set_ylabel(label)
    return fig

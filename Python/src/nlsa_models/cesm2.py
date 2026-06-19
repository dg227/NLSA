"""Computation and plotting functions for analysis of CESM2 data."""

import jax
import jax.numpy as jnp
import matplotlib.figure as mpf
import matplotlib.pyplot as plt
import nlsa.abstract_algebra as alg
import nlsa.function_algebra as fun
import nlsa.jax.delays as dl
import nlsa.jax.koopman as koop
import nlsa.jax.stats as stats
import nlsa.jax.vector_algebra as vec
import numpy as np
import os
import pandas as pd
import xarray as xr
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum, nonmember
from functools import partial
from jax import Array, jit, vmap
from jax.sharding import Sharding
from jax.typing import DTypeLike
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from nlsa.io_actions import timeit
from nlsa.jax.kernels import (
    KernelEigenbasis,
)
from nlsa.jax.koopman import KoopmanEigenbasis, GeneratorShardings
from nlsa.jax.stats import anomaly_correlation_coefficient, normalized_rmse
from nlsa.jax.vector_algebra import L2FnAlgebra, L2FnAlgebraShardings
from nlsa_models.core import (
    JaxEnv as JaxEnv,
    SkillScores as SkillScores,
    initialize_jax as initialize_jax,
    initialize_matplotlib as initialize_matplotlib,
)
from numpy.typing import ArrayLike
from pandas import DataFrame, Series, Timestamp
from pathlib import Path
from tabulate import tabulate
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    TypedDict,
    assert_never,
    final,
    runtime_checkable,
)
from xarray import CFTimeIndex, Dataset

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
type Mat = Array  # Matrix
type V = Array  # Vector in L2
type Vs = Array  # Collection of vectors in L2
type Vtsts = Array  # Collection of vectors in L2 with respect to test dataset
type F[*Ss, T] = Callable[[*Ss], T]  # Shorthand for Callables
type CESMVar = IndoPacificVar | PacificVar | Nino34MeanVar
type NPVector[N: int, D: np.floating[Any]] = np.ndarray[tuple[N], np.dtype[D]]
type NPMatrix[M: int, N: int, D: np.floating[Any]] = np.ndarray[
    tuple[M, N], np.dtype[D]
]


@runtime_checkable
class ImplementsGriddedVar(Protocol):
    """Gridded variable protocol."""

    @property
    def input_name(self) -> str:
        """Return input_name property of ImplementsGriddedVar Protocol."""
        ...

    @property
    def min_lon(self) -> Optional[float]:
        """Return min_lon property of ImplementsGriddedVar Protocol."""
        ...

    @property
    def max_lon(self) -> Optional[float]:
        """Return max_lon property of ImplementsGriddedVar Protocol."""
        ...

    @property
    def min_lat(self) -> Optional[float]:
        """Return min_lat property of ImplementsGriddedVar Protocol."""
        ...

    @property
    def max_lat(self) -> Optional[float]:
        """Return max_lat property of ImplementsGriddedVar Protocol."""
        ...

    @property
    def spatial_mean(self) -> bool:
        """Return spatial_mean property of ImplementsGriddedVar Protocol."""
        ...


@final
class IndoPacificVar(StrEnum):
    """Indo-Pacific fields."""

    SST = "ip_sst"
    SSH = "ip_ssh"

    _min_lon = nonmember(None)
    _max_lon = nonmember(None)
    _min_lat = nonmember(None)
    _max_lat = nonmember(None)
    _spatial_mean = nonmember(False)

    @property
    def input_name(self) -> str:
        """Return variable name in input file of IndoPacificVar object."""
        match self:
            case IndoPacificVar.SST:
                return "tos"
            case IndoPacificVar.SSH:
                return "zos"
            case _ as unreachable:
                assert_never(unreachable)

    @property
    def min_lon(self) -> Optional[float]:
        """Minimum longitude of IndoPacificVar object."""
        return self._min_lon

    @property
    def max_lon(self) -> Optional[float]:
        """Minimum longitude of IndoPacificVar object."""
        return self._max_lon

    @property
    def min_lat(self) -> Optional[float]:
        """Minimum longitude of IndoPacificVar object."""
        return self._min_lat

    @property
    def max_lat(self) -> Optional[float]:
        """Minimum longitude of IndoPacificVar object."""
        return self._max_lat

    @property
    def spatial_mean(self) -> bool:
        """Minimum longitude of IndoPacificVar object."""
        return self._spatial_mean


@final
class PacificVar(StrEnum):
    """Pacific fields."""

    SST = "pac_sst"
    SSH = "pac_ssh"

    _min_lon = nonmember(140)
    _max_lon = nonmember(280)
    _min_lat = nonmember(-30)
    _max_lat = nonmember(30)
    _spatial_mean = nonmember(False)

    @property
    def input_name(self) -> str:
        """Return variable name in input file of PacificVar object."""
        match self:
            case PacificVar.SST:
                return "tos"
            case PacificVar.SSH:
                return "zos"
            case _ as unreachable:
                assert_never(unreachable)

    @property
    def min_lon(self) -> Optional[float]:
        """Minimum longitude of PacificVar object."""
        return self._min_lon

    @property
    def max_lon(self) -> Optional[float]:
        """Minimum longitude of PacificVar object."""
        return self._max_lon

    @property
    def min_lat(self) -> Optional[float]:
        """Minimum longitude of PacificVar object."""
        return self._min_lat

    @property
    def max_lat(self) -> Optional[float]:
        """Minimum longitude of PacificVar object."""
        return self._max_lat

    @property
    def spatial_mean(self) -> bool:
        """Minimum longitude of PacificVar object."""
        return self._spatial_mean


@final
class Nino34MeanVar(StrEnum):
    """Nino 3.4 area-averaged variables."""

    SST = "nino34av_sst"
    SSH = "nino34av_ssh"

    _min_lon = nonmember(360 - 170)
    _max_lon = nonmember(360 - 120)
    _min_lat = nonmember(-5)
    _max_lat = nonmember(5)
    _spatial_mean = nonmember(True)

    @property
    def input_name(self) -> str:
        """Return variable name in input file of Nino34MeanVar object."""
        match self:
            case Nino34MeanVar.SST:
                return "tos"
            case Nino34MeanVar.SSH:
                return "zos"
            case _ as unreachable:
                assert_never(unreachable)

    @property
    def min_lon(self) -> Optional[float]:
        """Minimum longitude of Nino34MeanVar object."""
        return self._min_lon

    @property
    def max_lon(self) -> Optional[float]:
        """Minimum longitude of Nino34MeanVar object."""
        return self._max_lon

    @property
    def min_lat(self) -> Optional[float]:
        """Minimum longitude of Nino34MeanVar object."""
        return self._min_lat

    @property
    def max_lat(self) -> Optional[float]:
        """Minimum longitude of Nino34MeanVar object."""
        return self._max_lat

    @property
    def spatial_mean(self) -> bool:
        """Minimum longitude of Nino34MeanVar object."""
        return self._spatial_mean


class Covariate(NamedTuple):
    """NamedTuple for covariate specification."""

    vars: list[CESMVar]
    remove_climatology: bool = False
    standardize: bool = False
    climatology_date_range: Optional[tuple[str, str]] = None

    def __str__(self) -> str:
        """Create string representation of covariate variable."""
        obs_vars = "_".join(self.vars)
        if self.climatology_date_range is not None:
            clim = "clim" + "_".join(self.climatology_date_range)
        else:
            clim = None
        anom = "anom" if self.remove_climatology else None
        std = "std" if self.standardize else None
        return "_".join(filter(None, (obs_vars, clim, anom, std)))


class Response(NamedTuple):
    """NamedTuple for response specification."""

    var: Nino34MeanVar
    remove_climatology: bool = False
    standardize: bool = False
    climatology_date_range: Optional[tuple[str, str]] = None

    def __str__(self) -> str:
        """Create string representation of response variable."""
        if self.climatology_date_range is not None:
            clim = "clim" + "_".join(self.climatology_date_range)
        else:
            clim = None
        anom = "anom" if self.remove_climatology else None
        std = "std" if self.standardize else None
        return "_".join(filter(None, (self.var, clim, anom, std)))


@dataclass(frozen=True, slots=True)
class DataPars:
    """Dataclass containing training and test data parameter values."""

    covariate: Covariate
    """Covariate function."""

    response: Response
    """Response function."""

    date_range: tuple[str, str]
    """Analysis time interval (in YYYY-MM-DD format)."""

    num_half_delays: int = 0
    """Half number of delays (to ensure even two-sided embedding window)."""

    delay_embedding_mode: Literal["explicit", "on_the_fly"] = "on_the_fly"
    """Delay embedding mmode."""

    num_before: int = 0
    """Number of extra samples before delay embedding."""

    num_after: int = 0
    """Number of extra after delay embedding."""

    velocity_covariate: bool = False
    """Include time tendencies (velocities) in covariate data."""

    velocity_fd_order: Optional[Literal[2, 4, 6, 8]] = None
    """Finite-difference order for velocity data."""

    eval_batch_size: Optional[int] = None
    """Number of batches for batchwise evaluation."""

    @property
    def num_total_samples(self) -> int:
        """Total number of samples in the analysis interval."""
        periods = pd.period_range(
            start=self.date_range[0], end=self.date_range[1], freq="M"
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
                    str(self.covariate),
                    str(self.response),
                    self.date_range[0],
                    self.date_range[1],
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


class Data(TypedDict):
    """TypedDict containing training data."""

    time: CFTimeIndex
    """Timestamps of the covariate/response data."""

    covariates: NPMatrix[int, int, np.floating[Any]]
    """Covariate variables."""

    responses: (
        NPVector[int, np.floating[Any]] | NPMatrix[int, int, np.floating[Any]]
    )
    """Response variables."""

    raw: Optional[Dataset]
    """Raw Nino 3.4 averaged data."""


def to_data_frame(
    pars: DataPars,
    koopman_basis: KoopmanEigenbasis[Yd, C, V, Cs, int | Array],
    timestamps: Series,
    which_eigs: int | tuple[int, int] | list[int],
    delay_timestamp_method: Literal["backward", "center"],
) -> DataFrame:
    """Extract Koopman eigenvectors to DataFrame."""
    match which_eigs:
        case int():
            idxs = jnp.arange(which_eigs)
        case (_, _):
            idxs = jnp.arange(which_eigs[0], which_eigs[1])
        case list():
            idxs = jnp.array(which_eigs)
    match delay_timestamp_method:
        case "backward":
            i0 = pars.delay_embedding_end
        case "center":
            i0 = pars.delay_embedding_center
    i1 = i0 + pars.num_samples
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


def to_skill_scores(
    dict_in: dict[str, ArrayLike], dtype: Optional[DTypeLike] = None
) -> SkillScores:
    """Convert dict of numpy ArrayLike objects to SkillScores TypedDict."""
    try:
        skill_scores: SkillScores = {
            "nrmses": jnp.array(dict_in["nrmses"], dtype),
            "accs": jnp.array(dict_in["accs"], dtype),
        }
        return skill_scores
    except ValueError as exc:
        raise ValueError("Incompatible keys/values") from exc


@timeit
def make_dataset(
    vars: ImplementsGriddedVar
    | Sequence[ImplementsGriddedVar] = IndoPacificVar.SST,
    date_range: Optional[tuple[str, str]] = None,
    standardize: bool = False,
    climatology_date_range: Optional[
        tuple[str, str] | tuple[Timestamp, Timestamp]
    ] = None,
    remove_climatology: bool = True,
    root_dir: Optional[str | Path] = None,
) -> Dataset:
    """Import data from NetCDF files into Xarray dataset."""
    # Open the NetCDF files
    if root_dir is None:
        pth = "*.nc"
    else:
        pth = Path(root_dir) / "*.nc"
    ds = xr.open_mfdataset(os.fspath(pth), parallel=False)

    # Build lists of input and output variables
    if isinstance(vars, ImplementsGriddedVar):
        _vars = [vars]
    else:
        _vars = vars

    # Extract and process input variables from input dataset
    for var in _vars:
        da = ds[var.input_name].sel(
            lon=slice(var.min_lon, var.max_lon),
            lat=slice(var.min_lat, var.max_lat),
        )
        if var.spatial_mean:
            ds[var] = da.mean(dim=("lon", "lat"))
        else:
            ds[var] = da

    # Assign default date range and climatology date range if not provided
    if date_range is not None:
        _date_range = date_range
    else:
        _date_range = (ds["time"].values[0], ds["time"].values[-1])
    if climatology_date_range is not None:
        _climatology_date_range = climatology_date_range
    else:
        _climatology_date_range = _date_range

    # Compute climatology if needed
    if remove_climatology:
        climatology_means = (
            ds[_vars]
            .sel(
                time=slice(
                    _climatology_date_range[0], _climatology_date_range[1]
                )
            )
            .groupby("time.month")
            .mean(dim="time")
        )

    # Extract requested date range
    if date_range is not None:
        ds = ds[_vars].sel(time=slice(_date_range[0], _date_range[1]))
    assert isinstance(ds, Dataset)

    # Remove climatology if requested
    if remove_climatology:
        ds = ds.groupby("time.month") - climatology_means

    # Standardize if requested
    if standardize:
        anomalies = ds[_vars] - ds[_vars].mean(dim="time")
        spatial_vars = [var for var in _vars if not var.spatial_mean]
        energies = anomalies**2
        for var in spatial_vars:
            space = [dim for dim in energies[var].dims if dim != "time"]
            energies[var] = energies[var].sum(dim=space)
        anomalies = anomalies / np.sqrt(energies.mean(dim="time"))
        if len(_vars) == 1:
            # I am not sure why, but xarray raises a TypeError if I don't do
            # this.
            ds[_vars[0]] = anomalies[_vars[0]]
        else:
            ds[_vars] = anomalies

    return ds


def extract_data_array[D: np.floating](
    var: ImplementsGriddedVar,
    ds: Dataset,
    dtype: Optional[type[D]] = None,
    atleast_2d: bool = False,
) -> NPVector[int, D] | NPMatrix[int, int, D]:
    """Extract data array from xarray dataset associated with a variable."""
    if var.spatial_mean:
        a = ds[var].transpose("time", ...).astype(dtype).to_numpy()
    else:
        space = [dim for dim in ds[var].dims if dim != "time"]
        a = (
            ds[var]
            .stack(space=space)
            .dropna(dim="space")
            .astype(dtype)
            .to_numpy()
        )
    if atleast_2d and len(a.shape) < 2:
        a = a.reshape((-1, 1))
    return a


def generate_data(
    pars: DataPars,
    dtype: Optional[type[np.floating[Any]]] = None,
    output_raw: bool | Literal["anomalies"] = False,
    root_dir: Optional[str | Path] = None,
) -> Data:
    """Extract CESM2 data."""
    print("Reading covariates:")
    print(pars.covariate.vars)
    ds = make_dataset(
        vars=pars.covariate.vars,
        date_range=pars.date_range,
        climatology_date_range=pars.covariate.climatology_date_range,
        remove_climatology=pars.covariate.remove_climatology,
        standardize=pars.covariate.standardize,
        root_dir=root_dir,
    )
    covariates = np.hstack(
        [
            extract_data_array(var, ds, dtype=dtype, atleast_2d=True)
            for var in pars.covariate.vars
        ]
    )
    print(f"Covariates array shape: {covariates.shape}")
    time = ds.indexes["time"]
    assert isinstance(time, CFTimeIndex)
    print("Reading response:")
    print(pars.response.var)
    ds = make_dataset(
        vars=pars.response.var,
        date_range=pars.date_range,
        climatology_date_range=pars.response.climatology_date_range,
        remove_climatology=pars.response.remove_climatology,
        standardize=pars.response.standardize,
        root_dir=root_dir,
    )
    response = extract_data_array(
        pars.response.var, ds, dtype=dtype, atleast_2d=False
    )
    print(f"Response array shape: {response.shape}")
    match output_raw:
        case True:
            raw_vars = [var for var in Nino34MeanVar]
            print("Reading raw data:")
            print(raw_vars)
            raw_ds = make_dataset(
                raw_vars,
                date_range=pars.date_range,
                root_dir=root_dir,
            )
        case "anomalies":
            raw_vars = [var for var in Nino34MeanVar]
            print("Reading raw anomaly data:")
            print(raw_vars)
            if pars.covariate.remove_climatology:
                raw_climatology_date_range = (
                    pars.covariate.climatology_date_range
                )
            elif pars.response.remove_climatology:
                raw_climatology_date_range = (
                    pars.response.climatology_date_range
                )
            else:
                raw_climatology_date_range = None
            raw_ds = make_dataset(
                [var for var in Nino34MeanVar],
                date_range=pars.date_range,
                root_dir=root_dir,
                remove_climatology=True,
                climatology_date_range=raw_climatology_date_range,
            )
        case _:
            raw_ds = None
    if pars.velocity_covariate:
        assert pars.velocity_fd_order is not None
        fd_op = jit(
            vmap(
                dl.make_fd_operator(
                    order=pars.velocity_fd_order, mode="central"
                ),
                in_axes=-1,
                out_axes=-1,
            )
        )
        vs = fd_op(covariates)
        data: Data = {
            "time": time,
            "covariates": np.stack((covariates, vs), axis=1).astype(dtype),
            "responses": response,
            "raw": raw_ds,
        }
    else:
        data: Data = {
            "time": time,
            "covariates": covariates,
            "responses": response,
            "raw": raw_ds,
        }
    return data


def make_l2_space[D: DTypeLike](
    pars: DataPars,
    dtype: D,
    data: Data,
    delay_embedding_mode: Literal["explicit", "on_the_fly"] = "on_the_fly",
    shardings: L2FnAlgebraShardings = L2FnAlgebraShardings(),
    jit: bool = False,
) -> L2FnAlgebra[tuple[int], D, Yd, R]:
    """Make L2 space over covariate data space."""
    i0 = pars.delay_embedding_origin
    i1 = i0 + pars.num_delay_samples
    incl: Callable[[F[Yd, R]], V]
    if pars.num_half_delays > 0:
        match delay_embedding_mode:
            case "on_the_fly":
                incl = dl.delay_eval_at(
                    jnp.asarray(
                        data["covariates"][i0:i1],
                        dtype=dtype,
                        device=shardings.data,
                    ),
                    num_delays=pars.num_delays,
                    batch_size=pars.eval_batch_size,
                    out_sharding=shardings.vectors,
                    jit=jit,
                )
            case "explicit":
                if pars.velocity_covariate:
                    hankel = jax.jit(
                        vmap(
                            partial(
                                dl.hankel,
                                num_delays=pars.num_delays,
                                flatten=True,
                            ),
                            in_axes=1,
                            out_axes=1,
                        )
                    )
                else:
                    hankel = jax.jit(
                        partial(
                            dl.hankel, num_delays=pars.num_delays, flatten=True
                        )
                    )
                incl = vec.batch_eval_at(
                    jnp.asarray(
                        hankel(data["covariates"][i0:i1]),
                        dtype=dtype,
                        device=shardings.data,
                    ),
                    batch_size=pars.eval_batch_size,
                    out_sharding=shardings.vectors,
                    jit=jit,
                )
    else:
        incl = vec.batch_eval_at(
            jnp.asarray(
                data["covariates"][i0:i1], dtype=dtype, device=shardings.data
            ),
            batch_size=pars.eval_batch_size,
            out_sharding=shardings.vectors,
            jit=jit,
        )
    mu = vec.make_normalized_counting_measure(pars.num_samples)
    return L2FnAlgebra(
        shape=(pars.num_samples,),
        dtype=dtype,
        measure=mu,
        inclusion_map=incl,
        sharding=shardings.vectors,
    )


def make_tangent_evaluation_functional_fd(
    pars: DataPars,
    dtype: DTypeLike,
    data: Data,
    fd_order: Literal[2, 4, 6, 8],
    delay_embedding_mode: Literal["explicit", "on_the_fly"] = "on_the_fly",
    shardings: L2FnAlgebraShardings = L2FnAlgebraShardings(),
    jit: bool = False,
) -> Callable[[F[Yd, TYd, R]], V]:
    """Make evaluation using finite-difference approx. of the generator."""
    if pars.velocity_covariate:
        fd_op = jax.jit(
            vmap(
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
        )
    else:
        fd_op = jax.jit(
            vmap(
                dl.make_fd_operator(
                    order=fd_order,
                    mode="central",
                    dt=1,
                    extrap=False,
                ),
                in_axes=-1,
                out_axes=-1,
            )
        )
    num_half_fd = fd_order // 2
    i0 = pars.delay_embedding_origin
    i1 = i0 + pars.num_delay_samples
    i0_fd = i0 - num_half_fd
    i1_fd = i1 + num_half_fd
    eval_tx: Callable[[F[Yd, TYd, R]], V]
    if pars.num_half_delays > 0:
        match delay_embedding_mode:
            case "on_the_fly":
                eval_tx = dl.delay_eval_at(
                    (
                        jnp.asarray(
                            data["covariates"][i0:i1],
                            dtype=dtype,
                            device=shardings.data,
                        ),
                        jnp.asarray(
                            fd_op(data["covariates"][i0_fd:i1_fd]),
                            dtype=dtype,
                            device=shardings.data,
                        ),
                    ),
                    num_delays=pars.num_delays,
                    batch_size=pars.eval_batch_size,
                    out_sharding=shardings.vectors,
                    jit=jit,
                )
            case "explicit":
                if pars.velocity_covariate:
                    hankel = jax.jit(
                        vmap(
                            partial(
                                dl.hankel,
                                num_delays=pars.num_delays,
                                flatten=True,
                            ),
                            in_axes=1,
                            out_axes=1,
                        )
                    )
                else:
                    hankel = jax.jit(
                        partial(
                            dl.hankel,
                            num_delays=pars.num_delays,
                            flatten=True,
                        )
                    )
                eval_tx = vec.batch_eval_at(
                    (
                        jnp.asarray(
                            hankel(data["covariates"][i0:i1]),
                            dtype=dtype,
                            device=shardings.data,
                        ),
                        jnp.asarray(
                            hankel(fd_op(data["covariates"][i0_fd:i1_fd])),
                            dtype=dtype,
                            device=shardings.data,
                        ),
                    ),
                    batch_size=pars.eval_batch_size,
                    out_sharding=shardings.vectors,
                    jit=jit,
                )
    else:
        eval_tx = vec.batch_eval_at(
            (
                jnp.asarray(
                    data["covariates"][i0:i1],
                    dtype=dtype,
                    device=shardings.data,
                ),
                jnp.asarray(
                    fd_op(data["covariates"][i0_fd:i1_fd]),
                    dtype=dtype,
                    device=shardings.data,
                ),
            ),
            batch_size=pars.eval_batch_size,
            out_sharding=shardings.vectors,
            jit=jit,
        )
    return eval_tx


def compute_generator_matrix[D: DTypeLike](
    pars: DataPars,
    fd_order: Literal[2, 4, 6, 8],
    l2_space: L2FnAlgebra[tuple[int], D, Yd, R],
    train_data: Data,
    basis: alg.ImplementsDimensionedL2FnFrame[Yd, R, V, Rs, int | Array],
    grad_batch_size: Optional[int] = None,
    gram_batch_size: Optional[int] = None,
    shardings: GeneratorShardings = GeneratorShardings(),
    jit: bool = True,
) -> Mat:
    """Compute generator matrix representation using finite differences."""
    eval_tx = make_tangent_evaluation_functional_fd(
        pars,
        dtype=l2_space.dtype,
        data=train_data,
        fd_order=fd_order,
        delay_embedding_mode=pars.delay_embedding_mode,
        shardings=shardings.tangents,
        jit=jit,
    )
    basis_idxs = jnp.arange(1, basis.dim)
    gen_mat = koop.compute_generator_matrix(
        l2_space,
        eval_tangents=eval_tx,
        basis=basis,
        basis_idxs=basis_idxs,
        grad_batch_size=grad_batch_size,
        gram_batch_size=gram_batch_size,
    )
    return gen_mat


def make_koopman_prediction_function(
    pars: DataPars,
    train_data: Data,
    koopman_basis: KoopmanEigenbasis[Yd, C, V, Cs, int | Array],
    shardings: Optional[Sharding] = None,
) -> F[R, Yd, Cs]:
    """Make prediction function for time series prediction."""
    i0 = pars.delay_embedding_end
    i1 = i0 + pars.num_samples
    f_coeffs = koopman_basis.anal(
        jnp.asarray(train_data["responses"][i0:i1], device=shardings)
    )

    def predict(t: R, y: Yd) -> C:
        phases = jnp.exp(1j * koopman_basis.efreqs * t)
        return koopman_basis.fn_synth(phases * f_coeffs)(y)

    return predict


def compute_response_skill_scores(
    pars: DataPars,
    test_data: Data,
    fys_pred: Vtsts,
    dropna: bool = False,
) -> SkillScores:
    """Compute NRMSE and ACC skill scores over the prediction ensemble."""
    num_pred_steps = fys_pred.shape[1] - 1
    i0 = pars.delay_embedding_end
    i1 = i0 + num_pred_steps + pars.num_samples
    hankel = jax.jit(partial(dl.hankel, num_delays=num_pred_steps))
    fxs_true = hankel(test_data["responses"][i0:i1])
    if dropna:
        mask = ~jnp.isnan(fys_pred).any(axis=1)
        fys_pred = fys_pred[mask]
        fxs_true = fxs_true[mask]
    nrmses = jax.jit(vmap(normalized_rmse, in_axes=1))(fxs_true, fys_pred)
    accs = jax.jit(vmap(anomaly_correlation_coefficient, in_axes=1))(
        fxs_true, fys_pred
    )
    scores: SkillScores = {"nrmses": nrmses, "accs": accs}
    return scores


def compute_covariate_skill_scores(
    pars: DataPars, test_data: Data, ys_pred: Vtsts, dropna: bool = False
) -> SkillScores:
    """Compute NRMSE and ACC skill scores over the prediction ensemble."""
    num_pred_steps = len(ys_pred) - 1
    i0 = pars.delay_embedding_end
    i1 = i0 + num_pred_steps + pars.num_samples
    hankel = jax.jit(
        fun.compose(
            partial(jnp.swapaxes, axis1=0, axis2=1),
            vmap(
                partial(dl.hankel, num_delays=num_pred_steps),
                in_axes=-1,
                out_axes=-1,
            ),
        )
    )
    normalized_rmses = jax.jit(
        vmap(vmap(stats.normalized_rmse, in_axes=1), in_axes=2)
    )
    anomaly_correlation_coefficients = jax.jit(
        vmap(vmap(stats.anomaly_correlation_coefficient, in_axes=1), in_axes=2)
    )
    ys_true = hankel(test_data["covariates"][i0:i1])
    if dropna:
        mask = ~jnp.isnan(ys_pred).any(axis=(1, 2))
        ys_pred = ys_pred[mask]
        ys_true = ys_true[mask]
    nrmses = normalized_rmses(ys_true, ys_pred)
    accs = anomaly_correlation_coefficients(ys_true, ys_pred)
    scores: SkillScores = {"nrmses": nrmses, "accs": accs}
    return scores


def compute_skill_scores(
    pars: DataPars,
    test_data: Data,
    preds: Vtsts,
    what: Literal["covariates", "responses"] = "responses",
    dropna: bool = False,
) -> SkillScores:
    """Compute NRMSE and ACC skill scores over the prediction ensemble."""
    match what:
        case "covariates":
            scores = compute_covariate_skill_scores(
                pars, test_data, preds, dropna=dropna
            )
        case "responses":
            scores = compute_response_skill_scores(
                pars, test_data, preds, dropna=dropna
            )
    return scores


def plot_bandwidth_function[D: DTypeLike](
    pars: DataPars,
    l2y: L2FnAlgebra[tuple[int], D, Yd, R],
    bandwidth_func: F[Yd, R],
    train_data: Data,
    test_pars: Optional[DataPars] = None,
    l2y_tst: Optional[L2FnAlgebra[tuple[int], D, Yd, R]] = None,
    test_data: Optional[Data] = None,
    delay_plot_mode: Literal["backward", "central"] = "central",
    plt_date_range: Optional[tuple[str, str]] = None,
    plt_date_range_tst: Optional[tuple[str, str]] = None,
    plt_step: int = 1,
    plt_step_tst: int = 1,
    i_fig: int = 1,
) -> Figure:
    """Plot bandwidth function on training and, optionally, test data."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    if test_pars is not None:
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
    match delay_plot_mode:
        case "backward":
            i0_dl = pars.delay_embedding_end
        case "central":
            i0_dl = pars.delay_embedding_center
    if plt_date_range is not None:
        i0 = train_data["time"].get_loc(plt_date_range[0])
        i1 = train_data["time"].get_loc(plt_date_range[1])
        assert isinstance(i0, int)
        assert isinstance(i1, int)
        i1 += 1
    else:
        i0 = i0_dl
        i1 = i0 + pars.num_samples
    j0 = i0 - i0_dl
    j1 = i1 - i0_dl
    bw_vals = l2y.incl(bandwidth_func)
    ax.plot(
        train_data["time"][i0:i1:plt_step].values,
        bw_vals[j0:j1:plt_step],
        "-",
    )
    ax.grid(True)
    ax.set_title("Kernel bandwidth function (training)")

    if test_pars is not None and l2y_tst is not None and test_data is not None:
        match delay_plot_mode:
            case "backward":
                i0_dl_tst = test_pars.delay_embedding_end
            case "central":
                i0_dl_tst = test_pars.delay_embedding_center
        if plt_date_range_tst is not None:
            i0_tst = train_data["time"].get_loc(plt_date_range_tst[0])
            i1_tst = train_data["time"].get_loc(plt_date_range_tst[1])
            assert isinstance(i0_tst, int)
            assert isinstance(i1_tst, int)
            i1_tst += 1
        else:
            i0_tst = i0_dl_tst
            i1_tst = i0_tst + test_pars.num_samples
        j0_tst = i0_tst - i0_dl_tst
        j1_tst = i1_tst - i0_dl_tst
        bw_vals_tst = l2y_tst.incl(bandwidth_func)
        ax_tst.plot(
            test_data["time"][i0_tst:i1_tst:plt_step_tst].values,
            bw_vals_tst[j0_tst:j1_tst:plt_step_tst],
            "-",
        )
        ax_tst.grid(True)
        ax_tst.set_title("Kernel bandwidth function (test)")
    return fig


def make_kernel_evecs_plotter[D: DTypeLike](
    pars: DataPars,
    train_data: Data,
    kernel_basis: KernelEigenbasis[Yd, R, V, Rs, int | Array],
    test_pars: Optional[DataPars] = None,
    l2y_tst: Optional[L2FnAlgebra[tuple[int], D, Yd, R]] = None,
    test_data: Optional[Data] = None,
    delay_plot_mode: Literal["backward", "central"] = "backward",
    plt_date_range: Optional[tuple[str, str]] = None,
    plt_date_range_tst: Optional[tuple[str, str]] = None,
    plt_step: int = 1,
    plt_step_tst: int = 1,
    i_fig: int = 1,
) -> tuple[Figure, F[int, None]]:
    """Make plotting function for kernel eigenfunctions."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    if test_pars is not None:
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
    match delay_plot_mode:
        case "backward":
            i0_dl = pars.delay_embedding_end
        case "central":
            i0_dl = pars.delay_embedding_center
    if plt_date_range is not None:
        i0 = train_data["time"].get_loc(plt_date_range[0])
        i1 = train_data["time"].get_loc(plt_date_range[1])
        assert isinstance(i0, int)
        assert isinstance(i1, int)
        i1 += 1
    else:
        i0 = i0_dl
        i1 = i0 + pars.num_samples
    j0 = i0 - i0_dl
    j1 = i1 - i0_dl
    if test_pars is not None and test_data is not None:
        match delay_plot_mode:
            case "backward":
                i0_dl_tst = test_pars.delay_embedding_end
            case "central":
                i0_dl_tst = test_pars.delay_embedding_center
        if plt_date_range_tst is not None:
            i0_tst = train_data["time"].get_loc(plt_date_range_tst[0])
            i1_tst = train_data["time"].get_loc(plt_date_range_tst[1])
            assert isinstance(i0_tst, int)
            assert isinstance(i1_tst, int)
            i1_tst += 1
        else:
            i0_tst = i0_dl_tst
            i1_tst = i0_tst + test_pars.num_samples
        j0_tst = i0_tst - i0_dl_tst
        j1_tst = i1_tst - i0_dl_tst

    def plot_eig(k: int):
        for figax in fig.axes:
            figax.cla()
        evec = kernel_basis.vec(k)
        ax.plot(train_data["time"][i0:i1:plt_step], evec[j0:j1:plt_step], "-")
        eta = kernel_basis.lapl_evl(k)
        ax.grid()
        ax.set_title(f"Eigenvector {k}: $\\eta_{{{k}}} = {eta: .3f}$")
        if (
            test_pars is not None
            and l2y_tst is not None
            and test_data is not None
            and kernel_basis is not None
        ):
            evec_tst = l2y_tst.incl(kernel_basis.fn(k))
            ax_tst.plot(
                test_data["time"][i0_tst:i1_tst:plt_step_tst],
                evec_tst[j0_tst:j1_tst:plt_step_tst],
                "-",
            )
            ax_tst.grid()
            ax_tst.set_title("Nystrom")

    return fig, plot_eig


def make_koopman_evecs_plotter[D: DTypeLike](
    pars: DataPars,
    train_data: Data,
    koopman_basis: KoopmanEigenbasis[Yd, C, V, Cs, int | Array],
    test_pars: Optional[DataPars] = None,
    l2y_tst: Optional[L2FnAlgebra[tuple[int], D, Yd, R]] = None,
    test_data: Optional[Data] = None,
    delay_plot_mode: Literal["backward", "central"] = "backward",
    plt_date_range: Optional[tuple[str, str]] = None,
    plt_date_range_tst: Optional[tuple[str, str]] = None,
    plt_step: int = 1,
    plt_step_tst: int = 1,
    i_fig: int = 1,
) -> tuple[Figure, F[int, None]]:
    """Make plotting function for Koopman eigenfunctions."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    if test_pars is not None:
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
    match delay_plot_mode:
        case "backward":
            i0_dl = pars.delay_embedding_end
        case "central":
            i0_dl = pars.delay_embedding_center
    if plt_date_range is not None:
        i0 = train_data["time"].get_loc(plt_date_range[0])
        i1 = train_data["time"].get_loc(plt_date_range[1])
        assert isinstance(i0, int)
        assert isinstance(i1, int)
        i1 += 1
    else:
        i0 = i0_dl
        i1 = i0 + pars.num_samples
    j0 = i0 - i0_dl
    j1 = i1 - i0_dl
    if test_pars is not None and test_data is not None:
        match delay_plot_mode:
            case "backward":
                i0_dl_tst = test_pars.delay_embedding_end
            case "central":
                i0_dl_tst = pars.delay_embedding_center
        if plt_date_range_tst is not None:
            i0_tst = train_data["time"].get_loc(plt_date_range_tst[0])
            i1_tst = train_data["time"].get_loc(plt_date_range_tst[1])
            assert isinstance(i0_tst, int)
            assert isinstance(i1_tst, int)
            i1_tst += 1
        else:
            i0_tst = i0_dl_tst
            i1_tst = i0_tst + test_pars.num_samples
        j0_tst = i0_tst - i0_dl_tst
        j1_tst = i1_tst - i0_dl_tst

    def plot_eig(k: int):
        for ax in fig.axes:
            ax.cla()
        evec = koopman_basis.vec(k)
        efreq = koopman_basis.efreq(k) / (2 * jnp.pi) * 12
        eperiod = koopman_basis.eperiod(k) / 12

        ax = axs[0]
        ax.plot(evec.real[j0:j1:plt_step], evec.imag[j0:j1:plt_step], "-")
        ax.set_xlabel(f"$\\mathrm{{Re}}\\zeta_{{{k}}}$")
        ax.set_ylabel(f"$\\mathrm{{Im}}\\zeta_{{{k}}}$")
        ax.set_title(
            f"Eigenfrequency $\\nu_{{{k}}} = {efreq: .3f}$ cycles/year"
        )
        ax.grid()

        ax = axs[1]
        ax.plot(
            train_data["time"][i0:i1:plt_step],
            evec.real[j0:j1:plt_step],
            "-",
            label=f"$\\mathrm{{Re}}\\zeta_{{{k}}}$",
        )
        ax.plot(
            train_data["time"][i0:i1:plt_step],
            evec.imag[j0:j1:plt_step],
            "-",
            label=f"$\\mathrm{{Im}}\\zeta_{{{k}}}$",
        )
        ax.set_title(f"Eigenperiod $T_{{{k}}} = {eperiod: .3f}$ years")
        ax.grid()
        ax.legend()

        if (
            test_pars is not None
            and l2y_tst is not None
            and test_data is not None
            and koopman_basis is not None
        ):
            evec_tst = l2y_tst.incl(koopman_basis.fn(k))

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
                test_data["time"][i0_tst:i1_tst:plt_step_tst],
                evec_tst.real[j0_tst:j1_tst:plt_step_tst],
                "-",
                label=f"$\\mathrm{{Re}}\\zeta_{{{k}}}$",
            )
            ax.plot(
                test_data["time"][i0_tst:i1_tst:plt_step_tst],
                evec_tst.imag[j0_tst:j1_tst:plt_step_tst],
                "-",
                label=f"$\\mathrm{{Im}}\\zeta_{{{k}}}$",
            )
            ax.grid()
            ax.legend()

    return fig, plot_eig


def make_koopman_lifecycle_plotter[D: DTypeLike](
    pars: DataPars,
    l2y: L2FnAlgebra[tuple[int], D, Yd, R],
    train_data: Data,
    koopman_basis: KoopmanEigenbasis[Yd, C, V, Cs, int | Array],
    delay_plot_mode: Literal["backward", "central"] = "backward",
    plt_date_range: Optional[tuple[str, str]] = None,
    plt_step: int = 1,
    center_colormap: bool = False,
    i_fig: int = 1,
) -> tuple[Figure, F[int, None]]:
    """Make plotting function for [T, h] lifecycle."""
    xs = train_data["raw"]
    assert xs is not None
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    figsize = plt.rcParams["figure.figsize"]
    fig, axs = plt.subplots(
        2,
        2,
        num=i_fig,
        figsize=(1.5 * figsize[0], 1.75 * figsize[1]),
        constrained_layout=True,
        sharey="row",
    )
    for ax in axs.flat:
        ax.set_box_aspect(1)
    match delay_plot_mode:
        case "backward":
            i0_dl = pars.delay_embedding_end
        case "central":
            i0_dl = pars.delay_embedding_center
    if plt_date_range is not None:
        i0 = train_data["time"].get_loc(plt_date_range[0])
        i1 = train_data["time"].get_loc(plt_date_range[1])
        assert isinstance(i0, int)
        assert isinstance(i1, int)
        i1 += 1
    else:
        i0 = i0_dl
        i1 = i0 + pars.num_samples
    j0 = i0 - i0_dl
    j1 = i1 - i0_dl
    var = xs[[Nino34MeanVar.SST, Nino34MeanVar.SSH]].var()
    vmax = xs[[Nino34MeanVar.SST, Nino34MeanVar.SSH]].max()
    vmin = xs[[Nino34MeanVar.SST, Nino34MeanVar.SSH]].min()

    def plot_lifecycle(k: int):
        for ax in fig.axes:
            ax.cla()
        evec = koopman_basis.vec(k)
        dual_evec = koopman_basis.dual_vec(k)
        efreq = koopman_basis.efreq(k) / (2 * jnp.pi) * 12
        eperiod = koopman_basis.eperiod(k) / 12
        amax = max(
            float(jnp.max(jnp.abs(evec.real))),
            float(jnp.max(jnp.abs(evec.imag))),
        )
        r2_t = jnp.abs(
            l2y.innerp(
                dual_evec,
                jnp.asarray(
                    xs[Nino34MeanVar.SST]
                    .isel(time=slice(i0_dl, i0_dl + pars.num_samples))
                    .to_numpy()
                ),
            )
        ) ** 2 / jnp.asarray(var[Nino34MeanVar.SST].to_numpy())
        r2_h = jnp.abs(
            l2y.innerp(
                dual_evec,
                jnp.asarray(
                    xs[Nino34MeanVar.SSH]
                    .isel(time=slice(i0_dl, i0_dl + pars.num_samples))
                    .to_numpy()
                ),
            )
        ) ** 2 / jnp.asarray(var[Nino34MeanVar.SSH].to_numpy())

        ax_re = axs[0, 0]
        ax_re.plot(
            xs[Nino34MeanVar.SST].isel(time=slice(j0, j1, plt_step)),
            xs[Nino34MeanVar.SSH].isel(time=slice(j0, j1, plt_step)),
            color="lightgreen",
            linewidth=0.5,
        )
        sc_re = ax_re.scatter(
            xs[Nino34MeanVar.SST].isel(time=slice(j0, j1, plt_step)),
            xs[Nino34MeanVar.SSH].isel(time=slice(j0, j1, plt_step)),
            c=evec.real[j0:j1:plt_step],
            s=8,
            vmin=-amax,
            vmax=amax,
            cmap="seismic",
        )
        ax_re.set_xlabel("$T$")
        ax_re.set_ylabel("$h$")
        ax_re.set_title(
            f"Eigenfrequency $\\nu_{{{k}}} = {efreq: .3f}$ cycles/year"
        )

        ax_im = axs[0, 1]
        ax_im.plot(
            xs[Nino34MeanVar.SST].isel(time=slice(j0, j1, plt_step)),
            xs[Nino34MeanVar.SSH].isel(time=slice(j0, j1, plt_step)),
            color="lightgreen",
            linewidth=0.5,
        )
        sc_im = ax_im.scatter(
            xs[Nino34MeanVar.SST].isel(time=slice(j0, j1, plt_step)),
            xs[Nino34MeanVar.SSH].isel(time=slice(j0, j1, plt_step)),
            c=evec.imag[j0:j1:plt_step],
            s=8,
            vmin=-amax,
            vmax=amax,
            cmap="seismic",
        )
        ax_im.set_xlabel("$T$")
        ax_im.set_ylabel("$h$")
        ax_im.set_title(f"Eigenperiod $T_{{{k}}} = {eperiod: .3f}$ years")

        ax_t = axs[1, 0]
        ax_t.plot(
            evec.real[j0:j1:plt_step],
            evec.imag[j0:j1:plt_step],
            color="lightgreen",
            linewidth=0.5,
        )
        if center_colormap:
            sc_t = ax_t.scatter(
                evec.real[j0:j1:plt_step],
                evec.imag[j0:j1:plt_step],
                c=xs[Nino34MeanVar.SST].isel(time=slice(j0, j1, plt_step)),
                s=8,
                norm=TwoSlopeNorm(
                    vcenter=0,
                    vmin=float(vmin[Nino34MeanVar.SST].values),
                    vmax=float(vmax[Nino34MeanVar.SST].values),
                ),
                cmap="seismic",
            )
        else:
            sc_t = ax_t.scatter(
                evec.real[j0:j1:plt_step],
                evec.imag[j0:j1:plt_step],
                c=xs[Nino34MeanVar.SST].isel(time=slice(j0, j1, plt_step)),
                s=8,
                vmin=vmin[Nino34MeanVar.SST],
                vmax=vmax[Nino34MeanVar.SST],
                cmap="seismic",
            )
        ax_t.set_xlabel(f"$\\mathrm{{Re}}\\zeta_{{{k}}}$")
        ax_t.set_ylabel(f"$\\mathrm{{Im}}\\zeta_{{{k}}}$")
        ax_t.set_title(f"$R^2_T = {r2_t:.3f}$")

        ax_h = axs[1, 1]
        ax_h.plot(
            evec.real[j0:j1:plt_step],
            evec.imag[j0:j1:plt_step],
            color="lightgreen",
            linewidth=0.5,
        )
        if center_colormap:
            sc_h = ax_h.scatter(
                evec.real[j0:j1:plt_step],
                evec.imag[j0:j1:plt_step],
                c=xs[Nino34MeanVar.SSH].isel(time=slice(j0, j1, plt_step)),
                s=8,
                norm=TwoSlopeNorm(
                    vcenter=0,
                    vmin=float(vmin[Nino34MeanVar.SSH].values),
                    vmax=float(vmax[Nino34MeanVar.SSH].values),
                ),
                cmap="seismic",
            )
        else:
            sc_h = ax_h.scatter(
                evec.real[j0:j1:plt_step],
                evec.imag[j0:j1:plt_step],
                c=xs[Nino34MeanVar.SSH].isel(time=slice(j0, j1, plt_step)),
                s=8,
                vmin=vmin[Nino34MeanVar.SSH],
                vmax=vmax[Nino34MeanVar.SSH],
                cmap="seismic",
            )
        ax_h.set_xlabel(f"$\\mathrm{{Re}}\\zeta_{{{k}}}$")
        ax_h.set_ylabel(f"$\\mathrm{{Im}}\\zeta_{{{k}}}$")
        ax_h.set_title(f"$R^2_h = {r2_h:.3f}$")

        if len(fig.axes) > 4:
            fig.colorbar(
                sc_re,
                ax=ax_re,
                cax=fig.axes[4],
                label=f"$\\mathrm{{Re}}\\zeta_{{{k}}}$",
                location="left",
            )
            fig.colorbar(
                sc_im,
                ax=ax_im,
                cax=fig.axes[5],
                label=f"$\\mathrm{{Im}}\\zeta_{{{k}}}$",
                location="right",
            )
            fig.colorbar(
                sc_t,
                ax=ax_t,
                cax=fig.axes[6],
                label="$T$",
                location="left",
            )
            fig.colorbar(
                sc_h,
                ax=ax_h,
                cax=fig.axes[7],
                label="$h$",
                location="right",
            )
        else:
            fig.colorbar(
                sc_re,
                ax=ax_re,
                label=f"$\\mathrm{{Re}}\\zeta_{{{k}}}$",
                location="left",
            )
            fig.colorbar(
                sc_im,
                ax=ax_im,
                label=f"$\\mathrm{{Im}}\\zeta_{{{k}}}$",
                location="right",
            )
            fig.colorbar(
                sc_t,
                ax=ax_t,
                label="$T$",
                location="left",
            )
            fig.colorbar(
                sc_h,
                ax=ax_h,
                label="$h$",
                location="right",
            )

    return fig, plot_lifecycle


def make_running_pred_plotter(
    test_pars: DataPars,
    test_data: Data,
    preds: Vtsts,
    plt_date_range_tst: Optional[tuple[str, str]] = None,
    plt_step_tst: int = 1,
    i_fig: int = 1,
) -> tuple[Figure, F[int, None]]:
    """Make plotting function for prediction over different lead times."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, axs = plt.subplots(
        1,
        2,
        num=i_fig,
        figsize=tuple(mpf.figaspect(0.5)),
        constrained_layout=True,
    )

    def plot_pred(i_step: int):
        i0_dl_tst = test_pars.delay_embedding_end
        if plt_date_range_tst is not None:
            plt_periods_tst = pd.period_range(
                start=plt_date_range_tst[0],
                end=plt_date_range_tst[1],
                freq="M",
            )
            i0_periods_tst = pd.period_range(
                start=test_data["time"][0],
                end=plt_date_range_tst[0],
                freq="M",
            )
            num_plt_tst = len(plt_periods_tst)
            i0_tst = i0_dl_tst + len(i0_periods_tst)
        else:
            num_plt_tst = test_pars.num_samples
            i0_tst = i0_dl_tst
        i1_tst = i0_tst + num_plt_tst
        i0_pred = i0_tst + i_step
        i1_pred = i1_tst + i_step
        j0_tst = i0_tst - test_pars.delay_embedding_end
        j1_tst = i1_tst - test_pars.delay_embedding_end
        err = (
            preds[j0_tst:j1_tst, i_step]
            - test_data["responses"][i0_pred:i1_pred]
        )
        for ax in axs:
            ax.cla()

        ax = axs[0]
        ax.plot(
            test_data["time"][i0_tst:i1_tst:plt_step_tst],
            test_data["responses"][i0_pred:i1_pred:plt_step_tst],
            "-",
            label="True",
        )
        ax.plot(
            test_data["time"][i0_tst:i1_tst:plt_step_tst],
            preds[j0_tst:j1_tst:plt_step_tst, i_step],
            "-",
            label="Prediction",
        )
        ax.set_xlabel("Verification time")
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(test_pars.response.var)
        ax.set_title(f"Prediction; lead time = {i_step} months")

        ax = axs[1]
        ax.plot(
            test_data["time"][i0_tst:i1_tst:plt_step_tst],
            err[::plt_step_tst],
            "-",
        )
        ax.set_xlabel("Verification time")
        ax.set_title("Error")
        ax.grid(True)

    return fig, plot_pred


def make_pred_timeseries_plotter(
    pars: DataPars, test_data: Data, preds: Vtsts, i_fig: int = 1
) -> tuple[Figure, F[int, None]]:
    """Make plotting function over different initial conditions."""
    if plt.fignum_exists(i_fig):
        plt.close(i_fig)
    fig, ax = plt.subplots(num=i_fig, constrained_layout=True)
    num_pred_steps = preds.shape[1] - 1
    ts = jnp.arange(num_pred_steps + 1)
    timestep_str = "months"

    def plot_pred(i_init: int):
        i0_tst = pars.delay_embedding_end + i_init
        i1_tst = i0_tst + num_pred_steps + 1
        init_timestamp = test_data["time"][i0_tst]
        ax.cla()
        ax.plot(ts, test_data["responses"][i0_tst:i1_tst], "o-", label="True")
        ax.plot(ts, preds[i_init, :], "o-", label="Prediction")
        ax.grid()
        ax.legend()
        ax.set_xlabel(f"Lead time ({timestep_str})")
        ax.set_title(f"Initialization time = {init_timestamp}")

    return fig, plot_pred


def plot_forecast_skill_scores(
    pars: DataPars, scores: SkillScores, i_fig: int = 1
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
    timestep_str = "months"
    for ax, score, label in zip(
        axs, (scores["nrmses"], scores["accs"]), labels
    ):
        ax.plot(ts, score, "o-")
        ax.grid()
        if ax.get_subplotspec().is_first_row():
            ax.set_title(pars.response.var)
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel(f"Lead time ({timestep_str})")
        ax.set_ylabel(label)
    return fig

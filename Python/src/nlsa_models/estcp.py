"""Computation and plotting functions for analysis of ERA5 data."""

import jax
import jax.numpy as jnp
import matplotlib.figure as mpf
import matplotlib.pyplot as plt
import nlsa.function_algebra as fun
import nlsa.jax.delays as dl
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
from enum import StrEnum, auto
from functools import partial, reduce
from jax import Array, NamedSharding, vmap
from jax.typing import DTypeLike
from matplotlib.figure import Figure
from nlsa.io_actions import timeit
from nlsa.jax.kernels import KernelEigen, KernelPars
from nlsa.jax.koopman import KoopmanEigen, KoopmanEigenbasis, KoopmanPars
from nlsa.jax.stats import anomaly_correlation_coefficient, normalized_rmse
from nlsa.jax.vector_algebra import (
    L2FnAlgebra,
    L2FnAlgebraShardings,
    L2VectorAlgebra,
)
from nlsa_models.core import (
    JaxEnv as JaxEnv,
    SkillScores as SkillScores,
    initialize_jax as initialize_jax,
    initialize_matplotlib as initialize_matplotlib,
)
from numpy.typing import ArrayLike
from pandas import DataFrame, DatetimeIndex, Series
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    assert_never,
    cast,
    final,
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
type F[*Ss, T] = Callable[[*Ss], T]  # Shorthand for Callables
type NPVector[N: int, D: np.dtype[np.floating[Any]]] = np.ndarray[tuple[N], D]
type NPMatrix[M: int, N: int, D: np.dtype[np.floating[Any]]] = np.ndarray[
    tuple[M, N], D
]
type TimeSampling = Literal["monthly", "daily"]
type SpaceSampling = Literal["pointwise", "coarsened", "area_averaged"]
type RollingMode = Literal["forward", "backward", "center"]


@runtime_checkable
class GriddedDataSpecs[T: TimeSampling, S: SpaceSampling](Protocol):
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
    def min_lon(self) -> Optional[float]:
        """Return min_lon property of GriddedDataSpecs protocol."""
        ...

    @property
    def max_lon(self) -> Optional[float]:
        """Return max_lon property of GriddedDataSpecs protocol."""
        ...

    @property
    def step_lon(self) -> Optional[int]:
        """Return step_lon property of GriddedDataSpecs protocol."""
        ...

    @property
    def min_lat(self) -> Optional[float]:
        """Return min_lat property of GriddedDataSpecs protocol."""
        ...

    @property
    def max_lat(self) -> Optional[float]:
        """Return max_lat property of GriddedDataSpecs protocol."""
        ...

    @property
    def step_lat(self) -> Optional[int]:
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
    def rolling_window(self) -> Optional[int]:
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
    def input_path(self) -> Optional[str | Path]:
        """Return input_path property of GriddedDataSpecs protocol."""
        ...

    @property
    def file_format(self) -> Optional[Literal["grib", "nc"]]:
        """Return file format property of GriddedDataSpecs protocol."""
        ...


class ERA5Var(StrEnum):
    """ERA5 variables."""

    AVG_TNLWRF = "avg_tnlwrf"
    """Outgoing longwave radiation."""

    MSL = "msl"
    """Mean sea level pressure."""

    SST = "sst"
    """Sea surface temperature."""

    SP = "sp"
    """Surface pressure."""

    CAPE = "cape"
    """Convectively available potential energy."""

    TCWV = "tcwv"
    """Total column water vapor."""

    T2M = "t2m"
    """2 meter temperature."""

    TSR = "tsr"
    """Top net shortwave (solar) radiation."""

    TTR = "ttr"
    """Top net longwave (thermal) radiation."""

    CP = "cp"
    """Convective precipitation."""

    LSP = "lsp"
    """Large-scale precipitation."""

    TENV = "10v"
    """10-meter V wind component."""

    SSRO = "ssro"
    """Subsurface runoff"""

    LAI_HV = "lai_hv"
    """Leaf area index, high vegetation."""

    LAI_LW = "lai_lv"
    """Lead area index, low vegetation."""


@dataclass(frozen=True, slots=True)
class ERA5Domain[S: SpaceSampling]:
    """Domain data for daily ERA5 NW hemisphere metadata."""

    min_lon: Optional[float] = None
    max_lon: Optional[float] = None
    min_lat: Optional[float] = None
    max_lat: Optional[float] = None
    step_lon: Optional[int] = None
    step_lat: Optional[int] = None
    sampling: S = cast(S, "pointwise")

    def __str__(self) -> str:
        """Create string representation of ERA5Domain object."""
        lon = "-".join(
            map(
                str,
                filter(None, [self.min_lon, self.max_lon, self.step_lon]),
            )
        )
        lat = "-".join(
            map(
                str,
                filter(None, [self.min_lat, self.max_lat, self.step_lat]),
            )
        )
        return "_".join((lon, lat))


@dataclass(frozen=True, slots=True)
class Time[T: TimeSampling]:
    """Temporal and spatial sampling of ERA5 data."""

    date_range: tuple[str, str]
    sampling: T
    rolling_window: Optional[int] = None
    rolling_mode: RollingMode = "center"
    custom_climatology_date_range: Optional[tuple[str, str]] = None

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


@dataclass(frozen=True, slots=True)
class ERA5IO:
    """ERA5 IO specs."""

    input_path: Optional[str | Path] = None
    file_format: Literal["grib", "nc"] = "nc"


@final
@dataclass(frozen=True, slots=True)
class ERA5DataSpecs[T: TimeSampling, S: SpaceSampling](GriddedDataSpecs[T, S]):
    """ERA5 dataset."""

    vars: Sequence[ERA5Var]
    domain: ERA5Domain[S]
    time: Time[T]
    io: ERA5IO
    climatology: Climatology = Climatology()

    @property
    def varnames(self) -> Sequence[str]:
        """Return varnames property of ERA5DataSpecs object."""
        return [var.value for var in self.vars]

    @property
    def input_varnames(self) -> Sequence[str]:
        """Return input_varnames property of ERA5DataSpecs object."""
        return [var.value for var in self.vars]

    @property
    def min_lon(self) -> Optional[float]:
        """Return min_lon property of ERA5DataSpecs object."""
        return self.domain.min_lon

    @property
    def max_lon(self) -> Optional[float]:
        """Return max_lon property of ERA5DataSpecs object."""
        return self.domain.max_lon

    @property
    def step_lon(self) -> Optional[int]:
        """Return step_lon property of ERA5DataSpecs object."""
        return self.domain.step_lon

    @property
    def min_lat(self) -> Optional[float]:
        """Return min_lat property of ERA5DataSpecs object."""
        return self.domain.min_lat

    @property
    def max_lat(self) -> Optional[float]:
        """Return max_lat property of ERA5DataSpecs object."""
        return self.domain.max_lat

    @property
    def step_lat(self) -> Optional[int]:
        """Return step_lat property of ERA5DataSpecs object."""
        return self.domain.step_lat

    @property
    def space_sampling(self) -> SpaceSampling:
        """Return space_sampling property of ERA5DataSpecs Protocol."""
        return self.domain.sampling

    @property
    def time_sampling(self) -> TimeSampling:
        """Return sampling frequency of ERA5DataSpecs object."""
        return self.time.sampling

    @property
    def rolling_window(self) -> Optional[int]:
        """Return rolling_window property of ERA5DataSpecs object."""
        return self.time.rolling_window

    @property
    def rolling_mode(self) -> RollingMode:
        """Return rolling_mode property of ERA5DataSpecs object."""
        return self.time.rolling_mode

    @property
    def standardize(self) -> bool:
        """Return standardize property of ERA5DataSpecs object."""
        return self.climatology.standardize

    @property
    def remove_climatology(self) -> bool:
        """Return standardize property of ERA5DataSpecs object."""
        return self.climatology.remove

    @property
    def date_range(self) -> tuple[str, str]:
        """Return date_range property of ERA5DataSpecs object."""
        return self.time.date_range

    @property
    def climatology_date_range(self) -> tuple[str, str]:
        """Return climatology_date_range property of ERA5DataSpecs object."""
        return self.time.climatology_date_range

    @property
    def input_path(self) -> Optional[str | Path]:
        """Return input_path property of ERA5DataSpecs object."""
        return self.io.input_path

    @property
    def file_format(self) -> Literal["grib", "nc"]:
        """Return file_format property of ERA5DataSpecs object."""
        return self.io.file_format

    def __str__(self) -> str:
        """Create string representation of ERA5DataSpecs object."""
        return "_".join(
            filter(
                None,
                (
                    *self.varnames,
                    *map(str, (self.domain, self.time, self.climatology)),
                ),
            )
        )


@runtime_checkable
class StationDataSpecs[T: TimeSampling](Protocol):
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
    def rolling_window(self) -> Optional[int]:
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
    def input_path(self) -> Optional[str | Path]:
        """Return input_path property of StationDataSpecs protocol."""
        ...

    @property
    def file_format(self) -> Optional[Literal["csv"]]:
        """Return file format property of StationDataSpecs protocol."""
        ...

    @property
    def input_year(self) -> str:
        """Return input_year property of WS44DataSpecs object."""
        ...

    @property
    def input_month(self) -> str:
        """Return input_month property of WS44DataSpecs object."""
        ...

    @property
    def input_day(self) -> str:
        """Return input_day property of WS44DataSpecs object."""
        ...


class WS44Station(StrEnum):
    """Station names based on ICAO codes."""

    KABQ = auto()
    """Kirtland AFB, AZ"""

    KBLH = auto()
    """Blythe Airport, CA"""

    KBUR = auto()
    """Hollywood Burbank Airport, CA"""

    KCDS = auto()
    """Childress Municipal Airport, TX"""

    KCNM = auto()
    """Cavern City Air Terminal, NM"""

    KCRQ = auto()
    """McClellan-Palomar Airport, CA"""

    KCVN = auto()
    """Clovis Regional Airport, NM"""

    KCVS = auto()
    """Canon AFB, NM"""

    KDAG = auto()
    """Barstow-Daggett Airport, CA"""

    KDLF = auto()
    """Laughlin AFB, TX"""

    KDMA = auto()
    """Davis-Mohnan AFB, AZ"""

    KDMN = auto()
    """Deming Municipal Airport, NM"""

    KDRT = auto()
    """Del Rio International Airport, TX"""

    KDUG = auto()
    """Bisbee Douglas International Airport, AZ"""

    KEDW = auto()
    """Edwards AFB, CA"""

    KELP = auto()
    """El Pasto International Airport, TX"""

    KFHU = auto()
    """Libby Army Airfield, AZ"""

    KGDP = auto()
    """Guadalupe Pass Weather Station, TX"""

    KHHR = auto()
    """Hawthorne Municipal Airport, CA"""

    KHMN = auto()
    """Holloman AFB, NM"""

    KHOB = auto()
    """Lea County Regional Airport, NM"""

    KINK = auto()
    """Winkler County Airport, TX"""

    KIPL = auto()
    """Imperial County Airport, CA"""

    KIWA = auto()
    """Phoenix-Mesa Gateway (former Williams AFB)"""

    KLAX = auto()
    """Los Angeles International Airport, CA"""

    KLBB = auto()
    """Lubbock, TX"""

    KLGB = auto()
    """Long Beach Airport, CA"""

    KLPC = auto()
    """Lompoc Airport, CA"""

    KLRU = auto()
    """Las Cruces International Airport, NM"""

    KLSV = auto()
    """Nellis AFB, NM"""

    KLUF = auto()
    """Luke AFB, AZ"""

    KMAF = auto()
    """Midland International Air & Space Port, TX"""

    KMRF = auto()
    """Marfa Municipal, TX"""

    KNJK = auto()
    """El Centro NAF, CA"""

    KNSI = auto()
    """San Nicolas Island NOLF, CA"""

    KNTD = auto()
    """Point Mugu NAS, CA"""

    KNYL = auto()
    """Marine Corps Air Station, Yuma, AZ"""

    KNZY = auto()
    """North Island NAS, CA"""

    KONT = auto()
    """Ontario International Airport, CA"""

    KOXR = auto()
    """Oxnard Airport, CA"""

    KPHX = auto()
    """Phoenix Sky Harbor International Airport, AZ"""

    KPRC = auto()
    """Prescott Regional Airport, AZ"""

    KPSP = auto()
    """Palm Springs International Airport, CA"""

    KRIV = auto()
    """March Air Reserve Base, CA"""

    KROW = auto()
    """Roswell International Air Center, NM"""

    KREE = auto()
    """Reese AFB, TX"""

    KSAD = auto()
    """Safford Regional Airport, AZ"""

    KSAN = auto()
    """San Diego International Airport, CA"""

    KSBA = auto()
    """Santa Barbara Municipal Airport, CA"""

    KSMX = auto()
    """Santa Maria Public Airport, CA"""

    KSDB = auto()
    """Sandberg Airstrip, CA"""

    KSEE = auto()
    """Gillespie Field, CA"""

    KSUU = auto()
    """Travis Air Force Base, CA"""

    KSJT = auto()
    """San Angelo Rgnl Mathis Field, TX"""

    KSMO = auto()
    """Santa Monica Municipal Airport, CA"""

    KSNA = auto()
    """John Wayne Airport, CA"""

    KSRR = auto()
    """Sierra Blanca Regional Airport, NM"""

    KTCS = auto()
    """Truth or Consequences Municipal Airport, NM"""

    KTUS = auto()
    """Tucson International Airport, AZ"""

    KTRM = auto()
    """Jacqueline Cohran Regional Airport, CA"""

    KWJF = auto()
    """General William J. Fox Airfield, CA"""

    KVBG = auto()
    """Vanderberg AFB, CA"""

    KVNY = auto()
    """Van Nuys Airport, XA"""

    PGUA = auto()
    """Andersen AFB, Guam"""

    PKWA = auto()
    """Bucholz Army Airfield (Kwajalein Atoll)"""

    PWAK = auto()
    """Wake Island Airfield (Wake Atoll)"""


class WS44Var(StrEnum):
    """Station variables."""

    TEMP_C_MAX = auto()
    """Max temperature."""

    TEMP_C_MIN = auto()
    """Min temperature."""

    TEMP_C_MEAN = auto()
    """Mean temperature."""

    SLP_MAX = auto()
    """Max sea level pressure."""

    SLP_MIN = auto()
    """Min sea level pressure."""

    SLP_MEAN = auto()
    """Mean sea level pressure."""

    STP_MAX = auto()
    """Max station pressure."""

    STP_MIN = auto()
    """Min station pressure."""

    STP_MEAN = auto()
    """Mean station pressure."""

    U_WIND_MPS_MEAN = auto()
    """Mean zonal winds."""

    V_WIND_MPS_MEAN = auto()
    """Mean meridional winds."""

    DAILY_PRECIP_IN = auto()
    """Precipitation."""


@dataclass(frozen=True, slots=True)
class WS44IO:
    """Station IO specs."""

    input_path: Optional[str | Path] = None
    file_format: Literal["csv"] = "csv"


@final
@dataclass(frozen=True, slots=True)
class WS44DataSpecs[T: TimeSampling](StationDataSpecs[T]):
    """WS44 station dataset."""

    vars: Sequence[WS44Var]
    stations: Sequence[WS44Station]
    time: Time[T]
    io: WS44IO
    climatology: Climatology = Climatology()

    @property
    def varnames(self) -> list[WS44Var]:
        """Return varnames property of WS44DataSpecs object."""
        return [var for var in self.vars]

    @property
    def input_varnames(self) -> Sequence[str]:
        """Return input_varnames property of WS44DataSpecs object."""
        return [var.value for var in self.vars]

    @property
    def station_names(self) -> Sequence[str]:
        """Return station_names property of WS44DataSpecs object."""
        return [station.value.upper() for station in self.stations]

    @property
    def time_sampling(self) -> TimeSampling:
        """Return sampling frequency of WS44DataSpecs object."""
        return self.time.sampling

    @property
    def rolling_window(self) -> Optional[int]:
        """Return rolling_window property of WS44DataSpecs object."""
        return self.time.rolling_window

    @property
    def rolling_mode(self) -> RollingMode:
        """Return rolling_mode property of WS44DataSpecs object."""
        return self.time.rolling_mode

    @property
    def standardize(self) -> bool:
        """Return standardize property of WS44DataSpecs object."""
        return self.climatology.standardize

    @property
    def remove_climatology(self) -> bool:
        """Return standardize property of WS44DataSpecs object."""
        return self.climatology.remove

    @property
    def date_range(self) -> tuple[str, str]:
        """Return date_range property of WS44DataSpecs object."""
        return self.time.date_range

    @property
    def climatology_date_range(self) -> tuple[str, str]:
        """Return climatology_date_range property of WS44DataSpecs object."""
        return self.time.climatology_date_range

    @property
    def input_path(self) -> Optional[str | Path]:
        """Return input_path property of WS44DataSpecs object."""
        return self.io.input_path

    @property
    def file_format(self) -> Literal["csv"]:
        """Return file_format property of WS44DataSpecs object."""
        return self.io.file_format

    @property
    def input_year(self) -> str:
        """Return input_year property of WS44DataSpecs object."""
        return "YEAR"

    @property
    def input_month(self) -> str:
        """Return input_month property of WS44DataSpecs object."""
        return "MO"

    @property
    def input_day(self) -> str:
        """Return input_day property of WS44DataSpecs object."""
        return "DAY"

    def __str__(self) -> str:
        """Create string representation of WS44DataSpecs object."""
        return "_".join(
            (
                *map(str.lower, self.varnames),
                *self.stations,
                *map(str, (self.time, self.climatology)),
            )
        )


class WS44HighLatStation(StrEnum):
    """Station names based on ICAO codes."""

    BGTL = auto()
    """Pituffik Space Base Airport."""


class WS44HighLatVar(StrEnum):
    """High latitude station variables."""

    MAXTEMP = auto()
    """Max temperature."""

    MINTEMP = auto()
    """Min temperature."""

    MEANTEMP = auto()
    """Mean temperature."""

    MAXSLP = auto()
    """Max sea level pressure."""

    MINSLP = auto()
    """Min sea level pressure."""

    MEANSLP = auto()
    """Mean sea level pressure."""

    MAXSTP = auto()
    """Max station pressure."""

    MINSTP = auto()
    """Min station pressure."""

    MEANSTP = auto()
    """Mean station pressure."""

    AVG_U_WIND = auto()
    """Mean zonal winds."""

    AVG_V_WIND = auto()
    """Mean meridional winds."""

    PRECIP_IN = auto()
    """Precipitation."""


@dataclass(frozen=True, slots=True)
class WS44HighLatIO:
    """Station IO specs."""

    input_path: Optional[str | Path] = None
    file_format: Literal["csv"] = "csv"


@final
@dataclass(frozen=True, slots=True)
class WS44HighLatDataSpecs[T: TimeSampling](StationDataSpecs[T]):
    """WS44 high latitude station dataset."""

    vars: Sequence[WS44HighLatVar]
    stations: Sequence[WS44HighLatStation]
    time: Time[T]
    io: WS44HighLatIO
    climatology: Climatology = Climatology()

    @property
    def varnames(self) -> list[WS44HighLatVar]:
        """Return varnames property of WS44HighLatDataSpecs object."""
        return [var for var in self.vars]

    @property
    def input_varnames(self) -> Sequence[str]:
        """Return input_varnames property of WS44HighLatDataSpecs object."""
        return [var.value for var in self.vars]

    @property
    def station_names(self) -> Sequence[str]:
        """Return station_names property of WS44HighLatDataSpecs object."""
        return [station.value.upper() for station in self.stations]

    @property
    def time_sampling(self) -> TimeSampling:
        """Return sampling frequency of WS44HighLatDataSpecs object."""
        return self.time.sampling

    @property
    def rolling_window(self) -> Optional[int]:
        """Return rolling_window property of WS44HighLatDataSpecs object."""
        return self.time.rolling_window

    @property
    def rolling_mode(self) -> RollingMode:
        """Return rolling_mode property of WS44HighLatDataSpecs object."""
        return self.time.rolling_mode

    @property
    def standardize(self) -> bool:
        """Return standardize property of WS44HighLatDataSpecs object."""
        return self.climatology.standardize

    @property
    def remove_climatology(self) -> bool:
        """Return standardize property of WS44HighLatDataSpecs object."""
        return self.climatology.remove

    @property
    def date_range(self) -> tuple[str, str]:
        """Return date_range property of WS44HighLatDataSpecs object."""
        return self.time.date_range

    @property
    def climatology_date_range(self) -> tuple[str, str]:
        """Return climatology_date_range of WS44HighLatDataSpecs object."""
        return self.time.climatology_date_range

    @property
    def input_path(self) -> Optional[str | Path]:
        """Return input_path property of WS44HighLatDataSpecs object."""
        return self.io.input_path

    @property
    def file_format(self) -> Literal["csv"]:
        """Return file_format property of WS44HighLatDataSpecs object."""
        return self.io.file_format

    @property
    def input_year(self) -> str:
        """Return input_year property of WS44HighLatDataSpecs object."""
        return "YEAR"

    @property
    def input_month(self) -> str:
        """Return input_month property of WS44HighLatDataSpecs object."""
        return "MO"

    @property
    def input_day(self) -> str:
        """Return input_day property of WS44HighLatDataSpecs object."""
        return "DAY"

    def __str__(self) -> str:
        """Create string representation of WS44HighLatDataSpecs object."""
        return "_".join(
            (
                *map(str.lower, self.varnames),
                *self.stations,
                *map(str, (self.time, self.climatology)),
            )
        )


class Covariate[T: TimeSampling](NamedTuple):
    """NamedTuple for covariate specification."""

    specs: Sequence[GriddedDataSpecs[T, SpaceSampling] | StationDataSpecs[T]]

    def __str__(self) -> str:
        """Create string representation of covariate variables."""
        return "_".join(map(str, self.specs))


class Response[T: TimeSampling](NamedTuple):
    """NamedTuple for response specification."""

    specs: GriddedDataSpecs[T, Literal["area_averaged"]] | StationDataSpecs[T]

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
    """Number of extra after delay embedding."""

    velocity_covariate: bool = False
    """Include time tendencies (velocities) in covariate data."""

    velocity_fd_order: Optional[Literal[2, 4, 6, 8]] = None
    """Finite-difference order for velocity data."""

    eval_batch_size: Optional[int] = None
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


class Data(NamedTuple):
    """NamedTuple containing JAX arrays fof the covariate/response vars."""

    covariates: Array
    """Covariate variables."""

    responses: Array
    """Response variables."""


class NPData(NamedTuple):
    """NamedTuple containing time, covariate, and response data."""

    time: DatetimeIndex
    """Timestamps of the covariate/response data."""

    covariates: NPMatrix[int, int, np.dtype[np.floating[Any]]]
    """Covariate variables."""

    responses: (
        NPVector[int, np.dtype[np.floating[Any]]]
        | NPMatrix[int, int, np.dtype[np.floating[Any]]]
    )
    """Response variables."""

    def to_device(
        self,
        dtype: Optional[DTypeLike] = None,
        shardings: Optional[NamedSharding | Device] = None,
    ) -> Data:
        """Put NDArray data to on-device JAX arrays."""
        return Data(
            covariates=jnp.asarray(
                self.covariates, dtype=dtype, device=shardings
            ),
            responses=jnp.asarray(
                self.responses, dtype=dtype, device=shardings
            ),
        )


def to_data_frame[T: TimeSampling](
    pars: DataPars[T],
    koopman_basis: KoopmanEigenbasis[Yd, C, V, Cs, int | Array],
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
def read_gridded_dataset[T: TimeSampling](
    specs: GriddedDataSpecs[T, SpaceSampling],
) -> Dataset:
    """Import gridded data into Xarray dataset."""
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
        xr.open_mfdataset(
            os.fspath(pth),
            parallel=True,
            compat="no_conflicts",
        )
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


def read_station_dataframe(
    station: str,
    sampling: TimeSampling,
    vars: Sequence[str],
    date_range: Optional[tuple[str, str]] = None,
    climatology_date_range: Optional[tuple[str, str]] = None,
    remove_climatology: bool = False,
    rolling_window: Optional[int] = None,
    rolling_mode: Literal["forward", "backward", "center"] = "center",
    standardize: bool = False,
    input_dir: Optional[str | Path] = None,
    input_year: Optional[str] = "YEAR",
    input_month: Optional[str] = "MO",
    input_day: Optional[str] = "DAY",
) -> DataFrame:
    """Import station data from CSV file into DataFrame.

    This function identifies missing dates, and fills missing values using
    linear interpolation. Data standardization and/or climatology removal
    (deseasonalization) are performed upon request.
    """
    # Read the CSV file
    input_file = station + "_dailystats.csv"
    if input_dir is None:
        df = pd.read_csv(input_file)
        print(f"Input dataset read from {input_file}")
    else:
        pth = Path(input_dir) / input_file
        df = pd.read_csv(pth)
        print(f"Input dataset read from {pth}")

    # Convert YEAR, MO, DAY to a datetime column
    df = df.rename(
        columns={input_year: "YEAR", input_month: "MONTH", input_day: "DAY"}
    )
    df["Date"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]])

    # Drop redundant columns
    df = df[["Date"] + list(vars)]

    # Generate full date range
    full_dates = pd.date_range(
        start=df["Date"].min(), end=df["Date"].max(), freq="D"
    )

    # Assign default date range and climatology date range if not provided
    if date_range is not None:
        _date_range = date_range
    else:
        _date_range = (full_dates[0], full_dates[-1])
    if climatology_date_range is not None:
        _climatology_date_range = climatology_date_range
    else:
        _climatology_date_range = _date_range

    # Reindex dataset to include all dates (missing dates get NaN values)
    df = df.set_index("Date").reindex(full_dates)

    # Identify missing dates
    for var in vars:
        missing_dates = df[df[var].isna()]
        if missing_dates.empty:
            print(f"No missing days in {station} for {var}.")
        else:
            df[var] = df[var].interpolate(method="linear")
            print(
                f"Interpolated {len(missing_dates)}/{len(df)} missing dates "
                f"in {station} for {var}:"
            )
            print(missing_dates)

    # Perform monthly averaging if requested
    if sampling == "monthly":
        df = df.resample("MS").mean()
        print("Monthly-averaged data.")

    # Remove climatology if requested
    if remove_climatology:
        climatology_mask = (df.index >= _climatology_date_range[0]) & (
            df.index <= _climatology_date_range[1]
        )
        assert isinstance(df.index, DatetimeIndex)
        match sampling:
            case "daily":
                df["clim_index"] = df.index.day_of_year
                df = df.reset_index()
                df_clim = df[climatology_mask]
                climatology = df_clim.groupby("clim_index").mean()
                df = df.merge(
                    climatology,
                    on="clim_index",
                    suffixes=("", "_mean"),
                    how="left",
                )
                df = df.set_index("index")
            case "monthly":
                df["clim_index"] = df.index.month
                df = df.reset_index()
                df_clim = df[climatology_mask]
                climatology = df_clim.groupby("clim_index").mean()
                df = df.merge(
                    climatology,
                    on="clim_index",
                    suffixes=("", "_mean"),
                    how="left",
                )
                df = df.set_index("index")
        for var in vars:
            df[var] -= df[f"{var}_mean"]
        df = df.drop(
            columns=["clim_index", "index_mean"]
            + [f"{var}_mean" for var in vars]
        )
        print(
            f"Removed {_climatology_date_range[0]} "
            f"to {_climatology_date_range[1]} climatology."
        )

    # Perform rolling average if requested
    if rolling_window is not None:
        match rolling_mode:
            case "backward":
                df = df.rolling(window=rolling_window, min_periods=1).mean()
            case "center":
                df = df.rolling(
                    window=rolling_window, min_periods=1, center=True
                ).mean()
            case "forward":
                df = (
                    df.iloc[::-1]
                    .rolling(window=rolling_window, min_periods=1)
                    .mean()
                    .iloc[::-1]
                )

    # Perform standardization if requested
    if standardize:
        df_std = df[
            (df.index >= _climatology_date_range[0])
            & (df.index <= _climatology_date_range[1])
        ]
        scaler = StandardScaler()
        scaler.fit(df_std[vars])
        df[vars] = np.asarray(scaler.transform(df[vars]))
        print(
            f"Standardized data based on {_climatology_date_range[0]} "
            f" to {_climatology_date_range[1]} climatology"
        )

    # Extract requested date range
    if date_range is not None:
        df = df[(df.index >= _date_range[0]) & (df.index <= _date_range[1])]
    return df


def read_high_lat_station_dataframe(
    station: str,
    sampling: TimeSampling,
    vars: Sequence[str],
    date_range: Optional[tuple[str, str]] = None,
    climatology_date_range: Optional[tuple[str, str]] = None,
    remove_climatology: bool = False,
    rolling_window: Optional[int] = None,
    rolling_mode: Literal["forward", "backward", "center"] = "center",
    standardize: bool = False,
    input_dir: Optional[str | Path] = None,
    input_date: str = "obs_date",
) -> DataFrame:
    """Import station data from CSV files into DataFrame.

    This function identifies missing dates, and fills missing values using
    linear interpolation. Data standardization and/or climatology removal
    (deseasonalization) are performed upon request.
    """
    # Read the CSV files
    if input_dir is None:
        pth = Path.cwd() / station
    else:
        pth = Path(input_dir) / station
    df_list = [
        pd.read_csv(file, parse_dates=[input_date])
        .drop(columns=["platformid"])
        .drop_duplicates(subset=[input_date])
        for file in pth.glob("*.csv")
    ]
    df = reduce(
        lambda left, right: pd.merge(left, right, on=input_date, how="outer"),
        df_list,
    )
    df = df.sort_values(input_date).reset_index(drop=True)
    df = df.rename(columns={input_date: "Date"})

    # Drop redundant columns
    df = df[["Date"] + list(vars)]

    # Generate full date range
    full_dates = pd.date_range(
        start=df["Date"].min(), end=df["Date"].max(), freq="D"
    )

    # Assign default date range and climatology date range if not provided
    if date_range is not None:
        _date_range = date_range
    else:
        _date_range = (full_dates[0], full_dates[-1])
    if climatology_date_range is not None:
        _climatology_date_range = climatology_date_range
    else:
        _climatology_date_range = _date_range

    # Reindex dataset to include all dates (missing dates get NaN values)
    df = df.set_index("Date").reindex(full_dates)

    # Identify missing dates
    for var in vars:
        missing_dates = df[df[var].isna()]
        if missing_dates.empty:
            print(f"No missing days in {station} for {var}.")
        else:
            df[var] = df[var].interpolate(method="linear")
            print(
                f"Interpolated {len(missing_dates)}/{len(df)} missing dates "
                f"in {station} for {var}:"
            )
            print(missing_dates)

    # Perform monthly averaging if requested
    if sampling == "monthly":
        df = df.resample("MS").mean()
        print("Monthly-averaged data.")

    # Remove climatology if requested
    if remove_climatology:
        climatology_mask = (df.index >= _climatology_date_range[0]) & (
            df.index <= _climatology_date_range[1]
        )
        assert isinstance(df.index, DatetimeIndex)
        match sampling:
            case "daily":
                df["clim_index"] = df.index.day_of_year
                df = df.reset_index()
                df_clim = df[climatology_mask]
                climatology = df_clim.groupby("clim_index").mean()
                df = df.merge(
                    climatology,
                    on="clim_index",
                    suffixes=("", "_mean"),
                    how="left",
                )
                df = df.set_index("index")
            case "monthly":
                df["clim_index"] = df.index.month
                df = df.reset_index()
                df_clim = df[climatology_mask]
                climatology = df_clim.groupby("clim_index").mean()
                df = df.merge(
                    climatology,
                    on="clim_index",
                    suffixes=("", "_mean"),
                    how="left",
                )
                df = df.set_index("index")
        for var in vars:
            df[var] -= df[f"{var}_mean"]
        df = df.drop(
            columns=["clim_index", "index_mean"]
            + [f"{var}_mean" for var in vars]
        )
        print(
            f"Removed {_climatology_date_range[0]} "
            f"to {_climatology_date_range[1]} climatology."
        )

    # Perform rolling average if requested
    if rolling_window is not None:
        match rolling_mode:
            case "backward":
                df = df.rolling(window=rolling_window, min_periods=1).mean()
            case "center":
                df = df.rolling(
                    window=rolling_window, min_periods=1, center=True
                ).mean()
            case "forward":
                df = (
                    df.iloc[::-1]
                    .rolling(window=rolling_window, min_periods=1)
                    .mean()
                    .iloc[::-1]
                )

    # Perform standardization if requested
    if standardize:
        df_std = df[
            (df.index >= _climatology_date_range[0])
            & (df.index <= _climatology_date_range[1])
        ]
        scaler = StandardScaler()
        scaler.fit(df_std[vars])
        df[vars] = np.asarray(scaler.transform(df[vars]))
        print(
            f"Standardized data based on {_climatology_date_range[0]} "
            f" to {_climatology_date_range[1]} climatology"
        )

    # Extract requested date range
    if date_range is not None:
        df = df[(df.index >= _date_range[0]) & (df.index <= _date_range[1])]
    return df


# TODO: Consider cleaning this up by adding a multi_file as a property
# of the StationDataSpecs Protocol. Then the high_lat reader could be used
# for multi-file dataframes.
def read_station_dataframes[T: TimeSampling](
    specs: StationDataSpecs[T],
) -> Iterator[DataFrame]:
    """Import station data into an iterator of Pandas dataframes."""
    for station in specs.station_names:
        if station in WS44HighLatStation.__members__:
            df = read_high_lat_station_dataframe(
                station=station,
                sampling=specs.time_sampling,
                vars=specs.varnames,
                date_range=specs.date_range,
                climatology_date_range=specs.climatology_date_range,
                remove_climatology=specs.remove_climatology,
                rolling_window=specs.rolling_window,
                rolling_mode=specs.rolling_mode,
                standardize=specs.standardize,
                input_dir=specs.input_path,
            )
        else:
            df = read_station_dataframe(
                station=station,
                sampling=specs.time_sampling,
                vars=specs.varnames,
                date_range=specs.date_range,
                climatology_date_range=specs.climatology_date_range,
                remove_climatology=specs.remove_climatology,
                rolling_window=specs.rolling_window,
                rolling_mode=specs.rolling_mode,
                standardize=specs.standardize,
                input_dir=specs.input_path,
            )
        yield df


def extract_data[T: TimeSampling, D: np.dtype[np.floating[Any]]](
    pars: DataPars[T],
    dtype: Optional[D] = None,
) -> NPData:
    """Extract gridded and station data."""

    def from_dataset(ds: Dataset) -> NPMatrix[int, int, D]:
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
        dtype: Optional[D] = None,
    ) -> NPMatrix[int, int, D]:
        """Extract numpy array from Pandas dataframe."""
        a = df.to_numpy().astype(dtype)
        return a

    def to_2darray(
        specs: GriddedDataSpecs[T, SpaceSampling] | StationDataSpecs[T],
    ) -> NPMatrix[int, int, D]:
        """Extract gridded or station data from specs to numpy array."""
        match specs:
            case GriddedDataSpecs():
                a = from_dataset(read_gridded_dataset(specs))
            case StationDataSpecs():
                a = np.hstack(
                    [
                        from_dataframe(df)
                        for df in read_station_dataframes(specs)
                    ]
                )
        return a

    print("Reading covariates:")
    print(*pars.covariate.specs, sep="\n")
    covariates = np.hstack(
        [to_2darray(specs) for specs in pars.covariate.specs]
    )
    print(f"Covariates array shape: {covariates.shape}")

    print("Reading response:")
    print(pars.response.specs)
    response = to_2darray(pars.response.specs).ravel()
    print(f"Response array shape: {response.shape}")
    match pars.covariate.specs[0].time_sampling:
        case "daily":
            freq = "D"
        case "monthly":
            freq = "MS"

    time = pd.date_range(
        start=pars.covariate.specs[0].date_range[0],
        end=pars.covariate.specs[0].date_range[1],
        freq=freq,
    )
    assert isinstance(time, DatetimeIndex)
    if pars.velocity_covariate:
        assert pars.velocity_fd_order is not None
        fd_op = jax.jit(
            vmap(
                dl.make_fd_operator(
                    order=pars.velocity_fd_order, mode="central"
                ),
                in_axes=-1,
                out_axes=-1,
            )
        )
        vs = np.asarray(fd_op(covariates), dtype=dtype)
        data = NPData(
            time=time,
            covariates=np.stack((covariates, vs), axis=1).astype(dtype),
            responses=response,
        )
    else:
        data = NPData(
            time=time,
            covariates=covariates,
            responses=response,
        )
    return data


def make_data_driven_l2_space[T: TimeSampling, D: DTypeLike](
    pars: DataPars[T],
    dtype: D,
    delay_embedding_mode: Literal["explicit", "on_the_fly"] = "on_the_fly",
    shardings: L2FnAlgebraShardings = L2FnAlgebraShardings(),
    jit: bool = False,
) -> Callable[[Data], L2FnAlgebra[tuple[int], D, Yd, R]]:
    """Make implementation function for L2 space over covariate data space."""

    def impl_l2(data: Data) -> L2FnAlgebra[tuple[int], D, Yd, R]:
        i0 = pars.delay_embedding_origin
        i1 = i0 + pars.num_delay_samples
        incl: Callable[[F[Yd, R]], V]
        if pars.num_half_delays > 0:
            match delay_embedding_mode:
                case "on_the_fly":
                    incl = dl.delay_eval_at(
                        jnp.asarray(
                            data.covariates[i0:i1],
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
                        hankel = vmap(
                            partial(
                                dl.hankel,
                                num_delays=pars.num_delays,
                                flatten=True,
                            ),
                            in_axes=1,
                            out_axes=1,
                        )
                    else:
                        hankel = partial(
                            dl.hankel,
                            num_delays=pars.num_delays,
                            flatten=True,
                        )
                    if jit:
                        hankel = jax.jit(hankel)
                    incl = vec.batch_eval_at(
                        jnp.asarray(
                            hankel(data.covariates[i0:i1]),
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
                    data.covariates[i0:i1],
                    dtype=dtype,
                    device=shardings.data,
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

    return impl_l2


def make_data_driven_tangent_evaluation_functional_fd[T: TimeSampling](
    pars: DataPars[T],
    dtype: DTypeLike,
    fd_order: Literal[2, 4, 6, 8],
    delay_embedding_mode: Literal["explicit", "on_the_fly"] = "on_the_fly",
    shardings: L2FnAlgebraShardings = L2FnAlgebraShardings(),
    jit: bool = False,
) -> Callable[[Data], Callable[[F[Yd, TYd, R]], V]]:
    """Make evaluation using finite-difference approx. of the generator."""

    def impl_eval_tx(data: Data) -> Callable[[F[Yd, TYd, R]], V]:
        if pars.velocity_covariate:
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
            fd_op = jax.jit(fd_op)

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
                                data.covariates[i0:i1],
                                dtype=dtype,
                                device=shardings.data,
                            ),
                            jnp.asarray(
                                fd_op(data.covariates[i0_fd:i1_fd]),
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
                        hankel = vmap(
                            partial(
                                dl.hankel,
                                num_delays=pars.num_delays,
                                flatten=True,
                            ),
                            in_axes=1,
                            out_axes=1,
                        )

                    else:
                        hankel = partial(
                            dl.hankel,
                            num_delays=pars.num_delays,
                            flatten=True,
                        )
                    if jit:
                        hankel = jax.jit(hankel)
                    eval_tx = vec.batch_eval_at(
                        (
                            jnp.asarray(
                                hankel(jnp.asarray(data.covariates[i0:i1])),
                                dtype=dtype,
                                device=shardings.data,
                            ),
                            jnp.asarray(
                                hankel(
                                    fd_op(
                                        jnp.asarray(
                                            data.covariates[i0_fd:i1_fd]
                                        )
                                    )
                                ),
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
                        data.covariates[i0:i1],
                        dtype=dtype,
                        device=shardings.data,
                    ),
                    jnp.asarray(
                        fd_op(jnp.asarray(data.covariates[i0_fd:i1_fd])),
                        dtype=dtype,
                        device=shardings.data,
                    ),
                ),
                batch_size=pars.eval_batch_size,
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
def compute_kaf_response_coeffs[N: int, D: DTypeLike, T: TimeSampling](
    pars: tuple[DataPars[T], KernelPars],
    impl_l2: Callable[[Data], L2FnAlgebra[tuple[N], D, Yd, R]],
    train_data: Data,
    kernel: Callable[[Yd, Yd], R] | Callable[[Data, Yd, Yd], R],
    kernel_eigen: KernelEigen[R, Rs, V, Vs],
    num_steps: int,
    which_eigs: int | tuple[int, int] | list[int] | None = None,
    jit: bool = True,
) -> Array:
    """Compute basis expansion coefficients for kernel analog forecast."""
    data_pars, kernel_pars = pars
    i0 = data_pars.delay_embedding_end
    i1 = i0 + num_steps + data_pars.num_samples
    impl_basis = knl.make_data_driven_eigenbasis(
        kernel_pars, impl_l2, kernel, which_eigs
    )
    anal = knl.make_kaf_analysis_operator(
        impl_basis,
        num_steps,
        which_samples=(i0, i1),
        jit=jit,
    )
    return anal(train_data, train_data.responses, kernel_eigen)


def compute_koopman_response_coeffs[
    N: int,
    D: DTypeLike,
    L: int,
    T: TimeSampling,
](
    pars: tuple[DataPars[T], KernelPars, KoopmanPars],
    c_l: L2VectorAlgebra[tuple[L], D],
    impl_l2: Callable[[Data], L2FnAlgebra[tuple[N], D, Yd, R]],
    train_data: Data,
    kernel: Callable[[Yd, Yd], R] | Callable[[Data, Yd, Yd], R],
    kernel_eigen: KernelEigen[R, Rs, V, Vs],
    koopman_eigen: KoopmanEigen[C, Cs, Css],
    which_eigs: int | tuple[int, int] | list[int] | None = None,
    jit: bool = True,
) -> Array:
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
    anal = koop.make_koopman_analysis_operator(
        impl_koopman_basis,
        which_samples=(i0, i1),
        jit=jit,
    )
    return anal(train_data, train_data.responses, kernel_eigen, koopman_eigen)


def compute_response_skill_scores[T: TimeSampling](
    pars: DataPars[T],
    test_data: Data,
    fys_pred: Vtsts,
    dropna: bool = False,
) -> SkillScores:
    """Compute NRMSE and ACC skill scores over the prediction ensemble."""
    num_pred_steps = fys_pred.shape[1] - 1
    i0 = pars.delay_embedding_end
    i1 = i0 + num_pred_steps + pars.num_samples
    hankel = jax.jit(partial(dl.hankel, num_delays=num_pred_steps))
    fxs_true = hankel(jnp.asarray(test_data.responses[i0:i1]))
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


def compute_covariate_skill_scores[T: TimeSampling](
    pars: DataPars[T], test_data: Data, ys_pred: Vtsts, dropna: bool = False
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
    ys_true = hankel(test_data.covariates[i0:i1])
    if dropna:
        mask = ~jnp.isnan(ys_pred).any(axis=(1, 2))
        ys_pred = ys_pred[mask]
        ys_true = ys_true[mask]
    nrmses = normalized_rmses(ys_true, ys_pred)
    accs = anomaly_correlation_coefficients(ys_true, ys_pred)
    scores: SkillScores = {"nrmses": nrmses, "accs": accs}
    return scores


def compute_skill_scores[T: TimeSampling](
    pars: DataPars[T],
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


def plot_bandwidth_function[T: TimeSampling, D: DTypeLike](
    pars: DataPars[T],
    impl_l2y: Callable[[Data], L2FnAlgebra[tuple[int], D, Yd, R]],
    bandwidth_func: Callable[[Data, Yd], R],
    train_data: NPData,
    train_shardings: Optional[NamedSharding | Device] = None,
    test_pars: Optional[DataPars[T]] = None,
    impl_l2y_tst: Optional[
        Callable[[Data], L2FnAlgebra[tuple[int], D, Yd, R]]
    ] = None,
    test_data: Optional[NPData] = None,
    test_shardings: Optional[NamedSharding | Device] = None,
    delay_plot_mode: Literal["backward", "central"] = "central",
    plt_date_range: Optional[tuple[str, str]] = None,
    plt_date_range_tst: Optional[tuple[str, str]] = None,
    plt_step: int = 1,
    plt_step_tst: int = 1,
    i_fig: int = 1,
) -> Figure:
    """Plot bandwidth function on training and, optionally, test data."""

    @jax.jit
    def bandwidths(xs_train: Data) -> V:
        l2y = impl_l2y(xs_train)
        return l2y.incl(partial(bandwidth_func, xs_train))

    @jax.jit
    def bandwidths_tst(xs_train: Data, xs_tst: Data) -> Vtst:
        if impl_l2y_tst is not None:
            l2y_tst = impl_l2y(xs_tst)
            return l2y_tst.incl(partial(bandwidth_func, xs_train))
        else:
            return jnp.zeros(shape=())

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
        i0 = train_data.time.get_loc(plt_date_range[0])
        i1 = train_data.time.get_loc(plt_date_range[1])
        assert isinstance(i0, int)
        assert isinstance(i1, int)
        i1 += 1
    else:
        i0 = i0_dl
        i1 = i0 + pars.num_samples
    j0 = i0 - i0_dl
    j1 = i1 - i0_dl
    bw_vals = bandwidths(train_data.to_device(shardings=train_shardings))
    assert isinstance(bw_vals, Array)
    ax.plot(
        train_data.time[i0:i1:plt_step].values,
        bw_vals[j0:j1:plt_step],
        "-",
    )
    ax.grid(True)
    ax.set_title("Kernel bandwidth function (training)")

    if (
        impl_l2y_tst is not None
        and test_data is not None
        and test_pars is not None
    ):
        match delay_plot_mode:
            case "backward":
                i0_dl_tst = test_pars.delay_embedding_end
            case "central":
                i0_dl_tst = test_pars.delay_embedding_center
        if plt_date_range_tst is not None:
            i0_tst = train_data.time.get_loc(plt_date_range_tst[0])
            i1_tst = train_data.time.get_loc(plt_date_range_tst[1])
            assert isinstance(i0_tst, int)
            assert isinstance(i1_tst, int)
            i1_tst += 1
        else:
            i0_tst = i0_dl_tst
            i1_tst = i0_tst + test_pars.num_samples
        j0_tst = i0_tst - i0_dl_tst
        j1_tst = i1_tst - i0_dl_tst
        bw_vals_tst = bandwidths_tst(
            train_data.to_device(shardings=train_shardings),
            test_data.to_device(shardings=test_shardings),
        )
        assert isinstance(bw_vals_tst, Array)
        ax_tst.plot(
            test_data.time[i0_tst:i1_tst:plt_step_tst].values,
            bw_vals_tst[j0_tst:j1_tst:plt_step_tst],
            "-",
        )
        ax_tst.grid(True)
        ax_tst.set_title("Kernel bandwidth function (test)")
    return fig


def make_kernel_evecs_plotter[T: TimeSampling, D: DTypeLike](
    pars: tuple[DataPars[T], KernelPars],
    impl_l2: Callable[[Data], L2FnAlgebra[tuple[int], D, Yd, R]],
    train_data: NPData,
    kernel_eigen: KernelEigen[Rs, Vs, V, R],
    train_shardings: Optional[NamedSharding | Device] = None,
    test_pars: Optional[DataPars[T]] = None,
    impl_l2_tst: Optional[
        Callable[[Data], L2FnAlgebra[tuple[int], D, Yd, R]]
    ] = None,
    test_data: Optional[NPData] = None,
    test_shardings: Optional[NamedSharding | Device] = None,
    kernel: Optional[Callable[[Data, Yd, Yd], R]] = None,
    delay_plot_mode: Literal["backward", "central"] = "backward",
    plt_date_range: Optional[tuple[str, str]] = None,
    plt_date_range_tst: Optional[tuple[str, str]] = None,
    plt_step: int = 1,
    plt_step_tst: int = 1,
    i_fig: int = 1,
) -> tuple[Figure, F[int, None]]:
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

    @jax.jit
    def efunc(
        _train_data: Data,
        _kernel_eigen: KernelEigen[Rs, Vs, V, R],
        _test_data: Data,
        j: int | Array,
    ) -> Vtst:
        assert impl_kernel_basis is not None
        assert impl_l2_tst is not None
        l2y_tst = impl_l2_tst(_test_data)
        eigenbasis = impl_kernel_basis(_train_data, _kernel_eigen)
        return l2y_tst.incl(eigenbasis.fn(j))

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
        ax_tst = None
    match delay_plot_mode:
        case "backward":
            i0_dl = data_pars.delay_embedding_end
        case "central":
            i0_dl = data_pars.delay_embedding_center
    if plt_date_range is not None:
        i0 = train_data.time.get_loc(plt_date_range[0])
        i1 = train_data.time.get_loc(plt_date_range[1])
        assert isinstance(i0, int)
        assert isinstance(i1, int)
        i1 += 1
    else:
        i0 = i0_dl
        i1 = i0 + data_pars.num_samples
    j0 = i0 - i0_dl
    j1 = i1 - i0_dl
    if test_pars is not None:
        match delay_plot_mode:
            case "backward":
                i0_dl_tst = test_pars.delay_embedding_end
            case "central":
                i0_dl_tst = test_pars.delay_embedding_center
        if test_data is not None and plt_date_range_tst is not None:
            i0_tst = test_data.time.get_loc(plt_date_range_tst[0])
            i1_tst = test_data.time.get_loc(plt_date_range_tst[1])
            assert isinstance(i0_tst, int)
            assert isinstance(i1_tst, int)
            i1_tst += 1
        else:
            i0_tst = i0_dl_tst
            i1_tst = i0_tst + test_pars.num_samples
        j0_tst = i0_tst - i0_dl_tst
        j1_tst = i1_tst - i0_dl_tst
    else:
        i0_tst, i1_tst, j0_tst, j1_tst = None, None, None, None

    def plot_eig(k: int):
        evec = kernel_eigen.evecs[k]
        if test_data is not None:
            evec_tst = efunc(
                train_data.to_device(shardings=train_shardings),
                kernel_eigen,
                test_data.to_device(shardings=test_shardings),
                k,
            )
            assert isinstance(evec_tst, Array)
            assert ax_tst is not None
        for figax in fig.axes:
            figax.cla()
        ax.plot(train_data.time[i0:i1:plt_step], evec[j0:j1:plt_step], "-")
        eta = lapl_evals[k]
        ax.grid()
        ax.set_title(f"Eigenvector {k}: $\\eta_{{{k}}} = {eta: .3f}$")
        if test_pars is not None and test_data is not None:
            ax_tst.plot(
                test_data.time[i0_tst:i1_tst:plt_step_tst],
                evec_tst[j0_tst:j1_tst:plt_step_tst],
                "-",
            )
            ax_tst.grid()
            ax_tst.set_title("Nystrom")

    return fig, plot_eig


def make_koopman_evecs_plotter[
    N: int,
    Ntst: int,
    T: TimeSampling,
    D: DTypeLike,
    L: int,
](
    pars: tuple[DataPars[T], KernelPars, KoopmanPars],
    c_l: L2VectorAlgebra[tuple[L], D],
    impl_l2: Callable[[Data], L2FnAlgebra[tuple[N], D, Yd, R]],
    train_data: NPData,
    kernel_eigen: KernelEigen[Rs, Vs, V, R],
    koopman_eigen: KoopmanEigen[C, Cs, Css],
    train_shardings: Optional[NamedSharding | Device] = None,
    test_pars: Optional[DataPars[T]] = None,
    impl_l2_tst: Optional[
        Callable[[Data], L2FnAlgebra[tuple[Ntst], D, Yd, R]]
    ] = None,
    test_data: Optional[NPData] = None,
    test_shardings: Optional[NamedSharding | Device] = None,
    kernel: Optional[
        Callable[[Yd, Yd], R] | Callable[[Data, Yd, Yd], R]
    ] = None,
    delay_plot_mode: Literal["backward", "central"] = "backward",
    plt_date_range: Optional[tuple[str, str]] = None,
    plt_date_range_tst: Optional[tuple[str, str]] = None,
    plt_step: int = 1,
    plt_step_tst: int = 1,
    i_fig: int = 1,
) -> tuple[Figure, F[int, None]]:
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

    @jax.jit
    def efunc(
        _train_data: Data,
        _kernel_eigen: KernelEigen[Rs, Vs, V, R],
        _koopman_eigen: KoopmanEigen[C, Cs, Css],
        _test_data: Data,
        j: Array,
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
        axs_tst = None
    match delay_plot_mode:
        case "backward":
            i0_dl = data_pars.delay_embedding_end
        case "central":
            i0_dl = data_pars.delay_embedding_center
    if plt_date_range is not None:
        i0 = train_data.time.get_loc(plt_date_range[0])
        i1 = train_data.time.get_loc(plt_date_range[1])
        assert isinstance(i0, int)
        assert isinstance(i1, int)
        i1 += 1
    else:
        i0 = i0_dl
        i1 = i0 + data_pars.num_samples
    j0 = i0 - i0_dl
    j1 = i1 - i0_dl
    if test_pars is not None and test_data is not None:
        match delay_plot_mode:
            case "backward":
                i0_dl_tst = test_pars.delay_embedding_end
            case "central":
                i0_dl_tst = data_pars.delay_embedding_center
        if plt_date_range_tst is not None:
            i0_tst = train_data.time.get_loc(plt_date_range_tst[0])
            i1_tst = train_data.time.get_loc(plt_date_range_tst[1])
            assert isinstance(i0_tst, int)
            assert isinstance(i1_tst, int)
            i1_tst += 1
        else:
            i0_tst = i0_dl_tst
            i1_tst = i0_tst + test_pars.num_samples
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
        efreq = koopman_eigen.efreqs[k] / (2 * jnp.pi) * 12
        eperiod = koopman_eigen.eperiods[k] / 12

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
            train_data.time[i0:i1:plt_step],
            evec.real[j0:j1:plt_step],
            "-",
            label=f"$\\mathrm{{Re}}\\zeta_{{{k}}}$",
        )
        ax.plot(
            train_data.time[i0:i1:plt_step],
            evec.imag[j0:j1:plt_step],
            "-",
            label=f"$\\mathrm{{Im}}\\zeta_{{{k}}}$",
        )
        ax.set_title(f"Eigenperiod $T_{{{k}}} = {eperiod: .3f}$ years")
        ax.grid()
        ax.legend()

        if test_data is not None:
            evec_tst = efunc(
                train_data.to_device(shardings=train_shardings),
                kernel_eigen,
                koopman_eigen,
                test_data.to_device(shardings=test_shardings),
                k,
            )
            assert isinstance(evec_tst, Array)
            assert axs_tst is not None

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
                test_data.time[i0_tst:i1_tst:plt_step_tst],
                evec_tst.real[j0_tst:j1_tst:plt_step_tst],
                "-",
                label=f"$\\mathrm{{Re}}\\zeta_{{{k}}}$",
            )
            ax.plot(
                test_data.time[i0_tst:i1_tst:plt_step_tst],
                evec_tst.imag[j0_tst:j1_tst:plt_step_tst],
                "-",
                label=f"$\\mathrm{{Im}}\\zeta_{{{k}}}$",
            )
            ax.grid()
            ax.legend()

    return fig, plot_eig


def make_running_pred_plotter[T: TimeSampling](
    test_pars: DataPars[T],
    test_data: NPData,
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
                start=test_data.time[0],
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
            preds[j0_tst:j1_tst, i_step] - test_data.responses[i0_pred:i1_pred]
        )
        for ax in axs:
            ax.cla()

        ax = axs[0]
        ax.plot(
            test_data.time[i0_tst:i1_tst:plt_step_tst],
            test_data.responses[i0_pred:i1_pred:plt_step_tst],
            "-",
            label="True",
        )
        ax.plot(
            test_data.time[i0_tst:i1_tst:plt_step_tst],
            preds[j0_tst:j1_tst:plt_step_tst, i_step],
            "-",
            label="Prediction",
        )
        ax.set_xlabel("Verification time")
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(test_pars.response.specs)
        ax.set_title(f"Prediction; lead time = {i_step} months")

        ax = axs[1]
        ax.plot(
            test_data.time[i0_tst:i1_tst:plt_step_tst],
            err[::plt_step_tst],
            "-",
        )
        ax.set_xlabel("Verification time")
        ax.set_title("Error")
        ax.grid(True)

    return fig, plot_pred


def make_pred_timeseries_plotter[T: TimeSampling](
    pars: DataPars[T], test_data: NPData, preds: Vtsts, i_fig: int = 1
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
        init_timestamp = test_data.time[i0_tst]
        ax.cla()
        ax.plot(ts, test_data.responses[i0_tst:i1_tst], "o-", label="True")
        ax.plot(ts, preds[i_init, :], "o-", label="Prediction")
        ax.grid()
        ax.legend()
        ax.set_xlabel(f"Lead time ({timestep_str})")
        ax.set_title(f"Initialization time = {init_timestamp}")

    return fig, plot_pred


def plot_forecast_skill_scores[T: TimeSampling](
    pars: DataPars[T], scores: SkillScores, i_fig: int = 1
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
            ax.set_title(pars.response.specs)
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel(f"Lead time ({timestep_str})")
        ax.set_ylabel(label)
    return fig

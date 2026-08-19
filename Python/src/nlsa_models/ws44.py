"""Computation and plotting functions for analysis of 14WS station data."""

import numpy as np
import pandas as pd
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from functools import reduce
from jax import Array
from nlsa_models.climate import (
    Climatology,
    ImplementsStationDataSpecs,
    RollingMode,
    Time,
    TimeSampling,
)
from pandas import DataFrame, DatetimeIndex
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    final,
)

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


class Station(StrEnum):
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


class Var(StrEnum):
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
class IO:
    """Station IO specs."""

    input_path: str | Path | None = None
    file_format: Literal["csv"] = "csv"


@final
@dataclass(frozen=True, slots=True)
class DataSpecs[T: TimeSampling](
    ImplementsStationDataSpecs[T, Iterator[DataFrame]]
):
    """WS44 station dataset."""

    vars: Sequence[Var]
    stations: Sequence[Station]
    time: Time[T]
    io: IO
    climatology: Climatology = Climatology()

    @property
    def varnames(self) -> list[Var]:
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
    def rolling_window(self) -> int | None:
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
    def input_path(self) -> str | Path | None:
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

    def read_data(self) -> Iterator[DataFrame]:
        """Read station data into an Iterator of DataFrames."""
        for station in self.station_names:
            df = read_station_dataframe(
                station=station,
                sampling=self.time_sampling,
                vars=self.varnames,
                date_range=self.date_range,
                climatology_date_range=self.climatology_date_range,
                remove_climatology=self.remove_climatology,
                rolling_window=self.rolling_window,
                rolling_mode=self.rolling_mode,
                standardize=self.standardize,
                input_dir=self.input_path,
            )
            yield df


class HighLatStation(StrEnum):
    """Station names based on ICAO codes."""

    BGTL = auto()
    """Pituffik Space Base Airport."""


class HighLatVar(StrEnum):
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
class HighLatIO:
    """Station IO specs."""

    input_path: str | Path | None = None
    file_format: Literal["csv"] = "csv"


@final
@dataclass(frozen=True, slots=True)
class HighLatDataSpecs[T: TimeSampling](
    ImplementsStationDataSpecs[T, Iterator[DataFrame]]
):
    """WS44 high latitude station dataset."""

    vars: Sequence[HighLatVar]
    stations: Sequence[HighLatStation]
    time: Time[T]
    io: HighLatIO
    climatology: Climatology = Climatology()

    @property
    def varnames(self) -> list[HighLatVar]:
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
    def rolling_window(self) -> int | None:
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
    def input_path(self) -> str | Path | None:
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

    def read_data(self) -> Iterator[DataFrame]:
        """Read WS44 high lat. station data into an Iterator of DataFrames."""
        for station in self.station_names:
            df = read_high_lat_station_dataframe(
                station=station,
                sampling=self.time_sampling,
                vars=self.varnames,
                date_range=self.date_range,
                climatology_date_range=self.climatology_date_range,
                remove_climatology=self.remove_climatology,
                rolling_window=self.rolling_window,
                rolling_mode=self.rolling_mode,
                standardize=self.standardize,
                input_dir=self.input_path,
            )
            yield df


def read_station_dataframe(
    station: str,
    sampling: TimeSampling,
    vars: Sequence[str],
    date_range: tuple[str, str] | None = None,
    climatology_date_range: tuple[str, str] | None = None,
    remove_climatology: bool = False,
    rolling_window: int | None = None,
    rolling_mode: Literal["forward", "backward", "center"] = "center",
    standardize: bool = False,
    input_dir: str | Path | None = None,
    input_year: str | None = "YEAR",
    input_month: str | None = "MO",
    input_day: str | None = "DAY",
) -> DataFrame:
    """Import WS44 station data from CSV file into DataFrame.

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
    date_range: tuple[str, str] | None = None,
    climatology_date_range: tuple[str, str] | None = None,
    remove_climatology: bool = False,
    rolling_window: int | None = None,
    rolling_mode: Literal["forward", "backward", "center"] = "center",
    standardize: bool = False,
    input_dir: str | Path | None = None,
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
# def read_station_dataframes[T: TimeSampling](
#     specs: ImplementsStationDataSpecs[T, Iterator[DataFrame]],
# ) -> Iterator[DataFrame]:
#     """Import station data into an iterator of Pandas dataframes."""
#     for station in specs.station_names:
#         if station in WS44HighLatStation.__members__:
#             df = read_ws44_high_lat_station_dataframe(
#                 station=station,
#                 sampling=specs.time_sampling,
#                 vars=specs.varnames,
#                 date_range=specs.date_range,
#                 climatology_date_range=specs.climatology_date_range,
#                 remove_climatology=specs.remove_climatology,
#                 rolling_window=specs.rolling_window,
#                 rolling_mode=specs.rolling_mode,
#                 standardize=specs.standardize,
#                 input_dir=specs.input_path,
#             )
#         else:
#             df = read_ws44_station_dataframe(
#                 station=station,
#                 sampling=specs.time_sampling,
#                 vars=specs.varnames,
#                 date_range=specs.date_range,
#                 climatology_date_range=specs.climatology_date_range,
#                 remove_climatology=specs.remove_climatology,
#                 rolling_window=specs.rolling_window,
#                 rolling_mode=specs.rolling_mode,
#                 standardize=specs.standardize,
#                 input_dir=specs.input_path,
#             )
#         yield df


# def extract_data[T: TimeSampling](
#     pars: DataPars[T],
#     dtype: np.dtype[np.floating[Any]] | None = None,
# ) -> NPData:
#     """Extract gridded and station data."""

#     def from_dataset(
#         ds: Dataset,
#     ) -> NPMatrix[int, int, np.dtype[np.floating[Any]]]:
#         """Extract numpy array from xarray dataset."""
#         a = (
#             ds.to_stacked_array(new_dim="stacked_dim", sample_dims=["time"])
#             .dropna(dim="stacked_dim")
#             .astype(dtype)
#             .to_numpy()
#         )
#         return a

#     def from_dataframe(
#         df: DataFrame,
#         dtype: np.dtype[np.floating[Any]] | None = None,
#     ) -> NPMatrix[int, int, np.dtype[np.floating[Any]]]:
#         """Extract numpy array from Pandas dataframe."""
#         a = df.to_numpy().astype(dtype)
#         return a

#     def to_2darray(
#         specs: ImplementsGriddedDataSpecs[T, SpaceSampling]
#         | ImplementsStationDataSpecs[T],
#     ) -> NPMatrix[int, int, np.dtype[np.floating[Any]]]:
#         """Extract gridded or station data from specs to numpy array."""
#         match specs:
#             case ImplementsGriddedDataSpecs():
#                 a = from_dataset(read_gridded_dataset(specs))
#             case ImplementsStationDataSpecs():
#                 a = np.hstack(
#                     [
#                         from_dataframe(df)
#                         for df in read_station_dataframes(specs)
#                     ]
#                 )
#         return a

#     print("Reading covariates:")
#     print(*pars.covariate.specs, sep="\n")
#     covariates = np.hstack(
#         [to_2darray(specs) for specs in pars.covariate.specs]
#     )
#     print(f"Covariates array shape: {covariates.shape}")

#     print("Reading response:")
#     print(pars.response.specs)
#     response = to_2darray(pars.response.specs).ravel()
#     print(f"Response array shape: {response.shape}")
#     match pars.covariate.specs[0].time_sampling:
#         case "daily":
#             freq = "D"
#         case "monthly":
#             freq = "MS"

#     time = pd.date_range(
#         start=pars.covariate.specs[0].date_range[0],
#         end=pars.covariate.specs[0].date_range[1],
#         freq=freq,
#     )
#     assert isinstance(time, DatetimeIndex)
#     if pars.velocity_covariate:
#         assert pars.velocity_fd_order is not None
#         fd_op = typestable_jit(
#             vmap(
#                 dl.make_fd_operator(
#                     order=pars.velocity_fd_order, mode="central"
#                 ),
#                 in_axes=-1,
#                 out_axes=-1,
#             )
#         )
#         vs = np.asarray(fd_op(jnp.asarray(covariates)), dtype=dtype)
#         data = NPData(
#             time=time,
#             covariates=np.stack((covariates, vs), axis=1).astype(dtype),
#             responses=response,
#         )
#     else:
#         data = NPData(
#             time=time,
#             covariates=covariates,
#             responses=response,
#         )
#     return data

"""Computation and plotting functions for analysis of ERA5 data."""

import nlsa_models.climate as clim
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from nlsa_models.climate import (
    Climatology,
    ImplementsGriddedDataSpecs,
    RollingMode,
    SpaceSampling,
    Time,
    TimeSampling,
)
from pathlib import Path
from typing import Literal, final
from xarray import Dataset


@dataclass(frozen=True, slots=True)
class Domain[S: SpaceSampling]:
    """Domain data for daily ERA5 NW hemisphere metadata."""

    sampling: S
    min_lon: float | None = None
    max_lon: float | None = None
    min_lat: float | None = None
    max_lat: float | None = None
    step_lon: int | None = None
    step_lat: int | None = None

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


class Var(StrEnum):
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
class IO:
    """ERA5 IO specs."""

    input_path: str | Path | None = None
    file_format: Literal["grib", "nc"] = "nc"


@final
@dataclass(frozen=True, slots=True)
class DataSpecs[T: TimeSampling, S: SpaceSampling](
    ImplementsGriddedDataSpecs[T, S, Dataset]
):
    """Specifications of ERA5 dataset."""

    vars: Sequence[Var]
    domain: Domain[S]
    time: Time[T]
    io: IO
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
    def min_lon(self) -> float | None:
        """Return min_lon property of ERA5DataSpecs object."""
        return self.domain.min_lon

    @property
    def max_lon(self) -> float | None:
        """Return max_lon property of ERA5DataSpecs object."""
        return self.domain.max_lon

    @property
    def step_lon(self) -> int | None:
        """Return step_lon property of ERA5DataSpecs object."""
        return self.domain.step_lon

    @property
    def min_lat(self) -> float | None:
        """Return min_lat property of ERA5DataSpecs object."""
        return self.domain.min_lat

    @property
    def max_lat(self) -> float | None:
        """Return max_lat property of ERA5DataSpecs object."""
        return self.domain.max_lat

    @property
    def step_lat(self) -> int | None:
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
    def rolling_window(self) -> int | None:
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
    def input_path(self) -> str | Path | None:
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

    def read_data(self) -> Dataset:
        """Read ERA5 data into Xarray Dataset."""
        return clim.read_gridded_dataset(self)


def nino34_domain[S: SpaceSampling](
    sampling: S = "pointwise",
    step_lon: int | None = None,
    step_lat: int | None = None,
) -> Domain[S]:
    """ERA5 Nino 3.4 domain."""
    return Domain(
        min_lon=360 - 170,
        max_lon=360 - 120,
        step_lon=step_lon,
        min_lat=-5,
        max_lat=5,
        step_lat=step_lat,
        sampling=sampling,
    )


def indo_pacific_domain[S: SpaceSampling](
    sampling: S = "pointwise",
    step_lon: int | None = None,
    step_lat: int | None = None,
) -> Domain[S]:
    """ERA5 Indo-Pacific domain."""
    return Domain(
        min_lon=28,
        max_lon=360 - 70,
        step_lon=step_lon,
        min_lat=-60,
        max_lat=20,
        step_lat=step_lat,
        sampling=sampling,
    )

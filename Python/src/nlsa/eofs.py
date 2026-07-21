"""Provide generic functions and classes for EOF computations."""

from dataclasses import dataclass
from typing import (
    Literal,
)


@dataclass(frozen=True, slots=True)
class EEOFPars:
    """Dataclass containing extended EOF parameters."""

    num_delays: int
    """Number of delays."""

    delay_step: int = 1
    """Delay step."""

    eigensolver: Literal["svd"] = "svd"
    """Eigensolver used for SVD."""

    num_eigs: int | None = None
    """Number of kernel eigenvalue/eigenvector pairs to compute."""

    batch_size: int | None = None
    """Maximum batch size for matrix-matrix products."""

    def __str__(self) -> str:
        """Create string representation of EEOF parameters."""
        return "_".join(("eeof", self.eigensolver, f"neigs{self.num_eigs}"))

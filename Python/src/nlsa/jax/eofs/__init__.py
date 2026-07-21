"""Provide classes and functions for EOF computations in JAX."""

from nlsa.eofs import EEOFPars as EEOFPars

from nlsa.jax.eofs._eofs import (
    EEOFEigen as EEOFEigen,
    EEOFEigenShardings as EEOFEigenShardings,
    EEOFEigenbasis as EEOFEigenbasis,
    compute_eigen as compute_eigen,
    make_data_driven_eigenbasis as make_data_driven_eigenbasis,
    make_eigenbasis as make_eigenbasis,
    make_svd_eeof_eigensolver as make_svd_eeof_eigensolver,
    plot_eeof_spectrum as plot_eeof_spectrum,
)

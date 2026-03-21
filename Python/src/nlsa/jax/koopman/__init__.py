"""Provide classes and functions for kernel computations in JAX."""

from nlsa.koopman import (
    KoopmanEigenbasis as KoopmanEigenbasis,
    KoopmanParsDiff as KoopmanParsDiff,
    KoopmanParsQz as KoopmanParsQz,
    plot_operator_matrix as plot_operator_matrix,
)
from nlsa.jax.koopman._koopman import (
    GeneratorShardings as GeneratorShardings,
    KoopmanEigen as KoopmanEigen,
    KoopmanEigenShardings as KoopmanEigenShardings,
    QzShardings as QzShardings,
    compute_generator_matrix as compute_generator_matrix,
    compute_qz_matrix as compute_qz_matrix,
    compute_eigen_diff as compute_eigen_diff,
    compute_eigen_qz as compute_eigen_qz,
    make_eigenbasis as make_eigenbasis,
    make_eigenbasis_antisym as make_eigenbasis_antisym,
    plot_generator_spectrum as plot_generator_spectrum,
    slice_koopman_eigen as slice_koopman_eigen,
    to_gen_evals as to_gen_evals,
    to_koopman_eigen as to_koopman_eigen,
)

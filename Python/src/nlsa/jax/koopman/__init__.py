"""Provide classes and functions for kernel computations in JAX."""

from nlsa.koopman import (
    KoopmanEigen as KoopmanEigen,
    KoopmanEigenbasis as KoopmanEigenbasis,
    KoopmanPars as KoopmanPars,
    KoopmanParsDiff as KoopmanParsDiff,
    KoopmanParsGauss as KoopmanParsGauss,
    KoopmanParsLapl as KoopmanParsLapl,
    KoopmanParsTransf as KoopmanParsTransf,
    plot_operator_matrix as plot_operator_matrix,
)
from nlsa.jax.koopman._koopman import (
    GeneratorShardings as GeneratorShardings,
    IntegralTransformShardings as IntegralTransformShardings,
    KoopmanEigenShardings as KoopmanEigenShardings,
    compute_generator_eigen_diff as compute_generator_eigen_diff,
    compute_generator_matrix as compute_generator_matrix,
    compute_integral_transform_eigen_comp as compute_integral_transform_eigen_comp,
    compute_integral_transform_matrix as compute_integral_transform_matrix,
    compute_koopman_preds as compute_koopman_preds,
    invert_dawson as invert_dawson,
    invert_qz as invert_qz,
    make_data_driven_eigenbasis as make_data_driven_eigenbasis,
    make_eigenbasis as make_eigenbasis,
    make_eigenbasis_antisym as make_eigenbasis_antisym,
    make_eigenbasis_asym as make_eigenbasis_asym,
    make_generator_builder as make_generator_builder,
    make_generator_eigensolver_diff as make_generator_eigensolver_diff,
    make_koopman_analysis_operator as make_koopman_analysis_operator,
    make_koopman_prediction_function as make_koopman_prediction_function,
    plot_generator_spectrum as plot_generator_spectrum,
)

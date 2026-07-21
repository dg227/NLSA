"""Provide classes and functions for kernel computations in JAX."""

from nlsa.koopman import (
    KoopmanPars as KoopmanPars,
    KoopmanParsDiff as KoopmanParsDiff,
    KoopmanParsTransf as KoopmanParsTransf,
    plot_operator_matrix as plot_operator_matrix,
    plot_generator_spectrum as plot_generator_spectrum,
)
from nlsa.jax.koopman._koopman import (
    GeneratorShardings as GeneratorShardings,
    IntegralTransformShardings as IntegralTransformShardings,
    KoopmanEigen as KoopmanEigen,
    KoopmanEigenbasis as KoopmanEigenbasis,
    KoopmanEigenShardings as KoopmanEigenShardings,
    compute_diffusion_regularized_generator_eigen as compute_diffusion_regularized_generator_eigen,
    compute_generator_matrix as compute_generator_matrix,
    compute_integral_transform_eigen_comp as compute_integral_transform_eigen_comp,
    compute_integral_transform_matrix as compute_integral_transform_matrix,
    compute_koopman_preds as compute_koopman_preds,
    evaluate_eigenfunction as evaluate_eigenfunction,
    invert_dawson as invert_dawson,
    invert_qz as invert_qz,
    make_data_driven_eigenbasis as make_data_driven_eigenbasis,
    make_eigenbasis as make_eigenbasis,
    make_eigenbasis_antisym as make_eigenbasis_antisym,
    make_eigenbasis_asym as make_eigenbasis_asym,
    make_eigenfunction_evaluation_functional as make_eigenfunction_evaluation_functional,
    make_generator_builder as make_generator_builder,
    make_diffusion_regularized_generator_eigensolver as make_diffusion_regularized_generator_eigensolver,
    make_koopman_analysis_operator as make_koopman_analysis_operator,
    make_koopman_prediction_function as make_koopman_prediction_function,
)

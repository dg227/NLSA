"""Provide classes and functions for dynamics computations in JAX."""

from nlsa.dynamics import (
    cocycle_orbit as cocycle_orbit,
    from_autonomous as from_autonomous,
    orbit as orbit,
    semigroup as semigroup,
)
from nlsa.jax.dynamics._dynamics import (
    make_fin_orbit as make_fin_orbit,
    vgrad as vgrad,
    flow as flow,
    make_posfreq_to_l2 as make_posfreq_to_l2,
    make_l2_to_posfreq as make_l2_to_posfreq,
    make_l63_vector_field as make_l63_vector_field,
    make_stepanoff_vector_field as make_stepanoff_vector_field,
    make_rotation_vector_field as make_rotation_vector_field,
    make_rotation_map as make_rotation_map,
    make_t2_rotation_generator_fourier as make_t2_rotation_generator_fourier,
    make_stepanoff_generator_fourier as make_stepanoff_generator_fourier,
    stepanoff_generator_matrix as stepanoff_generator_matrix,
)

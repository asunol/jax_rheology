import pytest
import jax
import jax.numpy as jnp
from diff_rheo.parameters import (
    Parameter, GaussianParameter, StaticParameter
)
from diff_rheo._forcing import VelocityGradient, AppliedStress
from diff_rheo.models import Newtonian, OldroydB
from diff_rheo._solver import DiffraxSolver

@pytest.fixture
def rng_key():
    """Provides a reusable JAX random key."""
    return jax.random.PRNGKey(0)

@pytest.fixture
def t_scalar() -> float:
    return 2.0

@pytest.fixture
def t_array() -> jax.Array:
    return jnp.linspace(0.0, 5.0, 10)

@pytest.fixture
def general_flow() -> VelocityGradient:
    return VelocityGradient.from_components(
        grad_u_11=5.0,                  # A constant component
        grad_u_12=lambda t: 2.0 * t     # A time-dependent component
    )

@pytest.fixture
def applied_stress_flow() -> AppliedStress:
    return AppliedStress.from_components(
        sigma_11=100.0,                 # A constant component
        sigma_12=lambda t: 5.0 * t      # A time-dependent component
    )

@pytest.fixture
def shear_stress_flow() -> AppliedStress:
    return AppliedStress.from_components(
        sigma_12=lambda t: 5.0 * t      # A time-dependent component
    )

@pytest.fixture
def newtonian_model():
    """Provides a simple Newtonian model instance."""
    return Newtonian(viscosity=Parameter(1.5))

@pytest.fixture
def oldroydb_model():
    """Provides a simple Oldroyd-B model instance."""
    return OldroydB(
        polymer_viscosity=Parameter(2.0),
        relaxation_time=Parameter(1.0),
        solvent_viscosity=Parameter(0.1)
    )

@pytest.fixture
def stochastic_oldroydb_model():
    """Provides an Oldroyd-B model with a random parameter."""
    return OldroydB(
        polymer_viscosity=GaussianParameter(mean=2.0, std=0.2),
        relaxation_time=Parameter(1.0),
        solvent_viscosity=StaticParameter(0.1)
    )

@pytest.fixture
def oldroydb_model_factory():
    """Factory to create a customizable Oldroyd-B model instance."""
    def _create_model(polymer_viscosity, relaxation_time, solvent_viscosity):
        return OldroydB(
            polymer_viscosity=Parameter(polymer_viscosity),
            relaxation_time=Parameter(relaxation_time),
            solvent_viscosity=Parameter(solvent_viscosity)
        )
    return _create_model

@pytest.fixture
def newtonian_model_factory():
    """Factory to create a customizable Newtonian model instance."""
    def _create_model(viscosity):
        return Newtonian(
            viscosity=Parameter(viscosity)
        )
    return _create_model

@pytest.fixture
def shear_strain_flow_factory():
    def _create_flow(shear_rate):
        return VelocityGradient.from_components(
            grad_u_12=lambda t: 2*shear_rate
        )
    return _create_flow

@pytest.fixture
def mock_solver():
    """Provides a standard DiffraxSolver instance for testing."""
    return DiffraxSolver()

@pytest.fixture
def initial_conditions_strain_input():
    return jnp.zeros((3,3))

@pytest.fixture
def initial_conditions_stress_input():
    return jnp.zeros(7)
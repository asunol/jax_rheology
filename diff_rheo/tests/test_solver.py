# In a new file: tests/test_integrators.py

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from diff_rheo._solver import DiffraxSolver, ODESolution
from diff_rheo.models import AbstractConstitutiveModel

class SimpleDecayModel(AbstractConstitutiveModel):
    decay_rate: jax.Array
    observation_noise: jax.Array

    def __init__(self, decay_rate=0.5):
        self.decay_rate = jnp.array(decay_rate)
        self.observation_noise = jnp.array(0.0)

    def rhs(self, t, y, args):
        return -self.decay_rate * y

@pytest.fixture
def decay_model():
    return SimpleDecayModel()

@pytest.fixture
def time_range():
    return jnp.linspace(0.0, 5.0, 100)

@pytest.fixture
def initial_condition():
    return jnp.array(1.0)

@pytest.mark.parametrize("solver_name", ["tsit5", "dopri5", "heun"])
def test_solver_initialization_and_correctness(solver_name, decay_model, time_range, initial_condition):
    """
    Tests that the solver initializes correctly and accurately solves a known ODE.
    """
    solver = DiffraxSolver(solver=solver_name)
    solution = solver.integrate(decay_model.rhs, initial_condition, time_range, args=())

    # Check the solution object
    assert isinstance(solution, ODESolution)
    assert solution.result, f"Solver '{solver_name}' failed to integrate."
    assert solution.ys.shape == (time_range.shape[0],)

    # Check accuracy against the analytical solution
    analytical_solution = initial_condition * jnp.exp(-decay_model.decay_rate * time_range)
    assert jnp.allclose(solution.ys, analytical_solution, rtol=1e-3, atol=1e-3)

def test_unknown_solver_raises_error():
    """
    Ensures that providing an invalid solver name raises a ValueError.
    """
    with pytest.raises(ValueError, match="Unknown solver: invalid_solver_name"):
        DiffraxSolver(solver="invalid_solver_name")

def test_solver_is_jittable(decay_model, time_range, initial_condition):
    """
    Ensures the entire integration process can be JIT-compiled.
    """
    solver = DiffraxSolver()

    @eqx.filter_jit
    def jitted_integrate(model, y0, ts):
        # The 'args' tuple must be static for JIT, so we pass model parameters through the closure.
        return solver.integrate(model.rhs, y0, ts, args=())

    try:
        jitted_solution = jitted_integrate(decay_model, initial_condition, time_range)
        eager_solution = solver.integrate(decay_model.rhs, initial_condition, time_range, args=())

        assert jitted_solution.result
        assert jnp.allclose(jitted_solution.ys, eager_solution.ys)
    except Exception as e:
        pytest.fail(f"JIT compilation failed for ODESolver.integrate: {e}")

def test_gradients_through_solver(decay_model, time_range, initial_condition):
    """
    Tests that we can take gradients through the ODE solve with respect to
    model parameters. This is the most critical test for model fitting.
    """
    solver = DiffraxSolver()

    # Define a simple loss function that depends on the final state of the ODE
    @eqx.filter_grad
    def loss_fn(model, y0, ts):
        solution = solver.integrate(model.rhs, y0, ts, args=())
        # A simple loss: how far is the final state from a target value?
        return (solution.ys[-1] - 0.5)**2

    # Compute the gradient with respect to the model's parameters
    grads = loss_fn(decay_model, initial_condition, time_range)

    # Check that the gradient is not None and has the correct structure
    assert isinstance(grads, SimpleDecayModel)
    assert grads.decay_rate is not None
    assert grads.decay_rate != 0.0
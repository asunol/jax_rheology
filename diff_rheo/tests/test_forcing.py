import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from diff_rheo._forcing import VelocityGradient, AppliedStress

def test_factory_init(general_flow, t_scalar):
    """
    Tests if the factory correctly creates a callable `forcing_fn`
    that evaluates constants and functions as expected.
    """
    # At t=2.0, grad_u_11 should be 5.0 and grad_u_21 should be 4.0
    grad = general_flow.gradient(t_scalar)
    assert jnp.isclose(grad[0, 0], 5.0)
    assert jnp.isclose(grad[0, 1], 4.0)
    assert jnp.isclose(grad[2, 2], 0.0) # Default component

def test_factory_with_invalid_key():
    """
    Tests that the factory raises a ValueError for an invalid component name.
    """
    with pytest.raises(ValueError, match="Invalid velocity gradient component"):
        VelocityGradient.from_components(grad_u_44=1.0) # Invalid key

def test_gradient_with_scalar_time(general_flow, t_scalar):
    """Tests the gradient() method with a single time value."""
    grad = general_flow.gradient(t_scalar)
    assert grad.shape == (3, 3)
    assert jnp.isclose(grad[0, 1], 2.0 * t_scalar)

def test_gradient_with_vector_time(general_flow, t_array):
    """Tests the gradient() method with an array of times (vmap)."""
    grads = general_flow.gradient(t_array)
    assert grads.shape == (t_array.shape[0], 3, 3)
    # Check a value from the middle of the vectorized computation
    expected_g12_at_t4 = 2.0 * t_array[4]
    assert jnp.isclose(grads[4, 0, 1], expected_g12_at_t4)

def test_rate_of_strain_with_scalar_time(general_flow, t_scalar):
    """Tests the rate_of_strain() method with a single time value."""
    strain = general_flow.rate_of_strain(t_scalar)
    grad = general_flow.gradient(t_scalar)

    # D = 0.5 * (L + L^T)
    expected_strain = 0.5 * (grad + grad.T)

    assert strain.shape == (3, 3)
    assert jnp.allclose(strain, expected_strain)
    # Specifically check the symmetrized component D_12 = D_21
    assert jnp.isclose(strain[1,0], 0.5 * grad[0, 1]) # 0.5 * (grad_21 + grad_12)

def test_rate_of_strain_with_vector_time(general_flow, t_array):
    """Tests the rate_of_strain() method with an array of times (vmap)."""
    strains = general_flow.rate_of_strain(t_array)
    assert strains.shape == (t_array.shape[0], 3, 3)

@pytest.mark.parametrize("method_name", ["gradient", "rate_of_strain"])
def test_jit_compatibility(general_flow, t_scalar, method_name):
    """Ensures both public methods can be JIT-compiled."""
    # Get the method from the instance by name
    method_to_test = getattr(general_flow, method_name)
    
    jitted_method = jax.jit(method_to_test)
    
    # Check that it runs without error and gives the correct result
    result_jitted = jitted_method(t_scalar)
    result_eager = method_to_test(t_scalar)

    assert jnp.allclose(result_jitted, result_eager)

class SinusoidalComponent(eqx.Module):
    """A simple component function with its own trainable parameters."""
    amplitude: jax.Array
    frequency: jax.Array

    def __init__(self, amplitude: float = 1.0, frequency: float = 1.0):
        self.amplitude = jnp.asarray(amplitude)
        self.frequency = jnp.asarray(frequency)

    def __call__(self, t: jax.Array) -> jax.Array:
        """Evaluates the function at time t."""
        return self.amplitude * jnp.sin(self.frequency * t)

def test_gradient_wrt_nested_module_params():
    """
    Tests that we can differentiate through a VelocityGradient with respect
    to the parameters of a nested component module.
    """
    # 1. Setup: Create the component and the main flow object
    component_module = SinusoidalComponent(amplitude=5.0, frequency=2.0)
    flow = VelocityGradient.from_components(grad_u_12=component_module)
    @eqx.filter_grad
    def compute_loss_and_grads(model: VelocityGradient) -> float:
        # Pick a time and get the specific gradient component
        t_val = jnp.pi / 4.0
        g12 = model.gradient(t_val)[0, 1]
        
        # Return a simple scalar loss
        return g12**2

    # 3. Compute the gradients
    grads = compute_loss_and_grads(flow)
    
    assert isinstance(grads.forcing_fn.grad_u_12.amplitude, jax.Array)
    assert grads.forcing_fn.grad_u_12.amplitude != 0
    
    # Check that the frequency parameter has a calculated gradient
    assert isinstance(grads.forcing_fn.grad_u_12.frequency, jax.Array)
    assert grads.forcing_fn.grad_u_12.frequency != 0


def test_stress_factory_init(applied_stress_flow, t_scalar):
    """
    Tests if the factory correctly creates and assigns the component functions.
    """
    # At t=2.0, sigma_11 should be 100.0 and sigma_12 should be 10.0
    stress_tensor = applied_stress_flow.stress(t_scalar)
    assert jnp.isclose(stress_tensor[0, 0], 100.0)
    assert jnp.isclose(stress_tensor[0, 1], 10.0)
    assert jnp.isclose(stress_tensor[2, 2], 0.0) # Default component

def test_stress_factory_with_invalid_key():
    """
    Tests that the factory raises a ValueError for an invalid component name.
    """
    with pytest.raises(ValueError, match="Invalid stress component"):
        AppliedStress.from_components(sigma_44=1.0) # Invalid key

## Method Behavior Tests
#-------------------------------------------------------------------------------

def test_stress_with_scalar_time(applied_stress_flow, t_scalar):
    """Tests the stress() method with a single time value."""
    stress_tensor = applied_stress_flow.stress(t_scalar)
    assert stress_tensor.shape == (3, 3)
    assert jnp.isclose(stress_tensor[0, 1], 5.0 * t_scalar)

def test_stress_with_vector_time(applied_stress_flow, t_array):
    """Tests the stress() method with an array of times (vmap)."""
    stress_tensors = applied_stress_flow.stress(t_array)
    assert stress_tensors.shape == (t_array.shape[0], 3, 3)
    
    # Check a value from the middle of the vectorized computation
    expected_s12_at_t4 = 5.0 * t_array[4]
    assert jnp.isclose(stress_tensors[4, 0, 1], expected_s12_at_t4)

## JAX Integration and Derivative Tests
#-------------------------------------------------------------------------------

def test_stress_jit_compatibility(applied_stress_flow, t_scalar):
    """Ensures the stress() method can be JIT-compiled."""
    jitted_stress_fn = jax.jit(applied_stress_flow.stress)
    
    result_jitted = jitted_stress_fn(t_scalar)
    result_eager = applied_stress_flow.stress(t_scalar)

    assert jnp.allclose(result_jitted, result_eager)

def test_stress_time_derivative(t_array):
    """
    Tests that we can correctly evaluate the time derivative d(sigma)/dt.
    """
    # 1. Create a model with a known analytical derivative
    # For sigma_11 = t^3, the derivative is 3*t^2
    model = AppliedStress.from_components(
        sigma_11=lambda t: t**3,
        sigma_22=10.0 # Constant, derivative should be 0
    )

    # 2. Get the time-derivative function by applying jax.grad to the method
    #    Note: jax.grad works element-wise on PyTree outputs.
    stress_dt_fn = jax.jacobian(model.stress)
    
    # 3. Use vmap to apply the derivative function across the time array
    derivative_tensors = jax.vmap(stress_dt_fn)(t_array)
    
    # 4. Check the results
    assert derivative_tensors.shape == (t_array.shape[0], 3, 3)
    
    # Check the derivative of the t^3 component
    expected_dt_s11 = 3 * t_array**2
    assert jnp.allclose(derivative_tensors[:, 0, 0], expected_dt_s11)
    
    # Check that the derivative of the constant component is zero
    assert jnp.allclose(derivative_tensors[:, 1, 1], 0.0)
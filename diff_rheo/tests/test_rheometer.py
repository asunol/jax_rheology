import pytest
import jax.numpy as jnp
import equinox as eqx

from diff_rheo._rheometer import VirtualRheometer
from diff_rheo.models import OldroydB, AbstractConstitutiveModel
from diff_rheo.parameters import StaticParameter, AbstractParameter, AbstractRandomParameter
from diff_rheo._data_types import SimulationData
from diff_rheo._protocols import (
    GeneralizedNewtonianStrainRateProtocol,
    ViscoelasticStrainRateProtocol,
    ViscoelasticShearStressProtocol
)

class TestVirtualRheometerInstantiation:
    """
    Tests the logic of the `from_model` factory method to ensure correct
    protocol selection and error handling.
    """
    def test_selects_gn_strain_rate_protocol(self, newtonian_model):
        rheometer = VirtualRheometer.setup(
            model=newtonian_model,
            experiment_type="strain_rate_response"
        )
        assert isinstance(rheometer.protocol, GeneralizedNewtonianStrainRateProtocol)

    def test_selects_ve_strain_rate_protocol(self, oldroydb_model, mock_solver):
        rheometer = VirtualRheometer.setup(
            model=oldroydb_model,
            experiment_type="strain_rate_response",
            solver=mock_solver
        )
        assert isinstance(rheometer.protocol, ViscoelasticStrainRateProtocol)

    def test_selects_ve_shear_stress_protocol(self, oldroydb_model, mock_solver):
        rheometer = VirtualRheometer.setup(
            model=oldroydb_model,
            experiment_type="shear_stress_response",
            solver=mock_solver
        )
        assert isinstance(rheometer.protocol, ViscoelasticShearStressProtocol)

    def test_raises_for_unsupported_gn_experiment(self, newtonian_model):
        with pytest.raises(NotImplementedError, match="Shear stress response not implemented"):
            VirtualRheometer.setup(
                model=newtonian_model,
                experiment_type="shear_stress_response"
            )

    def test_raises_for_missing_solver_in_ve_model(self, oldroydb_model):
        with pytest.raises(ValueError, match="Solver must be provided for viscoelastic models"):
            VirtualRheometer.setup(
                model=oldroydb_model,
                experiment_type="strain_rate_response",
                solver=None # Explicitly pass None
            )

    def test_raises_for_unknown_experiment_type(self, newtonian_model):
        with pytest.raises(ValueError, match="Unknown experiment type"):
            VirtualRheometer.setup(
                model=newtonian_model,
                experiment_type="invalid_experiment_type"
            )

    def test_raises_for_unknown_model_type(self):
        class UnknownModel(AbstractConstitutiveModel):
            pass
        
        with pytest.raises(ValueError, match="Unknown model type"):
            VirtualRheometer.setup(
                model=UnknownModel(),
                experiment_type="strain_rate_response"
            )


class TestVirtualRheometerJAX:
    """
    Test suite for the VirtualRheometer class, focusing on JAX compatibility.
    """

    def test_run_experiment_jittable(self, oldroydb_model, general_flow, t_array, initial_conditions_strain_input, mock_solver):
        """
        Tests that the standard run_experiment method can be JIT-compiled.
        """
        rheometer = VirtualRheometer.setup(
            model=oldroydb_model,
            experiment_type="strain_rate_response",
            solver=mock_solver
        )

        # Define a simple wrapper to JIT
        @eqx.filter_jit
        def jitted_run(forcing, time, y0):
            return rheometer.run_experiment(oldroydb_model, forcing, time, y0)

        # Run once to compile and execute
        jitted_data = jitted_run(general_flow, t_array, initial_conditions_strain_input)
        
        # Run eagerly to get a baseline for comparison
        eager_data = rheometer.run_experiment(oldroydb_model, general_flow, t_array, initial_conditions_strain_input)

        assert isinstance(jitted_data, SimulationData)
        assert jitted_data.result
        # Verify that the JIT-compiled result matches the eager execution
        assert jnp.allclose(jitted_data.solution.ys, eager_data.solution.ys)

    def test_run_experiment_differentiable(self, oldroydb_model, general_flow, t_array, initial_conditions_strain_input, mock_solver):
        """
        Tests that we can differentiate through the run_experiment method
        with respect to the model's parameters.
        """
        rheometer = VirtualRheometer.setup(
            model=oldroydb_model,
            experiment_type="strain_rate_response",
            solver=mock_solver
        )

        # The loss function must take the object to be differentiated (the model)
        # as its first argument.
        @eqx.filter_grad
        def loss_fn(model, forcing, time, y0):
            # Create a temporary rheometer with the potentially perturbed model
            exp_data = rheometer.run_experiment(model, forcing, t_array, initial_conditions_strain_input)
            # A simple scalar loss
            return jnp.mean(exp_data.solution.ys**2)

        # Compute gradients with respect to the model
        grads = loss_fn(oldroydb_model, general_flow, t_array, initial_conditions_strain_input)

        # Check that gradients were computed for trainable parameters
        assert isinstance(grads, OldroydB)
        assert grads.polymer_viscosity.value is not None and grads.polymer_viscosity.value != 0.0
        assert grads.relaxation_time.value is not None and grads.relaxation_time.value != 0.0
        assert grads.solvent_viscosity.value is not None and grads.solvent_viscosity.value != 0.0

    def test_run_ensemble_jittable(self, stochastic_oldroydb_model, general_flow, t_array, initial_conditions_strain_input, mock_solver, rng_key):
        """
        Tests that the vmap-based run_ensemble method can be JIT-compiled.
        """
        rheometer = VirtualRheometer.setup(
            model=stochastic_oldroydb_model,
            experiment_type="strain_rate_response",
            solver=mock_solver
        )

        @eqx.filter_jit
        def jitted_ensemble_run(model, forcing, time, y0, key):
            return rheometer.run_ensemble(model, forcing, time, y0, key, size = 10)

        # Run the jitted function
        jitted_data = jitted_ensemble_run(stochastic_oldroydb_model, general_flow, t_array, initial_conditions_strain_input, rng_key)

        assert isinstance(jitted_data, SimulationData)
        assert jitted_data.solution.ys.shape[0] == 10

    def test_run_ensemble_differentiable(self, stochastic_oldroydb_model, general_flow, t_array, initial_conditions_strain_input, mock_solver, rng_key):
        """
        Tests that we can differentiate through the run_ensemble method
        with respect to the stochastic model's underlying parameters (mean, std).
        """
        rheometer = VirtualRheometer.setup(
            model=stochastic_oldroydb_model,
            experiment_type="strain_rate_response",
            solver=mock_solver
        )

        @eqx.filter_grad
        def loss_fn(model, rheometer, forcing, time, y0, key):
            ensemble_data = rheometer.run_ensemble(model, forcing, time, y0, key, size=10)
            
            # Calculate the mean prediction across the ensemble batch
            mean_prediction = jnp.mean(ensemble_data.data, axis=0)
            
            # A simple scalar loss against a target of zero
            return jnp.mean(mean_prediction**2)

        # Compute gradients with respect to the stochastic model's parameters
        grads = loss_fn(stochastic_oldroydb_model, rheometer, general_flow, t_array, initial_conditions_strain_input, rng_key)

        # Check that gradients were computed for the underlying distribution parameters
        assert isinstance(grads, OldroydB)
        for param_name, param_value in vars(grads).items():
            if isinstance(param_value, AbstractParameter):  # Regular parameters
                if isinstance(param_value, StaticParameter):
                    assert param_value.value is None, f"Static param {param_name} gradient value is not None"
                elif isinstance(param_value, AbstractRandomParameter):
                    assert param_value.mean is not None, f"Random param {param_name} gradient value is None"
                    assert param_value.mean != 0.0, f"Random param {param_name} gradient is zero"
                    assert param_value.log_std is not None, f"Random param {param_name} std gradient is not None"
                    assert param_value.log_std != 0.0, f"Random param {param_name} std gradient is not zero"
                else:
                    assert param_value.value is not None, f"Parameter {param_name} gradient value is None"
                    assert param_value.value != 0.0, f"Parameter {param_name} gradient is zero"
    
    def test_run_ensemble_on_deterministic_model_yields_identical_results(self, oldroydb_model, general_flow, t_array, initial_conditions_strain_input, mock_solver, rng_key):
        """
        Tests that when run_ensemble is used with a non-stochastic model,
        all simulations in the batch produce identical results.
        """
        rheometer = VirtualRheometer.setup(
            model=oldroydb_model,  # Using the deterministic model fixture
            experiment_type="strain_rate_response",
            solver=mock_solver
        )

        ensemble_size = 2
        ensemble_data = rheometer.run_ensemble(
            oldroydb_model,
            general_flow,
            t_array,
            initial_conditions_strain_input,
            rng_key,
            size=ensemble_size
        )

        assert ensemble_data.solution.ys.shape[0] == ensemble_size
        assert jnp.allclose(ensemble_data.data[0], ensemble_data.data[1])

import pytest
import jax.numpy as jnp
import equinox as eqx
from diff_rheo.models import Newtonian, OldroydB
from diff_rheo._protocols import (
    GeneralizedNewtonianStrainRateProtocol,
    GeneralizedNewtonianShearStressProtocol,
    ViscoelasticStrainRateProtocol,
    ViscoelasticShearStressProtocol
)
from diff_rheo.parameters import AbstractRandomParameter, StaticParameter, AbstractParameter
from diff_rheo._solver import DiffraxSolver
from diff_rheo._data_types import SimulationData

class TestGeneralizedNewtonianProtocols:
    def test_strain_rate_protocol(self, newtonian_model, general_flow, t_array):
        protocol = GeneralizedNewtonianStrainRateProtocol()
        solution = protocol.run(newtonian_model, general_flow, t_array)

        assert isinstance(solution, SimulationData)
        assert solution.result
        assert solution.data.shape == (t_array.shape[0], 3,3)

    def test_shear_stress_protocol(self, newtonian_model, applied_stress_flow, t_array):
        protocol = GeneralizedNewtonianShearStressProtocol()
        with pytest.raises(NotImplementedError):
            protocol.run(newtonian_model, applied_stress_flow, t_array)

    def test_gn_strain_rate_analytical(self, newtonian_model_factory, shear_strain_flow_factory, t_array):
        model = newtonian_model_factory(viscosity=1.0)
        flow = shear_strain_flow_factory(shear_rate=1.0)
        protocol = GeneralizedNewtonianStrainRateProtocol()
        solution = protocol.run(model, flow, t_array)
        final_stress = solution.data[-1]
        assert jnp.allclose(final_stress, jnp.array([[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))

    def test_gn_protocol_is_jittable(self, newtonian_model, general_flow, t_array):
        """
        Ensures the GN protocol's run method can be JIT-compiled.
        """
        protocol = GeneralizedNewtonianStrainRateProtocol()
        def run_simulation(model, flow, ts):
            return protocol.run(model, flow, ts)

        jitted_run = eqx.filter_jit(run_simulation)

        # Ensure it runs without error and results are consistent
        eager_solution = run_simulation(newtonian_model, general_flow, t_array)
        jitted_solution = jitted_run(newtonian_model, general_flow, t_array)

        assert jnp.allclose(eager_solution.data, jitted_solution.data)

    def test_gn_protocol_is_differentiable(self, newtonian_model, general_flow, t_array):
        protocol = GeneralizedNewtonianStrainRateProtocol()
        @eqx.filter_grad
        def loss_fn(model, flow, ts):
            solution = protocol.run(model, flow, ts)
            # A simple scalar loss: the sum of the final stress tensor
            return jnp.sum(solution.data[-1]**2)

        # Compute gradients
        grads = loss_fn(newtonian_model, general_flow, t_array)

        # Check that a gradient was computed for the trainable parameter
        assert isinstance(grads, Newtonian)
        assert grads.viscosity.value is not None
        assert grads.viscosity.value != 0.0

VE_PROTOCOL_TEST_CASES = [
    pytest.param(
        ViscoelasticStrainRateProtocol(),
        "general_flow",              # Use the VelocityGradient fixture
        "initial_conditions_strain_input", # Use the 3x3 zero matrix
        id="StrainRateProtocol"      # A descriptive ID for the test run
    ),
    pytest.param(
        ViscoelasticShearStressProtocol(),
        "shear_stress_flow",         # Use the AppliedStress fixture
        "initial_conditions_stress_input", # Use the 7-element zero vector
        id="ShearStressProtocol"
    )
]

class TestViscoelasticProtocols:
    @pytest.mark.parametrize("model_fixture", ["oldroydb_model", "stochastic_oldroydb_model"])
    def test_strain_rate_protocol(self, model_fixture, general_flow, t_array, initial_conditions_strain_input, mock_solver, request):
        model = request.getfixturevalue(model_fixture)
        protocol = ViscoelasticStrainRateProtocol()
        solution = protocol.run(model, general_flow, t_array, initial_conditions_strain_input, mock_solver)

        assert isinstance(solution, SimulationData)
        assert solution.result
        assert solution.data.shape == (t_array.shape[0], 3,3)

    def test_shear_stress_protocol(self, oldroydb_model, shear_stress_flow, t_array, initial_conditions_stress_input, mock_solver):
        protocol = ViscoelasticShearStressProtocol()
        solution = protocol.run(oldroydb_model, shear_stress_flow, t_array, initial_conditions_stress_input, mock_solver)

        assert isinstance(solution, SimulationData)
        assert solution.result
        assert solution.data.shape == (t_array.shape[0], 7)

    def test_ve_strain_rate_protocol_analytical(self, oldroydb_model_factory, shear_strain_flow_factory):
        solvent_viscosity = 1.0 
        polymer_viscosity = 2.0 
        relaxation_time = 3.0 
        shear_rate = 0.5
        
        ucm_model = oldroydb_model_factory(polymer_viscosity=polymer_viscosity, relaxation_time=relaxation_time, solvent_viscosity=solvent_viscosity)

        t_eval = jnp.linspace(0, 10 * relaxation_time, 200) 
        y0 = jnp.zeros((3, 3)) 
        solver = DiffraxSolver()

        flow = shear_strain_flow_factory(shear_rate=shear_rate)

        protocol = ViscoelasticStrainRateProtocol()
        solution = protocol.run(ucm_model, flow, t_eval, y0, solver)

        expected_shear_stress = 2*(solvent_viscosity + polymer_viscosity)*shear_rate
        expected_N1 = 8*relaxation_time*polymer_viscosity*shear_rate**2  # UCM: N1 = 2*eta_p*lambda*gammadot^2 > 0

        final_stress_tensor = solution.data[-1]
        simulated_shear_stress = final_stress_tensor[0, 1]
        simulated_N1 = final_stress_tensor[0, 0] - final_stress_tensor[1, 1]

        assert jnp.allclose(simulated_shear_stress, expected_shear_stress, rtol=1e-3)
        assert jnp.allclose(simulated_N1, expected_N1, rtol=1e-3)

    @pytest.mark.parametrize("protocol, flow_fixture_name, y0_fixture_name", VE_PROTOCOL_TEST_CASES)
    @pytest.mark.parametrize("model_fixture", ["oldroydb_model", "stochastic_oldroydb_model"])
    def test_ve_protocols_are_jittable(self, model_fixture, t_array, mock_solver, protocol, flow_fixture_name, y0_fixture_name, request):
        """
        A single, parameterized test to ensure all VE protocols can be JIT-compiled.
        """
        # Dynamically get the correct fixture values based on the parameterized names
        flow = request.getfixturevalue(flow_fixture_name)
        y0 = request.getfixturevalue(y0_fixture_name)
        model = request.getfixturevalue(model_fixture)

        def run_simulation(model, flow, ts, y0, solver):
            return protocol.run(model, flow, ts, y0, solver)

        jitted_run = eqx.filter_jit(run_simulation)

        try:
            eager_solution = run_simulation(model, flow, t_array, y0, mock_solver)
            jitted_solution = jitted_run(model, flow, t_array, y0, mock_solver)
        except Exception as e:
            pytest.fail(f"JIT compilation failed for {type(protocol).__name__} with flow {type(flow).__name__}: {e}")

        assert jitted_solution.result
        assert jnp.allclose(eager_solution.data, jitted_solution.data, rtol=1e-5, atol=1e-5)


    @pytest.mark.parametrize("protocol, flow_fixture_name, y0_fixture_name", VE_PROTOCOL_TEST_CASES)
    @pytest.mark.parametrize("model_fixture", ["oldroydb_model", "stochastic_oldroydb_model"])
    def test_ve_protocols_are_differentiable(self, model_fixture, t_array, mock_solver, protocol, flow_fixture_name, y0_fixture_name, request):
        """
        A single, parameterized test to ensure all VE protocols are differentiable.
        """
        # Dynamically get the correct fixture values
        flow = request.getfixturevalue(flow_fixture_name)
        y0 = request.getfixturevalue(y0_fixture_name)
        model = request.getfixturevalue(model_fixture)

        @eqx.filter_grad
        def loss_fn(model, flow, ts, y0, solver):
            solution = protocol.run(model, flow, ts, y0, solver)
            # A simple scalar loss based on the final state
            # Note: The output shape differs, but sum(**2) works for both
            return jnp.sum(solution.data[-1]**2)

        # Compute the gradients of the loss w.r.t. the model parameters
        grads = loss_fn(model, flow, t_array, y0, mock_solver)

        # Assert that gradients were computed for all trainable parameters
        assert isinstance(grads, OldroydB)
        # Iterate through all attributes of the model that are parameters
        for param_name, param_value in vars(grads).items():
            if isinstance(param_value, AbstractParameter):  # Regular parameters
                if isinstance(param_value, StaticParameter):
                    assert param_value.value is None, f"Static param {param_name} gradient value is not None"
                elif isinstance(param_value, AbstractRandomParameter):
                    assert param_value.mean is not None, f"Random param {param_name} gradient value is None"
                    assert param_value.mean != 0.0, f"Random param {param_name} gradient is zero"
                    assert param_value.log_std is not None, f"Random param {param_name} std gradient is not None"
                    assert param_value.log_std == 0.0, f"Random param {param_name} std gradient is not zero"
                else:
                    assert param_value.value is not None, f"Parameter {param_name} gradient value is None"
                    assert param_value.value != 0.0, f"Parameter {param_name} gradient is zero"

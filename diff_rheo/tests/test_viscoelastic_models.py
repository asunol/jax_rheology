import pytest
import jax.numpy as jnp
import jax
import equinox as eqx
from pathlib import Path

from diff_rheo.models import (
    OldroydB, Giesekus,
    GeneralizedOldroydB, LinearPTT,
    ExponentialPTT, GeneralizedPTT,
    FENECR, FENEP, XPomPom, RUDE
)
from diff_rheo._forcing import VelocityGradient, AppliedStress
from diff_rheo._utils import _vector_to_symmetric_matrix
from .utils import parse_analytical_solutions, general_ptt_f_ptt

solutions_filepath = Path(__file__).parent.parent / "docs" / "constitutive_equations" / "viscoelastic_models.txt"
ANALYTICAL_SOLUTIONS = parse_analytical_solutions(solutions_filepath)

@pytest.fixture(scope="module")
def model_params():
    """Provides a dictionary of model parameters for the tests."""
    return {
        "polymer_viscosity": 2.0,
        "relaxation_time": 1.0,
        "solvent_viscosity": 0.1,
        "extension_length": 2.0,
        "alpha": 0.5,
        "beta": 0.5,
        "epsilon": 0.1,
        "zeta": 0.2,
        "F_function": lambda stress, rate_of_strain: 2 * stress + rate_of_strain,
        "relaxation_time_s": 1.0,
        "n": 0.5,
        "q": 0.5,
    }

@pytest.fixture(scope="module")
def extra_stress_matrix():
    return jnp.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])

@pytest.fixture(scope="module")
def strain_evolution_vector():
    return jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

@pytest.fixture(scope="module")
def test_time():
    return 1.0

@pytest.fixture
def shear_velocity_gradient():
    return VelocityGradient.from_components(
        grad_u_12=lambda t: 1.0
    )

@pytest.fixture
def shear_stress():
    return AppliedStress.from_components(
        sigma_12=lambda t: 1.0
    )

@pytest.mark.parametrize(
    "model_cls, model_name, required_params",
    [
        (OldroydB, "Oldroyd_B", []),
        (Giesekus, "Giesekus", ["alpha"]),
        (LinearPTT, "PTT_Linear", ["epsilon", "zeta"]),
        (ExponentialPTT, "PTT_Exponential", ["epsilon", "zeta"]),
        (GeneralizedOldroydB, "F_Form", ["F_function"]),
        (GeneralizedPTT, "PTT_General", ["epsilon", "zeta", "alpha", "beta"]),
        (FENECR, "FENE_CR", ["extension_length"]),
        (FENEP, "FENE_P", ["extension_length"]),
        (XPomPom, "XPomPom", ["alpha", "relaxation_time_s", "n", "q"]),
    ]
)
def test_extra_stress_models(shear_velocity_gradient, model_cls, model_name, required_params, model_params, extra_stress_matrix, test_time):
    specific_params = {k: model_params[k] for k in required_params}
    
    model = model_cls(
        polymer_viscosity=model_params["polymer_viscosity"],
        relaxation_time=model_params["relaxation_time"],
        solvent_viscosity=model_params["solvent_viscosity"],
        **specific_params
    )
    
    model_solutions = ANALYTICAL_SOLUTIONS[model_name]['extra_stress']

    # reference formulas use the conjugate (lower-convected) velocity-gradient
    # convention; evaluate them at u^T so they match the upper-convected model.
    u = shear_velocity_gradient.gradient(test_time).T
    tau = extra_stress_matrix
    
    eval_context = {
        **model_params, "jnp": jnp,
        "tau11": tau[0, 0], "tau12": tau[0, 1], "tau13": tau[0, 2],
        "tau22": tau[1, 1], "tau23": tau[1, 2], "tau33": tau[2, 2],
        "u11": u[0, 0], "u12": u[0, 1], "u13": u[0, 2],
        "u21": u[1, 0], "u22": u[1, 1], "u23": u[1, 2],
        "u31": u[2, 0], "u32": u[2, 1], "u33": u[2, 2],
    }
    
    if model_name == "F_Form":
        rate_of_strain = shear_velocity_gradient.rate_of_strain(test_time)
        f_matrix = model_params["F_function"](tau, rate_of_strain)
        eval_context.update({
            "f11": f_matrix[0, 0], "f12": f_matrix[0, 1], "f13": f_matrix[0, 2],
            "f22": f_matrix[1, 1], "f23": f_matrix[1, 2], "f33": f_matrix[2, 2],
        })
    elif model_name == "PTT_General":
        f_ptt = general_ptt_f_ptt(extra_stress_matrix, model_params)
        eval_context.update({
            "f_ptt": f_ptt,
        })

    solution_vector = jnp.array([eval(model_solutions[f'dtau{comp}'], eval_context) for comp in ["11", "22", "33", "12", "13", "23"]])
    
    expected_stress_rate = _vector_to_symmetric_matrix(solution_vector)
    actual_stress_rate = model.extra_stress_response_rhs(test_time, extra_stress_matrix, shear_velocity_gradient)
    
    assert jnp.allclose(actual_stress_rate, expected_stress_rate)

@pytest.mark.parametrize(
    "model_cls, model_name, required_params",
    [
        (OldroydB, "Oldroyd_B", []),
        (Giesekus, "Giesekus", ["alpha"]),
        (LinearPTT, "PTT_Linear", ["epsilon", "zeta"]),
        (ExponentialPTT, "PTT_Exponential", ["epsilon", "zeta"]),
        (GeneralizedPTT, "PTT_General", ["epsilon", "zeta", "alpha", "beta"]),
        (FENECR, "FENE_CR", ["extension_length"]),
        (FENEP, "FENE_P", ["extension_length"]),
        (XPomPom, "XPomPom", ["alpha", "relaxation_time_s", "n", "q"]),
    ]
)
def test_shear_stress_models(shear_stress, model_cls, model_name, required_params, model_params, strain_evolution_vector, test_time):
    """
    A single, parameterized test for all shear_stress_experiment_rhs methods.
    """
    specific_params = {k: model_params[k] for k in required_params}

    model = model_cls(
        polymer_viscosity=model_params["polymer_viscosity"],
        relaxation_time=model_params["relaxation_time"],
        solvent_viscosity=model_params["solvent_viscosity"],
        **specific_params
    )

    model_solutions = ANALYTICAL_SOLUTIONS[model_name]['strain_solution']

    s11, s22, s33, s13, s23, u12, _ = strain_evolution_vector
    s12 = shear_stress.stress(test_time)[0, 1]
    ds12 = jax.jacobian(shear_stress.stress)(test_time)[0, 1]

    # the reference strain-solution uses the conjugate convection convention;
    # evaluate it on the 1<->2 axis-swapped state and swap the result back below.
    eval_context = {
        **model_params, "jnp": jnp,
        "s11": s22, "s12": s12, "s13": s23, "s22": s11, "s23": s13, "s33": s33,
        "u12": u12, "ds12": ds12,
    }

    extra_stress_matrix = _vector_to_symmetric_matrix(jnp.array([s11, s22, s33, s12-model_params["solvent_viscosity"]*u12, s13, s23]))

    if model_name == "PTT_General":
        f_ptt = general_ptt_f_ptt(extra_stress_matrix, model_params)
        eval_context.update({
            "f_ptt": f_ptt,
        })

    solution_vector = jnp.array([
        eval(model_solutions['ds22'], eval_context),
        eval(model_solutions['ds11'], eval_context),
        eval(model_solutions['ds33'], eval_context),
        eval(model_solutions['ds23'], eval_context),
        eval(model_solutions['ds13'], eval_context),
        eval(model_solutions['du12'], eval_context),
        u12,
    ])

    expected_rhs = solution_vector.reshape((7,))
    actual_rhs = model.shear_stress_experiment_rhs(test_time, strain_evolution_vector, shear_stress)

    assert jnp.allclose(actual_rhs, expected_rhs)

class TestRUDE():
    @pytest.fixture
    def mock_tensors(self):
        """Provides mock strain and stress tensors for testing."""
        return jnp.zeros((3, 3)), jnp.zeros((3, 3))

    def test_zero_initialization(self):
        """
        Tests if the model initializes with all weights as zeros when no key is provided.
        """
        model = RUDE.zero_init()
        weights = [p for p in jax.tree_util.tree_leaves(model) if eqx.is_array(p)]
        for w in weights:
            assert jnp.allclose(w, 0.0)

    def test_random_initialization(self, rng_key):
        """
        Tests if the model initializes with non-zero weights when a key is provided.
        """
        model = RUDE(key=rng_key)
        weights = [p for p in jax.tree_util.tree_leaves(model) if eqx.is_array(p)]
        
        assert len(weights) > 0
        all_zero = all(jnp.allclose(w, 0.0) for w in weights)
        assert not all_zero

    def test_forward_pass_shape(self, mock_tensors):
        """
        Tests if the forward pass runs and returns an output of the correct shape.
        """
        strain, stress = mock_tensors
        model = RUDE.zero_init()
        output = model(strain, stress)
        
        # The output should be a flattened 9-element vector
        assert output.shape == (3, 3)

    def test_jit_compilation(self, mock_tensors, rng_key):
        """
        Tests that the model's __call__ method can be JIT-compiled without errors.
        """
        strain, stress = mock_tensors
        model = RUDE(key=rng_key)

        # Wrap the model's call in a JIT-compiled function
        jitted_model_call = eqx.filter_jit(model)

        try:
            # Execute the JIT-compiled function
            output = jitted_model_call(strain, stress)
            # Check that the output is valid
            assert output is not None
            assert output.shape == (3, 3)
        except Exception as e:
            pytest.fail(f"JIT compilation failed with an exception: {e}")

    def test_rude_in_oldroydb(self, rng_key, model_params, shear_velocity_gradient, shear_stress, test_time, extra_stress_matrix):
        """
        Tests if the RUDE model can be used in the Oldroyd-B model.
        """
        model = RUDE(key=rng_key)
        rude_model = GeneralizedOldroydB(F_function=model, polymer_viscosity=model_params["polymer_viscosity"], relaxation_time=model_params["relaxation_time"], solvent_viscosity=model_params["solvent_viscosity"])
        extra_stress = rude_model.extra_stress_response_rhs(test_time, extra_stress_matrix, shear_velocity_gradient)

        assert extra_stress.shape == (3, 3)

    def test_rude_oldroydb_differentiation(self, rng_key, model_params, shear_velocity_gradient, test_time, extra_stress_matrix):
        """
        Tests if the GeneralizedOldroydB with a RUDE model can be differentiated
        with respect to the RUDE model's weights.
        """
        # 1. Set up the model
        rude_f = RUDE(key=rng_key)
        model = GeneralizedOldroydB(
            F_function=rude_f,
            polymer_viscosity=model_params["polymer_viscosity"],
            relaxation_time=model_params["relaxation_time"],
            solvent_viscosity=model_params["solvent_viscosity"]
        )

        # 2. Define a scalar loss function to differentiate
        def loss_fn(m):
            # The loss is the sum of the output tensor's elements.
            # This provides a simple scalar value for jax.grad.
            stress_rate = m.extra_stress_response_rhs(test_time, extra_stress_matrix, shear_velocity_gradient)
            return jnp.sum(stress_rate)

        # 3. Compute the gradient of the loss with respect to the entire model PyTree
        grads = eqx.filter_grad(loss_fn)(model)

        # 4. Assert that the gradients for the RUDE model's weights exist and are not all zero
        # The gradients for the RUDE model are nested inside the full model's gradients
        rude_grads = grads.F_function
        
        # Extract just the weight arrays from the RUDE gradient PyTree
        weight_grads = [
            p.weight for p in jax.tree_util.tree_leaves(rude_grads, is_leaf=lambda x: isinstance(x, eqx.nn.Linear))
        ]

        assert len(weight_grads) > 0, "No linear layer gradients found."
        
        # Check that at least one gradient is non-zero, proving differentiation worked.
        is_any_grad_nonzero = any(not jnp.allclose(g, 0.0) for g in weight_grads)
        assert is_any_grad_nonzero, "All gradients for RUDE weights are zero."


@pytest.mark.parametrize("model_cls, extra", [
    (OldroydB, {}),
    (Giesekus, {"alpha": 0.4}),
    (LinearPTT, {"epsilon": 0.2, "zeta": 0.1}),
    (ExponentialPTT, {"epsilon": 0.2, "zeta": 0.1}),
])
def test_stress_controlled_roundtrip(model_cls, extra):
    """Forward<->reverse round trip with a SIGN-CHANGING oscillatory drive.

    Regression test for the reverse-mode shear-rate sign bug: the algebraic
    stress inversion in ``_calculate_du12_dt`` must use the *signed* shear rate
    (2*D12), not the non-negative magnitude returned by
    ``_rate_of_strain_to_strain_rate``.  With the magnitude, the damping term
    flips sign whenever gamma_dot < 0 and the stress-controlled solve diverges
    for any oscillatory drive.  Here we drive a model forward with
    gamma_dot(t) = g0*sin(omega t), feed the resulting sigma12(t) back into the
    stress-controlled protocol, and require it to recover the strain.
    """
    import diff_rheo as dr

    model = model_cls(polymer_viscosity=2.0, relaxation_time=1.0,
                      solvent_viscosity=0.4, **extra)
    solver = dr.DiffraxSolver(rtol=1e-8, atol=1e-10, max_steps=200000, throw=False)
    g0, omega, t_end, n = 0.5, 0.8, 30.0, 400
    t = jnp.linspace(0.0, t_end, n)

    # forward: impose oscillatory shear rate, record sigma12 and analytic strain
    fwd = dr.VirtualRheometer.setup(model, "strain_rate_response", solver)
    L = dr.VelocityGradient.from_components(grad_u_12=lambda tt: g0 * jnp.sin(omega * tt))
    sim_f = fwd.run_experiment(model, L, t, jnp.zeros((3, 3)))
    sigma12 = sim_f.data[:, 0, 1]
    gamma_true = (g0 / omega) * (1.0 - jnp.cos(omega * t))

    # reverse: impose that sigma12, predict strain
    rev_data = dr.ShearStressData(time=t, data=gamma_true, forcing_data=sigma12,
                                  initial_condition=jnp.zeros(7))
    rev = dr.VirtualRheometer.setup(model, "shear_stress_response", solver)
    sim_r = rev.run_experiment(model, rev_data.get_forcing_function(), t, jnp.zeros(7))
    gamma_pred = rev_data.extract_from_simulation(sim_r)

    assert jnp.all(jnp.isfinite(gamma_pred)), "stress-controlled solve diverged"
    rel = jnp.sqrt(jnp.mean((gamma_pred - gamma_true) ** 2)) / (
        jnp.sqrt(jnp.mean(gamma_true ** 2)) + 1e-12)
    assert rel < 1e-2, f"reverse strain does not match forward (rel err {rel:.2e})"
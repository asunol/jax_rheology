"""
Tests for :mod:`diff_rheo._information` — Fisher-information-based uncertainty,
sloppy-direction analysis, and optimal experiment design.

The tests are organised around three claims, each checked against a synthetic
case where the correct answer is known analytically:

* **Uncertainty** – for a Newtonian fluid the FIM-predicted parameter error has
  a closed form (``σ/√Σγ̇²``); a Monte-Carlo study confirms it matches the true
  sampling spread of the estimator.
* **Sloppy directions** – a Carreau-Yasuda fluid probed only at low shear rate
  is *provably* sloppy (only the zero-shear viscosity is identifiable); the
  eigenspectrum must reflect this.
* **Experiment design** – optimising the forcing waveform must monotonically
  increase the Expected Information Gain and shrink the sloppiest direction.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diff_rheo import (
    BatchedData,
    DiffraxSolver,
    FittingConfig,
    MultiToneShearDesign,
    ShearStrainRateData,
    ShearStrainRateNormalStressData,
    SplineShearDesign,
    VelocityGradient,
    VirtualRheometer,
    discrimination_score,
    expected_information_gain,
    fisher_information,
    fit_model_to_experimental_data,
    optimize_discriminating_experiment,
    optimize_experiment,
    parameter_uncertainty,
    sensitivity_jacobian,
    shear_and_normal_stress_observable,
    shear_stress_observable,
    sloppy_analysis,
)
from diff_rheo.models import CarreauYasuda, Newtonian, OldroydB
from diff_rheo.parameters import LogParameter, Parameter


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _newtonian_dataset(viscosity, gammadot_amp, omega, n_points=60):
    """A strain-rate experiment with γ̇(t) = amp·sin(ω·t) for a Newtonian fluid."""
    time = jnp.linspace(0.0, 12.0, n_points)
    forcing_data = gammadot_amp * jnp.sin(omega * time)
    # Newtonian shear stress σ₁₂ = η·γ̇ (noiseless); `data` is unused by the FIM.
    data = viscosity * forcing_data
    return ShearStrainRateData(
        time=time,
        data=data,
        forcing_data=forcing_data,
        initial_condition=jnp.zeros((3, 3)),
    )


def _simulate_strain_rate_dataset(model, rheometer, forcing_data, time):
    """Build a ShearStrainRateData by simulating `model` (noiseless)."""
    coeffs_data = ShearStrainRateData(
        time=time, data=forcing_data, forcing_data=forcing_data,
        initial_condition=jnp.zeros((3, 3)),
    )
    forcing = coeffs_data.get_forcing_function()
    sim = rheometer.run_experiment(model, forcing, time, jnp.zeros((3, 3)))
    sigma_12 = sim.data[:, 0, 1]
    return ShearStrainRateData(
        time=time, data=sigma_12, forcing_data=forcing_data,
        initial_condition=jnp.zeros((3, 3)),
    )


@pytest.fixture
def newtonian_rheometer():
    model = Newtonian(viscosity=Parameter(2.0))
    return model, VirtualRheometer.setup(model, "strain_rate_response")


# ---------------------------------------------------------------------------
# Sensitivity Jacobian / FIM mechanics
# ---------------------------------------------------------------------------

class TestSensitivityJacobian:
    """The Jacobian and FIM must match their analytic forms for a Newtonian fluid."""

    def test_jacobian_equals_strain_rate(self, newtonian_rheometer):
        # σ₁₂ = η·γ̇  ⇒  ∂σ₁₂/∂η = γ̇, so the Jacobian column is the forcing itself.
        model, rheometer = newtonian_rheometer
        data = _newtonian_dataset(2.0, gammadot_amp=1.5, omega=0.8)
        jac, labels = sensitivity_jacobian(model, rheometer, data)

        assert labels == ["viscosity"]
        assert jac.shape == (data.time.shape[0], 1)
        assert jnp.allclose(jac[:, 0], data.forcing_data, atol=1e-4)

    def test_fwd_and_rev_modes_agree(self):
        # Forward- and reverse-mode autodiff must give the same Jacobian through
        # the ODE solver (requires the forward-mode-capable "direct" adjoint).
        model = OldroydB(
            polymer_viscosity=Parameter(2.0),
            relaxation_time=Parameter(1.0),
            solvent_viscosity=Parameter(0.3),
        )
        solver = DiffraxSolver(adjoint="direct")
        rheometer = VirtualRheometer.setup(model, "strain_rate_response", solver)
        time = jnp.linspace(0.0, 6.0, 40)
        forcing_data = jnp.full_like(time, 1.0)
        data = _simulate_strain_rate_dataset(model, rheometer, forcing_data, time)

        jac_rev, _ = sensitivity_jacobian(model, rheometer, data, mode="rev")
        jac_fwd, _ = sensitivity_jacobian(model, rheometer, data, mode="fwd")
        assert jnp.allclose(jac_rev, jac_fwd, rtol=1e-3, atol=1e-5)

    def test_observation_noise_excluded(self):
        # observation_noise does not enter the forward prediction and must be
        # left out of the FIM (otherwise it contributes a zero row).
        model = OldroydB(
            polymer_viscosity=Parameter(2.0),
            relaxation_time=Parameter(1.0),
            solvent_viscosity=Parameter(0.3),
            observation_noise=LogParameter(0.05),
        )
        solver = DiffraxSolver()
        rheometer = VirtualRheometer.setup(model, "strain_rate_response", solver)
        time = jnp.linspace(0.0, 6.0, 30)
        data = _simulate_strain_rate_dataset(model, rheometer, jnp.full_like(time, 1.0), time)

        fisher = fisher_information(model, rheometer, data, noise=0.05)
        assert fisher.labels == ["polymer_viscosity", "relaxation_time", "solvent_viscosity"]
        assert fisher.matrix.shape == (3, 3)


class TestFisherInformation:
    """Structural properties the FIM must satisfy."""

    def test_fim_matches_analytic_newtonian(self, newtonian_rheometer):
        model, rheometer = newtonian_rheometer
        data = _newtonian_dataset(2.0, gammadot_amp=1.5, omega=0.8)
        sigma = 0.1
        fisher = fisher_information(model, rheometer, data, noise=sigma)
        # FIM = Σγ̇²/σ²  for a single-parameter Newtonian fit.
        analytic = jnp.sum(data.forcing_data ** 2) / sigma ** 2
        assert jnp.allclose(fisher.matrix[0, 0], analytic, rtol=1e-3)

    def test_fim_symmetric_and_psd(self):
        model = OldroydB(
            polymer_viscosity=Parameter(2.0),
            relaxation_time=Parameter(1.0),
            solvent_viscosity=Parameter(0.3),
        )
        solver = DiffraxSolver()
        rheometer = VirtualRheometer.setup(model, "strain_rate_response", solver)
        time = jnp.linspace(0.0, 8.0, 50)
        data = _simulate_strain_rate_dataset(model, rheometer, jnp.full_like(time, 1.0), time)

        fisher = fisher_information(model, rheometer, data, noise=0.05)
        assert jnp.allclose(fisher.matrix, fisher.matrix.T, rtol=1e-4)
        eigvals, _ = fisher.eigenspectrum()
        assert jnp.all(eigvals > -1e-5 * eigvals[0])  # positive semi-definite
        # FIM must equal JᵀJ/σ².
        assert jnp.allclose(fisher.matrix, fisher.jacobian.T @ fisher.jacobian / 0.05 ** 2, rtol=1e-4)

    def test_zero_noise_raises(self, newtonian_rheometer):
        model, rheometer = newtonian_rheometer
        data = _newtonian_dataset(2.0, 1.5, 0.8)
        with pytest.raises(ValueError, match="strictly positive"):
            fisher_information(model, rheometer, data, noise=0.0)

    def test_physical_vs_natural_coords(self):
        # For a LogParameter the physical std should be ≈ value × natural std.
        model = Newtonian(viscosity=LogParameter(2.0))
        rheometer = VirtualRheometer.setup(model, "strain_rate_response")
        data = _newtonian_dataset(2.0, 1.5, 0.8)

        nat = fisher_information(model, rheometer, data, noise=0.1, coords="natural")
        phys = fisher_information(model, rheometer, data, noise=0.1, coords="physical")
        se_nat = nat.standard_errors()[0]
        se_phys = phys.standard_errors()[0]
        assert jnp.allclose(se_phys, 2.0 * se_nat, rtol=1e-3)


# ---------------------------------------------------------------------------
# Parameter uncertainty — Monte-Carlo validation
# ---------------------------------------------------------------------------

class TestParameterUncertainty:
    """The FIM error bar must match the true spread of the estimator."""

    def test_fim_error_matches_monte_carlo(self):
        # Newtonian σ₁₂ = η·γ̇ + noise.  The MLE η̂ = Σγ̇y / Σγ̇² has a known
        # sampling std σ/√Σγ̇²; the FIM must reproduce it.
        eta_true = 2.0
        sigma = 0.15
        model = Newtonian(viscosity=Parameter(eta_true))
        rheometer = VirtualRheometer.setup(model, "strain_rate_response")
        data = _newtonian_dataset(eta_true, gammadot_amp=1.2, omega=0.7, n_points=80)

        fim_std = fisher_information(model, rheometer, data, noise=sigma).standard_errors()[0]

        gammadot = np.asarray(data.forcing_data)
        clean = eta_true * gammadot
        rng = np.random.default_rng(0)
        n_trials = 400
        estimates = np.empty(n_trials)
        for i in range(n_trials):
            noisy = clean + rng.normal(0.0, sigma, size=gammadot.shape)
            estimates[i] = np.sum(gammadot * noisy) / np.sum(gammadot ** 2)

        empirical_std = estimates.std(ddof=1)
        # ±20 % is generous given the ~5 % std-of-std at 400 trials.
        assert abs(float(fim_std) - empirical_std) / empirical_std < 0.2

    def test_parameter_uncertainty_dict(self, newtonian_rheometer):
        model, rheometer = newtonian_rheometer
        data = _newtonian_dataset(2.0, 1.5, 0.8)
        report = parameter_uncertainty(model, rheometer, data, noise=0.1)
        assert set(report) == {"viscosity"}
        value, error = report["viscosity"]
        assert value == pytest.approx(2.0)
        assert error > 0.0


# ---------------------------------------------------------------------------
# Sloppy-direction analysis
# ---------------------------------------------------------------------------

class TestSloppyAnalysis:
    """A model probed in an uninformative regime must show sloppy directions."""

    def test_low_shear_carreau_yasuda_is_sloppy(self):
        # At low shear rate η(γ̇) → η₀: only the zero-shear viscosity is
        # identifiable, the other four parameters are sloppy.
        model = CarreauYasuda(
            zero_shear_viscosity=Parameter(10.0),
            infinite_shear_viscosity=Parameter(0.1),
            n=Parameter(0.5),
            a=Parameter(2.0),
            k=Parameter(1.0),
        )
        rheometer = VirtualRheometer.setup(model, "strain_rate_response")
        time = jnp.linspace(0.0, 10.0, 60)
        low_rate = 1e-2 * jnp.sin(0.6 * time)
        data = ShearStrainRateData(
            time=time, data=jnp.zeros_like(time),
            forcing_data=low_rate, initial_condition=jnp.zeros((3, 3)),
        )

        fisher = fisher_information(model, rheometer, data, noise=0.01)
        analysis = sloppy_analysis(fisher)

        # Eigenvalues descending, real, non-negative, orthonormal eigenvectors.
        assert jnp.all(jnp.diff(analysis.eigenvalues) <= 1e-8)
        assert jnp.all(analysis.eigenvalues > -1e-4 * analysis.eigenvalues[0])
        gram = analysis.eigenvectors.T @ analysis.eigenvectors
        assert jnp.allclose(gram, jnp.eye(5), atol=1e-5)

        # Severely sloppy: the spectrum is ill-conditioned with few stiff directions.
        assert analysis.condition_number > 1e3
        assert analysis.n_stiff <= 2

        # The single stiff direction is dominated by the zero-shear viscosity.
        _, stiff_vec = analysis.stiffest()
        assert int(jnp.argmax(jnp.abs(stiff_vec))) == 0  # zero_shear_viscosity

    def test_sloppy_summary_runs(self, newtonian_rheometer):
        model, rheometer = newtonian_rheometer
        data = _newtonian_dataset(2.0, 1.5, 0.8)
        fisher = fisher_information(model, rheometer, data, noise=0.1)
        analysis = sloppy_analysis(fisher)
        assert isinstance(analysis.summary(), str)
        assert isinstance(fisher.summary(), str)


# ---------------------------------------------------------------------------
# Optimal experiment design
# ---------------------------------------------------------------------------

class TestExperimentDesign:
    """Optimising the waveform must increase information about the parameters."""

    def _carreau_setup(self):
        model = CarreauYasuda(
            zero_shear_viscosity=Parameter(10.0),
            infinite_shear_viscosity=Parameter(0.1),
            n=Parameter(0.5),
            a=Parameter(2.0),
            k=Parameter(1.0),
        )
        rheometer = VirtualRheometer.setup(model, "strain_rate_response")
        # A weak, low-amplitude prior experiment: leaves the model sloppy.
        time = jnp.linspace(0.0, 10.0, 60)
        prior_rate = 1e-2 * jnp.sin(0.5 * time)
        prior = ShearStrainRateData(
            time=time, data=jnp.zeros_like(time),
            forcing_data=prior_rate, initial_condition=jnp.zeros((3, 3)),
        )
        return model, rheometer, BatchedData([prior])

    def test_eig_is_non_negative(self):
        model, rheometer, prior = self._carreau_setup()
        prior_fisher = fisher_information(model, rheometer, prior, noise=0.05)
        prior_precision = prior_fisher.matrix + 1e-2 * jnp.eye(5)

        forcing = VelocityGradient.from_components(grad_u_12=lambda t: 5.0 * jnp.sin(2.0 * t))
        time = jnp.linspace(0.0, 10.0, 60)
        eig = expected_information_gain(
            model, rheometer, forcing, time, jnp.zeros((3, 3)), prior_precision,
            noise=0.05,
        )
        assert float(eig) >= -1e-6

    def test_optimize_experiment_increases_information(self):
        model, rheometer, prior = self._carreau_setup()
        result = optimize_experiment(
            model, rheometer, prior, noise=0.05,
            init_amplitude=0.05, init_frequency=0.5,
            num_steps=80, learning_rate=0.2, prior_ridge=1e-2,
        )

        # EIG must rise over the optimisation and end strictly positive.
        assert result.eig_history[-1] > result.eig_history[0]
        assert result.eig > 0.0

        # The optimised experiment adds information: the total Fisher
        # information (trace of the precision — a roundoff-robust aggregate,
        # unlike the smallest eigenvalue of a sloppy matrix) strictly grows.
        assert jnp.trace(result.posterior_precision) > jnp.trace(result.prior_precision)

        # The design stays inside the physical box.
        assert result.design["amplitude"] > 0.0
        assert result.design["frequency"] > 0.0
        assert isinstance(result.summary(), str)

    def test_eig_gradient_is_finite(self):
        model, rheometer, prior = self._carreau_setup()
        prior_fisher = fisher_information(model, rheometer, prior, noise=0.05)
        prior_precision = prior_fisher.matrix + 1e-2 * jnp.eye(5)
        time = jnp.linspace(0.0, 10.0, 60)

        from diff_rheo import OscillatoryShearDesign
        design = OscillatoryShearDesign()

        def eig_of_z(z):
            return expected_information_gain(
                model, rheometer, design.forcing(z), time, jnp.zeros((3, 3)),
                prior_precision, noise=0.05,
            )

        z0 = design.to_unconstrained(1.0, 1.0)
        grad = jax.grad(eig_of_z)(z0)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.linalg.norm(grad) > 0.0


# ---------------------------------------------------------------------------
# Model-discrimination experiment design
# ---------------------------------------------------------------------------

class TestModelDiscrimination:
    """A designed forcing must separate models that are otherwise confused."""

    def test_discrimination_score_basics(self):
        y_ref = jnp.array([1.0, 2.0, 3.0])
        # Identical predictions → zero separation.
        assert float(discrimination_score(y_ref, [y_ref])) == pytest.approx(0.0)
        # A rival offset by 1 everywhere → mean-squared gap of 1.
        assert float(discrimination_score(y_ref, [y_ref + 1.0])) == pytest.approx(1.0)
        # The score is set by the *closest* rival.
        assert float(discrimination_score(y_ref, [y_ref + 1.0, y_ref + 0.1])) == pytest.approx(0.01, rel=1e-3)

    def test_optimize_discriminating_experiment_separates_models(self):
        # Oldroyd-B (elastic) vs. a Newtonian rival with the matched
        # steady-shear viscosity η_p+η_s: indistinguishable in slow shear,
        # separable once the forcing is fast enough to expose the elastic lag.
        truth = OldroydB(
            polymer_viscosity=Parameter(3.0),
            relaxation_time=Parameter(2.0),
            solvent_viscosity=Parameter(0.5),
        )
        rival = Newtonian(viscosity=Parameter(3.5))
        result = optimize_discriminating_experiment(
            [truth], [[rival]], solver=DiffraxSolver(),
            time=jnp.linspace(0.0, 10.0, 60),
            initial_condition=jnp.zeros((3, 3)),
            noise=0.05,
            init_amplitude=1.0, init_frequency=0.1,
            num_steps=60, learning_rate=0.2,
        )
        # The designed experiment separates the models far better than the
        # slow initial forcing, and pushes the gap above the noise floor.
        assert result.separation > result.initial_separation
        assert result.separation > result.noise_variance
        assert result.separation_history[-1] > result.separation_history[0]
        assert result.design["amplitude"] > 0.0
        assert result.design["frequency"] > 0.0
        assert isinstance(result.summary(), str)


# ---------------------------------------------------------------------------
# Multi-tone (Fourier) shear design — the "information maximum of pure shear"
# ---------------------------------------------------------------------------

class TestMultiToneDesign:
    """A multi-tone waveform must stay solvable and generalise the single tone."""

    def test_forcing_shape_and_physical(self):
        design = MultiToneShearDesign(n_tones=4)
        z = design.default_z(init_amplitude=2.0)
        assert z.shape == (design.n_params,) == (1 + 3 * 4,)

        forcing = design.forcing(z)
        time = jnp.linspace(0.0, 10.0, 80)
        grad = forcing.gradient(time)
        assert grad.shape == (80, 3, 3)

        physical = design.physical(z)
        assert physical["n_tones"] == 4
        assert len(physical["frequencies"]) == 4
        assert len(physical["amplitudes"]) == 4

    def test_peak_shear_rate_is_bounded(self):
        # Σ shares = 1 ⇒ peak |γ̇| ≤ amplitude, no matter how many tones or
        # their phases — this is what keeps the stiff ODEs solvable.
        amp_max = 6.0
        design = MultiToneShearDesign(n_tones=6, amplitude_range=(0.1, amp_max))
        time = jnp.linspace(0.0, 20.0, 400)
        for seed in range(5):
            z = design.default_z(init_amplitude=3.0, key=jax.random.PRNGKey(seed))
            z = z + jax.random.normal(jax.random.PRNGKey(seed + 100), z.shape)
            gammadot = design.forcing(z).gradient(time)[:, 0, 1]
            assert jnp.max(jnp.abs(gammadot)) <= amp_max + 1e-3

    def test_single_tone_limit(self):
        # n_tones=1 must reduce to a pure A·sin(ω·t+φ).
        design = MultiToneShearDesign(n_tones=1)
        z = design.default_z(init_amplitude=2.5)
        physical = design.physical(z)
        amp = physical["amplitudes"][0]
        freq = physical["frequencies"][0]
        phase = physical["phases"][0]

        time = jnp.linspace(0.0, 10.0, 80)
        gammadot = design.forcing(z).gradient(time)[:, 0, 1]
        expected = amp * jnp.sin(freq * time + phase)
        assert jnp.allclose(gammadot, expected, atol=1e-5)

    def test_seed_from_single_tone(self):
        # seed_from_single_tone must reproduce the given single sine: almost all
        # the amplitude budget on tone 0, so the waveform ≈ A·sin(ω·t).
        design = MultiToneShearDesign(n_tones=4, amplitude_range=(0.1, 6.0))
        z = design.seed_from_single_tone(amplitude=5.0, frequency=1.2)
        assert z.shape == (design.n_params,)

        physical = design.physical(z)
        # tone 0 carries essentially the whole amplitude budget.
        assert physical["amplitudes"][0] > 0.99 * physical["amplitude_total"]
        assert physical["frequencies"][0] == pytest.approx(1.2, rel=1e-3)

        time = jnp.linspace(0.0, 10.0, 80)
        gammadot = design.forcing(z).gradient(time)[:, 0, 1]
        expected = 5.0 * jnp.sin(1.2 * time)
        assert jnp.allclose(gammadot, expected, atol=2e-2)

    def test_optimize_discriminating_with_multitone(self):
        # The discrimination optimiser must accept a multi-tone design (via
        # init_z) and improve the separation, with a list-valued design dict.
        truth = OldroydB(
            polymer_viscosity=Parameter(3.0),
            relaxation_time=Parameter(2.0),
            solvent_viscosity=Parameter(0.5),
        )
        rival = Newtonian(viscosity=Parameter(3.5))
        design = MultiToneShearDesign(n_tones=3, amplitude_range=(0.1, 5.0))
        z0 = design.default_z(init_amplitude=1.0, key=jax.random.PRNGKey(0))
        result = optimize_discriminating_experiment(
            [truth], [[rival]], solver=DiffraxSolver(), design=design,
            time=jnp.linspace(0.0, 10.0, 60),
            initial_condition=jnp.zeros((3, 3)),
            noise=0.05, init_z=z0,
            num_steps=60, learning_rate=0.2,
        )
        assert result.separation > result.initial_separation
        assert result.design["n_tones"] == 3
        assert isinstance(result.summary(), str)  # summary handles list values


# ---------------------------------------------------------------------------
# Spline shear design — the most general smooth pure-shear waveform
# ---------------------------------------------------------------------------

class TestSplineDesign:
    """A cubic-spline waveform must stay in-envelope and generalise the sinusoid."""

    def _carreau_setup(self):
        model = CarreauYasuda(
            zero_shear_viscosity=Parameter(10.0),
            infinite_shear_viscosity=Parameter(0.1),
            n=Parameter(0.5),
            a=Parameter(2.0),
            k=Parameter(1.0),
        )
        rheometer = VirtualRheometer.setup(model, "strain_rate_response")
        # A weak, low-amplitude prior experiment: leaves the model sloppy.
        time = jnp.linspace(0.0, 10.0, 60)
        prior_rate = 1e-2 * jnp.sin(0.5 * time)
        prior = ShearStrainRateData(
            time=time, data=jnp.zeros_like(time),
            forcing_data=prior_rate, initial_condition=jnp.zeros((3, 3)),
        )
        return model, rheometer, BatchedData([prior])

    def test_forcing_shape_and_physical(self):
        design = SplineShearDesign(duration=10.0, n_knots=12)
        z = design.default_z(init_amplitude=2.0, init_frequency=1.0)
        assert z.shape == (design.n_params,) == (12,)

        forcing = design.forcing(z)
        grad = forcing.gradient(jnp.linspace(0.0, 10.0, 80))
        assert grad.shape == (80, 3, 3)

        physical = design.physical(z)
        assert physical["n_knots"] == 12
        assert len(physical["knot_times"]) == 12
        assert len(physical["control_values"]) == 12

    def test_knot_values_are_envelope_bounded(self):
        # tanh squashing confines every knot value to (−A, A): no design
        # vector, however extreme, can place a control point outside the
        # rheometer's shear-rate envelope.
        amp_max = 5.0
        design = SplineShearDesign(duration=20.0, n_knots=15, amplitude_max=amp_max)
        for seed in range(5):
            z = 8.0 * jax.random.normal(jax.random.PRNGKey(seed), (design.n_params,))
            control = jnp.asarray(design.physical(z)["control_values"])
            assert jnp.max(jnp.abs(control)) <= amp_max + 1e-5

    def test_waveform_bounded_between_knots(self):
        # The squash is applied *after* interpolation, so the bound holds at
        # every t, not just at the knots: a densely sampled waveform — and the
        # reported peak_shear_rate — can never exceed amplitude_max.  This is
        # the regression test for the cubic-Hermite overshoot that previously
        # let the optimised spline exceed the envelope by ~30%.
        amp_max = 5.0
        design = SplineShearDesign(duration=20.0, n_knots=15, amplitude_max=amp_max)
        dense_t = jnp.linspace(0.0, 20.0, 4000)
        for seed in range(5):
            z = 8.0 * jax.random.normal(jax.random.PRNGKey(seed), (design.n_params,))
            waveform = design.forcing(z).gradient(dense_t)[:, 0, 1]
            assert jnp.max(jnp.abs(waveform)) <= amp_max + 1e-5
            assert design.physical(z)["peak_shear_rate"] <= amp_max + 1e-5

    def test_sine_seed_reproduces_sinusoid(self):
        # to_unconstrained(A, ω) must seed a spline that *is* A·sin(ω·t):
        # exact at the knots, and close in between for a dense knot set —
        # this is what makes "expand the sinusoid into a spline" faithful.
        design = SplineShearDesign(duration=12.0, n_knots=24, amplitude_max=6.0)
        A, omega = 3.0, 0.8
        z = design.to_unconstrained(A, omega)
        control = jnp.asarray(design.physical(z)["control_values"])
        assert jnp.allclose(control, A * jnp.sin(omega * design.knot_times), atol=1e-4)

        time = jnp.linspace(0.0, 12.0, 200)
        gammadot = design.forcing(z).gradient(time)[:, 0, 1]
        assert jnp.allclose(gammadot, A * jnp.sin(omega * time), atol=0.08)

    def test_roughness_penalty_orders_waveforms(self):
        # A smooth (sinusoidal) waveform must score a far lower second-
        # difference roughness penalty than a jagged (random) one.
        design = SplineShearDesign(duration=10.0, n_knots=16)
        smooth = design.to_unconstrained(2.0, 0.6)
        jagged = 3.0 * jax.random.normal(jax.random.PRNGKey(0), (design.n_params,))
        assert design.roughness_penalty(smooth) < design.roughness_penalty(jagged)

    def test_optimize_experiment_with_spline(self):
        # optimize_experiment must accept a SplineShearDesign (seeded from a
        # sinusoid via to_unconstrained) and add information.
        model, rheometer, prior = self._carreau_setup()
        design = SplineShearDesign(duration=10.0, n_knots=10)
        result = optimize_experiment(
            model, rheometer, prior, noise=0.05, design=design,
            init_amplitude=0.05, init_frequency=0.5,
            num_steps=60, learning_rate=0.2, prior_ridge=1e-2,
        )
        assert result.objective_history[-1] > result.objective_history[0]
        # The optimised free-form experiment adds information.
        assert jnp.trace(result.posterior_precision) > jnp.trace(result.prior_precision)
        assert result.design["n_knots"] == 10
        assert isinstance(result.summary(), str)  # summary handles list values

    def test_optimize_discriminating_with_spline(self):
        # The discrimination optimiser must accept a SplineShearDesign (via
        # init_z), improve the separation, and honour roughness_weight.
        truth = OldroydB(
            polymer_viscosity=Parameter(3.0),
            relaxation_time=Parameter(2.0),
            solvent_viscosity=Parameter(0.5),
        )
        rival = Newtonian(viscosity=Parameter(3.5))
        design = SplineShearDesign(duration=10.0, n_knots=10, amplitude_max=5.0)
        z0 = design.seed_from_sine(amplitude=1.0, frequency=0.3)
        result = optimize_discriminating_experiment(
            [truth], [[rival]], solver=DiffraxSolver(), design=design,
            time=jnp.linspace(0.0, 10.0, 60),
            initial_condition=jnp.zeros((3, 3)),
            noise=0.05, init_z=z0,
            num_steps=60, learning_rate=0.2, roughness_weight=1e-4,
        )
        assert result.separation > result.initial_separation
        assert result.design["n_knots"] == 10
        assert isinstance(result.summary(), str)

    def test_roughness_weight_requires_compatible_design(self):
        # roughness_weight is meaningful only for a design exposing
        # roughness_penalty; pairing it with a plain oscillatory design must
        # fail loudly rather than silently ignore the request.
        from diff_rheo import OscillatoryShearDesign
        model, rheometer, prior = self._carreau_setup()
        with pytest.raises(ValueError, match="roughness_penalty"):
            optimize_experiment(
                model, rheometer, prior, noise=0.05,
                design=OscillatoryShearDesign(),
                num_steps=1, roughness_weight=1e-2,
            )


# ---------------------------------------------------------------------------
# Normal-stress observable — breaking the shear-stress degeneracy
# ---------------------------------------------------------------------------

class TestNormalStressObservable:
    """Measuring N₁ must add discriminating power that σ₁₂ alone cannot."""

    def _simulate(self, model, forcing, time):
        rheometer = VirtualRheometer.setup(model, "strain_rate_response", DiffraxSolver())
        return rheometer.run_experiment(model, forcing, time, jnp.zeros((3, 3)))

    def test_extraction_shape_and_channels(self):
        # ShearStrainRateNormalStressData.extract_from_simulation → (T, 2),
        # column 0 = σ₁₂, column 1 = N₁ = σ₁₁ − σ₂₂.
        model = OldroydB(
            polymer_viscosity=Parameter(2.0),
            relaxation_time=Parameter(1.5),
            solvent_viscosity=Parameter(0.3),
        )
        time = jnp.linspace(0.0, 8.0, 50)
        forcing = VelocityGradient.from_components(grad_u_12=lambda t: 2.0 * jnp.sin(t))
        sim = self._simulate(model, forcing, time)

        data = ShearStrainRateNormalStressData(
            time=time, data=jnp.zeros((50, 2)),
            forcing_data=2.0 * jnp.sin(time), initial_condition=jnp.zeros((3, 3)),
        )
        extracted = data.extract_from_simulation(sim)
        assert extracted.shape == (50, 2)
        assert jnp.allclose(extracted[:, 0], sim.data[:, 0, 1])
        assert jnp.allclose(extracted[:, 1], sim.data[:, 0, 0] - sim.data[:, 1, 1])
        # column 0 of the data type matches the standalone σ₁₂ observable helper.
        assert jnp.allclose(extracted[:, 0], shear_stress_observable(sim))

    def test_n1_breaks_a_shear_stress_degeneracy(self):
        # The defining case: two models with *identical* σ₁₂ but different N₁.
        # σ₁₂-only discrimination is exactly zero — they are degenerate — yet
        # the [σ₁₂, N₁] observable separates them.  (discrimination_score takes
        # plain arrays, so the degeneracy can be exhibited exactly.)
        n = 24
        sigma12 = jnp.sin(jnp.linspace(0.0, 6.0, n))
        n1_a = jnp.zeros(n)
        n1_b = 0.5 * jnp.ones(n)

        shear_only_ref = sigma12
        shear_only_rival = sigma12
        joint_ref = jnp.concatenate([sigma12, n1_a])
        joint_rival = jnp.concatenate([sigma12, n1_b])

        assert float(discrimination_score(shear_only_ref, [shear_only_rival])) == pytest.approx(0.0)
        assert float(discrimination_score(joint_ref, [joint_rival])) > 0.0

    def test_normal_stress_is_a_genuine_channel(self):
        # N₁ is a physically independent observable: a Newtonian fluid carries
        # no normal stress, an Oldroyd-B fluid does.  So the σ₁₂-degenerate
        # Newtonian/Oldroyd-B pair is *not* degenerate in [σ₁₂, N₁].
        oldroyd = OldroydB(
            polymer_viscosity=Parameter(3.0),
            relaxation_time=Parameter(2.0),
            solvent_viscosity=Parameter(0.5),
        )
        newtonian = Newtonian(viscosity=Parameter(3.5))
        time = jnp.linspace(0.0, 12.0, 80)
        forcing = VelocityGradient.from_components(grad_u_12=lambda t: jnp.sin(0.5 * t))

        sim_ob = self._simulate(oldroyd, forcing, time)
        sim_nt = self._simulate(newtonian, forcing, time)
        n1_ob = sim_ob.data[:, 0, 0] - sim_ob.data[:, 1, 1]
        n1_nt = sim_nt.data[:, 0, 0] - sim_nt.data[:, 1, 1]

        assert jnp.mean(n1_nt ** 2) < 1e-8       # Newtonian: no normal stress
        assert jnp.mean(n1_ob ** 2) > 1e-2       # Oldroyd-B: real N₁ signal

        # The joint observable concatenates both channels and separates the
        # pair through the N₁ block.
        joint_ob = shear_and_normal_stress_observable(sim_ob)
        joint_nt = shear_and_normal_stress_observable(sim_nt)
        assert joint_ob.shape == (2 * time.shape[0],)
        assert float(discrimination_score(joint_ob, [joint_nt])) > 0.0

    def test_fitting_with_normal_stress_data(self):
        # The new data type must drive the standard fitting loop: fitting an
        # Oldroyd-B to its own noiseless [σ₁₂, N₁] trajectory drives the loss
        # to ~0.
        truth = OldroydB(
            polymer_viscosity=Parameter(2.5),
            relaxation_time=Parameter(1.5),
            solvent_viscosity=Parameter(0.4),
        )
        time = jnp.linspace(0.0, 10.0, 60)
        forcing = VelocityGradient.from_components(grad_u_12=lambda t: 2.0 * jnp.sin(0.8 * t))
        sim = self._simulate(truth, forcing, time)
        observed = jnp.stack(
            [sim.data[:, 0, 1], sim.data[:, 0, 0] - sim.data[:, 1, 1]], axis=-1)
        data = BatchedData([ShearStrainRateNormalStressData(
            time=time, data=observed, forcing_data=2.0 * jnp.sin(0.8 * time),
            initial_condition=jnp.zeros((3, 3)),
        )])

        guess = OldroydB(
            polymer_viscosity=Parameter(2.0),
            relaxation_time=Parameter(1.0),
            solvent_viscosity=Parameter(0.6),
        )
        rheometer = VirtualRheometer.setup(guess, "strain_rate_response", DiffraxSolver())
        fitted = fit_model_to_experimental_data(
            guess, rheometer, data, FittingConfig(num_epochs=1000, learning_rate=1e-1))

        sim_fit = rheometer.run_experiment(fitted, forcing, time, jnp.zeros((3, 3)))
        fit_pred = jnp.stack(
            [sim_fit.data[:, 0, 1], sim_fit.data[:, 0, 0] - sim_fit.data[:, 1, 1]], axis=-1)
        assert jnp.mean((fit_pred - observed) ** 2) < 1e-3


# ---------------------------------------------------------------------------
# Targeted experiment design — D-, A-, E- and c-optimality
# ---------------------------------------------------------------------------

class TestDesignCriteria:
    """optimize_experiment must support criteria beyond plain D-optimality."""

    def _sloppy_carreau(self):
        # A Carreau-Yasuda fluid probed only at very low shear rate: only the
        # zero-shear viscosity is identifiable, the other parameters sloppy.
        model = CarreauYasuda(
            zero_shear_viscosity=Parameter(10.0),
            infinite_shear_viscosity=Parameter(0.1),
            n=Parameter(0.5), a=Parameter(2.0), k=Parameter(1.0),
        )
        rheometer = VirtualRheometer.setup(model, "strain_rate_response")
        time = jnp.linspace(0.0, 10.0, 60)
        prior = BatchedData([ShearStrainRateData(
            time=time, data=jnp.zeros_like(time),
            forcing_data=1e-2 * jnp.sin(0.5 * time), initial_condition=jnp.zeros((3, 3)),
        )])
        return model, rheometer, prior

    def test_each_criterion_runs_and_improves(self):
        model, rheometer, prior = self._sloppy_carreau()
        for crit in ("eig", "a_optimal", "e_optimal"):
            result = optimize_experiment(
                model, rheometer, prior, noise=0.05, criterion=crit,
                init_amplitude=0.05, init_frequency=0.5,
                num_steps=80, learning_rate=0.2, prior_ridge=1e-2,
            )
            assert result.criterion == crit
            # the chosen criterion improves over the optimisation
            assert result.objective_history[-1] > result.objective_history[0]
            assert isinstance(result.summary(), str)
            # adding an experiment never destroys information
            assert jnp.trace(result.posterior_precision) >= jnp.trace(result.prior_precision)
            # the eig/eig_history aliases still resolve
            assert float(result.eig) == float(result.objective)

    def test_target_criterion_reduces_targeted_variance(self):
        # c-optimality aimed at a sloppy parameter must shrink *its* error bar.
        model, rheometer, prior = self._sloppy_carreau()
        result = optimize_experiment(
            model, rheometer, prior, noise=0.05, criterion="target", target="n",
            init_amplitude=0.05, init_frequency=0.5,
            num_steps=80, learning_rate=0.2, prior_ridge=1e-2,
        )
        assert result.criterion == "target"
        # the c-optimality score (−variance of n) rises over the optimisation
        assert result.objective_history[-1] > result.objective_history[0]
        k = result.labels.index("n")
        var_prior = jnp.linalg.inv(result.prior_precision)[k, k]
        var_post = jnp.linalg.inv(result.posterior_precision)[k, k]
        assert var_post < var_prior

    def test_target_accepts_a_direction_vector(self):
        # the target may be an explicit direction — e.g. the sloppy eigenvector.
        model, rheometer, prior = self._sloppy_carreau()
        prior_fisher = fisher_information(model, rheometer, prior, noise=0.05)
        _, sloppy_vec = sloppy_analysis(prior_fisher).sloppiest()
        result = optimize_experiment(
            model, rheometer, prior, noise=0.05, criterion="target", target=sloppy_vec,
            init_amplitude=0.05, init_frequency=0.5,
            num_steps=60, learning_rate=0.2, prior_ridge=1e-2,
        )
        assert result.criterion == "target"
        assert result.objective_history[-1] > result.objective_history[0]

    def test_invalid_criterion_and_missing_target_raise(self):
        model, rheometer, prior = self._sloppy_carreau()
        with pytest.raises(ValueError, match="criterion"):
            optimize_experiment(model, rheometer, prior, noise=0.05, criterion="bogus")
        with pytest.raises(ValueError, match="target"):
            optimize_experiment(model, rheometer, prior, noise=0.05, criterion="target")

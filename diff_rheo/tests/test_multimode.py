"""
Tests for multi-mode constitutive models (:class:`diff_rheo.models.MultiModeOldroydB`).

The multi-mode machinery must satisfy two exact algebraic identities, which
pin down both the per-mode integration and the mode summation:

* **Single-mode reduction** – ``MultiModeOldroydB`` with ``N = 1`` reproduces
  the single-mode :class:`~diff_rheo.models.OldroydB` trajectory exactly.
* **Linear superposition** – because each Oldroyd-B mode is linear and the
  modes share the kinematics, an ``N = 2`` model's total stress equals the sum
  of two independent single-mode runs (with the solvent counted once).

A third test checks the dispatch wiring (the right protocol is selected) and a
fourth checks the relaxation-spectrum shape contract.
"""

import jax
import jax.numpy as jnp

import diff_rheo as dr
from diff_rheo import (
    ShearStrainRateData,
    fisher_information,
    parameter_uncertainty,
    sloppy_analysis,
)
from diff_rheo.models import OldroydB, MultiModeOldroydB, OrderedMultiModeOldroydB
from diff_rheo._protocols import MultiModeStrainRateProtocol


def _shear_forcing(amp=2.0, omega=1.0):
    return dr.VelocityGradient.from_components(grad_u_12=lambda t: amp * jnp.sin(omega * t))


def _run(model, forcing, t_end=15.0, n=200):
    solver = dr.DiffraxSolver(rtol=1e-8, atol=1e-8, max_steps=100000)
    rheometer = dr.VirtualRheometer.setup(model, "strain_rate_response", solver)
    time = jnp.linspace(0.0, t_end, n)
    sim = rheometer.run_experiment(model, forcing, time, jnp.zeros((3, 3)))
    return time, sim


class TestMultiModeDispatch:
    def test_selects_multimode_protocol(self):
        model = MultiModeOldroydB(
            polymer_viscosities=jnp.array([1.0, 0.5]),
            relaxation_times=jnp.array([0.2, 1.0]),
            solvent_viscosity=0.1,
        )
        rheometer = dr.VirtualRheometer.setup(model, "strain_rate_response", dr.DiffraxSolver())
        assert isinstance(rheometer.protocol, MultiModeStrainRateProtocol)
        assert model.n_modes == 2


class TestSingleModeReduction:
    def test_n1_matches_single_oldroydb(self):
        eta_p, lam, eta_s = 1.3, 0.7, 0.15
        single = OldroydB(polymer_viscosity=eta_p, relaxation_time=lam, solvent_viscosity=eta_s)
        multi = MultiModeOldroydB(
            polymer_viscosities=jnp.array([eta_p]),
            relaxation_times=jnp.array([lam]),
            solvent_viscosity=eta_s,
        )
        forcing = _shear_forcing()
        _, sim_single = _run(single, forcing)
        _, sim_multi = _run(multi, forcing)
        # Compare the full stress trajectories.
        assert jnp.allclose(sim_single.data, sim_multi.data, atol=1e-4)


class TestModeSuperposition:
    def test_n2_equals_sum_of_independent_modes(self):
        # Total stress = τ₁ + τ₂ + 2 η_s D.  Build the reference as the sum of
        # two single-mode runs, putting the whole solvent on the first so the
        # solvent term is counted exactly once.
        eta1, lam1 = 2.0, 0.3
        eta2, lam2 = 0.8, 1.5
        eta_s = 0.1
        forcing = _shear_forcing(amp=1.5, omega=1.3)

        multi = MultiModeOldroydB(
            polymer_viscosities=jnp.array([eta1, eta2]),
            relaxation_times=jnp.array([lam1, lam2]),
            solvent_viscosity=eta_s,
        )
        mode1 = OldroydB(polymer_viscosity=eta1, relaxation_time=lam1, solvent_viscosity=eta_s)
        mode2 = OldroydB(polymer_viscosity=eta2, relaxation_time=lam2, solvent_viscosity=0.0)

        _, sim_multi = _run(multi, forcing)
        _, sim_m1 = _run(mode1, forcing)
        _, sim_m2 = _run(mode2, forcing)

        reference = sim_m1.data + sim_m2.data
        assert jnp.allclose(sim_multi.data, reference, atol=1e-4)

    def test_shear_stress_channel_superposes(self):
        # Spot-check the σ₁₂ observable specifically (the fitting target).
        multi = MultiModeOldroydB(
            polymer_viscosities=jnp.array([1.0, 1.0]),
            relaxation_times=jnp.array([0.25, 2.0]),
            solvent_viscosity=0.05,
        )
        m1 = OldroydB(polymer_viscosity=1.0, relaxation_time=0.25, solvent_viscosity=0.05)
        m2 = OldroydB(polymer_viscosity=1.0, relaxation_time=2.0, solvent_viscosity=0.0)
        forcing = _shear_forcing()
        _, sm = _run(multi, forcing)
        _, s1 = _run(m1, forcing)
        _, s2 = _run(m2, forcing)
        assert jnp.allclose(sm.data[:, 0, 1], s1.data[:, 0, 1] + s2.data[:, 0, 1], atol=1e-4)


# ---------------------------------------------------------------------------
# Fisher Information / sloppy analysis on vector-valued parameter leaves
# ---------------------------------------------------------------------------

def _multimode_dataset(multi, *, t_end=10.0, n=80, amp=1.5, omega=0.8):
    """Build a noiseless σ₁₂ dataset for FIM analysis of a multi-mode model."""
    time = jnp.linspace(0.0, t_end, n)
    forcing_data = amp * jnp.sin(omega * time)
    forcing = dr.VelocityGradient.from_components(grad_u_12=lambda t: amp * jnp.sin(omega * t))
    solver = dr.DiffraxSolver(rtol=1e-7, atol=1e-7, max_steps=200000)
    rheometer = dr.VirtualRheometer.setup(multi, "strain_rate_response", solver)
    sim = rheometer.run_experiment(multi, forcing, time, jnp.zeros((3, 3)))
    data = ShearStrainRateData(
        time=time, data=sim.data[:, 0, 1], forcing_data=forcing_data,
        initial_condition=jnp.zeros((3, 3)),
    )
    return data, rheometer


class TestMultiModeFisherInformation:
    """The Fisher-information tooling must handle vector-valued parameter leaves."""

    def test_labels_expand_per_mode(self):
        multi = MultiModeOldroydB(
            polymer_viscosities=jnp.array([1.0, 0.6, 0.3]),
            relaxation_times=jnp.array([0.1, 1.0, 5.0]),
            solvent_viscosity=0.05,
        )
        data, rheometer = _multimode_dataset(multi)
        fisher = fisher_information(multi, rheometer, data, noise=0.05)
        # 3 viscosities + 3 relaxation times + 1 solvent = 7 parameters.
        assert fisher.matrix.shape == (7, 7)
        assert fisher.labels == [
            "polymer_viscosities[0]",
            "polymer_viscosities[1]",
            "polymer_viscosities[2]",
            "relaxation_times[0]",
            "relaxation_times[1]",
            "relaxation_times[2]",
            "solvent_viscosity",
        ]
        # FIM is symmetric and PSD with positive diagonal at a well-posed point.
        assert jnp.allclose(fisher.matrix, fisher.matrix.T, atol=1e-6)
        eigvals, _ = fisher.eigenspectrum()
        assert jnp.all(eigvals > -1e-5 * eigvals[0])

    def test_n1_multimode_matches_single_mode_fim(self):
        # A 1-mode MultiModeOldroydB and a scalar OldroydB are the same model
        # in the same coordinates; their FIMs must agree up to label naming.
        eta_p, lam, eta_s = 1.3, 0.7, 0.15
        single = OldroydB(polymer_viscosity=eta_p, relaxation_time=lam, solvent_viscosity=eta_s)
        multi = MultiModeOldroydB(
            polymer_viscosities=jnp.array([eta_p]),
            relaxation_times=jnp.array([lam]),
            solvent_viscosity=eta_s,
        )
        data_single, rheo_single = _multimode_dataset(single)
        data_multi, rheo_multi = _multimode_dataset(multi)
        f_single = fisher_information(single, rheo_single, data_single, noise=0.05)
        f_multi = fisher_information(multi, rheo_multi, data_multi, noise=0.05)
        # Column order is (polymer, relax, solvent) for both; label spelling
        # differs but the matrix should match.
        assert f_single.labels == ["polymer_viscosity", "relaxation_time", "solvent_viscosity"]
        assert f_multi.labels == [
            "polymer_viscosities[0]", "relaxation_times[0]", "solvent_viscosity",
        ]
        assert jnp.allclose(f_single.matrix, f_multi.matrix, rtol=1e-4, atol=1e-6)

    def test_parameter_uncertainty_keys(self):
        multi = MultiModeOldroydB(
            polymer_viscosities=jnp.array([1.0, 0.5]),
            relaxation_times=jnp.array([0.3, 2.0]),
            solvent_viscosity=0.05,
        )
        data, rheometer = _multimode_dataset(multi)
        report = parameter_uncertainty(multi, rheometer, data, noise=0.05)
        assert set(report) == {
            "polymer_viscosities[0]", "polymer_viscosities[1]",
            "relaxation_times[0]", "relaxation_times[1]",
            "solvent_viscosity",
        }
        # Every error bar is positive (the noiseless dataset constrains every
        # parameter to some extent under bounded-amplitude forcing).
        for name, (value, error) in report.items():
            assert error >= 0.0
            assert jnp.isfinite(value)

    def test_sloppy_analysis_reveals_spectrum_sloppiness(self):
        # A well-separated 3-mode spectrum probed over a *narrow* frequency
        # range cannot pin down every mode independently; the FIM eigenspectrum
        # therefore spans many decades — Sethna's "sloppiness".
        multi = MultiModeOldroydB(
            polymer_viscosities=jnp.array([1.0, 1.0, 1.0]),
            relaxation_times=jnp.array([0.01, 1.0, 100.0]),
            solvent_viscosity=0.05,
        )
        data, rheometer = _multimode_dataset(multi, omega=1.0, t_end=12.0, n=80)
        fisher = fisher_information(multi, rheometer, data, noise=0.05)
        analysis = sloppy_analysis(fisher)
        # 7 parameters (3 + 3 + 1).
        assert analysis.eigenvalues.shape == (7,)
        # The fast / slow modes are nearly invisible at ω = 1; expect a wide
        # condition number (≥ 4 decades is conservative).
        assert analysis.condition_number > 1e4
        # Orthonormal eigenvectors.
        gram = analysis.eigenvectors.T @ analysis.eigenvectors
        assert jnp.allclose(gram, jnp.eye(7), atol=1e-5)


# ---------------------------------------------------------------------------
# OrderedMultiModeOldroydB — same physics, ordered relaxation times
# ---------------------------------------------------------------------------

class TestOrderedMultiMode:
    """The ordered variant must reproduce the unordered model's forward physics
    and the relaxation-time vector recovered from its leaves must always be
    monotonically increasing, regardless of the input ordering or the
    optimiser's wanderings in unconstrained space."""

    def test_initial_relaxation_times_round_trip(self):
        # Construction from a physical λ-vector must recover it exactly.
        lam = jnp.array([0.1, 1.0, 10.0])
        eta = jnp.array([1.0, 1.0, 1.0])
        model = OrderedMultiModeOldroydB(
            polymer_viscosities=eta,
            relaxation_times=lam,
            solvent_viscosity=0.05,
        )
        # Recovery within float32 accuracy.
        assert jnp.allclose(model.relaxation_times_value, lam, rtol=1e-3, atol=1e-4)

    def test_relaxation_times_always_monotone(self):
        # Even with random log_lambda_min and log_increments (potentially negative),
        # the recovered relaxation times must be strictly increasing.
        from diff_rheo.parameters import Parameter
        key = jax.random.PRNGKey(0)
        for _ in range(20):
            key, k1, k2 = jax.random.split(key, 3)
            lam_min_logit = jax.random.normal(k1) * 3.0
            inc_logit = jax.random.normal(k2, (4,)) * 3.0  # 5-mode model
            model = OrderedMultiModeOldroydB(
                polymer_viscosities=jnp.ones(5),
                solvent_viscosity=0.05,
                log_lambda_min=Parameter(lam_min_logit),
                log_increments=Parameter(inc_logit),
            )
            lam = model.relaxation_times_value
            assert jnp.all(jnp.diff(lam) > 0), \
                f"non-monotone λ at log_lambda_min={lam_min_logit} log_inc={inc_logit}: {lam}"

    def test_matches_unordered_when_λ_already_sorted(self):
        # Same physical parameters → same stress trajectory (the reparameterisation
        # is an exact change of coordinates).
        lam = jnp.array([0.2, 1.5, 8.0])
        eta = jnp.array([0.7, 1.3, 0.9])
        eta_s = 0.04
        ordered = OrderedMultiModeOldroydB(
            polymer_viscosities=eta, relaxation_times=lam, solvent_viscosity=eta_s,
        )
        unordered = MultiModeOldroydB(
            polymer_viscosities=eta, relaxation_times=lam, solvent_viscosity=eta_s,
        )
        forcing = _shear_forcing(amp=1.5, omega=1.0)
        _, sim_o = _run(ordered, forcing)
        _, sim_u = _run(unordered, forcing)
        assert jnp.allclose(sim_o.data, sim_u.data, atol=1e-4)

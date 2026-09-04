"""
Tests for extensional kinematics (:func:`diff_rheo.extensional_forcing` and
:class:`diff_rheo.ExtensionalStrainRateData`).

Three claims, each checked against a known answer:

* **Kinematics** – the uniaxial / planar / biaxial velocity gradients are the
  expected traceless (incompressible) diagonal tensors.
* **Oldroyd-B steady state** – under a constant sub-critical extension rate the
  simulated tensile stress σ_E = σ₁₁ − σ₂₂ converges to the closed-form
  Oldroyd-B value, and the low-rate Trouton ratio η_E/η₀ → 3.
* **Degeneracy breaking** – a strain-hardening-bounded model (linear PTT) has a
  far smaller σ_E than Oldroyd-B near the ε̇·λ → 1/2 catastrophe, the divergence
  that no simple-shear experiment can see.
"""

import jax.numpy as jnp
import pytest

import diff_rheo as dr
from diff_rheo.models import OldroydB, LinearPTT


def _run(model, forcing, t_end=40.0, n=400):
    solver = dr.DiffraxSolver(rtol=1e-7, atol=1e-7, max_steps=200000)
    rheometer = dr.VirtualRheometer.setup(model, "strain_rate_response", solver)
    time = jnp.linspace(0.0, t_end, n)
    sim = rheometer.run_experiment(model, forcing, time, jnp.zeros((3, 3)))
    return time, sim


class TestExtensionalKinematics:
    def test_uniaxial_is_traceless_diagonal(self):
        L = dr.uniaxial_extension(2.0).gradient(0.0)
        assert jnp.allclose(jnp.diag(L), jnp.array([2.0, -1.0, -1.0]))
        assert jnp.allclose(L - jnp.diag(jnp.diag(L)), 0.0)  # off-diagonal zero
        assert abs(float(jnp.trace(L))) < 1e-6                # incompressible

    def test_planar_is_traceless_diagonal(self):
        L = dr.planar_extension(2.0).gradient(0.0)
        assert jnp.allclose(jnp.diag(L), jnp.array([2.0, 0.0, -2.0]))
        assert abs(float(jnp.trace(L))) < 1e-6

    def test_biaxial_is_traceless_diagonal(self):
        L = dr.biaxial_extension(1.5).gradient(0.0)
        assert jnp.allclose(jnp.diag(L), jnp.array([1.5, 1.5, -3.0]))
        assert abs(float(jnp.trace(L))) < 1e-6

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            dr.extensional_forcing(1.0, mode="shear")


class TestOldroydBExtensionalViscosity:
    """Steady uniaxial Oldroyd-B tensile stress has a closed form."""

    @staticmethod
    def _analytic_eta_E(eta_p, lam, eta_s, rate):
        # η_E = 3η_s + η_p[2/(1−2λε̇) + 1/(1+λε̇)]   (Oldroyd-B, uniaxial)
        return 3.0 * eta_s + eta_p * (2.0 / (1.0 - 2.0 * lam * rate) + 1.0 / (1.0 + lam * rate))

    def test_steady_state_matches_closed_form(self):
        eta_p, lam, eta_s = 2.0, 1.0, 0.1
        rate = 0.3 / lam   # λε̇ = 0.3, comfortably sub-critical
        model = OldroydB(polymer_viscosity=eta_p, relaxation_time=lam, solvent_viscosity=eta_s)
        time, sim = _run(model, dr.uniaxial_extension(rate), t_end=60.0, n=600)
        sigma_E = sim.data[:, 0, 0] - sim.data[:, 1, 1]
        eta_E_sim = float(sigma_E[-1]) / rate
        eta_E_exact = self._analytic_eta_E(eta_p, lam, eta_s, rate)
        assert jnp.allclose(eta_E_sim, eta_E_exact, rtol=1e-2)

    def test_low_rate_trouton_ratio_is_three(self):
        eta_p, lam, eta_s = 2.0, 1.0, 0.5
        eta_0 = eta_p + eta_s
        rate = 1e-3 / lam   # vanishing Wi → linear (Trouton) limit
        model = OldroydB(polymer_viscosity=eta_p, relaxation_time=lam, solvent_viscosity=eta_s)
        _, sim = _run(model, dr.uniaxial_extension(rate), t_end=80.0, n=400)
        sigma_E = sim.data[:, 0, 0] - sim.data[:, 1, 1]
        trouton = (float(sigma_E[-1]) / rate) / eta_0
        assert jnp.allclose(trouton, 3.0, rtol=2e-2)


class TestExtensionBreaksDegeneracy:
    def test_ptt_bounds_extensional_stress(self):
        # Near ε̇λ → 1/2 Oldroyd-B's tensile stress runs away; PTT's strain-
        # hardening function caps it.  Extension exposes a difference that
        # simple shear (σ₁₂) cannot.
        eta_p, lam, eta_s = 2.0, 1.0, 0.1
        rate = 0.45 / lam   # close to the Oldroyd-B catastrophe
        ob = OldroydB(polymer_viscosity=eta_p, relaxation_time=lam, solvent_viscosity=eta_s)
        ptt = LinearPTT(polymer_viscosity=eta_p, relaxation_time=lam, solvent_viscosity=eta_s,
                        epsilon=0.2, zeta=0.0)
        _, sim_ob = _run(ob, dr.uniaxial_extension(rate), t_end=40.0)
        _, sim_ptt = _run(ptt, dr.uniaxial_extension(rate), t_end=40.0)
        sE_ob = float((sim_ob.data[:, 0, 0] - sim_ob.data[:, 1, 1])[-1])
        sE_ptt = float((sim_ptt.data[:, 0, 0] - sim_ptt.data[:, 1, 1])[-1])
        assert sE_ptt < 0.5 * sE_ob


class TestExtensionalData:
    def test_data_roundtrips_through_pipeline(self):
        eta_p, lam, eta_s = 1.5, 0.8, 0.1
        rate = 0.25 / lam
        model = OldroydB(polymer_viscosity=eta_p, relaxation_time=lam, solvent_viscosity=eta_s)
        time, sim = _run(model, dr.uniaxial_extension(rate), t_end=20.0, n=120)
        sigma_E = sim.data[:, 0, 0] - sim.data[:, 1, 1]

        data = dr.ExtensionalStrainRateData(
            time=time, data=sigma_E,
            forcing_data=rate * jnp.ones_like(time),
            initial_condition=jnp.zeros((3, 3)),
            mode="uniaxial",
        )
        # The reconstructed forcing reproduces the experiment and the extractor
        # recovers σ_E to high accuracy.
        rheometer = dr.VirtualRheometer.setup(
            model, "strain_rate_response", dr.DiffraxSolver(rtol=1e-7, atol=1e-7)
        )
        sim2 = rheometer.run_experiment(model, data.get_forcing_function(), time, data.initial_condition)
        extracted = data.extract_from_simulation(sim2)
        assert extracted.shape == sigma_E.shape
        assert jnp.allclose(extracted, sigma_E, atol=1e-3)

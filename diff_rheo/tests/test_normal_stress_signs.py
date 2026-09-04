"""Analytic normal-stress regression tests (startup of steady simple shear).

These guard against the upper-convected-derivative sign/convention bug in which
the strain-rate RHS used ``sigma @ L + L^T @ sigma`` (lower-convected) instead of
``L @ sigma + sigma @ L^T``, which flips the sign of N1 (and N2) while leaving
sigma12 correct.  In steady simple shear at rate gdot with eta_p = lambda = 1:

    Oldroyd-B / White-Metzner(n=m=1):  sigma12 = gdot,  N1 = 2*gdot^2 > 0,  N2 = 0
    Multi-mode Oldroyd-B:              N1 = 2*sum_k(eta_k*lam_k)*gdot^2 > 0,  N2 = 0
    Giesekus / PTT:                    N1 > 0,  N2 < 0
"""
import jax
import jax.numpy as jnp
import pytest

import diff_rheo as dr
from diff_rheo.models import OldroydB, Giesekus, LinearPTT, MultiModeOldroydB
from diff_rheo.models._viscoelastic import WhiteMetzner
from diff_rheo.parameters import LogParameter, StaticParameter, TanhParameter

jax.config.update("jax_enable_x64", True)


def _steady_shear(model, gdot=1.0, T=60.0, n=1200):
    t = jnp.linspace(0.0, T, n)
    exp = dr.ShearStrainRateData(time=t, data=jnp.zeros_like(t),
                                 forcing_data=gdot * jnp.ones_like(t),
                                 initial_condition=jnp.zeros((3, 3)))
    solver = dr.DiffraxSolver(rtol=1e-8, atol=1e-10, max_steps=1_000_000, throw=False)
    rheo = dr.VirtualRheometer.setup(model, "strain_rate_response", solver)
    s = rheo.run_experiment(model, exp.get_forcing_function(), exp.time, exp.initial_condition).data[-1]
    return s[0, 1], s[0, 0] - s[1, 1], s[1, 1] - s[2, 2]   # sigma12, N1, N2


def test_oldroyd_b_steady_shear_normal_stresses():
    """N1 = 2*eta_p*lambda*gdot^2 > 0 and N2 = 0 exactly (textbook UCM)."""
    m = OldroydB(polymer_viscosity=LogParameter(1.0), relaxation_time=LogParameter(1.0),
                 solvent_viscosity=StaticParameter(1e-6))
    s12, n1, n2 = _steady_shear(m, gdot=1.0)
    assert jnp.allclose(s12, 1.0, atol=1e-3)
    assert jnp.allclose(n1, 2.0, atol=1e-3)          # POSITIVE, magnitude exact
    assert jnp.allclose(n2, 0.0, atol=1e-3)


def test_oldroyd_b_n1_scaling():
    m = OldroydB(polymer_viscosity=LogParameter(1.0), relaxation_time=LogParameter(1.0),
                 solvent_viscosity=StaticParameter(1e-6))
    for g in (0.5, 2.0):
        _, n1, _ = _steady_shear(m, gdot=g)
        assert jnp.allclose(n1, 2.0 * g * g, atol=1e-3)


def test_white_metzner_reduces_to_oldroyd_b():
    m = WhiteMetzner(polymer_viscosity=LogParameter(1.0), relaxation_time=LogParameter(1.0),
                     solvent_viscosity=StaticParameter(1e-6), K=LogParameter(1.0), L=LogParameter(1.0),
                     n=LogParameter(1.0), m=LogParameter(1.0), a=LogParameter(2.0), b=LogParameter(2.0))
    s12, n1, n2 = _steady_shear(m, gdot=1.0)
    assert jnp.allclose(s12, 1.0, atol=1e-3)
    assert jnp.allclose(n1, 2.0, atol=1e-3)
    assert jnp.allclose(n2, 0.0, atol=1e-3)


def test_multimode_oldroyd_b_n1_is_sum_over_modes():
    etas, lams = jnp.array([0.6, 0.4]), jnp.array([1.0, 3.0])
    m = MultiModeOldroydB(polymer_viscosities=etas, relaxation_times=lams,
                          solvent_viscosity=StaticParameter(1e-6))
    _, n1, n2 = _steady_shear(m, gdot=1.0)
    assert jnp.allclose(n1, 2.0 * jnp.sum(etas * lams), atol=1e-3)   # = 3.6, POSITIVE
    assert jnp.allclose(n2, 0.0, atol=1e-3)


@pytest.mark.parametrize("model", [
    Giesekus(polymer_viscosity=LogParameter(1.0), relaxation_time=LogParameter(1.0),
             solvent_viscosity=StaticParameter(1e-6), alpha=TanhParameter(0.3, max_value=1.0)),
    LinearPTT(polymer_viscosity=LogParameter(1.0), relaxation_time=LogParameter(1.0),
              solvent_viscosity=StaticParameter(1e-6), epsilon=LogParameter(0.2),
              zeta=TanhParameter(0.1, max_value=2.0)),
])
def test_nonlinear_models_have_positive_n1_negative_n2(model):
    """Shear-thinning viscoelastic models: N1 > 0 and N2 < 0 in steady shear."""
    _, n1, n2 = _steady_shear(model, gdot=1.0)
    assert n1 > 0.0
    assert n2 < 0.0

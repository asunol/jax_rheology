"""Tests for the NumPyro / NUTS Bayesian-inference module (``diff_rheo._bayesian``).

The end-to-end sampling tests are guarded with :func:`pytest.importorskip` so the
suite still passes in environments where ``numpyro`` / ``arviz`` are not
installed.  They deliberately use very short chains and time grids -- this is a
smoke test of the wiring (shapes, finiteness, the BayesianFit API), not a
posterior-accuracy benchmark.
"""

import jax
import jax.numpy as jnp
import pytest

import diff_rheo as dr
from diff_rheo import _bayesian
from diff_rheo.models import Giesekus, OldroydB
from diff_rheo.parameters import GaussianParameter, Parameter, StaticParameter


# ---------------------------------------------------------------------------
# free_parameter_names -- pure, no numpyro required
# ---------------------------------------------------------------------------

def test_free_parameter_names_lists_inferable_parameters():
    model = OldroydB(
        polymer_viscosity=Parameter(2.0),
        relaxation_time=Parameter(1.0),
        solvent_viscosity=Parameter(0.1),
    )
    assert _bayesian.free_parameter_names(model) == [
        "polymer_viscosity", "relaxation_time", "solvent_viscosity"
    ]


def test_free_parameter_names_excludes_static_parameters():
    # A StaticParameter is frozen and must not appear among the inferable names.
    model = OldroydB(
        polymer_viscosity=Parameter(2.0),
        relaxation_time=Parameter(1.0),
        solvent_viscosity=StaticParameter(0.1),
    )
    names = _bayesian.free_parameter_names(model)
    assert "solvent_viscosity" not in names
    assert names == ["polymer_viscosity", "relaxation_time"]


def test_free_parameter_names_rejects_random_parameters():
    # NUTS replaces variational inference; a random-parameter template is wrong.
    model = OldroydB(
        polymer_viscosity=GaussianParameter(mean=2.0, std=0.2),
        relaxation_time=Parameter(1.0),
        solvent_viscosity=Parameter(0.1),
    )
    with pytest.raises(ValueError, match="random/variational"):
        _bayesian.free_parameter_names(model)


# ---------------------------------------------------------------------------
# Priors -- require numpyro
# ---------------------------------------------------------------------------

def test_default_priors_cover_every_free_parameter():
    pytest.importorskip("numpyro")
    model = Giesekus(
        polymer_viscosity=Parameter(2.0),
        relaxation_time=Parameter(1.0),
        solvent_viscosity=Parameter(0.1),
        alpha=Parameter(0.3),
    )
    priors = _bayesian.default_priors(model)
    assert set(priors) == {
        "polymer_viscosity", "relaxation_time", "solvent_viscosity", "alpha"
    }


def test_default_noise_prior_is_positive_support():
    pytest.importorskip("numpyro")
    import numpyro.distributions as dist

    prior = _bayesian.default_noise_prior()
    assert isinstance(prior, dist.Distribution)
    # A half-normal is supported on the non-negative reals.
    assert float(prior.sample(jax.random.PRNGKey(0))) >= 0.0


# ---------------------------------------------------------------------------
# End-to-end NUTS smoke test -- requires numpyro + arviz
# ---------------------------------------------------------------------------

def _tiny_oldroydb_dataset():
    """A small, cheap synthetic Oldroyd-B shear dataset for the smoke tests."""
    true = OldroydB(
        polymer_viscosity=2.0, relaxation_time=1.0, solvent_viscosity=0.2
    )
    solver = dr.DiffraxSolver(max_steps=10000, throw=False)
    rheo = dr.VirtualRheometer.setup(true, "strain_rate_response", solver)
    time = jnp.linspace(0.0, 6.0, 20)
    ic = jnp.zeros((3, 3))
    forcing = dr.VelocityGradient.from_components(
        grad_u_12=lambda t: 2.0 * jnp.sin(t)
    )
    sim = rheo.run_experiment(true, forcing, time, ic)
    clean = sim.data[:, 0, 1]
    noisy = clean + 0.02 * jax.random.normal(jax.random.PRNGKey(1), clean.shape)
    data = dr.ShearStrainRateData(
        time=time, data=noisy,
        forcing_data=2.0 * jnp.sin(time), initial_condition=ic,
    )
    return dr.BatchedData([data]), rheo, solver


def test_run_nuts_end_to_end():
    pytest.importorskip("numpyro")
    pytest.importorskip("arviz")

    data, rheo, solver = _tiny_oldroydb_dataset()
    template = OldroydB(
        polymer_viscosity=1.0, relaxation_time=1.0, solvent_viscosity=0.5
    )
    fit = dr.run_nuts(
        template, rheo, data,
        num_warmup=25, num_samples=25, num_chains=1,
        key=jax.random.PRNGKey(2), progress_bar=False,
    )

    # --- BayesianFit basics ------------------------------------------------
    assert isinstance(fit, dr.BayesianFit)
    assert fit.param_names == [
        "polymer_viscosity", "relaxation_time", "solvent_viscosity"
    ]
    draws = fit.posterior("polymer_viscosity")
    assert draws.shape == (25,)
    assert jnp.all(draws > 0.0)                       # LogNormal prior support
    assert jnp.all(fit.posterior("sigma") > 0.0)      # noise std is positive

    means = fit.posterior_mean()
    assert set(means) >= set(fit.param_names) | {"sigma"}

    # posterior_model rebuilds a usable deterministic constitutive model.
    post_model = fit.posterior_model()
    assert isinstance(post_model, OldroydB)

    # posterior_mass_below is a probability.
    mass = fit.posterior_mass_below("polymer_viscosity", 10.0)
    assert 0.0 <= mass <= 1.0

    # --- WAIC / PSIS-LOO ---------------------------------------------------
    waic = fit.waic()
    loo = fit.loo()
    assert jnp.isfinite(_bayesian._elpd(waic, "elpd_waic"))
    assert jnp.isfinite(_bayesian._elpd(loo, "elpd_loo"))
    assert jnp.isfinite(fit.max_pareto_k())
    assert fit.divergence_count() >= 0

    # az.summary should yield one row per sampled site.
    summary = fit.summary()
    assert len(summary) >= len(fit.param_names)


def test_compare_models_ranks_two_fits():
    pytest.importorskip("numpyro")
    pytest.importorskip("arviz")

    data, rheo_ob, solver = _tiny_oldroydb_dataset()
    ob_template = OldroydB(
        polymer_viscosity=1.0, relaxation_time=1.0, solvent_viscosity=0.5
    )
    ob_fit = dr.run_nuts(
        ob_template, rheo_ob, data,
        num_warmup=20, num_samples=20, num_chains=1,
        key=jax.random.PRNGKey(3), progress_bar=False,
    )

    gk_template = Giesekus(
        polymer_viscosity=1.0, relaxation_time=1.0,
        solvent_viscosity=0.5, alpha=0.1,
    )
    rheo_gk = dr.VirtualRheometer.setup(gk_template, "strain_rate_response", solver)
    gk_fit = dr.run_nuts(
        gk_template, rheo_gk, data,
        num_warmup=20, num_samples=20, num_chains=1,
        key=jax.random.PRNGKey(4), progress_bar=False,
    )

    table = dr.compare_models({"OldroydB": ob_fit, "Giesekus": gk_fit}, ic="loo")
    assert len(table) == 2
    assert {"OldroydB", "Giesekus"} == set(table.index)

"""
Loss functions, fitting routines, and model evaluation utilities.

This module is the primary user-facing API for fitting constitutive models to
experimental data.  It provides:

**Deterministic (MAP) fitting**

* :func:`data_fitting_loss` – mean-squared-error loss over a batch of experiments.
* :func:`fit_model_to_experimental_data` – convenience wrapper around
  :class:`~diff_rheo._fitting.ModelFitter` for MAP fitting.

**Variational inference (ELBO)**

* :func:`kl_divergence` – KL divergence between variational posteriors and a
  standard Normal prior, accumulated over all
  :class:`~diff_rheo.parameters.GaussianParameter` /
  :class:`~diff_rheo.parameters.LogGaussianParameter` leaves.
* :func:`trajectory_log_likelihood` – log-likelihood based on the *differential*
  (incremental) trajectory, helpful for non-stationary signals.
* :func:`trajectory_log_likelihood_direct` – log-likelihood based on the direct
  trajectory values.
* :func:`variational_inference_loss` / :func:`variational_inference_loss_direct` –
  ELBO = KL - log-likelihood, minimised during variational training.
* :func:`fit_variational_inference` – convenience wrapper for ELBO fitting.

**Model evaluation**

* :func:`model_bic` – Bayesian Information Criterion via the ensemble
  log-likelihood.
* :func:`calculate_bic_from_l2` – BIC derived from the L2/MSE loss, assuming
  i.i.d. Gaussian noise.
* :func:`display_results` – colour-coded console table comparing fitted
  parameters to ground truth.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax as opt
from jax.scipy.stats import norm
from typing import Union, Iterator
from colorama import init, Fore, Style

from ._data_types import ExperimentalData, FittingConfig, BatchedData
from ._rheometer import VirtualRheometer
from .parameters import GaussianParameter, LogGaussianParameter
from ._fitting import ModelFitter
from .models import AbstractConstitutiveModel


# ---------------------------------------------------------------------------
# Deterministic (MAP) loss
# ---------------------------------------------------------------------------

def data_fitting_loss(
    model: AbstractConstitutiveModel,
    rheometer: VirtualRheometer,
    reference: Iterator[ExperimentalData],
    *args,
    **kwargs,
) -> tuple[jax.Array, None]:
    """Compute the mean-squared-error loss over a batch of experiments.

    Iterates over all experiments yielded by ``reference`` and accumulates the
    per-experiment MSE loss.  Each experiment is evaluated independently via the
    JIT-compiled :func:`_data_fitting_loss_single` helper.

    Parameters
    ----------
    model : AbstractConstitutiveModel
        The constitutive model with current parameter values.
    rheometer : VirtualRheometer
        The virtual rheometer used to run simulations.
    reference : Iterator[ExperimentalData]
        An iterator of experimental observations (e.g. from
        :meth:`~diff_rheo._data_types.BatchedData.fitting_schedule`).

    Returns
    -------
    tuple[jax.Array, None]
        ``(loss, None)`` where ``loss`` is the summed MSE over all experiments.
        The ``None`` auxiliary output is required by the
        ``eqx.filter_value_and_grad(has_aux=True)`` interface used in
        :class:`~diff_rheo._fitting.ModelFitter`.

    Notes
    -----
    For a single experiment, the loss is
    ``mean((predicted_signal - observed_signal)²)``.
    """
    loss = 0
    n = 0
    for ref in reference:
        loss += _data_fitting_loss_single(model, rheometer, ref, *args, **kwargs)
        n += 1
    return loss, None


@eqx.filter_jit
def _data_fitting_loss_single(
    model: AbstractConstitutiveModel,
    rheometer: VirtualRheometer,
    reference: ExperimentalData,
    *args,
    **kwargs,
) -> jax.Array:
    """JIT-compiled MSE loss for a single experiment.

    Reconstructs the forcing function from the stored discrete data, runs the
    simulation, extracts the relevant observable, and returns the MSE.

    Parameters
    ----------
    model : AbstractConstitutiveModel
    rheometer : VirtualRheometer
    reference : ExperimentalData
        A single experimental observation.

    Returns
    -------
    jax.Array
        Scalar MSE: ``mean((predicted - observed)²)``.
    """
    forcing = reference.get_forcing_function()
    time = reference.time
    data = reference.data
    simulated_data = rheometer.run_experiment(model, forcing, time, reference.initial_condition)
    predicted_data = reference.extract_from_simulation(simulated_data)
    return jnp.mean((predicted_data - data)**2)


# ---------------------------------------------------------------------------
# Variational inference: KL divergence and log-likelihoods
# ---------------------------------------------------------------------------

def kl_divergence(model: AbstractConstitutiveModel) -> jax.Array:
    """Compute the total KL divergence between variational posteriors and a N(0,1) prior.

    Traverses all leaves of the Equinox pytree ``model`` and accumulates the
    closed-form KL divergence for every
    :class:`~diff_rheo.parameters.GaussianParameter` and
    :class:`~diff_rheo.parameters.LogGaussianParameter` found.

    For a single Gaussian variational distribution q = N(μ, σ²) against a
    standard Normal prior p = N(0, 1):

        D_KL(q ‖ p) = -log(σ) + (σ² + μ²)/2 - 1/2

    Parameters
    ----------
    model : AbstractConstitutiveModel
        A model containing one or more stochastic parameter leaves.

    Returns
    -------
    jax.Array
        Scalar total KL divergence.  Returns ``0.0`` if the model has no
        stochastic parameters.

    Notes
    -----
    A small epsilon (``1e-6``) is added to ``σ`` to prevent ``log(0)`` during
    early training when the standard deviation may collapse to zero.
    """
    gaussians = [
        x for x in jax.tree_util.tree_leaves(model)
        if isinstance(x, (GaussianParameter, LogGaussianParameter))
    ]

    total_kl = 0.0
    for param in gaussians:
        mean = param.mean
        std = param.std
        # Guard against degenerate std=0 at initialisation
        std = jnp.where(std == 0, 1e-6, std)
        kl = jnp.sum(
            -jnp.log(std) + 0.5 * (std**2 + mean**2) - 0.5
        )
        total_kl += kl

    return total_kl


@eqx.filter_jit
def trajectory_log_likelihood(
    model: AbstractConstitutiveModel,
    rheometer: VirtualRheometer,
    reference: ExperimentalData,
    key: jax.random.PRNGKey,
    ensemble_size: int,
) -> jax.Array:
    """Estimate the log-likelihood of the data using a differential (incremental) trajectory.

    Runs ``ensemble_size`` stochastic forward simulations by sampling model
    parameters, then fits a Gaussian to the *increments* (first differences) of
    the predicted ensemble.  The log-likelihood of the observed increments under
    this Gaussian is returned.

    Using differences rather than absolute values makes the likelihood less
    sensitive to initial condition offset errors.

    The total predictive variance combines:

    * **Parametric uncertainty**: variance of the ensemble predictions.
    * **Observation noise**: ``model.observation_noise.get_value()²``.

    Parameters
    ----------
    model : AbstractConstitutiveModel
        A model with stochastic parameter leaves.
    rheometer : VirtualRheometer
    reference : ExperimentalData
        A single experimental observation.
    key : jax.random.PRNGKey
        PRNG key for sampling ensemble members.
    ensemble_size : int
        Number of forward simulations to run.

    Returns
    -------
    jax.Array
        Scalar log-likelihood (larger is better fit).
    """
    forcing = reference.get_forcing_function()
    simulated_solutions = rheometer.run_ensemble(model, forcing, reference.time, reference.initial_condition, key, ensemble_size)
    predicted_ensemble = eqx.filter_vmap(reference.extract_from_simulation)(simulated_solutions)
    ensemble_diff = jnp.diff(predicted_ensemble, axis=1)
    pred_mean_diff = jnp.mean(ensemble_diff, axis=0)
    pred_std_diff = jnp.std(ensemble_diff, axis=0)
    pred_std_diff = jnp.where(pred_std_diff == 0, 1e-6, pred_std_diff)

    observation_noise = model.observation_noise.get_value()
    pred_std_diff = jnp.sqrt(pred_std_diff**2 + jnp.ones_like(pred_std_diff) * observation_noise**2)
    ref_data = jnp.diff(reference.data, axis=0)

    log_likelihood = jnp.sum(
        norm.logpdf(ref_data, loc=pred_mean_diff, scale=pred_std_diff)
    )

    return log_likelihood


@eqx.filter_jit
def trajectory_log_likelihood_direct(
    model: AbstractConstitutiveModel,
    rheometer: VirtualRheometer,
    reference: ExperimentalData,
    key: jax.random.PRNGKey,
    ensemble_size: int,
) -> jax.Array:
    """Estimate the log-likelihood of the data using direct trajectory values.

    Runs ``ensemble_size`` stochastic forward simulations, then fits a Gaussian
    to the predicted ensemble at each time point.  The log-likelihood of the
    observed data under this Gaussian is returned.

    Unlike :func:`trajectory_log_likelihood`, this function operates on the
    absolute trajectory values rather than their increments, which may be more
    appropriate when the initial condition is well-constrained.

    The total predictive standard deviation combines parametric uncertainty
    from the ensemble spread and the ``model.observation_noise`` parameter:

        σ_total² = σ_obs² + σ_params²

    Parameters
    ----------
    model : AbstractConstitutiveModel
        A model with stochastic parameter leaves.
    rheometer : VirtualRheometer
    reference : ExperimentalData
        A single experimental observation.
    key : jax.random.PRNGKey
    ensemble_size : int
        Number of forward simulations to run.

    Returns
    -------
    jax.Array
        Scalar log-likelihood (larger is better fit).
    """
    forcing = reference.get_forcing_function()

    simulated_solutions = rheometer.run_ensemble(
        model,
        forcing,
        reference.time,
        reference.initial_condition,
        key,
        ensemble_size,
    )

    predicted_ensemble = eqx.filter_vmap(reference.extract_from_simulation)(simulated_solutions)

    pred_mean = jnp.mean(predicted_ensemble, axis=0)
    pred_var_from_params = jnp.var(predicted_ensemble, axis=0)
    pred_std_from_params = jnp.sqrt(pred_var_from_params + 1e-8)

    observation_noise = model.observation_noise.get_value()

    total_std = jnp.sqrt(observation_noise**2 + pred_std_from_params**2)
    total_std = jnp.where(total_std == 0, 1e-6, total_std)

    log_likelihood = jnp.sum(
        norm.logpdf(reference.data, loc=pred_mean, scale=total_std)
    )

    return log_likelihood


# ---------------------------------------------------------------------------
# ELBO losses
# ---------------------------------------------------------------------------

def variational_inference_loss(
    model: AbstractConstitutiveModel,
    rheometer: VirtualRheometer,
    reference: Iterator[ExperimentalData],
    key: jax.random.PRNGKey,
    ensemble_size: int,
) -> tuple[jax.Array, None]:
    """Compute the ELBO loss using the differential trajectory log-likelihood.

    The Evidence Lower BOund (ELBO) is:

        ELBO = E_q[log p(data | θ)] - D_KL(q(θ) ‖ p(θ))

    We minimise the negative ELBO (i.e. KL - log-likelihood).  The
    log-likelihood term uses :func:`trajectory_log_likelihood` (differential).

    Parameters
    ----------
    model : AbstractConstitutiveModel
    rheometer : VirtualRheometer
    reference : Iterator[ExperimentalData]
        Iterator over training experiments.
    key : jax.random.PRNGKey
    ensemble_size : int
        Number of samples for Monte Carlo log-likelihood estimation.

    Returns
    -------
    tuple[jax.Array, None]
        ``(loss, None)`` where ``loss = KL - mean(log-likelihood)``.
    """
    kl_div = kl_divergence(model)
    log_likelihood = 0
    n = 0
    for ref in reference:
        log_likelihood += trajectory_log_likelihood(model, rheometer, ref, key, ensemble_size)
        n += 1
    log_likelihood /= n

    return (kl_div - log_likelihood, None)


def variational_inference_loss_direct(
    model: AbstractConstitutiveModel,
    rheometer: VirtualRheometer,
    reference: Iterator[ExperimentalData],
    key: jax.random.PRNGKey,
    ensemble_size: int,
) -> tuple[jax.Array, None]:
    """Compute the ELBO loss using the direct trajectory log-likelihood.

    Similar to :func:`variational_inference_loss` but uses
    :func:`trajectory_log_likelihood_direct` (direct values rather than
    increments) and additionally includes a log-prior over the observation noise
    parameter:

        loss = KL - mean(log-likelihood) - log p(log_noise)

    where ``log p(log_noise) = N(log_noise; -4.0, 1.0)`` is a weakly
    informative prior that keeps the noise level from growing too large.

    Parameters
    ----------
    model : AbstractConstitutiveModel
    rheometer : VirtualRheometer
    reference : Iterator[ExperimentalData]
        Iterator over training experiments.
    key : jax.random.PRNGKey
    ensemble_size : int

    Returns
    -------
    tuple[jax.Array, None]
        ``(loss, None)``.
    """
    kl_div = kl_divergence(model)

    log_likelihood = 0
    n = 0
    for ref in reference:
        log_likelihood += trajectory_log_likelihood_direct(model, rheometer, ref, key, ensemble_size)
        n += 1
    log_likelihood /= n

    # Weakly informative log-Normal prior on the observation noise level
    noise_log_sigma = model.observation_noise.get_value()
    log_prior_noise = norm.logpdf(noise_log_sigma, loc=-4.0, scale=1.0)

    loss = kl_div - log_likelihood - log_prior_noise

    return (loss, None)


# ---------------------------------------------------------------------------
# Convenience fitting functions
# ---------------------------------------------------------------------------

def fit_model_to_experimental_data(
    model,
    rheometer: VirtualRheometer,
    reference: BatchedData,
    config: FittingConfig,
) -> AbstractConstitutiveModel:
    """Fit a model to experimental data using MAP (mean-squared-error) optimisation.

    Constructs a :class:`~diff_rheo._fitting.ModelFitter` with
    :func:`data_fitting_loss` and runs the training loop.

    Parameters
    ----------
    model : AbstractConstitutiveModel
        The model to fit.  Parameters should be initialised with
        :class:`~diff_rheo.parameters.LogParameter` or
        :class:`~diff_rheo.parameters.Parameter` for MAP fitting.
    rheometer : VirtualRheometer
        The virtual rheometer matching the model type and experiment type.
    reference : BatchedData
        The experimental observations.
    config : FittingConfig
        Training configuration.

    Returns
    -------
    AbstractConstitutiveModel
        The fitted model with optimised parameter values.
    """
    fitter = ModelFitter(rheometer, opt.adam(config.learning_rate), config, data_fitting_loss)
    fit, loss, aux = fitter.train(model, reference)
    return fit


def fit_variational_inference(
    model,
    rheometer: VirtualRheometer,
    reference: BatchedData,
    config: FittingConfig,
    direct: bool = False,
) -> AbstractConstitutiveModel:
    """Fit a model using variational inference (ELBO optimisation).

    Constructs a :class:`~diff_rheo._fitting.ModelFitter` with the appropriate
    ELBO loss and runs the training loop.  The model should contain
    :class:`~diff_rheo.parameters.GaussianParameter` or
    :class:`~diff_rheo.parameters.LogGaussianParameter` attributes whose
    mean and variance are optimised to approximate the posterior.

    Parameters
    ----------
    model : AbstractConstitutiveModel
        A model with stochastic parameter leaves.
    rheometer : VirtualRheometer
    reference : BatchedData
        The experimental observations.
    config : FittingConfig
        Training configuration.  ``config.ensemble_size`` controls how many
        samples are used per ELBO estimate (larger = lower variance but slower).
    direct : bool
        If ``False`` (default), use :func:`variational_inference_loss`
        (differential trajectory).  If ``True``, use
        :func:`variational_inference_loss_direct` (direct values).

    Returns
    -------
    AbstractConstitutiveModel
        The fitted model with optimised variational distribution parameters.
    """
    if direct:
        fitter = ModelFitter(rheometer, opt.adam(config.learning_rate), config, variational_inference_loss_direct)
    else:
        fitter = ModelFitter(rheometer, opt.adam(config.learning_rate), config, variational_inference_loss)
    fit, loss, aux = fitter.train(model, reference)
    return fit


# ---------------------------------------------------------------------------
# Model selection and evaluation
# ---------------------------------------------------------------------------

def model_bic(
    model: AbstractConstitutiveModel,
    rheometer: VirtualRheometer,
    reference: BatchedData,
    config: FittingConfig,
    direct: bool = False,
) -> jax.Array:
    """Compute the Bayesian Information Criterion (BIC) using ensemble log-likelihood.

    BIC = k · log(n) - 2 · log L̂

    where ``k`` is the number of trainable parameters, ``n`` is the total
    number of data points, and ``log L̂`` is the maximised log-likelihood
    estimated via the ensemble.

    Parameters
    ----------
    model : AbstractConstitutiveModel
        The fitted model (after optimisation).
    rheometer : VirtualRheometer
    reference : BatchedData
        The experimental observations.
    config : FittingConfig
        Uses ``config.key`` and ``config.ensemble_size`` for the log-likelihood
        estimate.
    direct : bool
        If ``True``, use the direct trajectory log-likelihood; otherwise use
        the differential trajectory log-likelihood.

    Returns
    -------
    jax.Array
        Scalar BIC value.  Lower BIC indicates a better model (better fit
        relative to model complexity).

    Notes
    -----
    Useful for comparing models of different complexity (e.g. Oldroyd-B vs
    Giesekus) fitted to the same dataset.  See also
    :func:`calculate_bic_from_l2` for a simpler BIC estimate from L2 loss.
    """
    key = config.key
    ensemble_size = config.ensemble_size
    log_likelihood = 0
    n = 0
    for ref in reference.data:
        if direct:
            log_likelihood += trajectory_log_likelihood_direct(model, rheometer, ref, key, ensemble_size)
        else:
            log_likelihood += trajectory_log_likelihood(model, rheometer, ref, key, ensemble_size)
        n += len(ref.data)
    p = model.trainable_count
    log_likelihood /= len(reference)
    return p * jnp.log(n) - 2 * log_likelihood


@eqx.filter_jit
def _predict_single(model, rheometer, reference):
    """JIT-compiled prediction for a single experiment (used by :func:`calculate_bic_from_l2`)."""
    forcing = reference.get_forcing_function()
    time = reference.time
    sim = rheometer.run_experiment(model, forcing, time, reference.initial_condition)
    return reference.extract_from_simulation(sim)


def calculate_bic_from_l2(
    fitted_model: "AbstractConstitutiveModel",
    rheometer: "VirtualRheometer",
    reference_data: "BatchedData",
    include_noise_param: bool = True,
) -> jax.Array:
    """Compute BIC from the L2 loss, assuming i.i.d. Gaussian noise.

    A simpler alternative to :func:`model_bic` that does not require an
    ensemble.  Estimates the maximised log-likelihood by assuming that residuals
    are i.i.d. Gaussian with MLE variance ``σ̂² = SSE / N``:

        log L̂ = -N/2 · (log(2π σ̂²) + 1)

    Then computes:

        BIC = k · log(N) - 2 · log L̂

    where ``k`` is the number of trainable floating-point array elements (not
    just the number of parameter objects).

    Parameters
    ----------
    fitted_model : AbstractConstitutiveModel
        The fitted model.
    rheometer : VirtualRheometer
    reference_data : BatchedData
        The experimental observations.
    include_noise_param : bool
        Reserved for future use (currently ignored).

    Returns
    -------
    jax.Array
        Scalar BIC value.
    """
    total_sse = jnp.array(0.0)
    total_n = 0

    for ref in reference_data.data:
        pred = _predict_single(fitted_model, rheometer, ref)
        residuals = pred - ref.data
        total_sse = total_sse + jnp.sum(residuals**2)
        total_n += residuals.size

    # MLE variance under i.i.d. Gaussian noise
    mse = total_sse / total_n
    mse = jnp.maximum(mse, jnp.finfo(mse.dtype).tiny)

    # Maximised Gaussian log-likelihood at σ̂² = SSE/N
    log_likelihood = -0.5 * total_n * (jnp.log(2 * jnp.pi * mse) + 1.0)

    # Count all trainable floating-point parameters (array elements, not objects)
    params, _ = eqx.partition(fitted_model, eqx.is_inexact_array)
    k = sum(x.size for x in jax.tree_util.tree_leaves(params))

    bic = k * jnp.log(total_n) - 2.0 * log_likelihood
    return bic


def display_results(
    fit_model: AbstractConstitutiveModel,
    ground_truth: Union[AbstractConstitutiveModel, dict],
):
    """Print a colour-coded comparison table of fitted vs ground-truth parameters.

    Displays each parameter's ground-truth value, fitted mean, and (if
    available) fitted standard deviation.  Rows are coloured:

    * **Green** – error < 5 %
    * **Yellow** – error 5–15 %
    * **Red** – error > 15 %

    Parameters
    ----------
    fit_model : AbstractConstitutiveModel
        The fitted model (after optimisation).
    ground_truth : AbstractConstitutiveModel | dict
        Either another model instance or a ``dict`` mapping parameter names
        to their true scalar values.
    """
    init()  # Initialise colorama for cross-platform ANSI colour support

    print("\nParameter Comparison: Ground Truth vs Fitted Values")
    print("-" * 70)
    print(f"{Fore.CYAN}{'Parameter':<20} {'Ground Truth':<15} {'Fitted Mean':<15} {'Fitted Std':<15}{Style.RESET_ALL}")
    print("-" * 70)

    fitted_params = fit_model.parameter_values

    if isinstance(ground_truth, AbstractConstitutiveModel):
        ground_truth_params = ground_truth.parameter_values
    else:
        ground_truth_params = ground_truth

    for param_name, ground_truth_value in ground_truth_params.items():
        if param_name in fitted_params:
            if isinstance(fitted_params[param_name], tuple):
                fitted_mean, fitted_std = fitted_params[param_name]
            else:
                fitted_mean = fitted_params[param_name]
                fitted_std = None

            error_percent = abs((fitted_mean - ground_truth_value) / ground_truth_value) * 100

            if error_percent < 5:
                color = Fore.GREEN
            elif error_percent < 15:
                color = Fore.YELLOW
            else:
                color = Fore.RED

            std_str = f"{fitted_std:.4f}" if fitted_std is not None else "---"

            print(f"{Fore.BLUE}{param_name:<20}{Style.RESET_ALL} "
                  f"{ground_truth_value:<15.4f} "
                  f"{color}{fitted_mean:<15.4f}{Style.RESET_ALL} "
                  f"{Fore.MAGENTA}{std_str:<15}{Style.RESET_ALL}")

    print("-" * 70)

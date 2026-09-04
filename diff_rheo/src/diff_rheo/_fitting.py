"""
Training loop for fitting constitutive models to experimental data.

The :class:`ModelFitter` orchestrates the gradient-based optimisation loop,
using :mod:`optax` for parameter updates and :mod:`equinox` for JAX-compatible
gradient filtering.

This module is intentionally low-level.  Most users should call the higher-level
convenience functions in :mod:`diff_rheo._core`:

* :func:`~diff_rheo._core.fit_model_to_experimental_data` – MAP fitting
* :func:`~diff_rheo._core.fit_variational_inference` – variational ELBO fitting
"""

import optax
import equinox as eqx
from tqdm import tqdm
import jax
from typing import Callable
from ._rheometer import VirtualRheometer
from ._data_types import BatchedData, FittingConfig
from .models import AbstractConstitutiveModel


class ModelFitter:
    """Orchestrates a gradient-based optimisation loop for model fitting.

    Wraps an :mod:`optax` optimiser and a loss function into a training loop
    that supports both deterministic MAP fitting and stochastic variational
    inference.

    Parameters
    ----------
    rheometer : VirtualRheometer
        The virtual rheometer used to simulate predictions.
    optimizer : optax.GradientTransformation
        An optax optimiser (e.g. :func:`optax.adam`).
    config : FittingConfig
        Training configuration (number of epochs, learning rate, etc.).
    loss_fn : Callable
        The loss function with signature
        ``loss_fn(model, rheometer, data_iterator, **kwargs) -> (loss, aux)``.
        Must return a 2-tuple; the second element (``aux``) is ignored by
        the training loop but required for ``eqx.filter_value_and_grad``'s
        ``has_aux=True`` interface.
    """

    def __init__(
        self,
        rheometer: VirtualRheometer,
        optimizer: optax.GradientTransformation,
        config: FittingConfig,
        loss_fn: Callable,
    ):
        self.rheometer = rheometer
        self.optimizer = optimizer
        self.config = config
        self.loss_fn = loss_fn

    @staticmethod
    def _make_step(
        model,
        rheometer,
        opt_state,
        optimizer,
        loss_fn,
        data_iterator,
        *args,
        **kwargs,
    ):
        """Perform a single optimisation step.

        Computes the loss and its gradient with respect to all inexact-array
        (floating-point) leaves of ``model`` using
        :func:`equinox.filter_value_and_grad`, then applies the gradient
        update via the :mod:`optax` optimiser.

        Parameters
        ----------
        model : AbstractConstitutiveModel
            Current model parameter state.
        rheometer : VirtualRheometer
            The virtual rheometer (used inside ``loss_fn``).
        opt_state : optax.OptState
            Current optimiser state (e.g. Adam momentum accumulators).
        optimizer : optax.GradientTransformation
            The optimiser.
        loss_fn : Callable
            Loss function.
        data_iterator : Iterator[ExperimentalData]
            Iterator over the training data for this step.
        *args, **kwargs
            Forwarded to ``loss_fn`` (e.g. ``key``, ``ensemble_size``).

        Returns
        -------
        tuple
            ``(updated_model, updated_opt_state, loss_value, aux)``
        """
        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, rheometer, data_iterator, *args, **kwargs)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, aux

    def train(self, model: AbstractConstitutiveModel, reference: BatchedData) -> tuple:
        """Run the full training loop.

        Iterates for :attr:`FittingConfig.num_epochs` steps.  At each step:

        1. Splits the PRNG key (if variational).
        2. Queries :meth:`~diff_rheo._data_types.BatchedData.fitting_schedule`
           for the data to use this epoch (supports curriculum learning).
        3. Calls :meth:`_make_step` to compute the gradient and update
           parameters.

        Parameters
        ----------
        model : AbstractConstitutiveModel
            The model to train (with trainable parameter leaves).
        reference : BatchedData
            The collection of experimental observations used as the training
            target.

        Returns
        -------
        tuple
            ``(fitted_model, final_loss, aux)`` where ``fitted_model`` has the
            optimised parameter values.
        """
        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        key = self.config.key
        num_epochs = self.config.num_epochs
        verbose = self.config.verbose
        ensemble_size = self.config.ensemble_size

        for i in (pbar := tqdm(range(num_epochs), disable=not verbose)):
            step_kwargs = {"ensemble_size": ensemble_size}
            if key is not None:
                key, subkey = jax.random.split(key)
                step_kwargs["key"] = subkey
            data_iterator = reference.fitting_schedule(self.config, i)
            model, opt_state, loss, aux = self._make_step(model, self.rheometer, opt_state, self.optimizer, self.loss_fn, data_iterator, **step_kwargs)
            # pbar.set_description(f"Loss: {loss.astype(float):.4f}")

        return model, loss, aux

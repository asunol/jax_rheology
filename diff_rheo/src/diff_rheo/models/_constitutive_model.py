"""
Abstract base classes for all constitutive models.

This module defines the class hierarchy that all constitutive models in diff_rheo
must follow:

* :class:`AbstractConstitutiveModel` – root base class providing parameter
  management, JAX pytree integration, and instance sampling for variational
  inference.
* :class:`AbstractGeneralizedNewtonianModel` – for time-independent (purely
  viscous / inelastic) models where stress is an algebraic function of strain rate.
* :class:`AbstractViscoelasticModel` – for time-dependent models that require
  ODE integration of a stress evolution equation.

Parameter registration
----------------------
When you subclass :class:`AbstractConstitutiveModel` and pass keyword arguments
to ``__init__``, the constructor applies the following rules:

* An :class:`~diff_rheo.parameters.AbstractParameter` instance → stored as-is.
* A ``float`` or ``jax.Array`` → automatically wrapped in a
  :class:`~diff_rheo.parameters.LogParameter` (log-space, always positive).
* A ``Callable`` → stored as-is (used for custom F-functions, neural networks).

This means you can quickly create a model with plain floats and it will be
optimisable out of the box::

    model = OldroydB(polymer_viscosity=2.0, relaxation_time=1.0, solvent_viscosity=0.1)
    # All three parameters are now LogParameter instances.
"""

import equinox as eqx
import jax
from ..parameters import AbstractParameter, AbstractRandomParameter, LogParameter, StaticParameter, Parameter
from .._forcing import VelocityGradient, AppliedStress
from typing import Union, Callable, Optional
from abc import abstractmethod


class AbstractConstitutiveModel(eqx.Module):
    """Root base class for all constitutive models in diff_rheo.

    Provides:

    * **Parameter registration** – ``__init__`` stores keyword arguments as
      :class:`~diff_rheo.parameters.AbstractParameter` instances, auto-wrapping
      plain floats as :class:`~diff_rheo.parameters.LogParameter`.
    * **Parameter introspection** – :attr:`parameter_values` returns a dict of
      current scalar values; :attr:`trainable_count` counts optimisable params.
    * **Instance sampling** – :meth:`get_instance` draws a deterministic sample
      from stochastic parameter distributions (for variational inference).
    * **Observation noise** – every model has an ``observation_noise`` parameter
      (default :class:`~diff_rheo.parameters.StaticParameter` 0.0) used in
      variational inference likelihoods.

    Attributes
    ----------
    observation_noise : AbstractParameter
        Measurement noise level.  Defaults to ``StaticParameter(0.0)``.
    """

    observation_noise: AbstractParameter

    def __init__(self, *args, **kwargs):
        """Register keyword arguments as model parameters.

        Parameters
        ----------
        **kwargs
            Parameter name → value mappings.  Each value is handled as:

            * :class:`~diff_rheo.parameters.AbstractParameter` → stored directly.
            * ``float`` / ``jax.Array`` → wrapped in
              :class:`~diff_rheo.parameters.LogParameter`.
            * ``Callable`` → stored directly (e.g. for RUDE neural network).

        Raises
        ------
        ValueError
            If a value is not one of the accepted types.
        """
        if "observation_noise" not in kwargs:
            kwargs["observation_noise"] = StaticParameter(0.0)
        for key, value in kwargs.items():
            if isinstance(value, AbstractParameter):
                setattr(self, key, value)
            elif isinstance(value, (float, jax.Array)):
                setattr(self, key, LogParameter(value))
            elif isinstance(value, Callable):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter type: {type(value)}")

    @property
    def parameter_values(self) -> dict:
        """Dict of current parameter values (or ``(mean, std)`` for stochastic params).

        Returns
        -------
        dict[str, jax.Array | tuple[jax.Array, jax.Array]]
        """
        value_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, AbstractRandomParameter):
                value_dict[key] = value.get_expectation()
            elif isinstance(value, AbstractParameter):
                value_dict[key] = value.get_value()
        return value_dict

    @property
    def trainable_count(self) -> int:
        """Number of trainable (non-static) parameters.

        :class:`~diff_rheo.parameters.StaticParameter` instances are excluded.
        """
        trainable_params = 0
        for key, value in self.__dict__.items():
            if isinstance(value, AbstractParameter) and not isinstance(value, StaticParameter):
                trainable_params += 1
        return trainable_params

    def get_non_log_instance(self) -> "AbstractConstitutiveModel":
        """Return a copy with all parameters unwrapped to plain :class:`~diff_rheo.parameters.Parameter`.

        Static parameters are preserved.  Useful for inspection or downstream
        processing without log-space transformations.
        """
        new_instance_kwargs = {}
        for key, value in self.__dict__.items():
            if isinstance(value, AbstractParameter) and not isinstance(value, StaticParameter):
                new_instance_kwargs[key] = Parameter(value.get_value())
            elif isinstance(value, StaticParameter):
                new_instance_kwargs[key] = value
        return self.__class__(**new_instance_kwargs)

    def get_instance(self, key: Optional[jax.Array] = None) -> "AbstractConstitutiveModel":
        """Return a deterministic model instance by sampling stochastic parameters.

        For each :class:`~diff_rheo.parameters.AbstractRandomParameter` attribute:

        * If ``key`` is provided, draws a random sample.
        * If ``key`` is ``None``, uses the distribution mean.

        Non-random parameters are passed through unchanged.  If the model has
        no stochastic parameters, ``self`` is returned directly.

        Parameters
        ----------
        key : jax.Array | None
            JAX PRNG key for sampling.  Split internally so each stochastic
            parameter gets an independent subkey.

        Returns
        -------
        AbstractConstitutiveModel
            A new instance with only deterministic parameter leaves.
        """
        samplable_params = {
            k: v for k, v in self.__dict__.items() if isinstance(v, AbstractRandomParameter)
        }

        if not samplable_params:
            return self
        new_instance_kwargs = {}
        if key is not None:
            keys = jax.random.split(key, len(samplable_params))
            key_iterator = iter(keys)
        else:
            key_iterator = iter([])
        for attr_name, attr_value in self.__dict__.items():
            if attr_name in samplable_params:
                if key is not None:
                    new_instance_kwargs[attr_name] = attr_value.sample(next(key_iterator))
                else:
                    new_instance_kwargs[attr_name] = attr_value.get_nonrandom()
            else:
                new_instance_kwargs[attr_name] = attr_value

        return self.__class__(**new_instance_kwargs)


class AbstractGeneralizedNewtonianModel(AbstractConstitutiveModel):
    """Base class for time-independent (generalized Newtonian) fluid models.

    The stress is an instantaneous function of the rate-of-strain:

        σ(t) = 2 η(γ̇(t)) D(t)

    No history or ODE integration is required.

    Subclasses must implement :meth:`stress_response` and
    :meth:`strain_response`.
    """

    @abstractmethod
    def stress_response(
        self,
        t: Union[float, jax.Array],
        velocity_gradient: VelocityGradient,
        *args,
        **kwargs,
    ) -> jax.Array:
        """Compute the stress tensor at time ``t`` given the velocity gradient.

        Parameters
        ----------
        t : float | jax.Array
            Evaluation time.
        velocity_gradient : VelocityGradient
            The prescribed velocity gradient L(t).

        Returns
        -------
        jax.Array
            Stress tensor, shape ``(3, 3)``.
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    @abstractmethod
    def strain_response(
        self,
        t: Union[float, jax.Array],
        applied_stress: AppliedStress,
        *args,
        **kwargs,
    ) -> jax.Array:
        """Compute the strain-rate tensor at time ``t`` given the applied stress.

        Parameters
        ----------
        t : float | jax.Array
            Evaluation time.
        applied_stress : AppliedStress
            The prescribed stress tensor σ(t).

        Returns
        -------
        jax.Array
            Rate-of-strain tensor D, shape ``(3, 3)``.
        """
        raise NotImplementedError("This method should be implemented by the subclass")


class AbstractViscoelasticModel(AbstractConstitutiveModel):
    """Base class for viscoelastic fluid models requiring ODE integration.

    The total stress is split into solvent and polymer (extra-stress) parts:

        σ_total = 2 η_s D + τ

    where the solvent part is algebraic and τ satisfies a constitutive ODE.

    Subclasses must implement :meth:`extra_stress_response_rhs` (for
    strain-rate controlled experiments) and :meth:`shear_stress_experiment_rhs`
    (for stress-controlled experiments).

    Attributes
    ----------
    solvent_viscosity : AbstractParameter
        The Newtonian solvent viscosity η_s.
    """

    solvent_viscosity: AbstractParameter

    @abstractmethod
    def extra_stress_response_rhs(
        self,
        t: Union[float, jax.Array],
        stress: jax.Array,
        velocity_gradient: VelocityGradient,
        *args,
        **kwargs,
    ) -> jax.Array:
        """ODE RHS for the extra-stress in a strain-rate controlled experiment.

        Computes dτ/dt given the current extra-stress τ and velocity gradient.

        Parameters
        ----------
        t : float | jax.Array
            Current time.
        stress : jax.Array
            Current extra-stress tensor τ, shape ``(3, 3)``.
        velocity_gradient : VelocityGradient
            The prescribed velocity gradient.

        Returns
        -------
        jax.Array
            dτ/dt, shape ``(3, 3)``.
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    @abstractmethod
    def shear_stress_experiment_rhs(
        self,
        t: Union[float, jax.Array],
        current_values: jax.Array,
        applied_stress: AppliedStress,
        *args,
        **kwargs,
    ) -> jax.Array:
        """ODE RHS for the combined state in a stress-controlled experiment.

        State vector: ``[σ₁₁, σ₂₂, σ₃₃, σ₁₃, σ₂₃, γ̇, γ]`` (length 7).

        Parameters
        ----------
        t : float | jax.Array
            Current time.
        current_values : jax.Array
            Current state vector, shape ``(7,)``.
        applied_stress : AppliedStress
            The prescribed shear stress σ₁₂(t).

        Returns
        -------
        jax.Array
            Rate of change of state, shape ``(7,)``.
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    def solvent_stress(
        self,
        t: Union[float, jax.Array],
        velocity_gradient: VelocityGradient,
    ) -> jax.Array:
        """Compute the Newtonian solvent stress σ_s = 2 η_s D(t).

        Parameters
        ----------
        t : float | jax.Array
            Time point(s).
        velocity_gradient : VelocityGradient
            The velocity gradient L(t).

        Returns
        -------
        jax.Array
            Solvent stress tensor, same shape as the rate-of-strain output.
        """
        rate_of_strain = velocity_gradient.rate_of_strain(t)
        solvent_stress = 2 * self.solvent_viscosity.get_value() * rate_of_strain
        return solvent_stress
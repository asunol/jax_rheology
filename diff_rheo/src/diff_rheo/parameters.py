"""
Parameter classes for constitutive model parameters.

This module defines the parameter hierarchy used throughout diff_rheo. Parameters
wrap scalar values and control how they are stored, transformed, and sampled during
optimization and variational inference.

Parameter hierarchy
-------------------
AbstractParameter
├── Parameter                  – Plain trainable scalar (stored as-is)
├── StaticParameter            – Non-trainable constant (excluded from gradient updates)
├── LogParameter               – Log-space trainable scalar; always positive
├── TanhParameter              – Bounded trainable scalar in (0, max_value)
└── AbstractRandomParameter    – Base for variational / stochastic parameters
    ├── GaussianParameter      – Variational Normal distribution N(μ, σ²)
    └── LogGaussianParameter   – Variational Log-Normal distribution LogN(μ, σ²)

Choosing a parameter type
--------------------------
* Use ``Parameter`` for general unconstrained quantities.
* Use ``LogParameter`` for strictly positive quantities (viscosities, relaxation
  times, etc.).  The optimiser works in log-space, preventing negative values.
* Use ``TanhParameter`` for quantities bounded in (0, max_value), such as the
  Giesekus mobility parameter α ∈ (0, 1).
* Use ``StaticParameter`` for constants that should never be updated (e.g. a
  known solvent viscosity that is held fixed during fitting).
* Use ``GaussianParameter`` or ``LogGaussianParameter`` with
  :func:`~diff_rheo._core.fit_variational_inference` to obtain posterior
  uncertainty estimates over parameters.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Union
from abc import abstractmethod


class AbstractParameter(eqx.Module):
    """Abstract base class for all model parameters.

    Every parameter must implement :meth:`get_value`, which returns the
    current scalar value of the parameter as a JAX array.  The value
    returned is what the constitutive model equations actually use.
    """

    @abstractmethod
    def get_value(self, *args, **kwargs) -> jax.Array:
        """Return the current scalar value of this parameter.

        Returns
        -------
        jax.Array
            The parameter value, as a 0-D JAX array.
        """
        raise NotImplementedError("This method should be implemented by the subclass")


class AbstractRandomParameter(AbstractParameter):
    """Abstract base class for stochastic / variational parameters.

    Extends :class:`AbstractParameter` with methods required for variational
    inference: sampling from the distribution, obtaining a deterministic
    (non-random) version, and computing the distributional expectation and
    standard deviation.

    Used by :func:`~diff_rheo._core.fit_variational_inference` to perform
    ELBO optimisation over parameter posteriors.
    """

    @abstractmethod
    def sample(self, key: jax.Array, *args, **kwargs) -> AbstractParameter:
        """Draw a single sample from the parameter distribution.

        Parameters
        ----------
        key : jax.Array
            A JAX PRNG key used to generate the random sample.

        Returns
        -------
        AbstractParameter
            A deterministic parameter (e.g. :class:`Parameter` or
            :class:`LogParameter`) whose value is one draw from this
            distribution.
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    @abstractmethod
    def get_nonrandom(self) -> AbstractParameter:
        """Return a deterministic parameter at the distribution mean.

        Returns
        -------
        AbstractParameter
            A deterministic parameter whose value equals the expected value
            of this distribution (without any randomness).
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    @abstractmethod
    def get_expectation(self) -> jax.Array:
        """Return the mean and standard deviation of the distribution.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            ``(mean, std)`` of the distribution in the original (un-transformed)
            parameter space.
        """
        raise NotImplementedError("This method should be implemented by the subclass")


class Parameter(AbstractParameter):
    """A plain trainable scalar parameter stored without transformation.

    The value is stored and retrieved as-is; gradient updates are applied
    directly to the stored value.

    Parameters
    ----------
    value : float | int | jax.Array
        The initial value of the parameter.

    Examples
    --------
    >>> p = Parameter(1.5)
    >>> p.get_value()
    Array(1.5, dtype=float32)
    """

    value: jax.Array

    def __init__(self, value: Union[float, int, jax.Array]):
        self.value = jnp.array(value)

    def get_value(self, *args, **kwargs) -> jax.Array:
        """Return the parameter value directly."""
        return self.value


class StaticParameter(Parameter):
    """A non-trainable constant parameter.

    Unlike :class:`Parameter`, this stores its value as a plain Python
    ``float`` rather than a JAX array, which means Equinox/JAX will not
    include it in gradient computations.  Use this for known constants that
    should remain fixed throughout optimisation (e.g. a fixed solvent
    viscosity, or the ``observation_noise`` default of ``0.0``).

    Parameters
    ----------
    value : float | int | jax.Array
        The constant value; stored as ``float``.
    """

    value: float

    def __init__(self, value: Union[float, int, jax.Array]):
        self.value = float(value)


class LogParameter(AbstractParameter):
    """A trainable parameter stored in log-space.

    Internally stores ``log(value)``; :meth:`get_value` returns
    ``exp(stored)`` which is always strictly positive.  This is the
    default parameter type used when a constitutive model is initialised
    with a plain ``float`` (see :class:`~diff_rheo.models.AbstractConstitutiveModel`).

    Parameters
    ----------
    value : float | int | jax.Array
        The initial **physical** value (must be positive).

    Examples
    --------
    >>> p = LogParameter(2.0)
    >>> p.get_value()   # returns 2.0
    Array(2., dtype=float32)
    """

    value: jax.Array

    def __init__(self, value: Union[float, int, jax.Array]):
        self.value = jnp.log(jnp.array(value))

    def get_value(self, *args, **kwargs) -> jax.Array:
        """Return ``exp(stored_log_value)``, always positive."""
        return jnp.exp(self.value)


class TanhParameter(AbstractParameter):
    """A trainable parameter bounded in ``(0, max_value)`` via a tanh transform.

    Internally stores ``atanh(2*value/max_value - 1)``; :meth:`get_value`
    inverts the transform to recover a value strictly between ``0`` and
    ``max_value``.  Useful for parameters like the Giesekus mobility
    coefficient α which must lie in (0, 1).

    Parameters
    ----------
    value : float | int | jax.Array
        The initial **physical** value; must satisfy ``0 < value < max_value``.
    max_value : float
        Upper bound for the parameter.  Defaults to ``1.0``.

    Examples
    --------
    >>> p = TanhParameter(0.3, max_value=1.0)
    >>> p.get_value()   # ≈ 0.3
    """

    value: jax.Array
    max_value: float

    def __init__(self, value: Union[float, int, jax.Array], max_value: float = 1.0):
        self.value = jnp.atanh(2 * jnp.array(value) / max_value - 1)
        self.max_value = max_value

    def get_value(self, *args, **kwargs) -> jax.Array:
        """Return the bounded physical value via the inverse tanh transform."""
        return (jnp.tanh(self.value) + 1) * self.max_value / 2


class GaussianParameter(AbstractRandomParameter):
    """A variational parameter following a Normal distribution N(μ, σ²).

    Used in variational inference to represent uncertainty over a model
    parameter.  During ensemble sampling, each draw produces an independent
    :class:`Parameter` with value ``μ + σ * ε``, where ``ε ~ N(0,1)``.

    The standard deviation is stored in log-space (``log_std``) to ensure
    positivity during optimisation.

    Parameters
    ----------
    mean : float | int | jax.Array
        Initial mean of the variational distribution.
    std : float | int | jax.Array
        Initial standard deviation (must be positive).

    Notes
    -----
    The KL divergence against a standard Normal prior N(0,1) is computed in
    :func:`~diff_rheo._core.kl_divergence` and added to the ELBO loss.
    """

    mean: jax.Array
    log_std: jax.Array

    def __init__(self, mean: Union[float, int, jax.Array], std: Union[float, int, jax.Array]):
        self.mean = jnp.array(mean)
        self.log_std = jnp.log(jnp.array(std))

    @property
    def std(self) -> jax.Array:
        """Standard deviation, recovered as ``exp(log_std)``."""
        return jnp.exp(self.log_std)

    def get_value(self, *args, **kwargs) -> jax.Array:
        """Return the distribution mean (point estimate)."""
        return self.mean

    def sample(self, key: jax.Array, *args, **kwargs) -> AbstractParameter:
        """Sample a value from N(μ, σ²) and wrap it as a :class:`Parameter`."""
        value = self.mean + self.std * jax.random.normal(key)
        return Parameter(value)

    def get_nonrandom(self) -> AbstractParameter:
        """Return a deterministic :class:`Parameter` at the mean."""
        return Parameter(self.mean)

    def get_expectation(self) -> tuple[jax.Array, jax.Array]:
        """Return ``(mean, std)`` of the Normal distribution."""
        return self.mean, self.std


class LogGaussianParameter(AbstractRandomParameter):
    """A variational parameter following a Log-Normal distribution LogN(μ, σ²).

    The underlying Normal is parameterised in log-space: if ``X ~ N(μ, σ²)``
    then ``exp(X)`` follows a Log-Normal.  This guarantees the sampled
    physical parameter is always strictly positive, making it suitable for
    viscosities, relaxation times, and similar quantities.

    Internally stores ``log(mean_physical)`` as the location and ``log(std)``
    as the log-scale of the underlying Normal.

    Parameters
    ----------
    mean : float | int | jax.Array
        Initial **expected value** of the Log-Normal in physical space
        (i.e. ``E[exp(X)]``).
    std : float | int | jax.Array
        Initial standard deviation of the **underlying Normal** (must be
        positive).

    Notes
    -----
    * ``get_value()`` returns the Log-Normal mean:
      ``exp(μ + σ²/2)``.
    * ``get_expectation()`` returns ``(lognormal_mean, lognormal_std)``
      in physical space.
    * The KL divergence against a standard Normal prior (in log-space) is
      computed by :func:`~diff_rheo._core.kl_divergence`.
    """

    mean: jax.Array
    log_std: jax.Array

    def __init__(self, mean: Union[float, int, jax.Array], std: Union[float, int, jax.Array]):
        self.mean = jnp.log(jnp.array(mean))
        self.log_std = jnp.log(jnp.array(std))

    @property
    def std(self) -> jax.Array:
        """Standard deviation of the underlying Normal, recovered as ``exp(log_std)``."""
        return jnp.exp(self.log_std)

    def get_value(self, *args, **kwargs) -> jax.Array:
        """Return the Log-Normal expected value: ``exp(μ + σ²/2)``."""
        return jnp.exp(self.mean + self.std**2 / 2)

    def sample(self, key: jax.Array, *args, **kwargs) -> AbstractParameter:
        """Sample a value from LogN and wrap it as a :class:`LogParameter`.

        The sampled value is ``exp(μ + σ * ε)`` where ``ε ~ N(0,1)``.
        """
        value = jnp.exp(self.mean + self.std * jax.random.normal(key))
        return LogParameter(value)

    def get_nonrandom(self) -> AbstractParameter:
        """Return a deterministic :class:`LogParameter` at the Log-Normal mean."""
        return LogParameter(jnp.exp(self.mean + self.std**2 / 2))

    def get_expectation(self) -> tuple[jax.Array, jax.Array]:
        """Return ``(lognormal_mean, lognormal_std)`` in physical parameter space.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            * mean: ``exp(μ + σ²/2)``
            * std:  ``exp(μ + σ²/2) * sqrt(exp(σ²) - 1)``
        """
        lognormal_mean = jnp.exp(self.mean + self.std**2 / 2)
        lognormal_std = lognormal_mean * jnp.sqrt(jnp.exp(self.std**2) - 1)
        return lognormal_mean, lognormal_std

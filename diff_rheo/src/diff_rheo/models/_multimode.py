"""
Multi-mode (relaxation-spectrum) constitutive models.

A real polymer melt or solution relaxes over a *spectrum* of timescales, not a
single one.  The standard way to capture this is a sum of independent
viscoelastic modes that share the same kinematics::

    τ(t) = Σₖ τₖ(t),     λₖ ∇τₖ + τₖ = 2 ηₖ D       (k = 1 … N)

Each mode obeys its own Oldroyd-B / UCM equation with its own modulus and
relaxation time; the modes are coupled only through the common velocity
gradient.  Because the modes have *different* λₖ, the sum does not collapse to
a single closed equation — the integrated state is genuinely a stack of ``N``
independent 3×3 stress tensors.

Fitting a multi-mode model is the classic ill-posed "relaxation-spectrum
inversion" problem: many ``{ηₖ, λₖ}`` reproduce the same linear response, so
the modes are sloppy.  This makes multi-mode models a natural marquee target
for the Fisher-information / sloppy-direction machinery in
:mod:`diff_rheo._information` and for posterior model selection over the number
of modes.

The :class:`AbstractMultiModeModel` marker tells
:meth:`~diff_rheo._rheometer.VirtualRheometer.setup` to select the dedicated
:class:`~diff_rheo._protocols.MultiModeStrainRateProtocol`, which integrates
the stacked state and sums the modes (plus the Newtonian solvent) into the
total stress.
"""

from typing import Union

import jax
import jax.numpy as jnp
import equinox as eqx

from ._constitutive_model import AbstractViscoelasticModel
from ..parameters import AbstractParameter, LogParameter, Parameter, StaticParameter
from .._forcing import VelocityGradient, AppliedStress


class AbstractMultiModeModel(AbstractViscoelasticModel):
    """Base class for models whose integrated state is a stack of ``N`` modes.

    Subclasses store per-mode parameters as array-valued parameters and
    implement :meth:`extra_stress_response_rhs` to act on a stress of shape
    ``(n_modes, 3, 3)``, returning the per-mode rate of change of the same
    shape.  The total polymer stress is the sum over modes, assembled by
    :class:`~diff_rheo._protocols.MultiModeStrainRateProtocol`.
    """

    @property
    def n_modes(self) -> int:
        """Number of relaxation modes ``N``."""
        raise NotImplementedError("Subclasses must implement n_modes")


class MultiModeOldroydB(AbstractMultiModeModel):
    """Multi-mode Oldroyd-B / Upper-Convected Maxwell model.

    A sum of ``N`` independent Oldroyd-B modes sharing the kinematics::

        λₖ ∇τₖ + τₖ = 2 ηₖ D,    σ = 2 η_s D + Σₖ τₖ

    where ∇ is the upper-convected derivative ``∇τ = dτ/dt − L·τ − τ·Lᵀ``.

    Parameters
    ----------
    polymer_viscosities : jax.Array | AbstractParameter
        Per-mode polymer viscosities ``ηₖ`` (Pa·s), shape ``(N,)``.  A bare
        array is auto-wrapped in a :class:`~diff_rheo.parameters.LogParameter`,
        so all modes stay strictly positive under optimisation.
    relaxation_times : jax.Array | AbstractParameter
        Per-mode relaxation times ``λₖ`` (s), shape ``(N,)``.
    solvent_viscosity : float | AbstractParameter
        Newtonian solvent viscosity η_s (Pa·s) — a single scalar shared by all
        modes.

    Notes
    -----
    The discrete relaxation spectrum ``{ηₖ, λₖ}`` reproduces a single-mode
    Oldroyd-B exactly for ``N = 1``.  The linear viscoelastic moduli are
    ``G'(ω) = Σₖ ηₖ/λₖ · (ωλₖ)²/(1+(ωλₖ)²)`` and the analogous ``G''`` — the
    quantities a SAOS frequency sweep measures.
    """

    polymer_viscosities: AbstractParameter
    relaxation_times: AbstractParameter
    solvent_viscosity: AbstractParameter

    @property
    def n_modes(self) -> int:
        return int(self.polymer_viscosities.get_value().shape[0])

    @eqx.filter_jit
    def extra_stress_response_rhs(
        self,
        t: Union[float, jax.Array],
        stress: jax.Array,
        velocity_gradient: VelocityGradient,
        *args,
        **kwargs,
    ) -> jax.Array:
        """Per-mode extra-stress ODE RHS for a strain-rate experiment.

        Parameters
        ----------
        stress : jax.Array
            Stacked per-mode extra stress, shape ``(N, 3, 3)``.

        Returns
        -------
        jax.Array
            Stacked dτₖ/dt, shape ``(N, 3, 3)``.
        """
        u_grad = velocity_gradient.gradient(t)
        rate_of_strain = velocity_gradient.rate_of_strain(t)
        eta = self.polymer_viscosities.get_value()   # (N,)
        lam = self.relaxation_times.get_value()       # (N,)

        def single_mode(tau_k, eta_k, lam_k):
            return (2.0 * eta_k * rate_of_strain - tau_k) / lam_k \
                + u_grad @ tau_k + tau_k @ u_grad.T

        return jax.vmap(single_mode)(stress, eta, lam)

    def shear_stress_experiment_rhs(
        self,
        t: Union[float, jax.Array],
        current_values: jax.Array,
        applied_stress: AppliedStress,
        *args,
        **kwargs,
    ) -> jax.Array:
        """Stress-controlled multi-mode integration is not yet implemented."""
        raise NotImplementedError(
            "Stress-controlled (shear_stress_response) protocol is not yet "
            "implemented for multi-mode models; use strain_rate_response."
        )


class OrderedMultiModeOldroydB(AbstractMultiModeModel):
    """Multi-mode Oldroyd-B with strictly ordered relaxation times.

    Same physics as :class:`MultiModeOldroydB`, but reparameterises the
    relaxation-time vector to enforce ``λ₁ < λ₂ < ... < λ_N`` for free,
    eliminating the ``N!`` permutation symmetry that traps gradient descent
    in mode-swapped local minima.  This addresses the basin-of-attraction
    failure documented in §7.5 of the information-geometry write-up: the
    Fisher information is sufficient for identification, but the symmetric
    loss landscape has 3! = 6 equivalent global minima per truly distinct
    solution, plus a runaway-mode basin with comparable SSE.

    Reparameterisation
    ------------------
    Stores

    * ``log_lambda_min`` -- scalar, ``log`` of the fastest relaxation time.
    * ``log_increments`` -- vector of length ``N-1``; pushed through
      ``softplus`` so each entry is *strictly positive*, then accumulated.

    The relaxation times are recovered as

    .. code::

        λ₁ = exp(log_lambda_min)
        λ_{k+1} = λ_k · exp(softplus(log_increments[k-1]))    (k = 1, ..., N-1)

    With ``softplus(x) = log(1 + exp(x)) > 0``, the increments are always
    positive, so ``λ_k`` is monotonically increasing.  The polymer viscosities
    and solvent viscosity are stored as ordinary LogParameters (positive,
    unconstrained order).

    Parameters
    ----------
    polymer_viscosities : jax.Array | AbstractParameter
        Per-mode polymer viscosities ``ηₖ``, shape ``(N,)``.  A bare array is
        auto-wrapped in :class:`~diff_rheo.parameters.LogParameter`.
    relaxation_times : jax.Array, optional
        Initial **physical** ordered relaxation times, shape ``(N,)``.
        Internally split into ``log_lambda_min`` (scalar) and
        ``log_increments`` (vector of length ``N-1`` parameterising the
        cumulative log-ratios via softplus).  If omitted, pass
        ``log_lambda_min`` and ``log_increments`` directly.
    solvent_viscosity : float | AbstractParameter
        Newtonian solvent viscosity.
    log_lambda_min, log_increments : AbstractParameter, optional
        Direct parameterisation (mutually exclusive with ``relaxation_times``).

    Notes
    -----
    The :attr:`relaxation_times` property computes the ordered vector from
    the underlying ``log_lambda_min`` and ``log_increments``, so the rest of
    the multi-mode machinery (the protocol, the Fisher tooling) sees the
    same interface as :class:`MultiModeOldroydB`.  The trainable leaves
    exposed to the optimiser are ``log_lambda_min``, ``log_increments``,
    ``polymer_viscosities`` and ``solvent_viscosity`` -- the same parameter
    count as the unordered model, just a different chart of the manifold.
    """

    polymer_viscosities: AbstractParameter
    log_lambda_min: AbstractParameter
    log_increments: AbstractParameter
    solvent_viscosity: AbstractParameter

    def __init__(
        self,
        polymer_viscosities,
        solvent_viscosity,
        *,
        relaxation_times=None,
        log_lambda_min=None,
        log_increments=None,
        observation_noise=None,
    ):
        # Inherited from AbstractConstitutiveModel; must be set even though the
        # multi-mode protocol does not use it for the forward solve.
        self.observation_noise = (
            observation_noise if isinstance(observation_noise, AbstractParameter)
            else StaticParameter(0.0)
        )
        # Polymer / solvent viscosities follow the standard auto-wrap convention.
        if isinstance(polymer_viscosities, AbstractParameter):
            self.polymer_viscosities = polymer_viscosities
        else:
            self.polymer_viscosities = LogParameter(jnp.asarray(polymer_viscosities))

        if isinstance(solvent_viscosity, AbstractParameter):
            self.solvent_viscosity = solvent_viscosity
        else:
            self.solvent_viscosity = LogParameter(jnp.asarray(solvent_viscosity))

        # Relaxation times: either pass the physical vector and we split it,
        # or pass the unconstrained scalar + increment leaves directly.
        if relaxation_times is not None:
            if log_lambda_min is not None or log_increments is not None:
                raise ValueError(
                    "Pass either `relaxation_times` OR "
                    "(`log_lambda_min`, `log_increments`), not both."
                )
            lam = jnp.asarray(relaxation_times)
            if lam.ndim != 1:
                raise ValueError(f"relaxation_times must be 1-D, got shape {lam.shape}.")
            lam_sorted = jnp.sort(lam)
            log_lam_min = jnp.log(lam_sorted[0])
            log_ratios = jnp.log(lam_sorted[1:] / lam_sorted[:-1])  # > 0
            # Invert softplus to recover the unconstrained logit
            # softplus(x) = y  =>  x = log(exp(y) - 1).  Add tiny floor so the
            # log is finite when two modes are exactly equal at init.
            inv_softplus = jnp.log(jnp.expm1(jnp.maximum(log_ratios, 1e-6)))
            self.log_lambda_min = Parameter(log_lam_min)
            self.log_increments = Parameter(inv_softplus)
        else:
            if log_lambda_min is None or log_increments is None:
                raise ValueError(
                    "Provide `relaxation_times` or both "
                    "`log_lambda_min` and `log_increments`."
                )
            self.log_lambda_min = (
                log_lambda_min if isinstance(log_lambda_min, AbstractParameter)
                else Parameter(jnp.asarray(log_lambda_min))
            )
            self.log_increments = (
                log_increments if isinstance(log_increments, AbstractParameter)
                else Parameter(jnp.asarray(log_increments))
            )

    @property
    def relaxation_times_value(self) -> jax.Array:
        """The ordered ``(N,)`` relaxation-time vector recovered from the leaves."""
        lam_min = jnp.exp(self.log_lambda_min.get_value())
        increments = jax.nn.softplus(self.log_increments.get_value())
        log_ratios = jnp.cumsum(increments)
        ratios = jnp.exp(log_ratios)
        return lam_min * jnp.concatenate([jnp.ones((1,), dtype=ratios.dtype), ratios])

    @property
    def n_modes(self) -> int:
        return int(self.polymer_viscosities.get_value().shape[0])

    @eqx.filter_jit
    def extra_stress_response_rhs(
        self,
        t: Union[float, jax.Array],
        stress: jax.Array,
        velocity_gradient: VelocityGradient,
        *args,
        **kwargs,
    ) -> jax.Array:
        u_grad = velocity_gradient.gradient(t)
        rate_of_strain = velocity_gradient.rate_of_strain(t)
        eta = self.polymer_viscosities.get_value()
        lam = self.relaxation_times_value

        def single_mode(tau_k, eta_k, lam_k):
            return (2.0 * eta_k * rate_of_strain - tau_k) / lam_k \
                + u_grad @ tau_k + tau_k @ u_grad.T

        return jax.vmap(single_mode)(stress, eta, lam)

    def shear_stress_experiment_rhs(
        self,
        t: Union[float, jax.Array],
        current_values: jax.Array,
        applied_stress: AppliedStress,
        *args,
        **kwargs,
    ) -> jax.Array:
        raise NotImplementedError(
            "Stress-controlled (shear_stress_response) protocol is not yet "
            "implemented for multi-mode models; use strain_rate_response."
        )

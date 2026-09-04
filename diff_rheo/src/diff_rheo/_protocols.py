"""
Experiment protocols that define how a constitutive model is simulated.

A *protocol* encapsulates the physics of a particular type of rheometer
experiment: it knows how to combine a constitutive model, a forcing function,
and a solver to produce a :class:`~diff_rheo._data_types.SimulationData` result.

Available protocols
-------------------
* :class:`GeneralizedNewtonianStrainRateProtocol` – algebraic evaluation of
  σ = 2η(γ̇) D for inelastic (non-time-dependent) models.
* :class:`GeneralizedNewtonianShearStressProtocol` – stress-controlled version
  (not yet implemented; raises :exc:`NotImplementedError`).
* :class:`ViscoelasticStrainRateProtocol` – integrates the extra-stress ODE
  for viscoelastic models under a prescribed velocity gradient.
* :class:`ViscoelasticShearStressProtocol` – integrates the combined stress +
  strain ODE for viscoelastic models under a prescribed shear stress.

Protocol selection is handled automatically by
:meth:`~diff_rheo._rheometer.VirtualRheometer.setup` based on the model type
and experiment type string.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from abc import abstractmethod
from ._solver import AbstractODESolver
from ._data_types import ODESolution, SimulationData
from .models import AbstractConstitutiveModel, AbstractGeneralizedNewtonianModel, AbstractViscoelasticModel, AbstractMultiModeModel
from ._forcing import AbstractForcing, VelocityGradient, AppliedStress


class AbstractProtocol(eqx.Module):
    """Abstract base class for experiment protocols.

    Subclasses implement :meth:`run` to execute a single simulation and
    return a :class:`~diff_rheo._data_types.SimulationData`.
    """

    @abstractmethod
    def run(
        self,
        model: AbstractConstitutiveModel,
        forcing: AbstractForcing,
        time_range: jax.Array,
        initial_condition: jax.Array,
        solver: AbstractODESolver,
    ) -> SimulationData:
        """Run a single virtual rheometer experiment.

        Parameters
        ----------
        model : AbstractConstitutiveModel
            The constitutive model whose parameters are used for this simulation.
        forcing : AbstractForcing
            The applied forcing (a :class:`~diff_rheo._forcing.VelocityGradient`
            or :class:`~diff_rheo._forcing.AppliedStress`).
        time_range : jax.Array
            Time points at which to save the solution, shape ``(T,)``.
        initial_condition : jax.Array
            Initial state for the ODE solver (e.g. zero stress tensor).
        solver : AbstractODESolver
            The ODE solver to use for time integration.

        Returns
        -------
        SimulationData
            The simulation result containing the solved trajectory.
        """
        pass


class GeneralizedNewtonianStrainRateProtocol(AbstractProtocol):
    """Algebraic stress evaluation for generalized Newtonian models.

    No ODE integration is needed; the stress is computed directly from the
    instantaneous rate-of-strain tensor via σ(t) = 2 η(γ̇(t)) D(t).
    The computation is vectorised over the time array using
    :func:`equinox.filter_vmap`.

    Notes
    -----
    Selected automatically by :meth:`~diff_rheo._rheometer.VirtualRheometer.setup`
    for :class:`~diff_rheo.models.AbstractGeneralizedNewtonianModel` with
    ``experiment_type="strain_rate_response"``.
    """

    @eqx.filter_jit
    def run(
        self,
        model: AbstractGeneralizedNewtonianModel,
        forcing: VelocityGradient,
        time_range: jax.Array,
        *args,
        **kwargs,
    ) -> SimulationData:
        """Evaluate stress at each time point in ``time_range``.

        Returns
        -------
        SimulationData
            ``solution.ys`` shape ``(T, 3, 3)`` – stress tensor at each time step.
        """
        stress_response = eqx.filter_vmap(model.stress_response, in_axes=(0, None))(time_range, forcing)
        solution = ODESolution(ys=stress_response, ts=time_range, result=True, stats={}, raw_solution=None)
        return SimulationData(forcing_function=forcing, solution=solution, experiment_type="strain_rate_response")


class GeneralizedNewtonianShearStressProtocol(AbstractProtocol):
    """Stress-controlled protocol for generalized Newtonian models.

    .. warning::
        Not implemented.  Raises :exc:`NotImplementedError` when called.
        Inverting from prescribed stress to strain rate requires a root-find
        which is not yet supported.
    """

    @eqx.filter_jit
    def run(
        self,
        model: AbstractGeneralizedNewtonianModel,
        forcing: AppliedStress,
        time_range: jax.Array,
        *args,
        **kwargs,
    ) -> SimulationData:
        raise NotImplementedError("Shear stress response not implemented for Generalized Newtonian model")


class ViscoelasticStrainRateProtocol(AbstractProtocol):
    """ODE integration protocol for strain-rate controlled viscoelastic experiments.

    Splits the total stress into solvent and polymer (extra) contributions:

        σ_total(t) = σ_s(t) + τ(t)

    where σ_s = 2 η_s D(t) is algebraic and τ(t) is obtained by integrating
    the constitutive extra-stress ODE:

        dτ/dt = f(t, τ, L)

    The initial extra-stress is derived from the supplied initial condition
    minus the solvent contribution at ``t₀``.

    Notes
    -----
    Selected automatically by :meth:`~diff_rheo._rheometer.VirtualRheometer.setup`
    for :class:`~diff_rheo.models.AbstractViscoelasticModel` with
    ``experiment_type="strain_rate_response"``.
    """

    @eqx.filter_jit
    def run(
        self,
        model: AbstractViscoelasticModel,
        velocity_gradient: VelocityGradient,
        time_range: jax.Array,
        initial_condition: jax.Array,
        solver: AbstractODESolver,
    ) -> SimulationData:
        """Integrate the extra-stress ODE and assemble total stress.

        Parameters
        ----------
        initial_condition : jax.Array
            Total stress initial condition, shape ``(3, 3)``.

        Returns
        -------
        SimulationData
            ``solution.ys`` shape ``(T, 3, 3)`` – total stress tensor at each
            saved time point.
        """
        model_instance = model.get_instance()
        solvent_stress = model_instance.solvent_stress(time_range, velocity_gradient)
        initial_extra_stress = initial_condition - solvent_stress[0]
        extra_stress = solver.integrate(model_instance.extra_stress_response_rhs, initial_extra_stress, time_range, velocity_gradient)
        total_stress = ODESolution(ys=extra_stress.ys + solvent_stress, ts=time_range, result=extra_stress.result, stats=extra_stress.stats, raw_solution=extra_stress.raw_solution)
        return SimulationData(forcing_function=velocity_gradient, solution=total_stress, experiment_type="strain_rate_response")


class MultiModeStrainRateProtocol(AbstractProtocol):
    """Strain-rate protocol for multi-mode viscoelastic models.

    Integrates a *stacked* extra-stress state of shape ``(N, 3, 3)`` — one 3×3
    tensor per relaxation mode — under a prescribed velocity gradient, then
    assembles the total stress as the sum over modes plus the Newtonian
    solvent contribution::

        σ_total(t) = 2 η_s D(t) + Σₖ τₖ(t)

    The supplied (3×3) initial condition is the *total* stress at ``t₀``; the
    excess over the solvent stress is placed entirely in mode 0 so that the
    reconstructed total stress matches the initial condition at ``t₀`` (for the
    usual start-from-rest case the excess is zero).

    Notes
    -----
    The output ``solution.ys`` has shape ``(T, 3, 3)`` — identical to the
    single-mode :class:`ViscoelasticStrainRateProtocol` — so every downstream
    observable (σ₁₂, N₁, σ_E) works unchanged.

    Selected automatically by :meth:`~diff_rheo._rheometer.VirtualRheometer.setup`
    for :class:`~diff_rheo.models.AbstractMultiModeModel` with
    ``experiment_type="strain_rate_response"``.
    """

    @eqx.filter_jit
    def run(
        self,
        model: AbstractMultiModeModel,
        velocity_gradient: VelocityGradient,
        time_range: jax.Array,
        initial_condition: jax.Array,
        solver: AbstractODESolver,
    ) -> SimulationData:
        model_instance = model.get_instance()
        solvent_stress = model_instance.solvent_stress(time_range, velocity_gradient)
        n_modes = model_instance.n_modes
        initial_extra_total = initial_condition - solvent_stress[0]
        initial_modes = jnp.zeros((n_modes, 3, 3)).at[0].set(initial_extra_total)
        modes = solver.integrate(
            model_instance.extra_stress_response_rhs, initial_modes, time_range, velocity_gradient
        )
        total = jnp.sum(modes.ys, axis=1) + solvent_stress
        solution = ODESolution(
            ys=total, ts=time_range, result=modes.result,
            stats=modes.stats, raw_solution=modes.raw_solution,
        )
        return SimulationData(forcing_function=velocity_gradient, solution=solution, experiment_type="strain_rate_response")


class ViscoelasticShearStressProtocol(AbstractProtocol):
    """ODE integration protocol for stress-controlled viscoelastic experiments.

    Integrates a combined state vector ``[σ₁₁, σ₂₂, σ₃₃, σ₁₃, σ₂₃, γ̇, γ]``
    (length 7) under a prescribed shear stress waveform σ₁₂(t).  The full ODE
    RHS is provided by
    :meth:`~diff_rheo.models.AbstractViscoelasticModel.shear_stress_experiment_rhs`.

    Notes
    -----
    Selected automatically by :meth:`~diff_rheo._rheometer.VirtualRheometer.setup`
    for :class:`~diff_rheo.models.AbstractViscoelasticModel` with
    ``experiment_type="shear_stress_response"``.
    """

    @eqx.filter_jit
    def run(
        self,
        model: AbstractViscoelasticModel,
        forcing: AppliedStress,
        time_range: jax.Array,
        initial_condition: jax.Array,
        solver: AbstractODESolver,
    ) -> SimulationData:
        """Integrate the combined stress + strain ODE under prescribed shear stress.

        Parameters
        ----------
        initial_condition : jax.Array
            Initial state vector of length 7:
            ``[σ₁₁, σ₂₂, σ₃₃, σ₁₃, σ₂₃, γ̇, γ]``.

        Returns
        -------
        SimulationData
            ``solution.ys`` shape ``(T, 7)`` – full state vector at each time
            point.  ``ys[:, -1]`` is the accumulated shear strain γ.
        """
        model_instance = model.get_instance()
        stress_and_strain_response = solver.integrate(model_instance.shear_stress_experiment_rhs, initial_condition, time_range, forcing)
        return SimulationData(forcing_function=forcing, solution=stress_and_strain_response, experiment_type="shear_stress_response")
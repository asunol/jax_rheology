"""
Virtual rheometer that executes constitutive model simulations.

The :class:`VirtualRheometer` is the central entry-point for running
simulations.  It combines a constitutive model type, an experiment type, and
an ODE solver into a single reusable object.

Typical usage
-------------
1. Create a model instance (e.g. :class:`~diff_rheo.models.OldroydB`).
2. Construct a rheometer via :meth:`VirtualRheometer.setup`.
3. Run a single deterministic experiment via :meth:`run_experiment`.
4. For uncertainty quantification with stochastic parameters, run an ensemble
   via :meth:`run_ensemble`.
"""

import equinox as eqx
import jax

from typing import Optional
from ._solver import AbstractODESolver
from ._data_types import SimulationData
from .models import AbstractConstitutiveModel, AbstractGeneralizedNewtonianModel, AbstractViscoelasticModel, AbstractMultiModeModel
from ._forcing import AbstractForcing
from ._protocols import AbstractProtocol, GeneralizedNewtonianStrainRateProtocol, ViscoelasticStrainRateProtocol, ViscoelasticShearStressProtocol, MultiModeStrainRateProtocol


class VirtualRheometer(eqx.Module):
    """A virtual rheometer that simulates constitutive model experiments.

    Encapsulates the experiment protocol and ODE solver so that the same
    rheometer object can be reused across many training steps with different
    model parameter values or forcing functions.

    Do not construct directly; use the :meth:`setup` factory method instead,
    which automatically selects the correct :class:`~diff_rheo._protocols.AbstractProtocol`
    based on the model type and experiment type.

    Attributes
    ----------
    protocol : AbstractProtocol
        The experiment protocol (determines how the simulation is run).
    solver : AbstractODESolver | None
        The ODE solver used for time integration.  ``None`` for generalized
        Newtonian models which require no time integration.
    """

    protocol: AbstractProtocol
    solver: Optional[AbstractODESolver]

    @classmethod
    def setup(
        cls,
        model: AbstractConstitutiveModel,
        experiment_type: str,
        solver: Optional[AbstractODESolver] = None,
    ) -> "VirtualRheometer":
        """Factory method that creates a properly configured :class:`VirtualRheometer`.

        Selects the appropriate protocol based on the model and experiment type,
        and validates that a solver is provided when required.

        Parameters
        ----------
        model : AbstractConstitutiveModel
            A constitutive model instance.  Used only to determine the model
            class; the actual parameter values are not stored in the rheometer.
        experiment_type : str
            One of:

            * ``"strain_rate_response"`` – prescribe velocity gradient L(t),
              measure stress σ(t).
            * ``"shear_stress_response"`` – prescribe shear stress σ₁₂(t),
              measure strain γ(t).

        solver : AbstractODESolver | None
            An ODE solver (e.g. :class:`~diff_rheo._solver.DiffraxSolver`).
            **Required** for all viscoelastic models.  May be ``None`` for
            generalized Newtonian models.

        Returns
        -------
        VirtualRheometer

        Raises
        ------
        ValueError
            If ``experiment_type`` is not one of the supported strings, if the
            model type is unrecognised, or if no solver is provided for a
            viscoelastic model.
        NotImplementedError
            If ``experiment_type="shear_stress_response"`` is requested for a
            generalized Newtonian model (not yet supported).

        Examples
        --------
        >>> model = OldroydB(polymer_viscosity=2.0, relaxation_time=1.0, solvent_viscosity=0.1)
        >>> solver = DiffraxSolver()
        >>> rheometer = VirtualRheometer.setup(model, "strain_rate_response", solver)
        """
        if experiment_type not in ["strain_rate_response", "shear_stress_response"]:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        if isinstance(model, AbstractGeneralizedNewtonianModel):
            if experiment_type == "strain_rate_response":
                protocol = GeneralizedNewtonianStrainRateProtocol()
            elif experiment_type == "shear_stress_response":
                raise NotImplementedError("Shear stress response not implemented for Generalized Newtonian model")
        elif isinstance(model, AbstractMultiModeModel):
            # Checked before AbstractViscoelasticModel: multi-mode models subclass it
            # but integrate a stacked (N, 3, 3) state and need their own protocol.
            if experiment_type == "strain_rate_response":
                protocol = MultiModeStrainRateProtocol()
            elif experiment_type == "shear_stress_response":
                raise NotImplementedError("Shear stress response not implemented for multi-mode models")
            if solver is None:
                raise ValueError("Solver must be provided for viscoelastic models")
        elif isinstance(model, AbstractViscoelasticModel):
            if experiment_type == "strain_rate_response":
                protocol = ViscoelasticStrainRateProtocol()
            elif experiment_type == "shear_stress_response":
                protocol = ViscoelasticShearStressProtocol()
            if solver is None:
                raise ValueError("Solver must be provided for viscoelastic models")
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
        return cls(protocol, solver)

    @eqx.filter_jit
    def run_experiment(
        self,
        model: AbstractConstitutiveModel,
        forcing: AbstractForcing,
        time_range: jax.Array,
        initial_condition: jax.Array,
    ) -> SimulationData:
        """Run a single deterministic simulation.

        Delegates to the configured protocol.  JIT-compiled for performance.

        Parameters
        ----------
        model : AbstractConstitutiveModel
            The model with current parameter values to simulate.
        forcing : AbstractForcing
            The forcing function for this experiment (velocity gradient or
            applied stress).
        time_range : jax.Array
            Sorted array of time points to save, shape ``(T,)``.
        initial_condition : jax.Array
            Initial state for the ODE solver.

        Returns
        -------
        SimulationData
            The simulation result.
        """
        return self.protocol.run(model, forcing, time_range, initial_condition, self.solver)

    def run_ensemble(
        self,
        model: AbstractConstitutiveModel,
        forcing: AbstractForcing,
        time_range: jax.Array,
        initial_condition: jax.Array,
        key: jax.random.PRNGKey,
        size: int = 1,
    ) -> SimulationData:
        """Run an ensemble of simulations by sampling stochastic model parameters.

        For each ensemble member, independently samples all
        :class:`~diff_rheo.parameters.AbstractRandomParameter` attributes of
        ``model`` using a different subkey, then runs a deterministic simulation
        with the sampled parameter values.  All ``size`` runs are vectorised via
        :func:`equinox.filter_vmap`.

        Parameters
        ----------
        model : AbstractConstitutiveModel
            A model with one or more :class:`~diff_rheo.parameters.GaussianParameter`
            or :class:`~diff_rheo.parameters.LogGaussianParameter` attributes.
        forcing : AbstractForcing
            The forcing function shared across all ensemble members.
        time_range : jax.Array
            Sorted array of time points to save, shape ``(T,)``.
        initial_condition : jax.Array
            Shared initial state for all ensemble members.
        key : jax.random.PRNGKey
            PRNG key used to generate ``size`` independent subkeys.
        size : int
            Number of ensemble members to simulate.  Defaults to ``1``.

        Returns
        -------
        SimulationData
            A batched ``SimulationData`` where each array field has an
            additional leading dimension of size ``size`` (e.g.
            ``solution.ys`` has shape ``(size, T, ...)``)."""
        subkeys = jax.random.split(key, size)

        def _run_single(subkey):
            # Sample a single deterministic model instance from the distribution
            model_instance = model.get_instance(subkey)
            # Run the protocol with that deterministic instance
            return self.protocol.run(model_instance, forcing, time_range, initial_condition, self.solver)

        data = eqx.filter_vmap(_run_single)(subkeys)
        return data

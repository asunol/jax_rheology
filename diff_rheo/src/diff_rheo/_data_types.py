"""
Core data structures for experimental data and simulation results.

This module defines the data containers used throughout diff_rheo to represent:

* **ODE solver output** – :class:`ODESolution`, :class:`SimulationData`
* **Experimental observations** – :class:`ExperimentalData` and its concrete
  subclasses :class:`ShearStrainRateData` and :class:`ShearStressData`
* **Training configuration** – :class:`FittingConfig`
* **Multi-experiment datasets** – :class:`BatchedData`

Data flow overview
------------------
1. Experimental data (time, measured signal, applied forcing) is wrapped in a
   concrete :class:`ExperimentalData` subclass.
2. ``BatchedData.from_data(*experiments)`` groups multiple experiments.
3. During fitting, ``BatchedData.fitting_schedule()`` yields individual
   :class:`ExperimentalData` objects that are passed to the loss function.
4. The loss function calls ``reference.get_forcing_function()`` to reconstruct a
   callable forcing from the stored discrete data (via cubic Hermite
   interpolation), then feeds it to the :class:`~diff_rheo._rheometer.VirtualRheometer`.
5. Predictions are extracted from the resulting :class:`SimulationData` via
   ``reference.extract_from_simulation()``.
"""

import equinox as eqx
import jax
from typing import Any, Callable, Iterator
from ._forcing import AbstractForcing, VelocityGradient, AppliedStress, extensional_forcing
import diffrax as dfx
from dataclasses import dataclass


class ODESolution(eqx.Module):
    """Container for the output of an ODE integration step.

    Wraps the raw :mod:`diffrax` solution together with convenience
    accessors used by the rest of the pipeline.

    Attributes
    ----------
    ys : jax.Array
        Solution values at each saved time point.  Shape depends on the
        state dimension, e.g. ``(T, 3, 3)`` for a 3×3 stress tensor
        integrated over ``T`` time points.
    ts : jax.Array
        Time points at which the solution is saved, shape ``(T,)``.
    result : bool
        ``True`` if the solver completed successfully
        (:attr:`diffrax.RESULTS.successful`), ``False`` otherwise.
    stats : dict
        Solver statistics returned by :mod:`diffrax` (e.g. number of steps,
        function evaluations).
    raw_solution : Any
        The raw :class:`diffrax.Solution` object, for access to additional
        solver metadata.
    """

    ys: jax.Array
    ts: jax.Array
    result: bool
    stats: dict
    raw_solution: Any


@dataclass
class FittingConfig:
    """Configuration for a model-fitting run.

    Passed to :func:`~diff_rheo._core.fit_model_to_experimental_data` and
    :func:`~diff_rheo._core.fit_variational_inference` to control the
    optimisation loop.

    Attributes
    ----------
    num_epochs : int
        Total number of gradient-descent steps to perform.
    learning_rate : float
        Step size for the Adam optimiser.
    ensemble_size : int
        Number of parameter samples drawn per ELBO estimate during variational
        inference.  Set to ``1`` for deterministic (MAP) fitting.
        Larger values reduce gradient variance but increase compute cost.
    verbose : bool
        If ``True``, display a :mod:`tqdm` progress bar during training.
    key : jax.random.PRNGKey | None
        JAX PRNG key used to split subkeys for stochastic sampling.  Required
        for variational inference; may be ``None`` for deterministic fitting.
    schedule_lr : bool
        Whether the training loop should anneal the learning rate over epochs.
        Defaults to ``False`` (constant learning rate).

    Examples
    --------
    >>> config = FittingConfig(num_epochs=1000, learning_rate=1e-2,
    ...                        ensemble_size=100, verbose=True,
    ...                        key=jax.random.PRNGKey(0))
    """

    num_epochs: int
    learning_rate: float
    ensemble_size: int = 1
    verbose: bool = False
    key: jax.random.PRNGKey = None
    schedule_lr: bool = False


class SimulationData(eqx.Module):
    """Output of a virtual rheometer experiment.

    Bundles the :class:`ODESolution` with the forcing function that produced
    it and a label describing the experiment type.  Concrete
    :class:`ExperimentalData` subclasses know how to extract the relevant
    scalar observable (e.g. shear stress σ₁₂) from this container via
    :meth:`~ExperimentalData.extract_from_simulation`.

    Attributes
    ----------
    forcing_function : AbstractForcing
        The forcing applied during the simulation (e.g. a
        :class:`~diff_rheo._forcing.VelocityGradient` for strain-rate
        controlled experiments).
    solution : ODESolution
        The integrated ODE solution.
    experiment_type : str
        One of ``"strain_rate_response"`` or ``"shear_stress_response"``,
        indicating whether the forcing was a prescribed velocity gradient or a
        prescribed shear stress.
    """

    forcing_function: AbstractForcing
    solution: ODESolution
    experiment_type: str

    @property
    def result(self) -> bool:
        """``True`` if the underlying ODE solve succeeded."""
        return self.solution.result

    @property
    def time(self) -> jax.Array:
        """Time array from the ODE solution, shape ``(T,)``."""
        return self.solution.ts

    @property
    def data(self) -> jax.Array:
        """Solution array ``ys`` from the ODE solution."""
        return self.solution.ys

    @property
    def forcing_data(self) -> jax.Array:
        """Evaluate the forcing function at all saved time points."""
        return self.forcing_function.gradient(self.time)


class ExperimentalData(eqx.Module):
    """Abstract base class for experimental observations from a rheometer.

    Stores the raw time-series data measured in a rheology experiment and
    provides methods to:

    1. Reconstruct a smooth :class:`~diff_rheo._forcing.AbstractForcing`
       callable from the stored discrete forcing signal via cubic Hermite
       interpolation (:meth:`get_forcing_function`).
    2. Extract the simulated observable that corresponds to the measured
       quantity from a :class:`SimulationData` result
       (:meth:`extract_from_simulation`).

    Both methods must be implemented by concrete subclasses.

    Attributes
    ----------
    time : jax.Array
        Measurement time points, shape ``(T,)``.
    data : jax.Array
        Measured observable values (e.g. shear stress), shape ``(T,)`` or
        ``(T, ...)``.
    forcing_data : jax.Array
        Discrete samples of the applied forcing (e.g. strain rate or shear
        stress waveform), shape ``(T,)`` or ``(T, ...)``.  Used to reconstruct
        the interpolated forcing function.
    initial_condition : jax.Array
        Initial state for the ODE solver (e.g. zero stress tensor for a
        start-from-rest experiment).

    See Also
    --------
    ShearStrainRateData : Strain-rate controlled shear experiment.
    ShearStressData : Stress-controlled shear experiment.
    """

    time: jax.Array
    data: jax.Array
    forcing_data: jax.Array
    initial_condition: jax.Array

    def __init__(self, time: jax.Array, data: jax.Array, forcing_data: jax.Array, initial_condition: jax.Array):
        self.time = time
        self.data = data
        self.forcing_data = forcing_data
        self.initial_condition = initial_condition

    def _generate_forcing_function(self) -> Callable:
        """Build a cubic Hermite interpolant from the stored discrete forcing.

        Uses :func:`diffrax.backward_hermite_coefficients` and
        :class:`diffrax.CubicInterpolation` so the resulting callable is
        compatible with JAX JIT compilation and differentiation.

        Returns
        -------
        Callable
            A function ``f(t) -> jax.Array`` that evaluates the smoothly
            interpolated forcing at arbitrary time ``t``.
        """
        coeffs = dfx.backward_hermite_coefficients(self.time, self.forcing_data)
        return dfx.CubicInterpolation(self.time, coeffs).evaluate

    def get_forcing_function(self) -> AbstractForcing:
        """Reconstruct the :class:`~diff_rheo._forcing.AbstractForcing` for this experiment.

        Must be implemented by subclasses to wrap the interpolated scalar
        signal in the appropriate forcing type (e.g. a
        :class:`~diff_rheo._forcing.VelocityGradient` or
        :class:`~diff_rheo._forcing.AppliedStress`).

        Returns
        -------
        AbstractForcing
            The forcing object suitable for passing to
            :meth:`~diff_rheo._rheometer.VirtualRheometer.run_experiment`.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def extract_from_simulation(self, simulation: SimulationData) -> jax.Array:
        """Extract the simulated observable corresponding to the measured quantity.

        Must be implemented by subclasses to select the relevant component
        from the full simulation output (e.g. shear stress component σ₁₂ from
        the 3×3 stress tensor).

        Parameters
        ----------
        simulation : SimulationData
            The result of running the virtual rheometer.

        Returns
        -------
        jax.Array
            The extracted quantity, with the same shape as :attr:`data`.
        """
        raise NotImplementedError("Subclasses must implement this method")


class ShearStrainRateData(ExperimentalData):
    """Experimental data from a strain-rate controlled shear experiment.

    The applied forcing is the (1,2) component of the velocity gradient
    (i.e. the shear strain rate ``du₁/dx₂``).  The measured observable is
    the resulting shear stress σ₁₂.

    The forcing is reconstructed as a :class:`~diff_rheo._forcing.VelocityGradient`
    with only the ``grad_u_12`` component set; all other components are zero.
    The observable is extracted as ``simulation.data[:, 0, 1]`` (the (0,1)
    component of the 3×3 stress tensor, i.e. σ₁₂).

    Notes
    -----
    ``forcing_data`` should be a 1-D array of the applied shear strain rate
    sampled at each time point.
    """

    def get_forcing_function(self) -> VelocityGradient:
        """Reconstruct the velocity gradient forcing with shear component only."""
        return VelocityGradient.from_components(grad_u_12=self._generate_forcing_function())

    def extract_from_simulation(self, simulation: SimulationData) -> jax.Array:
        """Extract σ₁₂ (row 0, column 1) from the simulated stress tensor."""
        return simulation.data[:, 0, 1]


class ShearStrainRateNormalStressData(ShearStrainRateData):
    """Strain-rate shear experiment that also measures the normal stress.

    Identical forcing to :class:`ShearStrainRateData` (the (1,2) component of
    the velocity gradient), but the measured observable is the **two-channel**
    signal ``[σ₁₂, N₁]`` where ``N₁ = σ₁₁ − σ₂₂`` is the first normal-stress
    difference.

    The shear stress σ₁₂ alone cannot tell apart constitutive models that are
    degenerate in simple shear — Giesekus with small α, the linear PTT model
    with small ε, and Oldroyd-B all produce nearly identical σ₁₂(t).  They
    differ in N₁: Oldroyd-B gives ``N₁ ∝ γ̇²`` exactly, whereas Giesekus and
    PTT shear-thin in N₁.  Adding the normal-stress channel therefore breaks
    the shear-stress degeneracy that no choice of shear waveform can resolve.

    Notes
    -----
    ``data`` must have shape ``(T, 2)``: column 0 is σ₁₂, column 1 is N₁.
    ``forcing_data`` is the 1-D applied shear strain rate, exactly as for
    :class:`ShearStrainRateData`.  The MSE fitting loss and the L2-BIC are
    shape-agnostic, so this type drops straight into the existing fitting and
    model-selection pipeline.
    """

    def extract_from_simulation(self, simulation: SimulationData) -> jax.Array:
        """Extract ``[σ₁₂, N₁]`` from the simulated 3×3 stress tensor."""
        stress = simulation.data
        shear = stress[:, 0, 1]
        first_normal = stress[:, 0, 0] - stress[:, 1, 1]
        return jax.numpy.stack([shear, first_normal], axis=-1)


class ExtensionalStrainRateData(ExperimentalData):
    """Experimental data from a rate-controlled *extensional* experiment.

    The applied forcing is the extension rate ε̇(t); the velocity gradient is
    the traceless diagonal extension tensor (see
    :func:`~diff_rheo._forcing.extensional_forcing`).  The measured observable
    is the tensile stress difference ``σ_E = σ₁₁ − σ₂₂`` — the quantity from
    which the extensional viscosity ``η_E = σ_E / ε̇`` is obtained.

    Extension is the kinematics that breaks the shear-only degeneracies: in
    simple shear Oldroyd-B, FENE-P, PTT and Giesekus can all be tuned to nearly
    identical σ₁₂(t), but their extensional responses diverge sharply (Oldroyd-B
    has an unbounded steady η_E that blows up as ε̇·λ → 1/2, while the
    finite-extensibility / network models saturate).  Measuring σ_E therefore
    discriminates models no shear waveform can.

    Parameters
    ----------
    mode : str
        Extension geometry passed to
        :func:`~diff_rheo._forcing.extensional_forcing` — ``"uniaxial"``
        (default), ``"planar"`` or ``"biaxial"``.

    Notes
    -----
    ``forcing_data`` is the 1-D applied extension rate sampled at each time
    point; ``data`` is the 1-D σ_E trajectory.  The MSE loss and L2-BIC are
    shape-agnostic, so this type drops straight into the fitting / selection
    pipeline.
    """

    mode: str = eqx.field(static=True)

    def __init__(self, time, data, forcing_data, initial_condition, mode: str = "uniaxial"):
        super().__init__(time, data, forcing_data, initial_condition)
        self.mode = mode

    def get_forcing_function(self) -> VelocityGradient:
        """Reconstruct the extensional velocity gradient from the rate signal."""
        return extensional_forcing(self._generate_forcing_function(), mode=self.mode)

    def extract_from_simulation(self, simulation: SimulationData) -> jax.Array:
        """Extract the tensile stress difference σ_E = σ₁₁ − σ₂₂."""
        stress = simulation.data
        return stress[:, 0, 0] - stress[:, 1, 1]


class ShearStressData(ExperimentalData):
    """Experimental data from a stress-controlled shear experiment.

    The applied forcing is the (1,2) component of the stress tensor (the
    prescribed shear stress σ₁₂).  The measured observable is the resulting
    cumulative shear strain ``γ`` (the last element of the solver state
    vector).

    The forcing is reconstructed as an :class:`~diff_rheo._forcing.AppliedStress`
    with only the ``sigma_12`` component set.  The observable is extracted
    as ``simulation.data[:, -1]``, which corresponds to the accumulated
    shear strain γ in the ODE state vector.

    Notes
    -----
    ``forcing_data`` should be a 1-D array of the applied shear stress
    sampled at each time point.
    """

    def get_forcing_function(self) -> AppliedStress:
        """Reconstruct the applied stress forcing with shear component only."""
        return AppliedStress.from_components(sigma_12=self._generate_forcing_function())

    def extract_from_simulation(self, simulation: SimulationData) -> jax.Array:
        """Extract accumulated shear strain γ (last element of state vector)."""
        return simulation.data[:, -1]


class BatchedData:
    """Container for a collection of :class:`ExperimentalData` experiments.

    Groups multiple rheology experiments (potentially with different
    waveforms, amplitudes, or frequencies) into a single object that the
    training loop iterates over.  A custom ``fitting_schedule`` can be
    implemented in a subclass to implement curriculum learning (e.g.
    introducing more data as training progresses).

    Attributes
    ----------
    data : list[ExperimentalData]
        The list of individual experiments.

    Examples
    --------
    >>> batched = BatchedData.from_data(exp1, exp2, exp3)
    >>> for ref in batched.fitting_schedule(config, epoch=0):
    ...     loss += compute_loss(model, ref)
    """

    data: list[ExperimentalData]

    def __init__(self, data: list[ExperimentalData]):
        self.data = data

    @classmethod
    def from_data(cls, *args: ExperimentalData) -> "BatchedData":
        """Construct a :class:`BatchedData` from positional :class:`ExperimentalData` arguments.

        Parameters
        ----------
        *args : ExperimentalData
            Any number of :class:`ExperimentalData` (or subclass) instances.

        Returns
        -------
        BatchedData

        Raises
        ------
        ValueError
            If any argument is not an instance of :class:`ExperimentalData`.
        """
        data_list = []
        for arg in args:
            if isinstance(arg, ExperimentalData):
                data_list.append(arg)
            else:
                raise ValueError(f"Invalid data type: {type(arg)}")
        return cls(data_list)

    def fitting_schedule(self, config: FittingConfig, epoch: int, *args, **kwargs) -> Iterator[ExperimentalData]:
        """Iterate over experiments for a given training epoch.

        The default implementation yields **all** experiments every epoch.
        Override this method in a subclass to implement curriculum strategies
        (e.g. only use a subset of experiments in early epochs).

        Parameters
        ----------
        config : FittingConfig
            The training configuration (available for epoch-dependent logic).
        epoch : int
            The current epoch index (0-based).

        Yields
        ------
        ExperimentalData
            Individual experiments to include in this epoch's loss computation.
        """
        for i in range(0, len(self.data)):
            yield self.data[i]

    def __len__(self) -> int:
        """Return the number of experiments in the dataset."""
        return len(self.data)

"""
ODE solver wrappers for integrating constitutive model equations.

This module provides an abstraction layer over :mod:`diffrax` that is used
by the :mod:`~diff_rheo._protocols` to integrate the stress evolution ODEs
for viscoelastic models.

The key class is :class:`DiffraxSolver`, which wraps :func:`diffrax.diffeqsolve`
and returns a standardised :class:`~diff_rheo._data_types.ODESolution` container.

Extending
---------
To add a new solver backend, subclass :class:`AbstractODESolver` and implement
:meth:`AbstractODESolver.integrate`.
"""

import equinox as eqx
import jax
import diffrax as dfx

from typing import Callable, Tuple, Any
from abc import abstractmethod
from ._data_types import ODESolution


class AbstractODESolver(eqx.Module):
    """Abstract base class for ODE solvers used by the virtual rheometer.

    Subclasses must implement :meth:`integrate` to numerically integrate a
    given right-hand-side function over a time range and return an
    :class:`~diff_rheo._data_types.ODESolution`.
    """

    @abstractmethod
    def integrate(
        self,
        rhs_function: Callable,
        initial_condition: jax.Array,
        time_range: jax.Array,
        args: Tuple[Any],
    ) -> ODESolution:
        """Integrate an ODE from ``time_range[0]`` to ``time_range[-1]``.

        Parameters
        ----------
        rhs_function : Callable
            The right-hand side ``f(t, y, args) -> dy/dt`` of the ODE.
            Must be compatible with JAX JIT compilation.
        initial_condition : jax.Array
            The initial state ``y(t₀)``.
        time_range : jax.Array
            Array of time points at which to save the solution, shape ``(T,)``.
            The solver integrates from ``time_range[0]`` to ``time_range[-1]``.
        args : tuple
            Extra arguments forwarded verbatim to ``rhs_function`` (e.g. the
            forcing function).

        Returns
        -------
        ODESolution
            Standardised solution container with ``ys``, ``ts``, ``result``,
            ``stats``, and ``raw_solution`` fields.
        """
        pass


class DiffraxSolver(AbstractODESolver):
    """Adaptive-step ODE solver backed by :mod:`diffrax`.

    Uses a PID step-size controller for adaptive integration and saves the
    solution at the user-specified time points via ``diffrax.SaveAt``.

    Supported stepping methods (passed as the ``solver`` string):

    * ``"tsit5"`` (default) – Tsitouras 5th-order Runge-Kutta.  A good
      general-purpose choice for non-stiff problems.
    * ``"dopri5"`` – Dormand-Prince 4(5)th-order.  Similar accuracy to
      Tsit5, occasionally preferred for certain problem structures.
    * ``"heun"`` – Heun's method (2nd-order).  Use for quick experiments or
      when stiffness requires very small steps anyway.

    Parameters
    ----------
    solver : str
        Stepping method identifier.  One of ``"tsit5"``, ``"dopri5"``,
        ``"heun"``.  Defaults to ``"tsit5"``.
    rtol : float
        Relative tolerance for the PID step-size controller.  Defaults to
        ``1e-5``.
    atol : float
        Absolute tolerance for the PID step-size controller.  Defaults to
        ``1e-5``.
    dt0 : float
        Initial step size hint passed to :func:`diffrax.diffeqsolve`.
        Defaults to ``0.1``.
    max_steps : int
        Maximum number of solver steps before aborting.  Increase this for
        long time windows or stiff systems.  Defaults to ``10000``.
    adjoint : str
        Autodiff strategy for differentiating through the solve:

        * ``"recursive_checkpoint"`` (default) – memory-efficient reverse-mode
          autodiff (:class:`diffrax.RecursiveCheckpointAdjoint`).  Use for
          ordinary gradient-based fitting.  **Reverse-mode only.**
        * ``"direct"`` – :class:`diffrax.DirectAdjoint`, which additionally
          supports *forward-mode* and higher-order autodiff.  Required by the
          information-geometry tools in :mod:`diff_rheo._information` when used
          with ``mode="fwd"`` (e.g. :func:`~diff_rheo.optimize_experiment`).
    **kwargs
        Additional keyword arguments forwarded verbatim to
        :func:`diffrax.diffeqsolve`.

    Examples
    --------
    Default solver::

        solver = DiffraxSolver()

    Tighter tolerances for high-accuracy integration::

        solver = DiffraxSolver(rtol=1e-8, atol=1e-8, max_steps=50000)

    Notes
    -----
    The :meth:`integrate` method is decorated with ``@eqx.filter_jit`` so
    it is JIT-compiled on the first call for each unique traced signature.
    """

    solver: dfx.AbstractSolver
    stepsize_controller: dfx.PIDController
    adjoint: dfx.AbstractAdjoint
    diffeqsolve_kwargs: dict

    def __init__(
        self,
        solver: str = "tsit5",
        rtol: float = 1e-5,
        atol: float = 1e-5,
        dt0: float = 0.1,
        max_steps: int = 10000,
        adjoint: str = "recursive_checkpoint",
        **kwargs,
    ):
        if solver == "tsit5":
            self.solver = dfx.Tsit5()
        elif solver == "dopri5":
            self.solver = dfx.Dopri5()
        # Need to add constant step size controller
        # elif solver == "euler":
        #     self.solver = dfx.Euler()
        elif solver == "heun":
            self.solver = dfx.Heun()
        else:
            raise ValueError(f"Unknown solver: {solver}")

        if adjoint == "recursive_checkpoint":
            self.adjoint = dfx.RecursiveCheckpointAdjoint()
        elif adjoint == "direct":
            self.adjoint = dfx.DirectAdjoint()
        else:
            raise ValueError(f"Unknown adjoint: {adjoint}")

        self.stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)
        self.diffeqsolve_kwargs = {"dt0": dt0, "max_steps": max_steps, **kwargs}

    @eqx.filter_jit
    def integrate(
        self,
        rhs_function: Callable,
        initial_condition: jax.Array,
        time_range: jax.Array,
        args: Tuple[Any],
    ) -> ODESolution:
        """Integrate the ODE using the configured diffrax solver.

        Wraps :func:`diffrax.diffeqsolve` with :class:`diffrax.SaveAt` so the
        solution is stored at each point in ``time_range``.

        Parameters
        ----------
        rhs_function : Callable
            RHS function ``f(t, y, args) -> dy/dt``.
        initial_condition : jax.Array
            Initial state ``y(time_range[0])``.
        time_range : jax.Array
            Sorted array of save times, shape ``(T,)``.
        args : tuple
            Extra arguments forwarded to ``rhs_function`` (typically the
            :class:`~diff_rheo._forcing.AbstractForcing` object).

        Returns
        -------
        ODESolution
            ``ys`` has shape ``(T, *state_shape)``; ``result`` is ``True``
            iff the solver terminated successfully.
        """
        term = dfx.ODETerm(rhs_function)
        saveat = dfx.SaveAt(ts=time_range)
        sol = dfx.diffeqsolve(
            term,
            self.solver,
            t0=time_range[0],
            t1=time_range[-1],
            saveat=saveat,
            y0=initial_condition,
            stepsize_controller=self.stepsize_controller,
            adjoint=self.adjoint,
            args=args,
            **self.diffeqsolve_kwargs,
        )

        return ODESolution(
            ys=sol.ys,
            ts=sol.ts,
            result=(sol.result == dfx.RESULTS.successful),
            stats=sol.stats,
            raw_solution=sol,
        )

"""
Forcing function representations for rheology experiments.

This module defines the two types of external forcing that can be applied in a
virtual rheometer experiment:

* :class:`VelocityGradient` – a 3×3 velocity gradient tensor L(t), used in
  strain-rate-controlled experiments.
* :class:`AppliedStress` – a symmetric 3×3 stress tensor σ(t), used in
  stress-controlled experiments.

Both classes use an internal ``_Component_Container`` (an Equinox module) to
store each tensor component as a callable ``f(t) -> scalar``.  Components can
be set to arbitrary Python callables (e.g. ``lambda t: 2*jnp.sin(t)``) or to
constant values.  The factory method ``from_components(**kwargs)`` constructs
the appropriate container; any unspecified component defaults to zero.

Usage
-----
Construct a simple oscillatory shear velocity gradient::

    import jax.numpy as jnp
    from diff_rheo import VelocityGradient

    def shear_rate(t):
        return 2.0 * jnp.sin(t)

    L = VelocityGradient.from_components(grad_u_12=shear_rate)
    L_matrix = L.gradient(t=1.0)   # 3×3 array
    D_matrix  = L.rate_of_strain(t=1.0)  # symmetric part

Apply a prescribed shear stress::

    from diff_rheo import AppliedStress

    sigma = AppliedStress.from_components(sigma_12=lambda t: 1.0)
    sigma_matrix = sigma.stress(t=0.5)   # 3×3 symmetric array
"""

import jax.numpy as jnp
import jax
import equinox as eqx
from typing import Callable


class _Component_Container(eqx.Module):
    """Internal Equinox module that stores tensor components as callables."""

    def get_component(self, t: float | jax.Array) -> jax.Array:
        raise NotImplementedError("Subclasses must implement this method")


class AbstractForcing(eqx.Module):
    """Abstract base class for all forcing function types.

    Subclasses store a :class:`_Component_Container` holding the individual
    tensor components and expose high-level methods to evaluate the full
    tensor at a given time ``t``.

    Attributes
    ----------
    forcing_fn : _Component_Container
        Internal container of per-component callable functions.
    """

    forcing_fn: _Component_Container

    @classmethod
    def from_components(cls, **kwargs) -> "AbstractForcing":
        """Construct a forcing object from named tensor components.

        Must be implemented by subclasses.  Keyword argument names should match
        the component naming convention of the subclass (e.g. ``grad_u_12`` for
        :class:`VelocityGradient` or ``sigma_12`` for :class:`AppliedStress`).

        Parameters
        ----------
        **kwargs
            Named components.  Values may be either a ``float`` / scalar
            (treated as a time-constant) or a ``Callable[[float], jax.Array]``.

        Returns
        -------
        AbstractForcing
        """
        raise NotImplementedError("Subclasses must implement this method")


class VelocityGradient(AbstractForcing):
    """A time-dependent 3×3 velocity gradient tensor L(t) = ∂u_i/∂x_j.

    Stores the nine components of L as individual callables and exposes
    ``gradient(t)`` and ``rate_of_strain(t)`` for evaluation.  Commonly used
    to drive strain-rate controlled experiments.

    The rate-of-strain tensor D(t) is the symmetric part of L:

        D(t) = 0.5 * (L(t) + L(t)ᵀ)

    For a simple shear flow with shear rate γ̇(t), only ``grad_u_12`` is
    non-zero: L₁₂ = γ̇(t).

    Parameters
    ----------
    forcing_fn : _Component_Container
        Internal container storing each of the nine ``grad_u_ij`` callables.

    Examples
    --------
    Oscillatory shear at amplitude 2.0 and frequency ω = 1.0::

        L = VelocityGradient.from_components(
            grad_u_12=lambda t: 2.0 * jnp.sin(t)
        )
        L_at_t1 = L.gradient(1.0)        # shape (3, 3)
        D_at_t1 = L.rate_of_strain(1.0)  # shape (3, 3), symmetric

    Evaluate over an array of times::

        ts = jnp.linspace(0, 5, 100)
        D_series = L.rate_of_strain(ts)  # shape (100, 3, 3)
    """

    forcing_fn: _Component_Container

    def __init__(self, forcing_fn: _Component_Container):
        """Construct directly from a pre-built component container."""
        self.forcing_fn = forcing_fn

    @classmethod
    def from_components(cls, **kwargs):
        """Build a :class:`VelocityGradient` from named components.

        Valid keyword arguments are ``grad_u_ij`` where ``i, j ∈ {1, 2, 3}``,
        e.g. ``grad_u_12``, ``grad_u_21``, ``grad_u_33``.  Any component not
        provided defaults to the zero function.

        Parameters
        ----------
        **kwargs
            Mapping from component name to either:

            * A callable ``f(t) -> scalar`` – used directly.
            * A numeric scalar – wrapped as a constant function.

        Returns
        -------
        VelocityGradient

        Raises
        ------
        ValueError
            If an unrecognised component name is provided.

        Examples
        --------
        >>> L = VelocityGradient.from_components(grad_u_12=lambda t: jnp.sin(t))
        """
        component_list = [f"grad_u_{i}{j}" for i in range(1, 4) for j in range(1, 4)]
        components = {}

        class _Gradient_Component_Container(_Component_Container):
            grad_u_11: Callable
            grad_u_12: Callable
            grad_u_13: Callable
            grad_u_21: Callable
            grad_u_22: Callable
            grad_u_23: Callable
            grad_u_31: Callable
            grad_u_32: Callable
            grad_u_33: Callable

            def get(self, t: float | jax.Array) -> jax.Array:
                return jnp.array([
                    [self.grad_u_11(t), self.grad_u_12(t), self.grad_u_13(t)],
                    [self.grad_u_21(t), self.grad_u_22(t), self.grad_u_23(t)],
                    [self.grad_u_31(t), self.grad_u_32(t), self.grad_u_33(t)]
                ])

        for key, value in kwargs.items():
            if key not in component_list:
                raise ValueError(f"Invalid velocity gradient component {key}")

            if callable(value):
                components[key] = value
            else:
                components[key] = lambda t, val=value: jnp.asarray(val)
        for key in component_list:
            if key not in components:
                components[key] = lambda t: jnp.asarray(0.0)

        forcing_fn = _Gradient_Component_Container(**components)

        return cls(forcing_fn)

    def gradient(self, t: float | jax.Array) -> jax.Array:
        """Evaluate the 3×3 velocity gradient tensor L at time ``t``.

        Automatically vmaps over an array ``t`` so the call signature is
        uniform for both scalar and batched inputs.

        Parameters
        ----------
        t : float | jax.Array
            Evaluation time(s).  May be a scalar or a 1-D array.

        Returns
        -------
        jax.Array
            Shape ``(3, 3)`` for scalar ``t``, or ``(T, 3, 3)`` for a 1-D
            array of ``T`` time points.
        """
        if isinstance(t, jnp.ndarray) and t.ndim >= 1:
            return jax.vmap(self.gradient)(t)
        return self.forcing_fn.get(t)

    def rate_of_strain(self, t: float | jax.Array) -> jax.Array:
        """Evaluate the rate-of-strain tensor D(t) = 0.5 * (L + Lᵀ) at time ``t``.

        Automatically vmaps over an array ``t``.

        Parameters
        ----------
        t : float | jax.Array
            Evaluation time(s).

        Returns
        -------
        jax.Array
            Shape ``(3, 3)`` for scalar ``t``, or ``(T, 3, 3)`` for a 1-D
            time array.  The result is always symmetric.
        """
        if isinstance(t, jnp.ndarray) and t.ndim >= 1:
            return jax.vmap(self.rate_of_strain)(t)

        grad_u = self.gradient(t)
        return 0.5 * (grad_u + grad_u.T)


class AppliedStress(AbstractForcing):
    """A time-dependent symmetric 3×3 stress tensor σ(t) applied externally.

    Used to drive stress-controlled experiments (e.g. creep tests).  Only
    the six independent components of the symmetric stress tensor need to
    be specified: ``sigma_11``, ``sigma_12``, ``sigma_13``, ``sigma_22``,
    ``sigma_23``, ``sigma_33``.  Unspecified components default to zero.

    Parameters
    ----------
    forcing_fn : _Component_Container
        Internal container storing each ``sigma_ij`` callable.

    Examples
    --------
    Constant shear stress of 1.0 Pa::

        sigma = AppliedStress.from_components(sigma_12=1.0)
        sigma_matrix = sigma.stress(t=0.0)  # shape (3, 3)

    Step stress (applied after t = 1)::

        sigma = AppliedStress.from_components(
            sigma_12=lambda t: jnp.where(t > 1.0, 1.0, 0.0)
        )
    """

    forcing_fn: _Component_Container

    def __init__(self, forcing_fn: _Component_Container):
        """Construct directly from a pre-built component container."""
        self.forcing_fn = forcing_fn

    @classmethod
    def from_components(cls, **kwargs):
        """Build an :class:`AppliedStress` from named symmetric components.

        Valid keyword arguments are ``sigma_ij`` for the upper-triangular
        entries of the symmetric stress tensor: ``sigma_11``, ``sigma_12``,
        ``sigma_13``, ``sigma_22``, ``sigma_23``, ``sigma_33``.  The
        (2,1), (3,1), and (3,2) components are automatically set equal to
        their symmetric counterparts.

        Parameters
        ----------
        **kwargs
            Mapping from component name to callable or scalar value.

        Returns
        -------
        AppliedStress

        Raises
        ------
        ValueError
            If an unrecognised component name is provided.
        """
        component_list = [f"sigma_{i}{j}" for i in range(1, 4) for j in range(1, 4) if i <= j]
        components = {}

        class _Stress_Component_Container(_Component_Container):
            sigma_11: Callable
            sigma_12: Callable
            sigma_13: Callable
            sigma_22: Callable
            sigma_23: Callable
            sigma_33: Callable

            def get(self, t: float | jax.Array) -> jax.Array:
                return jnp.array([
                    [self.sigma_11(t), self.sigma_12(t), self.sigma_13(t)],
                    [self.sigma_12(t), self.sigma_22(t), self.sigma_23(t)],
                    [self.sigma_13(t), self.sigma_23(t), self.sigma_33(t)]
                ])

        for key, value in kwargs.items():
            if key not in component_list:
                raise ValueError(f"Invalid stress component {key}")

            if callable(value):
                components[key] = value
            else:
                components[key] = lambda t, val=value: jnp.asarray(val)
        for key in component_list:
            if key not in components:
                components[key] = lambda t: jnp.asarray(0.0)

        forcing_fn = _Stress_Component_Container(**components)

        return cls(forcing_fn)

    def stress(self, t: float | jax.Array) -> jax.Array:
        """Evaluate the 3×3 symmetric stress tensor σ at time ``t``.

        Automatically vmaps over an array ``t``.

        Parameters
        ----------
        t : float | jax.Array
            Evaluation time(s).

        Returns
        -------
        jax.Array
            Shape ``(3, 3)`` for scalar ``t``, or ``(T, 3, 3)`` for a 1-D
            array of ``T`` time points.  The result is always symmetric.
        """
        if isinstance(t, jnp.ndarray) and t.ndim >= 1:
            return jax.vmap(self.stress)(t)
        return self.forcing_fn.get(t)


# ---------------------------------------------------------------------------
# Extensional kinematics
# ---------------------------------------------------------------------------
#
# Shear is only one of the canonical homogeneous flows.  Extensional flows have
# a *diagonal* velocity gradient and stretch material lines exponentially, so
# they probe a constitutive model where simple shear is blind: models that are
# nearly degenerate in σ₁₂ (Oldroyd-B vs FENE-P / PTT / Giesekus) diverge
# dramatically in their extensional viscosity.  Oldroyd-B famously has an
# *unbounded* steady extensional viscosity that blows up as the Weissenberg
# number ε̇·λ → 1/2, whereas every finite-extensibility / network model caps it.
#
# Each helper builds a :class:`VelocityGradient` whose diagonal is the
# traceless (incompressible) extension tensor for a prescribed extension rate
# ε̇(t).  The :class:`ViscoelasticStrainRateProtocol` already integrates an
# arbitrary velocity gradient, so these drop straight into the existing solver.

VALID_EXTENSION_MODES = ("uniaxial", "planar", "biaxial")


def _as_rate_fn(rate: "float | Callable") -> Callable:
    """Coerce a scalar or callable extension rate into a callable ``ε̇(t)``."""
    if callable(rate):
        return rate
    return lambda t, v=rate: jnp.asarray(v)


def extensional_forcing(rate: "float | Callable", mode: str = "uniaxial") -> VelocityGradient:
    """Build a :class:`VelocityGradient` for a homogeneous extensional flow.

    Parameters
    ----------
    rate : float | Callable
        The extension rate ε̇ (a constant) or a callable ``ε̇(t)``.
    mode : str
        One of:

        * ``"uniaxial"`` – stretch along x₁, equal compression on x₂, x₃:
          ``L = diag(ε̇, −ε̇/2, −ε̇/2)`` (fibre spinning).
        * ``"planar"`` – stretch along x₁, compress along x₃, x₂ neutral:
          ``L = diag(ε̇, 0, −ε̇)`` (film/sheet drawing).
        * ``"biaxial"`` – equal stretch along x₁, x₂, compression on x₃:
          ``L = diag(ε̇, ε̇, −2ε̇)`` (bubble/blow moulding).

    Returns
    -------
    VelocityGradient
        A traceless (volume-preserving) diagonal velocity gradient.
    """
    rate_fn = _as_rate_fn(rate)
    if mode == "uniaxial":
        return VelocityGradient.from_components(
            grad_u_11=lambda t: rate_fn(t),
            grad_u_22=lambda t: -0.5 * rate_fn(t),
            grad_u_33=lambda t: -0.5 * rate_fn(t),
        )
    elif mode == "planar":
        return VelocityGradient.from_components(
            grad_u_11=lambda t: rate_fn(t),
            grad_u_33=lambda t: -rate_fn(t),
        )
    elif mode == "biaxial":
        return VelocityGradient.from_components(
            grad_u_11=lambda t: rate_fn(t),
            grad_u_22=lambda t: rate_fn(t),
            grad_u_33=lambda t: -2.0 * rate_fn(t),
        )
    raise ValueError(
        f"Unknown extension mode {mode!r}; expected one of {VALID_EXTENSION_MODES}."
    )


def uniaxial_extension(rate: "float | Callable") -> VelocityGradient:
    """Uniaxial extensional flow ``L = diag(ε̇, −ε̇/2, −ε̇/2)``."""
    return extensional_forcing(rate, mode="uniaxial")


def planar_extension(rate: "float | Callable") -> VelocityGradient:
    """Planar extensional flow ``L = diag(ε̇, 0, −ε̇)``."""
    return extensional_forcing(rate, mode="planar")


def biaxial_extension(rate: "float | Callable") -> VelocityGradient:
    """Biaxial extensional flow ``L = diag(ε̇, ε̇, −2ε̇)``."""
    return extensional_forcing(rate, mode="biaxial")

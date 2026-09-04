"""
Generalized Newtonian fluid models.

These models compute stress as an instantaneous algebraic function of the
rate-of-strain tensor; no ODE integration is required.

Available models
----------------
* :class:`Newtonian` – constant viscosity.
* :class:`CarreauYasuda` – shear-thinning / thickening with five parameters.
* :class:`PowerLaw` – simple power-law viscosity.

All models inherit from
:class:`~diff_rheo.models._constitutive_model.AbstractGeneralizedNewtonianModel`
and accept parameter values as plain floats (automatically wrapped in
:class:`~diff_rheo.parameters.LogParameter`) or as explicit
:class:`~diff_rheo.parameters.AbstractParameter` instances.
"""

from typing import Union
import jax
import jax.numpy as jnp
from ._constitutive_model import AbstractGeneralizedNewtonianModel
from ..parameters import AbstractParameter
from .._forcing import VelocityGradient, AppliedStress
from .._utils import _rate_of_strain_to_strain_rate


class Newtonian(AbstractGeneralizedNewtonianModel):
    """Newtonian fluid with constant viscosity.

    Constitutive equation: σ = 2 η D.

    Parameters
    ----------
    viscosity : float | AbstractParameter
        Dynamic viscosity η (Pa·s).
    """

    viscosity: AbstractParameter

    def stress_response(
        self,
        t: Union[float, jax.Array],
        velocity_gradient: VelocityGradient,
        *args,
        **kwargs,
    ) -> jax.Array:
        """Return σ = 2 η D at time ``t``.

        Returns
        -------
        jax.Array
            Stress tensor, shape ``(3, 3)``.
        """
        return 2 * self.viscosity.get_value() * velocity_gradient.rate_of_strain(t)

    def strain_response(
        self,
        t: Union[float, jax.Array],
        applied_stress: AppliedStress,
        *args,
        **kwargs,
    ) -> jax.Array:
        """Return D = σ / (2 η) at time ``t``.

        Returns
        -------
        jax.Array
            Rate-of-strain tensor, shape ``(3, 3)``.
        """
        return applied_stress.stress(t) / (2 * self.viscosity.get_value())


class CarreauYasuda(AbstractGeneralizedNewtonianModel):
    """Carreau-Yasuda generalized Newtonian model.

    Effective viscosity:

        η(γ̇) = η_∞ + (η₀ - η_∞) · [1 + (k|γ̇|)ᵃ]^((n-1)/a)

    Parameters
    ----------
    zero_shear_viscosity : float | AbstractParameter
        η₀ – zero-shear-rate viscosity (Pa·s).
    infinite_shear_viscosity : float | AbstractParameter
        η_∞ – infinite-shear-rate viscosity (Pa·s).
    n : float | AbstractParameter
        Power-law index (n < 1 → shear-thinning).
    a : float | AbstractParameter
        Yasuda parameter controlling transition breadth.
    k : float | AbstractParameter
        Time constant (s); sets the onset strain rate.

    Notes
    -----
    :meth:`strain_response` is not implemented (inversion is not analytically tractable).
    """

    zero_shear_viscosity: AbstractParameter
    infinite_shear_viscosity: AbstractParameter
    n: AbstractParameter
    a: AbstractParameter
    k: AbstractParameter

    def stress_response(
        self,
        t: Union[float, jax.Array],
        velocity_gradient: VelocityGradient,
        *args,
        **kwargs,
    ) -> jax.Array:
        """Return σ = 2 η(γ̇) D at time ``t``.

        Returns
        -------
        jax.Array
            Stress tensor, shape ``(3, 3)``.
        """
        rate_of_strain = velocity_gradient.rate_of_strain(t)
        strain_rate = _rate_of_strain_to_strain_rate(rate_of_strain)
        effective_viscosity = self.effective_viscosity(strain_rate)
        return 2 * effective_viscosity * rate_of_strain

    def effective_viscosity(self, strain_rate: jax.Array) -> jax.Array:
        """Compute η(γ̇) via the Carreau-Yasuda equation.

        Parameters
        ----------
        strain_rate : jax.Array
            Scalar strain rate γ̇ = √(2 D:D).

        Returns
        -------
        jax.Array
            Effective viscosity (Pa·s).
        """
        zero_shear_viscosity = self.zero_shear_viscosity.get_value()
        infinite_shear_viscosity = self.infinite_shear_viscosity.get_value()
        n = self.n.get_value()
        a = self.a.get_value()
        k = self.k.get_value()
        inner_base = k * jnp.abs(strain_rate) + 1e-8

        term = jnp.exp(((n - 1.0) / a) * jnp.log1p(jnp.power(inner_base, a)))

        return infinite_shear_viscosity + (zero_shear_viscosity - infinite_shear_viscosity) * term

    def strain_response(
        self,
        t: Union[float, jax.Array],
        applied_stress: AppliedStress,
        *args,
        **kwargs,
    ) -> jax.Array:
        raise NotImplementedError("Strain response not implemented for CarreauYasuda model")


class PowerLaw(AbstractGeneralizedNewtonianModel):
    """Power-law generalized Newtonian model.

    Effective viscosity: η(γ̇) = η₀ · |γ̇|^(n-1).

    Stress: σ = 2 η₀ |γ̇|^(n-1) D.

    .. warning::
        Viscosity diverges as γ̇ → 0 for n < 1.  Consider :class:`CarreauYasuda`
        if a low-shear plateau is needed.

    Parameters
    ----------
    n : float | AbstractParameter
        Power-law index.
    zero_shear_viscosity : float | AbstractParameter
        Consistency coefficient η₀ (Pa·sⁿ).

    Notes
    -----
    :meth:`strain_response` is not implemented.
    """

    n: AbstractParameter
    zero_shear_viscosity: AbstractParameter

    def stress_response(
        self,
        t: Union[float, jax.Array],
        velocity_gradient: VelocityGradient,
        *args,
        **kwargs,
    ) -> jax.Array:
        """Return σ = 2 η₀ |γ̇|^(n-1) D at time ``t``.

        Returns
        -------
        jax.Array
            Stress tensor, shape ``(3, 3)``.
        """
        zero_shear_viscosity = self.zero_shear_viscosity.get_value()
        n = self.n.get_value()
        rate_of_strain = velocity_gradient.rate_of_strain(t)
        strain_rate = _rate_of_strain_to_strain_rate(rate_of_strain)
        effective_viscosity = zero_shear_viscosity * jnp.power(strain_rate, n - 1)
        return 2 * effective_viscosity * rate_of_strain

    def strain_response(
        self,
        t: Union[float, jax.Array],
        applied_stress: AppliedStress,
        *args,
        **kwargs,
    ) -> jax.Array:
        raise NotImplementedError("Strain response not implemented for PowerLaw model")
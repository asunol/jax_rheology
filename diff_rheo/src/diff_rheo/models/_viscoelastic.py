"""
Viscoelastic constitutive models.

All models subclass
:class:`~diff_rheo.models._constitutive_model.AbstractViscoelasticModel` and
require ODE integration.

Model hierarchy
---------------
Most models extend :class:`OldroydB` by overriding:

* :meth:`~OldroydB._non_linear_term` – additional nonlinear stress contribution.
* :meth:`~OldroydB.effective_polymer_viscosity` / :meth:`~OldroydB.effective_relaxation_time` –
  strain-rate dependent material functions.

Available models
----------------
* :class:`OldroydB` – Upper-Convected Maxwell / Oldroyd-B, 3 parameters.
* :class:`GeneralizedOldroydB` – Oldroyd-B with a user-supplied nonlinear F function.
* :class:`Giesekus` – quadratic stress anisotropy, 4 parameters.
* :class:`LinearPTT` – linear Phan-Thien-Tanner, 5 parameters.
* :class:`ExponentialPTT` – exponential PTT, 5 parameters.
* :class:`GeneralizedPTT` – PTT with Mittag-Leffler extensional function, 7 parameters.
* :class:`FENECR` – Finitely Extensible Nonlinear Elastic (FENE-CR), 4 parameters.
* :class:`FENEP` – FENE-P dumbbell model, 4 parameters.
* :class:`WhiteMetzner` – Oldroyd-B with rate-dependent viscosity/relaxation, 9 parameters.
* :class:`XPomPom` – eXtended Pom-Pom for branched polymer melts, 7 parameters.
"""

from typing import Union, Callable
import jax
import jax.numpy as jnp
import equinox as eqx

from ._constitutive_model import AbstractViscoelasticModel
from ..parameters import AbstractParameter
from .._forcing import VelocityGradient, AppliedStress
from .._utils import _rate_of_strain_to_strain_rate, _flatten_symmetric_array, _vector_to_symmetric_matrix, _generalized_mittag_leffler_function


class OldroydB(AbstractViscoelasticModel):
    """Oldroyd-B (Upper-Convected Maxwell) viscoelastic model.

    The constitutive equation for the extra (polymer) stress τ is:

        λ ∇τ + τ = 2 η_p D

    where ∇ denotes the upper-convected derivative:

        ∇τ = dτ/dt - L·τ - τ·Lᵀ

    The total stress is σ = 2 η_s D + τ.

    This is the simplest linear viscoelastic model that captures elastic
    normal stresses in shear.  It predicts a constant shear viscosity
    (η₀ = η_s + η_p) and a first normal stress difference N₁ = 2 η_p λ γ̇².

    Parameters
    ----------
    polymer_viscosity : float | AbstractParameter
        Polymer (extra) viscosity η_p (Pa·s).
    relaxation_time : float | AbstractParameter
        Relaxation time λ (s).
    solvent_viscosity : float | AbstractParameter
        Newtonian solvent viscosity η_s (Pa·s).

    Notes
    -----
    The Oldroyd-B model reduces to the Upper-Convected Maxwell (UCM) model
    when ``solvent_viscosity = 0``.

    Most other viscoelastic models in this module (Giesekus, PTT, etc.)
    extend :class:`OldroydB` by overriding :meth:`_non_linear_term`.
    """

    polymer_viscosity: AbstractParameter
    relaxation_time: AbstractParameter
    solvent_viscosity: AbstractParameter

    def effective_polymer_viscosity(self, strain_rate: jax.Array) -> jax.Array:
        """Return the effective polymer viscosity (constant for Oldroyd-B).

        Overridden by :class:`WhiteMetzner` to provide strain-rate dependence.
        """
        return self.polymer_viscosity.get_value()

    def effective_relaxation_time(self, strain_rate: jax.Array) -> jax.Array:
        """Return the effective relaxation time (constant for Oldroyd-B).

        Overridden by :class:`WhiteMetzner` to provide strain-rate dependence.
        """
        return self.relaxation_time.get_value()

    def _non_linear_term(self, stress: jax.Array, rate_of_strain: jax.Array) -> jax.Array:
        """Return the nonlinear stress correction term (zero for Oldroyd-B).

        Overridden by :class:`Giesekus`, :class:`LinearPTT`, :class:`ExponentialPTT`,
        :class:`GeneralizedPTT`, and :class:`GeneralizedOldroydB` to add
        model-specific nonlinear contributions to the stress evolution ODE.

        Parameters
        ----------
        stress : jax.Array
            Current extra-stress tensor τ, shape ``(3, 3)``.
        rate_of_strain : jax.Array
            Rate-of-strain tensor D, shape ``(3, 3)``.

        Returns
        -------
        jax.Array
            Nonlinear correction tensor, shape ``(3, 3)``.
        """
        return jnp.zeros_like(stress)

    @eqx.filter_jit
    def _calculate_polymer_stress_rate_tensor(
        self,
        stress: jax.Array,
        u_grad: jax.Array,
        rate_of_strain: jax.Array,
    ) -> jax.Array:
        """Compute the full extra-stress ODE RHS for the UCM/Oldroyd-B framework.

        Implements:

            dτ/dt = (2 η_p(γ̇) D - τ - F(τ, D)) / λ(γ̇) + τ·L + Lᵀ·τ

        where the upper-convected terms ``τ·L + Lᵀ·τ`` account for the
        frame-objective (convected) derivative, and F(τ, D) is the nonlinear
        correction from :meth:`_non_linear_term`.

        Parameters
        ----------
        stress : jax.Array
            Current extra-stress τ, shape ``(3, 3)``.
        u_grad : jax.Array
            Velocity gradient L, shape ``(3, 3)``.
        rate_of_strain : jax.Array
            Rate-of-strain D, shape ``(3, 3)``.

        Returns
        -------
        jax.Array
            dτ/dt, shape ``(3, 3)``.
        """
        strain_rate = _rate_of_strain_to_strain_rate(rate_of_strain)
        polymer_viscosity = self.effective_polymer_viscosity(strain_rate)
        relaxation_time = self.effective_relaxation_time(strain_rate)
        non_linear_term = self._non_linear_term(stress, rate_of_strain)
        return (2*polymer_viscosity*rate_of_strain - stress - non_linear_term)/relaxation_time + u_grad @ stress + stress @ u_grad.T

    def _calculate_du12_dt(
        self,
        stress: jax.Array,
        rate_of_strain: jax.Array,
        ds12: jax.Array,
    ) -> jax.Array:
        """Compute the rate of change of the shear rate γ̇ in a stress-controlled experiment.

        Derived from the analytical inversion of the Oldroyd-B ODE for the
        shear component under prescribed σ₁₂(t).

        Parameters
        ----------
        stress : jax.Array
            Total stress tensor, shape ``(3, 3)``.
        rate_of_strain : jax.Array
            Rate-of-strain tensor D (encodes the current γ̇ via its (1,2) component).
        ds12 : jax.Array
            Time derivative of the applied σ₁₂.

        Returns
        -------
        jax.Array
            dγ̇/dt (scalar).
        """
        # |gammadot| (a non-negative invariant) drives the rate-dependent
        # material functions; the SIGNED shear rate (2*D12) is what enters the
        # algebraic stress inversion below.  Using the magnitude in the inversion
        # flips the damping sign when gammadot < 0 and destabilises the solve.
        gammadot = _rate_of_strain_to_strain_rate(rate_of_strain)
        u12 = 2.0 * rate_of_strain[0, 1]
        relaxation_time = self.effective_relaxation_time(gammadot)
        polymer_viscosity = self.effective_polymer_viscosity(gammadot)
        solvent_viscosity = self.solvent_viscosity.get_value()

        stress = stress[jnp.array([1, 0, 2])][:, jnp.array([1, 0, 2])]  # UCM fix: 1<->2 axis swap
        non_linear_term = self._non_linear_term(stress, rate_of_strain)

        s11, _, _, s12, _, _ = _flatten_symmetric_array(stress)
        _, _, _, nl12, _, _ = _flatten_symmetric_array(non_linear_term)

        du12=(nl12+ds12*relaxation_time+s12-(polymer_viscosity+relaxation_time*s11+solvent_viscosity)*u12)/(relaxation_time*solvent_viscosity)

        return du12

    @eqx.filter_jit
    def extra_stress_response_rhs(
        self,
        t: Union[float, jax.Array],
        stress: jax.Array,
        velocity_gradient: VelocityGradient,
        *args,
        **kwargs,
    ) -> jax.Array:
        """ODE RHS for the extra-stress in a strain-rate controlled experiment.

        Parameters
        ----------
        t : float | jax.Array
            Current time.
        stress : jax.Array
            Current extra-stress τ, shape ``(3, 3)``.
        velocity_gradient : VelocityGradient
            Prescribed velocity gradient L(t).

        Returns
        -------
        jax.Array
            dτ/dt, shape ``(3, 3)``.
        """
        u_grad = velocity_gradient.gradient(t)
        rate_of_strain = velocity_gradient.rate_of_strain(t)
        dtau_dt = self._calculate_polymer_stress_rate_tensor(stress, u_grad, rate_of_strain)
        return dtau_dt

    @eqx.filter_jit
    def shear_stress_experiment_rhs(
        self,
        t: Union[float, jax.Array],
        current_values: jax.Array,
        applied_stress: AppliedStress,
        *args,
        **kwargs,
    ) -> jax.Array:
        """ODE RHS for the combined state in a stress-controlled experiment.

        State vector: ``[σ₁₁, σ₂₂, σ₃₃, σ₁₃, σ₂₃, γ̇, γ]``.
        σ₁₂ is read from ``applied_stress`` at time ``t`` and is not part
        of the integrated state.

        Parameters
        ----------
        t : float | jax.Array
            Current time.
        current_values : jax.Array
            State vector, shape ``(7,)``.
        applied_stress : AppliedStress
            Prescribed σ₁₂(t).

        Returns
        -------
        jax.Array
            d(state)/dt, shape ``(7,)``.
        """
        s11, s22, s33, s13, s23, u12, _ = current_values
        s12 = applied_stress.stress(t)[0,1]
        ds12 = jax.jacobian(applied_stress.stress)(t)[0,1]

        total_stress = _vector_to_symmetric_matrix(jnp.array([s11,s22,s33,s12,s13,s23]))
        u_grad = jnp.array([[0.0, u12, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
        rate_of_strain = _vector_to_symmetric_matrix(jnp.array([0.0, 0.0, 0.0, u12/2, 0.0, 0.0]))

        solvent_viscosity = self.solvent_viscosity.get_value()
        polymer_stress = total_stress - solvent_viscosity * _vector_to_symmetric_matrix(jnp.array([0.0, 0.0, 0.0, u12, 0.0, 0.0]))

        du12 = self._calculate_du12_dt(total_stress, rate_of_strain, ds12)

        dsigma = self._calculate_polymer_stress_rate_tensor(polymer_stress, u_grad, rate_of_strain)

        ds11, ds22, ds33, _, ds13, ds23 = _flatten_symmetric_array(dsigma)

        return jnp.array([ds11, ds22, ds33, ds13, ds23, du12, u12]).reshape((7,))


class GeneralizedOldroydB(OldroydB):
    """Oldroyd-B model with a user-supplied nonlinear F function.

    Extends the standard Oldroyd-B ODE by replacing the zero nonlinear term
    with a callable ``F_function(stress, rate_of_strain) -> correction_tensor``.
    This provides a generic framework for data-driven or custom constitutive
    models.

    The :class:`~diff_rheo.models._rude.RUDE` neural network model uses this
    class as its backbone, with the RUDE network providing F.

    Parameters
    ----------
    polymer_viscosity : float | AbstractParameter
        Polymer viscosity η_p (Pa·s).
    relaxation_time : float | AbstractParameter
        Relaxation time λ (s).
    solvent_viscosity : float | AbstractParameter
        Solvent viscosity η_s (Pa·s).
    F_function : Callable[[jax.Array, jax.Array], jax.Array]
        A function ``F(τ, D) -> correction`` that returns the nonlinear
        stress correction tensor with shape ``(3, 3)``.  Must be JAX-
        compatible (JIT-compilable, differentiable).
    """

    polymer_viscosity: AbstractParameter
    relaxation_time: AbstractParameter
    solvent_viscosity: AbstractParameter
    F_function: Callable

    @eqx.filter_jit
    def _non_linear_term(self, stress: jax.Array, rate_of_strain: jax.Array) -> jax.Array:
        """Evaluate the nonlinear correction F(τ, D) using the stored callable."""
        F = self.F_function(stress, rate_of_strain)
        return F
        
class WhiteMetzner(OldroydB):
    """White-Metzner viscoelastic model with strain-rate dependent material functions.

    Generalises Oldroyd-B by making both the polymer viscosity and relaxation
    time functions of the local strain rate γ̇:

        η_p(γ̇) = η_p0 · [1 + (K γ̇)^a]^((n-1)/a)
        λ(γ̇)   = λ₀  · [1 + (L γ̇)^b]^((m-1)/b)

    These functional forms are Power-Law / Carreau-type expressions, enabling
    the model to match shear-thinning viscosity and relaxation time data.

    Parameters
    ----------
    polymer_viscosity : float | AbstractParameter
        Reference polymer viscosity η_p0 (Pa·s).
    relaxation_time : float | AbstractParameter
        Reference relaxation time λ₀ (s).
    solvent_viscosity : float | AbstractParameter
        Solvent viscosity η_s (Pa·s).
    K : float | AbstractParameter
        Viscosity time constant (s).
    n : float | AbstractParameter
        Viscosity power-law index.
    a : float | AbstractParameter
        Viscosity Yasuda parameter.
    L : float | AbstractParameter
        Relaxation time constant (s).
    m : float | AbstractParameter
        Relaxation time power-law index.
    b : float | AbstractParameter
        Relaxation time Yasuda parameter.

    References
    ----------
    White, J.L. & Metzner, A.B. (1963). J. Appl. Polym. Sci., 7(5), 1867-1889.
    """

    polymer_viscosity: AbstractParameter
    relaxation_time: AbstractParameter
    solvent_viscosity: AbstractParameter
    K: AbstractParameter
    L: AbstractParameter
    n: AbstractParameter
    m: AbstractParameter
    a: AbstractParameter
    b: AbstractParameter

    def effective_polymer_viscosity(self, strain_rate: jax.Array) -> jax.Array:
        K = self.K.get_value()
        n = self.n.get_value()
        a = self.a.get_value()
        return self.polymer_viscosity.get_value() * (1 + (K * strain_rate) ** a) ** ((n - 1) / a)
    
    def effective_relaxation_time(self, strain_rate: jax.Array) -> jax.Array:
        L = self.L.get_value()
        m = self.m.get_value()
        b = self.b.get_value()
        return self.relaxation_time.get_value() * (1 + (L * strain_rate) ** b) ** ((m - 1) / b)

class Giesekus(OldroydB):
    """Giesekus viscoelastic model with quadratic stress nonlinearity.

    Extends Oldroyd-B by adding an anisotropic drag (mobility) term:

        F(τ, D) = α λ / η_p · τ·τ

    so the extra-stress ODE becomes:

        λ ∇τ + (1 + α λ / η_p · τ) · τ = 2 η_p D

    This introduces shear-rate dependent viscosity and normal stress
    differences that are more realistic than Oldroyd-B.

    Parameters
    ----------
    polymer_viscosity : float | AbstractParameter
        Polymer viscosity η_p (Pa·s).
    relaxation_time : float | AbstractParameter
        Relaxation time λ (s).
    solvent_viscosity : float | AbstractParameter
        Solvent viscosity η_s (Pa·s).
    alpha : float | AbstractParameter
        Mobility / anisotropy parameter α ∈ (0, 1].  α = 0 recovers
        Oldroyd-B; larger α increases nonlinearity.

    References
    ----------
    Giesekus, H. (1982). J. Non-Newtonian Fluid Mech., 11(1-2), 69-109.
    """

    polymer_viscosity: AbstractParameter
    relaxation_time: AbstractParameter
    solvent_viscosity: AbstractParameter
    alpha: AbstractParameter

    def _non_linear_term(self, stress: jax.Array, rate_of_strain: jax.Array) -> jax.Array:
        alpha = self.alpha.get_value()
        strain_rate = _rate_of_strain_to_strain_rate(rate_of_strain)
        polymer_viscosity = self.effective_polymer_viscosity(strain_rate)
        relaxation_time = self.effective_relaxation_time(strain_rate)

        return alpha * relaxation_time/polymer_viscosity * stress@stress

    def _calculate_du12_dt(self, stress: jax.Array, rate_of_strain: jax.Array, ds12: jax.Array) -> jax.Array:
        stress = stress[jnp.array([1, 0, 2])][:, jnp.array([1, 0, 2])]  # UCM fix: 1<->2 axis swap
        gammadot = _rate_of_strain_to_strain_rate(rate_of_strain)
        u12 = 2.0 * rate_of_strain[0, 1]
        relaxation_time = self.effective_relaxation_time(gammadot)
        polymer_viscosity = self.effective_polymer_viscosity(gammadot)
        solvent_viscosity = self.solvent_viscosity.get_value()
        alpha = self.alpha.get_value()

        s11, s22, _, s12, s13, s23 = _flatten_symmetric_array(stress)

        du12=(ds12*polymer_viscosity*relaxation_time+s12*(polymer_viscosity+alpha*relaxation_time*(s11+s22))+alpha*relaxation_time*s13*s23-(polymer_viscosity**2+alpha*relaxation_time*(s11+s22)*solvent_viscosity+polymer_viscosity*(relaxation_time*s11+solvent_viscosity))*u12)/(polymer_viscosity*relaxation_time*solvent_viscosity)

        return du12

class LinearPTT(OldroydB):
    """Linear Phan-Thien-Tanner viscoelastic model.

    Extends Oldroyd-B with two additional terms in the stress ODE that
    capture the behaviour of entangled polymer networks:

        F(τ, D) = ζ λ (τ·D + D·τ) + ε λ / η_p · tr(τ) · τ

    The ε term (linear PTT function) relates to extensional behaviour;
    the ζ term (Gordon-Schowalter slip) affects both shear and extension.

    Parameters
    ----------
    polymer_viscosity : float | AbstractParameter
        η_p (Pa·s).
    relaxation_time : float | AbstractParameter
        λ (s).
    solvent_viscosity : float | AbstractParameter
        η_s (Pa·s).
    epsilon : float | AbstractParameter
        Extensional parameter ε (controls shear-thinning / extension hardening).
    zeta : float | AbstractParameter
        Slip parameter ζ (0 ≤ ζ ≤ 2).  ζ = 0 → upper-convected; ζ = 1 → corotational.

    References
    ----------
    Phan-Thien, N. & Tanner, R.I. (1977). J. Non-Newtonian Fluid Mech., 2(4), 353-365.
    """

    polymer_viscosity: AbstractParameter
    relaxation_time: AbstractParameter
    solvent_viscosity: AbstractParameter
    epsilon: AbstractParameter
    zeta: AbstractParameter

    def _non_linear_term(self, stress: jax.Array, rate_of_strain: jax.Array) -> jax.Array:
        epsilon = self.epsilon.get_value()
        zeta = self.zeta.get_value()
        strain_rate = _rate_of_strain_to_strain_rate(rate_of_strain)
        polymer_viscosity = self.effective_polymer_viscosity(strain_rate)
        relaxation_time = self.effective_relaxation_time(strain_rate)

        return zeta*relaxation_time*(stress@rate_of_strain + rate_of_strain@stress) + epsilon*relaxation_time/polymer_viscosity * jnp.trace(stress)*stress
    
    def _calculate_du12_dt(self, stress: jax.Array, rate_of_strain: jax.Array, ds12: jax.Array) -> jax.Array:
        stress = stress[jnp.array([1, 0, 2])][:, jnp.array([1, 0, 2])]  # UCM fix: 1<->2 axis swap
        gammadot = _rate_of_strain_to_strain_rate(rate_of_strain)
        u12 = 2.0 * rate_of_strain[0, 1]
        relaxation_time = self.effective_relaxation_time(gammadot)
        polymer_viscosity = self.effective_polymer_viscosity(gammadot)
        solvent_viscosity = self.solvent_viscosity.get_value()
        epsilon = self.epsilon.get_value()
        zeta = self.zeta.get_value()

        s11, s22, s33, s12, _, _ = _flatten_symmetric_array(stress)
        du12=(2*ds12*polymer_viscosity*relaxation_time+2*s12*(polymer_viscosity+epsilon*relaxation_time*(s11+s22+s33))-2*(polymer_viscosity**2+epsilon*relaxation_time*(s11+s22+s33)*solvent_viscosity+polymer_viscosity*(relaxation_time*s11+solvent_viscosity))*u12+polymer_viscosity*relaxation_time*(s11+s22)*u12*zeta)/(2*polymer_viscosity*relaxation_time*solvent_viscosity)
        return du12

class ExponentialPTT(OldroydB):
    """Exponential Phan-Thien-Tanner viscoelastic model.

    Uses an exponential PTT extensional function instead of the linear one:

        f_PTT = exp(ε λ / η_p · tr(τ))

    The nonlinear correction becomes:

        F(τ, D) = ζ λ (τ·D + D·τ) + (f_PTT - 1) · τ

    The exponential form provides stronger strain hardening in extensional
    flows compared to :class:`LinearPTT`.

    Parameters
    ----------
    polymer_viscosity : float | AbstractParameter
        η_p (Pa·s).
    relaxation_time : float | AbstractParameter
        λ (s).
    solvent_viscosity : float | AbstractParameter
        η_s (Pa·s).
    epsilon : float | AbstractParameter
        Extensional parameter ε.
    zeta : float | AbstractParameter
        Slip parameter ζ.

    References
    ----------
    Phan-Thien, N. (1978). J. Rheol., 22(3), 259-283.
    """

    polymer_viscosity: AbstractParameter
    relaxation_time: AbstractParameter
    solvent_viscosity: AbstractParameter
    epsilon: AbstractParameter
    zeta: AbstractParameter

    def _non_linear_term(self, stress: jax.Array, rate_of_strain: jax.Array) -> jax.Array:
        epsilon = self.epsilon.get_value()
        zeta = self.zeta.get_value()
        strain_rate = _rate_of_strain_to_strain_rate(rate_of_strain)
        polymer_viscosity = self.effective_polymer_viscosity(strain_rate)
        relaxation_time = self.effective_relaxation_time(strain_rate)

        exp_term = jnp.exp(epsilon*relaxation_time/polymer_viscosity * jnp.trace(stress))
        return zeta*relaxation_time*(stress@rate_of_strain + rate_of_strain@stress) + (exp_term-1)*stress

    def _calculate_du12_dt(self, stress: jax.Array, rate_of_strain: jax.Array, ds12: jax.Array) -> jax.Array:
        stress = stress[jnp.array([1, 0, 2])][:, jnp.array([1, 0, 2])]  # UCM fix: 1<->2 axis swap
        gammadot = _rate_of_strain_to_strain_rate(rate_of_strain)
        u12 = 2.0 * rate_of_strain[0, 1]
        relaxation_time = self.effective_relaxation_time(gammadot)
        polymer_viscosity = self.effective_polymer_viscosity(gammadot)
        solvent_viscosity = self.solvent_viscosity.get_value()
        epsilon = self.epsilon.get_value()
        zeta = self.zeta.get_value()

        s11, s22, s33, s12, _, _ = _flatten_symmetric_array(stress)
        du12=(2*ds12*relaxation_time-2*(polymer_viscosity+relaxation_time*s11)*u12+2*jnp.exp((epsilon*relaxation_time*(s11+s22+s33))/polymer_viscosity)*(s12-solvent_viscosity*u12)+relaxation_time*(s11+s22)*u12*zeta)/(2*relaxation_time*solvent_viscosity)
        return du12

class GeneralizedPTT(OldroydB):
    """Generalized Phan-Thien-Tanner model with Mittag-Leffler extensional function.

    Replaces the linear or exponential PTT extensional function with the
    generalized Mittag-Leffler function E_{α,β}:

        f_PTT = Γ(β) · E_{α,β}(ε λ / η_p · tr(τ))

    where Γ is the Euler Gamma function and E_{α,β} is the two-parameter
    Mittag-Leffler function (computed via series in
    :func:`~diff_rheo._utils._generalized_mittag_leffler_function`).

    * α = 1, β = 1 → recovers the exponential PTT.
    * α = 1, β = 1 with ε small → approaches linear PTT.

    Parameters
    ----------
    polymer_viscosity : float | AbstractParameter
        η_p (Pa·s).
    relaxation_time : float | AbstractParameter
        λ (s).
    solvent_viscosity : float | AbstractParameter
        η_s (Pa·s).
    epsilon : float | AbstractParameter
        Extensional parameter ε.
    zeta : float | AbstractParameter
        Slip parameter ζ.
    alpha : float | AbstractParameter
        Mittag-Leffler parameter α.
    beta : float | AbstractParameter
        Mittag-Leffler parameter β.

    Notes
    -----
    The Mittag-Leffler function is computed via a truncated power series
    (:func:`~diff_rheo._utils._generalized_mittag_leffler_function`).
    Convergence may be slow for large argument values.

    References
    ----------
    Ferrás, L.L. et al. (2019). J. Non-Newtonian Fluid Mech., 269, 88-99.
    """

    polymer_viscosity: AbstractParameter
    relaxation_time: AbstractParameter
    solvent_viscosity: AbstractParameter
    epsilon: AbstractParameter
    zeta: AbstractParameter
    alpha: AbstractParameter
    beta: AbstractParameter

    def _non_linear_term(self, stress: jax.Array, rate_of_strain: jax.Array) -> jax.Array:
        epsilon = self.epsilon.get_value()
        zeta = self.zeta.get_value()
        strain_rate = _rate_of_strain_to_strain_rate(rate_of_strain)
        polymer_viscosity = self.effective_polymer_viscosity(strain_rate)
        relaxation_time = self.effective_relaxation_time(strain_rate)
        alpha = self.alpha.get_value()
        beta = self.beta.get_value()

        argument = epsilon*relaxation_time/polymer_viscosity * jnp.trace(stress)
        normalization = jax.scipy.special.gamma(beta)
        mittag_leffler = _generalized_mittag_leffler_function(argument, alpha, beta)
        f_ptt = normalization * mittag_leffler

        return zeta*relaxation_time*(stress@rate_of_strain + rate_of_strain@stress) + (f_ptt-1)*stress

    def _calculate_du12_dt(self, stress: jax.Array, rate_of_strain: jax.Array, ds12: jax.Array) -> jax.Array:
        stress = stress[jnp.array([1, 0, 2])][:, jnp.array([1, 0, 2])]  # UCM fix: 1<->2 axis swap
        gammadot = _rate_of_strain_to_strain_rate(rate_of_strain)
        u12 = 2.0 * rate_of_strain[0, 1]
        relaxation_time = self.effective_relaxation_time(gammadot)
        polymer_viscosity = self.effective_polymer_viscosity(gammadot)
        solvent_viscosity = self.solvent_viscosity.get_value()
        epsilon = self.epsilon.get_value()
        zeta = self.zeta.get_value()
        alpha = self.alpha.get_value()
        beta = self.beta.get_value()

        s11, s22, s33, s12, _, _ = _flatten_symmetric_array(stress)
        argument = epsilon*relaxation_time/polymer_viscosity * jnp.trace(stress)
        normalization = jax.scipy.special.gamma(beta)
        mittag_leffler = _generalized_mittag_leffler_function(argument, alpha, beta)
        f_ptt = normalization * mittag_leffler

        du12=(2*ds12*relaxation_time+2*f_ptt*s12-2*(polymer_viscosity+relaxation_time*s11+f_ptt*solvent_viscosity)*u12+relaxation_time*(s11+s22)*u12*zeta)/(2*relaxation_time*solvent_viscosity)
        return du12

class FENECR(AbstractViscoelasticModel):
    """Finitely Extensible Nonlinear Elastic (FENE-CR) dumbbell model.

    Models polymer chains as finitely extensible dumbbells with a FENE spring
    force law.  The "CR" (Chilcott-Rallison) variant modifies the standard
    FENE-P model to give a constant viscosity in steady shear.

    The finite extensibility constraint is parameterised by ``extension_length``
    (often denoted ``b`` or ``L`` in literature): as the chain length approaches
    ``extension_length``, the spring constant diverges, preventing infinite extension.

    Parameters
    ----------
    polymer_viscosity : float | AbstractParameter
        Polymer viscosity η_p (Pa·s).
    relaxation_time : float | AbstractParameter
        Relaxation time λ (s).
    solvent_viscosity : float | AbstractParameter
        Solvent viscosity η_s (Pa·s).
    extension_length : float | AbstractParameter
        Maximum chain extensibility L (dimensionless; larger = more extensible).

    Notes
    -----
    The stress ODE components have been fully expanded to avoid redundant
    matrix operations for performance.

    References
    ----------
    Chilcott, M.D. & Rallison, J.M. (1988). J. Non-Newtonian Fluid Mech., 29, 381-432.
    """

    polymer_viscosity: AbstractParameter
    relaxation_time: AbstractParameter
    solvent_viscosity: AbstractParameter
    extension_length: AbstractParameter

    @eqx.filter_jit
    def extra_stress_response_rhs(self, t: Union[float, jax.Array], stress: jax.Array, velocity_gradient: VelocityGradient, *args, **kwargs) -> jax.Array:
        u_grad = velocity_gradient.gradient(t).T  # UCM fix: hand-coded RHS uses the conjugate convention
        u11, u12, u13, u21, u22, u23, u31, u32, u33 = u_grad.ravel()
        polymer_viscosity = self.polymer_viscosity.get_value()
        relaxation_time = self.relaxation_time.get_value()
        extension_length = self.extension_length.get_value()

        tau11, tau22, tau33, tau12, tau13, tau23 = _flatten_symmetric_array(stress)

        dtau11=(extension_length**4*polymer_viscosity**2*(tau11*(-1+2*relaxation_time*u11)+2*(polymer_viscosity*u11+relaxation_time*tau12*u21+relaxation_time*tau13*u31))-relaxation_time**2*tau11*(tau11**2+tau22**2+tau33**2-2*polymer_viscosity*tau33*u11+6*polymer_viscosity*tau12*u12+6*polymer_viscosity*tau13*u13+6*polymer_viscosity*tau12*u21-2*polymer_viscosity*tau33*u22+6*polymer_viscosity*tau23*u23+6*polymer_viscosity*tau13*u31+6*polymer_viscosity*tau23*u32+4*polymer_viscosity*tau33*u33+2*tau11*(tau22+tau33+2*polymer_viscosity*u11-polymer_viscosity*u22-polymer_viscosity*u33)+2*tau22*(tau33-polymer_viscosity*(u11-2*u22+u33)))+2*extension_length**2*polymer_viscosity*relaxation_time*(tau11**2*(-1+relaxation_time*u11)+polymer_viscosity*((tau22+tau33)*u11-3*(tau12*u21+tau13*u31))+tau11*(tau22*(-1+relaxation_time*u22)+relaxation_time*(tau12*(u12+u21)+tau13*(u13+u31)+tau23*(u23+u32))+polymer_viscosity*(-u11+u22+u33)+tau33*(-1+relaxation_time*u33))))/(extension_length**2*(-3+extension_length**2)*polymer_viscosity**2*relaxation_time)
        dtau22=(extension_length**4*polymer_viscosity**2*(tau22*(-1+2*relaxation_time*u22)+2*(relaxation_time*tau12*u12+polymer_viscosity*u22+relaxation_time*tau23*u32))-relaxation_time**2*tau22*(tau11**2+tau22**2+tau33**2-2*polymer_viscosity*tau33*u11+6*polymer_viscosity*tau12*u12+6*polymer_viscosity*tau13*u13+6*polymer_viscosity*tau12*u21-2*polymer_viscosity*tau33*u22+6*polymer_viscosity*tau23*u23+6*polymer_viscosity*tau13*u31+6*polymer_viscosity*tau23*u32+4*polymer_viscosity*tau33*u33+2*tau11*(tau22+tau33+2*polymer_viscosity*u11-polymer_viscosity*u22-polymer_viscosity*u33)+2*tau22*(tau33-polymer_viscosity*(u11-2*u22+u33)))+2*extension_length**2*polymer_viscosity*relaxation_time*(tau11*tau22*(-1+relaxation_time*u11)+polymer_viscosity*tau11*u22+tau22**2*(-1+relaxation_time*u22)+polymer_viscosity*(-3*tau12*u12+tau33*u22-3*tau23*u32)+tau22*(relaxation_time*(tau12*(u12+u21)+tau13*(u13+u31)+tau23*(u23+u32))+polymer_viscosity*(u11-u22+u33)+tau33*(-1+relaxation_time*u33))))/(extension_length**2*(-3+extension_length**2)*polymer_viscosity**2*relaxation_time)
        dtau33=(2*extension_length**2*polymer_viscosity*relaxation_time*(-(tau11*tau33)-tau22*tau33-tau33**2+polymer_viscosity*tau33*u11+relaxation_time*tau11*tau33*u11+relaxation_time*tau12*tau33*u12-3*polymer_viscosity*tau13*u13+relaxation_time*tau13*tau33*u13+relaxation_time*tau12*tau33*u21+polymer_viscosity*tau33*u22+relaxation_time*tau22*tau33*u22-3*polymer_viscosity*tau23*u23+relaxation_time*tau23*tau33*u23+relaxation_time*tau13*tau33*u31+relaxation_time*tau23*tau33*u32+polymer_viscosity*(tau11+tau22-tau33)*u33+relaxation_time*tau33**2*u33)+extension_length**4*polymer_viscosity**2*(-tau33+2*relaxation_time*tau13*u13+2*relaxation_time*tau23*u23+2*(polymer_viscosity+relaxation_time*tau33)*u33)-relaxation_time**2*tau33*(tau11**2+tau22**2+tau33**2-2*polymer_viscosity*tau33*u11+6*polymer_viscosity*tau12*u12+6*polymer_viscosity*tau13*u13+6*polymer_viscosity*tau12*u21-2*polymer_viscosity*tau33*u22+6*polymer_viscosity*tau23*u23+6*polymer_viscosity*tau13*u31+6*polymer_viscosity*tau23*u32+4*polymer_viscosity*tau33*u33+2*tau11*(tau22+tau33+2*polymer_viscosity*u11-polymer_viscosity*u22-polymer_viscosity*u33)+2*tau22*(tau33-polymer_viscosity*(u11-2*u22+u33))))/(extension_length**2*(-3+extension_length**2)*polymer_viscosity**2*relaxation_time)
        dtau12=(extension_length**4*polymer_viscosity**2*(polymer_viscosity*(u12+u21)+tau12*(-1+relaxation_time*(u11+u22))+relaxation_time*(tau11*u12+tau22*u21+tau23*u31+tau13*u32))-relaxation_time**2*tau12*(tau11**2+tau22**2+tau33**2-2*polymer_viscosity*tau33*u11+6*polymer_viscosity*tau12*u12+6*polymer_viscosity*tau13*u13+6*polymer_viscosity*tau12*u21-2*polymer_viscosity*tau33*u22+6*polymer_viscosity*tau23*u23+6*polymer_viscosity*tau13*u31+6*polymer_viscosity*tau23*u32+4*polymer_viscosity*tau33*u33+2*tau11*(tau22+tau33+2*polymer_viscosity*u11-polymer_viscosity*u22-polymer_viscosity*u33)+2*tau22*(tau33-polymer_viscosity*(u11-2*u22+u33)))+extension_length**2*polymer_viscosity*relaxation_time*(2*tau11*tau12*(-1+relaxation_time*u11)+polymer_viscosity*tau11*(-2*u12+u21)+2*relaxation_time*tau12**2*(u12+u21)+polymer_viscosity*(tau22*(u12-2*u21)+tau33*(u12+u21)-3*(tau23*u31+tau13*u32))+tau12*(2*tau22*(-1+relaxation_time*u22)+2*relaxation_time*(tau13*(u13+u31)+tau23*(u23+u32))-polymer_viscosity*(u11+u22-2*u33)+2*tau33*(-1+relaxation_time*u33))))/(extension_length**2*(-3+extension_length**2)*polymer_viscosity**2*relaxation_time)
        dtau13=(extension_length**4*polymer_viscosity**2*(polymer_viscosity*(u13+u31)+relaxation_time*(tau11*u13+tau23*u21+tau12*u23+tau33*u31)+tau13*(-1+relaxation_time*(u11+u33)))-relaxation_time**2*tau13*(tau11**2+tau22**2+tau33**2-2*polymer_viscosity*tau33*u11+6*polymer_viscosity*tau12*u12+6*polymer_viscosity*tau13*u13+6*polymer_viscosity*tau12*u21-2*polymer_viscosity*tau33*u22+6*polymer_viscosity*tau23*u23+6*polymer_viscosity*tau13*u31+6*polymer_viscosity*tau23*u32+4*polymer_viscosity*tau33*u33+2*tau11*(tau22+tau33+2*polymer_viscosity*u11-polymer_viscosity*u22-polymer_viscosity*u33)+2*tau22*(tau33-polymer_viscosity*(u11-2*u22+u33)))+extension_length**2*polymer_viscosity*relaxation_time*(2*relaxation_time*tau13**2*(u13+u31)+tau11*(2*tau13*(-1+relaxation_time*u11)+polymer_viscosity*(-2*u13+u31))+polymer_viscosity*(tau33*u13-3*tau23*u21-3*tau12*u23-2*tau33*u31+tau22*(u13+u31))+tau13*(2*tau22*(-1+relaxation_time*u22)+2*relaxation_time*(tau12*(u12+u21)+tau23*(u23+u32))-polymer_viscosity*(u11-2*u22+u33)+2*tau33*(-1+relaxation_time*u33))))/(extension_length**2*(-3+extension_length**2)*polymer_viscosity**2*relaxation_time)
        dtau23=(extension_length**2*polymer_viscosity*relaxation_time*(-2*tau23*tau33+2*polymer_viscosity*tau23*u11+2*tau11*tau23*(-1+relaxation_time*u11)-3*polymer_viscosity*tau13*u12+2*relaxation_time*tau12*tau23*u12-3*polymer_viscosity*tau12*u13+2*relaxation_time*tau13*tau23*u13+2*relaxation_time*tau12*tau23*u21-polymer_viscosity*tau23*u22+2*tau22*tau23*(-1+relaxation_time*u22)+2*relaxation_time*tau23**2*u23+polymer_viscosity*tau33*u23+2*relaxation_time*tau13*tau23*u31+2*relaxation_time*tau23**2*u32-2*polymer_viscosity*tau33*u32+polymer_viscosity*tau22*(-2*u23+u32)+polymer_viscosity*tau11*(u23+u32)-polymer_viscosity*tau23*u33+2*relaxation_time*tau23*tau33*u33)-relaxation_time**2*tau23*(tau11**2+tau22**2+tau33**2-2*polymer_viscosity*tau33*u11+6*polymer_viscosity*tau12*u12+6*polymer_viscosity*tau13*u13+6*polymer_viscosity*tau12*u21-2*polymer_viscosity*tau33*u22+6*polymer_viscosity*tau23*u23+6*polymer_viscosity*tau13*u31+6*polymer_viscosity*tau23*u32+4*polymer_viscosity*tau33*u33+2*tau11*(tau22+tau33+2*polymer_viscosity*u11-polymer_viscosity*u22-polymer_viscosity*u33)+2*tau22*(tau33-polymer_viscosity*(u11-2*u22+u33)))+extension_length**4*polymer_viscosity**2*(polymer_viscosity*(u23+u32)+relaxation_time*(tau13*u12+tau12*u13+tau22*u23+tau33*u32)+tau23*(-1+relaxation_time*(u22+u33))))/(extension_length**2*(-3+extension_length**2)*polymer_viscosity**2*relaxation_time)

        dstress = jnp.array([dtau11, dtau22, dtau33, dtau12, dtau13, dtau23]).reshape((6,))
        return _vector_to_symmetric_matrix(dstress)

    @eqx.filter_jit
    def shear_stress_experiment_rhs(self, t: Union[float, jax.Array], current_values: jax.Array, applied_stress: AppliedStress, *args, **kwargs) -> jax.Array:
        s22, s11, s33, s23, s13, u12, _ = current_values  # UCM fix: 1<->2 swap in
        s12 = applied_stress.stress(t)[0,1]
        ds12 = jax.jacobian(applied_stress.stress)(t)[0,1]

        extension_length = self.extension_length.get_value()
        polymer_viscosity = self.polymer_viscosity.get_value()
        relaxation_time = self.relaxation_time.get_value()
        solvent_viscosity = self.solvent_viscosity.get_value()

        ds11=(s11*(-((extension_length**2*polymer_viscosity+relaxation_time*(s11+s22+s33))**2/(-3+extension_length**2))+2*polymer_viscosity*relaxation_time**2*s12*u12-2*polymer_viscosity*relaxation_time**2*solvent_viscosity*u12**2))/(extension_length**2*polymer_viscosity**2*relaxation_time)
        ds22=(-((s22*(extension_length**2*polymer_viscosity+relaxation_time*(s11+s22+s33))**2)/((-3+extension_length**2)*relaxation_time))+2*polymer_viscosity*s12*(extension_length**2*polymer_viscosity+relaxation_time*s22)*u12-2*polymer_viscosity*(extension_length**2*polymer_viscosity+relaxation_time*s22)*solvent_viscosity*u12**2)/(extension_length**2*polymer_viscosity**2)
        ds33=(s33*(-((extension_length**2*polymer_viscosity+relaxation_time*(s11+s22+s33))**2/(-3+extension_length**2))+2*polymer_viscosity*relaxation_time**2*s12*u12-2*polymer_viscosity*relaxation_time**2*solvent_viscosity*u12**2))/(extension_length**2*polymer_viscosity**2*relaxation_time)
        du12=(ds12*extension_length**2*(-3+extension_length**2)*polymer_viscosity**2*relaxation_time-extension_length**4*polymer_viscosity**2*(-s12+(polymer_viscosity+relaxation_time*s11+solvent_viscosity)*u12)+relaxation_time**2*(s12-solvent_viscosity*u12)*((s11+s22+s33)**2+6*polymer_viscosity*u12*(s12-solvent_viscosity*u12))+extension_length**2*polymer_viscosity*relaxation_time*(-2*relaxation_time*s12**2*u12-(s22+s33)*(polymer_viscosity+2*solvent_viscosity)*u12-2*relaxation_time*solvent_viscosity**2*u12**3+2*s11*(s12+(polymer_viscosity-solvent_viscosity)*u12)+2*s12*(s22+s33+2*relaxation_time*solvent_viscosity*u12**2)))/(extension_length**2*(-3+extension_length**2)*polymer_viscosity**2*relaxation_time*solvent_viscosity)
        ds13=(s13*(-((extension_length**2*polymer_viscosity+relaxation_time*(s11+s22+s33))**2/(-3+extension_length**2))+2*polymer_viscosity*relaxation_time**2*s12*u12-2*polymer_viscosity*relaxation_time**2*solvent_viscosity*u12**2))/(extension_length**2*polymer_viscosity**2*relaxation_time)
        ds23=(-((s23*(extension_length**2*polymer_viscosity+relaxation_time*(s11+s22+s33))**2)/((-3+extension_length**2)*relaxation_time))+polymer_viscosity*(extension_length**2*polymer_viscosity*s13+2*relaxation_time*s12*s23)*u12-2*polymer_viscosity*relaxation_time*s23*solvent_viscosity*u12**2)/(extension_length**2*polymer_viscosity**2)

        return jnp.array([ds22, ds11, ds33, ds23, ds13, du12, u12]).reshape((7,))

class FENEP(AbstractViscoelasticModel):
    """Finitely Extensible Nonlinear Elastic (FENE-P) dumbbell model.

    The FENE-P (Peterlin) model uses a pre-averaged (Peterlin) FENE spring,
    yielding a closed-form constitutive equation.  Unlike FENE-CR, FENE-P
    predicts a shear-rate dependent viscosity.

    The finite extensibility parameter ``extension_length`` (L) limits the
    maximum polymer chain extension.  In the limit L → ∞ the model approaches
    the Oldroyd-B model.

    Parameters
    ----------
    polymer_viscosity : float | AbstractParameter
        Polymer viscosity η_p (Pa·s).
    relaxation_time : float | AbstractParameter
        Relaxation time λ (s).
    solvent_viscosity : float | AbstractParameter
        Solvent viscosity η_s (Pa·s).
    extension_length : float | AbstractParameter
        Maximum extensibility L (dimensionless).

    Notes
    -----
    The stress ODE components are fully expanded for computational efficiency.

    References
    ----------
    Bird, R.B., Dotson, P.J. & Johnson, N.L. (1980). J. Non-Newtonian Fluid
    Mech., 7(2-3), 213-235.
    """

    polymer_viscosity: AbstractParameter
    relaxation_time: AbstractParameter
    solvent_viscosity: AbstractParameter
    extension_length: AbstractParameter

    @eqx.filter_jit
    def extra_stress_response_rhs(self, t: Union[float, jax.Array], stress: jax.Array, velocity_gradient: VelocityGradient, *args, **kwargs) -> jax.Array:
        u_grad = velocity_gradient.gradient(t).T  # UCM fix: hand-coded RHS uses the conjugate convention
        u11, u12, u13, u21, u22, u23, u31, u32, u33 = u_grad.ravel()
        polymer_viscosity = self.polymer_viscosity.get_value()
        relaxation_time = self.relaxation_time.get_value()
        extension_length = self.extension_length.get_value()

        tau11, tau22, tau33, tau12, tau13, tau23 = _flatten_symmetric_array(stress)

        dtau11=(-((-3+extension_length**2)**2*relaxation_time**2*tau11**3)+2*(-3+extension_length**2)*relaxation_time*tau11**2*(extension_length**2*polymer_viscosity-extension_length**4*polymer_viscosity-(-3+extension_length**2)*relaxation_time*(tau22+tau33-extension_length**2*polymer_viscosity*u11))+extension_length**2*polymer_viscosity*(3*relaxation_time*(tau22+tau33)**2+2*extension_length**6*polymer_viscosity*(polymer_viscosity*u11+relaxation_time*tau12*u21+relaxation_time*tau13*u31)-extension_length**2*relaxation_time*(tau22**2+2*tau22*(tau33+3*polymer_viscosity*u22)+6*polymer_viscosity*(tau12*(u12-2*u21)+tau13*(u13-2*u31)+tau23*(u23+u32))+tau33*(tau33+6*polymer_viscosity*u33))+extension_length**4*polymer_viscosity*(tau22*(-1+2*relaxation_time*u22)+2*relaxation_time*(tau12*(u12-5*u21)+tau13*(u13-5*u31)+tau23*(u23+u32))+2*polymer_viscosity*(-2*u11+u22+u33)+tau33*(-1+2*relaxation_time*u33)))+tau11*(-(extension_length**6*(-2+extension_length**2)*polymer_viscosity**2)+(-3+extension_length**2)*relaxation_time*(3*relaxation_time*(tau22+tau33)**2+2*extension_length**6*polymer_viscosity**2*u11+2*extension_length**4*polymer_viscosity*(tau22*(-1+relaxation_time*u22)+relaxation_time*(tau12*(u12+u21)+tau13*(u13+u31)+tau23*(u23+u32))+polymer_viscosity*(-u11+u22+u33)+tau33*(-1+relaxation_time*u33))-extension_length**2*(relaxation_time*(tau22+tau33)**2+polymer_viscosity*(tau22*(-1+6*relaxation_time*u22)+6*relaxation_time*(tau12*(u12+u21)+tau13*(u13+u31)+tau23*(u23+u32))+tau33*(-1+6*relaxation_time*u33))))))/(extension_length**4*(-3+extension_length**2)**2*polymer_viscosity**2*relaxation_time)
        dtau22=(-((-3+extension_length**2)**2*relaxation_time**2*tau22**3)-(-3+extension_length**2)*relaxation_time*tau11**2*(-3*relaxation_time*tau22+extension_length**2*(polymer_viscosity+relaxation_time*tau22))+tau11*(-(extension_length**6*polymer_viscosity**2)+(-3+extension_length**2)*relaxation_time*(6*relaxation_time*tau22*(tau22+tau33)+2*extension_length**4*polymer_viscosity*(polymer_viscosity*u11+tau22*(-1+relaxation_time*u11))-extension_length**2*(2*polymer_viscosity*tau33+2*relaxation_time*tau22*(tau22+tau33)+polymer_viscosity*tau22*(-1+6*relaxation_time*u11))))+2*(-3+extension_length**2)*relaxation_time*tau22**2*(extension_length**2*polymer_viscosity-extension_length**4*polymer_viscosity+(-3+extension_length**2)*relaxation_time*(-tau33+extension_length**2*polymer_viscosity*u22))+extension_length**2*polymer_viscosity*(3*relaxation_time*tau33**2+2*extension_length**6*polymer_viscosity*(relaxation_time*tau12*u12+polymer_viscosity*u22+relaxation_time*tau23*u32)-extension_length**2*relaxation_time*(tau33**2+6*polymer_viscosity*(tau12*(-2*u12+u21)+tau13*(u13+u31)+tau23*(u23-2*u32))+6*polymer_viscosity*tau33*u33)+extension_length**4*polymer_viscosity*(2*relaxation_time*(tau12*(-5*u12+u21)+tau13*(u13+u31)+tau23*(u23-5*u32))+2*polymer_viscosity*(u11-2*u22+u33)+tau33*(-1+2*relaxation_time*u33)))+tau22*(-(extension_length**6*(-2+extension_length**2)*polymer_viscosity**2)+(-3+extension_length**2)*relaxation_time*(3*relaxation_time*tau33**2+2*extension_length**6*polymer_viscosity**2*u22+2*extension_length**4*polymer_viscosity*(relaxation_time*(tau12*(u12+u21)+tau13*(u13+u31)+tau23*(u23+u32))+polymer_viscosity*(u11-u22+u33)+tau33*(-1+relaxation_time*u33))-extension_length**2*(relaxation_time*tau33**2+6*polymer_viscosity*relaxation_time*(tau12*(u12+u21)+tau13*(u13+u31)+tau23*(u23+u32))+polymer_viscosity*tau33*(-1+6*relaxation_time*u33)))))/(extension_length**4*(-3+extension_length**2)**2*polymer_viscosity**2*relaxation_time)
        dtau33=(-9*relaxation_time**2*tau33*(tau11+tau22+tau33)**2+extension_length**8*polymer_viscosity**2*(-tau33+2*relaxation_time*tau13*u13+2*relaxation_time*tau23*u23+2*(polymer_viscosity+relaxation_time*tau33)*u33)+3*extension_length**2*relaxation_time*(2*relaxation_time*tau33*(tau11+tau22+tau33)**2+polymer_viscosity*(tau11**2+2*tau11*tau22+tau22**2+tau11*tau33*(-1+6*relaxation_time*u11)+tau22*tau33*(-1+6*relaxation_time*u22)+6*relaxation_time*tau33*(tau12*(u12+u21)+tau13*(u13+u31)+tau23*(u23+u32))+2*tau33**2*(-1+3*relaxation_time*u33)))+extension_length**6*polymer_viscosity*(polymer_viscosity*(-tau22+2*tau33+tau11*(-1+2*relaxation_time*u11)+2*relaxation_time*(-5*tau13*u13+tau12*(u12+u21)+tau22*u22-5*tau23*u23+tau13*u31+tau23*u32+tau33*(u11+u22-4*u33)))+2*polymer_viscosity**2*(u11+u22-2*u33)+2*relaxation_time*tau33*(-tau22-tau33+tau11*(-1+relaxation_time*u11)+relaxation_time*(tau12*(u12+u21)+tau22*u22+tau13*(u13+u31)+tau23*(u23+u32)+tau33*u33)))-extension_length**4*relaxation_time*(relaxation_time*tau33*(tau11+tau22+tau33)**2+6*polymer_viscosity**2*(tau11*u11-2*tau13*u13+tau12*(u12+u21)+tau22*u22-2*tau23*u23+tau13*u31+tau23*u32+tau33*(u11+u22-u33))+polymer_viscosity*(tau11**2+tau22**2+tau11*(2*tau22+tau33*(-7+12*relaxation_time*u11))+tau22*tau33*(-7+12*relaxation_time*u22)+4*tau33*(3*relaxation_time*(tau12*(u12+u21)+tau13*(u13+u31)+tau23*(u23+u32))+tau33*(-2+3*relaxation_time*u33)))))/(extension_length**4*(-3+extension_length**2)**2*polymer_viscosity**2*relaxation_time)
        dtau12=(3*relaxation_time**2*tau12*(tau11+tau22+tau33)**2+extension_length**6*polymer_viscosity**2*(polymer_viscosity*(u12+u21)+tau12*(-1+relaxation_time*(u11+u22))+relaxation_time*(tau11*u12+tau22*u21+tau23*u31+tau13*u32))+extension_length**4*polymer_viscosity*relaxation_time*(2*tau11*tau12*(-1+relaxation_time*u11)-3*polymer_viscosity*tau11*u12+2*relaxation_time*tau12**2*(u12+u21)-3*polymer_viscosity*(tau22*u21+tau23*u31+tau13*u32)+tau12*(2*tau22*(-1+relaxation_time*u22)+2*relaxation_time*(tau13*(u13+u31)+tau23*(u23+u32))-polymer_viscosity*(u11+u22-2*u33)+2*tau33*(-1+relaxation_time*u33)))-extension_length**2*relaxation_time*tau12*(relaxation_time*(tau11+tau22+tau33)**2+polymer_viscosity*(-3*(tau22+tau33)+tau11*(-3+6*relaxation_time*u11)+6*relaxation_time*(tau12*(u12+u21)+tau22*u22+tau13*(u13+u31)+tau23*(u23+u32)+tau33*u33))))/(extension_length**4*(-3+extension_length**2)*polymer_viscosity**2*relaxation_time)
        dtau13=(3*relaxation_time**2*tau13*(tau11+tau22+tau33)**2+extension_length**6*polymer_viscosity**2*(polymer_viscosity*(u13+u31)+relaxation_time*(tau11*u13+tau23*u21+tau12*u23+tau33*u31)+tau13*(-1+relaxation_time*(u11+u33)))+extension_length**4*polymer_viscosity*relaxation_time*(2*tau11*tau13*(-1+relaxation_time*u11)-3*polymer_viscosity*tau11*u13+2*relaxation_time*tau13**2*(u13+u31)-3*polymer_viscosity*(tau23*u21+tau12*u23+tau33*u31)+tau13*(2*tau22*(-1+relaxation_time*u22)+2*relaxation_time*(tau12*(u12+u21)+tau23*(u23+u32))-polymer_viscosity*(u11-2*u22+u33)+2*tau33*(-1+relaxation_time*u33)))-extension_length**2*relaxation_time*tau13*(relaxation_time*(tau11+tau22+tau33)**2+polymer_viscosity*(-3*(tau22+tau33)+tau11*(-3+6*relaxation_time*u11)+6*relaxation_time*(tau12*(u12+u21)+tau22*u22+tau13*(u13+u31)+tau23*(u23+u32)+tau33*u33))))/(extension_length**4*(-3+extension_length**2)*polymer_viscosity**2*relaxation_time)
        dtau23=(3*relaxation_time**2*tau23*(tau11+tau22+tau33)**2+extension_length**4*polymer_viscosity*relaxation_time*(-2*tau22*tau23-2*tau23*tau33+2*polymer_viscosity*tau23*u11+2*tau11*tau23*(-1+relaxation_time*u11)-3*polymer_viscosity*tau13*u12+2*relaxation_time*tau12*tau23*u12-3*polymer_viscosity*tau12*u13+2*relaxation_time*tau13*tau23*u13+2*relaxation_time*tau12*tau23*u21-polymer_viscosity*tau23*u22+2*relaxation_time*tau22*tau23*u22-3*polymer_viscosity*tau22*u23+2*relaxation_time*tau23**2*u23+2*relaxation_time*tau13*tau23*u31+2*relaxation_time*tau23**2*u32-3*polymer_viscosity*tau33*u32-polymer_viscosity*tau23*u33+2*relaxation_time*tau23*tau33*u33)+extension_length**6*polymer_viscosity**2*(polymer_viscosity*(u23+u32)+relaxation_time*(tau13*u12+tau12*u13+tau22*u23+tau33*u32)+tau23*(-1+relaxation_time*(u22+u33)))-extension_length**2*relaxation_time*tau23*(relaxation_time*(tau11+tau22+tau33)**2+polymer_viscosity*(-3*(tau22+tau33)+tau11*(-3+6*relaxation_time*u11)+6*relaxation_time*(tau12*(u12+u21)+tau22*u22+tau13*(u13+u31)+tau23*(u23+u32)+tau33*u33))))/(extension_length**4*(-3+extension_length**2)*polymer_viscosity**2*relaxation_time)

        dstress = jnp.array([dtau11, dtau22, dtau33, dtau12, dtau13, dtau23]).reshape((6,))
        return _vector_to_symmetric_matrix(dstress)

    @eqx.filter_jit
    def shear_stress_experiment_rhs(self, t: Union[float, jax.Array], current_values: jax.Array, applied_stress: AppliedStress, *args, **kwargs) -> jax.Array:

        s22, s11, s33, s23, s13, u12, _ = current_values  # UCM fix: 1<->2 swap in
        s12 = applied_stress.stress(t)[0,1]
        ds12 = jax.jacobian(applied_stress.stress)(t)[0,1]

        extension_length = self.extension_length.get_value()
        polymer_viscosity = self.polymer_viscosity.get_value()
        solvent_viscosity = self.solvent_viscosity.get_value()
        relaxation_time = self.relaxation_time.get_value()

        ds11=-((extension_length**8*polymer_viscosity**2*s11+9*relaxation_time**2*s11*(s11+s22+s33)**2+3*extension_length**2*relaxation_time*(-((s11+s22+s33)*(polymer_viscosity*(-2*s11+s22+s33)+2*relaxation_time*s11*(s11+s22+s33)))-6*polymer_viscosity*relaxation_time*s11*s12*u12+6*polymer_viscosity*relaxation_time*s11*solvent_viscosity*u12**2)+extension_length**4*relaxation_time*((s11+s22+s33)*(polymer_viscosity*(-8*s11+s22+s33)+relaxation_time*s11*(s11+s22+s33))+6*polymer_viscosity*(polymer_viscosity+2*relaxation_time*s11)*s12*u12-6*polymer_viscosity*(polymer_viscosity+2*relaxation_time*s11)*solvent_viscosity*u12**2)+extension_length**6*polymer_viscosity*(2*relaxation_time*s11*(s11+s22+s33+relaxation_time*u12*(-s12+solvent_viscosity*u12))+polymer_viscosity*(-2*s11+s22+s33+2*relaxation_time*u12*(-s12+solvent_viscosity*u12))))/(extension_length**4*(-3+extension_length**2)**2*polymer_viscosity**2*relaxation_time))
        ds22=-((9*relaxation_time**2*s22*(s11+s22+s33)**2+3*extension_length**2*relaxation_time*(-((s11+s22+s33)*(polymer_viscosity*(s11-2*s22+s33)+2*relaxation_time*s22*(s11+s22+s33)))-6*polymer_viscosity*relaxation_time*s12*s22*u12+6*polymer_viscosity*relaxation_time*s22*solvent_viscosity*u12**2)+extension_length**4*relaxation_time*((s11+s22+s33)*(polymer_viscosity*(s11-8*s22+s33)+relaxation_time*s22*(s11+s22+s33))-12*polymer_viscosity*s12*(polymer_viscosity-relaxation_time*s22)*u12+12*polymer_viscosity*(polymer_viscosity-relaxation_time*s22)*solvent_viscosity*u12**2)+extension_length**8*polymer_viscosity**2*(s22+2*relaxation_time*u12*(-s12+solvent_viscosity*u12))+extension_length**6*polymer_viscosity*(polymer_viscosity*(s11-2*s22+s33+10*relaxation_time*u12*(s12-solvent_viscosity*u12))+2*relaxation_time*s22*(s11+s22+s33+relaxation_time*u12*(-s12+solvent_viscosity*u12))))/(extension_length**4*(-3+extension_length**2)**2*polymer_viscosity**2*relaxation_time))
        ds33=-((extension_length**8*polymer_viscosity**2*s33+9*relaxation_time**2*s33*(s11+s22+s33)**2-3*extension_length**2*relaxation_time*((s11+s22+s33)*(polymer_viscosity*(s11+s22-2*s33)+2*relaxation_time*s33*(s11+s22+s33))+6*polymer_viscosity*relaxation_time*s12*s33*u12-6*polymer_viscosity*relaxation_time*s33*solvent_viscosity*u12**2)+extension_length**4*relaxation_time*((s11+s22+s33)*(polymer_viscosity*(s11+s22-8*s33)+relaxation_time*s33*(s11+s22+s33))+6*polymer_viscosity*s12*(polymer_viscosity+2*relaxation_time*s33)*u12-6*polymer_viscosity*(polymer_viscosity+2*relaxation_time*s33)*solvent_viscosity*u12**2)+extension_length**6*polymer_viscosity*(2*relaxation_time*s33*(s11+s22+s33+relaxation_time*u12*(-s12+solvent_viscosity*u12))+polymer_viscosity*(s11+s22-2*(s33+relaxation_time*u12*(s12-solvent_viscosity*u12)))))/(extension_length**4*(-3+extension_length**2)**2*polymer_viscosity**2*relaxation_time))
        du12=(ds12*extension_length**4*(-3+extension_length**2)*polymer_viscosity**2*relaxation_time-3*relaxation_time**2*(s11+s22+s33)**2*(s12-solvent_viscosity*u12)-extension_length**6*polymer_viscosity**2*(-s12+(polymer_viscosity+relaxation_time*s11+solvent_viscosity)*u12)+extension_length**4*polymer_viscosity*relaxation_time*(s11*(2*s12+3*polymer_viscosity*u12-2*solvent_viscosity*u12)-2*(s12-solvent_viscosity*u12)*(-s22-s33+relaxation_time*u12*(s12-solvent_viscosity*u12)))+extension_length**2*relaxation_time*(s12-solvent_viscosity*u12)*(relaxation_time*(s11+s22+s33)**2-3*polymer_viscosity*(s11+s22+s33+2*relaxation_time*u12*(-s12+solvent_viscosity*u12))))/(extension_length**4*(-3+extension_length**2)*polymer_viscosity**2*relaxation_time*solvent_viscosity)
        ds13=-((s13*(extension_length**6*polymer_viscosity**2+relaxation_time*(-3*relaxation_time*(s11+s22+s33)**2+2*extension_length**4*polymer_viscosity*(s11+s22+s33+relaxation_time*u12*(-s12+solvent_viscosity*u12))+extension_length**2*(relaxation_time*(s11+s22+s33)**2-3*polymer_viscosity*(s11+s22+s33+2*relaxation_time*u12*(-s12+solvent_viscosity*u12))))))/(extension_length**4*(-3+extension_length**2)*polymer_viscosity**2*relaxation_time))
        ds23=s13*u12-(s23*(extension_length**6*polymer_viscosity**2+relaxation_time*(-3*relaxation_time*(s11+s22+s33)**2+2*extension_length**4*polymer_viscosity*(s11+s22+s33+relaxation_time*u12*(-s12+solvent_viscosity*u12))+extension_length**2*(relaxation_time*(s11+s22+s33)**2-3*polymer_viscosity*(s11+s22+s33+2*relaxation_time*u12*(-s12+solvent_viscosity*u12))))))/(extension_length**4*(-3+extension_length**2)*polymer_viscosity**2*relaxation_time)

        return jnp.array([ds22, ds11, ds33, ds23, ds13, du12, u12]).reshape((7,))

class XPomPom(AbstractViscoelasticModel):
    """eXtended Pom-Pom (XPP) model for branched polymer melts.

    The Pom-Pom model was developed to capture the complex rheology of
    H-shaped (and more generally branched) polymer molecules.  The XPP
    formulation uses a single differential equation for the stress tensor,
    including both orientation relaxation (λ) and stretch relaxation (λ_s).

    Key dimensionless groups:

    * ``alpha`` (α) – anisotropy parameter (Giesekus-like quadratic term).
    * ``n`` – stretch exponent.
    * ``q`` – number of arms at each branch point.
    * ``relaxation_time_s`` (λ_s) – stretch relaxation time.

    Parameters
    ----------
    polymer_viscosity : float | AbstractParameter
        Polymer viscosity η_p (Pa·s).
    relaxation_time : float | AbstractParameter
        Orientation relaxation time λ (s).
    solvent_viscosity : float | AbstractParameter
        Solvent viscosity η_s (Pa·s).
    alpha : float | AbstractParameter
        Anisotropy/mobility parameter α.
    relaxation_time_s : float | AbstractParameter
        Stretch relaxation time λ_s (s); typically λ_s < λ.
    n : float | AbstractParameter
        Stretch exponent (controls approach to maximum stretch).
    q : float | AbstractParameter
        Number of dangling arms per branch point.

    References
    ----------
    Verbeeten, W.M.H., Peters, G.W.M. & Baaijens, F.P.T. (2001). J. Rheol.,
    45(4), 823-843.
    """

    polymer_viscosity: AbstractParameter
    relaxation_time: AbstractParameter
    solvent_viscosity: AbstractParameter
    alpha: AbstractParameter
    relaxation_time_s: AbstractParameter
    n: AbstractParameter
    q: AbstractParameter

    @eqx.filter_jit
    def extra_stress_response_rhs(self, t: Union[float, jax.Array], stress: jax.Array, velocity_gradient: VelocityGradient, *args, **kwargs) -> jax.Array:
        u_grad = velocity_gradient.gradient(t).T  # UCM fix: hand-coded RHS uses the conjugate convention
        u11, u12, u13, u21, u22, u23, u31, u32, u33 = u_grad.ravel()
        polymer_viscosity = self.polymer_viscosity.get_value()
        relaxation_time = self.relaxation_time.get_value()
        alpha = self.alpha.get_value()
        relaxation_time_s = self.relaxation_time_s.get_value()
        n = self.n.get_value()
        q = self.q.get_value()

        tau11, tau22, tau33, tau12, tau13, tau23 = _flatten_symmetric_array(stress)
        
        dtau11=(alpha*(-2*tau11**2-tau12**2-tau13**2+tau22**2+2*tau23**2+tau33**2)-(alpha*relaxation_time*(tau11**2*(tau22+tau33)+(tau12**2+tau13**2)*(tau22+tau33)-tau11*(tau12**2+tau13**2+tau22**2+2*tau23**2+tau33**2)))/polymer_viscosity-(2*jnp.exp((2*(-3+jnp.sqrt(9+(3*relaxation_time*(tau11+tau22+tau33))/polymer_viscosity)))/(3*q))*(polymer_viscosity+relaxation_time*tau11)*(-(3**((1+n)/2)*polymer_viscosity*jnp.sqrt(3+(relaxation_time*(tau11+tau22+tau33))/polymer_viscosity))+polymer_viscosity*(3+(relaxation_time*(tau11+tau22+tau33))/polymer_viscosity)**((2+n)/2)))/(relaxation_time*relaxation_time_s*(3+(relaxation_time*(tau11+tau22+tau33))/polymer_viscosity)**(n/2))+(6*polymer_viscosity**2*u11)/relaxation_time+2*relaxation_time*(tau11+tau22+tau33)*(tau11*u11+tau12*u21+tau13*u31)+(polymer_viscosity*(tau22+tau33+2*relaxation_time*tau22*u11+2*relaxation_time*tau33*u11+tau11*(-2+8*relaxation_time*u11)+6*relaxation_time*tau12*u21+6*relaxation_time*tau13*u31))/relaxation_time)/(3*polymer_viscosity+relaxation_time*(tau11+tau22+tau33))
        dtau22=(alpha*(tau11**2-tau12**2+2*tau13**2-2*tau22**2-tau23**2+tau33**2)+(alpha*relaxation_time*(tau11**2*tau22+tau22*(tau12**2+2*tau13**2+tau23**2)-tau11*(tau12**2+tau22**2+tau23**2)-(tau12**2+tau22**2+tau23**2)*tau33+tau22*tau33**2))/polymer_viscosity-(2*jnp.exp((2*(-3+jnp.sqrt(9+(3*relaxation_time*(tau11+tau22+tau33))/polymer_viscosity)))/(3*q))*(polymer_viscosity+relaxation_time*tau22)*(-(3**((1+n)/2)*polymer_viscosity*jnp.sqrt(3+(relaxation_time*(tau11+tau22+tau33))/polymer_viscosity))+polymer_viscosity*(3+(relaxation_time*(tau11+tau22+tau33))/polymer_viscosity)**((2+n)/2)))/(relaxation_time*relaxation_time_s*(3+(relaxation_time*(tau11+tau22+tau33))/polymer_viscosity)**(n/2))+(6*polymer_viscosity**2*u22)/relaxation_time+2*relaxation_time*(tau11+tau22+tau33)*(tau12*u12+tau22*u22+tau23*u32)+(polymer_viscosity*(tau11+tau33+6*relaxation_time*tau12*u12+2*relaxation_time*tau11*u22+2*relaxation_time*tau33*u22+tau22*(-2+8*relaxation_time*u22)+6*relaxation_time*tau23*u32))/relaxation_time)/(3*polymer_viscosity+relaxation_time*(tau11+tau22+tau33))
        dtau33=(alpha*(tau11**2+2*tau12**2-tau13**2+tau22**2-tau23**2-2*tau33**2)+(alpha*relaxation_time*(-((tau11+tau22)*(tau13**2+tau23**2))+(tau11**2+2*tau12**2+tau13**2+tau22**2+tau23**2)*tau33-(tau11+tau22)*tau33**2))/polymer_viscosity-(2*jnp.exp((2*(-3+jnp.sqrt(9+(3*relaxation_time*(tau11+tau22+tau33))/polymer_viscosity)))/(3*q))*(polymer_viscosity+relaxation_time*tau33)*(-(3**((1+n)/2)*polymer_viscosity*jnp.sqrt(3+(relaxation_time*(tau11+tau22+tau33))/polymer_viscosity))+polymer_viscosity*(3+(relaxation_time*(tau11+tau22+tau33))/polymer_viscosity)**((2+n)/2)))/(relaxation_time*relaxation_time_s*(3+(relaxation_time*(tau11+tau22+tau33))/polymer_viscosity)**(n/2))+(6*polymer_viscosity**2*u33)/relaxation_time+2*relaxation_time*(tau11+tau22+tau33)*(tau13*u13+tau23*u23+tau33*u33)+(polymer_viscosity*(tau11+tau22-2*tau33+6*relaxation_time*tau13*u13+6*relaxation_time*tau23*u23+2*relaxation_time*(tau11+tau22+4*tau33)*u33))/relaxation_time)/(3*polymer_viscosity+relaxation_time*(tau11+tau22+tau33))
        dtau12=(2*alpha*relaxation_time**2*relaxation_time_s*tau12**3+tau12*(jnp.exp((2*(-3+jnp.sqrt(9+(3*relaxation_time*(tau11+tau22+tau33))/polymer_viscosity)))/(3*q))*(-2*polymer_viscosity*relaxation_time**2*(tau11+tau22+tau33)+2*polymer_viscosity**2*relaxation_time*(-3+3**((1+n)/2)*(3+(relaxation_time*(tau11+tau22+tau33))/polymer_viscosity)**((1-n)/2)))+relaxation_time_s*(alpha*relaxation_time**2*(2*(tau13**2-tau11*tau22+tau23**2)-(tau11+tau22)*tau33+tau33**2)+3*polymer_viscosity**2*(-1+relaxation_time*(u11+u22))+polymer_viscosity*relaxation_time*(-3*alpha*(tau11+tau22)+relaxation_time*(tau11+tau22+tau33)*(u11+u22))))+relaxation_time_s*(3*polymer_viscosity+relaxation_time*(tau11+tau22+tau33))*(-(alpha*relaxation_time*tau13*tau23)+polymer_viscosity*(polymer_viscosity*(u12+u21)+relaxation_time*(tau11*u12+tau22*u21+tau23*u31+tau13*u32))))/(polymer_viscosity*relaxation_time*relaxation_time_s*(3*polymer_viscosity+relaxation_time*(tau11+tau22+tau33)))
        dtau13=(2*alpha*relaxation_time**2*relaxation_time_s*tau13**3+relaxation_time_s*(3*polymer_viscosity+relaxation_time*(tau11+tau22+tau33))*(-(alpha*relaxation_time*tau12*tau23)+polymer_viscosity*(polymer_viscosity*(u13+u31)+relaxation_time*(tau11*u13+tau23*u21+tau12*u23+tau33*u31)))+tau13*(jnp.exp((2*(-3+jnp.sqrt(9+(3*relaxation_time*(tau11+tau22+tau33))/polymer_viscosity)))/(3*q))*(-2*polymer_viscosity*relaxation_time**2*(tau11+tau22+tau33)+2*polymer_viscosity**2*relaxation_time*(-3+3**((1+n)/2)*(3+(relaxation_time*(tau11+tau22+tau33))/polymer_viscosity)**((1-n)/2)))+relaxation_time_s*(alpha*relaxation_time**2*(2*tau12**2-tau11*tau22+tau22**2+2*tau23**2-(2*tau11+tau22)*tau33)+3*polymer_viscosity**2*(-1+relaxation_time*(u11+u33))+polymer_viscosity*relaxation_time*(-3*alpha*(tau11+tau33)+relaxation_time*(tau11+tau22+tau33)*(u11+u33)))))/(polymer_viscosity*relaxation_time*relaxation_time_s*(3*polymer_viscosity+relaxation_time*(tau11+tau22+tau33)))
        dtau23=(2*alpha*relaxation_time**2*relaxation_time_s*tau12**2*tau23+2*alpha*relaxation_time**2*relaxation_time_s*tau23**3+relaxation_time*relaxation_time_s*tau12*(3*polymer_viscosity+relaxation_time*(tau11+tau22+tau33))*(-(alpha*tau13)+polymer_viscosity*u13)+polymer_viscosity*relaxation_time_s*(3*polymer_viscosity+relaxation_time*(tau11+tau22+tau33))*(polymer_viscosity*(u23+u32)+relaxation_time*(tau13*u12+tau22*u23+tau33*u32))+tau23*((2*jnp.exp((-6+2*jnp.sqrt(9+(3*relaxation_time*(tau11+tau22+tau33))/polymer_viscosity))/(3*q))*polymer_viscosity*relaxation_time*(3**((1+n)/2)*polymer_viscosity*jnp.sqrt(3+(relaxation_time*(tau11+tau22+tau33))/polymer_viscosity)-(3*polymer_viscosity+relaxation_time*(tau11+tau22+tau33))*(3+(relaxation_time*(tau11+tau22+tau33))/polymer_viscosity)**(n/2)))/(3+(relaxation_time*(tau11+tau22+tau33))/polymer_viscosity)**(n/2)+relaxation_time_s*(alpha*relaxation_time**2*(tau11**2+2*tau13**2-2*tau22*tau33-tau11*(tau22+tau33))+3*polymer_viscosity**2*(-1+relaxation_time*(u22+u33))+polymer_viscosity*relaxation_time*(-3*alpha*(tau22+tau33)+relaxation_time*(tau11+tau22+tau33)*(u22+u33)))))/(polymer_viscosity*relaxation_time*relaxation_time_s*(3*polymer_viscosity+relaxation_time*(tau11+tau22+tau33)))

        dstress = jnp.array([dtau11, dtau22, dtau33, dtau12, dtau13, dtau23])
        return _vector_to_symmetric_matrix(dstress)

    @eqx.filter_jit
    def shear_stress_experiment_rhs(self, t: Union[float, jax.Array], current_values: jax.Array, applied_stress: AppliedStress, *args, **kwargs) -> jax.Array:
        s22, s11, s33, s23, s13, u12, _ = current_values  # UCM fix: 1<->2 swap in
        s12 = applied_stress.stress(t)[0,1]
        ds12 = jax.jacobian(applied_stress.stress)(t)[0,1]

        polymer_viscosity = self.polymer_viscosity.get_value()
        relaxation_time = self.relaxation_time.get_value()
        alpha = self.alpha.get_value()
        relaxation_time_s = self.relaxation_time_s.get_value()
        n = self.n.get_value()
        q = self.q.get_value()
        solvent_viscosity = self.solvent_viscosity.get_value()

        ds11=-(((2*jnp.exp((2*(-3+jnp.sqrt(9+(3*relaxation_time*(s11+s22+s33))/polymer_viscosity)))/(3*q))*polymer_viscosity*(polymer_viscosity+relaxation_time*s11)*(-(3**((1+n)/2)*polymer_viscosity*jnp.sqrt(3+(relaxation_time*(s11+s22+s33))/polymer_viscosity))+polymer_viscosity*(3+(relaxation_time*(s11+s22+s33))/polymer_viscosity)**((2+n)/2)))/(3+(relaxation_time*(s11+s22+s33))/polymer_viscosity)**(n/2)+relaxation_time_s*(polymer_viscosity**2*(2*s11-s22-s33)+alpha*polymer_viscosity*relaxation_time*(2*s11**2+s13**2-s22**2-2*s23**2-s33**2+(s12-solvent_viscosity*u12)**2)+alpha*relaxation_time**2*(s11**2*(s22+s33)+(s22+s33)*(s13**2+(s12-solvent_viscosity*u12)**2)-s11*(s13**2+s22**2+2*s23**2+s33**2+(s12-solvent_viscosity*u12)**2))))/(polymer_viscosity*relaxation_time*relaxation_time_s*(3*polymer_viscosity+relaxation_time*(s11+s22+s33))))
        ds22=-(((2*jnp.exp((2*(-3+jnp.sqrt(9+(3*relaxation_time*(s11+s22+s33))/polymer_viscosity)))/(3*q))*polymer_viscosity*(polymer_viscosity+relaxation_time*s22)*(-(3**((1+n)/2)*polymer_viscosity*jnp.sqrt(3+(relaxation_time*(s11+s22+s33))/polymer_viscosity))+polymer_viscosity*(3+(relaxation_time*(s11+s22+s33))/polymer_viscosity)**((2+n)/2)))/(3+(relaxation_time*(s11+s22+s33))/polymer_viscosity)**(n/2)+relaxation_time_s*(-(polymer_viscosity**2*(s11-2*s22+s33+6*relaxation_time*u12*(s12-solvent_viscosity*u12)))+alpha*relaxation_time**2*(-(s11**2*s22)-2*s13**2*s22-s22*s23**2+s22**2*s33+s23**2*s33-s22*s33**2+s12**2*(-s22+s33)+2*s12*(s22-s33)*solvent_viscosity*u12+(-s22+s33)*solvent_viscosity**2*u12**2+s11*(s22**2+s23**2+(s12-solvent_viscosity*u12)**2))+polymer_viscosity*relaxation_time*(2*relaxation_time*(s11+s22+s33)*u12*(-s12+solvent_viscosity*u12)+alpha*(-s11**2-2*s13**2+2*s22**2+s23**2-s33**2+(s12-solvent_viscosity*u12)**2))))/(polymer_viscosity*relaxation_time*relaxation_time_s*(3*polymer_viscosity+relaxation_time*(s11+s22+s33))))
        ds33=-(((2*jnp.exp((2*(-3+jnp.sqrt(9+(3*relaxation_time*(s11+s22+s33))/polymer_viscosity)))/(3*q))*polymer_viscosity*(polymer_viscosity+relaxation_time*s33)*(-(3**((1+n)/2)*polymer_viscosity*jnp.sqrt(3+(relaxation_time*(s11+s22+s33))/polymer_viscosity))+polymer_viscosity*(3+(relaxation_time*(s11+s22+s33))/polymer_viscosity)**((2+n)/2)))/(3+(relaxation_time*(s11+s22+s33))/polymer_viscosity)**(n/2)+relaxation_time_s*(-(polymer_viscosity**2*(s11+s22-2*s33))+alpha*relaxation_time**2*(-(s11**2*s33)-2*s12**2*s33+(s22-s33)*(s13**2+s23**2-s22*s33)+s11*(s13**2+s23**2+s33**2)+4*s12*s33*solvent_viscosity*u12-2*s33*solvent_viscosity**2*u12**2)+alpha*polymer_viscosity*relaxation_time*(-s11**2+s13**2-s22**2+s23**2+2*s33**2-2*(s12-solvent_viscosity*u12)**2)))/(polymer_viscosity*relaxation_time*relaxation_time_s*(3*polymer_viscosity+relaxation_time*(s11+s22+s33))))
        du12=-((-(ds12*polymer_viscosity*relaxation_time*relaxation_time_s*(3*polymer_viscosity+relaxation_time*(s11+s22+s33)))+(2*jnp.exp((-6+2*jnp.sqrt(9+(3*relaxation_time*(s11+s22+s33))/polymer_viscosity))/(3*q))*polymer_viscosity*relaxation_time*(3**((1+n)/2)*polymer_viscosity*jnp.sqrt(3+(relaxation_time*(s11+s22+s33))/polymer_viscosity)-(3*polymer_viscosity+relaxation_time*(s11+s22+s33))*(3+(relaxation_time*(s11+s22+s33))/polymer_viscosity)**(n/2))*(s12-solvent_viscosity*u12))/(3+(relaxation_time*(s11+s22+s33))/polymer_viscosity)**(n/2)+relaxation_time_s*(3*polymer_viscosity**3*u12+polymer_viscosity**2*(-3*s12+relaxation_time*(4*s11+s22+s33)*u12+3*solvent_viscosity*u12)+polymer_viscosity*relaxation_time*(-3*alpha*(s12*(s11+s22)+s13*s23)+relaxation_time*s11*(s11+s22+s33)*u12+3*alpha*(s11+s22)*solvent_viscosity*u12)+alpha*relaxation_time**2*(2*s12**3-s13*s23*(s11+s22+s33)-6*s12**2*solvent_viscosity*u12+(-2*(s13**2-s11*s22+s23**2)+(s11+s22)*s33-s33**2)*solvent_viscosity*u12-2*solvent_viscosity**3*u12**3+s12*(2*s13**2-2*s11*s22-(s11+s22)*s33+s33**2+2*(s23**2+3*solvent_viscosity**2*u12**2)))))/(polymer_viscosity*relaxation_time*relaxation_time_s*(3*polymer_viscosity+relaxation_time*(s11+s22+s33))*solvent_viscosity))
        ds13=(2*alpha*relaxation_time**2*relaxation_time_s*s13**3-alpha*relaxation_time*relaxation_time_s*s23*(3*polymer_viscosity+relaxation_time*(s11+s22+s33))*(s12-solvent_viscosity*u12)+s13*(jnp.exp((2*(-3+jnp.sqrt(9+(3*relaxation_time*(s11+s22+s33))/polymer_viscosity)))/(3*q))*(-2*polymer_viscosity*relaxation_time**2*(s11+s22+s33)+2*polymer_viscosity**2*relaxation_time*(-3+3**((1+n)/2)*(3+(relaxation_time*(s11+s22+s33))/polymer_viscosity)**((1-n)/2)))-relaxation_time_s*(3*polymer_viscosity**2+3*alpha*polymer_viscosity*relaxation_time*(s11+s33)+alpha*relaxation_time**2*(-2*s23**2+s22*(-s22+s33)+s11*(s22+2*s33)-2*(s12-solvent_viscosity*u12)**2))))/(polymer_viscosity*relaxation_time*relaxation_time_s*(3*polymer_viscosity+relaxation_time*(s11+s22+s33)))
        ds23=(2*alpha*relaxation_time**2*relaxation_time_s*s12**2*s23+2*alpha*relaxation_time**2*relaxation_time_s*s23**3+relaxation_time*relaxation_time_s*s13*(3*polymer_viscosity+relaxation_time*(s11+s22+s33))*(polymer_viscosity+alpha*solvent_viscosity)*u12-alpha*relaxation_time*relaxation_time_s*s12*(3*polymer_viscosity*s13+relaxation_time*s13*(s11+s22+s33)+4*relaxation_time*s23*solvent_viscosity*u12)+s23*(jnp.exp((2*(-3+jnp.sqrt(9+(3*relaxation_time*(s11+s22+s33))/polymer_viscosity)))/(3*q))*(-2*polymer_viscosity*relaxation_time**2*(s11+s22+s33)+2*polymer_viscosity**2*relaxation_time*(-3+3**((1+n)/2)*(3+(relaxation_time*(s11+s22+s33))/polymer_viscosity)**((1-n)/2)))-relaxation_time_s*(3*polymer_viscosity**2+3*alpha*polymer_viscosity*relaxation_time*(s22+s33)+alpha*relaxation_time**2*(-s11**2+2*s22*s33+s11*(s22+s33)-2*(s13**2+solvent_viscosity**2*u12**2)))))/(polymer_viscosity*relaxation_time*relaxation_time_s*(3*polymer_viscosity+relaxation_time*(s11+s22+s33)))

        return jnp.array([ds22, ds11, ds33, ds23, ds13, du12, u12]).reshape((7,))
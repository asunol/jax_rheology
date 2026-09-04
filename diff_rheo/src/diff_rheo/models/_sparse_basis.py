"""Sparse tensor-basis closure for data-driven constitutive discovery.

The viscoelastic models in this package differ only in the nonlinear closure
``F(tau, D)`` of the Generalised Oldroyd-B equation

    dtau/dt - L.tau - tau.L^T + (tau - 2 eta_p D + F(tau, D)) / lambda = 0 .

Each textbook model is a *sparse* point in the space of closures:

    Oldroyd-B  :  F = 0
    Giesekus   :  F = (alpha lambda / eta_p) * tau.tau
    linear PTT :  F = (epsilon lambda / eta_p) * tr(tau) * tau

:class:`SparseTensorBasisF` represents ``F`` as a linear combination of a small
library of frame-invariant tensor-basis terms with scalar coefficients.
Fitting those coefficients with a sparsity penalty (see
``scripts/discover_constitutive.py``) discovers the smallest set of terms a
dataset supports --- the SINDy idea applied to constitutive laws.  Unlike
:class:`~diff_rheo.models.RUDE`, which uses a neural network for ``F``, this
closure is *interpretable*: a surviving coefficient names a physical term.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

#: Human-readable names of the tensor-basis library terms, in coefficient order.
BASIS_NAMES = (
    "tau.tau",            # Giesekus quadratic closure
    "tr(tau).tau",        # linear PTT closure
    "tr(tau).D",          # rate-coupled trace term
    "tau.D + D.tau",      # co-rotational stress-rate coupling
    "D.D",                # quadratic rate term
    "tr(tau.tau).tau",    # higher-order (quadratic-invariant) closure
)


class SparseTensorBasisF(eqx.Module):
    """A closure ``F(tau, D) = sum_m c_m B_m(tau, D)`` over a tensor-basis library.

    Plug an instance into :class:`~diff_rheo.models.GeneralizedOldroydB` as its
    ``F_function``.  The coefficient vector :attr:`coefficients` is a trainable
    array leaf; fitting it under an L1 / sequential-thresholding penalty yields
    the sparsest closure consistent with the data.

    Parameters
    ----------
    coefficients : jax.Array, optional
        Length-``len(BASIS_NAMES)`` coefficient vector.  Defaults to all zeros
        --- i.e. the closure ``F = 0`` (the Oldroyd-B / UCM limit), the natural
        starting point for sparse discovery.
    """

    coefficients: jax.Array

    def __init__(self, coefficients=None):
        if coefficients is None:
            self.coefficients = jnp.zeros(len(BASIS_NAMES))
        else:
            self.coefficients = jnp.asarray(coefficients, dtype=float)

    def basis(self, tau: jax.Array, rate_of_strain: jax.Array) -> jax.Array:
        """Return the stacked library tensors ``B_m(tau, D)``, shape ``(M, 3, 3)``."""
        tr_tau = jnp.trace(tau)
        return jnp.stack([
            tau @ tau,
            tr_tau * tau,
            tr_tau * rate_of_strain,
            tau @ rate_of_strain + rate_of_strain @ tau,
            rate_of_strain @ rate_of_strain,
            jnp.trace(tau @ tau) * tau,
        ])

    def __call__(self, stress: jax.Array, rate_of_strain: jax.Array) -> jax.Array:
        """Evaluate the closure ``F = sum_m c_m B_m``, shape ``(3, 3)``."""
        basis = self.basis(stress, rate_of_strain)
        return jnp.sum(self.coefficients[:, None, None] * basis, axis=0)

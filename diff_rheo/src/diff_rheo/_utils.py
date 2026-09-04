"""Internal utility functions shared across the diff_rheo package.

This module contains low-level helpers for tensor manipulation and noise
generation.  These functions are not part of the public API.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gamma


def _safe_sqrt(x: jax.Array, eps: float = 1e-10) -> jax.Array:
    """``sqrt`` regularised for stable automatic differentiation near ``x = 0``.

    ``jnp.sqrt`` has an infinite first derivative at zero — and worse
    higher-order derivatives — so differentiating a strain-rate magnitude at a
    point of *zero shear* produces ``nan`` / overflowing gradients.  This bites
    at the turning points of an oscillatory waveform, e.g. when differentiating
    the optimal-experiment-design objective (a second-order quantity) through
    the forcing.

    Evaluating ``sqrt(x + eps)`` keeps every derivative order finite.  With the
    tiny default ``eps`` the returned value is unchanged to ~1e-5 even at
    ``x = 0`` and to full floating-point precision everywhere else.

    Parameters
    ----------
    x : jax.Array
        Non-negative argument.
    eps : float
        Regularisation floor added inside the square root.

    Returns
    -------
    jax.Array
        ``sqrt(x + eps)``.
    """
    return jnp.sqrt(x + eps)


def _rate_of_strain_to_strain_rate(rate_of_strain: jax.Array) -> jax.Array:
    """Compute the scalar shear rate γ̇ from a rate-of-strain tensor D.

    The scalar shear rate is defined as the second invariant of D:

        γ̇ = sqrt(2 · D : D) = sqrt(2 · tr(D · Dᵀ))

    Accepts D in either full 3×3 or packed symmetric 6-vector form
    ``[D11, D22, D33, D12, D13, D23]``.

    Parameters
    ----------
    rate_of_strain : jax.Array
        Rate-of-strain tensor.  Shape ``(3, 3)`` or ``(6,)``.

    Returns
    -------
    jax.Array
        Scalar shear rate, shape ``()``.

    Raises
    ------
    ValueError
        If ``rate_of_strain`` does not have shape ``(3, 3)`` or ``(6,)``.
    """
    if rate_of_strain.shape == (3, 3):
        return _safe_sqrt(jnp.sum(jnp.ravel(rate_of_strain) @ jnp.ravel(rate_of_strain))*2)
    elif rate_of_strain.shape == (6,):
        a1, a2, a3, a4, a5, a6 = rate_of_strain
        return _safe_sqrt((a1**2 + a2**2 + a3**2 + 2 * a4**2 + 2 * a5**2 + 2 * a6**2)*2)
    else:
        raise ValueError("Array must be 3x3 or 6x1")


def _flatten_symmetric_array(array: jax.Array) -> jax.Array:
    """Pack a symmetric 3×3 matrix into a 6-vector.

    Extracts the diagonal entries followed by the upper-triangular off-diagonal
    entries, giving the order ``[a11, a22, a33, a12, a13, a23]``.

    This is the inverse of :func:`_vector_to_symmetric_matrix`.

    Parameters
    ----------
    array : jax.Array
        Symmetric 3×3 matrix, shape ``(3, 3)``.

    Returns
    -------
    jax.Array
        Packed 6-vector, shape ``(6,)``.
    """
    diagonal = jnp.diag(array)
    upper_triangular = array[jnp.triu_indices(3, k=1)]
    return jnp.concatenate([diagonal, upper_triangular])


def _vector_to_symmetric_matrix(vec: jax.Array) -> jax.Array:
    """Unpack a 6-vector into a symmetric 3×3 matrix.

    Assumes the packing order ``[s11, s22, s33, s12, s13, s23]`` and fills
    in the lower-triangular entries by symmetry.

    This is the inverse of :func:`_flatten_symmetric_array`.

    Parameters
    ----------
    vec : jax.Array
        Packed symmetric tensor, shape ``(6,)``.

    Returns
    -------
    jax.Array
        Symmetric 3×3 matrix, shape ``(3, 3)``.

    Raises
    ------
    AssertionError
        If ``vec`` does not have exactly 6 elements.
    """
    assert vec.shape == (6,), "Input must be a 6-element vector"

    s11, s22, s33, s12, s13, s23 = vec
    return jnp.array([[s11, s12, s13], [s12, s22, s23], [s13, s23, s33]])


def _generalized_mittag_leffler_function(
    z: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    eps_err: float = 1e-6,
    max_terms: int = 100,
) -> jax.Array:
    """Compute the generalised Mittag-Leffler function E_{α,β}(z).

    The two-parameter Mittag-Leffler function is defined by the power series:

        E_{α,β}(z) = Σ_{k=0}^{∞} z^k / Γ(k·α + β)

    The series is truncated when either the last term is smaller than
    ``eps_err`` in absolute value or ``max_terms`` terms have been computed.
    Iteration is implemented as a JAX ``while_loop`` for JIT compatibility.

    This function is used in the :class:`~diff_rheo.models.GeneralizedPTT`
    model (Ferrás et al. 2019), where it appears in the nonlinear relaxation
    modulus.

    Parameters
    ----------
    z : jax.Array
        Scalar argument of the function.
    alpha : jax.Array
        First parameter α (controls the fractional order).  Must satisfy
        α > 0.
    beta : jax.Array
        Second parameter β (shift in the Gamma function argument).
    eps_err : float
        Convergence tolerance; iteration stops when ``|term| < eps_err``.
        Defaults to ``1e-6``.
    max_terms : int
        Hard upper bound on the number of series terms.  Prevents infinite
        loops when the series converges slowly.  Defaults to ``100``.

    Returns
    -------
    jax.Array
        Approximate value of E_{α,β}(z), shape ``()``.

    Notes
    -----
    The initialisation ``(j=0, term=1.0, sum=0.0)`` means the loop starts by
    adding the ``k=0`` term ``z^0 / Γ(β) = 1/Γ(β)`` on the first iteration.
    The convergence check uses the *previous* term magnitude, so the loop runs
    at least once.
    """

    def body_fn(carry):
        """Computes one term of the series."""
        j, term, sum_so_far = carry
        next_term = (z**j) / gamma(j * alpha + beta)
        next_sum = sum_so_far + next_term
        return j + 1, next_term, next_sum

    def cond_fn(carry):
        """Condition to continue the loop. Stop if the term is small enough or max terms reached."""
        j, term, _ = carry
        return (j < max_terms) & (jnp.abs(term) > eps_err)

    init_val = (
        0,
        1.0,
        0.0,
    )  # (j, term, sum) initial values. Start with j=0, term = 1.0
    _, _, result = jax.lax.while_loop(cond_fn, body_fn, init_val)

    return result


def generate_autoregressive_noise(
    num_time_steps: int,
    key: jax.random.PRNGKey,
    autoregressive_coefficients: list,
    noise_level: float,
) -> jax.Array:
    """Generate an autoregressive AR(p) noise sequence.

    Produces a time series {n_t} following the AR(p) model:

        n_t = Σ_{j=1}^{p} φ_j · n_{t-j} + ε_t

    where ``ε_t ~ N(0, noise_level²)`` are i.i.d. Gaussian innovations and
    ``φ_j`` are the autoregressive coefficients.

    This function is implemented in NumPy (not JAX) due to the sequential
    dependency structure; it is intended for pre-generating synthetic
    experimental noise offline, not inside a JIT-compiled function.

    Parameters
    ----------
    num_time_steps : int
        Length of the noise sequence to generate.
    key : jax.random.PRNGKey
        PRNG key used to draw the Gaussian innovations.
    autoregressive_coefficients : list of float
        AR coefficients ``[φ₁, φ₂, ..., φ_p]`` ordered from lag-1 to lag-p.
        The model order p is inferred from the length of this list.
    noise_level : float
        Standard deviation of the Gaussian innovations ε_t.

    Returns
    -------
    jax.Array
        AR(p) noise sequence, shape ``(num_time_steps,)``.

    Notes
    -----
    Terms where the lag exceeds the current time index (i.e. ``i - j < 0``)
    are silently dropped rather than using a zero-padded history.  This is
    equivalent to assuming the process starts from rest (n_t = 0 for t < 0).
    """
    gaussian_noise = noise_level * jax.random.normal(key, (num_time_steps,))
    noise = np.zeros((num_time_steps,))
    noise[0] = gaussian_noise[0]
    for i in range(1, num_time_steps):
        noise_update = np.sum(
            np.array(
                [
                    autoregressive_coefficients[j - 1] * noise[i - j]
                    for j in range(1, len(autoregressive_coefficients) + 1)
                    if i - j >= 0
                ]
            )
        )
        noise[i] = noise_update + gaussian_noise[i]
    return jnp.array(noise)

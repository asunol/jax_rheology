"""RUDE: Recurrent Unit with Differential Equations.

Implementation of the RUDE (Rheology via Universal Differential Equations) model
from Lennon et al. 2023 (https://doi.org/10.1073/pnas.2304669120).

Architecture Overview
---------------------
RUDE is a **Tensor Basis Neural Network (TBNN)** that acts as the nonlinear
function F in a Generalised Oldroyd-B constitutive equation:

    dτ/dt - L·τ - τ·Lᵀ + F(γ̇, τ) = 0

where τ is the polymer extra stress, L is the velocity gradient, γ̇ is the
rate-of-strain tensor, and F is the learned closure function.

The TBNN represents F as a linear combination of nine integrity-basis tensors:

    F(γ̇, τ) = Σᵢ αᵢ(I₁,...,I₉) · Tᵢ

Tensor Basis
~~~~~~~~~~~~
The nine symmetric integrity-basis tensors are (following Rivlin & Ericksen):

* T₁ = I                                    (identity)
* T₂ = σ                                    (stress)
* T₃ = γ̇                                   (rate of strain)
* T₄ = σ·σ
* T₅ = γ̇·γ̇
* T₆ = σ·γ̇ + γ̇·σ
* T₇ = σ²·γ̇ + γ̇·σ²
* T₈ = σ·γ̇² + γ̇²·σ
* T₉ = σ²·γ̇² + γ̇²·σ²

Invariants
~~~~~~~~~~
The nine scalar invariants fed as input to the neural network are:

* I₁ = tr(σ)
* I₂ = tr(σ²)
* I₃ = tr(γ̇²)
* I₄ = tr(σ³)
* I₅ = tr(γ̇³)
* I₆ = tr(σ²·γ̇²)
* I₇ = tr(σ²·γ̇)
* I₈ = tr(σ·γ̇²)
* I₉ = tr(σ·γ̇)

Network Architecture
~~~~~~~~~~~~~~~~~~~~
The feedforward neural network maps the 9 invariants to 9 scalar coefficients:

    Input (9) → Linear(9→32) → Tanh → Linear(32→32) → Tanh → Linear(32→9)

All layers are bias-free (following Lennon et al.).

Status
------
This model is experimental and partially implemented:

* The TBNN architecture and forward pass are complete.
* The model is **not yet integrated** with the :class:`~diff_rheo._rheometer.VirtualRheometer`
  or :class:`~diff_rheo.models.AbstractViscoelasticModel` interface.
* Training utilities are rudimentary (``zero_init`` helper only).
* No validation against Lennon et al.'s results has been performed.

References
----------
Lennon, K. R., McKinley, G. H., & Swan, J. W. (2023).
Data-driven constitutive models for the rheology of complex fluids.
*PNAS*, 120(32), e2304669120.
https://doi.org/10.1073/pnas.2304669120
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from .._utils import _vector_to_symmetric_matrix
from typing import Callable, Any


class RUDE(eqx.Module):
    """Tensor Basis Neural Network (TBNN) for data-driven constitutive modelling.

    Implements the RUDE architecture from Lennon et al. 2023: a frame-invariant
    neural network closure for the Generalised Oldroyd-B equation.  The network
    maps the nine scalar invariants of (γ̇, τ) to nine scalar coefficients that
    multiply the corresponding integrity-basis tensors, producing the F(γ̇, τ)
    tensor in:

        dτ/dt - L·τ - τ·Lᵀ + F(γ̇, τ) = 0

    The representation is **exactly frame-invariant** by construction: the basis
    tensors transform correctly under a change of reference frame, and the
    invariants are independent of frame.

    Attributes
    ----------
    layers : eqx.nn.Sequential
        The feedforward neural network: Input(9) → Linear(32) → Tanh →
        Linear(32) → Tanh → Linear(9).  All layers use ``use_bias=False``.

    Notes
    -----
    Input tensors (strain-rate γ̇ and stress τ) can be either:

    * 3×3 JAX arrays (full tensor)
    * 6-element JAX arrays (Voigt/packed symmetric notation: [s11, s22, s33,
      s12, s13, s23]), which are automatically unpacked to 3×3.

    Examples
    --------
    Construct with random initialisation:

    >>> key = jax.random.PRNGKey(42)
    >>> model = RUDE(key)

    Construct with zero-initialised weights (produces zero output everywhere,
    useful as a starting point for training):

    >>> model = RUDE.zero_init()

    Evaluate F(γ̇, τ):

    >>> gamma_dot = jnp.zeros((3, 3))
    >>> tau = jnp.zeros((3, 3))
    >>> F = model(gamma_dot, tau)  # shape (3, 3)
    """

    layers: eqx.nn.Sequential

    def __init__(self, key: jax.random.PRNGKey):
        """Initialise RUDE with random layer weights.

        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key used to initialise the three linear layers.
        """
        key1, key2, key3 = jax.random.split(key, 3)
        self.layers = eqx.nn.Sequential([
            eqx.nn.Linear(9, 32, key=key1, use_bias=False),
            eqx.nn.Lambda(jax.nn.tanh),
            eqx.nn.Linear(32, 32, key=key2, use_bias=False),
            eqx.nn.Lambda(jax.nn.tanh),
            eqx.nn.Linear(32, 9, key=key3, use_bias=False),
        ])

    @classmethod
    def zero_init(cls) -> "RUDE":
        """Create a RUDE model with all weights initialised to zero.

        A zero-initialised model outputs F = 0 for all inputs, which
        corresponds to the Upper-Convected Maxwell equation with no
        nonlinear correction.  This is often a convenient starting point
        for training because it starts from a physically meaningful state.

        Returns
        -------
        RUDE
            A new RUDE instance with all linear layer weights set to zero.
        """
        key = jax.random.PRNGKey(0)
        return init_linear_weights(cls(key), zero_init, key)

    def __call__(self, strain: jax.Array, stress: jax.Array) -> jax.Array:
        """Evaluate the nonlinear closure tensor F(γ̇, τ).

        1. Computes the 9 scalar invariants from (strain, stress).
        2. Passes them through the neural network to get 9 coefficients α.
        3. Constructs the 9 integrity-basis tensors from (strain, stress).
        4. Returns F = Σᵢ αᵢ · Tᵢ.

        Parameters
        ----------
        strain : jax.Array
            Rate-of-strain tensor γ̇.  Shape ``(3, 3)`` or ``(6,)``.
        stress : jax.Array
            Polymer extra-stress tensor τ.  Shape ``(3, 3)`` or ``(6,)``.

        Returns
        -------
        jax.Array
            The closure tensor F(γ̇, τ), shape ``(3, 3)``.
        """
        x = self._calculate_invariants(strain, stress)
        x = self.layers(x)
        basis = self._generate_tensor_basis(stress, strain)
        f_tensor = jnp.sum(x[:, None, None] * basis, axis=0)
        return f_tensor

    def _generate_tensor_basis(self, gamma: jax.Array, sigma: jax.Array) -> jax.Array:
        """Construct the 9 integrity-basis tensors from γ̇ and σ.

        After preprocessing both inputs to 3×3 form, returns the stack:

            [I, σ, γ̇, σ², γ̇², σγ̇+γ̇σ, σ²γ̇+γ̇σ², σγ̇²+γ̇²σ, σ²γ̇²+γ̇²σ²]

        Parameters
        ----------
        gamma : jax.Array
            Rate-of-strain tensor.  Shape ``(3, 3)`` or ``(6,)``.
        sigma : jax.Array
            Stress tensor.  Shape ``(3, 3)`` or ``(6,)``.

        Returns
        -------
        jax.Array
            Stacked basis tensors, shape ``(9, 3, 3)``.
        """
        gamma, sigma = self._preprocess_tensors([gamma, sigma])
        T1 = jnp.identity(3)
        T2 = sigma
        T3 = gamma
        T4 = sigma @ sigma
        T5 = gamma @ gamma
        T6 = sigma @ gamma + gamma @ sigma
        T7 = sigma @ sigma @ gamma + gamma @ sigma @ sigma
        T8 = sigma @ gamma @ gamma + gamma @ gamma @ sigma
        T9 = sigma @ sigma @ gamma @ gamma + gamma @ gamma @ sigma @ sigma
        basis_tensors = jnp.stack([T1, T2, T3, T4, T5, T6, T7, T8, T9], axis=0)
        return basis_tensors

    def _calculate_invariants(self, gamma: jax.Array, sigma: jax.Array) -> jax.Array:
        """Compute the 9 scalar frame-invariants of (γ̇, σ).

        The invariants are traces of polynomial combinations of γ̇ and σ:

            I₁ = tr(σ),  I₂ = tr(σ²),  I₃ = tr(γ̇²),
            I₄ = tr(σ³), I₅ = tr(γ̇³), I₆ = tr(σ²γ̇²),
            I₇ = tr(σ²γ̇), I₈ = tr(σγ̇²), I₉ = tr(σγ̇)

        Parameters
        ----------
        gamma : jax.Array
            Rate-of-strain tensor.  Shape ``(3, 3)`` or ``(6,)``.
        sigma : jax.Array
            Stress tensor.  Shape ``(3, 3)`` or ``(6,)``.

        Returns
        -------
        jax.Array
            Nine scalar invariants, shape ``(9,)``, dtype ``float32``.
        """
        gamma, sigma = self._preprocess_tensors([gamma, sigma])
        l1 = jnp.trace(sigma)
        l2 = jnp.trace(sigma @ sigma)
        l3 = jnp.trace(gamma @ gamma)
        l4 = jnp.trace(sigma @ sigma @ sigma)
        l5 = jnp.trace(gamma @ gamma @ gamma)
        l6 = jnp.trace(sigma @ sigma @ gamma @ gamma)
        l7 = jnp.trace(sigma @ sigma @ gamma)
        l8 = jnp.trace(sigma @ gamma @ gamma)
        l9 = jnp.trace(sigma @ gamma)
        return jnp.array([l1, l2, l3, l4, l5, l6, l7, l8, l9]).astype("f")

    def _preprocess_tensors(self, tensor_list: list) -> list:
        """Convert tensors from packed 6-vector form to 3×3 matrices if needed.

        Accepts each tensor in either of two formats:

        * ``(3, 3)`` – returned unchanged.
        * ``(6,)``   – unpacked as [s11, s22, s33, s12, s13, s23] and
          reconstructed as a symmetric 3×3 matrix via
          :func:`~diff_rheo._utils._vector_to_symmetric_matrix`.

        Parameters
        ----------
        tensor_list : list of jax.Array
            Input tensors to preprocess.

        Returns
        -------
        list of jax.Array
            Tensors all in ``(3, 3)`` form.

        Raises
        ------
        ValueError
            If any tensor has a shape other than ``(3, 3)`` or ``(6,)``.
        """
        output_tensors = []
        for tensor in tensor_list:
            if tensor.shape == (3, 3):
                output_tensors.append(tensor)
            elif tensor.shape == (6,):
                output_tensors.append(_vector_to_symmetric_matrix(tensor))
            else:
                raise ValueError("Input tensors must be 3x3 or 6x1")
        return output_tensors


def init_linear_weights(model: RUDE, init_fn: Callable, key: jax.random.PRNGKey) -> RUDE:
    """Reinitialise all linear layer weights in a RUDE model.

    Traverses the Equinox pytree of ``model``, finds all
    :class:`equinox.nn.Linear` leaf modules, applies ``init_fn`` to each
    weight matrix (with an independent subkey), and returns a new model with
    the updated weights.

    Parameters
    ----------
    model : RUDE
        The model whose weights will be reinitialised.
    init_fn : Callable[[jax.Array, jax.random.PRNGKey], jax.Array]
        Initialisation function with signature ``(weight, key) -> new_weight``.
        See :func:`zero_init` for an example.
    key : jax.random.PRNGKey
        PRNG key; split internally to produce independent subkeys for each
        layer.

    Returns
    -------
    RUDE
        A new RUDE instance identical to ``model`` except for the reinitialised
        weight matrices.
    """
    def is_linear(x: Any) -> bool:
        return isinstance(x, eqx.nn.Linear)
    def get_weights(m: RUDE) -> list[jax.Array]:
        return [x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
    weights = get_weights(model)
    subkeys = jax.random.split(key, len(weights))
    new_weights = [init_fn(weight, subkey) for weight, subkey in zip(weights, subkeys)]
    return eqx.tree_at(get_weights, model, new_weights)


def zero_init(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
    """Return a zero array with the same shape and dtype as ``weight``.

    Used as the ``init_fn`` argument to :func:`init_linear_weights` when
    constructing a zero-initialised :class:`RUDE` model via
    :meth:`RUDE.zero_init`.

    Parameters
    ----------
    weight : jax.Array
        The weight array whose shape and dtype to match.
    key : jax.random.PRNGKey
        Unused; present for API compatibility with other init functions.

    Returns
    -------
    jax.Array
        Zero array of the same shape and dtype as ``weight``.
    """
    return jnp.zeros_like(weight)

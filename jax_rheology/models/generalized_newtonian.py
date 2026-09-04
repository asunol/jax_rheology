"""Generalized-Newtonian viscosity laws and the shared viscosity dispatch.

Three closed-form viscosity laws, each with its stress tensor and the body
force the solver actually consumes:

  1) Newtonian (constant viscosity)
  2) Carreau-Yasuda
  3) Power-law

Also here: the grid and tensor helpers all three share (rate-of-strain and
vorticity, invariants, tensor divergence), and the viscosity / ``nu0``
selection used by every generalized-Newtonian path.

The learned instantaneous closure is *not* implemented in this module. It
lives in :mod:`jax_rheology.models.tbnn_instantaneous`; the dispatch in
:func:`get_viscosity_field` and :func:`create_dynamic_nu0_fn` accepts
``'TBNN'`` and delegates to it, which is why that module is imported at the
bottom of this file rather than at the top (the two import each other).
"""

import jax
import jax.numpy as jnp
import jax.nn as jnn
from flax import linen as nn
from flax.core import freeze, unfreeze
from typing import Optional, Any, List, Tuple, Sequence
# Removed jdebug import as it's no longer needed

import jax_ib.base as ib
from jax_ib.base import diffusion
from jax.tree_util import tree_map, tree_leaves



# =============================================================================
# SECTION 1 -- Grid/Tensor utilities (shared by all three laws)
# =============================================================================

def compute_S_R(v):
    """Compute strain-rate S and spin R from velocity field container."""
    grad = ib.finite_differences.gradient_tensor(v)
    S = (grad + grad.T) / 2.
    R = (grad - grad.T) / 2.
    return S, R

def extract_data(T):
    """
    Correctly extracts data from a GridArrayTensor into a single, dense JAX array.

    This function unpacks the container object T using a list comprehension--the
    standard pattern for this data structure--and converts the result into a
    dense JAX array.
    """
    rows, cols = T.shape
    
    # This is the robust version of your original, working pattern.
    list_of_lists = [[T[i][j].data for j in range(cols)] for i in range(rows)]
    
    return jnp.array(list_of_lists)

def compute_invariants_2d(S, R):
    """Stable invariant calc: sum of squares (Tr(TTT)) style."""
    S_data = extract_data(S)
    R_data = extract_data(R)
    I1 = jnp.sum(S_data**2, axis=(0, 1))
    # For skew-symmetric R, Tr(R^2) < 0; we keep the classic sign convention:
    I2 = -jnp.sum(R_data**2, axis=(0, 1))
    return jnp.array([I1, I2])

def tensor_divergence(T):
    """div tau for a 2x2 GridArrayTensor -> (GridArray, GridArray).
    Uses central differences on the GridVariables held in the container.
    """
    dTxx_dx = ib.finite_differences.central_difference(T[0][0], axis=0)
    dTyx_dy = ib.finite_differences.central_difference(T[1][0], axis=1)
    dTxy_dx = ib.finite_differences.central_difference(T[0][1], axis=0)
    dTyy_dy = ib.finite_differences.central_difference(T[1][1], axis=1)
    fx = dTxx_dx + dTyx_dy
    fy = dTxy_dx + dTyy_dy
    return (fx, fy)

# =============================================================================
# SECTION 2 -- Newtonian (constant viscosity)
# =============================================================================

def newtonian_stress_tensor(S, eta: float):
    """Newtonian stress tau = 2 eta S for constant viscosity eta."""
    S_data = extract_data(S)
    return 2.0 * eta * S_data  # (2,2,H,W)

def newtonian_stress_forcing(pressure_gradient, permeability, model, params, factor, U_f, nu0):
    """
    Newtonian stress forcing (explicit div tau_Newt with implicit nu0grad^2u):
        return px - kappa(u-Uf) + [factor * div tau_Newt(u) - nu0 grad^2u]
    Notes:
      - `params` is the constant viscosity eta.
      - Use this in IMEX-style loops when you still want an explicit/implicit split.
    """
    eta = params if jnp.isscalar(params) else float(params)

    def forcing(v):
        S, _ = compute_S_R(v)
        stress_tensor = newtonian_stress_tensor(S, eta)

        # Pack dense stress -> GridArrayTensor w/ velocity BCs
        stress_grid = ib.finite_differences.GridArrayTensor([
            [
                ib.grids.GridVariable(
                    ib.grids.GridArray(stress_tensor[i, j], S[i][j].offset, v[0].grid),
                    v[0].bc
                )
                for j in range(S.shape[1])
            ]
            for i in range(S.shape[0])
        ])

        div_stress_full = tensor_divergence(stress_grid)
        aligned_div = tuple(
            ib.interpolation.linear(ib.grids.GridVariable(stress, u.bc), u.offset)
            for stress, u in zip(div_stress_full, v)
        )

        # Implicit part nu0grad^2u
        div_stress_implicit = tuple(diffusion.diffuse(u, nu0) for u in v)

        # Explicit correction
        rheology_corr = tuple(factor * alg.data - impl.data for alg, impl in zip(aligned_div, div_stress_implicit))

        # Final forcing
        return tuple(
            ib.grids.GridArray(
                pxn * jnp.ones_like(u.data) - permeability * (u.data - U_f) + corr,
                u.offset, u.grid
            )
            for corr, pxn, u in zip(rheology_corr, pressure_gradient, v)
        )
    return forcing


# =============================================================================
# SECTION 3 -- Carreau-Yasuda (CY)
# =============================================================================

def compute_shear_rate(S, eps=1e-6):
    """
    Compute shear rate magnitude from strain rate tensor.
    Shear rate = sqrt(2 * S_ij * S_ij)
    """
    S_data = extract_data(S)
    first_invariant = jnp.sum(S_data**2, axis=(0, 1))
    return jnp.sqrt(2 * first_invariant + eps**2)

def carreau_yasuda_viscosity(shear_rate, eta_inf, eta_0, lambda_, n, a):
    """
    Carreau-Yasuda: eta(gammadot) = eta_inf + (eta_0 - eta_inf) [1 + (lamgammadot)^a]^((n-1)/a)
    Log/exp + clamps keep AD stable for wide ranges.
    """
    g   = jnp.clip(shear_rate, 0.0, 1e12)
    a   = jnp.clip(a,    1e-6, 100.0)
    lam = jnp.clip(lambda_, 1e-12, 1e6)
    x = jnp.clip(lam * g, 0.0, 1e12)
    term = jnp.exp(((n - 1.0) / a) * jnp.log1p(jnp.power(x, a)))
    return eta_inf + (eta_0 - eta_inf) * term

# For compatibility - alias to the above
compute_carreau_yasuda_viscosity = carreau_yasuda_viscosity

def carreau_yasuda_stress_tensor(S, eta_inf, eta_0, lambda_, n, a):
    """tau = 2 eta(gammadot) S."""
    shear_rate = compute_shear_rate(S)
    viscosity = carreau_yasuda_viscosity(shear_rate, eta_inf, eta_0, lambda_, n, a)
    S_data = extract_data(S)
    return 2 * viscosity * S_data  # (2,2,H,W)

def carreau_yasuda_stress_forcing(pressure_gradient, permeability, model, params, factor, U_f, nu0):
    """
    Forcing with explicit div tau(CY) and implicit nu0grad^2u:
        return px - kappa(u-Uf) + [factor * div tau_CY(u) - nu0 grad^2u]
    """
    eta_inf, eta_0, lambda_, n, a = params[:5]

    def forcing(v):
        S, _ = compute_S_R(v)
        stress_tensor = carreau_yasuda_stress_tensor(S, eta_inf, eta_0, lambda_, n, a)

        # Pack dense stress -> GridArrayTensor w/ velocity BCs
        stress_grid = ib.finite_differences.GridArrayTensor([
            [
                ib.grids.GridVariable(
                    ib.grids.GridArray(stress_tensor[i, j], S[i][j].offset, v[0].grid),
                    v[0].bc
                )
                for j in range(S.shape[1])
            ]
            for i in range(S.shape[0])
        ])

        div_stress_full = tensor_divergence(stress_grid)
        aligned_div = tuple(
            ib.interpolation.linear(ib.grids.GridVariable(stress, u.bc), u.offset)
            for stress, u in zip(div_stress_full, v)
        )

        # Implicit part nu0grad^2u
        div_stress_implicit = tuple(diffusion.diffuse(u, nu0) for u in v)

        # Explicit correction
        rheology_corr = tuple(factor * alg.data - impl.data for alg, impl in zip(aligned_div, div_stress_implicit))

        # Final forcing
        return tuple(
            ib.grids.GridArray(
                pxn * jnp.ones_like(u.data) - permeability * (u.data - U_f) + corr,
                u.offset, u.grid
            )
            for corr, pxn, u in zip(rheology_corr, pressure_gradient, v)
        )
    return forcing

# =============================================================================
# SECTION 4 -- Power-law
# =============================================================================

def compute_power_law_viscosity(S, K, n, regularization=1e-6):
    """
    Apparent viscosity: eta_app = K * |gammadot|^(n-1)
    Uses |gammadot|^2 = S:S * 2 with a small regularization to avoid singularities.
    """
    S_data = extract_data(S)
    shear_rate_squared = jnp.sum(S_data**2, axis=(0, 1)) + regularization**2
    return K * shear_rate_squared**((n - 1) / 2)

def power_law_stress_tensor(S, K, n, regularization=1e-6):
    """tau = 2 eta_app S with regularized shear rate."""
    apparent_viscosity = compute_power_law_viscosity(S, K, n, regularization)
    S_data = extract_data(S)
    return 2 * apparent_viscosity * S_data

def power_law_stress_forcing(pressure_gradient, permeability, model, params, factor, U_f, nu0):
    """
    Forcing with explicit div tau(PL) and implicit nu0grad^2u:
        return px - kappa(u-Uf) + [factor * div tau_PL(u) - nu0 grad^2u]
    """
    K, n = params[0], params[1]

    def forcing(v):
        S, _ = compute_S_R(v)
        stress_tensor = power_law_stress_tensor(S, K, n, regularization=1e-8)

        stress_grid = ib.finite_differences.GridArrayTensor([
            [
                ib.grids.GridVariable(
                    ib.grids.GridArray(stress_tensor[i, j], S[i][j].offset, v[0].grid),
                    v[0].bc
                )
                for j in range(S.shape[1])
            ]
            for i in range(S.shape[0])
        ])

        div_stress_full = tensor_divergence(stress_grid)
        aligned_div = tuple(
            ib.interpolation.linear(ib.grids.GridVariable(stress, u.bc), u.offset)
            for stress, u in zip(div_stress_full, v)
        )

        div_stress_implicit = tuple(diffusion.diffuse(u, nu0) for u in v)
        rheology_corr = tuple(factor * alg.data - impl.data for alg, impl in zip(aligned_div, div_stress_implicit))

        return tuple(
            ib.grids.GridArray(
                pxn * jnp.ones_like(u.data) - permeability * (u.data - U_f) + corr,
                u.offset, u.grid
            )
            for corr, pxn, u in zip(rheology_corr, pressure_gradient, v)
        )
    return forcing


# =============================================================================
# SECTION 5 -- Viscosity access & nu0 selection (shared utilities)
# =============================================================================
def fully_implicit_forcing(pressure_gradient, permeability, U_f):
    """
    Create forcing function for fully implicit BE-IMEX (variable-coefficient solver).
    Returns only non-viscous forcing; viscous terms are handled implicitly inside the solver:
        return px - kappa(u-Uf)
    """
    def forcing(v):
        return tuple(
            ib.grids.GridArray(
                pxn * jnp.ones_like(u.data) - permeability * (u.data - U_f),
                u.offset,
                u.grid,
            )
            for pxn, u in zip(pressure_gradient, v)
        )
    return forcing

def get_viscosity_field(velocity, params, model_type, model=None, eta_floor=1e-3, eta_cap=1e3):
    """
    Compute the spatial viscosity field eta(x) for a given model type.
    Supported: 'newtonian', 'power_law', 'carreau_yasuda', 'TBNN' (bounded).
    """
    S, _ = compute_S_R(velocity)

    if model_type == 'newtonian':
        # Constant viscosity field for fully implicit solves, etc.
        grid_shape = velocity[0].array.data.shape
        # Scalar-cast form (kept commented): broke when ``params``
        # (the Newtonian viscosity) became a traced JAX value during
        # ``jax.grad`` w.r.t. ``nu_s``, because ``float(traced_array)``
        # triggers ConcretizationTypeError.
        # viscosity_field = jnp.full(grid_shape, params if jnp.isscalar(params) else float(params))
        viscosity_field = jnp.full(grid_shape, jnp.asarray(params, dtype=velocity[0].array.data.dtype))

    elif model_type == 'power_law':
        K, n = params[0], params[1]
        viscosity_field = compute_power_law_viscosity(S, K, n, regularization=1e-8)

    elif model_type == 'carreau_yasuda':
        eta_inf, eta_0, lambda_, n, a = params[:5]
        shear_rate = compute_shear_rate(S)
        viscosity_field = carreau_yasuda_viscosity(shear_rate, eta_inf, eta_0, lambda_, n, a)

    elif model_type in ('TBNN', 'tbnn_bounded'):
        if model is None:
            raise ValueError("TBNN model object must be provided for viscosity field calculation.")
        viscosity_field = tbnn_eta_bounded_from_v(velocity, model, params)

    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. Choose from: "
            "'newtonian', 'power_law', 'carreau_yasuda', 'TBNN'"
        )

    return jnp.clip(viscosity_field, eta_floor, eta_cap)

def _validate_nu0(nu0):
    """Ensure nu0 is finite and positive (small floor keeps operator well-posed)."""
    nu0 = jnp.where(jnp.isfinite(nu0), nu0, 1.0)
    return jnp.maximum(nu0, 1e-6)

def static_nu0_update_fn(velocity, params):
    """
    Static nu0 function (returns 0.0). Use when you *don't* want an IMEX split, e.g.,
    fully implicit variable-coefficient solves with `fully_implicit_forcing`.
    """
    return 0.0

def create_dynamic_nu0_fn(model_type, model=None, strategy: str = 'max', C: float = 1.0):
    """
    Factory for nu0(velocity, params) depending on the rheology:
      - 'newtonian'      : returns static 0.0 (use fully implicit, no split)
      - 'power_law'      : nu0 = C * max(eta_app) or mean heuristic
      - 'carreau_yasuda' : nu0 = C * max(eta(gammadot)) or mean heuristic
      - 'TBNN'/'tbnn_bounded': nu0 = C * quantile(eta, 0.99) or mean heuristic
    """
    if model_type == 'newtonian':
        return static_nu0_update_fn

    if model_type == 'power_law':
        def fn(velocity, params):
            S, _ = compute_S_R(velocity)
            viscosity_field = compute_power_law_viscosity(S, params[0], params[1])
            if strategy == 'max':
                nu0 = C * jnp.max(viscosity_field)
            elif strategy == 'mean':
                nu0 = (0.2 + jnp.mean(viscosity_field)) / 2.0
            else:
                raise ValueError(f"Unknown nu0 strategy: '{strategy}'.")
            return _validate_nu0(nu0)
        return fn

    if model_type == 'carreau_yasuda':
        def fn(velocity, params):
            S, _ = compute_S_R(velocity)
            shear_rate = compute_shear_rate(S)
            viscosity_field = carreau_yasuda_viscosity(shear_rate, *params)
            if strategy == 'max':
                nu0 = C * jnp.max(viscosity_field)
            elif strategy == 'mean':
                nu0 = (0.2 + jnp.mean(viscosity_field)) / 2.0
            else:
                raise ValueError(f"Unknown nu0 strategy: '{strategy}'.")
            return _validate_nu0(nu0)
        return fn

    if model_type in ('TBNN', 'tbnn_bounded'):
        if model is None:
            raise ValueError("TBNN model object must be provided for a dynamic nu0 function.")
        def fn(velocity, params):
            eta_field = tbnn_eta_bounded_from_v(velocity, model, params)
            if strategy == 'max':
                nu0 = C * jnp.quantile(eta_field, 0.99)
            elif strategy == 'mean':
                nu0 = (0.2 + jnp.mean(eta_field)) / 2.0
            else:
                raise ValueError(f"Unknown nu0 strategy: '{strategy}'.")
            return _validate_nu0(nu0)
        return fn

    # Fallback to a harmless static zero if an unknown type is passed.
    return static_nu0_update_fn
from jax_rheology.models.tbnn_instantaneous import tbnn_eta_bounded_from_v

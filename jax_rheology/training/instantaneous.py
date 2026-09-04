#!/usr/bin/env python
"""Train the instantaneous TBNN closure on constricted-channel flow.

The training loop behind the generalized-Newtonian results: a reference fluid
is simulated to steady state to give the observed velocity field, then the
mixture-of-sigmoids viscosity closure is fitted by differentiating through the
forward solver until its velocity field matches.

The closure is
:class:`~jax_rheology.models.tbnn_instantaneous.BoundedSlopeViscosity`, whose
viscosity is bounded and monotone by construction; see that module for the
functional form.

Supported reference fluids, with the parameters each expects:

- ``'newtonian'``      -- viscosity (scalar)
- ``'carreau_yasuda'`` -- ``[eta_inf, eta_0, lam, n, a]``
- ``'power_law'``      -- ``[K, n]``

Also here: the loss functions (masked field RMSE, and the shape loss that
compares profiles up to a scale), the observation masks that select which part
of the domain the loss sees, the optional two-stage schedule that fits the
high-shear plateau before the curvature, and the Carreau-Yasuda shape
pre-training used to start near a sensible curve.

Entry points are ``debug_one_step_gradient_constriction`` for a single
gradient step and ``demo_gradient_debugging_constriction`` for a full run;
the cluster runners in ``campaigns/`` call the latter.
"""

import jax
jax.config.update("jax_debug_nans", True)
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
_VERBOSE = ("--verbose" in sys.argv) or bool(os.environ.get("JAX_RHEOLOGY_VERBOSE"))
import time
import traceback
import functools
from jax import grad, jit, value_and_grad
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax import traverse_util

# Try Optax for Adam optimizer
try:
    import optax
    HAVE_OPTAX = True
except ImportError:
    HAVE_OPTAX = False
    print("Optax not available, will use fallback Adam implementation")

# Disable JIT for debugging (can be enabled for performance)
jax.config.update('jax_disable_jit', False)

# Set working directory and paths
from repo_paths import bootstrap, REPO_ROOT
bootstrap()
if _VERBOSE:
    print("Working directory:", os.getcwd())

# Import the rheology modules
from jax_rheology.core import flow_conditions
from jax_rheology.forward import generic as forward_simulation
from jax_rheology.core import params as parameter_utils
from jax_rheology import models
from jax_ib.base import particle_class as pc, grids, kinematics as ks, boundaries
from jax_ib.base import advection, diffusion
from jax_rheology.solvers import steppers as equations_rheology
from jax_rheology.solvers import pressure
import jax_cfd.base as cfd
import jax_ib.penalty.util_funs

# Import plotting utilities (file-path load so paper_figs/__init__.py is not imported)
import importlib.util as _ilu
_plot_src = REPO_ROOT / "paper_figs" / "training_plots.py"
_plot_spec = _ilu.spec_from_file_location("paper_figs_training_plots", _plot_src)
_plot_mod = _ilu.module_from_spec(_plot_spec)
assert _plot_spec.loader is not None
_plot_spec.loader.exec_module(_plot_mod)
plotting = _plot_mod
from jax_rheology.geometries.constricted_channel import setup_channel_constriction
from jax_rheology.training.observation import piv_downsample_THW, add_piv_noise_jax


if _VERBOSE:
    print("Instantaneous trainer initialized "
          "(mixture-of-sigmoids viscosity closure).")

# --mask-layout: canonical names are constriction_focused (paper
# weighting, solver-native (T, nx, ny)) and full_domain (PIV-twin
# transpose onto (T, ny, nx)). Hidden aliases keep old CLI strings
# runnable.
MASK_LAYOUT_ALIASES = {
    "legacy": "constriction_focused",
    "correct": "full_domain",
}
MASK_LAYOUT_CANONICAL = ("full_domain", "constriction_focused")


def resolve_mask_layout(mask_layout: str, *, log: bool = False) -> str:
    """Map deprecated aliases; return a canonical layout name."""
    alias_of = MASK_LAYOUT_ALIASES.get(mask_layout)
    if alias_of is not None:
        if log:
            print(f"DEPRECATED: --mask-layout {mask_layout} -> {alias_of}")
        return alias_of
    if mask_layout not in MASK_LAYOUT_CANONICAL:
        raise ValueError(
            "mask_layout must be 'full_domain' or 'constriction_focused' "
            f"(aliases: legacy, correct); got {mask_layout!r}"
        )
    return mask_layout

# =============================================================================
# VELOCITY SHAPE LOSS UTILITIES
# =============================================================================

def _stopgrad_mean(x):
    """Safe mean with stop_gradient so scale estimate doesn't backprop."""
    return jax.lax.stop_gradient(jnp.mean(x))

def normalize_field(x, mode='mean', eps=1e-8):
    """Normalize x to remove global scale. No grads flow through the scale."""
    if mode == 'mean':
        denom = _stopgrad_mean(jnp.abs(x)) + eps
    elif mode == 'l2':
        denom = jax.lax.stop_gradient(jnp.sqrt(jnp.mean(x**2)) + eps)
    elif mode == 'max':
        denom = jax.lax.stop_gradient(jnp.max(jnp.abs(x)) + eps)
    else:
        raise ValueError(f'Unknown shape_norm {mode}')
    return x / denom

def cosine_distance(a, b, eps=1e-8):
    """Cosine distance: 1 - cosine_similarity."""
    a = a - _stopgrad_mean(a)
    b = b - _stopgrad_mean(b)
    num = jnp.sum(a * b)
    den = (jnp.sqrt(jnp.sum(a*a)) + eps) * (jnp.sqrt(jnp.sum(b*b)) + eps)
    return 1.0 - num / den

def corr_distance(a, b, eps=1e-8):
    """Pearson correlation distance: 1 - corr."""
    a = a - jnp.mean(a)
    b = b - jnp.mean(b)
    num = jnp.sum(a * b)
    den = (jnp.sqrt(jnp.sum(a*a)) + eps) * (jnp.sqrt(jnp.sum(b*b)) + eps)
    return 1.0 - num / den

def rmse(a, b):
    """Root mean squared error."""
    return jnp.sqrt(jnp.mean((a - b)**2))

def velocity_shape_loss(u_pred, u_true, *,
                        norm_mode='mean',
                        metric='cosine',
                        use_speed=True):
    """
    Compare SHAPE of velocity ignoring global scale.
    
    Args:
        u_pred: Predicted velocity field (can be scalar or [..., 2] for 2D)
        u_true: True velocity field (same shape as u_pred)
        norm_mode: How to normalize ('mean', 'l2', or 'max')
        metric: Distance metric ('cosine', 'corr', or 'rmse')
        use_speed: If True, compare speed ||u||. If False, compare components.
    
    Returns:
        Shape distance (scalar)
    """
    # Accept either [..., 2] vector fields or scalar speed
    def _to_components(u):
        if u.ndim >= 1 and u.shape[-1] == 2:
            return u[..., 0], u[..., 1]
        return (u,)  # already scalar

    if use_speed:
        if u_pred.ndim >= 1 and u_pred.shape[-1] == 2:
            sp = jnp.sqrt(jnp.sum(u_pred**2, axis=-1))
            st = jnp.sqrt(jnp.sum(u_true**2, axis=-1))
        else:
            sp, st = u_pred, u_true
        sp_n = normalize_field(sp, norm_mode)
        st_n = normalize_field(st, norm_mode)
        if metric == 'cosine':
            return cosine_distance(sp_n.ravel(), st_n.ravel())
        elif metric == 'corr':
            return corr_distance(sp_n.ravel(), st_n.ravel())
        elif metric == 'rmse':
            return rmse(sp_n, st_n)
        else:
            raise ValueError(f'Unknown shape_metric {metric}')
    else:
        comps_p = _to_components(u_pred)
        comps_t = _to_components(u_true)
        losses = []
        for cp, ct in zip(comps_p, comps_t):
            cp_n = normalize_field(cp, norm_mode)
            ct_n = normalize_field(ct, norm_mode)
            if metric == 'cosine':
                d = cosine_distance(cp_n.ravel(), ct_n.ravel())
            elif metric == 'corr':
                d = corr_distance(cp_n.ravel(), ct_n.ravel())
            elif metric == 'rmse':
                d = rmse(cp_n, ct_n)
            else:
                raise ValueError(f'Unknown shape_metric {metric}')
            losses.append(d)
        return jnp.mean(jnp.stack(losses))

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

# Consistent bounds for TBNN viscosity initialization and decoding
ETA_LO_SCALE = 0.01  # eta_lo = 0.01 * eta_ref
ETA_HI_SCALE = 10.0   # eta_hi = 2.0 * eta_ref

# === TBNN Configuration ===
# Use defaults that align with models.py (BoundedSlopeViscosity - mixture-of-sigmoids)
DEFAULT_TBNN_CONFIG = {
    'M': 4,                    # number of sigmoid terms in mixture (default: 4)
    'eta_min': 1e-2,
    'eta_max': 10.0,
    'smax': 0.5,               # UNUSED in new model (kept for API compatibility)
    'zmax': 8.0,               # UNUSED in new model (kept for API compatibility)
    'gamma_ref': 1.0,
    's_floor': 0.0,            # minimum logistic width (0.5-0.8 for broader, smoother kernels)
    'alpha_temp': 1.0          # softmax temperature (>1 spreads weights, <1 sharpens)
}

DEFAULT_CONFIG = {
    'domain': ((0, 8.0), (0, 4.0)),
    'domain_size': (256, 128),
    'density': 1.0,
    'dt': 1e-5,
    'inner_steps': 400,
    'outer_steps': 50,
    'solver_type': 'bicgstab',
    'stepper_type': 'fully_implicit',
    'boundary_type': 'moving_wall',
    'use_preconditioner': False,
    'preconditioner_type': 'none'
}

# Functions moved to models.py - using centralized versions

def inspect_head(tbnn, params, gamma_probe=1.0):
    """Return (eta_inf, eta0, delta, mu_vec) from the model itself."""
    # Probe once; we capture sowed intermediates
    gamma_img_probe = jnp.asarray(gamma_probe, dtype=jnp.float32).reshape(1, 1)
    invariants_probe = jnp.stack([jnp.array([0.5*gamma_probe**2]).reshape(1,1),
                                  jnp.array([-0.5*gamma_probe**2]).reshape(1,1)], axis=0)
    _, inter = tbnn.apply(params, gamma_img_probe, invariants_probe,
                          mutable=['intermediates'])
    mu_vec   = np.array(inter['intermediates']['mu_snapshot'][-1])          # (K,)
    eta_inf  = float(np.array(inter['intermediates']['eta_inf_value'][-1]))
    eta0     = float(np.array(inter['intermediates']['eta0_value'][-1]))
    delta    = float(np.array(inter['intermediates']['delta_value'][-1]))
    return eta_inf, eta0, delta, mu_vec

def setup_tbnn_model(hidden_units, domain_size, eta_init=1.0, random_seed=42, #used to be 42
                     use_soft_newtonian_init=False, s_floor=0.0, alpha_temp=1.0,
                     freeze_eta0=False, eta0_fixed=1.0, eta0_eps=1e-6,
                     mu_min_gamma=None, mu_max_gamma=None, gate_gamma=None, gate_width_z=0.5,
                     tail_gate_gamma=None, tail_gate_width_z=0.5,
                     enable_pl_per_mode=False, pl_width_z=0.5,
                     log_head=False, log_mixing="add", freeze_centers=False, M=4):
    """Setup TBNN model with specified eta initialization using BoundedSlopeViscosity (mixture-of-sigmoids).
    
    Note: The new model uses a mixture-of-sigmoids approach for stable, monotone viscosity.
    Parameters smax and zmax are no longer used but kept for API compatibility.
    
    Args:
        s_floor: Minimum logistic width (0.5-0.8 for broader, smoother kernels)
        alpha_temp: Softmax temperature for mixture weights (>1 spreads, <1 sharpens)
        freeze_eta0: If True, freeze eta0 = eta_inf + delta (only learn curvature)
        eta0_fixed: Fixed value of eta0 when freeze_eta0=True
        eta0_eps: Small positive floor for delta when frozen
        mu_min_gamma: Lower bound on center locations (no curvature before this gammadot)
        mu_max_gamma: Optional upper bound on center locations
        gate_gamma: If set, multiply mixture by smooth gate starting at this gammadot
        gate_width_z: Gate smoothness in log(gammadot) space (default: 0.5)
        tail_gate_gamma: If set, delay tail (etainf) to this gammadot (prevents mid-shear drag)
        tail_gate_width_z: Tail gate smoothness in log(gammadot) space (default: 0.5)
        enable_pl_per_mode: If True, add per-mode power-law bumps for extra expressiveness
        pl_width_z: Smooth onset width for each mode's PL bump (default: 0.5)
        log_head: If True, learn in log-eta space (more stable, weights decades evenly) (default: False)
        log_mixing: Mixing mode for log_head - "add" or "geom" (default: "add")
        freeze_centers: If True, freeze mu centers (learn only s/alpha, not locations) (default: False)
        M: Number of sigmoid terms in the mixture (default: 4)
    """
    # Set random seed
    key = jax.random.PRNGKey(random_seed)
    
    # Create bounded slope viscosity model (mixture-of-sigmoids version)
    tbnn_config = DEFAULT_TBNN_CONFIG.copy()
    tbnn_config['M'] = M
    # Adjust eta bounds based on initialization
    tbnn_config['eta_min'] = max(eta_init * ETA_LO_SCALE, 1e-3)
    tbnn_config['eta_max'] = eta_init * ETA_HI_SCALE
    # Add new parametrization controls
    tbnn_config['s_floor'] = s_floor
    tbnn_config['alpha_temp'] = alpha_temp
    # Add eta0-freezing controls
    tbnn_config['freeze_eta0'] = freeze_eta0
    tbnn_config['eta0_fixed'] = eta0_fixed
    tbnn_config['eta0_eps'] = eta0_eps
    # Add curvature-control parameters
    tbnn_config['mu_min_gamma'] = mu_min_gamma
    tbnn_config['mu_max_gamma'] = mu_max_gamma
    tbnn_config['gate_gamma'] = gate_gamma
    tbnn_config['gate_width_z'] = gate_width_z
    tbnn_config['tail_gate_gamma'] = tail_gate_gamma
    tbnn_config['tail_gate_width_z'] = tail_gate_width_z
    # Add per-mode PL bump parameters
    tbnn_config['enable_pl_per_mode'] = enable_pl_per_mode
    tbnn_config['pl_width_z'] = pl_width_z
    # Add log head parameters
    tbnn_config['log_head'] = log_head
    tbnn_config['log_mixing'] = log_mixing
    # Add freeze_centers flag
    tbnn_config['freeze_centers'] = freeze_centers
    
    TBNN_model = models.build_tbnn_bounded_model(
        hidden_units=hidden_units,
        **tbnn_config
    )
    
    # Create dummy inputs with spatial dimensions matching domain_size
    H, W = domain_size[1], domain_size[0]  # domain_size is (nx, ny)
    
    dummy_gamma_dot = jnp.ones((H, W)) * 1.0  # Use shear rate = 1.0 for initialization
    dummy_invariants = jnp.ones((2, H, W))    # Two invariants (I1, I2)
    
    # Initialize parameters
    if use_soft_newtonian_init:
        # Use soft Newtonian initialization to start near constant viscosity
        print(f"   Using soft Newtonian initialization (eta0 = {eta_init})")
        params = models.init_tbnn_soft_newtonian(
            TBNN_model, key, H, W, eta_init,
            A_frac=0.05,         # Small amplitude for smooth start
            k_frac=0.2,          # Moderate slope to avoid saturation
            pair_modes=(0, 1)    # First two modes form +/- pair
        )
        init_method = "soft_newtonian"
    else:
        # Use random initialization
        print(f"    Using random initialization")
        params = TBNN_model.init(key, dummy_gamma_dot, dummy_invariants)
        init_method = "random"
    
    # Prepare parameters for optimization (flatten them)
    flattened_params, tree_def, shapes = parameter_utils.flatten_params(params)
    
    # Calculate parameter indices for efficient unflattening
    starts_static = []
    ends_static = []
    idx = 0
    for shape in shapes:
        size = int(np.prod(np.array(shape)))
        starts_static.append(idx)
        ends_static.append(idx + size)
        idx += size
    
    model_info = {
        'model': TBNN_model,
        'params': params,
        'flattened_params': flattened_params,
        'tree_def': tree_def,
        'shapes': shapes,
        'starts_static': starts_static,
        'ends_static': ends_static,
        'hidden_units': hidden_units,
        'num_params': len(flattened_params),
        'eta_init': eta_init,
        'eta_min': tbnn_config['eta_min'],
        'eta_max': tbnn_config['eta_max'],
        'tbnn_config': tbnn_config.copy(),  # Store TBNN config for later use
        'init_method': init_method  # Track initialization method
    }
    
    print(f"TBNN initialized with {model_info['num_params']} parameters ({init_method})")
    print(f"TBNN bounds: eta  in  [{tbnn_config['eta_min']:.3f}, {tbnn_config['eta_max']:.3f}]")
    
    return model_info

def _compute_eta(velocity, tbnn_model, params, tbnn_config=None):
    """Viscosity computation using bounded mixture-of-sigmoids viscosity model."""
    return models.tbnn_eta_bounded_from_v(velocity, tbnn_model, params)

def create_reference_trajectory_constriction(reference_model, reference_params, flow_cond, particles, steps):
    """Create reference trajectory for comparison with constriction geometry."""
    print(f"Creating {reference_model} reference with params: {reference_params}")
    
    # Setup nu0 update function based on model type
    if reference_model == 'newtonian':
        nu0_update_fn = models.create_dynamic_nu0_fn(model_type='newtonian', strategy='max', C=1.0)
        stress_forcing_fn = models.newtonian_stress_forcing
    elif reference_model == 'carreau_yasuda':
        nu0_update_fn = models.create_dynamic_nu0_fn(model_type='carreau_yasuda', strategy='max', C=1.0)
        stress_forcing_fn = models.carreau_yasuda_stress_forcing
    elif reference_model == 'power_law':
        nu0_update_fn = models.create_dynamic_nu0_fn(model_type='power_law', strategy='max', C=1.0)
        stress_forcing_fn = models.power_law_stress_forcing
    else:
        raise ValueError(f"Unknown reference model: {reference_model}")
    
    # Temporarily modify flow_cond for reference run
    ref_flow_cond = flow_cond.copy()
    ref_flow_cond['outer_steps'] = steps
    # MEM jax_rheology unflattens whenever tree_def is set (dropped
    # `model is not None`). CY/Newtonian/PL references pass a short
    # vector with model=None; drop the TBNN tree so params stay raw.
    for _k in ('tree_def', 'shapes', 'starts_static', 'ends_static'):
        ref_flow_cond.pop(_k, None)
    
    try:
        final_result, trajectory, _ = forward_simulation.forward_fluid_simulation(
            flow_cond=ref_flow_cond,
            flattened_params=reference_params,
            particles=particles,  # Include constriction particles
            stress_forcing_fn=stress_forcing_fn,
            model=None,
            nu0_update_fn=nu0_update_fn,
            nu0_baseline=0.0,
            stepper_type=ref_flow_cond.get('stepper_type', 'fully_implicit'),
            solver_type=ref_flow_cond.get('solver_type', 'bicgstab'),
            use_preconditioner=False,
            preconditioner_type='none'
        )
        
        # Extract velocity trajectory data - both x and y components
        # Handle both Field objects (with .data) and raw arrays
        if hasattr(trajectory[0], 'data'):
            ref_trajectory_x = trajectory[0].data  # x-velocity component (Field object)
            ref_trajectory_y = trajectory[1].data  # y-velocity component (Field object)
        else:
            ref_trajectory_x = trajectory[0]  # Already an array
            ref_trajectory_y = trajectory[1]  # Already an array
        
        print(f"Reference trajectory shapes: x={ref_trajectory_x.shape}, y={ref_trajectory_y.shape}")
        
        # Return both components as a tuple
        ref_trajectory = (ref_trajectory_x, ref_trajectory_y)
        return ref_trajectory, final_result
        
    except Exception as e:
        print(f"Error creating reference trajectory: {e}")
        traceback.print_exc()
        raise

def compute_tbnn_trajectory_loss_constriction(tbnn_params, flow_cond, model_info, particles, reference_trajectory,
                                             warmup: int = 0, tail: int = None, detach: bool = True,
                                             visc_loss: bool = False, visc_loss_config: dict = None,
                                             shape_loss: bool = False, shape_weight: float = 0.0,
                                             mask_layout: str = "full_domain",
                                             resolution_piv: bool = False, piv_W_win: tuple = (32, 16),
                                             piv_overlap: float = 0.75, piv_kernel: str = "hann",
                                             add_piv_noise: bool = False, piv_noise_p_percent: float = 0.0,
                                             piv_noise_corr_frac: float = 0.35, piv_noise_beta_grad: float = 0.5,
                                             piv_noise_use_bias: bool = True, piv_noise_seed: int = 1453):
    """Compute RMSE loss between TBNN simulation and reference trajectories with constriction.
    
    Uses full-field library-style loss with particle masking (from loss_functions.py approach).
    Includes both u_x and u_y components in loss calculation.
    
    Args:
        tbnn_params: TBNN parameters
        flow_cond: Flow conditions dictionary
        model_info: TBNN model information
        particles: Particle configuration
        reference_trajectory: Reference trajectory to match
        warmup: Number of initial steps to run without gradients (default: 0)
        tail: Number of final steps to use for loss computation (default: all steps after warmup)
        detach: Whether to detach gradients from warmup phase (default: True)
        visc_loss: If True, add log-log viscosity derivative regularization (default: False)
        visc_loss_config: Configuration dict for viscosity regularization penalties (default: None)
        shape_loss: If True, add a scale-invariant velocity-shape RMSE (default: False)
        shape_weight: Multiplier for the shape term added to the RMSE (default: 0.0)
        mask_layout: "constriction_focused" = no transpose (paper weighting,
            solver-native ``(T, nx, ny)``);
            "full_domain" = apply the PIV-twin transpose before the
            mask/reduction block so the mask is built on ``(T, ny, nx)``.
            Aliases: legacy -> constriction_focused, correct -> full_domain.
    
    Returns:
        Tuple of (loss, final_result, trajectory)
    """
    mask_layout = resolve_mask_layout(mask_layout)
    if resolution_piv and mask_layout == "constriction_focused":
        raise ValueError(
            "PIV observation with constriction_focused mask_layout was "
            "never used and is untested; use full_domain"
        )
    # Create TBNN nu0 update function using new bounded approach
    tbnn_nu0_fn = models.create_dynamic_nu0_fn(
        'TBNN', 
        model=model_info['model'], 
        strategy='max', 
        C=1.0
    )
    
    # Use TBNN stress forcing function (signature matches new models.py)
    tbnn_stress_forcing = models.TBNN_stress_forcing
    
    # Validate parameters and ensure full simulation
    steps_total = flow_cond['outer_steps']
    if tail is None:
        tail = steps_total - warmup
    
    # Ensure we simulate the full trajectory
    if warmup + tail != steps_total:
        print(f"WARNING: warmup({warmup}) + tail({tail}) = {warmup + tail} != outer_steps({steps_total})")
        print(f"   Auto-adjusting to ensure full {steps_total} step simulation")
        if tail is None or warmup + tail < steps_total:
            # Extend tail to cover remaining steps
            tail = steps_total - warmup
        elif warmup + tail > steps_total:
            # Reduce warmup to fit
            warmup = steps_total - tail
    
    assert tail > 0 and warmup >= 0 and warmup + tail == steps_total, f"Invalid warmup={warmup}, tail={tail}, total={steps_total}"
    
    try:
        # 1) Warmup phase (forward only, optionally detached)
        init_state_for_tail = None
        if warmup > 0:
            flow_w = dict(flow_cond)
            flow_w['outer_steps'] = warmup
            
            warm_final_result, _, _ = forward_simulation.forward_fluid_simulation(
                flow_cond=flow_w,
                flattened_params=tbnn_params,
                particles=particles,
                stress_forcing_fn=tbnn_stress_forcing,
                model=model_info['model'],
                nu0_update_fn=tbnn_nu0_fn,
                nu0_baseline=0.0,
                stepper_type=flow_w.get('stepper_type', 'fully_implicit'),
                solver_type=flow_w.get('solver_type', 'bicgstab'),
                use_preconditioner=False,
                preconditioner_type='none'
            )
            
            # Detach gradients from warmup phase if requested
            if detach:
                init_state_for_tail = jax.tree_map(jax.lax.stop_gradient, warm_final_result)
            else:
                init_state_for_tail = warm_final_result
        
        # 2) Tail phase (differentiable)
        flow_t = dict(flow_cond)
        flow_t['outer_steps'] = tail
        
        final_result, trajectory, _ = forward_simulation.forward_fluid_simulation(
            flow_cond=flow_t,
            flattened_params=tbnn_params,
            particles=particles,
            stress_forcing_fn=tbnn_stress_forcing,
            model=model_info['model'],
            nu0_update_fn=tbnn_nu0_fn,
            nu0_baseline=0.0,
            stepper_type=flow_t.get('stepper_type', 'fully_implicit'),
            solver_type=flow_t.get('solver_type', 'bicgstab'),
            use_preconditioner=False,
            preconditioner_type='none',
            initial_state=init_state_for_tail  # Start from warmup state if available
        )
        
        # 3) Extract trajectory data and compute loss on tail portion only
        # Handle both Field objects (with .data) and raw arrays
        if hasattr(trajectory[0], 'data'):
            sim_x, sim_y = trajectory[0].data, trajectory[1].data  # Field objects
        else:
            sim_x, sim_y = trajectory[0], trajectory[1]  # Already arrays
        # shapes (tail, H, W)
        
        # Handle reference trajectory format - now expects tuple of (x, y) components
        if isinstance(reference_trajectory, tuple) and len(reference_trajectory) == 2:
            reference_x, reference_y = reference_trajectory
        else:
            # Backward compatibility: if single array, assume it's x-component only
            reference_x = reference_trajectory
            reference_y = jnp.zeros_like(reference_trajectory)

        # mask_layout: "full_domain" transposes solver-native (T, nx, ny)
        # onto the PIV-twin (T, ny, nx) so the mask is built on that layout.
        # "constriction_focused" leaves arrays in solver-native (T, nx, ny).
        if mask_layout == "full_domain":
            # jax-cfd returns (T, nx, ny) = (T, 256, 128) for domain (8x4) with grid (256, 128)
            # We need (T, H, W) = (T, ny, nx) = (T, 128, 256) where H=height=y, W=width=x
            if sim_x.ndim == 3:
                sim_x = jnp.transpose(sim_x, (0, 2, 1))  # (T, 256, 128) -> (T, 128, 256)
                sim_y = jnp.transpose(sim_y, (0, 2, 1))
            # Apply same transpose to reference
            if reference_x.ndim == 3:
                reference_x = jnp.transpose(reference_x, (0, 2, 1))
                reference_y = jnp.transpose(reference_y, (0, 2, 1))
        elif mask_layout != "constriction_focused":
            raise ValueError(
                f"mask_layout must be 'full_domain' or 'constriction_focused'; got {mask_layout!r}"
            )
        
        # Library-style full-field loss with particle masking
        # Align reference to tail window 
        ref_x_tail = reference_x[-tail:] if len(reference_x.shape) > 2 else reference_x[-tail:]  
        ref_y_tail = reference_y[-tail:] if len(reference_y.shape) > 2 else reference_y[-tail:]
        
        # Ensure matching lengths
        min_len = min(sim_x.shape[0], ref_x_tail.shape[0], sim_y.shape[0], ref_y_tail.shape[0])
        sim_x_cut = sim_x[:min_len]
        sim_y_cut = sim_y[:min_len] 
        ref_x_cut = ref_x_tail[:min_len]
        ref_y_cut = ref_y_tail[:min_len]

        # --- OPTIONAL: PIV downsample both sim and ref; then add noise to REF only ---
        if resolution_piv:
            # vector-grid stride from overlap
            if isinstance(piv_W_win, (int, float)):
                W_x, W_y = int(piv_W_win), int(piv_W_win)
            else:
                W_x, W_y = int(piv_W_win[0]), int(piv_W_win[1])

            s_x = max(1, int(round(W_x * (1.0 - float(piv_overlap)))))
            s_y = max(1, int(round(W_y * (1.0 - float(piv_overlap)))))

            # physical extents (match your grid.domain)
            grid = flow_cond['grid']
            (x_min, x_max), (y_min, y_max) = grid.domain
            Lx = float(x_max - x_min)
            Ly = float(y_max - y_min)

            # downsample both streams to the SAME vector grid (T, H_ds, W_ds)
            # Store original dimensions for diagnostic
            orig_H, orig_W = sim_x_cut.shape[1], sim_x_cut.shape[2]
            
            sim_x_cut, sim_y_cut, x_c_vec, y_c_vec = piv_downsample_THW(
                sim_x_cut, sim_y_cut, W_x, W_y, s_x, s_y, x_min, y_min, Lx, Ly, kernel=piv_kernel
            )
            ref_x_cut, ref_y_cut, _, _ = piv_downsample_THW(
                ref_x_cut, ref_y_cut, W_x, W_y, s_x, s_y, x_min, y_min, Lx, Ly, kernel=piv_kernel
            )
            
            # Diagnostic: verify PIV grid dimensions
            jax.debug.print("PIV grid diagnostic: H={}, W={} | W_x={}, W_y={} | s_x={}, s_y={}", 
                           orig_H, orig_W, W_x, W_y, s_x, s_y)
            jax.debug.print("   Expected (Ny,Nx) = ({}, {})", (orig_H - W_y)//s_y + 1, (orig_W - W_x)//s_x + 1)
            jax.debug.print("   Actual   (Ny,Nx) = ({}, {})", sim_x_cut.shape[-2], sim_x_cut.shape[-1])

            # deterministic, spatially fixed noise (same for all frames) added ONLY to reference
            if add_piv_noise and piv_noise_p_percent > 0.0:
                key = jax.random.PRNGKey(int(piv_noise_seed))

                # Scale = (p% of U95) -- compute U95 on the *downsampled* ref speed
                ref_speed = jnp.sqrt(ref_x_cut**2 + ref_y_cut**2)
                u95 = jnp.asarray(jnp.percentile(ref_speed, 95.0), jnp.float32)
                sigma_base = (jnp.float32(piv_noise_p_percent) / 100.0) * (u95 + 1e-6)

                # original full grid (W_full, H_full) for spacing conversion
                H_full = sim_x.shape[1]
                W_full = sim_x.shape[2]

                ref_x_cut, ref_y_cut, _ = add_piv_noise_jax(
                    ref_x_cut, ref_y_cut,
                    W_x=W_x, W_y=W_y, s_x=s_x, s_y=s_y,
                    Lx=Lx, Ly=Ly,
                    key=key,
                    corr_frac=float(piv_noise_corr_frac),
                    sigma_base=float(sigma_base),
                    beta_grad=float(piv_noise_beta_grad),
                    use_bias=bool(piv_noise_use_bias),
                    full_grid_shape=(W_full, H_full)
                )
        # --- end PIV downsample + noise ---
        # Compute squared errors (full field)
        squared_error_x = jnp.square(sim_x_cut - ref_x_cut)  # (T,H,W)
        squared_error_y = jnp.square(sim_y_cut - ref_y_cut)  # (T,H,W) 
        err_all = squared_error_x + squared_error_y  # Equal weighting
        
        # Create particle mask for constriction geometry
        H, W = err_all.shape[1], err_all.shape[2]
        grid = flow_cond['grid']
        domain_bounds = grid.domain  # ((x_min, x_max), (y_min, y_max))
        dx = (domain_bounds[0][1] - domain_bounds[0][0]) / jnp.maximum(W - 1, 1)
        dy = (domain_bounds[1][1] - domain_bounds[1][0]) / jnp.maximum(H - 1, 1)

        center_x = (domain_bounds[0][1] + domain_bounds[0][0]) / 2
        radius = 1.5  # semicircle radius from setup

        # Coordinate grids
        x_coords = jnp.linspace(domain_bounds[0][0], domain_bounds[0][1], W)
        y_coords = jnp.linspace(domain_bounds[1][0], domain_bounds[1][1], H)
        X, Y = jnp.meshgrid(x_coords, y_coords, indexing='ij')
        X, Y = X.T, Y.T  # (H,W)

        dist_bottom = jnp.sqrt((X - center_x)**2 + (Y - 0.0)**2)
        dist_top    = jnp.sqrt((X - center_x)**2 + (Y - domain_bounds[1][1])**2)

        buffer = 0.1
        fluid_mask_2d = (dist_bottom > (radius + buffer)) & (dist_top > (radius + buffer))
        T = err_all.shape[0]
        mask = jnp.broadcast_to(fluid_mask_2d[None, :, :], (T, H, W)).astype(err_all.dtype)

        # Compute loss with particle masking
        denom = jnp.maximum(jnp.sum(mask), 1.0)
        loss  = jnp.sum(err_all * mask) / denom
        
        # --- OPTIONAL: velocity shape loss (scale-invariant RMSE, parallel to RMSE) ---
        if shape_loss and shape_weight > 0.0:
            eps = 1e-30
            # Use the SAME mask as RMSE
            # Compute optimal scalar a_t per time step to align sim -> ref in least-squares sense.
            # num = <sim, ref>_w , den = <sim, sim>_w  (both include x and y components)
            num = jnp.sum(mask * (sim_x_cut * ref_x_cut + sim_y_cut * ref_y_cut), axis=(1, 2))   # (T,)
            den = jnp.sum(mask * (sim_x_cut * sim_x_cut + sim_y_cut * sim_y_cut),     axis=(1, 2)) + eps  # (T,)
            a_t = jax.lax.stop_gradient(num / den)   # (T,)
            a   = a_t[:, None, None]                 # (T,1,1) broadcast

            # Scale predicted velocity and compute weighted RMSE on shape-only residual
            sx = a * sim_x_cut
            sy = a * sim_y_cut
            shape_err = (sx - ref_x_cut) ** 2 + (sy - ref_y_cut) ** 2               # (T,H,W)
            denom_shape = jnp.maximum(jnp.sum(mask), 1.0)                            # same normalization as RMSE
            shape_term = jnp.sum(mask * shape_err) / denom_shape

            loss = loss + shape_weight * shape_term
            jax.debug.print("   Shape loss: term={} (w={})", shape_term, shape_weight)
        # --- end shape loss ---
        
        # --- OPTIONAL: log-log viscosity derivative regularization ---
        if visc_loss:
            cfg = visc_loss_config or {}
            gmin = float(cfg.get("reg_gamma_min", 1e-2))
            gmax = float(cfg.get("reg_gamma_max", 1e2))
            npts = int(cfg.get("reg_num_points", 128))
            s_cap = float(cfg.get("reg_s_cap", 0.30))
            lam_s = float(cfg.get("reg_lambda_slope", 1e-3))
            lam_c = float(cfg.get("reg_lambda_curv", 3e-4))
            
            # Unflatten params for model.apply if needed
            if isinstance(tbnn_params, jnp.ndarray) and tbnn_params.ndim == 1:
                # tbnn_params is flattened - need to unflatten it
                structured_params = parameter_utils.unflatten_params(
                    tbnn_params, model_info['tree_def'], model_info['shapes']
                )
            else:
                # Already structured
                structured_params = tbnn_params
            
            # Make sure there's exactly ONE top-level "params"
            if isinstance(structured_params, dict):
                variables = structured_params if 'params' in structured_params else {'params': structured_params}
                variables = freeze(variables)
            else:
                # If it's already a FrozenDict from init(), keep as-is
                variables = structured_params
            
            reg_pen = _loglog_slope_curv_penalty_debug(
                model_info['model'], variables,   # pass the proper variables tree
                gamma_min=gmin, gamma_max=gmax, num_points=npts,
                s_cap=s_cap, lam_slope=lam_s, lam_curv=lam_c,
                curv_cap_abs=float(cfg.get("reg_curv_cap_abs", 1.0)),
                p_slope=float(cfg.get("reg_p_slope", 2.0)),
                p_curv=float(cfg.get("reg_p_curv", 2.0))
            )
            
            physics_loss = loss
            loss = loss + reg_pen
            jax.debug.print(
                "   Viscosity regularization: physics_loss={}, reg_penalty={}, total={}",
                physics_loss, reg_pen, loss
            )
        # --- end regularization ---
        
        return loss, final_result, trajectory
        
    except jax.errors.ConcretizationTypeError as e:
        # This is a programming/debug-print issue; never hide it.
        raise
    except Exception as e:
        import traceback
        print("TBNN trajectory sim failed inside compute_tbnn_trajectory_loss_constriction:", e)
        traceback.print_exc()
        # Keep grads alive so we can still see something useful
        l2 = 1e-6 * jnp.sum(jnp.square(tbnn_params))
        return 1_000.0 + l2, None, None
# VISUALIZATION FUNCTIONS - Now imported from plotting module
# =============================================================================
# All plotting functions live in paper_figs/training_plots.py (file-path loaded).
# Access via: plotting.plot_model_comparison_constriction(...)

# =============================================================================
# LOG-LOG VISCOSITY REGULARIZATION (MANUAL DEBUG VERSION)
# Compatible with NEW mixture-of-sigmoids model
# =============================================================================

def _eta_cy_scalar(g, etainf, eta0, lam, n, a):
    """Carreau-Yasuda viscosity model (scalar evaluation)."""
    return etainf + (eta0 - etainf) * jnp.power(1.0 + jnp.power(lam * g, a), (n - 1.0) / a)

def _tbnn_eta_on_gamma_grid_debug(tbnn_model, params, gamma_grid):
    """
    Evaluate eta(gammadot) from the TBNN head on a 1D grid of gammadot.
    Manual debug version - identical to loss_functions.py implementation.
    Works with the NEW mixture-of-sigmoids viscosity model.
    """
    gamma_grid = jnp.asarray(gamma_grid)
    N = gamma_grid.shape[0]

    # (H,W) = (1,N)
    gamma_field = gamma_grid.reshape(1, N)

    # invariants along the "pure shear" path used in plots
    I1 = 0.5 * gamma_grid**2
    I2 = -0.5 * gamma_grid**2
    invariants_aux = jnp.stack([I1, I2], axis=0).reshape(2, 1, N)  # (C,H,W) = (2,1,N)

    # Expect a single Flax variables tree with top-level "params"
    eta_field = tbnn_model.apply(params, gamma_field, invariants_aux)  # (1, N)
    return eta_field.reshape(N)  # (N,)


def _loglog_slope_curv_penalty_debug(tbnn_model,
                                     params,
                                     *,
                                     gamma_min: float,
                                     gamma_max: float,
                                     num_points: int,
                                     s_cap: float,
                                     lam_slope: float,
                                     lam_curv: float,
                                     curv_cap_abs: float = 1.0,
                                     p_slope: float = 2.0,
                                     p_curv: float = 2.0):
    """
    Build eta(gammadot) on a z-uniform grid with z=log gammadot. Compute
      dl/dz and d^2l/dz^2  (l=log eta),
    then penalize:
      - slope overflow summed over z (how much it's over s_cap),
      - curvature overflow only when |d^2l/dz^2| > curv_cap_abs (default 1.0),
    using L^p costs (default p=2 -> squared overflow).
    
    Manual debug version - identical to loss_functions.py implementation.
    """
    # z-uniform grid
    z_min = jnp.log(gamma_min)
    z_max = jnp.log(gamma_max)
    z = jnp.linspace(z_min, z_max, num_points)          # (N,)
    gamma = jnp.exp(z)                                  # (N,)

    # Evaluate eta(gammadot) from the TBNN head
    eta = _tbnn_eta_on_gamma_grid_debug(tbnn_model, params, gamma)  # (N,)
    ell = jnp.log(jnp.maximum(eta, 1e-30))                # l = log eta, safe for AD

    # First derivative (central difference) lives on z[1:-1]
    d1 = (ell[2:] - ell[:-2]) / (z[2:] - z[:-2])         # (N-2,)

    # Second derivative on z[1:-1] (three-point formula)
    dz = jnp.diff(z)                                     # (N-1,)
    d2 = 2.0 * (
        (ell[2:]   - ell[1:-1]) / dz[1:] -
        (ell[1:-1] - ell[:-2])  / dz[:-1]
    ) / (z[2:] - z[:-2])                                  # (N-2,)

    # --- SLOPE: summed overflow (how much above s_cap) ---
    slope_over = jnp.maximum(0.0, jnp.abs(d1) - s_cap)    # (N-2,)
    slope_cost_pointwise = slope_over ** p_slope          # squared by default
    pen_slope = lam_slope * jnp.sum(slope_cost_pointwise)

    # --- CURVATURE: only penalize when |d^2l/dz^2| > curv_cap_abs ---
    curv_over = jnp.maximum(0.0, jnp.abs(d2) - curv_cap_abs)
    curv_cost_pointwise = curv_over ** p_curv             # squared by default
    pen_curv = lam_curv * jnp.sum(curv_cost_pointwise)

    return pen_slope + pen_curv


# =============================================================================
# GRADIENT UPDATE TEST WITH CONSTRICTION
# =============================================================================

def _pretrain_tbnn_cy_shape_only(model_info, cy_params, *,
                                 n2_target=0.95, steps1=50, steps2=30,
                                 lr=1e-1, gmin=1e-2, gmax=1e2, N=120):
    """Two-stage CY fit with etainf/delta frozen; learns only mu/s/alpha on current model."""
    etainf, eta0, lam, n_cy, a = [float(x) for x in cy_params]
    tbnn = model_info['model']; params_in = model_info['params']
    gamma = jnp.geomspace(gmin, gmax, N)
    I1 = 0.5 * gamma**2; I2 = -0.5 * gamma**2
    inv_aux = jnp.stack([I1, I2], axis=0).reshape(2, 1, N)
    g_field = gamma.reshape(1, N)

    def _make_target(n_val):
        return _eta_cy_scalar(gamma, etainf, eta0, lam, n_val, a)

    def _run_stage(params0, n_val, steps, tag):
        eta_tgt = _make_target(n_val)
        def loss_fn(p):
            eta = tbnn.apply(p, g_field, inv_aux).reshape(-1)
            res = jnp.log(jnp.clip(eta, 1e-30, 1e30)) - jnp.log(jnp.clip(eta_tgt, 1e-30, 1e30))
            return jnp.mean(res**2)

        flat = traverse_util.flatten_dict(unfreeze(params0), keep_empty_nodes=True)
        labels = {}
        for k in flat.keys():
            name = "/".join(k)
            if name.endswith(("eta_inf_raw","delta_raw","log_eta_inf_raw","r_raw","log_range_raw")):
                labels[k] = "tail"   # freeze
            else:
                labels[k] = "mix"    # learn mu/s/alpha
        labels = traverse_util.unflatten_dict(labels)
        labels = freeze(labels) if isinstance(params0, FrozenDict) else labels
        tx = optax.multi_transform({"mix": optax.adam(lr), "tail": optax.chain(optax.scale(0.0))}, labels)
        state = tx.init(params0)

        @jit
        def step(p, s):
            L, g = value_and_grad(loss_fn)(p)
            upd, s = tx.update(g, s, p)
            p = optax.apply_updates(p, upd)
            return p, s, L

        p = params0; best = (1e9, p)
        for i in range(1, steps+1):
            p, state, L = step(p, state)
            if float(L) < best[0]: best = (float(L), p)
        print(f"   [pretrain] {tag}: best loss {best[0]:.4e} in {steps} steps")
        return best[1]

    p1 = _run_stage(params_in, n_cy,     steps1, f"CY stage 1 (n={n_cy:.3f})")
    p2 = _run_stage(p1,        n2_target, steps2, f"CY stage 2 (n={n2_target:.3f})")
    return p2


def debug_one_step_gradient_constriction(
    reference_model='carreau_yasuda',
    reference_params=None,
    tbnn_hidden_units=None,
    eta_init=1.0,
    pressure_gradient=2.5,
    dt=5e-5,
    inner_steps=None,
    outer_steps=100,
    domain=None,
    domain_size=None,
    learning_rate=1e-3,
    num_update_steps=1,
    # --- new: 2-stage training (etainf ON then freeze) ---
    two_stage_etainf_then_curv=False,
    stage1_steps_etainf=0,      # N1 (max steps for stage 1)
    stage2_steps_curv=0,        # N2
    stage1_etainf_only=True,    # If True, stage 1 = etainf ONLY (freeze curvature); if False, both learn
    stage1_reset_momentum=True, # If True, reset Adam momentum when switching to stage 2
    stage1_early_stop_on_flip=True,  # If True, end stage 1 early if detainf flips sign (momentum overshoot)
    run_new_forward=True,
    solver_type='bicgstab',
    stepper_type='fully_implicit',
    random_seed=42,
    save_plots=False,
    output_dir='./work/instantaneous_train',
    use_soft_newtonian_init=False,
    save_traj_info=False,
    use_warmup_tail=False,
    warmup_steps=None,
    tail_steps=None,
    visc_loss=False,
    reg_gamma_min=1e-2,
    reg_gamma_max=1e2,
    reg_num_points=128,
    reg_s_cap=0.30,
    reg_lambda_slope=1e-3,
    reg_lambda_curv=3e-4,
    reg_curv_cap_abs=1.0,
    reg_p_slope=2.0,
    reg_p_curv=2.0,
    s_floor=0.0,
    alpha_temp=1.0,
    global_scalar_lr_scale=10.0,
    freeze_eta0=False,
    eta0_fixed=1.0,
    eta0_eps=1e-6,
    mu_min_gamma=None,
    mu_max_gamma=None,
    gate_gamma=None,
    gate_width_z=0.5,
    tail_gate_gamma=None,
    tail_gate_width_z=0.5,
    enable_pl_per_mode=False,
    pl_width_z=0.5,
    pl_lr_scale=1.0,
    checkpoint_every=10,
    enable_grad_equalizer=False,
    equalize_target='mix',
    equalize_cap_ratio=0.5,
    log_head=False,
    log_mixing="add",
    # --- optional two-stage CY pretrain (shape-only, frozen plateaus)
    pretrain_cy=False,
    pretrain_cy_steps_1=50,
    pretrain_cy_steps_2=10,
    pretrain_cy_n2_target=0.95,
    # --- velocity shape loss ---
    shape_loss=False,
    shape_weight=0.0,
    freeze_centers=False,
    M=4,
    mask_layout="full_domain",
    resolution_piv=False,
    piv_W_win=(32, 16),
    piv_overlap=0.75,
    piv_kernel="hann",
    add_piv_noise=False,
    piv_noise_p_percent=0.0,
    piv_noise_corr_frac=0.35,
    piv_noise_beta_grad=0.5,
    piv_noise_use_bias=True,
    piv_noise_seed=1453,
):
    """
    Multi-step gradient update with constriction geometry (Mini Training)
    
    Uses the NEW mixture-of-sigmoids viscosity model:
        eta(z) = eta_inf + delta * (1 - sum_i alpha_i * sigmoid((z - mu_i)/s_i))
    where z = log(gamma_dot / gamma_ref)
    
    This function:
    1. Sets up TBNN and reference models with constriction geometry
    2. Computes loss and gradients for multiple update steps
    3. Prints and plots differences between models
    4. Applies multiple SGD steps to TBNN parameters
    5. Plots loss convergence over training steps
    6. Shows viscosity function evolution
    7. Optionally runs forward simulation with final updated TBNN
    
    Args:
        reference_model: 'newtonian', 'carreau_yasuda', or 'power_law'
        reference_params: Parameters for reference model
            - Newtonian: scalar viscosity (e.g., 0.5)
            - Carreau-Yasuda: [etainf, eta0, lam, n, a] (e.g., [0.02, 1.0, 5.0, 0.5, 2.0])
            - Power-law: [K, n] (e.g., [0.5, 0.8])
        tbnn_hidden_units: TBNN architecture (e.g., [30, 30, 30])
        eta_init: Initial viscosity for TBNN (e.g., 1.0)
        pressure_gradient: Applied pressure gradient (e.g., 2.5)
        dt: Time step (e.g., 5e-5)
        inner_steps: Inner time steps per outer step (e.g., 300)
        outer_steps: Number of outer time steps (e.g., 100)
        domain: Domain bounds ((x_min, x_max), (y_min, y_max))
        domain_size: Grid resolution (nx, ny)
        learning_rate: Adam learning rate (e.g., 1e-3)
        num_update_steps: Number of Adam optimization steps (e.g., 1, 5, 10)
        two_stage_etainf_then_curv: If True, enable 2-stage training (etainf ON then freeze) (default: False)
        stage1_steps_etainf: Max number of steps with etainf learnable (stage 1) (default: 0)
        stage2_steps_curv: Number of steps with etainf frozen (stage 2, curvature-only) (default: 0)
        stage1_etainf_only: If True, stage 1 = etainf ONLY (freeze curvature); if False, both etainf+curvature learn (default: True)
        stage1_reset_momentum: If True, reset Adam momentum when switching to stage 2 (default: True)
        stage1_early_stop_on_flip: If True, end stage 1 early if detainf flips sign (momentum overshoot) (default: False)
        run_new_forward: Whether to run forward sim with final updated TBNN
        save_plots: Whether to save plots to files (default: False)
        output_dir: Directory to save plots and results (default: './work/instantaneous_train')
        save_traj_info: If True, save all trajectory data and model states as .npy files (default: False)
        solver_type: 'bicgstab', 'gmres', or 'cg'
        stepper_type: Time integrator type
        random_seed: Random seed for reproducibility
        use_soft_newtonian_init: If True, initialize TBNN to behave like Newtonian fluid
        use_warmup_tail: If True, use warmup+tail approach to control gradient explosion (default: False)
        warmup_steps: Number of warmup steps if use_warmup_tail=True (default: None, auto-calculated)
        tail_steps: Number of tail steps if use_warmup_tail=True (default: None, auto-calculated)
        visc_loss: If True, add log-log viscosity derivative regularization to loss (default: False)
        reg_gamma_min: Min shear rate for viscosity regularization (default: 1e-2)
        reg_gamma_max: Max shear rate for viscosity regularization (default: 1e2)
        reg_num_points: Number of grid points for viscosity regularization (default: 128)
        reg_s_cap: Slope cap |d log eta / d log gammadot| for regularization (default: 0.30)
        reg_lambda_slope: Weight for slope overflow penalty (default: 1e-3)
        reg_lambda_curv: Weight for curvature overflow penalty (default: 3e-4)
        reg_curv_cap_abs: Curvature threshold to start penalizing (default: 1.0)
        reg_p_slope: Exponent for slope penalty (2.0 = squared, default: 2.0)
        reg_p_curv: Exponent for curvature penalty (2.0 = squared, default: 2.0)
        s_floor: Minimum logistic width in z-space (0.0 = default, 0.5-0.8 for broader/smoother kernels)
        alpha_temp: Softmax temperature for mixture weights (1.0 = default, >1 spreads weights, <1 sharpens)
        global_scalar_lr_scale: Learning rate multiplier for global scalars (eta_inf_raw, delta_raw) (default: 10.0)
        freeze_eta0: If True, freeze eta0 = eta_inf + delta to learn only curvature (default: False)
        eta0_fixed: Fixed value of eta0 when freeze_eta0=True (default: 1.0)
        eta0_eps: Small positive floor for delta when eta0 is frozen (default: 1e-6)
        mu_min_gamma: Lower bound on center locations - no curvature before this gammadot (default: None)
        mu_max_gamma: Optional upper bound on center locations (default: None)
        gate_gamma: If set, multiply mixture by smooth gate starting at this gammadot (default: None)
        gate_width_z: Gate smoothness in log(gammadot) space (default: 0.5)
        tail_gate_gamma: If set, delay tail (etainf) to this gammadot (prevents mid-shear drag) (default: None)
        tail_gate_width_z: Tail gate smoothness in log(gammadot) space (default: 0.5)
        enable_pl_per_mode: If True, add per-mode power-law bumps for extra expressiveness (default: False)
        pl_width_z: Smooth onset width for each mode's PL bump (default: 0.5)
        pl_lr_scale: Learning rate multiplier for pl_slope_raw parameters (default: 1.0)
        checkpoint_every: Save TBNN params every N steps (0 = no checkpoints) (default: 10)
        enable_grad_equalizer: If True, prevent tail/PL grads from dwarfing mixture params (default: False)
        equalize_target: Which group to protect - 'mix', 'tail', or 'pl' (default: 'mix')
        equalize_cap_ratio: Shrink other groups when they exceed ratio * ||target|| (default: 0.5)
        log_head: If True, learn in log-eta space (geometric blend, stable across decades) (default: False)
        log_mixing: Mixing mode - "add" = log1p(r*(1-F)), "geom" = L*(1-F) for true log-domain (default: "add")
        pretrain_cy: If True, run two-stage CY pretraining before main training (default: False)
        pretrain_cy_steps_1: Number of steps for stage 1 (fit exact CY target) (default: 50)
        pretrain_cy_steps_2: Number of steps for stage 2 (fit near-Newtonian CY) (default: 10)
        pretrain_cy_n2_target: Target power-law index for stage 2 (0.95 ~= Newtonian) (default: 0.95)
        shape_loss: If True, add scale-invariant velocity-shape RMSE (default: False)
        shape_weight: Multiplier for shape term added to RMSE (default: 0.0)
        freeze_centers: If True, freeze mu centers (learn only widths s and weights alpha) (default: False)
        M: Number of sigmoid terms in the mixture-of-sigmoids viscosity model (default: 4)
    
    Returns:
        Dictionary with test results and updated TBNN model
        
    Example:
        # Carreau-Yasuda reference with moderate parameters (single step)
        results = debug_one_step_gradient_constriction(
            reference_model='carreau_yasuda',
            reference_params=[0.02, 1.0, 5.0, 0.5, 2.0],
            tbnn_hidden_units=[20, 20],
            eta_init=1.0,
            pressure_gradient=2.5,
            learning_rate=1e-3,
            num_update_steps=1,
            run_new_forward=True
        )
        
        # Mini training with 5 steps
        results = debug_one_step_gradient_constriction(
            reference_model='newtonian',
            reference_params=0.5,
            tbnn_hidden_units=[30, 30, 30],
            eta_init=0.5,
            pressure_gradient=3.0,
            learning_rate=1e-3,
            num_update_steps=5
        )
        
        # Initialize TBNN near Newtonian instead of random
        results = debug_one_step_gradient_constriction(
            reference_model='newtonian',
            reference_params=1.0,
            tbnn_hidden_units=[20, 20],
            eta_init=1.0,
            learning_rate=1e-3,
            use_soft_newtonian_init=True  # Start near constant viscosity
        )
        
        # Use warmup+tail to control gradient explosion for long simulations
        results = debug_one_step_gradient_constriction(
            reference_model='carreau_yasuda',
            reference_params=[0.02, 1.0, 5.0, 0.5, 2.0],
            tbnn_hidden_units=[32, 32],
            eta_init=1.0,
            outer_steps=400,  # Long simulation
            learning_rate=1e-3,
            use_warmup_tail=True,  # Enable warmup+tail approach
            warmup_steps=300,      # Optional: specify exact warmup steps
            tail_steps=100         # Optional: specify exact tail steps
        )
        
        # Use warmup+tail to control gradient explosion for long simulations
        results_long_sim = debug_one_step_gradient_constriction(
            reference_model='carreau_yasuda',
            reference_params=[0.02, 1.0, 5.0, 0.5, 2.0],
            tbnn_hidden_units=[32, 32],
            eta_init=1.0,
            outer_steps=400,  # Long simulation
            learning_rate=1e-3,
            use_warmup_tail=True,  # Enable warmup+tail approach
            warmup_steps=300,      # Optional: specify exact warmup steps
            tail_steps=100         # Optional: specify exact tail steps
        )
        
        # Use viscosity regularization to penalize unrealistic eta(gammadot) curves
        results_with_visc_reg = debug_one_step_gradient_constriction(
            reference_model='carreau_yasuda',
            reference_params=[0.02, 1.0, 5.0, 0.5, 2.0],
            tbnn_hidden_units=[30, 30],
            eta_init=1.0,
            outer_steps=2000,
            use_warmup_tail=True,
            warmup_steps=1800,
            tail_steps=200,
            visc_loss=True,              # Enable viscosity regularization
            reg_gamma_min=1e-2,          # Shear rate range
            reg_gamma_max=1e2,
            reg_num_points=160,          # Grid resolution
            reg_s_cap=0.30,              # Slope cap
            reg_lambda_slope=1e-3,       # Slope penalty weight
            reg_lambda_curv=3e-4,        # Curvature penalty weight
            reg_curv_cap_abs=1.0,        # Curvature threshold
            learning_rate=1e-2
        )
        
        # Start with broad, smooth kernels and diffuse weights for safer training
        results_smooth_start = debug_one_step_gradient_constriction(
            reference_model='carreau_yasuda',
            reference_params=[0.02, 1.0, 5.0, 0.5, 2.0],
            tbnn_hidden_units=[30, 30],
            eta_init=1.0,
            outer_steps=100,
            learning_rate=1e-3,
            s_floor=0.7,                 # Broader kernels (smoother viscosity curves)
            alpha_temp=2.0,              # More diffuse weights (less sharp transitions)
            num_update_steps=5
        )
        
        # Customize learning rate scaling for global scalars
        results_custom_lr = debug_one_step_gradient_constriction(
            reference_model='carreau_yasuda',
            reference_params=[0.02, 1.0, 5.0, 0.5, 2.0],
            tbnn_hidden_units=[30, 30],
            eta_init=1.0,
            learning_rate=1e-3,
            global_scalar_lr_scale=5.0,  # Use 5x LR for eta_inf_raw, delta_raw (default: 10.0)
            num_update_steps=5
        )
        
        # Freeze eta0 to learn only curvature (forces PDE to learn shape, not magnitude)
        results_freeze_eta0 = debug_one_step_gradient_constriction(
            reference_model='carreau_yasuda',
            reference_params=[0.02, 1.0, 5.0, 0.5, 2.0],
            tbnn_hidden_units=[30, 30],
            eta_init=1.0,
            learning_rate=1e-3,
            freeze_eta0=True,          # Freeze zero-shear viscosity
            eta0_fixed=1.0,            # Must match target's eta0
            eta0_eps=1e-6,             # Small epsilon for numerical stability
            num_update_steps=10
        )
        
        # Prevent curvature before a threshold (clean fix for f32 issues + stops Newtonian collapse)
        results_bounded_centers = debug_one_step_gradient_constriction(
            reference_model='carreau_yasuda',
            reference_params=[0.02, 1.0, 5.0, 0.5, 2.0],
            tbnn_hidden_units=[30, 30],
            eta_init=1.0,
            learning_rate=1e-3,
            freeze_eta0=True,          # Freeze eta0
            eta0_fixed=1.0,
            mu_min_gamma=1e-2,         # All centers >= log(0.01/gamma_ref) - no curvature before 0.01 s^-^1
            gate_gamma=1e-2,           # Additionally gate mixture on at 0.01
            gate_width_z=0.5,          # Smooth transition (~half-decade)
            num_update_steps=10
        )
        
        # Use two-stage CY pretraining to set up better initial parameters
        results_pretrained = debug_one_step_gradient_constriction(
            reference_model='carreau_yasuda',
            reference_params=[0.02, 1.0, 5.0, 0.5, 2.0],
            tbnn_hidden_units=[30, 30],
            eta_init=1.0,
            learning_rate=1e-3,
            freeze_eta0=True,          # Freeze eta0
            eta0_fixed=1.0,
            pretrain_cy=True,          # Enable two-stage CY pretraining
            pretrain_cy_steps_1=50,    # Stage 1: fit exact CY target
            pretrain_cy_steps_2=10,    # Stage 2: fit near-Newtonian CY (n=0.95)
            pretrain_cy_n2_target=0.95,# Target power-law index for stage 2
            num_update_steps=10
        )
        
        # Use velocity shape loss to learn spatial patterns before magnitudes converge
        results_shape = debug_one_step_gradient_constriction(
            reference_model='carreau_yasuda',
            reference_params=[0.02, 1.0, 5.0, 0.5, 2.0],
            tbnn_hidden_units=[30, 30],
            eta_init=1.0,
            learning_rate=1e-3,
            shape_loss=True,           # Enable scale-invariant shape loss
            shape_weight=0.1,          # Weight for shape term
            num_update_steps=10
        )
        
        # Freeze centers to learn only widths and weights (not locations)
        results_frozen_centers = debug_one_step_gradient_constriction(
            reference_model='carreau_yasuda',
            reference_params=[0.02, 1.0, 5.0, 0.5, 2.0],
            tbnn_hidden_units=[30, 30],
            eta_init=1.0,
            learning_rate=1e-3,
            freeze_centers=True,       # Freeze mu centers (learn only s and alpha)
            mu_min_gamma=1e-2,         # Initialize centers at 0.01 s^-^1 and above
            num_update_steps=10
        )
        
        # Use more mixture modes for extra expressiveness
        results_more_modes = debug_one_step_gradient_constriction(
            reference_model='carreau_yasuda',
            reference_params=[0.02, 1.0, 5.0, 0.5, 2.0],
            tbnn_hidden_units=[30, 30],
            eta_init=1.0,
            learning_rate=1e-3,
            M=8,                       # 8 sigmoid terms (more flexible viscosity)
            num_update_steps=10
        )
        
        # Use 2-stage training: learn etainf ONLY first, then freeze and refine curvature
        results_2stage_etainf_only = debug_one_step_gradient_constriction(
            reference_model='carreau_yasuda',
            reference_params=[0.02, 1.0, 5.0, 0.5, 2.0],
            tbnn_hidden_units=[30, 30],
            eta_init=1.0,
            learning_rate=1e-3,
            freeze_eta0=True,          # eta0 stays fixed throughout
            eta0_fixed=1.0,
            freeze_centers=True,       # mu centers frozen
            two_stage_etainf_then_curv=True,  # Enable 2-stage
            stage1_steps_etainf=10,    # Stage 1: learn etainf ONLY (max 10 steps)
            stage2_steps_curv=10,      # Stage 2: freeze etainf, learn curvature (10 steps)
            stage1_etainf_only=True,   # Stage 1 = etainf ONLY (freeze curvature)
            stage1_reset_momentum=True,      # Reset momentum at switch (recommended)
            stage1_early_stop_on_flip=True   # Stop stage 1 early if detainf flips sign (overshoot)
        )
        
        # Alternative: etainf + curvature in stage 1, then curvature-only in stage 2
        results_2stage_both = debug_one_step_gradient_constriction(
            reference_model='carreau_yasuda',
            reference_params=[0.02, 1.0, 5.0, 0.5, 2.0],
            tbnn_hidden_units=[30, 30],
            eta_init=1.0,
            learning_rate=1e-3,
            freeze_eta0=True,
            eta0_fixed=1.0,
            freeze_centers=True,
            two_stage_etainf_then_curv=True,
            stage1_steps_etainf=5,     # Stage 1: learn etainf + curvature together (5 steps)
            stage2_steps_curv=5,       # Stage 2: curvature-only (5 steps)
            stage1_etainf_only=False   # Stage 1 allows BOTH etainf and curvature to move
        )
    """
    print(f"\n{'='*80}")
    if two_stage_etainf_then_curv:
        total_train_steps = stage1_steps_etainf + stage2_steps_curv
        print(f"2-STAGE GRADIENT UPDATE WITH CONSTRICTION ({total_train_steps} steps, NEW MODEL)")
        stage1_desc = 'etainf ONLY (curvature frozen)' if stage1_etainf_only else 'etainf + curvature'
        print(f"  Stage 1: {stage1_desc} for {stage1_steps_etainf} steps")
        print(f"  Stage 2: curvature ONLY (etainf frozen) for {stage2_steps_curv} steps")
    elif num_update_steps == 1:
        print(f"ONE STEP GRADIENT UPDATE WITH CONSTRICTION (NEW MODEL)")
    else:
        print(f"MULTI-STEP GRADIENT UPDATE WITH CONSTRICTION ({num_update_steps} steps, NEW MODEL)")
    print(f"Reference Model: {reference_model.upper()}")
    print(f"TBNN Model: Mixture-of-Sigmoids Viscosity")
    print(f"{'='*80}")
    
    # Set defaults
    if reference_params is None:
        if reference_model == 'newtonian':
            reference_params = 0.5
        elif reference_model == 'carreau_yasuda':
            reference_params = jnp.array([0.02, 1.0, 5.0, 0.5, 2.0])
        elif reference_model == 'power_law':
            reference_params = jnp.array([0.5, 0.8])
    
    if tbnn_hidden_units is None:
        tbnn_hidden_units = [30, 30, 30]
    if domain is None:
        domain = DEFAULT_CONFIG['domain']
    if domain_size is None:
        domain_size = DEFAULT_CONFIG['domain_size']
    if inner_steps is None:
        inner_steps = DEFAULT_CONFIG['inner_steps']
    
    # Convert to JAX arrays
    if not isinstance(reference_params, jnp.ndarray):
        reference_params = jnp.array(reference_params) if hasattr(reference_params, '__iter__') else reference_params
    
    print(f"TBNN architecture: {tbnn_hidden_units}")
    print(f"TBNN mixture modes: M={M} sigmoid terms")
    print(f"Reference params: {reference_params}")
    print(f"TBNN eta init: {eta_init}")
    print(f"Pressure gradient: {pressure_gradient}")
    print(f"Time stepping: dt={dt}, inner={inner_steps}, outer={outer_steps}")
    print(f"Training: {num_update_steps} steps, Adam learning rate: {learning_rate}")
    print(f"  Global scalar LR scale: {global_scalar_lr_scale}x (eta_inf_raw, delta_raw)")
    if enable_pl_per_mode:
        print(f"  PL slope LR scale: {pl_lr_scale}x (pl_slope_raw)")
    print(f"Loss: Full-field library RMSE with particle masking")
    print(f"Viscosity regularization: {visc_loss}")
    if visc_loss:
        print(f"  gammadot range: [{reg_gamma_min}, {reg_gamma_max}], points: {reg_num_points}")
        print(f"  Slope cap: {reg_s_cap}, lam_slope: {reg_lambda_slope}, lam_curv: {reg_lambda_curv}")
    print(f"Velocity shape loss: {shape_loss}")
    if shape_loss:
        print(f"  Shape weight: {shape_weight} (scale-invariant RMSE)")
    mask_layout = resolve_mask_layout(mask_layout, log=True)
    print(f"Mask layout: {mask_layout}")
    print(f"PIV resolution mode: {resolution_piv}")
    if resolution_piv:
        print(f"  PIV window size: {piv_W_win} pixels")
        print(f"  PIV overlap: {piv_overlap} ({int(piv_overlap*100)}%)")
        print(f"  PIV kernel: {piv_kernel}")
        W_x, W_y = (int(piv_W_win), int(piv_W_win)) if isinstance(piv_W_win, (int, float)) else (int(piv_W_win[0]), int(piv_W_win[1]))
        s_x = max(1, int(round(W_x * (1.0 - float(piv_overlap)))))
        s_y = max(1, int(round(W_y * (1.0 - float(piv_overlap)))))
        print(f"  Vector spacing: {s_x}x{s_y} pixels ({s_x} streamwise, {s_y} wall-normal)")
        if add_piv_noise:
            print(f"   PIV noise ENABLED (applied to reference only):")
            print(f"     Noise level: {piv_noise_p_percent}% of U95")
            print(f"     Correlation: {piv_noise_corr_frac} x window half-width")
            print(f"     Heteroscedastic beta: {piv_noise_beta_grad}")
            print(f"     Static bias: {piv_noise_use_bias}")
            print(f"     Random seed: {piv_noise_seed} (deterministic)")
        else:
            print(f"  PIV noise: disabled (clean downsampled data)")
    print(f"Model parameters: s_floor={s_floor}, alpha_temp={alpha_temp}")
    if s_floor > 0 or alpha_temp != 1.0:
        print(f"  Using safer parametrization:")
        if s_floor > 0:
            print(f"     Min logistic width (s_floor): {s_floor} (broader kernels)")
        if alpha_temp != 1.0:
            print(f"     Softmax temperature (alpha_temp): {alpha_temp} ({'diffuse' if alpha_temp > 1.0 else 'sharp'} weights)")
    if pretrain_cy:
        print(f"   CY PRETRAIN ENABLED:")
        print(f"     Stage 1: {pretrain_cy_steps_1} steps (fit exact CY target)")
        print(f"     Stage 2: {pretrain_cy_steps_2} steps (fit near-Newtonian, n={pretrain_cy_n2_target})")
        print(f"     Freezes etainf/delta to learn only curvature parameters mu/s/alpha")
    if freeze_eta0:
        print(f"  eta0 FREEZE ENABLED:")
        print(f"     eta0_fixed: {eta0_fixed} (learn curvature only)")
        print(f"     eta0_eps: {eta0_eps}")
    if mu_min_gamma is not None or gate_gamma is not None or tail_gate_gamma is not None:
        print(f"  CURVATURE CONTROL ENABLED:")
        if mu_min_gamma is not None:
            print(f"     mu_min_gamma: {mu_min_gamma} (no curvature before this gammadot)")
        if mu_max_gamma is not None:
            print(f"     mu_max_gamma: {mu_max_gamma} (no curvature after this gammadot)")
        if gate_gamma is not None:
            print(f"     gate_gamma: {gate_gamma}, gate_width_z: {gate_width_z} (smooth gate)")
        if tail_gate_gamma is not None:
            print(f"     tail_gate_gamma: {tail_gate_gamma}, tail_gate_width_z: {tail_gate_width_z} (defer etainf)")
    if enable_pl_per_mode:
        print(f"  PER-MODE POWER-LAW BUMPS ENABLED:")
        print(f"     pl_width_z: {pl_width_z} (smooth onset per mode)")
    if freeze_centers:
        print(f"  FREEZE CENTERS ENABLED:")
        print(f"     mu centers frozen (learn only widths s and weights alpha)")
    print(f"Save plots: {save_plots}, Output dir: {output_dir}")
    print(f"Save trajectory info: {save_traj_info}")
    if save_traj_info and checkpoint_every > 0:
        print(f"  Checkpointing enabled: saving params every {checkpoint_every} steps")
    
    # Create output directory if saving plots or trajectory info
    if save_plots or save_traj_info:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
        
        # Create subdirectory for trajectory data
        if save_traj_info:
            traj_data_dir = os.path.join(output_dir, "trajectory_data")
            os.makedirs(traj_data_dir, exist_ok=True)
            print(f"Created trajectory data directory: {traj_data_dir}")
    
    # Setup grid and flow conditions
    grid = flow_conditions.create_grid(domain_size, domain)
    
    flow_cond = {
        'density': DEFAULT_CONFIG['density'],
        'base_viscosity': 0.0,
        'pressure_gradient': pressure_gradient,
        'dt': dt,
        'U_f': 0.0,
        'grid': grid,
        'amp_shear': 0.0,
        'freq_osc': 0.0,
        'nu0_baseline': 0.0,
        'inner_steps': inner_steps,
        'outer_steps': outer_steps,
        'boundary_type': DEFAULT_CONFIG['boundary_type'],
        'solver_type': solver_type,
        'stepper_type': stepper_type
    }
    
    # Setup constriction geometry
    print("\n1. Setting up constriction geometry...")
    particles = setup_channel_constriction(domain)
    print("Constriction geometry created")
    
    # Setup TBNN model
    print("\n2. Setting up TBNN model...")
    model_info = setup_tbnn_model(tbnn_hidden_units, domain_size, eta_init, random_seed, 
                                  use_soft_newtonian_init, s_floor, alpha_temp,
                                  freeze_eta0, eta0_fixed, eta0_eps,
                                  mu_min_gamma, mu_max_gamma, gate_gamma, gate_width_z,
                                  tail_gate_gamma, tail_gate_width_z,
                                  enable_pl_per_mode, pl_width_z,
                                  log_head, log_mixing, freeze_centers, M)
    
    # --- Read initial plateau values via introspection (works for all heads) ---
    eta_inf0, eta00, delta0, mu0 = inspect_head(model_info['model'], model_info['params'])
    print(f"   INIT: etainf={eta_inf0:.6g}, eta0={eta00:.6g}, Delta={delta0:.6g}")
    print(f"   INIT mu (z=log gammadot/gamma_ref): {np.array2string(mu0, precision=4, separator=', ')}")
    
    # --- Optional two-stage CY pretrain (shape-only; respects current clamps/gates) ---
    if pretrain_cy:
        if reference_model != 'carreau_yasuda' or not hasattr(reference_params, '__len__') or len(reference_params) < 5:
            raise ValueError("pretrain_cy=True requires reference_model='carreau_yasuda' with 5 params [etainf, eta0, lam, n, a].")
        print(f"\n2b. Pretraining shape to CY: n={float(reference_params[3]):.3f} -> n={float(pretrain_cy_n2_target):.3f}")
        new_params = _pretrain_tbnn_cy_shape_only(
            model_info, cy_params=reference_params,
            n2_target=float(pretrain_cy_n2_target),
            steps1=int(pretrain_cy_steps_1), steps2=int(pretrain_cy_steps_2),
            lr=float(max(1e-3, learning_rate))
        )
        model_info['params'] = new_params
        model_info['flattened_params'] = parameter_utils.flatten_params(new_params)[0]
        _ = plotting.plot_viscosity_strain_rate(
            model_info, "After CY Pretrain: TBNN Viscosity vs Strain Rate (Constriction)",
            save_plots, output_dir, "pretrain_cy_viscosity",
            reference_model=reference_model, reference_params=reference_params
        )
    
    # Add TBNN-specific fields to flow_cond
    flow_cond.update({
        'tree_def': model_info['tree_def'],
        'shapes': model_info['shapes'],
        'starts_static': model_info['starts_static'],
        'ends_static': model_info['ends_static']
    })
    
    # Plot initial viscosity behavior immediately after model setup
    print("\n3. Plotting initial TBNN viscosity behavior...")
    initial_visc_stats = plotting.plot_viscosity_strain_rate(
        model_info, "Initial TBNN Viscosity vs Strain Rate (Constriction)",
        save_plots, output_dir, "initial_viscosity_constriction",
        reference_model=reference_model, reference_params=reference_params
    )
    
    # Save initial viscosity data if requested
    if save_traj_info:
        np.save(os.path.join(traj_data_dir, "initial_strain_rates.npy"), 
                np.array(initial_visc_stats['strain_rates']))
        np.save(os.path.join(traj_data_dir, "initial_viscosities.npy"), 
                np.array(initial_visc_stats['viscosities']))
        print(f"   Saved initial viscosity data to {traj_data_dir}")
    
    # Create reference trajectory
    print("\n4. Creating reference trajectory...")
    try:
        reference_trajectory, ref_result = create_reference_trajectory_constriction(
            reference_model, reference_params, flow_cond, particles, outer_steps
        )
        print("wrote reference trajectory")
        
        # Save reference trajectory data if requested
        if save_traj_info:
            ref_traj_x, ref_traj_y = reference_trajectory
            np.save(os.path.join(traj_data_dir, "reference_trajectory_x.npy"), np.array(ref_traj_x))
            np.save(os.path.join(traj_data_dir, "reference_trajectory_y.npy"), np.array(ref_traj_y))
            np.save(os.path.join(traj_data_dir, "reference_velocity_x.npy"), np.array(ref_result.velocity[0].data))
            np.save(os.path.join(traj_data_dir, "reference_velocity_y.npy"), np.array(ref_result.velocity[1].data))
            print(f"   Saved reference trajectory data to {traj_data_dir}")
        
        # If PIV mode is enabled, create and plot PIV-downsampled reference for visualization
        if resolution_piv:
            print("\n4b. Creating PIV-downsampled reference trajectory for visualization...")
            try:
                # Extract reference trajectory components
                ref_traj_x, ref_traj_y = reference_trajectory
                
                # Transpose reference from jax-cfd format (T, nx, ny) to (T, H, W) = (T, ny, nx)
                if ref_traj_x.ndim == 3:
                    ref_traj_x = jnp.transpose(ref_traj_x, (0, 2, 1))  # (T, 256, 128) -> (T, 128, 256)
                    ref_traj_y = jnp.transpose(ref_traj_y, (0, 2, 1))
                
                # Downsample to PIV resolution (use last N frames)
                W_x, W_y = (int(piv_W_win), int(piv_W_win)) if isinstance(piv_W_win, (int, float)) else (int(piv_W_win[0]), int(piv_W_win[1]))
                s_x = max(1, int(round(W_x * (1.0 - float(piv_overlap)))))
                s_y = max(1, int(round(W_y * (1.0 - float(piv_overlap)))))
                
                grid = flow_cond['grid']
                (x_min, x_max), (y_min, y_max) = grid.domain
                Lx = float(x_max - x_min)
                Ly = float(y_max - y_min)
                
                # Downsample reference trajectory
                ref_piv_x, ref_piv_y, x_c, y_c = piv_downsample_THW(
                    ref_traj_x, ref_traj_y, W_x, W_y, s_x, s_y, x_min, y_min, Lx, Ly, kernel=piv_kernel
                )
                
                # Optionally add noise for visualization
                noise_params_dict = None
                if add_piv_noise and piv_noise_p_percent > 0.0:
                    key_vis = jax.random.PRNGKey(int(piv_noise_seed))
                    ref_piv_speed = jnp.sqrt(ref_piv_x**2 + ref_piv_y**2)
                    u95 = jnp.asarray(jnp.percentile(ref_piv_speed, 95.0), jnp.float32)
                    sigma_base = (jnp.float32(piv_noise_p_percent) / 100.0) * (u95 + 1e-6)
                    
                    H_full = ref_traj_x.shape[1]
                    W_full = ref_traj_x.shape[2]
                    
                    ref_piv_x, ref_piv_y, _ = add_piv_noise_jax(
                        ref_piv_x, ref_piv_y,
                        W_x=W_x, W_y=W_y, s_x=s_x, s_y=s_y,
                        Lx=Lx, Ly=Ly,
                        key=key_vis,
                        corr_frac=float(piv_noise_corr_frac),
                        sigma_base=float(sigma_base),
                        beta_grad=float(piv_noise_beta_grad),
                        use_bias=bool(piv_noise_use_bias),
                        full_grid_shape=(W_full, H_full)
                    )
                    noise_params_dict = {
                        'p_percent': piv_noise_p_percent,
                        'corr_frac': piv_noise_corr_frac,
                        'beta_grad': piv_noise_beta_grad,
                        'use_bias': piv_noise_use_bias
                    }
                
                # Plot comparison (transpose back to (nx, ny) for plotting function)
                piv_title = f"PIV Resolution Effect: {W_x}x{W_y} windows, {int(piv_overlap*100)}% overlap"
                if add_piv_noise and piv_noise_p_percent > 0.0:
                    piv_title += f" + {piv_noise_p_percent}% noise"
                
                # Plotting function expects (T, nx, ny) format, so transpose back
                ref_traj_x_plot = jnp.transpose(ref_traj_x, (0, 2, 1)) if ref_traj_x.ndim == 3 else ref_traj_x.T
                ref_traj_y_plot = jnp.transpose(ref_traj_y, (0, 2, 1)) if ref_traj_y.ndim == 3 else ref_traj_y.T
                ref_piv_x_plot = jnp.transpose(ref_piv_x, (0, 2, 1)) if ref_piv_x.ndim == 3 else ref_piv_x.T
                ref_piv_y_plot = jnp.transpose(ref_piv_y, (0, 2, 1)) if ref_piv_y.ndim == 3 else ref_piv_y.T
                
                piv_stats = plotting.plot_piv_resolution_comparison(
                    ref_full_res=(ref_traj_x_plot, ref_traj_y_plot),
                    ref_piv_res=(ref_piv_x_plot, ref_piv_y_plot),
                    domain=domain,
                    title=piv_title,
                    save_plots=save_plots,
                    output_dir=output_dir,
                    file_prefix="piv_resolution_effect",
                    noise_params=noise_params_dict
                )
                
                print(f"PIV resolution comparison plot created")
                print(f"   Full resolution: {ref_traj_x.shape[1]}x{ref_traj_x.shape[2]} -> PIV resolution: {ref_piv_x.shape[1]}x{ref_piv_x.shape[2]}")
                print(f"   Data reduction: {piv_stats['reduction_factor']:.1f}x fewer points")
                
            except Exception as e:
                print(f"PIV comparison plot failed: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"Failed to create reference trajectory: {e}")
        return None
    
    # Create viscosity loss configuration if enabled
    visc_loss_config = None
    if visc_loss:
        visc_loss_config = {
            'reg_gamma_min': reg_gamma_min,
            'reg_gamma_max': reg_gamma_max,
            'reg_num_points': reg_num_points,
            'reg_s_cap': reg_s_cap,
            'reg_lambda_slope': reg_lambda_slope,
            'reg_lambda_curv': reg_lambda_curv,
            'reg_curv_cap_abs': reg_curv_cap_abs,
            'reg_p_slope': reg_p_slope,
            'reg_p_curv': reg_p_curv
        }
    
    # Define loss function for gradient computation (with optional warmup+tail)
    def loss_fn(params):
        if use_warmup_tail:
            # Use warmup+tail approach to control gradient explosion
            total_steps = flow_cond['outer_steps']
            
            # Use provided parameters or auto-calculate
            if warmup_steps is not None:
                w_steps = warmup_steps
            else:
                w_steps = min(total_steps * 3 // 4, 300)  # Use 3/4 for warmup, up to 300
                
            if tail_steps is not None:
                t_steps = tail_steps
            else:
                t_steps = total_steps - w_steps
            
            loss, _, _ = compute_tbnn_trajectory_loss_constriction(
                params, flow_cond, model_info, particles, reference_trajectory,
                warmup=w_steps, tail=t_steps, detach=True,
                visc_loss=visc_loss, visc_loss_config=visc_loss_config,
                shape_loss=shape_loss, shape_weight=shape_weight,
                mask_layout=mask_layout,
                resolution_piv=resolution_piv, piv_W_win=piv_W_win, piv_overlap=piv_overlap, piv_kernel=piv_kernel,
                add_piv_noise=add_piv_noise, piv_noise_p_percent=piv_noise_p_percent,
                piv_noise_corr_frac=piv_noise_corr_frac, piv_noise_beta_grad=piv_noise_beta_grad,
                piv_noise_use_bias=piv_noise_use_bias, piv_noise_seed=piv_noise_seed
            )
        else:
            # Original behavior: all steps differentiable, no warmup
            loss, _, _ = compute_tbnn_trajectory_loss_constriction(
                params, flow_cond, model_info, particles, reference_trajectory,
                warmup=0, tail=None, detach=True,
                visc_loss=visc_loss, visc_loss_config=visc_loss_config,
                shape_loss=shape_loss, shape_weight=shape_weight,
                mask_layout=mask_layout,
                resolution_piv=resolution_piv, piv_W_win=piv_W_win, piv_overlap=piv_overlap, piv_kernel=piv_kernel,
                add_piv_noise=add_piv_noise, piv_noise_p_percent=piv_noise_p_percent,
                piv_noise_corr_frac=piv_noise_corr_frac, piv_noise_beta_grad=piv_noise_beta_grad,
                piv_noise_use_bias=piv_noise_use_bias, piv_noise_seed=piv_noise_seed
            )
        return loss
    
    # Gradient reporting function to verify freeze
    def grad_report(flat_grad):
        """Report gradient counts by parameter group to verify freezing."""
        g = parameter_utils.unflatten_params(flat_grad, model_info['tree_def'], model_info['shapes'])
        flat = traverse_util.flatten_dict(unfreeze(g), keep_empty_nodes=True)
        z = lambda a: int(jnp.sum(jnp.abs(a) > 1e-12))
        c = {"etainf": 0, "delta": 0, "pl": 0, "curv": 0}
        for k, v in flat.items():
            name = "/".join(k)
            if name.endswith(("eta_inf_raw","log_eta_inf_raw","eta_partition_logit")):
                c["etainf"] += z(v)  # etainf or its reparameterization (partition in add-mode)
            elif name.endswith(("delta_raw","r_raw","log_range_raw")):
                c["delta"] += z(v)
            elif name.endswith("pl_slope_raw"):
                c["pl"] += z(v)
            else:
                c["curv"] += z(v)
        print(f"   Gradient counts: etainf={c['etainf']}  delta/eta0={c['delta']}  PL={c['pl']}  curvature={c['curv']}")
        return c
    
    def extract_etainf_gradient(flat_grad):
        """Extract the etainf gradient WITH SIGN (handles all head variants)."""
        g = parameter_utils.unflatten_params(flat_grad, model_info['tree_def'], model_info['shapes'])
        flat = traverse_util.flatten_dict(unfreeze(g), keep_empty_nodes=True)
        for k, v in flat.items():
            name = "/".join(k)
            # Check all possible etainf-related parameter names - PRESERVE SIGN!
            if name.endswith("log_eta_inf_raw"):
                return float(v.item())  # log-head: gradient of log_eta_inf_raw (with sign)
            elif name.endswith("eta_partition_logit"):
                return float(v.item())  # add-freeze: gradient of partition logit (with sign)
            elif name.endswith("eta_inf_raw"):
                return float(v.item())  # linear-head: gradient of eta_inf_raw (with sign)
        return 0.0  # Not found or frozen
    
    # Compute initial loss and gradients
    print("\n5. Computing initial loss and gradients...")
    start_time = time.time()
    
    try:
        val_and_grad = jax.value_and_grad(loss_fn)
        initial_loss, gradient = val_and_grad(model_info['flattened_params'])
        grad_time = time.time() - start_time
        
        print(f"Initial loss: {float(initial_loss):.6e}")
        
        print(f"Gradient computation completed in {grad_time:.2f} seconds")
        print(f"Gradient shape: {gradient.shape}")
        
        grad_magnitude = jnp.linalg.norm(gradient)
        print(f"Gradient magnitude: {grad_magnitude:.6e}")
        
        # Check gradient properties
        has_nan_grad = jnp.any(jnp.isnan(gradient))
        has_inf_grad = jnp.any(jnp.isinf(gradient))
        num_nonzero = jnp.sum(jnp.abs(gradient) > 1e-12)
        
        print(f"Has NaN: {has_nan_grad}, Has Inf: {has_inf_grad}")
        print(f"Non-zero gradients: {int(num_nonzero)}/{len(gradient)}")
        
        # Report gradient distribution (before any freezing)
        print(f"Initial gradient distribution (before optimizer freezing):")
        grad_counts = grad_report(gradient)
        
        # Extract and print etainf gradient for monitoring (WITH SIGN to detect flips)
        initial_etainf_grad = extract_etainf_gradient(gradient)
        print(f"Initial etainf gradient: detainf = {initial_etainf_grad:+.3e} (sign matters!)")
        
        if has_nan_grad or has_inf_grad:
            print("WARNING: Gradient contains NaN or Inf values!")
            return None
            
    except Exception as e:
        print(f"Gradient computation failed: {e}")
        traceback.print_exc()
        return None
    
    # Run initial TBNN simulation for comparison
    print("\n6. Running initial TBNN simulation...")
    try:
        initial_tbnn_loss, initial_tbnn_result, initial_tbnn_trajectory = compute_tbnn_trajectory_loss_constriction(
            model_info['flattened_params'], flow_cond, model_info, particles, reference_trajectory,
            visc_loss=visc_loss, visc_loss_config=visc_loss_config,
            shape_loss=shape_loss, shape_weight=shape_weight,
            mask_layout=mask_layout,
            resolution_piv=resolution_piv, piv_W_win=piv_W_win, piv_overlap=piv_overlap, piv_kernel=piv_kernel,
            add_piv_noise=add_piv_noise, piv_noise_p_percent=piv_noise_p_percent,
            piv_noise_corr_frac=piv_noise_corr_frac, piv_noise_beta_grad=piv_noise_beta_grad,
            piv_noise_use_bias=piv_noise_use_bias, piv_noise_seed=piv_noise_seed
        )
        if initial_tbnn_result is not None:
            print("Initial TBNN simulation completed")
            
            # Save initial TBNN results if requested
            if save_traj_info:
                np.save(os.path.join(traj_data_dir, "initial_tbnn_velocity_x.npy"), 
                        np.array(initial_tbnn_result.velocity[0].data))
                np.save(os.path.join(traj_data_dir, "initial_tbnn_velocity_y.npy"), 
                        np.array(initial_tbnn_result.velocity[1].data))
                # Save initial TBNN trajectory
                if initial_tbnn_trajectory is not None:
                    if hasattr(initial_tbnn_trajectory[0], 'data'):
                        np.save(os.path.join(traj_data_dir, "initial_tbnn_trajectory_x.npy"), 
                                np.array(initial_tbnn_trajectory[0].data))
                        np.save(os.path.join(traj_data_dir, "initial_tbnn_trajectory_y.npy"), 
                                np.array(initial_tbnn_trajectory[1].data))
                    else:
                        np.save(os.path.join(traj_data_dir, "initial_tbnn_trajectory_x.npy"), 
                                np.array(initial_tbnn_trajectory[0]))
                        np.save(os.path.join(traj_data_dir, "initial_tbnn_trajectory_y.npy"), 
                                np.array(initial_tbnn_trajectory[1]))
                # Save initial model parameters (both formats)
                import pickle
                # Flattened array (for inspection)
                np.save(os.path.join(traj_data_dir, "initial_tbnn_params.npy"), 
                        np.array(model_info['flattened_params']))
                # Structured params (for forward simulations)
                with open(os.path.join(traj_data_dir, "initial_tbnn_params.pkl"), 'wb') as f:
                    pickle.dump(model_info['params'], f)
                # Reconstruction info (tree_def + shapes)
                with open(os.path.join(traj_data_dir, "initial_tbnn_tree_def.pkl"), 'wb') as f:
                    pickle.dump(model_info['tree_def'], f)
                np.save(os.path.join(traj_data_dir, "initial_tbnn_shapes.npy"), 
                        np.array(model_info['shapes'], dtype=object))
                print(f"   Saved initial TBNN simulation data to {traj_data_dir}")
                print(f"      (saved both .npy flat + .pkl structured + tree_def/shapes)")
        else:
            print("Initial TBNN simulation failed")
            return None
    except Exception as e:
        print(f"Initial TBNN simulation failed: {e}")
        return None
    
    # Plot initial comparison immediately after simulation
    print("\n7. Plotting initial model comparison...")
    initial_stats = plotting.plot_model_comparison_constriction(
        initial_tbnn_result, ref_result, domain, 
        f"Initial TBNN vs {reference_model.title()} Reference (Constriction)",
        save_plots, output_dir, "initial_comparison_constriction"
    )
    
    # Plot histogram of local strain rates immediately
    print("\n8. Analyzing local strain rate distribution...")
    strain_rate_stats = plotting.plot_strain_rate_histogram_constriction(
        initial_tbnn_result, domain, "Initial TBNN Training Example (Constriction)",
        save_plots, output_dir, "strain_rate_histogram_constriction"
    )
    
    # Multi-step training loop with Adam optimizer and failure detection
    print(f"\n9. Running {num_update_steps}-step Adam training (learning rate: {learning_rate})...")
    
    # Setup Adam optimizer - freeze both etainf and delta (which freezes eta0 too)
    def make_label_tree(params_pytree):
        """Label parameter groups: freeze etainf and delta, learn only curvature (mu, s, alpha)."""
        flat = traverse_util.flatten_dict(unfreeze(params_pytree), keep_empty_nodes=True)
        labeled = {}
        etainf_names = []
        delta_names = []
        pl_names = []
        for k in flat.keys():
            name = "/".join(k)
            if name.endswith(("eta_inf_raw","log_eta_inf_raw","eta_partition_logit")):
                labeled[k] = "etainf"          # etainf (or its reparameterization)
                etainf_names.append(name)
            elif name.endswith(("delta_raw","r_raw","log_range_raw")):
                labeled[k] = "delta"           # delta = eta0-etainf  (frozen -> fixes eta0 too)
                delta_names.append(name)
            elif name.endswith("pl_slope_raw"):
                labeled[k] = "pl_slopes"       # optional: per-mode PL bumps
                pl_names.append(name)
            else:
                labeled[k] = "others"          # mu, s, alpha, dense layers (curvature)
        if etainf_names:
            print(f"   FROZEN etainf: {', '.join(etainf_names)}")
        if delta_names:
            # Distinguish add vs geom freeze mode in prints
            if log_head and log_mixing == "geom" and freeze_eta0:
                print(f"   eta0 fixed via L = log(eta0_fixed)  log(etainf); etainf is learnable")
            elif freeze_eta0:
                print(f"   eta0 fixed via partition/delta; etainf is learnable: {', '.join(delta_names)}")
            else:
                print(f"   FROZEN delta (fixes eta0): {', '.join(delta_names)}")
        if pl_names:
            print(f"   PL slopes (LR x {pl_lr_scale}): {', '.join(pl_names)}")
        labels = traverse_util.unflatten_dict(labeled)
        labels = freeze(labels) if isinstance(params_pytree, FrozenDict) else labels
        return labels
    
    # Group-wise gradient equalizer (optional)
    def label_param_groups(params_pytree):
        """Label parameter groups for gradient equalization."""
        flat = traverse_util.flatten_dict(unfreeze(params_pytree), keep_empty_nodes=True)
        labels = {}
        for k, _ in flat.items():
            name = "/".join(k)
            if name.endswith(("eta_inf_raw","delta_raw","log_eta_inf_raw","r_raw","log_range_raw","eta_partition_logit")):
                labels[k] = "tail"      # etainf / delta / log variants / partition
            elif name.endswith("pl_slope_raw"):
                labels[k] = "pl"        # per-mode PL slopes
            else:
                labels[k] = "mix"       # Dense_* weights/biases for mu,s,alpha
        return freeze(traverse_util.unflatten_dict(labels))
    
    def equalize_group_grads(grads, group_labels, *, target="mix", cap_ratio=0.5, eps=1e-12):
        """Rescale gradients so tail/PL groups don't dwarf the target group."""
        gflat = traverse_util.flatten_dict(unfreeze(grads), keep_empty_nodes=True)
        lflat = traverse_util.flatten_dict(unfreeze(group_labels), keep_empty_nodes=True)

        # L2 norms per group
        def _accum(norms, k, g):
            grp = lflat[k]
            val = jnp.sqrt(jnp.sum(jnp.square(g)))
            return {**norms, grp: norms.get(grp, 0.0) + val}
        norms = {}
        for k, g in gflat.items():
            norms = _accum(norms, k, g)

        tgt = jnp.asarray(norms.get(target, 0.0))
        # scale factors: shrink 'tail' and 'pl' when they dwarf 'mix'
        s_tail = jnp.minimum(1.0, cap_ratio * tgt / (jnp.asarray(norms.get("tail", 0.0)) + eps))
        s_pl   = jnp.minimum(1.0, cap_ratio * tgt / (jnp.asarray(norms.get("pl",   0.0)) + eps))

        # apply scales
        for k in gflat.keys():
            grp = lflat[k]
            if grp == "tail":
                gflat[k] = gflat[k] * s_tail
            elif grp == "pl":
                gflat[k] = gflat[k] * s_pl
            # 'mix' stays untouched
        out = traverse_util.unflatten_dict(gflat)
        return freeze(out) if isinstance(grads, FrozenDict) else out
    
    # Use structured params for optimizer
    current_structured_params = model_info['params']
    
    # Label parameter groups if equalizer enabled
    PARAM_GROUPS = label_param_groups(current_structured_params) if enable_grad_equalizer else None
    if enable_grad_equalizer:
        print(f"    Gradient equalizer enabled: protecting '{equalize_target}' group (cap_ratio={equalize_cap_ratio})")
    
    # --- helper to build an optax multi-transform with stage control ---
    def _init_tx_for_stage(mode: str):
        """Create (tx, opt_state) for different training stages.
        
        Args:
            mode: 'etainf_only', 'curvature_only', or 'both'
        """
        labels = make_label_tree(current_structured_params)
        
        if mode == 'etainf_only':
            # Stage 1: ONLY etainf learns (freeze curvature)
            tx_local = optax.multi_transform(
                {
                    "others":     optax.chain(optax.scale(0.0)),                      # freeze mu, s, alpha
                    "pl_slopes":  optax.chain(optax.scale(0.0)),                      # freeze PL slopes
                    "etainf":     optax.adam(learning_rate * global_scalar_lr_scale), # etainf updates
                    "delta":      optax.chain(optax.scale(0.0)),                      # freeze delta
                },
                param_labels=labels
            )
        elif mode == 'curvature_only':
            # Stage 2: ONLY curvature learns (freeze etainf)
            tx_local = optax.multi_transform(
                {
                    "others":     optax.adam(learning_rate),                          # mu, s, alpha learn
                    "pl_slopes":  optax.adam(learning_rate * pl_lr_scale),           # PL slopes learn
                    "etainf":     optax.chain(optax.scale(0.0)),                      # freeze etainf
                    "delta":      optax.chain(optax.scale(0.0)),                      # freeze delta
                },
                param_labels=labels
            )
        else:  # 'both'
            # Both etainf and curvature learn
            tx_local = optax.multi_transform(
                {
                    "others":     optax.adam(learning_rate),                          # mu, s, alpha learn
                    "pl_slopes":  optax.adam(learning_rate * pl_lr_scale),           # PL slopes learn
                    "etainf":     optax.adam(learning_rate * global_scalar_lr_scale), # etainf learns
                    "delta":      optax.chain(optax.scale(0.0)),                      # freeze delta
                },
                param_labels=labels
            )
        
        opt_state_local = tx_local.init(current_structured_params)
        return tx_local, opt_state_local
    
    if HAVE_OPTAX:
        # Decide total steps & initial stage mode
        if two_stage_etainf_then_curv:
            # 2-stage mode: explicit control via stage parameters
            total_steps = int(stage1_steps_etainf + stage2_steps_curv)
            if stage1_steps_etainf > 0:
                current_stage_mode = 'etainf_only' if stage1_etainf_only else 'both'
            else:
                current_stage_mode = 'curvature_only'
        else:
            # Normal mode: check freeze_eta0 flag
            total_steps = int(num_update_steps)
            if freeze_eta0:
                # freeze_eta0=True means eta0 is frozen (via model), so etainf should be FREE to learn
                current_stage_mode = 'both'  # etainf + curvature both learn
            else:
                # freeze_eta0=False means both can move (old behavior: freeze both via optimizer)
                current_stage_mode = 'curvature_only'
        
        tx, opt_state = _init_tx_for_stage(current_stage_mode)
        if two_stage_etainf_then_curv:
            stage1_desc = 'etainf ONLY' if stage1_etainf_only else 'etainf + curvature'
            early_stop_msg = ' (early stop on detainf flip)' if stage1_early_stop_on_flip else ''
            print(f"   Using Optax Adam (2-stage): stage1 {stage1_desc} for up to {stage1_steps_etainf} step(s){early_stop_msg}, then curvature-only for {stage2_steps_curv} step(s)")
        else:
            if freeze_eta0:
                print(f"   Using Optax Adam: eta0 frozen (model), etainf + curvature learning (optimizer)")
            else:
                print(f"   Using Optax Adam: curvature-only (etainf & eta0 frozen via optimizer)")
    else:
        # Fallback manual Adam
        b1, b2, eps = 0.9, 0.999, 1e-8
        opt_state = {
            'm': jax.tree_util.tree_map(jnp.zeros_like, current_structured_params),
            'v': jax.tree_util.tree_map(jnp.zeros_like, current_structured_params),
            't': 0
        }
        print(f"   Using fallback Adam implementation")
    
    # Initialize training tracking
    loss_history = [float(initial_loss)]
    gradient_magnitudes = [float(grad_magnitude)]
    current_params = current_structured_params  # Use structured params
    current_model_info = model_info.copy()
    current_loss = initial_loss
    current_gradient_flat = gradient  # Already computed gradient at initial params
    
    # State tracking for rollback in case of simulation failure
    last_good_params = current_params
    last_good_model_info = current_model_info.copy()
    last_good_loss = float(initial_loss)
    last_good_step = 0
    simulation_failed = False
    failure_threshold = 900.0  # Consider loss > 900 as simulation failure
    
    # Stage 1 checkpoint (for 2-stage training)
    stage1_model_info = None
    stage1_viscosities = None
    stage1_early_stopped = False
    prev_etainf_grad = None  # Track previous gradient for sign flip detection
    
    print(f"   Step 0: Loss = {float(initial_loss):.6e}, |grad| = {float(grad_magnitude):.6e}")
    print(f"   Failure detection active (threshold: {failure_threshold})")
    print(f"   Optimizer will freeze etainf and delta (only curvature parameters will update)")
    
    # Define Adam step function that takes PRE-COMPUTED gradients
    def adam_step_optax(params, state, grads_flat):
        """One Adam step using Optax with pre-computed gradients."""
        # Unflatten gradients
        grads = parameter_utils.unflatten_params(grads_flat, model_info['tree_def'], model_info['shapes'])
        # Apply gradient equalizer if enabled
        if enable_grad_equalizer:
            grads = equalize_group_grads(grads, PARAM_GROUPS, target=equalize_target, cap_ratio=equalize_cap_ratio)
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params, state
    
    def adam_step_manual(params, state, grads_flat):
        """One Adam step using manual implementation with pre-computed gradients."""
        # Unflatten gradients
        grads = parameter_utils.unflatten_params(grads_flat, model_info['tree_def'], model_info['shapes'])
        
        # Apply gradient equalizer if enabled
        if enable_grad_equalizer:
            grads = equalize_group_grads(grads, PARAM_GROUPS, target=equalize_target, cap_ratio=equalize_cap_ratio)
        
        # Freeze etainf and delta by zeroing their gradients
        flat_g = traverse_util.flatten_dict(unfreeze(grads), keep_empty_nodes=True)
        for k in list(flat_g.keys()):
            name = "/".join(k)
            if name.endswith(("eta_inf_raw","log_eta_inf_raw","delta_raw","r_raw","log_range_raw","eta_partition_logit")):
                flat_g[k] = jnp.zeros_like(flat_g[k])   # freeze plateaus
            elif name.endswith("pl_slope_raw"):
                flat_g[k] = flat_g[k] * pl_lr_scale
        grads_scaled = freeze(traverse_util.unflatten_dict(flat_g))
        
        # Adam update
        state['t'] += 1
        m = jax.tree_util.tree_map(lambda m_, g_: b1*m_ + (1-b1)*g_, state['m'], grads_scaled)
        v = jax.tree_util.tree_map(lambda v_, g_: b2*v_ + (1-b2)*(g_**2), state['v'], grads_scaled)
        mhat = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1**state['t']), m)
        vhat = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2**state['t']), v)
        params = jax.tree_util.tree_map(lambda p_, m_, v_: p_ - learning_rate * (m_ / (jnp.sqrt(v_) + eps)), 
                                        params, mhat, vhat)
        state['m'], state['v'] = m, v
        return params, state
    
    adam_step = adam_step_optax if HAVE_OPTAX else adam_step_manual
    
    # Training loop - only compute gradients ONCE per step
    for step in range(total_steps):
        print(f"   Step {step+1}/{total_steps}:", end=" ")
        
        try:
            # Take Adam step using the CURRENT gradients (already computed)
            current_params, opt_state = adam_step(current_params, opt_state, current_gradient_flat)
            
            # Now compute loss and gradients at the UPDATED params (for next iteration and reporting)
            current_params_flat = parameter_utils.flatten_params(current_params)[0]
            new_loss, new_gradient_flat = value_and_grad(loss_fn)(current_params_flat)
            new_grad_magnitude = jnp.linalg.norm(new_gradient_flat)
            
            # Verify freezing at the right time:
            #   - single-stage: first step
            #   - two-stage: the first step AFTER we switch etainf OFF
            verify_now = ( (not two_stage_etainf_then_curv and step == 0) or
                           (two_stage_etainf_then_curv and (stage1_steps_etainf > 0) and (step+1 == stage1_steps_etainf+1)) )
            if verify_now:
                print("")  # newline
                print(f"      Verifying parameter freeze (etainf & delta) at this stage:")
                grad_counts_step1 = grad_report(new_gradient_flat)
                if grad_counts_step1["etainf"] == 0 and grad_counts_step1["delta"] == 0:
                    print(f"      Freeze verified: etainf and delta have zero gradients")
                else:
                    print(f"      WARNING: etainf or delta still have non-zero gradients!")
                print(f"   Step {step+1}/{total_steps}:", end=" ")
            
            # Check for simulation failure
            if float(new_loss) > failure_threshold:
                print(f"SIMULATION FAILURE! Loss = {float(new_loss):.6e} > {failure_threshold}")
                print(f"      Rolling back to last good state (Step {last_good_step})")
                simulation_failed = True
                break
            
            loss_history.append(float(new_loss))
            gradient_magnitudes.append(float(new_grad_magnitude))
            
            # Extract etainf gradient to monitor momentum effects (WITH SIGN to detect flips)
            etainf_grad = extract_etainf_gradient(new_gradient_flat)
            
            # Detect sign flip for early stopping
            sign_flipped = False
            if two_stage_etainf_then_curv and stage1_early_stop_on_flip and current_stage_mode != 'curvature_only':
                if prev_etainf_grad is not None and abs(etainf_grad) > 1e-12 and abs(prev_etainf_grad) > 1e-12:
                    # Check if signs are opposite (using multiplication: same sign = positive, opposite = negative)
                    if (etainf_grad * prev_etainf_grad) < 0:
                        sign_flipped = True
                        print(f"Loss = {float(new_loss):.6e}, |grad| = {float(new_grad_magnitude):.6e}, detainf = {etainf_grad:+.3e} FLIP!")
                    else:
                        print(f"Loss = {float(new_loss):.6e}, |grad| = {float(new_grad_magnitude):.6e}, detainf = {etainf_grad:+.3e}")
                else:
                    print(f"Loss = {float(new_loss):.6e}, |grad| = {float(new_grad_magnitude):.6e}, detainf = {etainf_grad:+.3e}")
                prev_etainf_grad = etainf_grad  # Update for next iteration
            else:
                print(f"Loss = {float(new_loss):.6e}, |grad| = {float(new_grad_magnitude):.6e}, detainf = {etainf_grad:+.3e}")
            
            # Check for NaN gradients
            if jnp.any(jnp.isnan(new_gradient_flat)) or jnp.any(jnp.isinf(new_gradient_flat)):
                print(f"       NaN/Inf gradients detected at step {step+1}, rolling back")
                simulation_failed = True
                break
            
            # Update current state for next iteration
            current_loss = new_loss
            current_gradient_flat = new_gradient_flat
            
            # Update last good state if this step was successful
            last_good_params = current_params
            last_good_model_info = current_model_info.copy()
            last_good_model_info['params'] = current_params
            last_good_model_info['flattened_params'] = current_params_flat
            last_good_loss = float(new_loss)
            last_good_step = step + 1
            
            # --- switch to curvature-only after N1 steps OR early if gradient flips ---
            switch_now = False
            early_stop_reason = ""
            
            if HAVE_OPTAX and two_stage_etainf_then_curv and current_stage_mode != 'curvature_only':
                # Normal switch at planned step count
                if (step+1) == stage1_steps_etainf:
                    switch_now = True
                    early_stop_reason = ""
                # Early switch if gradient flips (momentum overshoot detected)
                elif stage1_early_stop_on_flip and sign_flipped:
                    switch_now = True
                    early_stop_reason = "gradient sign flip"
                    stage1_early_stopped = True
            
            if switch_now:
                # Save stage 1 checkpoint (for plotting viscosity after etainf-only learning)
                print("")  # newline
                actual_stage1_steps = step + 1
                if early_stop_reason:
                    print(f"      STAGE 1 EARLY STOP at step {actual_stage1_steps} ({early_stop_reason})")
                    # Adjust stage 2 steps to use remaining budget
                    stage2_steps_curv = stage2_steps_curv + (stage1_steps_etainf - actual_stage1_steps)
                    print(f"       Allocating {stage2_steps_curv} steps to stage 2 (using saved budget)")
                else:
                    print(f"      STAGE 1 COMPLETE (after {actual_stage1_steps} steps)")
                
                try:
                    eta_inf_switch, eta0_switch, delta_switch, mu_switch = inspect_head(model_info['model'], current_params)
                    print(f"      Plateaus: etainf={eta_inf_switch:.6g}, eta0={eta0_switch:.6g}, Delta={delta_switch:.6g}")
                    print(f"      mu: {np.array2string(mu_switch, precision=4, separator=', ')}")
                except Exception as e:
                    print(f"      [inspect_head failed at switch]: {e}")
                
                # Save stage 1 model state for viscosity plotting
                stage1_model_info = model_info.copy()
                stage1_model_info['params'] = current_params
                stage1_model_info['flattened_params'] = current_params_flat
                
                # Compute stage 1 viscosity curve
                try:
                    stage1_viscosities = plotting.compute_tbnn_viscosities_vs_strain_rate(stage1_model_info)
                    print(f"      Stage 1 viscosity curve computed")
                except Exception as e:
                    print(f"      Failed to compute stage 1 viscosity: {e}")
                    stage1_viscosities = None
                
                # Switch to curvature-only mode
                tx, new_opt_state = _init_tx_for_stage('curvature_only')
                
                if stage1_reset_momentum:
                    opt_state = new_opt_state  # Reset Adam momentum
                    print(f"      Switched to STAGE 2 (curvature-only); Adam momentum RESET")
                else:
                    # Keep old momentum state (not recommended, but supported)
                    print(f"      Switched to STAGE 2 (curvature-only); momentum carried over")
                
                current_stage_mode = 'curvature_only'
                print(f"      Stage 2 steps: {stage2_steps_curv}")
            
            # Save checkpoint every N steps if requested
            if save_traj_info and checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
                import pickle
                checkpoint_step = step + 1
                checkpoint_dir = os.path.join(traj_data_dir, f"checkpoint_step_{checkpoint_step:03d}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                # Save flattened params
                np.save(os.path.join(checkpoint_dir, f"params_step_{checkpoint_step}.npy"), 
                        np.array(current_params_flat))
                # Save structured params
                with open(os.path.join(checkpoint_dir, f"params_step_{checkpoint_step}.pkl"), 'wb') as f:
                    pickle.dump(current_params, f)
                print(f" [checkpoint saved]", end="")
            
        except Exception as e:
            print(f"failed with error: {e}")
            print(f"       Training stopped at step {step+1}, rolling back to last good state")
            simulation_failed = True
            break
    
    # Handle simulation failure - rollback to last good state
    failed_viscosities = None
    failed_model_info = None
    if simulation_failed:
        print(f"\nROLLBACK ACTIVATED:")
        print(f"   Reverting to last good state from Step {last_good_step}")
        print(f"   Last good loss: {last_good_loss:.6e}")
        
        # Before rollback, compute the failed viscosity curve for analysis
        print(f"   Computing failed viscosity curve for analysis...")
        try:
            # Create a temporary model_info with the ACTUAL failed parameters from current_params
            failed_model_info = model_info.copy()  # Start with base structure
            failed_model_info['params'] = current_params  # Use the failed step's params
            failed_model_info['flattened_params'] = parameter_utils.flatten_params(current_params)[0]
            failed_viscosities = plotting.compute_tbnn_viscosities_vs_strain_rate(failed_model_info)
            
            # Analyze the failed viscosity
            min_failed_visc = float(jnp.min(failed_viscosities))
            max_failed_visc = float(jnp.max(failed_viscosities))
            mean_failed_visc = float(jnp.mean(failed_viscosities))
            
            print(f"   Failed viscosity range: [{min_failed_visc:.6f}, {max_failed_visc:.6f}]")
            print(f"   Failed viscosity mean: {mean_failed_visc:.6f}")
            
            # Diagnose the problem
            if max_failed_visc > 1e3:
                print(f"   DIAGNOSIS: Viscosity became too high (max: {max_failed_visc:.2e})")
            elif min_failed_visc < 1e-6:
                print(f"   DIAGNOSIS: Viscosity became too low (min: {min_failed_visc:.2e})")
            elif max_failed_visc / min_failed_visc > 1e6:
                print(f"   DIAGNOSIS: Viscosity range became extreme (ratio: {max_failed_visc/min_failed_visc:.2e})")
            else:
                print(f"   DIAGNOSIS: Viscosity appears reasonable, failure likely due to other factors")
                
        except Exception as e:
            print(f"   Could not compute failed viscosity curve: {e}")
            failed_viscosities = None
        
        # Restore last good parameters and model state
        current_params = last_good_params
        current_model_info = last_good_model_info
        
        # Truncate history to last good state
        if last_good_step < len(loss_history):
            loss_history = loss_history[:last_good_step + 1]
            gradient_magnitudes = gradient_magnitudes[:last_good_step + 1]
        
        print(f"   State restored to Step {last_good_step}")
        print(f"   training history truncated to {len(loss_history)-1} accepted steps")
    
    # Final training statistics (using potentially rolled-back state)
    final_loss = loss_history[-1]
    total_loss_reduction = loss_history[0] - final_loss
    relative_improvement = (total_loss_reduction / loss_history[0]) * 100 if loss_history[0] > 0 else 0
    
    if simulation_failed:
        print(f"\ntraining stopped after rollback; {len(loss_history)-1} accepted steps")
        print(f"Final state: Step {last_good_step} (rolled back from failed step)")
    else:
        if two_stage_etainf_then_curv:
            print(f"\ntwo-stage training finished after {len(loss_history)-1} steps")
            if stage1_early_stopped:
                print(f"   Stage 1 (etainf ON): ended early at step {actual_stage1_steps} (gradient sign flip)")
            else:
                print(f"   Stage 1 (etainf ON): {stage1_steps_etainf} steps")
            print(f"   Stage 2 (etainf OFF): {stage2_steps_curv} steps")
        else:
            print(f"\ntraining finished after {len(loss_history)-1} steps")
    
    print(f"Initial loss: {loss_history[0]:.6e}")
    print(f"Final loss: {final_loss:.6e}")
    print(f"Loss reduction: {total_loss_reduction:.6e} ({relative_improvement:.2f}%)")
    
    # Use flattened params for comparison (current_params is structured after Adam training)
    current_params_flat = parameter_utils.flatten_params(current_params)[0]
    print(f"Final parameter change: {jnp.linalg.norm(current_params_flat - model_info['flattened_params']):.6e}")
    
    # Plot training progress
    print(f"\n10. Plotting training progress...")
    stage1_end_val = stage1_steps_etainf if two_stage_etainf_then_curv else None
    training_stats = plotting.plot_training_progress(
        loss_history, gradient_magnitudes, learning_rate, total_steps,
        save_plots, output_dir, "training_progress_constriction",
        stage1_end=stage1_end_val
    )
    
    # Save training progress data if requested
    if save_traj_info:
        np.save(os.path.join(traj_data_dir, "loss_history.npy"), np.array(loss_history))
        np.save(os.path.join(traj_data_dir, "gradient_magnitudes.npy"), np.array(gradient_magnitudes))
        print(f"   Saved training progress data to {traj_data_dir}")
    
    # Use final model state (convert back to proper format)
    updated_model_info = last_good_model_info
    updated_params = updated_model_info['flattened_params']
    
    # --- Read final plateau values via introspection ---
    eta_inf_f, eta0_f, delta_f, mu_f = inspect_head(updated_model_info['model'], updated_model_info['params'])
    print(f"\n   FINAL: etainf={eta_inf_f:.6g}, eta0={eta0_f:.6g}, Delta={delta_f:.6g}")
    print(f"   FINAL mu (z=log gammadot/gamma_ref): {np.array2string(mu_f, precision=4, separator=', ')}")
    print(f"   Changes: Deltaetainf={(eta_inf_f-eta_inf0):+.3e}, Deltaeta0={(eta0_f-eta00):+.3e}, DeltaDelta={(delta_f-delta0):+.3e}")
    
    # Plot final viscosity behavior
    print(f"\n11. Plotting final TBNN viscosity behavior...")
    
    # Compute initial viscosity for comparison
    initial_viscosities = plotting.compute_tbnn_viscosities_vs_strain_rate(model_info)
    
    # Prepare additional viscosity curves
    additional_viscosities = [initial_viscosities]
    additional_labels = ['Initial TBNN']
    
    # Add stage 1 viscosity curve if 2-stage training
    if two_stage_etainf_then_curv and stage1_viscosities is not None:
        additional_viscosities.append(stage1_viscosities)
        stage1_label = f'After Stage 1 (step {stage1_steps_etainf}, etainf only)' if stage1_etainf_only else f'After Stage 1 (step {stage1_steps_etainf})'
        additional_labels.append(stage1_label)
        print(f"   Including stage 1 viscosity curve in plot")
    
    # Add failed viscosity curve if rollback occurred
    if simulation_failed and failed_viscosities is not None:
        additional_viscosities.append(failed_viscosities)
        additional_labels.append('Failed TBNN (before rollback)')
        print(f"   Including failed viscosity curve in plot")
    
    title_suffix = f"After {len(loss_history)-1} steps, Constriction"
    if two_stage_etainf_then_curv:
        title_suffix += f" (2-stage)"
    if simulation_failed:
        title_suffix += f" (rolled back from failure)"
    
    updated_visc_stats = plotting.plot_viscosity_strain_rate(
        updated_model_info, f"Final TBNN Viscosity vs Strain Rate ({title_suffix})",
        save_plots, output_dir, "final_viscosity_constriction",
        reference_model=reference_model, reference_params=reference_params,
        additional_viscosities=additional_viscosities, 
        additional_labels=additional_labels
    )
    
    # Save final viscosity data and model parameters if requested
    if save_traj_info:
        np.save(os.path.join(traj_data_dir, "final_strain_rates.npy"), 
                np.array(updated_visc_stats['strain_rates']))
        np.save(os.path.join(traj_data_dir, "final_viscosities.npy"), 
                np.array(updated_visc_stats['viscosities']))
        # Save final model parameters (both formats)
        import pickle
        # Flattened array (for inspection)
        np.save(os.path.join(traj_data_dir, "final_tbnn_params.npy"), 
                np.array(updated_model_info['flattened_params']))
        # Structured params (for forward simulations)
        with open(os.path.join(traj_data_dir, "final_tbnn_params.pkl"), 'wb') as f:
            pickle.dump(updated_model_info['params'], f)
        # Reconstruction info (tree_def + shapes)
        with open(os.path.join(traj_data_dir, "final_tbnn_tree_def.pkl"), 'wb') as f:
            pickle.dump(updated_model_info['tree_def'], f)
        np.save(os.path.join(traj_data_dir, "final_tbnn_shapes.npy"), 
                np.array(updated_model_info['shapes'], dtype=object))
        print(f"   Saved final viscosity data and model parameters to {traj_data_dir}")
        print(f"      (saved both .npy flat + .pkl structured + tree_def/shapes)")
        
        # Save stage 1 model if 2-stage training
        if two_stage_etainf_then_curv and stage1_model_info is not None:
            # Flattened array
            np.save(os.path.join(traj_data_dir, "stage1_tbnn_params.npy"), 
                    np.array(stage1_model_info['flattened_params']))
            # Structured params
            with open(os.path.join(traj_data_dir, "stage1_tbnn_params.pkl"), 'wb') as f:
                pickle.dump(stage1_model_info['params'], f)
            if stage1_viscosities is not None:
                np.save(os.path.join(traj_data_dir, "stage1_viscosities.npy"), 
                        np.array(stage1_viscosities))
            print(f"   Saved stage 1 model parameters to {traj_data_dir}")
            print(f"      (saved both .npy flat + .pkl structured + viscosities)")
        
        # Save failed model if rollback occurred
        if simulation_failed and failed_model_info is not None:
            # Flattened array
            np.save(os.path.join(traj_data_dir, "failed_tbnn_params.npy"), 
                    np.array(failed_model_info['flattened_params']))
            # Structured params
            with open(os.path.join(traj_data_dir, "failed_tbnn_params.pkl"), 'wb') as f:
                pickle.dump(failed_model_info['params'], f)
            if failed_viscosities is not None:
                np.save(os.path.join(traj_data_dir, "failed_viscosities.npy"), 
                        np.array(failed_viscosities))
            print(f"   Saved failed model parameters to {traj_data_dir}")
            print(f"      (saved both .npy flat + .pkl structured)")
    
    # Compare viscosity changes
    print(f"\nVISCOSITY CHANGE ANALYSIS:")
    print(f"   PLATEAU VALUES:")
    print(f"      Initial: etainf={eta_inf0:.6g}, eta0={eta00:.6g}, Delta={delta0:.6g}")
    print(f"      Final:   etainf={eta_inf_f:.6g}, eta0={eta0_f:.6g}, Delta={delta_f:.6g}")
    print(f"      Changes: Deltaetainf={(eta_inf_f-eta_inf0):+.3e}, Deltaeta0={(eta0_f-eta00):+.3e}, DeltaDelta={(delta_f-delta0):+.3e}")
    print(f"   mu CENTER POSITIONS (z=log gammadot/gamma_ref):")
    print(f"      Initial: {np.array2string(mu0, precision=4, separator=', ')}")
    print(f"      Final:   {np.array2string(mu_f, precision=4, separator=', ')}")
    mu_change = np.linalg.norm(mu_f - mu0)
    print(f"      Change magnitude: {mu_change:.3e} {'(FROZEN )' if mu_change < 1e-10 else '(MOVED!)'}")
    print(f"   STRAIN RATE VISCOSITIES:")
    print(f"   Initial low strain rate viscosity: {initial_visc_stats['low_strain_viscosity']:.6f}")
    print(f"   Updated low strain rate viscosity: {updated_visc_stats['low_strain_viscosity']:.6f}")
    print(f"   Change: {updated_visc_stats['low_strain_viscosity'] - initial_visc_stats['low_strain_viscosity']:.6e}")
    print(f"   Initial high strain rate viscosity: {initial_visc_stats['high_strain_viscosity']:.6f}")
    print(f"   Updated high strain rate viscosity: {updated_visc_stats['high_strain_viscosity']:.6f}")
    print(f"   Change: {updated_visc_stats['high_strain_viscosity'] - initial_visc_stats['high_strain_viscosity']:.6e}")
    
    # Run forward simulation with updated TBNN if requested
    updated_tbnn_result = None
    updated_stats = None
    
    if run_new_forward:
        print(f"\n12. Running forward simulation with updated TBNN...")
        try:
            updated_loss, updated_tbnn_result, updated_trajectory = compute_tbnn_trajectory_loss_constriction(
                updated_params, flow_cond, updated_model_info, particles, reference_trajectory,
                visc_loss=visc_loss, visc_loss_config=visc_loss_config,
                shape_loss=shape_loss, shape_weight=shape_weight,
                mask_layout=mask_layout,
                resolution_piv=resolution_piv, piv_W_win=piv_W_win, piv_overlap=piv_overlap, piv_kernel=piv_kernel,
                add_piv_noise=add_piv_noise, piv_noise_p_percent=piv_noise_p_percent,
                piv_noise_corr_frac=piv_noise_corr_frac, piv_noise_beta_grad=piv_noise_beta_grad,
                piv_noise_use_bias=piv_noise_use_bias, piv_noise_seed=piv_noise_seed
            )
            
            if updated_tbnn_result is not None:
                print("Updated TBNN simulation completed")
                print(f"Updated loss: {float(updated_loss):.6e}")
                print(f"Loss change: {float(updated_loss - initial_loss):.6e}")
                
                # Plot updated comparison
                print("\n13. Plotting updated model comparison...")
                updated_stats = plotting.plot_model_comparison_constriction(
                    updated_tbnn_result, ref_result, domain,
                    f"Updated TBNN vs {reference_model.title()} Reference (After Gradient Step, Constriction)",
                    save_plots, output_dir, "updated_comparison_constriction"
                )
                
                # Plot three-way flow field comparison
                print("\n14. Plotting three-way flow field comparison...")
                three_way_stats = plotting.plot_three_way_flow_comparison_constriction(
                    initial_tbnn_result, updated_tbnn_result, ref_result, domain,
                    f"Flow Field Evolution: Initial -> Final TBNN vs {reference_model.title()} Ground Truth (Constriction)",
                    save_plots, output_dir, "three_way_flow_comparison_constriction"
                )
                
                print(f"\nPERFORMANCE CHANGE ANALYSIS:")
                print(f"   Initial relative error: {initial_stats['relative_error_percent']:.2f}%")
                print(f"   Updated relative error: {updated_stats['relative_error_percent']:.2f}%")
                error_change = updated_stats['relative_error_percent'] - initial_stats['relative_error_percent']
                print(f"   Error change: {error_change:.2f}% ({'improvement' if error_change < 0 else 'degradation'})")
                print(f"   Flow field error reduction: {three_way_stats['relative_improvement']:.2f}%")
                
                # Save updated TBNN results if requested
                if save_traj_info:
                    np.save(os.path.join(traj_data_dir, "updated_tbnn_velocity_x.npy"), 
                            np.array(updated_tbnn_result.velocity[0].data))
                    np.save(os.path.join(traj_data_dir, "updated_tbnn_velocity_y.npy"), 
                            np.array(updated_tbnn_result.velocity[1].data))
                    # Save updated TBNN trajectory
                    if updated_trajectory is not None:
                        if hasattr(updated_trajectory[0], 'data'):
                            np.save(os.path.join(traj_data_dir, "updated_tbnn_trajectory_x.npy"), 
                                    np.array(updated_trajectory[0].data))
                            np.save(os.path.join(traj_data_dir, "updated_tbnn_trajectory_y.npy"), 
                                    np.array(updated_trajectory[1].data))
                        else:
                            np.save(os.path.join(traj_data_dir, "updated_tbnn_trajectory_x.npy"), 
                                    np.array(updated_trajectory[0]))
                            np.save(os.path.join(traj_data_dir, "updated_tbnn_trajectory_y.npy"), 
                                    np.array(updated_trajectory[1]))
                    print(f"   Saved updated TBNN simulation data to {traj_data_dir}")
                
            else:
                print("Updated TBNN simulation failed")
                
        except Exception as e:
            print(f"Updated TBNN simulation failed: {e}")
    else:
        print(f"\n12. Skipping new forward simulation (run_new_forward=False)")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"ONE STEP GRADIENT UPDATE WITH CONSTRICTION SUMMARY")
    print(f"{'='*60}")
    print(f"Reference model: {reference_model}")
    print(f"Reference params: {reference_params}")
    print(f"TBNN architecture: {tbnn_hidden_units}")
    print(f"Loss: Full-field library RMSE")
    print(f"Viscosity regularization: {visc_loss}")
    print(f"Velocity shape loss: {shape_loss}")
    print(f"Initial loss: {float(initial_loss):.6e}")
    print(f"Gradient magnitude: {grad_magnitude:.6e}")
    print(f"Learning rate: {learning_rate}")
    
    if run_new_forward and updated_tbnn_result is not None:
        print(f"Updated loss: {float(updated_loss):.6e}")
        print(f"Loss improvement: {float(initial_loss - updated_loss):.6e}")
        if updated_stats:
            print(f"Error change: {updated_stats['relative_error_percent'] - initial_stats['relative_error_percent']:.2f}%")
    
    # Return comprehensive results
    results = {
        'test_type': 'one_step_gradient_constriction',
        'reference_model': reference_model,
        'reference_params': reference_params,
        'tbnn_architecture': tbnn_hidden_units,
        'eta_init': eta_init,
        'visc_loss': visc_loss,
        'visc_loss_config': visc_loss_config if visc_loss else None,
        'shape_loss': shape_loss,
        'shape_weight': shape_weight if shape_loss else None,
        'mask_layout': mask_layout,
        'initial_loss': float(initial_loss),
        'gradient': gradient,
        'gradient_magnitude': float(grad_magnitude),
        'learning_rate': learning_rate,
        'initial_model_info': model_info,
        'updated_model_info': updated_model_info,
        'initial_tbnn_result': initial_tbnn_result,
        'reference_result': ref_result,
        'initial_comparison_stats': initial_stats,
        'strain_rate_stats': strain_rate_stats,
        'initial_viscosity_stats': initial_visc_stats,
        'updated_viscosity_stats': updated_visc_stats,
        'training_stats': training_stats,
        'simulation_failed': simulation_failed,
        'last_good_step': last_good_step if simulation_failed else len(loss_history)-1,
        'rollback_activated': simulation_failed,
        'failed_viscosities': failed_viscosities if simulation_failed else None,
        'failed_model_info': failed_model_info if simulation_failed else None,
        'particles': particles,
        'flow_cond': flow_cond,
        'config': {
            'domain': domain,
            'domain_size': domain_size,
            'pressure_gradient': pressure_gradient,
            'dt': dt,
            'inner_steps': inner_steps,
            'outer_steps': outer_steps,
            'solver_type': solver_type,
            'stepper_type': stepper_type,
            'learning_rate': learning_rate,
            'num_update_steps': num_update_steps,
            'random_seed': random_seed
        }
    }
    
    if run_new_forward:
        results.update({
            'updated_loss': float(updated_loss) if updated_tbnn_result else None,
            'updated_tbnn_result': updated_tbnn_result,
            'updated_comparison_stats': updated_stats,
            'three_way_flow_stats': three_way_stats if 'three_way_stats' in locals() else None,
            'loss_improvement': float(initial_loss - updated_loss) if updated_tbnn_result else None
        })
    
    # Create summary of saved files if trajectory info was saved
    if save_traj_info:
        summary_path = os.path.join(traj_data_dir, "saved_files_manifest.txt")
        with open(summary_path, 'w') as f:
            f.write("TRAJECTORY DATA FILES MANIFEST\n")
            f.write("="*60 + "\n\n")
            f.write("This directory contains all trajectory data and model states saved as .npy arrays.\n\n")
            
            f.write("VISCOSITY DATA:\n")
            f.write("  - initial_strain_rates.npy: Strain rate grid for viscosity evaluation\n")
            f.write("  - initial_viscosities.npy: Initial TBNN viscosity curve\n")
            f.write("  - final_strain_rates.npy: Strain rate grid (same as initial)\n")
            f.write("  - final_viscosities.npy: Final TBNN viscosity curve after training\n")
            if simulation_failed and failed_viscosities is not None:
                f.write("  - failed_viscosities.npy: Failed TBNN viscosity curve (before rollback)\n")
            f.write("\n")
            
            f.write("REFERENCE TRAJECTORY DATA:\n")
            f.write("  - reference_trajectory_x.npy: Reference X-velocity trajectory (time, H, W)\n")
            f.write("  - reference_trajectory_y.npy: Reference Y-velocity trajectory (time, H, W)\n")
            f.write("  - reference_velocity_x.npy: Reference final X-velocity field (H, W)\n")
            f.write("  - reference_velocity_y.npy: Reference final Y-velocity field (H, W)\n")
            f.write("\n")
            
            f.write("INITIAL TBNN DATA:\n")
            f.write("  - initial_tbnn_params.npy: Initial TBNN model parameters (flattened 1D array)\n")
            f.write("  - initial_tbnn_params.pkl: Initial TBNN parameters (structured, for forward sims)\n")
            f.write("  - initial_tbnn_tree_def.pkl: Tree structure for unflattening .npy params\n")
            f.write("  - initial_tbnn_shapes.npy: Parameter shapes for reconstruction\n")
            f.write("  - initial_tbnn_velocity_x.npy: Initial TBNN final X-velocity field (H, W)\n")
            f.write("  - initial_tbnn_velocity_y.npy: Initial TBNN final Y-velocity field (H, W)\n")
            f.write("  - initial_tbnn_trajectory_x.npy: Initial TBNN X-velocity trajectory (time, H, W)\n")
            f.write("  - initial_tbnn_trajectory_y.npy: Initial TBNN Y-velocity trajectory (time, H, W)\n")
            f.write("\n")
            
            f.write("TRAINING DATA:\n")
            f.write("  - loss_history.npy: Loss values at each training step\n")
            f.write("  - gradient_magnitudes.npy: Gradient magnitudes at each step\n")
            if checkpoint_every > 0:
                f.write(f"  - checkpoint_step_NNN/: TBNN params saved every {checkpoint_every} steps\n")
                f.write(f"    * params_step_N.npy: Flattened params at step N\n")
                f.write(f"    * params_step_N.pkl: Structured params at step N (ready for forward sim)\n")
            f.write("\n")
            
            f.write("FINAL TBNN DATA:\n")
            f.write("  - final_tbnn_params.npy: Final TBNN parameters after training (flattened 1D array)\n")
            f.write("  - final_tbnn_params.pkl: Final TBNN parameters (structured, for forward sims)\n")
            f.write("  - final_tbnn_tree_def.pkl: Tree structure for unflattening .npy params\n")
            f.write("  - final_tbnn_shapes.npy: Parameter shapes for reconstruction\n")
            if two_stage_etainf_then_curv and stage1_model_info is not None:
                f.write("  - stage1_tbnn_params.npy: Stage 1 TBNN parameters (after etainf-only learning, flattened)\n")
                f.write("  - stage1_tbnn_params.pkl: Stage 1 TBNN parameters (structured, for forward sims)\n")
                f.write("  - stage1_viscosities.npy: Stage 1 viscosity curve (eta vs strain rate)\n")
            if simulation_failed:
                f.write("  - failed_tbnn_params.npy: Failed TBNN parameters (flattened, before rollback)\n")
                f.write("  - failed_tbnn_params.pkl: Failed TBNN parameters (structured, before rollback)\n")
            if run_new_forward and updated_tbnn_result is not None:
                f.write("  - updated_tbnn_velocity_x.npy: Final TBNN X-velocity field (H, W)\n")
                f.write("  - updated_tbnn_velocity_y.npy: Final TBNN Y-velocity field (H, W)\n")
                f.write("  - updated_tbnn_trajectory_x.npy: Final TBNN X-velocity trajectory (time, H, W)\n")
                f.write("  - updated_tbnn_trajectory_y.npy: Final TBNN Y-velocity trajectory (time, H, W)\n")
            f.write("\n")
            
            f.write("NOTES:\n")
            f.write("  - All .npy files can be loaded with: numpy.load('filename.npy')\n")
            f.write("  - All .pkl files can be loaded with: pickle.load(open('filename.pkl', 'rb'))\n")
            f.write("  - Velocity field shapes: (H, W) where H=height, W=width\n")
            f.write("  - Trajectory shapes: (time_steps, H, W)\n")
            f.write("  - Parameter formats:\n")
            f.write("    * .npy files: Flattened 1D arrays (for inspection, plotting)\n")
            f.write("    * .pkl files: Structured FrozenDict (ready for model.apply() or forward sims)\n")
            f.write("    * tree_def.pkl + shapes.npy: Needed to reconstruct structured params from .npy\n")
            f.write("  - To use .pkl params for forward simulation:\n")
            f.write("      import pickle\n")
            f.write("      with open('final_tbnn_params.pkl', 'rb') as f:\n")
            f.write("          params = pickle.load(f)\n")
            f.write("      # Now use: tbnn_model.apply(params, gamma_dot, invariants)\n")
            f.write("  - To reconstruct from .npy (if needed):\n")
            f.write("      flat_params = np.load('final_tbnn_params.npy')\n")
            f.write("      with open('final_tbnn_tree_def.pkl', 'rb') as f:\n")
            f.write("          tree_def = pickle.load(f)\n")
            f.write("      shapes = np.load('final_tbnn_shapes.npy', allow_pickle=True)\n")
            f.write("      params = unflatten_params(flat_params, tree_def, shapes)\n")
            if simulation_failed:
                f.write(f"  - Training failed at step {last_good_step+1}, rolled back to step {last_good_step}\n")
        
        print(f"\nSaved trajectory data manifest to: {summary_path}")
    
    print("one-step gradient update on the constriction: PASS")
    return results


# =============================================================================
# MAIN EXECUTION AND DEMO
# =============================================================================

def demo_gradient_debugging_constriction():
    """Run demonstration of gradient debugging test with constriction geometry using NEW model.
    
    Uses the mixture-of-sigmoids viscosity model for improved stability.
    """
    print("TBNN GRADIENT DEBUGGING DEMONSTRATION WITH CONSTRICTION (NEW MODEL)")
    print("Using mixture-of-sigmoids viscosity model")
    print("Running gradient update test with constriction geometry")
    
    # Demo: One step gradient update
    print("\n" + "="*80)
    print("RUNNING DEMO: GRADIENT UPDATE WITH CONSTRICTION")
    print("="*80)
    
    test_results = debug_one_step_gradient_constriction(
        reference_model='carreau_yasuda',
        reference_params=[0.02, 1.0, 5.0, 0.5, 2.0],
        tbnn_hidden_units=[20, 20],
        eta_init=1.0,
        pressure_gradient=2.5,
        learning_rate=1e-3,
        run_new_forward=True,
        outer_steps=50  # Fewer steps for demo
    )
    
    # Summary
    print("\n" + "="*80)
    print("GRADIENT DEBUGGING DEMO WITH CONSTRICTION COMPLETED")
    print("="*80)
    
    if test_results:
        print(f"Gradient Update with Constriction: SUCCESS")
        print(f"   Initial loss: {test_results['initial_loss']:.6e}")
        print(f"   Gradient magnitude: {test_results['gradient_magnitude']:.6e}")
        if 'updated_loss' in test_results and test_results['updated_loss']:
            print(f"   Loss improvement: {test_results['loss_improvement']:.6e}")
    else:
        print(f"Gradient Update with Constriction: FAILED")
    
    return test_results

if __name__ == "__main__":
    # Run demonstration
    results = demo_gradient_debugging_constriction()

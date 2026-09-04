"""
TBNN viscosity fit: SHAPE-ONLY LEARNING (frozen etainf and eta0)
Learn only the curvature shape (mu, s, alpha) by freezing both plateaus
"""

import os, sys, math
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import matplotlib.pyplot as plt
from flax.core import freeze, unfreeze
from flax.core.frozen_dict import FrozenDict
from flax import traverse_util

from jax_rheology.models import build_tbnn_bounded_model, init_tbnn_soft_newtonian

# Try Optax
try:
    import optax
    HAVE_OPTAX = True
except Exception:
    HAVE_OPTAX = False


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


def fit_tbnn_to_viscosity(
    # Dataset parameters
    seed=1453,
    N_points=120,
    gmin=1e-2,
    gmax=1e+2,
    gamma_ref=1.0,
    
    # Model architecture
    hidden_units=[24, 24],
    M=4,
    eta_min=1e-2,
    eta_max=10.0,
    
    # MODEL PARAMETERS (no gating, no PL)
    s_floor=0.35,           # Smooth bumps
    alpha_temp=0.8,         # Moderate diffusion
    freeze_centers=True,    # FREEZE mu centers (non-trainable)
    
    # HARD-FREEZE eta0 (learn etainf via gap) - new approach
    freeze_eta0=True,       # HARD-FREEZE eta0 (learn etainf via gap)
    eta0_fixed=None,        # Fixed eta0 value (None = use target eta0_cy)
    
    # FREEZE BOTH PLATEAUS (etainf and eta0) - shape-only learning (old optimizer-based approach)
    freeze_both_plateaus=False,  # FREEZE etainf AND eta0 via optimizer (deprecated, use freeze_eta0 instead)
    
    # MANUAL mu PLACEMENT AND GATING
    mu_min_manual=None,         # start mu here (gammadot); if None, use gmin
    mu_max_manual=None,         # end mu here (gammadot); if None, use gmax
    gate_gamma_manual=None,     # gate turn-on (gammadot); None disables gate
    gate_width_z=0.6,           # smoothness in z = log(gammadot/gamma_ref)
    tail_gate_gamma_manual=None,# optional: delay tail leverage; None disables
    
    # Training
    learning_rate=1e-1,
    num_steps=5,
    print_every=1,
    use_log_loss=True,
    
    # DERIVATIVE REGULARIZATION (light)
    enable_deriv_regularization=True,
    reg_gamma_min=0.05,
    reg_gamma_max=5.0,
    reg_num_points=128,
    reg_s_cap=0.50,
    reg_lambda_slope=1e-3,
    reg_lambda_curv=3e-4,
    reg_curv_cap_abs=0.8,
    reg_p_slope=2.0,
    reg_p_curv=2.0,
    
    # Initialization
    init_mode='soft_newtonian',
    saved_params=None,
    
    # Target selection
    target_type="carreau_yasuda",
    eta0_cy=1.0,
    etainf_cy=0.02,
    lam_cy=5.0,
    a_cy=2.0,
    n_cy=0.5,
    
    # Plotting
    show_plots=True
):
    """
    Fit a TBNN model to viscosity data with optional plateau freezing.
    
    Parameters
    ----------
    seed : int
        Random seed for initialization
    N_points : int
        Number of data points
    gmin, gmax : float
        Shear rate range [1/s]
    gamma_ref : float
        Reference shear rate for normalization
    hidden_units : list
        Hidden layer sizes
    M : int
        Number of modes in TBNN
    eta_min, eta_max : float
        Viscosity bounds [Pa.s]
    s_floor : float
        Minimum mode width parameter
    alpha_temp : float
        Temperature/diffusion parameter
    freeze_centers : bool
        If True, freeze mu centers (non-trainable, stationary)
    freeze_eta0 : bool
        If True, hard-freeze eta0 and learn etainf via gap parameter (log-add only)
    eta0_fixed : float or None
        Fixed eta0 value when freeze_eta0=True (None = use target eta0_cy)
    freeze_both_plateaus : bool
        If True, freeze etainf and eta0 via optimizer (deprecated, use freeze_eta0)
    mu_min_manual : float or None
        Start mu centers here (gammadot [1/s]); if None, use gmin
    mu_max_manual : float or None
        End mu centers here (gammadot [1/s]); if None, use gmax
    gate_gamma_manual : float or None
        Gate turn-on location (gammadot [1/s]); None disables gate
    gate_width_z : float
        Gate smoothness in z = log(gammadot/gamma_ref) (default: 0.6)
    tail_gate_gamma_manual : float or None
        Delay tail leverage to this gammadot [1/s]; None disables (default: None)
    learning_rate : float
        Adam learning rate
    num_steps : int
        Training iterations
    print_every : int
        Print frequency
    use_log_loss : bool
        Use log-space loss
    enable_deriv_regularization : bool
        Enable derivative penalty
    reg_* : float
        Regularization parameters
    init_mode : str
        'soft_newtonian', 'from_saved', or 'random'
    saved_params : dict
        Pre-trained parameters (if init_mode='from_saved')
    target_type : str
        Target model type
    eta0_cy, etainf_cy, lam_cy, a_cy, n_cy : float
        Carreau-Yasuda parameters
    show_plots : bool
        Whether to display plots
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'trained_params': trained model parameters
        - 'loss_hist': training loss history
        - 'data_loss_hist': data loss history
        - 'reg_loss_hist': regularization loss history
        - 'eta_pred': final viscosity prediction
        - 'eta_target': target viscosity
        - 'gamma_vec': shear rate array
        - 'deriv_final': final derivatives (d1, d2, z)
        - 'checkpoint_params': intermediate checkpoints
        - 'fig': matplotlib figure (if show_plots=True)
    """
    
    # ========================= BUILD DATASET ======================================
    W = N_points
    H = 1
    gamma_vec = jnp.geomspace(gmin, gmax, W)
    
    def eta_carreau_yasuda(g):
        return etainf_cy + (eta0_cy - etainf_cy) * jnp.power(
            1.0 + jnp.power(lam_cy * g, a_cy), (n_cy - 1.0) / a_cy
        )
    
    eta_target_vec = eta_carreau_yasuda(gamma_vec)
    
    gamma_img = gamma_vec.reshape(1, -1)
    C = 2
    I1_vec = (gamma_vec**2) / 2.0
    I2_vec = -(gamma_vec**2) / 2.0
    inv_aux = jnp.stack([I1_vec, I2_vec], axis=0).reshape(C, H, W)
    
    if gamma_ref is None:
        gamma_ref = float(jnp.exp(jnp.mean(jnp.log(gamma_vec + 1e-30))))
    
    # ========================= BUILD MODEL ========================================
    # Set eta0_fixed to target if not explicitly provided
    eta0_val = eta0_fixed if eta0_fixed is not None else eta0_cy
    
    # --- Manual-only wiring (no "calculation mumbo jumbo") ---
    mu_min_eff = float(mu_min_manual) if (mu_min_manual is not None) else float(gmin)
    mu_max_eff = float(mu_max_manual) if (mu_max_manual is not None) else float(gmax)
    gate_gamma_eff = (None if gate_gamma_manual is None else float(gate_gamma_manual))
    tail_gate_gamma_eff = (None if tail_gate_gamma_manual is None else float(tail_gate_gamma_manual))
    
    # Optional: quick sanity prints
    print(f"mu range: [{mu_min_eff:.3e}, {mu_max_eff:.3e}] (manual)")
    if gate_gamma_eff is None:
        print("Gate: OFF (manual)")
    else:
        print(f"Gate: ON @ gammadot = {gate_gamma_eff:.3e} with width_z = {gate_width_z}")
    if tail_gate_gamma_eff is not None:
        print(f"Tail gate: ON @ gammadot = {tail_gate_gamma_eff:.3e}")
    
    tbnn = build_tbnn_bounded_model(
        hidden_units=hidden_units, M=M,
        eta_min=eta_min, eta_max=eta_max,
        gamma_ref=gamma_ref,
        s_floor=s_floor,
        alpha_temp=alpha_temp,
        
        # Locks / plateaus
        freeze_centers=freeze_centers,
        freeze_eta0=freeze_eta0,
        eta0_fixed=eta0_val,
        
        # Manual mu placement
        mu_min_gamma=mu_min_eff,
        mu_max_gamma=mu_max_eff,
        
        # Manual gating
        gate_gamma=gate_gamma_eff,
        gate_width_z=gate_width_z,
        tail_gate_gamma=tail_gate_gamma_eff,
        
        enable_pl_per_mode=False,
        log_head=True, log_mixing="add"
    )
    
    print(f"Model configuration:")
    print(f"   M={M}, hidden={hidden_units}")
    print(f"   s_floor={s_floor}, alpha_temp={alpha_temp}")
    print(f"   LOG-HEAD: mode='add'")
    if freeze_centers:
        print(f"   CENTERS FROZEN (mu stationary, non-trainable)")
    if freeze_eta0:
        print(f"   eta0 HARD-FROZEN at {eta0_val:.4f} (etainf learnable via gap)")
    if freeze_both_plateaus:
        print(f"   BOTH PLATEAUS FROZEN via optimizer (deprecated, prefer freeze_eta0)")
    
    key = random.PRNGKey(seed)
    
    # Initialize
    if init_mode == 'soft_newtonian':
        print(f"Using soft Newtonian initialization (eta0 ~= {eta0_cy})")
        params = init_tbnn_soft_newtonian(tbnn, key, H, W, eta0_cy,
                                           A_frac=0.05, k_frac=0.2, pair_modes=(0, 1))
        print("TBNN initialized")
    
    elif init_mode == 'from_saved':
        if saved_params is None:
            raise ValueError("init_mode='from_saved' but saved_params is None!")
        params = saved_params
        print(f" Loaded parameters from saved_params variable")
    
    elif init_mode == 'random':
        print(" Using random initialization")
        params = tbnn.init(key, gamma_img, inv_aux)
    
    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")
    
    # --- Read initial plateau values via introspection (works for all heads) ---
    eta_inf0, eta00, delta0, mu0 = inspect_head(tbnn, params)
    print(f"INIT: etainf={eta_inf0:.6g}, eta0={eta00:.6g}, Delta={delta0:.6g}")
    print("INIT mu (z=log gammadot/gamma_ref):", np.array2string(mu0, precision=4, separator=", "))
    
    # Peek at constants created during initialization (if freeze_centers=True)
    if freeze_centers:
        _, consts = tbnn.apply(params, gamma_img, inv_aux, mutable=['constants'])
        print(f"   mu centers (z=log gammadot/gamma_ref): {np.array(consts['constants']['mu_centers'])}")
    
    # ========================= DERIVATIVE REGULARIZATION ==========================
    def evaluate_eta_on_grid(params, gamma_grid):
        """Evaluate eta(gammadot) on a 1D grid of shear rates."""
        gamma_grid = jnp.asarray(gamma_grid)
        N = gamma_grid.shape[0]
        gamma_field = gamma_grid.reshape(1, N)
        
        I1 = 0.5 * gamma_grid**2
        I2 = -0.5 * gamma_grid**2
        invariants_aux = jnp.stack([I1, I2], axis=0).reshape(2, 1, N)
        
        eta_field = tbnn.apply(params, gamma_field, invariants_aux)
        return eta_field.reshape(N)
    
    def compute_slope_curv_penalty(params):
        """Compute log-log derivative regularization penalty."""
        z_min = jnp.log(reg_gamma_min)
        z_max = jnp.log(reg_gamma_max)
        z = jnp.linspace(z_min, z_max, reg_num_points)
        gamma_grid = jnp.exp(z)
        
        eta = evaluate_eta_on_grid(params, gamma_grid)
        ell = jnp.log(jnp.maximum(eta, 1e-30))
        
        d1 = (ell[2:] - ell[:-2]) / (z[2:] - z[:-2])
        
        dz = jnp.diff(z)
        d2 = 2.0 * (
            (ell[2:]   - ell[1:-1]) / dz[1:] -
            (ell[1:-1] - ell[:-2])  / dz[:-1]
        ) / (z[2:] - z[:-2])
        
        slope_over = jnp.maximum(0.0, jnp.abs(d1) - reg_s_cap)
        pen_slope = reg_lambda_slope * jnp.sum(slope_over ** reg_p_slope)
        
        curv_over = jnp.maximum(0.0, jnp.abs(d2) - reg_curv_cap_abs)
        pen_curv = reg_lambda_curv * jnp.sum(curv_over ** reg_p_curv)
        
        return pen_slope + pen_curv, (d1, d2, z[1:-1])
    
    # ========================= LOSS & OPTIMIZER ===================================
    def predict_eta(params, gamma_img, inv_aux):
        return tbnn.apply(params, gamma_img, inv_aux)
    
    def loss_fn(params):
        # NaN-safe loss with clipping
        pred = predict_eta(params, gamma_img, inv_aux).reshape(-1)
        pred = jnp.clip(pred, 1e-12, 1e12)  # clip before log
        tgt = eta_target_vec
        if use_log_loss:
            data_loss = jnp.mean((jnp.log(pred) - jnp.log(jnp.clip(tgt, 1e-30, 1e30)))**2)
        else:
            data_loss = jnp.mean((pred - tgt)**2)
        
        if enable_deriv_regularization:
            reg_penalty, _ = compute_slope_curv_penalty(params)
            total_loss = data_loss + reg_penalty
        else:
            total_loss = data_loss
        
        # NAN GUARD: replace non-finite with a large finite scalar (no NaN backprop)
        total_loss = jnp.where(jnp.isfinite(total_loss), total_loss, 1e6)
        return total_loss
    
    def loss_fn_detailed(params):
        """Return (total_loss, data_loss, reg_penalty) for logging."""
        pred = predict_eta(params, gamma_img, inv_aux).reshape(-1)
        tgt = eta_target_vec
        if use_log_loss:
            res = jnp.log(jnp.clip(pred, 1e-30, 1e30)) - jnp.log(jnp.clip(tgt, 1e-30, 1e30))
        else:
            res = pred - tgt
        data_loss = jnp.mean(res**2)
        
        if enable_deriv_regularization:
            reg_penalty, _ = compute_slope_curv_penalty(params)
            return data_loss + reg_penalty, data_loss, reg_penalty
        else:
            return data_loss, data_loss, 0.0
    
    # FREEZE BOTH PLATEAUS via optimizer
    def make_label_tree(params_pytree):
        """Label leaves to freeze etainf and delta, learn only mu/s/alpha."""
        flat = traverse_util.flatten_dict(unfreeze(params_pytree), keep_empty_nodes=True)
        labeled = {}
        frozen_names = []
        learnable_names = []
        for k in flat.keys():
            name = "/".join(k)
            if freeze_both_plateaus:
                if   name.endswith(("eta_inf_raw","log_eta_inf_raw")): 
                    labeled[k] = "eta_inf"
                    frozen_names.append(name)
                elif name.endswith(("delta_raw","r_raw","log_range_raw","eta_partition_logit")):
                    labeled[k] = "delta_fix"
                    frozen_names.append(name)
                elif name.endswith("pl_slope_raw"):
                    labeled[k] = "pl_slopes"
                    learnable_names.append(name)
                else:
                    labeled[k] = "others"
                    learnable_names.append(name)
            else:
                # Normal mode: everything learns (including eta_partition_logit when freeze_eta0=True)
                if name.endswith(("eta_inf_raw","delta_raw","log_eta_inf_raw","r_raw","log_range_raw","eta_partition_logit")):
                    labeled[k] = "global"
                    learnable_names.append(name)
                elif name.endswith("pl_slope_raw"):
                    labeled[k] = "pl_slopes"
                    learnable_names.append(name)
                else:
                    labeled[k] = "others"
                    learnable_names.append(name)
        
        if frozen_names:
            print(f"FROZEN params (LR=0): {', '.join(frozen_names)}")
        if learnable_names[:3]:  # Print first few
            print(f"Learnable params: {', '.join(learnable_names[:3])}...")
        
        labels = traverse_util.unflatten_dict(labeled)
        labels = freeze(labels) if isinstance(params_pytree, FrozenDict) else labels
        return labels
    
    if HAVE_OPTAX:
        if freeze_both_plateaus:
            # Freeze etainf and delta - only learn shape
            base_tx = optax.multi_transform(
                {
                    "others":    optax.adam(learning_rate),         # mu, s, alpha learn
                    "pl_slopes": optax.adam(learning_rate),
                    "eta_inf":   optax.chain(optax.scale(0.0)),     # hard-freeze etainf
                    "delta_fix": optax.chain(optax.scale(0.0)),     # hard-freeze delta
                },
                param_labels=make_label_tree(params)
            )
        else:
            # Normal mode
            base_tx = optax.multi_transform(
                {
                    "others": optax.adam(learning_rate),
                    "global": optax.adam(learning_rate),
                    "pl_slopes": optax.adam(learning_rate),
                },
                param_labels=make_label_tree(params)
            )
        # Add gradient clipping for stability
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),  # tame spikes
            base_tx
        )
        train_state = tx.init(params)
        
        @jit
        def step(params, state):
            loss, grads = value_and_grad(loss_fn)(params)
            updates, state = tx.update(grads, state, params)
            params = optax.apply_updates(params, updates)
            return params, state, loss
    else:
        # Fallback Adam
        b1, b2, eps = 0.9, 0.999, 1e-8
        train_state = {
            'm': jax.tree_util.tree_map(jnp.zeros_like, params),
            'v': jax.tree_util.tree_map(jnp.zeros_like, params),
            't': 0
        }
        
        def _freeze_tail_grads(grads):
            """Zero out gradients for etainf and delta if freezing both plateaus."""
            if not freeze_both_plateaus:
                return grads
            flat = traverse_util.flatten_dict(unfreeze(grads), keep_empty_nodes=True)
            for k in list(flat.keys()):
                name = "/".join(k)
                if name.endswith(("eta_inf_raw","delta_raw","log_eta_inf_raw","r_raw","log_range_raw","eta_partition_logit")):
                    flat[k] = jnp.zeros_like(flat[k])  # Zero gradient
            return freeze(traverse_util.unflatten_dict(flat))
        
        def _adam_update(p, g, m, v, t):
            m = jax.tree_util.tree_map(lambda m_, g_: b1*m_ + (1-b1)*g_, m, g)
            v = jax.tree_util.tree_map(lambda v_, g_: b2*v_ + (1-b2)*(g_**2), v, g)
            mhat = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1**t), m)
            vhat = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2**t), v)
            p = jax.tree_util.tree_map(lambda p_, m_, v_: p_ - learning_rate * (m_ / (jnp.sqrt(v_) + eps)), p, mhat, vhat)
            return p, m, v
        
        def step(params, state):
            loss, grads = value_and_grad(loss_fn)(params)
            grads = _freeze_tail_grads(grads)
            state['t'] += 1
            params, state['m'], state['v'] = _adam_update(params, grads, state['m'], state['v'], state['t'])
            return params, state, loss
    
    # ========================= TRAINING ===========================================
    print(f"\n{'='*70}")
    print(f"Training TBNN on {target_type} target - SHAPE-ONLY MODE")
    print(f"  Points: {W}, range: [{gmin:.1e}, {gmax:.1e}] 1/s")
    print(f"  Steps: {num_steps}, LR: {learning_rate}")
    if freeze_both_plateaus:
        print(f"  BOTH PLATEAUS FROZEN -> learning SHAPE (mu,s,alpha) only")
    if enable_deriv_regularization:
        print(f"  Derivative regularization: s_cap={reg_s_cap}, curv_cap={reg_curv_cap_abs}")
    print(f"{'='*70}\n")
    
    # Save initial prediction
    with jax.disable_jit():
        eta_pred_initial = predict_eta(params, gamma_img, inv_aux).reshape(-1)
        _, deriv_initial = compute_slope_curv_penalty(params)
    
    checkpoint_steps = [num_steps // 4, num_steps // 2, 3 * num_steps // 4]
    checkpoint_params = {}
    checkpoint_derivs = {}
    
    loss_hist = []
    data_loss_hist = []
    reg_loss_hist = []
    best = (math.inf, params)
    
    for it in range(1, num_steps+1):
        params, train_state, loss = step(params, train_state)
        
        # Fail fast on NaNs (so you know the first bad step)
        if not jnp.isfinite(loss):
            print(f" NaN/Inf at step {it}; stopping early.")
            break
        
        with jax.disable_jit():
            total_l, data_l, reg_l = loss_fn_detailed(params)
        
        if loss < best[0]:
            best = (float(loss), params)
        loss_hist.append(float(total_l))
        data_loss_hist.append(float(data_l))
        reg_loss_hist.append(float(reg_l))
        
        if it in checkpoint_steps:
            checkpoint_params[it] = params
            with jax.disable_jit():
                _, derivs = compute_slope_curv_penalty(params)
                checkpoint_derivs[it] = derivs
            print(f"step {it:5d} | loss {loss:.4e} (data {data_l:.4e} + reg {reg_l:.4e}) [checkpoint]")
            # Log plateaus & mu during training
            try:
                eta_inf_t, eta0_t, delta_t, mu_t = inspect_head(tbnn, params)
                print(f"    etainf={eta_inf_t:.6g}, eta0={eta0_t:.6g}, Delta={delta_t:.6g}")
            except Exception as e:
                print("   [inspect_head failed]", e)
        elif it % print_every == 0 or it == 1:
            print(f"step {it:5d} | loss {loss:.4e} (data {data_l:.4e} + reg {reg_l:.4e})")
            # Log plateaus & mu during training (every print_every steps)
            if it % (print_every * 5) == 0 or it == 1:  # Less frequent for regular prints
                try:
                    eta_inf_t, eta0_t, delta_t, mu_t = inspect_head(tbnn, params)
                    print(f"    etainf={eta_inf_t:.6g}, eta0={eta0_t:.6g}, Delta={delta_t:.6g}")
                except Exception as e:
                    print("   [inspect_head failed]", e)
    
    params = best[1]
    print(f"\nBest loss: {best[0]:.4e}")
    
    trained_params = params
    print(f"Trained parameters saved to output")
    
    # ========================= COMPUTE DERIVATIVES ================================
    with jax.disable_jit():
        eta_pred = predict_eta(params, gamma_img, inv_aux).reshape(-1)
        _, deriv_final = compute_slope_curv_penalty(params)
        
        eta_pred_checkpoints = {}
        for step_num, cp_params in checkpoint_params.items():
            eta_pred_checkpoints[step_num] = predict_eta(cp_params, gamma_img, inv_aux).reshape(-1)
    
    # --- Read final plateau values via introspection ---
    eta_inf_f, eta0_f, delta_f, mu_f = inspect_head(tbnn, params)
    print(f"FINAL: etainf={eta_inf_f:.6g}, eta0={eta0_f:.6g}, Delta={delta_f:.6g}")
    print("FINAL mu (z=log gammadot/gamma_ref):", np.array2string(mu_f, precision=4, separator=", "))
    print(f"  Changes: Deltaetainf={(eta_inf_f-eta_inf0):+.3e}, Deltaeta0={(eta0_f-eta00):+.3e}, DeltaDelta={(delta_f-delta0):+.3e}")
    
    def compute_target_derivatives():
        z = jnp.linspace(jnp.log(reg_gamma_min), jnp.log(reg_gamma_max), reg_num_points)
        gamma_grid = jnp.exp(z)
        eta_target = eta_carreau_yasuda(gamma_grid)
        ell = jnp.log(jnp.maximum(eta_target, 1e-30))
        
        d1 = (ell[2:] - ell[:-2]) / (z[2:] - z[:-2])
        
        dz = jnp.diff(z)
        d2 = 2.0 * (
            (ell[2:]   - ell[1:-1]) / dz[1:] -
            (ell[1:-1] - ell[:-2])  / dz[:-1]
        ) / (z[2:] - z[:-2])
        
        return d1, d2, z[1:-1]
    
    target_d1, target_d2, z_grid = compute_target_derivatives()
    gamma_grid_plot = jnp.exp(z_grid)
    
    # ========================= PLOTTING ===========================================
    fig = plt.figure(figsize=(18, 12))
    
    ax1 = plt.subplot(3, 2, 1)
    ax1.loglog(gamma_vec, eta_target_vec, 'k-', linewidth=2.5, label='Target', zorder=10)
    ax1.loglog(gamma_vec, np.array(eta_pred_initial), ':', alpha=0.5, color='gray', label='Initial')
    
    colors_cp = ['blue', 'green', 'orange']
    for i, (step_num, eta_cp) in enumerate(sorted(eta_pred_checkpoints.items())):
        ax1.loglog(gamma_vec, np.array(eta_cp), '--', alpha=0.6, 
                   color=colors_cp[i % len(colors_cp)], label=f'Step {step_num}')
    
    ax1.loglog(gamma_vec, np.array(eta_pred), 'r-', linewidth=2, label=f'Final (step {num_steps})')
    ax1.set_xlabel(r'$\dot{\gamma}$ [1/s]')
    ax1.set_ylabel(r'$\eta$ [Pa.s]')
    title_str = f'Shape-Only Learning (M={M}, frozen tail, log-add)'
    ax1.set_title(title_str)
    ax1.legend(fontsize=8)
    ax1.grid(True, which='both', ls=':', alpha=0.3)
    
    ax2 = plt.subplot(3, 2, 2)
    ax2.semilogx(gamma_grid_plot, np.array(target_d1), 'k-', linewidth=2.5, label='Target', zorder=10)
    ax2.semilogx(gamma_grid_plot, np.array(deriv_initial[0]), ':', alpha=0.5, color='gray', label='Initial')
    for i, (step_num, derivs) in enumerate(sorted(checkpoint_derivs.items())):
        ax2.semilogx(gamma_grid_plot, np.array(derivs[0]), '--', alpha=0.6,
                     color=colors_cp[i % len(colors_cp)], label=f'Step {step_num}')
    ax2.semilogx(gamma_grid_plot, np.array(deriv_final[0]), 'r-', linewidth=2, label=f'Final')
    ax2.axhline(reg_s_cap, ls='--', color='red', alpha=0.3)
    ax2.axhline(-reg_s_cap, ls='--', color='red', alpha=0.3)
    ax2.set_xlabel(r'$\dot{\gamma}$ [1/s]')
    ax2.set_ylabel(r'$d\log\eta / d\log\dot{\gamma}$')
    ax2.set_title('First Derivative')
    ax2.legend(fontsize=8)
    ax2.grid(True, which='both', ls=':', alpha=0.3)
    
    ax3 = plt.subplot(3, 2, 3)
    ax3.semilogx(gamma_grid_plot, np.array(target_d2), 'k-', linewidth=2.5, label='Target', zorder=10)
    ax3.semilogx(gamma_grid_plot, np.array(deriv_initial[1]), ':', alpha=0.5, color='gray', label='Initial')
    for i, (step_num, derivs) in enumerate(sorted(checkpoint_derivs.items())):
        ax3.semilogx(gamma_grid_plot, np.array(derivs[1]), '--', alpha=0.6,
                     color=colors_cp[i % len(colors_cp)], label=f'Step {step_num}')
    ax3.semilogx(gamma_grid_plot, np.array(deriv_final[1]), 'r-', linewidth=2, label=f'Final')
    ax3.axhline(reg_curv_cap_abs, ls='--', color='red', alpha=0.3)
    ax3.axhline(-reg_curv_cap_abs, ls='--', color='red', alpha=0.3)
    ax3.set_xlabel(r'$\dot{\gamma}$ [1/s]')
    ax3.set_ylabel(r'$d^2\log\eta / d(\log\dot{\gamma})^2$')
    ax3.set_title('Second Derivative')
    ax3.legend(fontsize=8)
    ax3.grid(True, which='both', ls=':', alpha=0.3)
    
    ax4 = plt.subplot(3, 2, 4)
    ax4.semilogy(loss_hist, 'k-', linewidth=2, label='Total Loss')
    ax4.semilogy(data_loss_hist, 'b--', linewidth=1.5, label='Data Loss')
    if enable_deriv_regularization:
        ax4.semilogy(reg_loss_hist, 'r:', linewidth=1.5, label='Reg Loss')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Loss')
    ax4.set_title('Training Loss')
    ax4.legend()
    ax4.grid(True, which='both', ls=':', alpha=0.3)
    
    ax5 = plt.subplot(3, 2, 5)
    slope_violation_final = np.maximum(0, np.abs(deriv_final[0]) - reg_s_cap)
    ax5.semilogx(gamma_grid_plot, slope_violation_final, 'r-', linewidth=2)
    ax5.fill_between(gamma_grid_plot, 0, slope_violation_final, alpha=0.2, color='red')
    ax5.set_xlabel(r'$\dot{\gamma}$ [1/s]')
    ax5.set_ylabel(r'Slope Overflow')
    ax5.set_title(f'Slope Penalty (lam={reg_lambda_slope})')
    ax5.grid(True, which='both', ls=':', alpha=0.3)
    
    ax6 = plt.subplot(3, 2, 6)
    curv_violation_final = np.maximum(0, np.abs(deriv_final[1]) - reg_curv_cap_abs)
    ax6.semilogx(gamma_grid_plot, curv_violation_final, 'r-', linewidth=2)
    ax6.fill_between(gamma_grid_plot, 0, curv_violation_final, alpha=0.2, color='red')
    ax6.set_xlabel(r'$\dot{\gamma}$ [1/s]')
    ax6.set_ylabel(r'Curvature Overflow')
    ax6.set_title(f'Curvature Penalty (lam={reg_lambda_curv})')
    ax6.grid(True, which='both', ls=':', alpha=0.3)
    
    plt.tight_layout()
    if show_plots:
        plt.show()
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"TRAINING SUMMARY - SHAPE-ONLY MODE")
    print(f"{'='*70}")
    if freeze_both_plateaus:
        print(f"BOTH PLATEAUS FROZEN (learned shape only)")
    print(f"Final total loss: {loss_hist[-1]:.4e}")
    print(f"  Data loss: {data_loss_hist[-1]:.4e}")
    if enable_deriv_regularization:
        print(f"  Reg loss: {reg_loss_hist[-1]:.4e}")
    print(f"\nDerivative statistics:")
    print(f"  Max |slope|: {float(np.max(np.abs(deriv_final[0]))):.4f} (cap: {reg_s_cap})")
    print(f"  Max |curvature|: {float(np.max(np.abs(deriv_final[1]))):.4f} (cap: {reg_curv_cap_abs})")
    print(f"\nModel: M={M}, hidden={hidden_units}, s_floor={s_floor}, alpha_temp={alpha_temp}")
    print(f"  log_head=True, log_mixing='add', no gating, no PL")
    print(f"{'='*70}")
    
    # Return results dictionary
    return {
        'trained_params': trained_params,
        'loss_hist': loss_hist,
        'data_loss_hist': data_loss_hist,
        'reg_loss_hist': reg_loss_hist,
        'eta_pred': eta_pred,
        'eta_target': eta_target_vec,
        'gamma_vec': gamma_vec,
        'deriv_final': deriv_final,
        'checkpoint_params': checkpoint_params,
        'eta_pred_checkpoints': eta_pred_checkpoints,
        'fig': fig
    }


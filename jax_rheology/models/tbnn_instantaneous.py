"""Instantaneous (algebraic) TBNN closure: bounded-slope viscosity.

The learned generalized-Newtonian closure used for the Carreau-Yasuda
results. A small Flax module maps the scalar strain-rate invariant to a
viscosity through a mixture of sigmoids in ``z = log(gammadot / gamma_ref)``,

    eta(z) = eta_inf + delta * (1 - sum_i alpha_i * sigmoid((z - mu_i) / s_i))

so eta is confined to ``[eta_inf, eta_inf + delta]`` and is smooth and
monotone by construction, which keeps reverse-mode gradients well behaved
across several decades of shear rate.

Provides the model definition (:class:`BoundedSlopeViscosity`), its
initialisation to a near-Newtonian start, the viscosity field evaluated from
a velocity field, and the stress forcing the solver consumes. The closed-form
viscosity laws and the shared tensor helpers live in
:mod:`jax_rheology.models.generalized_newtonian`, which dispatches to this
module when a run selects ``'TBNN'``.
"""
import jax
import jax.numpy as jnp
import jax.nn as jnn
from flax import linen as nn
from flax.core import freeze, unfreeze
from typing import Optional, Any, List, Tuple, Sequence

import jax_ib.base as ib
from jax_ib.base import diffusion

from jax_rheology.models.generalized_newtonian import (
    compute_S_R,
    compute_shear_rate,
    compute_invariants_2d,
    extract_data,
    tensor_divergence,
)

# =============================================================================
# SECTION 5 -- TBNN (bounded-slope viscosity)
# =============================================================================

def _logit(p):
    p = jnp.clip(p, 1e-6, 1-1e-6)
    return jnp.log(p) - jnp.log1p(-p)

def _softplus_inv(y):
    # inverse of softplus, numerically safe
    y = jnp.clip(y, 1e-12, 1e12)
    return jnp.log(jnp.expm1(y))

def _default_mu_centers(M, mu_min_gamma, mu_max_gamma, gamma_ref):
    """Evenly space mu centers in z=log(gammadot/gamma_ref). Fallback to [-2.5,2.5] if no bounds."""
    if (mu_min_gamma is not None) and (mu_max_gamma is not None):
        z_lo = jnp.log(jnp.asarray(mu_min_gamma) / gamma_ref)
        z_hi = jnp.log(jnp.asarray(mu_max_gamma) / gamma_ref)
    else:
        z_lo, z_hi = -2.5, 2.5
    return jnp.linspace(z_lo, z_hi, M)

def init_tbnn_soft_newtonian(tbnn_model, rng, H, W, eta0,
                             A_frac=0.05,        # unused in new head; kept for API compat
                             k_frac=0.2,         # unused; kept for API compat
                             pair_modes=(0,1)):  # unused; kept for API compat
    """
    Initialize the monotone-MLP head to be ~Newtonian at eta0, while keeping gradients alive.
    - Sets global learnable scalars: eta_inf ~= max(eta_min*1.02, 0.95*eta0), delta ~= 0.05*eta0
    - Sets last Dense_* bias so the mixture centers (mu) span a reasonable log-gammadot range,
      with moderate scales (s) and uniform logits.
    - Does NOT zero any kernels.
    """
    gamma_dummy = jnp.ones((H, W))
    I_dummy     = jnp.ones((2, H, W))  # (I1, I2) present; I2 unused for now

    params = tbnn_model.init(rng, gamma_dummy, I_dummy)  # {'params': ...}
    p = unfreeze(params)

    use_log  = getattr(tbnn_model, "log_head", False)
    mixing   = getattr(tbnn_model, "log_mixing", "add")

    # --- Set the global scalars (created via self.param in the Module) ---
    eta0 = float(eta0)
    eta_min = float(tbnn_model.eta_min)
    eta_max = float(tbnn_model.eta_max)

    # Choose low tail to force shape learning (0.10 * eta0) instead of conservative 0.95
    # If you want shape-only learning with frozen tail, use 0.10; for normal use 0.95
    # eta_inf_init = float(jnp.clip(0.10*eta0, eta_min*1.02, eta0*0.99))
    # choose a conservative tail to stabilize solver; small delta keeps curve ~flat
    eta_inf_init = float(jnp.clip(0.95 * eta0, eta_min * 1.02, eta0 * 0.995))
    delta_init   = float(jnp.clip(0.05 * eta0, 1e-4, max(1e-3, eta_max - eta_inf_init)))
    
    if use_log:
        if mixing == "add":
            if getattr(tbnn_model, 'freeze_eta0', False):
                # Hard-freeze mode: params are log_eta_inf_raw (dummy), eta_partition_logit
                # eta0 = eta0_fixed, use partition p to set (etainf, r) coherently
                eta0_val = float(tbnn_model.eta0_fixed if tbnn_model.eta0_fixed is not None else eta0)
                # Start with p ~= 0.95 (so etainf ~= 0.95.eta0, r ~= 0.053)
                p_init = max(min(eta_inf_init / eta0_val, 0.99), 0.01)
                logit_init = float(jnp.log(p_init / (1.0 - p_init)))  # inverse sigmoid
                p['params']['log_eta_inf_raw'] = jnp.asarray(0.0)  # dummy, not used
                p['params']['eta_partition_logit'] = jnp.asarray(logit_init)
            else:
                # Normal mode: params are log_eta_inf_raw, r_raw
                if not all(k in p['params'] for k in ("log_eta_inf_raw","r_raw")):
                    raise RuntimeError("Expect log_eta_inf_raw & r_raw for log_mixing='add'")
                r_init = max(eta0/eta_inf_init - 1.0, 1e-3)
                p['params']['log_eta_inf_raw'] = jnp.asarray(jnp.log(eta_inf_init))
                p['params']['r_raw']           = jnp.asarray(jnp.log(jnp.exp(r_init) - 1.0))

        elif mixing == "geom":
            # params: log_eta_inf_raw, log_range_raw  (L = softplus(log_range_raw))
            if getattr(tbnn_model, 'freeze_eta0', False):
                # Freeze mode: L is computed dynamically, don't create log_range_raw
                p['params']['log_eta_inf_raw'] = jnp.asarray(jnp.log(eta_inf_init))
                # log_range_raw not created - will be computed in __call__ as L = log(eta0_fixed) - log(etainf)
            else:
                # Normal mode: both learnable
                if not all(k in p['params'] for k in ("log_eta_inf_raw","log_range_raw")):
                    raise RuntimeError("Expect log_eta_inf_raw & log_range_raw for log_mixing='geom'")
                L_init = max(jnp.log(eta0) - jnp.log(eta_inf_init), 1e-3)
                p['params']['log_eta_inf_raw'] = jnp.asarray(jnp.log(eta_inf_init))
                p['params']['log_range_raw']   = jnp.asarray(jnp.log(jnp.exp(L_init) - 1.0))

        else:
            raise ValueError(f"unknown log_mixing={mixing}")

    else:
        # original linear head: eta_inf_raw, delta_raw
        delta_init = float(jnp.clip(eta0 - eta_inf_init, 1e-4, eta_max - eta_inf_init))
        if getattr(tbnn_model, 'freeze_eta0', False):
            eta_inf_init = min(eta_inf_init, float(tbnn_model.eta0_fixed) - 1e-6)
            delta_init   = max(float(tbnn_model.eta0_fixed) - eta_inf_init, 1e-6)
        p['params']['eta_inf_raw'] = jnp.asarray(_softplus_inv(eta_inf_init))
        p['params']['delta_raw']   = jnp.asarray(_softplus_inv(delta_init))

    # --- Initialize the final Dense_* bias to sane mixture params: [mu(K), logs(K), logits(K)] ---
    dense_keys = [k for k in p['params'] if k.startswith('Dense_')]
    if not dense_keys:
        raise RuntimeError("No Dense_* layers found in params.")
    last_key = max(dense_keys, key=lambda s: int(s.split('_')[1]))
    head = p['params'][last_key]

    out_dim = head['bias'].shape[0]
    K = tbnn_model.M  # number of logistics in the mixture
    expected = 3 * K
    assert out_dim == expected, f"Final Dense out_dim must equal 3*M (got {out_dim}, expected {expected})."

    # centers across a log-gamma window; honor mu_min_gamma if present
    if getattr(tbnn_model, "mu_min_gamma", None) is not None:
        z_lo = float(jnp.log(jnp.asarray(tbnn_model.mu_min_gamma) / tbnn_model.gamma_ref))
        # Prefer tail_gate_gamma; else fallback to mu_max_gamma; else give ~2.5 decades
        if getattr(tbnn_model, "tail_gate_gamma", None) is not None:
            z_hi = float(jnp.log(jnp.asarray(tbnn_model.tail_gate_gamma) / tbnn_model.gamma_ref))
            mu   = jnp.linspace(z_lo + 0.3, z_hi - 0.3, K)    # keep margins to avoid early/late spill
        elif getattr(tbnn_model, "mu_max_gamma", None) is not None:
            z_hi = float(jnp.log(jnp.asarray(tbnn_model.mu_max_gamma) / tbnn_model.gamma_ref))
            mu   = jnp.linspace(z_lo + 0.3, z_hi - 0.3, K)
        else:
            mu   = jnp.linspace(z_lo + 0.3, z_lo + 3.0, K)    # ~3 z-units (~1.3 decades)
    else:
        mu = jnp.linspace(-2.5, 2.5, K)

    # scales s ~= 0.7 (via softplus(logs)), and uniform logits
    logs   = jnp.full((K,), jnp.log(jnp.exp(0.7) - 1.0))  # softplus(logs) ~= 0.7
    logits = jnp.zeros((K,))

    bias = jnp.concatenate([mu, logs, logits])
    head['bias'] = bias
    p['params'][last_key] = head

    return freeze(p)

# === Bounded-slope viscosity helpers (AD-safe, no hard clips) ===
class BoundedSlopeViscosity(nn.Module):
    """
    Invariant-conditioned, monotone & bounded viscosity head (TBNN scalar coefficient).
    - MLP over invariants (I1, I2) -> mixture params {mu_i, s_i, alpha_i}
    - Viscosity: eta(z) = eta_inf + delta * (1 - sum_i alpha_i * sigmoid((z - mu_i)/s_i))
      with z = log(gamma_dot / gamma_ref)
    - Guarantees eta  in  [eta_inf, eta_inf + delta], smooth & monotone decreasing in z.
    - eta_inf and delta are learnable *global* scalars (via self.param).
    """
    hidden_units: List[int] = None
    M: int = 4                          # number of logistics in the mixture
    eta_min: float = 1e-2
    eta_max: float = 10.0
    gamma_ref: float = 1.0              # reference shear-rate for z = log(gamma/gamma_ref)

    # --- stability / training knobs ---
    s_floor: float = 0.0                # minimum logistic width in z-space (0.5-0.8 for broader kernels)
    alpha_temp: float = 1.0             # softmax temperature (>1 spreads weights, <1 sharpens)

    # --- NEW: keep curvature after a given shear-rate ---
    mu_min_gamma: Optional[float] = None   # all centers mu_i >= log(mu_min_gamma/gamma_ref)
    mu_max_gamma: Optional[float] = None   # (optional) also cap above
    gate_gamma:   Optional[float] = None   # if set, multiply mixture by gate G(z) starting here
    gate_width_z: float = 0.5              # gate smoothness in z = log(gammadot/gamma_ref)
    
    # --- NEW: delay the tail (etainf only kicks in at high gammadot) ---
    tail_gate_gamma: Optional[float] = None   # turn-on for tail
    tail_gate_width_z: float = 0.5           # smoothness in z = log(gammadot/gamma_ref)

    # --- keep existing freeze knobs ---
    freeze_eta0: bool = False           # hard-freeze eta0 (learn etainf via gap)
    eta0_fixed: Optional[float] = None  # explicit eta0 value if freezing
    eta0_eps: float = 1e-6              # keeps delta positive when frozen

    # --- NEW: per-mode power-law bump (toggle + width) ---
    enable_pl_per_mode: bool = False        # turn per-mode PL bump on/off
    pl_width_z: float = 0.5                 # smooth onset width in z for each mode

    # --- NEW: log head toggle ---
    log_head: bool = False                  # learn in log-eta space if True
    log_mixing: str = "add"                 # "add" = log1p(r*(1-F)), "geom" = log_eta_inf + L*(1-F)

    # --- NEW: freeze centers toggle ---
    freeze_centers: bool = False            # keep mu_i stationary (non-trainable)

    def setup(self):
        if self.hidden_units is None:
            self.hidden_units = [32, 32]

    @nn.compact
    def __call__(self, gamma_dot: jnp.ndarray, invariants_aux: jnp.ndarray) -> jnp.ndarray:
        # Inputs:
        #   gamma_dot: (H,W)
        #   invariants_aux: (C,H,W) objective scalars; we read I2 but only use I1 for now
        H, W = gamma_dot.shape
        C = invariants_aux.shape[0]

        # Flatten invariants -> (N,C); then use only I1 (column 0), but keep I2 in the tensor.
        I = jnp.stack([invariants_aux[c] for c in range(C)], axis=-1).reshape(-1, C)  # (N,C)
        X = I[:, :1]  # use only I1 now; shape (N,1)

        # Small MLP over invariants to produce mixture params
        h = X
        for units in (self.hidden_units or []):
            h = jnn.tanh(nn.Dense(units)(h))

        K = self.M
        raw = nn.Dense(3 * K)(h)  # [mu(K), logs(K), logits(K)]

        mu_raw, logs, logits = jnp.split(raw, 3, axis=-1)  # each (N,K)

        # ----- NEW: bound the center locations in z-space -----
        # z = log(gamma/gamma_ref); enforce mu in [z_lo, z_hi] or [z_lo, +inf)
        if self.mu_min_gamma is not None:
            z_lo = jnp.log(jnp.asarray(self.mu_min_gamma) / self.gamma_ref)
            if self.mu_max_gamma is not None:
                z_hi = jnp.log(jnp.asarray(self.mu_max_gamma) / self.gamma_ref)
                mu = z_lo + (z_hi - z_lo) * jnn.sigmoid(mu_raw)          # (N,K) in [z_lo, z_hi]
            else:
                mu = z_lo + jnn.softplus(mu_raw)                          # (N,K) in [z_lo, +inf)
        else:
            mu = mu_raw                                                   # unbounded (old behavior)

        # ----- NEW: optionally freeze mu centers (non-trainable) -----
        if self.freeze_centers:
            # Create/read non-trainable centers in 'constants'
            mu_const = self.variable(
                'constants', 'mu_centers',
                lambda: _default_mu_centers(self.M, self.mu_min_gamma, self.mu_max_gamma, self.gamma_ref)
            )
            # Broadcast to (N,K) and stop all gradients through mu
            mu = jnp.broadcast_to(jax.lax.stop_gradient(mu_const.value).reshape(1, -1), mu.shape)

        # scales/weights
        s = jnn.softplus(logs) + (self.s_floor if self.s_floor > 0 else 1e-4)
        alpha = jnn.softmax(logits / self.alpha_temp, axis=-1)

        # ---------- global scalars ----------
        if self.log_head:
            log_eta_inf_raw = self.param("log_eta_inf_raw", lambda rng: jnp.array(0.0))

            if self.log_mixing == "add":
                # log eta = log(eta_inf) + log(1 + r*(1-F))
                if self.freeze_eta0:
                    # ----- HARD-FREEZE eta0 via partition-of-unity reparameterization -----
                    # eta0 is fixed; learn a partition p  in  (0,1) to set (etainf, r) coherently
                    # etainf = p.eta0_fixed,  r = (1-p)/p  =>  eta0 = etainf.(1+r) = eta0_fixed exactly
                    logit = self.param("eta_partition_logit", nn.initializers.zeros, ())  # trainable scalar
                    p = jnn.sigmoid(logit)
                    # numerical guards
                    p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
                    
                    eta0_const = self.eta0_fixed if (self.eta0_fixed is not None) else self.eta_max
                    eta0_const = jnp.asarray(eta0_const, dtype=jnp.float32)
                    
                    eta_inf = eta0_const * p              # >= 0
                    r       = (1.0 - p) / p               # >= 0
                    
                    # keep within declared bounds (very gentle)
                    eta_inf = jnp.clip(eta_inf, self.eta_min + 1e-12, self.eta_max - 1e-12)
                    # r doesn't need explicit bounds; it's implied by p in (0,1)
                else:
                    # ----- ORIGINAL PATH: both etainf and eta0 learnable -----
                    r_raw = self.param("r_raw", lambda rng: jnp.array(jnp.log(jnp.exp(0.05) - 1.0)))
                    log_eta_inf = log_eta_inf_raw
                    r = jnn.softplus(r_raw) + 1e-12
                    # cap for stability
                    r = jnp.minimum(r, 30.0)
                    # compute eta_inf from log_eta_inf for non-freeze path
                    eta_inf = jnp.exp(log_eta_inf)

            elif self.log_mixing == "geom":
                # log eta = log(eta_inf) + L*(1-F)  with L = log(eta0) - log(eta_inf) >= 0
                log_eta_inf = log_eta_inf_raw
                if self.freeze_eta0 and (self.eta0_fixed is not None):
                    # enforce eta0 fixed: L = log(eta0_fixed) - log(etainf)
                    log_eta0_fixed = jnp.log(jnp.asarray(self.eta0_fixed, dtype=jnp.float32))
                    L = jax.lax.stop_gradient(log_eta0_fixed - log_eta_inf)
                    # Ensure L >= 0 (eta0 >= etainf)
                    L = jnp.maximum(L, 1e-12)
                else:
                    log_range_raw = self.param("log_range_raw", lambda rng: jnp.array(jnp.log(jnp.exp(0.1)-1.0)))
                    L = jnn.softplus(log_range_raw) + 1e-12  # >=0
                    # cap for stability
                    L = jnp.minimum(L, 30.0)
                eta_inf = jnp.exp(log_eta_inf) + 1e-12

            else:
                raise ValueError(f"unknown log_mixing={self.log_mixing}")

        else:
            # ----- original linear head -----
            eta_inf_raw = self.param("eta_inf_raw", lambda rng: jnp.array(0.0))
            delta_raw   = self.param("delta_raw",   lambda rng: jnp.array(0.0))
            eta_inf_free = jnn.softplus(eta_inf_raw) + 1e-6
            # cap for stability
            eta_inf_free = jnp.minimum(eta_inf_free, self.eta_max - 1e-6)
            if self.freeze_eta0:
                eta_inf = jnp.minimum(eta_inf_free, self.eta0_fixed - self.eta0_eps)
                delta   = jnp.maximum(self.eta0_fixed - eta_inf, self.eta0_eps)
            else:
                eta_inf = eta_inf_free
                delta   = jnn.softplus(delta_raw)
                # cap for stability
                delta = jnp.minimum(delta, self.eta_max - eta_inf)

        # ----- DEBUG: expose plateau values -----
        # Ensure the following three are defined in either branch:
        #   eta_inf_val  (scalar >= 0)
        #   eta0_val     (scalar >= eta_inf_val)
        #   delta_val    (eta0_val - eta_inf_val)
        if self.log_head:
            if self.log_mixing == "add":
                # eta_inf is already in linear space (from partition or exp(log_eta_inf_raw))
                eta_inf_val = eta_inf
                eta0_val    = eta_inf_val * (1.0 + r)
                delta_val   = eta0_val - eta_inf_val
            else:  # "geom" with L >= 0
                eta_inf_val = jnp.exp(log_eta_inf)
                eta0_val    = jnp.exp(log_eta_inf + L)
                delta_val   = eta0_val - eta_inf_val
        else:
            # linear head with eta_inf and delta (both >= 0)
            eta_inf_val = eta_inf
            delta_val   = delta
            eta0_val    = eta_inf + delta_val

        # Sow a compact snapshot for the caller (means across N for mu, finite-safe)
        mu_finite = jnp.where(jnp.isfinite(mu), mu, 0.0)
        self.sow('intermediates', 'mu_snapshot', jnp.mean(mu_finite, axis=0))  # (K,)
        self.sow('intermediates', 'eta_inf_value', eta_inf_val)
        self.sow('intermediates', 'eta0_value',    eta0_val)
        self.sow('intermediates', 'delta_value',   delta_val)

        # z for current gammadot - SAFER z computation
        eps = 1e-30
        z = jnp.log(jnp.clip(gamma_dot.reshape(-1, 1) / self.gamma_ref, eps, 1.0/eps))
        # clamp z to avoid huge arguments to sigmoid
        z = jnp.clip(z, -40.0, 40.0)  # (N,1)

        # mixture CDF - SAFER widths and sigmoid arguments
        s = jnp.maximum(s, self.s_floor if self.s_floor > 0 else 1e-3)
        q = (z - mu) / s
        # keep q in stable range for sigmoid
        q = jnp.clip(q, -40.0, 40.0)
        sigma = jnn.sigmoid(q)
        F_raw = jnp.sum(alpha * sigma, axis=-1, keepdims=True)  # (N,1)

        # ----- NEW: optional gate so F ~= 0 before a threshold - SAFER gate -----
        F = F_raw
        if self.gate_gamma is not None:
            z_gate = jnp.log(jnp.clip(self.gate_gamma / self.gamma_ref, eps, 1.0/eps))
            w = jnp.maximum(self.gate_width_z if self.gate_width_z is not None else 0.6, 1e-3)
            gate_arg = (z - z_gate) / w
            gate_arg = jnp.clip(gate_arg, -40.0, 40.0)
            G_low = jnn.sigmoid(gate_arg)  # (N,1), up in z
            F = G_low * F

        # NEW: tail gate -- defer etainf leverage to higher gammadot - SAFER tail gate
        if self.tail_gate_gamma is not None:
            z_tail = jnp.log(jnp.clip(self.tail_gate_gamma / self.gamma_ref, eps, 1.0/eps))
            w_tail = jnp.maximum(self.tail_gate_width_z if self.tail_gate_width_z is not None else 0.5, 1e-3)
            tail_arg = (z - z_tail) / w_tail
            tail_arg = jnp.clip(tail_arg, -40.0, 40.0)
            G_tail = jnn.sigmoid(tail_arg)
            F = F * G_tail

        # ---------- combine - SAFER mixing to avoid negative/zero multipliers ----------
        if self.log_head:
            if self.log_mixing == "add":
                # Bound the multiplier to avoid 1 + r*(1-F) <= 0
                # r >= 0 by construction; still guard numerically
                mult = 1.0 + r * (1.0 - F)
                mult = jnp.maximum(mult, 1e-12)
                eta  = eta_inf * mult
            else:  # "geom"
                ell = log_eta_inf + L * (1.0 - F)
                eta = jnp.exp(ell)
        else:
            # Linear head: eta_inf + delta*(1-F), naturally positive if eta_inf,delta >= 0
            eta = eta_inf + delta * (1.0 - F)
        
        # Final safety clamp to valid physical bounds
        eta = jnp.clip(eta, self.eta_min + 1e-12, self.eta_max - 1e-12)

        # === NEW: optional per-mode power-law bump in log-space (acts only after each mode) ===
        if self.enable_pl_per_mode:
            K = self.M
            # 1) Learnable per-mode slopes (strictly <= 0). Init ~ no effect.
            pl_slope_raw = self.param("pl_slope_raw", lambda rng: jnp.full((K,), -10.0))
            #    p_k = -softplus(raw); with raw=-10 => p_k ~= -4.5e-5 (tiny negative)
            p_k = -jnn.softplus(pl_slope_raw)                     # (K,) <= 0
            p_k = p_k.reshape((1, K))                              # (1,K) for broadcasting

            # 2) Smooth onset per mode (local to this mode): S_k ~ max(0, z - mu_k) with width pl_width_z
            w_pl = self.pl_width_z
            Zm   = (z - mu) / w_pl                                 # (N,K)
            S_k  = w_pl * (jnp.log1p(jnp.exp(Zm)) - jnp.log(2.0))  # (N,K), 0 before mu_k, ~ (z-mu_k) after

            # 3) Localized gate per mode to avoid "leaking" curvature too early:
            #    lam_k turns on after mu_k; multiply by alpha to respect mixture responsibility.
            lam_k    = jnn.sigmoid(Zm)                             # (N,K), ~0 before mu_k, ->1 after
            weight_k = alpha * lam_k                                # (N,K)

            # 4) Optional global gate: if user set gate_gamma, suppress PL before that too.
            if self.gate_gamma is not None:
                G_eff = jnn.sigmoid((z - jnp.log(jnp.asarray(self.gate_gamma) / self.gamma_ref)) / self.gate_width_z)  # (N,1)
            else:
                G_eff = 1.0

            # 5) Sum per-mode bumps (log-domain), then exponentiate back to eta.
            #    bump_total(z) = Sigma_k weight_k(z) * p_k * S_k(z)
            bump_total = G_eff * jnp.sum(weight_k * (p_k * S_k), axis=-1, keepdims=True)  # (N,1)

            ell = jnp.log(jnp.clip(eta, 1e-30, 1e30))              # log eta_base
            ell = ell + bump_total                                  # add PL bump
            eta = jnp.exp(ell)                                      # back to linear eta

        eta = jnp.clip(eta, self.eta_min, self.eta_max).reshape(H, W)
        return eta

def build_tbnn_bounded_model(hidden_units: List[int], M: int = 4,
                             eta_min: float = 1e-2, eta_max: float = 5.0,
                             smax: float = 0.5, zmax: float = 8.0,   # unused; kept for API compat
                             gamma_ref: float = 1.0,
                             log_head: bool = False,
                             freeze_centers: bool = False,
                             freeze_eta0: bool = False,
                             eta0_fixed: Optional[float] = None,
                             **kw) -> BoundedSlopeViscosity:
    """Build the monotone-MLP viscosity model (TBNN scalar coefficient).
    
    Additional kwargs:
        s_floor: minimum logistic width (0.5-0.8 for broader, smoother kernels)
        alpha_temp: softmax temperature (>1 spreads mixture weights, <1 sharpens)
        freeze_eta0: if True, hard-freeze eta0 and learn etainf via gap parameter (log-add only) (default: False)
        eta0_fixed: fixed value of eta0 when freeze_eta0=True (default: None, uses eta_max)
        eta0_eps: small positive floor for delta when frozen (default: 1e-6)
        mu_min_gamma: lower bound on center locations (no curvature before this gammadot)
        mu_max_gamma: optional upper bound on center locations
        gate_gamma: if set, multiply mixture by smooth gate starting at this gammadot
        gate_width_z: gate smoothness in log(gammadot) space (default: 0.5)
        tail_gate_gamma: if set, delay tail (etainf) to this gammadot (prevents mid-shear drag)
        tail_gate_width_z: tail gate smoothness in log(gammadot) space (default: 0.5)
        enable_pl_per_mode: if True, add per-mode log-slope bumps on top of the linear blend
        pl_width_z: smooth onset width in z-units for each mode's PL bump (default 0.5)
        log_head: if True, learn in log-eta space (more stable, weights decades evenly) (default: False)
        log_mixing: mixing mode for log_head - "add" = log1p(r*(1-F)), "geom" = L*(1-F) (default: "add")
        freeze_centers: if True, keep mu_i stationary (non-trainable) (default: False)
    """
    return BoundedSlopeViscosity(hidden_units=hidden_units, M=M,
                                 eta_min=eta_min, eta_max=eta_max,
                                 gamma_ref=gamma_ref, log_head=log_head,
                                 freeze_centers=freeze_centers,
                                 freeze_eta0=freeze_eta0,
                                 eta0_fixed=eta0_fixed,
                                 **kw)

def tbnn_eta_bounded_from_v(velocity, tbnn_model: BoundedSlopeViscosity, params,
                            *, gamma_ref: Optional[float] = None) -> jnp.ndarray:
    """Compute eta(x,y) from velocity via invariant-conditioned monotone head.
    Preserves dtype/shape of the grid data.
    """
    S, R = compute_S_R(velocity)
    shear_rate = compute_shear_rate(S)
    invariants = compute_invariants_2d(S, R)  # (2,H,W)  -> reads I2 but head uses only I1 for now

    prms = params if ('params' in params) else {'params': params}
    model = tbnn_model if gamma_ref is None else tbnn_model.replace(gamma_ref=gamma_ref)
    eta = model.apply(prms, shear_rate, invariants)  # (H,W)

    template = velocity[0].data
    eta = jnp.asarray(eta, dtype=template.dtype)
    if eta.shape != template.shape:
        eta = jnp.broadcast_to(eta, template.shape)
    return eta

def tbnn_forward_2d_bounded(v, tbnn_model, params):
    """tau = 2 eta S using the bounded viscosity head. Returns GridArrayTensor 2x2."""
    eta = tbnn_eta_bounded_from_v(v, tbnn_model, params)
    S, _ = compute_S_R(v)
    S_data = extract_data(S)
    tau_data = 2.0 * S_data * eta[None, None, :, :]

    return ib.finite_differences.GridArrayTensor([
        [ib.grids.GridVariable(
            ib.grids.GridArray(tau_data[i, j], S[i][j].offset, v[0].grid),
            v[0].bc
        ) for j in range(2)]
        for i in range(2)
    ])

# Back-compat: expose the new bounded forward under the old alias
tbnn_forward_2d_finalized = tbnn_forward_2d_bounded



def TBNN_stress_forcing(pressure_gradient, permeability, tbnn_model, params, factor, U_f, nu0):
    """IMEX explicit correction for TBNN bounded head: factor div tau - nu0 grad^2u."""
    def forcing(v):
        tau = tbnn_forward_2d_bounded(v, tbnn_model, params)
        div_full = tensor_divergence(tau)
        div_impl = tuple(diffusion.diffuse(u, nu0) for u in v)

        aligned_div = tuple(
            ib.interpolation.linear(ib.grids.GridVariable(stress, u.bc), u.offset)
            for stress, u in zip(div_full, v)
        )
        corr = tuple(factor * a_.data - i_.data for a_, i_ in zip(aligned_div, div_impl))
        return tuple(
            ib.grids.GridArray(pxn * jnp.ones_like(u.data) - permeability * (u.data - U_f) + c, u.offset, u.grid)
            for pxn, u, c in zip(pressure_gradient, v, corr)
        )
    return forcing


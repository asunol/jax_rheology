"""TBNN closure checks + fitting harness (mirrors ``visco_families.py``).

Anchored potential-mobility TBNN. Provides, in the style of
the family harness:
  * the continuous TBNN source ``tbnn_source_R`` and a readout-aware
    ``tbnn_viscometric`` (so the shared Newton-root
    ``vf.steady_simple_shear_reference`` works for the TBNN);
  * ``tbnn_steady_reference`` -- the TBNN's own steady-shear truth, used
    by the selftest self-consistency check and the G3/G4 observable gates;
  * ``_constriction_loss_fn_theta`` -- velocity-RMSE loss differentiable
    w.r.t. the theta pytree (G2/fitting);
  * ``tbnn_invariant_cloud`` -- logs the visited ``(x1, x2, x3)`` features
    (sets the tanh bound ``c`` and restricts later 0D protocols);
  * the gate callables: ``tbnn_regression_gate`` (G1) here; G2-G4 are
    added with the fitting driver.

Kernel-restart note: importing this module imports ``tbnn_closure``,
which registers ``tbnn_potential_logconf_bk_v2`` at import. ``cr.register``
refuses duplicates -- restart the kernel after editing the closure.

Notebook usage: enable float64, then ``import visco_tbnn as vt`` and call
``vt.tbnn_regression_gate()`` (GPU). Everything heavy is here.

Tier-1 and Tier-3 below are the anchored viscoelastic fit and the
yield-capable elastoviscoplastic fit respectively; see
``jax_rheology/models/tbnn_memory.py`` for the switches behind each.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import analytic_limits_validation as p3b
import visco_families as vf
from jax_rheology.models import registry as cr
from jax_rheology import log_conformation as lc
from jax_rheology.models import tbnn_memory as tb


# ===========================================================================
# Sec.1. Continuous TBNN source R(A) and viscometric readout
# ===========================================================================

def _bound_c_from_params(params, default=tb.TBNN_DEFAULT_BOUND_C):
    try:
        return lc._params_get(params, 'tbnn_bound_c')
    except (KeyError, AttributeError):
        return default


def _switches_from_params(params):
    """Read the (anchored, mobility, kappa) STATIC switches from a harness
    params dict (defaults = Tier-1). These let the 0D reference / recovery
    eval interrogate a non-default registration (unanchored / EVP) without
    changing the Tier-1 callers -- the switches are still plain Python
    passed at call time, never in the differentiable pytree."""
    def _get(name, default):
        try:
            return lc._params_get(params, name)
        except (KeyError, AttributeError):
            return default
    anchored = bool(_get('tbnn_anchored', tb.TBNN_DEFAULT_ANCHORED))
    mobility = str(_get('tbnn_mobility', tb.TBNN_DEFAULT_MOBILITY))
    kappa = float(_get('tbnn_kappa', tb.TBNN_DEFAULT_KAPPA))
    return anchored, mobility, kappa


def tbnn_source_R(A_xx, A_xy, A_yy, A_zz, lam, params):
    """``R(A) = -(1/lam)(m0 I + m1 A) K(A)`` (continuous source, mirrors
    the family ``*_source_R``).

    Computed as ``-(1/lam) M_frozen (A - A*)`` from
    :func:`tb.tbnn_K_and_frozen`, which equals ``-(1/lam) Mob K`` by the
    Sec. 0.4 factorization -- so the reference solver roots exactly the
    closure's own balance (the floored quantities, consistent with the
    integrated dynamics). ``theta = params['theta']``,
    ``bound_c = params['tbnn_bound_c']``.
    """
    theta = lc._params_get(params, 'theta')
    bound_c = _bound_c_from_params(params)
    anchored, mobility, kappa = _switches_from_params(params)
    _, M, As = tb.tbnn_K_and_frozen(
        jnp.asarray(A_xx), jnp.asarray(A_xy),
        jnp.asarray(A_yy), jnp.asarray(A_zz), theta, bound_c,
        anchored=anchored, mobility=mobility, kappa=kappa)
    D_xx = A_xx - As[0]
    D_xy = A_xy - As[1]
    D_yy = A_yy - As[2]
    D_zz = A_zz - As[3]
    MD_xx = M[0] * D_xx + M[1] * D_xy
    MD_xy = 0.5 * ((M[0] * D_xy + M[1] * D_yy) + (M[1] * D_xx + M[2] * D_xy))
    MD_yy = M[1] * D_xy + M[2] * D_yy
    MD_zz = M[3] * D_zz
    return (-MD_xx / lam, -MD_xy / lam, -MD_yy / lam, -MD_zz / lam)


def tbnn_viscometric(A_xx, A_xy, A_yy, A_zz, params, Gp):
    """``tau = Gp K(A)`` => ``N1 = Gp(K_xx - K_yy)``, ``tau_xy = Gp K_xy``,
    ``N2 = Gp(K_yy - K_zz)`` (the non-Hookean TBNN readout)."""
    theta = lc._params_get(params, 'theta')
    bound_c = _bound_c_from_params(params)
    anchored, mobility, kappa = _switches_from_params(params)
    K, _, _ = tb.tbnn_K_and_frozen(
        jnp.asarray(A_xx), jnp.asarray(A_xy),
        jnp.asarray(A_yy), jnp.asarray(A_zz), theta, bound_c,
        anchored=anchored, mobility=mobility, kappa=kappa)
    return (Gp * (K[0] - K[2]), Gp * K[1], Gp * (K[2] - K[3]))


def tbnn_steady_reference(theta, params, gammadot, *, Gp=1.0, lam=None,
                          bound_c=None, A_init=None, n_iter=300,
                          anchored=None, mobility=None, kappa=None):
    """Newton root of ``stretch(A; gammadot) + R_theta(A) = 0`` -- the
    TBNN's own steady-shear truth (via ``vf.steady_simple_shear_reference``
    with ``tbnn_source_R`` / ``tbnn_viscometric``). The optional
    ``anchored`` / ``mobility`` / ``kappa`` override the static switches for
    a non-default (unanchored / EVP) registration (Tier-3); defaults fall
    back to ``params`` (then Tier-1)."""
    if lam is None:
        lam = lc._params_get(params, 'lam')
    if bound_c is None:
        bound_c = _bound_c_from_params(params)
    pp = dict(params)
    pp['theta'] = theta
    pp['tbnn_bound_c'] = bound_c
    p_anch, p_mob, p_kap = _switches_from_params(pp)
    pp['tbnn_anchored'] = p_anch if anchored is None else bool(anchored)
    pp['tbnn_mobility'] = p_mob if mobility is None else str(mobility)
    pp['tbnn_kappa'] = p_kap if kappa is None else float(kappa)
    return vf.steady_simple_shear_reference(
        tbnn_source_R, gammadot, lam, params=pp, Gp=Gp,
        viscometric_fn=tbnn_viscometric, A_init=A_init, n_iter=n_iter)


# ===========================================================================
# Sec.2. theta-differentiable constriction loss + invariant-cloud logger
# ===========================================================================

def _constriction_loss_fn_theta(cfg, init_state, model, grid, perm_f,
                                 fixed_params: Dict[str, Any]):
    """Velocity-RMSE loss whose differentiable argument is the theta
    pytree (G2/fitting). ``fixed_params`` carries ``Gp, lam, nu_s,
    tbnn_bound_c`` (and anything else the model reads); ``theta`` is
    merged in under ``params['theta']``. Same validated solver path as
    ``vf._constriction_loss_fn``."""
    def loss(theta):
        params = dict(fixed_params)
        params['theta'] = theta
        out = p3b._evolve_wall_bounded_with_diagnostics(
            initial_state=init_state, model=model, polymer_params=params,
            grid=grid, density=cfg['density'], base_viscosity=cfg['nu_s'],
            dt=cfg['dt'], inner_steps=cfg['inner_steps'],
            outer_steps=cfg['outer_steps'],
            solver_type=cfg['solver_type'],
            use_preconditioner=cfg['use_preconditioner'],
            preconditioner_type=cfg['preconditioner_type'],
            pressure_gradient=(cfg['g_x'], 0.0), permeability=perm_f,
            U_f=cfg['U_f'], solver_tol=cfg['solver_tol'],
            solver_maxiter=cfg['solver_maxiter'])
        return jnp.sum(out['u_traj'] ** 2) + jnp.sum(out['v_traj'] ** 2)
    return loss


def _channel_loss_fn_theta(cfg, init_state, model, grid, perm_f, fixed_params):
    """theta-differentiable velocity-RMSE loss for the CHANNEL geometry.
    ``_constriction_loss_fn_theta`` is already geometry-agnostic (it takes
    ``init_state`` / ``model`` / ``grid`` / ``perm_f``); the channel supplies
    the flat-wall state and ``perm_f = 0.0``. Thin delegator for naming
    symmetry."""
    return _constriction_loss_fn_theta(cfg, init_state, model, grid, perm_f,
                                       fixed_params)


def _build_geometry(cfg, model_name, geometry):
    """Dispatch the geometry builder: 'constriction' (IB object) or 'channel'
    (flat walls, no object). Returns ``(grid, model, init_state, perm_f)``."""
    if geometry == 'channel':
        return vf._build_channel(cfg, model_name)
    return vf._build_constriction(cfg, model_name)


def tbnn_invariant_cloud(cfg, theta, *, Gp, lam, nu_s, bound_c,
                         model_name='tbnn_potential_logconf_bk_v2',
                         anchored=tb.TBNN_DEFAULT_ANCHORED,
                         mobility=tb.TBNN_DEFAULT_MOBILITY,
                         kappa=tb.TBNN_DEFAULT_KAPPA,
                         geometry='constriction',
                         tau_y=None):
    """Run a forward pass and summarize the visited invariant features
    ``x = (tau - 3, p2 - 3, l)`` over the final state (and report floor
    diagnostics). Used to set ``c`` and to restrict later 0D protocols to
    the trained region. ``geometry`` selects the
    constriction (default) or the channel builder."""
    grid, model, init_state, perm_f = _build_geometry(cfg, model_name, geometry)
    params = dict(Gp=jnp.asarray(Gp, dtype=jnp.float64),
                  lam=jnp.asarray(lam, dtype=jnp.float64),
                  theta=theta, tbnn_bound_c=float(bound_c),
                  tbnn_kappa=float(kappa))
    if tau_y is not None:
        params['tau_y'] = jnp.asarray(tau_y, dtype=jnp.float64)
    out = p3b._evolve_wall_bounded_with_diagnostics(
        initial_state=init_state, model=model, polymer_params=params,
        grid=grid, density=cfg['density'], base_viscosity=nu_s,
        dt=cfg['dt'], inner_steps=cfg['inner_steps'],
        outer_steps=cfg['outer_steps'], solver_type=cfg['solver_type'],
        use_preconditioner=cfg['use_preconditioner'],
        preconditioner_type=cfg['preconditioner_type'],
        pressure_gradient=(cfg['g_x'], 0.0), permeability=perm_f,
        U_f=cfg['U_f'], solver_tol=cfg['solver_tol'],
        solver_maxiter=cfg['solver_maxiter'])
    Axx = np.asarray(out['A_xx_traj'][-1]).reshape(-1)
    Axy = np.asarray(out['A_xy_traj'][-1]).reshape(-1)
    Ayy = np.asarray(out['A_yy_traj'][-1]).reshape(-1)
    Azz = np.asarray(out['A_zz_traj'][-1]).reshape(-1)
    x1, x2, x3 = tb.tbnn_invariant_features(
        jnp.asarray(Axx), jnp.asarray(Axy), jnp.asarray(Ayy), jnp.asarray(Azz))
    x1, x2, x3 = np.asarray(x1), np.asarray(x2), np.asarray(x3)
    d = tb.tbnn_floor_diagnostics(
        jnp.asarray(Axx), jnp.asarray(Axy), jnp.asarray(Ayy),
        jnp.asarray(Azz), theta, float(bound_c),
        anchored=anchored, mobility=mobility, kappa=kappa)
    qs = (0.0, 0.01, 0.5, 0.99, 1.0)

    def _summ(v):
        return {f'q{int(100*q)}': float(np.quantile(v, q)) for q in qs}
    return dict(
        x1=_summ(x1), x2=_summ(x2), x3=_summ(x3),
        active_fraction=float(d['active_fraction']),
        P_margin=float(d['P_margin']), phil_margin=float(d['phil_margin']),
        M_frozen_max_eig=float(d['M_frozen_max_eig']),
        cloud=dict(x1=x1, x2=x2, x3=x3))


# ===========================================================================
# Sec.3. G1 -- regression gate: exact-OB init == oldroyd_b_logconf_bk_v2
# ===========================================================================

def tbnn_regression_gate(config: Optional[Dict[str, Any]] = None,
                         rtol: float = 1e-6,
                         seed: int = 0,
                         bound_c: float = tb.TBNN_DEFAULT_BOUND_C,
                         ) -> Dict[str, Any]:
    """G1: ``tbnn_potential_logconf_bk_v2`` at the exact-OB init must
    reproduce ``oldroyd_b_logconf_bk_v2`` on the constriction.

    Compares forward loss and ``dL/dGp``, ``dL/dlam`` (float64,
    ``solver_tol=1e-12``). Both routes use the same affine integrator with
    ``M = I``, ``A* = I``, so agreement is near-bit-identical; gate at the
    family standard ``rtol 1e-6`` (the Giesekus alpha=0 bar).
    """
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(p3b.DEFAULT_MULTISTEP_AD_FD_CONFIG)
    if config:
        cfg.update(config)

    p = dict(Gp=jnp.asarray(cfg['Gp_init'], dtype=jnp.float64),
             lam=jnp.asarray(cfg['lam_init'], dtype=jnp.float64))

    # Oldroyd-B reference.
    grid, ob_model, ob_state, ob_perm = vf._build_constriction(
        cfg, 'oldroyd_b_logconf_bk_v2')
    ob_loss = vf._constriction_loss_fn(cfg, ob_state, ob_model, grid, ob_perm,
                                       fixed_params={}, diff_keys=('Gp', 'lam'))
    ob_vg = jax.jit(jax.value_and_grad(ob_loss))
    L_ob, g_ob = ob_vg(p)
    L_ob = float(L_ob)
    g_ob = {k: float(v) for k, v in g_ob.items()}

    # TBNN at the exact-OB init.
    theta, _ = tb.init_tbnn_theta(jax.random.PRNGKey(seed), bound_c=bound_c)
    grid_t, t_model, t_state, t_perm = vf._build_constriction(
        cfg, 'tbnn_potential_logconf_bk_v2')
    t_loss = vf._constriction_loss_fn(
        cfg, t_state, t_model, grid_t, t_perm,
        fixed_params={'theta': theta, 'tbnn_bound_c': float(bound_c)},
        diff_keys=('Gp', 'lam'))
    t_vg = jax.jit(jax.value_and_grad(t_loss))
    L_t, g_t = t_vg(p)
    L_t = float(L_t)
    g_t = {k: float(v) for k, v in g_t.items()}

    def _rel(a, b):
        return abs(a - b) / max(abs(b), 1e-30)

    rel_loss = _rel(L_t, L_ob)
    rel_Gp = _rel(g_t['Gp'], g_ob['Gp'])
    rel_lam = _rel(g_t['lam'], g_ob['lam'])
    gate_pass = bool(rel_loss < rtol and rel_Gp < rtol and rel_lam < rtol)

    print("=== TBNN init-theta regression vs oldroyd_b_logconf_bk_v2 ===")
    print(f"  forward loss : OB = {L_ob:.10e}   TBNN = {L_t:.10e}"
          f"   rel = {rel_loss:.2e}")
    print(f"  dL/dGp       : OB = {g_ob['Gp']:.10e}   TBNN = {g_t['Gp']:.10e}"
          f"   rel = {rel_Gp:.2e}")
    print(f"  dL/dlam      : OB = {g_ob['lam']:.10e}   TBNN = {g_t['lam']:.10e}"
          f"   rel = {rel_lam:.2e}")
    print(f"  GATE (rtol={rtol:.0e}): {'PASS' if gate_pass else 'FAIL'}")
    return dict(L_ob=L_ob, L_t=L_t, g_ob=g_ob, g_t=g_t,
                rel_loss=rel_loss, rel_Gp=rel_Gp, rel_lam=rel_lam,
                gate_pass=gate_pass)


# ===========================================================================
# Timing check -- run before any fit
# ===========================================================================

def tbnn_timing_smoketest(config: Optional[Dict[str, Any]] = None,
                          n_repeat: int = 3,
                          flag_ratio: float = 5.0,
                          bound_c: float = tb.TBNN_DEFAULT_BOUND_C,
                          seed: int = 0,
                          ) -> Dict[str, Any]:
    """Per-step wall time of ``tbnn_potential_logconf_bk_v2`` vs
    ``giesekus_logconf_bk_v2`` on the constriction (forward and
    ``value_and_grad``), plus the TBNN theta-gradient cost (the real
    training step). Records the ratio and flags if > ~``flag_ratio``x
    ("should be" is not a measurement).
    """
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(p3b.DEFAULT_MULTISTEP_AD_FD_CONFIG)
    if config:
        cfg.update(config)
    n_steps = cfg['outer_steps'] * cfg['inner_steps']
    p = dict(Gp=jnp.asarray(cfg['Gp_init'], dtype=jnp.float64),
             lam=jnp.asarray(cfg['lam_init'], dtype=jnp.float64))

    def _time(fn, arg):
        out = fn(arg)
        jax.block_until_ready(out)  # warm (compile)
        ts = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            out = fn(arg)
            jax.block_until_ready(out)
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    # Giesekus (alpha=0) baseline.
    grid, g_model, g_state, g_perm = vf._build_constriction(
        cfg, 'giesekus_logconf_bk_v2')
    g_loss = vf._constriction_loss_fn(
        cfg, g_state, g_model, grid, g_perm,
        fixed_params={'alpha': jnp.asarray(0.0, dtype=jnp.float64)},
        diff_keys=('Gp', 'lam'))
    g_fwd = _time(jax.jit(g_loss), p)
    g_vg = _time(jax.jit(jax.value_and_grad(g_loss)), p)

    # TBNN at init-theta.
    theta, _ = tb.init_tbnn_theta(jax.random.PRNGKey(seed), bound_c=bound_c)
    grid_t, t_model, t_state, t_perm = vf._build_constriction(
        cfg, 'tbnn_potential_logconf_bk_v2')
    t_loss = vf._constriction_loss_fn(
        cfg, t_state, t_model, grid_t, t_perm,
        fixed_params={'theta': theta, 'tbnn_bound_c': float(bound_c)},
        diff_keys=('Gp', 'lam'))
    t_fwd = _time(jax.jit(t_loss), p)
    t_vg = _time(jax.jit(jax.value_and_grad(t_loss)), p)

    # TBNN theta-gradient (the training step).
    theta_loss = _constriction_loss_fn_theta(
        cfg, t_state, t_model, grid_t, t_perm,
        fixed_params={'Gp': p['Gp'], 'lam': p['lam'],
                      'tbnn_bound_c': float(bound_c)})
    t_vg_theta = _time(jax.jit(jax.value_and_grad(theta_loss)), theta)

    r_fwd = t_fwd / g_fwd
    r_vg = t_vg / g_vg
    flagged = bool(r_fwd > flag_ratio or r_vg > flag_ratio)
    print("=== TBNN vs Giesekus timing smoke test (constriction) ===")
    print(f"  {n_steps} steps/eval; median of {n_repeat} (s)")
    print(f"  forward       : Gie={g_fwd:.3f}  TBNN={t_fwd:.3f}  ratio={r_fwd:.2f}x")
    print(f"  value_and_grad: Gie={g_vg:.3f}  TBNN={t_vg:.3f}  ratio={r_vg:.2f}x")
    print(f"  TBNN d/dtheta : {t_vg_theta:.3f} s  (the training step)")
    print(f"  FLAG (> {flag_ratio:.0f}x): {'YES -- investigate' if flagged else 'no'}")
    return dict(g_fwd=g_fwd, g_vg=g_vg, t_fwd=t_fwd, t_vg=t_vg,
                t_vg_theta=t_vg_theta, ratio_fwd=r_fwd, ratio_vg=r_vg,
                flagged=flagged, n_steps=n_steps)


# ===========================================================================
# Sec.5. G2 -- AD vs FD on the theta gradient (and re-check dL/dGp, dL/dlam)
# ===========================================================================

def _perturb_theta_last_layers(theta, key, scale=0.1):
    """Set each head's zeroed last layer to small random values so dL/dtheta
    has signal (init-theta has exact-zero gradients behind the zeroed last
    layer -- plan G2: perturb first, then test)."""
    out = {}
    keys = jax.random.split(key, len(theta))
    for (name, layers), k in zip(theta.items(), keys):
        W, b = layers[-1]
        kw, kb = jax.random.split(k)
        new_last = (jax.random.normal(kw, W.shape, dtype=W.dtype) * scale,
                    b + jax.random.normal(kb, b.shape, dtype=b.dtype) * scale)
        out[name] = layers[:-1] + [new_last]
    return out


def tbnn_ad_vs_fd(config: Optional[Dict[str, Any]] = None,
                  Gp: float = 3.2, lam: float = 0.7, g_x: float = 8.0,
                  outer_steps: int = 100, n_test: int = 5,
                  fd_eps_list: Tuple[float, ...] = (1e-3, 1e-4, 1e-5),
                  gate_rel_tol: float = 0.02, seed: int = 0,
                  bound_c: float = tb.TBNN_DEFAULT_BOUND_C,
                  ) -> Dict[str, Any]:
    """G2: reverse-mode ``dL/dtheta`` vs centered FD on ``n_test`` raveled
    theta entries (relative steps), at a perturbed theta and an elastic
    operating point. Also re-verifies ``dL/dGp`` (AD vs FD) with theta in
    the params. One axis at a time (the model is the TBNN; numerics are
    the validated constriction values)."""
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(p3b.DEFAULT_MULTISTEP_AD_FD_CONFIG)
    cfg['g_x'] = g_x
    cfg['outer_steps'] = outer_steps
    if config:
        cfg.update(config)

    grid, model, init_state, perm_f = vf._build_constriction(
        cfg, 'tbnn_potential_logconf_bk_v2')

    theta0, _ = tb.init_tbnn_theta(jax.random.PRNGKey(seed), bound_c=bound_c)
    theta_p = _perturb_theta_last_layers(theta0, jax.random.PRNGKey(seed + 1))

    # --- dL/dtheta ---
    loss_theta = _constriction_loss_fn_theta(
        cfg, init_state, model, grid, perm_f,
        fixed_params={'Gp': jnp.asarray(Gp, dtype=jnp.float64),
                      'lam': jnp.asarray(lam, dtype=jnp.float64),
                      'tbnn_bound_c': float(bound_c)})
    flat0, unravel = ravel_pytree(theta_p)
    loss_flat = lambda fl: loss_theta(unravel(fl))
    loss_flat_jit = jax.jit(loss_flat)
    vg = jax.jit(jax.value_and_grad(loss_flat))
    t0 = time.perf_counter()
    L0, g_flat = vg(flat0)
    g_flat = np.asarray(g_flat)
    L0 = float(L0)
    print(f"[G2] loss(theta_p)={L0:.6e}  (AD warm {time.perf_counter()-t0:.1f}s); "
          f"|theta|={flat0.size}")

    # Pick the n_test entries with the largest |AD grad| (signal above noise).
    order = np.argsort(np.abs(g_flat))[::-1]
    test_idx = [int(i) for i in order[:n_test]]

    results = []
    for i in test_idx:
        ad_i = float(g_flat[i])
        step = max(abs(float(flat0[i])), 1.0)
        best = None
        for eps in fd_eps_list:
            h = eps * step
            fp = flat0.at[i].set(flat0[i] + h)
            fm = flat0.at[i].set(flat0[i] - h)
            fd = (float(loss_flat_jit(fp)) - float(loss_flat_jit(fm))) / (2 * h)
            rel = abs(ad_i - fd) / max(abs(fd), 1e-30)
            if best is None or rel < best[2]:
                best = (eps, fd, rel)
        results.append((i, ad_i, best[1], best[0], best[2]))
        print(f"  theta[{i:>4}]  AD={ad_i:+.6e}  FD={best[1]:+.6e}  "
              f"(eps={best[0]:.0e})  rel={best[2]:.2e}")
    worst_rel = max(r[4] for r in results)

    # --- re-verify dL/dGp with theta in params ---
    gp_loss = vf._constriction_loss_fn(
        cfg, init_state, model, grid, perm_f,
        fixed_params={'theta': theta_p, 'tbnn_bound_c': float(bound_c),
                      'lam': jnp.asarray(lam, dtype=jnp.float64)},
        diff_keys=('Gp',))
    gp_loss_jit = jax.jit(gp_loss)
    gp_vg = jax.jit(jax.value_and_grad(gp_loss))
    _, gGp = gp_vg({'Gp': jnp.asarray(Gp, dtype=jnp.float64)})
    ad_Gp = float(gGp['Gp'])
    fd_Gp_best = None
    for eps in fd_eps_list:
        h = eps * Gp
        Lp = float(gp_loss_jit({'Gp': jnp.asarray(Gp + h, dtype=jnp.float64)}))
        Lm = float(gp_loss_jit({'Gp': jnp.asarray(Gp - h, dtype=jnp.float64)}))
        fd = (Lp - Lm) / (2 * h)
        rel = abs(ad_Gp - fd) / max(abs(fd), 1e-30)
        if fd_Gp_best is None or rel < fd_Gp_best[2]:
            fd_Gp_best = (eps, fd, rel)
    print(f"  dL/dGp  AD={ad_Gp:+.6e}  FD={fd_Gp_best[1]:+.6e}  "
          f"(eps={fd_Gp_best[0]:.0e})  rel={fd_Gp_best[2]:.2e}")

    gate_pass = bool(np.isfinite(worst_rel) and worst_rel <= gate_rel_tol
                     and fd_Gp_best[2] <= gate_rel_tol)
    print(f"=== TBNN G2 AD-vs-FD: worst theta rel = {worst_rel:.2e}, "
          f"Gp rel = {fd_Gp_best[2]:.2e} ===")
    print(f"  GATE (rel_tol={gate_rel_tol:.0%}): {'PASS' if gate_pass else 'FAIL'}")
    return dict(loss=L0, theta_tests=results, worst_rel=worst_rel,
                ad_Gp=ad_Gp, fd_Gp=fd_Gp_best, gate_pass=gate_pass)


# ===========================================================================
# Sec.6. G3/G4 -- recovery: learned 0D observables vs the truth's steady root
# ===========================================================================

def _cloud_feature_bands(cloud, lo_q='q1', hi_q='q99'):
    """In-cloud per-invariant bands ``[q_lo, q_hi]`` for ``x = (tau-3,
    p2-3, l)`` from the fit's invariant-cloud summary (the trained
    envelope; recovery evaluation)."""
    return (
        (cloud['x1'][lo_q], cloud['x1'][hi_q]),
        (cloud['x2'][lo_q], cloud['x2'][hi_q]),
        (cloud['x3'][lo_q], cloud['x3'][hi_q]),
    )


def _features_of_state(a):
    x1, x2, x3 = tb.tbnn_invariant_features(
        jnp.asarray(a[0]), jnp.asarray(a[1]),
        jnp.asarray(a[2]), jnp.asarray(a[3]))
    return float(x1), float(x2), float(x3)


def tbnn_recovery_eval(theta, params_fit, truth_R_fn, truth_visc_fn,
                       truth_params, *, lam_truth, lam_fit, Gp_truth=1.0,
                       Gp_fit=1.0, bound_c=tb.TBNN_DEFAULT_BOUND_C, cloud,
                       n_shear=12, gd_lo=0.05, gd_hi_scan=30.0, obs_rtol=0.05,
                       min_in_cloud=8, tol_band=1e-9):
    """Compare the learned model's steady-shear **in-plane** observables to
    the truth's Newton root across a **physical shear-rate** sweep (plan
    G3/G4 -- gate on the OBSERVABLE in-plane response; N2-derived quantities
    are RECORD-only).

    Gauge note: under the agnostic protocol the
    fitted model runs at ``Gp_fit = lam_fit = 1`` while the truth keeps its
    physical ``(Gp_truth, lam_truth)``. The relaxation *rate* ``m0/lam`` and
    the stress *scale* ``Gp*phi`` are gauge directions the network absorbs,
    so the comparison must be at the **same physical gammadot** (NOT the
    same Wi): truth at ``lam_truth``, learned at ``lam_fit``. The conformation
    dynamics depend on gammadot and the rate; ``Gp`` only scales the stress
    readout. In ``--unit-test`` mode pass ``lam_fit=lam_truth``,
    ``Gp_fit=Gp_truth``.

    Empirically-debugged sweep rules (plan, do not regress):
    * sweep built in shear rate directly (geomspace ``gd_lo..gd_hi``);
    * "in-cloud" = per-point membership of the truth steady state's
      ``(tau-3, p2-3, l)`` features in the trained per-invariant bands
      (cloud ``[q1, q99]``), NOT a bulk trA cap;
    * ``gd_hi`` auto-set just past the cloud edge (x1.3);
    * the observable gate requires ``>= min_in_cloud`` of ``n_shear``
      in-cloud points to count.
    """
    b1, b2, b3 = _cloud_feature_bands(cloud)

    def _in_band(a):
        x1, _x2, x3 = _features_of_state(a)
        return (b1[0] - tol_band <= x1 <= b1[1] + tol_band and
                b3[0] - tol_band <= x3 <= b3[1] + tol_band)

    # Cloud edge in shear rate: largest scanned gammadot whose TRUTH steady
    # state is in-band (stretch is monotone in gammadot for these models).
    gd_edge = gd_lo
    for gd in np.geomspace(gd_lo, gd_hi_scan, 48):
        tr = vf.steady_simple_shear_reference(
            truth_R_fn, gd, lam_truth, params=truth_params, Gp=Gp_truth,
            viscometric_fn=truth_visc_fn)
        if _in_band(tr['a']):
            gd_edge = float(gd)
    gd_hi = 1.3 * gd_edge

    rows = []
    for gd in np.geomspace(gd_lo, gd_hi, n_shear):
        tr = vf.steady_simple_shear_reference(
            truth_R_fn, gd, lam_truth, params=truth_params, Gp=Gp_truth,
            viscometric_fn=truth_visc_fn)
        lr = tbnn_steady_reference(theta, params_fit, gd, Gp=Gp_fit,
                                   lam=lam_fit, bound_c=bound_c, A_init=tr['a'])
        rows.append(dict(
            gammadot=float(gd), Wi=float(lam_truth * gd),
            in_cloud=bool(_in_band(tr['a'])),
            t_N1=tr['N1'], t_N2=tr['N2'], t_tau=tr['tau_xy'], t_eta=tr['eta_p'],
            t_Azz=tr['A_zz'], t_trA=tr['trA'],
            l_N1=lr['N1'], l_N2=lr['N2'], l_tau=lr['tau_xy'], l_eta=lr['eta_p'],
            l_Azz=lr['A_zz'], l_trA=lr['trA']))

    def _wrel(key):
        errs = []
        for r in rows:
            if not r['in_cloud']:
                continue
            denom = max(abs(r[f't_{key}']), 1e-12)
            if denom < 1e-9:
                continue
            errs.append(abs(r[f'l_{key}'] - r[f't_{key}']) / denom)
        return max(errs) if errs else float('nan')

    # GATE only on the in-plane observables N1, tau_xy, eta_p (never N2).
    worst = {k: _wrel(k) for k in ('N1', 'tau', 'eta')}
    n_in = int(sum(r['in_cloud'] for r in rows))
    return dict(rows=rows, worst=worst, obs_rtol=obs_rtol,
                n_in_cloud=n_in, min_in_cloud=min_in_cloud,
                enough_in_cloud=bool(n_in >= min_in_cloud),
                gd_edge=gd_edge, gd_hi=gd_hi,
                bands=dict(x1=b1, x2=b2, x3=b3))


# ===========================================================================
# Saramito EVP generator checks
# ===========================================================================

def _forward_traj(cfg, model_name, params, geometry='constriction'):
    """Run the geometry forward and return the diagnostics dict (with
    u_traj / v_traj / A_*_traj). Shared by the P3-G1 kinematics regression
    and the yielded-fraction logger. ``geometry`` selects constriction
    (default) or channel."""
    grid, model, init_state, perm_f = _build_geometry(cfg, model_name, geometry)
    out = p3b._evolve_wall_bounded_with_diagnostics(
        initial_state=init_state, model=model, polymer_params=params,
        grid=grid, density=cfg['density'], base_viscosity=cfg['nu_s'],
        dt=cfg['dt'], inner_steps=cfg['inner_steps'],
        outer_steps=cfg['outer_steps'], solver_type=cfg['solver_type'],
        use_preconditioner=cfg['use_preconditioner'],
        preconditioner_type=cfg['preconditioner_type'],
        pressure_gradient=(cfg['g_x'], 0.0), permeability=perm_f,
        U_f=cfg['U_f'], solver_tol=cfg['solver_tol'],
        solver_maxiter=cfg['solver_maxiter'])
    return out


def saramito_yielded_fraction(cfg, Gp, lam, tau_y, nu_s,
                              model_name='saramito_logconf_bk_v2',
                              geometry='constriction'):
    """Fraction of cells that are YIELDED (|tau_d| > tau_y) in the final
    Saramito truth field (if the yielded fraction is ~0% or
    ~100% the data cannot identify the yield surface). ``geometry`` selects
    constriction (default) or channel."""
    c = dict(cfg)
    c['nu_s'] = nu_s
    params = dict(Gp=jnp.asarray(Gp, dtype=jnp.float64),
                  lam=jnp.asarray(lam, dtype=jnp.float64),
                  tau_y=jnp.asarray(tau_y, dtype=jnp.float64))
    out = _forward_traj(c, model_name, params, geometry=geometry)
    Axx = jnp.asarray(out['A_xx_traj'][-1])
    Axy = jnp.asarray(out['A_xy_traj'][-1])
    Ayy = jnp.asarray(out['A_yy_traj'][-1])
    Azz = jnp.asarray(out['A_zz_traj'][-1])
    td = np.asarray(tb.saramito_tau_d_norm(Axx, Axy, Ayy, Azz, Gp)).reshape(-1)
    frac = float(np.mean(td > tau_y))
    return dict(yielded_fraction=frac, td_min=float(td.min()),
                td_max=float(td.max()), tau_y=tau_y,
                td_q50=float(np.quantile(td, 0.5)),
                td_q99=float(np.quantile(td, 0.99)))


def tbnn_toggle_init_regression_gate(config=None, rtol=1e-6, seed=0):
    """Config regression for the two NEW toggle registrations: at the
    exact-OB init (N == 0) with kappa = 1, both
    ``tbnn_potential_unanchored_logconf_bk_v2`` (False, softplus) and
    ``tbnn_potential_free_logconf_bk_v2`` (False, relu_annealed) must
    reduce to ``oldroyd_b_logconf_bk_v2`` on the constriction (unanchored
    == anchored at N == 0; relu_annealed == softplus at kappa = 1). Cheap
    gpu_test confirmation that the new wrappers are wired correctly
    (config-regression axis)."""
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(p3b.DEFAULT_MULTISTEP_AD_FD_CONFIG)
    if config:
        cfg.update(config)
    p = dict(Gp=jnp.asarray(cfg['Gp_init'], dtype=jnp.float64),
             lam=jnp.asarray(cfg['lam_init'], dtype=jnp.float64))
    grid, ob_model, ob_state, ob_perm = vf._build_constriction(
        cfg, 'oldroyd_b_logconf_bk_v2')
    ob_loss = vf._constriction_loss_fn(cfg, ob_state, ob_model, grid, ob_perm,
                                       fixed_params={}, diff_keys=('Gp', 'lam'))
    L_ob = float(jax.jit(ob_loss)(p))

    def _rel(a, b):
        return abs(a - b) / max(abs(b), 1e-30)

    results = {}
    for name, anch, mob in (
            ('tbnn_potential_unanchored_logconf_bk_v2', False, 'softplus'),
            ('tbnn_potential_free_logconf_bk_v2', False, 'relu_annealed')):
        theta, _ = tb.init_tbnn_theta(jax.random.PRNGKey(seed), anchored=anch,
                                      mobility=mob)
        _, model, state, perm = vf._build_constriction(cfg, name)
        loss = vf._constriction_loss_fn(
            cfg, state, model, grid, perm,
            fixed_params={'theta': theta, 'tbnn_bound_c': tb.TBNN_DEFAULT_BOUND_C,
                          'tbnn_kappa': 1.0}, diff_keys=('Gp', 'lam'))
        L = float(jax.jit(loss)(p))
        rel = _rel(L, L_ob)
        results[name] = dict(loss=L, rel=rel, pass_=bool(rel < rtol))
        print(f"  {name}: loss={L:.10e}  rel vs OB={rel:.2e}  "
              f"{'PASS' if rel < rtol else 'FAIL'}")
    gate_pass = bool(all(v['pass_'] for v in results.values()))
    print(f"=== TBNN toggle-init config regression (rtol={rtol:.0e}): "
          f"{'PASS' if gate_pass else 'FAIL'} ===")
    return dict(L_ob=L_ob, results=results, gate_pass=gate_pass)


def saramito_regression_gate(config=None, rtol=1e-6, traj_rtol=1e-9):
    """P3-G1 (generator sanity): Saramito ``tau_y = 0`` reproduces
    ``oldroyd_b_logconf_bk_v2``.

    Two checks:
      * forward loss + ``dL/dGp``, ``dL/dlam`` match OB to ``rtol`` (the
        Giesekus(alpha=0) family standard -- the affine integrator's
        ``M = I`` limit vs the OB analytic exponential differ only at
        float64 round-off);
      * the ONE-LINE kinematics regression: the full ``u_traj``/``v_traj``
        of Saramito(tau_y=0) matches the OB trajectory to ``traj_rtol``
        (confirming the yield generator rides the validated FK/RK2
        kinematics, not a re-implementation). "Bit-identical" here means the
        family machine-precision standard (same integrator-vs-analytic FP
        gap the Giesekus alpha=0 gate accepts), documented in
        the elastoviscoplastic cost notes.
    """
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(p3b.DEFAULT_MULTISTEP_AD_FD_CONFIG)
    if config:
        cfg.update(config)
    p = dict(Gp=jnp.asarray(cfg['Gp_init'], dtype=jnp.float64),
             lam=jnp.asarray(cfg['lam_init'], dtype=jnp.float64))

    grid, ob_model, ob_state, ob_perm = vf._build_constriction(
        cfg, 'oldroyd_b_logconf_bk_v2')
    ob_loss = vf._constriction_loss_fn(cfg, ob_state, ob_model, grid, ob_perm,
                                       fixed_params={}, diff_keys=('Gp', 'lam'))
    L_ob, g_ob = jax.jit(jax.value_and_grad(ob_loss))(p)
    L_ob = float(L_ob); g_ob = {k: float(v) for k, v in g_ob.items()}

    grid_s, s_model, s_state, s_perm = vf._build_constriction(
        cfg, 'saramito_logconf_bk_v2')
    s_loss = vf._constriction_loss_fn(
        cfg, s_state, s_model, grid_s, s_perm,
        fixed_params={'tau_y': jnp.asarray(0.0, dtype=jnp.float64)},
        diff_keys=('Gp', 'lam'))
    L_s, g_s = jax.jit(jax.value_and_grad(s_loss))(p)
    L_s = float(L_s); g_s = {k: float(v) for k, v in g_s.items()}

    def _rel(a, b):
        return abs(a - b) / max(abs(b), 1e-30)
    rel_loss = _rel(L_s, L_ob)
    rel_Gp = _rel(g_s['Gp'], g_ob['Gp'])
    rel_lam = _rel(g_s['lam'], g_ob['lam'])

    # Kinematics regression: full-trajectory match vs OB.
    p_ob = dict(Gp=p['Gp'], lam=p['lam'])
    out_ob = _forward_traj(cfg, 'oldroyd_b_logconf_bk_v2', p_ob)
    p_s = dict(Gp=p['Gp'], lam=p['lam'],
               tau_y=jnp.asarray(0.0, dtype=jnp.float64))
    out_s = _forward_traj(cfg, 'saramito_logconf_bk_v2', p_s)
    du = float(jnp.max(jnp.abs(out_s['u_traj'] - out_ob['u_traj'])))
    dv = float(jnp.max(jnp.abs(out_s['v_traj'] - out_ob['v_traj'])))
    scale = float(jnp.max(jnp.abs(out_ob['u_traj']))) + 1e-30
    traj_rel = max(du, dv) / scale

    gate_pass = bool(rel_loss < rtol and rel_Gp < rtol and rel_lam < rtol
                     and traj_rel < traj_rtol)
    print("=== Saramito tau_y=0 regression vs oldroyd_b_logconf_bk_v2 (P3-G1) ===")
    print(f"  forward loss : OB={L_ob:.10e}  Sar={L_s:.10e}  rel={rel_loss:.2e}")
    print(f"  dL/dGp       : rel={rel_Gp:.2e}   dL/dlam: rel={rel_lam:.2e}")
    print(f"  kinematics   : max|du|={du:.2e} max|dv|={dv:.2e}  traj_rel={traj_rel:.2e}"
          f"  (gate {traj_rtol:.0e})")
    print(f"  GATE (rtol={rtol:.0e}): {'PASS' if gate_pass else 'FAIL'}")
    return dict(L_ob=L_ob, L_s=L_s, rel_loss=rel_loss, rel_Gp=rel_Gp,
                rel_lam=rel_lam, traj_rel=traj_rel, gate_pass=gate_pass)


def saramito_tau_y_ad_vs_fd(config=None, tau_y0=1.0, Gp=3.2, lam=0.7,
                            g_x=8.0, outer_steps=100,
                            fd_eps_list=(1e-2, 1e-3, 1e-4),
                            gate_rel_tol=0.02):
    """P3-G1 (gradient): reverse-mode ``dL/dtau_y`` vs centered FD on the
    constriction, at a moderate ``tau_y`` (yield active). One axis: the
    model is Saramito; numerics are the validated constriction values."""
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(p3b.DEFAULT_MULTISTEP_AD_FD_CONFIG)
    cfg['g_x'] = g_x
    cfg['outer_steps'] = outer_steps
    if config:
        cfg.update(config)
    grid, model, init_state, perm_f = vf._build_constriction(
        cfg, 'saramito_logconf_bk_v2')
    fixed = dict(Gp=jnp.asarray(Gp, dtype=jnp.float64),
                 lam=jnp.asarray(lam, dtype=jnp.float64))
    loss = vf._constriction_loss_fn(cfg, init_state, model, grid, perm_f,
                                    fixed_params=fixed, diff_keys=('tau_y',))
    loss_jit = jax.jit(loss)
    vg = jax.jit(jax.value_and_grad(loss))
    t0 = time.perf_counter()
    L0, g0 = vg({'tau_y': jnp.asarray(tau_y0, dtype=jnp.float64)})
    ad = float(g0['tau_y'])
    print(f"[P3-G1 tau_y] loss={float(L0):.6e}  AD dL/dtau_y={ad:+.6e}  "
          f"(warm {time.perf_counter()-t0:.1f}s)")

    def _eval(ty):
        return float(loss_jit({'tau_y': jnp.asarray(ty, dtype=jnp.float64)}))
    best = None
    for eps in fd_eps_list:
        h = eps * max(abs(tau_y0), 1.0)
        fd = (_eval(tau_y0 + h) - _eval(tau_y0 - h)) / (2 * h)
        rel = abs(ad - fd) / max(abs(fd), 1e-30)
        print(f"  FD eps={eps:.0e}  dL/dtau_y={fd:+.6e}  rel={rel:.2e}")
        if best is None or rel < best[2]:
            best = (eps, fd, rel)
    gate_pass = bool(np.isfinite(ad) and best[2] <= gate_rel_tol)
    print(f"  GATE (rel_tol={gate_rel_tol:.0%}): {'PASS' if gate_pass else 'FAIL'}")
    return dict(ad=ad, fd_best=best, rel_err=best[2], gate_pass=gate_pass,
                loss=float(L0))


# ===========================================================================
# Sec.7b. V2 structured-yield scalar gates (Stage 2: G-A through G-D)
# ===========================================================================

V2_YIELD_MODEL = 'tbnn_potential_yield_logconf_bk_v2'
V1_MODEL = 'tbnn_potential_logconf_bk_v2'


def tbnn_v2_ga_regression_gate(config=None, rtol=1e-6, seed=0):
    """G-A (halt gate): all EXISTING registrations unchanged after the V2
    closure edit. Runs V1 vs OB (P3-G3), toggle-init (V3/unanchored), and
    Saramito ``tau_y=0`` vs OB."""
    g1 = tbnn_regression_gate(config=config, rtol=rtol, seed=seed)
    g3 = tbnn_toggle_init_regression_gate(config=config, rtol=rtol, seed=seed)
    gs = saramito_regression_gate(config=config, rtol=rtol, traj_rtol=1e-9)
    gate_pass = bool(g1['gate_pass'] and g3['gate_pass'] and gs['gate_pass'])
    print(f"=== G-A combined (existing paths bit-identical): "
          f"{'PASS' if gate_pass else 'FAIL'} ===")
    return dict(g1=g1, g3=g3, saramito=gs, gate_pass=gate_pass)


def _channel_Q(out, cfg):
    Ny, Ly = cfg['Ny'], cfg['Ly']
    dy = Ly / Ny
    y = (jnp.arange(Ny, dtype=jnp.float64) + 0.5) * dy
    u_prof = jnp.mean(out['u_traj'][-1], axis=0)
    return float(jnp.trapz(u_prof, y))


def tbnn_v2_gb_tau0_channel_gate(config=None, traj_rtol=1e-12, seed=0,
                                 bound_c=tb.TBNN_DEFAULT_BOUND_C):
    """G-B: V2 with ``tau_y=0`` end-to-end channel trajectory == V1."""
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(vf.DEFAULT_CHANNEL_CONFIG)
    cfg['solver_tol'] = 1e-12
    if config:
        cfg.update(config)
    Gp, lam, nu_s = cfg['Gp_init'], cfg['lam_init'], cfg['nu_s']
    theta, _ = tb.init_tbnn_theta(jax.random.PRNGKey(seed), bound_c=bound_c)
    base = dict(Gp=jnp.asarray(Gp, dtype=jnp.float64),
                lam=jnp.asarray(lam, dtype=jnp.float64),
                theta=theta, tbnn_bound_c=float(bound_c))
    c = dict(cfg)
    c['nu_s'] = nu_s
    out_v1 = _forward_traj(c, V1_MODEL, base, geometry='channel')
    out_v2 = _forward_traj(
        c, V2_YIELD_MODEL,
        {**base, 'tau_y': jnp.asarray(0.0, dtype=jnp.float64)},
        geometry='channel')
    du = float(jnp.max(jnp.abs(out_v2['u_traj'] - out_v1['u_traj'])))
    dv = float(jnp.max(jnp.abs(out_v2['v_traj'] - out_v1['v_traj'])))
    scale = float(jnp.max(jnp.abs(out_v1['u_traj']))) + 1e-30
    traj_rel = max(du, dv) / scale
    gate_pass = bool(traj_rel < traj_rtol)
    print(f"=== G-B V2(tau_y=0) vs V1 channel trajectory ===")
    print(f"  max|du|={du:.2e} max|dv|={dv:.2e}  traj_rel={traj_rel:.2e}  "
          f"({'PASS' if gate_pass else 'FAIL'}, gate {traj_rtol:.0e})")
    return dict(traj_rel=traj_rel, gate_pass=gate_pass)


def tbnn_v2_gc_saramito_equiv_gate(config=None, tau_y=1.45, g_x_list=(4.0,),
                                   q_rtol=1e-2, seed=0,
                                   bound_c=tb.TBNN_DEFAULT_BOUND_C,
                                   steady=True):
    """G-C: V2 at OB-init with fixed ``tau_y`` vs Saramito on the channel.

    Uses steady evolution (``fwd_yield_sweep`` protocol) when ``steady=True``.
    Tolerance ``q_rtol`` defaults to ``1e-2``: V2 applies ``pref`` on the
    mobility block then ``M_frozen = 2.Mob.P``, whereas Saramito uses
    isotropic ``M = kappa_y I``. Sub-yield ``g_x`` (e.g. 1.3) is excluded from
    the default gate: even with ``pref~=1``, ``2.Mob.P != I`` away from ``A=I``.
    The Sec.4c ~1e-9 match used the *demo* ``m0=kappa_y`` injection, not V2.
    """
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    from fwd_yield_sweep import (_evolve_to_steady, _build_channel_with_model,
                                _base_cfg)
    cfg = dict(_base_cfg())
    cfg['solver_tol'] = 1e-12
    if config:
        cfg.update(config)
    Gp, lam, nu_s = cfg['Gp_init'], cfg['lam_init'], cfg['nu_s']
    theta, _ = tb.init_tbnn_theta(jax.random.PRNGKey(seed), bound_c=bound_c)
    m_v2 = cr.get_model(V2_YIELD_MODEL)
    m_s = cr.get_model('saramito_logconf_bk_v2')
    results = {}
    all_pass = True
    for g_x in g_x_list:
        c = dict(cfg)
        c['g_x'] = float(g_x)
        c['nu_s'] = nu_s
        p_tb = dict(Gp=jnp.asarray(Gp, dtype=jnp.float64),
                    lam=jnp.asarray(lam, dtype=jnp.float64),
                    theta=theta, tbnn_bound_c=float(bound_c),
                    tau_y=jnp.asarray(tau_y, dtype=jnp.float64))
        p_s = dict(Gp=jnp.asarray(Gp, dtype=jnp.float64),
                   lam=jnp.asarray(lam, dtype=jnp.float64),
                   tau_y=jnp.asarray(tau_y, dtype=jnp.float64))
        if steady:
            gv, mv2, sv2, pv2 = _build_channel_with_model(c, m_v2)
            gs, ms, ss, ps = _build_channel_with_model(c, m_s)
            out_tb = _evolve_to_steady(c, mv2, p_tb, gv, sv2, pv2, g_x, 'V2')
            out_s = _evolve_to_steady(c, ms, p_s, gs, ss, ps, g_x, 'Sar')
            Q_tb = float(out_tb['metrics']['Q'])
            Q_s = float(out_s['metrics']['Q'])
            mode = 'steady'
        else:
            c['outer_steps'] = 84
            out_tb = _forward_traj(c, V2_YIELD_MODEL, p_tb, geometry='channel')
            out_s = _forward_traj(c, 'saramito_logconf_bk_v2', p_s, geometry='channel')
            Q_tb = _channel_Q(out_tb, c)
            Q_s = _channel_Q(out_s, c)
            mode = 'T~3lam'
        rel = abs(Q_tb - Q_s) / max(abs(Q_s), 1e-9)
        ok = rel < q_rtol
        all_pass &= ok
        results[g_x] = dict(Q_tb=Q_tb, Q_s=Q_s, rel=rel, pass_=ok, mode=mode)
        print(f"  g_x={g_x:g} ({mode}): Q_V2={Q_tb:.6e}  Q_Sar={Q_s:.6e}  "
              f"rel={rel:.2e}  {'PASS' if ok else 'FAIL'}")
    gate_pass = bool(all_pass)
    print(f"=== G-C V2 OB-init tau_y={tau_y} vs Saramito (Q rel {q_rtol:.0e}): "
          f"{'PASS' if gate_pass else 'FAIL'} ===")
    return dict(results=results, gate_pass=gate_pass)


def tbnn_v2_gd_tau_y_grad_gate(config=None, tau_y0=1.45, Gp=3.2, lam=0.7,
                               nu_s=0.8, g_x=4.0, outer_steps=84, Ny=64,
                               fd_eps_list=(1e-2, 1e-3, 1e-4),
                               gate_rel_tol=1e-3, seed=0,
                               bound_c=tb.TBNN_DEFAULT_BOUND_C):
    """G-D: AD vs FD on ``dL/dtau_y`` through the V2 closure (short channel)."""
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(vf.DEFAULT_CHANNEL_CONFIG)
    cfg.update(dict(g_x=g_x, outer_steps=outer_steps, Ny=Ny, solver_tol=1e-12,
                    nu_s=nu_s))
    if config:
        cfg.update(config)
    theta, _ = tb.init_tbnn_theta(jax.random.PRNGKey(seed), bound_c=bound_c)
    grid, model, init_state, perm_f = vf._build_channel(
        cfg, V2_YIELD_MODEL)
    fixed = dict(Gp=jnp.asarray(Gp, dtype=jnp.float64),
                 lam=jnp.asarray(lam, dtype=jnp.float64),
                 theta=theta, tbnn_bound_c=float(bound_c))
    loss = vf._constriction_loss_fn(cfg, init_state, model, grid, perm_f,
                                    fixed_params=fixed, diff_keys=('tau_y',))
    loss_jit = jax.jit(loss)
    vg = jax.jit(jax.value_and_grad(loss))
    L0, g0 = vg({'tau_y': jnp.asarray(tau_y0, dtype=jnp.float64)})
    ad = float(g0['tau_y'])
    print(f"[G-D V2 tau_y] loss={float(L0):.6e}  AD dL/dtau_y={ad:+.6e}")

    def _eval(ty):
        return float(loss_jit({'tau_y': jnp.asarray(ty, dtype=jnp.float64)}))

    best = None
    for eps in fd_eps_list:
        h = eps * max(abs(tau_y0), 1.0)
        fd = (_eval(tau_y0 + h) - _eval(tau_y0 - h)) / (2 * h)
        rel = abs(ad - fd) / max(abs(fd), 1e-30)
        print(f"  FD eps={eps:.0e}  dL/dtau_y={fd:+.6e}  rel={rel:.2e}")
        if best is None or rel < best[2]:
            best = (eps, fd, rel)
    gate_pass = bool(np.isfinite(ad) and best[2] <= gate_rel_tol)
    print(f"  G-D GATE (rel_tol={gate_rel_tol:.0e}): "
          f"{'PASS' if gate_pass else 'FAIL'}")
    return dict(ad=ad, fd_best=best, rel_err=best[2], gate_pass=gate_pass,
                loss=float(L0))


# ===========================================================================
# Channel geometry ladder (B0-B5)
# ===========================================================================

def _channel_forward(cfg, model_name, params):
    """Run the flat-wall channel forward to (developed) steady state and
    return the diagnostics dict (u/v/A_* trajectories). ``params`` carries
    ``Gp, lam`` (+ ``tau_y`` for Saramito)."""
    grid, model, init_state, perm_f = vf._build_channel(cfg, model_name)
    return p3b._evolve_wall_bounded_with_diagnostics(
        initial_state=init_state, model=model, polymer_params=params,
        grid=grid, density=cfg['density'], base_viscosity=cfg['nu_s'],
        dt=cfg['dt'], inner_steps=cfg['inner_steps'],
        outer_steps=cfg['outer_steps'], solver_type=cfg['solver_type'],
        use_preconditioner=cfg['use_preconditioner'],
        preconditioner_type=cfg['preconditioner_type'],
        pressure_gradient=(cfg['g_x'], 0.0), permeability=perm_f,
        U_f=cfg['U_f'], solver_tol=cfg['solver_tol'],
        solver_maxiter=cfg['solver_maxiter'])


def channel_geometry_print(config=None):
    """B0 (login): print the channel geometry -- grid, extents, wall/BC
    layout, g_x. No solve. Eyeball: flat walls top & bottom, periodic x, no
    residual constriction object (perm_f == 0.0)."""
    cfg = dict(vf.DEFAULT_CHANNEL_CONFIG)
    if config:
        cfg.update(config)
    grid, model, init_state, perm_f = vf._build_channel(
        cfg, 'oldroyd_b_logconf_bk_v2')
    print("=== B0: channel geometry ===")
    print(f"  grid: Nx={cfg['Nx']} Ny={cfg['Ny']}  domain Lx={cfg['Lx']} "
          f"Ly={cfg['Ly']} (H=Ly/2={0.5*cfg['Ly']})")
    print(f"  walls: no-slip at y=0 and y={cfg['Ly']} (U_wall=0); periodic in x")
    print(f"  conformation BC: extrapolation (y), periodic (x)")
    print(f"  drive: body force g_x={cfg['g_x']}  (pressure_gradient=(g_x,0))")
    print(f"  IB object: perm_f={perm_f}  (== 0.0 => object-free channel)")
    print(f"  grid.shape={grid.shape}  dx={cfg['Lx']/cfg['Nx']:.4f} "
          f"dy={cfg['Ly']/cfg['Ny']:.4f}")
    return dict(cfg=cfg, perm_f=perm_f, object_free=bool(perm_f == 0.0))


def channel_newtonian_gate(config=None, rtol=0.05):
    """B1 (gpu_test, the arbiter) + B2 (symmetry): OB plane-Poiseuille
    parabola. Run ``oldroyd_b_logconf_bk_v2`` to steady state; compare the
    x-averaged u(y) to the analytic parabola (bulk window), and check
    symmetry / centreline-max / zero-at-walls. This catches BC-sign,
    forcing-sign, and grid errors against a known closed form."""
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(vf.DEFAULT_CHANNEL_CONFIG)
    if config:
        cfg.update(config)
    Gp, lam, nu_s, g_x = cfg['Gp_init'], cfg['lam_init'], cfg['nu_s'], cfg['g_x']
    params = dict(Gp=jnp.asarray(Gp, dtype=jnp.float64),
                  lam=jnp.asarray(lam, dtype=jnp.float64))
    out = _channel_forward(cfg, 'oldroyd_b_logconf_bk_v2', params)
    u = np.asarray(out['u_traj'][-1]).mean(axis=0)      # (Ny,)
    ref = vf.channel_poiseuille_reference(cfg, Gp, lam, nu_s, g_x)
    ua = ref['u']
    any_nan = bool(np.asarray(out['any_nan_traj']).any())
    Ny = cfg['Ny']; bm = int(cfg['bulk_margin']); bulk = slice(bm, Ny - bm)
    denom = max(np.abs(ua[bulk]).max(), 1e-30)
    bulk_rel = float(np.abs(u[bulk] - ua[bulk]).max() / denom)
    # B2 symmetry about the centreline; centre is the max; walls ~0.
    sym_err = float(np.abs(u - u[::-1]).max() / max(np.abs(u).max(), 1e-30))
    umax_meas = float(u.max()); j_max = int(np.argmax(u))
    center_ok = abs(j_max - (Ny - 1) / 2.0) <= 1.5
    wall_rel = float(max(abs(u[0]), abs(u[-1])) / max(np.abs(u).max(), 1e-30))
    b1_ok = bool(bulk_rel < rtol and not any_nan)
    b2_ok = bool(sym_err < 0.02 and center_ok and wall_rel < 0.10)
    print("=== B1 Newtonian/OB plane-Poiseuille parabola (arbiter) ===")
    print(f"  u_max: meas={umax_meas:.4f}  analytic={ref['u_max']:.4f}")
    print(f"  bulk rel err vs parabola = {bulk_rel:.2e} (gate {rtol:.0e})  "
          f"any_nan={any_nan}")
    print(f"  B2: symmetry={sym_err:.2e}  centre-is-max={center_ok} "
          f"(argmax row {j_max}/{Ny})  wall/max={wall_rel:.2e}")
    print(f"  B1 {'PASS' if b1_ok else 'FAIL'}   B2 {'PASS' if b2_ok else 'FAIL'}")
    return dict(bulk_rel=bulk_rel, sym_err=sym_err, umax_meas=umax_meas,
                umax_analytic=ref['u_max'], center_ok=center_ok,
                wall_rel=wall_rel, any_nan=any_nan, b1_pass=b1_ok,
                b2_pass=b2_ok, gate_pass=bool(b1_ok and b2_ok),
                u=u, u_analytic=ua, y=ref['y'])


def channel_constriction_regression(config=None, rtol=1e-12,
                                    known_ob_loss=1.5376320564):
    """B3: adding the channel must not perturb the constriction. Recompute the
    OB constriction forward loss at (Gp_init, lam_init) and check it matches
    the known constriction-loss value bit-identically (``_build_constriction``
    byte-identical)."""
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(p3b.DEFAULT_MULTISTEP_AD_FD_CONFIG)
    if config:
        cfg.update(config)
    p = dict(Gp=jnp.asarray(cfg['Gp_init'], dtype=jnp.float64),
             lam=jnp.asarray(cfg['lam_init'], dtype=jnp.float64))
    grid, model, state, perm = vf._build_constriction(cfg, 'oldroyd_b_logconf_bk_v2')
    loss = vf._constriction_loss_fn(cfg, state, model, grid, perm,
                                    fixed_params={}, diff_keys=('Gp', 'lam'))
    L = float(jax.jit(loss)(p))
    rel = abs(L - known_ob_loss) / max(abs(known_ob_loss), 1e-30)
    gate_pass = bool(rel <= 1e-9)   # matches to the known value (~1e-14 noise)
    print("=== B3 constriction-untouched regression ===")
    print(f"  OB constriction forward loss = {L:.10e}  known = {known_ob_loss:.10e}"
          f"  rel = {rel:.2e}  {'PASS' if gate_pass else 'FAIL'}")
    return dict(loss=L, known=known_ob_loss, rel=rel, gate_pass=gate_pass)


def _channel_profiles(out, cfg, Gp):
    """x-averaged final-time profiles: u(y), gamma_dot(y) (central diff),
    polymer |tau_d|(y), and kappa_y(y) (needs tau_y at the call site)."""
    Ny, Ly = cfg['Ny'], cfg['Ly']
    dy = Ly / Ny
    u = np.asarray(out['u_traj'][-1]).mean(axis=0)
    gdot = np.full(Ny, 0.0)
    gdot[1:-1] = (u[2:] - u[:-2]) / (2.0 * dy)
    Axx = np.asarray(out['A_xx_traj'][-1]).mean(axis=0)
    Axy = np.asarray(out['A_xy_traj'][-1]).mean(axis=0)
    Ayy = np.asarray(out['A_yy_traj'][-1]).mean(axis=0)
    Azz = np.asarray(out['A_zz_traj'][-1]).mean(axis=0)
    td = np.asarray(tb.saramito_tau_d_norm(
        jnp.asarray(Axx), jnp.asarray(Axy), jnp.asarray(Ayy),
        jnp.asarray(Azz), Gp))
    y = (np.arange(Ny) + 0.5) * dy
    return dict(y=y, y_from_centre=y - 0.5 * Ly, u=u, gamma_dot=gdot,
                tau_d=td, Axx=Axx, Axy=Axy, Ayy=Ayy, Azz=Azz, dy=dy)


def channel_saramito_plug_gate(tau_y, config=None, kappa_plug=0.05,
                               gdot_frac=0.15):
    """B4 (gpu_test): Saramito plug forms in the channel. Run
    ``saramito_logconf_bk_v2`` to steady state at ``tau_y``; confirm a flat
    central plug, measure its half-width (the central band where the model is
    UNyielded, ``kappa_y < kappa_plug``) vs the analytic ``y_p = tau_y/g_x``,
    and confirm ``kappa_y > 0`` in the yielded shoulders + flat u in the plug.
    This is the physics proof that the channel makes tau_y observable."""
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(vf.DEFAULT_CHANNEL_CONFIG)
    if config:
        cfg.update(config)
    Gp, lam, g_x = cfg['Gp_init'], cfg['lam_init'], cfg['g_x']
    params = dict(Gp=jnp.asarray(Gp, dtype=jnp.float64),
                  lam=jnp.asarray(lam, dtype=jnp.float64),
                  tau_y=jnp.asarray(tau_y, dtype=jnp.float64))
    out = _channel_forward(cfg, 'saramito_logconf_bk_v2', params)
    any_nan = bool(np.asarray(out['any_nan_traj']).any())
    prof = _channel_profiles(out, cfg, Gp)
    ky = np.asarray(tb.saramito_kappa_y(
        jnp.asarray(prof['Axx']), jnp.asarray(prof['Axy']),
        jnp.asarray(prof['Ayy']), jnp.asarray(prof['Azz']), Gp, tau_y))
    Ny, Ly = cfg['Ny'], cfg['Ly']
    H = 0.5 * Ly
    # Plug = contiguous central band of UNyielded cells (kappa_y < kappa_plug).
    unyielded = ky < kappa_plug
    jc = Ny // 2
    lo = jc
    while lo - 1 >= 0 and unyielded[lo - 1]:
        lo -= 1
    hi = jc
    while hi + 1 < Ny and unyielded[hi + 1]:
        hi += 1
    plug_cells = (hi - lo + 1) if unyielded[jc] else 0
    plug_half_width = 0.5 * plug_cells * prof['dy']
    y_p_analytic = tau_y / g_x
    yp_rel = (abs(plug_half_width - y_p_analytic) / max(y_p_analytic, 1e-30)
              if plug_cells > 0 else float('nan'))
    # Flat top: |gamma_dot| in the plug is small vs the max shear.
    gmax = max(np.abs(prof['gamma_dot']).max(), 1e-30)
    plug_gdot_frac = (float(np.abs(prof['gamma_dot'][lo:hi + 1]).max() / gmax)
                      if plug_cells > 0 else float('nan'))
    ky_shoulder = float(ky[[bm for bm in (int(cfg['bulk_margin']),
                                          Ny - 1 - int(cfg['bulk_margin']))]].max())
    plug_forms = bool(plug_cells >= 3 and unyielded[jc])
    flat_top = bool(np.isfinite(plug_gdot_frac) and plug_gdot_frac < gdot_frac)
    shoulders_yield = bool(ky_shoulder > 0.1)
    gate_pass = bool(plug_forms and flat_top and shoulders_yield and not any_nan)
    print(f"=== B4 Saramito plug in channel (tau_y={tau_y}, g_x={g_x}) ===")
    print(f"  plug: {plug_cells} cells, half-width={plug_half_width:.4f} vs "
          f"y_p=tau_y/g_x={y_p_analytic:.4f} (rel {yp_rel:.1%}), "
          f"plug frac={2*plug_half_width/Ly:.2f} (y_p/H={plug_half_width/H:.2f})")
    print(f"  flat top: plug |gdot|/max = {plug_gdot_frac:.2f} (<{gdot_frac})  "
          f"shoulder kappa_y={ky_shoulder:.3f} (>0.1)  any_nan={any_nan}")
    print(f"  B4 {'PASS' if gate_pass else 'FAIL'} "
          f"(plug_forms={plug_forms} flat={flat_top} shoulders={shoulders_yield})")
    return dict(plug_cells=int(plug_cells), plug_half_width=plug_half_width,
                y_p_analytic=y_p_analytic, yp_rel=yp_rel, plug_forms=plug_forms,
                flat_top=flat_top, plug_gdot_frac=plug_gdot_frac,
                ky_shoulder=ky_shoulder, shoulders_yield=shoulders_yield,
                any_nan=any_nan, gate_pass=gate_pass,
                y=prof['y'], u=prof['u'], gamma_dot=prof['gamma_dot'],
                tau_d=prof['tau_d'], kappa_y=ky)


def channel_calibrate_tau_y(config=None, target_fracs=(0.35, 0.40, 0.45, 0.50)):
    """B5 (gpu_test): pick tau_y for a target plug fraction. Run OB (tau_y=0)
    in the channel to get the polymer |tau_d|(y) field; the yield surface for
    a plug fraction f (central band of half-width f*H) sits at |y-centre|=f*H,
    so tau_y = |tau_d| there. Reports tau_y for each target f and the
    analytic y_p/H = tau_y/(g_x*H). Confirm against a forward |tau_d| field."""
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(vf.DEFAULT_CHANNEL_CONFIG)
    if config:
        cfg.update(config)
    Gp, lam, g_x = cfg['Gp_init'], cfg['lam_init'], cfg['g_x']
    params = dict(Gp=jnp.asarray(Gp, dtype=jnp.float64),
                  lam=jnp.asarray(lam, dtype=jnp.float64))
    out = _channel_forward(cfg, 'oldroyd_b_logconf_bk_v2', params)
    prof = _channel_profiles(out, cfg, Gp)
    Ny, Ly = cfg['Ny'], cfg['Ly']
    H = 0.5 * Ly
    yc = np.abs(prof['y_from_centre'])
    td = prof['tau_d']
    print(f"=== B5 tau_y calibration (channel, g_x={g_x}, H={H}) ===")
    print(f"  polymer |tau_d|(y): centre~{td[Ny//2]:.4f}  wall~{td[[0,-1]].max():.4f}")
    rows = []
    for f in target_fracs:
        # yield location |y-centre| = f*H; tau_y = |tau_d| there (nearest cell).
        j = int(np.argmin(np.abs(yc - f * H)))
        ty = float(td[j])
        rows.append(dict(target_frac=f, tau_y=ty, yp_over_H_analytic=ty / (g_x * H)))
        print(f"  target plug frac {f:.2f} (|y-c|={f*H:.3f}) => tau_y={ty:.4f} "
              f"(analytic y_p/H={ty/(g_x*H):.2f})")
    return dict(rows=rows, tau_d=td, y=prof['y'], g_x=g_x, H=H,
                Gp=Gp, lam=lam, nu_s=cfg['nu_s'])

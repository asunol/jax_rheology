"""Family-extension harness -- Giesekus (and, later, FENE-P / linear PTT).

The *models* live in `jax_rheology/log_conformation.py` (the registered
presets and the shared affine relaxation integrator); this module is the
*test/validation* side:

  * the shared steady simple-shear **reference solver** -- a Newton root
    of `UC_stretch(A; gammadot) + R(A) = 0` (NOT a transcribed prefactor),
    used by the steady-shear physics check;
  * the continuous-source `R(A)` of each model (the analytic truth the
    reference solver roots);
  * the three per-model gates (Sec.4): the alpha=0 Oldroyd-B regression
    (Sec.4.1), AD-vs-FD on the new `alpha` partial (Sec.4.3), and the Couette
    Wi-sweep physics check vs the reference (Sec.4.2).

Notebook usage (see `visco_families.ipynb`): enable float64, then
`import visco_families as vf` and call `vf.run_all_giesekus_gates()`
or the individual `vf.giesekus_*` runners. Everything heavy is here;
the notebook is thin.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

import analytic_limits_validation as p3b
from jax_rheology.models import registry as cr
from jax_rheology import log_conformation as lc


# ===========================================================================
# Sec.1. Continuous-source R(A) -- the analytic truth the reference solver roots
# ===========================================================================
# A-space relaxation source `A^grad = R(A)`; these are the *continuous*
# forms (the registered models integrate a frozen-coefficient version of
# the same R over dt). Components are the full planar conformation
# `(A_xx, A_xy, A_yy, A_zz)`; `tr A = A_xx + A_yy + A_zz`.

def oldroyd_b_source_R(A_xx, A_xy, A_yy, A_zz, lam, params=None):
    """`R(A) = -(1/lam)(A - I)` (constitutive ref Sec.1)."""
    del params
    return (-(A_xx - 1.0) / lam,
            -A_xy / lam,
            -(A_yy - 1.0) / lam,
            -(A_zz - 1.0) / lam)


def giesekus_source_R(A_xx, A_xy, A_yy, A_zz, lam, params):
    """`R(A) = -(1/lam)[ (A - I) + alpha (A - I)(A - I) ]` (constitutive ref Sec.2).

    `alpha = params['alpha']`. The matrix square of the symmetric in-plane
    block `B = A - I` is `(B^2)_xx = B_xx^2 + B_xy^2`,
    `(B^2)_xy = B_xy(B_xx + B_yy)`, `(B^2)_yy = B_xy^2 + B_yy^2`; the
    out-of-plane channel is the scalar `(B^2)_zz = B_zz^2`.
    """
    alpha = params['alpha']
    B_xx = A_xx - 1.0
    B_xy = A_xy
    B_yy = A_yy - 1.0
    B_zz = A_zz - 1.0
    BB_xx = B_xx * B_xx + B_xy * B_xy
    BB_xy = B_xy * (B_xx + B_yy)
    BB_yy = B_xy * B_xy + B_yy * B_yy
    BB_zz = B_zz * B_zz
    return (-(B_xx + alpha * BB_xx) / lam,
            -(B_xy + alpha * BB_xy) / lam,
            -(B_yy + alpha * BB_yy) / lam,
            -(B_zz + alpha * BB_zz) / lam)


def fene_p_source_R(A_xx, A_xy, A_yy, A_zz, lam, params):
    """`R(A) = -(1/lam)[ f.A - a.I ]`, `f = L^2/(L^2-trA)`, `a = L^2/(L^2-3)`
    (constitutive ref Sec.4). `L^2 = params['Lsq']`. Uses the same smooth-
    floored Peterlin factor as the model (`lc._fene_p_peterlin_f`) so the
    reference and the solver share one definition of `f`.
    """
    Lsq = params['Lsq']
    f = lc._fene_p_peterlin_f(A_xx + A_yy + A_zz, Lsq)
    a = Lsq / (Lsq - 3.0)
    return (-(f * A_xx - a) / lam,
            -(f * A_xy) / lam,
            -(f * A_yy - a) / lam,
            -(f * A_zz - a) / lam)


# --- model-specific viscometric readouts for the steady-shear reference ----

def hookean_viscometric(A_xx, A_xy, A_yy, A_zz, params, Gp):
    """`tau = Gp(A - I)` => `N1 = Gp(A_xx-A_yy)`, `tau_xy = Gp.A_xy`,
    `N2 = Gp(A_yy-A_zz)`. Shared by Oldroyd-B / Giesekus / PTT."""
    del params
    return (Gp * (A_xx - A_yy), Gp * A_xy, Gp * (A_yy - A_zz))


def fene_p_viscometric(A_xx, A_xy, A_yy, A_zz, params, Gp):
    """`tau = Gp(f.A - a.I)` => the `a.I` cancels in the differences:
    `N1 = Gp.f.(A_xx-A_yy)`, `tau_xy = Gp.f.A_xy`, `N2 = Gp.f.(A_yy-A_zz)`.
    """
    Lsq = params['Lsq']
    f = lc._fene_p_peterlin_f(A_xx + A_yy + A_zz, Lsq)
    return (Gp * f * (A_xx - A_yy), Gp * f * A_xy, Gp * f * (A_yy - A_zz))


def saramito_source_R(A_xx, A_xy, A_yy, A_zz, lam, params):
    """Saramito Bingham EVP relaxation `R(A) = -(kappa_y/lam)(A - I)`.

    `kappa_y = max(0, 1 - tau_y/|tau_d|)`, `|tau_d|` the von-Mises deviator norm of
    `tau = Gp(A - I)` (over all four components). Uses the **canonical**
    `jax_rheology.models.tbnn_memory.saramito_kappa_y` so this continuous reference source and
    the registered `saramito_logconf_bk_v2` generator share ONE definition
    of the yield factor (no drift). `params`: `Gp`, `tau_y`.

    NOTE: valid on the **yielded branch** only. Below yield `kappa_y -> 0` and
    the rest state is non-unique (any sub-yield `A` has `R = 0`); the Newton
    root of `stretch + R = 0` at a prescribed `gammadot > 0` lands on the yielded
    branch by construction (steady flow requires yield). Gate physics on the
    yielded branch and on the 0D start-up/cessation trajectories, never on a
    below-yield "steady state" (see `saramito_yielded_shear_reference`).
    """
    from jax_rheology.models import tbnn_memory as tb  #  lazy: avoid registering
    Gp = params['Gp']                            # TBNN models on plain vf import
    tau_y = params['tau_y']
    ky = tb.saramito_kappa_y(A_xx, A_xy, A_yy, A_zz, Gp, tau_y)
    return (-ky * (A_xx - 1.0) / lam,
            -ky * A_xy / lam,
            -ky * (A_yy - 1.0) / lam,
            -ky * (A_zz - 1.0) / lam)


def saramito_yielded_shear_reference(gammadot, lam, Gp, tau_y,
                                     A_init=None, n_iter=300):
    """Yielded-branch-only steady simple-shear reference for Saramito.

    Thin wrapper over :func:`steady_simple_shear_reference` with
    `saramito_source_R` + the shared Hookean readout, augmented with the
    yield diagnostics `|tau_d|`, `kappa_y`, and a `yielded` flag (`|tau_d| > tau_y`).
    The Newton root is only physically meaningful **above yield**; at
    prescribed `gammadot > 0` the steady state is yielded by construction. Callers
    should discard points with `converged=False` or `yielded=False`.
    """
    from jax_rheology.models import tbnn_memory as tb
    params = {'Gp': Gp, 'tau_y': tau_y}
    # Warm-start on the YIELDED branch: cold-starting Newton from A = I puts
    # it exactly on the below-yield set (kappa_y = 0 => singular Jacobian).
    # The Oldroyd-B analytic steady state (the fully-yielded kappa_y -> 1
    # limit) is a robust in-branch guess: A_xx = 1 + 2(lam.gd)^2, A_xy =
    # lam.gd, A_yy = A_zz = 1.
    if A_init is None:
        wi = lam * gammadot
        A_init = np.array([1.0 + 2.0 * wi ** 2, wi, 1.0, 1.0], dtype=np.float64)
    ref = steady_simple_shear_reference(
        saramito_source_R, gammadot, lam, params=params, Gp=Gp,
        viscometric_fn=hookean_viscometric, A_init=A_init, n_iter=n_iter)
    a = ref['a']
    td = float(tb.saramito_tau_d_norm(a[0], a[1], a[2], a[3], Gp))
    ky = float(tb.saramito_kappa_y(a[0], a[1], a[2], a[3], Gp, tau_y))
    ref['tau_d_norm'] = td
    ref['kappa_y'] = ky
    ref['yielded'] = bool(td > tau_y)
    return ref


def ptt_source_R(A_xx, A_xy, A_yy, A_zz, lam, params):
    """Linear PTT `R(A) = -(f/lam)(A - I)`, `f = 1 + eps(tr A - 3)`
    (constitutive ref Sec.6). `eps = params['epsilon']`. Slip `zeta=0`, so the
    stretch backbone is the standard upper-convected one and PTT reuses
    the Hookean readout (`hookean_viscometric`).
    """
    epsilon = params['epsilon']
    f = 1.0 + epsilon * (A_xx + A_yy + A_zz - 3.0)
    return (-(f * (A_xx - 1.0)) / lam,
            -(f * A_xy) / lam,
            -(f * (A_yy - 1.0)) / lam,
            -(f * (A_zz - 1.0)) / lam)


# ===========================================================================
# Sec.2. Shared steady simple-shear reference solver (Newton root)
# ===========================================================================
# Steady homogeneous simple shear `u = (gammadot.y, 0)`, so the velocity
# gradient is `L = [[0, gammadot], [0, 0]]` (Hulsen convention `L_ij = d_j u_i`,
# matching `upper_convective_step_analytic`). The upper-convective
# stretch contributes `L.A + A.LT` to `dA/dt`:
#     stretch_xx = 2 gammadot A_xy,  stretch_xy = gammadot A_yy,
#     stretch_yy = 0,          stretch_zz = 0.
# Steady state: `stretch(A; gammadot) + R(A) = 0` (4 unknowns). This roots the
# model's *own* balance -- it reproduces Oldroyd-B's `N1 = 2 Gp lam^2 gammadot^2`
# exactly and gives the correct root for every family member by
# construction.

def _steady_shear_residual(a, R_fn, gammadot, lam, params):
    A_xx, A_xy, A_yy, A_zz = a[0], a[1], a[2], a[3]
    s_xx = 2.0 * gammadot * A_xy
    s_xy = gammadot * A_yy
    s_yy = 0.0
    s_zz = 0.0
    R_xx, R_xy, R_yy, R_zz = R_fn(A_xx, A_xy, A_yy, A_zz, lam, params)
    return jnp.array([s_xx + R_xx, s_xy + R_xy, s_yy + R_yy, s_zz + R_zz])


def steady_simple_shear_reference(R_fn: Callable,
                                  gammadot: float,
                                  lam: float,
                                  params: Optional[Dict[str, Any]] = None,
                                  Gp: float = 1.0,
                                  viscometric_fn: Optional[Callable] = None,
                                  A_init: Optional[np.ndarray] = None,
                                  n_iter: int = 200,
                                  tol: float = 1e-13,
                                  ) -> Dict[str, Any]:
    """Damped-Newton root of `stretch(A; gammadot) + R(A) = 0`.

    Returns the steady conformation and the model's viscometric
    functions `(N1, tau_xy, N2)`, computed by `viscometric_fn` (defaults
    to the Hookean readout `hookean_viscometric` shared by Oldroyd-B /
    Giesekus / PTT). FENE-P passes `fene_p_viscometric` so the `f`
    factor enters `N1`/`tau_xy`/`N2`. `A_init` warm-starts the Newton
    iteration (used by the Wi sweep for robustness at high Wi).
    """
    params = params or {}
    if viscometric_fn is None:
        viscometric_fn = hookean_viscometric
    resid = lambda a: _steady_shear_residual(a, R_fn, gammadot, lam, params)
    jac = jax.jacobian(resid)

    a = (jnp.asarray(A_init, dtype=jnp.float64) if A_init is not None
         else jnp.array([1.0, 0.0, 1.0, 1.0], dtype=jnp.float64))
    converged = False
    for _ in range(n_iter):
        F = resid(a)
        nF = float(jnp.max(jnp.abs(F)))
        if nF < tol:
            converged = True
            break
        J = jac(a)
        da = jnp.linalg.solve(J, -F)
        # Damped step: backtrack until the residual norm decreases.
        t = 1.0
        while t > 1e-6:
            a_try = a + t * da
            if float(jnp.max(jnp.abs(resid(a_try)))) < nF:
                break
            t *= 0.5
        a = a + t * da

    A_xx, A_xy, A_yy, A_zz = (float(a[0]), float(a[1]),
                              float(a[2]), float(a[3]))
    N1, tau_xy, N2 = viscometric_fn(A_xx, A_xy, A_yy, A_zz, params, Gp)
    N1, tau_xy, N2 = float(N1), float(tau_xy), float(N2)
    eta_p = tau_xy / gammadot if gammadot != 0.0 else float('nan')
    return dict(A_xx=A_xx, A_xy=A_xy, A_yy=A_yy, A_zz=A_zz,
                N1=N1, tau_xy=tau_xy, N2=N2, eta_p=eta_p,
                trA=A_xx + A_yy + A_zz,
                gammadot=gammadot, lam=lam, Wi=lam * gammadot,
                Gp=Gp, converged=converged,
                residual=float(jnp.max(jnp.abs(resid(a)))),
                a=np.asarray(a))


# ===========================================================================
# Sec.3. Constriction loss builder (shared by the regression + AD-FD gates)
# ===========================================================================

def _constriction_loss_fn(cfg, init_state, model, grid, perm_f,
                          fixed_params: Dict[str, Any],
                          diff_keys: Tuple[str, ...]):
    """Velocity-RMSE loss over `diff_keys`, with `fixed_params` merged in.

    Same validated path as `analytic_limits_validation`'s milestone
    cells (`_evolve_wall_bounded_with_diagnostics` on the constriction),
    so the only thing that changes between Oldroyd-B and Giesekus is the
    `model` and the presence of `alpha` in the params dict.
    """
    def loss(diff_params):
        params = dict(fixed_params)
        for k in diff_keys:
            params[k] = diff_params[k]
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


def _build_constriction(cfg, model_name):
    grid = p3b._build_grid(cfg['Nx'], cfg['Ny'], cfg['Lx'], cfg['Ly'])
    model = cr.get_model(model_name)
    domain = ((0.0, cfg['Lx']), (0.0, cfg['Ly']))
    init_state, perm_f = p3b._build_constriction_initial_state(
        grid=grid, model=model, wall_conformation_bc='extrapolation',
        obstacle_radius=cfg['obstacle_radius'], domain=domain,
        ib_smoothing_width=cfg['ib_smoothing_width'],
        ib_smoothing_scale=cfg['ib_smoothing_scale'])
    return grid, model, init_state, perm_f


# ===========================================================================
# Pressure-driven planar channel (the elastoviscoplastic geometry)
# ===========================================================================
# A yield-stress fluid in a body-force-driven planar channel forms a central
# rigid plug where the local shear stress is below yield. The fully-developed
# shear stress is LINEAR in the wall-normal coordinate, so the plug half-width
# is algebraic and exact: y_p = tau_y / g_x  (H = Ly/2, tau_w = g_x*H), and
# the plug (the flat top of u(y)) is a FIRST-ORDER velocity feature -- so
# velocity-only data pins tau_y almost directly (Buckingham-Reiner), unlike
# the saturated constriction. This is a GEOMETRY ADDITION only: same g_x
# body-force drive, periodic-x, flat no-slip walls, but NO constriction IB
# object (permeability = 0.0). Reuses the validated plane-Poiseuille
# path (p3b._build_wall_bounded_initial_state, U_wall=0) verbatim -- the
# closure, Saramito generator, and log_conformation are untouched.

DEFAULT_CHANNEL_CONFIG: Dict[str, Any] = dict(
    # Flat-wall channel: walls at y=0 and y=Ly (H = Ly/2, centreline Ly/2),
    # periodic in x (fully developed => x-invariant, so Nx is small). Fine in
    # y to resolve the plug. NO IB object => dt is NOT the IB-locked 1e-4; use
    # the Poiseuille CFL-safe dt (conformation advection is x-only, u_max small).
    Nx=32,
    Ny=128,
    Lx=1.0,
    Ly=2.0,          # H = Ly/2 = 1.0
    density=1.0,
    nu_s=0.8,        # match the constriction P3-G4 truth (polymer-dominant:
    Gp_init=3.2,     # eta_p = Gp*lam = 2.24 vs eta_s = 0.8, beta_p ~ 0.74,
    lam_init=0.7,    # so the plug is reasonably sharp).
    g_x=4.0,         # body-force drive (calibrated in B5 for a target plug).
    U_wall=0.0,      # no-slip both walls (Poiseuille).
    U_f=0.0,
    dt=2.5e-3,       # Poiseuille CFL-safe (no IB penalty here).
    inner_steps=10,
    outer_steps=200,  # T = 5 = ~7 lam at lam=0.7: developed (plug is a
                      # STEADY feature, unlike the short constriction horizon).
    solver_type='bicgstab',
    use_preconditioner=False,
    preconditioner_type='none',
    solver_tol=1.0e-12,
    solver_maxiter=500,
    bulk_margin=2,
    # Unused-but-present keys so the same forward/loss code paths work.
    obstacle_radius=0.0, ib_smoothing_width=0.0, ib_smoothing_scale=0.0,
)


def _build_channel(cfg, model_name):
    """Pressure-driven flat-wall channel (the elastoviscoplastic geometry). Mirrors
    :func:`_build_constriction` but strips the IB constriction object: flat
    no-slip walls (``U_wall = 0``) via the validated wall-bounded
    (plane-Poiseuille) initial state, periodic-x, driven by the ``g_x`` body
    force in the loss/forward. Returns ``perm_f = 0.0`` (no penalty object) so
    the shared forward driver treats it as object-free. ``_build_constriction``
    is left byte-identical (gate B3)."""
    grid = p3b._build_grid(cfg['Nx'], cfg['Ny'], cfg['Lx'], cfg['Ly'])
    model = cr.get_model(model_name)
    init_state = p3b._build_wall_bounded_initial_state(
        grid, 0.0, model, 'extrapolation')
    return grid, model, init_state, 0.0


def channel_poiseuille_reference(cfg, Gp, lam, nu_s, g_x, tau_y=None):
    """Analytic references for the channel (walls at y=0, y=Ly; H=Ly/2):

    * **OB plane-Poiseuille parabola** (B1 arbiter): total shear viscosity
      ``eta0 = nu_s + Gp*lam`` (Oldroyd-B has constant shear viscosity; N1
      varies across the channel but does not feed fully-developed axial
      momentum), so ``u(y) = (g_x/(2 eta0)) * y * (Ly - y)``, peak
      ``g_x*Ly^2/(8 eta0)`` at the centreline; ``gamma_dot(y) =
      (g_x/eta0)*(Ly/2 - y)``; total ``tau_xy(y) = eta0*gamma_dot(y)`` linear
      in y (slope -g_x), magnitude ``tau_w = g_x*H`` at the wall.
    * **Plug** (B4/B5, pure-Bingham estimate): ``y_p = tau_y/g_x`` from the
      centreline, ``plug_fraction = 2*y_p/Ly = tau_y/tau_w``. (The Saramito
      yield is on the polymer |tau_d|, so with a solvent the measured plug is
      approximate -- B4 measures it empirically; this is the analytic target.)
    """
    Ny, Ly = cfg['Ny'], cfg['Ly']
    dy = Ly / Ny
    y = (np.arange(Ny) + 0.5) * dy           # cell-centred, matches the solver
    eta0 = nu_s + Gp * lam
    H = 0.5 * Ly
    u = (g_x / (2.0 * eta0)) * y * (Ly - y)
    gdot = (g_x / eta0) * (0.5 * Ly - y)
    tau_xy_total = eta0 * gdot
    tau_w = g_x * H
    out = dict(y=y, y_from_centre=y - 0.5 * Ly, u=u, gamma_dot=gdot,
               tau_xy_total=tau_xy_total, eta0=eta0, H=H, tau_w=tau_w,
               u_max=float(g_x * Ly ** 2 / (8.0 * eta0)))
    if tau_y is not None:
        y_p = tau_y / g_x
        out.update(tau_y=tau_y, y_p=y_p, plug_fraction=min(2.0 * y_p / Ly, 1.0),
                   yp_over_H=min(y_p / H, 1.0))
    return out


def _channel_loss_fn(cfg, init_state, model, grid, perm_f, fixed_params,
                     diff_keys):
    """Velocity-RMSE loss for the channel geometry. The constriction loss
    builder is already geometry-agnostic (it takes ``init_state`` / ``model``
    / ``grid`` / ``perm_f`` and passes ``perm_f`` as ``permeability``); the
    channel simply supplies the flat-wall state and ``perm_f = 0.0``. This
    thin delegator exists for naming symmetry."""
    return _constriction_loss_fn(cfg, init_state, model, grid, perm_f,
                                 fixed_params=fixed_params, diff_keys=diff_keys)


# ===========================================================================
# Sec.4.1  Regression gate -- Giesekus(alpha=0) == Oldroyd-B (machine precision)
# ===========================================================================

def giesekus_regression_gate(config: Optional[Dict[str, Any]] = None,
                             rtol: float = 1e-6,
                             ) -> Dict[str, Any]:
    """At alpha=0, Giesekus must reproduce `oldroyd_b_logconf_bk_v2`.

    Compares forward loss and `dL/dGp`, `dL/dlam` between the two models
    on the validated short-T constriction config. With the affine
    exponential integrator the alpha=0 limit is the exact Oldroyd-B
    analytic exponential, so agreement is machine-precision (the gate
    chosen in alignment: ~1e-12 / FD-noise, not literal byte-equality).
    """
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(p3b.DEFAULT_MULTISTEP_AD_FD_CONFIG)
    if config:
        cfg.update(config)

    p = dict(Gp=jnp.asarray(cfg['Gp_init'], dtype=jnp.float64),
             lam=jnp.asarray(cfg['lam_init'], dtype=jnp.float64))

    # Oldroyd-B reference.
    grid, ob_model, ob_state, ob_perm = _build_constriction(
        cfg, 'oldroyd_b_logconf_bk_v2')
    ob_loss = _constriction_loss_fn(cfg, ob_state, ob_model, grid, ob_perm,
                                    fixed_params={}, diff_keys=('Gp', 'lam'))
    ob_vg = jax.jit(jax.value_and_grad(ob_loss))
    L_ob, g_ob = ob_vg(p)
    L_ob = float(L_ob)
    g_ob = {k: float(v) for k, v in g_ob.items()}

    # Giesekus at alpha = 0.
    grid_g, g_model, g_state, g_perm = _build_constriction(
        cfg, 'giesekus_logconf_bk_v2')
    g_loss = _constriction_loss_fn(
        cfg, g_state, g_model, grid_g, g_perm,
        fixed_params={'alpha': jnp.asarray(0.0, dtype=jnp.float64)},
        diff_keys=('Gp', 'lam'))
    g_vg = jax.jit(jax.value_and_grad(g_loss))
    L_g, g_g = g_vg(p)
    L_g = float(L_g)
    g_g = {k: float(v) for k, v in g_g.items()}

    def _rel(a, b):
        return abs(a - b) / max(abs(b), 1e-30)

    rel_loss = _rel(L_g, L_ob)
    rel_Gp = _rel(g_g['Gp'], g_ob['Gp'])
    rel_lam = _rel(g_g['lam'], g_ob['lam'])
    gate_pass = bool(rel_loss < rtol and rel_Gp < rtol and rel_lam < rtol)

    print("=== Giesekus alpha=0 regression vs oldroyd_b_logconf_bk_v2 ===")
    print(f"  forward loss : OB = {L_ob:.10e}   G(alpha=0) = {L_g:.10e}"
          f"   rel = {rel_loss:.2e}")
    print(f"  dL/dGp       : OB = {g_ob['Gp']:.10e}   G = {g_g['Gp']:.10e}"
          f"   rel = {rel_Gp:.2e}")
    print(f"  dL/dlam      : OB = {g_ob['lam']:.10e}   G = {g_g['lam']:.10e}"
          f"   rel = {rel_lam:.2e}")
    print(f"  GATE (rtol={rtol:.0e}): {'PASS' if gate_pass else 'FAIL'}")
    return dict(L_ob=L_ob, L_g=L_g, g_ob=g_ob, g_g=g_g,
                rel_loss=rel_loss, rel_Gp=rel_Gp, rel_lam=rel_lam,
                gate_pass=gate_pass)


# ===========================================================================
# Sec.4.3  Gradient gate -- AD vs FD on the new `alpha` partial
# ===========================================================================

def giesekus_alpha_ad_vs_fd(config: Optional[Dict[str, Any]] = None,
                            alpha0: float = 0.4,
                            Gp: float = 3.2,
                            lam: float = 0.7,
                            g_x: float = 8.0,
                            outer_steps: int = 100,
                            fd_eps_list: Tuple[float, ...] = (1e-3, 1e-4, 1e-5),
                            gate_rel_tol: float = 0.01,
                            ) -> Dict[str, Any]:
    """Reverse-mode AD vs centered FD for `dL/dalpha` on the constriction.

    Deviates on **one axis only** (family-extension Sec.4.3 / the Sec.3.2
    cautionary tale): the model is Giesekus, but geometry / precision
    (float64) / `solver_tol` are the validated constriction values. To
    give `alpha` a signal above FD noise we use an elastic truth point
    (`Gp`, `g_x`, `outer_steps` raised together as the *operating
    point*, not the numerics) -- at near-Newtonian settings `dL/dalpha` is
    genuinely ~0 (same conditioning story as `lam`, Sec.6).
    """
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(p3b.DEFAULT_MULTISTEP_AD_FD_CONFIG)
    cfg['g_x'] = g_x
    cfg['outer_steps'] = outer_steps
    if config:
        cfg.update(config)

    grid, model, init_state, perm_f = _build_constriction(
        cfg, 'giesekus_logconf_bk_v2')
    fixed = dict(Gp=jnp.asarray(Gp, dtype=jnp.float64),
                 lam=jnp.asarray(lam, dtype=jnp.float64))
    loss = _constriction_loss_fn(cfg, init_state, model, grid, perm_f,
                                 fixed_params=fixed, diff_keys=('alpha',))
    loss_jit = jax.jit(loss)
    vg = jax.jit(jax.value_and_grad(loss))

    a0 = {'alpha': jnp.asarray(alpha0, dtype=jnp.float64)}
    t0 = time.perf_counter()
    L0, g0 = vg(a0)
    ad = float(g0['alpha'])
    t_ad = time.perf_counter() - t0

    def _eval(alpha):
        return float(loss_jit({'alpha': jnp.asarray(alpha, dtype=jnp.float64)}))

    fd_at_eps = {}
    for eps in fd_eps_list:
        Lp = _eval(alpha0 + eps)
        Lm = _eval(alpha0 - eps)
        fd_at_eps[eps] = (Lp - Lm) / (2.0 * eps)
    eps_best, fd_best = min(fd_at_eps.items(),
                            key=lambda kv: abs(kv[1] - ad))
    rel_err = abs(ad - fd_best) / max(abs(fd_best), 1e-30)
    gate_pass = bool(np.isfinite(ad) and rel_err <= gate_rel_tol)

    print("=== Giesekus AD-vs-FD on dL/dalpha (constriction) ===")
    print(f"  truth point: Gp={Gp}, lam={lam}, alpha={alpha0}, g_x={g_x}, "
          f"outer_steps={outer_steps}  (T={outer_steps*cfg['inner_steps']*cfg['dt']:.3g}"
          f" = {outer_steps*cfg['inner_steps']*cfg['dt']/lam:.3g}lam)")
    print(f"  loss = {float(L0):.6e}   (AD warm {t_ad:.1f}s)")
    print(f"  AD   dL/dalpha = {ad:+.6e}")
    for eps in fd_eps_list:
        print(f"  FD eps={eps:.0e}  dL/dalpha = {fd_at_eps[eps]:+.6e}")
    print(f"  best FD eps={eps_best:.0e}  value={fd_best:+.6e}  "
          f"rel err vs AD = {100*rel_err:.4f}%")
    print(f"  GATE (rel_tol={gate_rel_tol:.0%}): "
          f"{'PASS' if gate_pass else 'FAIL'}")
    return dict(ad=ad, fd_at_eps=fd_at_eps, eps_best=eps_best,
                fd_best=fd_best, rel_err=rel_err, gate_pass=gate_pass,
                loss=float(L0))


# ===========================================================================
# Sec.4.2  Physics gate -- Couette Wi sweep vs the steady-shear reference
# ===========================================================================

def _couette_steady_measure(cfg, model_name, polymer_params, margin=4,
                            settle_fraction=0.7):
    """Run the Couette solver and return time-settled bulk A / N1 / tau_xy / N2."""
    grid = p3b._build_grid(cfg['Nx'], cfg['Ny'], cfg['Lx'], cfg['Ly'])
    model = cr.get_model(model_name)
    init_state = p3b._build_wall_bounded_initial_state(
        grid, cfg['U_wall'], model, 'extrapolation')
    out = p3b._evolve_wall_bounded_with_diagnostics(
        initial_state=init_state, model=model, polymer_params=polymer_params,
        grid=grid, density=cfg['density'], base_viscosity=cfg['nu_s'],
        dt=cfg['dt'], inner_steps=cfg['inner_steps'],
        outer_steps=cfg['outer_steps'],
        solver_type=cfg['solver_type'],
        use_preconditioner=cfg['use_preconditioner'],
        preconditioner_type=cfg['preconditioner_type'],
        pressure_gradient=(0.0, 0.0))
    n_settle = int(settle_fraction * cfg['outer_steps'])
    # Trajectories: (T, Nx, Ny). Bulk = interior y-rows, time-settled.
    def _bulk(traj):
        t = np.asarray(traj)[n_settle:, :, margin:-margin]
        return float(t.mean())
    A_xx = _bulk(out['A_xx_traj'])
    A_xy = _bulk(out['A_xy_traj'])
    A_yy = _bulk(out['A_yy_traj'])
    A_zz = _bulk(out['A_zz_traj'])
    # N1 / tau_xy from the model's OWN stress readout (correct for the
    # non-Hookean FENE-P; identical to Gp(A_xx-A_yy) for the linear
    # readout models). tau_zz is not in the trajectory => no N2 here.
    N1 = _bulk(np.asarray(out['tau_xx_traj']) - np.asarray(out['tau_yy_traj']))
    tau_xy = _bulk(out['tau_xy_traj'])
    min_lam = float(np.asarray(out['min_lam_traj'])[n_settle:].min())
    any_nan = bool(np.asarray(out['any_nan_traj']).any())
    return dict(A_xx=A_xx, A_xy=A_xy, A_yy=A_yy, A_zz=A_zz,
                trA=A_xx + A_yy + A_zz,
                N1=N1, tau_xy=tau_xy, min_lam=min_lam, any_nan=any_nan)


def giesekus_couette_physics(alpha: float = 0.3,
                             lam_list: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
                             config: Optional[Dict[str, Any]] = None,
                             rel_tol: float = 0.06,
                             ) -> Dict[str, Any]:
    """Couette Wi sweep for Giesekus, solver vs steady-shear reference.

    For each lam (=> Wi = lam.gammadot, gammadot = U_wall/Ly fixed): run the full
    BE-IMEX Couette solver with the Giesekus model and compare the
    measured steady bulk `N1` / `tau_xy` to the shared Newton-root
    reference (`steady_simple_shear_reference`). Also reports the
    Giesekus signatures the reference must show: shear-thinning `eta_p`
    (where Oldroyd-B is flat) and the high-Wi `-N2/N1 -> alpha/2` asymptote.
    Independent textbook cross-check is the alpha->0 limit (reference must
    reproduce Oldroyd-B `N1 = 2 Gp lam^2 gammadot^2`) plus the alpha/2 asymptote.
    """
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(p3b.DEFAULT_COUETTE_CONFIG)
    if config:
        cfg.update(config)
    gammadot = cfg['U_wall'] / cfg['Ly']
    Gp = cfg['Gp']

    rows = []
    A_warm = None
    for lam in lam_list:
        c = dict(cfg)
        c['lam'] = lam
        c['outer_steps'] = int(round(cfg['outer_steps'] * lam))  # T = 5lam
        polymer_params = dict(Gp=jnp.asarray(Gp, dtype=jnp.float64),
                              lam=jnp.asarray(lam, dtype=jnp.float64),
                              alpha=jnp.asarray(alpha, dtype=jnp.float64))
        t0 = time.perf_counter()
        meas = _couette_steady_measure(c, 'giesekus_logconf_bk_v2',
                                       polymer_params)
        t_run = time.perf_counter() - t0
        ref = steady_simple_shear_reference(
            giesekus_source_R, gammadot, lam, {'alpha': alpha}, Gp=Gp,
            A_init=A_warm)
        A_warm = ref['a']
        rel_N1 = abs(meas['N1'] - ref['N1']) / max(abs(ref['N1']), 1e-30)
        rel_txy = abs(meas['tau_xy'] - ref['tau_xy']) / max(abs(ref['tau_xy']), 1e-30)
        rows.append(dict(lam=lam, Wi=lam * gammadot,
                         N1_meas=meas['N1'], N1_ref=ref['N1'], rel_N1=rel_N1,
                         txy_meas=meas['tau_xy'], txy_ref=ref['tau_xy'],
                         rel_txy=rel_txy, N2_ref=ref['N2'],
                         eta_p_ref=ref['eta_p'],
                         # Relative polymer viscosity eta_p/eta_p0, eta_p0 = Gp.lam
                         # (the OB plateau). This is F(Wi); shear-thinning =>
                         # decreasing. Sweeping lam at fixed gammadot probes F(Wi)
                         # the same as a gammadot sweep would, once normalised.
                         eta_rel_ref=ref['eta_p'] / (Gp * lam),
                         min_lam=meas['min_lam'],
                         any_nan=meas['any_nan'], t_run=t_run,
                         ref_resid=ref['residual']))

    # Reference-side signatures.
    # (a) alpha->0 reproduces Oldroyd-B N1 = 2 Gp lam^2 gammadot^2 exactly.
    lam_chk = lam_list[-1]
    ref_a0 = steady_simple_shear_reference(
        giesekus_source_R, gammadot, lam_chk, {'alpha': 0.0}, Gp=Gp)
    N1_ob = 2.0 * Gp * lam_chk ** 2 * gammadot ** 2
    rel_ob_limit = abs(ref_a0['N1'] - N1_ob) / N1_ob
    # (b) zero-shear asymptote -N2/N1 -> alpha/2 (Giesekus's -Psi2/Psi1|0).
    lo = steady_simple_shear_reference(
        giesekus_source_R, 0.01, lam_chk, {'alpha': alpha}, Gp=Gp)
    ratio_lo = -lo['N2'] / lo['N1']
    asymptote = alpha / 2.0

    print(f"=== Giesekus Couette Wi sweep (alpha={alpha}) -- solver vs reference ===")
    print(f"{'lam':>5} {'Wi':>5} {'N1_meas':>10} {'N1_ref':>10} {'relN1':>7} "
          f"{'etap/Gplam':>8} {'N2_ref':>9} {'minLambda':>7}")
    for r in rows:
        flag = 'PASS' if (r['rel_N1'] < rel_tol and not r['any_nan']) else 'fail'
        print(f"{r['lam']:>5.2f} {r['Wi']:>5.2f} {r['N1_meas']:>10.4f} "
              f"{r['N1_ref']:>10.4f} {100*r['rel_N1']:>6.2f}% "
              f"{r['eta_rel_ref']:>8.4f} {r['N2_ref']:>9.4f} "
              f"{r['min_lam']:>7.3f}  {flag}")
    # Shear-thinning: relative viscosity eta_p/(Gp.lam) = F(Wi) decreases with
    # Wi (OB is flat at F==1). Absolute eta_p rises with lam here only because
    # the OB plateau eta_p0 = Gp.lam scales with lam -- normalise it out.
    eta_rel = [r['eta_rel_ref'] for r in rows]
    shear_thins = all(eta_rel[i] >= eta_rel[i + 1] - 1e-9
                      for i in range(len(eta_rel) - 1)) and eta_rel[0] > eta_rel[-1]
    print(f"\n  reference alpha->0 limit:  N1_ref={ref_a0['N1']:.4f}  "
          f"OB N1={N1_ob:.4f}  rel={100*rel_ob_limit:.3f}%  "
          f"(reproduces Oldroyd-B)")
    print(f"  zero-shear (Wi={0.01*lam_chk:.3f}) N2/N1 = {ratio_lo:.4f}  "
          f"-> alpha/2 = {asymptote:.4f}  (Giesekus 2/1|0 signature)")
    print(f"  shear-thinning eta_p/(Gp.lam)=F(Wi) (OB flat at 1.0): "
          f"{eta_rel[0]:.4f} -> {eta_rel[-1]:.4f}  ({'YES' if shear_thins else 'NO'})")

    all_pts = all(r['rel_N1'] < rel_tol and not r['any_nan'] for r in rows)
    ob_limit_ok = rel_ob_limit < 1e-6
    asymptote_ok = abs(ratio_lo - asymptote) < 0.02
    gate_pass = bool(all_pts and ob_limit_ok and asymptote_ok and shear_thins)
    print(f"\n  GATE: solver-vs-ref within {rel_tol:.0%}={all_pts}  "
          f"OB-limit={ob_limit_ok}  alpha/2-asymptote={asymptote_ok}  "
          f"shear-thinning={shear_thins}  ->  "
          f"{'PASS' if gate_pass else 'FAIL'}")
    return dict(rows=rows, rel_ob_limit=rel_ob_limit, ratio_lo=ratio_lo,
                asymptote=asymptote, shear_thins=shear_thins,
                gate_pass=gate_pass)


# ===========================================================================
# Sec.5. Convenience: run all three Giesekus gates
# ===========================================================================

def run_all_giesekus_gates() -> Dict[str, Any]:
    """Run the Sec.4.1 regression, Sec.4.3 AD-vs-FD(alpha), and Sec.4.2 physics gates."""
    reg = giesekus_regression_gate()
    print()
    adfd = giesekus_alpha_ad_vs_fd()
    print()
    phys = giesekus_couette_physics()
    print()
    overall = bool(reg['gate_pass'] and adfd['gate_pass'] and phys['gate_pass'])
    print(f"########## GIESEKUS OVERALL: {'PASS' if overall else 'FAIL'} "
          f"(regression={reg['gate_pass']}, ad_vs_fd={adfd['gate_pass']}, "
          f"physics={phys['gate_pass']}) ##########")
    return dict(regression=reg, ad_vs_fd=adfd, physics=phys, overall=overall)


# ===========================================================================
# Sec.6.1  FENE-P regression gate -- Oldroyd-B asymptotic limit (O(1/L^2))
# ===========================================================================

def fene_p_regression_gate(config: Optional[Dict[str, Any]] = None,
                           Lsq1: float = 1e6,
                           Lsq2: float = 2e6,
                           tol_limit: float = 1e-3,
                           rate_lo: float = 1.5,
                           rate_hi: float = 2.5,
                           ) -> Dict[str, Any]:
    """As `L^2 -> inf`, FENE-P must approach `oldroyd_b_logconf_bk_v2`.

    Unlike Giesekus(alpha=0) (which is *exactly* Oldroyd-B), FENE-P->OB is
    asymptotic with `O(1/L^2)` error. So this is a **limit + rate** gate:
      * at `Lsq1 = 1e6` the forward loss / grads match OB to `tol_limit`;
      * halving `1/L^2` (`Lsq1 -> Lsq2 = 2e6`) **halves** the discrepancy
        (ratio  in  [rate_lo, rate_hi]), confirming the genuine `O(1/L^2)`
        limit rather than a coincidental near-match.
    """
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(p3b.DEFAULT_MULTISTEP_AD_FD_CONFIG)
    if config:
        cfg.update(config)
    p = dict(Gp=jnp.asarray(cfg['Gp_init'], dtype=jnp.float64),
             lam=jnp.asarray(cfg['lam_init'], dtype=jnp.float64))

    grid, ob_model, ob_state, ob_perm = _build_constriction(
        cfg, 'oldroyd_b_logconf_bk_v2')
    ob_loss = _constriction_loss_fn(cfg, ob_state, ob_model, grid, ob_perm,
                                    fixed_params={}, diff_keys=('Gp', 'lam'))
    L_ob, g_ob = jax.jit(jax.value_and_grad(ob_loss))(p)
    L_ob = float(L_ob)
    g_ob = {k: float(v) for k, v in g_ob.items()}

    grid_f, f_model, f_state, f_perm = _build_constriction(
        cfg, 'fene_p_logconf_bk_v2')

    def _fene_at(Lsq):
        loss = _constriction_loss_fn(
            cfg, f_state, f_model, grid_f, f_perm,
            fixed_params={'Lsq': jnp.asarray(Lsq, dtype=jnp.float64)},
            diff_keys=('Gp', 'lam'))
        L, g = jax.jit(jax.value_and_grad(loss))(p)
        return float(L), {k: float(v) for k, v in g.items()}

    L1, g1 = _fene_at(Lsq1)
    L2, g2 = _fene_at(Lsq2)

    def _rel(a, b):
        return abs(a - b) / max(abs(b), 1e-30)

    e1 = _rel(L1, L_ob)
    e2 = _rel(L2, L_ob)
    e1_Gp = _rel(g1['Gp'], g_ob['Gp'])
    e2_Gp = _rel(g2['Gp'], g_ob['Gp'])
    ratio = e1 / max(e2, 1e-30)
    ratio_Gp = e1_Gp / max(e2_Gp, 1e-30)

    limit_ok = e1 < tol_limit
    rate_ok = rate_lo < ratio < rate_hi
    gate_pass = bool(limit_ok and rate_ok)

    print("=== FENE-P L^2->inf regression vs oldroyd_b_logconf_bk_v2 ===")
    print(f"  OB loss = {L_ob:.10e}")
    print(f"  L^2={Lsq1:.0e}:  loss={L1:.10e}  |Delta|/L_ob={e1:.2e}   "
          f"dGp rel={e1_Gp:.2e}")
    print(f"  L^2={Lsq2:.0e}:  loss={L2:.10e}  |Delta|/L_ob={e2:.2e}   "
          f"dGp rel={e2_Gp:.2e}")
    print(f"  O(1/L^2) rate: loss-err ratio={ratio:.3f}  dGp-err ratio="
          f"{ratio_Gp:.3f}  (expect ~= 2)")
    print(f"  GATE: limit(<{tol_limit:.0e})={limit_ok}  "
          f"rate in [{rate_lo},{rate_hi}]={rate_ok}  ->  "
          f"{'PASS' if gate_pass else 'FAIL'}")
    return dict(L_ob=L_ob, L1=L1, L2=L2, e1=e1, e2=e2, ratio=ratio,
                ratio_Gp=ratio_Gp, limit_ok=limit_ok, rate_ok=rate_ok,
                gate_pass=gate_pass)


# ===========================================================================
# Sec.6.2  FENE-P gradient gate -- AD vs FD on the new `Lsq` partial
# ===========================================================================

def fene_p_lsq_ad_vs_fd(config: Optional[Dict[str, Any]] = None,
                        Lsq0: float = 50.0,
                        Gp: float = 3.2,
                        lam: float = 0.7,
                        g_x: float = 8.0,
                        outer_steps: int = 100,
                        fd_frac_list: Tuple[float, ...] = (1e-2, 1e-3, 1e-4),
                        gate_rel_tol: float = 0.01,
                        ) -> Dict[str, Any]:
    """Reverse-mode AD vs centered FD for `dL/d(L^2)` on the constriction.

    One-axis deviation (family-extension Sec.4.3): Giesekus -> FENE-P model
    swap, validated constriction geometry / float64 / `tol=1e-12`. `L^2`
    is chosen *moderate* (finite-extensibility active) so the partial
    has signal -- too large => `dL/dL^2 -> 0` (the OB plateau, same
    weak-signal trap as `lam`/large-`Lsq`). FD steps are **relative**
    (`eps = frac.L^2`) since `L^2` is `O(10-100)`.
    """
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(p3b.DEFAULT_MULTISTEP_AD_FD_CONFIG)
    cfg['g_x'] = g_x
    cfg['outer_steps'] = outer_steps
    if config:
        cfg.update(config)

    grid, model, init_state, perm_f = _build_constriction(
        cfg, 'fene_p_logconf_bk_v2')
    fixed = dict(Gp=jnp.asarray(Gp, dtype=jnp.float64),
                 lam=jnp.asarray(lam, dtype=jnp.float64))
    loss = _constriction_loss_fn(cfg, init_state, model, grid, perm_f,
                                 fixed_params=fixed, diff_keys=('Lsq',))
    loss_jit = jax.jit(loss)
    vg = jax.jit(jax.value_and_grad(loss))

    g0 = {'Lsq': jnp.asarray(Lsq0, dtype=jnp.float64)}
    t0 = time.perf_counter()
    L0, grad0 = vg(g0)
    ad = float(grad0['Lsq'])
    t_ad = time.perf_counter() - t0

    def _eval(Lsq):
        return float(loss_jit({'Lsq': jnp.asarray(Lsq, dtype=jnp.float64)}))

    fd_at = {}
    for frac in fd_frac_list:
        eps = frac * Lsq0
        fd_at[frac] = (_eval(Lsq0 + eps) - _eval(Lsq0 - eps)) / (2.0 * eps)
    frac_best, fd_best = min(fd_at.items(), key=lambda kv: abs(kv[1] - ad))
    rel_err = abs(ad - fd_best) / max(abs(fd_best), 1e-30)
    gate_pass = bool(np.isfinite(ad) and rel_err <= gate_rel_tol)

    print("=== FENE-P AD-vs-FD on dL/d(L^2) (constriction) ===")
    print(f"  truth point: Gp={Gp}, lam={lam}, Lsq={Lsq0}, g_x={g_x}, "
          f"outer_steps={outer_steps}")
    print(f"  loss = {float(L0):.6e}   (AD warm {t_ad:.1f}s)")
    print(f"  AD   dL/dL^2 = {ad:+.6e}")
    for frac in fd_frac_list:
        print(f"  FD eps={frac:.0e}.L^2  dL/dL^2 = {fd_at[frac]:+.6e}")
    print(f"  best FD eps={frac_best:.0e}.L^2  value={fd_best:+.6e}  "
          f"rel err vs AD = {100*rel_err:.4f}%")
    print(f"  GATE (rel_tol={gate_rel_tol:.0%}): "
          f"{'PASS' if gate_pass else 'FAIL'}")
    return dict(ad=ad, fd_at=fd_at, frac_best=frac_best, fd_best=fd_best,
                rel_err=rel_err, gate_pass=gate_pass, loss=float(L0))


# ===========================================================================
# Sec.6.3  FENE-P physics gate -- Couette Wi sweep vs the steady-shear reference
# ===========================================================================

def fene_p_couette_physics(Lsq: float = 50.0,
                           lam_list: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
                           config: Optional[Dict[str, Any]] = None,
                           rel_tol: float = 0.06,
                           ) -> Dict[str, Any]:
    """Couette Wi sweep for FENE-P, solver vs steady-shear reference.

    Same structure as the Giesekus physics gate, with FENE-P's own
    `R(A)` and (non-Hookean) viscometric readout. Cross-checks the
    FENE-P signatures the reference must show:
      * **OB limit:** `L^2 -> inf` reproduces `N1 = 2 Gp lam^2 gammadot^2`;
      * **A_zz coupling:** the steady root satisfies `A_zz = a/f`
        exactly (machine precision) -- validates the trace coupling that
        the 4th component exists for;
      * **finite-extensibility cap:** `tr A < L^2` always, rising toward
        `L^2` with Wi (OB/Giesekus have unbounded `tr A`);
      * **shear-thinning:** `eta_p/(Gplam)` decreasing.
    """
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(p3b.DEFAULT_COUETTE_CONFIG)
    if config:
        cfg.update(config)
    gammadot = cfg['U_wall'] / cfg['Ly']
    Gp = cfg['Gp']
    a_const = Lsq / (Lsq - 3.0)

    rows = []
    A_warm = None
    for lam in lam_list:
        c = dict(cfg)
        c['lam'] = lam
        c['outer_steps'] = int(round(cfg['outer_steps'] * lam))  # T = 5lam
        polymer_params = dict(Gp=jnp.asarray(Gp, dtype=jnp.float64),
                              lam=jnp.asarray(lam, dtype=jnp.float64),
                              Lsq=jnp.asarray(Lsq, dtype=jnp.float64))
        t0 = time.perf_counter()
        meas = _couette_steady_measure(c, 'fene_p_logconf_bk_v2',
                                       polymer_params)
        t_run = time.perf_counter() - t0
        ref = steady_simple_shear_reference(
            fene_p_source_R, gammadot, lam, {'Lsq': Lsq}, Gp=Gp,
            viscometric_fn=fene_p_viscometric, A_init=A_warm)
        A_warm = ref['a']
        rel_N1 = abs(meas['N1'] - ref['N1']) / max(abs(ref['N1']), 1e-30)
        # A_zz = a/f self-consistency at the steady root (R_zz = 0).
        azz_consistency = abs(ref['A_zz'] - a_const / float(
            lc._fene_p_peterlin_f(ref['trA'], Lsq)))
        rows.append(dict(lam=lam, Wi=lam * gammadot,
                         N1_meas=meas['N1'], N1_ref=ref['N1'], rel_N1=rel_N1,
                         txy_meas=meas['tau_xy'], txy_ref=ref['tau_xy'],
                         N2_ref=ref['N2'], eta_rel_ref=ref['eta_p'] / (Gp * lam),
                         A_zz_ref=ref['A_zz'], trA_ref=ref['trA'],
                         azz_consistency=azz_consistency,
                         min_lam=meas['min_lam'], any_nan=meas['any_nan'],
                         t_run=t_run, ref_resid=ref['residual']))

    # OB limit: large L^2 reproduces N1 = 2 Gp lam^2 gammadot^2.
    lam_chk = lam_list[-1]
    ref_ob = steady_simple_shear_reference(
        fene_p_source_R, gammadot, lam_chk, {'Lsq': 1e6}, Gp=Gp,
        viscometric_fn=fene_p_viscometric)
    N1_ob = 2.0 * Gp * lam_chk ** 2 * gammadot ** 2
    rel_ob_limit = abs(ref_ob['N1'] - N1_ob) / N1_ob

    print(f"=== FENE-P Couette Wi sweep (L^2={Lsq}) -- solver vs reference ===")
    print(f"{'lam':>5} {'Wi':>5} {'N1_meas':>10} {'N1_ref':>10} {'relN1':>7} "
          f"{'etap/Gplam':>8} {'A_zz':>7} {'trA':>7} {'minLambda':>7}")
    for r in rows:
        flag = 'PASS' if (r['rel_N1'] < rel_tol and not r['any_nan']) else 'fail'
        print(f"{r['lam']:>5.2f} {r['Wi']:>5.2f} {r['N1_meas']:>10.4f} "
              f"{r['N1_ref']:>10.4f} {100*r['rel_N1']:>6.2f}% "
              f"{r['eta_rel_ref']:>8.4f} {r['A_zz_ref']:>7.4f} "
              f"{r['trA_ref']:>7.3f} {r['min_lam']:>7.3f}  {flag}")

    eta_rel = [r['eta_rel_ref'] for r in rows]
    shear_thins = all(eta_rel[i] >= eta_rel[i + 1] - 1e-9
                      for i in range(len(eta_rel) - 1)) and eta_rel[0] > eta_rel[-1]
    trA = [r['trA_ref'] for r in rows]
    trace_cap_ok = all(t < Lsq for t in trA) and trA[-1] > trA[0]
    azz_ok = max(r['azz_consistency'] for r in rows) < 1e-9
    azz_contracts = all(r['A_zz_ref'] < 1.0 for r in rows)

    print(f"\n  OB limit (L^2=1e6):  N1_ref={ref_ob['N1']:.4f}  "
          f"OB N1={N1_ob:.4f}  rel={100*rel_ob_limit:.3f}%")
    print(f"  A_zz = a/f self-consistency (max |Delta|): "
          f"{max(r['azz_consistency'] for r in rows):.2e}  "
          f"(A_zz<1 contracts: {azz_contracts})")
    print(f"  finite-extensibility: tr A  in  [{trA[0]:.2f}, {trA[-1]:.2f}] "
          f"< L^2={Lsq}  (rises with Wi: {trace_cap_ok})")
    print(f"  shear-thinning eta_p/(Gp.lam)=F(Wi): "
          f"{eta_rel[0]:.4f} -> {eta_rel[-1]:.4f}  ({'YES' if shear_thins else 'NO'})")

    all_pts = all(r['rel_N1'] < rel_tol and not r['any_nan'] for r in rows)
    ob_limit_ok = rel_ob_limit < 1e-4
    gate_pass = bool(all_pts and ob_limit_ok and azz_ok and azz_contracts
                     and trace_cap_ok and shear_thins)
    print(f"\n  GATE: solver-vs-ref<{rel_tol:.0%}={all_pts}  OB-limit={ob_limit_ok}"
          f"  A_zz=a/f={azz_ok}  trace-cap={trace_cap_ok}  "
          f"shear-thinning={shear_thins}  ->  {'PASS' if gate_pass else 'FAIL'}")
    return dict(rows=rows, rel_ob_limit=rel_ob_limit, shear_thins=shear_thins,
                trace_cap_ok=trace_cap_ok, azz_ok=azz_ok,
                azz_contracts=azz_contracts, gate_pass=gate_pass)


def fene_p_wall_stress_test(Lsq: float = 10.0,
                            lam_list: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
                            config: Optional[Dict[str, Any]] = None,
                            ) -> Dict[str, Any]:
    """Push the **full Couette solver** toward the finite-extension wall.

    Tight `L^2` + escalating Wi drives `tr A -> L^2`, where the Peterlin
    factor `f` (and the relaxation stiffness) blows up. This is a
    *diagnostic* sweep (not a pass/fail gate): for each Wi it reports the
    solver's bulk `tr A / L^2`, `N1` vs the steady-shear reference, and
    the solver-health flags (`minLambda`, `any_nan`). It answers "does the
    BE-IMEX log-conformation solver stay SPD/finite as the chains
    saturate, and does it still track the analytic root?".

    If the solver strains at the highest Wi, the lever is `outer_steps`
    (longer settle) or smaller `dt` (the config), not a model change --
    the reference is exact and shows the target.
    """
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(p3b.DEFAULT_COUETTE_CONFIG)
    if config:
        cfg.update(config)
    gammadot = cfg['U_wall'] / cfg['Ly']
    Gp = cfg['Gp']

    rows = []
    A_warm = None
    for lam in lam_list:
        c = dict(cfg)
        c['lam'] = lam
        c['outer_steps'] = int(round(cfg['outer_steps'] * lam))  # T = 5lam
        polymer_params = dict(Gp=jnp.asarray(Gp, dtype=jnp.float64),
                              lam=jnp.asarray(lam, dtype=jnp.float64),
                              Lsq=jnp.asarray(Lsq, dtype=jnp.float64))
        t0 = time.perf_counter()
        meas = _couette_steady_measure(c, 'fene_p_logconf_bk_v2',
                                       polymer_params)
        t_run = time.perf_counter() - t0
        ref = steady_simple_shear_reference(
            fene_p_source_R, gammadot, lam, {'Lsq': Lsq}, Gp=Gp,
            viscometric_fn=fene_p_viscometric, A_init=A_warm)
        A_warm = ref['a']
        rel_N1 = abs(meas['N1'] - ref['N1']) / max(abs(ref['N1']), 1e-30)
        rel_trA = abs(meas['trA'] - ref['trA']) / max(abs(ref['trA']), 1e-30)
        rows.append(dict(lam=lam, Wi=lam * gammadot,
                         trA_meas=meas['trA'], trA_ref=ref['trA'],
                         fill_meas=meas['trA'] / Lsq, fill_ref=ref['trA'] / Lsq,
                         rel_trA=rel_trA,
                         N1_meas=meas['N1'], N1_ref=ref['N1'], rel_N1=rel_N1,
                         A_zz_meas=meas['A_zz'], A_zz_ref=ref['A_zz'],
                         min_lam=meas['min_lam'], any_nan=meas['any_nan'],
                         t_run=t_run, ref_resid=ref['residual']))

    print(f"=== FENE-P near-wall solver stress test (L^2={Lsq}) ===")
    print(f"{'Wi':>5} {'trA_meas':>9} {'trA_ref':>9} {'trA/L^2':>7} "
          f"{'relTrA':>7} {'N1_meas':>9} {'N1_ref':>9} {'relN1':>7} "
          f"{'minLambda':>7} {'nan':>4}")
    for r in rows:
        print(f"{r['Wi']:>5.1f} {r['trA_meas']:>9.4f} {r['trA_ref']:>9.4f} "
              f"{r['fill_ref']:>7.3f} {100*r['rel_trA']:>6.2f}% "
              f"{r['N1_meas']:>9.4f} {r['N1_ref']:>9.4f} "
              f"{100*r['rel_N1']:>6.2f}% {r['min_lam']:>7.3f} "
              f"{str(r['any_nan']):>4}")
    healthy = all((not r['any_nan']) and r['min_lam'] > 0 for r in rows)
    print(f"\n  solver healthy (SPD, no NaN) at every Wi: {healthy}")
    print(f"  max trA/L^2 reached: {max(r['fill_ref'] for r in rows):.3f} "
          f"(reference)  {max(r['fill_meas'] for r in rows):.3f} (solver)")
    return dict(rows=rows, Lsq=Lsq, healthy=healthy)


def run_all_fene_p_gates() -> Dict[str, Any]:
    """Run the Sec.6.1 regression, Sec.6.2 AD-vs-FD(L^2), and Sec.6.3 physics gates."""
    reg = fene_p_regression_gate()
    print()
    adfd = fene_p_lsq_ad_vs_fd()
    print()
    phys = fene_p_couette_physics()
    print()
    overall = bool(reg['gate_pass'] and adfd['gate_pass'] and phys['gate_pass'])
    print(f"########## FENE-P OVERALL: {'PASS' if overall else 'FAIL'} "
          f"(regression={reg['gate_pass']}, ad_vs_fd={adfd['gate_pass']}, "
          f"physics={phys['gate_pass']}) ##########")
    return dict(regression=reg, ad_vs_fd=adfd, physics=phys, overall=overall)


# ===========================================================================
# Sec.7.1  Linear-PTT regression gate -- PTT(eps=0) == Oldroyd-B (machine precision)
# ===========================================================================

def ptt_regression_gate(config: Optional[Dict[str, Any]] = None,
                        rtol: float = 1e-6,
                        ) -> Dict[str, Any]:
    """At eps=0, linear PTT must reproduce `oldroyd_b_logconf_bk_v2`.

    Like Giesekus(alpha=0), eps=0 => f==1 => the affine exponential integrator is
    the *exact* Oldroyd-B analytic exponential, so forward loss and
    `dL/dGp`, `dL/dlam` agree to machine precision (~1e-12 / FD-noise).
    """
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(p3b.DEFAULT_MULTISTEP_AD_FD_CONFIG)
    if config:
        cfg.update(config)
    p = dict(Gp=jnp.asarray(cfg['Gp_init'], dtype=jnp.float64),
             lam=jnp.asarray(cfg['lam_init'], dtype=jnp.float64))

    grid, ob_model, ob_state, ob_perm = _build_constriction(
        cfg, 'oldroyd_b_logconf_bk_v2')
    ob_loss = _constriction_loss_fn(cfg, ob_state, ob_model, grid, ob_perm,
                                    fixed_params={}, diff_keys=('Gp', 'lam'))
    L_ob, g_ob = jax.jit(jax.value_and_grad(ob_loss))(p)
    L_ob = float(L_ob)
    g_ob = {k: float(v) for k, v in g_ob.items()}

    grid_p, p_model, p_state, p_perm = _build_constriction(
        cfg, 'ptt_linear_logconf_bk_v2')
    p_loss = _constriction_loss_fn(
        cfg, p_state, p_model, grid_p, p_perm,
        fixed_params={'epsilon': jnp.asarray(0.0, dtype=jnp.float64)},
        diff_keys=('Gp', 'lam'))
    L_p, g_p = jax.jit(jax.value_and_grad(p_loss))(p)
    L_p = float(L_p)
    g_p = {k: float(v) for k, v in g_p.items()}

    def _rel(a, b):
        return abs(a - b) / max(abs(b), 1e-30)

    rel_loss = _rel(L_p, L_ob)
    rel_Gp = _rel(g_p['Gp'], g_ob['Gp'])
    rel_lam = _rel(g_p['lam'], g_ob['lam'])
    gate_pass = bool(rel_loss < rtol and rel_Gp < rtol and rel_lam < rtol)

    print("=== linear PTT eps=0 regression vs oldroyd_b_logconf_bk_v2 ===")
    print(f"  forward loss : OB = {L_ob:.10e}   PTT(eps=0) = {L_p:.10e}"
          f"   rel = {rel_loss:.2e}")
    print(f"  dL/dGp       : OB = {g_ob['Gp']:.10e}   PTT = {g_p['Gp']:.10e}"
          f"   rel = {rel_Gp:.2e}")
    print(f"  dL/dlam      : OB = {g_ob['lam']:.10e}   PTT = {g_p['lam']:.10e}"
          f"   rel = {rel_lam:.2e}")
    print(f"  GATE (rtol={rtol:.0e}): {'PASS' if gate_pass else 'FAIL'}")
    return dict(L_ob=L_ob, L_p=L_p, g_ob=g_ob, g_p=g_p,
                rel_loss=rel_loss, rel_Gp=rel_Gp, rel_lam=rel_lam,
                gate_pass=gate_pass)


# ===========================================================================
# Sec.7.2  Linear-PTT gradient gate -- AD vs FD on the new `epsilon` partial
# ===========================================================================

def ptt_epsilon_ad_vs_fd(config: Optional[Dict[str, Any]] = None,
                         eps0: float = 0.15,
                         Gp: float = 3.2,
                         lam: float = 0.7,
                         g_x: float = 8.0,
                         outer_steps: int = 100,
                         fd_eps_list: Tuple[float, ...] = (1e-3, 1e-4, 1e-5),
                         gate_rel_tol: float = 0.01,
                         ) -> Dict[str, Any]:
    """Reverse-mode AD vs centered FD for `dL/deps` on the constriction.

    One-axis deviation (family-extension Sec.4.3): FENE-P -> PTT model swap,
    validated constriction geometry / float64 / `tol=1e-12`. `eps` is
    chosen *moderate* (PTT thinning active) so the partial has signal.
    """
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(p3b.DEFAULT_MULTISTEP_AD_FD_CONFIG)
    cfg['g_x'] = g_x
    cfg['outer_steps'] = outer_steps
    if config:
        cfg.update(config)

    grid, model, init_state, perm_f = _build_constriction(
        cfg, 'ptt_linear_logconf_bk_v2')
    fixed = dict(Gp=jnp.asarray(Gp, dtype=jnp.float64),
                 lam=jnp.asarray(lam, dtype=jnp.float64))
    loss = _constriction_loss_fn(cfg, init_state, model, grid, perm_f,
                                 fixed_params=fixed, diff_keys=('epsilon',))
    loss_jit = jax.jit(loss)
    vg = jax.jit(jax.value_and_grad(loss))

    g0 = {'epsilon': jnp.asarray(eps0, dtype=jnp.float64)}
    t0 = time.perf_counter()
    L0, grad0 = vg(g0)
    ad = float(grad0['epsilon'])
    t_ad = time.perf_counter() - t0

    def _eval(eps):
        return float(loss_jit({'epsilon': jnp.asarray(eps, dtype=jnp.float64)}))

    fd_at_eps = {}
    for eps in fd_eps_list:
        fd_at_eps[eps] = (_eval(eps0 + eps) - _eval(eps0 - eps)) / (2.0 * eps)
    eps_best, fd_best = min(fd_at_eps.items(), key=lambda kv: abs(kv[1] - ad))
    rel_err = abs(ad - fd_best) / max(abs(fd_best), 1e-30)
    gate_pass = bool(np.isfinite(ad) and rel_err <= gate_rel_tol)

    print("=== linear PTT AD-vs-FD on dL/deps (constriction) ===")
    print(f"  truth point: Gp={Gp}, lam={lam}, eps={eps0}, g_x={g_x}, "
          f"outer_steps={outer_steps}")
    print(f"  loss = {float(L0):.6e}   (AD warm {t_ad:.1f}s)")
    print(f"  AD   dL/deps = {ad:+.6e}")
    for eps in fd_eps_list:
        print(f"  FD eps_fd={eps:.0e}  dL/deps = {fd_at_eps[eps]:+.6e}")
    print(f"  best FD eps_fd={eps_best:.0e}  value={fd_best:+.6e}  "
          f"rel err vs AD = {100*rel_err:.4f}%")
    print(f"  GATE (rel_tol={gate_rel_tol:.0%}): "
          f"{'PASS' if gate_pass else 'FAIL'}")
    return dict(ad=ad, fd_at_eps=fd_at_eps, eps_best=eps_best,
                fd_best=fd_best, rel_err=rel_err, gate_pass=gate_pass,
                loss=float(L0))


# ===========================================================================
# Sec.7.3  Linear-PTT physics gate -- Couette Wi sweep vs steady-shear reference
# ===========================================================================

def ptt_couette_physics(epsilon: float = 0.25,
                        lam_list: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
                        config: Optional[Dict[str, Any]] = None,
                        rel_tol: float = 0.06,
                        ) -> Dict[str, Any]:
    """Couette Wi sweep for linear PTT, solver vs steady-shear reference.

    PTT shares the Hookean readout, so the reference uses
    `hookean_viscometric`. Cross-checks the PTT signatures:
      * **OB limit:** eps->0 reproduces `N1 = 2 Gp lam^2 gammadot^2`;
      * **shear-thinning:** `eta_p/(Gplam)` decreasing (PTT thins via f>1);
      * **N2 = 0:** linear PTT with zeta=0 has no second normal-stress diff;
      * **A_zz == 1:** the out-of-plane channel stays inert (the zz check;
        `tr A - 3 = A_xx + A_yy - 2`).
    """
    if not jax.config.read('jax_enable_x64'):
        raise RuntimeError("Enable float64 first (jax_enable_x64).")
    cfg = dict(p3b.DEFAULT_COUETTE_CONFIG)
    if config:
        cfg.update(config)
    gammadot = cfg['U_wall'] / cfg['Ly']
    Gp = cfg['Gp']

    rows = []
    A_warm = None
    for lam in lam_list:
        c = dict(cfg)
        c['lam'] = lam
        c['outer_steps'] = int(round(cfg['outer_steps'] * lam))  # T = 5lam
        polymer_params = dict(Gp=jnp.asarray(Gp, dtype=jnp.float64),
                              lam=jnp.asarray(lam, dtype=jnp.float64),
                              epsilon=jnp.asarray(epsilon, dtype=jnp.float64))
        t0 = time.perf_counter()
        meas = _couette_steady_measure(c, 'ptt_linear_logconf_bk_v2',
                                       polymer_params)
        t_run = time.perf_counter() - t0
        ref = steady_simple_shear_reference(
            ptt_source_R, gammadot, lam, {'epsilon': epsilon}, Gp=Gp,
            A_init=A_warm)
        A_warm = ref['a']
        rel_N1 = abs(meas['N1'] - ref['N1']) / max(abs(ref['N1']), 1e-30)
        rows.append(dict(lam=lam, Wi=lam * gammadot,
                         N1_meas=meas['N1'], N1_ref=ref['N1'], rel_N1=rel_N1,
                         txy_meas=meas['tau_xy'], txy_ref=ref['tau_xy'],
                         N2_ref=ref['N2'], eta_rel_ref=ref['eta_p'] / (Gp * lam),
                         A_zz_meas=meas['A_zz'], A_zz_ref=ref['A_zz'],
                         min_lam=meas['min_lam'], any_nan=meas['any_nan'],
                         t_run=t_run, ref_resid=ref['residual']))

    # OB limit: eps->0 reproduces N1 = 2 Gp lam^2 gammadot^2.
    lam_chk = lam_list[-1]
    ref_a0 = steady_simple_shear_reference(
        ptt_source_R, gammadot, lam_chk, {'epsilon': 0.0}, Gp=Gp)
    N1_ob = 2.0 * Gp * lam_chk ** 2 * gammadot ** 2
    rel_ob_limit = abs(ref_a0['N1'] - N1_ob) / N1_ob

    print(f"=== linear PTT Couette Wi sweep (eps={epsilon}) -- solver vs ref ===")
    print(f"{'lam':>5} {'Wi':>5} {'N1_meas':>10} {'N1_ref':>10} {'relN1':>7} "
          f"{'etap/Gplam':>8} {'N2_ref':>9} {'A_zz':>7} {'minLambda':>7}")
    for r in rows:
        flag = 'PASS' if (r['rel_N1'] < rel_tol and not r['any_nan']) else 'fail'
        print(f"{r['lam']:>5.2f} {r['Wi']:>5.2f} {r['N1_meas']:>10.4f} "
              f"{r['N1_ref']:>10.4f} {100*r['rel_N1']:>6.2f}% "
              f"{r['eta_rel_ref']:>8.4f} {r['N2_ref']:>9.2e} "
              f"{r['A_zz_ref']:>7.4f} {r['min_lam']:>7.3f}  {flag}")

    eta_rel = [r['eta_rel_ref'] for r in rows]
    shear_thins = all(eta_rel[i] >= eta_rel[i + 1] - 1e-9
                      for i in range(len(eta_rel) - 1)) and eta_rel[0] > eta_rel[-1]
    n2_zero = max(abs(r['N2_ref']) for r in rows) < 1e-9
    azz_inert = max(abs(r['A_zz_ref'] - 1.0) for r in rows) < 1e-9

    print(f"\n  eps->0 limit:  N1_ref={ref_a0['N1']:.4f}  OB N1={N1_ob:.4f}  "
          f"rel={100*rel_ob_limit:.3f}%  (reproduces Oldroyd-B)")
    print(f"  N2 = 0 (linear PTT, =0): max|N2_ref|="
          f"{max(abs(r['N2_ref']) for r in rows):.2e}  ({n2_zero})")
    print(f"  A_zz  1 (zz channel inert): max|A_zz1|="
          f"{max(abs(r['A_zz_ref'] - 1.0) for r in rows):.2e}  ({azz_inert})")
    print(f"  shear-thinning eta_p/(Gp.lam)=F(Wi): "
          f"{eta_rel[0]:.4f} -> {eta_rel[-1]:.4f}  ({'YES' if shear_thins else 'NO'})")

    all_pts = all(r['rel_N1'] < rel_tol and not r['any_nan'] for r in rows)
    ob_limit_ok = rel_ob_limit < 1e-6
    gate_pass = bool(all_pts and ob_limit_ok and n2_zero and azz_inert
                     and shear_thins)
    print(f"\n  GATE: solver-vs-ref<{rel_tol:.0%}={all_pts}  OB-limit={ob_limit_ok}"
          f"  N2=0={n2_zero}  A_zz1={azz_inert}  shear-thinning={shear_thins}"
          f"  ->  {'PASS' if gate_pass else 'FAIL'}")
    return dict(rows=rows, rel_ob_limit=rel_ob_limit, n2_zero=n2_zero,
                azz_inert=azz_inert, shear_thins=shear_thins,
                gate_pass=gate_pass)


def run_all_ptt_gates() -> Dict[str, Any]:
    """Run the Sec.7.1 regression, Sec.7.2 AD-vs-FD(eps), and Sec.7.3 physics gates."""
    reg = ptt_regression_gate()
    print()
    adfd = ptt_epsilon_ad_vs_fd()
    print()
    phys = ptt_couette_physics()
    print()
    overall = bool(reg['gate_pass'] and adfd['gate_pass'] and phys['gate_pass'])
    print(f"########## LINEAR PTT OVERALL: {'PASS' if overall else 'FAIL'} "
          f"(regression={reg['gate_pass']}, ad_vs_fd={adfd['gate_pass']}, "
          f"physics={phys['gate_pass']}) ##########")
    return dict(regression=reg, ad_vs_fd=adfd, physics=phys, overall=overall)

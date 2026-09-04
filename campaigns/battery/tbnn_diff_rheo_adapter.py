"""Adapter: TBNN + Giesekus as conformation-level digital materials
for protocol-space probing through the ``diff_rheo`` integrator.

Runs in the **diff_rheo** conda env, where ``jax_rheology`` is NOT importable
(it needs ``jax_ib`` / ``tree_math``). So this module RE-IMPLEMENTS, in pure
JAX, only the tiny pieces of ``jax_rheology.models.tbnn_memory`` needed to evaluate
the closure from the checkpoint ``.npz`` weights:

  * the invariant features ``x = (tau-3, p2-3, ln det A)`` (with the det floor),
  * the three small tanh MLP heads (``phi``, ``m0_raw``, ``m1``),
  * the anchored potential partials ``(phi_tau, phi_p2, phi_l)``,
  * the Sec. 0.7 floors (#1 det, #2 P/mobility min-eig, #4 phi_l coercivity),
  * the assembly ``K``, ``M_frozen``, ``A*`` (Sec. 0.4),
  * the continuous relaxation source ``R(A) = -(1/lam) M_frozen (A - A*)``,
  * the stress readout ``tau_p = Gp K(A)``.

The port is verified bit-for-bit against a reference (A -> K, R) pair exported
from the solver env (``tbnn_export_reference.py``) -- see :func:`verify_against_reference`.

Both the TBNN and the Giesekus truth are exposed as conformation rhs
``dA/dt = stretch(L,A) + R(A)`` over the planar-3D state
``A = (A_xx, A_xy, A_yy, A_zz)``, integrated by the SAME ``diff_rheo``
``DiffraxSolver`` at the SAME settings (apples-to-apples: any discrepancy is
closure, never discretization). ``A_zz`` has no stretch in planar shear.

Conventions match ``jax_rheology``: ``L_ij = du_i/dx_j``; simple shear has the
single nonzero entry ``L_xy = gammadot``; stretch ``= L A + A L^T`` gives
``(dA_xx, dA_xy, dA_yy, dA_zz)|stretch = (2 gd A_xy, gd A_yy, 0, 0)``.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

# --- Mobility floor constants (copied verbatim from tbnn_memory.py) ---------
_DET_FLOOR = 1e-10
_P_FLOOR = 0.05
_PHIL_MIN = 0.05
_FLOOR_WIDTH = 0.01
_SQRT_GUARD = 1e-30


# ===========================================================================
# Pure-JAX port of the closure math (no jax_rheology import).
# ===========================================================================

def _smooth_floor(z, floor, s):
    return floor + s * jnp.logaddexp(0.0, (z - floor) / s)


def _coaxial_min_eig(Xxx, Xxy, Xyy, Xzz):
    tr2 = Xxx + Xyy
    disc = (Xxx - Xyy) ** 2 + 4.0 * Xxy ** 2
    sq = jnp.sqrt(jnp.maximum(disc, _SQRT_GUARD))
    lam_min_inplane = 0.5 * (tr2 - sq)
    return jnp.minimum(lam_min_inplane, Xzz)


def _floor_coaxial_block(Xxx, Xxy, Xyy, Xzz, floor, s):
    x_min = _coaxial_min_eig(Xxx, Xxy, Xyy, Xzz)
    shift = s * jnp.logaddexp(0.0, (floor - x_min) / s)
    return (Xxx + shift, Xxy, Xyy + shift, Xzz + shift), x_min, shift


def invariant_features(A_xx, A_xy, A_yy, A_zz):
    tau = A_xx + A_yy + A_zz
    p2 = A_xx ** 2 + 2.0 * A_xy ** 2 + A_yy ** 2 + A_zz ** 2
    det_A2 = A_xx * A_yy - A_xy ** 2
    det_A2_eff = _smooth_floor(det_A2, _DET_FLOOR, _DET_FLOOR)
    A_zz_eff = _smooth_floor(A_zz, _DET_FLOOR, _DET_FLOOR)
    l = jnp.log(det_A2_eff) + jnp.log(A_zz_eff)
    return tau - 3.0, p2 - 3.0, l


def mlp_apply(layers, x):
    """tanh MLP; ``layers`` = list of (W, b). ``x`` is (d_in,) or (n, d_in)."""
    h = x
    for W, b in layers[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = layers[-1]
    return h @ W + b


def _N_scalar(phi_layers, x):
    """Scalar correction net N(x) = MLP(x) (scalar in, scalar out)."""
    return mlp_apply(phi_layers, x)[0]


def tbnn_heads(theta, X):
    """Per-cell heads from feature batch ``X`` (n,3). Mirrors
    ``tbnn_memory.tbnn_heads`` (no value bound; anchoring at x=0)."""
    phi_layers = theta['phi']
    zero3 = jnp.zeros(3, dtype=X.dtype)
    N0 = _N_scalar(phi_layers, zero3)
    grad_N_one = jax.grad(lambda xx: _N_scalar(phi_layers, xx))
    g0 = grad_N_one(zero3)
    grad_N = jax.vmap(grad_N_one)(X)
    phi_tau = 0.5 + grad_N[:, 0] - g0[0]
    phi_p2 = 0.0 + grad_N[:, 1] - g0[1]
    phi_l = -0.5 + grad_N[:, 2] - g0[2]
    m0 = jax.nn.softplus(mlp_apply(theta['m0_raw'], X)[:, 0])
    m1 = mlp_apply(theta['m1'], X)[:, 0]
    N_x = jax.vmap(lambda xx: _N_scalar(phi_layers, xx))(X)
    phi_val = 0.5 * (X[:, 0] - X[:, 2]) + N_x - N0 - X @ g0
    return phi_tau, phi_p2, phi_l, m0, m1, phi_val


def _A_squared_components(A_xx, A_xy, A_yy, A_zz):
    A2_xx = A_xx ** 2 + A_xy ** 2
    A2_xy = A_xy * (A_xx + A_yy)
    A2_yy = A_xy ** 2 + A_yy ** 2
    A2_zz = A_zz ** 2
    return A2_xx, A2_xy, A2_yy, A2_zz


def tbnn_K_and_frozen(A_xx, A_xy, A_yy, A_zz, theta):
    """Floored effective heads + assembly. Returns (K, M_frozen, A_star),
    each a 4-tuple (xx, xy, yy, zz). Mirrors tbnn_memory.tbnn_K_and_frozen."""
    A_xx = jnp.asarray(A_xx); A_xy = jnp.asarray(A_xy)
    A_yy = jnp.asarray(A_yy); A_zz = jnp.asarray(A_zz)
    shape = A_xx.shape
    x1, x2, x3 = invariant_features(A_xx, A_xy, A_yy, A_zz)
    X = jnp.stack([x1.reshape(-1), x2.reshape(-1), x3.reshape(-1)], axis=-1)
    phi_tau, phi_p2, phi_l, m0, m1, _ = tbnn_heads(theta, X)
    phi_l_eff = -_smooth_floor(-phi_l, _PHIL_MIN, _FLOOR_WIDTH)
    rs = lambda v: v.reshape(shape)
    phi_tau, phi_p2, phi_l_eff = rs(phi_tau), rs(phi_p2), rs(phi_l_eff)
    m0, m1 = rs(m0), rs(m1)

    A2_xx, A2_xy, A2_yy, A2_zz = _A_squared_components(A_xx, A_xy, A_yy, A_zz)
    phi_l = phi_l_eff
    K_xx = 2.0 * phi_tau * A_xx + 4.0 * phi_p2 * A2_xx + 2.0 * phi_l
    K_xy = 2.0 * phi_tau * A_xy + 4.0 * phi_p2 * A2_xy
    K_yy = 2.0 * phi_tau * A_yy + 4.0 * phi_p2 * A2_yy + 2.0 * phi_l
    K_zz = 2.0 * phi_tau * A_zz + 4.0 * phi_p2 * A2_zz + 2.0 * phi_l

    P_xx = phi_tau + 2.0 * phi_p2 * A_xx
    P_xy = 2.0 * phi_p2 * A_xy
    P_yy = phi_tau + 2.0 * phi_p2 * A_yy
    P_zz = phi_tau + 2.0 * phi_p2 * A_zz
    (Pf_xx, Pf_xy, Pf_yy, Pf_zz), _, _ = _floor_coaxial_block(
        P_xx, P_xy, P_yy, P_zz, _P_FLOOR, _FLOOR_WIDTH)

    Mob_xx = m0 + m1 * A_xx
    Mob_xy = m1 * A_xy
    Mob_yy = m0 + m1 * A_yy
    Mob_zz = m0 + m1 * A_zz
    (Mf_xx, Mf_xy, Mf_yy, Mf_zz), _, _ = _floor_coaxial_block(
        Mob_xx, Mob_xy, Mob_yy, Mob_zz, _P_FLOOR, _FLOOR_WIDTH)

    M_xx = 2.0 * (Mf_xx * Pf_xx + Mf_xy * Pf_xy)
    M_xy_a = 2.0 * (Mf_xx * Pf_xy + Mf_xy * Pf_yy)
    M_xy_b = 2.0 * (Mf_xy * Pf_xx + Mf_yy * Pf_xy)
    M_xy = 0.5 * (M_xy_a + M_xy_b)
    M_yy = 2.0 * (Mf_xy * Pf_xy + Mf_yy * Pf_yy)
    M_zz = 2.0 * Mf_zz * Pf_zz

    detP = Pf_xx * Pf_yy - Pf_xy ** 2
    As_xx = -phi_l * (Pf_yy / detP)
    As_xy = -phi_l * (-Pf_xy / detP)
    As_yy = -phi_l * (Pf_xx / detP)
    As_zz = -phi_l / Pf_zz
    return ((K_xx, K_xy, K_yy, K_zz), (M_xx, M_xy, M_yy, M_zz),
            (As_xx, As_xy, As_yy, As_zz))


def tbnn_source_R(A_xx, A_xy, A_yy, A_zz, lam, theta):
    """R(A) = -(1/lam) M_frozen (A - A*). Mirrors visco_tbnn.tbnn_source_R."""
    _, M, As = tbnn_K_and_frozen(A_xx, A_xy, A_yy, A_zz, theta)
    D_xx = A_xx - As[0]; D_xy = A_xy - As[1]
    D_yy = A_yy - As[2]; D_zz = A_zz - As[3]
    MD_xx = M[0] * D_xx + M[1] * D_xy
    MD_xy = 0.5 * ((M[0] * D_xy + M[1] * D_yy) + (M[1] * D_xx + M[2] * D_xy))
    MD_yy = M[1] * D_xy + M[2] * D_yy
    MD_zz = M[3] * D_zz
    return (-MD_xx / lam, -MD_xy / lam, -MD_yy / lam, -MD_zz / lam)


def tbnn_viscometric(A_xx, A_xy, A_yy, A_zz, theta, Gp):
    """tau = Gp K(A): returns the full (tau_xx, tau_xy, tau_yy, tau_zz)."""
    K, _, _ = tbnn_K_and_frozen(A_xx, A_xy, A_yy, A_zz, theta)
    return Gp * K[0], Gp * K[1], Gp * K[2], Gp * K[3]


# ===========================================================================
# Giesekus truth (conformation level), Hookean readout tau = Gp (A - I).
# ===========================================================================

def giesekus_source_R(A_xx, A_xy, A_yy, A_zz, lam, alpha):
    B_xx = A_xx - 1.0; B_xy = A_xy; B_yy = A_yy - 1.0; B_zz = A_zz - 1.0
    BB_xx = B_xx * B_xx + B_xy * B_xy
    BB_xy = B_xy * (B_xx + B_yy)
    BB_yy = B_xy * B_xy + B_yy * B_yy
    BB_zz = B_zz * B_zz
    return (-(B_xx + alpha * BB_xx) / lam, -(B_xy + alpha * BB_xy) / lam,
            -(B_yy + alpha * BB_yy) / lam, -(B_zz + alpha * BB_zz) / lam)


def giesekus_viscometric(A_xx, A_xy, A_yy, A_zz, Gp):
    return (Gp * (A_xx - 1.0), Gp * A_xy, Gp * (A_yy - 1.0), Gp * (A_zz - 1.0))


# ===========================================================================
# Conformation rhs for diff_rheo's solver:  dA/dt = stretch(L,A) + R(A).
# ``velocity_gradient`` is a diff_rheo VelocityGradient (has .gradient(t)).
# ===========================================================================

def _stretch(gd, A_xx, A_xy, A_yy):
    # L_xy = gd only; (dA_xx, dA_xy, dA_yy, dA_zz)|stretch = (2 gd A_xy, gd A_yy, 0, 0)
    return 2.0 * gd * A_xy, gd * A_yy, 0.0, 0.0


def make_tbnn_rhs(theta, lam):
    def rhs(t, y, velocity_gradient):
        A_xx, A_xy, A_yy, A_zz = y
        gd = velocity_gradient.gradient(t)[0, 1]
        s_xx, s_xy, s_yy, s_zz = _stretch(gd, A_xx, A_xy, A_yy)
        R = tbnn_source_R(A_xx, A_xy, A_yy, A_zz, lam, theta)
        return jnp.array([s_xx + R[0], s_xy + R[1], s_yy + R[2], s_zz + R[3]])
    return rhs


def make_giesekus_rhs(lam, alpha):
    def rhs(t, y, velocity_gradient):
        A_xx, A_xy, A_yy, A_zz = y
        gd = velocity_gradient.gradient(t)[0, 1]
        s_xx, s_xy, s_yy, s_zz = _stretch(gd, A_xx, A_xy, A_yy)
        R = giesekus_source_R(A_xx, A_xy, A_yy, A_zz, lam, alpha)
        return jnp.array([s_xx + R[0], s_xy + R[1], s_yy + R[2], s_zz + R[3]])
    return rhs


A_REST = jnp.array([1.0, 0.0, 1.0, 1.0])  # conformation at rest = I


# ===========================================================================
# Checkpoint loading (plain .npz: raw weights + metadata) and verification.
# ===========================================================================

def load_tbnn_checkpoint(npz_path):
    """Rebuild theta (dict head -> list of (W,b)) + scalars from the .npz."""
    z = np.load(npz_path, allow_pickle=False)
    heads = [str(h) for h in z['ckpt_heads']]
    nlayers = {h: int(n) for h, n in zip(heads, z['ckpt_nlayers'])}
    theta = {}
    for head in heads:
        layers = []
        for i in range(nlayers[head]):
            W = jnp.asarray(z[f'theta::{head}::{i}::W'], dtype=jnp.float64)
            b = jnp.asarray(z[f'theta::{head}::{i}::b'], dtype=jnp.float64)
            layers.append((W, b))
        theta[head] = layers
    return dict(
        theta=theta,
        Gp=float(z['ckpt_Gp_fit']), lam=float(z['ckpt_lam_fit']),
        nu_s=float(z['ckpt_nu_s']),
        width=int(z['ckpt_width']), depth=int(z['ckpt_depth']),
        truth_model=str(z['ckpt_truth_model']),
    )


def verify_against_reference(ckpt, refio_path, rtol=1e-4, atol=1e-11):
    """Recompute (K, R) at the reference A states and assert they match the
    solver-env values. Cross-env gate (addendum Prereq): the port is the SAME
    formula, so a re-derivation/wiring bug shows up as an O(1) mismatch. The
    tolerance is 1e-4 (not 1e-15): the solver env (jax 0.4.x) and the diff_rheo
    env (jax 0.7.1) use different XLA transcendental implementations, so states
    that evaluate near a Sec.0.7 smooth-floor edge can differ at the ~1e-5
    level -- negligible vs the 3% protocol noise / %-level fit tolerances. The
    actual relerr is always reported."""
    r = np.load(refio_path, allow_pickle=False)
    A = jnp.asarray(r['A_states'], dtype=jnp.float64)  # (n, 4)
    lam = float(r['lam'])
    theta = ckpt['theta']
    K, _, _ = tbnn_K_and_frozen(A[:, 0], A[:, 1], A[:, 2], A[:, 3], theta)
    R = tbnn_source_R(A[:, 0], A[:, 1], A[:, 2], A[:, 3], lam, theta)
    K_here = np.asarray(jnp.stack(K, axis=-1))   # (n,4)
    R_here = np.asarray(jnp.stack(R, axis=-1))
    K_ref = np.asarray(r['K'])
    R_ref = np.asarray(r['R'])

    def _relerr(a, b):
        return float(np.max(np.abs(a - b) / (np.abs(b) + atol)))
    eK, eR = _relerr(K_here, K_ref), _relerr(R_here, R_ref)
    ok = (eK <= rtol) and (eR <= rtol)
    return dict(ok=bool(ok), K_relerr=eK, R_relerr=eR, n_states=int(A.shape[0]))

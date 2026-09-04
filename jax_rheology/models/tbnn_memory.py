"""Viscoelastic TBNN closure: ONE toggleable potential-mobility body.

A single closure parameterized by two **static config switches**. The
learned relaxation is derived from a free energy ``phi`` and a two-term
mobility ``(m0, m1)``, expressed in the library's planar-3D state
``A = (A_xx, A_xy, A_yy, A_zz)``.

Two configurations are referred to throughout by the shorthand the campaigns
use: the **anchored** closure (``anchored=True``, equilibrium pinned at
``A = I``, fitted to a viscoelastic family truth) and the **yield-capable**
closure (``anchored=False`` with an annealed relu mobility, fitted to an
elastoviscoplastic truth). Older comments call these Tier-1 and Tier-3.

THE TOGGLE ARCHITECTURE (the core of this file)
=======================================================================
Three switches, **static config, fixed at registration time**, threaded
through ``init_tbnn_theta`` / ``tbnn_heads`` as plain Python (hashable),
passed at factory time -- **never** stored inside the differentiable
``theta`` pytree, **never** made learnable (if any became a pytree
leaf or an optimized variable you would get silent mode-mixing across the
batch -- the same class of trap the plan flags for the tanh-bound and
floor-pinning):

  * ``anchored: bool`` -- ``True`` hard-wires the equilibrium rest state
    at ``A = I`` via the value-and-gradient subtraction
    ``Dphi = N(x) - N(0) - x.grad N(0)``; ``False`` drops the subtraction
    (``phi = phi_OB + N(x)`` directly) and lets the network learn its own
    rest state. Coercivity in BOTH settings is the Sec. 0.7 floor #4
    (``phi_l`` clamped negative) -- the hard-coded ``-l/2`` barrier. (No
    ``c*tanh`` value bound: value bounds are rejected --
    they collapse the model to Oldroyd-B where they saturate -- and
    floor #4 is the coercivity barrier.)
  * ``mobility: 'softplus' | 'relu_annealed'`` -- ``'softplus'`` is
    strictly positive (``m0 = softplus(raw0)``, no yield);
    ``'relu_annealed'`` is ``m0 = kappa*softplus(raw0/kappa)`` with the
    temperature ``kappa`` annealed 1 -> 0.1 -> 0.02 during training. As
    ``kappa -> 0`` it approaches ``max(0, raw0)`` and the **zero-set of
    the mobility IS the learned yield surface** (EVP). At ``kappa = 1`` it
    is ``softplus`` to machine precision. In the ``relu_annealed`` mobility the
    Sec. 0.7 floor #2b relaxes from ``>= p_floor`` to PSD (``>= 0``) so
    ``M_frozen = 0`` is a valid frozen step (``E = exp(0) = I``, the
    identity below yield -- do NOT "fix" it). ``kappa`` is a runtime
    static float in ``params['tbnn_kappa']`` (like ``tbnn_bound_c``);
    changing it triggers a recompile (fine for a schedule of a few values)
    -- it is NOT a pytree leaf and is NOT optimized.
  * ``yield_mode: 'off' | 'scalar'`` -- ``'off'`` is a **static dead
    branch**: the Saramito yield prefactor is never evaluated and ``tau_y``
    does not exist as a parameter on any ``yield_mode='off'`` path (V1,
    V3, Giesekus/contraction fits cannot absorb a spurious yield). ``'scalar'``
    multiplies the WHOLE mobility block ``(m0 I + m1 A)`` by
    ``smoothmax(0, 1 - tau_y/|tau_d|)`` with ``|tau_d|`` from the
    **physical** polymer stress ``tau_p = Gp K(A)`` (developed-channel
    momentum balance pins the shear-stress profile; ``tau_y`` is in true
    physical units, directly comparable to truth; under the gauge fix
    ``Gp = 1`` the readout is unchanged). ``tau_y`` lives in ``params``
    next to ``Gp``/``lam``/``nu_s``, NOT in ``theta``. V2 uses plain
    ``softplus`` ``m0`` (no ``kappa`` anneal).

Modality table:

| anchored | mobility        | yield_mode | Model class                              | Registration                             |
|---------:|:----------------|:-----------|:-----------------------------------------|:-----------------------------------------|
| True     | softplus        | off        | Tier 1 / V1 -- viscoelastic, no yield    | ``tbnn_potential_logconf_bk_v2``         |
|          |                 |            | (Giesekus/FENE-P/PTT recovery).          |                                          |
|          |                 |            | Existing; must reproduce the trained path.|                                          |
| True     | softplus        | scalar     | V2 -- Tier 1 + structured yield scalar   | ``tbnn_potential_yield_logconf_bk_v2``   |
|          |                 |            | (``tau_y=0`` => V1 exactly).               |                                          |
| False    | softplus        | off        | Unanchored elastic -- learned rest       | ``tbnn_potential_unanchored_logconf_bk_v2`` |
|          |                 |            | state, still smooth (intermediate;       |                                          |
|          |                 |            | nematic-capable, 0D only)                |                                          |
| False    | relu_annealed   | off        | Tier 3 / V3 -- full EVP / yield stress   | ``tbnn_potential_free_logconf_bk_v2``    |

The four TBNN registrations are **thin wrappers that pin the config**;
there is exactly ONE closure body (do not fork the implementation per tier).
The registration factories are ``_make_tbnn_relaxation_fn(anchored,
mobility, yield_mode)`` / ``_make_tbnn_stress_readout_fn(anchored,
mobility)``. The
module-level ``_tbnn_relaxation_from_params`` / ``_tbnn_stress_readout_fn``
are the default ``(True, 'softplus')`` instances (Tier-1, back-compatible
names). The Saramito EVP data generator ``saramito_logconf_bk_v2`` (a
hard-coded Bingham truth, NOT a TBNN) also lives here.

Design contract:
  * Additive only -- this module *uses* ``log_conformation`` (imported as
    ``lc``) but edits nothing in it. The closure is just another
    ``relaxation_fn`` filling the family slot
    ``(A_xx, A_xy, A_yy, A_zz, velocity, dt, params) -> (...)`` plus a
    non-Hookean stress readout (the FENE-P pattern).
  * No eigendecompositions: only component-wise algebra, 2x2 adjugates,
    and the existing ``lc._exp_2x2_general`` (via
    ``lc._affine_exponential_relaxation_step``). The "min eigenvalue"
    used by the smooth floors is the closed-form scalar for a coaxial
    symmetric 2x2 block, never ``eig``.
  * float64 for any gradient-based task (the caller sets
    ``jax_enable_x64``; callers that differentiate should pin x64).

The state, invariants and features, with rest values
``tau = 3``, ``p2 = 3``, ``l = 0`` at ``A = I_3``::

    tau = tr A   = A_xx + A_yy + A_zz
    p2  = tr A^2 = A_xx^2 + 2 A_xy^2 + A_yy^2 + A_zz^2
    l   = ln det A = ln(A_xx A_yy - A_xy^2) + ln(A_zz)
    x   = (tau - 3, p2 - 3, l)

The learned objects (Tier 1, Variant A = anchored):
  * potential ``phi(x) = phi_OB(x) + Dphi_theta(x)``,
    ``phi_OB = 0.5 (x1 - x3) = 0.5 (tau - l - 3)``, with the
    value-and-gradient anchoring ``Dphi = N - N(0) - x . grad N(0)`` so
    ``dphi/dA = 0`` at ``A = I`` exactly;
  * mobility pair ``(m0, m1)``: ``m0 = softplus(raw0)`` (init 1),
    ``m1 = raw1`` (init 0, unbounded -- no value bound).

Stress and relaxation are both derived from ``phi``::

    K(A)     = 2 phi_tau A + 4 phi_p2 A^2 + 2 phi_l I       (coaxial, symmetric)
    tau_p    = Gp K(A)                                      (in-plane triple)
    dA/dt|relax = -(1/lam) (m0 I + m1 A) K(A)

routed through ``lc._affine_exponential_relaxation_step`` via the
factorization::

    P        = phi_tau I + 2 phi_p2 A
    M_frozen = 2 (m0 I + m1 A) P          (product of coaxial matrices)
    A*       = -phi_l P^{-1}              (P^{-1} via 2x2 adjugate)

so that ``M_frozen (A - A*) = (m0 I + m1 A) K(A)`` (verified on paper).
Oldroyd-B is the exact init: ``phi_OB, m0=1, m1=0`` give
``P = 0.5 I``, ``M_frozen = I``, ``A* = I``.

Smooth floors and AD-safety, all with width ``s`` much
smaller than the threshold (rule 0) so they are numerically inactive at
OB-init (the selftest asserts this to <= 1e-12):
  #1 ``ln det A_2`` argument floored at ``_DET_FLOOR``;
  #2a ``P`` min-eigenvalue floored to ``_P_FLOOR`` (all tiers) -- gates
      invertibility of ``A* = -phi_l P^{-1}``;
  #2b mobility block ``(m0 I + m1 A)`` min-eigenvalue floored to
      ``_P_FLOOR`` (Tiers 1-2; Tier 3 relaxes to PSD);
  #4 coercivity barrier: ``phi_l`` smoothly clamped above at
      ``-_PHIL_MIN`` so ``-phi_l >= _PHIL_MIN > 0`` (``A*`` SPD).

One ``effective_heads`` is the single source of truth for the floored
``(phi_tau, phi_p2, phi_l_eff, m0, m1)``, consumed by BOTH the
relaxation and the stress readout (floor #4).

Kernel-restart note: this module registers ``tbnn_potential_logconf_bk_v2``,
``tbnn_potential_yield_logconf_bk_v2``,
``tbnn_potential_unanchored_logconf_bk_v2``,
``tbnn_potential_free_logconf_bk_v2`` and the ``saramito_logconf_bk_v2``
EVP data generator at import. ``cr.register`` refuses duplicates, so
re-importing after an edit requires a fresh kernel (same rule as
``log_conformation``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp

from jax_rheology.models import registry as cr
from jax_rheology import log_conformation as lc

__all__ = [
    'tbnn_invariant_features',
    'init_mlp',
    'mlp_apply',
    'init_tbnn_theta',
    'tbnn_heads',
    'tbnn_effective_heads',
    'tbnn_assemble_from_heads',
    'tbnn_K_and_frozen',
    'tbnn_floor_diagnostics',
    '_make_tbnn_relaxation_fn',
    '_make_tbnn_stress_readout_fn',
    '_tbnn_relaxation_from_params',
    '_tbnn_stress_readout_fn',
    'tbnn_tau_zz_readout',
    # Saramito EVP data generator (hard-coded Bingham truth).
    'saramito_tau_d_norm',
    'saramito_kappa_y',
    '_saramito_relaxation_from_params',
    'SARAMITO_LOGCONF_BK_V2',
    # Toggle registrations (thin wrappers pinning the static config).
    'TBNN_POTENTIAL_LOGCONF_BK_V2',
    'TBNN_POTENTIAL_YIELD_LOGCONF_BK_V2',
    'TBNN_POTENTIAL_UNANCHORED_LOGCONF_BK_V2',
    'TBNN_POTENTIAL_FREE_LOGCONF_BK_V2',
    'TBNN_DEFAULT_BOUND_C',
    'TBNN_DEFAULT_ANCHORED',
    'TBNN_DEFAULT_MOBILITY',
    'TBNN_DEFAULT_YIELD_MODE',
    'TBNN_DEFAULT_KAPPA',
    'TBNN_MOBILITY_MODES',
    'TBNN_YIELD_MODES',
    'tau_d_norm_from_K',
    'yield_prefactor_scalar',
]


# ---------------------------------------------------------------------------
# Smooth-floor constants (width s << threshold).
# ---------------------------------------------------------------------------

_DET_FLOOR = 1e-10
"""Floor on ``det A_2`` (and ``A_zz``) inside the ``ln det`` feature
(floor #1). Relative width = the floor itself."""

_P_FLOOR = 0.05
"""Min-eigenvalue floor for ``P`` (#2a) and the mobility block (#2b)
Threshold; width ``_FLOOR_WIDTH`` is much smaller."""

_PHIL_MIN = 0.05
"""Coercivity barrier: ``-phi_l >= _PHIL_MIN`` so ``A*`` stays SPD
(floor #4)."""

_FLOOR_WIDTH = 0.01
"""Smoothing width ``s`` for floors #2 and #4 (rule 0:
residual ~ e^{-45} at OB-init, negligible vs the G1 rtol 1e-6)."""

_SQRT_GUARD = 1e-30
"""AD-safe guard on the eigen-gap sqrt argument (the ``max(., 0)``-style
floor of rule 2). Below this the sqrt gradient is killed
by ``jnp.maximum``; above it the gradient is finite."""

TBNN_DEFAULT_BOUND_C = 3.0
"""DEPRECATED / INERT. Formerly the ``c tanh(./c)`` saturation bound on
``Dphi`` and ``m1``; **removed** --
value bounds collapse the model to Oldroyd-B where they saturate, fatal
for low-L^2 FENE-P. Safety now lives entirely in the Sec. 0.7 floors. The
constant and the ``bound_c``/``tbnn_bound_c`` argument are retained only
for signature/config stability (ignored at runtime; reserved for a future
inference-time-only input clamp, never active during fitting)."""

# ---------------------------------------------------------------------------
# STATIC-CONFIG SWITCHES. Two settings each; what
# each opens up physically; and the trap. These are set ONCE at registration
# time (baked into the factory closures below), passed to the heads as plain
# Python keyword arguments, and are NEVER stored in the differentiable
# ``theta`` pytree and NEVER optimized. If either becomes a pytree leaf or a
# learnable variable you get silent mode-mixing across the batch.
# ---------------------------------------------------------------------------

TBNN_DEFAULT_ANCHORED = True
"""``anchored`` switch default (Tier-1 behaviour).
  * ``True``  : equilibrium rest state hard-wired at ``A = I`` via the
    value-and-gradient subtraction ``Dphi = N(x) - N(0) - x.grad N(0)`` ->
    ``dphi/dA = 0`` at ``A = I`` exactly (rest state FIXED).
  * ``False`` : drop the subtraction; ``phi = phi_OB + N(x)`` directly ->
    the network LEARNS its own rest state (nematic / EVP capable).
Coercivity is the Sec. 0.7 floor #4 in BOTH settings (no ``c*tanh`` bound).
TRAP: never a pytree leaf, never learnable (static config only)."""

TBNN_DEFAULT_MOBILITY = 'softplus'
"""``mobility`` switch default (Tier-1 behaviour).
  * ``'softplus'``      : ``m0 = softplus(raw0)`` -- strictly positive,
    NO yield (viscoelastic).
  * ``'relu_annealed'`` : ``m0 = kappa*softplus(raw0/kappa)`` -- can reach
    exactly zero as ``kappa -> 0``; the mobility zero-set IS the learned
    yield surface (EVP). Floor #2b relaxes to PSD in this mode.
TRAP: never a pytree leaf, never learnable (static config only)."""

TBNN_DEFAULT_KAPPA = 1.0
"""Default annealing temperature for the ``relu_annealed`` mobility. At
``kappa = 1`` it is ``softplus`` to machine precision. Runtime static float in
``params['tbnn_kappa']`` (like ``tbnn_bound_c``); NOT a pytree leaf, NOT
optimized. The annealing schedule (1 -> 0.1 -> 0.02) is experiment config;
save the kappa history; never evaluate a gate mid-anneal."""

TBNN_MOBILITY_MODES = ('softplus', 'relu_annealed')
"""Valid ``mobility`` switch settings."""

TBNN_DEFAULT_YIELD_MODE = 'off'
"""``yield_mode`` switch default (Tier-1 / V1 behaviour).
  * ``'off'``    : static dead branch -- no yield prefactor evaluated,
    no ``tau_y`` parameter on any off path.
  * ``'scalar'`` : Saramito-style prefactor gating the whole mobility block,
    applied AFTER the #2b floor (``mob_prefactor``) so ``pref = 0`` gives
    ``M_frozen = 0`` exactly; ``tau_y`` in ``params`` (scalar block), NOT in
    ``theta``.
TRAP: never a pytree leaf, never learnable (static config only)."""

TBNN_YIELD_MODES = ('off', 'scalar')
"""Valid ``yield_mode`` switch settings."""


# ---------------------------------------------------------------------------
# Smooth-floor helpers (component-wise; no eig).
# ---------------------------------------------------------------------------

def _smooth_floor(z: jnp.ndarray, floor: float, s: float) -> jnp.ndarray:
    """Smooth lower bound: ``>= floor``, exact (== z) when ``z >> floor``.

    ``floor + s * softplus((z - floor) / s)`` -- the AD-safe idiom of
    ``lc._fene_p_peterlin_f`` but with an independent width ``s`` so that
    ``s << floor``. ``jnp.logaddexp(0, .)`` is the
    numerically stable softplus.
    """
    return floor + s * jnp.logaddexp(0.0, (z - floor) / s)


def _coaxial_min_eig(Xxx: jnp.ndarray, Xxy: jnp.ndarray,
                     Xyy: jnp.ndarray, Xzz: jnp.ndarray) -> jnp.ndarray:
    """Smallest eigenvalue of a block-diagonal coaxial symmetric tensor.

    The in-plane block ``[[Xxx, Xxy], [Xxy, Xyy]]`` has eigenvalues
    ``0.5 (tr +/- sqrt((Xxx - Xyy)^2 + 4 Xxy^2))`` (
    rule 2, eigenvector-free); the out-of-plane channel contributes the
    scalar ``Xzz``. The sqrt argument is guarded with ``_SQRT_GUARD`` so
    the gradient is finite/AD-clean on the isotropic manifold.
    """
    tr2 = Xxx + Xyy
    disc = (Xxx - Xyy) ** 2 + 4.0 * Xxy ** 2
    sq = jnp.sqrt(jnp.maximum(disc, _SQRT_GUARD))
    lam_min_inplane = 0.5 * (tr2 - sq)
    return jnp.minimum(lam_min_inplane, Xzz)


def _floor_coaxial_block(Xxx, Xxy, Xyy, Xzz, floor, s):
    """Shift a coaxial symmetric block-diag tensor to ``min-eig >= floor``.

    ``X -> X + shift I`` with ``shift = s softplus((floor - x_min)/s)``
   . Keeps ``X`` coaxial with ``A`` (a scalar
    multiple of ``I`` added). Returns the shifted components plus the
    pre-shift ``x_min`` and the ``shift`` (for floor-activation
    monitoring).
    """
    x_min = _coaxial_min_eig(Xxx, Xxy, Xyy, Xzz)
    shift = s * jnp.logaddexp(0.0, (floor - x_min) / s)
    return (Xxx + shift, Xxy, Xyy + shift, Xzz + shift), x_min, shift


# ---------------------------------------------------------------------------
# Invariant features with the ln-det floor (#1).
# ---------------------------------------------------------------------------

def tbnn_invariant_features(A_xx, A_xy, A_yy, A_zz):
    """Network input features ``x = (tau - 3, p2 - 3, l)``.

    All component-wise (no eig). ``l = ln det A`` uses the smooth ``det``
    floor #1 so the log argument is positive and
    AD-clean even if a stretch substep transiently dents the SPD margin;
    the floor is inactive on the SPD states the solver maintains.
    """
    tau = A_xx + A_yy + A_zz
    p2 = A_xx ** 2 + 2.0 * A_xy ** 2 + A_yy ** 2 + A_zz ** 2
    det_A2 = A_xx * A_yy - A_xy ** 2
    det_A2_eff = _smooth_floor(det_A2, _DET_FLOOR, _DET_FLOOR)
    A_zz_eff = _smooth_floor(A_zz, _DET_FLOOR, _DET_FLOOR)
    l = jnp.log(det_A2_eff) + jnp.log(A_zz_eff)
    return tau - 3.0, p2 - 3.0, l


# ---------------------------------------------------------------------------
# Minimal pure-JAX MLP (explicit pytree params; no flax inside the registry
# pure functions).
# ---------------------------------------------------------------------------

Layer = Tuple[jnp.ndarray, jnp.ndarray]


def init_mlp(key, sizes: Tuple[int, ...]) -> List[Layer]:
    """Initialize a tanh MLP as a list of ``(W, b)`` leaves.

    He-ish scaling ``1/sqrt(fan_in)`` on the hidden weights; the caller
    zeroes the last layer for the exact-OB init (``init_tbnn_theta``).
    """
    layers: List[Layer] = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, d_in, d_out in zip(keys, sizes[:-1], sizes[1:]):
        W = jax.random.normal(k, (d_in, d_out), dtype=jnp.float64) * jnp.sqrt(1.0 / d_in)
        b = jnp.zeros((d_out,), dtype=jnp.float64)
        layers.append((W, b))
    return layers


def mlp_apply(layers: List[Layer], x: jnp.ndarray) -> jnp.ndarray:
    """Apply a tanh MLP. ``x`` may be ``(d_in,)`` or batched ``(n, d_in)``.

    Hidden layers use ``tanh``; the output layer is linear. Returns the
    raw output (shape ``(d_out,)`` or ``(n, d_out)``); the heads wrap it
    with ``softplus`` / ``c tanh`` as needed.
    """
    h = x
    for W, b in layers[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = layers[-1]
    return h @ W + b


def _zero_last_layer(layers: List[Layer], bias_value: float = 0.0) -> List[Layer]:
    """Zero a head's final ``(W, b)`` so its output is the constant
    ``bias_value`` (and its input-gradient is exactly zero) -- the
    exact-OB / anchored init."""
    W, b = layers[-1]
    new_last = (jnp.zeros_like(W), jnp.full_like(b, bias_value))
    return layers[:-1] + [new_last]


def _softplus_inv(y: float) -> float:
    """Inverse softplus: ``ln(exp(y) - 1)``. ``_softplus_inv(1) = ln(e - 1)``
    so a zeroed ``m0_raw`` last layer with this bias gives ``m0 = 1``."""
    return float(jnp.log(jnp.expm1(y)))


def init_tbnn_theta(key, *, width: int = 32, depth: int = 2,
                    bound_c: float = TBNN_DEFAULT_BOUND_C,
                    anchored: bool = TBNN_DEFAULT_ANCHORED,
                    mobility: str = TBNN_DEFAULT_MOBILITY,
                    yield_mode: str = TBNN_DEFAULT_YIELD_MODE,
                    kappa_init: float = TBNN_DEFAULT_KAPPA):
    """Exact-Oldroyd-B init of the theta pytree.

    The ``(anchored, mobility, kappa_init)`` static switches are recorded
    in ``config`` (plain Python, hashable) -- they are NEVER placed in the
    differentiable ``theta`` pytree (never a pytree leaf). At the
    exact-OB init the closure is analytically Oldroyd-B for BOTH mobility
    modes (``relu_annealed`` at ``kappa = 1`` is softplus to machine precision)
    and BOTH anchoring settings (``N == 0`` at init, so anchored and
    unanchored coincide -- ``phi = phi_OB``).

    Three small MLPs (``phi``, ``m0_raw``, ``m1``), each input dim 3,
    ``depth`` hidden layers of ``width``, scalar output, float64, tanh.
    The last layer of every head is zeroed so:
      * ``N_theta == 0`` -> ``Dphi == 0`` -> ``phi == phi_OB`` (and the
        anchoring subtraction is trivially satisfied);
      * ``m1 == 0``;
      * ``m0_raw == softplus^{-1}(1)`` -> ``m0 == 1`` exactly.
    With these, the closure is analytically Oldroyd-B (gate G1).

    Returns ``(theta, config)``. ``theta`` is arrays only
    (``{'phi', 'm0_raw', 'm1'}``) -- the differentiable pytree. ``config``
    is plain Python (``sizes``, ``bound_c``, ...) for the run-config
    record; only ``bound_c`` is consumed at runtime and is passed
    separately via ``params['tbnn_bound_c']`` (never inside the pytree).
    """
    sizes = (3,) + (int(width),) * int(depth) + (1,)
    k_phi, k_m0, k_m1 = jax.random.split(key, 3)
    theta = {
        'phi': _zero_last_layer(init_mlp(k_phi, sizes), 0.0),
        'm0_raw': _zero_last_layer(init_mlp(k_m0, sizes), _softplus_inv(1.0)),
        'm1': _zero_last_layer(init_mlp(k_m1, sizes), 0.0),
    }
    if mobility not in TBNN_MOBILITY_MODES:
        raise ValueError(f"mobility must be one of {TBNN_MOBILITY_MODES}, "
                         f"got {mobility!r}")
    if yield_mode not in TBNN_YIELD_MODES:
        raise ValueError(f"yield_mode must be one of {TBNN_YIELD_MODES}, "
                         f"got {yield_mode!r}")
    config = {'sizes': sizes, 'width': int(width), 'depth': int(depth),
              'bound_c': float(bound_c), 'anchored': bool(anchored),
              'mobility': str(mobility), 'yield_mode': str(yield_mode),
              'kappa_init': float(kappa_init)}
    return theta, config


def _set_last_bias(layers: List[Layer], bias_value: float) -> List[Layer]:
    """Set the last-layer bias to ``bias_value``; keep ``W`` and all hidden
    layers untouched. Used by the Giesekus theta-init so the head sits near
    a target constant while remaining state-dependent (and AD-alive) through
    the random last-layer weights. Distinct from ``_zero_last_layer``, which
    kills the input gradient."""
    W, b = layers[-1]
    return layers[:-1] + [(W, jnp.full_like(b, bias_value))]


def init_tbnn_theta_giesekus(key, *, width: int = 32, depth: int = 2,
                             bound_c: float = TBNN_DEFAULT_BOUND_C,
                             anchored: bool = TBNN_DEFAULT_ANCHORED,
                             mobility: str = TBNN_DEFAULT_MOBILITY,
                             yield_mode: str = TBNN_DEFAULT_YIELD_MODE,
                             kappa_init: float = TBNN_DEFAULT_KAPPA,
                             alpha: float = 0.3):
    """Giesekus(``alpha``) mobility init with Oldroyd-B potential (``N == 0``).

    Potential net: exact-OB (last layer of ``phi`` zeroed -> ``N == 0``).
    Mobility: standard random hidden layers; only the output bias is set so
    ``softplus(raw0) ~= 1-alpha`` and ``m1 ~= alpha`` near rest, with last-layer ``W``
    kept from the random init so gradients through the head stay intact.
    Default ``alpha = 0.3`` -> targets ``(m0, m1) = (0.7, 0.3)``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f'Giesekus alpha must be in (0,1), got {alpha!r}')
    sizes = (3,) + (int(width),) * int(depth) + (1,)
    k_phi, k_m0, k_m1 = jax.random.split(key, 3)
    theta = {
        'phi': _zero_last_layer(init_mlp(k_phi, sizes), 0.0),
        'm0_raw': _set_last_bias(init_mlp(k_m0, sizes),
                                 _softplus_inv(1.0 - alpha)),
        'm1': _set_last_bias(init_mlp(k_m1, sizes), alpha),
    }
    if mobility not in TBNN_MOBILITY_MODES:
        raise ValueError(f"mobility must be one of {TBNN_MOBILITY_MODES}, "
                         f"got {mobility!r}")
    if yield_mode not in TBNN_YIELD_MODES:
        raise ValueError(f"yield_mode must be one of {TBNN_YIELD_MODES}, "
                         f"got {yield_mode!r}")
    config = {'sizes': sizes, 'width': int(width), 'depth': int(depth),
              'bound_c': float(bound_c), 'anchored': bool(anchored),
              'mobility': str(mobility), 'yield_mode': str(yield_mode),
              'kappa_init': float(kappa_init),
              'theta_init': 'giesekus', 'giesekus_alpha': float(alpha)}
    return theta, config


def init_tbnn_theta_random(key, *, width: int = 32, depth: int = 2,
                           bound_c: float = TBNN_DEFAULT_BOUND_C,
                           anchored: bool = TBNN_DEFAULT_ANCHORED,
                           mobility: str = TBNN_DEFAULT_MOBILITY,
                           yield_mode: str = TBNN_DEFAULT_YIELD_MODE,
                           kappa_init: float = TBNN_DEFAULT_KAPPA,
                           scale: float = 1.0):
    """Fully random theta at the standard MLP init scale (no last-layer
    zeroing). ``scale`` multiplies every ``(W, b)`` after ``init_mlp``;
    ``scale = 1.0`` is the unmodified He-ish init. Perturbs both the
    potential net and the mobility heads.
    """
    sizes = (3,) + (int(width),) * int(depth) + (1,)
    k_phi, k_m0, k_m1 = jax.random.split(key, 3)
    s = float(scale)

    def _scaled(k):
        layers = init_mlp(k, sizes)
        if s == 1.0:
            return layers
        return [(W * s, b * s) for (W, b) in layers]

    theta = {'phi': _scaled(k_phi), 'm0_raw': _scaled(k_m0),
             'm1': _scaled(k_m1)}
    if mobility not in TBNN_MOBILITY_MODES:
        raise ValueError(f"mobility must be one of {TBNN_MOBILITY_MODES}, "
                         f"got {mobility!r}")
    if yield_mode not in TBNN_YIELD_MODES:
        raise ValueError(f"yield_mode must be one of {TBNN_YIELD_MODES}, "
                         f"got {yield_mode!r}")
    config = {'sizes': sizes, 'width': int(width), 'depth': int(depth),
              'bound_c': float(bound_c), 'anchored': bool(anchored),
              'mobility': str(mobility), 'yield_mode': str(yield_mode),
              'kappa_init': float(kappa_init),
              'theta_init': 'random', 'theta_init_scale': s}
    return theta, config


# ---------------------------------------------------------------------------
# Heads: phi (anchored) + its partials, and the mobility pair.
# ---------------------------------------------------------------------------

def _N_raw(phi_layers: List[Layer], x: jnp.ndarray, c: float) -> jnp.ndarray:
    """Unbounded scalar correction network ``N(x) = MLP(x)`` (scalar in,
    scalar out).

    Plan Sec. 0.6 (current revision): **no value bound** on ``Dphi``. A
    ``c tanh(./c)`` saturation was rejected -- where the raw output
    saturates, ``tanh' -> 0`` squashes the derivatives and the model
    collapses to exactly Oldroyd-B (fatal for low-L^2 FENE-P, where
    ``Dphi(tau)`` legitimately exceeds any ``c ~ 3-5`` exactly where the
    finite-extensibility physics lives). Safety lives in the Sec. 0.7
    floors on the derived quantities (``P``, mobility block, ``phi_l``),
    not in an output bound. ``c`` is accepted for signature stability but
    ignored (reserved for an inference-time-only input clamp, never active
    during fitting)."""
    del c
    return mlp_apply(phi_layers, x)[0]


def tbnn_heads(theta: Dict[str, Any], X: jnp.ndarray, bound_c: float,
               *, anchored: bool = TBNN_DEFAULT_ANCHORED,
               mobility: str = TBNN_DEFAULT_MOBILITY,
               kappa: float = TBNN_DEFAULT_KAPPA):
    """Per-cell heads from the feature batch ``X`` of shape ``(n, 3)``.

    Returns ``(phi_tau, phi_p2, phi_l, m0, m1, phi_val)`` each of shape
    ``(n,)``, the *unfloored* partials (floors #2/#4 are applied later in
    ``tbnn_effective_heads`` / ``tbnn_K_and_frozen``).

    ``anchored``:
      * ``True`` -- value-and-gradient subtraction:
        ``Dphi = N - N(0) - x . grad N(0)`` so the partials of ``Dphi``
        vanish at ``x = 0`` (``K(I) = 0`` exactly, rest state FIXED at
        ``A = I``). ``N(0)`` and ``grad N(0)`` are constants *per theta* --
        computed once and broadcast, NOT inside the per-cell vmap (plan
        Sec. 0.8 perf note (i)).
      * ``False`` -- drop the subtraction entirely: ``phi = phi_OB + N(x)``
        directly. No value bound (value bounds are rejected;
        ``c*tanh`` saturation; coercivity is floor #4 downstream). The rest
        state is whatever the data says -- ``K(I)`` is NOT re-imposed to 0.

    ``phi_OB = 0.5 (x1 - x3)`` contributes the constant partials
    ``(0.5, 0, -0.5)``; anchored, at ``x = 0`` the partials are exactly
    ``(0.5, 0, -0.5)`` (Oldroyd-B).

    ``mobility``:
      * ``'softplus'``      -- ``m0 = softplus(raw0)`` (strictly positive).
      * ``'relu_annealed'`` -- ``m0 = kappa*softplus(raw0/kappa)`` (reaches
        0 as ``kappa -> 0``; the zero-set is the yield surface). At
        ``kappa = 1`` this is ``softplus`` to machine precision.
    """
    phi_layers = theta['phi']

    grad_N_one = jax.grad(lambda xx: _N_raw(phi_layers, xx, bound_c))
    grad_N = jax.vmap(grad_N_one)(X)             # (n, 3)
    N_x = jax.vmap(lambda xx: _N_raw(phi_layers, xx, bound_c))(X)

    if anchored:
        # Tier-1 Variant A: value-and-gradient subtraction pins K(I) = 0.
        zero3 = jnp.zeros(3, dtype=X.dtype)
        N0 = _N_raw(phi_layers, zero3, bound_c)
        g0 = grad_N_one(zero3)                   # (3,), constant per theta
        phi_tau = 0.5 + grad_N[:, 0] - g0[0]
        phi_p2 = 0.0 + grad_N[:, 1] - g0[1]
        phi_l = -0.5 + grad_N[:, 2] - g0[2]
        phi_val = 0.5 * (X[:, 0] - X[:, 2]) + N_x - N0 - X @ g0
    else:
        # Swap 1 (unanchored): phi = phi_OB + N(x) directly, NO subtraction.
        # (grep target: the anchored subtraction is genuinely bypassed here.)
        phi_tau = 0.5 + grad_N[:, 0]
        phi_p2 = 0.0 + grad_N[:, 1]
        phi_l = -0.5 + grad_N[:, 2]
        phi_val = 0.5 * (X[:, 0] - X[:, 2]) + N_x

    # No value bound on m1: m1 is the
    # raw MLP output. ``bound_c`` is still threaded into ``_N_raw`` (ignored
    # there) only for signature stability.
    raw0 = mlp_apply(theta['m0_raw'], X)[:, 0]
    if mobility == 'softplus':
        m0 = jax.nn.softplus(raw0)
    elif mobility == 'relu_annealed':
        # Swap 2: annealed smooth-ReLU. kappa*softplus(raw0/kappa) -> softplus
        # at kappa=1 (division/multiply by 1.0 are exact), -> max(0, raw0) as
        # kappa -> 0. kappa is a static float (params['tbnn_kappa']), never
        # a pytree leaf.
        m0 = kappa * jax.nn.softplus(raw0 / kappa)
    else:
        raise ValueError(f"mobility must be one of {TBNN_MOBILITY_MODES}, "
                         f"got {mobility!r}")
    m1 = mlp_apply(theta['m1'], X)[:, 0]

    return phi_tau, phi_p2, phi_l, m0, m1, phi_val


def tbnn_effective_heads(A_xx, A_xy, A_yy, A_zz, theta, bound_c,
                         *, anchored: bool = TBNN_DEFAULT_ANCHORED,
                         mobility: str = TBNN_DEFAULT_MOBILITY,
                         kappa: float = TBNN_DEFAULT_KAPPA):
    """Floored heads on a full grid -- the single source of truth used by
    BOTH relaxation and stress readout (floor #4).

    Computes features, runs ``tbnn_heads`` over flattened cells (threading
    the ``anchored`` / ``mobility`` / ``kappa`` static switches), reshapes
    back to the grid, and applies the coercivity barrier (floor #4):
    ``phi_l_eff = -[smooth_floor(-phi_l, _PHIL_MIN)]`` so
    ``-phi_l_eff >= _PHIL_MIN > 0`` (exact when ``phi_l`` is safely
    negative -- e.g. OB's ``phi_l = -0.5``). Floor #4 is UNCHANGED across
    all tiers ( it *is* the hard-coded coercivity barrier, in
    both anchored and unanchored settings). Returns the grid-shaped
    ``(phi_tau, phi_p2, phi_l_eff, m0, m1, phi_val)``.
    """
    shape = A_xx.shape
    x1, x2, x3 = tbnn_invariant_features(A_xx, A_xy, A_yy, A_zz)
    X = jnp.stack([x1.reshape(-1), x2.reshape(-1), x3.reshape(-1)], axis=-1)
    phi_tau, phi_p2, phi_l, m0, m1, phi_val = tbnn_heads(
        theta, X, bound_c, anchored=anchored, mobility=mobility, kappa=kappa)

    # Floor #4: clamp phi_l from above at -_PHIL_MIN (coercivity barrier).
    phi_l_eff = -_smooth_floor(-phi_l, _PHIL_MIN, _FLOOR_WIDTH)

    rs = lambda v: v.reshape(shape)
    return (rs(phi_tau), rs(phi_p2), rs(phi_l_eff),
            rs(m0), rs(m1), rs(phi_val))


# ---------------------------------------------------------------------------
# K(A), frozen operator M_frozen, and offset A* with the
# P / mobility min-eigenvalue floors (#2a / #2b).
# ---------------------------------------------------------------------------

def _A_squared_components(A_xx, A_xy, A_yy, A_zz):
    """Components of ``A^2`` for the block-diagonal planar-3D ``A``
    (2x2 square + scalar ``A_zz^2``). Component-wise, no matmul kernel."""
    A2_xx = A_xx ** 2 + A_xy ** 2
    A2_xy = A_xy * (A_xx + A_yy)
    A2_yy = A_xy ** 2 + A_yy ** 2
    A2_zz = A_zz ** 2
    return A2_xx, A2_xy, A2_yy, A2_zz


def tbnn_assemble_from_heads(A_xx, A_xy, A_yy, A_zz,
                             phi_tau, phi_p2, phi_l_eff, m0, m1,
                             *, mob_floor: float = _P_FLOOR,
                             mob_prefactor=None):
    """Assemble ``K``, ``M_frozen``, ``A*`` from given (effective) heads.

    ``phi_l_eff`` is the *coercivity-floored* ``phi_l`` (floor #4 already
    applied) so this is the single assembly shared by relaxation and
    readout. Supplying analytic closed-form heads here (and comparing to
    the existing family relaxation/readout functions) is exactly the
    Sec. 0.5 recovery selftest.

    ``K = 2 phi_tau A + 4 phi_p2 A^2 + 2 phi_l_eff I``;
    ``P = phi_tau I + 2 phi_p2 A`` floored to ``min-eig >= _P_FLOOR``
    (#2a, all tiers -- gates invertibility of ``A*``);
    ``Mob = m0 I + m1 A`` floored to ``min-eig >= mob_floor`` (#2b:
    ``mob_floor = _P_FLOOR`` in Tiers 1-2, ``mob_floor = 0`` (PSD) in
    Tier 3 -- the mobility zero-set is the yield surface, and
    ``M_frozen = 0`` is a valid frozen step ``E = exp(0) = I``);
    ``M_frozen = 2 Mob_floored P_floored``;
    ``A* = -phi_l_eff P_floored^{-1}`` (2x2 adjugate; scalar zz). ``A*``
    stays finite even at yield because ``P`` is floored strictly positive.
    Satisfies ``M_frozen (A - A*) = Mob K``.

    ``mob_prefactor`` (default ``None``) is the yield gate, applied to the
    mobility block **after** the #2b floor:
    ``Mob_gated = mob_prefactor * Mob_floored``. It must be last, because
    anything downstream of the floor can lift a zero back off zero -- which
    is exactly the defect this keyword fixes. Scaling ``(m0, m1)`` *before*
    the floor (the pre-fix V2 path) could not arrest: at ``pref = 0`` the
    pre-floor block is zero, ``_floor_coaxial_block`` shifts it to
    ``_FLOOR_WIDTH * softplus(_P_FLOOR / _FLOOR_WIDTH) = 0.0500672``, and
    ``M_frozen -> 0.05 I`` instead of ``0`` for every ``theta`` and every
    ``tau_y``, leaving an Oldroyd-B residual at ~5 % of the OB relaxation
    rate. Gating after the floor keeps the floor's protection of the
    *learned* block -- ``m1`` is an unbounded MLP output and can go
    negative, and a negative eigenvalue in ``exp(-(dt/lam) M)`` amplifies --
    while still allowing an exact ``M_frozen = 0``.

    ``None`` **skips the multiply entirely** rather than multiplying by
    ``1.0``, so bit-identity for every pre-existing caller is structural
    (a Python branch on a static value) and not a floating-point
    coincidence. ``_P_FLOOR``, ``_mob_floor_for`` and ``_FLOOR_WIDTH`` are
    unchanged.
    """
    phi_l = phi_l_eff
    A2_xx, A2_xy, A2_yy, A2_zz = _A_squared_components(A_xx, A_xy, A_yy, A_zz)

    # K = 2 phi_tau A + 4 phi_p2 A^2 + 2 phi_l I.
    K_xx = 2.0 * phi_tau * A_xx + 4.0 * phi_p2 * A2_xx + 2.0 * phi_l
    K_xy = 2.0 * phi_tau * A_xy + 4.0 * phi_p2 * A2_xy
    K_yy = 2.0 * phi_tau * A_yy + 4.0 * phi_p2 * A2_yy + 2.0 * phi_l
    K_zz = 2.0 * phi_tau * A_zz + 4.0 * phi_p2 * A2_zz + 2.0 * phi_l

    # P = phi_tau I + 2 phi_p2 A  (coaxial symmetric block-diagonal).
    P_xx = phi_tau + 2.0 * phi_p2 * A_xx
    P_xy = 2.0 * phi_p2 * A_xy
    P_yy = phi_tau + 2.0 * phi_p2 * A_yy
    P_zz = phi_tau + 2.0 * phi_p2 * A_zz
    (Pf_xx, Pf_xy, Pf_yy, Pf_zz), _, _ = _floor_coaxial_block(
        P_xx, P_xy, P_yy, P_zz, _P_FLOOR, _FLOOR_WIDTH)

    # Mobility block Mob = m0 I + m1 A  (coaxial symmetric block-diagonal).
    Mob_xx = m0 + m1 * A_xx
    Mob_xy = m1 * A_xy
    Mob_yy = m0 + m1 * A_yy
    Mob_zz = m0 + m1 * A_zz
    (Mf_xx, Mf_xy, Mf_yy, Mf_zz), _, _ = _floor_coaxial_block(
        Mob_xx, Mob_xy, Mob_yy, Mob_zz, mob_floor, _FLOOR_WIDTH)

    # Yield gate, LAST: nothing may run downstream of it, or a zero gets
    # lifted back off zero. `None` skips the multiply so the pre-existing
    # callers match the no-yield path by construction (see the docstring).
    if mob_prefactor is not None:
        Mf_xx = mob_prefactor * Mf_xx
        Mf_xy = mob_prefactor * Mf_xy
        Mf_yy = mob_prefactor * Mf_yy
        Mf_zz = mob_prefactor * Mf_zz

    # M_frozen = 2 Mob_floored . P_floored  (2x2 matmul in-plane; commute,
    # so the product is symmetric -- symmetrize the off-diagonal for
    # round-off insurance, mirroring _affine_exponential_relaxation_step).
    M_xx = 2.0 * (Mf_xx * Pf_xx + Mf_xy * Pf_xy)
    M_xy_a = 2.0 * (Mf_xx * Pf_xy + Mf_xy * Pf_yy)
    M_xy_b = 2.0 * (Mf_xy * Pf_xx + Mf_yy * Pf_xy)
    M_xy = 0.5 * (M_xy_a + M_xy_b)
    M_yy = 2.0 * (Mf_xy * Pf_xy + Mf_yy * Pf_yy)
    M_zz = 2.0 * Mf_zz * Pf_zz

    # A* = -phi_l_eff P_floored^{-1}; P^{-1} = adj(P)/det(P) (2x2), scalar zz.
    detP = Pf_xx * Pf_yy - Pf_xy ** 2
    inv_xx = Pf_yy / detP
    inv_xy = -Pf_xy / detP
    inv_yy = Pf_xx / detP
    As_xx = -phi_l * inv_xx
    As_xy = -phi_l * inv_xy
    As_yy = -phi_l * inv_yy
    As_zz = -phi_l / Pf_zz

    K = (K_xx, K_xy, K_yy, K_zz)
    M_frozen = (M_xx, M_xy, M_yy, M_zz)
    A_star = (As_xx, As_xy, As_yy, As_zz)
    return K, M_frozen, A_star


def _mob_floor_for(mobility: str) -> float:
    """Mobility-block min-eigenvalue floor (#2b): strictly
    positive ``_P_FLOOR`` for the smooth (softplus) mobility, PSD (``0``)
    for the yield-capable ``relu_annealed`` mobility (the zero-set IS the
    yield surface)."""
    return 0.0 if mobility == 'relu_annealed' else _P_FLOOR


_SARAMITO_TDNORM_FLOOR = 1e-12
"""Floor on ``|tau_d|`` so ``tau_y/|tau_d|`` is finite at the rest state."""

_SARAMITO_SMOOTH = 1e-3
"""Smoothing width for the yield prefactor / ``kappa_y`` softplus floor."""

_YIELD_PREF_FLOOR = 0.0
"""Lower clamp on the V2 yield prefactor before it gates the mobility block.

``0.0`` allows an exact ``M_frozen = 0`` (true arrest), which is the point of
gating after the #2b floor. The risk it carries is in the ADJOINT, not the
forward: ``lc._affine_exponential_relaxation_step`` routes through
``lc._exp_2x2_general``, whose eigen-gap ``sqrt`` argument vanishes at
``M = 0``; ``_SQRT_GUARD`` keeps the forward finite but the derivative of the
guarded sqrt there is ~5e14, so ``0 * inf -> nan`` in the backward pass is a
live possibility. The Saramito generator hits ``M = 0`` constantly but is
never differentiated, so that path is untested. Gate G1 differentiates a
sub-yield forward and decides: if the adjoint is finite at ``0.0`` this stays
``0.0``; otherwise it moves to ``1e-12``, which costs a relaxation rate of
``1e-12/lam`` (utterly negligible against the 0.05 it replaces) and keeps the
sqrt argument off the guard. Recorded in the checkpoint metadata either way so
the value used by any fit is recoverable."""


def tau_d_norm_from_K(K_xx, K_xy, K_yy, K_zz, Gp):
    """von-Mises norm of the deviator of ``tau_p = Gp K(A)``.

    ``|tau_d| = Gp * sqrt(0.5 tau_d:tau_d)`` with the double-where
    sqrt guard (``val = 0`` at rest => inf derivative => ``0*inf = nan`` in
    AD without the guard). Used by the V2 yield prefactor; at OB-init
    ``K = A - I`` this matches :func:`saramito_tau_d_norm`.
    """
    trK = K_xx + K_yy + K_zz
    dxx = K_xx - trK / 3.0
    dyy = K_yy - trK / 3.0
    dzz = K_zz - trK / 3.0
    dxy = K_xy
    val = 0.5 * (dxx ** 2 + dyy ** 2 + dzz ** 2 + 2.0 * dxy ** 2)
    safe = val > 0.0
    val_safe = jnp.where(safe, val, 1.0)
    return Gp * jnp.where(safe, jnp.sqrt(val_safe), 0.0)


def yield_prefactor_scalar(tau_d_norm, tau_y,
                           smooth: float = _SARAMITO_SMOOTH):
    """Saramito yield prefactor ``smoothmax(0, 1 - tau_y/|tau_d|)``.

    Reuses the generator's smooth Peterlin floor (width ``_SARAMITO_SMOOTH``).
    ``tau_y = 0`` => prefactor ``== 1`` (machine-exact OB limit).
    """
    td_eff = jnp.maximum(tau_d_norm, _SARAMITO_TDNORM_FLOOR)
    g = 1.0 - tau_y / td_eff
    return smooth * jax.nn.softplus(g / smooth)


def tbnn_K_and_frozen(A_xx, A_xy, A_yy, A_zz, theta, bound_c,
                      *, anchored: bool = TBNN_DEFAULT_ANCHORED,
                      mobility: str = TBNN_DEFAULT_MOBILITY,
                      yield_mode: str = TBNN_DEFAULT_YIELD_MODE,
                      kappa: float = TBNN_DEFAULT_KAPPA,
                      tau_y=None, Gp=None):
    """Heads-from-theta + assembly. The closure's main
    entry point: computes the floored effective heads (threading the
    ``anchored`` / ``mobility`` / ``kappa`` static switches) and routes
    them through :func:`tbnn_assemble_from_heads`. Both the relaxation
    slot and the stress readout call this, so ``K`` (and hence the stress)
    is consistent with ``M_frozen``/``A*`` by construction. Oldroyd-B init
    (either mobility at ``kappa = 1``, either anchoring at ``N == 0``):
    ``M_frozen = I``, ``A* = I``, ``K = A - I``.
    """
    (phi_tau, phi_p2, phi_l_eff, m0, m1, _) = tbnn_effective_heads(
        A_xx, A_xy, A_yy, A_zz, theta, bound_c,
        anchored=anchored, mobility=mobility, kappa=kappa)
    mob_floor = _mob_floor_for(mobility)
    mob_prefactor = None
    if yield_mode == 'scalar':
        # Static branch: only compiled for V2 registrations. Prefactor on
        # the WHOLE mobility block. |tau_d| is read from the FIXED Hookean
        # map ``tau = Gp (A - I)`` (:func:`saramito_tau_d_norm`, the closed
        # form the Saramito generator itself uses) -- NOT from the learned
        # ``Gp K(A)``. This is deliberate: with the learned ruler, theta warps
        # K to carry mis-scaled (Gp, lam) and drags the yield criterion with
        # it, degenerating the prefactor (the v2_prod "two-ruler" failure:
        # 100% yielded, plug=0). The Hookean ruler depends on (A, Gp) only, so
        # the criterion is decoupled from theta. At OB-init K = A - I this is
        # identical to the old path (tau_y=0 => pref==1 => V1 bit-identity).
        #
        # The prefactor is handed to the assembly and applied AFTER the #2b
        # mobility floor. Scaling (m0, m1) here instead -- the pre-fix path --
        # put the floor downstream of the gate, so pref = 0 gave
        # M_frozen = 0.0500672 I rather than 0 and arrest was unreachable for
        # every theta and every tau_y. See tbnn_assemble_from_heads.
        td = saramito_tau_d_norm(A_xx, A_xy, A_yy, A_zz, Gp)
        pref = yield_prefactor_scalar(td, tau_y)
        mob_prefactor = jnp.maximum(pref, _YIELD_PREF_FLOOR)
    return tbnn_assemble_from_heads(
        A_xx, A_xy, A_yy, A_zz, phi_tau, phi_p2, phi_l_eff, m0, m1,
        mob_floor=mob_floor, mob_prefactor=mob_prefactor)


def tbnn_floor_diagnostics(A_xx, A_xy, A_yy, A_zz, theta, bound_c,
                           active_tol: float = 1e-12,
                           *, anchored: bool = TBNN_DEFAULT_ANCHORED,
                           mobility: str = TBNN_DEFAULT_MOBILITY,
                           kappa: float = TBNN_DEFAULT_KAPPA,
                           ) -> Dict[str, jnp.ndarray]:
    """Per-snapshot floor-activation and stiffness monitor.

    Reports, over the whole grid: the smooth-floor *shifts* for ``P``
    (#2a), the mobility block (#2b) and the ``det`` feature (#1); the
    coercivity clamp residual for ``phi_l`` (#4); the pre-floor
    min-eigenvalues and ``-phi_l`` with their margins to the thresholds;
    the min/max eigenvalues of ``M_frozen`` (frozen-split stiffness); and
    the fraction of cells with ANY active floor (> ``active_tol``).

    The recovery gates require ``active_fraction ~ 0`` at convergence
   ; the selftest asserts every shift/residual is
    <= 1e-12 at OB-init so future G1 residuals are attributable
   .
    """
    shape = A_xx.shape
    mob_floor = _mob_floor_for(mobility)
    x1, x2, x3 = tbnn_invariant_features(A_xx, A_xy, A_yy, A_zz)
    X = jnp.stack([x1.reshape(-1), x2.reshape(-1), x3.reshape(-1)], axis=-1)
    phi_tau, phi_p2, phi_l, m0, m1, _ = tbnn_heads(
        theta, X, bound_c, anchored=anchored, mobility=mobility, kappa=kappa)
    rs = lambda v: v.reshape(shape)
    phi_tau, phi_p2, phi_l = rs(phi_tau), rs(phi_p2), rs(phi_l)
    m0, m1 = rs(m0), rs(m1)

    # Floor #1 (det feature) shift.
    det_A2 = A_xx * A_yy - A_xy ** 2
    det_shift = _smooth_floor(det_A2, _DET_FLOOR, _DET_FLOOR) - det_A2

    # Floor #2a (P).
    P_xx = phi_tau + 2.0 * phi_p2 * A_xx
    P_xy = 2.0 * phi_p2 * A_xy
    P_yy = phi_tau + 2.0 * phi_p2 * A_yy
    P_zz = phi_tau + 2.0 * phi_p2 * A_zz
    (Pf_xx, Pf_xy, Pf_yy, Pf_zz), P_min, P_shift = _floor_coaxial_block(
        P_xx, P_xy, P_yy, P_zz, _P_FLOOR, _FLOOR_WIDTH)

    # Floor #2b (mobility block).
    Mob_xx = m0 + m1 * A_xx
    Mob_xy = m1 * A_xy
    Mob_yy = m0 + m1 * A_yy
    Mob_zz = m0 + m1 * A_zz
    (_, _, _, _), Mob_min, Mob_shift = _floor_coaxial_block(
        Mob_xx, Mob_xy, Mob_yy, Mob_zz, mob_floor, _FLOOR_WIDTH)

    # Floor #4 (coercivity clamp on phi_l).
    phi_l_eff = -_smooth_floor(-phi_l, _PHIL_MIN, _FLOOR_WIDTH)
    phil_clamp = phi_l_eff - phi_l  # >= 0 (clamped up toward -_PHIL_MIN)

    # M_frozen eigenvalues (stiffness): 2 Mob_f . P_f.
    (Mfb_xx, Mfb_xy, Mfb_yy, Mfb_zz), _, _ = _floor_coaxial_block(
        Mob_xx, Mob_xy, Mob_yy, Mob_zz, mob_floor, _FLOOR_WIDTH)
    Mfr_xx = 2.0 * (Mfb_xx * Pf_xx + Mfb_xy * Pf_xy)
    Mfr_xy = 2.0 * (Mfb_xx * Pf_xy + Mfb_xy * Pf_yy)
    Mfr_yy = 2.0 * (Mfb_xy * Pf_xy + Mfb_yy * Pf_yy)
    Mfr_zz = 2.0 * Mfb_zz * Pf_zz
    Mfrozen_min = _coaxial_min_eig(Mfr_xx, Mfr_xy, Mfr_yy, Mfr_zz)
    # max-eig = -min-eig(-M).
    Mfrozen_max = -_coaxial_min_eig(-Mfr_xx, -Mfr_xy, -Mfr_yy, -Mfr_zz)

    any_active = ((det_shift > active_tol) | (P_shift > active_tol) |
                  (Mob_shift > active_tol) | (phil_clamp > active_tol))

    return {
        'det_shift_max': jnp.max(det_shift),
        'P_shift_max': jnp.max(P_shift),
        'mob_shift_max': jnp.max(Mob_shift),
        'phil_clamp_max': jnp.max(phil_clamp),
        'P_min_eig': jnp.min(P_min),
        'mob_min_eig': jnp.min(Mob_min),
        'neg_phil_min': jnp.min(-phi_l),
        'P_margin': jnp.min(P_min) - _P_FLOOR,
        'mob_margin': jnp.min(Mob_min) - mob_floor,
        'phil_margin': jnp.min(-phi_l) - _PHIL_MIN,
        'M_frozen_min_eig': jnp.min(Mfrozen_min),
        'M_frozen_max_eig': jnp.max(Mfrozen_max),
        'active_fraction': jnp.mean(any_active.astype(jnp.float64)),
    }


# ---------------------------------------------------------------------------
# Relaxation slot + stress readout (the registry contract).
# ---------------------------------------------------------------------------

def _bound_c_from_params(params):
    try:
        return lc._params_get(params, 'tbnn_bound_c')
    except (KeyError, AttributeError):
        return TBNN_DEFAULT_BOUND_C


def _kappa_from_params(params):
    """Runtime annealing temperature ``kappa`` for the ``relu_annealed``
    mobility (static float in ``params['tbnn_kappa']``; default 1.0).
    NOT a pytree leaf, NOT optimized -- changing it triggers a recompile,
    which is the intended annealing mechanism."""
    try:
        return float(lc._params_get(params, 'tbnn_kappa'))
    except (KeyError, AttributeError, TypeError):
        return TBNN_DEFAULT_KAPPA


def _make_tbnn_relaxation_fn(anchored: bool, mobility: str,
                             yield_mode: str = TBNN_DEFAULT_YIELD_MODE):
    """Factory: build a TBNN relaxation slot with the ``(anchored,
    mobility, yield_mode)`` static config **pinned** at registration time
    (thin wrappers, ONE closure body). The returned
    ``relaxation_fn`` reads ``theta``, ``lam``, ``bound_c`` and (for the
    ``relu_annealed`` mobility only) ``kappa`` from ``params``; for
    ``yield_mode='scalar'`` it also reads ``tau_y`` and ``Gp``. It builds
    ``(M_frozen, A*)`` frozen at the post-advection ``A_pre`` (same freeze
    point as Giesekus / FENE-P) and calls the existing
    ``lc._affine_exponential_relaxation_step``. ``velocity`` is unused in
    all three tiers (the slot carries it for a future flow-coupling head).
    """
    if mobility not in TBNN_MOBILITY_MODES:
        raise ValueError(f"mobility must be one of {TBNN_MOBILITY_MODES}")
    if yield_mode not in TBNN_YIELD_MODES:
        raise ValueError(f"yield_mode must be one of {TBNN_YIELD_MODES}")
    read_kappa = (mobility == 'relu_annealed')
    read_yield = (yield_mode == 'scalar')

    def _relaxation_fn(A_xx, A_xy, A_yy, A_zz, velocity, dt, params):
        del velocity  # state-only coefficients (plan; flow-coupling OOS).
        lam = lc._params_get(params, 'lam')
        theta = lc._params_get(params, 'theta')
        bound_c = _bound_c_from_params(params)
        kappa = _kappa_from_params(params) if read_kappa else TBNN_DEFAULT_KAPPA
        if read_yield:
            tau_y = lc._params_get(params, 'tau_y')
            Gp = lc._params_get(params, 'Gp')
            _, M_frozen, A_star = tbnn_K_and_frozen(
                A_xx, A_xy, A_yy, A_zz, theta, bound_c,
                anchored=anchored, mobility=mobility, kappa=kappa,
                yield_mode='scalar', tau_y=tau_y, Gp=Gp)
        else:
            _, M_frozen, A_star = tbnn_K_and_frozen(
                A_xx, A_xy, A_yy, A_zz, theta, bound_c,
                anchored=anchored, mobility=mobility, kappa=kappa)
        M_xx, M_xy, M_yy, M_zz = M_frozen
        As_xx, As_xy, As_yy, As_zz = A_star
        return lc._affine_exponential_relaxation_step(
            A_xx, A_xy, A_yy, A_zz,
            M_xx, M_xy, M_yy, M_zz,
            As_xx, As_xy, As_yy, As_zz,
            dt, lam)

    return _relaxation_fn


def _tbnn_K_from_memory(memory_fields, params, *,
                        anchored: bool = TBNN_DEFAULT_ANCHORED,
                        mobility: str = TBNN_DEFAULT_MOBILITY):
    """Shared helper: ``Gp`` and ``K(A)`` components from the state fields
    (used by both the in-plane stress readout and the zz diagnostic).
    ``K`` depends only on the potential (``anchored``); the mobility/kappa
    do not enter the stress, but are threaded for uniformity."""
    A_xx_var, A_xy_var, A_yy_var, A_zz_var = memory_fields
    Gp = lc._params_get(params, 'Gp')
    theta = lc._params_get(params, 'theta')
    bound_c = _bound_c_from_params(params)
    kappa = _kappa_from_params(params) if mobility == 'relu_annealed' else TBNN_DEFAULT_KAPPA
    K, _, _ = tbnn_K_and_frozen(
        A_xx_var.array.data, A_xy_var.array.data,
        A_yy_var.array.data, A_zz_var.array.data, theta, bound_c,
        anchored=anchored, mobility=mobility, kappa=kappa)
    return Gp, K, A_xx_var.grid


def _make_tbnn_stress_readout_fn(anchored: bool, mobility: str):
    """Factory: build the TBNN stress readout ``tau_p = Gp K(A)`` (in-plane
    triple) with the ``(anchored, mobility)`` static config pinned. The
    potential-derived non-Hookean ``K`` (the FENE-P pattern) reaches
    momentum through the same plumbing ``Gp`` already exercises. At the
    exact-OB init ``K = A - I`` so this reduces to ``Gp (A - I)``."""
    def _stress_readout_fn(memory_fields, velocity, params):
        del velocity
        Gp, K, grid = _tbnn_K_from_memory(
            memory_fields, params, anchored=anchored, mobility=mobility)
        K_xx, K_xy, K_yy, _ = K
        return (
            lc.GridArray(Gp * K_xx, lc.CELL_CENTER_OFFSET_2D, grid),
            lc.GridArray(Gp * K_xy, lc.CELL_CENTER_OFFSET_2D, grid),
            lc.GridArray(Gp * K_yy, lc.CELL_CENTER_OFFSET_2D, grid),
        )
    return _stress_readout_fn


def tbnn_tau_zz_readout(memory_fields, params, *,
                        anchored: bool = TBNN_DEFAULT_ANCHORED,
                        mobility: str = TBNN_DEFAULT_MOBILITY):
    """Out-of-plane TBNN normal stress ``tau_zz = Gp K_zz`` (diagnostic
    only, for ``N2 = tau_yy - tau_zz``). Mirrors
    :func:`lc.fene_p_tau_zz_readout`; never feeds 2-D momentum. Pass the
    matching ``anchored`` / ``mobility`` for a non-default registration."""
    Gp, K, grid = _tbnn_K_from_memory(
        memory_fields, params, anchored=anchored, mobility=mobility)
    _, _, _, K_zz = K
    return lc.GridArray(Gp * K_zz, lc.CELL_CENTER_OFFSET_2D, grid)


# Default (Tier-1) instances -- back-compatible module-level names. These
# pin (anchored=True, mobility='softplus', yield_mode='off') so the default
# registration path is op-for-op identical to the pre-toggle Tier-1 code
# (same ops as the pre-toggle Tier-1 path).
_tbnn_relaxation_from_params = _make_tbnn_relaxation_fn(
    TBNN_DEFAULT_ANCHORED, TBNN_DEFAULT_MOBILITY, TBNN_DEFAULT_YIELD_MODE)
_tbnn_stress_readout_fn = _make_tbnn_stress_readout_fn(
    TBNN_DEFAULT_ANCHORED, TBNN_DEFAULT_MOBILITY)
_tbnn_yield_relaxation_from_params = _make_tbnn_relaxation_fn(
    True, 'softplus', 'scalar')


# ---------------------------------------------------------------------------
# Saramito EVP data generator (hard-coded Bingham-type truth; NOT a TBNN).
# In conformation form on the SAME
# affine integrator:
#     tau     = Gp (A - I)                          (shared Hookean readout)
#     tau_d   = tau - (tr tau / 3) I                (deviator over xx,xy,yy,zz)
#     |tau_d| = sqrt(1/2 tau_d:tau_d)
#     kappa_y = max(0, 1 - tau_y/|tau_d|)           (smooth-max softplus floor)
#     dA/dt   = -(kappa_y/lam)(A - I)  =>  M = kappa_y I, A* = I
# tau_y = 0 => kappa_y == 1 => exact Oldroyd-B. Below
# yield kappa_y = 0 => M = 0 => relaxation is the identity (exp(0) = I); the
# affine integrator already handles M = 0 cleanly -- do NOT "fix" it.
# (_SARAMITO_TDNORM_FLOOR and _SARAMITO_SMOOTH defined above the V2 helpers.)
# ---------------------------------------------------------------------------

def saramito_tau_d_norm(A_xx, A_xy, A_yy, A_zz, Gp):
    """von-Mises norm of the deviator of ``tau = Gp(A - I)``, in CLOSED
    FORM through the Hookean map:

        |tau_d| = Gp * sqrt( 1/2 [ p2 - 2 tau + 3 - (tau - 3)^2 / 3 ] )

    with ``tau = tr A`` and ``p2 = tr A^2`` (the same invariants the TBNN
    features use). Derived once here and reused for the yield-locus overlay
    (``|tau_d|(x) = tau_y`` is then a closed-form curve in ``(tau, p2)``).
    The selftest cross-checks this against the explicit component deviator.
    """
    trA = A_xx + A_yy + A_zz
    p2 = A_xx ** 2 + 2.0 * A_xy ** 2 + A_yy ** 2 + A_zz ** 2
    val = 0.5 * (p2 - 2.0 * trA + 3.0 - (trA - 3.0) ** 2 / 3.0)
    # AD-safe sqrt: input-safe / output-masked double-where (the project's
    # standard sqrt-at-zero guard). At the rest state
    # A = I, val = 0 and a naive sqrt has an INFINITE derivative -> 0*inf =
    # nan in the backward pass (this nan'd dL/dGp, dL/dlam at tau_y=0). The
    # dummy positive argument under the unselected branch keeps the gradient
    # finite while the forward value is exact (0 where val <= 0).
    safe = val > 0.0
    val_safe = jnp.where(safe, val, 1.0)
    return Gp * jnp.where(safe, jnp.sqrt(val_safe), 0.0)


def saramito_kappa_y(A_xx, A_xy, A_yy, A_zz, Gp, tau_y,
                     smooth: float = _SARAMITO_SMOOTH):
    """Saramito yield factor ``kappa_y = max(0, 1 - tau_y/|tau_d|)`` with a
    smooth Peterlin-style softplus floor at 0. ``tau_y = 0 => kappa_y == 1``
    (exact Oldroyd-B); below yield (``|tau_d| < tau_y``) ``kappa_y -> 0``
    smoothly (frozen relaxation)."""
    td = saramito_tau_d_norm(A_xx, A_xy, A_yy, A_zz, Gp)
    td_eff = jnp.maximum(td, _SARAMITO_TDNORM_FLOOR)
    g = 1.0 - tau_y / td_eff
    return smooth * jax.nn.softplus(g / smooth)


def _saramito_relaxation_from_params(A_xx, A_xy, A_yy, A_zz, velocity, dt,
                                     params):
    """Saramito (Bingham-type EVP) relaxation slot -- the yield-capable
    data generator. Frozen ``M = kappa_y I``,
    ``A* = I``, routed through the shared affine integrator. Reads
    ``lam``, ``Gp``, ``tau_y`` from ``params``. Below yield ``kappa_y = 0``
    ``=> M = 0`` and the integrator returns the identity step.

    The anchored V2 TBNN reproduces this exactly, above AND below yield, now
    that its yield prefactor gates the mobility block after the #2b floor
    (``tbnn_assemble_from_heads``'s ``mob_prefactor``): at OB-init the floored
    block is ``I`` and ``P_f = 0.5 I``, so ``M_frozen = pref I`` against this
    generator's ``M = kappa_y I`` with the same prefactor function. Before
    that fix the sentence here read "the EVP behaviour the anchored TBNN
    cannot produce", which was true of the code as written -- the floor sat
    downstream of the gate and pinned ``M_frozen`` at ``0.05 I`` below yield --
    but is no longer true of the closure. Gate G5 is the below-yield
    equivalence test; the pre-fix gate only compared above yield, which is why
    the defect went unseen."""
    del velocity
    lam = lc._params_get(params, 'lam')
    Gp = lc._params_get(params, 'Gp')
    tau_y = lc._params_get(params, 'tau_y')
    kappa_y = saramito_kappa_y(A_xx, A_xy, A_yy, A_zz, Gp, tau_y)
    zero = jnp.zeros_like(A_xy)
    one = jnp.ones_like(A_xy)
    return lc._affine_exponential_relaxation_step(
        A_xx, A_xy, A_yy, A_zz,
        kappa_y, zero, kappa_y, kappa_y,          # M = kappa_y I
        one, zero, one, one,                      # A* = I
        dt, lam)


# ---------------------------------------------------------------------------
# Registrations (kernel-restart on re-import; cr.register refuses duplicates).
# Three THIN WRAPPERS pinning the (anchored, mobility) static config through
# the ONE closure body above, plus the Saramito EVP data generator. Each
# registration comment gives its `= (anchored, mobility) -> model class`.
# ---------------------------------------------------------------------------

# = (anchored=True, mobility='softplus') -> Tier 1 (viscoelastic, no yield).
# Must match the pre-toggle Tier-1 path and the Oldroyd-B init vs
# oldroyd_b_logconf_bk_v2 (the safety net proving the switches did not
# disturb the trained model). Uses the default-config
# factory instances (_tbnn_relaxation_from_params / _tbnn_stress_readout_fn).
TBNN_POTENTIAL_LOGCONF_BK_V2: cr.ConstitutiveModel = cr.register(
    cr.ConstitutiveModel(
        name='tbnn_potential_logconf_bk_v2',
        state_spec=lc._LOGCONF_STATE_SPEC,
        evolution_fn=lc.make_logconf_evolution_fn(
            psi_kernel='bk', uc_method='analytic', advect_method='rk2',
            relaxation_fn=_tbnn_relaxation_from_params),
        stress_readout_fn=_tbnn_stress_readout_fn,
        coupling_mode='explicit_force',
        polymer_linearization_fn=None,
    ))
"""Tier-1 anchored potential-mobility TBNN closure -- ``= (anchored=True,
mobility='softplus')``.

The ``bk_v2`` kinematic backbone (BK eigenvalue-free Psi<->A +
Fattal-Kupferman analytic UC stretch + SSP-RK2 advection) with the
learned relaxation source ``R(A) = -(1/lam)(m0 I + m1 A) K(A)`` and the
potential-derived non-Hookean stress ``tau = Gp K(A)``,
``K = 2 phi_tau A + 4 phi_p2 A^2 + 2 phi_l_eff I``. Heads live in
``params['theta']`` (a pure array pytree from :func:`init_tbnn_theta`),
``bound_c`` in ``params['tbnn_bound_c']``. At the exact-OB init it is
analytically Oldroyd-B (``M_frozen = I``, ``A* = I``, ``K = A - I``;
gate G1). Routes through the shared SPD-safe affine integrator
``lc._affine_exponential_relaxation_step`` -- additive, no edits to
``log_conformation``. The default-config path matches the pre-toggle
Tier-1 closure."""


# = (anchored=True, mobility='softplus', yield_mode='scalar') -> V2.
# V2 = V1 + yield prefactor; tau_y=0 => V1 exactly.
TBNN_POTENTIAL_YIELD_LOGCONF_BK_V2: cr.ConstitutiveModel = cr.register(
    cr.ConstitutiveModel(
        name='tbnn_potential_yield_logconf_bk_v2',
        state_spec=lc._LOGCONF_STATE_SPEC,
        evolution_fn=lc.make_logconf_evolution_fn(
            psi_kernel='bk', uc_method='analytic', advect_method='rk2',
            relaxation_fn=_tbnn_yield_relaxation_from_params),
        stress_readout_fn=_tbnn_stress_readout_fn,
        coupling_mode='explicit_force',
        polymer_linearization_fn=None,
    ))
"""V2 structured-yield scalar TBNN closure -- ``= (anchored=True,
mobility='softplus', yield_mode='scalar')``.

Tier-1 anchored potential-mobility body plus a multiplicative Saramito-style
yield prefactor on the whole mobility block ``(m0 I + m1 A)``, applied
**after** the #2b min-eigenvalue floor so that ``pref = 0`` gives an exact
``M_frozen = 0`` (true arrest). ``tau_y`` is a scalar in ``params`` (NOT in
``theta``). Plain ``softplus`` ``m0`` (no ``kappa`` anneal). At OB-init with
fixed ``tau_y`` the closure matches the Saramito generator above and below
yield; ``tau_y = 0`` reproduces V1 to machine precision."""


# = (anchored=False, mobility='softplus') -> Unanchored elastic (learned
# rest state, still smooth; intermediate / nematic-capable, 0D only).
TBNN_POTENTIAL_UNANCHORED_LOGCONF_BK_V2: cr.ConstitutiveModel = cr.register(
    cr.ConstitutiveModel(
        name='tbnn_potential_unanchored_logconf_bk_v2',
        state_spec=lc._LOGCONF_STATE_SPEC,
        evolution_fn=lc.make_logconf_evolution_fn(
            psi_kernel='bk', uc_method='analytic', advect_method='rk2',
            relaxation_fn=_make_tbnn_relaxation_fn(False, 'softplus')),
        stress_readout_fn=_make_tbnn_stress_readout_fn(False, 'softplus'),
        coupling_mode='explicit_force',
        polymer_linearization_fn=None,
    ))
"""Unanchored elastic TBNN closure -- ``= (anchored=False,
mobility='softplus')``. Swap 1 only: drop the value-and-gradient anchor
(``phi = phi_OB + N(x)`` directly), still-positive softplus mobility (no
yield). The rest state is learned from data; coercivity is the Sec. 0.7
floor #4 barrier (no ``c*tanh`` bound, ). Intermediate
modality between Tier 1 and Tier 3; nematic-capable (0D showcase)."""


# = (anchored=False, mobility='relu_annealed') -> Tier 3 (full EVP / yield).
TBNN_POTENTIAL_FREE_LOGCONF_BK_V2: cr.ConstitutiveModel = cr.register(
    cr.ConstitutiveModel(
        name='tbnn_potential_free_logconf_bk_v2',
        state_spec=lc._LOGCONF_STATE_SPEC,
        evolution_fn=lc.make_logconf_evolution_fn(
            psi_kernel='bk', uc_method='analytic', advect_method='rk2',
            relaxation_fn=_make_tbnn_relaxation_fn(False, 'relu_annealed')),
        stress_readout_fn=_make_tbnn_stress_readout_fn(False, 'relu_annealed'),
        coupling_mode='explicit_force',
        polymer_linearization_fn=None,
    ))
"""Tier-3 unanchored + yield-capable TBNN closure -- ``= (anchored=False,
mobility='relu_annealed')``. Both swaps: unanchored potential (Swap 1) AND
the annealed smooth-ReLU mobility ``m0 = kappa*softplus(raw0/kappa)``
(Swap 2). The mobility zero-set (rigorously, the zero-set of
``min-eig(m0 I + m1 A)``) is the learned yield surface; floor #2b is PSD
here. ``kappa`` is annealed via ``params['tbnn_kappa']`` (static float,
never a pytree leaf). This is the reviewer-response deliverable (yield
stress / EVP capability)."""


# ---------------------------------------------------------------------------
# Saramito EVP data generator registration (hard-coded Bingham truth). Uses
# the SHARED Hookean readout (tau = Gp(A - I)); the only model-specific piece
# is _saramito_relaxation_from_params (M = kappa_y I, A* = I). tau_y = 0 =>
# exact Oldroyd-B. Params: Gp, lam, tau_y.
# ---------------------------------------------------------------------------

SARAMITO_LOGCONF_BK_V2: cr.ConstitutiveModel = cr.register(
    cr.ConstitutiveModel(
        name='saramito_logconf_bk_v2',
        state_spec=lc._LOGCONF_STATE_SPEC,
        evolution_fn=lc.make_logconf_evolution_fn(
            psi_kernel='bk', uc_method='analytic', advect_method='rk2',
            relaxation_fn=_saramito_relaxation_from_params),
        stress_readout_fn=lc._linear_conformation_stress_readout_fn,
        coupling_mode='explicit_force',
        polymer_linearization_fn=None,
    ))
"""Saramito Bingham-type EVP data generator. The
``bk_v2`` kinematic backbone with the yield-capable relaxation
``dA/dt = -(kappa_y/lam)(A - I)``, ``kappa_y = max(0, 1 - tau_y/|tau_d|)``
(smooth softplus floor), frozen ``M = kappa_y I``, ``A* = I``. Reuses the
shared Hookean stress readout ``tau = Gp(A - I)``. ``tau_y = 0`` reproduces
Oldroyd-B to machine precision; this is the yield
truth the unanchored Tier-3 TBNN is fit to. Additive, no edits to
``log_conformation``."""

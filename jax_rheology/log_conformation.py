"""Log-conformation viscoelastic models: Oldroyd-B, Giesekus, FENE-P, linear PTT.

The shared machinery is the 2D log-conformation formulation, following the
Hao-Pan / Fattal-Kupferman three-stage split

    (a) upper-convective:  d_t Psi = 2 B + (Omega.Psi - Psi.Omega)
    (b) advection:         d_t Psi + div (Psi u) = 0
    (c) relaxation:        d_t A = - (A - I) / lam   (analytic)

with stress readout  tau_p = G_p . (A - I).

All inputs and outputs that interact with the surrounding solver are
:class:`GridVariable`\\ s at cell-center offset ``(0.5, 0.5)``; the
internal math operates on raw ``jax.numpy`` arrays.

The reference for every algebraic identity is the Basilisk source
``log-conform-viscoelastic-scalar-2D.h`` (Sanjay 2024, vendored copy of
``http://basilisk.fr/src/log-conform.h``).

Numerical hazards:
  * Becker-swap risk: the closed-form 2x2 eigendecomposition is
    rank-deficient on the manifold A prop to I. Every site that can hit
    ``0/0`` (eigenvector normalisation, the ``Lambda_y - Lambda_x`` denominator
    in ``omega``) uses a double-where guard; the affected branches are
    marked with ``# BECKER-SWAP``. Prefer the Becker-Knechtges
    eigenvalue-free kernel below when you need a clean AD path through
    that manifold.
  * Slow-manifold drift: no SPD projection is applied here.
    The forward dynamics is exact for Oldroyd-B; spurious negative
    eigenvalues would indicate either timestep instability or an
    AD-induced perturbation that needs the dedicated guard.
  * Advection and relaxation stay first-order (forward Euler /
    analytic exponential) on the eig-path presets; the ``_bk_v2``
    presets raise those orders independently.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax.numpy as jnp

from jax_ib.base import grids
from jax_ib.base import finite_differences as fd
from jax_ib.base import advection as adv
from jax_ib.base import interpolation  # wall-aware grad_u at Dirichlet rows

from jax_rheology.models import registry as cr


GridArray = grids.GridArray
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector


# ---------------------------------------------------------------------------
# Tunable safety constants
# ---------------------------------------------------------------------------

# Threshold below which we treat A.xy^2 as "diagonal" and short-circuit the
# eigendecomposition to R = I. Mirrors Basilisk's `sq(A->x.y) < 1e-15`.
_AXY_SQ_DEGEN_THRESHOLD = 1e-15

# Threshold below which we treat (Lambda_x - Lambda_y) as "degenerate" and replace
# the upper-convective B by the symmetric strain-rate D. Mirrors
# Basilisk's `fabs(Lambda.x - Lambda.y) <= 1e-20`.
_LAMBDA_GAP_DEGEN_THRESHOLD = 1e-20


# ---------------------------------------------------------------------------
# Sec.5.A  Closed-form 2x2 symmetric eigendecomposition
# ---------------------------------------------------------------------------

def eig2x2_symmetric(Axx: jnp.ndarray,
                     Axy: jnp.ndarray,
                     Ayy: jnp.ndarray
                     ) -> Tuple[jnp.ndarray, jnp.ndarray,
                                jnp.ndarray, jnp.ndarray,
                                jnp.ndarray, jnp.ndarray]:
    """Eigendecomposition of the 2x2 symmetric matrix ``[[Axx, Axy], [Axy, Ayy]]``.

    Returns ``(lam_x, lam_y, R_xx, R_xy, R_yx, R_yy)`` where the columns
    of ``R = [[R_xx, R_xy], [R_yx, R_yy]]`` are eigenvectors associated
    with ``lam_x`` and ``lam_y`` respectively (the same convention as
    ``Lambda``/``R`` in the Basilisk source).

    The non-degenerate branch follows ``diagonalization_2D`` line for
    line:

        T   = Axx + Ayy
        D   = Axx.Ayy - Axy^2
        Lambda_x = T/2 + sqrt(T^2/4 - D)        (larger root)
        Lambda_y = T/2 - sqrt(T^2/4 - D)        (smaller root)

        column-i of R = normalise( (Axy, Lambda_i - Axx) ).

    On the degenerate manifold ``Axy^2 < _AXY_SQ_DEGEN_THRESHOLD`` the
    matrix is already diagonal, so we short-circuit to ``R = I`` and
    ``Lambda = diag(A)``.

    AD safety: every division uses the double-where
    pattern (``jnp.where(mask, 1, denom)`` inside, ``jnp.where(mask,
    safe_val, expr)`` outside) so neither the forward pass nor the VJP
    can hit a NaN on the degenerate manifold. The non-degenerate branch
    is *not* differentiable through the eigenvector swap that happens
    when the eigenvalues cross -- that is the Becker-swap issue the
    eigenvalue-free kernel below removes.
    """
    Axx = jnp.asarray(Axx)
    Axy = jnp.asarray(Axy)
    Ayy = jnp.asarray(Ayy)

    T = Axx + Ayy
    D = Axx * Ayy - Axy * Axy
    disc_raw = 0.25 * T * T - D

    # Double-where on the discriminant. ``jnp.maximum(disc_raw, 0)``
    # alone is forward-safe but NOT backward-safe: at points where
    # ``disc_raw <= 0`` the chain rule through ``sqrt(max(x, 0))``
    # multiplies ``1/(2.sqrt(0)) = inf`` against ``d_max/dx = 0`` and
    # produces NaN. This bites any gradient evaluation at rest
    # (``A ~= I``) and the bulk of any low-Wi run, where Axy ~= 0
    # and T/2 ~= Axx ~= Ayy makes ``disc_raw`` numerically zero from
    # cancellation.
    # The standard JAX fix is the input-safe / output-masked pair:
    # replace the input before sqrt so the trace never sees the
    # singular point, then mask the output back to the physically
    # correct value where the original input was singular.
    is_disc_bad = disc_raw <= 0.0
    disc_safe = jnp.where(is_disc_bad, 1.0, disc_raw)
    sqrt_disc = jnp.where(is_disc_bad, 0.0, jnp.sqrt(disc_safe))
    lam_x_nd = 0.5 * T + sqrt_disc
    lam_y_nd = 0.5 * T - sqrt_disc

    is_degen = Axy * Axy < _AXY_SQ_DEGEN_THRESHOLD

    # Eigenvector for lam_x: (Axy, lam_x - Axx). Normalise.  # BECKER-SWAP
    # Same double-where pattern as above on the modulus sqrt:
    # at degenerate points ``mod_x_sq ~= 0`` (Axy = 0 => vx_x = 0;
    # lam_x ~= Axx => vy_x ~= 0), so ``sqrt(mod_x_sq)`` would seed
    # ``inf . 0 = NaN`` into the gradient even though the outer
    # ``where`` masks the value.
    vx_x = Axy
    vy_x = lam_x_nd - Axx
    mod_x_sq = vx_x * vx_x + vy_x * vy_x
    mod_x_sq_safe = jnp.where(is_degen, 1.0, mod_x_sq)
    mod_x_safe = jnp.where(is_degen, 1.0, jnp.sqrt(mod_x_sq_safe))
    R_xx_nd = vx_x / mod_x_safe
    R_yx_nd = vy_x / mod_x_safe

    # Eigenvector for lam_y: (Axy, lam_y - Axx).                  # BECKER-SWAP
    vx_y = Axy
    vy_y = lam_y_nd - Axx
    mod_y_sq = vx_y * vx_y + vy_y * vy_y
    mod_y_sq_safe = jnp.where(is_degen, 1.0, mod_y_sq)
    mod_y_safe = jnp.where(is_degen, 1.0, jnp.sqrt(mod_y_sq_safe))
    R_xy_nd = vx_y / mod_y_safe
    R_yy_nd = vy_y / mod_y_safe

    # Outer where: on the degenerate manifold, replace with identity.
    R_xx = jnp.where(is_degen, 1.0, R_xx_nd)
    R_yx = jnp.where(is_degen, 0.0, R_yx_nd)
    R_xy = jnp.where(is_degen, 0.0, R_xy_nd)
    R_yy = jnp.where(is_degen, 1.0, R_yy_nd)

    lam_x = jnp.where(is_degen, Axx, lam_x_nd)
    lam_y = jnp.where(is_degen, Ayy, lam_y_nd)

    return lam_x, lam_y, R_xx, R_xy, R_yx, R_yy


# ---------------------------------------------------------------------------
# Sec.5.B  Psi <-> A change of variables
# ---------------------------------------------------------------------------

def Psi_from_A(Axx: jnp.ndarray,
               Axy: jnp.ndarray,
               Ayy: jnp.ndarray
               ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute ``Psi = log A`` componentwise via the spectral expansion

        Psi = R . diag(log Lambda_x, log Lambda_y) . R^T

    For an SPD conformation tensor the eigenvalues are strictly
    positive; we clamp by ``jnp.maximum(lam, tiny)`` before taking the
    log so a transient AD perturbation that pushes lam to zero does not
    produce NaN. The forward pass is unaffected on the SPD manifold.
    """
    lam_x, lam_y, R_xx, R_xy, R_yx, R_yy = eig2x2_symmetric(Axx, Axy, Ayy)
    log_lam_x = jnp.log(jnp.maximum(lam_x, 1e-300))
    log_lam_y = jnp.log(jnp.maximum(lam_y, 1e-300))
    Psi_xx = R_xx * R_xx * log_lam_x + R_xy * R_xy * log_lam_y
    Psi_yy = R_yx * R_yx * log_lam_x + R_yy * R_yy * log_lam_y
    Psi_xy = R_xx * R_yx * log_lam_x + R_yy * R_xy * log_lam_y
    return Psi_xx, Psi_xy, Psi_yy


def A_from_Psi(Psi_xx: jnp.ndarray,
               Psi_xy: jnp.ndarray,
               Psi_yy: jnp.ndarray
               ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute ``A = exp Psi`` componentwise via the spectral expansion

        A = R . diag(exp Lambda_x, exp Lambda_y) . R^T.
    """
    mu_x, mu_y, R_xx, R_xy, R_yx, R_yy = eig2x2_symmetric(Psi_xx, Psi_xy, Psi_yy)
    exp_mu_x = jnp.exp(mu_x)
    exp_mu_y = jnp.exp(mu_y)
    A_xx = R_xx * R_xx * exp_mu_x + R_xy * R_xy * exp_mu_y
    A_yy = R_yx * R_yx * exp_mu_x + R_yy * R_yy * exp_mu_y
    A_xy = R_xx * R_yx * exp_mu_x + R_yy * R_xy * exp_mu_y
    return A_xx, A_xy, A_yy


# ---------------------------------------------------------------------------
# Upper-convective increment (Fattal-Kupferman B + Omega split)
# ---------------------------------------------------------------------------

def _cell_centered_grad_u(velocity: GridVariableVector
                          ) -> Tuple[jnp.ndarray, jnp.ndarray,
                                     jnp.ndarray, jnp.ndarray]:
    """Return the four cell-centered velocity-gradient components.

    Uses :func:`jax_ib.base.finite_differences.gradient_tensor` which
    contains the staggered-grid offset logic: each velocity component
    is reduced to cell-center ``(0.5, 0.5)`` via a backward / forward /
    central-after-interp difference, picking the one matching the
    component's offset. The result tensor ``G`` is indexed
    ``G[derivative_axis, velocity_component]``, i.e.

        G[0, 0] = du_x/dx,  G[1, 0] = du_x/dy,
        G[0, 1] = du_y/dx,  G[1, 1] = du_y/dy.

    Returns the four components as plain ``jnp`` arrays (the
    cell-centered field values).
    """
    G = fd.gradient_tensor(velocity)
    dux_dx = G[0, 0].data
    dux_dy = G[1, 0].data
    duy_dx = G[0, 1].data
    duy_dy = G[1, 1].data
    return dux_dx, dux_dy, duy_dx, duy_dy


def _cell_centered_grad_u_wall_aware(velocity: GridVariableVector
                                     ) -> Tuple[jnp.ndarray, jnp.ndarray,
                                                jnp.ndarray, jnp.ndarray]:
    """Wall-aware version of :func:`_cell_centered_grad_u`.

    Bulk rows are identical to :func:`_cell_centered_grad_u` (same
    central-difference-with-linear-extrapolation-ghost stencil that
    :func:`fd.gradient_tensor` produces). The two wall rows of the
    *cross-shear* component on each Dirichlet axis are replaced with
    a 2nd-order one-sided stencil that uses the wall Dirichlet value
    directly:

      Top wall (j = N-1):
        (du/dy)|_{N-1} = (4.u_wall - 3.u_{N-1} - u_{N-2}) / (3.dy)
      Bottom wall (j = 0):
        (du/dy)|_0     = (-4.u_wall + 3.u_0 + u_1) / (3.dy)

    Derivation (top wall) -- Taylor expansion around y_{N-1} with
    offsets (-dy, 0, +dy/2) for u_{N-2}, u_{N-1}, u_wall; solve the
    3x3 linear system that kills u_0, u', u'' to 2nd order:

        alpha + beta + gamma = 0
        -alpha.dy + gamma.dy/2 = 1
        alpha.dy^2/2 + gamma.dy^2/8 = 0       (kills u'' term)

      => alpha = -1/(3dy), beta = -1/dy, gamma = 4/(3dy).

    Truncation error is (dy^2/12).u''' = O(dy^2), vs the existing
    centered + linear-extrapolation-ghost stencil whose leading
    error at the wall is -(dy/8).u'' = O(dy) (a Dy-order loss
    relative to the bulk).

    Wall axes are detected from ``velocity[*].bc.types``; only
    axes with ``('dirichlet', 'dirichlet')`` get patched. Periodic
    axes (e.g. the x-direction in Couette) keep the bulk central
    stencil unchanged.

    Cross-shear patch (the only one that matters for Couette):
      * y is wall-bounded for ``u_x``  =>  patch ``dux_dy`` at j=0 and j=Ny-1
      * x is wall-bounded for ``u_y``  =>  patch ``duy_dx`` at i=0 and i=Nx-1

    The diagonal components (``dux_dx``, ``duy_dy``) are left
    untouched because their stencils are forward/backward
    differences on the velocity face values -- already one-sided
    and already use the wall Dirichlet value directly through the
    pad. The cross-shear is the only component whose wall row
    composes a 2nd-order central diff with a 1st-order linear-
    extrap ghost, and that's exactly the truncation loss this fix
    addresses.

    Why this exists: the historical wall stencil has leading truncation
    ``-(dy/8).u''(y_wall)``, so a Couette-like shear carries an O(dy)
    error at the wall row that bleeds into bulk ``A_xy`` (~0.6% at
    typical resolution). The one-sided stencil restores O(dy^2) there.
    """
    G = fd.gradient_tensor(velocity)
    dux_dx = G[0, 0].data
    dux_dy = G[1, 0].data
    duy_dx = G[0, 1].data
    duy_dy = G[1, 1].data

    u_x_var, u_y_var = velocity

    if u_x_var.bc.types[1] == (
            boundaries_BCType_DIRICHLET, boundaries_BCType_DIRICHLET):
        dy = u_x_var.grid.step[1]
        u_wall_bot, u_wall_top = u_x_var.bc.bc_values[1]
        ux_centered = interpolation.linear(u_x_var, u_x_var.grid.cell_center)
        v = ux_centered.data
        Ny = v.shape[1]
        dux_dy_top = (4.0 * u_wall_top - 3.0 * v[:, Ny - 1] - v[:, Ny - 2]) / (3.0 * dy)
        dux_dy_bot = (-4.0 * u_wall_bot + 3.0 * v[:, 0] + v[:, 1]) / (3.0 * dy)
        dux_dy = dux_dy.at[:, Ny - 1].set(dux_dy_top)
        dux_dy = dux_dy.at[:, 0].set(dux_dy_bot)

    if u_y_var.bc.types[0] == (
            boundaries_BCType_DIRICHLET, boundaries_BCType_DIRICHLET):
        dx = u_y_var.grid.step[0]
        v_wall_left, v_wall_right = u_y_var.bc.bc_values[0]
        uy_centered = interpolation.linear(u_y_var, u_y_var.grid.cell_center)
        v = uy_centered.data
        Nx = v.shape[0]
        duy_dx_right = (4.0 * v_wall_right - 3.0 * v[Nx - 1, :] - v[Nx - 2, :]) / (3.0 * dx)
        duy_dx_left = (-4.0 * v_wall_left + 3.0 * v[0, :] + v[1, :]) / (3.0 * dx)
        duy_dx = duy_dx.at[Nx - 1, :].set(duy_dx_right)
        duy_dx = duy_dx.at[0, :].set(duy_dx_left)

    return dux_dx, dux_dy, duy_dx, duy_dy


# String literal for the Dirichlet BC type used by the wall-aware
# gradient helper. We avoid importing the BCType enum directly to
# keep the dependency graph one-way (jax_rheology -> jax_ib).
# Mirrors ``jax_ib.base.boundaries.BCType.DIRICHLET``.
boundaries_BCType_DIRICHLET = 'dirichlet'


def upper_convective_increment(velocity: GridVariableVector,
                               Axx: jnp.ndarray,
                               Axy: jnp.ndarray,
                               Ayy: jnp.ndarray,
                               dt: float,
                               grad_u_fn: Callable = _cell_centered_grad_u,
                               ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(DeltaPsi_xx, DeltaPsi_xy, DeltaPsi_yy)`` from the upper-convective stage.

    Fattal-Kupferman B + Omega split:

      1. Eigendecomposition of A -> (Lambda, R)
      2. Build M = R^T . (gradu)^T . R, then omega, OM, B in physical frame
            (Basilisk lines 247-262). Special case ``|Lambda_x - Lambda_y| <=
            _LAMBDA_GAP_DEGEN_THRESHOLD`` -> B = D (centered symmetric
            strain rate), Omega = 0 (Basilisk lines 235-240).
      3. Increment in Psi-space:
                DeltaPsi_xy = dt . (2 B_xy + OM . (Psi_yy - Psi_xx))
                DeltaPsi_xx = 2 dt . (B_xx + OM . Psi_xy^{old})
                DeltaPsi_yy = 2 dt . (B_yy - OM . Psi_xy^{old})

    The function returns increments (not new values); the caller adds
    them to Psi. Inputs are A components (cell-centered ``jnp`` arrays);
    Psi is recovered internally from A via :func:`Psi_from_A` so callers
    do not need to maintain Psi as a separate cached field.
    """
    # Eigendecomposition of A.
    lam_x, lam_y, R_xx, R_xy, R_yx, R_yy = eig2x2_symmetric(Axx, Axy, Ayy)

    # Psi^{old} for the commutator term in the Psi increment.
    Psi_xx_old, Psi_xy_old, Psi_yy_old = Psi_from_A(Axx, Axy, Ayy)

    # Cell-centered velocity gradient. ``grad_u_fn`` defaults to
    # :func:`_cell_centered_grad_u`; the factory can swap in
    # :func:`_cell_centered_grad_u_wall_aware` via
    # ``wall_stencil='oneside_2nd_order'``.
    # Central-only call (kept commented):
    # dux_dx, dux_dy, duy_dx, duy_dy = _cell_centered_grad_u(velocity)
    dux_dx, dux_dy, duy_dx, duy_dy = grad_u_fn(velocity)

    # M = R^T . (gradu)^T . R (Basilisk lines 247-254).
    # The C `foreach_dimension` macro expands into x.x and y.y formulas
    # built from R's first / second column respectively; M.x.y and
    # M.y.x are listed separately. All four are computed here as plain
    # tensor products.
    M_xx = (R_xx * R_xx * dux_dx
            + R_yx * R_yx * duy_dy
            + R_xx * R_yx * (dux_dy + duy_dx))
    M_yy = (R_xy * R_xy * dux_dx
            + R_yy * R_yy * duy_dy
            + R_xy * R_yy * (dux_dy + duy_dx))
    M_xy = (R_xx * R_xy * dux_dx
            + R_xy * R_yx * duy_dx
            + R_xx * R_yy * dux_dy
            + R_yx * R_yy * duy_dy)
    M_yx = (R_xx * R_xy * dux_dx
            + R_xx * R_yy * duy_dx
            + R_xy * R_yx * dux_dy
            + R_yx * R_yy * duy_dy)

    # omega (eigenbasis) and OM (physical frame).
    lambda_gap = lam_y - lam_x                                # BECKER-SWAP
    is_degen_B = jnp.abs(lambda_gap) <= _LAMBDA_GAP_DEGEN_THRESHOLD
    lambda_gap_safe = jnp.where(is_degen_B, 1.0, lambda_gap)
    omega_nd = (lam_y * M_xy + lam_x * M_yx) / lambda_gap_safe
    omega = jnp.where(is_degen_B, 0.0, omega_nd)
    det_R = R_xx * R_yy - R_xy * R_yx
    OM = det_R * omega

    # B in physical frame.
    B_xx_nd = M_xx * R_xx * R_xx + M_yy * R_xy * R_xy
    B_yy_nd = M_xx * R_yx * R_yx + M_yy * R_yy * R_yy
    B_xy_nd = M_xx * R_xx * R_yx + M_yy * R_yy * R_xy

    # Degenerate-eigenvalue branch: B = D (Basilisk lines 235-240).
    B_xx_d = dux_dx
    B_yy_d = duy_dy
    B_xy_d = 0.5 * (dux_dy + duy_dx)

    B_xx = jnp.where(is_degen_B, B_xx_d, B_xx_nd)
    B_yy = jnp.where(is_degen_B, B_yy_d, B_yy_nd)
    B_xy = jnp.where(is_degen_B, B_xy_d, B_xy_nd)

    # Psi increments.
    dPsi_xy = dt * (2.0 * B_xy + OM * (Psi_yy_old - Psi_xx_old))
    dPsi_xx = 2.0 * dt * (B_xx + OM * Psi_xy_old)
    dPsi_yy = 2.0 * dt * (B_yy - OM * Psi_xy_old)

    return dPsi_xx, dPsi_xy, dPsi_yy


# ---------------------------------------------------------------------------
# Analytic Oldroyd-B relaxation
# ---------------------------------------------------------------------------

def oldroyd_b_relaxation_analytic(Axx: jnp.ndarray,
                                  Axy: jnp.ndarray,
                                  Ayy: jnp.ndarray,
                                  dt: float,
                                  lam: float
                                  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply the analytic Oldroyd-B relaxation over a step ``dt``.

    Closed form of d_t A = - (A - I) / lam:

        A.x.y  <-  A.x.y . exp(-dt/lam)
        A.x.x  <-  (1 - exp(-dt/lam)) + A.x.x . exp(-dt/lam)
        A.y.y  <-  (1 - exp(-dt/lam)) + A.y.y . exp(-dt/lam)

    ``lam`` is a scalar (uniform lam). Spatially-varying lam is not
    implemented.
    """
    int_factor = jnp.exp(-dt / lam)
    Axy_new = Axy * int_factor
    Axx_new = (1.0 - int_factor) + Axx * int_factor
    Ayy_new = (1.0 - int_factor) + Ayy * int_factor
    return Axx_new, Axy_new, Ayy_new


# ---------------------------------------------------------------------------
# Psi-advection (forward-Euler, van-Leer TVD)
# ---------------------------------------------------------------------------

def _advect_psi_components_euler(psi_vars: Tuple[GridVariable, ...],
                                 velocity: GridVariableVector,
                                 dt: float
                                 ) -> Tuple[jnp.ndarray, ...]:
    """Advect an arbitrary tuple of Psi components by ``velocity`` for ``dt``.

    Uses van-Leer TVD limiting (matches the Basilisk
    ``advection`` call after the upper-convective update). Forward
    Euler in time: ``Psi <- Psi + dt . ( - div (Psi u) )``.

    Each component is transported by an **independent** van-Leer call --
    there is no cross-component coupling in the advection operator -- so
    advecting a 4-tuple ``(Psi_xx, Psi_xy, Psi_yy, Psi_zz)`` returns the same
    first three arrays byte-for-byte as advecting the legacy 3-tuple
    (the ``A_zz`` channel rides along without
    perturbing the in-plane channels). Returns one updated array per
    input component (cell-centered).
    """
    out = []
    for v in psi_vars:
        rate = adv.advect_van_leer_using_limiters(v, velocity, dt)
        out.append(v.array.data + dt * rate.data)
    return tuple(out)


# ---------------------------------------------------------------------------
# Sec.5.F  Cell-center offset shared by all log-conformation models
# ---------------------------------------------------------------------------
# Every registered ``oldroyd_b_logconf*`` model places ``A`` on this
# offset. The actual evolution_fn for each registered model is built
# in Sec.6 via the :func:`make_logconf_evolution_fn` factory, which
# composes the kinematic building blocks (Psi<->A kernel, upper-convective
# stretch method, advection scheme) and a model-specific relaxation
# slot. See Sec.6 for the curated set of registered models and the
# factory signature.

CELL_CENTER_OFFSET_2D = (0.5, 0.5)


def _linear_conformation_stress_readout_fn(memory_fields: Tuple[GridVariable, ...],
                                           velocity: GridVariableVector,
                                           params: Any,
                                           ) -> Tuple[GridArray, GridArray, GridArray]:
    """Polymer stress from the conformation tensor -- linear (Hookean) form.

    ``tau_p = G_p . (A - I)``: the elastic stress of a
    Hookean (linear-spring, infinitely-extensible) dumbbell. This is
    the **shared** readout for every family member whose stress is
    linear in ``A`` -- Oldroyd-B, Giesekus, and linear PTT all reuse it
    (shared Hookean form). FENE-P is the exception (finite
    extensibility => ``tau = G_p(f.A - a.I)``) and supplies its own
    readout. (Renamed from ``_oldroyd_b_stress_readout_fn``: the form is
    Oldroyd-B's but it is no longer Oldroyd-B-specific.)

    Returns three cell-centered :class:`GridArray`\\ s (no BC -- that is
    the caller's responsibility, matching :data:`StressReadoutFn`'s
    contract).

    The state carries a 4th component
    ``A_zz``, but ``tau_zz`` does **not** enter the 2-D momentum coupling
    -- the contract is the in-plane triple ``(tau_xx, tau_xy, tau_yy)`` that
    :func:`equations_rheology.polymer_force_to_faces` consumes. ``A_zz``
    is ignored here; ``tau_zz`` for the ``N2`` diagnostic is exposed
    separately via :func:`linear_conformation_tau_zz_readout`.
    """
    A_xx_var, A_xy_var, A_yy_var, A_zz_var = memory_fields
    del A_zz_var  # out-of-plane stress does not feed 2-D momentum.
    grid = A_xx_var.grid
    Gp = _params_get(params, 'Gp')
    tau_xx = Gp * (A_xx_var.array.data - 1.0)
    tau_xy = Gp * A_xy_var.array.data
    tau_yy = Gp * (A_yy_var.array.data - 1.0)
    return (
        GridArray(tau_xx, CELL_CENTER_OFFSET_2D, grid),
        GridArray(tau_xy, CELL_CENTER_OFFSET_2D, grid),
        GridArray(tau_yy, CELL_CENTER_OFFSET_2D, grid),
    )


def linear_conformation_tau_zz_readout(memory_fields: Tuple[GridVariable, ...],
                                       params: Any,
                                       ) -> GridArray:
    """Out-of-plane polymer normal stress ``tau_zz = G_p.(A_zz - 1)``.

    Diagnostic only -- used to form the second normal-stress difference
    ``N2 = tau_yy - tau_zz = G_p.(A_yy - A_zz)``,
    which is uniform across the family. Deliberately **not** part of
    :data:`StressReadoutFn`: ``tau_zz`` must never enter the 2-D momentum
    coupling. Uses the linear (Hookean) form ``tau = G_p(A - I)`` shared
    by Oldroyd-B / Giesekus / PTT; FENE-P, whose stress readout differs,
    supplies its own ``N2`` helper.
    """
    A_zz_var = memory_fields[3]
    Gp = _params_get(params, 'Gp')
    return GridArray(Gp * (A_zz_var.array.data - 1.0),
                     CELL_CENTER_OFFSET_2D, A_zz_var.grid)


# --- FENE-P: Peterlin factor + finite-extensibility stress readout ---------

_FENE_P_FLOOR_FRAC = 1e-3
"""Smooth floor on the Peterlin denominator: ``D_floor = frac . L^2``
. The guard only acts in the last ~0.1 % before
the finite-extension wall ``tr A = L^2``."""


def _fene_p_peterlin_f(trA: jnp.ndarray, Lsq: Any) -> jnp.ndarray:
    """Peterlin factor ``f = L^2/(L^2 - tr A)`` with an AD-safe smooth floor.

    The denominator ``D = L^2 - tr A`` must stay positive (a FENE-P
    dumbbell cannot stretch past ``L^2``), but an explicit stretch
    substep can transiently overshoot ``tr A -> L^2``. Rather than a hard
    clamp (kink + dead gradient), we floor ``D`` smoothly with softplus --
    the same AD-safe idiom as the Carreau / power-law regularisation in
    ``jax_rheology/models.py`` (``sqrt(.+eps^2)`` / softplus, "no hard
    clips"):

        ``D_eff = D_floor + s.softplus((D - D_floor)/s)``,  ``s = D_floor``,
        ``D_floor = _FENE_P_FLOOR_FRAC . L^2``.

    For ``D >> D_floor`` (``tr A`` well below the wall) ``softplus(z) ~= z``
    => ``D_eff ~= D`` and ``f`` is the *exact* Peterlin factor (no physics
    distortion). As ``tr A -> L^2`` (or overshoots past it) ``D_eff ->
    D_floor`` smoothly, so ``f -> L^2/D_floor`` -- large, finite, maximal
    restoring stiffness at the wall -- with a non-zero (sigmoid) gradient
    everywhere, so AD/inference never hits a dead region.
    """
    D = Lsq - trA
    D_floor = _FENE_P_FLOOR_FRAC * Lsq
    # s = D_floor; jnp.logaddexp(0, x) is the numerically stable softplus.
    D_eff = D_floor + D_floor * jnp.logaddexp(0.0, (D - D_floor) / D_floor)
    return Lsq / D_eff


def _fene_p_stress_readout_fn(memory_fields: Tuple[GridVariable, ...],
                              velocity: GridVariableVector,
                              params: Any,
                              ) -> Tuple[GridArray, GridArray, GridArray]:
    """FENE-P polymer stress -- finite-extensibility (non-Hookean) form.

    ``tau = G_p.( f.A - a.I )``, ``f = L^2/(L^2 - tr A)``, ``a = L^2/(L^2 - 3)``
    (constitutive ref Sec.4). Note the ``a`` on ``I`` (not ``-I``): at rest
    ``tr A = 3 => f = a => tau = 0``. This is the family's first
    **non-Hookean** readout -- Oldroyd-B / Giesekus / PTT share
    :func:`_linear_conformation_stress_readout_fn`; FENE-P does not.

    Reads **all four** components (``f`` needs ``tr A = A_xx+A_yy+A_zz``,
    so ``f`` sees all four components) but returns only the in-plane triple
    ``(tau_xx, tau_xy, tau_yy)`` for the 2-D momentum coupling; ``tau_zz`` (for
    ``N2``) is exposed separately via :func:`fene_p_tau_zz_readout`.
    """
    del velocity
    A_xx_var, A_xy_var, A_yy_var, A_zz_var = memory_fields
    grid = A_xx_var.grid
    Gp = _params_get(params, 'Gp')
    Lsq = _params_get(params, 'Lsq')
    A_xx = A_xx_var.array.data
    A_xy = A_xy_var.array.data
    A_yy = A_yy_var.array.data
    A_zz = A_zz_var.array.data
    f = _fene_p_peterlin_f(A_xx + A_yy + A_zz, Lsq)
    a = Lsq / (Lsq - 3.0)
    return (
        GridArray(Gp * (f * A_xx - a), CELL_CENTER_OFFSET_2D, grid),
        GridArray(Gp * (f * A_xy), CELL_CENTER_OFFSET_2D, grid),
        GridArray(Gp * (f * A_yy - a), CELL_CENTER_OFFSET_2D, grid),
    )


def fene_p_tau_zz_readout(memory_fields: Tuple[GridVariable, ...],
                          params: Any,
                          ) -> GridArray:
    """Out-of-plane FENE-P normal stress ``tau_zz = G_p.(f.A_zz - a)``.

    Diagnostic only (the FENE-P analog of
    :func:`linear_conformation_tau_zz_readout`), for the second
    normal-stress difference ``N2 = tau_yy - tau_zz``. Never enters the 2-D
    momentum coupling.
    """
    A_xx_var, A_xy_var, A_yy_var, A_zz_var = memory_fields
    Gp = _params_get(params, 'Gp')
    Lsq = _params_get(params, 'Lsq')
    A_xx = A_xx_var.array.data
    A_yy = A_yy_var.array.data
    A_zz = A_zz_var.array.data
    f = _fene_p_peterlin_f(A_xx + A_yy + A_zz, Lsq)
    a = Lsq / (Lsq - 3.0)
    return GridArray(Gp * (f * A_zz - a), CELL_CENTER_OFFSET_2D,
                     A_zz_var.grid)


def _params_get(params: Any, name: str) -> Any:
    """Pull ``name`` from ``params`` whether it is a dict or a dataclass.

    The evolution / readout functions take a single ``params`` object
    so the API is invariant across all constitutive models, but the
    same model can be called with either a simple dict or with a richer
    parameter dataclass from
    the surrounding solver. Both forms are accepted here.
    """
    if isinstance(params, dict):
        if name in params:
            return params[name]
        raise KeyError(
            f"params dict is missing required key {name!r}; got keys "
            f"{sorted(params)}.")
    if hasattr(params, name):
        return getattr(params, name)
    raise AttributeError(
        f"params object {params!r} has no attribute {name!r}.")


# ---------------------------------------------------------------------------
# Sec.5.G  Model record + registration
# ---------------------------------------------------------------------------

# The full *planar* conformation tensor is carried as four components
# ``(A_xx, A_xy, A_yy, A_zz)`` for every log-conformation model. The
# out-of-plane shear components ``A_xz, A_yz`` stay identically zero in
# planar flow and are not carried. ``A_zz`` is a scalar log-conformation
# channel (``Psi_zz = log A_zz``); the ``'spd'`` manifold rest state gives
# it ``A_zz = 1`` for free (``_rest_state_for_manifold`` maps any label
# whose first==last char to 1.0). For Oldroyd-B / Giesekus / PTT
# ``A_zz == 1`` (forcing-free fixed point, no z-stretch), so the channel
# is inert and the in-plane dynamics match the 3-component state
# byte-for-byte; FENE-P is the model that makes ``A_zz`` non-trivial.
_LOGCONF_STATE_SPEC: cr.StateSpec = (
    cr.FieldSpec(
        name='A',
        components=('xx', 'xy', 'yy', 'zz'),
        manifold='spd',
        offset=CELL_CENTER_OFFSET_2D,
    ),
)


# The registered ``*_logconf*`` models that share this
# ``_LOGCONF_STATE_SPEC`` and ``_linear_conformation_stress_readout_fn`` live
# in Sec.6 alongside the kinematic-pipeline factory.


# ===========================================================================
# Becker-Knechtges eigenvalue-free kernel (2D)
# ===========================================================================
#
# Replaces the four ``# BECKER-SWAP``-tagged helpers above with an
# eigenvalue-free formulation. The eig path stays as a regression
# baseline; both kernels are registered as separate models.
#
# ---------------------------------------------------------------------------
# PRIMARY CITATION (must cite in any publication using this kernel):
#
#   Becker, F., Rauthmann, K., Pauli, L., Knechtges, P.
#   "An Eigenvalue-Free Implementation of the Log-Conformation
#    Formulation."
#   arXiv:2308.09394 [math.NA], August 2023.
#   German Aerospace Center (DLR), Institute for Software
#    Technology, and MAGMA Giessereitechnologie GmbH.
#   https://arxiv.org/abs/2308.09394
#
# BibTeX:
#   @article{becker2023eigenvaluefree,
#     title  = {An Eigenvalue-Free Implementation of the
#               Log-Conformation Formulation},
#     author = {Becker, Florian and Rauthmann, Katharina
#               and Pauli, Lutz and Knechtges, Philipp},
#     journal= {arXiv preprint arXiv:2308.09394},
#     year   = {2023},
#     eprint = {2308.09394},
#     archivePrefix = {arXiv},
#     primaryClass  = {math.NA},
#     url    = {https://arxiv.org/abs/2308.09394},
#   }
#
# Supporting references used in this implementation:
#
#   The 2D eigenvalue-free formulation that the 2D specialisation
#   of [Becker 2023] recovers exactly.
#
#   [FK04] Fattal, R., Kupferman, R. "Constitutive laws for the
#          matrix-logarithm of the conformation tensor." J. Non-
#          Newtonian Fluid Mech. 123, 281-285 (2004).
#          DOI: 10.1016/j.jnnfm.2004.08.008
#          -- the log-conformation formulation; provides the B/Omega
#          split that BK eliminates.
#
#   [H05] Hulsen, M. A., Fattal, R., Kupferman, R. "Flow of
#         viscoelastic fluids past a cylinder at high Weissenberg
#         number: stabilized simulations using matrix logarithms."
#         J. Non-Newtonian Fluid Mech. 127, 27-39 (2005).
#         DOI: 10.1016/j.jnnfm.2005.01.002
#         -- the velocity-gradient convention this codebase uses
#         internally (``L = (gradu)^T``); see sign-convention note
#         in ``upper_convective_increment_bk`` docstring Sec.5.
#
# Sign-convention note: [Becker 2023] Sec.2 uses
# ``[gradu]_ij = d_i u_j``, while this codebase follows Hulsen 2005
# and the existing eig-path code (``M = R^T (gradu)^T R``), which is
# equivalent to taking the *transpose* as the working velocity-
# gradient tensor. Net effect: the antisymmetric part ``omega`` in
# this implementation has opposite sign from [Becker 2023] eq. 3,
# i.e. our ``omega_code = -omega_BK``. The kinematic increment
# ``[omega_code, Psi] + 2 f(ad Psi) eps(u)`` is then mathematically equal
# to ``[omega_BK, Psi_paper_with_swapped_omega] + 2 f(...)``; the total
# kinematic update is identical because [Becker 2023] eq. 3
# itself is invariant under simultaneous transposition of ``gradu``.
# The forward regression against the eig path (machine precision
# in float64) is the proof of that convention.
#
# Key structural difference from the eig path:
#   * The Fattal-Kupferman ``B + Omega`` split, and with it the
#     ``lam_gap``-divided ``omega = (lam_y.M_xy + lam_x.M_yx)/(lam_y-lam_x)``
#     term, is *gone*. The whole kinematic non-advection part is
#     ``[omega, Psi] + 2 f(ad Psi) eps(u)`` where ``f(ad Psi) eps(u)`` collapses
#     in 2D to scalar arithmetic plus one call to a smooth scalar
#     function ``h(x)``.
#   * ``Psi_from_A_bk`` and ``A_from_Psi_bk`` use 2D Cayley-Hamilton
#     with ``sinh(s)/s`` / ``atanh(r)/r`` Taylor extensions at the
#     degenerate manifold ``s = 0`` (analytic, not subgradient-
#     selected).
#   * No ``# BECKER-SWAP`` tags below -- the BK helpers are the
#     publication-grade target the tags pointed to.
#
# The eig path stays in this file as a regression baseline; both
# kernels are registered as separate ``ConstitutiveModel``
# records (``oldroyd_b_logconf`` for eig, ``oldroyd_b_logconf_bk``
# for BK).

# ---------------------------------------------------------------------------
# Sec.3.5.A  AD-safe scalar helpers for Cayley-Hamilton and the BK kernel
# ---------------------------------------------------------------------------
#
# All three helpers below are 2D specialisations of the smooth
# scalar functions that appear in the BK formulation. Each has an
# analytic Taylor extension at the degenerate manifold (``s=0`` or
# ``x=0``), so the AD path is uniformly finite -- no ``where``
# selects a subgradient. We still use ``jnp.where`` to switch
# between the direct closed form (efficient for large argument)
# and the Taylor expansion (stable for small argument), with the
# double-where pattern: the *input* to the direct formula is
# replaced with a safe non-singular value on the Taylor branch
# so the masked-out gradient is finite as well.

# Threshold for switching between direct evaluation and Taylor
# expansion. Picked so that the truncated Taylor series (kept to
# order ``x^6``) is accurate to ``~1e-16`` in float64.
_BK_TAYLOR_THRESHOLD: float = 1.0e-3


def _sinhc(s_sq: jnp.ndarray) -> jnp.ndarray:
    """Smooth evaluation of ``sinh(sqrts_sq) / sqrts_sq``.

    Argument is the *square* of ``s`` so callers never have to
    take ``sqrt`` themselves (and so the Taylor variable is exactly
    ``s_sq``, no derivative-leaking ``sqrt(0)`` in the trace).

    Mathematically ``sinhc(0) = 1`` and ``sinhc`` is entire (no
    singularities at any finite ``s``). Used inside
    :func:`A_from_Psi_bk` as ``beta_exp = exp(T/2).sinhc(s^2)``.
    """
    use_taylor = s_sq < _BK_TAYLOR_THRESHOLD
    # Taylor: sinh(s)/s = 1 + s^2/6 + s4/120 + s6/5040 + ...
    sinhc_taylor = (1.0
                     + s_sq / 6.0
                     + s_sq * s_sq / 120.0
                     + s_sq * s_sq * s_sq / 5040.0)
    # Direct, input-safe: replace the s in the sqrt by 1 on the
    # Taylor branch so the gradient through ``sqrt`` is finite.
    s_sq_safe = jnp.where(use_taylor, 1.0, s_sq)
    s_safe = jnp.sqrt(s_sq_safe)
    sinhc_direct = jnp.sinh(s_safe) / s_safe
    return jnp.where(use_taylor, sinhc_taylor, sinhc_direct)


def _cosh_from_sq(s_sq: jnp.ndarray) -> jnp.ndarray:
    """Smooth evaluation of ``cosh(sqrts_sq)``.

    Like :func:`_sinhc`, takes ``s_sq`` so the caller never has to
    materialise ``s = sqrts_sq``. Taylor: ``cosh(s) = 1 + s^2/2 +
    s4/24 + s6/720 + ...``. No degenerate singularity to worry about
    (``cosh`` is entire).
    """
    use_taylor = s_sq < _BK_TAYLOR_THRESHOLD
    cosh_taylor = (1.0
                    + s_sq / 2.0
                    + s_sq * s_sq / 24.0
                    + s_sq * s_sq * s_sq / 720.0)
    s_sq_safe = jnp.where(use_taylor, 1.0, s_sq)
    s_safe = jnp.sqrt(s_sq_safe)
    cosh_direct = jnp.cosh(s_safe)
    return jnp.where(use_taylor, cosh_taylor, cosh_direct)


def _atanh_over_r(r_sq: jnp.ndarray) -> jnp.ndarray:
    """Smooth evaluation of ``arctanh(sqrtr_sq) / sqrtr_sq``.

    Used in :func:`Psi_from_A_bk` to compute ``beta_log = arctanh(s/
    (T/2)) / s`` via ``beta_log = (1/(T/2)) . _atanh_over_r(r^2)``
    where ``r = s/(T/2)``.

    Taylor: ``arctanh(r)/r = 1 + r^2/3 + r4/5 + r6/7 + ...`` (radius
    of convergence ``|r| < 1``). For SPD ``A`` we always have
    ``r  in  [0, 1)`` so the direct formula is well-defined.
    """
    use_taylor = r_sq < _BK_TAYLOR_THRESHOLD
    atanh_over_r_taylor = (1.0
                            + r_sq / 3.0
                            + r_sq * r_sq / 5.0
                            + r_sq * r_sq * r_sq / 7.0)
    # Input-safe: replace r^2 with a finite value on the Taylor
    # branch so the gradient through ``arctanh`` and the division
    # by ``sqrt`` is finite there. The replacement value 0.25 keeps
    # the Direct branch in a well-conditioned regime.
    r_sq_safe = jnp.where(use_taylor, 0.25, r_sq)
    r_safe = jnp.sqrt(r_sq_safe)
    atanh_over_r_direct = jnp.arctanh(r_safe) / r_safe
    return jnp.where(use_taylor, atanh_over_r_taylor,
                      atanh_over_r_direct)


def _bk_h(x: jnp.ndarray) -> jnp.ndarray:
    """Becker-Knechtges scalar ``h(x) = (1/x).(sqrtx/tanh(sqrtx) - 1)``.

    Defined for ``x >= 0`` (which is the regime our 2D ``X11/4 =
    s^2_Psi`` always lives in). Taylor expansion (BK eq. 24):

      h(x) = 1/3 - x/45 + 2 x^2/945 - x^3/4725 + ...

    The radius of convergence is ``pi^2 ~= 9.87`` (pole at ``x =
    -pi^2``). For our SPD-bounded ``Psi`` we never approach this
    radius. At ``x = 0`` the function is smooth with ``h(0) =
    1/3``.

    The function ``sqrtx/tanh(sqrtx) = sqrtx . coth(sqrtx)`` is smooth in
    ``x`` (the ``sqrtx`` cancels with ``tanh``'s odd part), so this
    is just one more analytic-in-the-square scalar helper.
    """
    use_taylor = x < _BK_TAYLOR_THRESHOLD
    # Taylor at x=0: 1/3 - x/45 + 2x^2/945 - x^3/4725
    h_taylor = (1.0 / 3.0
                 - x / 45.0
                 + 2.0 * x * x / 945.0
                 - x * x * x / 4725.0)
    # Direct: input-safe replacement so sqrt has a finite gradient.
    x_safe = jnp.where(use_taylor, 1.0, x)
    sqrt_x = jnp.sqrt(x_safe)
    # sqrtx/tanh(sqrtx) - 1 = sqrtx . coth(sqrtx) - 1
    # Use ``/jnp.tanh`` rather than ``coth`` (not a JAX primitive).
    coth_arg = sqrt_x / jnp.tanh(sqrt_x)
    h_direct = (coth_arg - 1.0) / x_safe
    return jnp.where(use_taylor, h_taylor, h_direct)


# ---------------------------------------------------------------------------
# Sec.3.5.B  Psi <-> A change of variables via 2D Cayley-Hamilton
# ---------------------------------------------------------------------------

def Psi_from_A_bk(Axx: jnp.ndarray,
                   Axy: jnp.ndarray,
                   Ayy: jnp.ndarray
                   ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Eigenvalue-free 2D matrix logarithm.

    Becker, Rauthmann, Pauli, Knechtges, *An Eigenvalue-Free
    Implementation of the Log-Conformation Formulation*,
    arXiv:2308.09394 (2023). 2D Cayley-Hamilton specialisation;
    cite the paper in any publication using this kernel.

    For a 2x2 symmetric SPD matrix ``A`` with trace ``T``,
    determinant ``D = T^2/4 - s^2``, Cayley-Hamilton gives

      log A = (log D)/2 . I + beta . (A - (T/2).I)

    with ``beta = arctanh(s/(T/2)) / s = (1/(T/2)) . _atanh_over_r(r^2)``,
    ``r = s/(T/2)``. All three scalars are smooth at the degenerate
    manifold ``s = 0`` (where ``A prop to I`` and ``beta -> 2/T``).

    Replaces :func:`Psi_from_A` for AD-quality applications; the
    forward value agrees with the eig version to machine precision
    on SPD inputs.
    """
    T = Axx + Ayy
    T_half = 0.5 * T
    # s^2 = (T/2)^2 - det(A) = ((Axx - Ayy)/2)^2 + Axy^2 >= 0
    diff_half = 0.5 * (Axx - Ayy)
    s_sq = diff_half * diff_half + Axy * Axy
    # r^2 = s^2/(T/2)^2; for SPD A always in [0, 1).
    T_half_sq = T_half * T_half
    r_sq = s_sq / T_half_sq
    beta = _atanh_over_r(r_sq) / T_half
    # log det A = log((T/2)^2 - s^2) -- always finite for SPD A.
    det_A = T_half_sq - s_sq
    half_log_det = 0.5 * jnp.log(det_A)
    # log A = half_log_det . I + beta . (A - (T/2).I)
    Psi_xx = half_log_det + beta * (Axx - T_half)
    Psi_xy = beta * Axy
    Psi_yy = half_log_det + beta * (Ayy - T_half)
    return Psi_xx, Psi_xy, Psi_yy


def A_from_Psi_bk(Psi_xx: jnp.ndarray,
                   Psi_xy: jnp.ndarray,
                   Psi_yy: jnp.ndarray
                   ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Eigenvalue-free 2D matrix exponential.

    Becker, Rauthmann, Pauli, Knechtges, *An Eigenvalue-Free
    Implementation of the Log-Conformation Formulation*,
    arXiv:2308.09394 (2023). 2D Cayley-Hamilton specialisation;
    cite the paper in any publication using this kernel.

    For a 2x2 symmetric matrix ``Psi`` with trace ``T``, ``s^2 =
    ((Psi_xx - Psi_yy)/2)^2 + Psi_xy^2 >= 0``, Cayley-Hamilton gives

      exp Psi = exp(T/2) . [ cosh(s) . I  +  sinhc(s^2) . (Psi - (T/2).I) ]

    where ``sinhc(s^2) = sinh(s)/s`` is analytic at ``s = 0`` via
    Taylor. Replaces :func:`A_from_Psi`.
    """
    T = Psi_xx + Psi_yy
    T_half = 0.5 * T
    diff_half = 0.5 * (Psi_xx - Psi_yy)
    s_sq = diff_half * diff_half + Psi_xy * Psi_xy
    exp_T_half = jnp.exp(T_half)
    cosh_s = _cosh_from_sq(s_sq)
    sinhc_s = _sinhc(s_sq)
    # exp Psi = exp(T/2) . cosh(s) . I + exp(T/2) . sinhc(s^2) . (Psi - (T/2).I)
    alpha = exp_T_half * cosh_s
    beta = exp_T_half * sinhc_s
    A_xx = alpha + beta * (Psi_xx - T_half)
    A_xy = beta * Psi_xy
    A_yy = alpha + beta * (Psi_yy - T_half)
    return A_xx, A_xy, A_yy


# ---------------------------------------------------------------------------
# Sec.3.5.C  Upper-convective increment via the 2D BK algorithm
# ---------------------------------------------------------------------------

def upper_convective_increment_bk(velocity: GridVariableVector,
                                    Psi_xx: jnp.ndarray,
                                    Psi_xy: jnp.ndarray,
                                    Psi_yy: jnp.ndarray,
                                    dt: float,
                                    grad_u_fn: Callable = _cell_centered_grad_u,
                                    ) -> Tuple[jnp.ndarray, jnp.ndarray,
                                                jnp.ndarray]:
    """Becker-Knechtges 2D kinematic increment (forward Euler).

    Becker, Rauthmann, Pauli, Knechtges, *An Eigenvalue-Free
    Implementation of the Log-Conformation Formulation*,
    arXiv:2308.09394 (2023). This routine is the 2D collapse of
    their Algorithm 1 (eqs. 25, 30, 33, 39) -- see paper Sec.3 just
    before Algorithm 2 for the dimensional-reduction note. Cite the
    paper in any publication using this kernel.

    Computes ``DeltaPsi = dt . ([omega, Psi] + 2 f(ad Psi) eps(u))`` where ``omega =
    (gradu - gradu^T)/2`` is the vorticity tensor and ``eps(u) = (gradu +
    gradu^T)/2`` is the strain rate tensor (paper eq. 46). The 2D
    collapse of BK eq. 25 + eqs. 30, 33, 39 yields

      v1     = eps_xy.(Psi_xx - Psi_yy)  -  Psi_xy.(eps_xx - eps_yy)
      X11/4  = ((Psi_xx - Psi_yy)^2 + 4.Psi_xy^2) / 4    ( = s^2_Psi )
      h_val  = h(X11/4)
      f_xx   = eps_xx + (1/4).(-2.Psi_xy).h_val.v1
      f_xy   = eps_xy + (1/4).( Psi_xx - Psi_yy).h_val.v1
      f_yy   = eps_yy + (1/4).( 2.Psi_xy).h_val.v1

    Combined with the rotational commutator ``[omega, Psi]_xx =
    2.w.Psi_xy``, ``[omega, Psi]_xy = w.(Psi_yy - Psi_xx)``, ``[omega, Psi]_yy =
    -2.w.Psi_xy`` where ``w = omega_xy = (d_y u_x - d_x u_y)/2``. This
    sign on ``w`` follows the same velocity-gradient convention
    as :func:`upper_convective_increment` -- namely
    ``L = (gradu)^T`` (Hulsen 2005 Sec.2), so the antisymmetric part
    is ``omega = (L - L^T)/2 = ((gradu)^T - gradu)/2``, giving
    ``omega_xy = (d_y u_x - d_x u_y)/2``. Verified via random-input
    regression against the eig kernel (machine precision in float64):

      DeltaPsi_xx = dt . ( 2.w.Psi_xy + 2.eps_xx - Psi_xy.h_val.v1 )
      DeltaPsi_xy = dt . ( w.(Psi_yy - Psi_xx)
                     + 2.eps_xy + 0.5.(Psi_xx - Psi_yy).h_val.v1 )
      DeltaPsi_yy = dt . (-2.w.Psi_xy + 2.eps_yy + Psi_xy.h_val.v1 )

    No eigendecomposition, no ``# BECKER-SWAP``-tagged
    ``lam_gap``-divided ``omega`` term, no special-case branch on
    degenerate eigenvalues. ``h(x)`` is the BK scalar (Sec.3.5.A);
    smooth at the degenerate manifold ``Psi = 0``.

    Replaces :func:`upper_convective_increment`. Argument is
    ``Psi`` (not ``A``), so the evolution_fn calls
    :func:`Psi_from_A_bk` once at the start of its time step
    instead of indirectly through this function -- half the
    Psi-from-A overhead.
    """
    # ``grad_u_fn`` is injectable; default is the central stencil.
    # Central-only call (kept commented):
    # dux_dx, dux_dy, duy_dx, duy_dy = _cell_centered_grad_u(velocity)
    dux_dx, dux_dy, duy_dx, duy_dy = grad_u_fn(velocity)

    # eps(u) and the scalar w = omega_xy = ((gradu)^T - gradu)_xy / 2
    # = (d_y u_x - d_x u_y)/2.  Sign convention matches the eig
    # path (which builds M from L = (gradu)^T); see docstring Sec.5.
    eps_xx = dux_dx
    eps_yy = duy_dy
    eps_xy = 0.5 * (dux_dy + duy_dx)
    w = 0.5 * (dux_dy - duy_dx)

    # BK 2D scalars.
    Psi_diff = Psi_xx - Psi_yy
    eps_diff = eps_xx - eps_yy
    v1 = eps_xy * Psi_diff - Psi_xy * eps_diff
    x_arg = 0.25 * (Psi_diff * Psi_diff + 4.0 * Psi_xy * Psi_xy)  # = X11/4 = s^2_Psi
    h_val = _bk_h(x_arg)

    # (1/4) Y1 . h_val . v1 is added to eps(u) to get f(ad Psi).eps(u),
    # then doubled to get the term in eq. 46.
    coef = h_val * v1
    f_xx = eps_xx + 0.25 * (-2.0 * Psi_xy) * coef
    f_xy = eps_xy + 0.25 * Psi_diff * coef
    f_yy = eps_yy + 0.25 * (2.0 * Psi_xy) * coef

    # Rotational commutator [omega, Psi] (symmetric, since [antisym,
    # sym] = sym).
    comm_xx = 2.0 * w * Psi_xy
    comm_xy = w * (Psi_yy - Psi_xx)
    comm_yy = -2.0 * w * Psi_xy

    # DeltaPsi = dt . ([omega, Psi] + 2.f(ad Psi).eps(u))
    dPsi_xx = dt * (comm_xx + 2.0 * f_xx)
    dPsi_xy = dt * (comm_xy + 2.0 * f_xy)
    dPsi_yy = dt * (comm_yy + 2.0 * f_yy)
    return dPsi_xx, dPsi_xy, dPsi_yy


# ---------------------------------------------------------------------------
# Sec.3.5.D  BK model registration -> lives in Sec.6 (factory-built)
# ---------------------------------------------------------------------------
# ``oldroyd_b_logconf_bk`` is constructed by
# :func:`make_logconf_evolution_fn` with
# ``(psi_kernel='bk', uc_method='fe', advect_method='fe')`` -- see Sec.6.


# ===========================================================================
# Conformation-kernel accuracy upgrades (model-agnostic)
# ===========================================================================
#
# Two orthogonal, model-agnostic upgrades layered on the BK kernel:
#
#   1. **SSP-RK2 van-Leer advection** (Heun's method) for the
#      Psi-space transport stage. Second-order in time; roughly
#      doubles the stable CFL of the limited scheme (forward
#      Euler ~= 0.5, SSP-RK2 ~= 1.0). Sibling of
#      :func:`_advect_psi_components_euler`, which stays as
#      a regression baseline.
#
#   2. **Closed-form upper-convective stretch** in A-space:
#      ``A(t+dt) = exp(dt.L) . A(t) . exp(dt.L^T)``, replacing
#      the forward-Euler Psi-space increment
#      (:func:`upper_convective_increment_bk`). Fattal-Kupferman
#      2004 Sec.3.3. Unconditionally stable in the kinematic stretch
#      direction (no CFL ceiling from ||dt.L||), exact for the
#      frozen-L sub-PDE ``d_t A = LA + AL^T``, and a natural fit
#      with the BK Cayley-Hamilton A-space arithmetic. ``L`` here
#      is ``(gradu)^T``, matching the Hulsen-2005 velocity-gradient
#      convention the rest of this kernel uses (see the
#      sign-convention note above).
#
# Both upgrades preserve the model-agnostic interface: only the
# *kinematic* parts of the evolution change. The model-specific
# slot (the analytic relaxation stage for Oldroyd-B) is
# unchanged. Other SPD-conformation models (Giesekus, FENE-P,
# PTT, viscoelastic TBNN closures) inherit both upgrades without
# modification.
#
# References:
#   [FK04 Sec.3.3]  Fattal, R., Kupferman, R. (2004) -- closed-form
#                analytic integration of the frozen-L
#                upper-convective stretch.
#                DOI: 10.1016/j.jnnfm.2004.08.008
#   SSP-RK2 is Heun's method (two-stage, second-order).
#
# Composed with the BK kernel this is the publication-grade
# kinematic backbone: AD-clean (BK), second-order in time (RK2),
# unconditionally stable in stretch (FK analytic), and
# model-agnostic (the relaxation slot is the only thing a future
# model would replace).
#
# The combined kernel is registered as
# :data:`OLDROYD_B_LOGCONF_BK_V2`. The FE-time BK model
# (:data:`OLDROYD_B_LOGCONF_BK`) and the eig model
# (:data:`OLDROYD_B_LOGCONF`) stay as regression baselines.


# ---------------------------------------------------------------------------
# Sec.3.6.A  Signed-argument scalar helpers
# ---------------------------------------------------------------------------
#
# Generalises :func:`_sinhc` / :func:`_cosh_from_sq` from Sec.3.5.A
# to allow the squared-eigenvalue argument ``s^2 = (T/2)^2 - det M``
# to be *negative* (complex-conjugate eigenvalues of ``M``, which
# happens for rotation-dominated 2x2 matrices). Mathematically
# the relevant scalar function is entire -- the same Taylor
# series ``1 + x/6 + x^2/120 + ...`` works for any real x -- so the
# extension is straightforward.

def _sinhc_signed(x: jnp.ndarray) -> jnp.ndarray:
    """Smooth scalar ``sinh(sqrtx)/sqrtx`` extended to ``x < 0``.

    On ``x >= 0`` returns ``sinh(sqrtx)/sqrtx``. On ``x < 0`` returns
    ``sin(sqrt|x|)/sqrt|x|`` (the analytic continuation, since
    ``sinh(it)/it = sin(t)/t``). Entire as a function of ``x``;
    Taylor series ``1 + x/6 + x^2/120 + x^3/5040 + ...`` converges
    for all ``x``.
    """
    use_taylor = jnp.abs(x) < _BK_TAYLOR_THRESHOLD
    taylor = (1.0
              + x / 6.0
              + x * x / 120.0
              + x * x * x / 5040.0)
    abs_x = jnp.abs(x)
    abs_x_safe = jnp.where(use_taylor, 1.0, abs_x)
    sqrt_abs_x = jnp.sqrt(abs_x_safe)
    direct_pos = jnp.sinh(sqrt_abs_x) / sqrt_abs_x          # x > 0 branch
    direct_neg = jnp.sin(sqrt_abs_x) / sqrt_abs_x           # x < 0 branch
    direct = jnp.where(x >= 0.0, direct_pos, direct_neg)
    return jnp.where(use_taylor, taylor, direct)


def _cosh_signed(x: jnp.ndarray) -> jnp.ndarray:
    """Smooth scalar ``cosh(sqrtx)`` extended to ``x < 0``.

    On ``x >= 0`` returns ``cosh(sqrtx)``. On ``x < 0`` returns
    ``cos(sqrt|x|)``. Entire; Taylor series
    ``1 + x/2 + x^2/24 + x^3/720 + ...``.
    """
    use_taylor = jnp.abs(x) < _BK_TAYLOR_THRESHOLD
    taylor = (1.0
              + x / 2.0
              + x * x / 24.0
              + x * x * x / 720.0)
    abs_x = jnp.abs(x)
    abs_x_safe = jnp.where(use_taylor, 1.0, abs_x)
    sqrt_abs_x = jnp.sqrt(abs_x_safe)
    direct_pos = jnp.cosh(sqrt_abs_x)
    direct_neg = jnp.cos(sqrt_abs_x)
    direct = jnp.where(x >= 0.0, direct_pos, direct_neg)
    return jnp.where(use_taylor, taylor, direct)


# ---------------------------------------------------------------------------
# Sec.3.6.B  General 2x2 matrix exponential
# ---------------------------------------------------------------------------

def _exp_2x2_general(M_xx: jnp.ndarray,
                       M_xy: jnp.ndarray,
                       M_yx: jnp.ndarray,
                       M_yy: jnp.ndarray
                       ) -> Tuple[jnp.ndarray, jnp.ndarray,
                                   jnp.ndarray, jnp.ndarray]:
    """Closed-form ``exp`` of a general (non-symmetric) 2x2 matrix.

    Cayley-Hamilton: ``exp(M) = alpha.I + beta.M`` with

      T   = M_xx + M_yy
      det = M_xx.M_yy - M_xy.M_yx
      s^2  = (T/2)^2 - det                  (signed -- real or pure-imag eigenvalues)
      beta   = exp(T/2) . sinhc_signed(s^2)
      alpha   = exp(T/2) . cosh_signed(s^2)  -  (T/2).beta

    For incompressible flow ``T = tr(L) = 0`` and the formula
    simplifies to ``exp(M) = cosh_signed(s^2).I + sinhc_signed(s^2).M``;
    we keep the general form so the helper is usable for any
    2x2 matrix (e.g. compressible flow, debugging, future tensor
    fields whose trace is non-trivial).
    """
    T = M_xx + M_yy
    det_M = M_xx * M_yy - M_xy * M_yx
    s_sq = 0.25 * T * T - det_M
    sinhc_val = _sinhc_signed(s_sq)
    cosh_val = _cosh_signed(s_sq)
    exp_T_half = jnp.exp(0.5 * T)
    beta = exp_T_half * sinhc_val
    alpha = exp_T_half * cosh_val - 0.5 * T * beta
    E_xx = alpha + beta * M_xx
    E_xy = beta * M_xy
    E_yx = beta * M_yx
    E_yy = alpha + beta * M_yy
    return E_xx, E_xy, E_yx, E_yy


# ---------------------------------------------------------------------------
# Sec.3.6.C  Closed-form upper-convective stretch (A-space, FK 2004 Sec.3.3)
# ---------------------------------------------------------------------------

def upper_convective_step_analytic(velocity: GridVariableVector,
                                     Axx: jnp.ndarray,
                                     Axy: jnp.ndarray,
                                     Ayy: jnp.ndarray,
                                     dt: float,
                                     grad_u_fn: Callable = _cell_centered_grad_u,
                                     ) -> Tuple[jnp.ndarray, jnp.ndarray,
                                                 jnp.ndarray]:
    """Closed-form integration of ``d_t A = LA + AL^T`` (frozen ``L``).

    Fattal-Kupferman 2004 Sec.3.3. For a velocity-gradient tensor
    ``L = (gradu)^T`` (Hulsen 2005 convention, matching the rest of
    this kernel) held constant over the step ``dt``, the
    upper-convective stretch is integrated exactly as

      A(t+dt)  =  E . A(t) . E^T          with  E = exp(dt . L).

    Unlike :func:`upper_convective_increment_bk` (forward-Euler
    on Psi), this is the *exact* solution of the frozen-``L``
    sub-PDE -- unconditionally stable for any ``||dt.L||``, and
    preserves SPD of ``A`` for free (``E.A.E^T`` is SPD whenever
    ``A`` is and ``E`` is real, regardless of how large ``E``
    becomes).

    Returns the new A components directly (full step in A-space,
    not a Psi-increment). Replaces stages 1-3 of the BK
    evolution_fn in the ``_v2`` model.

    Cite Fattal-Kupferman 2004 (DOI 10.1016/j.jnnfm.2004.08.008)
    when using this helper.
    """
    # ``grad_u_fn`` is injectable; default is the central stencil.
    # Central-only call (kept commented):
    # dux_dx, dux_dy, duy_dx, duy_dy = _cell_centered_grad_u(velocity)
    dux_dx, dux_dy, duy_dx, duy_dy = grad_u_fn(velocity)
    # L = (gradu)^T with this codebase's convention ``[gradu]_ij = d_i u_j``
    # => L_ij = d_j u_i. Components:
    #   L_xx = d_x u_x = dux_dx
    #   L_xy = d_y u_x = dux_dy
    #   L_yx = d_x u_y = duy_dx
    #   L_yy = d_y u_y = duy_dy
    Lxx_dt = dt * dux_dx
    Lxy_dt = dt * dux_dy
    Lyx_dt = dt * duy_dx
    Lyy_dt = dt * duy_dy

    E_xx, E_xy, E_yx, E_yy = _exp_2x2_general(
        Lxx_dt, Lxy_dt, Lyx_dt, Lyy_dt)

    # Compute E.A. A is symmetric (A_yx = A_xy).
    EA_xx = E_xx * Axx + E_xy * Axy
    EA_xy = E_xx * Axy + E_xy * Ayy
    EA_yx = E_yx * Axx + E_yy * Axy
    EA_yy = E_yx * Axy + E_yy * Ayy

    # (E.A).E^T. Note (E^T)_kj = E_jk.
    Axx_new = EA_xx * E_xx + EA_xy * E_xy
    Axy_new = EA_xx * E_yx + EA_xy * E_yy
    Ayy_new = EA_yx * E_yx + EA_yy * E_yy
    # (We don't need Ayx_new -- A stays symmetric. Sanity-check
    # in the regression test: Ayx_new should equal Axy_new to
    # round-off.)
    return Axx_new, Axy_new, Ayy_new


# ---------------------------------------------------------------------------
# Sec.3.6.D  SSP-RK2 (Heun) van-Leer advection
# ---------------------------------------------------------------------------

def _advect_psi_components_ssp_rk2(psi_vars: Tuple[GridVariable, ...],
                                   velocity: GridVariableVector,
                                   dt: float
                                   ) -> Tuple[jnp.ndarray, ...]:
    """Advect an arbitrary tuple of Psi components by ``velocity`` with SSP-RK2.

    Heun's method (Shu-Osher SSP-RK2):

      R1 = R(Psin)
      Psi* = Psin + dt . R1
      R2 = R(Psi*)
      Psin+^1 = Psin + 12.dt.(R1 + R2)

    where ``R(Psi) = -div (Psi u)`` is the van-Leer-limited rate
    (same operator as :func:`_advect_psi_components_euler`,
    via :func:`adv.advect_van_leer_using_limiters`). Second
    order in time; CFL ceiling for the limited scheme is
    ~`1.0` (vs ~`0.5` for forward Euler), which is the
    wall-time win at higher resolution.

    As with the Euler variant, each component is transported by
    independent van-Leer calls, so the first three channels of a
    4-tuple are byte-identical to the legacy 3-component path while
    the ``A_zz`` channel rides along under the
    same second-order scheme (needed once ``A_zz`` becomes non-trivial
    for FENE-P). Returns one updated array per input component.
    """
    half_dt = 0.5 * dt

    # --- Stage 1: rate at Psin. ---
    rate1 = [adv.advect_van_leer_using_limiters(v, velocity, dt)
             for v in psi_vars]

    # Predictor Psi* = Psin + dt.R1 (same offset, same bc per component).
    star_vars = [
        GridVariable(
            GridArray(v.array.data + dt * r1.data, CELL_CENTER_OFFSET_2D, v.grid),
            v.bc)
        for v, r1 in zip(psi_vars, rate1)
    ]

    # --- Stage 2: rate at Psi*. ---
    rate2 = [adv.advect_van_leer_using_limiters(sv, velocity, dt)
             for sv in star_vars]

    # Corrector Psin+^1 = Psin + (dt/2).(R1 + R2).
    return tuple(
        v.array.data + half_dt * (r1.data + r2.data)
        for v, r1, r2 in zip(psi_vars, rate1, rate2)
    )


# ---------------------------------------------------------------------------
# Sec.3.6.E  v2 model registration -> lives in Sec.6 (factory-built)
# ---------------------------------------------------------------------------
# ``oldroyd_b_logconf_bk_v2`` is constructed
# by :func:`make_logconf_evolution_fn` with
# ``(psi_kernel='bk', uc_method='analytic', advect_method='rk2')`` --
# see Sec.6.


# ===========================================================================
# Sec.6 -- Composable evolution_fn factory + curated registered models
# ===========================================================================
#
# BK (eigenvalue-free Psi<->A), FK analytic UC stretch + SSP-RK2 advect,
# and the wall-aware gradu stencil are orthogonal kinematic axes on top
# of the original eig path. Registering every combination would
# balloon the registry. We do **not** do that. Instead:
#
#   1. :func:`make_logconf_evolution_fn` is a small factory that takes
#      named choices on each axis and composes the right three-stage
#      pipeline at construction time (no runtime dispatch -- selection
#      collapses at trace time, JIT-friendly).
#   2. We register only a **curated** set of presets -- the ones with
#      a defensible "use me when..." docstring. Today that's three:
#        * ``oldroyd_b_logconf``      -- eig-path baseline.
#        * ``oldroyd_b_logconf_bk``   -- BK kernel, AD-clean, FE in time.
#        * ``oldroyd_b_logconf_bk_v2``-- BK + analytic UC + SSP-RK2.
#   3. Any other combo (e.g. BK + analytic UC + FE advect, the
#      "always-on accuracy win, no CFL cost" preset) can be
#      constructed ad-hoc by calling the factory directly. Tests and
#      experiment notebooks use this path so they don't have to
#      promote experimental variants to the registry.
#
# The factory's ``relaxation_fn`` slot is the model-specific axis --
# Oldroyd-B is the default; FENE-P / Giesekus / PTT / a learned TBNN
# closure all plug in here without touching the kinematic backbone.


def _oldroyd_b_relaxation_from_params(A_xx: jnp.ndarray,
                                        A_xy: jnp.ndarray,
                                        A_yy: jnp.ndarray,
                                        A_zz: jnp.ndarray,
                                        velocity: GridVariableVector,
                                        dt: float,
                                        params: Any,
                                        ) -> Tuple[jnp.ndarray, jnp.ndarray,
                                                    jnp.ndarray, jnp.ndarray]:
    """Thin adapter so :func:`oldroyd_b_relaxation_analytic` fits the
    factory's uniform ``relaxation_fn`` signature.

    The slot is
    ``relaxation_fn(A_xx, A_xy, A_yy, A_zz, velocity, dt, params)
    -> (A_xx, A_xy, A_yy, A_zz)``. The algebraic family (Oldroyd-B,
    Giesekus, FENE-P, PTT) ignores ``velocity``; only the learned
    viscoelastic-TBNN closure reads it. Carrying it here keeps one
    uniform relaxation signature across the whole family.

    ``A_zz`` is the out-of-plane diagonal. For Oldroyd-B its
    relaxation is the same scalar exponential as the in-plane
    diagonals -- ``d_t A_zz = -(A_zz - 1)/lam`` -- and since the
    rest/forcing-free value is ``A_zz = 1`` in planar flow, this maps
    ``1 |-> 1`` exactly (the channel is inert).

    Future constitutive models replace this with their own callable
    of the same signature; the rest of the pipeline (kinematic stretch,
    advection, Psi<->A conversions) is inherited unchanged via the factory.
    """
    del velocity  # Oldroyd-B relaxation is a function of A only.
    lam = _params_get(params, 'lam')
    A_xx_new, A_xy_new, A_yy_new = oldroyd_b_relaxation_analytic(
        A_xx, A_xy, A_yy, dt, lam)
    # Scalar out-of-plane channel: same analytic exponential as the
    # in-plane diagonals (closed form of d_t A_zz = -(A_zz - 1)/lam).
    int_factor = jnp.exp(-dt / lam)
    A_zz_new = (1.0 - int_factor) + A_zz * int_factor
    return A_xx_new, A_xy_new, A_yy_new, A_zz_new


# ---------------------------------------------------------------------------
# Sec.6.0  Shared SPD-safe relaxation integrator (affine, exact, AD-clean)
# ---------------------------------------------------------------------------
# Oldroyd-B relaxation is exactly exponential, but the rest of the family
# (Giesekus, FENE-P, PTT, and the eventual learned TBNN closure) is **not**.
# The family relaxation step owns its own
# SPD-preserving integrator. We use one shared helper:
#
#   freeze the model's coefficient(s) at the explicit (pre-relaxation)
#   value and integrate the resulting *constant-coefficient affine* ODE
#         dA/dt = -(1/lam) . M . (A - A*)          (frozen M, A*)
#   exactly:
#         A_new = A* + exp(-(dt/lam) M) . (A - A*).
#
# This is the linearly-implicit step of Sec.3 done as a matrix exponential
# rather than a backward-Euler solve, which buys two things over the
# literal Sec.3 backward-Euler form:
#   * the alpha->0 / eps->0 / L^2->inf Oldroyd-B limit is the *exact* analytic
#     exponential (M = I, A* = I => A_new = I + e^{-dt/lam}(A - I)), so the
#     regression gate is machine-precision, not O(dt^2)-off;
#   * SPD is preserved by construction (see below).
#
# The **affine offset** ``A*`` (the frozen steady state, ``B* = A* - I``)
# is carried from the start so FENE-P -- whose relaxation has a non-zero
# offset, ``A* = (a/f).I`` -- routes through the *same* helper without a
# refactor. Giesekus and PTT have ``A* = I`` (zero offset); Giesekus
# supplies a full 2x2 matrix ``M = I + alpha(A - I)``, PTT/FENE-P a scalar
# ``M = f.I``.
#
# SPD-safety: in the eigenbasis of ``B = A - I`` (shared with ``M`` for
# Giesekus, trivial for the scalar-M models), each eigenvalue maps
# ``b |-> b* + e^{-(dt/lam).m}(b - b*)`` with ``m > 0`` and ``b* > -1``; a
# contraction of an SPD-realisable state toward an SPD ``A*`` stays SPD
# for the physical parameter ranges (alpha in [0,1], finite L^2, eps>=0).


def _affine_exponential_relaxation_step(A_xx: jnp.ndarray,
                                        A_xy: jnp.ndarray,
                                        A_yy: jnp.ndarray,
                                        A_zz: jnp.ndarray,
                                        M_xx: jnp.ndarray,
                                        M_xy: jnp.ndarray,
                                        M_yy: jnp.ndarray,
                                        M_zz: jnp.ndarray,
                                        Astar_xx: jnp.ndarray,
                                        Astar_xy: jnp.ndarray,
                                        Astar_yy: jnp.ndarray,
                                        Astar_zz: jnp.ndarray,
                                        dt: float,
                                        lam: float,
                                        ) -> Tuple[jnp.ndarray, jnp.ndarray,
                                                   jnp.ndarray, jnp.ndarray]:
    """Exact step of ``dA/dt = -(1/lam) M (A - A*)`` for frozen ``M``, ``A*``.

    ``A_new = A* + exp(-(dt/lam) M) . (A - A*)``. The in-plane block uses
    the closed-form 2x2 matrix exponential (:func:`_exp_2x2_general`);
    ``A_zz`` is the decoupled scalar channel. The in-plane ``M`` is a
    symmetric 2x2 ``[[M_xx, M_xy], [M_xy, M_yy]]``; for the whole
    family ``M`` is a function of ``A_pre`` (so ``exp(-(dt/lam)M)`` and
    ``A_pre - A*`` commute) and the product is symmetric -- we symmetrise
    the off-diagonal anyway as round-off insurance (AD-clean).

    Shared by Giesekus / FENE-P / linear PTT. The
    upper-convective stretch and advection are the inherited backbone;
    this helper owns only the relaxation sub-step.
    """
    c = -dt / lam
    E_xx, E_xy, E_yx, E_yy = _exp_2x2_general(
        c * M_xx, c * M_xy, c * M_xy, c * M_yy)

    D_xx = A_xx - Astar_xx
    D_xy = A_xy - Astar_xy
    D_yy = A_yy - Astar_yy

    ED_xx = E_xx * D_xx + E_xy * D_xy
    # E.D off-diagonals (equal up to round-off since E and D commute);
    # symmetrise to keep A_new exactly symmetric.
    ED_xy_a = E_xx * D_xy + E_xy * D_yy
    ED_xy_b = E_yx * D_xx + E_yy * D_xy
    ED_xy = 0.5 * (ED_xy_a + ED_xy_b)
    ED_yy = E_yx * D_xy + E_yy * D_yy

    A_xx_new = Astar_xx + ED_xx
    A_xy_new = Astar_xy + ED_xy
    A_yy_new = Astar_yy + ED_yy

    E_zz = jnp.exp(c * M_zz)
    A_zz_new = Astar_zz + E_zz * (A_zz - Astar_zz)
    return A_xx_new, A_xy_new, A_yy_new, A_zz_new


def _giesekus_relaxation_from_params(A_xx: jnp.ndarray,
                                     A_xy: jnp.ndarray,
                                     A_yy: jnp.ndarray,
                                     A_zz: jnp.ndarray,
                                     velocity: GridVariableVector,
                                     dt: float,
                                     params: Any,
                                     ) -> Tuple[jnp.ndarray, jnp.ndarray,
                                                jnp.ndarray, jnp.ndarray]:
    """Giesekus relaxation.

    A-space source ``R(A) = -(1/lam)[ (A - I) + alpha (A - I)(A - I) ]`` with
    mobility ``alpha = params['alpha']  in  [0, 1]``. Factor the quadratic as

        R(A) = -(1/lam) . M . (A - I),   M = I + alpha (A_pre - I)   (frozen),

    so the relaxation is the shared affine step with steady state
    ``A* = I`` (zero offset). At ``alpha = 0`` => ``M = I`` => exact Oldroyd-B
    analytic exponential (machine-precision regression gate). Stress
    readout is the shared Hookean ``tau = Gp(A - I)``
    (:func:`_linear_conformation_stress_readout_fn`).
    """
    del velocity  # algebraic model: function of A and tr A only.
    lam = _params_get(params, 'lam')
    alpha = _params_get(params, 'alpha')

    B_xx = A_xx - 1.0
    B_xy = A_xy
    B_yy = A_yy - 1.0
    B_zz = A_zz - 1.0

    # Frozen linear operator M = I + alpha.(A_pre - I).
    M_xx = 1.0 + alpha * B_xx
    M_xy = alpha * B_xy
    M_yy = 1.0 + alpha * B_yy
    M_zz = 1.0 + alpha * B_zz

    # Steady state A* = I (B* = 0).
    return _affine_exponential_relaxation_step(
        A_xx, A_xy, A_yy, A_zz,
        M_xx, M_xy, M_yy, M_zz,
        1.0, 0.0, 1.0, 1.0,
        dt, lam)


def _fene_p_relaxation_from_params(A_xx: jnp.ndarray,
                                   A_xy: jnp.ndarray,
                                   A_yy: jnp.ndarray,
                                   A_zz: jnp.ndarray,
                                   velocity: GridVariableVector,
                                   dt: float,
                                   params: Any,
                                   ) -> Tuple[jnp.ndarray, jnp.ndarray,
                                              jnp.ndarray, jnp.ndarray]:
    """FENE-P relaxation.

    A-space source ``R(A) = -(1/lam)[ f.A - a.I ]`` with Peterlin factor
    ``f = L^2/(L^2 - tr A)`` (AD-safe smooth floor, :func:`_fene_p_peterlin_f`),
    ``a = L^2/(L^2 - 3)`` constant, ``L^2 = params['Lsq']``,
    ``tr A = A_xx + A_yy + A_zz``. This is affine in ``A`` with a
    **non-zero** offset:

        R(A) = -(f/lam).(A - A*),   M = f.I (scalar),   A* = (a/f).I,

    so it routes through the shared SPD-safe integrator
    (:func:`_affine_exponential_relaxation_step`) with frozen ``f``. The
    update ``A_new = (a/f)(1-w).I + w.A_pre``, ``w = e^{-(dt/lam)f}``, is a
    convex combination of two SPD states => SPD-safe. At ``L^2 -> inf``,
    ``f, a -> 1`` => exact Oldroyd-B analytic exponential (asymptotic,
    ``O(1/L^2)`` -- the check is a limit, not a byte-for-byte match).

    Unlike Oldroyd-B / Giesekus / PTT, ``A_zz`` is genuinely active here:
    ``f`` depends on the full trace, and the steady out-of-plane value is
    ``A_zz = a/f < 1`` in shear (it contracts). Stress readout is the
    FENE-P-specific ``tau = Gp(f.A - a.I)``
    (:func:`_fene_p_stress_readout_fn`), **not** the shared Hookean form.
    """
    del velocity  # algebraic model: function of A and tr A only.
    lam = _params_get(params, 'lam')
    Lsq = _params_get(params, 'Lsq')

    f = _fene_p_peterlin_f(A_xx + A_yy + A_zz, Lsq)
    a = Lsq / (Lsq - 3.0)
    astar = a / f  # steady-state diagonal A* = (a/f).I

    # M = f.I (scalar; M_xy = 0), A* = (a/f).I (non-zero affine offset).
    return _affine_exponential_relaxation_step(
        A_xx, A_xy, A_yy, A_zz,
        f, 0.0, f, f,
        astar, 0.0, astar, astar,
        dt, lam)


def _ptt_relaxation_from_params(A_xx: jnp.ndarray,
                                A_xy: jnp.ndarray,
                                A_yy: jnp.ndarray,
                                A_zz: jnp.ndarray,
                                velocity: GridVariableVector,
                                dt: float,
                                params: Any,
                                ) -> Tuple[jnp.ndarray, jnp.ndarray,
                                           jnp.ndarray, jnp.ndarray]:
    """Linear PTT relaxation.

    A-space source ``R(A) = -(f/lam).(A - I)`` with the **linear** PTT
    trace function ``f = 1 + eps.(tr A - 3)``, ``eps = params['epsilon']``,
    ``tr A = A_xx + A_yy + A_zz``, slip ``zeta = 0``. This is affine in
    ``A`` with a **scalar** frozen operator and **zero** offset (the
    Giesekus structural case, with ``f`` scalar instead of the Giesekus
    matrix ``M``):

        R(A) = -(f/lam).(A - A*),   M = f.I (scalar),   A* = I,

    so it routes through the shared SPD-safe integrator
    (:func:`_affine_exponential_relaxation_step`) with frozen ``f``. The
    update ``A_new = I + e^{-(dt/lam)f}.(A - I)`` is a convex pull of an
    SPD state toward ``I`` => SPD-safe. No regularisation is needed:
    ``tr A >= 3`` in these flows => ``f >= 1 > 0`` always (``f`` is linear
    and bounded in ``tr A``, never singular -- contrast FENE-P's
    ``1/(L^2-tr A)``). At ``eps = 0`` => ``f == 1`` => **exact** Oldroyd-B
    analytic exponential (exact limit, like Giesekus ``alpha=0``).

    ``A_zz == 1`` is a fixed point (``R_zz = -(f/lam)(A_zz-1) = 0`` and no
    z-stretch), so the out-of-plane channel rides along inert -- the
    ``tr A - 3 = A_xx + A_yy - 2`` identity is the built-in ``zz`` check.
    Stress readout is the shared Hookean ``tau = Gp(A - I)``
    (:func:`_linear_conformation_stress_readout_fn`).
    """
    del velocity  # algebraic model: function of A and tr A only.
    lam = _params_get(params, 'lam')
    epsilon = _params_get(params, 'epsilon')

    f = 1.0 + epsilon * (A_xx + A_yy + A_zz - 3.0)

    # M = f.I (scalar; M_xy = 0), A* = I (zero offset, B* = 0).
    return _affine_exponential_relaxation_step(
        A_xx, A_xy, A_yy, A_zz,
        f, 0.0, f, f,
        1.0, 0.0, 1.0, 1.0,
        dt, lam)


def make_logconf_evolution_fn(*,
                                 psi_kernel: str = 'bk',
                                 uc_method: str = 'analytic',
                                 advect_method: str = 'rk2',
                                 wall_stencil: str = 'central',
                                 relaxation_fn=_oldroyd_b_relaxation_from_params,
                                 ):
    """Compose a log-conformation evolution_fn from named pieces.

    Three orthogonal kinematic axes plus a model-specific slot:

      * ``psi_kernel``   : ``'eig'`` | ``'bk'`` -- Psi<->A conversion.
        ``'eig'`` is the eigendecomp path
        (:func:`Psi_from_A` / :func:`A_from_Psi`); ``'bk'`` is the
        eigenvalue-free Cayley-Hamilton path
        (:func:`Psi_from_A_bk` / :func:`A_from_Psi_bk`), AD-clean
        on the ``A ~= I`` degenerate manifold.
      * ``uc_method``    : ``'fe'`` | ``'analytic'`` --
        upper-convective stretch integrator. ``'fe'`` is first-order
        forward-Euler on Psi (the original Hulsen/Basilisk path,
        :func:`upper_convective_increment` for ``eig`` and
        :func:`upper_convective_increment_bk` for ``bk``);
        ``'analytic'`` is the Fattal-Kupferman 2004 Sec.3.3 closed form
        :math:`A(t+dt) = E\\, A\\, E^T,\\ E = \\exp(dt\\, L)`
        (:func:`upper_convective_step_analytic`),
        unconditionally stable, exact for frozen ``L``, no eigenvalue
        smearing.
      * ``advect_method``: ``'fe'`` | ``'rk2'`` -- van-Leer transport
        integrator. ``'fe'`` is forward-Euler
        (:func:`_advect_psi_components_euler`); ``'rk2'`` is
        SSP-RK2 / Heun (:func:`_advect_psi_components_ssp_rk2`),
        second-order in time, ~2x the limited-scheme
        CFL ceiling.
      * ``wall_stencil``: ``'central'`` | ``'oneside_2nd_order'`` --
        velocity-gradient stencil at Dirichlet wall rows.
        ``'central'`` is
        :func:`_cell_centered_grad_u` (central diff composed with
        linear-extrapolation Dirichlet ghost -- formally O(dy^2) but
        only O(dy) in practice at the wall row, since the ghost is
        a 1st-order extrapolation through the wall);
        ``'oneside_2nd_order'`` swaps in
        :func:`_cell_centered_grad_u_wall_aware` which uses a 3-point
        one-sided stencil (wall + two interior cells) that kills the
        u'' truncation term exactly. Why: (i) close the ~0.6% bulk
        ``A_xy`` bias from the O(dy) wall stencil, (ii) shrink the
        first-step ``grad(u^0)`` impulse that seeds the Lie-split
        explicit-coupling cliff. Whether (ii) actually moves the
        cliff ``dt`` bracket is empirical, not assumed.
      * ``relaxation_fn`` : model-specific A-space stage with
        signature ``(A_xx, A_xy, A_yy, A_zz, velocity, dt, params) ->
        (A_xx, A_xy, A_yy, A_zz)``. Default is
        :func:`_oldroyd_b_relaxation_from_params` (analytic exponential
        relax for Oldroyd-B). ``A_zz`` is the out-of-plane conformation
        diagonal; ``velocity`` is threaded so the learned
        viscoelastic-TBNN closure can read ``gradu`` invariants --
        the algebraic family ignores it. Other constitutive models
        plug in here.

    Selection happens at *construction* time -- the returned
    ``evolution_fn`` calls only the chosen building blocks, with no
    runtime branches, so JIT sees a single static pipeline per
    returned function.

    Invalid combinations (e.g. ``psi_kernel='eig'`` with
    ``uc_method='analytic'``) are not policed -- the analytic UC step
    produces ``A`` directly, then the chosen ``psi_kernel`` is used
    only for the Psi<->A conversions surrounding advection; the result
    is well-defined but not particularly interesting.
    """
    if psi_kernel not in ('eig', 'bk'):
        raise ValueError(
            f"psi_kernel must be 'eig' or 'bk'; got {psi_kernel!r}.")
    if uc_method not in ('fe', 'analytic'):
        raise ValueError(
            f"uc_method must be 'fe' or 'analytic'; got {uc_method!r}.")
    if advect_method not in ('fe', 'rk2'):
        raise ValueError(
            f"advect_method must be 'fe' or 'rk2'; got {advect_method!r}.")
    if wall_stencil not in ('central', 'oneside_2nd_order'):
        raise ValueError(
            "wall_stencil must be 'central' or 'oneside_2nd_order'; "
            f"got {wall_stencil!r}.")

    psi_from_A_fn = Psi_from_A_bk if psi_kernel == 'bk' else Psi_from_A
    A_from_psi_fn = A_from_Psi_bk if psi_kernel == 'bk' else A_from_Psi
    advect_fn = (_advect_psi_components_ssp_rk2
                 if advect_method == 'rk2'
                 else _advect_psi_components_euler)
    # Velocity-gradient kernel for the upper-convective stage.
    # The wall-aware variant matches the central one in the bulk;
    # only the wall-row cross-shear components differ.
    grad_u_impl = (_cell_centered_grad_u_wall_aware
                    if wall_stencil == 'oneside_2nd_order'
                    else _cell_centered_grad_u)

    if uc_method == 'analytic':
        def stage1(Axx, Axy, Ayy, velocity, dt):
            A_xx_s, A_xy_s, A_yy_s = upper_convective_step_analytic(
                velocity, Axx, Axy, Ayy, dt, grad_u_fn=grad_u_impl)
            return psi_from_A_fn(A_xx_s, A_xy_s, A_yy_s)
    elif psi_kernel == 'bk':
        def stage1(Axx, Axy, Ayy, velocity, dt):
            Pxx, Pxy, Pyy = psi_from_A_fn(Axx, Axy, Ayy)
            dPxx, dPxy, dPyy = upper_convective_increment_bk(
                velocity, Pxx, Pxy, Pyy, dt, grad_u_fn=grad_u_impl)
            return Pxx + dPxx, Pxy + dPxy, Pyy + dPyy
    else:
        # eig + FE: the original eig-path. ``upper_convective_increment``
        # takes A directly (uses an eigendecomposition of A internally).
        def stage1(Axx, Axy, Ayy, velocity, dt):
            Pxx, Pxy, Pyy = psi_from_A_fn(Axx, Axy, Ayy)
            dPxx, dPxy, dPyy = upper_convective_increment(
                velocity, Axx, Axy, Ayy, dt, grad_u_fn=grad_u_impl)
            return Pxx + dPxx, Pxy + dPxy, Pyy + dPyy

    def evolution_fn(memory_fields: Tuple[GridVariable, ...],
                      velocity: GridVariableVector,
                      params: Any,
                      dt: float,
                      ) -> Tuple[GridVariable, ...]:
        if len(memory_fields) != 4:
            raise ValueError(
                f"log-conformation expects 4 memory fields "
                f"(A_xx, A_xy, A_yy, A_zz); got {len(memory_fields)}.")
        A_xx_var, A_xy_var, A_yy_var, A_zz_var = memory_fields
        grid = A_xx_var.grid
        bc_A = A_xx_var.bc
        Axx = A_xx_var.array.data
        Axy = A_xy_var.array.data
        Ayy = A_yy_var.array.data
        Azz = A_zz_var.array.data

        # Stage 1 -- in-plane upper-convective stretch (FE on Psi or FK
        # analytic). Unchanged 2x2 path. ``A_zz`` has no stretch in
        # planar flow (no z-velocity, no
        # z-gradient), so its log-conformation channel is just
        # ``Psi_zz = log A_zz`` carried straight through the stretch
        # stage (the BK Psi<->A kernel is a 2x2 op and is left untouched).
        Psi_xx_star, Psi_xy_star, Psi_yy_star = stage1(
            Axx, Axy, Ayy, velocity, dt)
        Psi_zz_star = jnp.log(Azz)

        # Wrap as GridVariables so advection's interpolation can read BCs.
        Psi_xx_var = GridVariable(
            GridArray(Psi_xx_star, CELL_CENTER_OFFSET_2D, grid), bc_A)
        Psi_xy_var = GridVariable(
            GridArray(Psi_xy_star, CELL_CENTER_OFFSET_2D, grid), bc_A)
        Psi_yy_var = GridVariable(
            GridArray(Psi_yy_star, CELL_CENTER_OFFSET_2D, grid), bc_A)
        Psi_zz_var = GridVariable(
            GridArray(Psi_zz_star, CELL_CENTER_OFFSET_2D, grid), bc_A)

        # Stage 2 -- van-Leer advection of Psi (FE or SSP-RK2 in time).
        # The ``A_zz`` channel is advected like the others; per-component
        # independence keeps the in-plane channels byte-identical.
        Psi_xx_new, Psi_xy_new, Psi_yy_new, Psi_zz_new = advect_fn(
            (Psi_xx_var, Psi_xy_var, Psi_yy_var, Psi_zz_var), velocity, dt)

        # Stage 3 -- back to A-space, then model-specific relaxation.
        A_xx_pre, A_xy_pre, A_yy_pre = A_from_psi_fn(
            Psi_xx_new, Psi_xy_new, Psi_yy_new)
        A_zz_pre = jnp.exp(Psi_zz_new)
        A_xx_new, A_xy_new, A_yy_new, A_zz_new = relaxation_fn(
            A_xx_pre, A_xy_pre, A_yy_pre, A_zz_pre, velocity, dt, params)

        return (
            GridVariable(GridArray(A_xx_new, CELL_CENTER_OFFSET_2D, grid), bc_A),
            GridVariable(GridArray(A_xy_new, CELL_CENTER_OFFSET_2D, grid), bc_A),
            GridVariable(GridArray(A_yy_new, CELL_CENTER_OFFSET_2D, grid), bc_A),
            GridVariable(GridArray(A_zz_new, CELL_CENTER_OFFSET_2D, grid), bc_A),
        )

    return evolution_fn


# ---------------------------------------------------------------------------
# Sec.6.A  Curated registered models
# ---------------------------------------------------------------------------
#
# Three presets cover the production use cases. Other combinations
# (e.g. ``bk + analytic UC + FE advect``, the "always-on accuracy
# upgrade with no CFL cost" preset) are available by calling
# :func:`make_logconf_evolution_fn` directly from a notebook or test
# without polluting the registry. Promote a combo to a registered
# preset only once it has a defensible "use me when..." docstring.

OLDROYD_B_LOGCONF: cr.ConstitutiveModel = cr.register(cr.ConstitutiveModel(
    name='oldroyd_b_logconf',
    state_spec=_LOGCONF_STATE_SPEC,
    evolution_fn=make_logconf_evolution_fn(
        psi_kernel='eig', uc_method='fe', advect_method='fe'),
    stress_readout_fn=_linear_conformation_stress_readout_fn,
    coupling_mode='explicit_force',
    polymer_linearization_fn=None,
))
"""Eig-path baseline: eig kernel + FE-on-Psi stretch + FE van-Leer.

Kept as the regression reference (it's the literal Hulsen-Basilisk
port). New training / production work should default to one of the
``_bk*`` siblings -- the eig kernel needs a ``double-where`` subgradient
fix on the ``A ~= I`` degenerate manifold which the BK siblings remove
entirely.
"""


OLDROYD_B_LOGCONF_BK: cr.ConstitutiveModel = cr.register(cr.ConstitutiveModel(
    name='oldroyd_b_logconf_bk',
    state_spec=_LOGCONF_STATE_SPEC,
    evolution_fn=make_logconf_evolution_fn(
        psi_kernel='bk', uc_method='fe', advect_method='fe'),
    stress_readout_fn=_linear_conformation_stress_readout_fn,
    coupling_mode='explicit_force',
    polymer_linearization_fn=None,
))
"""BK eigenvalue-free Psi<->A, FE-on-Psi stretch, FE van-Leer.

AD-clean (no degenerate-manifold subgradient choice). First-order in
time on both the stretch and the transport -- the conservative
"same truncation order as the eig path, but without the AD
quirks" preset. Use this when you want the same time discretization
as the eig path with no time-order surprises.
"""


OLDROYD_B_LOGCONF_BK_V2: cr.ConstitutiveModel = cr.register(cr.ConstitutiveModel(
    name='oldroyd_b_logconf_bk_v2',
    state_spec=_LOGCONF_STATE_SPEC,
    evolution_fn=make_logconf_evolution_fn(
        psi_kernel='bk', uc_method='analytic', advect_method='rk2'),
    stress_readout_fn=_linear_conformation_stress_readout_fn,
    coupling_mode='explicit_force',
    polymer_linearization_fn=None,
))
"""BK kernel + Fattal-Kupferman analytic stretch + SSP-RK2 advection.

The full kinematic upgrade. Closed-form integration of the frozen-``L``
upper-convective stretch (unconditionally stable, exact, kills the
eigenvalue-smearing artifact of FE-on-Psi) composed with second-order-
in-time SSP-RK2 van-Leer advection (~2x the limited-scheme CFL
ceiling). Cost ~4x per AD step at the same ``dt`` versus the FE
baselines; the wall-time win comes from being able to take a larger
``dt``. The CFL-relaxation crossover where v2 becomes net-faster is
problem- and hardware-dependent.
"""


OLDROYD_B_LOGCONF_BK_V2_WS2: cr.ConstitutiveModel = cr.register(cr.ConstitutiveModel(
    name='oldroyd_b_logconf_bk_v2_ws2',
    state_spec=_LOGCONF_STATE_SPEC,
    evolution_fn=make_logconf_evolution_fn(
        psi_kernel='bk', uc_method='analytic', advect_method='rk2',
        wall_stencil='oneside_2nd_order'),
    stress_readout_fn=_linear_conformation_stress_readout_fn,
    coupling_mode='explicit_force',
    polymer_linearization_fn=None,
))
"""``bk_v2`` plus the wall-aware velocity-gradient stencil.

Matches ``oldroyd_b_logconf_bk_v2`` in the bulk; differs
only on the two wall rows where ``du_x/dy`` (and ``du_y/dx`` if
``x`` is also Dirichlet) is now computed by the 2nd-order
one-sided stencil
``(+/-4.u_wall -/+ 3.u_{wall-adjacent} -/+ u_{one-in}) / (3.dy)``
instead of central-FD-with-extrapolation-ghost. Why:

  1. Bulk-accuracy fix. The central-plus-ghost wall stencil has
     leading truncation ``-(dy/8).u''(y_wall)`` -- i.e., it is *one
     order less accurate at the wall row than the bulk centered
     stencil*. For Couette this stencil happens to be exact on the
     analytic linear profile (``u'' = 0``), but during the transient
     and in any not-perfectly-linear discrete solution the wall-row
     ``gradu`` carries an O(dy) error that bleeds into the bulk
     ``A_xy`` via the conformation transport.
  2. Cliff-seed reduction (testable, not assumed). The Lie-split
     composition order in ``memory_be_imex_stepper`` converts the
     first-step ``gradu^0`` into a ``G_p``-proportional polymer body
     force on step 1. Sharper wall ``gradu`` resolution should
     shrink that seed; whether the cliff ``dt`` bracket actually
     moves is empirical.

Cost: a few extra ``.at[].set()`` ops per UC step, no change to
asymptotic complexity. Same AD-cleanliness as ``bk_v2`` (the new
stencil is bilinear in the wall and interior u-values; trivially
differentiable).
"""


# ---------------------------------------------------------------------------
# Sec.6.B  Family extension -- Giesekus
# ---------------------------------------------------------------------------

GIESEKUS_LOGCONF_BK_V2: cr.ConstitutiveModel = cr.register(cr.ConstitutiveModel(
    name='giesekus_logconf_bk_v2',
    state_spec=_LOGCONF_STATE_SPEC,
    evolution_fn=make_logconf_evolution_fn(
        psi_kernel='bk', uc_method='analytic', advect_method='rk2',
        relaxation_fn=_giesekus_relaxation_from_params),
    stress_readout_fn=_linear_conformation_stress_readout_fn,
    coupling_mode='explicit_force',
    polymer_linearization_fn=None,
))
"""Giesekus: the ``bk_v2`` kinematic backbone
(BK eigenvalue-free Psi<->A + Fattal-Kupferman analytic UC stretch +
SSP-RK2 advection) with the Giesekus relaxation source
``R(A) = -(1/lam)[(A - I) + alpha (A - I)^2]``, ``alpha = params['alpha']``.

Shares the 4-component ``_LOGCONF_STATE_SPEC`` and the Hookean stress
readout ``tau = Gp(A - I)`` with Oldroyd-B; the only model-specific piece
is :func:`_giesekus_relaxation_from_params`, which routes through the
shared SPD-safe affine integrator (:func:`_affine_exponential_relaxation_step`)
with frozen ``M = I + alpha(A - I)`` and steady state ``A* = I``. At
``alpha = 0`` it reduces to the exact Oldroyd-B analytic exponential, so the
``alpha = 0`` regression against ``oldroyd_b_logconf_bk_v2`` is
machine-precision. Giesekus has bounded
extensional viscosity and a non-zero second normal-stress difference
``N2 = Gp(A_yy - A_zz) < 0`` (carried by ``A_yy < 1``; ``A_zz == 1``).
"""


# ---------------------------------------------------------------------------
# Sec.6.C  Family extension -- FENE-P
# ---------------------------------------------------------------------------

FENE_P_LOGCONF_BK_V2: cr.ConstitutiveModel = cr.register(cr.ConstitutiveModel(
    name='fene_p_logconf_bk_v2',
    state_spec=_LOGCONF_STATE_SPEC,
    evolution_fn=make_logconf_evolution_fn(
        psi_kernel='bk', uc_method='analytic', advect_method='rk2',
        relaxation_fn=_fene_p_relaxation_from_params),
    stress_readout_fn=_fene_p_stress_readout_fn,
    coupling_mode='explicit_force',
    polymer_linearization_fn=None,
))
"""FENE-P: the ``bk_v2`` kinematic backbone with
the finite-extensibility Peterlin relaxation
``R(A) = -(1/lam)[f.A - a.I]``, ``f = L^2/(L^2 - tr A)``, ``a = L^2/(L^2 - 3)``,
``L^2 = params['Lsq']``.

This is the family member the 4-component state was built for: it
shares the 4-component ``_LOGCONF_STATE_SPEC`` but ``A_zz`` is genuinely
active (the Peterlin trace couples all diagonals; steady ``A_zz = a/f <
1`` in shear). It is also the first member with a **non-Hookean** stress
readout -- :func:`_fene_p_stress_readout_fn` (``tau = Gp(f.A - a.I)``),
**not** the shared linear form. Relaxation routes through the shared
SPD-safe affine integrator with frozen ``M = f.I`` and **non-zero**
offset ``A* = (a/f).I`` (the offset the integrator was designed to carry
from the start). At ``L^2 -> inf`` it reduces to Oldroyd-B asymptotically
(``O(1/L^2)``); FENE-P has bounded extensional viscosity (finite
extensibility caps ``tr A < L^2``).
"""


# ---------------------------------------------------------------------------
# Sec.6.D  Family extension -- linear PTT
# ---------------------------------------------------------------------------

PTT_LINEAR_LOGCONF_BK_V2: cr.ConstitutiveModel = cr.register(
    cr.ConstitutiveModel(
        name='ptt_linear_logconf_bk_v2',
        state_spec=_LOGCONF_STATE_SPEC,
        evolution_fn=make_logconf_evolution_fn(
            psi_kernel='bk', uc_method='analytic', advect_method='rk2',
            relaxation_fn=_ptt_relaxation_from_params),
        stress_readout_fn=_linear_conformation_stress_readout_fn,
        coupling_mode='explicit_force',
        polymer_linearization_fn=None,
    ))
"""Linear PTT: the ``bk_v2`` backbone with the
Phan-Thien-Tanner trace relaxation ``R(A) = -(f/lam)(A - I)``,
``f = 1 + eps.(tr A - 3)``, ``eps = params['epsilon']``, slip ``zeta = 0``.

Structurally the simplest family member: scalar frozen ``M = f.I``,
**zero** offset ``A* = I`` (the Giesekus case with scalar ``f``), and it
**reuses the shared Hookean readout** ``tau = Gp(A - I)``
(:func:`_linear_conformation_stress_readout_fn`) -- no FENE-P-style
custom stress. ``A_zz == 1`` is inert (fixed point), so the 4th component
is a sanity check rather than an active DOF. No regularisation: ``f >= 1``
always (``tr A >= 3``), never singular. At ``eps = 0`` it is **exactly**
Oldroyd-B. Like FENE-P it shear-thins and has bounded extensional
viscosity, but via the linear trace function rather than finite
extensibility, and with ``N2 = 0`` (``zeta = 0``).
"""


__all__ = (
    # Building blocks (eig path).
    'eig2x2_symmetric',
    'Psi_from_A',
    'A_from_Psi',
    'upper_convective_increment',
    'oldroyd_b_relaxation_analytic',
    # Building blocks (BK eigenvalue-free).
    'Psi_from_A_bk',
    'A_from_Psi_bk',
    'upper_convective_increment_bk',
    # Building blocks (analytic UC stretch).
    'upper_convective_step_analytic',
    # Composable factory + curated registered models (Sec.6).
    'make_logconf_evolution_fn',
    'OLDROYD_B_LOGCONF',
    'OLDROYD_B_LOGCONF_BK',
    'OLDROYD_B_LOGCONF_BK_V2',
    'OLDROYD_B_LOGCONF_BK_V2_WS2',
    # Shared readouts + family members.
    'linear_conformation_tau_zz_readout',
    'GIESEKUS_LOGCONF_BK_V2',
    'fene_p_tau_zz_readout',
    'FENE_P_LOGCONF_BK_V2',
    'PTT_LINEAR_LOGCONF_BK_V2',
)

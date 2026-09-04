"""Offline post-processing diagnostics for the lid-driven cavity.

Pure NumPy / offline (login node / ``test`` partition) -- outside the
differentiable inner loop. Provides the cavity metrics:

  * ``vorticity_cell_centered`` -- omega = dv/dx - du/dy at cell centers.
  * ``streamfunction`` -- solve del^2 psi = -omega with psi=0 on all four
    walls (closed cavity), via a Dirichlet-Dirichlet fast-diagonalization
    Poisson solve (reuses the same fast-diag infra as the pressure solve).
  * ``psi_min_and_center`` -- the primary-vortex strength psi_min and its
    (x, y) location (well-resolved on a moderate uniform grid).
  * ``centerline_profiles`` -- u(y) on x=L/2 and v(x) on y=L/2, interpolated
    to the geometric centerlines for point-matching published Newtonian tables.

Staggering note: ``u`` lives on x-faces (offset (1, 0.5)), ``v`` on y-faces
(offset (0.5, 1)). We interpolate both to cell centers with a simple
two-point average before central-differencing the vorticity; on a uniform
grid this is 2nd-order and adequate for psi_min-to-<1-cell and centerline
overlays. Absolute psi_min magnitude converges to the spectral value under
refinement (Richardson check in the runner).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _to_cell_center(u: np.ndarray, v: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Average the staggered ``u`` (x-faces) and ``v`` (y-faces) to centers."""
    # u[i,j] at x-face -> center i is average of faces (i-1, i).
    u_c = 0.5 * (u + np.roll(u, 1, axis=0))
    # v[i,j] at y-face -> center j is average of faces (j-1, j).
    v_c = 0.5 * (v + np.roll(v, 1, axis=1))
    return u_c, v_c


def vorticity_cell_centered(u: np.ndarray, v: np.ndarray, grid) -> np.ndarray:
    """omega = dv/dx - du/dy at cell centers (central differences)."""
    dx, dy = grid.step
    u_c, v_c = _to_cell_center(u, v)
    dvdx = (np.roll(v_c, -1, axis=0) - np.roll(v_c, 1, axis=0)) / (2.0 * dx)
    dudy = (np.roll(u_c, -1, axis=1) - np.roll(u_c, 1, axis=1)) / (2.0 * dy)
    return dvdx - dudy


def streamfunction(u: np.ndarray, v: np.ndarray, grid) -> np.ndarray:
    """Solve del^2 psi = -omega with psi = 0 on all walls (closed cavity).

    Cell-centered psi via fast diagonalization with homogeneous Dirichlet
    on both axes (non-singular). Returns psi as an (Nx, Ny) array.
    """
    import jax.numpy as jnp
    from jax_ib.base import boundaries, array_utils, grids as ib_grids
    from jax_cfd.base import fast_diagonalization

    omega = vorticity_cell_centered(u, v, grid)
    rhs = -np.asarray(omega)

    D = boundaries.BCType.DIRICHLET
    psi_bc = boundaries.ConstantBoundaryConditions(
        0.0, ((0.0, 0.0), (0.0, 0.0)), ((D, D), (D, D)), lambda t: 0.0)
    laplacians = array_utils.laplacian_matrix_w_boundaries(
        grid, grid.cell_center, psi_bc)
    pinv = fast_diagonalization.pseudoinverse(
        laplacians, rhs.dtype, hermitian=True, circulant=False,
        implementation='matmul')
    psi = np.asarray(pinv(jnp.asarray(rhs)))
    return psi


def streamfunction_nodal(u: np.ndarray, v: np.ndarray, grid):
    """MAC-native streamfunction: psi at interior cell corners, psi=0 on walls.

    More accurate than :func:`streamfunction` (which center-averages the
    staggered velocity before differencing, smoothing the vortex-core
    extremum). Vorticity is formed directly at cell corners from the
    staggered faces:

        omega_node[i,j] = (v[i+1,j] - v[i,j])/dx - (u[i,j+1] - u[i,j])/dy

    (u on x-faces, v on y-faces), giving (Nx-1) x (Ny-1) interior corner
    nodes. The Dirichlet Poisson  -del^2 psi = omega,  psi=0 on the walls,
    is then solved exactly by a type-1 discrete sine transform on a uniform
    grid. Returns (psi_nodes[(Nx-1),(Ny-1)], x_nodes, y_nodes).
    """
    from scipy.fft import dstn, idstn
    dx, dy = grid.step
    # corner vorticity (interior nodes)
    dvdx = (v[1:, :-1] - v[:-1, :-1]) / dx      # (Nx-1, Ny-1)
    dudy = (u[:-1, 1:] - u[:-1, :-1]) / dy       # (Nx-1, Ny-1)
    omega = dvdx - dudy
    m, n = omega.shape
    # DST-I Poisson: eigenvalues of the 1-D Dirichlet Laplacian.
    kx = np.arange(1, m + 1)
    ky = np.arange(1, n + 1)
    lamx = (2.0 / dx**2) * (1.0 - np.cos(np.pi * kx / (m + 1)))
    lamy = (2.0 / dy**2) * (1.0 - np.cos(np.pi * ky / (n + 1)))
    denom = lamx[:, None] + lamy[None, :]
    rhs_hat = dstn(omega, type=1)               # -del^2 psi = omega
    psi_hat = rhs_hat / denom
    psi = idstn(psi_hat, type=1)
    x0, y0 = grid.domain[0][0], grid.domain[1][0]
    x_nodes = x0 + (np.arange(1, m + 1)) * dx
    y_nodes = y0 + (np.arange(1, n + 1)) * dy
    return psi, x_nodes, y_nodes


def psi_min_and_center_nodal(u, v, grid):
    """psi_min and its (x, y) location from the nodal streamfunction."""
    psi, xn, yn = streamfunction_nodal(u, v, grid)
    idx = int(np.argmin(psi))
    i, j = np.unravel_index(idx, psi.shape)
    return float(psi[i, j]), (float(xn[i]), float(yn[j])), psi


def psi_min_and_center(psi: np.ndarray, grid
                       ) -> Tuple[float, Tuple[float, float], Tuple[int, int]]:
    """Return (psi_min, (x, y) location, (i, j) index) of the primary vortex."""
    idx = int(np.argmin(psi))
    i, j = np.unravel_index(idx, psi.shape)
    xc = np.asarray(grid.axes(grid.cell_center)[0])
    yc = np.asarray(grid.axes(grid.cell_center)[1])
    return float(psi[i, j]), (float(xc[i]), float(yc[j])), (int(i), int(j))


def centerline_profiles(u: np.ndarray, v: np.ndarray, grid):
    """u(y) on x=L/2 and v(x) on y=L/2, interpolated to the centerlines.

    Returns ``(y_u, u_center, x_v, v_center)``:
      * ``y_u``: y-coordinates of the u samples (u's y-face-cell positions);
        ``u_center``: u interpolated to the vertical centerline x=L/2.
      * ``x_v``: x-coordinates of the v samples; ``v_center``: v interpolated
        to the horizontal centerline y=L/2.
    """
    Lx = grid.domain[0][1] - grid.domain[0][0]
    Ly = grid.domain[1][1] - grid.domain[1][0]
    x_mid = 0.5 * (grid.domain[0][0] + grid.domain[0][1])
    y_mid = 0.5 * (grid.domain[1][0] + grid.domain[1][1])

    # u lives on x-faces: x = grid.axes(cell_faces[0])[0], y = ...[1].
    xu = np.asarray(grid.axes(grid.cell_faces[0])[0])
    yu = np.asarray(grid.axes(grid.cell_faces[0])[1])
    # interpolate u(x=x_mid, y) linearly in x for each y-row.
    u_center = np.array([np.interp(x_mid, xu, u[:, j]) for j in range(u.shape[1])])

    xv = np.asarray(grid.axes(grid.cell_faces[1])[0])
    yv = np.asarray(grid.axes(grid.cell_faces[1])[1])
    v_center = np.array([np.interp(y_mid, yv, v[i, :]) for i in range(v.shape[0])])

    return yu, u_center, xv, v_center


def classify_steadiness(ke_hist, max_Axx_hist, psi_min_hist, *,
                            ke_relstd_tol=5e-3, stress_reltrend_tol=1e-2):
    """Classify a cavity run from scalar time histories (offline, host NumPy).

    Returns ``'STEADY'`` only if, over the **last 25%** of frames:

      (1) relative std of ``ke_hist`` < ``ke_relstd_tol``;
      (2) ``max_Axx_hist`` is not still trending:
          ``|linfit_slope * T_window / mean(max_Axx_window)|``
          < ``stress_reltrend_tol``;
      (3) the same trend test on ``psi_min_hist`` < ``stress_reltrend_tol``.

    Returns ``'SURVIVED'`` if no NaN/Inf but not all three hold.
    Returns ``'BLEW_UP'`` if any NaN/Inf appears in any history.
    """
    ke = np.asarray(ke_hist, dtype=np.float64)
    axx = np.asarray(max_Axx_hist, dtype=np.float64)
    psi = np.asarray(psi_min_hist, dtype=np.float64)
    if (ke.size == 0 or axx.size == 0 or psi.size == 0
            or ke.shape != axx.shape or ke.shape != psi.shape):
        raise ValueError('ke_hist, max_Axx_hist, psi_min_hist must be same-length '
                         f'non-empty 1-D arrays; got {ke.shape}, {axx.shape}, '
                         f'{psi.shape}')
    if (not np.all(np.isfinite(ke)) or not np.all(np.isfinite(axx))
            or not np.all(np.isfinite(psi))):
        return 'BLEW_UP'

    n = ke.size
    n_win = max(int(np.ceil(n * 0.25)), 2)
    T_window = float(n_win - 1)

    ke_w = ke[-n_win:]
    ke_relstd = float(np.std(ke_w) / max(abs(np.mean(ke_w)), 1e-30))

    def _rel_trend(y):
        y_w = y[-n_win:]
        x = np.arange(n_win, dtype=np.float64)
        slope = float(np.polyfit(x, y_w, 1)[0])
        return abs(slope * T_window / max(abs(np.mean(y_w)), 1e-30))

    if (ke_relstd < ke_relstd_tol
            and _rel_trend(axx) < stress_reltrend_tol
            and _rel_trend(psi) < stress_reltrend_tol):
        return 'STEADY'
    return 'SURVIVED'


# ---------------------------------------------------------------------------
# Reference benchmark tables (for the runner to overlay / tabulate).
# ---------------------------------------------------------------------------
# Published Newtonian cavity -- u on the vertical centerline x=0.5.
GHIA_Y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
                   0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                   0.9688, 0.9766, 1.0000])
GHIA_U = {
    100: np.array([0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                   -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                   0.68717, 0.73722, 0.78871, 0.84123, 1.0]),
    400: np.array([0.0, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299,
                   -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093,
                   0.55892, 0.61756, 0.68439, 0.75837, 1.0]),
    1000: np.array([0.0, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
                    -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304,
                    0.46604, 0.51117, 0.57492, 0.65928, 1.0]),
}
# Published Newtonian cavity -- v on the horizontal centerline y=0.5.
GHIA_X = np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
                   0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
                   0.9609, 0.9688, 1.0000])
GHIA_V = {
    100: np.array([0.0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507,
                   0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313,
                   -0.08864, -0.07391, -0.05906, 0.0]),
    400: np.array([0.0, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124, 0.30203,
                   0.30174, 0.05186, -0.38598, -0.44993, -0.23827, -0.22847,
                   -0.19254, -0.15663, -0.12146, 0.0]),
    1000: np.array([0.0, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075,
                    0.32235, 0.02526, -0.31966, -0.42665, -0.51550, -0.39188,
                    -0.33714, -0.27669, -0.21388, 0.0]),
}

# Spectral Re=1000 Newtonian cavity, +x-lid convention (mirrored
# from the published -x-lid tables). Primary vortex + centerline extrema.
BOTELLA_PEYRET_RE1000 = dict(
    psi_min=-0.1189366,
    omega=-2.067753,
    center=(0.5308, 0.5652),
    u_min=-0.3885698,     # extreme u on x=0.5
    u_min_y=0.1717,
)

"""Grid reconstruction, staggered->centre interpolation and plot-time masking.

Everything here is *plot-time only*.  No function in this module is ever
applied to an array before a metric is computed; the figure modules compute
metrics from the raw loaded arrays and only then call into here.

Boundary-condition rendering rules (field-plot conventions) are implemented by
:func:`plot_grid`, which returns a node grid that includes the wall
coordinates.  Because the wall nodes carry the exact boundary value (0 for
no-slip velocity), a filled contour drawn on that grid renders the zero at the
true wall rather than half a cell inside the fluid.  Solid regions are cut on
cell edges, so the grey patch and the coloured field share an edge exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Contraction geometry.  Domain and solid set are transcribed from
# jax_rheology/contraction_geometry.py (READ-ONLY):
#   Omega = [-L_up, L_down] x [-R*H, R*H],  S = {x >= 0, H < |y| <= R*H}
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractionGrid:
    xc: np.ndarray          # (nx,) cell centres in x
    yc: np.ndarray          # (ny,) cell centres in y (also the y-station of u)
    xfu: np.ndarray         # (nx,) x-faces where u lives (right face of cell)
    H: float
    R: float
    L_up: float
    L_down: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.xc.size, self.yc.size)

    @property
    def dx(self) -> float:
        return float(self.xc[1] - self.xc[0])

    @property
    def dy(self) -> float:
        return float(self.yc[1] - self.yc[0])

    @property
    def x_edges(self) -> np.ndarray:
        return np.concatenate([[self.xc[0] - self.dx / 2],
                               self.xc + self.dx / 2])

    @property
    def y_edges(self) -> np.ndarray:
        return np.concatenate([[self.yc[0] - self.dy / 2],
                               self.yc + self.dy / 2])

    @property
    def yfv(self) -> np.ndarray:
        """y-faces where v lives (top face of each cell)."""
        return self.yc + self.dy / 2

    def solid_mask(self) -> np.ndarray:
        """Cell-centre solid mask, S = {x >= 0, H < |y| <= R H}."""
        X, Y = np.meshgrid(self.xc, self.yc, indexing="ij")
        return (X >= 0.0) & (np.abs(Y) > self.H) & (np.abs(Y) <= self.R * self.H)

    def fluid_mask(self) -> np.ndarray:
        return ~self.solid_mask()

    def step_x_snapped(self) -> float:
        """The contraction plane x=0 snapped OUTWARD to the nearest grid line.

        The solver's step at x=0 does not land on a cell edge at 128x256, so
        the drawn mask is moved to the first cell edge at or beyond the first
        solid cell centre.  Drawing the grey patch there makes it coincide
        exactly with the edge of the coloured field.
        """
        edges = self.x_edges
        first_solid = int(np.argmax(self.xc >= 0.0))
        return float(edges[first_solid])

    def solid_rectangles(self) -> list[tuple[float, float, float, float]]:
        """(x0, y0, width, height) of the two solid blocks.

        Drawn from the geometric contraction plane x = 0, not from the snapped
        cell edge the mask uses.  At 128x256 those differ by 0.047 H (a third
        of a cell), and starting the patch at the snapped edge leaves a visible
        strip of field between x = 0 and the block.  Starting it at 0 instead
        covers that strip, which is the geometry the config defines.
        """
        x0 = 0.0
        x1 = float(self.x_edges[-1])
        y_hi = self.R * self.H
        return [
            (x0, self.H, x1 - x0, y_hi - self.H),
            (x0, -y_hi, x1 - x0, y_hi - self.H),
        ]


def contraction_grid_from_archive(arch: dict) -> ContractionGrid:
    """Reconstruct the grid from the coordinate arrays stored in the archive."""
    return ContractionGrid(
        xc=np.asarray(arch["xc"], float), yc=np.asarray(arch["yc"], float),
        xfu=np.asarray(arch["xfu"], float),
        H=float(arch.get("H", 1.0)), R=float(arch.get("R", 4.0)),
        L_up=6.0, L_down=12.0,
    )


def contraction_grid_from_config(cfg: dict) -> ContractionGrid:
    """Reconstruct the grid from a run config (nx, ny, H, ratio, L_up, L_down)."""
    H = float(cfg["H"])
    R = float(cfg["ratio"])
    L_up, L_down = float(cfg["L_up"]) * H, float(cfg["L_down"]) * H
    nx, ny = int(cfg["nx"]), int(cfg["ny"])
    dx = (L_down + L_up) / nx
    dy = 2.0 * R * H / ny
    xc = -L_up + (np.arange(nx) + 0.5) * dx
    yc = -R * H + (np.arange(ny) + 0.5) * dy
    return ContractionGrid(xc=xc, yc=yc, xfu=xc + dx / 2, H=H, R=R,
                           L_up=L_up, L_down=L_down)


# --------------------------------------------------------------------------
# Staggered -> centre interpolation
# --------------------------------------------------------------------------

def u_faces_to_centres(u: np.ndarray, *, inlet: np.ndarray | float | None = None
                       ) -> np.ndarray:
    """u lives on the right x-face of each cell; average onto cell centres.

    ``inlet`` is the value on the left face of the first cell (the domain
    inlet, not a wall).  Default is edge clamping, i.e. the first cell centre
    inherits the first face value; the inlet column sits at x = -6 H, far
    outside every plotted window.
    """
    if inlet is None:
        left = u[:1, :]
    else:
        left = np.broadcast_to(np.atleast_1d(inlet), (u.shape[1],))[None, :]
    shifted = np.concatenate([left, u[:-1, :]], axis=0)
    return 0.5 * (shifted + u)


def v_faces_to_centres(v: np.ndarray) -> np.ndarray:
    """v lives on the top y-face of each cell; average onto cell centres.

    The bottom face of the first cell is a no-slip wall, so v = 0 there.
    """
    below = np.zeros((v.shape[0], 1))
    shifted = np.concatenate([below, v[:, :-1]], axis=1)
    return 0.5 * (shifted + v)


# --------------------------------------------------------------------------
# Plot grid: cell centres plus the wall nodes the field-plot rules require
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PlotGrid:
    X: np.ndarray
    Y: np.ndarray
    xn: np.ndarray
    yn: np.ndarray
    solid: np.ndarray       # boolean, True strictly inside a solid block


def _interp_to_nodes(xc, yc, F, xn, yn):
    """Separable linear interpolation onto the node grid, clamped at edges."""
    out = np.empty((xn.size, yc.size), float)
    for j in range(yc.size):
        out[:, j] = np.interp(xn, xc, F[:, j])
    res = np.empty((xn.size, yn.size), float)
    for i in range(xn.size):
        res[i, :] = np.interp(yn, yc, out[i, :])
    return res


def contraction_plot_grid(grid: ContractionGrid) -> PlotGrid:
    """Node grid: cell centres + inlet/outlet + all wall coordinates + y=0."""
    x_step = grid.step_x_snapped()
    xn = np.unique(np.concatenate([
        [float(grid.x_edges[0])], grid.xc, [x_step], [float(grid.x_edges[-1])]]))
    y_hi = grid.R * grid.H
    yn = np.unique(np.concatenate([
        [-y_hi], grid.yc, [-grid.H, 0.0, grid.H], [y_hi]]))
    X, Y = np.meshgrid(xn, yn, indexing="ij")
    # Nodes *on* the snapped step face and on |y| = H belong to the boundary,
    # not to the interior of the block: keeping them finite lets the filled
    # contour reach the grid line the grey patch starts on, with no sliver.
    solid = (X > x_step + 1e-12) & (np.abs(Y) > grid.H + 1e-12)
    return PlotGrid(X=X, Y=Y, xn=xn, yn=yn, solid=solid)


def contraction_to_plot(field: np.ndarray, grid: ContractionGrid,
                        pg: PlotGrid, *, velocity: bool) -> np.ndarray:
    """Interpolate a cell-centre field onto the plot grid and apply the field-plot rules.

    ``velocity=True`` forces the field to exactly zero on every no-slip wall
    (outer walls |y| = R H, the step faces |y| = H for x >= x_step, and the
    step front x = x_step for |y| >= H) and inside the solid blocks.  Scalars
    such as ``tr A`` are only extended to the wall, never zeroed.
    """
    F = _interp_to_nodes(grid.xc, grid.yc, np.asarray(field, float),
                         pg.xn, pg.yn)
    x_step = grid.step_x_snapped()
    y_hi = grid.R * grid.H
    if velocity:
        wall = np.isclose(np.abs(pg.Y), y_hi)
        wall |= np.isclose(np.abs(pg.Y), grid.H) & (pg.X >= x_step - 1e-12)
        wall |= np.isclose(pg.X, x_step) & (np.abs(pg.Y) >= grid.H - 1e-12)
        F = np.where(wall, 0.0, F)
        F = np.where(pg.solid, 0.0, F)
    F = np.where(pg.solid, np.nan, F)
    return F


def mirrored_split(top: np.ndarray, bottom: np.ndarray,
                   pg: PlotGrid) -> np.ndarray:
    """Truth above the centreline, learned below (prompt N1a / SN4a)."""
    return np.where(pg.Y >= 0.0, top, bottom)


# --------------------------------------------------------------------------
# ROI band.  Transcribed from passc_contraction_figures.py::_roi so the band
# drawn here is the same band the archived metrics were computed on.
# --------------------------------------------------------------------------

def roi_fields(args: dict, grid: ContractionGrid):
    X, Y = np.meshgrid(grid.xc, grid.yc, indexing="ij")
    ubar = float(args["ratio"]) * float(args["U"])
    delta = ubar * float(args["truth_lam"])
    x_on = float(args["roi_xc"]) - float(args["roi_a"])
    x_off = float(args["roi_xc"]) + float(args["roi_c"]) * delta
    sig = lambda v: 1.0 / (1.0 + np.exp(-v))
    psi_x = sig((X - x_on) / float(args["roi_ell"]))
    psi_x = psi_x * sig((x_off - X) / float(args["roi_ell"]))
    psi_y = np.exp(-(Y ** 2) / (2.0 * float(args["roi_sigma_y"]) ** 2))
    activation = psi_x * psi_y
    raw = 1.0 + float(args["roi_kappa"]) * activation
    return raw / raw.mean(), activation, x_on, x_off


# --------------------------------------------------------------------------
# Cavity geometry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CavityGrid:
    n: int
    L: float

    @property
    def centres(self) -> np.ndarray:
        return (np.arange(self.n) + 0.5) * self.L / self.n

    @property
    def edges(self) -> np.ndarray:
        return np.arange(self.n + 1) * self.L / self.n


def cavity_grid_from_config(cfg: dict) -> CavityGrid:
    return CavityGrid(n=int(cfg["cells"]), L=float(cfg["L"]))


def cavity_plot_grid(grid: CavityGrid) -> PlotGrid:
    c = grid.centres
    xn = np.unique(np.concatenate([[0.0], c, [grid.L]]))
    X, Y = np.meshgrid(xn, xn, indexing="ij")
    return PlotGrid(X=X, Y=Y, xn=xn, yn=xn, solid=np.zeros(X.shape, bool))


def cavity_to_plot(field: np.ndarray, grid: CavityGrid, pg: PlotGrid, *,
                   velocity: bool, lid_component: str | None = None,
                   U_lid: float = 0.0) -> np.ndarray:
    """Cell-centre cavity field onto the wall-inclusive node grid.

    Velocity is zeroed on the three stationary walls.  On the lid the
    x-component takes ``U_lid`` and the y-component zero.
    """
    c = grid.centres
    F = _interp_to_nodes(c, c, np.asarray(field, float), pg.xn, pg.yn)
    if velocity:
        stationary = (np.isclose(pg.X, 0.0) | np.isclose(pg.X, grid.L)
                      | np.isclose(pg.Y, 0.0))
        F = np.where(stationary, 0.0, F)
        lid = np.isclose(pg.Y, grid.L) & ~np.isclose(pg.X, 0.0) \
            & ~np.isclose(pg.X, grid.L)
        F = np.where(lid, U_lid if lid_component == "u" else 0.0, F)
    return F


# --------------------------------------------------------------------------
# EVP channel
# --------------------------------------------------------------------------

EVP_NY = 64
EVP_LY = 2.0
EVP_H = 1.0


def evp_y(ny: int = EVP_NY, Ly: float = EVP_LY, H: float = EVP_H):
    """Cell centres and centred coordinate; y is NOT stored in the npz."""
    y = (np.arange(ny) + 0.5) * (Ly / ny)
    return y, y - H


def evp_profile_with_walls(u: np.ndarray, ny: int = EVP_NY,
                           Ly: float = EVP_LY, H: float = EVP_H):
    """Append the two no-slip walls at yhat = -H, +H with u = 0 exactly."""
    _, yhat = evp_y(ny, Ly, H)
    yy = np.concatenate([[-H], yhat, [H]])
    uu = np.concatenate([[0.0], np.asarray(u, float), [0.0]])
    return yy, uu


def plug_halfwidth(yhat: np.ndarray, u: np.ndarray, *,
                   frac: float = 0.01) -> float:
    """Kinematic plug half-width: |yhat| where |du/dyhat| first exceeds
    ``frac`` of its channel maximum, scanning outward from the centreline."""
    du = np.gradient(np.asarray(u, float), np.asarray(yhat, float))
    a = np.abs(du)
    thresh = frac * a.max()
    core = np.abs(yhat) <= np.abs(yhat).max()
    idx = np.argsort(np.abs(yhat))
    for i in idx:
        if core[i] and a[i] > thresh:
            return float(abs(yhat[i]))
    return float(np.abs(yhat).max())

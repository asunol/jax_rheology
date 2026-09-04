"""Boundary conditions for velocity, pressure, and the conformation tensor.

Extends the jax_ib boundary types with what the viscoelastic solvers need:
conditions on the conformation field (:class:`ConformationBoundaryConditions`,
with wall handling selectable between extrapolation and Neumann), the moving
lid of the driven cavity, and the inflow/outflow pair of the contraction.
:class:`BCSpec` bundles the velocity, pressure, and conformation conditions
that one geometry needs so a forward driver can pass them as a single object.
"""

import dataclasses
from typing import Literal, Optional, Tuple

import numpy as np
import jax.numpy as jnp
from jax import lax
from jax.tree_util import register_pytree_node_class
from jax_ib.base import grids, boundaries


def create_bc(grid):
    """A helper function to correctly create the new BC object."""
    ndim = grid.ndim
    periodic_types = ((boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),) * ndim
    periodic_values = ((None, None),) * ndim
    dummy_fn = lambda t: 0.0
    return boundaries.ConstantBoundaryConditions(time_stamp=0.0, values=periodic_values, types=periodic_types, boundary_fn=dummy_fn)


def create_bc_ndim(ndim: int):
    """Create corrected periodic BCs using only the dimensionality.

    This avoids requiring a Grid object at call sites that only know the domain.
    """
    periodic_types = ((boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),) * ndim
    periodic_values = ((None, None),) * ndim
    dummy_fn = lambda t: 0.0
    return boundaries.ConstantBoundaryConditions(time_stamp=0.0, values=periodic_values, types=periodic_types, boundary_fn=dummy_fn)


# ---------------------------------------------------------------------------
# Wall conformation BC option
# ---------------------------------------------------------------------------
#
# Wall BCs for the conformation field are *model-agnostic*: every
# constitutive model whose memory field carries cell-centered SPD /
# symmetric data on a wall-bounded grid faces the same closure choice.
# The plan picks linear extrapolation of ``Psi = log(A)`` (RheoTool
# practice; automatically SPD-safe because ``A = exp(Psi)`` is SPD for
# any real symmetric Psi) as the default, with homogeneous Neumann
# retained as a debug / fallback baseline.
#
# Linear extrapolation of ``Psi = log(A)`` is the default (RheoTool;
# SPD-safe because ``A = exp(Psi)``). Homogeneous Neumann is the
# debug / fallback baseline and maps onto the existing
# ``HomogeneousBoundaryConditions`` if a caller asks for it on a
# wall-bounded axis.

WallConformationBC = Literal['extrapolation', 'neumann']
WALL_CONFORMATION_BC_OPTIONS = ('extrapolation', 'neumann')
DEFAULT_WALL_CONFORMATION_BC: WallConformationBC = 'extrapolation'

# String tag used in the per-axis ``types`` tuple of
# :class:`ConformationBoundaryConditions` to mark "linear extrapolation
# of the cell-centered field to the wall ghost cell". This is *not* a
# member of ``jax_ib.base.boundaries.BCType`` -- the three string
# constants there (``'periodic'``, ``'dirichlet'``, ``'neumann'``) are
# what the unmodified ``jax_ib._pad`` knows how to handle. The
# extrapolation tag is consumed entirely inside our override.
EXTRAPOLATION = 'extrapolation'


@register_pytree_node_class
class ConformationBoundaryConditions(boundaries.ConstantBoundaryConditions):
    """Constant-style BC that additionally supports a ``'extrapolation'`` tag.

    On any axis pair tagged ``EXTRAPOLATION`` (per side), :meth:`_pad`
    fills ghost cells with a **linear extrapolation in Psi** from the
    two interior cells nearest the wall:

        Psi(ghost at index ``-k``,  k>=1) = (k + 1) . Psi(0)   - k . Psi(1)
        Psi(ghost at index ``N+k-1``)    = (k + 1) . Psi(N-1) - k . Psi(N-2)

    Equivalently, fit a straight line through the two cell-center
    values closest to the wall and continue it past the wall. This is
    the default for the cell-centered log-conformation field at
    grid-aligned walls (and for any future constitutive memory field
    that lives on the same manifold). Two
    properties that matter:

      * **SPD-safe.** ``A = exp(Psi)`` is SPD for any real symmetric
        ``Psi``, so a linear extrapolation of ``Psi`` can never produce a
        ghost-cell ``A`` with a negative eigenvalue -- the failure mode
        that makes ``A``-space extrapolation fragile.
      * **Unbounded ghost-cell width.** Unlike ``BCType.NEUMANN`` in
        ``jax_ib`` (which raises ``ValueError: Padding past 1 ghost
        cell is not defined in neumann case.`` for ``|width| > 1``), the
        extrapolation formula is well-defined for any positive integer
        ``k``. (In practice the van-Leer limiter only calls
        ``c.shift(+/-1)``, so even Neumann's 1-ghost-cell limit ought
        to suffice; the unbounded width is bookkeeping insurance,
        not a hard requirement of the advection.)

    The class is otherwise identical to
    :class:`boundaries.ConstantBoundaryConditions` -- same
    ``__init__`` signature, same ``tree_flatten`` / ``tree_unflatten``
    inherited from the parent, same ``_trim_padding`` /
    ``pad_and_impose_bc`` machinery. The single override is
    ``_pad``, which dispatches the extrapolation case and forwards
    everything else (PERIODIC, DIRICHLET, NEUMANN) to ``super()._pad``.
    """

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Round-trip ``ConformationBoundaryConditions`` through a pytree.

        We override the parent's :meth:`tree_unflatten` only so the
        reconstructed instance is a :class:`ConformationBoundaryConditions`
        and not a vanilla :class:`ConstantBoundaryConditions` (which
        wouldn't know the extrapolation tag).
        """
        return cls(*children, *aux_data)

    def _pad(self, u, width, axis):
        """Pad ``u`` along ``axis`` by ``width`` cells.

        Dispatches the extrapolation case; everything else (PERIODIC,
        DIRICHLET, NEUMANN) is delegated to the parent so existing
        callers see no behavior change on the non-extrapolation axes.

        Assumes ``u.shape == u.grid.shape`` on the padded axis, i.e.
        no pre-existing ghost cells. That holds for every call site we
        actually hit: ``GridVariable.shift`` pads then trims back to
        grid size, ``advect_van_leer_using_limiters`` only ever calls
        ``c.shift(+/-1)`` on grid-sized arrays, and the kernel wraps the
        three Psi components fresh each step. The parent's
        ``_trim_padding`` dance is only relevant if a caller chains
        ``_pad`` calls without intervening trims, which we don't do --
        and reproducing that dance with an alternate ghost-cell formula
        complicates the proof without adding capability.
        """
        bc_type = self.types[axis][0] if width < 0 else self.types[axis][1]
        if bc_type != EXTRAPOLATION:
            return super()._pad(u, width, axis)

        new_offset = list(u.offset)
        new_offset[axis] -= max(-width, 0)

        if width < 0:
            n = -width
            slice0 = lax.slice_in_dim(u.data, 0, 1, axis=axis)
            slice1 = lax.slice_in_dim(u.data, 1, 2, axis=axis)
            # Ghost index ``-k`` (1 <= k <= n) = (k+1).Psi(0) - k.Psi(1).
            # Concatenation order needs the farthest ghost first.
            ghosts = [
                (k + 1) * slice0 - k * slice1
                for k in range(n, 0, -1)
            ]
            new_data = jnp.concatenate(ghosts + [u.data], axis=axis)
        else:
            n = width
            N = u.data.shape[axis]
            sliceN1 = lax.slice_in_dim(u.data, N - 1, N, axis=axis)
            sliceN2 = lax.slice_in_dim(u.data, N - 2, N - 1, axis=axis)
            # Ghost index ``N + k - 1`` (1 <= k <= n) = (k+1).Psi(N-1) - k.Psi(N-2).
            ghosts = [
                (k + 1) * sliceN1 - k * sliceN2
                for k in range(1, n + 1)
            ]
            new_data = jnp.concatenate([u.data] + ghosts, axis=axis)

        return grids.GridArray(new_data, tuple(new_offset), u.grid)


@register_pytree_node_class
class CavityLidBoundaryConditions(boundaries.ConstantBoundaryConditions):
    """All-Dirichlet velocity BC that injects a **spatially-varying lid**.

    The regularized lid-driven-cavity forcing
    ``u(x) = 16 U (x/L)^2 (1 - x/L)^2`` is a
    function of ``x`` along the top ``vx`` face, not a constant. The stock
    :class:`boundaries.ConstantBoundaryConditions` cannot carry it: its
    Dirichlet ``_pad`` fills ghost cells with
    ``jnp.pad(..., constant_values=self.bc_values)``, and ``jnp.pad``'s
    ``constant_values`` must be scalar -- the scalar RHS-lift cannot
    carry an array.

    This subclass carries a 1-D ``lid_profile`` array (one value per
    ``vx`` x-face) as an extra pytree child and overrides ``_pad`` for the
    **top (``y``-high) face only**, where ``vx`` is cell-centered in ``y``
    (offset ``0.5``). For that one case it forms the inhomogeneous
    Dirichlet ghost row

        ghost_top(x) = 2 . lid_profile(x) - u_interior_top(x)

    by taking the *homogeneous* reflection from the parent (with the
    stored scalar top value kept at ``0``) and adding ``2 . lid_profile``
    to the newly-padded top ghost row(s). Every other face (the two side
    walls and the bottom wall, all ``vx = 0``, and every ``vy`` face) is
    delegated verbatim to ``super()._pad``.

    Why this is the robust mechanism (a ``_pad`` override rather than
    strong assignment): the lid value enters
    the solver *only* through ``u.bc._pad`` -- the wall BC-lift
    (``fully_implicit_rheology_stepper`` Step 4 builds
    ``div_nu_symgrad_vector`` of a zero-interior field carrying this BC),
    the Step-7/9 rewrap, and the pressure projection's ``impose_bc`` all
    call it. Overriding ``_pad`` therefore propagates the profile
    everywhere with **no change to the stepper, solver, or projection**.
    It also inherits SPD-irrelevant, scalar-shape-safe bookkeeping from
    the parent for the three homogeneous faces.

    Assumes (like :class:`ConformationBoundaryConditions`) that the padded
    ``vx`` array is grid-sized on the padded axis and that the van-Leer /
    finite-difference stack only ever calls ``shift(+/-1)`` (width ``+/-1``);
    that is the only call pattern the cavity driver hits.
    """

    def __init__(self, time_stamp, values, types, boundary_fn, lid_profile):
        super().__init__(time_stamp, values, types, boundary_fn)
        object.__setattr__(self, 'lid_profile', lid_profile)

    def tree_flatten(self):
        children = (self.time_stamp, self.bc_values, self.lid_profile)
        aux_data = (self.types, self.boundary_fn)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        time_stamp, bc_values, lid_profile = children
        types, boundary_fn = aux_data
        return cls(time_stamp, bc_values, types, boundary_fn, lid_profile)

    def _pad(self, u, width, axis):
        # Only the top (y-high) cell-centered face carries the lid; every
        # other face is homogeneous Dirichlet and goes to the parent.
        is_top_lid = (axis == 1 and width > 0
                      and np.isclose(u.offset[axis] % 1, 0.5))
        padded = super()._pad(u, width, axis)
        if not is_top_lid:
            return padded
        # Parent used the stored scalar top value (kept 0) -> the top
        # `width` ghost rows currently hold the homogeneous reflection
        # (-u_interior). Add 2*lid_profile to convert to the inhomogeneous
        # Dirichlet ghost 2*lid - u_interior.
        w = int(width)
        lid = jnp.asarray(self.lid_profile)
        add = 2.0 * lid[:, None]  # (nx, 1) broadcasts over the w ghost rows
        new_data = padded.data.at[:, -w:].add(add)
        return grids.GridArray(new_data, padded.offset, padded.grid)


@register_pytree_node_class
class ContractionBoundaryConditions(boundaries.ConstantBoundaryConditions):
    """Constant BC that supports a *mixed* edge-aligned axis (Dirichlet on
    one face, Neumann on the other) without corrupting the staggered-grid
    shape bookkeeping.

    Why this is needed: the 4:1 contraction's
    streamwise velocity is Dirichlet at the inlet (low x-face) and Neumann
    (zero-gradient) at the outlet (high x-face). The parent
    :meth:`ConstantBoundaryConditions.pad_and_impose_bc` assumes that if a
    face-aligned axis is Dirichlet on its *lower* face it is Dirichlet on
    *both* faces (the periodic-x / wall-y channel never violated this), so
    it pads the array back to ``N+1`` along that axis. For the mixed
    inlet/outlet axis the outlet is a real DOF and nothing was trimmed, so
    that pad produces an inconsistent shape (``add got incompatible
    shapes``) inside ``fd.laplacian`` / ``fd.divergence``.

    The fix: treat a mixed (non-two-sided-Dirichlet) edge axis as
    "all grid points present" -- exactly the Neumann convention -- and let
    the per-face ``_pad`` (inherited, unchanged) supply the inlet Dirichlet
    ghost and the outlet Neumann ghost during ``shift``. Two-sided
    Dirichlet axes (the no-slip walls in y) keep the parent's trim/pad
    behavior verbatim, so the wall handling is identical to the channel.

    Only :meth:`pad_and_impose_bc` is overridden; ``_pad``, ``_trim``,
    ``trim_boundary`` (which already keys its trim on
    ``types[axis][1] == DIRICHLET`` per face) are inherited unchanged.
    """

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

    def pad_and_impose_bc(self, u, offset_to_pad_to=None):
        if offset_to_pad_to is None:
            offset_to_pad_to = u.offset
        for axis in range(u.grid.ndim):
            lo, hi = self.types[axis]
            both_dirichlet = (lo == boundaries.BCType.DIRICHLET and
                              hi == boundaries.BCType.DIRICHLET)
            # Only the two-sided-Dirichlet edge axis gets the parent's
            # trim/pad restoration; mixed (e.g. Dirichlet/Neumann) edge axes
            # are left at grid size (all points present).
            if both_dirichlet and np.isclose(u.offset[axis], 1.0):
                if np.isclose(offset_to_pad_to[axis], 1.0):
                    u = self._pad(u, 1, axis)
                elif np.isclose(offset_to_pad_to[axis], 0.0):
                    u = self._pad(u, -1, axis)
        return grids.GridVariable(u, self)


def _conformation_bc_with_axes(ndim: int,
                                 wall_axes,
                                 wall_axis_pair,
                                 non_wall_axis_pair):
    """Build a ConformationBoundaryConditions with the given per-axis types.

    ``wall_axes`` are the axes that get ``wall_axis_pair``; the rest get
    ``non_wall_axis_pair``. Used by :func:`create_conformation_bc` to
    factor periodic-vs-wall axis logic into one place.
    """
    type_list = []
    for axis in range(ndim):
        if axis in wall_axes:
            type_list.append(wall_axis_pair)
        else:
            type_list.append(non_wall_axis_pair)
    types = tuple(type_list)
    values = ((0.0, 0.0),) * ndim
    return ConformationBoundaryConditions(
        time_stamp=0.0,
        values=values,
        types=types,
        boundary_fn=lambda t: 0.0,
    )


def validate_wall_conformation_bc(option: str) -> WallConformationBC:
    """Check that ``option`` is one of the supported wall-conformation BCs.

    Raises ``ValueError`` on anything else. Returns the option string
    unchanged.
    """
    if option not in WALL_CONFORMATION_BC_OPTIONS:
        raise ValueError(
            f"Unknown wall_conformation_bc={option!r}. "
            f"Choose from {WALL_CONFORMATION_BC_OPTIONS}."
        )
    return option  # type: ignore[return-value]


def create_conformation_bc(grid,
                           boundary_type: str = 'periodic',
                           wall_conformation_bc: WallConformationBC
                                   = DEFAULT_WALL_CONFORMATION_BC,
                           wall_axes: 'tuple[int, ...] | None' = None):
    """Build a BC for one component of a cell-centered memory field.

    Periodic case: returns the corrected periodic BC used by the
    velocity solver, which is also the right choice for the
    conformation field in periodic domains.

    Wall-bounded case:

      * ``wall_conformation_bc='extrapolation'`` (the default) returns
        a :class:`ConformationBoundaryConditions` with the new
        ``EXTRAPOLATION`` tag on the wall axes and ``PERIODIC`` on the
        rest. ``_pad`` then fills ghost cells by linear extrapolation
        in ``Psi`` from the two cells nearest the wall -- the established
        RheoTool default, and SPD-safe because ``A = exp(Psi)`` is SPD
        for any real symmetric ``Psi``.
      * ``wall_conformation_bc='neumann'`` returns a
        :class:`ConformationBoundaryConditions` with
        ``BCType.NEUMANN`` on the wall axes. Useful as a debug /
        comparison baseline. ``BCType.NEUMANN`` in ``jax_ib`` supports
        only ``|width| = 1`` of padding; the van-Leer limiter only
        ever does ``c.shift(+/-1)``, so this is fine for stage 2 even
        with the 1-ghost-cell limit.

    Construction note (matters for ``jax.lax.scan``):
        ``HomogeneousBoundaryConditions`` overrides ``__init__`` to take
        only the ``types`` tuple, but inherits its parent's
        ``tree_unflatten`` (which assumes the 4-argument
        ``ConstantBoundaryConditions`` constructor). Round-tripping a
        ``HomogeneousBoundaryConditions`` through pytree
        flatten/unflatten -- which ``jax.lax.scan`` does on every step
        for the carry -- raises ``TypeError``. We therefore build a
        :class:`ConformationBoundaryConditions` directly (which inherits
        the 4-argument ``ConstantBoundaryConditions`` constructor and
        adds the matching ``tree_unflatten``); it carries the same
        physical content but round-trips cleanly. The same pattern is
        used by :func:`create_bc_ndim` and by ``jax_ib``'s
        :func:`Moving_wall_boundary_conditions`.

    Args:
        grid: A :class:`jax_ib.base.grids.Grid`.
        boundary_type: ``'periodic'`` or any non-periodic tag
            (e.g. ``'moving_wall'``, ``'channel'``).
        wall_conformation_bc: Either ``'extrapolation'`` (default) or
            ``'neumann'``. Ignored when ``boundary_type='periodic'``.
        wall_axes: Which axes are wall-bounded. ``None`` (default)
            means "all axes are walls" -- consistent with how
            ``Far_field_boundary_conditions`` treats every axis as
            Dirichlet. For ``boundary_type='moving_wall'`` on a 2D grid
            the conventional choice is ``wall_axes=(1,)``: periodic
            in x, walls in y, matching ``Moving_wall_boundary_conditions``.
    """
    validate_wall_conformation_bc(wall_conformation_bc)
    if boundary_type == 'periodic':
        return create_bc_ndim(grid.ndim)

    ndim = grid.ndim
    if wall_axes is None:
        wall_axes = tuple(range(ndim))

    non_wall_pair = (boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC)
    if wall_conformation_bc == 'extrapolation':
        wall_pair = (EXTRAPOLATION, EXTRAPOLATION)
    else:  # 'neumann'
        wall_pair = (boundaries.BCType.NEUMANN, boundaries.BCType.NEUMANN)

    return _conformation_bc_with_axes(
        ndim=ndim,
        wall_axes=wall_axes,
        wall_axis_pair=wall_pair,
        non_wall_axis_pair=non_wall_pair,
    )


# ---------------------------------------------------------------------------
# BC-spec: per-field, per-axis BC *types* as data
# ---------------------------------------------------------------------------
#
# Historically the matrix-free velocity operator hard-coded its boundary
# type inline (``x-periodic, y-Dirichlet`` for the moving-wall family;
# all-periodic for the periodic family -- see ``pressure.py``). To support
# the 4:1 contraction (``x``: Dirichlet inlet / Neumann outlet; ``y``:
# Dirichlet walls) the boundary type must become **data passed into** the
# operator. :class:`BCSpec` is that data: a small, static (frozen,
# hashable) record of per-axis, per-face BC *types* for each field.
#
# Design notes:
#   * Only BC *types* live here. Nonzero BC *values* (e.g. the inlet
#     velocity ``U(t)``) are lifted onto the solver RHS by the stepper
#     (``equations_rheology.fully_implicit_rheology_stepper`` Step 4), so
#     the matrix-free operator always runs with the *homogeneous* version
#     of the BC. That is why :meth:`velocity_operator_bc` returns a
#     ``HomogeneousBoundaryConditions``.
#   * The **default** specs reproduce the historical hard-coded
#     behavior exactly:
#       - :meth:`BCSpec.channel`  -> x-periodic, y-Dirichlet  (moving wall)
#       - :meth:`BCSpec.periodic` -> all-periodic
#     so a call site passing ``bc_spec=None`` and a call site passing the
#     matching default ``BCSpec`` produce the same operator BCs.
#   * :meth:`pressure_bc` is for the contraction Poisson solve (Neumann
#     inlet, Dirichlet outlet). Existing channel / periodic paths still
#     derive the pressure BC from the velocity field.

BCTypePair = Tuple[str, str]
BCTypeTuple = Tuple[BCTypePair, ...]


def _zero_bc_fn(t):
  """Module-level zero boundary function.

  Used as the ``boundary_fn`` for BC objects that are **rebuilt every step
  inside ``jax.lax.scan``** (e.g. the pressure BC produced by
  :meth:`BCSpec.pressure_bc` in the projection). ``boundary_fn`` is carried
  as pytree *aux_data*; ``jax.lax.scan`` compares the carry treedef across
  iterations by equality, and two distinct ``lambda`` objects are never
  equal -- so a fresh lambda each step raises a carry-structure mismatch.
  A single module-level function has stable identity and avoids that.
  """
  del t
  return 0.0


@dataclasses.dataclass(frozen=True)
class BCSpec:
    """Static, per-field boundary-condition type specification.

    Each ``*_types`` field is a tuple of ``(low_face, high_face)`` pairs,
    one pair per spatial axis, holding ``jax_ib.base.boundaries.BCType``
    string constants (``'periodic'``, ``'dirichlet'``, ``'neumann'``) or,
    for conformation, the ``EXTRAPOLATION`` tag.

    Frozen + tuple-valued so the spec is hashable and can be captured as a
    static Python closure constant by the (traced) matrix-free operators
    without becoming a JAX tracer.
    """

    velocity_types: BCTypeTuple
    pressure_types: BCTypeTuple
    conformation_types: Optional[BCTypeTuple] = None

    # -- field-specific BC builders --------------------------------------
    def velocity_operator_bc(self, grid):
        """BC wrapped around the matrix-free velocity operator's ghost cells.

        Homogeneous (values = 0) because nonzero BC values are RHS-lifted
        by the stepper. The all-periodic case returns the corrected
        periodic :func:`create_bc` object so it is byte-identical to the
        legacy periodic operator path.
        """
        P = boundaries.BCType.PERIODIC
        if all(pair == (P, P) for pair in self.velocity_types):
            return create_bc(grid)
        # A mixed edge axis (different type per face, e.g. the contraction's
        # Dirichlet-inlet / Neumann-outlet x-axis) needs the shape-safe
        # ContractionBoundaryConditions. The pure two-sided-Dirichlet
        # channel keeps the legacy HomogeneousBoundaryConditions object so
        # the moving-wall default is unchanged.
        has_mixed_face = any(pair[0] != pair[1] for pair in self.velocity_types)
        ndim = len(self.velocity_types)
        if has_mixed_face:
            zero_values = ((0.0, 0.0),) * ndim
            return ContractionBoundaryConditions(
                0.0, zero_values, self.velocity_types, _zero_bc_fn)
        return boundaries.HomogeneousBoundaryConditions(self.velocity_types)

    def pressure_bc(self, grid):
        """Pressure BC for the projection / Poisson solve.

        Built as a :class:`boundaries.ConstantBoundaryConditions` with
        zero values. Channel / periodic paths still derive the pressure
        BC from the velocity field; this is used by the contraction
        Poisson solver.
        """
        del grid
        ndim = len(self.pressure_types)
        values = ((0.0, 0.0),) * ndim
        return boundaries.ConstantBoundaryConditions(
            time_stamp=0.0, values=values, types=self.pressure_types,
            boundary_fn=_zero_bc_fn)

    def conformation_bc(self, grid):
        """Conformation/stress field BC.

        Honors the ``EXTRAPOLATION`` tag via
        :class:`ConformationBoundaryConditions`.
        """
        del grid
        if self.conformation_types is None:
            raise ValueError("BCSpec has no conformation_types set.")
        ndim = len(self.conformation_types)
        values = ((0.0, 0.0),) * ndim
        return ConformationBoundaryConditions(
            time_stamp=0.0, values=values, types=self.conformation_types,
            boundary_fn=_zero_bc_fn)

    # -- default factories (must reproduce legacy hard-coded BCs) --------
    @classmethod
    def periodic(cls, ndim: int) -> 'BCSpec':
        """All-periodic spec -- the legacy fully-periodic default."""
        P = boundaries.BCType.PERIODIC
        per = ((P, P),) * ndim
        return cls(velocity_types=per, pressure_types=per,
                   conformation_types=per)

    @classmethod
    def channel(cls, ndim: int, wall_axes: 'tuple[int, ...] | None' = None
                ) -> 'BCSpec':
        """x-periodic, y-Dirichlet spec -- the legacy moving-wall default.

        ``wall_axes`` default (``None``) means "every axis except 0 is a
        wall", matching ``Moving_wall_boundary_conditions`` on a 2-D grid
        (periodic in x, no-slip walls in y). On wall axes: velocity
        Dirichlet, pressure Neumann, conformation extrapolation; on the
        remaining (periodic) axes everything is periodic.
        """
        if wall_axes is None:
            wall_axes = tuple(ax for ax in range(ndim) if ax != 0)
        P = boundaries.BCType.PERIODIC
        D = boundaries.BCType.DIRICHLET
        N = boundaries.BCType.NEUMANN
        vt, pt, ct = [], [], []
        for ax in range(ndim):
            if ax in wall_axes:
                vt.append((D, D)); pt.append((N, N)); ct.append((EXTRAPOLATION,
                                                                 EXTRAPOLATION))
            else:
                vt.append((P, P)); pt.append((P, P)); ct.append((P, P))
        return cls(velocity_types=tuple(vt), pressure_types=tuple(pt),
                   conformation_types=tuple(ct))

    @classmethod
    def cavity(cls, ndim: int = 2) -> 'BCSpec':
        """Square lid-driven cavity spec.

        A closed box with no-slip Dirichlet velocity on **all four**
        walls (the top wall's tangential value is the lid; it is set on
        the velocity field's own BC, not here). Mapped per field:

          * velocity      x: (Dirichlet, Dirichlet)  y: (Dirichlet, Dirichlet)
          * pressure      x: (Neumann,   Neumann)    y: (Neumann,   Neumann)
          * conformation  x: (extrap,    extrap)     y: (extrap,    extrap)

        Two consequences that distinguish the cavity from the
        contraction:

          * **Singular pressure Poisson.** All-Dirichlet velocity =>
            all-Neumann pressure => the discrete pressure operator has a
            constant null space (pressure defined up to an additive
            constant). Nothing pins it (there is no Dirichlet outlet as
            in :meth:`contraction`), so the cavity pressure solve must
            use the pseudoinverse / mean-subtraction path
            (:func:`pressure.solve_fast_diag_cavity`).
          * **No mixed per-face axis.** Every velocity axis is
            two-sided Dirichlet, so :meth:`velocity_operator_bc` returns
            a plain ``HomogeneousBoundaryConditions(((D,D),(D,D)))`` --
            the cavity must *not* take the ``ContractionBoundaryConditions``
            (mixed-face) branch.

        Conformation is ``EXTRAPOLATION`` on every wall: unlike the
        contraction there is no inflow face, so there is no Dirichlet
        ``A = I`` boundary -- every wall is a no-inflow wall and gets the
        SPD-safe linear-Psi extrapolation.
        """
        if ndim != 2:
            raise NotImplementedError(
                "BCSpec.cavity is only defined for the 2-D square cavity; "
                f"got ndim={ndim}.")
        D = boundaries.BCType.DIRICHLET
        N = boundaries.BCType.NEUMANN
        velocity_types = ((D, D), (D, D))
        pressure_types = ((N, N), (N, N))
        conformation_types = ((EXTRAPOLATION, EXTRAPOLATION),
                              (EXTRAPOLATION, EXTRAPOLATION))
        return cls(velocity_types=velocity_types,
                   pressure_types=pressure_types,
                   conformation_types=conformation_types)

    @classmethod
    def contraction(cls, ndim: int = 2) -> 'BCSpec':
        """4:1 planar contraction spec.

        Axis 0 (``x``, streamwise): inlet (low face) Dirichlet, outlet
        (high face) Neumann / zero-gradient. Axis 1 (``y``): no-slip walls
        (Dirichlet). Mapped per field:

          * velocity   x: (Dirichlet inlet, Neumann outlet)  y: (Dir, Dir)
          * pressure   x: (Neumann inlet,  Dirichlet outlet=0) y: (Neu, Neu)
          * conform.   x: (Dirichlet inlet=I, zero-grad outlet) y: (extrap)

        Note the pressure outlet is the **only** Dirichlet face on the
        pressure field -- it pins the Poisson nullspace (``p=0`` at the
        outlet), which is why the contraction Poisson solve is
        non-singular. BC *values* (inlet ``U``, etc.) are not stored here;
        they live on the velocity field's own BC and are RHS-lifted.
        """
        if ndim != 2:
            raise NotImplementedError(
                "BCSpec.contraction is only defined for the 2-D planar "
                f"contraction; got ndim={ndim}.")
        P = boundaries.BCType.PERIODIC  # noqa: F841 (kept for symmetry)
        D = boundaries.BCType.DIRICHLET
        N = boundaries.BCType.NEUMANN
        velocity_types = ((D, N), (D, D))
        pressure_types = ((N, D), (N, N))
        # Conformation outlet is EXTRAPOLATION rather than NEUMANN: the
        # log-conf van-Leer TVD scheme shifts by 2 cells, and jax_ib's
        # Neumann _pad only supports 1 ghost cell, whereas EXTRAPOLATION
        # (the unbounded-width tag handled by ConformationBoundaryConditions)
        # supports any width. Physically they coincide at a developed
        # outlet (zero-gradient == linear extrapolation). Inlet stays
        # Dirichlet (A=I -> Psi=0, tau=0).
        conformation_types = ((D, EXTRAPOLATION), (EXTRAPOLATION, EXTRAPOLATION))
        return cls(velocity_types=velocity_types,
                   pressure_types=pressure_types,
                   conformation_types=conformation_types)
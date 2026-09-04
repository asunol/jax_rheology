"""Assemble the starting state of a run: grid, boundary conditions, initial fields.

One place to answer "what does this geometry start from": the grid, the
velocity and pressure boundary conditions, the initial velocity and pressure,
and, for a memory model, the initial memory fields at rest (the conformation
tensor at the identity). The layout of those memory fields is read from the
model's registry record, so adding a constitutive model does not change this
module.
"""
import sys
from typing import Any, Optional, Tuple

import jax.numpy as jnp
from jax_ib.base import grids, boundaries
from jax_rheology.core import boundaries as bnew
from jax_rheology.models import registry as cr

def create_grid(domain_size, domain):
    """Creates a Grid object."""
    return grids.Grid(domain_size, domain=domain)

def get_boundary_conditions(domain, boundary_type="periodic", amp_shear=0.0, freq_osc=0.0,
                            stepper_type: str = None, solver_type: str = None):
    """Defines boundary conditions based on domain and parameters.

    If fully-implicit (stepper_type == 'fully_implicit'), use corrected periodic BCs
    for periodic domains to avoid zero-flow in iterative solvers.
    """

    
    if boundary_type == "periodic":
        # Use corrected periodic BCs only for fully implicit schemes (Test 1B)
        if stepper_type == 'fully_implicit':
            ndim = 2 if hasattr(domain, "__len__") else 2
            bc = bnew.create_bc_ndim(ndim)
            velocity_bc = (bc, bc)
            return velocity_bc
        def Boundary_fn(t):
            return 0.0
        
        #-- CFD Boundary condition parameters
        freq=[0.]*4
        A=[0.]*4
        B=[0.]*4
        C=[0.]*4
        D=[0.]*4

        bc_fns = [Boundary_fn for a,b,c,d,f in zip(A,B,C,D,freq)]
        vx_bc=((bc_fns[0](0.0), bc_fns[1](0.0)), (bc_fns[2](0.0),bc_fns[3](0.0)))
        vy_bc=((0.0, 0.0), (0, 0.0))

        #velocity_bc = (boundaries.new_periodic_boundary_conditions(ndim=2,bc_vals=vx_bc,bc_fn=bc_fns,time_stamp=0.0),
        #           boundaries.new_periodic_boundary_conditions(ndim=2,bc_vals=vy_bc,bc_fn=bc_fns,time_stamp=0.0))

        ## Notice that I have changed the boundary conditions for the fluid velocity to specify
        velocity_bc = (boundaries.new_periodic_boundary_conditions(ndim = 2, bc_vals = vx_bc, bc_fn = bc_fns, time_stamp = 0.0),
                       boundaries.new_periodic_boundary_conditions(ndim = 2, bc_vals = vy_bc, bc_fn = bc_fns, time_stamp = 0.0))

    else:
        def boundary_fn(t):
            return 0.0

        def boundary_fn_shear(t):
            shear_rate = amp_shear * jnp.cos(jnp.pi * t * freq_osc)
            return shear_rate

        # For x-component: [left, right, bottom, top]
        bc_fns_x = [boundary_fn, boundary_fn, boundary_fn, boundary_fn_shear]
        # For y-component: All zero
        bc_fns_y = [boundary_fn, boundary_fn, boundary_fn, boundary_fn]

        # Initialize with correct values at t=0
        vx_bc = ((bc_fns_x[0](0.0), bc_fns_x[1](0.0)), (bc_fns_x[2](0.0), bc_fns_x[3](0.0)))
        vy_bc = ((0.0, 0.0), (0.0, 0.0))
        # Create with Moving_wall_boundary_conditions for time-dependent BCs
        velocity_bc = (
            boundaries.Moving_wall_boundary_conditions(ndim=2, bc_vals=vx_bc, bc_fn=bc_fns_x, time_stamp=0.0),
            boundaries.Moving_wall_boundary_conditions(ndim=2, bc_vals=vy_bc, bc_fn=bc_fns_y, time_stamp=0.0)
        )
    
    return velocity_bc

def get_initial_velocity(grid, boundary_type='periodic', amp_shear=0.0, freq_osc=0.0,
                         background_flow=0.0, stepper_type: str = None, solver_type: str = None):
    """Defines initial velocity field."""

    vx_fn = lambda x, y: jnp.zeros_like(x + y)  # Zero initial velocity
    
    vy_fn = lambda x, y: jnp.zeros_like(x + y)  # Always zero y-velocity initially
    
    velocity_fns = (vx_fn, vy_fn)
    
    # Create GridArray objects for velocity
    v0 = tuple(grid.eval_on_mesh(v_fn, offset) for v_fn, offset in zip(velocity_fns, grid.cell_faces))
    
    # Get boundary conditions (use correct type)
    if boundary_type == 'periodic':
        velocity_bc = get_boundary_conditions(grid.domain, 'periodic', amp_shear, freq_osc,
                                              stepper_type=stepper_type, solver_type=solver_type)
    else:
        velocity_bc = get_boundary_conditions(grid.domain, 'moving_wall', amp_shear, freq_osc,
                                              stepper_type=stepper_type, solver_type=solver_type)
    
    # Create GridVariable objects with proper boundary conditions
    v0 = tuple(
        grids.GridVariable(u, bc) 
        for u, bc in zip(v0, velocity_bc)
    )
    
    return v0

def get_initial_pressure(grid, v0):
    """Defines initial pressure field."""
    pressure0 = grids.GridVariable(
        grids.GridArray(jnp.zeros_like(v0[0].data), grid.cell_center, grid),
        boundaries.get_pressure_bc_from_velocity(v0)
    )
    return pressure0


def _rest_state_for_manifold(manifold: str,
                             components: Tuple[str, ...]) -> Tuple[float, ...]:
    """Return per-component rest-state values for the given manifold.

    ``'spd'`` is the identity tensor (diagonals 1, off-diagonals 0) --
    the rest state every log-conformation model starts from. The other
    manifold tags are accepted in the type system but raise here until
    a registered model actually needs them.
    """
    if manifold == 'spd':
        # Identity tensor: diagonals -> 1.0, off-diagonals -> 0.0.
        return tuple(1.0 if (c and c[0] == c[-1] and c[0] != '') else 0.0
                     for c in components)
    if manifold in ('symmetric', 'traceless_symmetric',
                    'unconstrained', 'scalar'):
        raise NotImplementedError(
            f"Rest state for manifold={manifold!r} is not implemented yet; "
            "no model needing this manifold has been registered. The "
            "this is filled in when a model needs it; "
            "extension."
        )
    raise ValueError(f"Unknown manifold tag {manifold!r}.")


def get_initial_memory(grid,
                       state_spec: Optional[cr.StateSpec],
                       bc: Optional[Any] = None,
                       ) -> Optional[Tuple[grids.GridVariable, ...]]:
    """Build the initial ``memory_fields`` tuple for a constitutive model.

    Iterates over the :class:`FieldSpec`\\ s in ``state_spec``;
    for each one, creates one :class:`GridVariable` per component at
    the field's declared offset, populated with the manifold-tagged
    rest state. The components of one field are concatenated into the
    same flat output tuple as the components of every other field, in
    declaration order -- exactly the layout that
    ``All_Variables.memory_fields`` carries.

    Returns ``None`` if ``state_spec`` is ``None`` or empty (no memory
    fields). Non-empty specs use the manifold rest state (``'spd'`` ->
    identity).

    Args:
        grid: A :class:`jax_ib.base.grids.Grid` (2D in scope).
        state_spec: Either ``None`` / empty tuple (no memory fields)
            or a :class:`StateSpec` describing the fields.
        bc: Boundary condition applied uniformly to every component of
            every field. ``None`` means use the corrected periodic BC
            (``boundaries.create_bc_ndim(grid.ndim)``).

    Returns:
        ``None`` for the empty case; otherwise a flat tuple of
        :class:`GridVariable`\\ s with ``len`` equal to the total
        component count across all fields.
    """
    if state_spec is None or len(state_spec) == 0:
        return None

    # Shared BC for every component. None means "use corrected periodic"
    # (the historical default). Wall-bounded conformation fields pass
    # `bc=` explicitly.
    if bc is None:
        bc = bnew.create_bc_ndim(grid.ndim)

    components: list = []
    for field in state_spec:
        rest_values = _rest_state_for_manifold(field.manifold,
                                               field.components)
        offset = field.offset
        for comp_label, rest_val in zip(field.components, rest_values):
            del comp_label  # only the rest value matters here
            # Use jnp default dtype so this matches the rest of the
            # solver (float32 by default; float64 if x64 is enabled).
            data = jnp.full(grid.shape, float(rest_val))
            arr = grids.GridArray(data, offset, grid)
            components.append(grids.GridVariable(arr, bc))
    return tuple(components)

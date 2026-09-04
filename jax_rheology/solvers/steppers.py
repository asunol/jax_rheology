"""Time steppers: advance velocity, and the memory fields when there are any.

One step of the momentum equation, in the variants the campaigns need:

* :func:`semi_implicit_rheology_stepper` -- explicit rheological stress with an
  implicit Newtonian part, for the generalized-Newtonian runs.
* :func:`fully_implicit_rheology_stepper` -- the variable-viscosity operator
  solved implicitly, which is what the stiff shear-thinning cases need.
* :func:`memory_be_imex_stepper` -- backward-Euler IMEX for the viscoelastic
  runs: the memory fields advance through the constitutive model's own
  evolution function, the polymer stress enters the momentum equation as a
  body force, and velocity is then projected to be divergence free.

The memory steppers are model-agnostic: what the memory fields are and how
they evolve comes from the model's registry record, so a new constitutive
model needs no change here.
"""

import functools
from typing import Callable, Optional
import dataclasses

import jax
import jax.numpy as jnp

from jax_ib.base import advection
from jax_ib.base import diffusion
from jax_ib.base import grids
from jax_ib.base import pressure
from jax_rheology.solvers import pressure as pressure_new
from jax_rheology.solvers.pressure import create_jacobi_preconditioner_fn, create_final_jacobi_preconditioner_fn, create_helmholtz_preconditioner_fn
from jax_cfd.base import pressure as pressureCFD
from jax_ib.base import time_stepping
from jax_ib.base import boundaries
from jax_ib.base import finite_differences
import tree_math
from jax_rheology.core import state as particle_class
from jax_cfd.base import equations as equationsCFD

GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
ConvectFn = Callable[[GridVariableVector], GridArrayVector]
DiffuseFn = Callable[[GridVariable, float], GridArray]
ForcingFn = Callable[[GridVariableVector], GridArrayVector]
BCFn =  Callable[[particle_class.All_Variables, float], particle_class.All_Variables]
BCFn_new =  Callable[[GridVariableVector, float], GridVariableVector]
IBMFn =  Callable[[particle_class.All_Variables, float], GridVariableVector]
GradPFn = Callable[[GridVariable], GridArrayVector]

PosFn =  Callable[[particle_class.All_Variables, float], particle_class.All_Variables]

DragFn =  Callable[[particle_class.All_Variables], particle_class.All_Variables]


def _wrap_term_as_vector(fun, *, name):
  return tree_math.unwrap(jax.named_call(fun, name=name), vector_argnums=0)


def viscoelastic_step(all_vars: particle_class.All_Variables,
                      model,
                      params,
                      dt: float) -> particle_class.All_Variables:
    """Advance the constitutive-model memory by one step.

    Model-agnostic; the body has no Oldroyd-B-specific code. Every
    rheological-model-with-memory ships its own ``evolution_fn`` (the
    Oldroyd-B logconf one lives in ``jax_rheology.log_conformation``);
    this helper is just the plumbing that calls it on the
    ``memory_fields`` slot of :class:`All_Variables` and threads the
    updated tuple back into a fresh ``All_Variables`` via
    ``dataclasses.replace``.

    Args:
        all_vars: Current solver state. Must carry a populated
            ``memory_fields`` slot (i.e. came out of
            ``get_initial_memory`` or a prior ``viscoelastic_step``).
        model: A :class:`constitutive_registry.ConstitutiveModel`
            record. Only ``model.evolution_fn`` is used here.
        params: Whatever parameter container the model's evolution
            function expects (dict or dataclass -- ``_params_get``
            accepts both).
        dt: Time step.

    Returns:
        A new ``All_Variables`` with updated ``memory_fields``. Every
        other slot -- ``velocity``, ``pressure``, ``particles``,
        ``Drag``, ``Step_count``, ``MD_var``, ``memory_layout`` -- is
        carried through unchanged.
    """
    new_memory_fields = model.evolution_fn(
        all_vars.memory_fields, all_vars.velocity, params, dt)
    return dataclasses.replace(all_vars, memory_fields=new_memory_fields)


def polymer_force_to_faces(tau_components, tau_bc, target_offsets,
                             interp_bc=None):
    """Compute ``div tau`` at each velocity-face offset.

    Helper used by every ``coupling_mode='explicit_force'`` constitutive
    model -- not Oldroyd-B-specific. Same construction the
    ``*_stress_forcing`` factories in :mod:`jax_rheology.models` already
    use for the explicit-stress path: pack the symmetric 2-D stress
    tensor as a :class:`jax_ib.base.finite_differences.GridArrayTensor`
    of :class:`GridVariable`\\ s (so :func:`tensor_divergence` can read
    BCs through ``central_difference``), divide-free the divergence at
    cell centers, then linearly interpolate each component to its
    target face offset.

    The off-diagonal ``tau_xy`` slot is symmetric -- ``T[0][1]`` and
    ``T[1][0]`` are *the same* :class:`GridVariable`, not two parallel
    copies.

    ``tau_bc`` is the BC used to wrap the three cell-centered stress
    components for the differencing pass. **Critically, this must be
    the conformation/stress BC, not the velocity BC** -- wrapping tau
    with the velocity BC is a latent bug on Couette: the velocity BC
    carries ``bc_values = ((0, 0), (0, U_wall))`` and the Dirichlet pad
    therefore set tau's top-ghost cell to ``2.U_wall`` even when the
    interior tau was identically zero, producing a phantom polymer
    force of magnitude O(U_wall/Deltay). In periodic geometries the
    velocity BC has ``bc_values = (None, None)`` everywhere and the
    bug is invisible. On wall-bounded geometries the right BC is the
    conformation field's BC -- the cell-centered Psi-extrapolation --
    which gives ghost values that linearly continue the interior tau
    field.

    ``interp_bc`` is the BC used when interpolating the cell-centered
    ``div tau`` components to their target velocity face offsets. Default
    (``None``) reuses ``tau_bc``; callers that want to interpolate
    using the velocity BC explicitly (e.g. to match an existing
    Newtonian forcing path's offset-interpolation semantics) can pass
    it through.

    Args:
        tau_components: A 3-tuple ``(tau_xx, tau_xy, tau_yy)`` of cell-centered
            :class:`grids.GridArray`\\ s -- the output of any
            ``model.stress_readout_fn``.
        tau_bc: A :class:`jax_ib.base.boundaries.BoundaryConditions`
            object for the stress field. For Oldroyd-B this is the
            same Psi-extrapolation BC carried on
            ``all_vars.memory_fields[i].bc``.
        target_offsets: A sequence of velocity-face offsets, one per
            momentum component, e.g. ``[(0.0, 0.5), (0.5, 0.0)]`` for
            a standard 2-D staggered grid.
        interp_bc: Optional BC for the interpolation-to-face step;
            defaults to ``tau_bc``.

    Returns:
        A tuple of :class:`grids.GridArray`\\ s, one per momentum
        component, each living at the corresponding ``target_offsets``
        entry. These are the per-cell *forces* ``(div tau)_i`` -- caller
        is responsible for any ``/ rho`` normalisation.
    """
    from jax_rheology.models import tensor_divergence

    if interp_bc is None:
        interp_bc = tau_bc

    tau_xx_arr, tau_xy_arr, tau_yy_arr = tau_components

    tau_xx_var = grids.GridVariable(tau_xx_arr, tau_bc)
    tau_xy_var = grids.GridVariable(tau_xy_arr, tau_bc)
    tau_yy_var = grids.GridVariable(tau_yy_arr, tau_bc)

    tau_tensor = finite_differences.GridArrayTensor([
        [tau_xx_var, tau_xy_var],
        [tau_xy_var, tau_yy_var],
    ])

    div_cc = tensor_divergence(tau_tensor)

    from jax_ib.base import interpolation as ib_interpolation
    return tuple(
        ib_interpolation.linear(grids.GridVariable(div_arr, interp_bc), offset).array
        for div_arr, offset in zip(div_cc, target_offsets)
    )


def navier_stokes_explicit_terms(
    density: float,
    viscosity: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,
    diffuse: DiffuseFn = diffusion.diffuse,
    forcing: Optional[ForcingFn] = None,
    
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Returns a function that performs a time step of Navier Stokes."""
  del grid  # unused

  if convect is None:
    def convect(v):  # pylint: disable=function-redefined
      return tuple(
          advection.advect_van_leer_using_limiters(u, v, dt) for u in v)

  def diffuse_velocity(v, *args):
    return tuple(diffuse(u, *args) for u in v)

  convection = _wrap_term_as_vector(convect, name='convection')
  diffusion_ = _wrap_term_as_vector(diffuse_velocity, name='diffusion')
  if forcing is not None:
    forcing = _wrap_term_as_vector(forcing, name='forcing')

  @tree_math.wrap
  @functools.partial(jax.named_call, name='navier_stokes_momentum')
  def _explicit_terms(v):
    dv_dt = convection(v)
    if viscosity is not None and viscosity > 0:
      dv_dt += diffusion_(v, viscosity / density)
    if forcing is not None:
      dv_dt += forcing(v) / density
    
    return dv_dt

  def explicit_terms_with_same_bcs(v):
    dv_dt = _explicit_terms(v)
    return tuple(grids.GridVariable(a, u.bc) for a, u in zip(dv_dt, v))

  return explicit_terms_with_same_bcs






def explicit_Reserve_BC(
    ReserveBC: BCFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:

   def Reserve_boundary(v, *args):
    return ReserveBC(v, *args)
   Reserve_bc_ = _wrap_term_as_vector(Reserve_boundary, name='Reserve_BC')
   
   @tree_math.wrap
  # @functools.partial(jax.named_call, name='master_BC_fn')
   def _Reserve_bc(v):
       
       return Reserve_bc_(v,step_time)

   return _Reserve_bc

def explicit_update_BC(
    updateBC: BCFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:

   def Update_boundary(v, *args):
    return updateBC(v, *args)
   Update_bc_ = _wrap_term_as_vector(Update_boundary, name='Update_BC')
   
   @tree_math.wrap
  # @functools.partial(jax.named_call, name='master_BC_fn')
   def _Update_bc(v):
       
       return Update_bc_(v,step_time)

   return _Update_bc


def explicit_IBM_Force(
    cal_IBM_force: IBMFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:

   def IBM_FORCE(v, *args):
    return cal_IBM_force(v, *args)
   IBM_FORCE_ = _wrap_term_as_vector(IBM_FORCE, name='IBM_FORCE')
   
   @tree_math.wrap
  # @functools.partial(jax.named_call, name='master_BC_fn')
   def _IBM_FORCE(v):
       
       return IBM_FORCE_(v,step_time)

   return _IBM_FORCE



def explicit_Update_position(
    cal_Update_Position: PosFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:

   def Update_Position(v, *args):
    return cal_Update_Position(v, *args)
   Update_Position_ = _wrap_term_as_vector(Update_Position, name='Update_Position')
   
   @tree_math.wrap
  # @functools.partial(jax.named_call, name='master_BC_fn')
   def _Update_Position(v):
       
       return Update_Position_(v,step_time)

   return _Update_Position


def explicit_Calc_Drag(
    cal_Drag: DragFn ,
    step_time: float,
) -> Callable[[GridVariableVector], GridVariableVector]:

   def Calculate_Drag(v, *args):
    return cal_Drag(v, *args)
   Calculate_Drag_ = _wrap_term_as_vector(Calculate_Drag, name='Calculate_Drag')
   
   @tree_math.wrap
  # @functools.partial(jax.named_call, name='master_BC_fn')
   def _Calculate_Drag(v):
       
       return Calculate_Drag_(v,step_time)

   return _Calculate_Drag

def explicit_Pressure_Gradient(
    cal_Pressure_Grad: GradPFn,
) -> Callable[[GridVariableVector], GridVariableVector]:

   def Pressure_Grad(v):
    return cal_Pressure_Grad(v)
   Pressure_Grad_ = _wrap_term_as_vector(Pressure_Grad, name='Pressure_Grad')
   
   @tree_math.wrap
  # @functools.partial(jax.named_call, name='master_BC_fn')
   def _Pressure_Grad(v):
       
       return Pressure_Grad_(v)

   return _Pressure_Grad

def semi_implicit_rheology_stepper(
    density: float,
    viscosity: float,
    nu0: float,
    dt: float,
    grid: grids.Grid,
    convect: Optional[ConvectFn] = None,
    forcing: Optional[ForcingFn] = None,
    pressure_solve: Callable = pressure_new.solve_fast_diag,
    helmholtz_solve: Callable = pressure_new.solve_helmholtz_fast_diag,
) -> Callable[[particle_class.All_Variables], particle_class.All_Variables]:
    """Returns a function that performs a single IMEX time step for rheology."""

    explicit_terms_fn = navier_stokes_explicit_terms(
        density=density,
        viscosity=viscosity,
        dt=dt,
        grid=grid,
        convect=convect,
        diffuse=diffusion.diffuse,
        forcing=forcing,
    )

    pressure_projection = jax.named_call(
        pressure_new.projection_and_update_pressure, name='pressure_projection'
    )
    
    named_helmholtz_solver = jax.named_call(
        helmholtz_solve, name='helmholtz_solve'
    )

    @jax.named_call
    def _imex_step(all_vars: particle_class.All_Variables) -> particle_class.All_Variables:
        """Performs one step of the IMEX scheme."""
        v = all_vars.velocity

        # Step 1: Calculate explicit forcing terms
        explicit_update = explicit_terms_fn(v)

        # Step 2: Form the right-hand side of the Helmholtz equation
        # RHS = un + Deltat * F(un)
        rhs = tuple(
            grids.GridArray(u.data + dt * du_dt.data, u.offset, u.grid)
            for u, du_dt in zip(v, explicit_update)
        )

        # Step 3: Solve the Helmholtz equation for intermediate velocity u*
        # (I - Deltat * nu0 * grad^2)u* = RHS
        intermediate_v_array = tuple(
            named_helmholtz_solver(r, alpha=1.0, beta=nu0 * dt) for r in rhs
        )
        
        # Convert the solver output to GridVariables.
        # The spectral Helmholtz solver for wall boundaries is designed to
        # implicitly satisfy the Dirichlet (no-slip) conditions.
        intermediate_v = tuple(
            grids.GridVariable(u_arr, v_comp.bc)
            for u_arr, v_comp in zip(intermediate_v_array, v)
        )

        # Step 4: Project to get the final divergence-free velocity
        temp_state = dataclasses.replace(all_vars, velocity=intermediate_v)
        projected_state = pressure_projection(temp_state, pressure_solve)
        
        # Step 5: Enforce boundary conditions ONCE on the final velocity.
        # This is standard practice after a projection step.
        final_velocity = tuple(u.impose_bc() for u in projected_state.velocity)
        final_state = dataclasses.replace(projected_state, velocity=final_velocity)
        
        # COMMENTED OUT: Previous aggressive boundary condition enforcement
        # This was likely fighting the solver rather than helping it.
        # The code below was an attempt to fix NaN accumulation issues but
        # may have been introducing numerical noise instead.
        
        # # CRITICAL FIX: Robust boundary condition enforcement
        # # First pass: Create GridVariables and impose boundary conditions
        # intermediate_v = tuple(
        #     grids.GridVariable(u_arr, v_comp.bc).impose_bc()
        #     for u_arr, v_comp in zip(intermediate_v_array, v)
        # )
        # 
        # # Second pass: Double-enforce boundary conditions for wall cases
        # # This prevents accumulation of boundary errors over timesteps
        # intermediate_v = tuple(u.impose_bc() for u in intermediate_v)
        #
        #
        # # CRITICAL: Explicit zero enforcement for wall boundaries
        # # This ensures wall boundaries are exactly zero, preventing NaN accumulation
        # if hasattr(intermediate_v[0].bc, 'types'):
        #     # Check if we have wall boundaries (non-periodic)
        #     has_wall_bc = any(
        #         hasattr(u.bc, 'types') and any(
        #             bc_type != 'PERIODIC' for row in u.bc.types for bc_type in row
        #         ) for u in intermediate_v if hasattr(u.bc, 'types')
        #     )
        #     
        #     if has_wall_bc:
        #         # Zero-enforce wall boundaries explicitly
        #         enforced_v = []
        #         for u in intermediate_v:
        #             u_data = u.array.data
        #             # Bottom wall (y=0): set to zero
        #             u_data = u_data.at[:, 0].set(0.0)
        #             # Top wall (y=-1): set to boundary value (usually zero for vy, flow value for vx)
        #             if len(u_data.shape) == 2:  # 2D case
        #                 u_data = u_data.at[:, -1].set(0.0)  # Conservative: set to zero
        #             
        #             # Create new GridVariable with zero-enforced data
        #             u_enforced = grids.GridVariable(
        #                 grids.GridArray(u_data, u.array.offset, u.array.grid), u.bc
        #             ).impose_bc()
        #             enforced_v.append(u_enforced)
        #         
        #         intermediate_v = tuple(enforced_v)
        #
        # # Step 4: Project to get the final divergence-free velocity
        # temp_state = dataclasses.replace(all_vars, velocity=intermediate_v)
        # projected_state = pressure_projection(temp_state, pressure_solve)
        # 
        # # Step 5: Enforce boundary conditions on the final velocity (triple enforcement)
        # final_velocity = tuple(u.impose_bc() for u in projected_state.velocity)
        # # Additional enforcement to ensure corner stability
        # final_velocity = tuple(u.impose_bc() for u in final_velocity)
        # final_state = dataclasses.replace(projected_state, velocity=final_velocity)
        
        return final_state

    return _imex_step


def fully_implicit_rheology_stepper(
    density: float,
    viscosity: float,  # Not used, but kept for API compatibility
    dt: float,
    grid: grids.Grid,
    model_type: str,
    params,  # Model parameters
    model=None,  # Required for TBNN
    convect: Optional[ConvectFn] = None,
    forcing: Optional[ForcingFn] = None,
    add_tbnn_residual: bool = False,
    pressure_solve: Callable = pressure_new.solve_fast_diag,
    solver_type: str = 'gmres',  # Solver type: 'gmres' (default) or 'cg' for comparison
    pressure_gradient: Optional[list] = None,  # Add pressure gradient parameter
    permeability: float = 0.0,  # Add permeability parameter
    U_f: float = 0.0,  # Add background flow parameter
    use_preconditioner: bool = True,  # Enable/disable preconditioner
    preconditioner_type: str = 'jacobi',  # Type of preconditioner ('jacobi', 'final_jacobi', or 'helmholtz')
    polymer_rate_fn: Optional[Callable[
        [particle_class.All_Variables], 'tuple[grids.GridArray, ...]'
    ]] = None,  # Optional explicit polymer force / rho at velocity face offsets
    solver_tol: Optional[float] = None,  # Override BiCGSTAB / GMRES / CG tol (None = solver's own default)
    solver_maxiter: Optional[int] = None,  # Override iterative-solver maxiter (None = solver's own default)
    bc_spec=None,  # Optional BCSpec for the wall operator (None = legacy x-periodic, y-Dirichlet)
    devss_viscosity: float = 0.0,  # BSD numerical viscosity (0 = off; opt-in only)
    grad_div_gamma: float = 1.0,  # Grad-div stabilizer; 1.0 = Sept27 / paper GNF
    grad_div_bc: str = 'velocity',  # 'velocity' = wrap div with v.bc (Sept27); 'neumann' = periodic+Neumann (memory-era)
) -> Callable[[particle_class.All_Variables], particle_class.All_Variables]:
    """Returns a function that performs a single BE-IMEX time step for rheology.
    
    This implements the fully implicit Backward Euler scheme where the entire
    non-linear viscous term div [nu(x)(gradu+gradu^T)] is treated implicitly using 
    variable-coefficient iterative solvers. GMRES is the default and recommended
    solver as it handles non-symmetric operators correctly, while CG and BiCGSTAB
    are available for comparison/validation purposes.
    
    Args:
        density: Fluid density
        viscosity: Unused (kept for API compatibility with nu0_split version)
        dt: Time step size
        grid: Grid object
        model_type: Rheology model ('power_law', 'carreau_yasuda', 'TBNN', 'newtonian')
        params: Model parameters array
        model: Model object (required for TBNN)
        convect: Convection function
        forcing: Additional forcing function (usually None for BE-IMEX)
        pressure_solve: Pressure projection solver
        solver_type: Solver type ('gmres' for GMRES, 'cg' for Conjugate Gradient, 'bicgstab' for BiCGSTAB)
        pressure_gradient: Pressure gradient vector [px, py, ...] (default: [0.0, 0.0])
        permeability: Permeability field scalar (default: 0.0)
        U_f: Background flow velocity (default: 0.0)
        use_preconditioner: Enable preconditioning (default: True)
        preconditioner_type: Type of preconditioner ('jacobi', 'final_jacobi', or 'helmholtz', default: 'jacobi')
        
    Returns:
        Function that advances one time step using BE-IMEX scheme
    """
    from jax_rheology.models import get_viscosity_field, fully_implicit_forcing#, tbnn_residual_divergence # removed tbnn_residual_divergence from models.py, need to add back in if needed
    
    # Select solver based on solver_type parameter. Special case for
    # the constant-viscosity wall-bounded path on bicgstab: route to
    # the per-component variant. The joint (u_x, u_y) flat-vector
    # solver has a BiCGSTAB-on-imbalanced-block-rhs breakdown that
    # bites whenever one component's rhs is much smaller than the
    # other (the case for low-Wi Oldroyd-B variant-A on Couette).
    # For model_type='newtonian' the operator decouples exactly
    # between components (gradnu = 0), so the per-component split is
    # mathematically identical to the joint solve. For variable-nu
    # models the gradnu . gradu cross-term re-couples components and the
    # joint solver is required. See the docstring of
    # ``solve_varvisc_bicgstab_moving_wall_vector_per_component`` in
    # ``pressure.py`` for the full story.
    if solver_type == 'gmres':
        varvisc_solve_wall = pressure_new.solve_varvisc_gmres_moving_wall_vector
        varvisc_solve_periodic = pressure_new.solve_varvisc_gmres_periodic_vector
    elif solver_type == 'cg':
        varvisc_solve_wall = pressure_new.solve_varvisc_cg_moving_wall_vector
        varvisc_solve_periodic = pressure_new.solve_varvisc_cg_periodic_vector
    elif solver_type == 'bicgstab':
        if model_type == 'newtonian':
            varvisc_solve_wall = pressure_new.solve_varvisc_bicgstab_moving_wall_vector_per_component
        else:
            varvisc_solve_wall = pressure_new.solve_varvisc_bicgstab_moving_wall_vector
        varvisc_solve_periodic = pressure_new.solve_varvisc_bicgstab_periodic_vector
    else:
        raise ValueError(f"Unknown solver_type: '{solver_type}'. Choose 'gmres', 'cg', or 'bicgstab'.")

    # Optionally tighten the iterative solver's tol / maxiter when a
    # high-accuracy gradient path needs it. The dispatch above bound
    # module-level functions; here we wrap them with the requested
    # overrides so the rest of the stepper logic is untouched. ``None``
    # keeps the solver's own default (e.g. ``tol=1e-7`` for the wall
    # path, ``tol=1e-12`` for the periodic path) -- the production
    # defaults the rest of the stack is tuned against.
    if solver_tol is not None or solver_maxiter is not None:
        import functools
        _solver_kwargs = {}
        if solver_tol is not None:
            _solver_kwargs['tol'] = solver_tol
        if solver_maxiter is not None:
            _solver_kwargs['maxiter'] = solver_maxiter
        varvisc_solve_wall = functools.partial(
            varvisc_solve_wall, **_solver_kwargs)
        varvisc_solve_periodic = functools.partial(
            varvisc_solve_periodic, **_solver_kwargs)
    
    # Non-viscous forcing function (pressure gradient + permeability penalties only)
    if forcing is None:
        # Default: no additional forcing beyond viscous effects
        def default_forcing(v):
            return tuple(grids.GridArray(jnp.zeros_like(u.data), u.offset, u.grid) for u in v)
        forcing = default_forcing
    
    # Use fully implicit forcing that excludes viscous terms
    # Use the provided parameters or default to zero
    if pressure_gradient is None:
        pressure_gradient = [0.0] * grid.ndim
    
    nonviscous_forcing = fully_implicit_forcing(
        pressure_gradient=pressure_gradient,  # Use actual pressure gradient
        permeability=permeability,  # Use actual permeability
        U_f=U_f  # Use actual background flow
    )
    
    pressure_projection = jax.named_call(
        pressure_new.projection_and_update_pressure, name='pressure_projection'
    )
    
    named_varvisc_wall_solver = jax.named_call(
        varvisc_solve_wall, name='varvisc_wall_solve'
    )
    
    named_varvisc_periodic_solver = jax.named_call(
        varvisc_solve_periodic, name='varvisc_periodic_solve'
    )
    
    # Statically compute the component size once. This happens outside the JIT path.
    import numpy as np
    component_size = np.prod(grid.shape)
    if grad_div_bc not in ('velocity', 'neumann'):
        raise ValueError(
            f"grad_div_bc must be 'velocity' or 'neumann'; got {grad_div_bc!r}")

    @jax.named_call
    def _be_imex_step(
        all_vars: particle_class.All_Variables,
        grad_div_gamma: float = grad_div_gamma,
        grad_div_bc: str = grad_div_bc,
    ) -> particle_class.All_Variables:
        """Performs one step of the BE-IMEX scheme."""
        v = all_vars.velocity
        # Step 1: Compute spatially-varying viscosity field eta(x) from current velocity
        eta_field = get_viscosity_field(v, params, model_type, model=model)
        # Step 2: Convert to kinematic viscosity nu(x) = eta(x)/rho - never mind, might just be normal viscosity
        nu_field = eta_field #/ density
        if devss_viscosity != 0.0:
            nu_field = nu_field + devss_viscosity

        # Step 3: Compute non-viscous forcing terms (all as rates du/dt)
        # This includes convection, pressure gradients, body forces, etc.
        # but excludes viscous stresses (handled implicitly)
        if convect is not None:
            # FIX 1: The convect() function already returns -(u.grad)u.
            # The extra negation made the term positive, causing instability.
            convective_rate = convect(v)
        else:
            # Default: no convection
            convective_rate = tuple(
                grids.GridArray(jnp.zeros_like(u.data), u.offset, u.grid) for u in v
            )
        
        # Additional forcing (non-viscous terms: pressure gradient, permeability, etc.)
        additional_forcing_rate = nonviscous_forcing(v)

        # Optional TBNN residual: div (tau_TBNN - 2 eta_equiv S)
        if add_tbnn_residual and (model_type == 'TBNN'):
            # Use the same eta_field just computed (before division by density)
            # tbnn_residual_divergence returns a vector of GridArray forces
            # residual_force = tbnn_residual_divergence(v, model, params, eta_field)
            residual_force = tuple(
                grids.GridArray(jnp.zeros_like(u.data), u.offset, u.grid) for u in v
            )
        else:
            # Zero residual (shape/BC-preserving)
            residual_force = tuple(
                grids.GridArray(jnp.zeros_like(u.data), u.offset, u.grid) for u in v
            )

        # Optional explicit polymer body force div tau_p / rho at velocity face
        # offsets. Parameterised on the core BE-IMEX stepper so any
        # ``coupling_mode='explicit_force'`` model can use it: the
        # matrix-free implicit operator stays Newtonian-solvent-only and
        # constitutive coupling rides in this same explicit rate slot.
        if polymer_rate_fn is not None:
            polymer_rate = polymer_rate_fn(all_vars)
        else:
            polymer_rate = tuple(
                grids.GridArray(jnp.zeros_like(u.data), u.offset, u.grid) for u in v
            )

        # Combine all non-viscous rates
        total_nonviscous_rates = tuple(
            grids.GridArray(
                conv_rate.data + add_force_rate.data + res_force.data + poly_rate.data,
                conv_rate.offset,
                conv_rate.grid
            )
            for conv_rate, add_force_rate, res_force, poly_rate
            in zip(convective_rate, additional_forcing_rate, residual_force,
                   polymer_rate)
        )
        if devss_viscosity != 0.0:
            bsd_rate = tuple(
                diffusion.diffuse(u_comp, devss_viscosity) for u_comp in v)
            total_nonviscous_rates = tuple(
                grids.GridArray(
                    r.data - b.data, r.offset, r.grid)
                for r, b in zip(total_nonviscous_rates, bsd_rate))

        # Step 5: Detect boundary conditions to choose appropriate solver.
        # ``BCType.PERIODIC`` is the lowercase string ``'periodic'`` in
        # jax_ib (see ``jax_ib/jax_ib/base/boundaries.py:36``), so we
        # compare against that. The earlier upper-case check was a
        # latent bug: it routed every BC (periodic included) to the
        # wall-solver branch. Periodic wraps still need the periodic
        # solver, so we fix the comparison here.
        has_wall_bc = False
        if hasattr(v[0].bc, 'types'):
            has_wall_bc = any(
                any(bc_type != boundaries.BCType.PERIODIC for bc_type in bc_pair)
                for bc_pair in v[0].bc.types
            )

        # Step 4: Form RHS for the implicit-viscous solve.
        #
        # The PERIODIC branch keeps the original "absolute" form,
        #     rhs = v + dt . rates,                                    (A)
        # and the solver returns v^{n+1} directly. The Newtonian-limit
        # path is built around this form.
        #
        # The WALL branch uses the same absolute form **plus an
        # explicit BC-lift on the rhs**:
        #     rhs = v + dt . rates + (dt/rho) . L_BC ,                    (B)
        # where ``L_BC`` is the viscous-stress divergence of a field
        # whose interior is identically zero but which carries v^n's
        # boundary values at the ghost cells. The wall solvers
        # ``solve_varvisc_*_moving_wall_vector`` build their matrix-free
        # operator with ``HomogeneousBoundaryConditions`` baked in
        # (see ``pressure.py:1078-1080``); the operator they realise is
        # the homogeneous part ``A_homo = I - (dt/rho).L_homo``. The full
        # equation ``A v^{n+1} = v + dt.rates`` with v^{n+1} having
        # nonhomogeneous BC decomposes as
        #     A_homo(v^{n+1}) + (dt/rho).L_BC = v + dt.rates
        # because ``L`` is affine in the BC values: L(u) = L_homo(u_int)
        # + L_BC, where L_BC depends only on the BC values, not on
        # ``u_int``. Moving L_BC to the rhs gives (B). The solver
        # returns the interior of v^{n+1} (carrying homogeneous-BC
        # ghosts internally); Step 7 then wraps that interior with
        # v's actual BC, so the GridVariable v^{n+1} has the right
        # ghosts re-imposed via the Dirichlet pad.
        #
        # Why this absolute form and not the increment form Deltav =
        # v^{n+1} - v^n: BiCGSTAB on the joint flat (u_x, u_y) vector
        # breaks down when one of the two component-rhs blocks has a
        # norm comparable to the absolute convergence tolerance -- and
        # the increment-form rhs is precisely that pathological
        # profile, with magnitude ~``dt . viscous_lift`` on one or two
        # boundary rows and ~``dt . polymer_rate`` (often <~ 1e-6) in
        # the bulk. The absolute formulation keeps the rhs dominated
        # by ``v^n`` everywhere, which BiCGSTAB tolerates well; this
        # matches the solver-stress profile in
        # ``tbnn_gradient_debug_constriction_new_piv``, where the
        # whole codebase's bicgstab convention was originally tuned.
        # (See the existing comment in
        # ``solve_varvisc_bicgstab_periodic_vector`` calling out the
        # same ``HomogeneousBoundaryConditions``-kills-the-flow trap
        # the periodic sibling already paved around.)
        if has_wall_bc:
            from jax_rheology.solvers.pressure import div_nu_symgrad_vector
            v_zero_with_bc = tuple(
                grids.GridVariable(
                    grids.GridArray(jnp.zeros_like(u.data), u.offset, u.grid),
                    u.bc,
                )
                for u in v
            )
            L_BC_force = div_nu_symgrad_vector(v_zero_with_bc, nu_field)
            rhs_vector = tuple(
                grids.GridArray(
                    u.data + dt * (nonvisc_rate.data
                                    + L_BC_i.data / density),
                    u.offset,
                    u.grid,
                )
                for u, nonvisc_rate, L_BC_i
                in zip(v, total_nonviscous_rates, L_BC_force)
            )
        else:
            rhs_vector = tuple(
                grids.GridArray(
                    u.data + dt * nonvisc_rate.data,
                    u.offset,
                    u.grid,
                )
                for u, nonvisc_rate in zip(v, total_nonviscous_rates)
            )
        
        # Step 5.5: Conditionally create preconditioner for improved convergence
        # This significantly improves convergence for extreme shear-thinning cases
        preconditioner_fn = None
        if use_preconditioner:
            if preconditioner_type == 'helmholtz':
                # Use the advanced Helmholtz-based preconditioner
                # Calculate nu0 = max(nu(x)) for representative constant viscosity
                #nu0 = jnp.max(nu_field)
                nu0 = jnp.quantile(nu_field, 0.90)
                # Select appropriate Helmholtz solver based on boundary conditions
                if has_wall_bc:
                    helmholtz_solver = pressure_new.solve_helmholtz_fast_diag_moving_wall
                else:
                    helmholtz_solver = pressure_new.solve_helmholtz_fast_diag
                
                # Create Helmholtz preconditioner instance, passing in the static size
                preconditioner_fn = create_helmholtz_preconditioner_fn(
                    grid, nu0, dt, helmholtz_solver, component_size
                )
            elif preconditioner_type == 'final_jacobi':
                # Use the corrected Jacobi preconditioner that properly matches the nugrad^2u operator
                preconditioner_fn = create_final_jacobi_preconditioner_fn(
                    v, nu_field, dt, density
                )
            else:  # Default to 'jacobi'
                # Use the simple Jacobi preconditioner (original version)
                preconditioner_fn = create_jacobi_preconditioner_fn(
                    grid, nu_field, dt, density
                )
        
        # Step 6: Solve variable-coefficient system
        # (I - Deltat/rho * div [nu(x)(gradu+gradu^T)]) u* = RHS. Both branches now
        # use the absolute formulation (rhs encodes v + dt.rates [+
        # BC-lift on the wall branch]); the solver returns the
        # interior of v^{n+1}, and Step 7 rewraps it with v's bc to
        # re-impose the ghosts.
        if has_wall_bc:
            intermediate_v_array = named_varvisc_wall_solver(
                rhs_vector, nu_field, dt, density,
                preconditioner_fn=preconditioner_fn,
                bc_spec=bc_spec,
            )
        else:
            intermediate_v_array = named_varvisc_periodic_solver(
                rhs_vector, nu_field, dt, density,
                preconditioner_fn=preconditioner_fn
            )

        # Step 7: Convert solver output to GridVariables with proper boundary conditions
        intermediate_v_raw = tuple(
            grids.GridVariable(u_arr, v_comp.bc)
            for u_arr, v_comp in zip(intermediate_v_array, v)
        )

        # Step 7.5: Apply Grad-Div stabilization to damp pressure oscillations
        # v* <- v* - dt*gamma*grad(div(v*)). Term is mathematically zero on a
        # converged solve; meant to damp high-frequency modes invisible to the
        # pressure solver.
        #
        # Two corrections that matter in practice:
        #   1. gamma can be 0.0. The fast-diag pressure solver used in
        #      this codebase delivers machine-precision divergence-free output
        #      on its own (see step 8), so the stabilization is unnecessary
        #      and, when enabled with a wrong BC, was a latent bug.
        #   2. The cell-centered divergence scalar must NOT be wrapped with
        #      the velocity Dirichlet BC. Doing so injects the wall velocity
        #      (e.g. U_wall=1.0) as a phantom "divergence value" into the top
        #      ghost cell of div(v*), producing grad(div) ~ U_wall/dy at the
        #      wall and a spurious force dt*gamma*U_wall/dy per step that
        #      pushes the velocity off the analytic Couette profile and
        #      eventually NaNs out the conformation field. The principled BC
        #      is homogeneous Neumann on the y-walls and periodic in x, since
        #      div(v) is a cell-centered scalar with no Dirichlet wall data
        #      (same lesson as the polymer-stress BC fix above).
        # Factory default is gamma=1.0 / velocity-BC wrap so the
        # instantaneous (GNF) entrypoints reproduce the paper. Memory
        # steppers pass grad_div_gamma=0.0, grad_div_bc='neumann'.
        gamma = grad_div_gamma
        div_u_star = finite_differences.divergence(intermediate_v_raw)
        if grad_div_bc == 'velocity':
            div_u_var = grids.GridVariable(div_u_star, intermediate_v_raw[0].bc)
        else:
            div_u_bc = boundaries.periodic_and_neumann_boundary_conditions()
            div_u_var = grids.GridVariable(div_u_star, div_u_bc)
        grad_div_correction = finite_differences.gradient_tensor(div_u_var)

        intermediate_v = tuple(
            grids.GridVariable(
                u.array - dt * gamma * grad_div.data,
                u.bc
            )
            for u, grad_div in zip(intermediate_v_raw, grad_div_correction)
        )
        
        # Step 8: Apply pressure projection to enforce incompressibility.
        # bc_spec threads the contraction pressure BC through (None =
        # legacy velocity-derived pressure BC).
        temp_state = dataclasses.replace(all_vars, velocity=intermediate_v)
        projected_state = pressure_projection(temp_state, pressure_solve, bc_spec)
        
        # Step 9: Final boundary condition enforcement
        # This is standard practice after projection to ensure no slip at walls
        final_velocity = tuple(u.impose_bc() for u in projected_state.velocity)
        final_state = dataclasses.replace(projected_state, velocity=final_velocity)
        
        return final_state
    
    return _be_imex_step


def memory_be_imex_stepper(
    density: float,
    dt: float,
    grid: grids.Grid,
    model,
    params,
    base_viscosity: float,
    convect: Optional[ConvectFn] = None,
    pressure_solve: Callable = pressure_new.solve_fast_diag,
    solver_type: str = 'gmres',
    pressure_gradient: Optional[list] = None,
    permeability: float = 0.0,
    U_f: float = 0.0,
    use_preconditioner: bool = True,
    preconditioner_type: str = 'jacobi',
    solver_tol: Optional[float] = None,
    solver_maxiter: Optional[int] = None,
    bc_spec=None,
    devss_viscosity: float = 0.0,
) -> Callable[[particle_class.All_Variables], particle_class.All_Variables]:
    """BE-IMEX stepper for any constitutive model with explicit-force coupling.

    Explicit-force coupling (polymer stress as a body force; the
    implicit operator stays Newtonian-solvent-only). One outer step
    does:

      1. ``viscoelastic_step(all_vars, model, params, dt)`` -- advance
         ``memory_fields`` using the registered ``model.evolution_fn``.
         The velocity field is the source of ``gradu`` here, and is read
         but not modified by this stage.
      2. ``model.stress_readout_fn(new_memory_fields, velocity, params)``
         -- compute the polymer stress ``(tau_xx, tau_xy, tau_yy)`` at cell
         centers from the freshly-updated memory.
      3. :func:`polymer_force_to_faces` -- ``div tau`` interpolated to each
         velocity face offset.
      4. Divide by ``density`` to obtain the polymer rate at face
         offsets.
      5. Run the BE-IMEX velocity step with this rate folded into
         ``total_nonviscous_rates``. The matrix-free implicit operator
         still sees only the **Newtonian solvent viscosity**
         ``base_viscosity`` -- variant a's defining property is that the
         polymer term is explicit and the implicit operator's
         preconditioner / spectrum is unchanged from the Newtonian
         baseline.

    No mention of Oldroyd-B, ``A``, log-conformation, or SPD-ness
    anywhere in this body -- the whole constitutive choice is captured
    by the ``model`` record and its registered function pointers.

    Args:
        density: Fluid density ``rho``.
        dt: Time step.
        grid: The simulation :class:`grids.Grid`.
        model: A :class:`constitutive_registry.ConstitutiveModel`
            record. ``model.coupling_mode`` must be
            ``'explicit_force'``. Implicit-block coupling
            (``'implicit_block'``) is not implemented here.
        params: Parameter container threaded into ``model.evolution_fn``
            *and* ``model.stress_readout_fn``. For Oldroyd-B that
            carries ``lam`` and ``Gp``.
        base_viscosity: Newtonian solvent viscosity ``nu_s`` for the
            implicit operator. In the Oldroyd-B context this is the
            solvent viscosity (a constant); polymer contributions never
            enter the implicit operator in this variant.
        convect: Convection rate function ``v -> -(u.grad)u`` already in
            the library convention; default matches ``forward_simulation``
            (``advect_upwind``).
        pressure_solve: Pressure projection solver (defaults to fast-diag
            periodic; pass the moving-wall variant for wall BCs).
        solver_type, pressure_gradient, permeability, U_f,
        use_preconditioner, preconditioner_type: Forwarded to
            :func:`fully_implicit_rheology_stepper` and treated exactly
            as in the existing ``stepper_type='fully_implicit'`` path --
            no special handling required. Grad-div is pinned to the
            memory-era values ``grad_div_gamma=0.0``,
            ``grad_div_bc='neumann'`` (paper instantaneous GNF uses the
            factory defaults ``1.0`` / ``'velocity'``).

    Returns:
        A callable ``step_fn(all_vars) -> all_vars`` that advances one
        outer step, JIT- and AD-safe (the only new ingredient relative
        to the existing BE-IMEX path is the closure that calls
        ``viscoelastic_step`` and then the polymer-rate hook; both are
        pure-JAX and ride entirely inside the existing
        ``@jax.checkpoint`` boundary in ``forward_simulation``).
    """
    if getattr(model, 'coupling_mode', None) != 'explicit_force':
        raise NotImplementedError(
            f"memory_be_imex_stepper only supports coupling_mode='explicit_force'; "
            f"got {getattr(model, 'coupling_mode', None)!r}. Variant b "
            "(implicit_block) is not implemented."
        )

    def _polymer_rate_fn(all_vars: particle_class.All_Variables):
        """Polymer ``div tau / rho`` at velocity face offsets, read off ``all_vars``.

        Called from inside ``_be_imex_step`` on the *post-viscoelastic-step*
        ``all_vars``, so ``all_vars.memory_fields`` already reflects the
        new conformation (memory first, then momentum).

        The BC threaded into :func:`polymer_force_to_faces` is the
        **conformation field's** BC (carried on
        ``all_vars.memory_fields[0].bc``), not the velocity BC. See
        the docstring of ``polymer_force_to_faces`` for why this
        matters on wall-bounded geometries -- the velocity BC carries
        the wall **velocity** value (e.g. ``U_wall`` for Couette),
        which the Dirichlet pad would impose on tau as a phantom wall
        stress. The conformation BC (Psi-extrapolation by default) is
        the physically right closure for the stress ghost cells.
        """
        tau_components = model.stress_readout_fn(
            all_vars.memory_fields, all_vars.velocity, params)
        tau_bc = all_vars.memory_fields[0].bc
        target_offsets = [u.offset for u in all_vars.velocity]
        polymer_face_force = polymer_force_to_faces(
            tau_components, tau_bc, target_offsets)
        return tuple(
            grids.GridArray(force.data / density, force.offset, force.grid)
            for force in polymer_face_force
        )

    be_imex_step = fully_implicit_rheology_stepper(
        density=density,
        viscosity=base_viscosity,
        dt=dt,
        grid=grid,
        model_type='newtonian',
        params=base_viscosity,
        model=None,
        convect=convect,
        forcing=None,
        add_tbnn_residual=False,
        pressure_solve=pressure_solve,
        solver_type=solver_type,
        pressure_gradient=pressure_gradient,
        permeability=permeability,
        U_f=U_f,
        use_preconditioner=use_preconditioner,
        preconditioner_type=preconditioner_type,
        polymer_rate_fn=_polymer_rate_fn,
        solver_tol=solver_tol,
        solver_maxiter=solver_maxiter,
        bc_spec=bc_spec,
        devss_viscosity=devss_viscosity,
        grad_div_gamma=0.0,
        grad_div_bc='neumann',
    )

    @jax.named_call
    def _memory_be_imex_step(all_vars: particle_class.All_Variables
                              ) -> particle_class.All_Variables:
        all_vars_after_visc = viscoelastic_step(all_vars, model, params, dt)
        return be_imex_step(all_vars_after_visc)

    return _memory_be_imex_step

"""Forward driver for the generalized-Newtonian and instantaneous-closure runs.

Advances a velocity field with the rheology steppers over an outer/inner step
schedule and returns the trajectory. Takes the constitutive model as a name
plus parameters, so the same driver serves the closed-form laws and a learned
closure, and the parameters can be a flat array for differentiation or the
original pytree.
"""

import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import jax_ib.base as ib  # Make sure this import is present
from jax_ib.base import advection
from jax_ib.base import convolution_functions, particle_motion, IBM_Force
from jax_rheology.core import state as pc
import jax_ib.penalty.util_funs  # Import this module specifically for permeability calculation
from jax_rheology.core import flow_conditions
from jax_rheology import models
from jax_rheology.core import params as parameter_utils
from jax_rheology.solvers import pressure  #  Corrected import
from jax_rheology.solvers import steppers as equations_rheology  #  Add this import
from jax_rheology.models import registry as constitutive_registry
from jax_rheology.core import boundaries as bnew
import numpy as np
from typing import Callable
from jax_cfd.base.funcutils import scan

def _identity(x):
    return x

def _velocity_profile(x):
    # return x.velocity  # <- current
    u, v = x.velocity
    return (u.data, v.data)  # (H, W) arrays; scan will stack to (T, H, W)


def forward_fluid_simulation(flow_cond, flattened_params, particles, stress_forcing_fn, model, 
                          nu0_update_fn: Callable,
                          nu0_baseline=None, trajectory_ref=None, starts_static=None, ends_static=None,
                          helmholtz_solver_type='fast_diag', stepper_type: str = 'nu0_split',
                          solver_type: str = 'bicgstab', use_preconditioner: bool = False, preconditioner_type: str = 'jacobi',
                          initial_state=None, initial_nu0=None, bc_spec=None,
                          permeability=None, devss_viscosity: float = 0.0):
    """
    Forward simulation function with dictionary-based flow conditions.
    
    Args:
        flow_cond: Dictionary containing flow conditions
        flattened_params: Flattened parameter array
        particles: Particle configuration
        stress_forcing_fn: Function to compute stress forcing
        model: TBNN model
        nu0_baseline: Baseline implicit viscosity nu0 (if None, use flow_cond value)
        trajectory_ref: Reference trajectory (optional)
        starts_static: List of start indices for parameter blocks (optional)
        ends_static: List of end indices for parameter blocks (optional)
        helmholtz_solver_type: Type of Helmholtz solver for wall boundaries ('fast_diag' or 'cg')
        stepper_type: Type of time stepper ('nu0_split' or 'fully_implicit')
        solver_type: Type of variable-coefficient solver ('gmres', 'cg', or 'bicgstab', default: 'gmres')
        use_preconditioner: Enable/disable preconditioner for fully_implicit stepper (default: True)
        preconditioner_type: Type of preconditioner ('jacobi', 'final_jacobi', or 'helmholtz', default: 'jacobi')
        initial_state: Optional initial state to start simulation from (default: None, creates new state)
        initial_nu0: Optional initial nu0 value (default: None, computes from nu0_update_fn)
    
    Returns:
        Tuple of (final_result, trajectory, perm_f)
    """
    # Extract flow conditions from dictionary
    density = flow_cond.get('density', 1.0)
    base_viscosity = flow_cond.get('base_viscosity', flow_cond.get('viscosity', 0.0))
    pressure_gradient = flow_cond.get('pressure_gradient', 0.0)
    dt = flow_cond.get('dt', 1e-5)
    U_f = flow_cond.get('U_f', 0.0)
    grid = flow_cond.get('grid')
    tree_def = flow_cond.get('tree_def')
    shapes = flow_cond.get('shapes')
    inner_steps = flow_cond.get('inner_steps', 300)
    outer_steps = flow_cond.get('outer_steps', 200)
    amp_shear = flow_cond.get('amp_shear', 0.0)
    freq_osc = flow_cond.get('freq_osc', 0.0)
    
    boundary_type = flow_cond.get('boundary_type', 'periodic')
    background_flow = flow_cond.get('U_f', 0.0)
    
    # Use provided nu0_baseline or flow_cond value
    if nu0_baseline is None:
        nu0_baseline = flow_cond.get('nu0_baseline', 0.0)
    
    # Use provided starts_static/ends_static or from flow_cond
    if starts_static is None:
        starts_static = flow_cond.get('starts_static')
    if ends_static is None:
        ends_static = flow_cond.get('ends_static')
    
    params = flattened_params # will replace for tbnn
    # Handle structured-params unflattening (TBNN MLP, or a plain
    # scalar dict like {'Gp', 'lam', 'nu_s'}).
    # Older guard that required ``model is not None`` (kept commented):
    # if model is not None and flattened_params is not None and tree_def is not None and shapes is not None:
    if flattened_params is not None and tree_def is not None and shapes is not None:
        # Slice sizes must be static. Computing them with jnp.prod makes the
        # index a traced scalar, which cannot be sliced at trace time; wrapping
        # that in a bare except would hide the failure and let params fall back
        # to the raw flat array, surfacing later as a missing-key error in the
        # model code rather than here.
        if starts_static is not None and ends_static is not None:
            params = parameter_utils.unflatten_params(
                flattened_params, tree_def, shapes,
                starts=starts_static, ends=ends_static)
        else:
            idx = 0
            leaves = []
            for shape in shapes:
                # Plain-Python size so the slice index stays static and
                # ``flattened_params[idx:idx+size]`` works at trace time.
                size = int(np.prod(shape)) if shape else 1
                leaves.append(flattened_params[idx:idx+size].reshape(shape))
                idx += size
            params = jax.tree_util.tree_unflatten(tree_def, leaves)
    
    # Setup permeability
    w = 0.0015  # Width of logistic smoothing
    K = 20000   # Logistic scaling factor
    def logistjax(G, K, w):
        return K * jax.scipy.special.expit(G / w)
    
    smoothening_fn = lambda G, K: logistjax(G, K, w)
    
    # Calculate permeability
    if particles is not None:
        try:
            perm_f = jax_ib.penalty.util_funs.perm_vmap_multiple_particles(grid, particles, smoothening_fn, K)
        except Exception as e:
            print(f"Warning: Permeability calculation failed: {e}")
            perm_f = 0.0
    else:
        perm_f = 0.0

    # Precomputed-permeability override: the 4:1 contraction supplies a
    # box-SDF penalty field directly rather than via particle shapes, so
    # callers pass it in. None keeps the particle-derived behavior.
    if permeability is not None:
        perm_f = permeability
    
    # Choose appropriate pressure and Helmholtz solvers based on boundary conditions
    if boundary_type == "periodic":
        pressure_solver = pressure.solve_fast_diag
        helmholtz_solver_fn = pressure.solve_helmholtz_fast_diag
    elif boundary_type == "contraction":
        # 4:1 contraction: Neumann inlet / Dirichlet outlet(p=0) in x,
        # Neumann walls in y. bc_spec carries the
        # matching velocity/pressure types; passed by the caller.
        pressure_solver = pressure.solve_fast_diag_contraction
        helmholtz_solver_fn = pressure.solve_helmholtz_fast_diag_moving_wall
    elif boundary_type == "moving_wall":
        pressure_solver = pressure.solve_fast_diag_moving_wall
        # Select Helmholtz solver based on solver type
        if helmholtz_solver_type == 'fast_diag':
            helmholtz_solver_fn = pressure.solve_helmholtz_fast_diag_moving_wall
        elif helmholtz_solver_type == 'cg':
            helmholtz_solver_fn = pressure.solve_helmholtz_cg_moving_wall
        else:
            raise ValueError(f"Unsupported Helmholtz solver type: '{helmholtz_solver_type}'. Use 'fast_diag' or 'cg'")
    else:
        raise ValueError(f"Unsupported boundary type: '{boundary_type}'")
    
    # Define convect function
    def convect(v):
        return tuple(advection.advect_upwind(u, v, dt) for u in v)
    
    # Initial state
    if initial_state is None:
        v0 = flow_conditions.get_initial_velocity(
            grid,
            boundary_type=boundary_type,
            amp_shear=amp_shear,
            freq_osc=freq_osc,
            stepper_type=stepper_type,
            solver_type=solver_type,
        )
        pressure0 = flow_conditions.get_initial_pressure(grid, v0)

        # Create state
        Intermediate_calcs = [0]
        Step_counter = 0
        MD_state = [0]

        # Memory-field initialisation for ``stepper_type='memory'``:
        # when the caller did not pre-build an ``All_Variables``, we
        # look up the constitutive model record by name and ask
        # ``flow_conditions.get_initial_memory`` to build the rest-state
        # tuple -- the same pattern the ``fully_implicit`` branch uses
        # for v0/pressure0. The model's ``state_spec`` rides into
        # ``All_Variables.memory_layout`` as aux_data so any downstream
        # code (e.g. the ``stress_readout_fn`` or a
        # ``polymer_linearization_fn``) can recover field layout without
        # re-introspecting the model.
        memory_fields = None
        memory_layout = None
        if stepper_type == 'memory':
            model_name = flow_cond.get('model_name')
            if model_name is None:
                raise ValueError(
                    "stepper_type='memory' requires flow_cond['model_name'] "
                    "(e.g. 'oldroyd_b_logconf')."
                )
            constitutive_model = constitutive_registry.get_model(model_name)

            # Conformation-field BC selection: for
            # periodic domains the corrected-periodic BC is the only
            # choice; for wall-bounded domains we honor
            # ``flow_cond['wall_conformation_bc']`` (default
            # ``'extrapolation'``) and pass that through
            # :func:`jax_rheology.core.boundaries.create_conformation_bc`. ``wall_axes``
            # follows the convention of ``Moving_wall_boundary_conditions``
            # -- periodic-x, walls-in-y on a 2-D moving-wall geometry.
            wall_axes = None
            if boundary_type == 'moving_wall':
                wall_axes = tuple(
                    ax for ax in range(grid.ndim) if ax != 0
                )
            wall_conformation_bc_opt = flow_cond.get(
                'wall_conformation_bc', bnew.DEFAULT_WALL_CONFORMATION_BC)
            memory_bc = bnew.create_conformation_bc(
                grid,
                boundary_type=boundary_type,
                wall_conformation_bc=wall_conformation_bc_opt,
                wall_axes=wall_axes,
            )
            memory_fields = flow_conditions.get_initial_memory(
                grid, constitutive_model.state_spec, bc=memory_bc)
            memory_layout = constitutive_model.state_spec

        initial_state = pc.All_Variables(
            particles, v0, pressure0,
            Intermediate_calcs, Step_counter, MD_state,
            memory_fields=memory_fields,
            memory_layout=memory_layout,
        )

    # Initial nu0 for IMEX scheme
    if initial_nu0 is None:
        initial_dynamic_nu0 = nu0_update_fn(initial_state.velocity, params)
        initial_nu0 = nu0_baseline + initial_dynamic_nu0

    # Main simulation loop using jax.lax.scan
    if stepper_type == 'nu0_split':
        
        # This function defines one full outer step. It correctly recalculates nu0
        # and rebuilds the inner stepper each time it is called.
        # The checkpoint decorator solves the memory issue.
        @jax.checkpoint
        def outer_step_fn(carry, _):
            current_state, current_nu0 = carry

            # A. Calculate Total nu0 for the IMEX scheme.
            # `new_nu0` represents the total viscosity to be treated implicitly. It's the
            # sum of the baseline Newtonian viscosity and a dynamic component from the
            # non-Newtonian model. The stress forcing functions calculate the full
            # rheological stress and subtract this `new_nu0` part, which is then
            # handled by the implicit Helmholtz solve. The `viscosity=0.0` setting
            # in the stepper below prevents double-counting.
            dynamic_nu0 = nu0_update_fn(current_state.velocity, params)
            proposed_nu0 = nu0_baseline + dynamic_nu0

            # Stability cap: ensure beta * lambda_max < 1 for Helmholtz (alpha - betagrad^2)
            # Accurate lambda_max for mixed boundary conditions
            if boundary_type == "periodic":
                # Pure periodic: lam_max ~= 4 * sum_d (1 / h_d^2)
                inv_h2_sum = sum(1.0 / (h * h) for h in grid.step)
                lambda_max = 4.0 * inv_h2_sum
            elif boundary_type == "moving_wall":
                # Mixed periodic-x, Dirichlet-y: different eigenvalue structure
                hx, hy = grid.step[0], grid.step[1]
                # Max eigenvalue is approximately: lam_x_max + lam_y_max
                # For periodic: lam_x_max ~= 4/hx^2 * sin^2(pi*nx/2nx) ~= 4/hx^2  
                # For Dirichlet: lam_y_max ~= 4/hy^2 * sin^2(pi*ny/2ny) ~= 4/hy^2
                lambda_max = 4.0 / (hx * hx) + 4.0 / (hy * hy)
            else:
                # Fallback to conservative estimate
                inv_h2_sum = sum(1.0 / (h * h) for h in grid.step)
                lambda_max = 4.0 * inv_h2_sum
                
            beta_cap = 0.8  # Slightly more conservative safety margin
            nu0_cap = beta_cap / (dt * lambda_max)
            new_nu0 = jnp.minimum(proposed_nu0, nu0_cap)

            # B. Re-create Forcing Function
            forcing_function = stress_forcing_fn(
                [pressure_gradient, 0.0], perm_f, model, params, 1.0, U_f, new_nu0
            )

            # D. Set Helmholtz Solver from pre-selected function
            helmholtz_solver = helmholtz_solver_fn

            # C. Re-create Time Stepper
            step_fn_inner = cfd.funcutils.repeated(
                equations_rheology.semi_implicit_rheology_stepper(
                    density=density,
                    viscosity=0.0,  # Explicit Newtonian viscosity is now handled by nu0
                    nu0=new_nu0,
                    dt=dt,
                    grid=grid,
                    convect=convect,
                    forcing=forcing_function,
                    pressure_solve=pressure_solver,
                    helmholtz_solve=helmholtz_solver,
                ),
                steps=inner_steps,
            )

            # E. Execute Inner Steps
            final_state = step_fn_inner(current_state)
            
            frame = _velocity_profile(final_state)
            
            return (final_state, new_nu0), frame
    
    elif stepper_type in ('fully_implicit', 'fully_implicit_plus_residual'):
        # Determine model type from stress forcing function name
        stress_fn_name = stress_forcing_fn.__name__ if hasattr(stress_forcing_fn, '__name__') else str(stress_forcing_fn)
        if 'TBNN' in stress_fn_name or 'tbnn' in stress_fn_name: model_type = 'TBNN'
        elif 'power_law' in stress_fn_name or 'power' in stress_fn_name: model_type = 'power_law'
        elif 'carreau' in stress_fn_name or 'yasuda' in stress_fn_name: model_type = 'carreau_yasuda'
        else: model_type = 'newtonian'
        
        # PERFORMANCE FIX: For this stepper, the inner function is constant,
        # so we define it once outside the loop.
        inner_stepper = cfd.funcutils.repeated(
            equations_rheology.fully_implicit_rheology_stepper(
                density=density, viscosity=base_viscosity, dt=dt, grid=grid,
                model_type=model_type, params=params, model=model,
                convect=convect, forcing=None,
                add_tbnn_residual=(stepper_type=='fully_implicit'),
                pressure_solve=pressure_solver,
                solver_type=solver_type, pressure_gradient=[pressure_gradient, 0.0],
                permeability=perm_f, U_f=U_f, use_preconditioner=use_preconditioner,
                preconditioner_type=preconditioner_type,
                bc_spec=bc_spec,
                devss_viscosity=devss_viscosity,
            ),
            steps=inner_steps,
        )

        # CRITICAL FIX: The function that executes the inner loop is checkpointed.
        @jax.checkpoint
        def outer_step_fn(carry, _):
            current_state, nu0_like = carry
            final_state = inner_stepper(current_state)
            frame = _velocity_profile(final_state)
            
            return (final_state, jax.lax.stop_gradient(jnp.zeros_like(nu0_like))), frame

    elif stepper_type == 'memory':
        # Constitutive-model dispatch: the only model-specific string
        # the caller passes is the model name. Polymer parameters live
        # in ``flow_cond['polymer_params']`` or, when present, in the
        # differentiable ``params`` pytree.
        model_name = flow_cond.get('model_name')
        if model_name is None:
            raise ValueError(
                "stepper_type='memory' requires flow_cond['model_name'] "
                "(e.g. 'oldroyd_b_logconf')."
            )
        constitutive_model = constitutive_registry.get_model(model_name)

        # Polymer parameters (Gp, lam) and the Newtonian solvent
        # viscosity (nu_s) flow through the ``params`` pytree so they
        # are differentiable via ``jax.grad`` w.r.t. the same
        # ``flattened_params`` array the adjoint already supports.
        # flow_cond-only path (kept commented):
        # polymer_params = flow_cond.get('polymer_params', {})
        # effective_base_viscosity = base_viscosity
        if isinstance(params, dict) and (
                'Gp' in params or 'lam' in params or 'nu_s' in params):
            fallback_polymer = flow_cond.get('polymer_params', {})
            polymer_params = {
                'Gp':  params.get('Gp',  fallback_polymer.get('Gp')),
                'lam': params.get('lam', fallback_polymer.get('lam')),
            }
            effective_base_viscosity = params.get('nu_s', base_viscosity)
        else:
            polymer_params = flow_cond.get('polymer_params', {})
            effective_base_viscosity = base_viscosity

        inner_stepper = cfd.funcutils.repeated(
            equations_rheology.memory_be_imex_stepper(
                density=density,
                dt=dt,
                grid=grid,
                model=constitutive_model,
                params=polymer_params,
                base_viscosity=effective_base_viscosity,
                convect=convect,
                pressure_solve=pressure_solver,
                solver_type=solver_type,
                pressure_gradient=[pressure_gradient, 0.0],
                permeability=perm_f,
                U_f=U_f,
                use_preconditioner=use_preconditioner,
                preconditioner_type=preconditioner_type,
                bc_spec=bc_spec,
                devss_viscosity=devss_viscosity,
            ),
            steps=inner_steps,
        )

        @jax.checkpoint
        def outer_step_fn(carry, _):
            current_state, nu0_like = carry
            final_state = inner_stepper(current_state)
            frame = _velocity_profile(final_state)
            return (final_state,
                    jax.lax.stop_gradient(jnp.zeros_like(nu0_like))), frame

    else:
        raise ValueError(
            f"Unknown stepper_type: '{stepper_type}'. "
            "Choose 'nu0_split', 'fully_implicit', "
            "'fully_implicit_plus_residual', or 'memory'."
        )

    # Run the simulation using scan
    initial_carry = (initial_state, initial_nu0)
    (final_result, _), trajectory = jax.lax.scan(
        outer_step_fn, initial_carry, xs=None, length=outer_steps
    )

    return final_result, trajectory, perm_f

def trajectory_check(step_fn, steps, post_process=_identity, *, start_with_input=False):
    """Original trajectory check function"""
    @jax.checkpoint
    def step(carry_in, _):
        carry_out = step_fn(carry_in)
        frame = post_process(carry_in if start_with_input else carry_out)
        return carry_out, frame

    def multistep(values):
        return scan(step, values, xs=None, length=steps)

    return multistep
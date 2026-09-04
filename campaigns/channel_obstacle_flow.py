#!/usr/bin/env python
"""
Channel Flow with Obstacle - Multiple Rheological Models

This script simulates channel flow with a circular obstacle in the center,
supporting Newtonian, Power-Law, Carreau-Yasuda, and TBNN rheological models.

USAGE:
Notebook:
    from channel_obstacle_flow import *
    
    # Newtonian flow
    run_channel_obstacle('newtonian', 1.0)
    
    # Power-Law flow  
    run_channel_obstacle('power_law', 0.5, 0.8)  # K=0.5, n=0.8
    
    # Carreau-Yasuda flow
    run_channel_obstacle('carreau_yasuda', 0.02, 0.5, 5.0, 0.8, 2.0)  # eta_inf, eta_0, lambda, n, a
    
    # TBNN flow (initialize fresh)
    run_channel_obstacle('tbnn', 'init', (256, 128), 42)  # 'init', domain_size, random_seed
    
    # TBNN flow (load from file)
    run_channel_obstacle('tbnn', '/path/to/params.pkl', (256, 128), 42)

Command line:
    python channel_obstacle_flow.py newtonian 1.0
    python channel_obstacle_flow.py power_law 0.5 0.8
    python channel_obstacle_flow.py carreau_yasuda 0.02 0.5 5.0 0.8 2.0
    python channel_obstacle_flow.py tbnn init
    python channel_obstacle_flow.py tbnn /path/to/params.pkl --random-seed 42
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
_VERBOSE = ("--verbose" in sys.argv) or bool(os.environ.get("JAX_RHEOLOGY_VERBOSE"))
import time
import argparse

# Set working directory and paths
from repo_paths import bootstrap, REPO_ROOT
bootstrap()
if _VERBOSE:
    print("Working directory:", os.getcwd())

# Import the rheology modules
from jax_rheology.core import flow_conditions
from jax_rheology import models
from jax_rheology.forward import generic as forward_simulation
from jax_rheology.core import params as parameter_utils
from jax_ib.base import particle_class as pc, grids, kinematics as ks
from jax_ib.base import advection, diffusion
from jax_rheology.solvers import steppers as equations_rheology
from jax_rheology.solvers import pressure
import jax_cfd.base as cfd
import jax_ib.penalty.util_funs
import pickle

# Disable JIT for debugging if needed
jax.config.update('jax_disable_jit', False)

if _VERBOSE:
    print("Obstacle channel solver initialized.")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Common configuration for all simulations
COMMON_CONFIG = {
    'domain': ((0, 12.0), (0, 4.0)),  # 8x4 rectangular channel
    'domain_size': (256, 128),  # 128x128 grid
    'density': 1.0,
    'pressure_gradient': 2.5,
    'dt': 1e-4,
    'inner_steps': 200,
    'outer_steps': 300,
    'solver_type': 'bicgstab',  # Always use BiCGSTAB as requested
    'stepper_type': 'fully_implicit'
}

# =============================================================================
# OBSTACLE SETUP FOR CHANNEL FLOW
# =============================================================================

def setup_channel_obstacle(domain):
    """Create a circular obstacle in the center of the channel."""
    def param_rot_ellipse(geometry_param, theta):
        A = geometry_param[0]
        B = geometry_param[1] 
        phi = geometry_param[2]
        excc = jnp.sqrt(1-jnp.round((B/A)**2, 6))
        return B/jnp.sqrt(1-(excc*jnp.cos(theta-phi))**2)
    
    # Place obstacle in the center of the channel
    center_x = (domain[0][1] + domain[0][0]) / 4  # Middle of channel length
    center_y = (domain[1][1] + domain[1][0]) / 2  # Middle of channel height
    radius = 0.4  # Smaller radius to fit nicely in channel
    
    particle_geometry_param = jnp.array([[radius, radius, 0.0]])  # Circular
    particle_center_position = jnp.array([[center_x, center_y]])
    displacement_param = jnp.array([[0.0, 0.0]])
    rotation_param = jnp.array([[0.0, 0.0, 0.0, 0]])
    mygrids = pc.Grid1d(100, domain=(0, 2*jnp.pi))
    
    particles = pc.particle(
        particle_center_position, particle_geometry_param,
        displacement_param, rotation_param, mygrids,
        param_rot_ellipse, ks.displacement, ks.rotation
    )
    
    return particles

# =============================================================================
# TBNN MODEL SETUP
# =============================================================================

# Default TBNN configuration matching the reference settings
DEFAULT_TBNN_SETTINGS = {
    'tbnn_hidden_units': [16],
    'M': 12,
    'eta_init': 1.0,
    'freeze_eta0': True,
    'eta0_fixed': 1.0,
    'eta0_eps': 1e-5,
    'mu_min_gamma': 1e-1,
    'mu_max_gamma': 1e1,
    'gate_gamma': 1e-1,
    'gate_width_z': 0.5,
    's_floor': 0.35,
    'alpha_temp': 0.8,
    'log_head': True,
    'log_mixing': "add",
    'enable_pl_per_mode': False,
    'pl_width_z': 0.5,
    'freeze_centers': False,
    'tail_gate_gamma': None,
    'tail_gate_width_z': 0.5,
}

def setup_tbnn_model_for_channel(domain_size, random_seed=42, **kwargs):
    """Setup TBNN model for channel flow using default settings.
    
    Args:
        domain_size: Grid resolution (nx, ny)
        random_seed: Random seed for initialization
        **kwargs: Override any DEFAULT_TBNN_SETTINGS
    
    Returns:
        Dictionary with TBNN model info
    """
    # Merge defaults with overrides
    settings = DEFAULT_TBNN_SETTINGS.copy()
    settings.update(kwargs)
    
    # Extract settings
    hidden_units = settings['tbnn_hidden_units']
    M = settings['M']
    eta_init = settings['eta_init']
    
    # Set random seed
    key = jax.random.PRNGKey(random_seed)
    
    # Create TBNN configuration
    tbnn_config = {
        'M': M,
        'eta_min': max(eta_init * 0.01, 1e-3),
        'eta_max': eta_init * 10.0,
        'gamma_ref': 1.0,
        's_floor': settings['s_floor'],
        'alpha_temp': settings['alpha_temp'],
        'freeze_eta0': settings['freeze_eta0'],
        'eta0_fixed': settings['eta0_fixed'],
        'eta0_eps': settings['eta0_eps'],
        'mu_min_gamma': settings['mu_min_gamma'],
        'mu_max_gamma': settings['mu_max_gamma'],
        'gate_gamma': settings['gate_gamma'],
        'gate_width_z': settings['gate_width_z'],
        'tail_gate_gamma': settings['tail_gate_gamma'],
        'tail_gate_width_z': settings['tail_gate_width_z'],
        'enable_pl_per_mode': settings['enable_pl_per_mode'],
        'pl_width_z': settings['pl_width_z'],
        'log_head': settings['log_head'],
        'log_mixing': settings['log_mixing'],
        'freeze_centers': settings['freeze_centers'],
    }
    
    # Build TBNN model
    TBNN_model = models.build_tbnn_bounded_model(
        hidden_units=hidden_units,
        **tbnn_config
    )
    
    # Create dummy inputs
    H, W = domain_size[1], domain_size[0]  # domain_size is (nx, ny)
    dummy_gamma_dot = jnp.ones((H, W)) * 1.0
    dummy_invariants = jnp.ones((2, H, W))
    
    # Initialize parameters with soft Newtonian init
    print(f"   Using soft Newtonian initialization (eta0 = {eta_init})")
    params = models.init_tbnn_soft_newtonian(
        TBNN_model, key, H, W, eta_init,
        A_frac=0.05,
        k_frac=0.2,
        pair_modes=(0, 1)
    )
    
    # Flatten parameters
    flattened_params, tree_def, shapes = parameter_utils.flatten_params(params)
    
    # Calculate parameter indices
    starts_static = []
    ends_static = []
    idx = 0
    for shape in shapes:
        size = int(np.prod(np.array(shape)))
        starts_static.append(idx)
        ends_static.append(idx + size)
        idx += size
    
    model_info = {
        'model': TBNN_model,
        'params': params,
        'flattened_params': flattened_params,
        'tree_def': tree_def,
        'shapes': shapes,
        'starts_static': starts_static,
        'ends_static': ends_static,
        'hidden_units': hidden_units,
        'num_params': len(flattened_params),
        'eta_init': eta_init,
        'eta_min': tbnn_config['eta_min'],
        'eta_max': tbnn_config['eta_max'],
        'tbnn_config': tbnn_config.copy(),
        'init_method': 'soft_newtonian'
    }
    
    print(f"TBNN initialized with {model_info['num_params']} parameters (soft_newtonian)")
    print(f"TBNN bounds: eta  in  [{tbnn_config['eta_min']:.3f}, {tbnn_config['eta_max']:.3f}]")
    print(f"TBNN config: M={M}, hidden={hidden_units}, log_head={settings['log_head']}")
    
    return model_info

# =============================================================================
# MODEL-SPECIFIC CONFIGURATIONS
# =============================================================================

def get_model_config(model_name, *params):
    """Get model configuration based on model name and parameters."""
    
    if model_name.lower() == 'newtonian':
        if len(params) != 1:
            raise ValueError("Newtonian model requires 1 parameter: viscosity")
        viscosity = float(params[0])
        
        return {
            'model_type': 'newtonian',
            'forcing_fn': models.newtonian_stress_forcing,
            'model_params': viscosity,
            'base_viscosity': viscosity,
            'nu0_baseline': viscosity,
            'description': f'Newtonian (mu={viscosity:.3f})'
        }
    
    elif model_name.lower() == 'power_law':
        if len(params) != 2:
            raise ValueError("Power-Law model requires 2 parameters: K, n")
        K, n = float(params[0]), float(params[1])
        
        return {
            'model_type': 'power_law', 
            'forcing_fn': models.power_law_stress_forcing,
            'model_params': jnp.array([K, n]),
            'base_viscosity': K,
            'nu0_baseline': 0.0,  # Pure non-Newtonian
            'description': f'Power-Law (K={K:.3f}, n={n:.3f})'
        }
    
    elif model_name.lower() == 'carreau_yasuda':
        if len(params) != 5:
            raise ValueError("Carreau-Yasuda model requires 5 parameters: eta_inf, eta_0, lambda, n, a")
        eta_inf, eta_0, lambda_, n, a = map(float, params)
        
        return {
            'model_type': 'carreau_yasuda',
            'forcing_fn': models.carreau_yasuda_stress_forcing,
            'model_params': jnp.array([eta_inf, eta_0, lambda_, n, a]),
            'base_viscosity': 0.0,  # Carreau-Yasuda handles all viscosity
            'nu0_baseline': 0.0,
            'description': f'Carreau-Yasuda (eta_inf={eta_inf:.3f}, eta_0={eta_0:.3f}, lam={lambda_:.3f}, n={n:.3f}, a={a:.3f})'
        }
    
    elif model_name.lower() == 'tbnn':
        # TBNN model: params = (tbnn_params_path_or_init, domain_size, random_seed, **tbnn_settings)
        # The first param should be either:
        #   - A path to .pkl file (string)
        #   - 'init' to initialize fresh
        # The second param should be domain_size tuple
        # The third param should be random_seed (optional)
        # Additional params are TBNN settings overrides
        
        if len(params) < 2:
            raise ValueError("TBNN model requires at least 2 parameters: (params_path_or_init, domain_size)")
        
        params_path = params[0]
        domain_size = params[1]
        random_seed = params[2] if len(params) > 2 else 42
        
        # Extract any TBNN setting overrides from remaining params
        tbnn_overrides = {}
        if len(params) > 3:
            # If there are additional params, they should be keyword arguments
            # For now, use defaults - user can modify DEFAULT_TBNN_SETTINGS if needed
            pass
        
        if isinstance(params_path, str) and params_path.lower() == 'init':
            # Initialize fresh TBNN
            print(f"Initializing fresh TBNN model...")
            model_info = setup_tbnn_model_for_channel(domain_size, random_seed, **tbnn_overrides)
            tbnn_params = model_info['flattened_params']
            tbnn_model = model_info['model']
            tree_def = model_info['tree_def']
            shapes = model_info['shapes']
            starts_static = model_info['starts_static']
            ends_static = model_info['ends_static']
            description = f"TBNN (fresh init, M={model_info['tbnn_config']['M']}, hidden={model_info['hidden_units']})"
        else:
            # Load from .pkl file
            print(f"Loading TBNN parameters from: {params_path}")
            if not os.path.exists(params_path):
                raise FileNotFoundError(f"TBNN parameter file not found: {params_path}")
            
            # Check if user accidentally provided tree_def or shapes file
            if 'tree_def' in os.path.basename(params_path):
                raise ValueError(
                    f"ERROR: You provided a tree_def file, but you need the params file!\n"
                    f"  You gave: {params_path}\n"
                    f"  You need: {params_path.replace('tree_def', 'params')}\n"
                    f"  The tree_def file contains metadata, not the actual parameters."
                )
            if 'shapes' in os.path.basename(params_path):
                raise ValueError(
                    f"ERROR: You provided a shapes file, but you need the params file!\n"
                    f"  You gave: {params_path}\n"
                    f"  You need: {params_path.replace('shapes.npy', 'params.pkl')}\n"
                    f"  The shapes file contains metadata, not the actual parameters."
                )
            
            with open(params_path, 'rb') as f:
                params_dict = pickle.load(f)
            
            # Validate that we loaded actual parameters, not metadata
            from jax.tree_util import PyTreeDef
            if isinstance(params_dict, PyTreeDef):
                raise ValueError(
                    f"ERROR: The file contains a PyTreeDef (metadata), not parameters!\n"
                    f"  You gave: {params_path}\n"
                    f"  Make sure you're loading '*_params.pkl', not '*_tree_def.pkl'\n"
                    f"  The correct file should be in the same directory."
                )
            
            # Build TBNN model with same architecture
            # We need to know the architecture to rebuild the model
            # For now, use defaults - user should ensure .pkl was saved with compatible architecture
            model_info = setup_tbnn_model_for_channel(domain_size, random_seed, **tbnn_overrides)
            tbnn_model = model_info['model']
            tree_def = model_info['tree_def']
            shapes = model_info['shapes']
            starts_static = model_info['starts_static']
            ends_static = model_info['ends_static']
            
            # Replace with loaded parameters
            model_info['params'] = params_dict
            try:
                tbnn_params, _, _ = parameter_utils.flatten_params(params_dict)
                model_info['flattened_params'] = tbnn_params
            except AttributeError as e:
                raise ValueError(
                    f"ERROR: Failed to flatten loaded parameters. This usually means you loaded metadata instead of actual parameters.\n"
                    f"  File: {params_path}\n"
                    f"  Make sure the file contains a FrozenDict of parameters, not a PyTreeDef or array shapes.\n"
                    f"  Original error: {e}"
                )
            
            print(f"Loaded TBNN parameters: {len(tbnn_params)} values")
            description = f"TBNN (loaded from {os.path.basename(params_path)})"
        
        return {
            'model_type': 'TBNN',
            'forcing_fn': models.TBNN_stress_forcing,
            'model_params': tbnn_params,
            'base_viscosity': 0.0,  # TBNN handles all viscosity
            'nu0_baseline': 0.0,
            'description': description,
            'tbnn_model': tbnn_model,
            'tree_def': tree_def,
            'shapes': shapes,
            'starts_static': starts_static,
            'ends_static': ends_static,
            'model_info': model_info
        }
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: newtonian, power_law, carreau_yasuda, tbnn")

# =============================================================================
# SIMULATION RUNNER
# =============================================================================

def run_channel_obstacle(
    model_name,
    *params,
    show_plots=True,
    save_trajectory=True,
    output_dir=str(REPO_ROOT / 'work' / 'reference_trajectories'),
    pressure_gradient=None,
    preconditioner=None,
    dt=None,
    inner_steps=None,
    outer_steps=None,
):
    """
    Run channel flow with obstacle simulation for specified rheological model.
    
    Args:
        model_name: 'newtonian', 'power_law', or 'carreau_yasuda'
        *params: Model-specific parameters
            - Newtonian: viscosity
            - Power-Law: K, n  
            - Carreau-Yasuda: eta_inf, eta_0, lambda, n, a
        show_plots: Whether to display plots (default: True)
        save_trajectory: Whether to save trajectory to .npy file (default: True)
        output_dir: Directory to save trajectory file (default: repo reference_trajectories)
        pressure_gradient: Pressure gradient to override default (default: None uses 2.5)
        preconditioner: Preconditioner type ('helmholtz', 'jacobi', 'None', or None for no preconditioning)
        dt: Time step override (default: None uses 1e-4)
        inner_steps: Inner steps override (default: None uses 200)
        outer_steps: Outer steps override (default: None uses 300)
    
    Returns:
        Dictionary with simulation results
    """
    print(f"\n{'='*80}")
    print(f"CHANNEL FLOW WITH OBSTACLE - {model_name.upper()} MODEL")
    print(f"{'='*80}")
    
    # Get model configuration
    model_config = get_model_config(model_name, *params)
    
    # Handle parameter overrides
    actual_pressure_gradient = pressure_gradient if pressure_gradient is not None else COMMON_CONFIG['pressure_gradient']
    actual_dt = dt if dt is not None else COMMON_CONFIG['dt']
    actual_inner_steps = inner_steps if inner_steps is not None else COMMON_CONFIG['inner_steps']
    actual_outer_steps = outer_steps if outer_steps is not None else COMMON_CONFIG['outer_steps']
    
    # Handle preconditioner settings
    use_preconditioner = preconditioner is not None and preconditioner.lower() != 'none'
    preconditioner_type = preconditioner.lower() if use_preconditioner else 'jacobi'
    
    print(f"Model: {model_config['description']}")
    print(f"Domain: {COMMON_CONFIG['domain']}, Grid: {COMMON_CONFIG['domain_size']}")
    print(f"Parameters: rho={COMMON_CONFIG['density']}, gradp={actual_pressure_gradient}")
    print(f"Time stepping: dt={actual_dt}, inner={actual_inner_steps}, outer={actual_outer_steps}")
    print(f"Solver: {COMMON_CONFIG['solver_type'].upper()}")
    if use_preconditioner:
        print(f"Preconditioner: {preconditioner_type.upper()}")
    else:
        print(f"Preconditioner: None")
    
    # Create grid and obstacle
    grid = flow_conditions.create_grid(COMMON_CONFIG['domain_size'], COMMON_CONFIG['domain'])
    particles = setup_channel_obstacle(COMMON_CONFIG['domain'])
    
    # Create flow conditions
    flow_cond = {
        'density': COMMON_CONFIG['density'],
        'base_viscosity': model_config['base_viscosity'],
        'pressure_gradient': actual_pressure_gradient,
        'dt': actual_dt,
        'U_f': 0.0,
        'grid': grid,
        'tree_def': None,
        'shapes': None,
        'starts_static': None,
        'ends_static': None,
        'amp_shear': 0.0,
        'freq_osc': 0.0,
        'nu0_baseline': model_config['nu0_baseline'],
        'inner_steps': actual_inner_steps,
        'outer_steps': actual_outer_steps,
        'boundary_type': 'moving_wall'  # Channel flow boundary conditions
    }
    
    # For TBNN, add model-specific fields to flow_cond
    if model_config['model_type'] == 'TBNN':
        flow_cond['tree_def'] = model_config['tree_def']
        flow_cond['shapes'] = model_config['shapes']
        flow_cond['starts_static'] = model_config['starts_static']
        flow_cond['ends_static'] = model_config['ends_static']
    
    # Create nu0 update function
    if model_config['model_type'] == 'TBNN':
        nu0_update_fn = models.create_dynamic_nu0_fn(
            'TBNN',
            model=model_config['tbnn_model'],
            strategy='max',
            C=1.0
        )
    else:
        nu0_update_fn = models.create_dynamic_nu0_fn(
            model_type=model_config['model_type'],
            strategy='max', 
            C=1.0
        )
    
    # Run simulation
    print("\nRunning simulation...")
    start_time = time.time()
    
    try:
        # For TBNN, pass the model; for others, pass None
        model_to_pass = model_config.get('tbnn_model', None) if model_config['model_type'] == 'TBNN' else None
        
        final_result, trajectory, perm_f = forward_simulation.forward_fluid_simulation(
            flow_cond=flow_cond,
            flattened_params=model_config['model_params'],
            particles=particles,
            stress_forcing_fn=model_config['forcing_fn'],
            model=model_to_pass,
            nu0_update_fn=nu0_update_fn,
            nu0_baseline=model_config['nu0_baseline'],
            stepper_type=COMMON_CONFIG['stepper_type'],
            solver_type=COMMON_CONFIG['solver_type'],
            use_preconditioner=use_preconditioner,
            preconditioner_type=preconditioner_type
        )
        
        end_time = time.time()
        
        print(f"Simulation completed in {end_time - start_time:.1f} seconds")
        
        # Check for NaN values
        # Handle both Field objects (with .data) and raw arrays
        if hasattr(final_result.velocity[0], 'data'):
            vx_final = final_result.velocity[0].data
            vy_final = final_result.velocity[1].data
        else:
            vx_final = final_result.velocity[0]
            vy_final = final_result.velocity[1]
        has_nan = jnp.any(jnp.isnan(vx_final)) or jnp.any(jnp.isnan(vy_final))
        
        if has_nan:
            print("WARNING: Simulation contains NaN values!")
            return None
        
        # Print velocity ranges
        print(f"Final vx range: [{float(jnp.min(vx_final)):.6f}, {float(jnp.max(vx_final)):.6f}]")
        print(f"Final vy range: [{float(jnp.min(vy_final)):.6f}, {float(jnp.max(vy_final)):.6f}]")
        
        # Create trajectory array in format expected by loss functions
        # Format: (time_steps, 2, grid_x, grid_y) where axis 1 contains [vx, vy]
        # Handle both Field objects (with .data) and raw arrays
        if hasattr(trajectory[0], 'data'):
            traj_x = trajectory[0].data
            traj_y = trajectory[1].data
        else:
            traj_x = trajectory[0]
            traj_y = trajectory[1]
        trajectory_array = jnp.stack([traj_x, traj_y], axis=1)
        print(f"Trajectory shape: {trajectory_array.shape}")
        
        # Create result dictionary
        results = {
            'model_name': model_name,
            'model_config': model_config,
            'final_result': final_result,
            'trajectory': trajectory_array,
            'runtime_seconds': end_time - start_time,
            'flow_cond': flow_cond,
            'particles': particles,
            'success': True
        }
        
        # Save trajectory if requested
        if save_trajectory:
            save_trajectory_file(results, output_dir)
        
        # Create plots if requested
        if show_plots:
            plot_channel_obstacle_results(results)
        
        # Print completion summary
        print("")
        print(f"   Model: {model_config['description']}")
        print(f"   Runtime: {end_time - start_time:.1f} seconds")
        print(f"   Final velocity range: vx=[{float(jnp.min(vx_final)):.4f}, {float(jnp.max(vx_final)):.4f}], vy=[{float(jnp.min(vy_final)):.4f}, {float(jnp.max(vy_final)):.4f}]")
        if save_trajectory:
            print(f"   Trajectory saved with shape: {trajectory_array.shape}")
        print(f"   Results stored in returned dictionary")
        
        return results
        
    except Exception as e:
        print(f"Simulation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# TRAJECTORY SAVING
# =============================================================================

def save_trajectory_file(results, output_dir='.'):
    """Save trajectory to .npy file with descriptive name."""
    model_name = results['model_name']
    model_config = results['model_config'] 
    trajectory = results['trajectory']
    
    # Create descriptive filename
    if model_name.lower() == 'newtonian':
        # Newtonian: viscosity
        params_str = f"visc{model_config['model_params']:.3f}"
    elif model_name.lower() == 'power_law':
        # Power-Law: K and n
        K, n = model_config['model_params']
        params_str = f"K{K:.3f}_n{n:.3f}"
    elif model_name.lower() == 'carreau_yasuda':
        # Carreau-Yasuda: main parameters
        eta_inf, eta_0, lambda_, n, a = model_config['model_params']
        params_str = f"etainf{eta_inf:.3f}_eta0{eta_0:.3f}_lam{lambda_:.3f}_n{n:.3f}_a{a:.3f}"
    elif model_name.lower() == 'tbnn':
        # TBNN: get info from description or use generic name
        if 'model_info' in model_config:
            M = model_config['model_info']['tbnn_config'].get('M', 12)
            hidden = model_config['model_info']['hidden_units']
            params_str = f"M{M}_hidden{'_'.join(map(str, hidden))}"
        else:
            # Fallback to generic TBNN name
            params_str = "tbnn"
    
    # Always add simulation parameters to filename for metadata extraction
    flow_cond = results['flow_cond']
    pressure_grad = flow_cond['pressure_gradient']
    dt_val = flow_cond['dt']
    inner_val = flow_cond['inner_steps']
    outer_val = flow_cond['outer_steps']
    
    # Always include these parameters (not just non-defaults)
    params_str += f"_pg{pressure_grad:.2f}"
    params_str += f"_dt{dt_val:.0e}"
    params_str += f"_in{inner_val}"
    params_str += f"_out{outer_val}"
    
    filename = f"channel_obstacle_{model_name}_{params_str}_trajectory.npy"
    filepath = os.path.join(output_dir, filename)
    
    # Save trajectory
    np.save(filepath, np.array(trajectory))
    print(f"Trajectory saved: {filepath}")
    print(f"   Shape: {trajectory.shape}")
    print(f"   Format: (time_steps, velocity_components[vx,vy], grid_x, grid_y)")

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_channel_obstacle_results(results):
    """Create comprehensive visualization of channel obstacle flow."""
    final_result = results['final_result']
    model_config = results['model_config']
    
    # Handle both Field objects (with .data) and raw arrays
    if hasattr(final_result.velocity[0], 'data'):
        vx_final = final_result.velocity[0].data
        vy_final = final_result.velocity[1].data
    else:
        vx_final = final_result.velocity[0]
        vy_final = final_result.velocity[1]
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    domain = COMMON_CONFIG['domain']
    
    # X-velocity field
    im1 = axes[0,0].imshow(vx_final.T, origin='lower', cmap='coolwarm', 
                           extent=[domain[0][0], domain[0][1], domain[1][0], domain[1][1]],
                           aspect='auto')
    axes[0,0].set_title(f'X-velocity\n{model_config["description"]}')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Y-velocity field
    im2 = axes[0,1].imshow(vy_final.T, origin='lower', cmap='coolwarm',
                           extent=[domain[0][0], domain[0][1], domain[1][0], domain[1][1]],
                           aspect='auto')
    axes[0,1].set_title(f'Y-velocity\n{model_config["description"]}')
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Velocity magnitude
    vel_mag = jnp.sqrt(vx_final**2 + vy_final**2)
    im3 = axes[0,2].imshow(vel_mag.T, origin='lower', cmap='viridis',
                           extent=[domain[0][0], domain[0][1], domain[1][0], domain[1][1]],
                           aspect='auto')
    axes[0,2].set_title(f'Velocity Magnitude\n{model_config["description"]}')
    axes[0,2].set_xlabel('x')
    axes[0,2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[0,2])
    
    # Streamlines
    try:
        x = jnp.linspace(domain[0][0], domain[0][1], vx_final.shape[0])
        y = jnp.linspace(domain[1][0], domain[1][1], vx_final.shape[1])
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        
        skip = max(1, vx_final.shape[0] // 20)
        X_stream = np.array(X[::skip, ::skip])
        Y_stream = np.array(Y[::skip, ::skip])
        vx_stream = np.array(vx_final[::skip, ::skip])
        vy_stream = np.array(vy_final[::skip, ::skip])
        
        axes[1,0].streamplot(X_stream.T, Y_stream.T, vx_stream.T, vy_stream.T, 
                            density=1.5, color='k', arrowsize=1.2)
        axes[1,0].set_title('Streamlines')
        axes[1,0].set_xlabel('x')
        axes[1,0].set_ylabel('y')
        axes[1,0].set_xlim(domain[0])
        axes[1,0].set_ylim(domain[1])
        
        # Add obstacle outline
        obstacle_center = [4.0, 2.0]  # Center of domain
        obstacle_radius = 0.4
        circle = plt.Circle(obstacle_center, obstacle_radius, fill=False, color='red', linewidth=2)
        axes[1,0].add_patch(circle)
        
    except Exception as e:
        axes[1,0].text(0.5, 0.5, f'Streamlines failed\n{str(e)[:50]}...', 
                       ha='center', va='center', transform=axes[1,0].transAxes)
    
    # Velocity profiles at different x-positions
    x_positions = [0.25, 0.5, 0.75]  # Relative positions along channel
    y_coords = jnp.linspace(domain[1][0], domain[1][1], vx_final.shape[1])
    
    for i, x_rel in enumerate(x_positions):
        x_idx = int(x_rel * vx_final.shape[0])
        profile = vx_final[x_idx, :]
        x_pos = domain[0][0] + x_rel * (domain[0][1] - domain[0][0])
        axes[1,1].plot(profile, y_coords, linewidth=2, 
                      label=f'x={x_pos:.1f}', alpha=0.8)
    
    axes[1,1].set_xlabel('vx velocity')
    axes[1,1].set_ylabel('y position')
    axes[1,1].set_title('Velocity Profiles at Different x-positions')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()
    
    # Local viscosity field (log scale heatmap)
    try:
        from jax_rheology.models import get_viscosity_field
        import jax_ib.base as ib
        
        # Get viscosity field based on model type
        model_type = model_config['model_type']
        if model_type == 'newtonian':
            # For Newtonian, create constant viscosity field
            viscosity_field = jnp.full_like(vx_final, model_config['model_params'])
        else:
            # Create proper velocity field using the original result's grid structure
            # Extract grid from final_result
            original_vx = final_result.velocity[0]
            original_vy = final_result.velocity[1]
            
            # Use the original grid structure with final velocity data
            v = [ib.grids.GridVariable(ib.grids.GridArray(vx_final, original_vx.offset, original_vx.grid), original_vx.bc),
                 ib.grids.GridVariable(ib.grids.GridArray(vy_final, original_vy.offset, original_vy.grid), original_vy.bc)]
            
            viscosity_field = get_viscosity_field(v, model_config['model_params'], model_type)
        
        # Create log-scale viscosity heatmap
        log_viscosity = jnp.log10(jnp.maximum(viscosity_field, 1e-8))  # Avoid log(0)
        
        im4 = axes[1,2].imshow(log_viscosity.T, origin='lower', cmap='plasma',
                              extent=[domain[0][0], domain[0][1], domain[1][0], domain[1][1]],
                              aspect='auto')
        axes[1,2].set_title(f'Local Viscosity (log₁₀ scale)\n{model_config["description"]}')
        axes[1,2].set_xlabel('x')
        axes[1,2].set_ylabel('y')
        cbar = plt.colorbar(im4, ax=axes[1,2])
        cbar.set_label('log₁₀(η)')
        
        # Add obstacle outline
        obstacle_center = [4.0, 2.0]  # Center of domain
        obstacle_radius = 0.4
        circle = plt.Circle(obstacle_center, obstacle_radius, fill=False, color='white', linewidth=2)
        axes[1,2].add_patch(circle)
        
    except Exception as e:
        # Fallback if viscosity calculation fails
        axes[1,2].text(0.5, 0.5, f'Viscosity plot failed\n{str(e)[:50]}...', 
                       ha='center', va='center', transform=axes[1,2].transAxes,
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
        axes[1,2].set_title('Local Viscosity (log scale)')
        print(f"   Viscosity plot failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Print flow statistics
    print(f"\nFLOW STATISTICS:")
    print(f"   Max velocity: {float(jnp.max(vel_mag)):.6f}")
    print(f"   Mean velocity: {float(jnp.mean(vel_mag)):.6f}")
    
    # Add viscosity statistics if available
    try:
        from jax_rheology.models import get_viscosity_field
        import jax_ib.base as ib
        
        model_type = model_config['model_type']
        if model_type == 'newtonian':
            visc_field = jnp.full_like(vx_final, model_config['model_params'])
        else:
            # Use proper grid structure for statistics too
            original_vx = final_result.velocity[0]
            original_vy = final_result.velocity[1]
            v = [ib.grids.GridVariable(ib.grids.GridArray(vx_final, original_vx.offset, original_vx.grid), original_vx.bc),
                 ib.grids.GridVariable(ib.grids.GridArray(vy_final, original_vy.offset, original_vy.grid), original_vy.bc)]
            visc_field = get_viscosity_field(v, model_config['model_params'], model_type)
        
        print(f"   Viscosity range: [{float(jnp.min(visc_field)):.6e}, {float(jnp.max(visc_field)):.6e}]")
        print(f"   Mean viscosity: {float(jnp.mean(visc_field)):.6e}")
        if model_type != 'newtonian':
            visc_ratio = float(jnp.max(visc_field) / jnp.min(visc_field))
            print(f"   Viscosity ratio (max/min): {visc_ratio:.2f}")
    except:
        pass
    
    # Compute divergence for flow quality assessment
    domain_size = vx_final.shape
    dx = (domain[0][1] - domain[0][0]) / domain_size[0]
    dy = (domain[1][1] - domain[1][0]) / domain_size[1]
    
    dvx_dx = jnp.gradient(vx_final, dx, axis=0)
    dvy_dy = jnp.gradient(vy_final, dy, axis=1)  
    divergence = dvx_dx + dvy_dy
    max_div = float(jnp.max(jnp.abs(divergence)))
    mean_div = float(jnp.mean(jnp.abs(divergence)))
    
    print(f"   Max |grad.v|: {max_div:.6e}")
    print(f"   Mean |grad.v|: {mean_div:.6e}")
    
    if max_div < 1e-6:
        print("   divergence within tolerance")
    elif max_div < 1e-4:
        print("   GOOD divergence control") 
    elif max_div < 1e-3:
        print("   ACCEPTABLE divergence control")
    else:
        print("   POOR divergence control")

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(description='Channel flow with obstacle simulation')
    parser.add_argument('--verbose', action='store_true',
                        help='Print import and environment banners at startup.')
    parser.add_argument('model', choices=['newtonian', 'power_law', 'carreau_yasuda', 'tbnn'],
                       help='Rheological model type')
    parser.add_argument('params', nargs='+',
                       help='Model parameters (for TBNN: params_path or "init")')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plots (useful for cluster runs)')
    parser.add_argument('--output-dir', default=str(REPO_ROOT / 'work' / 'reference_trajectories'),
                       help='Output directory for trajectory files')
    parser.add_argument('--pressure-gradient', type=float, default=None,
                       help='Pressure gradient override (default: 2.5)')
    parser.add_argument('--preconditioner', type=str, default=None,
                       help='Preconditioner type: helmholtz, jacobi, or none (default: none)')
    parser.add_argument('--dt', type=float, default=None,
                       help='Time step override (default: 1e-4)')
    parser.add_argument('--inner-steps', type=int, default=None,
                       help='Inner steps override (default: 200)')
    parser.add_argument('--outer-steps', type=int, default=None,
                       help='Outer steps override (default: 300)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for TBNN initialization (default: 42)')
    
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(parser)
    
    # Validate parameter count (convert to float for numeric models)
    if args.model == 'tbnn':
        # TBNN: first param is path or 'init', rest are optional
        if len(args.params) < 1:
            print(f"Error: TBNN model requires at least 1 parameter: params_path or 'init'")
            print("  Examples:")
            print("    python channel_obstacle_flow.py tbnn init")
            print("    python channel_obstacle_flow.py tbnn /path/to/params.pkl")
            sys.exit(1)
        
        # For TBNN, pass string param and domain_size
        tbnn_params_path = args.params[0]
        domain_size = COMMON_CONFIG['domain_size']
        random_seed = args.random_seed
        processed_params = [tbnn_params_path, domain_size, random_seed]
    else:
        # Convert to float for other models
        try:
            numeric_params = [float(p) for p in args.params]
        except ValueError:
            print(f"Error: Non-TBNN models require numeric parameters")
            sys.exit(1)
        
        expected_params = {
            'newtonian': 1,
            'power_law': 2, 
            'carreau_yasuda': 5
        }
        
        if len(numeric_params) != expected_params[args.model]:
            print(f"Error: {args.model} model requires {expected_params[args.model]} parameters")
            print("  newtonian: viscosity")
            print("  power_law: K, n")
            print("  carreau_yasuda: eta_inf, eta_0, lambda, n, a")
            sys.exit(1)
        
        processed_params = numeric_params
    
    # Run simulation
    print(f"Running {args.model} simulation with parameters: {args.params}")
    if args.pressure_gradient is not None:
        print(f"Using pressure gradient override: {args.pressure_gradient}")
    if args.preconditioner is not None:
        print(f"Using preconditioner: {args.preconditioner}")
    if args.dt is not None:
        print(f"Using time step override: {args.dt}")
    if args.inner_steps is not None:
        print(f"Using inner steps override: {args.inner_steps}")
    if args.outer_steps is not None:
        print(f"Using outer steps override: {args.outer_steps}")
    
    results = run_channel_obstacle(
        args.model, 
        *processed_params,
        show_plots=not args.no_plots,
        save_trajectory=True,
        output_dir=args.output_dir,
        pressure_gradient=args.pressure_gradient,
        preconditioner=args.preconditioner,
        dt=args.dt,
        inner_steps=args.inner_steps,
        outer_steps=args.outer_steps
    )
    
    if results:
        print("")
        print(f"Runtime: {results['runtime_seconds']:.1f} seconds")
    else:
        print(f"\nSimulation failed!")
        sys.exit(1)

# =============================================================================
# CONVENIENCE FUNCTIONS FOR NOTEBOOK USE
# =============================================================================

def run_newtonian_demo(viscosity=1.0, pressure_gradient=None, preconditioner=None, dt=None, inner_steps=None, outer_steps=None):
    """Quick Newtonian demo."""
    return run_channel_obstacle('newtonian', viscosity, pressure_gradient=pressure_gradient, preconditioner=preconditioner, dt=dt, inner_steps=inner_steps, outer_steps=outer_steps)

def run_power_law_demo(K=0.5, n=0.8, pressure_gradient=None, preconditioner=None, dt=None, inner_steps=None, outer_steps=None):
    """Quick Power-Law demo.""" 
    return run_channel_obstacle('power_law', K, n, pressure_gradient=pressure_gradient, preconditioner=preconditioner, dt=dt, inner_steps=inner_steps, outer_steps=outer_steps)

def run_carreau_yasuda_demo(eta_inf=0.02, eta_0=0.5, lambda_=5.0, n=0.8, a=2.0, pressure_gradient=None, preconditioner=None, dt=None, inner_steps=None, outer_steps=None):
    """Quick Carreau-Yasuda demo."""
    return run_channel_obstacle('carreau_yasuda', eta_inf, eta_0, lambda_, n, a, pressure_gradient=pressure_gradient, preconditioner=preconditioner, dt=dt, inner_steps=inner_steps, outer_steps=outer_steps)

def run_tbnn_demo(params_path='init', random_seed=42, pressure_gradient=None, preconditioner=None, dt=None, inner_steps=None, outer_steps=None, save_trajectory=True, show_plots=True):
    """Quick TBNN demo.
    
    Args:
        params_path: Path to .pkl file or 'init' to initialize fresh (default: 'init')
        random_seed: Random seed for initialization (default: 42)
        pressure_gradient: Pressure gradient override (default: None uses 2.5)
        preconditioner: Preconditioner type (default: None)
        dt: Time step override (default: None uses 1e-4)
        inner_steps: Inner steps override (default: None uses 200)
        outer_steps: Outer steps override (default: None uses 300)
        save_trajectory: Whether to save trajectory to .npy file (default: True)
        show_plots: Whether to display plots (default: True)
    
    Example:
        # Initialize fresh TBNN
        results = run_tbnn_demo('init', random_seed=1453)
        
        # Load trained TBNN without saving trajectory
        results = run_tbnn_demo('/path/to/final_tbnn_params.pkl', save_trajectory=False)
    """
    domain_size = COMMON_CONFIG['domain_size']
    return run_channel_obstacle('tbnn', params_path, domain_size, random_seed, 
                                pressure_gradient=pressure_gradient, preconditioner=preconditioner, 
                                dt=dt, inner_steps=inner_steps, outer_steps=outer_steps,
                                save_trajectory=save_trajectory, show_plots=show_plots)

def run_comparison_demo(include_tbnn=False):
    """Run comparison of rheological models.
    
    Args:
        include_tbnn: If True, include TBNN in comparison (default: False)
    """
    print("Running comparison of rheological models...")
    
    results = {}
    
    # Newtonian
    print("\n" + "="*50)
    results['newtonian'] = run_channel_obstacle('newtonian', 0.5, show_plots=False)
    
    # Power-Law
    print("\n" + "="*50)  
    results['power_law'] = run_channel_obstacle('power_law', 0.5, 0.8, show_plots=False)
    
    # Carreau-Yasuda
    print("\n" + "="*50)
    results['carreau_yasuda'] = run_channel_obstacle('carreau_yasuda', 0.02, 0.5, 5.0, 0.8, 2.0, show_plots=False)
    
    # TBNN (optional)
    if include_tbnn:
        print("\n" + "="*50)
        domain_size = COMMON_CONFIG['domain_size']
        results['tbnn'] = run_channel_obstacle('tbnn', 'init', domain_size, 42, show_plots=False)
    
    # Create comparison plot
    if all(r is not None for r in results.values()):
        plot_model_comparison(results)
    
    return results

def plot_model_comparison(results):
    """Plot comparison of different rheological models."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    domain = COMMON_CONFIG['domain']
    y_coords = jnp.linspace(domain[1][0], domain[1][1], 128)
    
    colors = {'newtonian': 'blue', 'power_law': 'red', 'carreau_yasuda': 'green'}
    
    for i, (model_name, result) in enumerate(results.items()):
        if result is None:
            continue
        
        # Handle both Field objects (with .data) and raw arrays
        if hasattr(result['final_result'].velocity[0], 'data'):
            vx_final = result['final_result'].velocity[0].data
            vy_final = result['final_result'].velocity[1].data
        else:
            vx_final = result['final_result'].velocity[0]
            vy_final = result['final_result'].velocity[1]
        vel_mag = jnp.sqrt(vx_final**2 + vy_final**2)
        
        # Velocity magnitude fields
        im = axes[0, i].imshow(vel_mag.T, origin='lower', cmap='viridis',
                              extent=[domain[0][0], domain[0][1], domain[1][0], domain[1][1]],
                              aspect='auto')
        axes[0, i].set_title(f'{model_name.title()}\n{result["model_config"]["description"]}')
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('y')
        plt.colorbar(im, ax=axes[0, i])
        
        # Velocity profiles at channel center
        mid_x = vx_final.shape[0] // 2
        profile = vx_final[mid_x, :]
        axes[1, 1].plot(profile, y_coords, color=colors[model_name], 
                       linewidth=2, label=model_name.title())
    
    axes[1, 1].set_xlabel('vx velocity')
    axes[1, 1].set_ylabel('y position')
    axes[1, 1].set_title('Velocity Profile Comparison (Channel Center)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Hide unused subplots
    axes[1, 0].set_visible(False)
    axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

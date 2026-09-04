#!/usr/bin/env python
"""
Porous Media Flow - Multiple Rheological Models

This script simulates porous media flow with two circular obstacles,
supporting Newtonian, Power-Law, Carreau-Yasuda, and TBNN rheological models.
Uses periodic boundary conditions and nu0_split stepper.

Obstacle configuration:
    - Obstacle 1: center at (3, 1), radius = 1.0
    - Obstacle 2: center at (1, 3), radius = 0.5

USAGE:
Notebook:
    from porous_media_flow import *
    
    # Newtonian flow
    run_porous_media('newtonian', 1.0)
    # OR use convenience function:
    run_porous_media_newtonian_demo(viscosity=1.0)
    
    # Power-Law flow  
    run_porous_media('power_law', 0.5, 0.8)  # K=0.5, n=0.8
    # OR: run_porous_media_power_law_demo(K=0.5, n=0.8)
    
    # Carreau-Yasuda flow
    run_porous_media('carreau_yasuda', 0.02, 0.5, 5.0, 0.8, 2.0)  # eta_inf, eta_0, lambda, n, a
    # OR: run_porous_media_carreau_yasuda_demo(eta_inf=0.02, eta_0=0.5, lambda_=5.0, n=0.8, a=2.0)
    
    # TBNN flow (initialize fresh)
    run_porous_media('tbnn', 'init', (128, 128), 42)  # 'init', domain_size, random_seed
    # OR: run_porous_media_tbnn_demo('init', random_seed=42)
    
    # TBNN flow (load from file)
    run_porous_media('tbnn', '/path/to/params.pkl', (128, 128), 42)
    # OR: run_porous_media_tbnn_demo('/path/to/params.pkl')
    
    # Compare two models (e.g., TBNN vs Carreau-Yasuda)
    comparison = run_demo_comparison(
        ground_truth_model='carreau_yasuda',
        ground_truth_params=(0.02, 1.0, 5.0, 0.7, 2.0),
        comparison_model='tbnn',
        comparison_params=('path/to/final_tbnn_params.pkl', 42),  # (params_path, random_seed)
        domain_size=(256, 256),  # Simulation resolution - applies to both models
        dt=1e-4,
        inner_steps=400,
        outer_steps=200,
        pressure_gradient=10.0,
        num_bins=20,
        save_trajectory=False
    )
    # Returns: {'ground_truth': <results>, 'comparison': <results>, 'metrics': <dict>}
    # Creates 2x2 plot:
    #   - Top left: Ground truth x-velocity
    #   - Bottom left: Comparison x-velocity
    #   - Top right: Percentage difference map
    #   - Bottom right: Relative error vs strain rate (log-log, binned)

Command line:
    python porous_media_flow.py newtonian 1.0
    python porous_media_flow.py power_law 0.5 0.8
    python porous_media_flow.py carreau_yasuda 0.02 0.5 5.0 0.8 2.0
    python porous_media_flow.py tbnn init
    python porous_media_flow.py tbnn /path/to/params.pkl --random-seed 42
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
    print("Porous media solver initialized.")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Common configuration for all simulations
COMMON_CONFIG = {
    'domain': ((0, 4.0), (0, 4.0)),  # 4x4 square domain for porous media
    'domain_size': (128, 128),  # 128x128 grid
    'density': 1.0,
    'pressure_gradient': 2.5,
    'dt': 1e-4,
    'inner_steps': 200,
    'outer_steps': 300,
    'helmholtz_solver_type': 'fast_diag',  # Helmholtz solver for nu0_split ('fast_diag' or 'cg')
    'stepper_type': 'nu0_split',  # nu0_split stepper for porous media
    'boundary_type': 'periodic'  # Periodic boundary conditions
}

# =============================================================================
# OBSTACLE SETUP FOR POROUS MEDIA
# =============================================================================

def setup_porous_media_obstacle(domain):
    """Create two circular obstacles in the porous media domain.
    
    Configuration:
        - Obstacle 1: center at (3, 1), radius = 1.0
        - Obstacle 2: center at (1, 3), radius = 0.5
    """
    def param_rot_ellipse(geometry_param, theta):
        A = geometry_param[0]
        B = geometry_param[1] 
        phi = geometry_param[2]
        excc = jnp.sqrt(1-jnp.round((B/A)**2, 6))
        return B/jnp.sqrt(1-(excc*jnp.cos(theta-phi))**2)
    
    # Two obstacles with different sizes and positions
    # Obstacle 1: larger obstacle at (3, 1)
    # Obstacle 2: smaller obstacle at (1, 3)
    particle_geometry_param = jnp.array([
        [1.0, 1.0, 0.0],  # Obstacle 1: radius = 1.0 (circular)
        [0.5, 0.5, 0.0]   # Obstacle 2: radius = 0.5 (circular)
    ])
    
    particle_center_position = jnp.array([
        [3.0, 1.0],  # Obstacle 1 position
        [1.0, 3.0]   # Obstacle 2 position
    ])
    
    displacement_param = jnp.array([
        [0.0, 0.0],
        [0.0, 0.0]
    ])
    
    rotation_param = jnp.array([
        [0.0, 0.0, 0.0, 0],
        [0.0, 0.0, 0.0, 0]
    ])
    
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

def setup_tbnn_model_for_porous_media(domain_size, random_seed=42, **kwargs):
    """Setup TBNN model for porous media flow using default settings.
    
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
            model_info = setup_tbnn_model_for_porous_media(domain_size, random_seed, **tbnn_overrides)
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
            model_info = setup_tbnn_model_for_porous_media(domain_size, random_seed, **tbnn_overrides)
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

def run_porous_media(
    model_name,
    *params,
    show_plots=True,
    save_trajectory=True,
    output_dir=str(REPO_ROOT / 'work' / 'reference_trajectories'),
    pressure_gradient=None,
    helmholtz_solver_type=None,
    dt=None,
    inner_steps=None,
    outer_steps=None,
):
    """
    Run porous media flow simulation for specified rheological model.
    
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
        helmholtz_solver_type: Helmholtz solver type ('fast_diag' or 'cg', default: None uses 'fast_diag')
        dt: Time step override (default: None uses 1e-4)
        inner_steps: Inner steps override (default: None uses 200)
        outer_steps: Outer steps override (default: None uses 300)
    
    Returns:
        Dictionary with simulation results
    """
    print(f"\n{'='*80}")
    print(f"POROUS MEDIA FLOW - {model_name.upper()} MODEL")
    print(f"{'='*80}")
    
    # Get model configuration
    model_config = get_model_config(model_name, *params)
    
    # Handle parameter overrides
    actual_pressure_gradient = pressure_gradient if pressure_gradient is not None else COMMON_CONFIG['pressure_gradient']
    actual_dt = dt if dt is not None else COMMON_CONFIG['dt']
    actual_inner_steps = inner_steps if inner_steps is not None else COMMON_CONFIG['inner_steps']
    actual_outer_steps = outer_steps if outer_steps is not None else COMMON_CONFIG['outer_steps']
    actual_helmholtz_solver_type = helmholtz_solver_type if helmholtz_solver_type is not None else COMMON_CONFIG['helmholtz_solver_type']
    
    print(f"Model: {model_config['description']}")
    print(f"Domain: {COMMON_CONFIG['domain']}, Grid: {COMMON_CONFIG['domain_size']}")
    print(f"Parameters: rho={COMMON_CONFIG['density']}, gradp={actual_pressure_gradient}")
    print(f"Time stepping: dt={actual_dt}, inner={actual_inner_steps}, outer={actual_outer_steps}")
    print(f"Stepper: {COMMON_CONFIG['stepper_type']}")
    print(f"Boundary: {COMMON_CONFIG['boundary_type']}")
    print(f"Helmholtz solver: {actual_helmholtz_solver_type}")
    
    # Create grid and obstacle
    grid = flow_conditions.create_grid(COMMON_CONFIG['domain_size'], COMMON_CONFIG['domain'])
    particles = setup_porous_media_obstacle(COMMON_CONFIG['domain'])
    
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
        'boundary_type': COMMON_CONFIG['boundary_type']  # Periodic boundary conditions
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
            stepper_type=COMMON_CONFIG['stepper_type'],  # nu0_split stepper
            helmholtz_solver_type=actual_helmholtz_solver_type,  # 'fast_diag' or 'cg'
            solver_type='bicgstab',  # Not used for nu0_split, only for fully_implicit
            use_preconditioner=False,  # Not used for nu0_split
            preconditioner_type='jacobi'  # Not used for nu0_split
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
            plot_porous_media_results(results)
        
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
    
    filename = f"porous_media_{model_name}_{params_str}_trajectory.npy"
    filepath = os.path.join(output_dir, filename)
    
    # Save trajectory
    np.save(filepath, np.array(trajectory))
    print(f"Trajectory saved: {filepath}")
    print(f"   Shape: {trajectory.shape}")
    print(f"   Format: (time_steps, velocity_components[vx,vy], grid_x, grid_y)")

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_porous_media_results(results):
    """Create comprehensive visualization of porous media flow."""
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
    
    # Tile the velocity field 3x3 to show periodic nature
    vx_tiled = jnp.tile(vx_final, (3, 3))
    vy_tiled = jnp.tile(vy_final, (3, 3))
    
    # Extended domain for tiled visualization
    domain_width = domain[0][1] - domain[0][0]
    domain_height = domain[1][1] - domain[1][0]
    extended_domain = (
        (domain[0][0] - domain_width, domain[0][1] + domain_width),
        (domain[1][0] - domain_height, domain[1][1] + domain_height)
    )
    
    # Obstacle positions and radii
    obstacles = [
        {'center': (3.0, 1.0), 'radius': 1.0},
        {'center': (1.0, 3.0), 'radius': 0.5}
    ]
    
    # Helper function to add obstacles in periodic pattern
    def add_obstacles_periodic(ax, facecolor='gray', edgecolor='black', alpha=0.3):
        """Add obstacles tiled 3x3 to show periodic nature"""
        for dx in [-domain_width, 0, domain_width]:
            for dy in [-domain_height, 0, domain_height]:
                for obs in obstacles:
                    center = (obs['center'][0] + dx, obs['center'][1] + dy)
                    circle = plt.Circle(center, obs['radius'], 
                                       facecolor=facecolor, edgecolor=edgecolor, 
                                       alpha=alpha, linewidth=1.5)
                    ax.add_patch(circle)
    
    # X-velocity field (tiled)
    im1 = axes[0,0].imshow(vx_tiled.T, origin='lower', cmap='coolwarm', 
                           extent=[extended_domain[0][0], extended_domain[0][1], 
                                  extended_domain[1][0], extended_domain[1][1]],
                           aspect='auto')
    axes[0,0].set_title(f'X-velocity (periodic tiled)\n{model_config["description"]}')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0,0])
    add_obstacles_periodic(axes[0,0])
    
    # Y-velocity field (tiled)
    im2 = axes[0,1].imshow(vy_tiled.T, origin='lower', cmap='coolwarm',
                           extent=[extended_domain[0][0], extended_domain[0][1],
                                  extended_domain[1][0], extended_domain[1][1]],
                           aspect='auto')
    axes[0,1].set_title(f'Y-velocity (periodic tiled)\n{model_config["description"]}')
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0,1])
    add_obstacles_periodic(axes[0,1])
    
    # Velocity magnitude (tiled)
    vel_mag_tiled = jnp.sqrt(vx_tiled**2 + vy_tiled**2)
    im3 = axes[0,2].imshow(vel_mag_tiled.T, origin='lower', cmap='viridis',
                           extent=[extended_domain[0][0], extended_domain[0][1],
                                  extended_domain[1][0], extended_domain[1][1]],
                           aspect='auto')
    axes[0,2].set_title(f'Velocity Magnitude (periodic tiled)\n{model_config["description"]}')
    axes[0,2].set_xlabel('x')
    axes[0,2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[0,2])
    add_obstacles_periodic(axes[0,2])
    
    # Streamlines (tiled)
    try:
        x_tiled = jnp.linspace(extended_domain[0][0], extended_domain[0][1], vx_tiled.shape[0])
        y_tiled = jnp.linspace(extended_domain[1][0], extended_domain[1][1], vy_tiled.shape[1])
        X_tiled, Y_tiled = jnp.meshgrid(x_tiled, y_tiled, indexing='ij')
        
        skip = max(1, vx_tiled.shape[0] // 40)
        X_stream = np.array(X_tiled[::skip, ::skip])
        Y_stream = np.array(Y_tiled[::skip, ::skip])
        vx_stream = np.array(vx_tiled[::skip, ::skip])
        vy_stream = np.array(vy_tiled[::skip, ::skip])
        
        axes[1,0].streamplot(X_stream.T, Y_stream.T, vx_stream.T, vy_stream.T, 
                            density=1.2, color='k', arrowsize=1.0, linewidth=0.8)
        axes[1,0].set_title('Streamlines (periodic tiled)')
        axes[1,0].set_xlabel('x')
        axes[1,0].set_ylabel('y')
        axes[1,0].set_xlim(extended_domain[0])
        axes[1,0].set_ylim(extended_domain[1])
        
        # Add obstacles with gray fill
        add_obstacles_periodic(axes[1,0], facecolor='gray', edgecolor='red', alpha=0.5)
        
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
    
    # Local viscosity field (log scale heatmap, tiled)
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
        
        # Tile the viscosity field
        viscosity_tiled = jnp.tile(viscosity_field, (3, 3))
        
        # Create log-scale viscosity heatmap
        log_viscosity_tiled = jnp.log10(jnp.maximum(viscosity_tiled, 1e-8))  # Avoid log(0)
        
        im4 = axes[1,2].imshow(log_viscosity_tiled.T, origin='lower', cmap='plasma',
                              extent=[extended_domain[0][0], extended_domain[0][1],
                                     extended_domain[1][0], extended_domain[1][1]],
                              aspect='auto')
        axes[1,2].set_title(f'Local Viscosity (log₁₀, periodic tiled)\n{model_config["description"]}')
        axes[1,2].set_xlabel('x')
        axes[1,2].set_ylabel('y')
        cbar = plt.colorbar(im4, ax=axes[1,2])
        cbar.set_label('log₁₀(η)')
        
        # Add obstacles with gray fill
        add_obstacles_periodic(axes[1,2], facecolor='gray', edgecolor='white', alpha=0.5)
        
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
    vel_mag = jnp.sqrt(vx_final**2 + vy_final**2)
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
    parser = argparse.ArgumentParser(description='Porous media flow simulation')
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
    parser.add_argument('--helmholtz-solver-type', type=str, default=None,
                       help='Helmholtz solver type: fast_diag or cg (default: fast_diag)')
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
            print("    python porous_media_flow.py tbnn init")
            print("    python porous_media_flow.py tbnn /path/to/params.pkl")
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
    if args.helmholtz_solver_type is not None:
        print(f"Using Helmholtz solver type: {args.helmholtz_solver_type}")
    if args.dt is not None:
        print(f"Using time step override: {args.dt}")
    if args.inner_steps is not None:
        print(f"Using inner steps override: {args.inner_steps}")
    if args.outer_steps is not None:
        print(f"Using outer steps override: {args.outer_steps}")
    
    results = run_porous_media(
        args.model, 
        *processed_params,
        show_plots=not args.no_plots,
        save_trajectory=True,
        output_dir=args.output_dir,
        pressure_gradient=args.pressure_gradient,
        helmholtz_solver_type=args.helmholtz_solver_type,
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

def run_porous_media_newtonian_demo(viscosity=1.0, pressure_gradient=None, helmholtz_solver_type=None, dt=None, inner_steps=None, outer_steps=None):
    """Quick porous media Newtonian demo."""
    return run_porous_media('newtonian', viscosity, pressure_gradient=pressure_gradient, helmholtz_solver_type=helmholtz_solver_type, dt=dt, inner_steps=inner_steps, outer_steps=outer_steps)

def run_porous_media_power_law_demo(K=0.5, n=0.8, pressure_gradient=None, helmholtz_solver_type=None, dt=None, inner_steps=None, outer_steps=None):
    """Quick porous media Power-Law demo.""" 
    return run_porous_media('power_law', K, n, pressure_gradient=pressure_gradient, helmholtz_solver_type=helmholtz_solver_type, dt=dt, inner_steps=inner_steps, outer_steps=outer_steps)

def run_porous_media_carreau_yasuda_demo(eta_inf=0.02, eta_0=0.5, lambda_=5.0, n=0.8, a=2.0, pressure_gradient=None, helmholtz_solver_type=None, dt=None, inner_steps=None, outer_steps=None):
    """Quick porous media Carreau-Yasuda demo."""
    return run_porous_media('carreau_yasuda', eta_inf, eta_0, lambda_, n, a, pressure_gradient=pressure_gradient, helmholtz_solver_type=helmholtz_solver_type, dt=dt, inner_steps=inner_steps, outer_steps=outer_steps)

def run_porous_media_tbnn_demo(params_path='init', random_seed=42, pressure_gradient=None, helmholtz_solver_type=None, dt=None, inner_steps=None, outer_steps=None, save_trajectory=True, show_plots=True):
    """Quick porous media TBNN demo.
    
    Args:
        params_path: Path to .pkl file or 'init' to initialize fresh (default: 'init')
        random_seed: Random seed for initialization (default: 42)
        pressure_gradient: Pressure gradient override (default: None uses 2.5)
        helmholtz_solver_type: Helmholtz solver type ('fast_diag' or 'cg', default: None uses 'fast_diag')
        dt: Time step override (default: None uses 1e-4)
        inner_steps: Inner steps override (default: None uses 200)
        outer_steps: Outer steps override (default: None uses 300)
        save_trajectory: Whether to save trajectory to .npy file (default: True)
        show_plots: Whether to display plots (default: True)
    
    Example:
        # Initialize fresh TBNN
        results = run_porous_media_tbnn_demo('init', random_seed=1453)
        
        # Load trained TBNN without saving trajectory
        results = run_porous_media_tbnn_demo('/path/to/final_tbnn_params.pkl', save_trajectory=False)
    """
    domain_size = COMMON_CONFIG['domain_size']
    return run_porous_media('tbnn', params_path, domain_size, random_seed, 
                            pressure_gradient=pressure_gradient, helmholtz_solver_type=helmholtz_solver_type, 
                            dt=dt, inner_steps=inner_steps, outer_steps=outer_steps,
                            save_trajectory=save_trajectory, show_plots=show_plots)

def run_demo_comparison(
    ground_truth_model,
    ground_truth_params,
    comparison_model,
    comparison_params,
    domain_size=(128, 128),
    dt=1e-4,
    inner_steps=400,
    outer_steps=200,
    pressure_gradient=10.0,
    num_bins=20,
    save_trajectory=False
):
    """
    Compare two rheological models in porous media flow.
    
    Args:
        ground_truth_model: Model type ('carreau_yasuda', 'power_law', 'newtonian', or 'tbnn')
        ground_truth_params: Parameters for ground truth model
            - For 'carreau_yasuda': (eta_inf, eta_0, lambda, n, a)
            - For 'power_law': (K, n)
            - For 'newtonian': (viscosity,)
            - For 'tbnn': (params_path, random_seed) - domain_size is auto-added
        comparison_model: Model type to compare against ground truth
        comparison_params: Parameters for comparison model (same format as ground_truth_params)
        domain_size: Grid resolution (nx, ny) for both simulations (default: (128, 128))
        dt: Time step (default: 1e-4)
        inner_steps: Inner steps (default: 400)
        outer_steps: Outer steps (default: 200)
        pressure_gradient: Pressure gradient (default: 10.0)
        num_bins: Number of bins for strain rate analysis (default: 20)
        save_trajectory: Whether to save trajectories (default: False)
    
    Returns:
        Dictionary with both results and comparison metrics
    
    Example:
        # Compare TBNN against Carreau-Yasuda at 256x256 resolution
        results = run_demo_comparison(
            ground_truth_model='carreau_yasuda',
            ground_truth_params=(0.02, 1.0, 5.0, 0.7, 2.0),
            comparison_model='tbnn',
            comparison_params=('work/instantaneous_train/<run>/final_tbnn_params.pkl', 42),
            domain_size=(256, 256),
            dt=1e-4,
            inner_steps=400,
            outer_steps=200,
            pressure_gradient=10.0,
            num_bins=20
        )
    """
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON: {comparison_model.upper()} vs {ground_truth_model.upper()} (Ground Truth)")
    print(f"{'='*80}\n")
    print(f"Resolution: {domain_size[0]}x{domain_size[1]}")
    
    # Temporarily override domain size
    original_domain_size = COMMON_CONFIG['domain_size']
    COMMON_CONFIG['domain_size'] = domain_size
    
    try:
        # Process parameters - add domain_size to TBNN params if needed
        def process_params(model_type, params):
            if model_type.lower() == 'tbnn':
                # TBNN params should be (params_path, random_seed) or (params_path, domain_size, random_seed)
                if len(params) == 2:
                    # Add domain_size: (params_path, random_seed) -> (params_path, domain_size, random_seed)
                    return (params[0], domain_size, params[1])
                elif len(params) == 3:
                    # Replace domain_size: (params_path, old_domain, random_seed) -> (params_path, domain_size, random_seed)
                    return (params[0], domain_size, params[2])
                else:
                    return params
            return params
        
        gt_params_processed = process_params(ground_truth_model, ground_truth_params)
        comp_params_processed = process_params(comparison_model, comparison_params)
        
        # Run ground truth simulation
        print(f"Running ground truth model ({ground_truth_model})...")
        gt_results = run_porous_media(
            ground_truth_model, *gt_params_processed,
            dt=dt, inner_steps=inner_steps, outer_steps=outer_steps,
            pressure_gradient=pressure_gradient, save_trajectory=save_trajectory,
            show_plots=False
        )
        
        # Run comparison simulation
        print(f"\nRunning comparison model ({comparison_model})...")
        comp_results = run_porous_media(
            comparison_model, *comp_params_processed,
            dt=dt, inner_steps=inner_steps, outer_steps=outer_steps,
            pressure_gradient=pressure_gradient, save_trajectory=save_trajectory,
            show_plots=False
        )
    finally:
        # Restore original domain size
        COMMON_CONFIG['domain_size'] = original_domain_size
    
    # Create comparison plot
    print(f"\nGenerating comparison plots...")
    comparison_dict = plot_model_comparison_detailed(
        gt_results, comp_results, ground_truth_model, comparison_model, num_bins
    )
    
    return {
        'ground_truth': gt_results,
        'comparison': comp_results,
        'metrics': comparison_dict
    }


def run_porous_media_comparison_demo(include_tbnn=False):
    """Run comparison of rheological models.
    
    Args:
        include_tbnn: If True, include TBNN in comparison (default: False)
    """
    print("Running comparison of rheological models...")
    
    results = {}
    
    # Newtonian
    print("\n" + "="*50)
    results['newtonian'] = run_porous_media('newtonian', 0.5, show_plots=False)
    
    # Power-Law
    print("\n" + "="*50)  
    results['power_law'] = run_porous_media('power_law', 0.5, 0.8, show_plots=False)
    
    # Carreau-Yasuda
    print("\n" + "="*50)
    results['carreau_yasuda'] = run_porous_media('carreau_yasuda', 0.02, 0.5, 5.0, 0.8, 2.0, show_plots=False)
    
    # TBNN (optional)
    if include_tbnn:
        print("\n" + "="*50)
        domain_size = COMMON_CONFIG['domain_size']
        results['tbnn'] = run_porous_media('tbnn', 'init', domain_size, 42, show_plots=False)
    
    # Create comparison plot
    if all(r is not None for r in results.values()):
        plot_model_comparison(results)
    
    return results

def plot_model_comparison_detailed(gt_results, comp_results, gt_name, comp_name, num_bins=20):
    """
    Create detailed comparison plot between two models.
    
    Args:
        gt_results: Ground truth simulation results
        comp_results: Comparison simulation results
        gt_name: Name of ground truth model
        comp_name: Name of comparison model
        num_bins: Number of bins for strain rate analysis
        
    Returns:
        Dictionary with comparison metrics
    """
    # Extract velocity fields
    if hasattr(gt_results['final_result'].velocity[0], 'data'):
        vx_gt = gt_results['final_result'].velocity[0].data
        vy_gt = gt_results['final_result'].velocity[1].data
    else:
        vx_gt = gt_results['final_result'].velocity[0]
        vy_gt = gt_results['final_result'].velocity[1]
    
    if hasattr(comp_results['final_result'].velocity[0], 'data'):
        vx_comp = comp_results['final_result'].velocity[0].data
        vy_comp = comp_results['final_result'].velocity[1].data
    else:
        vx_comp = comp_results['final_result'].velocity[0]
        vy_comp = comp_results['final_result'].velocity[1]
    
    domain = COMMON_CONFIG['domain']
    
    # Tile the velocity fields 3x3 for periodic visualization
    vx_gt_tiled = jnp.tile(vx_gt, (3, 3))
    vx_comp_tiled = jnp.tile(vx_comp, (3, 3))
    vy_gt_tiled = jnp.tile(vy_gt, (3, 3))
    vy_comp_tiled = jnp.tile(vy_comp, (3, 3))
    
    # Extended domain for tiled visualization (shift both axes by -2 to center: -6 to 6 instead of -4 to 8)
    domain_width = domain[0][1] - domain[0][0]
    domain_height = domain[1][1] - domain[1][0]
    x_shift = -2.0  # Shift to center the periodic domain
    y_shift = -2.0  # Shift to center the periodic domain
    extended_domain = (
        (domain[0][0] - domain_width + x_shift, domain[0][1] + domain_width + x_shift),
        (domain[1][0] - domain_height + y_shift, domain[1][1] + domain_height + y_shift)
    )
    
    # Obstacle positions and radii
    obstacles = [
        {'center': (3.0, 1.0), 'radius': 1.0},
        {'center': (1.0, 3.0), 'radius': 0.5}
    ]
    
    # Helper function to add obstacles in periodic pattern
    def add_obstacles_periodic(ax, facecolor='gray', edgecolor='black', alpha=1.0, linewidth=2):
        """Add obstacles tiled 3x3 to show periodic nature"""
        for dx in [-domain_width, 0, domain_width]:
            for dy in [-domain_height, 0, domain_height]:
                for obs in obstacles:
                    center = (obs['center'][0] + dx + x_shift, obs['center'][1] + dy + y_shift)
                    circle = plt.Circle(center, obs['radius'], 
                                       facecolor=facecolor, edgecolor=edgecolor, 
                                       alpha=alpha, linewidth=linewidth)
                    ax.add_patch(circle)
    
    # Calculate absolute differences for Vx and Vy
    vx_diff = vx_comp - vx_gt
    vx_abs_diff = jnp.abs(vx_diff)
    vx_abs_diff_tiled = jnp.tile(vx_abs_diff, (3, 3))
    
    vy_diff = vy_comp - vy_gt
    vy_abs_diff = jnp.abs(vy_diff)
    vy_abs_diff_tiled = jnp.tile(vy_abs_diff, (3, 3))
    
    # Calculate strain rate magnitude: gamma_dot = sqrt(2 * D:D) where D is strain rate tensor
    # For 2D: D_xx = du/dx, D_yy = dv/dy, D_xy = 0.5*(du/dy + dv/dx)
    dx = (domain[0][1] - domain[0][0]) / vx_gt.shape[0]
    dy = (domain[1][1] - domain[1][0]) / vx_gt.shape[1]
    
    # Ground truth strain rate
    dux_dx_gt = jnp.gradient(vx_gt, dx, axis=0)
    duy_dy_gt = jnp.gradient(vy_gt, dy, axis=1)
    dux_dy_gt = jnp.gradient(vx_gt, dy, axis=1)
    duy_dx_gt = jnp.gradient(vy_gt, dx, axis=0)
    
    # Strain rate tensor components
    D_xx_gt = dux_dx_gt
    D_yy_gt = duy_dy_gt
    D_xy_gt = 0.5 * (dux_dy_gt + duy_dx_gt)
    
    # Strain rate magnitude: gamma_dot = sqrt(2 * (D_xx^2 + D_yy^2 + 2*D_xy^2))
    gamma_dot_gt = jnp.sqrt(2 * (D_xx_gt**2 + D_yy_gt**2 + 2*D_xy_gt**2))
    
    # Relative errors in x and y velocities
    relative_error_x = jnp.abs(vx_diff) / (jnp.abs(vx_gt) + 1e-10)
    relative_error_y = jnp.abs(vy_diff) / (jnp.abs(vy_gt) + 1e-10)
    
    # Create mask to exclude points inside obstacles
    # Generate meshgrid of coordinates
    nx, ny = vx_gt.shape
    x_coords = jnp.linspace(domain[0][0], domain[0][1], nx)
    y_coords = jnp.linspace(domain[1][0], domain[1][1], ny)
    X, Y = jnp.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Create mask: True for points OUTSIDE all obstacles
    obstacle_mask = jnp.ones_like(X, dtype=bool)
    for obs in obstacles:
        # Calculate distance from obstacle center
        dist_sq = (X - obs['center'][0])**2 + (Y - obs['center'][1])**2
        # Points inside this obstacle (False means inside)
        outside_this_obstacle = dist_sq > obs['radius']**2
        # Update mask: keep only points outside ALL obstacles
        obstacle_mask = obstacle_mask & outside_this_obstacle
    
    # Create figure with publication-quality settings
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 30,
        'axes.titlesize': 26,
        'xtick.labelsize': 26,
        'ytick.labelsize': 26,
        'legend.fontsize': 22,
        'figure.titlesize': 28
    })
    
    # Calculate common velocity ranges for consistent colorbars
    vx_min = min(float(jnp.min(vx_gt_tiled)), float(jnp.min(vx_comp_tiled)))
    vx_max = max(float(jnp.max(vx_gt_tiled)), float(jnp.max(vx_comp_tiled)))
    vy_min = min(float(jnp.min(vy_gt_tiled)), float(jnp.min(vy_comp_tiled)))
    vy_max = max(float(jnp.max(vy_gt_tiled)), float(jnp.max(vy_comp_tiled)))
    
    # Calculate binned strain rate data first (needed for all plots)
    # Apply obstacle mask before flattening
    gamma_flat = gamma_dot_gt[obstacle_mask].flatten()
    error_x_flat = relative_error_x[obstacle_mask].flatten()
    error_y_flat = relative_error_y[obstacle_mask].flatten()
    
    # Remove invalid values for x errors
    valid_mask_x = jnp.isfinite(gamma_flat) & jnp.isfinite(error_x_flat) & (gamma_flat > 1e-10)
    gamma_valid_x = gamma_flat[valid_mask_x]
    error_x_valid = error_x_flat[valid_mask_x]
    
    # Remove invalid values for y errors
    valid_mask_y = jnp.isfinite(gamma_flat) & jnp.isfinite(error_y_flat) & (gamma_flat > 1e-10)
    gamma_valid_y = gamma_flat[valid_mask_y]
    error_y_valid = error_y_flat[valid_mask_y]
    
    # Create log-spaced bins based on combined data range
    gamma_min = float(min(jnp.min(gamma_valid_x), jnp.min(gamma_valid_y)))
    gamma_max = float(max(jnp.max(gamma_valid_x), jnp.max(gamma_valid_y)))
    bins = jnp.logspace(jnp.log10(gamma_min), jnp.log10(gamma_max), num_bins + 1)
    
    # Bin the x-error data
    bin_centers_x = []
    bin_errors_x = []
    bin_stds_x = []
    
    for i in range(num_bins):
        mask = (gamma_valid_x >= bins[i]) & (gamma_valid_x < bins[i+1])
        if jnp.sum(mask) > 0:
            bin_centers_x.append(jnp.sqrt(bins[i] * bins[i+1]))  # Geometric mean
            bin_errors_x.append(float(jnp.mean(error_x_valid[mask])))
            bin_stds_x.append(float(jnp.std(error_x_valid[mask])))
    
    bin_centers_x = np.array(bin_centers_x)
    bin_errors_x = np.array(bin_errors_x)
    bin_stds_x = np.array(bin_stds_x)
    
    # Bin the y-error data
    bin_centers_y = []
    bin_errors_y = []
    bin_stds_y = []
    
    for i in range(num_bins):
        mask = (gamma_valid_y >= bins[i]) & (gamma_valid_y < bins[i+1])
        if jnp.sum(mask) > 0:
            bin_centers_y.append(jnp.sqrt(bins[i] * bins[i+1]))  # Geometric mean
            bin_errors_y.append(float(jnp.mean(error_y_valid[mask])))
            bin_stds_y.append(float(jnp.std(error_y_valid[mask])))
    
    bin_centers_y = np.array(bin_centers_y)
    bin_errors_y = np.array(bin_errors_y)
    bin_stds_y = np.array(bin_stds_y)
    
    # Figure size for all plots - wider to accommodate colorbar while keeping plot square
    fig_size = (10, 8)  # Width x Height - extra width for colorbar
    dpi = 600
    
    # PLOT 1: Ground truth x-velocity (tiled)
    fig1 = plt.figure(figsize=fig_size, dpi=dpi)
    ax1 = fig1.add_subplot(111)
    im1 = ax1.imshow(vx_gt_tiled.T, origin='lower', cmap='coolwarm',
                     extent=[extended_domain[0][0], extended_domain[0][1], 
                            extended_domain[1][0], extended_domain[1][1]],
                     aspect='equal', vmin=vx_min, vmax=vx_max)
    ax1.set_title('Ground Truth Velocity in x Direction', fontsize=26, fontweight='bold', pad=25)
    ax1.set_xlabel('x', fontsize=30)
    ax1.set_ylabel('y', fontsize=30)
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-6, 6)
    ax1.set_xticks([-6, -4, -2, 0, 2, 4, 6])
    ax1.set_yticks([-6, -4, -2, 0, 2, 4, 6])
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('$u_x$', fontsize=30, rotation=0, labelpad=20)
    cbar1.ax.tick_params(labelsize=26)
    add_obstacles_periodic(ax1, facecolor='gray', edgecolor='black', linewidth=2)
    plt.tight_layout()
    plt.show()
    
    # PLOT 2: Comparison model x-velocity (tiled)
    fig2 = plt.figure(figsize=fig_size, dpi=dpi)
    ax2 = fig2.add_subplot(111)
    im2 = ax2.imshow(vx_comp_tiled.T, origin='lower', cmap='coolwarm',
                     extent=[extended_domain[0][0], extended_domain[0][1],
                            extended_domain[1][0], extended_domain[1][1]],
                     aspect='equal', vmin=vx_min, vmax=vx_max)
    ax2.set_title(f'{comp_name.upper()} Predicted Velocity in x Direction', fontsize=26, fontweight='bold', pad=25)
    ax2.set_xlabel('x', fontsize=30)
    ax2.set_ylabel('y', fontsize=30)
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-6, 6)
    ax2.set_xticks([-6, -4, -2, 0, 2, 4, 6])
    ax2.set_yticks([-6, -4, -2, 0, 2, 4, 6])
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('$u_x$', fontsize=30, rotation=0, labelpad=20)
    cbar2.ax.tick_params(labelsize=26)
    add_obstacles_periodic(ax2, facecolor='gray', edgecolor='black', linewidth=2)
    plt.tight_layout()
    plt.show()
    
    # PLOT 3: Absolute difference in Vx (tiled)
    vmax_abs_x = jnp.percentile(vx_abs_diff_tiled, 95)  # Use 95th percentile for better visualization
    fig3 = plt.figure(figsize=fig_size, dpi=dpi)
    ax3 = fig3.add_subplot(111)
    im3 = ax3.imshow(vx_abs_diff_tiled.T, origin='lower', cmap='YlOrRd',
                     extent=[extended_domain[0][0], extended_domain[0][1],
                            extended_domain[1][0], extended_domain[1][1]],
                     aspect='equal', vmin=0, vmax=vmax_abs_x)
    ax3.set_title('Absolute Difference in x Direction', fontsize=26, fontweight='bold', pad=25)
    ax3.set_xlabel('x', fontsize=30)
    ax3.set_ylabel('y', fontsize=30)
    ax3.set_xlim(-6, 6)
    ax3.set_ylim(-6, 6)
    ax3.set_xticks([-6, -4, -2, 0, 2, 4, 6])
    ax3.set_yticks([-6, -4, -2, 0, 2, 4, 6])
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('$u_x$', fontsize=30, rotation=0, labelpad=30)
    cbar3.ax.tick_params(labelsize=26)
    add_obstacles_periodic(ax3, facecolor='gray', edgecolor='white', linewidth=2)
    plt.tight_layout()
    plt.show()
    
    # PLOT 4: Ground truth y-velocity (tiled)
    fig4 = plt.figure(figsize=fig_size, dpi=dpi)
    ax4 = fig4.add_subplot(111)
    im4 = ax4.imshow(vy_gt_tiled.T, origin='lower', cmap='coolwarm',
                     extent=[extended_domain[0][0], extended_domain[0][1], 
                            extended_domain[1][0], extended_domain[1][1]],
                     aspect='equal', vmin=vy_min, vmax=vy_max)
    ax4.set_title('Ground Truth Velocity in y Direction', fontsize=26, fontweight='bold', pad=25)
    ax4.set_xlabel('x', fontsize=30)
    ax4.set_ylabel('y', fontsize=30)
    ax4.set_xlim(-6, 6)
    ax4.set_ylim(-6, 6)
    ax4.set_xticks([-6, -4, -2, 0, 2, 4, 6])
    ax4.set_yticks([-6, -4, -2, 0, 2, 4, 6])
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label('$u_y$', fontsize=30, rotation=0, labelpad=20)
    cbar4.ax.tick_params(labelsize=26)
    add_obstacles_periodic(ax4, facecolor='gray', edgecolor='black', linewidth=2)
    plt.tight_layout()
    plt.show()
    
    # PLOT 5: Comparison model y-velocity (tiled)
    fig5 = plt.figure(figsize=fig_size, dpi=dpi)
    ax5 = fig5.add_subplot(111)
    im5 = ax5.imshow(vy_comp_tiled.T, origin='lower', cmap='coolwarm',
                     extent=[extended_domain[0][0], extended_domain[0][1],
                            extended_domain[1][0], extended_domain[1][1]],
                     aspect='equal', vmin=vy_min, vmax=vy_max)
    ax5.set_title(f'{comp_name.upper()} Predicted Velocity in y Direction', fontsize=26, fontweight='bold', pad=25)
    ax5.set_xlabel('x', fontsize=30)
    ax5.set_ylabel('y', fontsize=30)
    ax5.set_xlim(-6, 6)
    ax5.set_ylim(-6, 6)
    ax5.set_xticks([-6, -4, -2, 0, 2, 4, 6])
    ax5.set_yticks([-6, -4, -2, 0, 2, 4, 6])
    cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cbar5.set_label('$u_y$', fontsize=30, rotation=0, labelpad=20)
    cbar5.ax.tick_params(labelsize=26)
    add_obstacles_periodic(ax5, facecolor='gray', edgecolor='black', linewidth=2)
    plt.tight_layout()
    plt.show()
    
    # PLOT 6: Absolute difference in Vy (tiled)
    vmax_abs_y = jnp.percentile(vy_abs_diff_tiled, 95)  # Use 95th percentile for better visualization
    fig6 = plt.figure(figsize=fig_size, dpi=dpi)
    ax6 = fig6.add_subplot(111)
    im6 = ax6.imshow(vy_abs_diff_tiled.T, origin='lower', cmap='YlOrRd',
                     extent=[extended_domain[0][0], extended_domain[0][1],
                            extended_domain[1][0], extended_domain[1][1]],
                     aspect='equal', vmin=0, vmax=vmax_abs_y)
    ax6.set_title('Absolute Difference in y Direction', fontsize=26, fontweight='bold', pad=25)
    ax6.set_xlabel('x', fontsize=30)
    ax6.set_ylabel('y', fontsize=30)
    ax6.set_xlim(-6, 6)
    ax6.set_ylim(-6, 6)
    ax6.set_xticks([-6, -4, -2, 0, 2, 4, 6])
    ax6.set_yticks([-6, -4, -2, 0, 2, 4, 6])
    cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    cbar6.set_label('$u_y$', fontsize=30, rotation=0, labelpad=30)
    cbar6.ax.tick_params(labelsize=26)
    add_obstacles_periodic(ax6, facecolor='gray', edgecolor='white', linewidth=2)
    plt.tight_layout()
    plt.show()
    
    # PLOT 7: Relative error vs strain rate for both x and y (binned) - keep this square too
    fig7 = plt.figure(figsize=(8, 8), dpi=dpi)
    ax7 = fig7.add_subplot(111)
    
    # Calculate median errors from binned data
    median_error_x = np.median(bin_errors_x)
    median_error_y = np.median(bin_errors_y)
    
    # Plot y errors first (larger error bars) so x errors are visible on top
    ax7.errorbar(bin_centers_y, bin_errors_y, yerr=bin_stds_y, 
                 fmt='s-', linewidth=3, markersize=10, capsize=7, capthick=2,
                 label=f'$u_y$, median error = {median_error_y:.4f}', color='#ff7f0e')
    ax7.errorbar(bin_centers_x, bin_errors_x, yerr=bin_stds_x, 
                 fmt='o-', linewidth=3, markersize=10, capsize=7, capthick=2,
                 label=f'$u_x$, median error = {median_error_x:.4f}', color='#1f77b4')
    ax7.set_xscale('log')
    ax7.set_yscale('log')
    ax7.set_xlabel(r'Local Strain Rate $\dot{\gamma}$', fontsize=30)
    ax7.set_ylabel('Relative Error', fontsize=30)
    ax7.set_title(f'Relative Error vs Strain Rate', fontsize=26, fontweight='bold', pad=25)
    
    # Set y-axis limits for better spacing
    ax7.set_ylim(1e-4, 5e-1)
    
    # Remove grid
    ax7.grid(False)
    
    # Configure ticks: inside, on both sides, keep size and frequency
    ax7.tick_params(axis='both', which='major', labelsize=26, width=1.5, length=8,
                    direction='in', top=True, right=True)
    ax7.tick_params(axis='both', which='minor', labelsize=20, width=1, length=4,
                    direction='in', top=True, right=True)
    
    # Legend in bottom left
    ax7.legend(loc='lower left', fontsize=20, frameon=True, fancybox=True, shadow=True)
    
    ax7.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    
    # Compute summary statistics (using only points outside obstacles)
    relative_error_x_masked = relative_error_x[obstacle_mask]
    relative_error_y_masked = relative_error_y[obstacle_mask]
    vx_diff_masked = vx_diff[obstacle_mask]
    vy_diff_masked = vy_diff[obstacle_mask]
    
    mean_error_x = float(jnp.mean(relative_error_x_masked))
    max_error_x = float(jnp.max(relative_error_x_masked))
    median_error_x = float(jnp.median(relative_error_x_masked))
    rmse_x = float(jnp.sqrt(jnp.mean(vx_diff_masked**2)))
    
    mean_error_y = float(jnp.mean(relative_error_y_masked))
    max_error_y = float(jnp.max(relative_error_y_masked))
    median_error_y = float(jnp.median(relative_error_y_masked))
    rmse_y = float(jnp.sqrt(jnp.mean(vy_diff_masked**2)))
    
    print(f"\nCOMPARISON STATISTICS (excluding obstacle regions):")
    print(f"   Ground Truth: {gt_name}")
    print(f"   Comparison: {comp_name}")
    print(f"\n   Vx Errors:")
    print(f"     Mean Relative Error: {mean_error_x:.4f} ({mean_error_x*100:.2f}%)")
    print(f"     Median Relative Error: {median_error_x:.4f} ({median_error_x*100:.2f}%)")
    print(f"     Max Relative Error: {max_error_x:.4f} ({max_error_x*100:.2f}%)")
    print(f"     RMSE: {rmse_x:.6f}")
    print(f"\n   Vy Errors:")
    print(f"     Mean Relative Error: {mean_error_y:.4f} ({mean_error_y*100:.2f}%)")
    print(f"     Median Relative Error: {median_error_y:.4f} ({median_error_y*100:.2f}%)")
    print(f"     Max Relative Error: {max_error_y:.4f} ({max_error_y*100:.2f}%)")
    print(f"     RMSE: {rmse_y:.6f}")
    print(f"\n   Strain Rate Range: [{gamma_min:.2e}, {gamma_max:.2e}] 1/s")
    print(f"   Points analyzed: {jnp.sum(obstacle_mask)} / {obstacle_mask.size} ({100*jnp.sum(obstacle_mask)/obstacle_mask.size:.1f}%)")
    
    return {
        'mean_error_x': mean_error_x,
        'median_error_x': median_error_x,
        'max_error_x': max_error_x,
        'rmse_x': rmse_x,
        'mean_error_y': mean_error_y,
        'median_error_y': median_error_y,
        'max_error_y': max_error_y,
        'rmse_y': rmse_y,
        'bin_centers_x': bin_centers_x,
        'bin_errors_x': bin_errors_x,
        'bin_stds_x': bin_stds_x,
        'bin_centers_y': bin_centers_y,
        'bin_errors_y': bin_errors_y,
        'bin_stds_y': bin_stds_y,
        'strain_rate_range': (gamma_min, gamma_max)
    }


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

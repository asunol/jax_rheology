#!/usr/bin/env python
"""
Channel Flow with Constriction - Multiple Rheological Models

This script simulates channel flow with two semicircular obstacles at the walls creating
a constriction in the center, supporting Newtonian, Power-Law, and Carreau-Yasuda rheological models.

USAGE:
Notebook:
    from channel_constriction_flow import *
    
    # Newtonian flow
    run_channel_constriction('newtonian', 1.0)
    
    # Power-Law flow  
    run_channel_constriction('power_law', 0.5, 0.8)  # K=0.5, n=0.8
    
    # Carreau-Yasuda flow
    run_channel_constriction('carreau_yasuda', 0.02, 0.5, 5.0, 0.8, 2.0)  # eta_inf, eta_0, lambda, n, a

Command line:
    python channel_constriction_flow.py newtonian 1.0
    python channel_constriction_flow.py power_law 0.5 0.8
    python channel_constriction_flow.py carreau_yasuda 0.02 0.5 5.0 0.8 2.0
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
from jax_ib.base import particle_class as pc, grids, kinematics as ks
from jax_ib.base import advection, diffusion
from jax_rheology.solvers import steppers as equations_rheology
from jax_rheology.solvers import pressure
import jax_cfd.base as cfd
import jax_ib.penalty.util_funs

# Disable JIT for debugging if needed
jax.config.update('jax_disable_jit', False)

if _VERBOSE:
    print("Constriction channel solver initialized.")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_extract_velocity_data(velocity_component):
    """
    Safely extract velocity data from either GridVariable/GridArray or bare JAX Array.
    
    This handles the case where velocity components might be returned as:
    1. GridVariable with .array.data 
    2. GridArray with .data
    3. Bare JAX Array (no .data attribute)
    """
    if hasattr(velocity_component, 'data'):
        # GridArray case
        return velocity_component.data
    elif hasattr(velocity_component, 'array') and hasattr(velocity_component.array, 'data'):
        # GridVariable case 
        return velocity_component.array.data
    else:
        # Bare JAX Array case
        return velocity_component

# =============================================================================
# CONFIGURATION
# =============================================================================

# Common configuration for all simulations
COMMON_CONFIG = {
    'domain': ((0, 8.0), (0, 4.0)),  # 8x4 rectangular channel
    'domain_size': (256, 128),  # 128x128 grid
    'density': 1.0,
    'pressure_gradient': 2.5,
    'dt': 1e-4,
    'inner_steps': 400,
    'outer_steps': 300,
    'solver_type': 'bicgstab',  # Always use BiCGSTAB as requested
    'stepper_type': 'fully_implicit'
}

# =============================================================================
# CONSTRICTION SETUP FOR CHANNEL FLOW
# =============================================================================

def setup_channel_constriction(domain):
    """Create two semicircular obstacles at the walls to form a constriction."""
    def param_rot_ellipse(geometry_param, theta):
        A = geometry_param[0]
        B = geometry_param[1] 
        phi = geometry_param[2]
        excc = jnp.sqrt(1-jnp.round((B/A)**2, 6))
        return B/jnp.sqrt(1-(excc*jnp.cos(theta-phi))**2)
    
    # Center of channel in x-direction
    center_x = (domain[0][1] + domain[0][0]) / 2  # x = 4.0
    radius = 1.5  # Radius of semicircles
    
    # Bottom semicircle: center at (4.0, 0.0), creates obstacle from y=0 to y=1.5
    bottom_center_y = 0.0
    
    # Top semicircle: center at (4.0, 4.0), creates obstacle from y=2.5 to y=4.0
    top_center_y = domain[1][1]  # y = 4.0
    
    # Create two circular particles positioned to act as semicircles
    particle_geometry_param = jnp.array([
        [radius, radius, 0.0],  # Bottom semicircle
        [radius, radius, 0.0]   # Top semicircle
    ])
    
    particle_center_position = jnp.array([
        [center_x, bottom_center_y],  # Bottom semicircle center
        [center_x, top_center_y]      # Top semicircle center
    ])
    
    displacement_param = jnp.array([
        [0.0, 0.0],  # Bottom semicircle
        [0.0, 0.0]   # Top semicircle
    ])
    
    rotation_param = jnp.array([
        [0.0, 0.0, 0.0, 0],  # Bottom semicircle
        [0.0, 0.0, 0.0, 0]   # Top semicircle
    ])
    
    mygrids = pc.Grid1d(100, domain=(0, 2*jnp.pi))
    
    particles = pc.particle(
        particle_center_position, particle_geometry_param,
        displacement_param, rotation_param, mygrids,
        param_rot_ellipse, ks.displacement, ks.rotation
    )
    
    return particles

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
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: newtonian, power_law, carreau_yasuda")

# =============================================================================
# SIMULATION RUNNER
# =============================================================================

def run_channel_constriction(
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
    Run channel flow with constriction simulation for specified rheological model.
    
    Args:
        model_name: 'newtonian', 'power_law', or 'carreau_yasuda'
        *params: Model-specific parameters
            - Newtonian: viscosity
            - Power-Law: K, n  
            - Carreau-Yasuda: eta_inf, eta_0, lambda, n, a
        show_plots: Whether to display plots (default: True)
        save_trajectory: Whether to save trajectory to .npy file (default: True)
        output_dir: Directory to save trajectory file (default: work/reference_trajectories)
        pressure_gradient: Pressure gradient to override default (default: None uses 2.5)
        preconditioner: Preconditioner type ('helmholtz', 'jacobi', 'None', or None for no preconditioning)
        dt: Time step override (default: None uses 1e-4)
        inner_steps: Inner steps override (default: None uses 200)
        outer_steps: Outer steps override (default: None uses 300)
    
    Returns:
        Dictionary with simulation results
    """
    print(f"\n{'='*80}")
    print(f"CHANNEL FLOW WITH CONSTRICTION - {model_name.upper()} MODEL")
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
    
    # Create grid and constriction obstacles
    grid = flow_conditions.create_grid(COMMON_CONFIG['domain_size'], COMMON_CONFIG['domain'])
    particles = setup_channel_constriction(COMMON_CONFIG['domain'])
    
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
    
    # Create nu0 update function
    nu0_update_fn = models.create_dynamic_nu0_fn(
        model_type=model_config['model_type'],
        strategy='max', 
        C=1.0
    )
    
    # Run simulation
    print("\nRunning simulation...")
    start_time = time.time()
    
    try:
        final_result, trajectory, perm_f = forward_simulation.forward_fluid_simulation(
            flow_cond=flow_cond,
            flattened_params=model_config['model_params'],
            particles=particles,
            stress_forcing_fn=model_config['forcing_fn'],
            model=None,
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
        vx_final = safe_extract_velocity_data(final_result.velocity[0])
        vy_final = safe_extract_velocity_data(final_result.velocity[1])
        has_nan = jnp.any(jnp.isnan(vx_final)) or jnp.any(jnp.isnan(vy_final))
        
        if has_nan:
            print("WARNING: Simulation contains NaN values!")
            return None
        
        # Print velocity ranges
        print(f"Final vx range: [{float(jnp.min(vx_final)):.6f}, {float(jnp.max(vx_final)):.6f}]")
        print(f"Final vy range: [{float(jnp.min(vy_final)):.6f}, {float(jnp.max(vy_final)):.6f}]")
        
        # Create trajectory array in format expected by loss functions
        # Format: (time_steps, 2, grid_x, grid_y) where axis 1 contains [vx, vy]
        trajectory_array = jnp.stack([safe_extract_velocity_data(trajectory[0]), safe_extract_velocity_data(trajectory[1])], axis=1)
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
            plot_channel_constriction_results(results)
        
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
    
    filename = f"channel_constriction_{model_name}_{params_str}_trajectory.npy"
    filepath = os.path.join(output_dir, filename)
    
    # Save trajectory
    np.save(filepath, np.array(trajectory))
    print(f"Trajectory saved: {filepath}")
    print(f"   Shape: {trajectory.shape}")
    print(f"   Format: (time_steps, velocity_components[vx,vy], grid_x, grid_y)")

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_channel_constriction_results(results):
    """Create comprehensive visualization of channel constriction flow."""
    final_result = results['final_result']
    model_config = results['model_config']
    
    vx_final = safe_extract_velocity_data(final_result.velocity[0])
    vy_final = safe_extract_velocity_data(final_result.velocity[1])
    
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
        
        # Add constriction obstacles outline
        center_x = 4.0
        radius = 1.5
        
        # Bottom semicircle outline (only the upper half that's in domain)
        theta_bottom = np.linspace(0, np.pi, 50)
        x_bottom = center_x + radius * np.cos(theta_bottom)
        y_bottom = 0.0 + radius * np.sin(theta_bottom)
        axes[1,0].plot(x_bottom, y_bottom, 'r-', linewidth=2, label='Constriction')
        
        # Top semicircle outline (only the lower half that's in domain) 
        theta_top = np.linspace(np.pi, 2*np.pi, 50)
        x_top = center_x + radius * np.cos(theta_top)
        y_top = 4.0 + radius * np.sin(theta_top)
        axes[1,0].plot(x_top, y_top, 'r-', linewidth=2)
        
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
            # Handle the case where original_vx might not have offset/grid attributes
            if hasattr(original_vx, 'offset') and hasattr(original_vx, 'grid'):
                v = [ib.grids.GridVariable(ib.grids.GridArray(vx_final, original_vx.offset, original_vx.grid), original_vx.bc),
                     ib.grids.GridVariable(ib.grids.GridArray(vy_final, original_vy.offset, original_vy.grid), original_vy.bc)]
            elif hasattr(original_vx, 'array'):
                # GridVariable case
                v = [ib.grids.GridVariable(ib.grids.GridArray(vx_final, original_vx.array.offset, original_vx.array.grid), original_vx.bc),
                     ib.grids.GridVariable(ib.grids.GridArray(vy_final, original_vy.array.offset, original_vy.array.grid), original_vy.bc)]
            else:
                # Fallback: create minimal velocity structure for viscosity calculation
                # This may not work perfectly but prevents crashes
                v = [vx_final, vy_final]
            
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
        
        # Add constriction obstacles outline
        center_x = 4.0
        radius = 1.5
        
        # Bottom semicircle outline
        theta_bottom = np.linspace(0, np.pi, 50)
        x_bottom = center_x + radius * np.cos(theta_bottom)
        y_bottom = 0.0 + radius * np.sin(theta_bottom)
        axes[1,2].plot(x_bottom, y_bottom, 'white', linewidth=2)
        
        # Top semicircle outline
        theta_top = np.linspace(np.pi, 2*np.pi, 50)
        x_top = center_x + radius * np.cos(theta_top)
        y_top = 4.0 + radius * np.sin(theta_top)
        axes[1,2].plot(x_top, y_top, 'white', linewidth=2)
        
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
            if hasattr(original_vx, 'offset') and hasattr(original_vx, 'grid'):
                v = [ib.grids.GridVariable(ib.grids.GridArray(vx_final, original_vx.offset, original_vx.grid), original_vx.bc),
                     ib.grids.GridVariable(ib.grids.GridArray(vy_final, original_vy.offset, original_vy.grid), original_vy.bc)]
            elif hasattr(original_vx, 'array'):
                v = [ib.grids.GridVariable(ib.grids.GridArray(vx_final, original_vx.array.offset, original_vx.array.grid), original_vx.bc),
                     ib.grids.GridVariable(ib.grids.GridArray(vy_final, original_vy.array.offset, original_vy.array.grid), original_vy.bc)]
            else:
                v = [vx_final, vy_final]
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
    
    # Print constriction-specific statistics
    print(f"\nCONSTRICTION ANALYSIS:")
    # Velocity at throat (center of constriction)
    center_x_idx = vx_final.shape[0] // 2  # x = 4.0
    throat_start_idx = int(1.5 / 4.0 * vx_final.shape[1])  # y = 1.5
    throat_end_idx = int(2.5 / 4.0 * vx_final.shape[1])    # y = 2.5
    throat_velocities = vx_final[center_x_idx, throat_start_idx:throat_end_idx]
    max_throat_vel = float(jnp.max(throat_velocities))
    mean_throat_vel = float(jnp.mean(throat_velocities))
    
    print(f"   Throat max velocity: {max_throat_vel:.6f}")
    print(f"   Throat mean velocity: {mean_throat_vel:.6f}")
    print(f"   Constriction height: 1.0 (from y=1.5 to y=2.5)")
    print(f"   Area ratio: 0.25 (constriction/full channel)")
    
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
    parser = argparse.ArgumentParser(description='Channel flow with constriction simulation')
    parser.add_argument('--verbose', action='store_true',
                        help='Print import and environment banners at startup.')
    parser.add_argument('model', choices=['newtonian', 'power_law', 'carreau_yasuda'],
                       help='Rheological model type')
    parser.add_argument('params', nargs='+', type=float,
                       help='Model parameters')
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
    
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(parser)
    
    # Validate parameter count
    expected_params = {
        'newtonian': 1,
        'power_law': 2, 
        'carreau_yasuda': 5
    }
    
    if len(args.params) != expected_params[args.model]:
        print(f"Error: {args.model} model requires {expected_params[args.model]} parameters")
        print("  newtonian: viscosity")
        print("  power_law: K, n")
        print("  carreau_yasuda: eta_inf, eta_0, lambda, n, a")
        sys.exit(1)
    
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
    
    results = run_channel_constriction(
        args.model, 
        *args.params,
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
    return run_channel_constriction('newtonian', viscosity, pressure_gradient=pressure_gradient, preconditioner=preconditioner, dt=dt, inner_steps=inner_steps, outer_steps=outer_steps)

def run_power_law_demo(K=0.5, n=0.8, pressure_gradient=None, preconditioner=None, dt=None, inner_steps=None, outer_steps=None):
    """Quick Power-Law demo.""" 
    return run_channel_constriction('power_law', K, n, pressure_gradient=pressure_gradient, preconditioner=preconditioner, dt=dt, inner_steps=inner_steps, outer_steps=outer_steps)

def run_carreau_yasuda_demo(eta_inf=0.02, eta_0=0.5, lambda_=5.0, n=0.8, a=2.0, pressure_gradient=None, preconditioner=None, dt=None, inner_steps=None, outer_steps=None):
    """Quick Carreau-Yasuda demo."""
    return run_channel_constriction('carreau_yasuda', eta_inf, eta_0, lambda_, n, a, pressure_gradient=pressure_gradient, preconditioner=preconditioner, dt=dt, inner_steps=inner_steps, outer_steps=outer_steps)

def run_comparison_demo():
    """Run comparison of all three models with similar rheological properties."""
    print("Running comparison of all three rheological models...")
    
    results = {}
    
    # Newtonian
    print("\n" + "="*50)
    results['newtonian'] = run_channel_constriction('newtonian', 0.5, show_plots=False)
    
    # Power-Law
    print("\n" + "="*50)  
    results['power_law'] = run_channel_constriction('power_law', 0.5, 0.8, show_plots=False)
    
    # Carreau-Yasuda
    print("\n" + "="*50)
    results['carreau_yasuda'] = run_channel_constriction('carreau_yasuda', 0.02, 0.5, 5.0, 0.8, 2.0, show_plots=False)
    
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
            
        vx_final = safe_extract_velocity_data(result['final_result'].velocity[0])
        vy_final = safe_extract_velocity_data(result['final_result'].velocity[1])
        vel_mag = jnp.sqrt(vx_final**2 + vy_final**2)
        
        # Velocity magnitude fields
        im = axes[0, i].imshow(vel_mag.T, origin='lower', cmap='viridis',
                              extent=[domain[0][0], domain[0][1], domain[1][0], domain[1][1]],
                              aspect='auto')
        axes[0, i].set_title(f'{model_name.title()}\n{result["model_config"]["description"]}')
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('y')
        plt.colorbar(im, ax=axes[0, i])
        
        # Velocity profiles at channel center (constriction)
        mid_x = vx_final.shape[0] // 2
        profile = vx_final[mid_x, :]
        axes[1, 1].plot(profile, y_coords, color=colors[model_name], 
                       linewidth=2, label=model_name.title())
    
    axes[1, 1].set_xlabel('vx velocity')
    axes[1, 1].set_ylabel('y position')
    axes[1, 1].set_title('Velocity Profile Comparison (Constriction Center)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Hide unused subplots
    axes[1, 0].set_visible(False)
    axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

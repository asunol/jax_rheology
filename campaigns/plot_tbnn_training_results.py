#!/usr/bin/env python
"""Diagnostic plots for a completed instantaneous training run.

Reads one run directory written by the instantaneous trainer and draws the
per-run figures used to judge the fit. Nothing here trains or differentiates.

    from plot_tbnn_training_results import plot_individual_flow_fields

    result = plot_individual_flow_fields(run_dir)                       # draw
    result = plot_individual_flow_fields(run_dir, save_fig=True,
                                         show_fig=False)               # write

Thirteen figures are returned, keyed in the ``figures`` dict and written as
``flow_field_NN_<name>.png``:

     1. reference_ux      ground-truth u_x
     2. reference_uy      ground-truth u_y
     3. trained_ux        u_x from the trained closure
     4. trained_uy        u_y from the trained closure
     5. error_ux          percentage error in u_x
     6. error_uy          percentage error in u_y
     7. abs_diff_ux       absolute difference in u_x
     8. abs_diff_uy       absolute difference in u_y
     9. strain_rate       local strain rate of the reference field
    10. viscosity         local viscosity of the reference fluid
    11. loss_history      loss against iteration
    12. throat_profiles   velocity profiles across the constriction throat
    13. binned_rmse       relative error binned by local strain rate
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import jax
import jax.numpy as jnp
from jax import vmap

# Set publication-quality defaults (slightly smaller than porous_media_flow.py)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['font.size'] = 18
rcParams['axes.labelsize'] = 26
rcParams['axes.titlesize'] = 22
rcParams['xtick.labelsize'] = 22
rcParams['ytick.labelsize'] = 22
rcParams['legend.fontsize'] = 19
rcParams['figure.titlesize'] = 24

# PIV operator: one copy in jax_rheology.training.observation (trainer canonical).
from jax_rheology.training.observation import piv_downsample_THW, add_piv_noise_jax

def parse_piv_params_from_summary(results_dir):
    """
    Parse PIV parameters from iteration_summary_constriction.txt.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Dictionary with PIV parameters, or None if file not found or no PIV used
    """
    summary_file = None
    for name in ('iteration_summary_constriction.txt', 'iteration_summary.txt'):
        cand = os.path.join(results_dir, name)
        if os.path.exists(cand):
            summary_file = cand
            break

    if summary_file is None:
        return None
    
    piv_params = {}
    in_piv_section = False
    
    with open(summary_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if 'PIV resolution and noise:' in line:
                in_piv_section = True
                continue
            
            if in_piv_section:
                # End of section
                if line.startswith('Model architecture:') or line.startswith('2-stage training:'):
                    break
                
                # Parse parameters
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'resolution_piv':
                        piv_params['resolution_piv'] = (value.lower() == 'true')
                    elif key == 'add_piv_noise':
                        piv_params['add_piv_noise'] = (value.lower() == 'true')
                    elif key == 'piv_W_win':
                        # Parse tuple like "(32, 16)"
                        piv_params['piv_W_win'] = eval(value)
                    elif key == 'piv_overlap':
                        piv_params['piv_overlap'] = float(value)
                    elif key == 'piv_kernel':
                        piv_params['piv_kernel'] = value
                    elif key == 'piv_noise_p_percent':
                        piv_params['piv_noise_p_percent'] = float(value)
                    elif key == 'piv_noise_corr_frac':
                        piv_params['piv_noise_corr_frac'] = float(value)
                    elif key == 'piv_noise_beta_grad':
                        piv_params['piv_noise_beta_grad'] = float(value)
                    elif key == 'piv_noise_use_bias':
                        piv_params['piv_noise_use_bias'] = (value.lower() == 'true')
                    elif key == 'piv_noise_seed':
                        piv_params['piv_noise_seed'] = int(value)
    
    # Check if PIV was actually used
    if piv_params.get('resolution_piv') and piv_params.get('add_piv_noise'):
        return piv_params
    else:
        return None


def reconstruct_noisy_reference(ref_vx, ref_vy, piv_params, domain, domain_size):
    """
    Reconstruct noisy reference data by applying PIV downsampling and noise.
    EXACTLY matches the logic in parse_job_folders.py
    
    Args:
        ref_vx, ref_vy: Clean reference velocity fields as loaded from disk
        piv_params: Dictionary with PIV parameters
        domain: Domain bounds ((x_min, x_max), (y_min, y_max))
        domain_size: Grid resolution (nx, ny)
        
    Returns:
        Tuple of (noisy_vx, noisy_vy) in (H_piv, W_piv) format (2D, no time), or None if failed
    """
    try:
        # Extract parameters
        W_x, W_y = piv_params['piv_W_win']
        piv_overlap = piv_params['piv_overlap']
        piv_kernel = piv_params['piv_kernel']
        piv_noise_p_percent = piv_params['piv_noise_p_percent']
        piv_noise_corr_frac = piv_params['piv_noise_corr_frac']
        piv_noise_beta_grad = piv_params['piv_noise_beta_grad']
        piv_noise_use_bias = piv_params['piv_noise_use_bias']
        piv_noise_seed = piv_params['piv_noise_seed']
        
        x_min, x_max = domain[0]
        y_min, y_max = domain[1]
        nx, ny = domain_size
        
        print(f"   Reconstructing noisy reference with PIV parameters:")
        print(f"      Window: {W_x}x{W_y}, Overlap: {piv_overlap}, Noise: {piv_noise_p_percent}%, seed={piv_noise_seed}")
        print(f"      Input shape: ref_vx={ref_vx.shape}, ref_vy={ref_vy.shape}")
        
        # CRITICAL: Transpose if needed (EXACTLY like parse_job_folders.py lines 413-417)
        # jax-cfd outputs (nx, ny) but we need (ny, nx) = (H, W) for PIV
        if ref_vx.shape[0] == nx and ref_vx.shape[1] == ny:
            ref_vx = ref_vx.T  # (256, 128) -> (128, 256) = (H, W)
            ref_vy = ref_vy.T
            print(f"      Transposed to (H, W) format: {ref_vx.shape}")
        
        # Convert to JAX arrays and add time dimension (EXACTLY like parse_job_folders.py line 420-421)
        ref_vx_jax = jnp.array(ref_vx)[None, ...]  # (1, H, W)
        ref_vy_jax = jnp.array(ref_vy)[None, ...]
        
        # Calculate stride (EXACTLY like parse_job_folders.py lines 426-427)
        s_x = max(1, int(round(W_x * (1.0 - piv_overlap))))
        s_y = max(1, int(round(W_y * (1.0 - piv_overlap))))
        
        Lx = float(x_max - x_min)
        Ly = float(y_max - y_min)
        
        print(f"      Stride: {s_x}x{s_y}")
        print(f"      Domain: Lx={Lx}, Ly={Ly}")
        
        # Apply PIV downsampling (EXACTLY like parse_job_folders.py lines 437-440)
        ref_piv_x, ref_piv_y, x_c, y_c = piv_downsample_THW(
            ref_vx_jax, ref_vy_jax, W_x, W_y, s_x, s_y, 
            x_min, y_min, Lx, Ly, kernel=piv_kernel
        )
        
        print(f"      Downsampled: {ref_vx.shape} -> {ref_piv_x.shape}")
        print(f"      Reduction factor: {(ref_vx.shape[0]*ref_vx.shape[1])/(ref_piv_x.shape[1]*ref_piv_x.shape[2]):.1f}x")
        
        # Add PIV noise if specified (EXACTLY like parse_job_folders.py lines 446-472)
        if piv_noise_p_percent > 0.0:
            print(f"       Adding PIV noise...")
            print(f"      Noise level: {piv_noise_p_percent}% of U95")
            
            # Use the seed from the summary file for exact reproducibility
            key_noise = jax.random.PRNGKey(piv_noise_seed)
            
            # Compute U95 (EXACTLY like parse lines 452-455)
            ref_speed = jnp.sqrt(ref_piv_x**2 + ref_piv_y**2)
            u95 = jnp.asarray(jnp.percentile(ref_speed, 95.0), jnp.float32)
            sigma_base = (jnp.float32(piv_noise_p_percent) / 100.0) * (u95 + 1e-6)
            
            # Get full grid shape (EXACTLY like parse lines 457-458)
            # ref_vx is now (H, W) after transpose
            H_full = ref_vx.shape[0]
            W_full = ref_vx.shape[1]
            
            # Add noise (EXACTLY like parse lines 460-470)
            ref_piv_x, ref_piv_y, _ = add_piv_noise_jax(
                ref_piv_x, ref_piv_y,
                W_x=W_x, W_y=W_y, s_x=s_x, s_y=s_y,
                Lx=Lx, Ly=Ly,
                key=key_noise,
                corr_frac=piv_noise_corr_frac,
                sigma_base=float(sigma_base),
                beta_grad=piv_noise_beta_grad,
                use_bias=piv_noise_use_bias,
                full_grid_shape=(W_full, H_full)  # (W, H) format
            )
            
            print(f"      Noise added (sigma = {float(sigma_base):.4f})")
        
        # Remove time dimension (EXACTLY like parse lines 475-476)
        ref_piv_x = ref_piv_x[0]  # (H_ds, W_ds)
        ref_piv_y = ref_piv_y[0]
        
        # Convert back to numpy for plotting (EXACTLY like parse lines 479-480)
        ref_piv_x_np = np.array(ref_piv_x)
        ref_piv_y_np = np.array(ref_piv_y)
        
        # Get PIV resolution for output
        H_piv, W_piv = ref_piv_x_np.shape
        
        print(f"      Noisy reference reconstructed: {ref_piv_x_np.shape}")
        print(f"      PIV RESOLUTION: {W_piv}x{H_piv} (width x height)")
        
        return (ref_piv_x_np, ref_piv_y_np)
        
    except Exception as e:
        print(f" Failed to reconstruct noisy reference: {e}")
        import traceback
        traceback.print_exc()
        return None


def add_constriction_obstacles(ax, domain=((0, 8.0), (0, 4.0)), center_x=4.0, radius=1.5, 
                               facecolor='gray', edgecolor='black', alpha=1.0, linewidth=1):
    """
    Add semicircular obstacle patches to represent the constriction geometry.
    
    Args:
        ax: Matplotlib axis object
        domain: Domain bounds ((x_min, x_max), (y_min, y_max))
        center_x: X-coordinate of semicircle centers
        radius: Radius of semicircles
        facecolor: Fill color for obstacles
        edgecolor: Edge color for obstacles
        alpha: Opacity (0-1)
        linewidth: Edge line width
    """
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]
    
    # Bottom semicircle (centered at y=0, extends upward)
    bottom_semi = patches.Wedge(
        center=(center_x, y_min), 
        r=radius, 
        theta1=0, 
        theta2=180,
        facecolor=facecolor, 
        edgecolor=edgecolor, 
        alpha=alpha,
        linewidth=linewidth,
        zorder=10
    )
    ax.add_patch(bottom_semi)
    
    # Top semicircle (centered at y=y_max, extends downward)
    top_semi = patches.Wedge(
        center=(center_x, y_max), 
        r=radius, 
        theta1=180, 
        theta2=360,
        facecolor=facecolor, 
        edgecolor=edgecolor, 
        alpha=alpha,
        linewidth=linewidth,
        zorder=10
    )
    ax.add_patch(top_semi)


def load_trajectory_data(results_dir):
    """
    Load all available trajectory data from a results directory.
    
    Args:
        results_dir: Path to results directory containing trajectory_data/
        
    Returns:
        Dictionary with all loaded data arrays
    """
    traj_dir = os.path.join(results_dir, 'trajectory_data')
    if not os.path.isdir(traj_dir):
        # Flat data-bundle folder (descriptive names, no trajectory_data/).
        traj_dir = results_dir
    if not os.path.isdir(traj_dir):
        raise ValueError(f"Trajectory data directory not found: {traj_dir}")
    
    data = {}
    
    # Load reference (ground truth) data
    ref_files = ['reference_velocity_x.npy', 'reference_velocity_y.npy',
                 'reference_trajectory_x.npy', 'reference_trajectory_y.npy']
    for fname in ref_files:
        fpath = os.path.join(traj_dir, fname)
        if os.path.exists(fpath):
            key = fname.replace('.npy', '')
            data[key] = np.load(fpath)
            print(f"Loaded {fname}: shape {data[key].shape}")
    
    # Load noisy reference data (if PIV noise was used during training)
    noisy_ref_files = ['noisy_reference_velocity_x.npy', 'noisy_reference_velocity_y.npy']
    for fname in noisy_ref_files:
        fpath = os.path.join(traj_dir, fname)
        if os.path.exists(fpath):
            key = fname.replace('.npy', '')
            data[key] = np.load(fpath)
            print(f"Loaded {fname}: shape {data[key].shape} [PIV noise data]")
    
    # Load initial TBNN data
    init_files = ['initial_tbnn_velocity_x.npy', 'initial_tbnn_velocity_y.npy',
                  'initial_tbnn_trajectory_x.npy', 'initial_tbnn_trajectory_y.npy']
    for fname in init_files:
        fpath = os.path.join(traj_dir, fname)
        if os.path.exists(fpath):
            key = fname.replace('.npy', '')
            data[key] = np.load(fpath)
            print(f"Loaded {fname}: shape {data[key].shape}")
    
    # Load final/updated TBNN data (bundle uses trained_tbnn_velocity_*)
    final_files = ['final_tbnn_velocity_x.npy', 'final_tbnn_velocity_y.npy',
                   'updated_tbnn_velocity_x.npy', 'updated_tbnn_velocity_y.npy',
                   'trained_tbnn_velocity_x.npy', 'trained_tbnn_velocity_y.npy',
                   'updated_tbnn_trajectory_x.npy', 'updated_tbnn_trajectory_y.npy']
    for fname in final_files:
        fpath = os.path.join(traj_dir, fname)
        if os.path.exists(fpath):
            key = fname.replace('.npy', '')
            data[key] = np.load(fpath)
            print(f"Loaded {fname}: shape {data[key].shape}")
    
    # Load training history
    history_files = ['loss_history.npy', 'gradient_magnitudes.npy']
    for fname in history_files:
        fpath = os.path.join(traj_dir, fname)
        if os.path.exists(fpath):
            key = fname.replace('.npy', '')
            data[key] = np.load(fpath)
            print(f"Loaded {fname}: shape {data[key].shape}")
    
    if 'trained_tbnn_velocity_x' in data and 'updated_tbnn_velocity_x' not in data:
        data['updated_tbnn_velocity_x'] = data['trained_tbnn_velocity_x']
        data['updated_tbnn_velocity_y'] = data['trained_tbnn_velocity_y']
    return data


def plot_individual_flow_fields(results_dir, domain=((0, 8.0), (0, 4.0)), 
                               domain_size=(256, 128), dpi=600,
                               save_fig=False, show_fig=True, num_bins=20,
                               noise_used=None, noisy_reference_data=None):
    """
    Plot all flow field visualizations (13 total figures).
    
    Creates 13 separate publication-quality figures:
    1. Ground truth u_x
    2. Ground truth u_y
    3. Trained TBNN u_x
    4. Trained TBNN u_y
    5. Percentage error in u_x
    6. Percentage error in u_y
    7. Absolute difference in u_x
    8. Absolute difference in u_y
    9. Local strain rate field (from reference/ground-truth training velocity)
    10. Local viscosity field (Carreau-Yasuda; from reference/ground-truth training velocity)
    11. Training loss vs iteration
    12. Velocity profiles at throat
    13. Binned relative error vs strain rate
    
    Args:
        results_dir: Path to results directory (e.g., iteration_XX_TIMESTAMP/)
        domain: Domain bounds ((x_min, x_max), (y_min, y_max))
        domain_size: Grid resolution (nx, ny)
        dpi: DPI for saved figures (default: 600)
        save_fig: Whether to save figures to disk (default: False)
        show_fig: Whether to display figures in notebook (default: True)
        num_bins: Number of bins for RMSE plot (default: 20)
        noise_used: Whether PIV noise was used in training (default: None = auto-detect)
        noisy_reference_data: Tuple of (noisy_ref_vx, noisy_ref_vy) if noise was used 
                             (default: None = auto-load from trajectory_data/)
        
    Returns:
        Dictionary with all figure objects and data
        
    Examples:
        # Display all plots in notebook (don't save) - auto-detects noise
        result = plot_individual_flow_fields(file_path)
        
        # Save to disk (don't display)
        result = plot_individual_flow_fields(file_path, save_fig=True, show_fig=False)
        
        # Both save and display
        result = plot_individual_flow_fields(file_path, save_fig=True, show_fig=True)
        
        # Noisy reference data is automatically loaded from trajectory_data/ if available
    """
    # Load data
    data = load_trajectory_data(results_dir)
    
    # Try to parse PIV parameters from summary file
    piv_params = parse_piv_params_from_summary(results_dir)
    
    # Auto-detect if PIV noise was used
    if noise_used is None:
        # First check if noisy data files exist
        if 'noisy_reference_velocity_x' in data and 'noisy_reference_velocity_y' in data:
            noise_used = True
        # Otherwise check if PIV noise is indicated in summary
        elif piv_params is not None:
            noise_used = True
        else:
            noise_used = False
    
    # Auto-load or reconstruct noisy reference data
    if noisy_reference_data is None and noise_used:
        # First try to load from saved files
        if 'noisy_reference_velocity_x' in data and 'noisy_reference_velocity_y' in data:
            noisy_reference_data = (data['noisy_reference_velocity_x'], 
                                   data['noisy_reference_velocity_y'])
            print(f"   Loaded saved noisy reference data")
        # Otherwise reconstruct from PIV parameters
        elif piv_params is not None:
            print(f"    Noisy reference data not found in trajectory_data/")
            print(f"   Attempting to reconstruct from PIV parameters in summary file...")
            
            # Need clean reference data for reconstruction
            if 'reference_velocity_x' in data and 'reference_velocity_y' in data:
                noisy_reference_data = reconstruct_noisy_reference(
                    data['reference_velocity_x'],
                    data['reference_velocity_y'],
                    piv_params,
                    domain,
                    domain_size
                )
                
                if noisy_reference_data is not None:
                    print("   reconstructed noisy reference data")
                else:
                    print(f"   Failed to reconstruct noisy reference data")
                    noise_used = False
            else:
                print(f"   Cannot reconstruct: clean reference data not found")
                noise_used = False
    
    print(f"\n{'='*60}")
    noise_tag = " [noise = True]" if noise_used else ""
    print(f"PLOTTING ALL FLOW FIELD VISUALIZATIONS (7 PLOTS){noise_tag}")
    print(f"{'='*60}")
    print(f"Results directory: {results_dir}")
    print(f"Save to disk: {save_fig}")
    print(f"Display in notebook: {show_fig}")
    if noise_used:
        has_noisy_data = noisy_reference_data is not None
        if has_noisy_data:
            print("PIV noise: True (using noisy reference data)")
        else:
            print(f"PIV noise: True (but noisy reference data unavailable)")
    
    # Determine which fields to use (prefer 'updated' over 'final')
    if 'updated_tbnn_velocity_x' in data:
        tbnn_vx = data['updated_tbnn_velocity_x']
        tbnn_vy = data['updated_tbnn_velocity_y']
        tbnn_label = 'Trained TBNN'
    elif 'final_tbnn_velocity_x' in data:
        tbnn_vx = data['final_tbnn_velocity_x']
        tbnn_vy = data['final_tbnn_velocity_y']
        tbnn_label = 'Trained TBNN'
    else:
        raise ValueError("No final TBNN velocity data found!")
    
    if 'reference_velocity_x' not in data:
        raise ValueError("No reference velocity data found!")
    
    ref_vx = data['reference_velocity_x']
    ref_vy = data['reference_velocity_y']
    
    print(f"\nReference velocity: {ref_vx.shape}")
    print(f"Trained TBNN velocity: {tbnn_vx.shape}")
    
    # Extract domain info
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]
    nx, ny = domain_size
    
    # Calculate aspect ratio (should be 2:1 for default domain)
    aspect_ratio = (x_max - x_min) / (y_max - y_min)
    print(f"\nDomain aspect ratio: {aspect_ratio:.2f}:1")
    
    # Check if data needs transposing (velocity should be (ny, nx) for plotting)
    if ref_vx.shape[0] == nx and ref_vx.shape[1] == ny:
        print(f" Transposing velocity data from {ref_vx.shape} to ({ny}, {nx}) for plotting")
        ref_vx = ref_vx.T
        ref_vy = ref_vy.T
        tbnn_vx = tbnn_vx.T
        tbnn_vy = tbnn_vy.T
        print(f"After transpose - Reference: {ref_vx.shape}, TBNN: {tbnn_vx.shape}")
    
    # Compute percentage errors (like porous_media_flow.py)
    error_vx = 100.0 * (tbnn_vx - ref_vx) / (np.abs(ref_vx) + 1e-10)  # Percentage error
    error_vy = 100.0 * (tbnn_vy - ref_vy) / (np.abs(ref_vy) + 1e-10)  # Percentage error
    
    # Create coordinate grids
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    # Figure size - maintain aspect ratio
    fig_width = 10  # inches
    fig_height = fig_width / aspect_ratio
    
    # Determine consistent color limits
    vx_min = min(ref_vx.min(), tbnn_vx.min())
    vx_max = max(ref_vx.max(), tbnn_vx.max())
    vy_min = min(ref_vy.min(), tbnn_vy.min())
    vy_max = max(ref_vy.max(), tbnn_vy.max())
    
    # Helper function to create a single plot
    def create_single_plot(field, title, vmin, vmax, cmap, filename, add_obstacles=True):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        # Plot field
        im = ax.pcolormesh(X, Y, field, cmap=cmap, vmin=vmin, vmax=vmax, 
                          shading='auto', rasterized=True)
        
        # Add obstacles
        if add_obstacles:
            add_constriction_obstacles(ax, domain=domain, facecolor='gray', 
                                       edgecolor='black', alpha=0.8, linewidth=1.5)
        
        # Formatting
        ax.set_xlabel('x', fontsize=26)
        ax.set_ylabel('y', fontsize=26)
        ax.set_title(title, fontsize=22, fontweight='bold', pad=20)
        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Add colorbar with label
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=22)
        
        # Extract label from title for colorbar
        if 'Percentage Error' in title:
            cbar.set_label('% Error', fontsize=26, rotation=90, labelpad=20)
        elif 'Absolute Difference' in title:
            # Use the same label as the velocity component (u_x or u_y)
            if 'u_x' in title or '$u_x$' in title:
                cbar.set_label('$u_x$', fontsize=26, rotation=0, labelpad=20)
            elif 'u_y' in title or '$u_y$' in title:
                cbar.set_label('$u_y$', fontsize=26, rotation=0, labelpad=20)
        elif 'u_x' in title or '$u_x$' in title:
            cbar.set_label('$u_x$', fontsize=26, rotation=0, labelpad=20)
        elif 'u_y' in title or '$u_y$' in title:
            cbar.set_label('$u_y$', fontsize=26, rotation=0, labelpad=20)
        
        plt.tight_layout()
        
        # Save
        if save_fig:
            output_path = os.path.join(results_dir, filename)
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"   Saved: {filename}")
        
        # Display (notebook will auto-show if in interactive mode)
        if not show_fig:
            plt.close(fig)
        # else: leave figure open for notebook display
        
        return fig
    
    # Create flow field plots (1-6)
    print(f"\nCreating flow field plots (1-6 of 13)...")
    if save_fig:
        print(f"   Saving to: {results_dir}")
    if show_fig:
        print(f"    Displaying in notebook...")
    
    figures = {}
    
    figures['ref_ux'] = create_single_plot(
        ref_vx, 'Ground Truth $u_x$', vx_min, vx_max, 'RdBu_r', 
        'flow_field_01_reference_ux.png'
    )
    
    figures['ref_uy'] = create_single_plot(
        ref_vy, 'Ground Truth $u_y$', vy_min, vy_max, 'RdBu_r',
        'flow_field_02_reference_uy.png'
    )
    
    figures['tbnn_ux'] = create_single_plot(
        tbnn_vx, f'{tbnn_label} $u_x$', vx_min, vx_max, 'RdBu_r',
        'flow_field_03_trained_ux.png'
    )
    
    figures['tbnn_uy'] = create_single_plot(
        tbnn_vy, f'{tbnn_label} $u_y$', vy_min, vy_max, 'RdBu_r',
        'flow_field_04_trained_uy.png'
    )
    
    # For percentage errors, use symmetric colormap centered at zero
    vmax_pct_x = np.percentile(np.abs(error_vx), 95)  # Use 95th percentile for better visualization
    vmax_pct_y = np.percentile(np.abs(error_vy), 95)
    
    figures['error_ux'] = create_single_plot(
        error_vx, 'Percentage Error in $u_x$', -vmax_pct_x, vmax_pct_x, 'hot',
        'flow_field_05_error_ux.png'
    )
    
    figures['error_uy'] = create_single_plot(
        error_vy, 'Percentage Error in $u_y$', -vmax_pct_y, vmax_pct_y, 'hot',
        'flow_field_06_error_uy.png'
    )
    
    # Compute and display error metrics
    rmse_vx = np.sqrt(np.mean((tbnn_vx - ref_vx)**2))
    rmse_vy = np.sqrt(np.mean((tbnn_vy - ref_vy)**2))
    
    rel_error_vx = rmse_vx / (np.abs(ref_vx).max() + 1e-10)
    rel_error_vy = rmse_vy / (np.abs(ref_vy).max() + 1e-10)
    
    # Mean percentage errors
    mean_pct_error_x = np.mean(np.abs(error_vx))
    mean_pct_error_y = np.mean(np.abs(error_vy))
    max_pct_error_x = np.max(np.abs(error_vx))
    max_pct_error_y = np.max(np.abs(error_vy))
    
    print(f"\nERROR METRICS:")
    print(f"   RMSE u_x: {rmse_vx:.6e} (relative to max: {rel_error_vx*100:.2f}%)")
    print(f"   RMSE u_y: {rmse_vy:.6e} (relative to max: {rel_error_vy*100:.2f}%)")
    print(f"   Mean percentage error u_x: {mean_pct_error_x:.2f}%")
    print(f"   Mean percentage error u_y: {mean_pct_error_y:.2f}%")
    print(f"   Max percentage error u_x: {max_pct_error_x:.2f}%")
    print(f"   Max percentage error u_y: {max_pct_error_y:.2f}%")
    
    # Create additional plots (7-13)
    print(f"\nCreating additional analysis plots (7-13 of 13)...")
    
    # Plot 7: Absolute difference u_x (absolute value)
    abs_diff_vx = np.abs(tbnn_vx - ref_vx)
    vmax_abs_x = np.percentile(abs_diff_vx, 95)
    figures['abs_diff_ux'] = create_single_plot(
        abs_diff_vx, 'Absolute Difference in $u_x$', 0, vmax_abs_x, 'YlOrRd',
        'flow_field_07_abs_diff_ux.png'
    )
    
    # Plot 8: Absolute difference u_y (absolute value)
    abs_diff_vy = np.abs(tbnn_vy - ref_vy)
    vmax_abs_y = np.percentile(abs_diff_vy, 95)
    figures['abs_diff_uy'] = create_single_plot(
        abs_diff_vy, 'Absolute Difference in $u_y$', 0, vmax_abs_y, 'YlOrRd',
        'flow_field_08_abs_diff_uy.png'
    )
    
    # Plot 9: Local strain rate (from reference/ground-truth training velocity)
    strain_rate_fig = plot_strain_rate_field(
        ref_vx, ref_vy, domain, domain_size,
        dpi=dpi, save_fig=save_fig, show_fig=show_fig, output_dir=results_dir
    )
    figures['strain_rate'] = strain_rate_fig
    
    # Plot 10: Local viscosity (from reference/ground-truth training velocity using Carreau-Yasuda model)
    # Default Carreau-Yasuda parameters (matching your training setup)
    cy_params = {
        'eta_inf': 0.02,
        'eta_0': 1.0,
        'lambda_': 5.0,
        'n': 0.7,
        'a': 2.0
    }
    viscosity_fig = plot_viscosity_field(
        ref_vx, ref_vy, domain, domain_size, cy_params,
        dpi=dpi, save_fig=save_fig, show_fig=show_fig, output_dir=results_dir
    )
    figures['viscosity'] = viscosity_fig
    
    # Plot 11: Loss vs iteration
    if 'loss_history' in data:
        fig_loss = plot_loss_history(data['loss_history'], dpi=dpi, save_fig=save_fig, 
                                     show_fig=show_fig, output_dir=results_dir)
        figures['loss_history'] = fig_loss
    
    # Plot 12: Velocity profiles at throat
    throat_fig = plot_throat_velocity_profiles(
        ref_vx, ref_vy, tbnn_vx, tbnn_vy, data, domain, domain_size,
        dpi=dpi, save_fig=save_fig, show_fig=show_fig, output_dir=results_dir,
        noise_used=noise_used, noisy_reference_data=noisy_reference_data
    )
    figures['throat_profiles'] = throat_fig
    
    # Plot 13: Binned RMSE
    rmse_result = plot_binned_rmse(results_dir, domain=domain, domain_size=domain_size,
                                    num_bins=num_bins, dpi=dpi, save_fig=save_fig, show_fig=show_fig)
    figures['binned_rmse'] = rmse_result['figure']
    
    print(f"\nwrote {len(figures)} plots to {results_dir}")
    
    return {
        'figures': figures, 
        'data': data,
        'binned_rmse_data': rmse_result
    }


def plot_strain_rate_field(vx, vy, domain, domain_size, dpi=600, 
                           save_fig=False, show_fig=True, output_dir='.'):
    """
    Plot local strain rate field on log scale (from trained TBNN velocity).
    
    Strain rate gammadot = sqrt(2 * D:D) where D is the rate-of-deformation tensor.
    
    Args:
        vx, vy: Velocity field components (ny, nx)
        domain: Domain bounds ((x_min, x_max), (y_min, y_max))
        domain_size: Grid resolution (nx, ny)
        dpi: DPI for saved figure
        save_fig: Whether to save figure
        show_fig: Whether to display figure
        output_dir: Directory to save figure
        
    Returns:
        Figure object
    """
    print(f"   Creating local strain rate field...")
    
    # Extract domain info
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]
    nx, ny = domain_size
    
    # Calculate grid spacings
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    
    # Compute velocity gradients using central differences
    # For interior points, use central difference
    # For boundaries, use forward/backward difference
    
    # du_x/dx
    dux_dx = np.zeros_like(vx)
    dux_dx[:, 1:-1] = (vx[:, 2:] - vx[:, :-2]) / (2 * dx)
    dux_dx[:, 0] = (vx[:, 1] - vx[:, 0]) / dx  # Forward diff
    dux_dx[:, -1] = (vx[:, -1] - vx[:, -2]) / dx  # Backward diff
    
    # du_y/dy
    duy_dy = np.zeros_like(vy)
    duy_dy[1:-1, :] = (vy[2:, :] - vy[:-2, :]) / (2 * dy)
    duy_dy[0, :] = (vy[1, :] - vy[0, :]) / dy  # Forward diff
    duy_dy[-1, :] = (vy[-1, :] - vy[-2, :]) / dy  # Backward diff
    
    # du_x/dy
    dux_dy = np.zeros_like(vx)
    dux_dy[1:-1, :] = (vx[2:, :] - vx[:-2, :]) / (2 * dy)
    dux_dy[0, :] = (vx[1, :] - vx[0, :]) / dy
    dux_dy[-1, :] = (vx[-1, :] - vx[-2, :]) / dy
    
    # du_y/dx
    duy_dx = np.zeros_like(vy)
    duy_dx[:, 1:-1] = (vy[:, 2:] - vy[:, :-2]) / (2 * dx)
    duy_dx[:, 0] = (vy[:, 1] - vy[:, 0]) / dx
    duy_dx[:, -1] = (vy[:, -1] - vy[:, -2]) / dx
    
    # Compute strain rate tensor components
    # D_xx = du_x/dx
    # D_yy = du_y/dy
    # D_xy = 0.5 * (du_x/dy + du_y/dx)
    D_xx = dux_dx
    D_yy = duy_dy
    D_xy = 0.5 * (dux_dy + duy_dx)
    
    # Strain rate: gammadot = sqrt(2 * (D_xx^2 + D_yy^2 + 2*D_xy^2))
    strain_rate = np.sqrt(2.0 * (D_xx**2 + D_yy**2 + 2.0*D_xy**2))
    
    # Log scale for visualization
    log_strain_rate = np.log10(np.maximum(strain_rate, 1e-8))  # Avoid log(0)
    
    # Create plot
    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = fig.add_subplot(111)
    
    # Create heatmap
    im = ax.pcolormesh(
        np.linspace(x_min, x_max, nx),
        np.linspace(y_min, y_max, ny),
        log_strain_rate,
        cmap='plasma',
        shading='auto'
    )
    
    # Add obstacles
    add_constriction_obstacles(ax, domain)
    
    ax.set_xlabel('x', fontsize=26)
    ax.set_ylabel('y', fontsize=26)
    # No title
    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add colorbar with label (smaller colorbar, no units)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.ax.tick_params(labelsize=22)
    cbar.set_label(r'$\log_{10}(\dot{\gamma})$', fontsize=26, rotation=90, labelpad=20)
    
    # Remove grid
    ax.grid(False)
    
    # Configure ticks: inside, on both sides
    ax.tick_params(axis='both', which='major', labelsize=22, width=1.5, length=8,
                   direction='in', top=True, right=True)
    ax.tick_params(axis='both', which='minor', labelsize=18, width=1, length=4,
                   direction='in', top=True, right=True)
    
    plt.tight_layout()
    
    if save_fig:
        output_path = os.path.join(output_dir, 'flow_field_09_strain_rate.png')
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"   Saved: flow_field_09_strain_rate.png")
    
    if not show_fig:
        plt.close(fig)
    
    return fig


def plot_viscosity_field(vx, vy, domain, domain_size, cy_params, dpi=600,
                         save_fig=False, show_fig=True, output_dir='.'):
    """
    Plot local viscosity field on log scale using Carreau-Yasuda model.
    
    Computes viscosity from velocity field using:
    eta = eta_inf + (eta_0 - eta_inf) * [1 + (lam*gammadot)^a]^((n-1)/a)
    
    Args:
        vx, vy: Velocity field components (ny, nx)
        domain: Domain bounds ((x_min, x_max), (y_min, y_max))
        domain_size: Grid resolution (nx, ny)
        cy_params: Dictionary with Carreau-Yasuda parameters
                   {'eta_inf', 'eta_0', 'lambda_', 'n', 'a'}
        dpi: DPI for saved figure
        save_fig: Whether to save figure
        show_fig: Whether to display figure
        output_dir: Directory to save figure
        
    Returns:
        Figure object
    """
    print(f"   Creating local viscosity field...")
    
    # Extract domain info
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]
    nx, ny = domain_size
    
    # Calculate grid spacings
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    
    # Compute velocity gradients (same as strain rate calculation)
    # du_x/dx
    dux_dx = np.zeros_like(vx)
    dux_dx[:, 1:-1] = (vx[:, 2:] - vx[:, :-2]) / (2 * dx)
    dux_dx[:, 0] = (vx[:, 1] - vx[:, 0]) / dx
    dux_dx[:, -1] = (vx[:, -1] - vx[:, -2]) / dx
    
    # du_y/dy
    duy_dy = np.zeros_like(vy)
    duy_dy[1:-1, :] = (vy[2:, :] - vy[:-2, :]) / (2 * dy)
    duy_dy[0, :] = (vy[1, :] - vy[0, :]) / dy
    duy_dy[-1, :] = (vy[-1, :] - vy[-2, :]) / dy
    
    # du_x/dy
    dux_dy = np.zeros_like(vx)
    dux_dy[1:-1, :] = (vx[2:, :] - vx[:-2, :]) / (2 * dy)
    dux_dy[0, :] = (vx[1, :] - vx[0, :]) / dy
    dux_dy[-1, :] = (vx[-1, :] - vx[-2, :]) / dy
    
    # du_y/dx
    duy_dx = np.zeros_like(vy)
    duy_dx[:, 1:-1] = (vy[:, 2:] - vy[:, :-2]) / (2 * dx)
    duy_dx[:, 0] = (vy[:, 1] - vy[:, 0]) / dx
    duy_dx[:, -1] = (vy[:, -1] - vy[:, -2]) / dx
    
    # Compute strain rate
    D_xx = dux_dx
    D_yy = duy_dy
    D_xy = 0.5 * (dux_dy + duy_dx)
    strain_rate = np.sqrt(2.0 * (D_xx**2 + D_yy**2 + 2.0*D_xy**2))
    
    # Apply Carreau-Yasuda model
    eta_inf = cy_params['eta_inf']
    eta_0 = cy_params['eta_0']
    lambda_ = cy_params['lambda_']
    n = cy_params['n']
    a = cy_params['a']
    
    # eta = eta_inf + (eta_0 - eta_inf) * [1 + (lam*gammadot)^a]^((n-1)/a)
    viscosity = eta_inf + (eta_0 - eta_inf) * np.power(
        1.0 + np.power(lambda_ * strain_rate, a),
        (n - 1.0) / a
    )
    
    # Normalize by eta_0 and take log10
    viscosity_normalized = viscosity / eta_0
    log_viscosity_normalized = np.log10(np.maximum(viscosity_normalized, 1e-8))  # Avoid log(0)
    
    # Create plot
    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = fig.add_subplot(111)
    
    # Create heatmap
    im = ax.pcolormesh(
        np.linspace(x_min, x_max, nx),
        np.linspace(y_min, y_max, ny),
        log_viscosity_normalized,
        cmap='plasma',
        shading='auto'
    )
    
    # Add obstacles
    add_constriction_obstacles(ax, domain)
    
    ax.set_xlabel('x', fontsize=26)
    ax.set_ylabel('y', fontsize=26)
    # No title
    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add colorbar with label (smaller colorbar, normalized viscosity)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.ax.tick_params(labelsize=22)
    cbar.set_label(r'$\log_{10}(\eta/\eta_0)$', fontsize=26, rotation=90, labelpad=20)
    
    # Remove grid
    ax.grid(False)
    
    # Configure ticks: inside, on both sides
    ax.tick_params(axis='both', which='major', labelsize=22, width=1.5, length=8,
                   direction='in', top=True, right=True)
    ax.tick_params(axis='both', which='minor', labelsize=18, width=1, length=4,
                   direction='in', top=True, right=True)
    
    plt.tight_layout()
    
    if save_fig:
        output_path = os.path.join(output_dir, 'flow_field_10_viscosity.png')
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"   Saved: flow_field_10_viscosity.png")
    
    if not show_fig:
        plt.close(fig)
    
    return fig


def plot_loss_history(loss_history, dpi=600, save_fig=False, show_fig=True, output_dir='.'):
    """
    Plot training loss vs iteration (log-y scale).
    
    Args:
        loss_history: Array of loss values at each iteration
        dpi: DPI for saved figure
        save_fig: Whether to save figure
        show_fig: Whether to display figure
        output_dir: Directory to save figure
        
    Returns:
        Figure object
    """
    print(f"   Creating loss history plot...")
    
    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    ax = fig.add_subplot(111)
    
    iterations = np.arange(len(loss_history))
    
    ax.semilogy(iterations, loss_history, 'o-', linewidth=3, markersize=8, color='#1f77b4')
    
    ax.set_xlabel('Iteration', fontsize=26)
    ax.set_ylabel('Loss', fontsize=26)
    ax.set_title('Training Loss vs Iteration', fontsize=22, fontweight='bold', pad=20)
    
    # Remove grid
    ax.grid(False)
    
    # Configure ticks: inside, on both sides
    ax.tick_params(axis='both', which='major', labelsize=22, width=1.5, length=8,
                   direction='in', top=True, right=True)
    ax.tick_params(axis='both', which='minor', labelsize=18, width=1, length=4,
                   direction='in', top=True, right=True)
    
    plt.tight_layout()
    
    if save_fig:
        output_path = os.path.join(output_dir, 'flow_field_11_loss_history.png')
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"   Saved: flow_field_11_loss_history.png")
    
    if not show_fig:
        plt.close(fig)
    
    return fig


def plot_throat_velocity_profiles(ref_vx, ref_vy, trained_vx, trained_vy, data, 
                                   domain, domain_size, dpi=600, 
                                   save_fig=False, show_fig=True, output_dir='.',
                                   noise_used=False, noisy_reference_data=None):
    """
    Plot velocity profiles at the throat (x=4.0, narrowest point).
    Compares ground truth, initial TBNN, and final TBNN.
    If noise was used, also shows the noisy reference that was actually fitted to.
    
    Args:
        ref_vx, ref_vy: Reference velocity fields (clean ground truth) in (H, W) format
        trained_vx, trained_vy: Trained TBNN velocity fields in (H, W) format
        data: Dictionary with loaded data (including initial_tbnn_velocity_x/y)
        domain: Domain bounds
        domain_size: Grid resolution
        dpi: DPI for saved figure
        save_fig: Whether to save figure
        show_fig: Whether to display figure
        output_dir: Directory to save figure
        noise_used: Whether PIV noise was used in training (adds label to title)
        noisy_reference_data: Tuple of (noisy_ref_vx, noisy_ref_vy) in (H_piv, W_piv) format
                             PIV downsampled resolution (e.g., 29x29)
        
    Returns:
        Figure object
    """
    noise_label = " (with noisy reference)" if noise_used and noisy_reference_data is not None else ""
    print(f"   Creating throat velocity profiles{noise_label}...")
    
    # Extract domain info
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]
    nx, ny = domain_size
    
    # Find throat location (center of domain)
    throat_x = (x_max + x_min) / 2  # x = 4.0 for default domain
    x_coords = np.linspace(x_min, x_max, nx)
    y_coords = np.linspace(y_min, y_max, ny)
    
    # Find index closest to throat
    throat_idx = np.argmin(np.abs(x_coords - throat_x))
    
    # Extract velocity profiles at throat
    ref_profile = ref_vx[:, throat_idx]
    trained_profile = trained_vx[:, throat_idx]
    
    # Get initial TBNN profile if available
    if 'initial_tbnn_velocity_x' in data:
        init_vx = data['initial_tbnn_velocity_x']
        # Check if needs transposing
        if init_vx.shape[0] == nx and init_vx.shape[1] == ny:
            init_vx = init_vx.T
        init_profile = init_vx[:, throat_idx]
    else:
        init_profile = None
    
    # Get noisy reference profile if provided
    noisy_profile = None
    if noise_used and noisy_reference_data is not None:
        noisy_vx, noisy_vy = noisy_reference_data
        
        print(f"   Noisy reference shape: {noisy_vx.shape}")
        
        # Noisy data is PIV downsampled in (H_piv, W_piv) format
        # Throat is at physical location throat_x
        ny_piv, nx_piv = noisy_vx.shape  # PIV resolution (H_piv, W_piv)
        
        # Create PIV coordinate grid
        x_coords_piv = np.linspace(x_min, x_max, nx_piv)
        y_coords_piv = np.linspace(y_min, y_max, ny_piv)
        
        # Find throat index in PIV coordinates
        throat_idx_piv = np.argmin(np.abs(x_coords_piv - throat_x))
        
        print(f"   PIV resolution: {nx_piv} points in x-direction, {ny_piv} points in y-direction")
        print(f"   Throat location: x={throat_x:.1f}, PIV index={throat_idx_piv}, full-res index={throat_idx}")
        
        # Extract profile at throat in PIV coordinates
        noisy_profile = noisy_vx[:, throat_idx_piv]
        
        print(f"   Extracted noisy profile at throat: shape {noisy_profile.shape}, range [{noisy_profile.min():.4f}, {noisy_profile.max():.4f}]")
    
    # Create plot
    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    ax = fig.add_subplot(111)
    
    # Plot profiles
    # Ground truth: dash-dot pattern (long dash, short dot) so it's visible under trained TBNN
    ax.plot(ref_profile, y_coords, 'k-.', linewidth=3, label='Ground Truth (clean)', zorder=3)
    # Noisy reference: red circles to show what was actually fitted to (PIV resolution)
    if noisy_profile is not None:
        ax.plot(noisy_profile, y_coords_piv, 'ro', markersize=6, label='Noisy Reference (fitted to)', 
                alpha=0.7, zorder=4)
    if init_profile is not None:
        ax.plot(init_profile, y_coords, '--', linewidth=3, color='#ff7f0e', 
                label='Initial TBNN', alpha=0.7, zorder=2)
    ax.plot(trained_profile, y_coords, '-', linewidth=3, color='#2ca02c', 
            label='Trained TBNN', zorder=2)
    
    ax.set_xlabel('$u_x$', fontsize=26)
    ax.set_ylabel('y', fontsize=26)
    title_text = f'Velocity Profile at Throat ($x = {throat_x:.1f}$)'
    if noise_used:
        title_text += ' [noise = True]'
    ax.set_title(title_text, fontsize=22, fontweight='bold', pad=20)
    
    ax.legend(loc='best', fontsize=19, frameon=True, fancybox=True, shadow=True)
    
    # Remove grid
    ax.grid(False)
    
    # Configure ticks: inside, on both sides
    ax.tick_params(axis='both', which='major', labelsize=22, width=1.5, length=8,
                   direction='in', top=True, right=True)
    ax.tick_params(axis='both', which='minor', labelsize=18, width=1, length=4,
                   direction='in', top=True, right=True)
    
    plt.tight_layout()
    
    if save_fig:
        output_path = os.path.join(output_dir, 'flow_field_12_throat_profiles.png')
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"   Saved: flow_field_12_throat_profiles.png")
    
    if not show_fig:
        plt.close(fig)
    
    return fig


def plot_binned_rmse(results_dir, domain=((0, 8.0), (0, 4.0)),
                     domain_size=(256, 128), num_bins=20, dpi=600,
                     save_fig=False, show_fig=True):
    """
    Plot binned RMSE vs local strain rate (log-log plot).
    
    Args:
        results_dir: Path to results directory
        domain: Domain bounds ((x_min, x_max), (y_min, y_max))
        domain_size: Grid resolution (nx, ny)
        num_bins: Number of logarithmically-spaced bins (default: 20)
        dpi: DPI for saved figure (default: 600)
        save_fig: Whether to save figure (default: False)
        show_fig: Whether to display figure (default: True)
        
    Returns:
        Dictionary with binned data and metrics
    """
    print(f"\n{'='*60}")
    print(f"PLOTTING BINNED RMSE VS STRAIN RATE")
    print(f"{'='*60}")
    
    # Load data
    data = load_trajectory_data(results_dir)
    
    # Determine which fields to use
    if 'updated_tbnn_velocity_x' in data:
        tbnn_vx = data['updated_tbnn_velocity_x']
        tbnn_vy = data['updated_tbnn_velocity_y']
    elif 'final_tbnn_velocity_x' in data:
        tbnn_vx = data['final_tbnn_velocity_x']
        tbnn_vy = data['final_tbnn_velocity_y']
    else:
        raise ValueError("No final TBNN velocity data found!")
    
    ref_vx = data['reference_velocity_x']
    ref_vy = data['reference_velocity_y']
    
    # Extract domain info
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]
    nx, ny = domain_size
    
    # Check if data needs transposing
    if ref_vx.shape[0] == nx and ref_vx.shape[1] == ny:
        ref_vx = ref_vx.T
        ref_vy = ref_vy.T
        tbnn_vx = tbnn_vx.T
        tbnn_vy = tbnn_vy.T
    
    # Calculate spatial grid spacing
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    
    # Compute strain rate from reference velocity (ground truth)
    # Using central differences
    def _central_diff_x(a):
        aL = np.concatenate([a[:, :1], a[:, :-1]], axis=1)
        aR = np.concatenate([a[:, 1:], a[:, -1:]], axis=1)
        return (aR - aL) / (2.0 * dx)
    
    def _central_diff_y(a):
        aB = np.concatenate([a[:1, :], a[:-1, :]], axis=0)
        aT = np.concatenate([a[1:, :], a[-1:, :]], axis=0)
        return (aT - aB) / (2.0 * dy)
    
    # Strain rate components for reference (ground truth)
    dux_dx = _central_diff_x(ref_vx)
    duy_dy = _central_diff_y(ref_vy)
    dux_dy = _central_diff_y(ref_vx)
    duy_dx = _central_diff_x(ref_vy)
    
    # Strain rate tensor components: D_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
    D_xx = dux_dx
    D_yy = duy_dy
    D_xy = 0.5 * (dux_dy + duy_dx)
    
    # Strain rate magnitude: gamma_dot = sqrt(2 * D:D)
    gamma_dot = np.sqrt(2.0 * (D_xx**2 + D_yy**2 + 2*D_xy**2))
    
    # Compute relative errors (as fractions, 0-1, for compatibility with porous_media style)
    rel_error_x = np.abs(tbnn_vx - ref_vx) / (np.abs(ref_vx) + 1e-10)
    rel_error_y = np.abs(tbnn_vy - ref_vy) / (np.abs(ref_vy) + 1e-10)
    
    # Create mask to exclude constriction regions
    x_coords = np.linspace(x_min, x_max, nx)
    y_coords = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    center_x = (x_max + x_min) / 2
    radius = 1.5
    dist_bottom = np.sqrt((X - center_x)**2 + (Y - y_min)**2)
    dist_top = np.sqrt((X - center_x)**2 + (Y - y_max)**2)
    
    buffer = 0.1
    fluid_mask = (dist_bottom > (radius + buffer)) & (dist_top > (radius + buffer))
    
    # Flatten and mask data
    gamma_flat = gamma_dot[fluid_mask].flatten()
    error_x_flat = rel_error_x[fluid_mask].flatten()
    error_y_flat = rel_error_y[fluid_mask].flatten()
    
    # Remove invalid values for x
    valid_mask_x = np.isfinite(gamma_flat) & np.isfinite(error_x_flat) & (gamma_flat > 1e-10)
    gamma_valid_x = gamma_flat[valid_mask_x]
    error_x_valid = error_x_flat[valid_mask_x]
    
    # Remove invalid values for y
    valid_mask_y = np.isfinite(gamma_flat) & np.isfinite(error_y_flat) & (gamma_flat > 1e-10)
    gamma_valid_y = gamma_flat[valid_mask_y]
    error_y_valid = error_y_flat[valid_mask_y]
    
    # Create log-spaced bins
    gamma_min = min(np.min(gamma_valid_x), np.min(gamma_valid_y))
    gamma_max = max(np.max(gamma_valid_x), np.max(gamma_valid_y))
    bins = np.logspace(np.log10(gamma_min), np.log10(gamma_max), num_bins + 1)
    
    # Bin the x-error data
    bin_centers_x = []
    bin_errors_x = []
    bin_stds_x = []
    
    for i in range(num_bins):
        mask = (gamma_valid_x >= bins[i]) & (gamma_valid_x < bins[i+1])
        if np.sum(mask) > 0:
            bin_centers_x.append(np.sqrt(bins[i] * bins[i+1]))  # Geometric mean
            bin_errors_x.append(float(np.mean(error_x_valid[mask])))
            bin_stds_x.append(float(np.std(error_x_valid[mask])))
    
    bin_centers_x = np.array(bin_centers_x)
    bin_errors_x = np.array(bin_errors_x)
    bin_stds_x = np.array(bin_stds_x)
    
    # Bin the y-error data
    bin_centers_y = []
    bin_errors_y = []
    bin_stds_y = []
    
    for i in range(num_bins):
        mask = (gamma_valid_y >= bins[i]) & (gamma_valid_y < bins[i+1])
        if np.sum(mask) > 0:
            bin_centers_y.append(np.sqrt(bins[i] * bins[i+1]))  # Geometric mean
            bin_errors_y.append(float(np.mean(error_y_valid[mask])))
            bin_stds_y.append(float(np.std(error_y_valid[mask])))
    
    bin_centers_y = np.array(bin_centers_y)
    bin_errors_y = np.array(bin_errors_y)
    bin_stds_y = np.array(bin_stds_y)
    
    # Calculate median errors
    median_error_x = np.median(bin_errors_x)
    median_error_y = np.median(bin_errors_y)
    
    print(f"\nBinned data computed:")
    print(f"   Strain rate range: [{gamma_min:.2e}, {gamma_max:.2e}] 1/s")
    print(f"   Number of bins: {num_bins}")
    print(f"   Median relative error u_x: {median_error_x:.6f} ({median_error_x*100:.2f}%)")
    print(f"   Median relative error u_y: {median_error_y:.6f} ({median_error_y*100:.2f}%)")
    print(f"   Error range u_x: [{np.min(bin_errors_x):.6f}, {np.max(bin_errors_x):.6f}]")
    print(f"   Error range u_y: [{np.min(bin_errors_y):.6f}, {np.max(bin_errors_y):.6f}]")
    
    # Create plot
    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = fig.add_subplot(111)
    
    # Plot y errors first (larger error bars) so x errors are visible on top
    ax.errorbar(bin_centers_y, bin_errors_y, yerr=bin_stds_y, 
                fmt='s-', linewidth=3, markersize=10, capsize=7, capthick=2,
                label=f'$u_y$, median error = {median_error_y:.4f}', color='#ff7f0e')
    ax.errorbar(bin_centers_x, bin_errors_x, yerr=bin_stds_x, 
                fmt='o-', linewidth=3, markersize=10, capsize=7, capthick=2,
                label=f'$u_x$, median error = {median_error_x:.4f}', color='#1f77b4')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Local Strain Rate $\dot{\gamma}$ (1/s)', fontsize=26)
    ax.set_ylabel('Relative Error', fontsize=26)
    ax.set_title('Relative Error vs Strain Rate', fontsize=22, fontweight='bold', pad=20)
    
    # Set y-axis limits for better spacing (adjust based on actual data range)
    y_min = min(np.min(bin_errors_x[bin_errors_x > 0]), np.min(bin_errors_y[bin_errors_y > 0]))
    y_max = max(np.max(bin_errors_x), np.max(bin_errors_y))
    ax.set_ylim(y_min * 0.5, y_max * 2.0)
    
    # Remove grid
    ax.grid(False)
    
    # Configure ticks: inside, on both sides
    ax.tick_params(axis='both', which='major', labelsize=22, width=1.5, length=8,
                   direction='in', top=True, right=True)
    ax.tick_params(axis='both', which='minor', labelsize=18, width=1, length=4,
                   direction='in', top=True, right=True)
    
    # Legend in lower left
    ax.legend(loc='lower left', fontsize=19, frameon=True, fancybox=True, shadow=True)
    
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    # Save
    if save_fig:
        output_path = os.path.join(results_dir, 'flow_field_13_binned_rmse.png')
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"\nSaved: flow_field_13_binned_rmse.png")
    
    # Display
    if not show_fig:
        plt.close(fig)
    
    return {
        'figure': fig,
        'bin_centers_x': bin_centers_x,
        'bin_errors_x': bin_errors_x,
        'bin_stds_x': bin_stds_x,
        'bin_centers_y': bin_centers_y,
        'bin_errors_y': bin_errors_y,
        'bin_stds_y': bin_stds_y,
        'median_error_x': median_error_x,
        'median_error_y': median_error_y,
        'strain_rate_range': (gamma_min, gamma_max)
    }




# Convenience alias
plot_results = plot_individual_flow_fields


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path as _Path

    _p = argparse.ArgumentParser(description="Plot Fig 2 / instantaneous flow-field panels")
    _p.add_argument("results_dir", nargs="?", default=None,
                    help="Run folder (frozen iteration_* or data-bundle fig2/fig4 folder)")
    _p.add_argument("--data-root", default=os.environ.get("TBNN_DATA_BUNDLE") or os.environ.get("DATA_ROOT"),
                    help="Unpacked data_bundle/ (or the tokens bundle / data_bundle). "
                         "Default Fig 2 folder is <root>/fig2_instantaneous_demo/cy_n0p7_lam5/")
    _args = _p.parse_args()
    if _args.results_dir:
        results_dir = _args.results_dir
    elif _args.data_root:
        root = _args.data_root
        if root in ("bundle", "data_bundle"):
            from repo_paths import REPO_ROOT
            root = str(REPO_ROOT / "data_bundle")
        results_dir = str(_Path(root) / "fig2_instantaneous_demo" / "cy_n0p7_lam5") + "/"
    else:
        from repo_paths import FROZEN_INST
        results_dir = str(FROZEN_INST / 'tbnn_debug_results_constriction_new' / 'iteration_12_20251008_050525') + '/'
    
    print(f"Loading results from: {results_dir}")
    
    # Plot all flow fields (creates 7 separate figures automatically)
    # When run as script: save to disk, don't show interactively
    result = plot_individual_flow_fields(results_dir, dpi=600, save_fig=True, show_fig=False)
    
    print(f"\nCreated {len(result['figures'])} plots total!")
    print("")


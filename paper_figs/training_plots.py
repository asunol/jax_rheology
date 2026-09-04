#!/usr/bin/env python
"""Constriction training plots: loss, fields, and PIV-resolution overlays.

Used by the instantaneous trainer to write the per-run diagnostic figures.
Nothing here trains or differentiates; it only draws arrays the trainer
already computed.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os

# Import models for strain rate computation
from jax_rheology import models


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def add_constriction_outline(ax, domain, color='white', linewidth=1, alpha=0.8):
    """Add constriction outline to plots."""
    center_x = (domain[0][1] + domain[0][0]) / 2  # x = 4.0 for default domain
    radius = 1.5
    
    # Bottom semicircle
    theta_bottom = jnp.linspace(0, jnp.pi, 50)
    x_bottom = center_x + radius * jnp.cos(theta_bottom)
    y_bottom = radius * jnp.sin(theta_bottom)
    ax.plot(x_bottom, y_bottom, color=color, linewidth=linewidth, alpha=alpha)
    
    # Top semicircle
    theta_top = jnp.linspace(jnp.pi, 2*jnp.pi, 50)
    x_top = center_x + radius * jnp.cos(theta_top)
    y_top = domain[1][1] + radius * jnp.sin(theta_top)
    ax.plot(x_top, y_top, color=color, linewidth=linewidth, alpha=alpha)


def plot_model_comparison_constriction(tbnn_result, ref_result, domain, title="TBNN vs Reference Constriction", 
                                      save_plots=False, output_dir='.', file_prefix='model_comparison_constriction'):
    """Plot side-by-side comparison of TBNN and reference flow fields with constriction."""
    print(f"\nMODEL COMPARISON VISUALIZATION: {title}")
    
    # Extract velocity fields
    tbnn_vx = tbnn_result.velocity[0].data
    tbnn_vy = tbnn_result.velocity[1].data
    ref_vx = ref_result.velocity[0].data
    ref_vy = ref_result.velocity[1].data
    
    # Compute velocity magnitudes
    tbnn_vel_mag = jnp.sqrt(tbnn_vx**2 + tbnn_vy**2)
    ref_vel_mag = jnp.sqrt(ref_vx**2 + ref_vy**2)
    
    # Compute differences
    diff_vx = tbnn_vx - ref_vx
    diff_vy = tbnn_vy - ref_vy
    diff_mag = jnp.sqrt(diff_vx**2 + diff_vy**2)
    
    print(f"   Reference velocity range: vx=[{float(jnp.min(ref_vx)):.4f}, {float(jnp.max(ref_vx)):.4f}], vy=[{float(jnp.min(ref_vy)):.6f}, {float(jnp.max(ref_vy)):.6f}]")
    print(f"   TBNN velocity range: vx=[{float(jnp.min(tbnn_vx)):.4f}, {float(jnp.max(tbnn_vx)):.4f}], vy=[{float(jnp.min(tbnn_vy)):.6f}, {float(jnp.max(tbnn_vy)):.6f}]")
    print(f"   Difference range: vx=[{float(jnp.min(diff_vx)):.6f}, {float(jnp.max(diff_vx)):.6f}], vy=[{float(jnp.min(diff_vy)):.6f}, {float(jnp.max(diff_vy)):.6f}]")
    print(f"   Max difference magnitude: {float(jnp.max(diff_mag)):.6f}")
    print(f"   Mean difference magnitude: {float(jnp.mean(diff_mag)):.6f}")
    
    # Create comprehensive comparison plot
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    extent = [domain[0][0], domain[0][1], domain[1][0], domain[1][1]]
    
    # Row 1: X-velocity comparison
    im1 = axes[0,0].imshow(ref_vx.T, origin='lower', cmap='coolwarm', extent=extent, aspect='auto')
    add_constriction_outline(axes[0,0], domain)
    axes[0,0].set_title('Reference: X-velocity')
    axes[0,0].set_xlabel('x'); axes[0,0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0,0])
    
    im2 = axes[0,1].imshow(tbnn_vx.T, origin='lower', cmap='coolwarm', extent=extent, aspect='auto')
    add_constriction_outline(axes[0,1], domain)
    axes[0,1].set_title('TBNN: X-velocity')  
    axes[0,1].set_xlabel('x'); axes[0,1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0,1])
    
    im3 = axes[0,2].imshow(diff_vx.T, origin='lower', cmap='RdBu_r', extent=extent, aspect='auto')
    add_constriction_outline(axes[0,2], domain)
    axes[0,2].set_title('Difference: TBNN - Ref (X-vel)')
    axes[0,2].set_xlabel('x'); axes[0,2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[0,2])
    
    # Velocity profiles at channel center
    mid_x_idx = ref_vx.shape[0] // 2
    y_coords = jnp.linspace(domain[1][0], domain[1][1], ref_vx.shape[1])
    ref_profile = ref_vx[mid_x_idx, :]
    tbnn_profile = tbnn_vx[mid_x_idx, :]
    
    axes[0,3].plot(ref_profile, y_coords, 'b-', linewidth=3, label='Reference', alpha=0.8)
    axes[0,3].plot(tbnn_profile, y_coords, 'r--', linewidth=3, label='TBNN', alpha=0.8)
    axes[0,3].set_xlabel('vx velocity')
    axes[0,3].set_ylabel('y position')
    axes[0,3].set_title('Velocity Profiles (Channel Center)')
    axes[0,3].grid(True, alpha=0.3)
    axes[0,3].legend()
    
    # Row 2: Velocity magnitude comparison
    im7 = axes[1,0].imshow(ref_vel_mag.T, origin='lower', cmap='viridis', extent=extent, aspect='auto')
    add_constriction_outline(axes[1,0], domain)
    axes[1,0].set_title('Reference: Velocity Magnitude')
    axes[1,0].set_xlabel('x'); axes[1,0].set_ylabel('y')
    plt.colorbar(im7, ax=axes[1,0])
    
    im8 = axes[1,1].imshow(tbnn_vel_mag.T, origin='lower', cmap='viridis', extent=extent, aspect='auto')
    add_constriction_outline(axes[1,1], domain)
    axes[1,1].set_title('TBNN: Velocity Magnitude')
    axes[1,1].set_xlabel('x'); axes[1,1].set_ylabel('y')
    plt.colorbar(im8, ax=axes[1,1])
    
    im9 = axes[1,2].imshow(diff_mag.T, origin='lower', cmap='plasma', extent=extent, aspect='auto')
    add_constriction_outline(axes[1,2], domain)
    axes[1,2].set_title('Difference Magnitude')
    axes[1,2].set_xlabel('x'); axes[1,2].set_ylabel('y')
    plt.colorbar(im9, ax=axes[1,2])
    
    # Statistics summary
    ref_max_vel = float(jnp.max(ref_vel_mag))
    tbnn_max_vel = float(jnp.max(tbnn_vel_mag))
    max_diff = float(jnp.max(diff_mag))
    mean_diff = float(jnp.mean(diff_mag))
    rel_error = (mean_diff / ref_max_vel) * 100 if ref_max_vel > 0 else 0
    
    stats_text = f"""COMPARISON STATISTICS:
    
Reference Max Velocity: {ref_max_vel:.6f}
TBNN Max Velocity: {tbnn_max_vel:.6f}

Max Difference: {max_diff:.6f}
Mean Difference: {mean_diff:.6f}
Relative Error: {rel_error:.2f}%

TBNN Status:
{'close to reference' if rel_error < 10 else 'still learning'}
{'good match' if rel_error < 5 else 'differences present'}"""
    
    axes[1,3].text(0.05, 0.95, stats_text, transform=axes[1,3].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1,3].set_title('Comparison Statistics')
    axes[1,3].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots:
        filename = f"{file_prefix}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"   Saved plot: {filepath}")
    
    plt.show()
    
    return {
        'ref_max_vel': ref_max_vel,
        'tbnn_max_vel': tbnn_max_vel,
        'max_difference': max_diff,
        'mean_difference': mean_diff,
        'relative_error_percent': rel_error
    }


def plot_strain_rate_histogram_constriction(final_result, domain, title="Strain Rate Distribution Constriction",
                                           save_plots=False, output_dir='.', file_prefix='strain_rate_histogram_constriction'):
    """Plot histogram of local strain rates in the computational domain with constriction."""
    print(f"\nSTRAIN RATE DISTRIBUTION ANALYSIS: {title}")
    
    # Extract velocity field
    vx = final_result.velocity[0].data
    vy = final_result.velocity[1].data
    
    print(f"   Domain: {domain}")
    print(f"   Grid shape: {vx.shape}")
    
    try:
        # Compute strain rate tensor from velocity field
        velocity_field = final_result.velocity
        
        # Compute strain rate and rotation tensors
        S, R = models.compute_S_R(velocity_field)
        
        # Compute shear rate magnitude: gammadot = sqrt(2 * S:S)
        shear_rate_field = models.compute_shear_rate(S, eps=1e-8)
        
        # Convert to numpy for analysis
        strain_rates = np.array(shear_rate_field).flatten()
        
        # Remove any NaN or infinite values
        valid_strain_rates = strain_rates[np.isfinite(strain_rates)]
        
        print(f"   Total points: {len(strain_rates)}")
        print(f"   Valid points: {len(valid_strain_rates)}")
        print(f"   Strain rate range: [{np.min(valid_strain_rates):.6f}, {np.max(valid_strain_rates):.6f}]")
        print(f"   Mean strain rate: {np.mean(valid_strain_rates):.6f}")
        print(f"   Median strain rate: {np.median(valid_strain_rates):.6f}")
        print(f"   95th percentile: {np.percentile(valid_strain_rates, 95):.6f}")
        
        # Create histogram
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Linear scale histogram
        axes[0].hist(valid_strain_rates, bins=50, alpha=0.7, color='blue', 
                    edgecolor='black', density=True)
        axes[0].set_xlabel('Strain Rate (s⁻¹)')
        axes[0].set_ylabel('Probability Density')
        axes[0].set_title(f'Strain Rate Distribution (Linear Scale)\n{title}')
        axes[0].grid(True, alpha=0.3)
        
        # Add vertical lines for statistics
        axes[0].axvline(np.mean(valid_strain_rates), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(valid_strain_rates):.4f}')
        axes[0].axvline(np.median(valid_strain_rates), color='orange', linestyle='--', 
                       label=f'Median: {np.median(valid_strain_rates):.4f}')
        axes[0].legend()
        
        # Log scale histogram (if strain rates span multiple orders)
        if np.max(valid_strain_rates) / np.min(valid_strain_rates) > 10:
            # Use log bins for wide range
            log_bins = np.logspace(np.log10(np.max([np.min(valid_strain_rates), 1e-6])), 
                                  np.log10(np.max(valid_strain_rates)), 50)
            axes[1].hist(valid_strain_rates, bins=log_bins, alpha=0.7, color='green', 
                        edgecolor='black', density=True)
            axes[1].set_xscale('log')
            axes[1].set_xlabel('Strain Rate (s⁻¹, log scale)')
            axes[1].set_ylabel('Probability Density')
            axes[1].set_title(f'Strain Rate Distribution (Log Scale)\n{title}')
            axes[1].grid(True, alpha=0.3)
            
            # Add statistical lines
            axes[1].axvline(np.mean(valid_strain_rates), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(valid_strain_rates):.4f}')
            axes[1].axvline(np.median(valid_strain_rates), color='orange', linestyle='--', 
                           label=f'Median: {np.median(valid_strain_rates):.4f}')
            axes[1].legend()
        else:
            # If range is narrow, show cumulative distribution instead
            sorted_rates = np.sort(valid_strain_rates)
            cumulative = np.arange(1, len(sorted_rates) + 1) / len(sorted_rates)
            axes[1].plot(sorted_rates, cumulative, 'g-', linewidth=2)
            axes[1].set_xlabel('Strain Rate (s⁻¹)')
            axes[1].set_ylabel('Cumulative Probability')
            axes[1].set_title(f'Cumulative Distribution\n{title}')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots:
            filename = f"{file_prefix}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"   Saved plot: {filepath}")
        
        plt.show()
        
        # Analyze strain rate regimes
        low_strain = np.sum(valid_strain_rates < 0.1) / len(valid_strain_rates) * 100
        med_strain = np.sum((valid_strain_rates >= 0.1) & (valid_strain_rates < 10)) / len(valid_strain_rates) * 100
        high_strain = np.sum(valid_strain_rates >= 10) / len(valid_strain_rates) * 100
        
        print(f"\n   STRAIN RATE REGIME ANALYSIS:")
        print(f"   Low strain rates (<0.1 s^-^1): {low_strain:.1f}%")
        print(f"   Medium strain rates (0.1-10 s^-^1): {med_strain:.1f}%")
        print(f"   High strain rates (>10 s^-^1): {high_strain:.1f}%")
        
        return {
            'strain_rates': valid_strain_rates,
            'min_strain_rate': np.min(valid_strain_rates),
            'max_strain_rate': np.max(valid_strain_rates),
            'mean_strain_rate': np.mean(valid_strain_rates),
            'median_strain_rate': np.median(valid_strain_rates),
            'percentile_95': np.percentile(valid_strain_rates, 95),
            'low_strain_percent': low_strain,
            'med_strain_percent': med_strain,
            'high_strain_percent': high_strain
        }
        
    except Exception as e:
        print(f"   Strain rate analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_training_progress(loss_history, gradient_magnitudes, learning_rate, num_steps,
                          save_plots=False, output_dir='.', file_prefix='training_progress_constriction',
                          stage1_end=None):
    """Plot training progress: loss convergence and gradient evolution.
    
    Args:
        stage1_end: If provided, marks the end of stage 1 (etainf ON) with a vertical line
    """
    print(f"\nTRAINING PROGRESS ANALYSIS")
    
    # Convert to numpy arrays for plotting
    losses = np.array(loss_history)
    grad_mags = np.array(gradient_magnitudes)
    steps = np.arange(len(losses))
    
    print(f"   Training steps completed: {len(losses)-1}")
    print(f"   Loss reduction: {losses[0]:.6e} -> {losses[-1]:.6e}")
    print(f"   Final gradient magnitude: {grad_mags[-1]:.6e}")
    
    # Create training plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Loss convergence (linear scale)
    axes[0,0].plot(steps, losses, 'b-', linewidth=2, marker='o', markersize=4)
    if stage1_end is not None and stage1_end > 0:
        axes[0,0].axvline(stage1_end, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Stage 1 to 2')
        axes[0,0].legend()
    axes[0,0].set_xlabel('Training Step')
    axes[0,0].set_ylabel('Loss')
    title_suffix = ' (2-stage)' if stage1_end is not None else ''
    axes[0,0].set_title(f'Loss Convergence (Linear Scale){title_suffix}\nLR={learning_rate:.0e}')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_yscale('linear')
    
    # Loss convergence (log scale) if beneficial
    if len(losses) > 2 and losses[0] / losses[-1] > 2:
        axes[0,1].semilogy(steps, losses, 'b-', linewidth=2, marker='o', markersize=4)
        if stage1_end is not None and stage1_end > 0:
            axes[0,1].axvline(stage1_end, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Stage 1 to 2')
            axes[0,1].legend()
        axes[0,1].set_xlabel('Training Step')
        axes[0,1].set_ylabel('Loss (log scale)')
        axes[0,1].set_title(f'Loss Convergence (Log Scale){title_suffix}')
        axes[0,1].grid(True, alpha=0.3)
    else:
        # If log scale doesn't help, show loss reduction instead
        loss_reduction = losses[0] - losses
        axes[0,1].plot(steps, loss_reduction, 'g-', linewidth=2, marker='s', markersize=4)
        if stage1_end is not None and stage1_end > 0:
            axes[0,1].axvline(stage1_end, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Stage 1 to 2')
            axes[0,1].legend()
        axes[0,1].set_xlabel('Training Step')
        axes[0,1].set_ylabel('Loss Reduction')
        axes[0,1].set_title(f'Cumulative Loss Reduction{title_suffix}')
        axes[0,1].grid(True, alpha=0.3)
    
    # Gradient magnitude evolution
    if len(grad_mags) > 1:
        # Ensure dimensions match
        min_len = min(len(steps), len(grad_mags))
        axes[1,0].plot(steps[:min_len], grad_mags[:min_len], 'r-', linewidth=2, marker='^', markersize=4)
        if stage1_end is not None and stage1_end > 0:
            axes[1,0].axvline(stage1_end, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Stage 1 to 2')
            axes[1,0].legend()
    else:
        axes[1,0].text(0.5, 0.5, 'Not enough gradient data\nfor evolution plot', 
                       ha='center', va='center', transform=axes[1,0].transAxes)
    axes[1,0].set_xlabel('Training Step')
    axes[1,0].set_ylabel('Gradient Magnitude')
    axes[1,0].set_title('Gradient Magnitude Evolution')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_yscale('log')
    
    # Training statistics summary
    if len(losses) > 1:
        final_improvement = (losses[0] - losses[-1]) / losses[0] * 100
        avg_improvement_per_step = final_improvement / (len(losses) - 1)
        
        # Detect convergence patterns
        if len(losses) >= 3:
            last_few_changes = np.abs(np.diff(losses[-3:]))
            is_converging = np.all(last_few_changes < 0.01 * losses[0])
        else:
            is_converging = False
        
        stage_info = ""
        if stage1_end is not None and stage1_end > 0:
            stage_info = f"\nMode: 2-stage (switch @ step {stage1_end})"
        
        stats_text = f"""TRAINING STATISTICS:{stage_info}

Steps: {len(losses)-1}
Learning Rate: {learning_rate:.0e}

Initial Loss: {losses[0]:.4e}
Final Loss: {losses[-1]:.4e}
Total Improvement: {final_improvement:.2f}%
Avg per Step: {avg_improvement_per_step:.2f}%

Initial |∇|: {grad_mags[0]:.4e}
Final |∇|: {grad_mags[-1]:.4e}

Status: {'Converging' if is_converging else 'Still Learning'}
"""
    else:
        stats_text = "Single step training - no progress data"
    
    axes[1,1].text(0.05, 0.95, stats_text, transform=axes[1,1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1,1].set_title('Training Statistics')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots:
        filename = f"{file_prefix}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"   Saved plot: {filepath}")
    
    plt.show()
    
    # Convergence analysis
    if len(losses) > 2:
        print(f"\n   CONVERGENCE ANALYSIS:")
        recent_change = abs(losses[-1] - losses[-2]) / losses[-2] * 100
        print(f"   Last step loss change: {recent_change:.3f}%")
        
        if recent_change < 0.1:
            print("   Training appears to be converging")
        elif recent_change < 1.0:
            print("   Training is slowing down")
        else:
            print("   Training is still making good progress")
    
    return {
        'loss_history': losses,
        'gradient_magnitudes': grad_mags,
        'total_steps': len(losses) - 1,
        'final_loss': losses[-1],
        'initial_loss': losses[0],
        'loss_reduction': losses[0] - losses[-1],
        'relative_improvement': (losses[0] - losses[-1]) / losses[0] * 100 if losses[0] > 0 else 0,
        'final_gradient_magnitude': grad_mags[-1] if len(grad_mags) > 0 else 0
    }


def compute_tbnn_viscosities_vs_strain_rate(model_info, strain_rates=None):
    """Compute TBNN viscosities for a range of strain rates using mixture-of-sigmoids model."""
    if strain_rates is None:
        strain_rates = jnp.logspace(-2, 2, 100)  # 0.01 to 100 s^-1
    
    viscosities = []
    
    for gamma_dot in strain_rates:
        try:
            # For simple shear with rate gammadot: 
            # S = [[0, gammadot/2], [gammadot/2, 0]]
            # R = [[0, gammadot/2], [-gammadot/2, 0]] (skew-symmetric part)
            
            # Compute invariants as done in compute_invariants_2d:
            # I1 = sum(S^2) = 2 * (gammadot/2)^2 = gammadot^2/2
            # I2 = -sum(R^2) = -2 * (gammadot/2)^2 = -gammadot^2/2
            I1_invariant = (gamma_dot**2) / 2.0  # tr(S^T S)
            I2_invariant = -(gamma_dot**2) / 2.0  # -tr(R^T R)
            
            # Create invariants with spatial dimensions (1x1 grid for single point analysis)
            # TBNN expects shape (2, H, W)
            invariants_aux = jnp.array([
                [[I1_invariant]],  # I1 with shape (1, 1)
                [[I2_invariant]]   # I2 with shape (1, 1)
            ])  # Final shape: (2, 1, 1)
            
            # Create gamma_dot field (1x1 grid)
            gamma_dot_field = jnp.array([[gamma_dot]])  # Shape: (1, 1)
            
            # Use the bounded mixture-of-sigmoids viscosity model to compute viscosity
            params = model_info['params']
            eta_field = model_info['model'].apply(params, gamma_dot_field, invariants_aux)
            
            # Extract scalar viscosity value
            viscosity = float(eta_field[0, 0])
            viscosities.append(viscosity)
            
        except Exception as e:
            print(f"Warning: Failed to compute viscosity for gammadot={gamma_dot:.3f}: {e}")
            # Use a default value to maintain array shape
            viscosities.append(float(model_info['eta_init']))
    
    return jnp.array(viscosities)


def plot_viscosity_strain_rate(model_info, title="TBNN Viscosity vs Strain Rate",
                              save_plots=False, output_dir='.', file_prefix='viscosity_strain_rate',
                              reference_model=None, reference_params=None,
                              additional_viscosities=None, additional_labels=None):
    """Plot TBNN viscosity as function of strain rate."""
    print(f"\n VISCOSITY-STRAIN RATE ANALYSIS: {title}")
    
    # Create range of strain rates for testing (log scale from 0.01 to 100)
    strain_rates = jnp.logspace(-2, 2, 100)  # 0.01 to 100 s^-1
    
    # Compute TBNN viscosities
    viscosities = compute_tbnn_viscosities_vs_strain_rate(model_info, strain_rates)
    
    # Compute ground truth viscosity if reference model is provided
    ground_truth_viscosities = None
    if reference_model is not None and reference_params is not None:
        ground_truth_viscosities = []
        for gamma_dot in strain_rates:
            if reference_model == 'newtonian':
                viscosity = reference_params  # Constant viscosity
            elif reference_model == 'carreau_yasuda':
                # Carreau-Yasuda: eta = eta_inf + (eta_0 - eta_inf) [1 + (lamgammadot)^a]^((n-1)/a)
                eta_inf, eta_0, lambda_, n, a = reference_params[:5]
                term = (1 + (lambda_ * gamma_dot)**a)**((n - 1) / a)
                viscosity = eta_inf + (eta_0 - eta_inf) * term
            elif reference_model == 'power_law':
                # Power-law: eta = K * gammadot^(n-1)
                K, n = reference_params[:2]
                viscosity = K * (gamma_dot + 1e-8)**(n - 1)  # Small regularization
            else:
                viscosity = 1.0  # Default fallback
            ground_truth_viscosities.append(float(viscosity))
        ground_truth_viscosities = jnp.array(ground_truth_viscosities)
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Linear scale plot
    axes[0].plot(strain_rates, viscosities, 'b-', linewidth=2, label='TBNN Viscosity')
    
    # Add ground truth if available
    if ground_truth_viscosities is not None:
        axes[0].plot(strain_rates, ground_truth_viscosities, 'r-', linewidth=2, 
                    label=f'Ground Truth ({reference_model.title()})', alpha=0.8)
    
    # Add additional viscosity curves if provided
    if additional_viscosities is not None and additional_labels is not None:
        colors = ['g--', 'm:', 'c-.', 'y-']
        for i, (add_visc, add_label) in enumerate(zip(additional_viscosities, additional_labels)):
            color = colors[i % len(colors)]
            axes[0].plot(strain_rates, add_visc, color, linewidth=2, 
                        label=add_label, alpha=0.7)
    
    axes[0].set_xlabel('Strain Rate (s⁻¹)')
    axes[0].set_ylabel('Viscosity (Pa·s)')
    axes[0].set_title(f'Viscosity vs Strain Rate\n{title}')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Log-log scale plot
    axes[1].loglog(strain_rates, viscosities, 'b-', linewidth=2, label='TBNN Viscosity')
    
    # Add ground truth if available
    if ground_truth_viscosities is not None:
        axes[1].loglog(strain_rates, ground_truth_viscosities, 'r-', linewidth=2, 
                      label=f'Ground Truth ({reference_model.title()})', alpha=0.8)
    
    # Add additional viscosity curves if provided
    if additional_viscosities is not None and additional_labels is not None:
        colors = ['g--', 'm:', 'c-.', 'y-']
        for i, (add_visc, add_label) in enumerate(zip(additional_viscosities, additional_labels)):
            color = colors[i % len(colors)]
            axes[1].loglog(strain_rates, add_visc, color, linewidth=2, 
                          label=add_label, alpha=0.7)
    
    axes[1].set_xlabel('Strain Rate (s⁻¹)')
    axes[1].set_ylabel('Viscosity (Pa·s)')
    axes[1].set_title(f'Viscosity vs Strain Rate (Log-Log)\n{title}')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots:
        filename = f"{file_prefix}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"   Saved plot: {filepath}")
    
    plt.show()
    
    # Print analysis
    eta_init = model_info.get('eta_init', 1.0)
    print(f"   Viscosity range: [{float(jnp.min(viscosities)):.6f}, {float(jnp.max(viscosities)):.6f}]")
    print(f"   Initialized viscosity: {eta_init:.6f}")
    print(f"   Low strain rate viscosity: {float(viscosities[0]):.6f}")
    print(f"   High strain rate viscosity: {float(viscosities[-1]):.6f}")
    
    # Check for shear-thinning/thickening behavior
    low_strain_visc = float(viscosities[:10].mean())
    high_strain_visc = float(viscosities[-10:].mean())
    
    if high_strain_visc < low_strain_visc * 0.9:
        print("    Shear-thinning behavior detected")
    elif high_strain_visc > low_strain_visc * 1.1:
        print("    Shear-thickening behavior detected")
    else:
        print("   Nearly Newtonian behavior")
    
    return {
        'strain_rates': strain_rates,
        'viscosities': viscosities,
        'eta_init': eta_init,
        'low_strain_viscosity': low_strain_visc,
        'high_strain_viscosity': high_strain_visc
    }


def plot_piv_resolution_comparison(ref_full_res, ref_piv_res, domain, 
                                   title="PIV Resolution Effect on Reference Data",
                                   save_plots=False, output_dir='.', file_prefix='piv_resolution_comparison',
                                   noise_params=None):
    """
    Plot comparison of full-resolution vs PIV-downsampled (and possibly noisy) reference data.
    
    Args:
        ref_full_res: Full-resolution reference velocity (tuple of (vx, vy) arrays)
        ref_piv_res: PIV-downsampled reference velocity (tuple of (vx, vy) arrays)
        domain: Domain bounds
        noise_params: Dict with noise parameters if noise was added (for title)
    """
    print(f"\nPIV RESOLUTION COMPARISON: {title}")
    
    # Extract full-resolution data
    ref_full_vx, ref_full_vy = ref_full_res
    # Handle both (T,H,W) and (H,W) formats
    if ref_full_vx.ndim == 3:
        ref_full_vx = ref_full_vx[-1]  # Last timestep
        ref_full_vy = ref_full_vy[-1]
    
    # Extract PIV-resolution data
    ref_piv_vx, ref_piv_vy = ref_piv_res
    if ref_piv_vx.ndim == 3:
        ref_piv_vx = ref_piv_vx[-1]  # Last timestep
        ref_piv_vy = ref_piv_vy[-1]
    
    # Compute magnitudes
    ref_full_mag = jnp.sqrt(ref_full_vx**2 + ref_full_vy**2)
    ref_piv_mag = jnp.sqrt(ref_piv_vx**2 + ref_piv_vy**2)
    
    print(f"   Full resolution shape: {ref_full_vx.shape}")
    print(f"   PIV resolution shape: {ref_piv_vx.shape}")
    print(f"   Resolution reduction: {ref_full_vx.shape[0] * ref_full_vx.shape[1] / (ref_piv_vx.shape[0] * ref_piv_vx.shape[1]):.1f}x fewer points")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    extent_full = [domain[0][0], domain[0][1], domain[1][0], domain[1][1]]
    
    # Determine extent for PIV data (might be different due to window centering)
    # For simplicity, use same extent (imshow will interpolate)
    extent_piv = extent_full
    
    # Row 1: X-velocity
    im1 = axes[0,0].imshow(ref_full_vx.T, origin='lower', cmap='coolwarm', extent=extent_full, aspect='auto')
    add_constriction_outline(axes[0,0], domain)
    axes[0,0].set_title(f'Full Resolution: X-velocity\n{ref_full_vx.shape[0]}×{ref_full_vx.shape[1]} points')
    axes[0,0].set_xlabel('x'); axes[0,0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0,0])
    
    im2 = axes[0,1].imshow(ref_piv_vx.T, origin='lower', cmap='coolwarm', extent=extent_piv, aspect='auto', interpolation='nearest')
    add_constriction_outline(axes[0,1], domain)
    noise_label = " + Noise" if noise_params else ""
    axes[0,1].set_title(f'PIV Resolution{noise_label}: X-velocity\n{ref_piv_vx.shape[0]}×{ref_piv_vx.shape[1]} vectors')
    axes[0,1].set_xlabel('x'); axes[0,1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Velocity profiles at channel center
    mid_x_idx_full = ref_full_vx.shape[0] // 2
    mid_x_idx_piv = ref_piv_vx.shape[0] // 2
    y_coords_full = jnp.linspace(domain[1][0], domain[1][1], ref_full_vx.shape[1])
    y_coords_piv = jnp.linspace(domain[1][0], domain[1][1], ref_piv_vx.shape[1])
    
    axes[0,2].plot(ref_full_vx[mid_x_idx_full, :], y_coords_full, 'b-', linewidth=2, label='Full Resolution', alpha=0.8)
    axes[0,2].plot(ref_piv_vx[mid_x_idx_piv, :], y_coords_piv, 'ro', markersize=6, label=f'PIV Resolution{noise_label}', alpha=0.8)
    axes[0,2].set_xlabel('vx velocity')
    axes[0,2].set_ylabel('y position')
    axes[0,2].set_title('Velocity Profiles (Channel Center)')
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].legend()
    
    # Row 2: Velocity magnitude
    im4 = axes[1,0].imshow(ref_full_mag.T, origin='lower', cmap='viridis', extent=extent_full, aspect='auto')
    add_constriction_outline(axes[1,0], domain)
    axes[1,0].set_title('Full Resolution: Velocity Magnitude')
    axes[1,0].set_xlabel('x'); axes[1,0].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1,0])
    
    im5 = axes[1,1].imshow(ref_piv_mag.T, origin='lower', cmap='viridis', extent=extent_piv, aspect='auto', interpolation='nearest')
    add_constriction_outline(axes[1,1], domain)
    axes[1,1].set_title(f'PIV Resolution{noise_label}: Velocity Magnitude')
    axes[1,1].set_xlabel('x'); axes[1,1].set_ylabel('y')
    plt.colorbar(im5, ax=axes[1,1])
    
    # Statistics
    stats_text = f"""PIV RESOLUTION COMPARISON:

Full Resolution:
  Grid: {ref_full_vx.shape[0]} × {ref_full_vx.shape[1]}
  Total points: {ref_full_vx.shape[0] * ref_full_vx.shape[1]}
  Max velocity: {float(jnp.max(ref_full_mag)):.6f}

PIV Resolution{noise_label}:
  Grid: {ref_piv_vx.shape[0]} × {ref_piv_vx.shape[1]}
  Total vectors: {ref_piv_vx.shape[0] * ref_piv_vx.shape[1]}
  Max velocity: {float(jnp.max(ref_piv_mag)):.6f}
  
Data Reduction:
  Factor: {ref_full_vx.shape[0] * ref_full_vx.shape[1] / (ref_piv_vx.shape[0] * ref_piv_vx.shape[1]):.1f}x fewer points
  ({100 * ref_piv_vx.shape[0] * ref_piv_vx.shape[1] / (ref_full_vx.shape[0] * ref_full_vx.shape[1]):.1f}% of original)
"""
    
    if noise_params:
        stats_text += f"""
Noise Parameters:
  Level: {noise_params['p_percent']:.1f}% of U95
  Correlation: {noise_params['corr_frac']:.2f}
  β (gradient): {noise_params['beta_grad']:.2f}
  Static bias: {noise_params['use_bias']}
"""
    
    axes[1,2].text(0.05, 0.95, stats_text, transform=axes[1,2].transAxes, 
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1,2].set_title('Resolution Statistics')
    axes[1,2].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots:
        filename = f"{file_prefix}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"   Saved plot: {filepath}")
    
    plt.show()
    
    return {
        'full_res_shape': ref_full_vx.shape,
        'piv_res_shape': ref_piv_vx.shape,
        'reduction_factor': ref_full_vx.shape[0] * ref_full_vx.shape[1] / (ref_piv_vx.shape[0] * ref_piv_vx.shape[1]),
        'full_max_vel': float(jnp.max(ref_full_mag)),
        'piv_max_vel': float(jnp.max(ref_piv_mag))
    }


def plot_three_way_flow_comparison_constriction(initial_result, final_result, ref_result, domain, 
                                  title="Flow Field Comparison: Initial vs Final vs Ground Truth (Constriction)",
                                  save_plots=False, output_dir='.', file_prefix='three_way_comparison_constriction'):
    """Plot three-way comparison of flow fields: Initial TBNN, Final TBNN, and Ground Truth with constriction."""
    print(f"\nThree-way flow field comparison: {title}")
    
    # Extract velocity fields
    initial_vx = initial_result.velocity[0].data
    initial_vy = initial_result.velocity[1].data
    final_vx = final_result.velocity[0].data
    final_vy = final_result.velocity[1].data
    ref_vx = ref_result.velocity[0].data
    ref_vy = ref_result.velocity[1].data
    
    # Compute velocity magnitudes
    initial_vel_mag = jnp.sqrt(initial_vx**2 + initial_vy**2)
    final_vel_mag = jnp.sqrt(final_vx**2 + final_vy**2)
    ref_vel_mag = jnp.sqrt(ref_vx**2 + ref_vy**2)
    
    # Compute differences from ground truth
    diff_initial_vx = initial_vx - ref_vx
    diff_final_vx = final_vx - ref_vx
    diff_initial_mag = jnp.sqrt((initial_vx - ref_vx)**2 + (initial_vy - ref_vy)**2)
    diff_final_mag = jnp.sqrt((final_vx - ref_vx)**2 + (final_vy - ref_vy)**2)
    
    print(f"   Reference velocity range: vx=[{float(jnp.min(ref_vx)):.4f}, {float(jnp.max(ref_vx)):.4f}]")
    print(f"   Initial TBNN velocity range: vx=[{float(jnp.min(initial_vx)):.4f}, {float(jnp.max(initial_vx)):.4f}]")
    print(f"   Final TBNN velocity range: vx=[{float(jnp.min(final_vx)):.4f}, {float(jnp.max(final_vx)):.4f}]")
    print(f"   Initial TBNN max error: {float(jnp.max(diff_initial_mag)):.6f}")
    print(f"   Final TBNN max error: {float(jnp.max(diff_final_mag)):.6f}")
    print(f"   Error improvement: {float(jnp.max(diff_initial_mag) - jnp.max(diff_final_mag)):.6f}")
    
    # Create comprehensive comparison plot
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    
    extent = [domain[0][0], domain[0][1], domain[1][0], domain[1][1]]
    
    # Row 1: X-velocity fields
    im1 = axes[0,0].imshow(ref_vx.T, origin='lower', cmap='coolwarm', extent=extent, aspect='auto')
    add_constriction_outline(axes[0,0], domain)
    axes[0,0].set_title('Ground Truth: X-velocity')
    axes[0,0].set_xlabel('x'); axes[0,0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0,0])
    
    im2 = axes[0,1].imshow(initial_vx.T, origin='lower', cmap='coolwarm', extent=extent, aspect='auto')
    add_constriction_outline(axes[0,1], domain)
    axes[0,1].set_title('Initial TBNN: X-velocity')  
    axes[0,1].set_xlabel('x'); axes[0,1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0,1])
    
    im3 = axes[0,2].imshow(final_vx.T, origin='lower', cmap='coolwarm', extent=extent, aspect='auto')
    add_constriction_outline(axes[0,2], domain)
    axes[0,2].set_title('Final TBNN: X-velocity')
    axes[0,2].set_xlabel('x'); axes[0,2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[0,2])
    
    # Row 2: Velocity magnitude fields
    im4 = axes[1,0].imshow(ref_vel_mag.T, origin='lower', cmap='viridis', extent=extent, aspect='auto')
    add_constriction_outline(axes[1,0], domain)
    axes[1,0].set_title('Ground Truth: Velocity Magnitude')
    axes[1,0].set_xlabel('x'); axes[1,0].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1,0])
    
    im5 = axes[1,1].imshow(initial_vel_mag.T, origin='lower', cmap='viridis', extent=extent, aspect='auto')
    add_constriction_outline(axes[1,1], domain)
    axes[1,1].set_title('Initial TBNN: Velocity Magnitude')
    axes[1,1].set_xlabel('x'); axes[1,1].set_ylabel('y')
    plt.colorbar(im5, ax=axes[1,1])
    
    im6 = axes[1,2].imshow(final_vel_mag.T, origin='lower', cmap='viridis', extent=extent, aspect='auto')
    add_constriction_outline(axes[1,2], domain)
    axes[1,2].set_title('Final TBNN: Velocity Magnitude')
    axes[1,2].set_xlabel('x'); axes[1,2].set_ylabel('y')
    plt.colorbar(im6, ax=axes[1,2])
    
    # Row 3: Error fields and velocity profiles
    im7 = axes[2,0].imshow(diff_initial_mag.T, origin='lower', cmap='plasma', extent=extent, aspect='auto')
    add_constriction_outline(axes[2,0], domain)
    axes[2,0].set_title('Initial TBNN: Error Magnitude')
    axes[2,0].set_xlabel('x'); axes[2,0].set_ylabel('y')
    plt.colorbar(im7, ax=axes[2,0])
    
    im8 = axes[2,1].imshow(diff_final_mag.T, origin='lower', cmap='plasma', extent=extent, aspect='auto')
    add_constriction_outline(axes[2,1], domain)
    axes[2,1].set_title('Final TBNN: Error Magnitude')
    axes[2,1].set_xlabel('x'); axes[2,1].set_ylabel('y')
    plt.colorbar(im8, ax=axes[2,1])
    
    # Velocity profiles at channel center
    mid_x_idx = ref_vx.shape[0] // 2
    y_coords = jnp.linspace(domain[1][0], domain[1][1], ref_vx.shape[1])
    ref_profile = ref_vx[mid_x_idx, :]
    initial_profile = initial_vx[mid_x_idx, :]
    final_profile = final_vx[mid_x_idx, :]
    
    axes[2,2].plot(ref_profile, y_coords, 'r-', linewidth=3, label='Ground Truth', alpha=0.8)
    axes[2,2].plot(initial_profile, y_coords, 'b--', linewidth=2, label='Initial TBNN', alpha=0.7)
    axes[2,2].plot(final_profile, y_coords, 'g-', linewidth=2, label='Final TBNN', alpha=0.8)
    axes[2,2].set_xlabel('vx velocity')
    axes[2,2].set_ylabel('y position')
    axes[2,2].set_title('Velocity Profiles (Channel Center)')
    axes[2,2].grid(True, alpha=0.3)
    axes[2,2].legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots:
        filename = f"{file_prefix}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"   Saved plot: {filepath}")
    
    plt.show()
    
    # Compute statistics
    initial_error = float(jnp.mean(diff_initial_mag))
    final_error = float(jnp.mean(diff_final_mag))
    error_reduction = initial_error - final_error
    relative_improvement = (error_reduction / initial_error) * 100 if initial_error > 0 else 0
    
    print(f"   Mean error - Initial: {initial_error:.6f}, Final: {final_error:.6f}")
    print(f"   Error reduction: {error_reduction:.6f} ({relative_improvement:.2f}%)")
    
    return {
        'initial_error': initial_error,
        'final_error': final_error,
        'error_reduction': error_reduction,
        'relative_improvement': relative_improvement
    }


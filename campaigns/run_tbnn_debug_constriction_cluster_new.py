#!/usr/bin/env python
"""
Cluster runner: train the instantaneous closure on the constricted channel.

Fits the mixture-of-sigmoids viscosity closure to a reference fluid's steady
velocity field by differentiating through the forward solver, and writes
every figure and result to the run directory for later analysis. This is the
full-resolution runner; the PIV-observation variant is
``run_tbnn_debug_constriction_cluster_new_piv.py``.

USAGE:
    python run_tbnn_debug_constriction_cluster_new.py [iteration_number] [options]
    
EXAMPLES:
    # Basic usage
    python run_tbnn_debug_constriction_cluster_new.py 1
    
    # Custom training configuration
    python run_tbnn_debug_constriction_cluster_new.py 2 --num-steps 20 --learning-rate 5e-4
    
    # With specific initialization
    python run_tbnn_debug_constriction_cluster_new.py 3 --init-method powerlawish_current
    
    # Larger architecture and longer simulation
    python run_tbnn_debug_constriction_cluster_new.py 4 --architecture 64 64 64 --outer-steps 200
    
    # With viscosity regularization
    python run_tbnn_debug_constriction_cluster_new.py 5 --visc-loss --reg-lambda-slope 1e-3 --reg-lambda-curv 3e-4
    
    # With warmup/tail for long simulations
    python run_tbnn_debug_constriction_cluster_new.py 6 --outer-steps 400 --use-warmup-tail --warmup-steps 300 --tail-steps 100
    
    # Save trajectory data and model states
    python run_tbnn_debug_constriction_cluster_new.py 7 --save-traj-info
    
    # With TBNN model options (sigmoid viscosity model)
    python run_tbnn_debug_constriction_cluster_new.py 8 --s-floor 0.7 --alpha-temp 2.0 --global-scalar-lr-scale 10.0
    
    # Freeze eta0 to learn only curvature (forces PDE to learn shape, not magnitude)
    python run_tbnn_debug_constriction_cluster_new.py 9 --freeze-eta0 --eta0-fixed 1.0 --num-steps 10
    
    # Use curvature control to prevent drops at low shear rates (clean fix for f32 issues)
    python run_tbnn_debug_constriction_cluster_new.py 10 --freeze-eta0 --eta0-fixed 1.0 \
        --mu-min-gamma 5e-2 --gate-gamma 5e-2 --gate-width-z 0.5 --num-steps 10
    
    # Enable per-mode power-law bumps for extra expressiveness
    python run_tbnn_debug_constriction_cluster_new.py 11 --freeze-eta0 --eta0-fixed 1.0 \
        --mu-min-gamma 1e-1 --gate-gamma 1e-1 --enable-pl-per-mode --pl-width-z 0.5 --pl-lr-scale 1.0 --num-steps 10
    
    # Full example with all options (including eta0 freeze + curvature control + PL bumps)
    python run_tbnn_debug_constriction_cluster_new.py 13 --num-steps 5 --learning-rate 4e-2 \
        --architecture 30 30 --inner-steps 200 --outer-steps 200 --use-warmup-tail --warmup-steps 150 --tail-steps 50 \
        --visc-loss --reg-s-cap 0.55 --reg-lambda-slope 1e-3 --reg-lambda-curv 3e-4 --reg-gamma-min 5e-2 --reg-gamma-max 5e0 \
        --s-floor 0.7 --alpha-temp 2.0 --global-scalar-lr-scale 10.0 \
        --freeze-eta0 --eta0-fixed 1.0 --eta0-eps 1e-6 \
        --mu-min-gamma 5e-2 --mu-max-gamma 5e0 --gate-gamma 5e-2 --gate-width-z 0.5 \
        --enable-pl-per-mode --pl-width-z 0.5 --pl-lr-scale 1.0 \
        --save-traj-info
    
    # Use two-stage CY pretraining to set up better initial parameters (works with freeze-eta0)
    python run_tbnn_debug_constriction_cluster_new.py 14 --freeze-eta0 --eta0-fixed 1.0 \
        --pretrain-cy --pretrain-cy-steps-1 50 --pretrain-cy-steps-2 10 --pretrain-cy-n2-target 0.95 \
        --num-steps 10 --learning-rate 1e-3
    
    # Use 2-stage training: learn etainf ONLY first (up to 10 steps), then freeze and refine curvature
    python run_tbnn_debug_constriction_cluster_new.py 15 --freeze-eta0 --eta0-fixed 1.0 \
        --two-stage-etainf-then-curv --stage1-steps-etainf 10 --stage2-steps-curv 10 \
        --stage1-etainf-only --stage1-reset-momentum --stage1-early-stop-on-flip \
        --freeze-centers --mu-min-gamma 1e-1 --mu-max-gamma 1e1 \
        --learning-rate 1e-1 --global-scalar-lr-scale 10.0
"""

import os
import sys
import time
import argparse
from datetime import datetime

# Set working directory and add paths
from repo_paths import bootstrap, REPO_ROOT
bootstrap()

# Import the constriction debugging functions
from jax_rheology.training.instantaneous import (
    debug_one_step_gradient_constriction,
    resolve_mask_layout,
)

def main():
    """Main function for cluster execution with constriction geometry."""
    parser = argparse.ArgumentParser(description='Run TBNN gradient debugging with constriction geometry on cluster')
    parser.add_argument('--verbose', action='store_true',
                        help='Print import and environment banners at startup.')
    parser.add_argument('iteration', nargs='?', type=int, default=1,
                       help='Iteration number (1-10)')
    parser.add_argument('--num-steps', type=int, default=10,
                       help='Number of gradient update steps (default: 10)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--architecture', nargs='*', type=int, default=[48, 48],
                       help='TBNN hidden units (default: 48 48). Use empty to run shallow network with no hidden layers.')
    parser.add_argument('--inner-steps', type=int, default=400,
                       help='Number of inner time steps per outer step (default: 400)')
    parser.add_argument('--outer-steps', type=int, default=100,
                       help='Number of outer time steps (default: 100)')
    parser.add_argument('--pressure-gradient', type=float, default=2.5,
                       help='Applied pressure gradient (default: 2.5)')
    parser.add_argument('--init-method', type=str, default=None, 
                       choices=['powerlawish_current', 'powerlawish', 'powerlawish_variant'],
                       help='Initialization method (default: None for standard init)')
    parser.add_argument('--use-soft-newtonian-init', action='store_true', default=False,
                       help='Use soft Newtonian initialization (default: False)')
    parser.add_argument('--save-traj-info', action='store_true', default=False,
                       help='Save trajectory data and model states as .npy arrays (default: False)')
    
    # Warmup/tail options
    parser.add_argument('--use-warmup-tail', action='store_true', default=False,
                       help='Use warmup+tail approach to control gradient explosion (default: False)')
    parser.add_argument('--warmup-steps', type=int, default=None,
                       help='Number of warmup steps if use-warmup-tail is True (default: auto-calculated)')
    parser.add_argument('--tail-steps', type=int, default=None,
                       help='Number of tail steps if use-warmup-tail is True (default: auto-calculated)')
    
    # Loss computation options
    parser.add_argument('--loss-mode', type=str, default='original',
                       choices=['original', 'library'],
                       help='Loss mode: original (spatial averaging) or library (full field, default: original)')
    
    # Viscosity regularization options
    parser.add_argument('--visc-loss', action='store_true', default=False,
                       help='Enable log-log viscosity derivative regularization (default: False)')
    parser.add_argument('--reg-gamma-min', type=float, default=1e-2,
                       help='Min shear rate for viscosity regularization (default: 1e-2)')
    parser.add_argument('--reg-gamma-max', type=float, default=1e2,
                       help='Max shear rate for viscosity regularization (default: 1e2)')
    parser.add_argument('--reg-num-points', type=int, default=128,
                       help='Number of grid points for viscosity regularization (default: 128)')
    parser.add_argument('--reg-s-cap', type=float, default=0.30,
                       help='Slope cap |d log eta / d log gammadot| for regularization (default: 0.30)')
    parser.add_argument('--reg-lambda-slope', type=float, default=1e-3,
                       help='Weight for slope overflow penalty (default: 1e-3)')
    parser.add_argument('--reg-lambda-curv', type=float, default=3e-4,
                       help='Weight for curvature overflow penalty (default: 3e-4)')
    parser.add_argument('--reg-curv-cap-abs', type=float, default=1.0,
                       help='Curvature threshold to start penalizing (default: 1.0)')
    parser.add_argument('--reg-p-slope', type=float, default=2.0,
                       help='Exponent for slope penalty (default: 2.0)')
    parser.add_argument('--reg-p-curv', type=float, default=2.0,
                       help='Exponent for curvature penalty (default: 2.0)')
    
    # TBNN model options
    parser.add_argument('--s-floor', type=float, default=0.0,
                       help='Minimum logistic width for sigmoid viscosity model (default: 0.0)')
    parser.add_argument('--alpha-temp', type=float, default=1.0,
                       help='Softmax temperature for sigmoid viscosity model (default: 1.0)')
    parser.add_argument('--global-scalar-lr-scale', type=float, default=10.0,
                       help='Learning rate scale for global scalar parameters (default: 10.0)')
    
    # eta0-freezing options
    parser.add_argument('--freeze-eta0', action='store_true', default=False,
                       help='Freeze eta0 = eta_inf + delta to learn only curvature (default: False)')
    parser.add_argument('--eta0-fixed', type=float, default=1.0,
                       help='Fixed value of eta0 when freeze-eta0 is True (default: 1.0)')
    parser.add_argument('--eta0-eps', type=float, default=1e-6,
                       help='Small positive floor for delta when eta0 is frozen (default: 1e-6)')
    
    # Curvature control options (prevent curvature at low shear rates)
    parser.add_argument('--mu-min-gamma', type=float, default=None,
                       help='Lower bound on center locations - no curvature before this gammadot (default: None)')
    parser.add_argument('--mu-max-gamma', type=float, default=None,
                       help='Optional upper bound on center locations (default: None)')
    parser.add_argument('--gate-gamma', type=float, default=None,
                       help='If set, multiply mixture by smooth gate starting at this gammadot (default: None)')
    parser.add_argument('--gate-width-z', type=float, default=0.5,
                       help='Gate smoothness in log(gammadot) space (default: 0.5)')
    parser.add_argument('--tail-gate-gamma', type=float, default=None,
                       help='Delay tail (etainf) to this gammadot (prevents mid-shear drag) (default: None)')
    parser.add_argument('--tail-gate-width-z', type=float, default=0.5,
                       help='Tail gate smoothness in log(gammadot) space (default: 0.5)')
    
    # Per-mode power-law bump options
    parser.add_argument('--enable-pl-per-mode', action='store_true', default=False,
                       help='Enable per-mode power-law bumps for extra expressiveness (default: False)')
    parser.add_argument('--pl-width-z', type=float, default=0.5,
                       help='Smooth onset width for each mode\'s PL bump (default: 0.5)')
    parser.add_argument('--pl-lr-scale', type=float, default=1.0,
                       help='Learning rate multiplier for pl_slope_raw parameters (default: 1.0)')
    
    # Checkpoint options
    parser.add_argument('--checkpoint-every', type=int, default=10,
                       help='Save TBNN params every N steps (0 = no checkpoints) (default: 10)')
    
    # Gradient equalizer options
    parser.add_argument('--enable-grad-equalizer', action='store_true', default=False,
                       help='Prevent tail/PL grads from dwarfing mixture params (default: False)')
    parser.add_argument('--equalize-target', type=str, default='mix',
                       choices=['mix', 'tail', 'pl'],
                       help='Which group to protect (default: mix)')
    parser.add_argument('--equalize-cap-ratio', type=float, default=0.5,
                       help='Shrink other groups when they exceed ratio * ||target|| (default: 0.5)')
    
    # Log-head options
    parser.add_argument('--log-head', action='store_true', default=False,
                       help='Learn in log-eta space (geometric blend, stable across decades) (default: False)')
    parser.add_argument('--log-mixing', type=str, default='add',
                       choices=['add', 'geom'],
                       help='Mixing mode for log_head - "add" or "geom" (default: add)')
    
    # CY pretraining options
    parser.add_argument('--pretrain-cy', action='store_true', default=False,
                       help='Run two-stage CY pretraining before main training (default: False)')
    parser.add_argument('--pretrain-cy-steps-1', type=int, default=50,
                       help='Number of steps for stage 1 (fit exact CY target) (default: 50)')
    parser.add_argument('--pretrain-cy-steps-2', type=int, default=10,
                       help='Number of steps for stage 2 (fit near-Newtonian CY) (default: 10)')
    parser.add_argument('--pretrain-cy-n2-target', type=float, default=0.95,
                       help='Target power-law index for stage 2 (0.95 ~= Newtonian) (default: 0.95)')
    
    # Velocity shape loss options
    parser.add_argument('--shape-loss', action='store_true', default=False,
                       help='Enable scale-invariant velocity shape loss (default: False)')
    parser.add_argument('--shape-weight', type=float, default=0.0,
                       help='Weight for velocity shape loss term (default: 0.0)')
    
    # Model architecture options
    parser.add_argument('--M', type=int, default=4,
                       help='Number of sigmoid terms in mixture-of-sigmoids viscosity model (default: 4)')
    parser.add_argument('--freeze-centers', action='store_true', default=False,
                       help='Freeze mu centers (learn only widths s and weights alpha) (default: False)')
    parser.add_argument('--mask-layout', type=str, default='full_domain',
                       metavar='{full_domain,constriction_focused}',
                       help='Loss-mask orientation: full_domain = PIV-twin transpose '
                            'to (T,ny,nx) before the mask block (default); '
                            'constriction_focused = untransposed (T,nx,ny) paper weighting')
    
    # 2-stage training options (etainf learning then freeze)
    parser.add_argument('--two-stage-etainf-then-curv', action='store_true', default=False,
                       help='Enable 2-stage training: etainf learning then freeze (default: False)')
    parser.add_argument('--stage1-steps-etainf', type=int, default=0,
                       help='Max steps for stage 1 with etainf learnable (default: 0)')
    parser.add_argument('--stage2-steps-curv', type=int, default=0,
                       help='Steps for stage 2 with etainf frozen (curvature-only) (default: 0)')
    parser.add_argument('--stage1-etainf-only', action='store_true', default=True,
                       help='Stage 1 = etainf ONLY (freeze curvature); if not set, both learn (default: True)')
    parser.add_argument('--stage1-reset-momentum', action='store_true', default=True,
                       help='Reset Adam momentum when switching to stage 2 (default: True)')
    parser.add_argument('--stage1-early-stop-on-flip', action='store_true', default=False,
                       help='End stage 1 early if detainf flips sign (momentum overshoot) (default: False)')
    
    # Reference model (Carreau-Yasuda) parameters
    parser.add_argument('--ref-eta-inf', type=float, default=0.02,
                       help='Reference Carreau-Yasuda etainf (default: 0.02)')
    parser.add_argument('--ref-eta-0', type=float, default=1.0,
                       help='Reference Carreau-Yasuda eta0 (default: 1.0)')
    parser.add_argument('--ref-lambda', type=float, default=5.0,
                       help='Reference Carreau-Yasuda lam (default: 5.0)')
    parser.add_argument('--ref-n', type=float, default=0.5,
                       help='Reference Carreau-Yasuda n (default: 0.5)')
    parser.add_argument('--ref-a', type=float, default=2.0,
                       help='Reference Carreau-Yasuda a (default: 2.0)')
    
    # Output directory naming
    parser.add_argument('--output-prefix', type=str, default='iteration',
                       help='Prefix for output directory naming (default: iteration)')
    parser.add_argument('--results-root', type=str, default='./work/instantaneous_train',
                       help='Root directory for run artefacts (default: ./work/instantaneous_train)')
    
    from jax_rheology.io.config import parse_with_config
    args, _cfg = parse_with_config(parser)
    args.mask_layout = resolve_mask_layout(args.mask_layout, log=True)
    
    # Create unique output directory for this iteration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.results_root}/{args.output_prefix}_{args.iteration:02d}_{timestamp}"
    
    print(f"TBNN GRADIENT DEBUGGING - CLUSTER EXECUTION (CONSTRICTION GEOMETRY)")
    print(f"Iteration: {args.iteration}")
    print(f"Reference model: Carreau-Yasuda [etainf={args.ref_eta_inf}, eta0={args.ref_eta_0}, lam={args.ref_lambda}, n={args.ref_n}, a={args.ref_a}]")
    print(f"Architecture: {args.architecture}")
    print(f"Mixture modes: M={args.M} sigmoid terms")
    print(f"Training steps: {args.num_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Global scalar LR scale: {args.global_scalar_lr_scale}")
    print(f"Simulation: inner_steps={args.inner_steps}, outer_steps={args.outer_steps}")
    print(f"Pressure gradient: {args.pressure_gradient}")
    print(f"Init method: {args.init_method}")
    print(f"Loss mode: {args.loss_mode}")
    print(f"TBNN model options:")
    print(f"  - s_floor: {args.s_floor}, alpha_temp: {args.alpha_temp}")
    if args.freeze_eta0:
        print(f"  - eta0 FROZEN at {args.eta0_fixed} (learning curvature only)")
        print(f"  - eta0_eps: {args.eta0_eps}")
    if args.mu_min_gamma is not None or args.gate_gamma is not None or args.tail_gate_gamma is not None:
        print(f"  - CURVATURE CONTROL:")
        if args.mu_min_gamma is not None:
            print(f"    mu_min_gamma: {args.mu_min_gamma} (no curvature before this gammadot)")
        if args.mu_max_gamma is not None:
            print(f"    mu_max_gamma: {args.mu_max_gamma} (no curvature after this gammadot)")
        if args.gate_gamma is not None:
            print(f"    gate_gamma: {args.gate_gamma}, gate_width_z: {args.gate_width_z}")
        if args.tail_gate_gamma is not None:
            print(f"    tail_gate_gamma: {args.tail_gate_gamma}, tail_gate_width_z: {args.tail_gate_width_z} (defer etainf)")
    if args.enable_pl_per_mode:
        print(f"  - PER-MODE POWER-LAW BUMPS ENABLED")
        print(f"    pl_width_z: {args.pl_width_z}, pl_lr_scale: {args.pl_lr_scale}")
    if args.enable_grad_equalizer:
        print(f"  -  GRADIENT EQUALIZER ENABLED")
        print(f"    target: {args.equalize_target}, cap_ratio: {args.equalize_cap_ratio}")
    if args.log_head:
        print(f"  - LOG-HEAD ENABLED")
        print(f"    log_mixing: {args.log_mixing}")
    if args.pretrain_cy:
        print(f"  -  CY PRETRAIN ENABLED")
        print(f"    Stage 1: {args.pretrain_cy_steps_1} steps (fit exact CY target)")
        print(f"    Stage 2: {args.pretrain_cy_steps_2} steps (fit near-Newtonian, n={args.pretrain_cy_n2_target})")
        print(f"    Freezes etainf/delta to learn only curvature parameters mu/s/alpha")
    if args.freeze_centers:
        print(f"  - CENTERS FROZEN (learn only widths s and weights alpha)")
    print(f"Mask layout: {args.mask_layout}")
    print(f"Velocity shape loss: {args.shape_loss}")
    if args.shape_loss:
        print(f"  - Shape weight: {args.shape_weight} (scale-invariant RMSE)")
    print(f"Viscosity regularization: {args.visc_loss}")
    if args.visc_loss:
        print(f"  - Gamma range: [{args.reg_gamma_min}, {args.reg_gamma_max}]")
        print(f"  - Slope cap: {args.reg_s_cap}, lam_slope: {args.reg_lambda_slope}, lam_curv: {args.reg_lambda_curv}")
    print(f"Warmup/tail: {args.use_warmup_tail}")
    if args.use_warmup_tail:
        print(f"  - Warmup steps: {args.warmup_steps}, Tail steps: {args.tail_steps}")
    print(f"Save trajectory info: {args.save_traj_info}")
    if args.save_traj_info and args.checkpoint_every > 0:
        print(f"  Checkpointing: saving params every {args.checkpoint_every} steps")
    print(f"Output directory: {output_dir}")
    print(f"Timestamp: {timestamp}")
    print("="*80)
    
    # Test 1: Multi-step gradient update with Carreau-Yasuda reference (Constriction)
    print("\n" + "="*60)
    print("RUNNING TEST 1: MULTI-STEP GRADIENT UPDATE (CONSTRICTION)")
    print("="*60)
    
    start_time = time.time()
    
    try:
        results1 = debug_one_step_gradient_constriction(
            reference_model='carreau_yasuda',
            reference_params=[args.ref_eta_inf, args.ref_eta_0, args.ref_lambda, args.ref_n, args.ref_a],  # [etainf, eta0, lam, n, a]
            tbnn_hidden_units=args.architecture,
            eta_init=1.0,
            pressure_gradient=args.pressure_gradient,
            dt=1e-5,
            inner_steps=args.inner_steps,
            outer_steps=args.outer_steps,
            learning_rate=args.learning_rate,
            num_update_steps=args.num_steps,
            run_new_forward=True,
            solver_type='bicgstab',
            stepper_type='fully_implicit',
            random_seed=42 + args.iteration,  # Unique seed per iteration
            save_plots=True,  # KEY: Save all plots
            output_dir=output_dir,
            use_soft_newtonian_init=args.use_soft_newtonian_init,
            save_traj_info=args.save_traj_info,
            # Warmup/tail options
            use_warmup_tail=args.use_warmup_tail,
            warmup_steps=args.warmup_steps,
            tail_steps=args.tail_steps,
            # Viscosity regularization options
            visc_loss=args.visc_loss,
            reg_gamma_min=args.reg_gamma_min,
            reg_gamma_max=args.reg_gamma_max,
            reg_num_points=args.reg_num_points,
            reg_s_cap=args.reg_s_cap,
            reg_lambda_slope=args.reg_lambda_slope,
            reg_lambda_curv=args.reg_lambda_curv,
            reg_curv_cap_abs=args.reg_curv_cap_abs,
            reg_p_slope=args.reg_p_slope,
            reg_p_curv=args.reg_p_curv,
            # TBNN model options
            s_floor=args.s_floor,
            alpha_temp=args.alpha_temp,
            global_scalar_lr_scale=args.global_scalar_lr_scale,
            # eta0-freezing options
            freeze_eta0=args.freeze_eta0,
            eta0_fixed=args.eta0_fixed,
            eta0_eps=args.eta0_eps,
            # Curvature control options
            mu_min_gamma=args.mu_min_gamma,
            mu_max_gamma=args.mu_max_gamma,
            gate_gamma=args.gate_gamma,
            gate_width_z=args.gate_width_z,
            tail_gate_gamma=args.tail_gate_gamma,
            tail_gate_width_z=args.tail_gate_width_z,
            # Per-mode PL bump options
            enable_pl_per_mode=args.enable_pl_per_mode,
            pl_width_z=args.pl_width_z,
            pl_lr_scale=args.pl_lr_scale,
            # Checkpoint options
            checkpoint_every=args.checkpoint_every,
            # Gradient equalizer options
            enable_grad_equalizer=args.enable_grad_equalizer,
            equalize_target=args.equalize_target,
            equalize_cap_ratio=args.equalize_cap_ratio,
            # Log-head options
            log_head=args.log_head,
            log_mixing=args.log_mixing,
            # CY pretraining options
            pretrain_cy=args.pretrain_cy,
            pretrain_cy_steps_1=args.pretrain_cy_steps_1,
            pretrain_cy_steps_2=args.pretrain_cy_steps_2,
            pretrain_cy_n2_target=args.pretrain_cy_n2_target,
            # Velocity shape loss options
            shape_loss=args.shape_loss,
            shape_weight=args.shape_weight,
            # Model architecture options
            M=args.M,
            freeze_centers=args.freeze_centers,
            mask_layout=args.mask_layout,
            # 2-stage training options
            two_stage_etainf_then_curv=args.two_stage_etainf_then_curv,
            stage1_steps_etainf=args.stage1_steps_etainf,
            stage2_steps_curv=args.stage2_steps_curv,
            stage1_etainf_only=args.stage1_etainf_only,
            stage1_reset_momentum=args.stage1_reset_momentum,
            stage1_early_stop_on_flip=args.stage1_early_stop_on_flip
        )
        
        test1_time = time.time() - start_time
        
        if results1:
            print(f"\nTEST 1 COMPLETED in {test1_time:.1f}s")
            training_stats = results1.get('training_stats', {})
            print(f"   Final loss: {training_stats.get('final_loss', 'N/A'):.6e}")
            print(f"   Total improvement: {training_stats.get('relative_improvement', 0):.2f}%")
            if results1.get('rollback_activated', False):
                print(f"   Rollback activated at step {results1.get('last_good_step', 'N/A')}")
        else:
            print(f"\nTEST 1 FAILED after {test1_time:.1f}s")
            
    except Exception as e:
        print(f"\nTEST 1 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results1 = None
        test1_time = time.time() - start_time
    
    # Test 2 removed - finite difference checks are no longer used
    results2 = None
    
    # Save summary results to file
    total_time = test1_time
    summary_file = os.path.join(output_dir, "iteration_summary_constriction.txt")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(summary_file, 'w') as f:
            f.write(f"TBNN GRADIENT DEBUGGING RESULTS (CONSTRICTION GEOMETRY)\n")
            f.write(f"======================================================\n\n")
            f.write(f"Iteration: {args.iteration}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Architecture: {args.architecture}\n")
            f.write(f"Training steps: {args.num_steps}\n")
            f.write(f"Learning rate: {args.learning_rate}\n")
            f.write(f"Inner steps: {args.inner_steps}\n")
            f.write(f"Outer steps: {args.outer_steps}\n")
            f.write(f"Pressure gradient: {args.pressure_gradient}\n")
            f.write(f"Random seed: {42 + args.iteration}\n")
            f.write(f"Init method: {args.init_method}\n")
            f.write(f"Reference model:\n")
            f.write(f"  type: carreau_yasuda\n")
            f.write(f"  η∞: {args.ref_eta_inf}\n")
            f.write(f"  η₀: {args.ref_eta_0}\n")
            f.write(f"  lam: {args.ref_lambda}\n")
            f.write(f"  n: {args.ref_n}\n")
            f.write(f"  a: {args.ref_a}\n")
            f.write(f"  params: [{args.ref_eta_inf}, {args.ref_eta_0}, {args.ref_lambda}, {args.ref_n}, {args.ref_a}]  # [η∞, η₀, λ, n, a]\n")
            f.write(f"Solver configuration:\n")
            f.write(f"  dt: 1e-5\n")
            f.write(f"  solver_type: bicgstab\n")
            f.write(f"  stepper_type: fully_implicit\n")
            f.write(f"  eta_init: 1.0\n")
            f.write(f"  use_soft_newtonian_init: {args.use_soft_newtonian_init}\n")
            f.write(f"  domain: ((0, 8.0), (0, 4.0))\n")
            f.write(f"  domain_size: (256, 128)\n")
            f.write(f"  density: 1.0\n")
            f.write(f"  run_new_forward: True\n")
            f.write(f"TBNN model options:\n")
            f.write(f"  s_floor: {args.s_floor}\n")
            f.write(f"  alpha_temp: {args.alpha_temp}\n")
            f.write(f"  global_scalar_lr_scale: {args.global_scalar_lr_scale}\n")
            f.write(f"η₀-freezing:\n")
            f.write(f"  freeze_eta0: {args.freeze_eta0}\n")
            if args.freeze_eta0:
                f.write(f"  eta0_fixed: {args.eta0_fixed}\n")
                f.write(f"  eta0_eps: {args.eta0_eps}\n")
            f.write(f"Curvature control:\n")
            f.write(f"  mu_min_gamma: {args.mu_min_gamma}\n")
            f.write(f"  mu_max_gamma: {args.mu_max_gamma}\n")
            f.write(f"  gate_gamma: {args.gate_gamma}\n")
            f.write(f"  gate_width_z: {args.gate_width_z}\n")
            f.write(f"  tail_gate_gamma: {args.tail_gate_gamma}\n")
            f.write(f"  tail_gate_width_z: {args.tail_gate_width_z}\n")
            f.write(f"Per-mode PL bumps:\n")
            f.write(f"  enable_pl_per_mode: {args.enable_pl_per_mode}\n")
            if args.enable_pl_per_mode:
                f.write(f"  pl_width_z: {args.pl_width_z}\n")
                f.write(f"  pl_lr_scale: {args.pl_lr_scale}\n")
            f.write(f"Warmup/tail: {args.use_warmup_tail}\n")
            if args.use_warmup_tail:
                f.write(f"  Warmup steps: {args.warmup_steps}\n")
                f.write(f"  Tail steps: {args.tail_steps}\n")
            f.write(f"Viscosity regularization: {args.visc_loss}\n")
            if args.visc_loss:
                f.write(f"  Gamma range: [{args.reg_gamma_min}, {args.reg_gamma_max}]\n")
                f.write(f"  Num points: {args.reg_num_points}\n")
                f.write(f"  Slope cap: {args.reg_s_cap}\n")
                f.write(f"  Lambda slope: {args.reg_lambda_slope}\n")
                f.write(f"  Lambda curv: {args.reg_lambda_curv}\n")
                f.write(f"  Curv cap abs: {args.reg_curv_cap_abs}\n")
                f.write(f"  P slope: {args.reg_p_slope}\n")
                f.write(f"  P curv: {args.reg_p_curv}\n")
            f.write(f"Geometry: Channel with constriction\n")
            f.write(f"Save trajectory info: {args.save_traj_info}\n")
            f.write(f"Checkpoint options:\n")
            f.write(f"  checkpoint_every: {args.checkpoint_every}\n")
            if args.save_traj_info and args.checkpoint_every > 0:
                f.write(f"  Status: ENABLED - saving every {args.checkpoint_every} steps\n")
            f.write(f"Gradient equalizer:\n")
            f.write(f"  enable_grad_equalizer: {args.enable_grad_equalizer}\n")
            if args.enable_grad_equalizer:
                f.write(f"  target: {args.equalize_target}\n")
                f.write(f"  cap_ratio: {args.equalize_cap_ratio}\n")
            f.write(f"Log-head learning:\n")
            f.write(f"  log_head: {args.log_head}\n")
            if args.log_head:
                f.write(f"  log_mixing: {args.log_mixing}\n")
            f.write(f"CY pretraining:\n")
            f.write(f"  pretrain_cy: {args.pretrain_cy}\n")
            if args.pretrain_cy:
                f.write(f"  pretrain_cy_steps_1: {args.pretrain_cy_steps_1}\n")
                f.write(f"  pretrain_cy_steps_2: {args.pretrain_cy_steps_2}\n")
                f.write(f"  pretrain_cy_n2_target: {args.pretrain_cy_n2_target}\n")
            f.write(f"Velocity shape loss:\n")
            f.write(f"  shape_loss: {args.shape_loss}\n")
            if args.shape_loss:
                f.write(f"  shape_weight: {args.shape_weight}\n")
            f.write(f"Model architecture:\n")
            f.write(f"  M: {args.M}\n")
            f.write(f"  freeze_centers: {args.freeze_centers}\n")
            f.write(f"Mask layout:\n")
            f.write(f"  mask_layout: {args.mask_layout}\n")
            f.write(f"2-stage training:\n")
            f.write(f"  two_stage_etainf_then_curv: {args.two_stage_etainf_then_curv}\n")
            if args.two_stage_etainf_then_curv:
                f.write(f"  stage1_steps_etainf: {args.stage1_steps_etainf}\n")
                f.write(f"  stage2_steps_curv: {args.stage2_steps_curv}\n")
                f.write(f"  stage1_etainf_only: {args.stage1_etainf_only}\n")
                f.write(f"  stage1_reset_momentum: {args.stage1_reset_momentum}\n")
                f.write(f"  stage1_early_stop_on_flip: {args.stage1_early_stop_on_flip}\n")
            f.write("\n")
            
            f.write(f"TEST 1 RESULTS (CONSTRICTION):\n")
            f.write(f"-------------------------------\n")
            if results1:
                training_stats = results1.get('training_stats', {})
                f.write(f"Status: SUCCESS\n")
                f.write(f"Runtime: {test1_time:.1f} seconds\n")
                f.write(f"Initial loss: {results1.get('initial_loss', 'N/A'):.6e}\n")
                f.write(f"Final loss: {training_stats.get('final_loss', 'N/A'):.6e}\n")
                f.write(f"Loss improvement: {training_stats.get('relative_improvement', 0):.2f}%\n")
                f.write(f"Final gradient magnitude: {training_stats.get('final_gradient_magnitude', 'N/A'):.6e}\n")
                f.write(f"Rollback activated: {results1.get('rollback_activated', False)}\n")
                if results1.get('rollback_activated', False):
                    f.write(f"Last good step: {results1.get('last_good_step', 'N/A')}\n")
                f.write(f"Initialization method: {results1.get('initial_model_info', {}).get('init_method', 'N/A')}\n")
            else:
                f.write(f"Status: FAILED\n")
                f.write(f"Runtime: {test1_time:.1f} seconds\n")
            
            f.write(f"\nTEST 2 (Finite Difference Check):\n")
            f.write(f"-------------------------------\n")
            f.write(f"Status: REMOVED (no longer used)\n")
            
            f.write(f"\nOVERALL:\n")
            f.write(f"--------\n")
            f.write(f"Total runtime: {total_time:.1f} seconds\n")
            f.write(f"Test 1 success: {results1 is not None}\n")
            f.write(f"Test 2 success: {results2 is not None}\n")
            f.write(f"Both tests success: {results1 is not None and results2 is not None}\n")
        
        print(f"\nSummary saved to: {summary_file}")
        
    except Exception as e:
        print(f"\nFailed to save summary: {e}")
    
    # Final summary
    print(f"\n" + "="*80)
    print(f"CLUSTER EXECUTION COMPLETED - ITERATION {args.iteration} (CONSTRICTION)")
    print(f"="*80)
    print(f"Gradient Update Test: {'SUCCESS' if results1 else 'FAILED'} ({test1_time:.1f}s)")
    if results1 and results1.get('rollback_activated', False):
        print(f"   Note: Rollback activated at step {results1.get('last_good_step', 'N/A')}")
    print(f"Total runtime: {total_time:.1f} seconds")
    print(f"Output directory: {output_dir}")
    
    # Provide overall success/failure exit code
    if results1:
        print("constriction gradient test: PASS")
        sys.exit(0)
    else:
        print(f"TEST 1 FAILED (CONSTRICTION GEOMETRY)!")
        sys.exit(1)

if __name__ == "__main__":
    main()

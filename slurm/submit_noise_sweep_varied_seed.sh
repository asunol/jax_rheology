#!/bin/bash
#
# Submit 20 TBNN training jobs with PIV noise sweep (DIFFERENT SEEDS)
# Sweeps over: 4 noise levels, 5 replicates each
# freeze_centers = False only
#
# Best hyperparameters from initial runs:
#   - learning_rate: 5e-2
#   - architecture: [16]
#   - stage2_steps: 50
#

echo "=========================================="
echo "TBNN PIV Noise Sweep (Varied Seeds) - 20 Jobs"
echo "Start time: $(date)"
echo "=========================================="

# Create directories
mkdir -p ./jobs
mkdir -p ./job_scripts

# Fixed parameters (best from initial sweep)
LEARNING_RATE=5e-2
ARCHITECTURE="16"
STAGE2_STEPS=50
M=12
OUTER_STEPS=2000
WARMUP_STEPS=1990
TAIL_STEPS=10
STAGE1_STEPS=20
GLOBAL_SCALAR_LR=10.0
PRESSURE_GRAD=5.0

# PIV parameters (fixed window)
PIV_OVERLAP=0.75
PIV_KERNEL="hann"
PIV_NOISE_CORR_FRAC=0.35
PIV_NOISE_BETA_GRAD=0.5
WINDOW_X=32
WINDOW_Y=16

# Sweep parameters
FREEZE_CENTERS=False
REPLICATES=(1 2 3 4 5)
NOISE_LEVELS=(0.5 1.0 2.0 4.0)

# Counter for job number
JOB_NUM=1
TOTAL_JOBS=20

echo "Sweep configuration:"
echo "  Noise levels: ${NOISE_LEVELS[@]} % of U95"
echo "  Replicates: ${#REPLICATES[@]} (each with different seed)"
echo "  freeze_centers: False"
echo "  Fixed window: ${WINDOW_X}×${WINDOW_Y}"
echo "  Total: 4 noise levels × 5 replicates = $TOTAL_JOBS jobs"
echo "=========================================="
echo ""

# Function to create and submit a job
submit_job() {
    local job_num=$1
    local noise_p=$2
    local replicate=$3
    
    # Calculate unique seed for each replicate
    # Base seed 1453, add 1000*replicate to ensure different noise patterns
    PIV_NOISE_SEED=$((1453 + 1000 * replicate))
    
    # Create descriptive job name
    JOB_NAME="tbnn_noise${noise_p}_rep${replicate}_seed${PIV_NOISE_SEED}"
    SCRIPT_FILE="./job_scripts/job_${job_num}_${JOB_NAME}.sh"
    
    echo "Creating job $job_num/$TOTAL_JOBS: $JOB_NAME (seed=${PIV_NOISE_SEED})"
    
    # Generate individual job script
    cat > $SCRIPT_FILE << EOF
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -p gpu,seas_gpu
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -n 1
#SBATCH -t 0-12:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=asunol@seas.harvard.edu
#SBATCH -o ./jobs/%j_${JOB_NAME}.out
#SBATCH -e ./jobs/%j_${JOB_NAME}.err

# Print job information
echo "=========================================="
echo "TBNN PIV Noise Sweep - Job ${job_num}/$TOTAL_JOBS"
echo "Job Name: ${JOB_NAME}"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Start time: \$(date)"
echo "=========================================="
echo ""
echo "Fixed Parameters (best from initial sweep):"
echo "  architecture: [$ARCHITECTURE]"
echo "  learning_rate: $LEARNING_RATE"
echo "  stage2_steps_curv: $STAGE2_STEPS"
echo "  stage1_steps_etainf: $STAGE1_STEPS"
echo ""
echo "Sweep Parameters:"
echo "  window: ${WINDOW_X}×${WINDOW_Y}"
echo "  noise_p: ${noise_p}% of U95"
echo "  freeze_centers: False"
echo "  replicate: $replicate"
echo "  PIV_NOISE_SEED: ${PIV_NOISE_SEED}"
echo "=========================================="

# Activate environment
source activate cfd_md_optimization

# Run the training
python campaigns/run_tbnn_debug_constriction_cluster_new_piv.py ${job_num} \\
    --M $M \\
    --architecture $ARCHITECTURE \\
    --num-steps 1 \\
    --learning-rate $LEARNING_RATE \\
    --global-scalar-lr-scale $GLOBAL_SCALAR_LR \\
    --inner-steps 400 \\
    --outer-steps $OUTER_STEPS \\
    --pressure-gradient $PRESSURE_GRAD \\
    --ref-eta-inf 0.02 \\
    --ref-eta-0 1.0 \\
    --ref-lambda 5.0 \\
    --ref-n 0.7 \\
    --ref-a 2.0 \\
    --use-soft-newtonian-init \\
    --use-warmup-tail \\
    --warmup-steps $WARMUP_STEPS \\
    --tail-steps $TAIL_STEPS \\
    --visc-loss \\
    --reg-gamma-min 1e-1 \\
    --reg-gamma-max 1e1 \\
    --reg-num-points 128 \\
    --reg-s-cap 0.60 \\
    --reg-lambda-slope 1e-3 \\
    --reg-lambda-curv 3e-4 \\
    --reg-curv-cap-abs 0.5 \\
    --shape-loss \\
    --shape-weight 0.5 \\
    --s-floor 0.35 \\
    --alpha-temp 0.8 \\
    --freeze-eta0 \\
    --eta0-fixed 1.0 \\
    --eta0-eps 1e-5 \\
    --mu-min-gamma 1e-1 \\
    --mu-max-gamma 1e1 \\
    --gate-gamma 1e-1 \\
    --gate-width-z 0.5 \\
    --enable-grad-equalizer \\
    --equalize-target mix \\
    --equalize-cap-ratio 1.0 \\
    --log-head \\
    --log-mixing add \\
    --two-stage-etainf-then-curv \\
    --stage1-steps-etainf $STAGE1_STEPS \\
    --stage2-steps-curv $STAGE2_STEPS \\
    --stage1-etainf-only \\
    --stage1-reset-momentum \\
    --stage1-early-stop-on-flip \\
    --resolution-piv \\
    --piv-W-win $WINDOW_X $WINDOW_Y \\
    --piv-overlap $PIV_OVERLAP \\
    --piv-kernel $PIV_KERNEL \\
    --add-piv-noise \\
    --piv-noise-p-percent $noise_p \\
    --piv-noise-corr-frac $PIV_NOISE_CORR_FRAC \\
    --piv-noise-beta-grad $PIV_NOISE_BETA_GRAD \\
    --piv-noise-use-bias \\
    --piv-noise-seed $PIV_NOISE_SEED \\
    --save-traj-info

EXIT_CODE=\$?

echo ""
echo "=========================================="
echo "Job ${JOB_NAME} completed"
echo "Exit code: \$EXIT_CODE"
echo "End time: \$(date)"
echo "=========================================="

exit \$EXIT_CODE
EOF

    # Make script executable
    chmod +x $SCRIPT_FILE
    
    # Submit the job
    SUBMIT_OUTPUT=$(sbatch $SCRIPT_FILE)
    JOBID=$(echo $SUBMIT_OUTPUT | awk '{print $4}')
    echo "  ✓ Submitted as SLURM job $JOBID (seed=${PIV_NOISE_SEED})"
    echo ""
    
    # Small delay to avoid overwhelming scheduler
    sleep 0.5
}

# ============================================================================
# PIV Noise Level Sweep (fixed window 32×16, varied seeds)
# ============================================================================
echo "=========================================="
echo "PIV Noise Level Sweep (window=32×16, varied seeds)"
echo "=========================================="

for noise_p in "${NOISE_LEVELS[@]}"; do
  for replicate in "${REPLICATES[@]}"; do
    submit_job $JOB_NUM "$noise_p" "$replicate"
    JOB_NUM=$((JOB_NUM + 1))
  done
done

echo ""
echo "=========================================="
echo "ALL $TOTAL_JOBS JOBS SUBMITTED"
echo "=========================================="
echo "Job scripts saved in: ./job_scripts/"
echo "Output logs will be in: ./jobs/"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check specific job: squeue -j <JOBID>"
echo "Cancel all jobs: scancel -u \$USER"
echo ""
echo "Results will be in: ./work/instantaneous_train/"
echo "Each iteration gets its own timestamped directory with prefix 'noise_iteration_'"
echo ""
echo "Sweep summary:"
echo "  Noise levels: 0.5, 1.0, 2.0, 4.0 % of U95"
echo "  Replicates per noise level: 5 (each with different seed)"
echo "  Seeds: 2453, 3453, 4453, 5453, 6453"
echo "  freeze_centers: False"
echo "  Fixed window: ${WINDOW_X}×${WINDOW_Y}"
echo "  Total: 4 × 5 = $TOTAL_JOBS jobs"
echo ""
echo "Fixed hyperparameters (best from initial runs):"
echo "  learning_rate: $LEARNING_RATE"
echo "  architecture: [$ARCHITECTURE]"
echo "  stage2_steps: $STAGE2_STEPS"
echo "=========================================="


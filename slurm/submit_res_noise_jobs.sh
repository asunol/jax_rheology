#!/bin/bash
#
# Generate and submit 54 TBNN training jobs with PIV resolution and noise sweeps
# Sweeps over: PIV window size, noise level, freeze_centers, 3 replicates each
#
# Best hyperparameters from initial runs:
#   - learning_rate: 5e-2
#   - architecture: [16]
#   - stage2_steps: 50
#

echo "=========================================="
echo "TBNN PIV Resolution + Noise Sweep - 54 Jobs"
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

# PIV parameters
PIV_OVERLAP=0.75
PIV_KERNEL="hann"
PIV_NOISE_CORR_FRAC=0.35
PIV_NOISE_BETA_GRAD=0.5
PIV_NOISE_SEED=1453

# Sweep parameters
FREEZE_CENTERS_VALS=(False True)
REPLICATES=(1 2 3)

# Define PIV window sweep (no noise)
declare -a WINDOW_SWEEP=("16 8" "24 12" "32 16" "48 24" "64 32")

# Define noise sweep (fixed window 32x16)
declare -a NOISE_SWEEP=(0.5 1.0 2.0 4.0)

# Counter for job number
JOB_NUM=1
TOTAL_JOBS=54

echo "Sweep configuration:"
echo "  Window sweep (no noise): ${#WINDOW_SWEEP[@]} windows"
echo "  Noise sweep (window=32x16): ${#NOISE_SWEEP[@]} noise levels"
echo "  Replicates: ${#REPLICATES[@]}"
echo "  freeze_centers: ${#FREEZE_CENTERS_VALS[@]} values"
echo "  Total: (5 + 4) × 3 × 2 = $TOTAL_JOBS jobs"
echo "=========================================="
echo ""

# Function to create and submit a job
submit_job() {
    local job_num=$1
    local freeze_centers=$2
    local replicate=$3
    local condition=$4  # "window" or "noise"
    local window_x=$5
    local window_y=$6
    local noise_p=$7
    
    # Set freeze-centers flag
    if [ "$freeze_centers" = "True" ]; then
        FREEZE_FLAG="--freeze-centers"
        FREEZE_DESC="frozen"
    else
        FREEZE_FLAG=""
        FREEZE_DESC="free"
    fi
    
    # Set condition-specific parameters
    if [ "$condition" = "window" ]; then
        PIV_FLAGS="--resolution-piv --piv-W-win $window_x $window_y --piv-overlap $PIV_OVERLAP --piv-kernel $PIV_KERNEL"
        CONDITION_DESC="win${window_x}x${window_y}_clean"
    else  # noise
        PIV_FLAGS="--resolution-piv --piv-W-win $window_x $window_y --piv-overlap $PIV_OVERLAP --piv-kernel $PIV_KERNEL --add-piv-noise --piv-noise-p-percent $noise_p --piv-noise-corr-frac $PIV_NOISE_CORR_FRAC --piv-noise-beta-grad $PIV_NOISE_BETA_GRAD --piv-noise-use-bias --piv-noise-seed $PIV_NOISE_SEED"
        CONDITION_DESC="win${window_x}x${window_y}_noise${noise_p}"
    fi
    
    # Create descriptive job name
    JOB_NAME="tbnn_piv_${CONDITION_DESC}_${FREEZE_DESC}_rep${replicate}"
    SCRIPT_FILE="./job_scripts/job_${job_num}_${JOB_NAME}.sh"
    
    echo "Creating job $job_num/$TOTAL_JOBS: $JOB_NAME"
    
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
echo "TBNN PIV Resolution + Noise - Job ${job_num}/$TOTAL_JOBS"
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
echo "  condition: $condition"
echo "  window: ${window_x}×${window_y}"
echo "  noise_p: $noise_p"
echo "  freeze_centers: $freeze_centers"
echo "  replicate: $replicate"
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
    $FREEZE_FLAG \\
    --log-head \\
    --log-mixing add \\
    --two-stage-etainf-then-curv \\
    --stage1-steps-etainf $STAGE1_STEPS \\
    --stage2-steps-curv $STAGE2_STEPS \\
    --stage1-etainf-only \\
    --stage1-reset-momentum \\
    --stage1-early-stop-on-flip \\
    $PIV_FLAGS \\
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
    echo "  ✓ Submitted as SLURM job $JOBID"
    echo ""
    
    # Small delay to avoid overwhelming scheduler
    sleep 0.5
}

# ============================================================================
# SWEEP 1: PIV Window Size (no noise)
# ============================================================================
echo "=========================================="
echo "SWEEP 1: PIV Window Size (no noise)"
echo "=========================================="

for freeze_centers in "${FREEZE_CENTERS_VALS[@]}"; do
  for replicate in "${REPLICATES[@]}"; do
    for window in "${WINDOW_SWEEP[@]}"; do
      # Split window string into W_x and W_y
      read -r window_x window_y <<< "$window"
      
      submit_job $JOB_NUM "$freeze_centers" "$replicate" "window" "$window_x" "$window_y" "0"
      
      JOB_NUM=$((JOB_NUM + 1))
    done
  done
done

# ============================================================================
# SWEEP 2: PIV Noise Level (fixed window 32×16)
# ============================================================================
echo "=========================================="
echo "SWEEP 2: PIV Noise Level (window=32×16)"
echo "=========================================="

for freeze_centers in "${FREEZE_CENTERS_VALS[@]}"; do
  for replicate in "${REPLICATES[@]}"; do
    for noise_p in "${NOISE_SWEEP[@]}"; do
      
      submit_job $JOB_NUM "$freeze_centers" "$replicate" "noise" "32" "16" "$noise_p"
      
      JOB_NUM=$((JOB_NUM + 1))
    done
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
echo "Each iteration gets its own timestamped directory"
echo ""
echo "Sweep summary:"
echo "  SWEEP 1 (Window, no noise): 5 windows × 3 replicates × 2 freeze_centers = 30 jobs"
echo "    Windows: (16,8), (24,12), (32,16), (48,24), (64,32)"
echo "  SWEEP 2 (Noise, window=32×16): 4 noise levels × 3 replicates × 2 freeze_centers = 24 jobs"
echo "    Noise: p = 0.5, 1.0, 2.0, 4.0 % of U95"
echo "  Total: 30 + 24 = $TOTAL_JOBS jobs"
echo ""
echo "Fixed hyperparameters (best from initial runs):"
echo "  learning_rate: $LEARNING_RATE"
echo "  architecture: [$ARCHITECTURE]"
echo "  stage2_steps: $STAGE2_STEPS"
echo "=========================================="

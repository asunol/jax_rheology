#!/bin/bash
#
# Generate and submit 24 TBNN training jobs with parameter sweeps
# Sweeps over: freeze_centers, architecture, learning_rate, stage2_steps
#

echo "=========================================="
echo "TBNN 2-Stage Training - 24 Job Sweep"
echo "Start time: $(date)"
echo "=========================================="

# Create directories
mkdir -p ./jobs
mkdir -p ./job_scripts

# Define parameter arrays
FREEZE_CENTERS_VALS=(False True)
ARCHITECTURES=("" "16")  # Empty string = [], "16" = [16]
LEARNING_RATES=(5e-3 1e-2 5e-2)
STAGE2_STEPS=(5 50)

# Fixed parameters (from your base config)
M=12
OUTER_STEPS=2000
WARMUP_STEPS=1990
TAIL_STEPS=10
STAGE1_STEPS=20
GLOBAL_SCALAR_LR=10.0
PRESSURE_GRAD=5.0

# Counter for job number
JOB_NUM=1

# Loop through all combinations
for freeze_centers in "${FREEZE_CENTERS_VALS[@]}"; do
  for arch in "${ARCHITECTURES[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      for stage2 in "${STAGE2_STEPS[@]}"; do
        
        echo "Creating job $JOB_NUM/24:"
        echo "  freeze_centers=$freeze_centers, arch=$arch, lr=$lr, stage2=$stage2"
        
        # Set architecture flag
        if [ -z "$arch" ]; then
          ARCH_FLAG="--architecture"
          ARCH_DESC="shallow"
        else
          ARCH_FLAG="--architecture $arch"
          ARCH_DESC="arch${arch}"
        fi
        
        # Set freeze-centers flag
        if [ "$freeze_centers" = "True" ]; then
          FREEZE_FLAG="--freeze-centers"
          FREEZE_DESC="frozen"
        else
          FREEZE_FLAG=""
          FREEZE_DESC="free"
        fi
        
        # Create descriptive job name
        JOB_NAME="tbnn_2stage_${FREEZE_DESC}_${ARCH_DESC}_lr${lr}_s2_${stage2}"
        SCRIPT_FILE="./job_scripts/job_${JOB_NUM}_${JOB_NAME}.sh"
        
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
echo "TBNN 2-Stage Training - Job ${JOB_NUM}/24"
echo "Job Name: ${JOB_NAME}"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Start time: \$(date)"
echo "=========================================="
echo ""
echo "Parameters:"
echo "  freeze_centers: $freeze_centers"
echo "  architecture: $arch"
echo "  learning_rate: $lr"
echo "  stage2_steps_curv: $stage2"
echo "  stage1_steps_etainf: $STAGE1_STEPS"
echo "=========================================="

# Activate environment
source activate cfd_md_optimization

# Run the training
python campaigns/run_tbnn_debug_constriction_cluster_new.py ${JOB_NUM} \\
    --M $M \\
    $ARCH_FLAG \\
    --num-steps 1 \\
    --learning-rate $lr \\
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
    --stage2-steps-curv $stage2 \\
    --stage1-etainf-only \\
    --stage1-reset-momentum \\
    --stage1-early-stop-on-flip \\
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
        echo "  ✓ Submitted job $JOB_NUM as SLURM job $JOBID"
        echo "    Script: $SCRIPT_FILE"
        
        # Increment job counter
        JOB_NUM=$((JOB_NUM + 1))
        
        # Small delay to avoid overwhelming scheduler
        sleep 0.5
        
      done
    done
  done
done

echo ""
echo "=========================================="
echo "ALL 24 JOBS SUBMITTED"
echo "=========================================="
echo "Total jobs: 24"
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
echo "Parameter sweep:"
echo "  freeze_centers: ${FREEZE_CENTERS_VALS[@]}"
echo "  architectures: [] [16]"
echo "  learning_rates: ${LEARNING_RATES[@]}"
echo "  stage2_steps: ${STAGE2_STEPS[@]}"
echo "  Total combinations: 2 × 2 × 3 × 2 = 24 jobs"
echo "=========================================="


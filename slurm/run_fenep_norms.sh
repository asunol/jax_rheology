#!/bin/bash
#SBATCH -J fenep_norms
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G
#SBATCH -p gpu_test
#SBATCH -t 01:00:00
#SBATCH -o ./jobs/%j_%x.out
#SBATCH -e ./jobs/%j_%x.err

# ===========================================================================
# FENE7 campaign -- Phase 0 (ONE gpu_test job, ~15 min).
# Fresh truth forwards FENE-P L^2=12 at U=0.5 AND U=4.0, ramp 1.0, T=2.0,
# production grid (128x256, dt=1e-4, inner=50, outer=400) -> per-tap
# max_t|dp*| norms per rate, per-rate alpha scales, and w_bal_v3 (init-balance
# of L_vel vs normalized L_press) -> ONE reference_values/fenep_rate_balance_norms.json (single
# source of truth; config hash verified by every production job). The dump
# path also writes a matching _health.json (KE/max_Axx/psi_min/dp3 vs t per rate).
# NEVER reuse fene6 T=3 data. Forwards fit the gpu_test MIG slice (no backward).
# ===========================================================================
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p ./jobs ./reference_values ./work/fenep_rset
echo "=== fenep_norms  Job ${SLURM_JOB_ID:-local}  $(date) ==="
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate cfd_md_optimization
echo "[env] python = $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda      # fail fast if the GPU cannot init (no CPU fallback)
python -c "import jax; print('[jax] backend =', jax.default_backend(), jax.devices())"

python -u campaigns/visco_opt_tbnn_contraction_run.py \
    --truth-model fene_p --scheme s4 --loss-weight roi \
    --truth-gp 3.2 --truth-lam 0.7 --truth-nus 0.8 --truth-lsq 12.0 \
    --gp-init 1.0 --lam-init 1.0 --nus-init 1.0 \
    --nx 128 --ny 256 --dt 1e-4 --inner 50 --outer 400 --ramp-time 1.0 \
    --lr 5e-4 --warmup 20 --clip 1.0 --stage1-ftol 1e-9 --width 32 --depth 2 \
    --U-list 0.5,4 --w-p-scale 1.0 --n-sub 8 \
    --dump-norms ./reference_values/fenep_rate_balance_norms.json \
    --out-dir ./work/fenep_rset --run-name _phase0_dump
DUMP_EC=$?
echo "=== fenep_norms done  exit $DUMP_EC  $(date) ==="
if [ "$DUMP_EC" -ne 0 ]; then
    echo "[HALT] norms dump exited $DUMP_EC -- STOP."; exit "$DUMP_EC"
fi
ls -la ./reference_values/fenep_rate_balance_norms.json ./reference_values/fenep_rate_balance_norms_health.json
exit 0

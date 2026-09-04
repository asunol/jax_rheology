#!/bin/bash
#SBATCH -J fenep_bal
#SBATCH -p gpu,seas_gpu
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem-per-gpu=80G
#SBATCH -o ./jobs/%j_%x.out
#SBATCH -e ./jobs/%j_%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=asunol@seas.harvard.edu

# FENE8 balanced arm: R2 config + --rate-balance equal. Uses the MODIFIED
# script (not FENE8_PINNED). Self-resubmits via train_ckpt like fene7.
set -u
RUN=${1:?run name}
SEED=${2:?seed}
shift 2 || true
EXTRA="$*"
OUT=./work/fenep_contraction
NORMS=./reference_values/fenep_rate_balance_norms.json
TIME_BUDGET=${TIME_BUDGET:-165600}
MAX_RESUB=${MAX_RESUB:-6}

cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p ./jobs "$OUT/$RUN"
# R2 resume guard: refuse to start if another job for this run is already queued
# (covers external double-submit; self-resubmit path also checks below).
# shellcheck source=slurm/fenep_submit_guard.sh
source slurm/fenep_submit_guard.sh
# When THIS job has already been allocated, skip the "refuse if SUBMITTED is me"
# false positive by temporarily clearing the check against our own job id.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -f "$OUT/$RUN/SUBMITTED" ]; then
  _prev=$(tr -d '[:space:]' < "$OUT/$RUN/SUBMITTED" || true)
  if [ "$_prev" != "$SLURM_JOB_ID" ]; then
    if ! fenep_guard_refuse_if_queued "$RUN"; then
      echo "[guard] another job owns $RUN; exiting without work."
      exit 3
    fi
  fi
fi
echo "$SLURM_JOB_ID" > "$OUT/$RUN/SUBMITTED"
echo "=== fenep_two_rate_balanced run=$RUN seed=$SEED extra='$EXTRA' Job $SLURM_JOB_ID $(date) ==="
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate cfd_md_optimization
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
python -c "import jax; print('[jax] backend =', jax.default_backend())"

python -u campaigns/visco_opt_tbnn_contraction_run.py \
    --truth-model fene_p --scheme s4 --loss-weight roi \
    --truth-gp 3.2 --truth-lam 0.7 --truth-nus 0.8 --truth-lsq 12.0 \
    --gp-init 1.0 --lam-init 1.0 --nus-init 1.0 \
    --nx 128 --ny 256 --dt 1e-4 --inner 50 --outer 400 --ramp-time 1.0 \
    --lr 5e-4 --warmup 20 --clip 1.0 --stage1-ftol 1e-9 --width 32 --depth 2 \
    --seed "$SEED" --curriculum-ulist 0.5,4 --curriculum-gate-after 1 \
    --w-p-scale 1.0 --rate-balance equal \
    --norms-json "$NORMS" --n-sub 8 --ckpt-every 25 \
    --time-budget-s "$TIME_BUDGET" --out-dir "$OUT" --run-name "$RUN" $EXTRA
EXIT=$?
echo "=== fenep_two_rate_balanced $RUN done exit $EXIT $(date) ==="

if [ -f "$OUT/$RUN/GATE_FAILED" ] || [ -f "$OUT/$RUN/STEP_GUARD_STOP" ]; then
  echo "[terminal] gate/step-guard stop; not resubmitting."
  exit "$EXIT"
fi
if [ ! -f "$OUT/$RUN/DONE" ] && [ "$EXIT" -eq 0 ]; then
  N=$(cat "$OUT/$RUN/resubmit_count" 2>/dev/null || echo 0)
  if [ "$N" -lt "$MAX_RESUB" ]; then
    # Clear our SUBMITTED marker so the guard does not refuse ourselves,
    # then refuse if any OTHER job for this tag is already queued.
    rm -f "$OUT/$RUN/SUBMITTED"
    if fenep_guard_refuse_if_queued "$RUN"; then
      echo $((N + 1)) > "$OUT/$RUN/resubmit_count"
      sbatch slurm/run_fenep_two_rate_balanced.sh "$RUN" "$SEED" $EXTRA
    else
      echo "[guard] skip self-resubmit for $RUN; another job already queued."
    fi
  fi
fi
exit $EXIT

#!/bin/bash
#SBATCH -J fenep_u05
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

# ===========================================================================
# FENE7 curriculum dual-rate campaign -- one production fit (of seven).
# Scheme s4, 600 theta-steps, 128x256, dt=1e-4, inner=50, outer=400 (T=2.0),
# ramp 1.0, ROI velocity loss (+ per-tap-normalized pressure when velp).
# Curriculum: stage-1 trains on U=0.5; a forward envelope gate at U=4 (as-is +
# lam x1.5) activates the dual {0.5,4} in-run. Resumable across 48h windows via
# train_ckpt.pkl (+ DONE marker); self-resubmits (afterany-style) while DONE is
# absent -> covers the ~40-45h curriculum runs.
#
#   sbatch run_fenep_single_rate_prod.sh <run_name> <cur|u05> <vel|velp> [EXTRA CLI ...]
#     e.g. sbatch run_fenep_single_rate_prod.sh fene7_cur_velp    cur velp
#          sbatch run_fenep_single_rate_prod.sh fene7_u05_vel     u05 vel
#          sbatch run_fenep_single_rate_prod.sh fene7_cur_velp_s1 cur velp --seed 1
#          sbatch run_fenep_single_rate_prod.sh fene7_cur_velp_lo cur velp --lam-init 0.35
# ===========================================================================
set -u
RUN=${1:?run name}
RATEKEY=${2:?cur|u05}
MODE=${3:?vel|velp}
shift 3 || true
EXTRA="$*"
OUT=./work/fenep_contraction
NORMS=./reference_values/fenep_rate_balance_norms.json
TIME_BUDGET=${TIME_BUDGET:-165600}     # 46h: margin for largest s4 block + finalize
MAX_RESUB=${MAX_RESUB:-6}

case "$RATEKEY" in
    cur)  RATE="--curriculum-ulist 0.5,4 --curriculum-gate-after 1" ;;
    u05)  RATE="--U 0.5" ;;
    *) echo "bad rate key $RATEKEY (cur|u05)"; exit 2 ;;
esac
case "$MODE" in
    vel)  WP="--w-p 0" ;;
    velp) WP="--w-p-scale 1.0" ;;
    *) echo "bad mode $MODE (vel|velp)"; exit 2 ;;
esac

cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p ./jobs "$OUT/$RUN"
echo "=== fenep_single_rate run=$RUN rate=$RATEKEY mode=$MODE extra='$EXTRA'  Job $SLURM_JOB_ID  $(date) ==="
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate cfd_md_optimization
echo "[env] python = $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
python -c "import jax; print('[jax] backend =', jax.default_backend())"

if [ ! -f "$NORMS" ]; then
    echo "[HALT] $NORMS missing -- run Phase 0 (run_fenep_norms.sh) first."
    exit 3
fi

# The fit itself is the config; only the per-run knobs are set here.
python -u experiments/contraction_train.py --config experiments/configs/fenep_single_rate_u05.yaml \
    --seed 0 $RATE $WP \
    --time-budget-s "$TIME_BUDGET" --out-dir "$OUT" --run-name "$RUN" $EXTRA
EXIT=$?
echo "=== fenep_single_rate $RUN done  exit $EXIT  $(date) ==="

# GATE_FAILED / STEP_GUARD_STOP are terminal -- do NOT resubmit those.
if [ -f "$OUT/$RUN/GATE_FAILED" ]; then
    echo "[curriculum] GATE_FAILED present -- terminal, not resubmitting."
    exit "$EXIT"
fi
if [ -f "$OUT/$RUN/STEP_GUARD_STOP" ]; then
    echo "[guard] STEP_GUARD_STOP present -- terminal, not resubmitting."
    exit "$EXIT"
fi

# chained continuation: resubmit while DONE is absent and we stopped cleanly
if [ ! -f "$OUT/$RUN/DONE" ] && [ "$EXIT" -eq 0 ]; then
    N=$(cat "$OUT/$RUN/resubmit_count" 2>/dev/null || echo 0)
    if [ "$N" -lt "$MAX_RESUB" ]; then
        echo $((N + 1)) > "$OUT/$RUN/resubmit_count"
        echo "[resume] recipe incomplete; resubmitting ($((N + 1))/$MAX_RESUB)."
        sbatch slurm/run_fenep_single_rate_prod.sh "$RUN" "$RATEKEY" "$MODE" $EXTRA
    else
        echo "[resume] hit MAX_RESUB=$MAX_RESUB; NOT resubmitting (investigate)."
    fi
fi
exit $EXIT

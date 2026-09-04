#!/bin/bash
#SBATCH -J gie_prod
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

# ---------------------------------------------------------------------------
# Giesekus ROI recovery re-run at production numerics (point A).
# Velocity ROI only, w_p=0, single U=0.5. 128x256, dt=1e-4, T=2.0.
# Resumable across 48h windows via train_ckpt.pkl + DONE marker.
# Block-boundary archives: ckpt_block{k}.pkl (written by the driver).
#
#   sbatch run_giesekus_contraction.sh <scheme s1|s1b|s4>
# ---------------------------------------------------------------------------
SCHEME=${1:?scheme s1|s1b|s4}
OUT=./work/giesekus_contraction
RUN="gie_A_${SCHEME}"
TIME_BUDGET=${TIME_BUDGET:-147600}
MAX_RESUB=${MAX_RESUB:-8}

cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p ./jobs "$OUT/$RUN"
echo "=== gie_prod scheme=$SCHEME run=$RUN  Job $SLURM_JOB_ID  $(date) ==="
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate cfd_md_optimization
echo "[env] python = $(which python)"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# NOTE: --nx/--ny (not --cells-per-H 24) — square-cell cph=24 is ~432x192;
# production 4x y-res is the non-square 128x256 grid (Phase 0 documented this).
# --w-p 0 and no --w-p-scale => velocity-only, bit-identical path, no pressure.csv.
python -u campaigns/visco_opt_tbnn_contraction_run.py \
    --truth-model giesekus --scheme "$SCHEME" --loss-weight roi \
    --truth-gp 3.2 --truth-lam 0.7 --truth-nus 0.8 --truth-alpha 0.3 \
    --gp-init 1.0 --lam-init 1.0 --nus-init 1.0 \
    --U 0.5 --nx 128 --ny 256 \
    --dt 1e-4 --inner 50 --outer 400 --ramp-time 0.7 \
    --lr 1e-4 --w-p 0 --ckpt-every 25 \
    --width 32 --depth 2 --seed 0 \
    --time-budget-s "$TIME_BUDGET" \
    --out-dir "$OUT" --run-name "$RUN"
EXIT=$?
echo "=== gie_prod $RUN done  exit $EXIT  $(date) ==="

if [ ! -f "$OUT/$RUN/DONE" ] && [ "$EXIT" -eq 0 ]; then
    N=$(cat "$OUT/$RUN/resubmit_count" 2>/dev/null || echo 0)
    if [ "$N" -lt "$MAX_RESUB" ]; then
        echo $((N + 1)) > "$OUT/$RUN/resubmit_count"
        echo "[resume] recipe incomplete; resubmitting ($((N + 1))/$MAX_RESUB)."
        sbatch slurm/run_giesekus_contraction.sh "$SCHEME"
    else
        echo "[resume] hit MAX_RESUB=$MAX_RESUB; NOT resubmitting."
    fi
fi
exit $EXIT

#!/bin/bash
#SBATCH -J fenep_dir
#SBATCH -p gpu,seas_gpu
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -n 1
#SBATCH -t 0-12:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem-per-gpu=80G
#SBATCH -o ./jobs/%j_%x.out
#SBATCH -e ./jobs/%j_%x.err

# Direct FENE-P fits / truth-init gates. Usage:
#   sbatch run_fenep_direct.sh <run_name> <regime> <start_key> [EXTRA...]
# regime: u05 | dual_legacy | dual_equal | gate_u05 | gate_dual_legacy | gate_dual_equal
# start_key: I1|I2|I3|I4|I5|truth
set -u
RUN=${1:?run name}
REGIME=${2:?regime}
START=${3:?start}
shift 3 || true
EXTRA="$*"
OUT=./work/fenep_direct
cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p ./jobs "$OUT/$RUN"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate cfd_md_optimization
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
echo "=== fenep_direct run=$RUN regime=$REGIME start=$START extra='$EXTRA' Job $SLURM_JOB_ID $(date) ==="

case "$REGIME" in
  u05|gate_u05)
    RATE=(--U 0.5 --w-p 0)
    ;;
  dual_legacy|gate_dual_legacy)
    RATE=(--U-list 0.5,4 --w-p-scale 1.0 --rate-balance legacy --norms-json ./reference_values/fenep_rate_balance_norms.json)
    ;;
  dual_equal|gate_dual_equal)
    RATE=(--U-list 0.5,4 --w-p-scale 1.0 --rate-balance equal --norms-json ./reference_values/fenep_rate_balance_norms.json)
    ;;
  *) echo "bad regime $REGIME"; exit 2 ;;
esac

INIT=()
GATE=()
case "$START" in
  truth)
    INIT=(--truth-init --gate-iters 10)
    GATE=()
    ;;
  I1) INIT=(--init-gp 2.0 --init-lam 1.5 --init-nus 0.5 --init-lsq 50) ;;
  I2) INIT=(--init-gp 3.0 --init-lam 0.4 --init-nus 1.0 --init-lsq 8) ;;
  I3) INIT=(--init-gp 4.5 --init-lam 0.5 --init-nus 0.9 --init-lsq 20) ;;
  I4) INIT=(--init-gp 1.0 --init-lam 3.0 --init-nus 1.5 --init-lsq 200) ;;
  I5)
    # log-uniform in [0.5,8]x[0.1,3]x[0.2,2]x[5,100]; seed from EXTRA or 0
    SEED=0
    for tok in $EXTRA; do
      case "$tok" in --seed) ;; esac
    done
    # parse --seed N from EXTRA if present
    set -- $EXTRA
    while [ $# -gt 0 ]; do
      if [ "$1" = "--seed" ]; then SEED=$2; shift 2; else shift; fi
    done
    read GP LAM NUS LSQ < <(python - <<PY
import math, numpy as np
rng = np.random.default_rng(int("$SEED"))
def lu(a,b):
    return float(math.exp(rng.uniform(math.log(a), math.log(b))))
print(lu(0.5,8), lu(0.1,3), lu(0.2,2), lu(5,100))
PY
)
    INIT=(--init-gp "$GP" --init-lam "$LAM" --init-nus "$NUS" --init-lsq "$LSQ" --seed "$SEED")
    ;;
  *) echo "bad start $START"; exit 2 ;;
esac

python -u campaigns/visco_opt_fenep_direct_contraction_run.py \
  --truth-gp 3.2 --truth-lam 0.7 --truth-nus 0.8 --truth-lsq 12.0 \
  --nx 128 --ny 256 --dt 1e-4 --inner 50 --outer 400 --ramp-time 1.0 \
  --loss-weight roi --maxiter 60 \
  "${RATE[@]}" "${INIT[@]}" \
  --out-dir "$OUT" --run-name "$RUN" $EXTRA
EXIT=$?
echo "=== fenep_direct $RUN done exit $EXIT $(date) ==="
exit $EXIT

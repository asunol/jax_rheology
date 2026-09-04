#!/bin/bash
#SBATCH -J evp_fix
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=48G
#SBATCH -o ./jobs/%j_evp_fix_%x.out
#SBATCH -e ./jobs/%j_evp_fix_%x.err

# One arm of the 8-run matrix. Partition / wall / job-name are set by the
# launcher (launch_evp_fix_matrix.sh) via sbatch flags; everything below is
# common to all eight arms.
#
# Positional args:  $1 = drive set (A|B)   $2 = horizon (3lam|7lam)
#                   $3 = init (agn|br)
set -u
DSET="$1"; HKEY="$2"; INIT="$3"

mkdir -p ./jobs ./work/evp_channel
export MPLCONFIGDIR=/tmp/mpl_$USER
# --export=ALL can leak a login-shell JAX_PLATFORMS=cpu in here and silently
# demote the job to CPU. This is a GPU job: make sure it stays one.
unset JAX_PLATFORMS

PY=${TBNN_PY:?set TBNN_PY to a Python from environment.yml}
RUN="evp_fix_${DSET}_${HKEY}_${INIT}"
TARGETS="./reference_values/evp_targets_geom${DSET}_${HKEY}.json"

case "$DSET" in
  A) GXLIST="1.8,2.5,4.0" ;;
  B) GXLIST="1.6,1.8,2.5" ;;
  *) echo "bad drive set $DSET"; exit 64 ;;
esac
case "$HKEY" in
  3lam) OUTER=84  ;;
  7lam) OUTER=200 ;;
  *) echo "bad horizon $HKEY"; exit 64 ;;
esac
# agnostic init = all four scalars start at 1.0 (--no-br-init); br = the
# per-arm Buckingham-Reiner init the runner derives from THIS arm's own truth
# flow rates.
case "$INIT" in
  agn) INIT_FLAG="--no-br-init" ;;
  br)  INIT_FLAG="" ;;
  *) echo "bad init $INIT"; exit 64 ;;
esac

# lambda_q = multiple x lambda0_new with multiple == 1.0 (G4: v2_prod2's
# lambda_q was exactly its lambda0). Read from the per-arm targets JSON so the
# launched value and the recorded one cannot drift.
LAMQ=$($PY -c "import json,sys; print(repr(json.load(open('${TARGETS}'))['lambda_q']))")
if [ -z "$LAMQ" ]; then echo "could not read lambda_q from ${TARGETS}"; exit 65; fi

# Wall/budget: v2_prod2 measured 46.03 s/grad at outer=84 over 498 grads
# (wall_opt 22921 s = 6.4 h). outer=200 is 200/84 = 2.381x that.
WALL_S=$(( $(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*TimeLimit=\([0-9:-]*\).*/\1/p' | awk -F'[-:]' 'NF==4{print ($1*86400)+($2*3600)+($3*60)+$4} NF==3{print ($1*3600)+($2*60)+$3}') ))
BUDGET=$(python3 -c "print(int(0.92*${WALL_S}))")

echo "=== ${RUN} job ${SLURM_JOB_ID} part=${SLURM_JOB_PARTITION} $(date) ==="
echo "    drives=${GXLIST} outer=${OUTER} init=${INIT} lambda_q=${LAMQ}"
echo "    targets=${TARGETS}  wall=${WALL_S}s budget=${BUDGET}s"

$PY -u campaigns/visco_opt_tbnn_evp_run.py \
  --geometry channel \
  --g-x-list "${GXLIST}" \
  --outer-steps "${OUTER}" \
  --inner-steps 10 \
  --Nx 32 --Ny 64 \
  --fit-solver-tol 1e-8 \
  --yield-mode scalar \
  --targets-json "${TARGETS}" \
  --lambda-q "${LAMQ}" \
  ${INIT_FLAG} \
  --truth-gp 3.2 --truth-lam 0.7 --truth-nus 0.8 --truth-tau-y 1.45 \
  --init ob --seed 0 --theta-seed 0 \
  --width 32 --depth 2 \
  --lr 5e-4 --scalar-lr2 2e-3 --clip 1.0 \
  --adam-steps-per-kappa 60 --tail-step-mult 2 --warmup 10 \
  --stage1-maxiter 60 --stage1-ftol 1e-9 \
  --n-shear 12 \
  --scalar-bound-lo 0.02 --scalar-bound-hi 20.0 \
  --kappa-schedule "1.0,0.3,0.1,0.05,0.02" \
  --time-budget-s "${BUDGET}" --wall-time-s "${WALL_S}" \
  --out-dir ./work/evp_channel \
  --run-name "${RUN}" \
  --resume
EXIT_CODE=$?
echo "=== ${RUN} done exit ${EXIT_CODE} $(date) ==="
exit $EXIT_CODE

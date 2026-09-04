#!/bin/bash
#SBATCH -J evp_fix_seed
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=48G
#SBATCH -o ./jobs/%j_evp_fix_seed_%x.out
#SBATCH -e ./jobs/%j_evp_fix_seed_%x.err

# One seed replica of evp_fix_A_3lam_agn. Partition/wall/job-name set by launcher.
# Positional: $1 = seed integer (used for both --seed and --theta-seed)
#             run name = evp_fix_A_3lam_agn_s${SEED}
set -u
SEED="$1"

mkdir -p ./jobs ./work/evp_channel
export MPLCONFIGDIR=/tmp/mpl_$USER
unset JAX_PLATFORMS

PY=${TBNN_PY:?set TBNN_PY to a Python from environment.yml}
RUN="evp_fix_A_3lam_agn_s${SEED}"
TARGETS="./reference_values/evp_targets_geomA_3lam.json"
GXLIST="1.8,2.5,4.0"
OUTER=84

LAMQ=$($PY -c "import json; print(repr(json.load(open('${TARGETS}'))['lambda_q']))")
if [ -z "$LAMQ" ]; then echo "could not read lambda_q from ${TARGETS}"; exit 65; fi

WALL_S=$(( $(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*TimeLimit=\([0-9:-]*\).*/\1/p' | awk -F'[-:]' 'NF==4{print ($1*86400)+($2*3600)+($3*60)+$4} NF==3{print ($1*3600)+($2*60)+$3}') ))
BUDGET=$(python3 -c "print(int(0.92*${WALL_S}))")

echo "=== ${RUN} job ${SLURM_JOB_ID} part=${SLURM_JOB_PARTITION} $(date) ==="
echo "    drives=${GXLIST} outer=${OUTER} agnostic init  theta_seed=${SEED} seed=${SEED}"
echo "    lambda_q=${LAMQ} targets=${TARGETS}  wall=${WALL_S}s budget=${BUDGET}s"
echo "    replica of evp_fix_A_3lam_agn (which used theta_seed=0); this is ensemble seed ${SEED}"

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
  --no-br-init \
  --truth-gp 3.2 --truth-lam 0.7 --truth-nus 0.8 --truth-tau-y 1.45 \
  --init ob --seed "${SEED}" --theta-seed "${SEED}" \
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

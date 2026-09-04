#!/bin/bash
#SBATCH -J bic_fit
#SBATCH -p test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 0-12:00
#SBATCH --mem 48G
#SBATCH -o ./work/bic_battery/logs/%A_%a_%x.out
#SBATCH -e ./work/bic_battery/logs/%A_%a_%x.err
# Array index map written by submit_bic_battery.sh
#SBATCH --array=0-0

set -u
cd "$(dirname "$(readlink -f "$0")")/.."
MAP=work/bic_battery/fit_array_map.txt
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=True
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
PY=${TBNN_PYDR:?set TBNN_PYDR to a Python from environment_diff_rheo.yml}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MAP")
set -- $line
TARGET=$1; CAND=$2; RESTART=$3
echo "array=${SLURM_ARRAY_TASK_ID} target=${TARGET} cand=${CAND} restart=${RESTART} start=$(date -Is)"
exec "${PY}" -u campaigns/battery/tbnn_bic_final_battery.py fit \
  --target "${TARGET}" --candidate "${CAND}" --restart "${RESTART}"

#!/bin/bash
#SBATCH -J bic_batt
#SBATCH -p test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 0-12:00
#SBATCH --mem 48G
#SBATCH -o ./work/bic_battery/logs/%j_%x.out
#SBATCH -e ./work/bic_battery/logs/%j_%x.err

set -u
cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p work/bic_battery/logs
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=True
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
PY=${TBNN_PYDR:?set TBNN_PYDR to a Python from environment_diff_rheo.yml}
MODE="${1:-}"
echo "mode=${MODE} args=$* job=${SLURM_JOB_ID:-local} host=$(hostname) start=$(date -Is)"

case "${MODE}" in
  prepare)
    exec "${PY}" -u campaigns/battery/tbnn_bic_final_battery.py prepare --target "${2:?}"
    ;;
  fit)
    exec "${PY}" -u campaigns/battery/tbnn_bic_final_battery.py fit \
      --target "${2:?}" --candidate "${3:?}" --restart "${4:?}"
    ;;
  merge-target)
    exec "${PY}" -u campaigns/battery/tbnn_bic_final_battery.py merge-target --target "${2:?}"
    ;;
  list)
    exec "${PY}" -u campaigns/battery/tbnn_bic_final_battery.py list
    ;;
  *)
    echo "usage: $0 prepare TARGET | fit TARGET CANDIDATE RESTART | merge-target TARGET | list"
    exit 2
    ;;
esac

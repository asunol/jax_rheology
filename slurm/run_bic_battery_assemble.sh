#!/bin/bash
#SBATCH -J f8final
#SBATCH -p test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 0-02:00
#SBATCH --mem 32G
#SBATCH -o ./work/bic_battery/logs/%j_final_assemble.out
#SBATCH -e ./work/bic_battery/logs/%j_final_assemble.err

# PR18–PR21 final assembly. Arm with:
#   sbatch --dependency=afterany:36487574,36487575 run_bic_battery_assemble.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p work/bic_battery/logs
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
PY=${TBNN_PYDR:?set TBNN_PYDR to a Python from environment_diff_rheo.yml}

echo "=== bic_battery assemble job=$SLURM_JOB_ID dep=${SLURM_JOB_DEPENDENCY:-none} $(date -Is) ==="
# Soft-wait: if deps completed but DONE markers lag a moment, poll briefly
for r in fene8_bal_s3 fene8_bal_s4; do
  rb=work/bic_battery/readback/$r
  for i in $(seq 1 30); do
    if [ -f "$rb/DONE" ] && [ -f "work/bic_battery/targets/$r.json" ]; then
      echo "[ok] $r readback+target present"
      break
    fi
    echo "[wait] $r not ready (try $i/30); sleep 60"
    sleep 60
  done
  if [ ! -f "$rb/DONE" ]; then
    echo "[HALT] $r readback DONE missing after wait"
    exit 4
  fi
done

exec "$PY" -u campaigns/battery/battery_assemble.py

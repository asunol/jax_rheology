#!/bin/bash
# Submit FENE8 publication battery on partition `test`.
# QoS MaxJobs/MaxSubmitJobs=5 => array concurrency capped at %5.
PY=${TBNN_PYDR:?set TBNN_PYDR to a Python from environment_diff_rheo.yml}
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
OUT=work/bic_battery
mkdir -p "$OUT/logs" "$OUT/fits" "$OUT/targets" "$OUT/data" "$OUT/errors"
chmod +x slurm/run_bic_battery_cpu.sh slurm/run_bic_battery_array.sh

TARGETS=(R1 R2 R3 R4 R5 R6 R7 T1 T2 T3 gie_A_s1 gie_A_s1b gie_A_s4 \
         clean_analytic_fene_p clean_analytic_giesekus)
PANEL=(Newtonian OldroydB Giesekus FENEPConformation LinearPTT)

# Fit array map: 15*5*3 = 225 lines
MAP="$OUT/fit_array_map.txt"
: > "$MAP"
for t in "${TARGETS[@]}"; do
  for c in "${PANEL[@]}"; do
    for r in 0 1 2; do
      echo "$t $c $r" >> "$MAP"
    done
  done
done
NMAP=$(wc -l < "$MAP")
echo "[map] $NMAP fit tasks -> $MAP"

wait_slurm() {
  local i
  for i in $(seq 1 60); do
    if sbatch --test-only slurm/run_bic_battery_cpu.sh list >/tmp/bic_battery_test.out 2>&1; then
      return 0
    fi
    echo "[wait] slurm controller not ready (attempt $i); $(date -Is)"
    cat /tmp/bic_battery_test.out || true
    sleep 10
  done
  return 1
}

wait_slurm || { echo "SLURM_UNAVAILABLE"; exit 1; }

# Prepare array map
PMAP="$OUT/prepare_array_map.txt"
printf '%s\n' "${TARGETS[@]}" > "$PMAP"

# Rewrite prepare array script on the fly via sbatch --array
PREP_ID=$(sbatch --parsable --array=0-$((${#TARGETS[@]} - 1))%5 \
  --job-name=bic_prep \
  --partition=test --nodes=1 --ntasks=1 --cpus-per-task=16 \
  --time=0-04:00 --mem=48G \
  --output="$OUT/logs/%A_%a_prep.out" \
  --error="$OUT/logs/%A_%a_prep.err" \
  --wrap="cd \"$PWD\"; \
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16; \
T=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID+1))p\" $PMAP); \
$PY -u campaigns/battery/tbnn_bic_final_battery.py prepare --target \$T")
echo "[ok] prepare array $PREP_ID"

# Fit array depends on all prepare tasks
FIT_ID=$(sbatch --parsable --array=0-$((NMAP - 1))%5 \
  --dependency=afterok:${PREP_ID} \
  --job-name=bic_fit \
  --partition=test --nodes=1 --ntasks=1 --cpus-per-task=16 \
  --time=0-12:00 --mem=48G \
  --output="$OUT/logs/%A_%a_fit.out" \
  --error="$OUT/logs/%A_%a_fit.err" \
  --wrap="cd \"$PWD\"; \
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16; \
line=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID+1))p\" $MAP); \
set -- \$line; T=\$1; C=\$2; R=\$3; \
$PY -u campaigns/battery/tbnn_bic_final_battery.py fit \
  --target \$T --candidate \$C --restart \$R")
echo "[ok] fit array $FIT_ID (afterok:$PREP_ID)"

# Merge + interim early-warning after fits
MERGE_ID=$(sbatch --parsable --array=0-$((${#TARGETS[@]} - 1))%5 \
  --dependency=afterok:${FIT_ID} \
  --job-name=bic_merge \
  --partition=test --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --time=0-01:00 --mem=8G \
  --output="$OUT/logs/%A_%a_merge.out" \
  --error="$OUT/logs/%A_%a_merge.err" \
  --wrap="cd \"$PWD\"; \
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True PYTHONDONTWRITEBYTECODE=1; \
T=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID+1))p\" $PMAP); \
$PY -u campaigns/battery/tbnn_bic_final_battery.py merge-target --target \$T")
echo "[ok] merge array $MERGE_ID (afterok:$FIT_ID)"

EARLY_ID=$(sbatch --parsable \
  --dependency=afterok:${MERGE_ID} \
  --job-name=bic_early \
  --partition=test --nodes=1 --ntasks=1 --cpus-per-task=2 \
  --time=0-00:30 --mem=4G \
  --output="$OUT/logs/%j_early.out" \
  --error="$OUT/logs/%j_early.err" \
  --wrap="cd \"$PWD\"; \
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True; \
$PY -u campaigns/battery/battery_early_warning.py")
echo "[ok] early-warning $EARLY_ID (afterok:$MERGE_ID)"

printf '%s\n' "$PREP_ID" "$FIT_ID" "$MERGE_ID" "$EARLY_ID" > "$OUT/slurm_job_ids.txt"
echo "SUBMITTED prep=$PREP_ID fit=$FIT_ID merge=$MERGE_ID early=$EARLY_ID"
squeue -u "$USER" -o '%.18i %.12P %.20j %.8T %.10M %R' | head -40

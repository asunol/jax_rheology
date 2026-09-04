#!/bin/bash
# R4: submit all 15 production direct FENE-P fits (gates already passed).
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
chmod +x slurm/run_fenep_direct.sh
mkdir -p work/fenep_direct jobs
submit() {
  local run=$1 regime=$2 start=$3
  shift 3 || true
  mkdir -p "work/fenep_direct/$run"
  if [ -f "work/fenep_direct/$run/DONE" ]; then
    echo "[skip] $run DONE"; return 0
  fi
  if [ -f "work/fenep_direct/$run/SUBMITTED" ]; then
    jid=$(tr -d '[:space:]' < "work/fenep_direct/$run/SUBMITTED")
    if squeue -j "$jid" -h 2>/dev/null | grep -q .; then
      echo "[skip] $run already queued as $jid"; return 0
    fi
  fi
  jid=$(sbatch --parsable slurm/run_fenep_direct.sh "$run" "$regime" "$start" "$@")
  echo "$jid" > "work/fenep_direct/$run/SUBMITTED"
  echo "[ok] $run -> $jid"
}
# regime x start
for start in I1 I2 I3 I4; do
  submit "direct_u05_${start}" u05 "$start"
  submit "direct_dual_legacy_${start}" dual_legacy "$start"
  submit "direct_dual_equal_${start}" dual_equal "$start"
done
# I5 with recorded seeds (distinct per regime)
submit "direct_u05_I5" u05 I5 --seed 501
submit "direct_dual_legacy_I5" dual_legacy I5 --seed 502
submit "direct_dual_equal_I5" dual_equal I5 --seed 503
echo '--- direct queue ---'
squeue -u "$USER" -n fenep_dir -o '%.18i %.12P %.10T %.10M %R' | head -40

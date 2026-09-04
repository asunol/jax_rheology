#!/bin/bash
# Launch the 8-arm matrix: 2 drive sets x 2 horizons x 2 inits.
#
# Observed concurrency on the production partitions (sacctmgr, this account):
#   gpu_test  QOS gpu_test : MaxJobsPU=2,     MaxSubmitPU=2   (gates only)
#   gpu       QOS normal   : MaxJobs=10100,   MaxSubmit=10100  -> no cap
#   seas_gpu  QOS normal   : MaxJobs=10100,   MaxSubmit=10100  -> no cap
# There is no per-user job-count limit on the production partitions, so all
# eight arms are submitted at once and Slurm stages them against available
# GPUs. They are split across `gpu` and `seas_gpu` to widen the pool.
#
# Wall: v2_prod2 measured 46.03 s/grad at outer=84 over 498 grads
# (wall_opt = 22921 s = 6.4 h). outer=200 is 200/84 = 2.381x -> ~110 s/grad,
# ~15.2 h of optimiser time. 3lam arms get 24 h, 7lam arms get 48 h.
# seas_gpu MaxTime is 2-00:00:00 and gpu MaxTime is 3-00:00:00, so both fit.
# Every arm passes --resume, so a wall-clock kill can be requeued without
# losing the fit.
set -eu
cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p jobs work/evp_channel

TARGET_DIR=./reference_values
for D in A B; do for H in 3lam 7lam; do
  if [ ! -f "${TARGET_DIR}/targets_${D}_${H}.json" ]; then
    echo "MISSING ${TARGET_DIR}/evp_targets_geom${D}_${H}.json -- run campaigns/evp_fix_targets.py first"
    exit 1
  fi
done; done
if [ ! -f "${TARGET_DIR}/consistency_check.json" ]; then
  echo "MISSING consistency_check.json -- Part 4 check has not run"; exit 1
fi
if ! python3 -c "import json,sys; sys.exit(0 if json.load(open('${TARGET_DIR}/consistency_check.json'))['pass_'] else 1)"; then
  echo "Part 4 consistency check FAILED -- refusing to launch"; exit 1
fi

# arm -> partition, alternating to spread across the two pools.
i=0
for D in A B; do
  for H in 3lam 7lam; do
    for I in agn br; do
      case "$H" in
        3lam) WALL=1-00:00:00 ;;
        7lam) WALL=2-00:00:00 ;;
      esac
      if [ $((i % 2)) -eq 0 ]; then PART=gpu; else PART=seas_gpu; fi
      NAME="evp_fix_${D}_${H}_${I}"
      JID=$(sbatch --parsable -p "$PART" -t "$WALL" -J "$NAME" \
                   slurm/submit_evp_fix_arm.sh "$D" "$H" "$I")
      echo "submitted ${NAME}  job=${JID}  part=${PART}  wall=${WALL}"
      i=$((i + 1))
    done
  done
done
echo
squeue -u "$USER" -n evp_fix_A_3lam_agn,evp_fix_A_3lam_br,evp_fix_A_7lam_agn,evp_fix_A_7lam_br,evp_fix_B_3lam_agn,evp_fix_B_3lam_br,evp_fix_B_7lam_agn,evp_fix_B_7lam_br \
  -o "%.10i %.10P %.22j %.2t %.10M %.12l %R"

#!/bin/bash
# Launch four seed replicas of evp_fix_A_3lam_agn (ensemble seeds 2..5).
# The existing run used theta_seed=0 / seed=0 and is seed 1 of 5.
set -eu
cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p jobs work/evp_channel

TARGETS=./reference_values/evp_targets_geomA_3lam.json
if [ ! -f "$TARGETS" ]; then
  echo "MISSING $TARGETS"; exit 1
fi

# Confirm original seed
ORIG_SEED=$(python3 -c "import json; print(json.load(open('work/evp_channel/evp_fix_A_3lam_agn/config.json'))['args']['theta_seed'])")
echo "original evp_fix_A_3lam_agn theta_seed=${ORIG_SEED}  (ensemble seed 1 of 5)"
echo "submitting ensemble seeds 2,3,4,5 with matching --theta-seed/--seed"
echo

WALL=1-00:00:00
i=0
for SEED in 2 3 4 5; do
  if [ $((i % 2)) -eq 0 ]; then PART=gpu; else PART=seas_gpu; fi
  NAME="evp_fix_A_3lam_agn_s${SEED}"
  JID=$(sbatch --parsable -p "$PART" -t "$WALL" -J "$NAME" \
               slurm/submit_evp_fix_seed.sh "$SEED")
  echo "submitted ${NAME}  job=${JID}  part=${PART}  wall=${WALL}  theta_seed=${SEED}"
  i=$((i + 1))
done
echo
squeue -u "$USER" -n evp_fix_A_3lam_agn_s2,evp_fix_A_3lam_agn_s3,evp_fix_A_3lam_agn_s4,evp_fix_A_3lam_agn_s5 \
  -o "%.10i %.10P %.28j %.2t %.10M %.12l %R"

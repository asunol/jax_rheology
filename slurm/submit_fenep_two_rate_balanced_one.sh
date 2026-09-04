#!/bin/bash
# Guarded single-run submit for fene8_bal_s*.
# Usage: ./submit_fenep_two_rate_balanced_one.sh <run_name> <seed>
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
RUN=${1:?run name}
SEED=${2:?seed}
# shellcheck source=slurm/fenep_submit_guard.sh
source slurm/fenep_submit_guard.sh
if [ -f "work/fenep_contraction/${RUN}/DONE" ]; then
  echo "[skip] $RUN already DONE"
  exit 0
fi
fenep_guard_refuse_if_queued "$RUN" || exit 3
jid=$(sbatch --parsable slurm/run_fenep_two_rate_balanced.sh "$RUN" "$SEED")
mkdir -p "work/fenep_contraction/${RUN}"
echo "$jid" > "work/fenep_contraction/${RUN}/SUBMITTED"
echo "[ok] $RUN -> $jid"
echo "$jid"

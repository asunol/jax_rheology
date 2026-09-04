#!/bin/bash
# Dispatcher for publication battery under test QOS MaxSubmitJobs=5.
# Keeps at most MAX_INFLIGHT individual jobs submitted; persists each fit
# independently. Safe to re-run (skips completed outputs).
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
OUT=work/bic_battery
mkdir -p "$OUT/logs" "$OUT/fits" "$OUT/targets" "$OUT/data" "$OUT/errors" "$OUT/dispatch"
MAX_INFLIGHT=${MAX_INFLIGHT:-5}
PY=${TBNN_PYDR:?set TBNN_PYDR to a Python from environment_diff_rheo.yml}
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True PYTHONDONTWRITEBYTECODE=1

TARGETS=(R1 R2 R3 R4 R5 R6 R7 T1 T2 T3 gie_A_s1 gie_A_s1b gie_A_s4 \
         clean_analytic_fene_p clean_analytic_giesekus)
PANEL=(Newtonian OldroydB Giesekus FENEPConformation LinearPTT)

QUEUE="$OUT/dispatch/queue.txt"
: > "$QUEUE"
for t in "${TARGETS[@]}"; do
  echo "prepare $t" >> "$QUEUE"
done
for t in "${TARGETS[@]}"; do
  for c in "${PANEL[@]}"; do
    for r in 0 1 2; do
      echo "fit $t $c $r" >> "$QUEUE"
    done
  done
done
for t in "${TARGETS[@]}"; do
  echo "merge $t" >> "$QUEUE"
done
echo "early" >> "$QUEUE"
echo "[dispatch] $(wc -l < "$QUEUE") work items; max_inflight=$MAX_INFLIGHT"

done_path() {
  local kind=$1
  case "$kind" in
    prepare) echo "$OUT/data/${2}.npz" ;;
    fit) echo "$OUT/fits/${2}/${3}_r${4}.json" ;;
    merge) echo "$OUT/targets/${2}.json" ;;
    early) echo "$OUT/interim_report_1.md" ;;
  esac
}

ready() {
  # prepare: always ready; fit needs data; merge needs 15 restart files; early needs all merges
  local kind=$1
  case "$kind" in
    prepare) return 0 ;;
    fit)
      [ -f "$OUT/data/${2}.npz" ] || return 1
      return 0
      ;;
    merge)
      local c r
      for c in "${PANEL[@]}"; do
        for r in 0 1 2; do
          [ -f "$OUT/fits/${2}/${c}_r${r}.json" ] || return 1
        done
      done
      return 0
      ;;
    early)
      local t
      for t in "${TARGETS[@]}"; do
        [ -f "$OUT/targets/${t}.json" ] || return 1
      done
      return 0
      ;;
  esac
}

submit_item() {
  local kind=$1; shift
  local tag name wrap
  tag=$(echo "${kind}_$*" | sha256sum | awk '{print substr($1,1,10)}')
  name="f8_${tag}"
  case "$kind" in
    prepare)
      wrap="export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True OMP_NUM_THREADS=16; \
$PY -u campaigns/battery/tbnn_bic_final_battery.py prepare --target $1"
      sbatch --parsable -J "$name" -p test -N 1 -n 1 -c 16 -t 0-04:00 --mem=48G \
        -o "$OUT/logs/%j_${name}.out" -e "$OUT/logs/%j_${name}.err" \
        --wrap="cd $PWD; $wrap"
      ;;
    fit)
      wrap="export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True OMP_NUM_THREADS=16; \
$PY -u campaigns/battery/tbnn_bic_final_battery.py fit --target $1 --candidate $2 --restart $3"
      sbatch --parsable -J "$name" -p test -N 1 -n 1 -c 16 -t 0-12:00 --mem=48G \
        -o "$OUT/logs/%j_${name}.out" -e "$OUT/logs/%j_${name}.err" \
        --wrap="cd $PWD; $wrap"
      ;;
    merge)
      wrap="export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True; \
$PY -u campaigns/battery/tbnn_bic_final_battery.py merge-target --target $1"
      sbatch --parsable -J "$name" -p test -N 1 -n 1 -c 4 -t 0-01:00 --mem=8G \
        -o "$OUT/logs/%j_${name}.out" -e "$OUT/logs/%j_${name}.err" \
        --wrap="cd $PWD; $wrap"
      ;;
    early)
      wrap="export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True; \
$PY -u campaigns/battery/battery_early_warning.py"
      sbatch --parsable -J "$name" -p test -N 1 -n 1 -c 2 -t 0-00:30 --mem=4G \
        -o "$OUT/logs/%j_${name}.out" -e "$OUT/logs/%j_${name}.err" \
        --wrap="cd $PWD; $wrap"
      ;;
  esac
}

STATE="$OUT/dispatch/state.log"
INFLIGHT_FILE="$OUT/dispatch/inflight.tsv"
touch "$INFLIGHT_FILE"
echo "[dispatch] start $(date -Is)" | tee -a "$STATE"

prune_inflight() {
  local tmp jid rest
  tmp=$(mktemp)
  while IFS=$'\t' read -r jid rest; do
    [ -z "${jid:-}" ] && continue
    if squeue -j "$jid" -h 2>/dev/null | grep -q .; then
      printf '%s\t%s\n' "$jid" "$rest" >> "$tmp"
    fi
  done < "$INFLIGHT_FILE"
  mv "$tmp" "$INFLIGHT_FILE"
}

while true; do
  prune_inflight
  inflight=$(wc -l < "$INFLIGHT_FILE")
  echo "[dispatch] $(date -Is) inflight=$inflight" | tee -a "$STATE"

  remaining=0
  submitted_this_round=0
  while IFS= read -r line; do
    set -- $line
    kind=$1; shift || true
    outp=$(done_path "$kind" "$@")
    if [ -f "$outp" ]; then
      continue
    fi
    remaining=$((remaining + 1))
    if ! ready "$kind" "$@"; then
      continue
    fi
    # already submitted and still queued/running?
    if awk -F'\t' -v key="$line" '$2 == key {found=1} END{exit !found}' "$INFLIGHT_FILE"; then
      continue
    fi
    if [ "$inflight" -ge "$MAX_INFLIGHT" ]; then
      break
    fi
    jid=$(submit_item "$kind" "$@") || { echo "[dispatch] submit failed for $line"; sleep 5; break; }
    printf '%s\t%s\n' "$jid" "$line" >> "$INFLIGHT_FILE"
    echo "[dispatch] submitted $jid :: $line" | tee -a "$STATE"
    inflight=$((inflight + 1))
    submitted_this_round=$((submitted_this_round + 1))
  done < "$QUEUE"

  if [ "$remaining" -eq 0 ]; then
    echo "[dispatch] ALL DONE $(date -Is)" | tee -a "$STATE"
    exit 0
  fi
  if [ "$submitted_this_round" -eq 0 ]; then
    sleep 60
  else
    sleep 20
  fi
done

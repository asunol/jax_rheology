#!/bin/bash
# Shared pre-submit guard for FENE8 run tags.
# Usage: source this file, then:
#   fenep_guard_refuse_if_queued <run_tag> || exit 3
# Refuses if any running/pending Slurm job's .out first-line or job name
# clearly belongs to this run tag (best-effort via squeue + SUBMITTED file
# + live outs mentioning run=<tag>).

fenep_guard_refuse_if_queued() {
  local tag="${1:?run tag}"
  local jid state out cmd self="${SLURM_JOB_ID:-}"
  # 1) Active job recorded in SUBMITTED still in queue (ignore ourselves)
  if [ -f "work/fenep_contraction/${tag}/SUBMITTED" ]; then
    jid=$(tr -d '[:space:]' < "work/fenep_contraction/${tag}/SUBMITTED")
    if [ -n "$jid" ] && [ "$jid" != "$self" ] \
       && squeue -j "$jid" -h 2>/dev/null | grep -q .; then
      echo "[guard] REFUSE submit for $tag: SUBMITTED job $jid still in queue" >&2
      return 1
    fi
  fi
  # 2) Any other queued job matching this run tag
  while read -r jid; do
    [ -z "$jid" ] && continue
    [ "$jid" = "$self" ] && continue
    out=$(ls -t "jobs/${jid}_"*.out 2>/dev/null | head -1)
    if [ -n "$out" ] && head -5 "$out" 2>/dev/null | grep -q "run=${tag}"; then
      state=$(squeue -j "$jid" -h -o '%T' 2>/dev/null || true)
      echo "[guard] REFUSE submit for $tag: job $jid ($state) matches run=${tag}" >&2
      return 1
    fi
    cmd=$(scontrol show job "$jid" 2>/dev/null | tr ' ' '\n' | grep '^Command=' | head -1 || true)
    if echo "$cmd" | grep -q -- "$tag"; then
      echo "[guard] REFUSE submit for $tag: job $jid Command mentions tag" >&2
      return 1
    fi
  done < <(squeue -u "${USER}" -h -o '%i' 2>/dev/null || true)
  echo "[guard] OK to submit $tag"
  return 0
}

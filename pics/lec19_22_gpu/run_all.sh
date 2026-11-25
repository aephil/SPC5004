#!/usr/bin/env bash
# Run every lec19-22 job in order. Each job is resumable: if anything stops, run this again and it carries on.
# A failed job does not stop the later ones; failures are listed at the end and in results/status.txt.
# Smoke test first (a few minutes, writes to results-quick/):   L19_QUICK=1 ./run_all.sh
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
R=results; [ -n "${L19_QUICK:-}" ] && R=results-quick
mkdir -p "$R"
: > "$R/status.txt"
for job in j00_probe j19_train j19_sample j19_measure j20_img2img j21_editing j22_control; do
  echo "=== $job  $(date '+%H:%M:%S')"
  start=$(date +%s)
  if python "$job.py" 2>&1 | tee "$R/$job.log"; then s=ok; else s=FAILED; fi
  echo "$job $s $(( $(date +%s) - start ))s" | tee -a "$R/status.txt"
done
python collect.py
echo; cat "$R/status.txt"

#!/usr/bin/env bash
# Run every lec18 job in order. Each step is resumable: if anything stops, just run this again.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
python probe_memory.py        # ~20-25 min: what fits, peak memory, seconds per step
python prepare_data.py        # ~10 min: download ~490 prints, caption them with BLIP
python train_and_sample.py    # the long one: 6,400 steps in 8 stages, sampling after each
python collect.py             # loss curve, memorisation check, zips results/ into lec18_results.zip

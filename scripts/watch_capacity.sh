#!/usr/bin/env bash
# Poll RunPod for 8-GPU capacity and record when it appears. Read-only: it calls
# `avail` (free) and never rents. `avail` exits 0 when some host reports stock.
#
#   scripts/watch_capacity.sh out/capacity.log 300
#
# On the first hit for a GPU type it appends a HIT line and touches
# <log>.hit-<slug>, so a later session can tell "capacity appeared at 03:12" from
# "capacity is there now" without re-reading the whole log.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=${1:-out/capacity.log}
INTERVAL=${2:-300}
mkdir -p "$(dirname "$LOG")"

# The A100 is the runbook's box and keeps wall-clock comparability with the DDP
# anchor; the H100 is a fallback at roughly twice the price. The 4090 pool is
# deliberately absent -- it has never pulled our image (session 2026-08-18).
GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA H100 80GB HBM3")

while true; do
    for gpu in "${GPUS[@]}"; do
        slug=$(echo "$gpu" | tr ' ' '-')
        out=$(uv run --script scripts/runpod_session.py --gpu-type "$gpu" --gpu-count 8 avail 2>&1)
        rc=$?
        ts=$(date -Is)
        if [ $rc -eq 0 ]; then
            echo "$ts HIT $gpu" >> "$LOG"
            echo "$out" | sed 's/^/    /' >> "$LOG"
            [ -e "$LOG.hit-$slug" ] || { echo "$ts" > "$LOG.hit-$slug"; }
        else
            echo "$ts none $gpu" >> "$LOG"
        fi
    done
    sleep "$INTERVAL"
done

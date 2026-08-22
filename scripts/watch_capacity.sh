#!/usr/bin/env bash
# Poll RunPod and Prime Intellect for 8-GPU capacity and record when it appears.
# Read-only: it calls `avail` (free) and never rents. `avail` exits 0 when some
# host reports stock, on both venues.
#
#   scripts/watch_capacity.sh out/capacity.log 300
#
# WATCH_GPUS, WATCH_PRIME_GPUS and WATCH_PRIME_SOCKET override what is polled, so
# the same script can hunt the PCIe shape the transport study still needs
# (decisions.md §22). RunPod sells the PCIe card as its own GPU type; Prime sells a
# socket field on the same enum, so both memory sizes are worth polling there:
#
#   WATCH_GPUS="NVIDIA A100 80GB PCIe" WATCH_PRIME_SOCKET=PCIe \
#   WATCH_PRIME_GPUS=$'A100_80GB\nA100_40GB' \
#       scripts/watch_capacity.sh out/capacity-pcie.log 300
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
# WATCH_GPUS is newline-separated so GPU names can contain spaces.
if [ -n "${WATCH_GPUS:-}" ]; then
    mapfile -t GPUS <<< "$WATCH_GPUS"
fi
PRIME_SOCKET=${WATCH_PRIME_SOCKET:-SXM4}
# Prime Intellect GPU enums to poll, newline-separated. A PCIe hunt wants both
# memory sizes: the socket is what is scarce, not the capacity of the card.
PRIME_GPUS=("A100_80GB")
if [ -n "${WATCH_PRIME_GPUS:-}" ]; then
    mapfile -t PRIME_GPUS <<< "$WATCH_PRIME_GPUS"
fi

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

    # Prime Intellect: one shape at a time (8x A100_80GB, §20), socket pinned so a
    # PCIe hunt cannot be satisfied by SXM4 stock. Skipped rather than logged as a
    # miss when no key is set, so the log stays honest.
    if [ -n "${PRIME_API_KEY:-}" ]; then
        for pgpu in "${PRIME_GPUS[@]}"; do
            out=$(uv run --script scripts/prime_session.py \
                --gpu-type "$pgpu" --socket "$PRIME_SOCKET" avail 2>&1)
            rc=$?
            ts=$(date -Is)
            if [ $rc -eq 0 ]; then
                echo "$ts HIT prime $pgpu $PRIME_SOCKET" >> "$LOG"
                echo "$out" | sed 's/^/    /' >> "$LOG"
                [ -e "$LOG.hit-prime-$pgpu" ] || { echo "$ts" > "$LOG.hit-prime-$pgpu"; }
            else
                echo "$ts none prime $pgpu $PRIME_SOCKET" >> "$LOG"
            fi
        done
    fi

    sleep "$INTERVAL"
done

#!/usr/bin/env bash
# Rent the first 8xA100-PCIe that appears, measure it, tear it down -- unattended.
#
#   scripts/pcie_hunt.sh out/pcie-hunt.log 300
#
# The PCIe point is the transport study's largest hole and the SKU is almost never
# in stock, so waiting for a human to notice an opening is how the opening gets
# missed. RunPod's `avail` precheck has false negatives, so the deploy call itself
# is the capacity probe: a *rejected* deploy creates nothing and costs nothing.
#
# SECURE only. The community tier has never run this image -- 8x 4090 (2026-08-18)
# and 2x A100 PCIe (2026-08-22, container created and never started, $1.26) -- and
# it is also the only tier whose hosts carry no public IP.
#
# Cost safety, in order: a wall-clock ceiling on `up`, `guard` started in the same
# breath as the pod, an EXIT trap that terminates whatever exists, and `verify` as
# the last word. The one case that is not free is a deploy that *succeeds* and
# then fails to boot: `up` terminates it, but the host is bad and retrying it
# immediately would burn the ceiling in a loop, so that path backs off hard.
#
# Prime Intellect is deliberately absent: its PCIe-labelled 8xA100 is cloudId
# `gpu_8x_a100`, lambdalabs' SXM4 box under a PCIe label (§22, re-confirmed
# 2026-08-22), so renting it would buy the same wrong fabric a second time.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=${1:-out/pcie-hunt.log}
INTERVAL=${2:-300}
BAD_HOST_BACKOFF=${BAD_HOST_BACKOFF:-1800}
MAX_HOURS=${MAX_HOURS:-1}
IMAGE=${IMAGE:-ghcr.io/adamdivak/distrain:c8c72e1}
NAME=${NAME:-distrain-pcie8}
GPU=${GPU:-NVIDIA A100 80GB PCIe}
ARTIFACTS=${ARTIFACTS:-out/runpod-pcie}
mkdir -p "$(dirname "$LOG")" "$ARTIFACTS"

[ -f .env ] && { set -a; . ./.env; set +a; }
SESSION=out/runpod/session-pcie8.json
RP=(uv run --script scripts/runpod_session.py --gpu-type "$GPU" --gpu-count 8
    --name "$NAME" --session-file "$SESSION")

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

teardown() {
    log "teardown: terminating anything under $NAME"
    # --yes: `down` prompts for the pod id otherwise, and an unattended prompt
    # reads EOF, fails, and leaves the pod billing to the ceiling.
    "${RP[@]}" down --yes >> "$LOG" 2>&1
    if "${RP[@]}" verify >> "$LOG" 2>&1; then log "verify: CLEAN"
    else log "verify: SOMETHING IS STILL BILLING"; fi
}

while true; do
    log "attempting SECURE"
    ATTEMPT=$(mktemp)
    if "${RP[@]}" --skip-capacity-check --cloud-type SECURE up \
            --max-hours "$MAX_HOURS" --volume-gb 0 --container-disk-gb 200 \
            --image "$IMAGE" > "$ATTEMPT" 2>&1; then
        cat "$ATTEMPT" >> "$LOG"
        log "GOT ONE -- pod is billing"
        trap teardown EXIT INT TERM

        # No --max-hours: the ceiling `up` stamped is the one to enforce.
        "${RP[@]}" guard --terminate-on-low-balance >> out/pcie-hunt-guard.log 2>&1 &
        GUARD=$!
        log "guard started (pid $GUARD)"

        IP=$(python3 -c "import json;print(json.load(open('$SESSION'))['ip'])")
        PORT=$(python3 -c "import json;print(json.load(open('$SESSION'))['port'])")
        SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=accept-new
             -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 "root@$IP")
        log "pod at $IP:$PORT"

        "${SSH[@]}" 'NPROC=8 bash -s' < scripts/pcie_measure.sh 2>&1 \
            | tee "$ARTIFACTS/session.log"
        RC=${PIPESTATUS[0]}
        log "measurement exited $RC"

        rsync -avz -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null" \
            "root@$IP:/workspace/session_out/" "$ARTIFACTS/session_out/" >> "$LOG" 2>&1
        log "artifacts pulled to $ARTIFACTS/session_out"

        kill "$GUARD" 2>/dev/null
        trap - EXIT INT TERM
        teardown
        [ "$RC" -eq 2 ] && log "TOPOLOGY GATE FAILED -- the PCIe SKU was not a PCIe fabric."
        rm -f "$ATTEMPT"
        exit "$RC"
    fi

    # A deploy that created a pod and then failed to boot has already cost money
    # and will cost it again on the same host. Anything else is a free miss.
    if grep -q "pod: created" "$ATTEMPT"; then
        cat "$ATTEMPT" >> "$LOG"
        log "BAD HOST: a pod was created but never came up. up terminated it."
        teardown
        log "backing off ${BAD_HOST_BACKOFF}s so a broken host cannot be retried in a loop"
        rm -f "$ATTEMPT"
        sleep "$BAD_HOST_BACKOFF"
        continue
    fi
    log "no SECURE capacity"
    rm -f "$ATTEMPT"
    sleep "$INTERVAL"
done

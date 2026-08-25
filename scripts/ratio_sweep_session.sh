#!/usr/bin/env bash
# Rent one small box, run scripts/ratio_sweep.sh on it, tear it down.
#
#   scripts/ratio_sweep_session.sh
#   GPU="NVIDIA A100-SXM4-80GB" TCP_SWEEP=1 scripts/ratio_sweep_session.sh
#
# One attempt, not a hunt: this measurement is wanted now and the SKU is small.
# A rejected deploy creates nothing and costs nothing, so the deploy call is also
# the capacity probe -- RunPod's `avail` precheck has false negatives (§24).
#
# Cost safety, in the order it takes effect (decisions.md §9):
#   1. a wall-clock ceiling stamped by `up`
#   2. `guard --terminate-on-low-balance` started in the same breath as the pod
#   3. an EXIT/INT/TERM trap that terminates whatever exists
#   4. `verify` as the last word, which exits non-zero if anything still bills
set -uo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-NVIDIA A100 80GB PCIe}
GPU_COUNT=${GPU_COUNT:-2}
CLOUD=${CLOUD:-SECURE}
MAX_HOURS=${MAX_HOURS:-1}
IMAGE=${IMAGE:-ghcr.io/adamdivak/distrain:c8c72e1}
NAME=${NAME:-distrain-ratio}
ARTIFACTS=${ARTIFACTS:-out/ratio-sweep}
LOG=${LOG:-out/ratio-sweep.log}
MICROS=${MICROS:-"8 16 30 60"}
TCP_SWEEP=${TCP_SWEEP:-0}
TCP_MICROS=${TCP_MICROS:-"8 60"}

mkdir -p "$(dirname "$LOG")" "$ARTIFACTS"
[ -f .env ] && { set -a; . ./.env; set +a; }
SESSION=out/runpod/session-ratio.json
RP=(uv run --script scripts/runpod_session.py --gpu-type "$GPU" --gpu-count "$GPU_COUNT"
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

log "attempting $CLOUD ${GPU_COUNT}x $GPU (ceiling ${MAX_HOURS}h)"
ATTEMPT=$(mktemp)
if ! "${RP[@]}" --skip-capacity-check --cloud-type "$CLOUD" up \
        --max-hours "$MAX_HOURS" --volume-gb 0 --container-disk-gb 200 \
        --image "$IMAGE" > "$ATTEMPT" 2>&1; then
    cat "$ATTEMPT" | tee -a "$LOG"
    if grep -q "pod: created" "$ATTEMPT"; then
        log "BAD HOST: a pod was created but never came up; up terminated it."
        teardown
    else
        log "no $CLOUD capacity for ${GPU_COUNT}x $GPU -- nothing was created, nothing billed."
    fi
    rm -f "$ATTEMPT"
    exit 1
fi
cat "$ATTEMPT" >> "$LOG"
rm -f "$ATTEMPT"
log "pod is billing"
trap teardown EXIT INT TERM

# No --max-hours here: the ceiling `up` stamped is the one to enforce.
"${RP[@]}" guard --terminate-on-low-balance >> out/ratio-sweep-guard.log 2>&1 &
GUARD=$!
log "guard started (pid $GUARD)"

IP=$(python3 -c "import json;print(json.load(open('$SESSION'))['ip'])")
PORT=$(python3 -c "import json;print(json.load(open('$SESSION'))['port'])")
SSHOPTS=(-p "$PORT" -o StrictHostKeyChecking=accept-new
         -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30)
log "pod at $IP:$PORT"

ssh "${SSHOPTS[@]}" "root@$IP" \
    "NPROC=$GPU_COUNT MICROS='$MICROS' TCP_SWEEP=$TCP_SWEEP TCP_MICROS='$TCP_MICROS' bash -s" \
    < scripts/ratio_sweep.sh 2>&1 | tee "$ARTIFACTS/session.log"
RC=${PIPESTATUS[0]}
log "sweep exited $RC"

rsync -avz -e "ssh ${SSHOPTS[*]}" \
    "root@$IP:/workspace/session_out/" "$ARTIFACTS/session_out/" >> "$LOG" 2>&1
log "artifacts pulled to $ARTIFACTS/session_out"

kill "$GUARD" 2>/dev/null
trap - EXIT INT TERM
teardown
exit "$RC"

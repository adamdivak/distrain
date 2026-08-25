#!/usr/bin/env bash
# The DDP-mode matrix as a function of the compute-to-communication ratio.
# Runs ON a rented pod:
#
#   ssh <pod> 'NPROC=2 bash -s' < scripts/ratio_sweep.sh | tee out/<session>/session.log
#
# Why this exists (decisions.md §26): the four-way mode comparison was only ever
# measured at one operating point, 8 sequences per rank, where the collective is
# 96% of the step and no implementation can differ by more than the 49.5 ms of
# compute. That is not a comparison, it is a floor. The ratio that decides which
# mode wins is
#
#     compute / comm  ∝  tokens per rank per backward
#
# and the parameter count cancels out of it -- compute and gradient bytes both
# scale linearly in N -- so the knob is the micro-batch, not the model, and not
# gradient accumulation (which raises compute per optimizer step but not the
# overlap window, because the reduction only starts on the last micro-step).
#
# Every arm therefore runs accumulation 1 with global batch = MICRO x NPROC:
# micro-batch is the only thing that moves. Absolute step times across arms are
# not comparable -- different batches -- but the *ranking of the four modes at
# each ratio* is exactly the question.
set -uo pipefail
cd /workspace
OUT=/workspace/session_out
mkdir -p "$OUT"
N=${NPROC:-2}
MICROS=${MICROS:-"8 16 30 60"}
MODES=${MODES:-"ddp_naive ddp_bucketed ddp_interleaved ddp_torch"}
STEPS=${STEPS:-16}
WARMUP=${WARMUP:-6}
# Set to 1 to repeat the sweep with NCCL forced onto TCP sockets.
TCP_SWEEP=${TCP_SWEEP:-0}
TCP_MICROS=${TCP_MICROS:-"8 60"}

echo "=== 0. identity ($N ranks, micro-batches: $MICROS) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$OUT/gpus.txt"
python -c "import torch; print('torch', torch.__version__)" | tee -a "$OUT/gpus.txt"

echo "=== 1. topology (recorded, not gated: this sweep is about the ratio) ==="
nvidia-smi topo -m | tee "$OUT/topo.txt"
nvidia-smi topo -p2p r 2>&1 | tee "$OUT/topo_p2p.txt"

echo "=== 2. roofline (the MFU denominator for this card) ==="
python scripts/measure_roofline.py 2>&1 | tee "$OUT/roofline.txt"

echo "=== 3. effective all-reduce bandwidth ==="
if [ -x /opt/nccl-tests/build/all_reduce_perf ]; then
    /opt/nccl-tests/build/all_reduce_perf -b 8M -e 512M -f 2 -g "$N" 2>&1 | tee "$OUT/nccl_tests.txt"
else
    # allreduce_bw.py is torchrun-launched, not rank-argument driven.
    python -m torch.distributed.run --standalone --nproc_per_node="$N" \
        scripts/allreduce_bw.py 2>&1 | tee "$OUT/nccl_tests.txt"
fi

echo "=== 4. data (2 chunks; step-time measurements, not a converged run) ==="
mkdir -p /workspace/data/fineweb10B
ln -sfn /workspace/data/fineweb10B /workspace/reference/modded_nanogpt/fineweb10B
python reference/modded_nanogpt/cached_fineweb10B.py 2 > "$OUT/data_download.log" 2>&1
ls -la /workspace/data/fineweb10B | tail -3

# Quoted: an unquoted glob is expanded by the pod's shell and argparse rejects
# the second shard as a stray positional (2026-08-21, ~$1 and a relaunch).
GLOBS='-- --train-glob "data/fineweb10B/fineweb_train_*.bin" --val-glob "data/fineweb10B/fineweb_val_*.bin"'

sweep() {  # $1 = label for the output directory
    local tag=$1
    for micro in $MICROS; do
        echo "=== 5.$tag micro-batch $micro ($((micro * N)) global, accum 1) ==="
        # bench_ddp_modes.py defaults global batch to per-gpu x nproc with
        # accumulation 1, which is what this sweep wants; only the data globs
        # are forwarded.
        eval python scripts/bench_ddp_modes.py \
            --nproc "$N" --no-single --steps "$STEPS" --warmup "$WARMUP" \
            --per-gpu-batch "$micro" --timeout 1800 \
            --modes $MODES \
            --out-dir "session_out/sweep-$tag-m$micro" "$GLOBS" 2>&1 \
            | tee "$OUT/sweep-$tag-m$micro.log"
    done
}

echo "=== 5. native fabric sweep ==="
sweep native

if [ "$TCP_SWEEP" = "1" ]; then
    echo "=== 6. forced TCP/loopback sweep (P2P and SHM disabled) ==="
    # The same control §22 used: verify by effect, then measure. NCCL prints the
    # transport it chose, so the log is the evidence the sweep was throttled.
    export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
    MICROS=$TCP_MICROS
    sweep tcp
    unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE
fi

echo "=== done: $(ls "$OUT" | wc -l) artifacts in $OUT ==="

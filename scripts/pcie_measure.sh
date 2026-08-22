#!/usr/bin/env bash
# The PCIe transport point, start to finish, run ON the rented pod.
#
#   ssh <pod> 'NPROC=8 bash -s' < scripts/pcie_measure.sh | tee out/runpod-pcie/session.log
#
# Everything lands in /workspace/session_out. The first thing it does is the
# topology gate: a provider's "PCIe" label is metadata, not a fabric guarantee
# (decisions.md §22 -- a PCIe-labelled offer once delivered a full NV12 mesh), so
# any NV# link between GPUs aborts before a single number is produced.
#
# NPROC is the rank count on the box. Every arm holds global batch 480; the
# micro-batch is 30 sequences per device, which is what transport.csv's NVLink and
# forced-TCP rows were measured at, so only the rank count and the fabric differ
# from them. The accumulation depth absorbs the rest. An anchor-matched 60 arm
# follows, for the micro-batch the converged run used.
set -uo pipefail
cd /workspace
OUT=/workspace/session_out
mkdir -p "$OUT"
N=${NPROC:-8}
MICRO=${MICRO:-30}
ACCUM=$(( 480 / (MICRO * N) ))
ACCUM60=$(( 480 / (60 * N) ))

echo "=== 0. identity ($N ranks, $MICRO seqs/device x $ACCUM accum = global batch 480) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$OUT/gpus.txt"
nproc

echo "=== 1. topology gate ==="
nvidia-smi topo -m | tee "$OUT/topo.txt"
# Only the GPU-to-GPU block matters: the legend at the bottom names NV# too.
if grep -E '^GPU[0-9]' "$OUT/topo.txt" | grep -qE '\bNV[0-9]+\b'; then
    echo "GATE FAILED: NVLink (NV#) links present -- this is not a PCIe fabric."
    echo "No number from this box may be published as PCIe. Release it."
    exit 2
fi
echo "GATE PASSED: no NV# links; GPU-to-GPU traffic crosses PCIe."

echo "=== 2. roofline (the MFU denominator) ==="
python scripts/measure_roofline.py 2>&1 | tee "$OUT/roofline.txt"

echo "=== 3. effective all-reduce bandwidth ==="
# Same invocation as the NVLink anchor, so busbw is comparable point to point.
/opt/nccl-tests/build/all_reduce_perf -b 8M -e 512M -f 2 -g "$N" 2>&1 | tee "$OUT/nccl_tests.txt"

echo "=== 4. data (2 chunks; these are step-time measurements, not a converged run) ==="
mkdir -p /workspace/data/fineweb10B
ln -sfn /workspace/data/fineweb10B /workspace/reference/modded_nanogpt/fineweb10B
python reference/modded_nanogpt/cached_fineweb10B.py 2 > "$OUT/data_download.log" 2>&1
ls -la /workspace/data/fineweb10B | tail -4

TG='data/fineweb10B/fineweb_train_*.bin'
VG='data/fineweb10B/fineweb_val_*.bin'
# Quoted: unquoted globs are expanded by the pod's shell and argparse rejects the
# second shard as a stray positional (2026-08-21, ~$1 and a relaunch).
GLOBS="--train-glob \"$TG\" --val-glob \"$VG\""
BENCH="python scripts/bench_ddp_modes.py --nproc $N --no-single --steps 25 --warmup 10 \
    --per-gpu-batch $MICRO --timeout 1800"
TRAIN="-- --global-batch-seqs 480 --grad-accum-steps $ACCUM $GLOBS"

echo "=== 5a. headline: $N ranks, global batch 480, compiled ddp_torch ==="
eval "$BENCH" --modes ddp_torch --out-dir "$OUT/bench-b480" "$TRAIN" 2>&1 | tail -30

echo "=== 5b. the middle regime: does overlap beat compilation here? ==="
# §21 found the compile-vs-overlap lead dissolves as bandwidth falls. PCIe at
# batch 480 is a real instance of the regime where compute and comm are
# comparable, so this tests that claim on a fabric instead of on netem.
eval "$BENCH" --modes ddp_interleaved --out-dir "$OUT/bench-b480-il" "$TRAIN" 2>&1 | tail -30
eval "$BENCH" --modes ddp_interleaved ddp_torch --no-compile \
    --out-dir "$OUT/bench-b480-nc" "$TRAIN" 2>&1 | tail -30

echo "=== 5c. anchor-matched chunking: 60 seqs/device (what the converged run used) ==="
eval python scripts/bench_ddp_modes.py --nproc "$N" --no-single --modes ddp_torch \
    --steps 25 --warmup 10 --per-gpu-batch 60 --timeout 1800 \
    --out-dir "$OUT/bench-b480-micro60" \
    -- --global-batch-seqs 480 --grad-accum-steps "$ACCUM60" "$GLOBS" 2>&1 | tail -30

echo "=== 5d. same-box single-GPU baseline (self-contained scaling ratio) ==="
eval python scripts/bench_ddp_modes.py --nproc 1 --no-single --modes ddp_torch \
    --steps 15 --warmup 5 --per-gpu-batch "$MICRO" --timeout 1800 \
    --out-dir "$OUT/bench-1gpu-b480" \
    -- --global-batch-seqs 480 --grad-accum-steps $(( 480 / MICRO )) "$GLOBS" 2>&1 | tail -30

echo "=== 6. transport confirmation (what NCCL actually used) ==="
grep -hiE "via|transport|NVLS|Socket|P2P|Ring" "$OUT"/bench-b480/*.log 2>/dev/null \
    | sort -u | head -30 | tee "$OUT/nccl_transport.txt"

echo "=== done; everything is under $OUT ==="
ls -R "$OUT" | head -60

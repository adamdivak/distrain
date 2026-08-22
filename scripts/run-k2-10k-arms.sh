#!/usr/bin/env bash
# The matched-schedule K=2 arms, so the DiLoCo merge penalty can be compared
# against K=8 without the schedule confound (decisions.md §23 measured K=2 on a
# 6000-step trapezoid; §21 measured K=8 on a 10000-step one, and no normalized
# x axis makes those two comparable).
#
# Both arms run 10000 steps on aurora's 3090, sequentially -- one GPU.
#
#   nohup scripts/run-k2-10k-arms.sh > out/k2-10k-runner.log 2>&1 &
#
# Why the DiLoCo arm can be *resumed* rather than rerun: lr_at() holds the LR
# constant from warmup_steps to max_steps-warmdown_steps, so the plateau is
# 250..5000 under the 6000-step schedule and 250..9000 under the 10000-step one.
# The permanent keep at step 4000 sits inside both, so re-entering there with
# --max-steps 10000 yields a genuine unbroken 10000-step trapezoid instead of a
# restart-after-warmdown artifact. This is the case train.py's
# checkpoint_keep_every comment was written for. Everything else matches the
# original arm A config byte for byte (recovered from the checkpoint's cfg).
#
# The reference has no such usable checkpoint -- checkpoints/ckpt.pt is
# rotary-calibration-3B at next_step=8500 on a *9000*-step schedule, already
# past a warmdown restart -- so it runs from scratch.
#
# val_every must be a multiple of outer_sync_every (train.py enforces it), so
# both arms validate every 500 rather than the reference running a denser grid;
# that also removes §23's mismatched-val-grid caveat.
set -euo pipefail
cd "$(dirname "$0")/.."

# Both are load-bearing on this box, and both were paid for once already
# (session 2026-08-19 §3): the diag eval runs on *every* rank, so two ranks
# sharing one 3090 allocate the logits tensor twice, and the allocator
# fragments badly enough that the reported val's contiguous request fails even
# when the free total is sufficient.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

UV=$(command -v uv || echo "$HOME/.local/bin/uv")
DATA=(--train-glob 'data/fineweb10B/fineweb_train_*.bin'
      --val-glob   'data/fineweb10B/fineweb_val_*.bin')
# Identical to arm A except max-steps; see the checkpoint cfg dump in the session log.
COMMON=(--global-batch-seqs 480 --grad-accum-steps 60 --seq-len 1024
        --learning-rate 0.0018 --max-steps 10000
        --warmup-steps 250 --warmdown-steps 1000
        --val-every 500 --eval-batch-seqs 32 --val-tokens 10485760
        --checkpoint-every 500 --checkpoint-keep-every 2000
        --compile)

echo "=== $(date -Is) arm 1/2: K=2 DiLoCo, resumed from step 4000 -> 10000 ==="
PYTHONUNBUFFERED=1 "$UV" run torchrun --nproc_per_node=2 -m distrain.train \
    "${DATA[@]}" "${COMMON[@]}" \
    --device cuda:0 --distributed-mode diloco --distributed-backend gloo \
    --outer-sync-every 500 --outer-lr 0.7 --outer-moment 0.5 \
    --diag-val-every 500 --diag-eval-batch-seqs 8 \
    --resume-from checkpoints/diloco-b480-mom05-step004000-rank0.pt \
    --run-name diloco-k2-10k \
    > out/diloco-k2-10k.log 2>&1
echo "=== $(date -Is) arm 1/2 done ==="

echo "=== $(date -Is) arm 2/2: single-GPU reference, 10000 steps from scratch ==="
PYTHONUNBUFFERED=1 "$UV" run python -m distrain.train \
    "${DATA[@]}" "${COMMON[@]}" \
    --run-name ref-1gpu-10k \
    > out/ref-1gpu-10k.log 2>&1
echo "=== $(date -Is) arm 2/2 done ==="

# Runbook — first 2-GPU session (RunPod community cloud)

Purpose, in priority order: **(1)** first NCCL contact with the hand-rolled DDP
layer, **(2)** time the three modes against each other
(`scripts/bench_ddp_modes.py`), **(3)** measured roofline for the GPU class.
Community GeForce numbers are not reported results (`project_brief.md` §8) — this
session exists so the cluster session starts with NCCL-proven code.

**Ceilings: 3 h wall clock, $5.** Attended session; set a timer at provision time.
If something fights back for more than ~15 min, record the state, terminate, and
regroup on aurora — debugging on rented hardware is the one forbidden move.

Estimated happy path: ~45–60 min, ≲ $1.50.

## 0. Before renting (free, on aurora)

- [ ] `uv run pytest -q` green, work committed.
- [ ] aurora's public key added in the RunPod console → Settings → SSH Public Keys
      (`cat ~/.ssh/id_ed25519.pub`), so the session can be driven from aurora.

## 1. Provision (console, ~5 min)

- Community Cloud → **2× RTX 4090** on one host (fallback: 2× RTX 3090; either is
  fine — the interconnect is PCIe/host either way, which is the *point*: comm cost
  is visible). ~$0.7–1.4/hr for the pair.
- Template: any RunPod PyTorch/CUDA ≥ 12.6 image — the real env comes from `uv`
  + `uv.lock`, the template only supplies the driver stack and sshd.
  (True image parity via a registry push of our pinned image is deferred to the
  cluster session; `uv.lock` already pins every wheel bit-for-bit.)
- Container/volume disk: 30 GB. Deploy, then copy the **SSH over exposed TCP**
  line — `ssh root@<IP> -p <PORT>` — everything below uses `<IP>`/`<PORT>`.

## 2. Push code (from aurora — the pod cannot reach aurora, so aurora pushes)

```bash
rsync -avz -e "ssh -p <PORT>" \
  --exclude data/ --exclude .venv/ --exclude out/ --exclude checkpoints/ \
  --exclude .git/ --exclude trackio/ \
  ~/work/distrain/ root@<IP>:/workspace/distrain/
```

## 3. Pod setup (~5 min, on the pod)

```bash
nvidia-smi                      # 2 GPUs visible, note driver version
nvidia-smi topo -m | tee /workspace/distrain/topo.txt   # expect PHB/PIX, no NVLink
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
cd /workspace/distrain && uv sync
uv run python scripts/make_synthetic_shards.py --out data/synthetic --shards 2
```

Synthetic data only — timing needs no FineWeb, and 19 GiB has no business here.

## 4. NCCL smoke — the moment the biggest "unproven" dies (~5 min)

```bash
cd /workspace/distrain
NCCL_DEBUG=INFO uv run torchrun --standalone --nproc_per_node=2 -m distrain.train \
  --distributed-mode ddp_naive --max-steps 5 --global-batch-seqs 16 --no-trackio
```

- `NCCL_DEBUG=INFO` shows the transport chosen (expect SHM/net, P2P disabled on
  GeForce — record the line).
- Repeat with `ddp_bucketed` and `ddp_interleaved`.
- **Cross-check losses**: all three modes, plus `--nproc_per_node=1` with the same
  `--global-batch-seqs 16`, should print near-identical per-step losses (same
  seed, same data, same global batch; bf16/atomics wiggle ~1e-3 is fine, anything
  larger is a real divergence — stop and record).
- If anything hangs: `TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_NCCL_DESYNC_DEBUG=1`
  (decisions.md §6), one retry, then terminate — do not debug here.

## 5. Roofline for this GPU class (~3 min)

```bash
uv run python scripts/measure_roofline.py | tee /workspace/distrain/roofline.txt
```

Record the measured bf16 TFLOP/s into `_PEAK_BF16` (with provenance) back on
aurora — datasheet 4090 values stay `UNVERIFIED` until this ran.

## 6. The measurement (~15 min)

```bash
uv run python scripts/bench_ddp_modes.py --nproc 2 --steps 50 --warmup 10
```

Optional, if time is comfortable — bucket-size sensitivity:

```bash
uv run python scripts/bench_ddp_modes.py --nproc 2 --steps 50 --warmup 10 \
  --no-single --modes ddp_bucketed ddp_interleaved -- --ddp-bucket-size 1048576
uv run python scripts/bench_ddp_modes.py --nproc 2 --steps 50 --warmup 10 \
  --no-single --modes ddp_bucketed ddp_interleaved -- --ddp-bucket-size 104857600
```

Expected shape: naive < bucketed ≤ interleaved on throughput, with the gap wide
on PCIe GeForce. Interleaved ≈ bucketed would mean overlap is not buying anything
on this transport — that is a finding, not a failure.

## 7. Optional: `all_reduce_perf` bandwidth ceiling (timebox 10 min)

```bash
cd /workspace && git clone https://github.com/NVIDIA/nccl-tests && cd nccl-tests
make -j              # needs the template's CUDA toolkit; skip on any friction
./build/all_reduce_perf -b 8M -e 256M -f 2 -g 2 | tee /workspace/distrain/nccl_tests.txt
```

## 8. Pull everything off before teardown (from aurora)

```bash
rsync -avz -e "ssh -p <PORT>" \
  root@<IP>:/workspace/distrain/out/bench/ ~/work/distrain/out/bench-runpod/
rsync -avz -e "ssh -p <PORT>" \
  "root@<IP>:/workspace/distrain/*.txt" ~/work/distrain/out/bench-runpod/
```

Nothing on the pod survives termination. Pull first, verify locally, then tear down.

## 9. Teardown

Console: **Stop** the pod, then **Terminate** it — a stopped pod still bills for
storage. Confirm the billing page shows no running spend before closing the tab.

## Afterwards (aurora, free)

- Record the 4090 (or 3090-cloud) roofline in `mfu.py`.
- Commit results + a short findings note; update README known gaps (NCCL proven,
  modes timed).
- Feed what the session showed (transport, bandwidth, mode gaps) into the plan
  for the cluster session: 8-GPU Track A matrix, netem curve, image-parity via
  registry push if wanted.

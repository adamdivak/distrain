# Runbook — first 8-GPU session (RunPod, registry image)

Purpose, in priority order: **(1)** image parity — the pod boots our pinned
image from GHCR, closing the README's known gap; **(2)** measured roofline +
`nccl-tests` for the GPU class; **(3)** the mode matrix at 8 ranks —
`bench_ddp_modes`, four modes × compile on/off; **(4)** the **first converged
Track A run**: 9000 steps / 4.42B tokens, whose first unsmoothed 3.28 crossing
is the tokens-to-target measurement (decisions §13 — the local calibration
bracketed it at 2.95–4.4B but never observed it).

**Ceilings: 8 h wall clock, $80.** Attended session; set a timer at provision
time. If something fights back for more than ~15 min, record the state,
terminate, and regroup on aurora — debugging on rented hardware is the one
forbidden move. Estimated happy path: ~3.5 h, **$30–55** (decisions §13 budget
arithmetic).

## 0. Before renting (free, on aurora)

- [ ] `uv run pytest -q` green, work committed.
- [ ] Image built **from the committed tree** and pushed:

  ```bash
  # one-time: token with package scopes, then registry login
  gh auth refresh -h github.com -s write:packages,read:packages
  gh auth token | docker login ghcr.io -u adamdivak --password-stdin

  SHA=$(git rev-parse --short HEAD)
  IMAGE=ghcr.io/adamdivak/distrain:$SHA scripts/container.sh build
  docker tag  ghcr.io/adamdivak/distrain:$SHA ghcr.io/adamdivak/distrain:latest
  docker push ghcr.io/adamdivak/distrain:$SHA
  docker push ghcr.io/adamdivak/distrain:latest
  ```

  Record `$SHA` — it is the provenance of every number from the session.
- [ ] Quick boot rehearsal passed (already scripted; rerun after any image change):
  `docker run --rm -d -e PUBLIC_KEY="$(cat ~/.ssh/id_ed25519.pub)" -p 2299:22
  ghcr.io/adamdivak/distrain:$SHA /workspace/scripts/pod-entry.sh`, then
  `ssh -p 2299 root@localhost 'python -c "import distrain.model"'`.
- [ ] RunPod console, one-time: **Settings → SSH Public Keys** has aurora's key;
  **Settings → Container Registries** has a GHCR credential — username
  `adamdivak`, password = a GitHub PAT with **`read:packages` only** (the
  package is private because the repo is; RunPod needs its own pull
  credential, not the login above).

## 1. Provision (console, ~5 min)

- Secure Cloud → **8× A100 80 GB** on one host (SXM preferred — NVLink is the
  interesting contrast with the 2×3090 PCIe numbers). Fallback: 8× A100 PCIe.
  H100 only if neither exists (~3× the price for the same measurements).
- **Custom image**: `ghcr.io/adamdivak/distrain:<SHA>`, with the GHCR
  credential selected.
- **Container Start Command**: `/workspace/scripts/pod-entry.sh`
  (starts sshd from the injected `PUBLIC_KEY`, idles; without it the image's
  default CMD prints a GPU report and exits — the pod would boot-loop).
- **Expose TCP port 22**; container disk **80 GB**.
- **No volume mounted at `/workspace`** — it would shadow the baked code and
  silently run whatever was on the volume instead of the pushed image. That
  would fake the parity check, not fail it. If a volume is wanted, mount it
  at `/data`.
- Deploy → copy the SSH-over-exposed-TCP line; `<IP>`/`<PORT>` below.
  `SSH` abbreviates `ssh -p <PORT> root@<IP>`.

## 2. First contact + start the data download (~5 min)

```bash
SSH 'nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; \
     nvidia-smi topo -m | tee /workspace/session_out/topo.txt; nproc; df -h /'
```

Expect 8 GPUs and `NV#` links in the topology (SXM). Then immediately start
the FineWeb fetch in the background — it is the long pole and nothing below
needs it until step 6. 55 chunks = 5.5B tokens (~10.5 GiB), margin over the
4.42B the run consumes; the val shard is always fetched:

```bash
SSH 'mkdir -p /workspace/session_out /workspace/data/fineweb10B && \
     ln -sfn /workspace/data/fineweb10B /workspace/reference/modded_nanogpt/fineweb10B && \
     cd /workspace && nohup python reference/modded_nanogpt/cached_fineweb10B.py 55 \
       > session_out/data_download.log 2>&1 &'
```

## 3. Parity check — the point of the registry push (~4 min)

```bash
SSH 'cd /workspace && python -c "import torch; print(torch.__version__, torch.version.cuda)" && \
     pytest -q 2>&1 | tail -3'
```

125 tests green **on the baked code, no rsync, no uv sync** = parity proven;
record the image digest from the console next to `$SHA` in the session log.
(If anything needed patching mid-session, note the file and diff in the
session log — the results' provenance is then `$SHA` + that recorded delta.)

## 4. Roofline + bandwidth ceiling (~5 min)

```bash
SSH 'cd /workspace && python scripts/measure_roofline.py | tee session_out/roofline.txt'
SSH '/opt/nccl-tests/build/all_reduce_perf -b 8M -e 512M -f 2 -g 8 \
     | tee /workspace/session_out/nccl_tests.txt'
```

The measured bf16 TFLOP/s is the MFU denominator for every number from this
box (the `_PEAK_BF16` A100 entry is an UNVERIFIED datasheet 312). If it
deviates: edit `mfu.py` on aurora, `rsync` that one file to the pod, record
the delta in the session log, commit the same change on aurora. MFU is
display-only — raw ms/step is unaffected either way.

## 5. NCCL smoke at 8 ranks, then the bench matrix (~45 min)

```bash
SSH 'cd /workspace && python scripts/make_synthetic_shards.py --out data/synthetic --shards 2'
SSH 'cd /workspace && NCCL_DEBUG=INFO torchrun --standalone --nproc_per_node=8 -m distrain.train \
     --distributed-mode ddp_naive --max-steps 5 --global-batch-seqs 64 --log-every 1 --no-trackio'
# repeat: ddp_bucketed, ddp_interleaved, ddp_torch — losses must agree to ~1e-3
```

Record the transport NCCL chooses (expect NVLink/P2P, unlike the 2×3090's
Socket+SHM). Then the measurement, both compile settings (the 2×3090 finding —
compile defeats hook overlap — is the hypothesis this retests at 8 ranks on a
fast interconnect):

```bash
SSH 'cd /workspace && nohup python scripts/bench_ddp_modes.py --nproc 8 --steps 50 --warmup 10 \
     --out-dir session_out/bench > session_out/bench.log 2>&1 &'
SSH 'cd /workspace && python scripts/bench_ddp_modes.py --nproc 8 --steps 30 --warmup 5 \
     --no-compile --out-dir session_out/bench'
```

## 6. The converged Track A run (~1.5–2 h)

Wait for the data download (`wc -l session_out/data_download.log`, 56 files).
Pick the fastest configuration from step 5 for `<mode>`/compile. Per-rank
batch is 60 seqs at accum 1 (global 480 unchanged — architecture *and* batch
identical across configs); if it OOMs, `--grad-accum-steps 2` and note it.

```bash
SSH 'cd /workspace && PYTHONUNBUFFERED=1 nohup torchrun --standalone --nproc_per_node=8 -m distrain.train \
  --train-glob "data/fineweb10B/fineweb_train_*.bin" \
  --val-glob "data/fineweb10B/fineweb_val_*.bin" \
  --distributed-mode <mode> --compile \
  --grad-accum-steps 1 --max-steps 9000 \
  --val-every 250 --checkpoint-every 250 \
  --run-name a100x8-fineweb-4.4B > session_out/train.log 2>&1 &'
```

Sanity gates, in order — stop early rather than pay for a broken run:

- step-0 val loss must print **10.8265** (= ln 50304): the zero-init-head
  fingerprint that the right code is running.
- val at step 250 ≈ 5.7, step 500 ≈ 4.9 (aurora's curve; large deviation =
  stop and pull logs).
- the number the session exists for: the loop's
  **`reached target 3.28 at step N after Xs`** line — first unsmoothed
  crossing, training time already excludes validation. Expected near the end
  of the warmdown, steps ~8500–9000; the run continues to 9000 regardless
  (the full curve is the artifact, the crossing is the headline).

## 7. Pull everything, then teardown (from aurora)

```bash
rsync -avz -e "ssh -p <PORT>" root@<IP>:/workspace/session_out/ \
      ~/work/distrain/out/runpod-8gpu/
rsync -avz -e "ssh -p <PORT>" root@<IP>:/root/.cache/huggingface/trackio/ \
      ~/work/distrain/out/runpod-8gpu/trackio/
rsync -avz -e "ssh -p <PORT>" root@<IP>:/workspace/checkpoints/ckpt.pt \
      ~/work/distrain/out/runpod-8gpu/   # final model, ~1.9 GB, optional
```

Nothing survives termination — pull first, verify locally (`ls -la`, open the
train log tail), then console **Stop → Terminate** and confirm the billing
page shows no running spend.

## Afterwards (aurora, free)

- Record the measured A100 roofline in `mfu.py` `_PEAK_BF16` (provenance:
  "measured on runpod 8×A100, <date>").
- Session log in `docs/sessions/`, same format as the 2×3090 one; commit with
  the artifacts' home noted.
- Update README (image parity gap closed, tokens-to-3.28 measured) and
  decisions §13 (the measured crossing replaces the bracket; budget actuals).
- Decide the compile × overlap question with 8-rank data — it shapes the
  Track A matrix and the DiLoCo/netem phase, which is next.

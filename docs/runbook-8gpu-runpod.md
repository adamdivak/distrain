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
  IMAGE=ghcr.io/adamdivak/distrain:$SHA    scripts/container.sh push
  IMAGE=ghcr.io/adamdivak/distrain:latest  scripts/container.sh push
  ```

  Use `container.sh push`, not `docker push`. BuildKit's default provenance
  attestation makes the tag an OCI *index*, and clients that only know the
  Docker media types then 404 on the manifest and report it as "image does not
  exist or you don't have permission" — Prime Intellect's image check does
  exactly that (decisions §20). `push` turns the attestation off and forces
  Docker media types; a plain `docker push` of an already-built image copies
  whatever the local store holds and cannot fix it after the fact.

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

## 1. Provision (~5 min)

[`scripts/runpod_session.py`](../scripts/runpod_session.py) does all of it —
network volume, pod, GHCR pull credential, start command, SSH wait — and is
idempotent, so a re-run reuses a live session instead of renting a second box:

```bash
uv run --script scripts/runpod_session.py avail            # which DCs have 8xA100 now
uv run --script scripts/runpod_session.py up --dry-run     # the plan + the price, free
uv run --script scripts/runpod_session.py up --max-hours 8 # rents; prints the ssh line
uv run --script scripts/runpod_session.py guard &          # the ceiling, enforced
```

`up` refuses to start what the balance cannot fund for `--max-hours`, and
terminates the pod itself if anything fails after creation. `guard` terminates
at the ceiling and warns on a low balance — credit exhaustion killed a
nearly-converged run on 2026-08-16. `ssh` prints (or with `--exec` runs) the
session's ssh line; `down` terminates. `SSH` below abbreviates that line.

What the script pins, and why the console path must match it if it is ever used
instead (Secure Cloud → 8× A100 80 GB on one host, SXM preferred for NVLink;
fallback 8× A100 PCIe, H100 only if neither exists):

- **Custom image** `ghcr.io/adamdivak/distrain:<SHA>` with the GHCR credential.
- **Container Start Command** `/workspace/scripts/pod-entry.sh` (starts sshd
  from the injected `PUBLIC_KEY`, idles; without it the image's default CMD
  prints a GPU report and exits — the pod would boot-loop).
- **Expose TCP port 22**; container disk **80 GB**.
- **No volume mounted at `/workspace`** — it would shadow the baked code and
  silently run whatever was on the volume instead of the pushed image. That
  would fake the parity check, not fail it. A network volume mounts at `/data`
  (`--volume-gb 0` to skip it; it bills until deleted, and it pins the pod to
  its data center).

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
     --distributed-mode ddp_naive --max-steps 5 --warmup-steps 0 --warmdown-steps 0 \
     --global-batch-seqs 64 --log-every 1 --no-trackio'
# --warmup/--warmdown 0: lr_at asserts the trapezoid fits inside max_steps
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
- val at step 250 ≈ 5.40, step 500 ≈ 4.51 (aurora's `rotary-calibration-3B`
  curve, from the trackio DB; large deviation = stop and pull logs).
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

Nothing survives termination — pull first, check locally (`ls -la`, open the
train log tail), then tear down and prove it:

```bash
uv run --script scripts/runpod_session.py down     # terminates, then runs verify
uv run --script scripts/runpod_session.py verify   # exits non-zero if anything bills
```

`verify` fails on anything metered hourly — running pods, *stopped* pods (their
disks still bill), serverless endpoints, or a non-zero account
`currentSpendPerHr` that no listed pod explains. A network volume is reported
with its ~$/month but does not fail the check; `--strict`, or
`down --delete-volume`, when it should be gone too.

## Afterwards (aurora, free)

- Record the measured A100 roofline in `mfu.py` `_PEAK_BF16` (provenance:
  "measured on runpod 8×A100, <date>").
- Session log in `docs/sessions/`, same format as the 2×3090 one; commit with
  the artifacts' home noted.
- Update README (image parity gap closed, tokens-to-3.28 measured) and
  decisions §13 (the measured crossing replaces the bracket; budget actuals).
- Decide the compile × overlap question with 8-rank data — it shapes the
  Track A matrix and the DiLoCo/netem phase, which is next.

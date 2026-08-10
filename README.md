# distrain

A distributed-training scaling study: a nanoGPT-style pretraining loop with a
distributed layer written from scratch, used to measure how training scales from
1 GPU to multiple GPUs and multiple nodes — on **both** raw throughput and
time-to-target-loss, and to quantify where and why the two diverge.

- [`project_brief.md`](project_brief.md) — goals, tracks, budget, providers. The
  original statement of intent, deliberately left unedited.
- [`docs/decisions.md`](docs/decisions.md) — everything settled since the brief,
  including the metric definitions that must not drift once runs cost money.
- [`reference/PROVENANCE.md`](reference/PROVENANCE.md) — vendored upstream code, and
  the exact definition of the 3.28 target.

## Status

Single-device training works end to end on CPU, MPS and CUDA; the three hand-rolled
DDP modes are correct (cursor-ordered launches, measured-order bucket rebuild) and
PyTorch's own DDP is wired in as a fourth, baseline mode. All four are NCCL-proven
and timed on a rented 2×3090 ([session log](docs/sessions/2026-08-09-runpod-2x3090.md));
headline finding: `torch.compile` defeats hook-based overlap, so uncompiled
interleaved is the fastest configuration on that box.

The model is no longer vanilla GPT-2: rotary embeddings (replacing `wpe`),
QK-norm, ReLU², zero-init residual projections, untied zero-init head, trapezoid
LR at 0.0018 — the early modded-nanogpt improvements, adopted because the vanilla
architecture measured val 3.50 after 3B tokens (~10B needed to 3.28, 3× the
assumed cost). See [`docs/decisions.md`](docs/decisions.md) §13. 125 tests
pass; the multi-rank ones run on gloo/CPU so they are exercisable without a GPU.

Validation runs on rank 0 only and the loss is broadcast, so every rank tests the same
value against the 3.28 target without N ranks paying for the same number.

| Piece | State |
|---|---|
| Shard IO + world-size-independent sharding | done, [`data.py`](src/distrain/data.py) |
| 124M GPT: SDPA, rotary, QK-norm, ReLU², zero-init, untied head | done, [`model.py`](src/distrain/model.py) |
| FLOPs / MFU / HFU accounting | done, [`mfu.py`](src/distrain/mfu.py) |
| Single-device loop, trapezoid LR, trackio logging | done, [`train.py`](src/distrain/train.py) |
| Pinned Docker image (aurora + cloud parity) | builds + tests pass on aurora, [`Dockerfile`](Dockerfile) |
| DDP modes 1–3 (naive, bucketed, interleaved) | done, [`distributed_synchronizer.py`](src/distrain/distributed_synchronizer.py); NCCL-proven, timed on 2×3090 |
| DDP mode 4 — `ddp_torch`, the upstream baseline | done, wraps `DistributedDataParallel` behind the same seam |
| Checkpointing — basic single-file save/resume | done, in [`train.py`](src/distrain/train.py); DCP deferred — [`docs/decisions.md`](docs/decisions.md) §12 |
| Bench harness for mode timing | done, [`scripts/bench_ddp_modes.py`](scripts/bench_ddp_modes.py) |
| FSDP2, DiLoCo, run matrix | not started |

The distributed layer is a single seam in the training loop — `finalize_gradients()`
between the accumulation loop and gradient clipping — behind which all four modes
switch at runtime. See [`docs/decisions.md`](docs/decisions.md) §6 for the conventions
it rests on and §10 for how the multi-rank tests are built.

Measured on aurora (RTX 3090): the modernized model (162M params with the untied
head) at seq-1024, batch 8, bf16 + compile → **130.4 ms/step, 82.6% MFU**.
Correctness only — a consumer card's numbers do not transfer (`project_brief.md` §8).

## Machines

| Machine | Role |
|---|---|
| `aurora` (RTX 3090, via Tailscale) | default development — editing, tests, all CUDA work, `~/work/distrain` |
| rented cloud nodes | all reported results — none yet |
| MacBook Pro (arm64) | fallback editor, CPU/MPS correctness only |

Day-to-day work happens directly on aurora (`ssh adam@aurora`); git is for
milestones. The Mac fallback workflow is at the [end of this README](#mac-fallback).

## Setup

Requires [uv](https://docs.astral.sh/uv/). The interpreter is uv-managed by design —
see [`docs/decisions.md`](docs/decisions.md) §2.

```bash
uv sync --extra dev
```

```bash
uv run pytest
```

On macOS this installs a CPU/MPS torch; on Linux, the CUDA build from the pinned
`cu126` index.

## Container

The pinned Docker image ([`Dockerfile`](Dockerfile)) is the reproducibility unit:
the *same* image on aurora and on rented cloud nodes, so cross-provider numbers are
comparable (`project_brief.md` §3). It bakes the exact `uv` environment on a pinned
CUDA 12.6 base; torch still comes from the `cu126` wheels, so the base supplies only
the toolchain and the driver ABI (injected by the NVIDIA Container Toolkit).

One-time host setup (installs the NVIDIA Container Toolkit, wires it into Docker,
adds you to the `docker` group — needs sudo, so run it yourself):

```bash
scripts/setup-docker-nvidia.sh   # then log out/in so 'docker' group applies
```

Then everything goes through one helper:

```bash
scripts/container.sh build        # (re)build the image
scripts/container.sh test         # pytest in the container, on GPU
scripts/container.sh smoke        # torch + GPU visibility check
scripts/container.sh run torchrun --nproc_per_node=1 -m distrain.train --max-steps 20
```

`run`/`test`/`shell` bind-mount the working tree at `/workspace` so rsync'd edits are
live without a rebuild (the venv lives at `/opt/venv`, outside the mount). `--no-mount`
runs the code baked into the image — the reproducible mode for reported results.

## Data

The full FineWeb10B set (104 shards, ~19 GiB; see
[`reference/PROVENANCE.md`](reference/PROVENANCE.md)) lives on aurora at
`data/fineweb10B`, fetched with:

```bash
ln -sfn ../../data/fineweb10B reference/modded_nanogpt/fineweb10B
uv run --extra data python reference/modded_nanogpt/cached_fineweb10B.py
```

The symlink makes the vendored script land shards in machine-local `data/`, which
git, rsync and the image build all ignore.

Synthetic shards exist for quick smokes and for machines without the real data:

```bash
uv run python scripts/make_synthetic_shards.py --out data/synthetic --shards 2
```

## Training

A real-data run on aurora — the Track A 124M model, GPT-2 global batch (480 seqs
via 60×8 accumulation), checkpointed and resumable:

```bash
PYTHONUNBUFFERED=1 nohup uv run python -m distrain.train \
  --train-glob 'data/fineweb10B/fineweb_train_*.bin' \
  --val-glob 'data/fineweb10B/fineweb_val_*.bin' \
  --grad-accum-steps 60 --max-steps 6000 \
  --val-every 250 --checkpoint-every 250 \
  --compile --run-name <name> > out/train.log 2>&1 &
```

`--checkpoint-every N` writes `checkpoints/ckpt.pt` (rank 0, atomic) every N steps;
`--resume` continues from it with the same command line — same command line matters,
because the LR schedule derives from `--max-steps`. `PYTHONUNBUFFERED=1` keeps the
log readable in real time instead of flushing every few hours.

A quick synthetic smoke (defaults are the 124M model at seq-1024):

```bash
uv run python -m distrain.train --global-batch-seqs 8 --max-steps 20 --compile
```

Local testing of a distributed run:

```bash
uv run torchrun --nproc_per_node=2 -m distrain.train --device cuda:0 \
  --distributed-backend gloo --distributed-mode ddp_naive
```

### Watching a run

Metrics live in a local SQLite store (`~/.cache/huggingface/trackio/distrain.db`).
The dashboard:

```bash
uv run trackio show --project distrain
```

It serves on `localhost:7860` on aurora; from another machine, forward the port
first (`ssh -L 7860:localhost:7860 adam@aurora`) and open http://localhost:7860.
System metrics (GPU/CPU/RAM, 10 s cadence) are logged automatically via the
`trackio[gpu]` extra.

### Timing the DDP modes

For any machine with ≥2 GPUs — a single-process baseline plus all three modes,
warmup excluded, raw per-step times and a comparison table written to
`out/bench/<timestamp>/`:

```bash
uv run python scripts/bench_ddp_modes.py --nproc 2 --steps 50 --warmup 10
```

Everything after `--` is forwarded to `distrain.train` (data globs, model size,
`--ddp-bucket-size`, ...). A hung mode is recorded as a result, not a crash — the
remaining modes still run. Harness mechanics are covered by
[`tests/test_bench.py`](tests/test_bench.py) on gloo/CPU, so the first paid
session only exercises NCCL, not the script.

## Measuring a new GPU

Before trusting MFU on any GPU class this project has not used before:

```bash
uv run python scripts/measure_roofline.py
```

Record the result in `_PEAK_BF16` in [`mfu.py`](src/distrain/mfu.py). Datacenter
entries there are datasheet values marked `UNVERIFIED` and should not be trusted
until measured — an unmeasured 3090 figure once produced a 158% MFU.

## Next steps

([`docs/decisions.md`](docs/decisions.md) §13 has the full reasoning.) In order:

1. **Overnight calibration run** (the 500-step sanity run passed, and a
   controlled A/B put rotary 0.74 val ahead at step 500 — decisions §13) —
   measures tokens-to-3.28 for the modernized model; that number, not the
   estimated ~2.2×, drives the final budget arithmetic.
2. **First larger-GPU session** (single 8-GPU node, RunPod has capacity):
   roofline + `nccl-tests`, image-parity check, then
   [`scripts/bench_ddp_modes.py`](scripts/bench_ddp_modes.py) across the four
   modes × compile on/off — the 2×3090 session showed compile kills hook-based
   overlap, and `ddp_torch` (which gets dynamo's DDPOptimizer graph breaks)
   is the interesting comparison. Then the first converged Track A run.
3. DiLoCo and the netem bandwidth curve after that.

## Known gaps

- **Image parity with the cloud is unproven.** The pinned image builds and its
  tests pass on aurora, but the 2×3090 session used a `uv` env on the provider's
  template (RunPod pods cannot run Docker-in-Docker); parity needs a session
  whose pod boots our image from a registry.
- **H100/A100/L40S peaks are unverified datasheet values.** Run the roofline
  script first thing on any rented node. (Both 3090s measured so far differ by
  9% — per-box measurement is mandatory.)
- **Mode timings exist only at 2 ranks over SHM.** Naive-vs-bucketed differences
  grow with world size and transport latency; the cluster session reruns the
  bench at 8 ranks. The compile-vs-overlap question (session log, decisions §13)
  is open and shapes the Track A matrix.
- **No spot-preemption recovery.** Basic single-file save/resume exists
  (`--checkpoint-every N`, `--resume`), enough to interrupt a local run; the
  DCP/preemption hardening is deliberately deferred
  ([`docs/decisions.md`](docs/decisions.md) §12) — it only matters if spot is chosen.
- **Tokens-to-3.28 for the modernized model is unmeasured** — the calibration
  run (next steps, item 2) exists to replace the ~2.2× estimate before any paid
  converged run.

## Mac fallback

The MacBook (arm64, no NVIDIA) can run everything except CUDA: tiny CPU/MPS configs
exercise the loop, data path, checkpointing and the gloo multi-rank tests. Keep the
batch small — fp32 logits for 480 sequences would need well over 100 GB. No
performance number from the Mac transfers anywhere (`project_brief.md` §8), and
Docker is not used there. Edit on the Mac, run on aurora:

```bash
scripts/sync-aurora.sh
ssh adam@aurora 'cd ~/work/distrain && uv run pytest -q'
```

`sync-aurora.sh` is for iteration (git stays for milestones); it excludes `data/`
and `.venv/`, which aurora owns.

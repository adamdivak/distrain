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

Single-device training works end to end on CPU, MPS and CUDA, and all three hand-rolled
DDP modes are correct, including under gradient accumulation. Mode 3 launches its
collectives through a cursor in fixed bucket order (rank-invariant by construction, not
by luck), and rebuilds its buckets after the first step in the measured
gradient-arrival order — rank 0's, broadcast — so the fixed order also preserves
overlap. 112 tests pass; the multi-rank ones run on gloo/CPU so they are exercisable
without a GPU. The first real-data run (124M, FineWeb, ~3B tokens) is underway on
aurora.

Validation runs on rank 0 only and the loss is broadcast, so every rank tests the same
value against the 3.28 target without N ranks paying for the same number.

| Piece | State |
|---|---|
| Shard IO + world-size-independent sharding | done, [`data.py`](src/distrain/data.py) |
| 124M GPT, SDPA, bf16 + `torch.compile` | done, [`model.py`](src/distrain/model.py) |
| FLOPs / MFU / HFU accounting | done, [`mfu.py`](src/distrain/mfu.py) |
| Single-device loop, trackio logging | done, [`train.py`](src/distrain/train.py) |
| Pinned Docker image (aurora + cloud parity) | builds + tests pass on aurora, [`Dockerfile`](Dockerfile) |
| DDP mode 1 — naive per-parameter all-reduce | done, [`distributed_synchronizer.py`](src/distrain/distributed_synchronizer.py) |
| DDP mode 2 — bucketed all-reduce | done |
| DDP mode 3 — bucketed + backward-hook overlap | done: cursor-ordered launches, arrival-order bucket rebuild; overlap not yet timed |
| Checkpointing — basic single-file save/resume | done, in [`train.py`](src/distrain/train.py); DCP deferred — [`docs/decisions.md`](docs/decisions.md) §12 |
| FSDP2, DiLoCo, run matrix | not started |

The distributed layer is a single seam in the training loop — `finalize_gradients()`
between the accumulation loop and gradient clipping — behind which all three modes
switch at runtime. See [`docs/decisions.md`](docs/decisions.md) §6 for the conventions
it rests on and §10 for how the multi-rank tests are built.

Measured on aurora (RTX 3090): 124M at seq-1024, batch 8, bf16 + compile →
**123.9 ms/step, 68.4% MFU**. Correctness only — a consumer card's numbers do not
transfer (`project_brief.md` §8).

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

De-risking is resequenced so the expensive cloud session starts with proven code
([`docs/decisions.md`](docs/decisions.md) §12). In order:

1. **Overnight real-FineWeb run on aurora** — *running* (started 2026-08-08, ~3B
   tokens). Validates the real data path and the token-budget assumption (2–3B vs
   5B to 3.28) before any paid run.
2. ~~Fix the mode-3 launch order~~ — *done*: launch cursor plus measured-order
   bucket rebuild ([`docs/decisions.md`](docs/decisions.md) §6).
3. **Cheap 2-GPU session** (~$5, Vast/RunPod community) — the next action. First
   NCCL contact, time the three DDP modes, verify mode 3 overlaps. Not for
   reported numbers.

## Known gaps

- **Image parity with the cloud is unproven.** The pinned image builds and its
  tests pass on aurora (toolkit installed via
  [`scripts/setup-docker-nvidia.sh`](scripts/setup-docker-nvidia.sh)), but "same
  image everywhere" stays a claim until the first rented node runs it.
- **H100/A100/L40S peaks are unverified datasheet values.** Run the roofline script
  first thing on any rented node, alongside `nccl-tests`.
- **The three DDP modes have never been timed against each other**, which is the
  measurement they exist for. It needs ≥2 GPUs — next-steps item 3 covers it, and
  [`scripts/bench_ddp_modes.py`](scripts/bench_ddp_modes.py) is ready to run there;
  correctness tests cannot tell a mode that overlaps from one that does not.
- **No spot-preemption recovery.** Basic single-file save/resume exists
  (`--checkpoint-every N`, `--resume`), enough to interrupt a local run; the
  DCP/preemption hardening is deliberately deferred
  ([`docs/decisions.md`](docs/decisions.md) §12) — it only matters if spot is chosen.
- **NCCL is untestable on aurora and stays unproven until first rented hardware.**
  Two ranks cannot share one GPU: NCCL rejects it outright (`ncclInvalidUsage`,
  duplicate GPU in the communicator), and there is no flag around it. What *does* run
  locally is `torchrun --nproc_per_node=2 --device cuda:0 --distributed-backend gloo`,
  which exercises `torchrun`, `cuda:{local_rank}` placement and collectives on CUDA
  tensors — everything except NCCL itself. Next-steps item 3 closes this.

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

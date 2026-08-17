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

**The first Track A number exists** ([session log](docs/sessions/2026-08-16-runpod-8xa100.md)):
on a rented 8×A100-SXM4 node, the 124M/162M model reached the 3.28 val-loss
target at **4.92B tokens / 3147.1 s of training time** (clean 10000-step
trapezoid, first unsmoothed crossing, ~79% MFU against the measured 269.9
TFLOP/s roofline, `ddp_torch --compile`). A clean 9000-step schedule ends
measurably short at 3.2849 / 4.42B tokens.

Single-device training works end to end on CPU, MPS and CUDA; the three hand-rolled
DDP modes are correct (cursor-ordered launches, measured-order bucket rebuild) and
PyTorch's own DDP is wired in as a fourth, baseline mode. All four are NCCL-proven
and timed on a rented 2×3090 ([session log](docs/sessions/2026-08-09-runpod-2x3090.md))
and at 8 ranks on the A100 node. The compile × overlap question resolved
per-transport: over NVLink every compiled config beats every uncompiled one
(`ddp_torch --compile` fastest); over the 3090s' Socket+SHM, uncompiled
interleaved wins — overlap buys more than compilation only once communication
dominates.

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
| Checkpointing — per-step files, retention anchors, async off-box mirror | done, in [`train.py`](src/distrain/train.py); survived a real pod termination. DCP deferred — [`docs/decisions.md`](docs/decisions.md) §12 |
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
| rented cloud nodes | all reported results |
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

For cloud sessions the image is pushed to **`ghcr.io/adamdivak/distrain:<git-sha>`**
(private, like the repo) and the pod boots it directly with
[`scripts/pod-entry.sh`](scripts/pod-entry.sh) as the start command — it starts
sshd from the provider-injected key and exports the baked env to SSH sessions.
The image also carries a prebuilt `nccl-tests` (`/opt/nccl-tests/build/`) and the
`data` extra, so a pod needs zero setup beyond booting. Build/push/provision
steps: [`docs/runbook-8gpu-runpod.md`](docs/runbook-8gpu-runpod.md).

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

1. **DiLoCo + the netem bandwidth curve** — the slow-transport side of the
   study, against the fast-interconnect anchor measured on 2026-08-16
   (3147.1 s / 4.92B tokens to 3.28 on 8×A100 NVLink). Only DiLoCo needs
   converged runs on the curve: DDP's crossing step is transport-invariant,
   so its time-to-target at each bandwidth is step-time × 9999 — see
   [`docs/decisions.md`](docs/decisions.md) §15. Rerun the bench
   matrix once under netem: the per-transport compile × overlap result
   predicts uncompiled interleaved retakes the lead once comm dominates.
2. Track B (FSDP2 at ~7B on one 8-GPU node) after that.

## Known gaps

- **H100/L40S/A100-PCIe peaks are unverified datasheet values.** A100-SXM4
  is measured (269.9). Run the roofline script first thing on any new GPU
  class; per-box measurement is mandatory (the two 3090s differ by 9%).
- **No spot-preemption recovery.** Per-step checkpoints with retention
  anchors, `--resume`/`--resume-from` and async off-box mirroring exist and
  survived a real pod termination; DCP/preemption hardening is deliberately
  deferred ([`docs/decisions.md`](docs/decisions.md) §12) — it only matters
  if spot is chosen.
- **Trackio curves from cloud sessions live in per-session DB copies**
  (`out/runpod-8gpu/trackio/`), not aurora's dashboard DB — a merge story
  is unbuilt.

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

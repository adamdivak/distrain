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

Single-device training works end to end on CPU, MPS and CUDA, and multi-rank DDP is
correct in its naive form, including under gradient accumulation. 86 tests pass; the
multi-rank ones run on gloo/CPU so they are exercisable without a GPU.

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
| DDP mode 2 — bucketed all-reduce | **next** |
| DDP mode 3 — bucketed + backward-hook overlap | not started |
| Checkpointing (`torch.distributed.checkpoint`) | not started |
| FSDP2 / TP, DiLoCo, run matrix | not started |

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
| MacBook Pro (arm64) | editor, CPU/MPS correctness work |
| `aurora` (RTX 3090, via Tailscale) | all CUDA work, `~/work/distrain` |
| rented cloud nodes | all reported results — none yet |

Edit on the Mac, run on aurora:

```bash
scripts/sync-aurora.sh
```

```bash
ssh adam@aurora 'cd ~/work/distrain && uv run pytest -q'
```

Git is for milestones; `sync-aurora.sh` is for iteration. It excludes `data/` and
`.venv/`, which aurora owns.

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

Local work runs on synthetic shards, so no download is needed:

```bash
uv run python scripts/make_synthetic_shards.py --out data/synthetic --shards 2
```

Real FineWeb shards are 190 MiB each and the full pull is ~19 GiB; see
[`reference/PROVENANCE.md`](reference/PROVENANCE.md) before fetching them. Nothing
real has been downloaded yet.

## Training

```bash
uv run python -m distrain.train --global-batch-seqs 8 --max-steps 20 --compile
```

Defaults are the 124M Track A model at seq-1024 with a ~0.5M-token global batch.
Batch must be far smaller on the Mac — fp32 logits for 480 sequences would need well
over 100 GB.

## Measuring a new GPU

Before trusting MFU on any GPU class this project has not used before:

```bash
uv run python scripts/measure_roofline.py
```

Record the result in `_PEAK_BF16` in [`mfu.py`](src/distrain/mfu.py). Datacenter
entries there are datasheet values marked `UNVERIFIED` and should not be trusted
until measured — an unmeasured 3090 figure once produced a 158% MFU.

## Known gaps

- **Image parity with the cloud is unproven.** The pinned image builds and its
  tests pass on aurora (toolkit installed via
  [`scripts/setup-docker-nvidia.sh`](scripts/setup-docker-nvidia.sh)), but "same
  image everywhere" stays a claim until the first rented node runs it.
- **H100/A100/L40S peaks are unverified datasheet values.** Run the roofline script
  first thing on any rented node, alongside `nccl-tests`.
- **No checkpointing yet**, so no spot-preemption recovery.
- **NCCL is untestable on aurora and stays unproven until the first rented node.**
  Two ranks cannot share one GPU: NCCL rejects it outright (`ncclInvalidUsage`,
  duplicate GPU in the communicator), and there is no flag around it. What *does* run
  locally is `torchrun --nproc_per_node=2 --device cuda:0 --distributed-backend gloo`,
  which exercises `torchrun`, `cuda:{local_rank}` placement and collectives on CUDA
  tensors — everything except NCCL itself. So a 2-process NCCL job belongs at the very
  top of the first cloud session, before anything with a clock on it.

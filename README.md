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

Single-device training works end to end on CPU, MPS and CUDA. 79 tests pass on both
the Mac and aurora.

| Piece | State |
|---|---|
| Shard IO + world-size-independent sharding | done, [`data.py`](src/distrain/data.py) |
| 124M GPT, SDPA, bf16 + `torch.compile` | done, [`model.py`](src/distrain/model.py) |
| FLOPs / MFU / HFU accounting | done, [`mfu.py`](src/distrain/mfu.py) |
| Single-device loop, trackio logging | done, [`train.py`](src/distrain/train.py) |
| Hand-rolled DDP (3 modes) | **next** |
| Checkpointing (`torch.distributed.checkpoint`) | not started |
| FSDP2 / TP, DiLoCo, run matrix | not started |

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

- **NVIDIA Container Toolkit is not installed on aurora**, so `docker run --gpus all`
  does not work and image parity with the cloud is unproven.
- **H100/A100/L40S peaks are unverified datasheet values.** Run the roofline script
  first thing on any rented node, alongside `nccl-tests`.
- **No checkpointing yet**, so no spot-preemption recovery.

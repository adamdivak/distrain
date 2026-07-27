# distrain

A distributed-training scaling study: a nanoGPT-style pretraining loop with a
distributed layer written from scratch, used to measure how training scales from
1 GPU to multiple GPUs and multiple nodes — on **both** raw throughput and
time-to-target-loss, and to quantify where and why the two diverge.

- [`project_brief.md`](project_brief.md) — goals, tracks, budget, providers.
- [`docs/decisions.md`](docs/decisions.md) — decisions made since the brief, including
  the metric definitions that must not drift once runs start.

Status: **setup**. No training code yet.

## Development setup

Requires [uv](https://docs.astral.sh/uv/). On macOS this installs a CPU/MPS build of
torch, which is for correctness work only — it produces no transferable performance
number.

```bash
uv sync --extra dev
```

Run the tests:

```bash
uv run pytest
```

On Linux (`aurora` and rented nodes), the same command installs the CUDA build from
the pinned `cu126` wheel index.

## Data

Local work runs on synthetic shards, so no download is needed:

```bash
uv run python scripts/make_synthetic_shards.py --out data/synthetic --shards 2
```

Real FineWeb shards are 190 MiB each; see [`reference/PROVENANCE.md`](reference/PROVENANCE.md)
for sizes and the exact definition of the 3.28 target before pulling them.

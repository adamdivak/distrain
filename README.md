# distrain

A distributed-training scaling study: a nanoGPT-style pretraining loop with a
distributed layer written from scratch, used to measure how training scales from
1 GPU to multiple GPUs and multiple nodes — on **both** raw throughput and
time-to-target-loss, and to quantify where and why the two diverge.

- [`project_brief.md`](project_brief.md) — goals, tracks, budget, providers. The
  original statement of intent, deliberately left unedited.
- [`docs/decisions.md`](docs/decisions.md) — everything settled since the brief,
  including the metric definitions that must not drift once runs cost money.
- [`docs/writeup.md`](docs/writeup.md) — the article draft; its collected numbers,
  rendered figures, and remaining plot gaps are indexed in
  [`docs/writeup_data/README.md`](docs/writeup_data/README.md).
- [`reference/PROVENANCE.md`](reference/PROVENANCE.md) — vendored upstream code, and
  the exact definition of the 3.28 target.

## Status

**The scaling headline exists end to end**
([session log](docs/sessions/2026-08-21-prime-single-gpu-scaling.md)): on one
A100 the 162M model reaches 3.28 in **7.19 h** (2589.1 ms/step at global batch
480, 60.1% MFU); on 8 of them over NVLink, **0.94 h** — **7.66×, 95.8% scaling
efficiency**, measured on one box at one micro-batch. Forcing NCCL off P2P and
shared memory and onto TCP/loopback costs **3.8×** (3.38 h) — a tax, not a
dealbreaker, since eight badly-connected GPUs still beat one well-connected one
by 2.1×. §14's 0.88
scaling figure turns out to be an artefact of the bench's 8-seq default batch:
**scaling efficiency is a function of the batch and must be quoted with it**
(`docs/decisions.md` §22).

**The DDP-vs-DiLoCo comparison exists**
([session log](docs/sessions/2026-08-21-prime-diloco-k8.md)): on 8×A100-SXM4,
at identical global batch and an identical 4.92B-token budget, DDP reaches
**3.2730** in 3147.1 s while untuned DiLoCo (K=8, H=500, outer lr 0.7, μ=0.5)
reaches **3.5183** in 3250.3 s — **+0.245 val loss and no wall-clock saving**
on a full-NVLink fabric, which is the transport where DiLoCo has nothing to buy.
DiLoCo's tokens-to-3.28 remains unmeasured and is now bounded as unreachable
inside this corpus (~24,900 steps needed vs a ~11,190-step wrap). The netem
sweep in the same session found that the compile-vs-overlap lead **dissolves**
rather than flipping as bandwidth falls: 0.12% spread across modes at 10 gbit.
See [`docs/decisions.md`](docs/decisions.md) §21.

**The first Track A number exists** ([session log](docs/sessions/2026-08-16-runpod-8xa100.md)):
on a rented 8×A100-SXM4 node, the 124M/162M model reached the 3.28 val-loss
target at **4.92B tokens / 3147.1 s of training time** (clean 10000-step
trapezoid, first unsmoothed crossing, ~62% MFU against the measured 269.9
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
| DiLoCo (5th mode, per-rank checkpoints, diag evals) | done; K=8 anchor measured on 8×A100, [`docs/decisions.md`](docs/decisions.md) §21 |
| FSDP2, run matrix | not started |

The distributed layer is a single seam in the training loop — `finalize_gradients()`
between the accumulation loop and gradient clipping — behind which all four modes
switch at runtime. See [`docs/decisions.md`](docs/decisions.md) §6 for the conventions
it rests on and §10 for how the multi-rank tests are built.

Measured on aurora (RTX 3090): the modernized model (162M params with the untied
head) at seq-1024, batch 8, bf16 + compile → **130.4 ms/step, 65.0% MFU**.
Correctness only — a consumer card's numbers do not transfer (`project_brief.md` §8).

## Writeup figures

The writeup numbers have been projected from the gitignored raw logs and benchmark
JSON into reviewable CSVs under [`docs/writeup_data/`](docs/writeup_data/). The
plotting script renders each figure as interactive HTML and static SVG/PNG under
[`docs/plots/`](docs/plots/):

```bash
uv run python scripts/collect_writeup_data.py
uv run --extra plots python scripts/plot_writeup.py
```

The rendered set covers both halves of the converged loss curve, measured 8-GPU
rentability, equal-token DDP-vs-DiLoCo quality and runtime, matched-batch 1→8 GPU
scaling, DDP transport sensitivity, transport MFU, four DDP implementations over
sockets, compilation versus overlap on A100 PCIe, DiLoCo outer-sync diagnostics,
the DiLoCo merge penalty at two replica counts, and DDP-vs-DiLoCo step time
across transports. Headline plotted values
include **7.19 h → 0.94 h** for 1→8 A100 scaling, **57.6% → 15.3% MFU** for
NVLink → TCP, and **3.2730 versus 3.5183** validation loss for DDP versus DiLoCo
at 4.915B tokens.

Two derived numbers now sit on the same reconstruction as the netem points and
answer the writeup's actual question. **8-GPU DDP stops beating a single A100
below 0.50 GB/s (4.0 Gbit/s) effective all-reduce bandwidth** — the measured
unthrottled socket sat only 1.8× above that cliff. And **DiLoCo starts beating
DDP end to end below 2.36 GB/s (18.8 Gbit/s)** once charged §18's 2.49× token
ratio, which is why its equal-token loss penalty is not the whole story. See
[`transport_crossovers.csv`](docs/writeup_data/transport_crossovers.csv).

**Still missing, in the order it would cost to buy:** a topology-verified A100
PCIe point — the largest hole, and **not a price problem**: RunPod sells the PCIe
card as its own GPU type at half the SXM4 anchor's rate and had no 8-GPU capacity
on 2026-08-22, while Prime Intellect has none at all
([`docs/decisions.md`](docs/decisions.md) §24). `scripts/pcie_hunt.sh` rents and
measures the first opening unattended. Then: a *measured* rather than
reconstructed DiLoCo slow-transport timing; complete price-per-convergence
comparisons; and a current-PyTorch rerun.
DiLoCo's time to 3.28 is blocked rather than unbought — it needs ~24,900 steps
against a corpus that wraps near 11,190. The procedure for each, and the PCIe
capacity-watch command, are in
[`docs/writeup_data/README.md`](docs/writeup_data/README.md#plots-still-missing);
the field-level manifest stays in
[`data_gaps.csv`](docs/writeup_data/data_gaps.csv).

Cost inputs belong in [`cost_inputs.csv`](docs/writeup_data/cost_inputs.csv),
which has the measured/projected runtimes prefilled and leaves rental rates,
average desktop wall power, electricity prices, capital allocation, and overrides
blank. The existing
3090 Trackio run supports a **21.30 h projected time to 3.28** (7.6684 s/step ×
9999); it did not itself cross the target and is labelled as an extrapolation.
That is the *same* evidence class as the 1×A100 **7.19 h** bar, which is also a
measured step time times the measured step-9999 crossing — so no direct 3090
convergence run is planned, and only the power and electricity fields are open.

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
`data` extra, so a pod needs zero setup beyond booting. Build/push steps:
[`docs/runbook-8gpu-runpod.md`](docs/runbook-8gpu-runpod.md).

### Renting the box

[`scripts/runpod_session.py`](scripts/runpod_session.py) provisions the pod over
the RunPod API instead of the console — ensure the network volume, ensure the
pod, boot the pinned image, wait for SSH:

```bash
uv run --script scripts/runpod_session.py status           # balance, pods, volumes
uv run --script scripts/runpod_session.py avail            # which DCs have 8xA100 now
uv run --script scripts/runpod_session.py up --dry-run     # the plan + the price, free
uv run --script scripts/runpod_session.py up --max-hours 8
uv run --script scripts/runpod_session.py guard &          # ceiling + balance watch
uv run --script scripts/runpod_session.py ssh --exec
uv run --script scripts/runpod_session.py down
uv run --script scripts/runpod_session.py verify           # is anything still billing?
```

`verify` is the end-of-session proof, and `down` ends by running it. It separates
what is metered **by the hour** — running pods, stopped pods (no GPU charge, but
their disks bill), serverless endpoints, and the account's own
`currentSpendPerHr` — from what is metered **by the month**: network volumes,
~$0.07/GB. The hourly tier must be empty and any of it exits non-zero; volumes
are reported with a price but pass, since a volume is the thing meant to outlive
a pod. `verify --strict` requires those gone too — the end of the project rather
than the end of a session.

It is idempotent (a second `up` reuses the live pod rather than renting another)
and it is where the cost kill-switch of [`docs/decisions.md`](docs/decisions.md)
§9 lives: `up` refuses a ceiling the balance cannot fund and terminates the pod
if provisioning fails partway; `guard` terminates at the wall-clock ceiling. The
`runpod` SDK comes from the script's own inline dependency block (`uv run
--script`), never from the training environment; the API key comes from
`RUNPOD_API_KEY` in the environment or the gitignored `.env`. The mechanics are
tested offline against a fake API in
[`tests/test_runpod_session.py`](tests/test_runpod_session.py) — no credentials,
nothing rented.

### The second venue

RunPod's 8×A100 stock is opportunistic, so
[`scripts/prime_session.py`](scripts/prime_session.py) mirrors the same contract
(`status`/`avail`/`up`/`guard`/`ssh`/`down`, wall-clock ceiling,
teardown-on-exception) against Prime Intellect, a compute exchange that resells
lambdalabs, vultr, hyperstack and others. It is stdlib-only — no SDK — and pins
the socket to SXM4 so a PCIe A100 cannot silently break comparability with the
§14 NVLink anchor. `up` refuses to provision without `--template-id` rather than
boot Prime Intellect's stock image.

Money added in their console lands on a **team** wallet, which a bare API key
cannot see — so the team id is picked up from `PRIME_TEAM_ID`, `.env`, or the
`prime` CLI's config and sent with every call (`--team-id ''` forces the personal
wallet). `status` prints which wallet it read. One step still has no API and must
be done in their console: adding a ghcr.io registry credential so a custom
template on the pinned image can be created. See
[`docs/decisions.md`](docs/decisions.md) §20 — including why SkyPilot was
evaluated and rejected as the broker.

`scripts/watch_capacity.sh` polls both venues.

A Prime Intellect pod is a **KVM VM with root**, so the session boots their stock
Ubuntu and runs our own container inside it — which pulls the byte-identical
aurora image, better parity than their registry (which rebuilds), and gives
`NET_ADMIN`, so netem runs in the same container as the training. That is the one
thing RunPod cannot do (§17). Two sessions are written up in
[`docs/runbook-prime-intellect.md`](docs/runbook-prime-intellect.md): the netem
curve (§15) and the DiLoCo K=8 converged anchor (§16, §19).

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

1. **Get one honest PCIe point**, the transport most people can actually rent.
   This is now waiting on stock rather than on a decision (§24): RunPod's
   `NVIDIA A100 80GB PCIe` is the real SKU at half the anchor's price and had no
   8-GPU capacity on 2026-08-22, and Prime Intellect has no PCIe 8×A100 at all.
   Leave `scripts/pcie_hunt.sh out/pcie-hunt.log 300` running — it probes by
   attempting the deploy (free when rejected), then guards, gates on
   `nvidia-smi topo -m`, measures, tears down and verifies without a human.
   Only bandwidth plus a batch-480 bench is needed, so it is minutes of rental
   once a box exists.
2. **Decide whether DiLoCo's tokens-to-3.28 is worth buying.** It needs
   ~24,900 steps at the measured ratio — past the 55-chunk corpus wrap
   (~11,190 steps), so it cannot be measured without disclosing a second
   epoch, and it costs ~2.3 h of 8×A100 on top. The equal-token endpoint
   (§21) may simply be the honest deliverable.
3. Track B (FSDP2 at ~7B on one 8-GPU node) after that.

**The netem ladder stops at 10 gbit, on purpose.** The 2026-08-21 sweep measured
unthrottled-socket, 40 gbit and 10 gbit; 1 gbit and 500 mbit overran their
timeouts because the schedule was budgeted off *nominal* rate, which netem misses
by ~8× ([`docs/decisions.md`](docs/decisions.md) §21). They are not worth
re-running. Nominal 10 gbit already delivers only 1.2 Gbit/s effective and puts
an 8-GPU DDP run at **21.1 h against one A100's 7.19 h** — far past the 0.50 GB/s
break-even where eight GPUs stop beating one. A lower point would extend a curve
whose conclusion is already settled.

## Known gaps

- **H100/L40S peaks are unverified datasheet values.** A100-SXM4 is measured at
  both memory sizes (269.9 / 270.1, 0.07% apart) and A100 80GB PCIe at 256.5
  (2026-08-22) — the PCIe part throttles to 223.9 at n=16384 where the SXM4
  cards peak, and its 312.0 datasheet figure was 22% high. Run the roofline
  script first thing on any new GPU class; per-box measurement is mandatory (the
  two 3090s differ by 9%). Check what the *generic* patterns in `_PEAK_BF16`
  already swallow: `"A100"` was silently catching 40 GB cards with a datasheet
  figure instead of refusing to start (`docs/decisions.md` §22).
- **PCIe is measured at 2 GPUs; the 8-GPU bar is out of stock, not mispriced.**
  A topology-verified 2×A100 80GB PCIe box gave **2.29 GB/s** effective
  all-reduce bandwidth — and revealed that the rentable node has **no GPU-to-GPU
  P2P at all** (`topo -p2p r` = `CNS`; NCCL routes via host memory), so none of
  it may be reported as PCIe P2P (§25). At 8 ranks that projects to 2.29 h to
  3.28, an optimistic bound since a wider ring crosses more host bridges.
  RunPod's `NVIDIA A100 80GB PCIe` is a distinct SKU at $11.12/h secure (half
  the SXM4 anchor) and had no 8-GPU capacity on 2026-08-22; Prime Intellect's
  every "PCIe" 8×A100 is `cloudId gpu_8x_a100`, lambdalabs' SXM4 box, which
  `prime_session.py` now refuses before renting (§24). The netem-derived points
  in `scripts/transport_curve.py` remain **upper bounds, +23% at the one point
  measured both ways**.
- **A registry credential is applied to whichever registry the image names.**
  Attaching the GHCR credential to a `runpod/pytorch` image fails the pull with
  `IMAGE_AUTH_ERROR` rather than being ignored — three dead pods on 2026-08-22.
  Use `--registry-auth-name ''` for a public image (§24).
- **DiLoCo's advantage on slow transport is reconstructed, not measured.**
  `diloco_transport.csv` divides each transport's *measured* all-reduce by
  H=500, which is sound — the outer sync moves the same tensor DDP all-reduces
  every step — but no DiLoCo run has actually been timed over a slow transport.
- **No spot-preemption recovery.** Per-step checkpoints with retention
  anchors, `--resume`/`--resume-from` and async off-box mirroring exist and
  survived a real pod termination; DCP/preemption hardening is deliberately
  deferred ([`docs/decisions.md`](docs/decisions.md) §12) — it only matters
  if spot is chosen.
- **Trackio curves from cloud sessions live in per-session DB copies**
  (`out/runpod-8gpu/trackio/`), not aurora's dashboard DB — a merge story
  is unbuilt. Worse on the container route: the DB lands in `~/.cache`
  *inside* the container, which no mount covers, so the 2026-08-21 curves
  died with the pod and `train.log` is the only record. Mount it next time.
- **A100-SXM4 peaks differ ~1.4% box to box** (269.9 measured on RunPod,
  266.0 on lambdalabs). MFU is reported against 269.9 throughout for
  cross-session comparability, which understates lambdalabs numbers slightly.

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

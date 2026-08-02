# Decisions log

Decisions settled *after* `project_brief.md` was written. The brief remains the
statement of intent; this file records the choices that turn it into code, plus
the things that must not drift once runs start costing money.

Each entry: what was decided, and why it matters. Anything that would silently
invalidate a cross-config comparison is marked **load-bearing**.

---

## 1. Machines and where code runs

| Machine | Role | Environment |
|---|---|---|
| MacBook Pro (arm64, no NVIDIA) | Editor + fast iteration | native `uv` venv, CPU / MPS |
| `aurora` (local, RTX 3090 24 GB) | All CUDA correctness work | native `uv` venv day-to-day; pinned image for parity checks |
| Rented cloud nodes | All reported results | pinned Docker image |

Reached as `adam@aurora` over Tailscale; the repo lives at `~/work/distrain`.
Driver 580.173.02, 16 cores, 31 GB RAM, 730 GB free — enough for the full 19 GiB
FineWeb10B pull.

The Mac must be able to run a tiny config end-to-end (small `n_layer`/`n_embd`,
short sequence, CPU or MPS) so the loop, data path, checkpointing and the
`gloo` multi-rank tests are exercisable without leaving the editor. It cannot
produce any performance number, and no `torch.compile`/bf16 claim from the Mac
transfers anywhere.

Docker is **not** used on the Mac — a CUDA image is meaningless on arm64 without
an NVIDIA GPU. Identical-image reproducibility is a claim about `aurora` and
cloud only, which is where every reported number comes from.

The image and its tooling now exist: `Dockerfile` bakes the pinned `uv` environment
on a pinned CUDA 12.6 devel base (torch still from the `cu126` wheels; the base
supplies the toolchain and, via the toolkit, the driver ABI). `scripts/container.sh`
is the single build/run/test/shell entry point, and `scripts/setup-docker-nvidia.sh`
does the one-time, sudo-requiring host setup (install the NVIDIA Container Toolkit,
`nvidia-ctk runtime configure --runtime=docker`, add the user to `docker`). This has
now been run on aurora (toolkit 1.19.1): the image builds and its tests pass in the
container on the GPU. Image parity with the *cloud* stays unproven only until the
first rented node runs the same image; the native `uv` venv remains the day-to-day
path on aurora, with the container the reproducibility unit for reported results.

## 1a. Development workflow

Edit on the Mac, run on aurora:

```bash
scripts/sync-aurora.sh                                    # rsync, well under a second
ssh adam@aurora 'cd ~/work/distrain && uv run pytest -q'
```

Git is for milestones, not for iteration — pushing and pulling per edit made the
feedback loop slow and the history unreadable. `sync-aurora.sh` excludes `data/`,
`.venv/` and outputs, and `rsync --delete` does not touch excluded paths, so
aurora keeps its own shards and environment.

Remote: `github.com/adamdivak/distrain`, **private for now**, to be made public
with the write-up.

## 2. Dependency pinning

`torch==2.13.0`, `numpy==2.5.1`, `tiktoken==0.13.0`, `trackio==0.33.0`, exact-pinned
in `pyproject.toml` with a `uv.lock` covering transitives. trackio in particular is
pinned because its API is young (brief §3).

torch resolves from PyPI on macOS and from the **cu126** wheel index on Linux.

aurora's driver is now confirmed as **580.173.02**, which is new enough for CUDA 13,
so cu130/cu132 would work there. cu126 is kept anyway: rented nodes are the binding
constraint and many ship older drivers, and the pin exists to make cross-provider
results comparable rather than to chase kernels. If it is ever bumped, bump *before*
the first cloud run — never between runs that get compared to each other.

**Python interpreter is uv-managed** (`python-preference = "only-managed"`). Ubuntu's
system `python3.12` ships without development headers unless `python3-dev` is
installed, and without `Python.h` Triton cannot build the kernels `torch.compile`
emits — compiled runs die at the first step. uv's CPython includes headers, so this
removes a root requirement and pins the interpreter itself.

## 3. MFU / HFU definition — **load-bearing**

Every throughput claim in both tracks depends on this being computed one way in
one place.

- **Numerator**: PaLM-style FLOPs per token, `6N + 12*L*H*Q*T` — i.e. including
  the attention term, not bare `6ND`. Measured: that term is **13% of total FLOPs**
  at 124M/seq-1024, which is larger than most effects this study aims to measure.
  Using `6ND` on Track A and the full formula on Track B would also make the two
  tracks incomparable.
- **MFU vs HFU**: report **true MFU** (excludes activation-recomputation FLOPs).
  When activation checkpointing is on — which Track B will need under FSDP2 — also
  log **HFU** (includes recompute) as a separate metric. They differ by 30%+ and
  are routinely conflated in published numbers.
- **Denominator**: bf16 **dense** peak, never the 2:4-sparsity marketing figure.
  H100 SXM = 989 TFLOP/s, *not* 1979.
- **Measure the roofline, don't cite it.** Datasheets mix tensor and non-tensor rates,
  dense and sparse figures, and FP16- vs FP32-accumulate variants. Run
  `scripts/measure_roofline.py` on each new GPU class and record the measured value
  with its provenance. Datacenter entries currently in the table are datasheet
  values marked `UNVERIFIED` until measured on first use.
- **Implementation**: one unit-tested function, with peak TFLOP/s in a table keyed
  by device name. Never hand-entered per run.

**Correction (2026-07-28).** The 3090 was first entered at 35.6 TFLOP/s on the belief
that GeForce cards run bf16 tensor matmuls at half rate. That is wrong: 35.6 is the
card's **FP32 non-tensor** rate. Measured on aurora: bf16 GEMM sustains **82.6–82.9
TFLOP/s**, while fp32 GEMM runs at 27.4 — about 77% of 35.6, which is what confirmed
the misidentification. The bad denominator produced a **158% MFU**, corrected to 68.4%.

Two mechanisms now exist so this fails loudly rather than silently:
- `warn_if_impossible()` in the training loop shouts once if MFU exceeds 100%.
- `scripts/measure_roofline.py` compares its measurement against the recorded table
  and flags a gap over 2% (below that is run-to-run noise).

Known convention detail, deliberately left alone: `12*L*H*Q*T` assumes a full T x T
attention matrix, while causal SDPA computes roughly half of it. PaLM and nanoGPT both
count it this way, so keeping it preserves comparability with published MFU figures at
the cost of a small overstatement.

## 4. Time-to-target-loss definition — **load-bearing**

The headline Track A metric. Pinned before any paid run, because ambiguity here
is on the same order as the DDP-vs-DiLoCo effect being measured:

- Fixed val eval every N steps, N chosen so eval overhead is < 1% of step time.
- Reported value is the **first** step/wall-clock at which val loss ≤ 3.28.
- **No smoothing** applied to the val curve.
- Eval batch, val split and eval determinism identical across every config.

## 5. Seed variance

Single runs per config give no error bar, and time-to-target-loss run-to-run
variance can be comparable to the effects being measured. Plan: 2–3 seeds on the
single-GPU baseline and on one DDP config only, to establish a variance estimate
that is quoted alongside the rest of the (single-run) matrix. Estimated cost
$20–40; accepted.

This is about how many seeds to *run*. How a seed is derived per rank is §11.

## 6. Hand-rolled DDP is three implementations, not one

Built as runtime-switchable modes so all three are measurable on identical hardware:

1. naive per-parameter all-reduce — **done** (`cfg.distributed_mode="ddp"`)
2. bucketed all-reduce — not started
3. bucketed with backward-hook compute/communication overlap — not started

This is an extra result axis for free (no extra provisioning) and it is the most
direct demonstration of the skill the project exists to build.

### The seam in the training loop

All three modes hide behind `DistributedSynchronizer`
(`src/distrain/distributed_synchronizer.py`), constructed once and called once per
optimizer step:

```python
dist_sync.finalize_gradients()   # after the accumulation loop, BEFORE clip_grad_norm_
```

Named for the *event* — "every gradient for this step is final" — rather than for
the batch hierarchy or the operation, because in mode 3 the method does not perform
the reduction at all: the backward hooks already did, and it only waits on work still
in flight. `sync_gradients()` would be a lie in the mode that matters most.

Clipping must happen **after** the reduce. Clipping local gradients and then averaging
gives a different (and wrong) result, and since post-reduce gradients are identical on
every rank, all ranks compute the same norm.

### Conventions — **load-bearing**

- **Two chunkings, two divisors.** The loss is a per-token *mean*, so the gradient of
  the global batch is the mean of the chunk gradients. Accumulation and sharding chunk
  the same global batch along different axes and each gets its own divisor:
  `(loss / grad_accum_steps).backward()` in the loop, `/ world_size` in the
  synchronizer. They compose to the micro-batch count. Do not move either.
  - Summing instead would be *nearly* invisible — AdamW is close to scale-invariant —
    but `clip_grad_norm_` is not, so clipping would engage harder as GPUs are added and
    the effect would masquerade as a legitimate large-batch result.
  - `/ world_size` assumes an **equal token count on every rank**. `ShardingPlan`'s
    divisibility check guarantees it today; a ragged tail would need a weighted mean.
- **`ReduceOp.AVG` is NCCL-only.** gloo is the correctness backend (it runs on the Mac),
  so it is always `SUM` followed by a manual divide.
- **Collective order must be rank-invariant.** Collectives are matched by order of
  invocation within the process group — there is no tag, no name, no header (only
  point-to-point `send`/`recv` take a `tag`). If rank 0 reduces a parameter that rank 1
  skips, every subsequent collective is paired with the wrong tensor: hang, crash, or
  silently wrong gradients of the right shape. Hence `finalize_gradients` iterates every
  parameter unconditionally and materialises `zeros_like` when `.grad is None`, rather
  than testing a per-rank predicate. Real DDP freezes bucket order in its constructor for
  the same reason, and makes `find_unused_parameters` an explicit, per-iteration opt-in.
  - Zero-filling is not a fudge: `grad is None` means that parameter was absent from the
    autograd graph, so its true contribution to the global mean gradient *is* zero.
  - `TORCH_DISTRIBUTED_DEBUG=DETAIL` validates cross-rank shape/dtype agreement and raises
    instead of hanging; `TORCH_NCCL_DESYNC_DEBUG=1` names the rank and collective that
    desynced. Both are worth setting when a distributed test hangs.
- **Replica equality comes from a broadcast, not from seeding.** `__init__` broadcasts
  every parameter (and buffer) from rank 0. Trusting two RNG streams to coincide is
  fragile in ways that are hard to anticipate; one collective at startup makes it an
  invariant. Seeds are `cfg.seed + cfg.rank` — rank-dependent so dropout decorrelates,
  never multiplied by `world_size` (see §11).
- **The broadcast must run under `torch.no_grad()`.** `dist.broadcast` is a dispatcher
  op (`c10d::broadcast_`, returning `(Tensor[], Work)`), not a raw memcpy. With grad mode
  on and a `requires_grad` input it routes through the Autograd key, finds no registered
  kernel, and the fallback wires a `NotImplemented` node into the graph — on the returned
  tensors that `dist.broadcast` discards, so it is invisible on the parameter itself and
  only surfaces as a warning at backward. `no_grad()` skips the Autograd key entirely so
  no node is created. Preferred over `.data`, which also works but additionally bypasses
  the version-counter safety net. The `all_reduce` in `finalize_gradients` needs no such
  guard: `.grad` does not require grad.
- **Tied weights are reduced once.** `wte.weight` and `lm_head.weight` are the same
  tensor; `named_parameters(remove_duplicate=True)` (the default) yields it once. Untying
  is a `GPTConfig` change, applied to every config in a study or none — untying *mid-run*
  would additionally invalidate the optimizer's param groups.

## 7. Data sharding — **load-bearing**

Shard deterministically by **global token index**, independent of world size
(brief §4). Enforced by a test asserting that concatenated per-rank token streams
are byte-identical across world sizes 1/2/4/8. Without this, changing GPU count
changes data ordering and silently confounds every time-to-target-loss comparison.

## 8. DiLoCo is in scope

Confirmed in scope for Track A. Renting well-connected multi-node clusters has
turned out to be the binding constraint (brief §7), which makes low-communication
methods more relevant to this project, not less — the slow-network curve is the
payoff plot.

## 9. Deferred to the cloud stage (recorded so they aren't rediscovered late)

- **netem needs `NET_ADMIN`** on the container, or `tc qdisc` cannot run at all.
  A provisioning flag to remember when writing the Docker / SkyPilot config.
- **Plot against measured bandwidth, not configured.** NCCL-over-TCP achieved
  bandwidth does not track the netem cap linearly. Re-run `all_reduce_perf` at
  every netem setting and use that as the x-axis.
- **Cost kill-switch.** Unattended provision → run → teardown needs a hard
  wall-clock ceiling and teardown-on-exception in the launcher, written before the
  first unattended run. Otherwise the $400 ceiling is enforced by luck.
- **Test DCP preemption recovery locally** by SIGKILLing ranks on `aurora`, before
  depending on it for spot instances.
- **Track B partially survives a 1-node world.** At ~7B, model states (~112 GB)
  exceed one 80 GB H100, so FSDP2 is genuinely required on 8 GPUs alone. Only the
  TP8×DP2 topology needs 2 nodes. This is the de-risked fallback if 2-node
  capacity never appears.

## 10. Distributed correctness harness

`tests/test_distributed.py`, gloo/CPU, so it runs on the Mac as well as aurora.

**Multi-rank via `torch.multiprocessing.spawn`, not `torchrun`.** Spawned children stay
in the pytest process tree — no shell, no orphan cleanup, and failures propagate as
exceptions. Production still goes through `torchrun`; both reach the same code path
because rank/world-size/local-rank are read from the environment in `main()` only.
Rendezvous is `file://{tmp_path}/rendezvous_world_size{N}` — a FileStore avoids port
collisions under parallel pytest, and the world size is in the filename because a single
test spawns groups of different sizes against the same `tmp_path`.

**Ranks hand results back through `tmp_path`**, `torch.save` in the worker and
`torch.load` in the parent. Assertions live in the parent process, where a failure is
readable; `assert_close(..., msg=lambda m, key=key: ...)` names the offending parameter
(bind `key` as a default argument — the closure would otherwise report the loop's last
key, which is ruff `B023`).

**Two tiers, deliberately:**

| Test | Worker | Question |
|---|---|---|
| gradients | small purpose-built loop | does the all-reduce compute the right average? |
| parameters | calls `train()` | does a real step — optimizer, clipping, autocast — stay identical? |

The gradient worker stays minimal because every line it copies from `train()` is a line
that can drift out of sync with it. The parameter test calls `train()` outright rather
than reimplementing it, which is why it needs no optimizer of its own. A one-step
parameter comparison would verify almost nothing: Adam's first update is
`m̂₁/√v̂₁ ≈ sign(g)`, so it only checks gradient *signs*. Hence 30 real steps there,
and 2 steps in the gradient worker — enough to catch state that survives across steps
(hooks firing twice, a wait that never resets) once modes 2 and 3 exist.

**Init equality is fault-injected on purpose.** `seed + rank` makes ranks genuinely
diverge before the broadcast, so `test_all_ranks_have_same_params_after_init` fails if
the broadcast is removed. Identical seeds would make that test pass whether or not the
broadcast exists — vacuous, and it would look like coverage.

**These tests only work because `dropout == 0`.** With dropout on, no seeding scheme
makes world sizes 1 and 2 bitwise comparable: one batch of 8 and two batches of 4 draw
different numbers of values in different shapes, so the RNG streams desynchronise
regardless. Assert it in the test config rather than inheriting it from a default.

## 11. Seeding policy — **load-bearing**

`torch.manual_seed(cfg.seed + cfg.rank)`, everywhere, CPU and CUDA generators alike.

- **`+ rank`**, so replicas do not draw identical dropout masks. A single batch of 8
  draws an independent mask per sequence; two ranks sharing an RNG state would give
  sequences 0–3 and 4–7 the same masks, which is not what the single-device baseline does.
- **Never `× world_size`.** That makes *rank 0's* stream a function of GPU count, so
  adding GPUs also reseeds the model and every measured difference is confounded by a
  different random init. It is the same failure as world-size-dependent data ordering
  (§7), on the parameters instead of the data. Rank *r* must always draw from `seed + r`.
- Parameter equality across ranks does **not** depend on any of this — the rank-0
  broadcast (§6) enforces it. The seed policy exists for the randomness that should
  differ, not the state that must not.

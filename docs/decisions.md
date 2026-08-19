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

**Default: work directly on aurora** (`ssh adam@aurora`, repo at `~/work/distrain`) —
editing, `uv run pytest -q`, training runs and the trackio dashboard all happen
there. Since the real FineWeb data, the GPU and the long-running jobs live on
aurora, editing anywhere else just adds a transport step.

Fallback: edit on the Mac and push over:

```bash
scripts/sync-aurora.sh                                    # rsync, well under a second
ssh adam@aurora 'cd ~/work/distrain && uv run pytest -q'
```

Git is for milestones, not for iteration — pushing and pulling per edit made the
feedback loop slow and the history unreadable. `sync-aurora.sh` excludes `data/`,
`.venv/` and outputs, and `rsync --delete` does not touch excluded paths, so
aurora keeps its own shards and environment. Careful with `--delete` when the
working tree on aurora is ahead of the Mac's, which is now the common case.

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

**The eval batch is its own config field** (`eval_batch_seqs`), not derived from
`global_batch_seqs // grad_accum_steps` as it was originally. Two reasons, and the
second is the one that would have bitten:

- Nothing in that formula names `world_size`, but the way configs are *swept* couples
  them anyway: strong scaling holds the global batch fixed and cuts `grad_accum_steps`
  as ranks are added, so the eval batch moved as a side effect of a knob turned for
  training reasons. The numerical effect is tiny (same sequences, same order, only the
  reduction order differs), but "identical across every config" should be true by
  construction, not by coincidence.
- Eval never shards — it is one rank's full pass over the val split. So the derived
  batch was the *whole* global step batch while each rank's training microbatch shrank
  with world size: at `global=480, accum=1`, training goes 480 → 60 sequences per GPU
  from 1 rank to 8 while eval stays at 480. Validation becomes the memory high-water
  mark of the run and never shrinks, so a config that trains fine OOMs in eval.

## 5. Seed variance

Single runs per config give no error bar, and time-to-target-loss run-to-run
variance can be comparable to the effects being measured. Plan: 2–3 seeds on the
single-GPU baseline and on one DDP config only, to establish a variance estimate
that is quoted alongside the rest of the (single-run) matrix. Estimated cost
$20–40; accepted.

This is about how many seeds to *run*. How a seed is derived per rank is §11.

## 6. Hand-rolled DDP is three implementations, not one

Built as runtime-switchable modes so all three are measurable on identical hardware:

1. naive per-parameter all-reduce — **done** (`ddp_naive`)
2. bucketed all-reduce — **done** (`ddp_bucketed`, `cfg.ddp_bucket_size` bytes)
3. bucketed with backward-hook compute/communication overlap — **done** (`ddp_interleaved`)

This is an extra result axis for free (no extra provisioning) and it is the most
direct demonstration of the skill the project exists to build.

All three are correctness-tested on gloo/CPU at world sizes 1 and 2, with and without
gradient accumulation. None has been *timed* against the others — that needs ≥2 GPUs,
so it is a first-cloud-session measurement, and no correctness test can distinguish a
mode 3 that truly overlaps from one that does not.

Buckets are built in **reverse registration order**, which approximates the order
backward produces gradients. At 124M this puts `wte` (154 MB, ~31% of all gradient
bytes) alone in the last bucket — it exceeds `bucket_size` so it cannot share one, and
it is also the last gradient backward produces, so mode 3 has no compute left to
overlap it against. That single bucket bounds the speedup mode 3 can show.

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
  - Modes 1 and 2 satisfy this by construction: both iterate a sequence fixed before the
    step. Mode 3 originally did **not** — it launched from hooks, so the collective
    sequence was autograd's completion order, identical across ranks only because every
    rank runs the same dense graph on the same shapes. **Resolved (2026-08-08)** the way
    real DDP does it, in two parts:
    - *Cursor.* Hooks only mark a bucket ready; `_reduce_all_ready_buckets_in_order`
      launches every consecutively-ready bucket strictly in bucket-index order. The
      launch sequence is now a function of the agreed bucket structure, never of
      autograd's scheduling. Worst case (buckets ordered opposite to completion) it
      degrades to mode-2 behaviour — performance lost, correctness never.
    - *Rebuild.* The first communicating step records per-parameter completion order;
      `_reorder_buckets` then broadcasts **rank 0's** recording and rebuilds the buckets
      from it on every rank — the analogue of DDP's `rebuild_buckets()` +
      `sync_bucket_indices()`. The broadcast is the load-bearing part: each rank's
      recording is only an observation, and adopting it locally would bake rank
      divergence into every later step. Hooks resolve their bucket through a
      `_param_to_bucket` map at fire time, so the rebuild swaps bucket dicts without
      re-registering anything.
    - Tested with a model whose execution order is the reverse of registration order:
      launch order holds on both steps, overlap is zero before the rebuild and full
      after, and the broadcast is fault-injected (rank 1's recording scrambled) so
      dropping it fails the cross-rank agreement test — mutation-verified. `wte` still
      arrives last and exceeds `bucket_size`, so no ordering gives mode 3 compute to
      overlap it against; the rebuild does not lift that bound.
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
- **Rank 0 evaluates, the loss is broadcast, and the broadcast tensor lives on the
  model's device.** Every rank running the identical full val pass is N× the cost for
  one number. `evaluate()` stays free of collectives so the call site alone determines
  rank-invariance; the broadcast lives in the synchronizer, which owns the process
  group. Rank 0 takes its own value back out of the tensor rather than keeping the one
  it put in, so all ranks compare a bit-identical float against 3.28 — otherwise ranks
  could disagree about which step first crossed, and that disagreement surfaces as an
  inconsistent result rather than a crash. The tensor is allocated on the model's
  device because **NCCL cannot broadcast a CPU tensor** and no test here can catch
  that: every multi-rank test is gloo, which accepts both.
- **Training ends with an explicit barrier before any rank tears down.** Once only
  rank 0 evaluates, the other ranks leave the final val broadcast first, destroy the
  process group and exit the process while rank 0 is still completing its side of that
  same collective. Rank 0's peer connection drops mid-teardown and gloo aborts it —
  SIGABRT, `terminate called without an active exception`, with no Python frame, after
  training has otherwise fully succeeded and printed its results. It reproduced in
  4 of 8 runs and vanished in 10 of 10 with the barrier. It belongs at the end of a
  *successful* run, not in `cleanup_distributed`: on the error path the surviving
  ranks would block there until the gloo timeout, turning one rank's crash into a hang.
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

**NCCL cannot be exercised on aurora at all.** Two ranks cannot share one GPU — NCCL
rejects a communicator with a duplicate device (`ncclInvalidUsage`) and there is no
flag around it; MPS does not help, being a scheduling layer rather than a device
multiplexer. The most that runs locally is

```bash
torchrun --nproc_per_node=2 -m distrain.train --device cuda:0 --distributed-backend gloo
```

which covers `torchrun`'s env plumbing, `cuda:{local_rank}` resolution and collectives
on CUDA tensors (gloo stages them through host memory) — everything except NCCL. Two
consequences: a 2-process NCCL job is the first thing to run on the first rented node,
before anything with a clock on it; and NCCL-only failures such as a CPU-tensor
collective cannot be caught by any local test, so they have to be designed out rather
than tested out.

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

## 12. De-risking resequenced (2026-08-08)

Everything unproven — NCCL, relative timing of the three DDP modes, image parity,
the token-budget assumption — was gated on one scarce event: the first 2×8 H100
session, while capacity has been the blocker since July. "Never debug on rented
hardware" had a loophole: first contact with NCCL *is* debugging, and it was
scheduled for the most expensive machine in the plan. Resequenced so the big
session becomes a confirmation run, in this order:

1. **Overnight real-FineWeb run on aurora.** *(Running since 2026-08-08.)* Pull the
   real shards and run to (near-)convergence locally. Validates the real data path
   end-to-end and checks the shakiest budget assumption (2–3B vs 5B tokens to 3.28,
   brief §6) for free, before any paid converged run.
2. **Fix the mode-3 launch-order gap** before renting anything. *(Done — cursor +
   measured-order rebuild, see §6.)* Completion-order launch is exactly the class
   of bug that surfaces as a hang on rented hardware — the expensive failure mode.
3. **Cheap 2-GPU session** (~$5: Vast/RunPod community, 2×3090 or 2×4090): first
   NCCL contact, time the three DDP modes against each other, verify mode 3
   actually overlaps. GeForce P2P-over-host scaling is unrepresentative, so
   nothing from this session is *reported* — it exists so the H100 session starts
   with known-good, NCCL-proven code. Relative mode comparison doesn't need
   representative hardware, and a bad interconnect makes overlap easier to see.

Consequences elsewhere:

- **Checkpointing demoted from "next" to when-needed.** It only serves spot
  preemption on Track A (Track B's ~40-step configs never converge), and
  on-demand fits the budget. A basic single-file `torch.save` save/resume now
  exists (`--checkpoint-every` / `--resume`, rank 0, atomic via `os.replace`) so
  a local overnight run survives an interrupt; resume is exact because the loop
  is a pure function of the step index once params + optimizer state are
  restored (dropout 0 → no RNG drawn; data order and LR derive from the step —
  which also means resume must reuse the same command line, `max_steps`
  included, or the LR schedule silently changes). DCP arrives with FSDP2's
  sharded states, and the preemption hardening (§9) waits until spot is
  actually chosen — which it may never be.
- **Track A plans around a single 8-GPU node.** `NCCL_P2P_DISABLE=1
  NCCL_SHM_DISABLE=1` forces socket transport intra-node and netem shapes it. A
  proxy for real inter-node TCP, but the x-axis is measured `all_reduce_perf`
  bandwidth anyway (§9), which is what makes a proxy transport legitimate. A
  2-node slot, if one appears, validates a few points on the curve instead of
  gating the whole plot. Single 8-GPU nodes are far easier to rent than 2×8.
- **Track B shrinks to its §9 fallback by default.** FSDP2 on one 8-GPU node is
  the plan (genuinely forced at ~7B); torchtitan comparison if time allows;
  TP8×DP2 only if 2-node capacity falls into our lap.
- **DiLoCo is scope-boxed.** Run at published hyperparameters, one config,
  labeled "untuned DiLoCo". Its time-to-target depends heavily on tuning (inner
  steps, outer LR/momentum) — a time sink and a fairness confound; tuning it is
  explicitly out of scope.

## 13. Track A model modernized with early modded-nanogpt improvements (2026-08-09)

**Trigger.** The first full-length local run (vanilla architecture, 3B tokens)
ended at val **3.50** with a flat tail — vanilla needs ~10B tokens to 3.28
(consistent with llm.c), 3× the assumed cost per converged run.

**Why this is allowed.** The load-bearing rule is *architecture identical across
configs* (CLAUDE.md), not *architecture equal to GPT-2*; 3.28 is defined by the
tokenizer and val slice, both untouched. What is given up is comparing our
absolute times to llm.c's 45 min — which different hardware already precluded.

**Adopted** (the record #2/#5/#8-era changes — the cheap, standard segment of the
speedrun history; est. ~2.2× fewer tokens, to be *measured* by a calibration run
before anything is rented):

- Trapezoid LR (linear warmup, plateau, linear warmdown to zero), peak 0.0018.
  `min_lr` is gone; `lr_at` asserts `warmup + warmdown < max_steps` so a run too
  short for its schedule fails loudly. The resume caveat shrinks: only the
  warmdown position depends on `max_steps` now.
- QK-norm (parameterless `rms_norm` over head_dim), ReLU² MLP.
- Zero-init residual projections (`c_proj`) and the **untied**, zero-init
  `lm_head` (`use_weight_tying` in `GPTConfig`, default off). Untying adds ~39M
  params ≈ +25% FLOPs/token; the MFU counter sees actual parameters, so the
  accounting follows automatically.

**Rotary embeddings — done (2026-08-10)**, completing the pack (the biggest
single early lever, ÷1.43 together with the LR tuning). Early-record design:
half-split rotation applied to q and k after QK-norm, base 10000, cos/sin
precomputed in fp32 for `block_size` as non-persistent buffers (static graph
under `torch.compile`, absent from checkpoints). `wpe` is gone, so
`num_params()` lost its `non_embedding` flag — there are no positional
parameters left to exclude. Tests pin the properties rotary is for: norm
preservation, position-0 identity, scores a function of relative position only,
and a 1-layer prefix-permutation test that fails if rotary is dropped (with
`wpe` gone the model would be position-blind). **Skipped deliberately:** Muon (cut for schedule reasons,
revisitable) and everything past record ~#9 — value embeddings, FlexAttention,
FP8, custom kernels — per the brief's no-micro-optimizations rule.

**Three port bugs, kept as lessons** (all found before any paid run):

- modded-nanogpt's `get_lr` is a **LambdaLR multiplier**; assigning its return
  value directly as the LR trained at 1.0 on the plateau — a 556× overshoot and
  an exploding loss. The old LR tests used `learning_rate=1.0`, the one value at
  which multiplier and LR are indistinguishable; the rewritten tests pin an LR
  ≠ 1.0 and a regression test names the failure.
- Zero-init done in submodule constructors was **silently undone** by
  `GPT.__init__`'s `apply(_init_weights)` and the residual-scaling pass after
  it. `apply` hands modules to the init fn without names by design; the
  canonical fixes are a marker attribute checked inside `_init_weights`, or a
  named `named_parameters()` post-pass after `apply` — the file already had one
  for the GPT-2 1/sqrt(2L) scaling, which zero-init supersedes, so it lives
  there now. `TestZeroInit` asserts the *finished* model, precisely because a
  constructor-time check passes while the model ships gaussian weights. Never
  zero a tied head — it is `wte`.
- `relu(x²)` instead of `relu(x)²`: after squaring, everything is non-negative,
  the ReLU becomes an identity and the activation turns symmetric — a silent
  quality bug, the failure class this project fears most.

Checkpoints from before this change no longer load (untied head, renamed
schedule fields). Expected and accepted: nothing durable had been trained.

**Sanity run + controlled rotary A/B (2026-08-10).** The 500-step sanity run
(`rotary-sanity-500`: warmup 100 / warmdown 100, real FineWeb) finished at val
4.89 — *above* the earlier `3090-fineweb-3B-modded` run's mid-plateau 4.57 at
the same step, which looked like a rotary regression. It wasn't: that baseline
ran a pre-commit working-tree state (its step-0 val of 10.68 proves a non-zero
head; the zero-init head forces exactly ln 50304 = 10.826), and mid-schedule
loss comparisons across different LR schedules are confounded anyway. A control
run of the committed no-rotary state (`norotary-sanity-500`, identical
seed/schedule/data — the only delta was the rotary commit) settled it: **4.89
vs 5.63 at step 499 — rotary is 0.74 ahead**, with the gap growing at every
checkpoint. Two lessons kept: (1) only matched-code, matched-schedule runs are
comparable — dashboard curves across code states mislead; (2) the zero-init
head deliberately trades early loss (it must grow from zero) for the
long-horizon win, so short-horizon comparisons penalize it by construction.

**Calibration result (2026-08-13).** `rotary-calibration-3B`, 6000 steps /
2.95B tokens, full trapezoid: final val **3.333** — the 3.28 target was *not*
crossed (vanilla measured 3.502 at the same token count). The run was
interrupted at step ~4990 and resumed from the 4750 checkpoint; the resume
omitted `--run-name`, so its last 1250 steps are logged under
`3090-fineweb-3B-modded` on the dashboard — the segments are one run (the
checkpoint's `wpe`-less state dict only loads on rotary code, and the val
curve is continuous). Two operational notes from this: pass `--run-name` on
every resume, and `checkpoints/ckpt.pt` is a single shared path — a new run
silently overwrites the previous run's checkpoint.

**Continuation outcome (2026-08-16).** The continuation (6000 → 9000 from the
step-6000 checkpoint) behaved exactly as predicted: re-entry bump to 3.464,
plateau grind back to 3.426 by step 8000, then the warmdown — 3.395 @ 8250,
**3.355 @ 8500** (4.18B tokens), per-250-step drops accelerating (−0.031,
−0.040) with 9.0e-4 of LR still to anneal. The process died at step 8650
(Claude Code session teardown killed the detached process — twice, `setsid
nohup` included) and the last 350 steps were abandoned rather than re-resumed.
Extrapolating the warmdown puts the 9000-step val at ~3.26–3.29: the crossing
sits right at the end of the composite schedule. Verdict, recorded in place of
a measured crossing: **tokens-to-3.28 is >2.95B (a clean 6000-step schedule
ended at 3.333) and ≈4.4B under a composite schedule that a clean one beats.**
The converged-run recipe is therefore a clean 9000-step trapezoid — 4.42B
tokens, expected to cross at or shortly before its end — versus ~10B for
vanilla: the modernization delivered the estimated ~2.2×.

**Budget arithmetic** (with N = 162.2M untied → 1.087e9 FLOPs/token by the §3
convention): one converged run is 4.42e9 × 1.087e9 ≈ **4.8e18 FLOPs**. On
8×A100 (312 TF dense each) at an assumed 35–45% MFU that is 1.3–1.7 h of node
time — **$15–27** at RunPod's ~$10–15/h. Cross-check from aurora: the 3090 run
took 19h at 84.7% MFU; an 8×A100 node has ~30× the peak, so ~1.4 h at 40% MFU.
Roofline + nccl-tests + the 4-mode × compile bench add ~1–2 h more. The first
8-GPU session, converged run included, fits in **~$30–55** — comfortably
inside the $150 target even with a retry.

**Next, in order:** first larger-GPU session — single 8-GPU node, *attended*
(the §9 kill-switch launcher gates unattended runs only, matching the 2×3090
precedent): roofline + `nccl-tests`, image-parity check, `bench_ddp_modes`
across the four modes × compile, then the first converged Track A run at
9000 steps → DiLoCo + netem after that. *(Executed 2026-08-16 — §14.)*

## 14. First 8×A100 session: tokens-to-3.28 measured (2026-08-16)

Full narrative in the [session log](sessions/2026-08-16-runpod-8xa100.md);
what supersedes §13's bracket:

- **Tokens-to-3.28 = 4.92B, measured**: first unsmoothed crossing at step
  9999 of a clean 10000-step trapezoid (val 3.2730), **3147.1 s** of
  training time on 8×A100-SXM4 at ~79% MFU. The clean 9000-step schedule
  ends at **3.2849 / 4.42B tokens** — short by 0.005: the §13 extrapolation
  (~3.26–3.29) was right and the 9000-step recipe sat on the wrong side.
  The converged-run recipe is now **10000 steps / 4.92B tokens**.
- **A 9000→10000 plateau extension is a clean run, not a composite.**
  Resuming the warmdown-start anchor (step 8000) under `--max-steps 10000`
  reproduces the clean longer schedule exactly — LR and data are functions
  of the step, and the 0–8000 history is shared by construction. This is
  the cheap recovery §13's post-warmdown continuation was not: re-enter at
  the *last checkpoint where LR was still at peak*, never after an anneal.
- **Checkpointing grew retention + mirroring** (prompted by a real loss:
  RunPod *terminates* pods when account credit drains, and the first pod
  died at ~step 7800 with the run's only `ckpt.pt`): per-step files
  `{run}-step{N}.pt`, rolling `keep_last` + permanent `keep_every` anchors
  (default 2000 = warmdown start), `--resume-from`, and an async off-box
  mirror (skip-if-busy, atomic, drained at exit). Operational rules that
  came with it: monitor the provider balance during paid runs, and mirror
  continuously off the pod — container disks do not survive termination.
- **Compile × overlap is a per-transport decision** (the §13 open question):
  on NVLink (154 GB/s bus) every compiled config beats every uncompiled one
  — `ddp_torch --compile` fastest, 0.88 scaling at 8 ranks — while
  uncompiled interleaved is best-in-class uncompiled (0.94). On Socket+SHM
  (2×3090) uncompiled interleaved beat everything. Track A fast-transport
  configs run `ddp_torch --compile`; expect the lead to flip back under
  netem, and re-verify with one bench rerun there.
- **Replication for free**: aurora 1×3090 (accum 60) and two 8×A100 DDP
  runs agree at every shared val point to ≤0.01 — the §7/§11
  world-size-independence invariants demonstrated on real data, not just
  gloo tests.
- **Budget actuals**: session ~$82 (~2.2 h + ~4.2 h of node time at
  $12.72/h; ceiling was $80, the ~$3 extension overshoot approved for the
  crossing). §13's estimate said $30–55: the overrun bought a credit-death
  lesson, a checkpoint system, and both the 9000- and 10000-step schedule
  endpoints. Project spend to date ≈ $83 of the $150 target.

## 15. The netem curve: converged runs only where the math changes — **load-bearing** (2026-08-17)

The naive netem plan — a converged run per (bandwidth × method) point — does
not fit the remaining budget and is not necessary:

- **DDP's val-vs-step curve is transport-invariant.** netem changes how long
  an all-reduce takes, never what it computes; data order and LR are pure
  functions of the step (§7, §13). The §14 replication (three runs, different
  hardware *and* chunkings, ≤0.01 at every shared val point) bounds the
  residual reduction-order noise. So the crossing stays at step 9999 — up to
  ±one val interval, since the crossing margin (3.2730 vs 3.28) is inside
  that ±0.01 band — and **DDP time-to-3.28 at bandwidth *b* is
  9999 × steady-state step time measured at *b***. A step-time measurement
  costs minutes per point; a throttled converged run costs hours (step time
  grows several-fold as comm dominates).
- **Guard per setting**: a few hundred steps at one throttled bandwidth,
  checking the early val gates (≈5.40 @ 250, ≈4.50 @ 500), confirms netem
  moved nothing but the clock. Cheap, and it catches a transport-dependent
  NCCL algorithm switch actually changing numerics.
- **Converged runs are reserved for methods that change the math.** DiLoCo's
  loss-vs-tokens curve genuinely differs from DDP's, so it needs (one or two)
  converged runs to anchor its tokens-to-target — but its wall-clock is
  roughly bandwidth-insensitive by construction (communication every H
  steps), so step-time measurements extend its point across the bandwidth
  axis the same way.

Consequence: the whole curve is one attended session — bench-style step-time
sweeps for DDP across netem settings, one sanity-gated throttled segment, and
DiLoCo's converged anchor — instead of a matrix of converged runs.

## 16. DiLoCo spec (2026-08-17)

**Algorithm: original DiLoCo ([arXiv 2311.08105](https://arxiv.org/abs/2311.08105)),
untuned, per the §12 scope-box.** Rejected alternatives, both for the same
reason (their contribution doesn't exist in this study's setting):
*Streaming DiLoCo* (2501.18512) syncs parameter fragments on a schedule with
overlap and 4-bit outer gradients — a peak-bandwidth optimization, write-up
mention only. *Decoupled DiLoCo* (2604.21428) makes the sync asynchronous via
a CPU-side synchronizer with quorum and grace windows — a fault-tolerance
system whose point requires stragglers/failures/multi-region; on one netem
node it degenerates to synchronous DiLoCo plus unused machinery.

**The algorithm.** K replicas start each round with identical params θ. Each
runs H inner AdamW steps on its own data shard, ending at θᵢ. The outer
gradient is Δᵢ = θ − θᵢ (the round's displacement, sign-flipped so it points
like a gradient); replicas all-reduce it, and the mean Δ is applied to the
round-start θ by an outer SGD-with-Nesterov-momentum step. Inner optimizer
state (Adam m, v) is per-replica: never communicated, never reset —
resetting forces Adam's cold start every round, and averaging would double
the comm volume to synchronize statistics that track each other anyway.

**Hyperparameters — published values, labeled "untuned DiLoCo" (§12):**
H = 500, outer lr 0.7, Nesterov momentum 0.9. Inner optimizer and trapezoid
schedule identical to the DDP configs (the paper used cosine; one honest
sentence in the write-up). Per-rank batch stays 60 seqs, so global token
throughput matches the DDP config exactly — but there is no gradient
averaging, so each replica's effective batch is 60, and K× more optimizer
steps happen per token globally. That is the method difference being
measured, not a confound.

**Mode design.** `diloco` is a fifth `DistributedSynchronizer` mode:

- `finalize_gradients()` is a **no-op**: no collective, and **no
  `/ world_size`** — a deliberate carve-out from §6, since nothing was
  summed across ranks. The accumulation divisor stays in the loop, untouched.
- One new seam call after `optimizer.step()`: `maybe_outer_step(step)`. At
  every H-th optimizer step (never micro-step): compute Δ per rank iterating
  parameters in fixed order (§6 collective-order discipline), all-reduce
  `SUM`, divide by K, outer-step the round-start params, copy the result
  into the model on every rank. Every rank computes the identical update
  from identical inputs, so replica equality is preserved by construction;
  the §6 init broadcast anchors round 0. For the other four modes the call
  is a no-op, so the training loop stays mode-blind.
- The outer optimizer is `torch.optim.SGD(nesterov=True, lr=0.7,
  momentum=0.9)` over a copy of the params, with the averaged Δ assigned
  into `.grad` — no hand-rolled momentum. Cost: two extra model-sized
  tensors per rank (round-start params + momentum buffer), trivial at 162M.
- **Checkpoints are per-rank in `diloco`** — the rank-0-only file is no
  longer complete state. It suffices for DDP because post-reduce gradients
  are identical on every rank, so every rank's AdamW state evolves
  identically; `diloco` breaks exactly that premise: each rank's inner
  (m, v) genuinely differs, and mid-round so do its params. Restoring
  rank 0's state onto every rank is a silent trajectory change, not a
  resume. So each rank saves `{run}-step{N}-rank{r}.pt` (its replica params
  + its inner optimizer state); the outer state (round-start params +
  momentum buffer) is identical everywhere by construction and lives in
  rank 0's file only. With that, resume is exact at any step, mid-round or
  boundary — round structure stays a pure function of the step index.
  §12's resume rule extends: H, the outer hypers *and the world size* must
  match the original command line — a `diloco` checkpoint set is
  K-specific. Retention and mirroring treat a step's K files as one unit.

**Validation and the crossing under DiLoCo (convention settled 2026-08-17).**
Mid-round, replicas have genuinely diverged, so rank 0's val loss measures
*rank 0's replica*, not "the model" — the shared model only exists just after
an outer step. This is not merely a labeling problem: the ≤ 3.28 check runs
in the loop at every eval, so a mid-round eval of one lucky replica could
cross before the synced model does and silently corrupt the headline metric.

The eval cadence itself is **mode-independent** (`step % val_every == 0`
plus the final step): the eval schedule is part of the measurement, and the
measurement must not vary with the communication mechanism being measured.
To make boundary vals see the shared model under that cadence, the outer
step fires at **`step > 0 and step % H == 0`** (and on the final step) — the
same iterations the val cadence uses, executed before the val block — with
`val_every % H == 0` required at startup (`lr_at`-style, fails loudly). Two
consequences, accepted: the *first* round is H+1 inner steps rather than H
(the price of aligning to a val grid that starts at step 0), and the
**step-0 val is rank 0's replica after one inner step** — the same one-step
init fingerprint every DDP run prints (data-independent ~10.8 under the
zero-init head), and incapable of recording a real crossing. Every val from
step H onward measures the freshly synced params. A per-replica diagnostic
eval can be added later behind an explicit flag; it must never share the
reported metric's path. The coarser crossing resolution (multiples of H) is
a property of the method.

**Invariants — load-bearing:**

- No `/ world_size` anywhere in the `diloco` per-step path.
- Outer-step collectives iterate parameters in fixed order, unconditionally
  (§6).
- **Never compare inner optimizer state across ranks** — (m, v) are
  *expected* to differ; the invariant is parameter equality at round
  boundaries only. A test asserting state equality would be asserting a bug.
- Data sharding is unchanged (§7): each replica consumes its own
  world-size-independent stream, so token accounting stays comparable
  with DDP.

**Test matrix (gloo/CPU, §10 harness):**

1. *Identity:* K = 1, outer lr 1.0, momentum 0 → **bitwise** equal to plain
   `train()` — the outer step reduces to copying θᵢ back over θ, so any
   deviation is delta/copy plumbing, not tuning.
2. *Degenerate averaging:* K = 2 with both ranks fed identical data →
   identical deltas → equal to the K = 1 run.
3. *Replica equality at boundaries*, fault-injected per §10: scramble one
   rank's delta before the reduce and the cross-rank agreement test must
   fail — otherwise it is vacuous.
4. *Round arithmetic:* the outer step fires exactly at positive
   optimizer-step multiples of H (never at step 0) and on the final step,
   including under gradient accumulation (H counts optimizer steps, not
   micro-batches).
5. *Resume:* checkpoint mid-round and at a boundary; both continue
   bit-identically **on every rank** — specifically rank ≠ 0's inner Adam
   state must round-trip, because a rank-0-only restore passes a
   parameter-equality check and still diverges.
6. *Eval placement:* `val_every % H != 0` fails at startup; at a boundary
   step the eval observes the post-outer-step (synced) params, not the
   pre-outer-step replica.

**Open measurement:** tokens-to-3.28 under untuned DiLoCo — expected above
DDP's 4.92B; how much above *is* the result. One converged anchor in the
netem session (§15), extendable via checkpoint anchors if the first schedule
guess lands short, exactly as §14 did.

## 17. netem cannot run on RunPod — measured (2026-08-18)

§9 recorded that netem needs `NET_ADMIN` on the container and filed it as "a
provisioning flag to remember." There is no such flag. Measured on a rented
1×3090 pod (EU-CZ-1, $0.50/h, ~10 min, ~$0.09), booting the pinned image
through `scripts/runpod_session.py`:

- `CapEff: 00000000a80425fb` — Docker's default set. `cap_net_raw` is present,
  **`cap_net_admin` is not**.
- `tc qdisc add dev eth0 root netem rate 100mbit delay 10ms` →
  `RTNETLINK answers: Operation not permitted`.
- The namespace workaround is closed too: `unshare -U`, `-n`, `-Ur`, `-Urn`
  and `-m` all fail with `Operation not permitted`, so a private netns whose
  `lo` we *could* throttle cannot be created. (`kernel.unprivileged_userns_clone`
  is 1, so this is the runtime's seccomp policy, not the kernel's.)
- The API has no way to ask for it: `capAdd`, `privileged`, `capabilities` and
  `securityOpt` are all rejected as unknown fields by
  `PodFindAndDeployOnDemandInput`, and the REST pod schema has no equivalent.

**Consequence for §15.** The netem half of the curve — DDP step-time sweeps
across bandwidths and the throttled sanity segment — has no home on RunPod
pods. DiLoCo's converged anchor is unaffected: it needs no throttling, so the
next session's headline measurement stands as planned. Options for the curve,
undecided: run it on `aurora` in Docker with `NET_ADMIN=1` (already supported
by `scripts/container.sh`; 1–2 GPUs, so it measures transport shape rather
than the 8-GPU anchor), or find a provider that permits privileged containers.
Deciding this is a prerequisite for the netem session, not for the DiLoCo one.

**Pricing, also measured.** `lowestPrice` from the GPU-type query is the
*community* rate and is only an availability signal; a SECURE pod bills
`securePrice × gpuCount` — the 3090 pod billed $0.50/h against a $0.22
`lowestPrice`, and 8×A100 is 8 × $1.59 = $12.72/h, exactly what §14 paid.
`runpod_session.py` budgets on `securePrice`. Account billing also lags a
termination by ~75 s, so `verify` re-reads the spend once before calling a
non-zero figure a leak.

## 18. Inner batch size is a DiLoCo knob, at fixed token budget (2026-08-18)

§16 fixes the global batch at 480 and *splits* it across K islands, so at K=8
each island trains on 60 sequences. Two different things can be meant by
"give each island a bigger batch", and they cost wildly different amounts:

- **The paper's regime (not adopted).** Each worker carries a full batch and
  the *step count is held fixed*, so total tokens scale as K x. A converged
  run at K=8 would be ~10000 steps x 3840 seqs x 1024 = **39B tokens**, four
  epochs of FineWeb10B and ~8x the DDP anchor's wall clock (~7 h on 8xA100,
  ~$89). Out of budget, out of corpus.
- **Bigger inner batch at fixed tokens (the variant worth running).** Hold the
  token budget at the DDP anchor's 4.92B and raise per-island batch `b`. The
  step count falls out of it: `S = T / (K*b*1024)`. Raising `b` 8x cuts S 8x,
  and with H fixed it cuts the number of outer steps 8x too. **Same data
  budget, same throughput, therefore the same wall clock and cost as the DDP
  anchor.**

**Why it should stabilise the outer step.** From the K analysis: the averaged
outer gradient's noise is `sigma_dbar ~ lr*sigma*sqrt(H/(K*b))`. It is
independent of K (a smaller per-island batch exactly cancels the sqrt(K)
averaging gain) but *not* of `b` -- raising the inner batch cuts outer-gradient
noise by sqrt(b). Since Nesterov at momentum 0.9 with outer lr 0.7 amplifies
the mean delta by up to `lr/(1-mu) = 7x` at steady state, feeding it a quieter
delta is the one lever that reduces overshoot without touching the outer
hyperparameters -- i.e. without breaking §12's "untuned DiLoCo" box.

**The trade, which is real and not a flaw.** At fixed tokens and fixed K you
cannot match DDP on both batch size and update count:

| config | per-island batch | updates per replica | global batch |
|---|---|---|---|
| DDP anchor | 480 | 10000 | 480 seqs |
| A (§16, implemented) | 60 | 10000 | 480 seqs |
| bigger-b variant | 480 | 1250 | 3840 seqs |

Config A matches DDP's step count at an eighth of its batch; the variant
matches DDP's batch at an eighth of its updates. At b=480 the global batch is
3.9M tokens/step, almost certainly past critical batch size for a 162M model,
so token efficiency would fall for reasons that have nothing to do with
DiLoCo. **Test intermediate values** -- b in {120, 240} gives 5000 / 2500
steps at the same 4.92B budget and stays in a sane batch regime -- rather than
jumping to the extreme and confounding two effects.

**What it buys the write-up.** The aurora K=2 smoke shows a sawtooth (gap
1.138 -> 0.489 -> 0.353 -> 0.532 -> 0.728 -> 0.298 at steps 500...3000, with a
descending envelope: 4.1396 @ 1500 -> 3.8804 @ 3000). That cannot distinguish
*"DiLoCo oscillates"* from *"published hyperparameters are mis-tuned for a
regime they were never fitted to"*. If the sawtooth flattens as `b` rises at
constant tokens, the oscillation is a regime artefact, and §12's "untuned
DiLoCo" label needs the batch regime stated beside it or it misleads.

**Corpus ceiling, independent of all this.** `DataLoader.sequence` wraps with
`% num_sequences`, so a run that outlasts its shards silently repeats data
rather than failing. At global batch 480 the full FineWeb10B corpus is
**~20,345 steps**; the 55-chunk subset the §14 runbook fetches wraps at
~11,190. Config A at the observed ~2.5x token ratio would need ~24,900 steps
to reach DDP's crossing -- past the corpus. Any run crossing that line must
disclose the second epoch, not absorb it. Raising `b` also raises this
ceiling in steps, since each step consumes more of the corpus.

## 19. The aurora excursion probe: two arms, one variable each (2026-08-19)

§18 framed the inner batch as `b` at K=8, where the §16 split gives b=60. The
aurora smoke runs K=2, so `b` is already 240 and the noise term collapses:
`sigma_dbar ~ lr*sigma*sqrt(H/(K*b))` and `K*b` **is** `global_batch_seqs`. At
fixed K the only knobs are H and the global batch, so "raise b" on aurora means
"raise the global batch and cut H to match". §18's `b in {120, 240}` is a K=8
prescription and does not transfer.

Two arms, each changing exactly one thing against the `diloco-smoke-b480`
baseline, both at the baseline's token budget and wall clock:

| arm | global batch | H | steps | rounds | outer moment | sigma_dbar |
|---|---|---|---|---|---|---|
| baseline (have) | 480 | 500 | 6000 | 12 | 0.9 | 1.00x |
| A `diloco-b480-mom05` | 480 | 500 | 6000 | 12 | **0.5** | 1.00x |
| B `diloco-b960-h250` | **960** | **250** | **3000** | 12 | 0.9 | **0.50x** |

Arm B holds tokens per round (240k seqs), round count, total tokens and wall
clock all identical to the baseline; only the inner batch and the inner step
count per round change, and `sqrt(H/B)` falls exactly 2x. Its trapezoid is
scaled with the step count (warmup 125, warmdown 500) so the LR multiplier
matches the baseline at equal token positions. Arm A keeps the baseline's
6000-step schedule verbatim, so its LR prefix is identical step for step.

**Inner LR is deliberately not scaled in arm B.** Doubling the batch at fixed
`learning_rate` leaves it mildly under-tuned per token, which biases *against*
the arm -- if the excursion damps anyway, the result is conservative. Scaling it
would confound the batch-size effect with an LR change.

**Why momentum is the other arm.** A sawtooth is a momentum signature; outer LR
alone would overshoot proportionally, not oscillate. At mu=0.5 the steady-state
amplification `lr/(1-mu)` is 1.4x instead of 7x. This deliberately leaves §12's
"untuned DiLoCo" box -- that box is already measured, and the rented K=8 run
needs chosen hyperparameters, not published ones.

**Both arms run `--diag-val-every 250`.** With H=500 that yields a mid-round and
a pre-sync point per round, so replica spread -- the quantity the noise algebra
is about -- is observed directly rather than inferred from the post-sync curve.
Cost is ~4% of wall clock and it never feeds the 3.28 check (§16).

**Sequential, not parallel.** One 3090, ~8.7 s/step either way: ~7.3 h to reach
step 3000 (where the baseline's excursion peaked and recovered), ~14.5 h for a
full schedule. `--checkpoint-every 500` on both, so the endpoint survives an
interruption -- the baseline's was lost to its omission.

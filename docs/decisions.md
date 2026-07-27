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
| `aurora` (local, RTX 3090) | All CUDA correctness work | pinned Docker image |
| Rented cloud nodes | All reported results | same pinned Docker image |

The Mac must be able to run a tiny config end-to-end (small `n_layer`/`n_embd`,
short sequence, CPU or MPS) so the loop, data path, checkpointing and the
`gloo` multi-rank tests are exercisable without leaving the editor. It cannot
produce any performance number, and no `torch.compile`/bf16 claim from the Mac
transfers anywhere.

Docker is **not** used on the Mac — a CUDA image is meaningless on arm64 without
an NVIDIA GPU. Identical-image reproducibility is a claim about `aurora` and
cloud only, which is where every reported number comes from.

## 2. Dependency pinning

`torch==2.13.0`, `numpy==2.5.1`, `tiktoken==0.13.0`, `trackio==0.33.0`, exact-pinned
in `pyproject.toml` with a `uv.lock` covering transitives. trackio in particular is
pinned because its API is young (brief §3).

torch resolves from PyPI on macOS and from the **cu126** wheel index on Linux.
cu126 rather than the newer cu130/cu132 because it has the widest NVIDIA driver
compatibility and `aurora`'s driver version is not yet confirmed. Once it is,
bump if the driver allows — but bump *before* the first cloud run, never between
runs that get compared to each other.

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
  H100 SXM = 989 TFLOP/s, *not* 1979. RTX 3090 ≈ 35, *not* 71.
- **Implementation**: one unit-tested function, with peak TFLOP/s in a table keyed
  by device name. Never hand-entered per run.

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

## 6. Hand-rolled DDP is three implementations, not one

Built as runtime-switchable modes so all three are measurable on identical hardware:

1. naive per-parameter all-reduce
2. bucketed all-reduce
3. bucketed with backward-hook compute/communication overlap

This is an extra result axis for free (no extra provisioning) and it is the most
direct demonstration of the skill the project exists to build.

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

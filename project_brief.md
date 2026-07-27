# Distributed Training Scaling Study — Project Brief

A self-funded ML project to gain hands-on multi-GPU / multi-node training experience,
producing a portfolio artifact for an ML-engineering / infra job search.

---

## 1. Goal

Reimplement a nanoGPT-style pretraining loop with a distributed training layer written
from scratch, then empirically characterize how training scales from 1 GPU to multiple
GPUs and multiple nodes — measuring **both** raw throughput scaling and time-to-target-loss,
and quantifying where and why they diverge.

This is honestly framed as an **engineering / systems study**, not a novel-method claim.
The primary skill gap being filled is **multi-GPU / multi-node distributed training**.

### Why this project
- Multi-GPU is the *point* of the task, not a bolt-on.
- Inherits a free, externally-defined baseline (the modded-nanogpt 3.28 val-loss target).
- Self-contained: no RAG, no RL, no external eval harness.
- Clean job-search story: "measured where DDP breaks down over commodity networking and
  what recovers it" beats "I fine-tuned a model."

### Explicitly NOT doing
- Not chasing the speedrun leaderboard record (years of work by strong people; second place at best).
- Not attempting a novel optimization technique.
- No RAG / RL (saved for a separate future project).

---

## 2. Two-track structure

**Track A — small model, converged, benchmark-comparable.**
- 124M params (GPT-2 small), FineWeb, GPT-2 BPE tokenizer, 3.28 val-loss target — all
  unchanged so results stay comparable to the leaderboard.
- DP only. Measure time-to-target-loss across fast interconnect, slow interconnect, and
  low-communication methods (DiLoCo).
- Produces the headline comparable number.
- **This track alone is a complete, presentable project.** Build it first; treat Track B
  as a planned extension so an overrun costs the extension, not the deliverable.

**Track B — large model, throughput only, NEVER converged.**
- ~7B params — large enough that model states (~112 GB at 16 B/param) exceed one 80 GB H100,
  so FSDP2 is *required*, not decorative. TP becomes warranted.
- Report MFU and scaling efficiency across parallelism strategies. ~40 steps per config
  (~10 warmup + ~30 measured) is enough for steady-state step time.
- Not comparable to the benchmark, and that's fine — MFU-across-configs is a standard,
  self-contained result.

### Key scaling concepts
- **Strong scaling**: fix global batch, split across GPUs. Clean infra measurement; hits
  comms-bound diminishing returns fast (a legitimate finding).
- **Weak scaling**: grow global batch with GPU count. More realistic; introduces LR-rescaling
  and critical-batch-size optimization confounds.
- Measure both, on two metrics: throughput/MFU (clean engineering) and time-to-target-loss
  (the real thing). **The gap between those two curves is the result.**

### Parallelism topology (the production pattern, for Track B)
- TP has the highest comms cost → confine it **intra-node** (NVLink/NVSwitch, ~900 GB/s).
- DP/PP have lower comms cost → carry the **inter-node** work.
- Rule: keep TP ≤ GPUs sharing fast intra-node interconnect. `TP8 x DP2` on 2×8 is the
  canonical shape.

---

## 3. Key technical decisions (settled)

| Area | Decision | Rationale |
|---|---|---|
| Base code | Start from original **nanoGPT** (or earliest `records/` files in modded-nanogpt) | Smallest thing that works; effort goes into the distributed layer, not absorbing others' architecture |
| Approach | From-scratch pretraining | Matches the skill gap |
| Dataset | **FineWeb** (NOT FineWeb-Edu), pre-tokenized GPT-2 BPE shards via `cached_fineweb10B.py` | No tokenization job; keeps comparability |
| Model A | 124M | Benchmark-comparable |
| Model B | ~7B | Forces FSDP2 to be genuinely necessary |
| Attention | `F.scaled_dot_product_attention` | Flash kernels for free, no dependency |
| Distributed | Hand-rolled `torch.distributed` **DDP first**, then FSDP2 (Track B) | Learn the layer before abstracting it |
| Library comparison | Then compare against **torchtitan** (PyTorch's own FSDP2/TP/PP reference) | Modern, readable; a real analysis axis. NOT DeepSpeed (ecosystem moving to FSDP2; equivalent at math layer) |
| Orchestration | **SkyPilot** (multi-cloud capacity search + managed spot w/ auto-recovery) | Availability is the binding constraint; spot recovery makes the budget work |
| Job launch | `torchrun --nnodes=2 --rdzv_backend=c10d` | Slurm/K8s are schedulers; unnecessary for one user / two nodes |
| Tracking | **trackio** (`import trackio as wandb`) | Local-first, free, optional public HF Space dashboard for the README. Pin the version (young API) |
| Checkpointing | `torch.distributed.checkpoint` (DCP) | Required for spot preemption recovery |
| Data loader | ~50-line memmap `.bin` shard reader | No HF `datasets` dependency |
| Env | Pinned Docker image + `uv`, same image locally and in cloud | Unpinned env silently invalidates cross-provider comparisons |

### NOT in the stack
HF `transformers`, `accelerate`, DeepSpeed, `datasets`, Lightning, Kubernetes.

---

## 4. Comparability — fragile in exactly two places
- **Tokenizer** and **validation split** define the 3.28 number. Use both verbatim; do not
  "improve" the tokenizer.
- **Shard data deterministically by global token index**, independent of world size.
  Otherwise changing GPU count changes data ordering and confounds every
  time-to-target-loss comparison — silently.
- FineWeb ≠ FineWeb-Edu (different filtering, different loss scale, not interchangeable).

---

## 5. The slow-network trick (cheaper AND better science)
Don't rent a bad network. On the *same* rented cluster:
- `NCCL_IB_DISABLE=1` forces NCCL onto TCP sockets.
- `tc qdisc netem` imposes arbitrary bandwidth caps and latency.

Identical hardware, transport as the only variable → plot time-to-target-loss vs. bandwidth
as a continuous curve, showing exactly where DDP falls off and DiLoCo takes over.

---

## 6. Budget (target $150, hard ceiling $400)
Full plan ≈ **$155** on current assumptions; local debugging on the 3090 drops it further.
- Track A converged runs: ~$6–21 each (~$81 total). A 124M run to target ≈ 28 GPU-min on 8×H100.
- Track B grid: ~$2–5 per config (~$25 total).
- **Biggest cost driver is launch overhead**, not training — batch Track B configs into one
  provisioned session.
- **Biggest lever is debugging discipline**: 25h of debugging is $10 on a cheap card vs
  ~$1,200 on a 2-node H100 cluster. Never debug on rented hardware.
- Shakiest assumption: Track A token budget. Evidence (runs hitting ~3.278 at 0.9B–1.7B
  tokens) suggests ~2–3B, not the 5B originally guessed. First converged run resolves this —
  run it early. (See `gpu_budget_model.xlsx`.)

---

## 7. Providers (self-serve, hourly, no sales call)
- **RunPod Clusters** — 2 nodes / 16 GPUs on demand, per-second billing, IB/RoCE, native Slurm. *(No capacity as of Jul 2026.)*
- **Prime Intellect** — compute exchange across ~12 clouds; good price-comparison layer. *(No capacity as of Jul 2026.)*
- **Nebius self-service** — up to 32 GPUs instantly, $25 min top-up, Quantum-2 IB, managed Slurm+K8s; also offers cheaper L40S / RTX PRO 6000. **Not yet checked — try this next.**
- Rule out Lambda 1-Click (16-GPU floor + 2-week minimum ≈ $16k).
- Capacity is the bottleneck. Relax GPU class (A100/L40S are less contended; slower GPUs
  make comms effects *more* visible). SkyPilot's multi-cloud search directly addresses this.
- Write for **unattended execution**: one script provisions → pulls data → runs matrix →
  writes durable results → tears down, so you can grab any slot the moment it opens.

---

## 8. Local 3090 — what it's for
**Good for (all correctness work, free and instant):**
model/data/loop correctness · bf16 + `torch.compile` (Ampere numerics match H100) · SDPA ·
checkpoint/resume · **distributed code path via `gloo`/CPU multi-rank** (verify N-rank grads
== single-process) · MFU instrumentation.

**Cannot tell you:** anything about interconnect/overlap (needs ≥2 GPUs) · Track B at all ·
**any transferable performance number** (GeForce bf16 is half-rate → ~35–70 vs ~990 TFLOP/s;
15–25× slower, plus thermal throttling). Use for correctness, never for results. A full local
converged run ≈ a day or two — fine as an overnight sanity check, useless for iterating.

Run the **same Docker image** locally as in cloud. (If a 2nd consumer card is ever added:
NVIDIA disables P2P on GeForce → NCCL via host memory → pathological, unrepresentative scaling.)

---

## 9. First cloud session (when capacity appears)
Keep it to minutes with known-good code:
1. Provision **one** node manually over SSH (by hand the first time — makes SkyPilot a
   convenience, not a black box).
2. Run `nccl-tests` (`all_reduce_perf`), record intra/inter-node bandwidth — this is the
   ceiling every later result is interpreted against.
3. Get a 2-process job talking.
4. *Then* write the SkyPilot YAML that reproduces steps 1–3 automatically.

---

## 10. Recommended first step for the coding session
1. `git clone` **karpathy/nanoGPT** and skim it; separately clone **KellerJordan/modded-nanogpt**
   for its `cached_fineweb10B.py` data script and the 3.28 target definition (use these two
   verbatim; ignore the current record's micro-optimizations).
2. Stand up the pinned Docker image + `uv` environment; get it running on the 3090.
3. Download a couple of FineWeb shards via `cached_fineweb10B.py`.
4. Get the **single-GPU** baseline training and reaching a sane loss on the 3090, with MFU
   instrumentation and trackio logging wired in from the start.
5. Add the hand-rolled DDP layer; verify multi-rank == single-process correctness on
   `gloo`/CPU. **Only then** think about renting anything.

Build order overall: correctness on 3090 → `nccl-tests` + single throughput number in a short
first cloud session → Track A matrix → Track B extension → analysis & write-up.

---

## Remaining open steps
- Select cloud provider (blocked on capacity — try Nebius next; consider A100/L40S).
- Set up infrastructure (register, pay, keys, provision + tear down a test node).
- Build & validate training stack (env + `nccl-tests`).
- Develop training code (model, data, distributed loop, MFU instrumentation).
- Small-scale validation runs (correctness, N-GPU == 1-GPU).
- Execute the run matrix (both tracks).
- Analyse & write up (scaling-efficiency plots, public trackio dashboard, portfolio README).

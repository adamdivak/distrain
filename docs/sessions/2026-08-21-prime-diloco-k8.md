# Session log — 8× A100-SXM4-80GB, Prime Intellect / lambdalabs (2026-08-21)

> **MFU correction (2026-08-22).** Every MFU figure in this log was computed with
> the pre-correction numerator (an untied `wte` charged 6N as if it were a matmul)
> and is **1.271× too high** — divide by 1.271. Wall clock, tokens and losses are
> unaffected. See [`../decisions.md`](../decisions.md) §3.

Runbook: [`../runbook-prime-intellect.md`](../runbook-prime-intellect.md),
Session B (DiLoCo K=8) plus a trimmed Session A (netem) in the same rental.
**First session on Prime Intellect that ran real work**, and the first to use
the stock-image route (§20): boot their Ubuntu, run our own container inside it.

| pod | image (digest) | where | when (UTC) | cost |
|---|---|---|---|---|
| `e980488c4af8…` | `c8c72e1` (`f915028c…`) | lambdalabs us-east-1, $22.32/h | 12:49 – 15:50 (guard) | $63.68 |

A short A6000 detour preceded it (`f87054dae9a4…`, 2× A6000 massedcompute
us-central-3, $1.08/h, ~10 min, **$0.06**): rented to run Session A cheaply,
then torn down on the call that a netem curve is worth more on the anchor's own
hardware than on a different card. Not wasted — it measured a second PI upstream
(see "Venue facts" below).

## Timeline

### 1. Provision and first contact

`up --allow-stock-image` on their `ubuntu_22_cuda_12`. Confirms §20 on a second
box and a second upstream: `systemd-detect-virt` → **`kvm`**, root `CapEff`
**`000001ffffffffff`** (full set), 240 vCPU, **19 TB** on `/`, driver 570.148.08.
Docker 28.3.1 and the NVIDIA container toolkit preinstalled — zero setup.

**Topology: full `NV12` mesh** — every GPU pair on 12 NVLink lanes, the same
fabric shape as the §14 anchor.

### 2. Image parity — the point of the stock-image route

```
pod:    ghcr.io/adamdivak/distrain@sha256:f915028c1c50ce23…
aurora: ghcr.io/adamdivak/distrain@sha256:f915028c1c50ce23…
```

**Byte-identical**, pulled in 1m43s. This is the parity their own registry
cannot give (`prime images push` rebuilds — §20). The read-only `read:packages`
PAT was used for the pod login; `gh auth token` never left aurora.

`pytest -q` on the baked code: **226 passed** (29m49s — the gloo multi-rank tests
spawn ranks that each default `OMP_NUM_THREADS` to all 240 cores, the same effect
§14 saw at 128 cores). Parity proven without rsync or `uv sync`.

### 3. Roofline and bandwidth ceiling

- **Roofline: 266.0 TFLOP/s** sustained bf16 (fp32 19.0) against the **269.9**
  recorded in `mfu.py` from the RunPod A100 node — **1.4% apart**, box-to-box
  variation of the kind the README already documents for the two 3090s (9%).
  **Deliberately not changed**: every MFU in this session is reported against
  269.9 so it stays directly comparable to the §14 anchor, at the cost of
  understating this box's MFU by ~1.4%. Changing the denominator mid-session
  would have required an image rebuild *and* broken the comparison the session
  exists to make.
- **NCCL all-reduce: 157.7 GB/s avg bus bandwidth**, 209.1 GB/s at 512 MB —
  the §14 anchor measured 154 GB/s, so the interconnect is a match.

### 4. Data and smokes

`cached_fineweb10B.py 55` → 56 files, 11 GB, ~7 min, started first per the
runbook and again not the long pole.

DiLoCo 8-rank smoke (H=4, synthetic): clean, outer steps visible at 303/213 ms
against an 87 ms inner step.

**Compiled smoke, added deliberately**: 306 ms/step at **80.6% MFU** at global
batch 480. §14's anchor ran `ddp_torch --compile`, so an uncompiled DiLoCo run
would have confounded the wall-clock half of the comparison with compilation
rather than method. The runbook's B3 command omitted `--compile`; it now
carries it.

### 5. A runbook bug the startup guard caught

The runbook's B3 command pairs `--val-every 250` with `--outer-sync-every 500`.
train.py refuses that combination:

> Validation must happen at intervals that are divisible by diloco
> synchronization intervals, otherwise evaluation would be performed on a
> non-synchronized local variation of the model.

`val_every` must be a **multiple** of H, not a divisor — otherwise a val on a
non-sync step measures one replica's local variation and can silently corrupt
the 3.28 check (§16). At H=500 the smallest legal cadence is 500, which costs
nothing since the crossing resolution under DiLoCo is a multiple of H anyway.
Cost of the bug: ~2 minutes, because it failed loudly at startup rather than
producing a plausible wrong number. Runbook corrected.

### 6. The converged run

```bash
torchrun --standalone --nproc_per_node=8 -m distrain.train \
  --distributed-mode diloco --outer-sync-every 500 --outer-lr 0.7 --outer-moment 0.5 \
  --global-batch-seqs 480 --grad-accum-steps 1 \
  --max-steps 10000 --val-every 500 --checkpoint-every 500 \
  --diag-val-every 250 --compile --run-name diloco-k8-a100
```

Launched 13:34:40 UTC. Steady state **~314 ms/step at ~79% MFU**, against the
§14 DDP anchor's 315 ms/step at ~79% — same hardware, same global batch, same
tokens per step (491,520). Global throughput matches the anchor exactly, so
wall-clock and token comparisons are both clean.

**The headline, against §14's DDP anchor — same box shape, same global batch,
same 4.92B tokens:**

| | val @ step 9999 | training time | reached 3.28? |
|---|---|---|---|
| DDP `ddp_torch --compile` (§14) | **3.2730** | 3147.1 s | yes, step 9999 |
| DiLoCo K=8, H=500, outer lr 0.7, μ=0.5 | **3.5183** | 3250.3 s | **no**, short by 0.238 |

`target_reached_step: None`. Untuned DiLoCo at K=8 costs **+0.245 val loss at
equal tokens** and buys nothing in wall clock on this transport — the ~103 s
excess is roughly the 40 diagnostic replica-evals, i.e. instrumentation rather
than method. That DiLoCo loses on a full-NVLink box is not a surprise but a
boundary condition: its design pays off when communication is expensive, which
is precisely what §15's netem curve exists to measure.

**Tokens-to-3.28 is therefore not measured for DiLoCo, and was not reachable
here.** §18's arithmetic says config A needs ~24,900 steps at the observed
ratio — ~2.3 h more compute, and past the 55-chunk corpus wrap at ~11,190 steps,
so it would silently begin a second epoch. Extending inside this rental was
declined for that reason; the deliverable is the loss-vs-tokens curve and the
equal-token endpoint, which is a clean comparison in its own right.

**Val curve** (post-sync, the reported metric):

| step | 500 | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 | 9000 | 9500 | 9999 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val | 6.5805 | 5.0144 | 4.1865 | 3.9361 | 3.8204 | 3.7481 | 3.6989 | 3.6634 | 3.6372 | 3.6152 | 3.5603 | 3.5183 |

**What the outer step is worth** — `--diag-val-every 250` evaluates every replica
just before the sync, so the averaging effect is observed, not inferred:

| step | replica mean | spread | synced | gap (synced − replicas) |
|---|---|---|---|---|
| 500 | 5.3708 | 0.0921 | 6.5805 | **+1.2097** |
| 1000 | 4.9496 | 0.0295 | 5.0144 | +0.0648 |
| 1500 | 4.5743 | 0.0383 | 4.4256 | −0.1487 |
| 4000 | 4.0562 | 0.0151 | 3.8204 | −0.2358 |
| 7000 | 3.9057 | 0.0244 | 3.6634 | −0.2423 |
| 9000 | 3.8629 | 0.0180 | 3.6152 | −0.2477 |
| 9500 | 3.7161 | 0.0057 | 3.5603 | −0.1558 |
| 9999 | 3.5636 | 0.0013 | 3.5183 | −0.0453 |

Three things this settles:

- **The first round is the dangerous one.** Averaging eight replicas that
  diverged from a barely-trained init costs +1.21 — the model-averaging
  barrier. It is neutral by step 1000 and beneficial from 1500 on. §18's K=2
  aurora baseline saw the same first-round shape (gap 1.138), so this is K-robust
  and not a K=8 pathology.
- **No excursion at μ=0.5, and no growing gap** — the runbook's stop condition
  never fired. The μ=0.9 sawtooth §19 diagnosed is absent at K=8, confirming
  §19's choice of μ=0.5 for the rented run was right.
- **Averaging recovers about half the deficit.** The steady-state gain from the
  outer step (−0.24) is almost exactly the size of DiLoCo's shortfall against DDP
  (+0.245), so a lone 60-sequence replica would trail DDP by roughly twice as
  much. Replica spread falls monotonically (0.092 → 0.0013): the replicas
  genuinely converge, they just converge to a worse point than DDP does at equal
  tokens.

### 7. Session A (trimmed) — netem in the same rental

Run on the leftover clock rather than in its own session, on the anchor's own
hardware at 8 ranks. Transport forced off NVLink and **verified before
trusting any number**: `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1` yields
**72 connections `via NET/Socket/0`**, so netem on `lo` actually shapes the
collective traffic.

Mean ms/step, 8 ranks, `bench_ddp_modes.py` (global batch 64 = 8 seqs/rank):

| rate | ddp_torch (compiled) | interleaved (compiled) | interleaved (uncompiled) | bucketed | naive |
|---|---|---|---|---|---|
| none (socket) | 1285.5 | **1180.4** | — | 1210.5 | 1596.0 |
| 40 gbit | **1897.2** | 1926.5 | 1899.1 | — | — |
| 10 gbit | **7314.6** | 7323.3 | 7317.9 | — | — |

**§14 predicted the compile-vs-overlap lead would "flip back" under netem. It
does not flip — it dissolves.** Uncompiled/overlapped interleaved does retake
the lead the moment the transport becomes sockets (1180 vs 1285 ms unthrottled,
matching the 2×3090 Socket+SHM result), but as soon as bandwidth binds, every
config converges: 1.5% spread at 40 gbit, **0.12% at 10 gbit**. Overlap cannot
hide communication that *is* the step, and compilation only accelerates compute
that has stopped being the bottleneck. The mode choice matters in the middle
regime, where compute and comm are comparable — not at the slow end, which is
where the study had assumed it would matter most.

**Three limitations, stated rather than smoothed over:**

- **The ladder stops at 10 gbit.** The 1 gbit and 500 mbit invocations exceeded
  their 600 s per-invocation timeout and wrote nothing. The step-time model
  behind the schedule was ~10× optimistic: nominal 10 gbit should move a 567 MB
  ring reduction in ~0.45 s, but the measured comm cost was ~6.1 s — netem's
  rate limiter over loopback with NCCL's many parallel socket channels is far
  from nominal throughput. Budget the low end off *measured* points, never off
  nominal rate.
- **The bench ran at global batch 64, not the anchor's 480**, so §15's
  `time-to-3.28 = 9999 × step_time` **cannot be applied to these numbers
  directly**. What transfers is the comm cost at each rate (step minus the
  ~0.09 s compute at this batch), which can be added to the anchor's 0.313 s
  compute — a reconstruction, and labeled as one, not a measurement.
- **The ceiling cut the session.** `guard` terminated at the 3 h deadline
  (15:50 UTC) with 500 mbit still queued, which also pre-empted the final
  artifact pull. What survived did so because the 5-minute watcher mirror had
  already copied `session_out/` off the box — the §14 "mirror continuously"
  rule earning its place a second time. The trackio DB did **not** survive: it
  lives in `~/.cache` inside the container, which is not a mounted path. Every
  number here comes from `train.log`, which was mirrored.

### 8. Cost

$89.00 → $25.26, i.e. **$63.74** for the session: 3 h of 8×A100 at $22.32/h
plus $0.06 for the A6000 probe. Project spend to date ≈ **$148** of the $150
target / $400 ceiling.

## Venue facts worth keeping

- **massedcompute is also a root KVM VM.** The A6000 probe measured `kvm`,
  full `CapEff`, and netem verified *by effect* (25 ms delay → 50.2 ms RTT).
  So §20's "PI pods are VMs" holds across at least three upstreams
  (nebius CPU, lambdalabs GPU, massedcompute GPU), not just lambdalabs.
- **No 2-GPU A100 exists on this venue.** Live A100 stock is 8-GPU only
  (lambdalabs 8×80GB SXM4 $22.32/h, 8×40GB PCIe $15.92/h, vultr $22.40/h),
  so a cheap 2-GPU netem session on *anchor-comparable* hardware is not
  purchasable here. Either accept a different card or run netem on the 8-GPU
  box — this session did the latter.
- **Team wallet, again.** `status` read `(team cmt057f9g…)`; a bare key would
  have seen $0 and `up` would have refused.

# 2026-08-19 — halving outer momentum removes the DiLoCo excursion

> **MFU correction (2026-08-22).** Every MFU figure in this log was computed with
> the pre-correction numerator (an untied `wte` charged 6N as if it were a matmul)
> and is **1.271× too high** — divide by 1.271. Wall clock, tokens and losses are
> unaffected. See [`../decisions.md`](../decisions.md) §3.

Arm A of the §19 probe (`diloco-b480-mom05`): the 2026-08-18 baseline with one
variable changed, outer momentum 0.9 → 0.5. Same seed, same 6000-step trapezoid,
same B=480 / H=500 / outer lr 0.7. Nothing was rented; $0 spent.

## 1. The measurement

`reference` is the single-GPU DDP anchor. Gaps are val loss above it. The
diagnostic columns come from `--diag-val-every 500`, which evaluates each replica
just *before* the outer sync, so replica spread is observed rather than inferred.

| step | reference | μ=0.9 baseline | μ=0.5 | gap μ=0.9 | gap μ=0.5 | pre-sync r0/r1 | spread | merge vs best replica |
|---|---|---|---|---|---|---|---|---|
| 0 | 10.8250 | 10.8250 | 10.8250 | 0 | 0 | 10.8250/10.8250 | 0 | — |
| 500 | 4.5076 | 5.6460 | **5.3353** | +1.138 | +0.828 | 4.7850/4.7204 | 0.0646 | −0.61 (hurt) |
| 1000 | 3.9778 | 4.4666 | **4.1773** | +0.489 | +0.199 | 4.1792/4.1887 | 0.0095 | +0.002 |
| 1500 | 3.7864 | 4.1396 | **3.9055** | +0.353 | +0.119 | 3.9596/3.9646 | 0.0050 | +0.054 |
| 2000 | 3.6925 | 4.2243 ↑ | **3.7583** | +0.532 | **+0.066** | 3.8358/3.8430 | 0.0072 | +0.078 |

The baseline's gap *grows* at step 2000 — that is the excursion. μ=0.5 has no
excursion at all, and its step-2000 gap (+0.066) already matches the baseline's
best-ever gap (+0.058, reached at step 4000) in half the steps. Token ratio at
step 2000 is ~1.21× (3.7583 ≈ reference step ~1650) against the baseline's ~1.35×
plateau.

Run reached step 2310 before being stopped for the night; last checkpoint step
2000. Perf unchanged from the baseline: ~8.8 s/step, MFU ~36.7%, 20.4 GiB, 398 W.

## 2. Why — and it is not "averaging two drifted replicas"

That was the first hypothesis after step 500 and the data killed it. From step
1000 to 1500 the spread *halved* (0.0095 → 0.0050) while the merge gain *grew*
(+0.002 → +0.054). Pure averaging predicts the opposite: less disagreement should
mean less to gain from merging.

The consistent reading is the Nesterov build-up. The first outer step applies
`lr·(1+μ)`; steady state approaches `lr/(1−μ)`, and after n rounds the partial sum
is `(1−μⁿ)/(1−μ)` of it. At μ=0.9 that is 1.33× → 7×; at μ=0.5, 1.05× → 1.4×.
Round 1 fires into an empty momentum buffer against replicas that have only just
diverged — a cold start — and at μ=0.9 the amplification then keeps climbing for
the rest of the run. `lr_eff = 1.0` is exactly plain parameter averaging, which is
the scale round 1 wants and the scale μ=0.9 leaves fastest.

Note the μ=0.5 curve is not "the excursion hasn't started yet": by round 4 μ=0.5
is at 94% of its final amplification, against μ=0.9's 34%.

## 3. Two OOMs on the 3090, both self-inflicted

Both came from the diag eval added the previous session, and both are fixed and
committed.

1. **Step 250.** The reported val runs on rank 0 alone; the diag eval runs on
   *every* rank at once. On a box where two ranks share one GPU that allocates the
   3.07 GiB logits tensor twice. Fix: `--diag-eval-batch-seqs`, a batch override
   for the diag path only (`717a680`). The reported path is untouched, and a test
   asserts the override does not change the value.
2. **Step 1000, in the *reported* val — caused by the fix for the first.** At
   batch 8 the diag pass leaves the caching allocator carved into ~0.77 GiB
   blocks; rank 0 then held 4.10 GiB reserved-but-unallocated while the 3.07 GiB
   contiguous request failed. Fix: `torch.cuda.empty_cache()` on every rank at the
   end of the diag block, plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   (`7b4f0fa`). Verified by replaying the failing diag→val sequence from the
   step-1000 checkpoint; the diag numbers reproduced bit-identically.

Two process lessons, both cheap and both paid for twice today:

- **A smoke test that shrinks the quantity under test proves nothing.** The first
  diag smoke ran at `--eval-batch-seqs 8 --val-tokens 262144` — it exercised the
  `all_gather` and said nothing about memory. Re-smoke at the production
  footprint.
- **Silence looks exactly like progress.** A run was reported healthy after 90
  minutes dead, because it was checked once at the 60-second mark. Monitors must
  match failure signatures (`OutOfMemory|Traceback|Killed|ChildFailedError`), not
  only the happy-path line.

## 4. RunPod capacity

`scripts/watch_capacity.sh` polled 8-GPU stock every 5 min, 10:59–23:54. It is
read-only and never rents.

| GPU | 8× hits | window | secure | community |
|---|---|---|---|---|
| H100 80GB HBM3 | 31 | 13:28 → 23:43, recurring all evening | $26.32/h | $21.52/h |
| A100-SXM4-80GB | 3 | 19:52 → 20:04 only | $12.72/h | $11.12/h |

Every hit was SECURE with capacity "Low"; COMMUNITY never had 8 of either. The
regions rotated (US-MO-1, AP-IN-2, CA-MTL-1), so this is opportunistic stock, not
a queue — the launcher has to be ready to fire when a hit lands.

**H100 is the realistically available box, not the A100.** The runbook assumes an
A100; at 2.07× the price it needs to be ~2× faster to break even on the same
science, which it plausibly is, but the roofline is `UNVERIFIED` for both and must
be measured on the rented box before any MFU number is reported.

## 5. Next session

*(Arm A was resumed and finished on 2026-08-21 — endpoint 3.3978, gap +0.031.
See `docs/decisions.md` §20.)*

- **Resume arm A from step 2000 → 6000.** The warmdown (5000–6000) is where the
  baseline lost most of its ground and where the plateau-derived token estimate
  proved optimistic; the μ=0.5 endpoint is the number that matters.
- **Arm B is unsettled.** §19 specifies `diloco-b960-h250` (the outer-gradient
  noise arm). Today's evidence points at round-one cold start instead, which makes
  **outer-LR warmup at μ=0.9** the more targeted test. Not yet decided.
- **DiLoCo has never run on NCCL** — including the `all_gather` in
  `gather_scalars`. Smoke it first on any rented box, before the paid run.
- Add `--ignore=tests/test_runpod_session.py` to the runbook's image-parity step.
- Update §19 to record what was actually measured.

Known flake: `test_params_equal_for_larger_world_size_with_grad_accum[ddp_naive]`
failed once under full-suite load with "terminate called without an active
exception"; passes in isolation and `test_ddp.py` is 29/29 with and without the
day's changes. Gloo process teardown, not the invariant.

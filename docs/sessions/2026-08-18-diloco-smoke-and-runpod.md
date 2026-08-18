# 2026-08-18 — first DiLoCo measurement (aurora), and a lost RunPod day

Two threads. The free one produced the result; the paid one produced none.

## 1. DiLoCo K=2 on aurora — the measurement

`diloco-smoke-b480`: real FineWeb, global batch 480 (2 gloo ranks on one 3090,
accum 60), H=500, outer lr 0.7 / Nesterov 0.9, trapezoid `--max-steps 6000
--warmup-steps 250 --warmdown-steps 1000` — i.e. the *same schedule* as the
`rotary-calibration-3B` reference, so the curves are directly comparable
(§13's matched-code/matched-schedule rule). Stopped by hand at step 5500.

**No checkpoints were written** (`--checkpoint-every` omitted at launch — a
mistake), so the run cannot be resumed and the step-6000 endpoint was never
measured. The val curve below is the whole artifact.

| step | reference | DiLoCo K=2 | gap | phase |
|---|---|---|---|---|
| 0 | 10.8250 | 10.8250 | 0.000 | init fingerprint (ln 50304) |
| 500 | 4.5076 | 5.6460 | +1.138 | round 1 |
| 1000 | 3.9778 | 4.4666 | +0.489 | |
| 1500 | 3.7864 | 4.1396 | +0.353 | |
| 2000 | 3.6925 | 4.2243 | +0.532 | excursion begins |
| 2500 | 3.6275 | 4.3558 | +0.728 | excursion peak |
| 3000 | 3.5825 | 3.8804 | +0.298 | recovery |
| 3500 | 3.5490 | 3.6480 | +0.099 | |
| 4000 | 3.5253 | 3.5837 | **+0.058** | best gap (plateau) |
| 4500 | 3.4982 | 3.5617 | +0.064 | |
| 5000 | 3.4869 | 3.5643 | +0.077 | warmdown starts |
| 5500 | 3.4056 | 3.5356 | +0.130 | warmdown |
| 5999 | **3.3327** | (never reached) | — | reference endpoint |

**Findings.**

- **Untuned DiLoCo works, but oscillates.** Loss rose for two consecutive
  rounds (2000, 2500) then fell below its previous best. Read as divergence
  mid-run; it was not. Each val point is exactly one round (val_every == H),
  so round-level oscillation at momentum 0.9 is expected and three points
  cannot distinguish it from divergence.
- **Mechanism.** PyTorch's Nesterov SGD approaches `lr/(1-mu) = 7x` the mean
  delta once round deltas correlate; round 1 applies only 1.33x (buffer starts
  at the gradient), which is why early rounds looked survivable and the
  excursion arrived later.
- **Plateau token ratio ~1.35x.** At step 4000 DiLoCo matched the reference's
  step-2980 loss (1.34x); at 4500, 1.39x.
- **But the anneal is where DiLoCo loses.** Between 5000 and 5500 the
  reference dropped 0.081, DiLoCo only 0.029 — the gap widened from 0.058 to
  0.130. Warmdown is when a coherent model settles, and DiLoCo instead keeps
  diverging for H steps then averages and extrapolates past the mean with
  momentum still loaded. **Any tokens-to-3.28 estimate taken from plateau
  points is therefore optimistic** — the crossing must also pay off the anneal
  deficit.

**Reference data gotcha.** The 6000-step trapezoid's curve is split across two
trackio run names: steps <=4750 under `rotary-calibration-3B`, the warmdown
(4750-5999, ending 3.3327) under `3090-fineweb-3B-modded`. `rotary-calibration-3B`
*also* holds the later 6000->9000 continuation (steps 6000-8500, including
§13's re-entry bump to 3.4638 at 6250). Splicing the two names naively mixes
two different schedules.

## 2. RunPod — six pods, ~$15, no science

| pod | GPUs | image | outcome |
|---|---|---|---|
| growing_blush_mole | 8x4090 | stock `runpod/pytorch:2.4.0` | booted (wrong image; RunPod pre-caches its own) |
| above_scarlet_koi | 8x4090 | ours, GHCR | pull stuck at 0% forever |
| distrain-4090x8 | 8x4090 | ours, GHCR | never exposed :22 in 20 min |
| distrain-4090x8 | 8x4090 | ours, GHCR | "failed to pull image: unexpected EOF"; 35 min |
| distrain-3090x1 | 1x3090 | ours, **Docker Hub** | **booted; 145/145 training tests green** |
| distrain-4090x8 | 8x4090 | ours, Docker Hub | never exposed :22 in 35 min |

- **Image parity is proven** (runbook priority #1): the baked image pulled from
  Docker Hub onto a rented node passes all 145 training tests with no rsync and
  no `uv sync`. Its 17 launcher-test failures are `git rev-parse` returning 128
  — the container has no `.git`. Parity step should use
  `pytest -q --ignore=tests/test_runpod_session.py`.
- **The blocker is the 8x4090 host pool, not the registry.** Our 7.9 GB image
  (19 layers; 3.72 / 2.55 / 1.44 GB largest) pulls onto a 1-GPU secure host but
  has never pulled onto an 8x4090 host, across both registries and windows up
  to 50 min. All 19 blobs verified present and correctly sized in GHCR, so the
  image is intact. Datacenter-class hosts (A100/H100/the 3090) are the ones
  that work.
- **Docker Hub mirror**: `docker.io/adamdivak/distrain:4bb78cd` and
  `:4bb78cd-dh` (identical content; the `-dh` tag exists only so
  `_ensure_template` derives a distinct template name). Push took 20 min.
- **`avail` has false negatives.** It reported no 8x4090 capacity in any data
  center while the console rented one, and while `--skip-capacity-check`
  successfully deployed three. Community hosts additionally carry no data
  center id, so a per-DC query can never see them.

## 3. Tooling changes (committed today)

- `--diag-val-every`: per-replica diagnostic eval on `diag/` keys, off by
  default, placed before the outer step so a boundary yields the pre-sync
  replicas alongside `val/loss`'s post-sync model. Never feeds the 3.28 check;
  its duration is subtracted from `step_time`. Untested on NCCL.
- `runpod_session.py`: live `dataCenters` query (the frozen list had 28 of the
  API's 49); `--cloud-type COMMUNITY` with its price and no-volume rules;
  `--skip-capacity-check` so the deploy call is the capacity signal.

## 4. Next session

1. **Re-run the K=2 smoke to its endpoint with `--checkpoint-every`**, or accept
   the curve above. The step-6000 endpoint against the reference's 3.3327 is
   the clean whole-schedule comparison and is currently missing.
2. **Size the anchor off warmdown-inclusive data**, not the 1.35x plateau ratio.
3. **§18 batch-size probe on aurora (free)**: `b`=120 (H=250, 1500 steps) and
   `b`=240 (H=125, 750 steps) at matched 1.47B tokens, testing whether a
   quieter delta damps the excursion.
4. **K=8 still unmeasured** — the one thing needing rented hardware. Wait for
   A100 (~$12.72/h, keeps wall-clock comparability with §14) or H100
   (~$26.32/h). Avoid the 8x4090 pool until the image is materially smaller.
5. Budget: ~$98 of the $150 target spent; RunPod balance $54.37.

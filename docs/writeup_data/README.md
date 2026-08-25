# Writeup data

These CSVs are the small, reviewable projection of the gitignored experiment
artifacts used by the writeup plots. Regenerate them from the raw logs and
benchmark JSON with:

```bash
uv run python scripts/collect_writeup_data.py
```

Render interactive Plotly HTML figures plus static SVG and PNG copies with:

```bash
uv run --extra plots python scripts/plot_writeup.py
```

## Collected numbers

| CSV | Contents |
|---|---|
| [`validation_curves.csv`](validation_curves.csv) | DDP and DiLoCo validation loss through step 9999 / 4.915B tokens, including training time. |
| [`training_curves.csv`](training_curves.csv) | The training-loss half of the same two runs, per logged step, with step time and MFU. |
| [`scaling.csv`](scaling.csv) | Matched-micro-batch 1×A100, 8×A100 NVLink, and 8×A100 forced TCP/loopback step times and target-time projections, plus the directly converged 8-GPU run. |
| [`transport.csv`](transport.csv) | Effective all-reduce bandwidth, batch-480 step time, and target time for NVLink, measured A100 PCIe, forced TCP/loopback, and nominal 40/10 Gbit netem. |
| [`pcie.csv`](pcie.csv) | The 2026-08-22 A100-PCIe session: five batch-480 arms on one box, with MFU against that card's own measured peak. |
| [`diloco_sync.csv`](diloco_sync.csv) | Per-rank validation spread and the effect of each DiLoCo outer merge. |
| [`diloco_transport.csv`](diloco_transport.csv) | DDP and DiLoCo step time at each measured transport, with DiLoCo's outer sync amortized over H=500. |
| [`diloco_k_penalty.csv`](diloco_k_penalty.csv) | Cost of the outer merge over the whole schedule, at K=2 and K=8 — as a validation-loss gap and as the token ratio to reach equal loss. |
| [`transport_crossovers.csv`](transport_crossovers.csv) | The two bandwidths where the ranking of the options flips. |
| [`capacity_availability.csv`](capacity_availability.csv) / [`capacity_timeline.csv`](capacity_timeline.csv) | Measured 8-GPU stock at both venues, per poll and summarized. |
| [`ddp_modes.csv`](ddp_modes.csv) | Every measured (implementation, transport, batch): four modes at batch 64 over sockets, two at the anchor batch on NVLink, real A100 PCIe and sockets. |
| [`ddp_mode_steps.csv`](ddp_mode_steps.csv) | Every individual timed step behind `ddp_modes.csv`, because the means are tail-driven. |
| [`ratio_sweep.csv`](ratio_sweep.csv) | Four DDP modes across four micro-batches on 2×A100-SXM4-80GB, native NVLink and forced TCP: the mode ranking as a function of compute per collective. |
| [`cost_inputs.csv`](cost_inputs.csv) | Prefilled runtimes and blank rate/energy fields for the progressive cost plots. This file is user-edited and is not overwritten by the collector. |
| [`data_gaps.csv`](data_gaps.csv) | Coverage manifest with exact missing measurements. |

The main collected results are:

- 1×A100: **2589.1 ms/step**, **7.19 h** projected to the measured step-9999 crossing.
- 8×A100 NVLink: **337.8 ms/step**, **0.94 h**, **7.66× speedup**, **95.8% scaling efficiency** at matched per-device micro-batch.
- 8×A100 forced TCP/loopback: **1.22–1.27 s/step**, **3.38–3.53 h**, depending on the selected DDP implementation.
- 1×RTX 3090: **21.22 h measured** — `ref-1gpu-10k` (aurora, 2026-08-24) crossed 3.28 at step 9999 at val 3.2678, 76403.7 training seconds at 7.6333 s/step and 66.6% MFU. Run as the K=2 DiLoCo arm's single-GPU reference, so it cost nothing extra; it confirms the earlier 21.299 h Trackio extrapolation to 0.36%. At 800 W and $0.23/kWh that is **$3.91** of electricity, against **$14.31** to rent one A100 for the same convergence.
- Transport MFU: **57.6% NVLink**, **15.3% TCP**, with lower bounds of **8.9%** and **2.6%** from the reconstructed nominal 40 and 10 Gbit points.
- A100 80GB PCIe, measured 2026-08-22 on 2 GPUs: **2.29 GB/s** effective
  all-reduce bandwidth, **1745.4 ms/step** at global batch 480, **79.8%** scaling
  at 2 ranks, and a measured **256.5 TFLOP/s** bf16 peak. The node has **no
  GPU-to-GPU P2P** (`topo -p2p r` = `CNS`), so NCCL stages through host memory —
  these are not PCIe P2P numbers. Projected to 8 ranks: **826 ms/step, 2.29 h**,
  an optimistic bound.
- Equal-token endpoint: DDP **3.2730 in 3147.1 s**; DiLoCo **3.5183 in 3250.3 s**.
- Socket DDP modes at global batch 64 (means): naive **1596.0 ms**, bucketed
  **1210.5 ms**, interleaved **1180.4 ms**, PyTorch DDP **1285.5 ms**. Only naive
  is separated: its *fastest* step (1367 ms) is slower than any other mode's
  slowest. The other three have minima of **1112.8 / 1132.7 / 1134.9 ms** — 2%
  apart — and their mean ordering is produced by late-run tails, not by the
  implementations. Bucketed's *median* (1140.8 ms) is faster than interleaved's
  (1176.7 ms), reversing the mean ranking.
- Ratio sweep, 2×A100-SXM4-80GB, 2026-08-24 ($2.13 over two rentals): on **NVLink**
  all four modes sit within **2.2%** at 8 seqs/rank and **1.7%** at 60 — naive
  included. On **forced TCP** `ddp_torch` wins at every ratio by **10.1%** (micro 8)
  rising to **32.4%** (micro 60). Subtracting compute, compiled interleaved exposes
  the whole collective while `ddp_torch` hides **9–11%** of it at micro 8 and **50%**
  at micro 60: overlap is bounded by the compute available to hide behind (§28).
- **Socket step times are host-specific.** Two boxes of the same SKU agreed to
  **3.1%** on every NVLink arm and differed by **20.8–45.8%** on every forced-TCP
  arm. Quote socket ratios only within one box; this qualifies the netem ladder
  and the forced-TCP control too.
- At the anchor batch of 480 the interleaved-versus-PyTorch gap **changes sign
  with the fabric**: **337.8 vs 340.3 ms on NVLink** (+0.8%, σ≈0.8), **1869.3 vs
  1745.4 ms on real A100 PCIe at 2 ranks** (−6.6%, and the minima are 7.5σ
  apart), **1218.4 vs 1270.3 ms over sockets** (+4.3%, but PyTorch DDP's fastest
  step is the faster of the two — a tail, not a finding). Under `torch.compile`
  our interleaved mode does not overlap at all and PyTorch DDP does, so the two
  are trading exposed communication against lost fusion; which wins is set by
  the fabric (`docs/decisions.md` §26).
- Rentability of an 8-GPU node, 2026-08-19 to 2026-08-21: RunPod A100-SXM4-80GB in
  stock on **2.8%** of 247 polls, RunPod H100 on **26.3%**, Prime Intellect A100 on
  **99.0%** of 103. The A100 that was available billed at $22.32/h; the one that was
  not billed at $12.72/h.
- DiLoCo merge penalty: mostly transient, and **the loss gap overstates it**. Over
  the plateau the gap decays from **+0.84 → +0.025** at K=2 and **+2.07 → +0.205**
  at K=8, most of it inside the first ~1B tokens. The gap then *widens* through the
  warmdown, to **+0.057** and **+0.245** — but that is the reference's own slope,
  not a regression: the reference falls about **10× faster** in the warmdown than on
  the plateau, so an unchanged token lag reads as a bigger loss gap. Inverting the
  reference curve instead: DiLoCo needs **1.15–1.25×** the reference's tokens for
  equal loss at K=2 and **2.7–3.4×**, rising with training, at K=8. Report the token
  ratio, not the loss gap — same conclusion as §23's crossing ratio. **The inversion
  is only valid while both sides sit on the same side of the warmdown**, which the
  `ratio_comparison` column records: at K=2 the warmdown points match the reference's
  own warmdown and are clean, ending at **1.05×**, but at K=8 they match reference
  points still on the plateau, so the apparent fall to 2.36× is a schedule-position
  artifact, not a catch-up. **3.41×** at step 9000 is the last clean K=8 value. Both
  arms now run the same 10000-step trapezoid on a 500-step validation
  grid, so K is no longer confounded with the schedule — see below.
- End to end, DiLoCo lands at **2.25–2.35 h across every transport** while DDP runs
  0.94 h on NVLink and 21.10 h at netem-10 — it loses on a good fabric and wins on
  a bad one.
- Crossovers, both derived from the same reconstruction as the netem points:
  8-GPU DDP stops beating a single A100 below **0.50 GB/s** (4.0 Gbit/s) effective
  all-reduce bandwidth, and DiLoCo starts beating DDP end to end below **2.36 GB/s**
  (18.8 Gbit/s) once charged §18's 2.49× token ratio.

## Rendered figures

Every entry is currently rendered as `.svg` under
[`docs/plots/`](../plots/).

| Figure | What it establishes | Evidence |
|---|---|---|
| [`loss_curves.svg`](../plots/loss_curves.svg) | Both halves of the converged 8×A100 run's loss curve. | Measured training and validation logs. |
| [`capacity_availability.svg`](../plots/capacity_availability.svg) | How often each 8-GPU SKU was actually rentable. | 597 measured availability polls. |
| [`validation_loss_vs_tokens.svg`](../plots/validation_loss_vs_tokens.svg) | DDP versus DiLoCo validation quality at equal tokens. | Measured curves. |
| [`matched_batch_scaling.svg`](../plots/matched_batch_scaling.svg) | 1×A100 versus 8×A100 NVLink/forced-TCP optimizer-step time. | Measured step times. |
| [`time_to_target_scaling.svg`](../plots/time_to_target_scaling.svg) | 7.19 h on 1×A100 versus 0.94 h on 8×A100 NVLink. | Measured step times × measured crossing step. |
| [`transport_sensitivity.svg`](../plots/transport_sensitivity.svg) | Target time versus effective all-reduce bandwidth, with the 1×A100 time and the 0.50 GB/s break-even marked. | NVLink/forced TCP measured; 40/10 Gbit reconstructed upper bounds. |
| [`transport_mfu.svg`](../plots/transport_mfu.svg) | Effective MFU lost as transport slows, against the 60.1% a single A100 reaches alone. | Measured points plus lower bounds derived from reconstructed times. |
| [`equal_token_runtime.svg`](../plots/equal_token_runtime.svg) | DDP and DiLoCo wall clock through the same 4.915B tokens. | Measured; explicitly not DiLoCo time-to-convergence. |
| [`ddp_mode_comparison.svg`](../plots/ddp_mode_comparison.svg) | Which implementation gaps survive the noise, and that the surviving one changes sign with the fabric. | 15 timed steps after 5 warmups, drawn individually; anchor-batch bars are 12–15-step means on all three measured fabrics. |
| [`ratio_sweep.svg`](../plots/ratio_sweep.svg) | That the implementation is worth ~0% on NVLink at any batch and up to 31% on sockets, and that the micro-batch is what unlocks it. | 10 timed steps after 6 warmups per arm, 24 arms on one box. |
| [`pcie_modes.svg`](../plots/pcie_modes.svg) | Compilation is worth 1.64× on real PCIe; overlap is worth nothing. | 15 measured steps after 10 warmups, 2×A100 PCIe. |
| [`diloco_outer_sync.svg`](../plots/diloco_outer_sync.svg) | Replica spread and whether each outer merge helps or hurts validation. | Measured pre/post-sync evaluations. |
| [`diloco_k_penalty.svg`](../plots/diloco_k_penalty.svg) | What the merge penalty costs at K=2 and K=8: the loss gap, then the tokens to reach equal loss. | Measured curves on a shared 10000-step schedule at 491,520 tokens/step. Each arm is drawn against a no-DiLoCo run of the same config on its own box; those two references are the same experiment and agree to within 0.016 at every step. |
| [`diloco_transport.svg`](../plots/diloco_transport.svg) | Where DiLoCo's low communication starts to pay: per step, then end to end with its 2.49× step penalty charged. | DDP measured/reconstructed; DiLoCo reconstructed at H=500; the step ratio is §18's estimate. |

### Entering costs

Fill the blank fields in [`cost_inputs.csv`](cost_inputs.csv). Keep one currency
across rows that will appear in the same plot.

- Cloud rental: enter `currency` and `hourly_rental_rate`; cost is
  `runtime_hours * hourly_rental_rate + fixed_cost`.
- Desktop: enter `average_psu_power_watts` (average power drawn at the wall, not
  the PSU's rated maximum) and `electricity_price_per_kwh`; optionally add
  `capital_cost_per_hour`. Derived energy is
  `average_psu_power_watts / 1000 * runtime_hours`, and cost is that energy times
  `electricity_price_per_kwh`, plus capital and fixed costs.
- Use `total_cost_override` when the provider supplied an authoritative billed
  amount or when a different accounting convention is required.

Rates currently in the file are a **2026-08-24 list-price snapshot**: RunPod
secure cloud and Prime Intellect on-demand, averaged where both venues quote the
shape, single-venue where only one does. Each row's `notes` records which quotes
it came from. They are list prices, not billed amounts -- prefer
`total_cost_override` for a row that has a real invoice.

Rows with `missing_measurement` deliberately have no runtime. The DiLoCo
equal-token row is not comparable to time-to-3.28 rows until DiLoCo converges.

The status columns are load-bearing:

- `measured_full_convergence` means a training run actually crossed 3.28.
- `extrapolated_from_measured_step_time` combines a directly measured step time
  with the independently measured step-9999 crossing.
- `reconstructed_upper_bound` adds communication measured at the benchmark
  batch to compute at the anchor batch. The reconstruction was calibrated at
  +23% against the direct forced-TCP/loopback measurement.

`data_gaps.csv` is the coverage manifest for the proposed writeup. In
particular, no price-per-convergence plot is generated: DiLoCo did not converge,
PCIe and multi-node were not measured, and a desktop-cost convention has not
been defined. Plotting those together would turn missing measurements into
unstated assumptions.

`ddp_modes.csv` is the appendix comparison, kept as a committed projection
because the raw JSON under `out/` is gitignored like the training logs. Two
caveats travel with it and are drawn into the figure rather than left in prose:

- **The four-way matrix exists at one operating point only** — global batch 64
  over forced TCP/loopback. The NVLink session ran the anchor batch and had
  rental time for two modes, not four, so naive and bucketed have no NVLink
  measurement. That is a gap in coverage, not a finding.
- **At 15 timed steps the means are tail-driven.** Three of the four modes show a
  rising tail in their last few steps, and each mode is a separate subprocess, so
  whichever one is running when the host jitters absorbs it. Compare minima or
  medians; `ddp_mode_steps.csv` carries every step so this is checkable.

Both implementations use the same 25 MB bucket cap — `ddp_torch` is constructed
with `bucket_cap_mb` from the same `--ddp-bucket-size` the hand-rolled modes use —
so the comparison is not a tuning artifact. Every row is compiled, and gradient
accumulation is excluded from communication on both sides (`no_sync()` for
`ddp_torch`, `set_last_iteration()` for ours).

What the two are actually trading is measured in `docs/decisions.md` §26 and
reproducible for free with `scripts/probe_compile_overlap.py`: under
`torch.compile` AOTAutograd fuses the backward into one node, so the hand-rolled
hooks all fire at its end and compiled `ddp_interleaved` is bucketed with an
async launch; DDPOptimizer splits the graph into 14 subgraphs at the bucket
boundaries and keeps the eager gradient-arrival profile, paying for it in fusion
lost at each seam. Never quote compiled interleaved as evidence about overlap.

## Plots still missing

Rows are ordered by what it takes to close them. Nothing here can be recovered
from artifacts already on disk — that work is done.

| Draft request | Missing numbers | Experiment or recovery needed |
|---|---|---|
| **8×A100 PCIe bars** | An 8-rank measurement. Bandwidth, step time, MFU and target time are measured at **2** ranks and projected to 8 (`docs/decisions.md` §25), which is an optimistic bound — an 8-way ring crosses more host bridges. | **Needs stock.** The hardware exists at RunPod for half the anchor's price and had no 8-GPU capacity on 2026-08-22; `scripts/pcie_hunt.sh` rents and measures the first opening unattended. See "The PCIe point" below. |
| DiLoCo on slow transport, measured | Directly measured inner-step time and outer-sync latency at TCP/40/10 Gbit. | `diloco_transport.csv` reconstructs all of these from the measured per-transport all-reduce ÷ H=500, so this buys a confirmation, not a new result. Would need ≥1,001 steps per transport at K=8, H=500, global batch 480, diagnostics off. |
| DiLoCo time to 3.28 | Crossing step, tokens, and training time. | Blocked, not merely unbought: at §18's ~2.5× token ratio it needs ~24,900 steps, past the corpus wrap near step 11,190, so it cannot be measured without disclosing a second epoch. The equal-token endpoint is the honest deliverable. |
| Price-per-convergence panels | A matching 1×A100 rental rate, the desktop power/price convention, and a convergence time for an 8-GPU PCIe box and for DiLoCo. | Fill the blank fields in `cost_inputs.csv`; the desktop row needs only average wall power and an electricity price. Do not mix simulated and differently priced machines silently. |
| Current-PyTorch comparison | Identical mode timings under a current PyTorch build. | Repeat the benchmark matrix in an image differing only in PyTorch. Needs ≥2 GPUs, so aurora can only do the single-device half. |
| Multi-node DDP | Everything. | Deliberately Part II. |

### Deliberately not measured

**The netem ladder stops at 10 gbit.** 1 gbit and 500 mbit overran their timeouts
in the 2026-08-21 sweep and will not be re-run. Nominal 10 gbit already delivers
only 1.2 Gbit/s effective and puts an 8-GPU DDP run at 21.1 h against one A100's
7.19 h — far past the 0.50 GB/s break-even in
[`transport_crossovers.csv`](transport_crossovers.csv) where eight GPUs stop
beating one. A lower point would extend a curve whose conclusion is settled.

**Seed error bars.** Not worth the aurora-days or the rental.

### Landed: the matched-schedule K=2 arms

The K=2 and K=8 DiLoCo arms used to run different schedule lengths (6000 versus
10000 steps), so no shared x axis made their penalties strictly comparable — a
normalized one implies equal progress at equal position, and a token axis leaves
them at different points in the LR trapezoid. `scripts/run-k2-10k-arms.sh`
removed the confound by putting both K=2 arms on the K=8 arm's schedule. Run on
aurora 2026-08-22 to 2026-08-24, ~36 h sequential on one GPU, $0:

1. `diloco-k2-10k` — the existing arm **resumed from its step-4000 checkpoint**
   with `--max-steps 10000`. `lr_at()` holds the LR constant from `warmup_steps`
   to `max_steps - warmdown_steps`, so the plateau is 250–5000 at 6000 steps and
   250–9000 at 10000; step 4000 lies inside both, and re-entering there yields a
   genuine unbroken 10000-step trapezoid rather than a restart-after-warmdown
   artifact. The resume reproduced the original step-4000 validation loss of
   **3.5524** exactly, which is the check that it re-entered cleanly. This is the
   case `train.py`'s `checkpoint_keep_every` comment was written for.
2. `ref-1gpu-10k` — a fresh 10000-step single-GPU reference. `checkpoints/ckpt.pt`
   could not serve: it is `rotary-calibration-3B` at `next_step=8500` on a
   **9000**-step schedule, already past a warmdown restart, which is exactly why
   that run's validation jumped 3.3664 → 3.4638 at step 6250. It ends at **3.2678**,
   reaching the 3.28 target at step 9999.

Both finished. The K=2 arm ends at **3.3248** against that reference — a gap of
**+0.057**, wider than the +0.031 §23 reported at 6000 steps, because the matched
schedule now runs the arm through the warmdown that the 6000-step arm stopped
short of. In token terms it is the opposite: **1.05×** the reference's tokens for
equal loss at the endpoint, against 1.25× on the plateau — and at K=2 that endpoint
figure is clean, because the reference reaches the same loss inside its own warmdown
(step 9553) rather than on the plateau. The K=8 arm's warmdown points are not clean
in that sense, so its last trustworthy ratio is 3.41× at step 9000; `ratio_comparison`
marks the difference and the plot draws those points dotted and open.
`token_ratio` is **interpolated** on the reference curve (`token_ratio_status` records this), not
an unsmoothed crossing like §23's, and the two references sit on different val
grids — 250 steps for K=8, 500 for K=2 — so K=2's warmdown ratio is the more
coarsely resolved of the two. `diloco_k_penalty.csv` splices the DiLoCo curve at step
4000: everything below comes from the 6000-step arm this one resumed, which is
the same trajectory on the shared plateau.

Both validate every 500 steps: `train.py` requires `val_every` to be a multiple
of `outer_sync_every`, so the DiLoCo arm cannot run a 250 grid. Matching the
reference to it also retires §23's mismatched-val-grid caveat. Both need
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and
`--diag-eval-batch-seqs 8`; without them the two ranks sharing one 3090 OOM, as
they did on 2026-08-19 and again on the first launch here.

### The PCIe point

This is the transport most people can actually rent. A topology-verified
**2×A100 80GB PCIe** box was measured on 2026-08-22 for $0.59 — 2.29 GB/s
effective all-reduce, 1745.4 ms at batch 480, projecting to 826 ms and 2.29 h to
3.28 at 8 ranks (`docs/decisions.md` §25). That node has **no GPU-to-GPU P2P at
all** (`topo -p2p r` = `CNS`; NCCL routes via host memory), so nothing here may be
called PCIe P2P. What is still missing is an 8-rank box, and as of 2026-08-22
that is a **capacity** problem, not a labelling one (§24):

- **Only one venue has the hardware.** RunPod sells `NVIDIA A100 80GB PCIe` as
  its own GPU type — a different SKU from `NVIDIA A100-SXM4-80GB`, so the card
  itself is the PCIe part. At $11.12/h secure it is *half* the SXM4 anchor's
  price, and it reported no 8-GPU capacity at the **deploy call** (not merely the
  advisory precheck) on both tiers.
- **Prime Intellect has none at all.** Every 8×A100 "PCIe" offer there carries
  `cloudId: gpu_8x_a100` — lambdalabs' 8×A100-**SXM4**-40GB instance, listed as
  PCIe in all four of its regions. `prime_session.py` now refuses these by
  `cloudId`, so a pinned `--socket PCIe` matches nothing instead of renting a
  mesh. This is the pre-rental fabric check §22 concluded did not exist.
- **Do not plan it on community capacity.** The only PCIe stock anywhere was
  2 GPUs on RunPod community, where four pods never produced a running container.

The wait is automated, and so is the measurement:

```bash
scripts/pcie_hunt.sh out/pcie-hunt.log 300     # rents, measures, tears down, verifies
```

It probes by attempting the deploy, because a rejected deploy creates nothing and
costs nothing while the advisory precheck has false negatives. On success it
starts `guard`, runs `scripts/pcie_measure.sh` — **topology gate first**: any
`NV#` link between GPUs aborts before a number is produced — then roofline,
`nccl-tests` bandwidth and the batch-480 benches, pulls the artefacts, terminates
and verifies. Ceiling 1 h / $11.12.

For a box that has stock but cannot boot the image that carries `nccl-tests`,
`scripts/allreduce_bw.py` measures the same bus bandwidth in plain torch. Bus
bandwidth is rank-normalized, so even a 2-GPU box places a point on this axis —
label it an optimistic bound for an 8-way host, whose ring crosses host bridges.

`scripts/watch_capacity.sh` still logs stock for the availability figure:

```bash
WATCH_GPUS="NVIDIA A100 80GB PCIe" WATCH_PRIME_SOCKET=PCIe \
WATCH_PRIME_GPUS=$'A100_80GB\nA100_40GB' \
    scripts/watch_capacity.sh out/capacity-pcie.log 300
```

`collect_writeup_data.py` folds `out/capacity-pcie.log` into the capacity CSVs
automatically when it exists, so a PCIe hunt also extends the availability
figure.

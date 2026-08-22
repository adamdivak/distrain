# Session log — 8× A100-SXM4-40GB, Prime Intellect / lambdalabs (2026-08-21, second session)

> **MFU correction (2026-08-22).** Every MFU figure in this log was computed with
> the pre-correction numerator (an untied `wte` charged 6N as if it were a matmul)
> and is **1.271× too high** — divide by 1.271. Wall clock, tokens and losses are
> unaffected. See [`../decisions.md`](../decisions.md) §3.

Runbook: [`../runbook-prime-intellect.md`](../runbook-prime-intellect.md) Session C.
Rented to buy two things the write-up was missing: a **measured single-GPU
time-to-3.28** on a datacenter GPU, and a **PCIe** point on the transport axis.
It delivered the first, disproved the premise of the second, and validated the
netem reconstruction along the way.

| pod | image (digest) | where | when (UTC) | cost |
|---|---|---|---|---|
| `f5313f834e74…` | `c8c72e1` (`f915028c…`) | lambdalabs us-east-1, $15.92/h | 16:49 – 17:20 | **$8.24** |

No image was built or pushed for this session: `git diff c8c72e1..HEAD` touches
only `README.md` and `docs/`, so the already-proven `c8c72e1` is *code-identical*
to HEAD. The §4 `pytest` parity run was skipped deliberately (~30 min against a
~40 min budget) — the digest comparison is the parity evidence, and this image's
suite went green on a rented box earlier the same day.

## 1. The topology gate fired — the "PCIe" offer is not PCIe

Prime Intellect's catalogue lists `A100_40GB` under both `SXM4` and `PCIe`
sockets, and `prime_session.py --socket PCIe` selected the latter. The box that
arrived:

```
NVIDIA A100-SXM4-40GB  x8, driver 570.148.08
GPU0..GPU7:  NV12 to every other GPU   (full mesh, 12 NVLink lanes per pair)
```

**`--socket PCIe` returned an SXM4 box on a full NVLink mesh.** The socket field
is provider metadata, not a fabric guarantee. Had the session trusted the label,
every number in it would have been published as "PCIe" while measuring NVLink —
which is precisely the failure the runbook's topology gate exists to prevent, and
the same discipline §21 applied when it verified `via NET/Socket/0` before
trusting a netem number.

Consequence: **PCIe was not measured and is not purchasable on this venue under
this label.** §20's claim that `--socket` "pins the socket to SXM4 so a PCIe A100
cannot silently break comparability" holds only in the direction that matters for
the anchor (an SXM4 request cannot yield PCIe *silently* — it yields SXM4). The
reverse is not true and the flag must not be read as a fabric guarantee.

The box was kept, because the single-GPU baseline — the higher-value gap — needs
exactly this hardware.

## 2. The box is the §14 anchor with less memory

| | this box (40GB) | §14 anchor (80GB) | delta |
|---|---|---|---|
| roofline, sustained bf16 | **270.1** TFLOP/s | 269.9 | **+0.07%** |
| NCCL all-reduce, avg bus BW | **151.0** GB/s | 154.4 | −2.2% |
| peak bus BW @ 512 MB | 195.5 GB/s | ~210 | −7% |

Compute is equivalent to within measurement noise and the fabric is the same
shape, so cross-box comparison is licensed on compute. The one real difference is
**memory**: 40 GB forces a micro-batch of 30 sequences where the anchor used 60.
Global batch (480), tokens/step (491,520), data order and comm volume per
optimizer step are all unchanged by that chunking.

`270.1` is committed to `_PEAK_BF16` as a measured `A100-SXM4-40GB` entry, ahead
of the generic `A100` pattern. That ordering matters: `"A100"` substring-matches
`NVIDIA A100-SXM4-40GB`, so **before this commit the box silently used the
`UNVERIFIED` 312.0 datasheet figure** rather than refusing to start. Every MFU
printed on the pod during this session is therefore ~14% too low; the figures
below are recomputed on aurora against 270.1.

## 3. The arms

All at **global batch 480** and **micro-batch 30 per device** — matching the
micro-batch across arms is what makes the ratio mean anything, since each device
then does identical compute chunks and only the rank count and communication
differ. 20 steps, 8 discarded as warmup.

| config | ms/step | MFU | time to 3.28 |
|---|---|---|---|
| **1 GPU**, 30 × 16 accum | **2589.1** | 76.4% | **7.19 h** |
| 1 GPU, `ddp_torch` at 1 rank | 2597.7 | 76.2% | 7.21 h |
| **8 GPU** NVLink, `ddp_interleaved` | **337.8** | 73.2% | **0.94 h** |
| 8 GPU NVLink, `ddp_torch` | 340.3 | 72.6% | 0.95 h |
| 8 GPU socket, `ddp_interleaved` | 1218.4 | 20.3% | 3.38 h |
| 8 GPU socket, `ddp_torch` | 1270.3 | 19.5% | 3.53 h |

Time-to-3.28 is `9999 × step time` per §15 — the val-vs-step curve does not
depend on rank count or transport, and §14's three-way replication bounds the
residual to ≤0.01. These are step-time measurements extended by a crossing step
measured elsewhere, not converged runs, and are labelled as such throughout.

**Headline: 7.66× from 8 GPUs, 95.8% scaling efficiency** (2589.1 / 337.8). A
1-rank `ddp_torch` costs 0.3% over unwrapped single-device — the DDP wrapper is
not where the loss is.

**The 0.88 in §14 badly understated scaling.** That figure came from the bench
default of 8 sequences per GPU, where a fixed 648 MB all-reduce sits on ~50 ms of
compute. At the anchor's own batch there is 6.6× more compute per optimizer step
and the same reduction, so the comm barely shows. Scaling efficiency is a
function of the batch, and quoting it without the batch is meaningless.

## 4. The reconstruction, validated

§21's netem sweep ran at global batch 64, so §15's `9999 × step_time` never
applied to it. `scripts/transport_curve.py` re-expresses each point as
comm-per-step and adds it to the anchor's compute. The socket arm above is the
one point where a reconstruction and a direct measurement at batch 480 both
exist:

- reconstructed `ddp_torch` on socket: **1566 ms**
- measured `ddp_torch` on socket: **1270 ms**

**The additive model runs +23%.** It is a conservative upper bound, not an
estimate — which is the useful direction to be wrong in, and now quantified
rather than asserted. The curve, with bandwidth reported as *measured effective
throughput* rather than netem's nominal rate:

| transport | effective bus BW | step @ batch 480 | time to 3.28 | vs NVLink | source |
|---|---|---|---|---|---|
| NVLink NV12 mesh | 151 GB/s | 338 ms | 0.94 h | 1.0× | measured |
| socket, unthrottled | 0.92 GB/s | 1270 ms | 3.53 h | 3.8× | measured |
| netem nominal 40 gbit | 0.61 GB/s | 2178 ms | 6.05 h | 6.4× | reconstructed |
| netem nominal 10 gbit | 0.16 GB/s | 7595 ms | 21.10 h | 22.5× | reconstructed |

Note what the middle column does to §21's netem labels: **nominal 10 gbit
delivered 1.2 Gbit/s effective** and nominal 40 gbit delivered 4.9 — the ~8×
shortfall §21 flagged, now expressed as the quantity that can be compared to a
real network.

## 5. Two arms failed, both to memory

- **Uncompiled at 8 ranks**: `ddp_interleaved` returned 2753.2 ms and `ddp_torch`
  OOM'd outright (`tried to allocate 5.76 GiB`). Uncompiled autograd holds more
  live activations, and 30 × 1024 on a 40 GB card is already near the edge. The
  2753 ms figure is **not a clean measurement** — it is a near-OOM allocator
  thrashing, 8× the compiled number where §14's ratio at small batch was 1.7×.
  Reported here for completeness and used for nothing. The compile-vs-overlap
  question *at the anchor's batch* stays open on this box.
- **Anchor-exact chunking** (60 × 1 accum, which would have been the direct
  bridge to the anchor's 315 ms) OOM'd, as the runbook predicted. 40 GB is the
  binding constraint.

Both were the lowest-priority arms and both were sequenced last, so neither cost
anything the session needed.

## 6. One self-inflicted cost

The first launch of all five arms died in 90 seconds: the data globs were
interpolated unquoted into the driver script, so the *host* shell expanded
`fineweb_train_*.bin` into two filenames and argparse rejected the second as a
stray positional. Cost ~$1 and one relaunch. The runbook's Session C listing now
carries the quoting and the reason.

## Session outcomes

1. **Single-GPU time-to-3.28 measured: 7.19 h** on one A100 at global batch 480
   (2589.1 ms/step, 76.4% MFU). The study's headline bullet no longer rests on a
   3090 extrapolation.
2. **8 GPUs deliver 7.66×, 95.8% scaling efficiency**, on identical hardware with
   identical per-device chunking — and §14's 0.88 is revealed as an artefact of
   the bench's small default batch.
3. **A slow transport costs 3.8×, not a dealbreaker**: full-socket TCP, the worst
   realistic case short of a throttled WAN, turns 0.94 h into 3.53 h. Still 2×
   faster than one GPU.
4. **The netem reconstruction is a +23% upper bound**, calibrated against a
   direct measurement rather than asserted.
5. **`--socket PCIe` is not a fabric guarantee** on Prime Intellect. Verify with
   `nvidia-smi topo -m` before believing any transport claim.
6. **A100-SXM4-40GB roofline 270.1 TFLOP/s**, committed — and the generic `A100`
   pattern was silently catching this device with a datasheet figure until now.
7. Cost **$8.24** against a $16 ceiling; project total ≈ **$156**.

## Still open

- **PCIe remains unmeasured.** Not purchasable on this venue under a label that
  can be trusted; would need a provider whose topology is verifiable before
  renting, or a different venue entirely.
- **Compile × overlap at the anchor's batch** needs a box with 80 GB cards; 40 GB
  turns the uncompiled arm into a memory experiment.

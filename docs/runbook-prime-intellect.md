# Runbook — Prime Intellect (netem curve, DiLoCo K=8, PCIe scaling)

Three sessions on one venue. **A** is the netem curve (§15), cheap, 2 GPUs, and
blocked everywhere else. **B** is the DiLoCo K=8 converged anchor (§16, §19),
8×A100, the headline number. **C** is the PCIe scaling arm — the measured
single-GPU baseline and a real slow-interconnect point. They share §§0–5; run A
first — it is a tenth of the price and it rehearses every mechanic B depends on.

Prime Intellect is an *exchange*: an offer is a (provider, data centre, socket)
triple, each priced and stocked separately. Everything below assumes the
lambdalabs upstream, which is what both probes on 2026-08-21 used.

**Why this venue and not RunPod**: PI pods are **KVM VMs with root**, so netem
works (RunPod's containers have no `cap_net_admin` — §17) and we run our own
container, which pulls the *byte-identical* aurora image. See decisions §20 for
the full measurement. RunPod remains cheaper for plain A100 DDP work when it has
stock.

---

## 0. Before renting (aurora, free)

- [ ] `uv run pytest -q` green, tree committed. `SHA=$(git rev-parse --short HEAD)`.
- [ ] Image pushed **from the committed tree**:

  ```bash
  gh auth token | docker login ghcr.io -u adamdivak --password-stdin
  IMAGE=ghcr.io/adamdivak/distrain:$SHA scripts/container.sh push
  ```

  `container.sh push`, never `docker push` — BuildKit's provenance attestation
  makes the tag an OCI index that many registry clients 404 on (decisions §20).
- [ ] `.env` has `PRIME_API_KEY`. The team id is picked up automatically from
  `~/.prime/config.json`; confirm the **funded** wallet is the one being read:

  ```bash
  uv run --script scripts/prime_session.py status   # must show "(team ...)" and a balance
  ```

  A bare key sees only the *personal* wallet, which is empty — console top-ups
  land on the team. `--team-id ''` forces personal if ever needed.
- [ ] A GitHub PAT with **`read:packages` only**, for the pod to pull with. Do not
  ship `gh auth token` — it carries `repo` scope over every private repo.

## 1. What the script pins, and the venue's traps

```bash
uv run --script scripts/prime_session.py avail          # live stock for the shape
uv run --script scripts/prime_session.py up --dry-run   # the plan and the price, free
uv run --script scripts/prime_session.py guard &        # wall-clock + balance kill-switch
uv run --script scripts/prime_session.py down           # terminates, then verifies
uv run --script scripts/prime_session.py verify         # is anything still billing?
```

`up` refuses a ceiling the balance cannot fund, and tears the pod down itself if
anything fails after creation. **Start `guard` immediately after `up`** — it is
the only thing that stops a forgotten pod, and an idle pod bills exactly like a
busy one (a probe on 2026-08-21 wasted ~$0.71 sitting idle).

Traps, all measured:

- **`--allow-stock-image` is the correct route here, not a fallback.** PI's
  `image` field is a closed enum; a custom image needs a `customTemplateId`, and
  templates have **no API, no CLI and no console UI** on this account. So boot
  their Ubuntu and run our container inside it — which yields *better* parity
  than their registry, since `docker pull` fetches the exact aurora bytes.
- **The socket is pinned to SXM4** (`--socket`). A PCIe A100 is a different
  interconnect and would look like the §14 anchor without being comparable to it.
  Renting PCIe *deliberately*, as an arm, is fine — that is what the flag is for.
- **`envVars` are rejected by some pod types**, so they are only sent with a
  template. Nothing depends on them; `guard` enforces the ceiling.
- **Nothing survives termination.** Mirror artefacts off the pod continuously
  (§14 lost a run's only checkpoint to a credit-death termination).

## 2. Provision and first contact (~5 min)

```bash
uv run --script scripts/prime_session.py \
  --gpu-type <TYPE> --gpu-count <N> --socket <SOCKET> --provider lambdalabs \
  --name distrain-<session> up --max-hours <H> --disk-gb 200 --allow-stock-image
uv run --script scripts/prime_session.py --name distrain-<session> guard &
```

`up` prints the ssh line; `SSH` abbreviates it below.

```bash
SSH 'systemd-detect-virt; sudo grep CapEff /proc/self/status; nproc; df -h /
     nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
     nvidia-smi topo -m'
```

Expect `kvm`, `CapEff: 000001ffffffffff` (full caps — netem depends on it), and
`NV#` links in the topology on an SXM box. Docker and the NVIDIA container
toolkit are **preinstalled** on lambdalabs GPU nodes; no setup needed.

## 3. Bring up our container (~10 min, mostly the pull)

```bash
SSH 'mkdir -p ~/session_out ~/data ~/checkpoints'
# read:packages PAT on stdin -- not gh auth token
SSH 'sudo docker login ghcr.io -u adamdivak --password-stdin' < /path/to/pat.txt
SSH 'sudo docker pull ghcr.io/adamdivak/distrain:'"$SHA"
SSH 'sudo docker image inspect ghcr.io/adamdivak/distrain:'"$SHA"' --format "{{index .RepoDigests 0}}"'
```

**Record that digest and compare it to aurora's** — equality is the parity proof,
and it is stronger than anything PI's own registry can offer. Then a long-lived
container to exec into:

```bash
SSH 'sudo docker run -d --name distrain --gpus all --ipc=host --cap-add=NET_ADMIN \
       -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       -v /home/ubuntu/session_out:/workspace/session_out \
       -v /home/ubuntu/data:/workspace/data \
       -v /home/ubuntu/checkpoints:/workspace/checkpoints \
       ghcr.io/adamdivak/distrain:'"$SHA"' sleep infinity'
```

`DEX` below = `SSH 'sudo docker exec distrain ...'`. Mount only *subdirectories*
of `/workspace` — a mount over `/workspace` itself shadows the baked code and
fakes the parity check rather than failing it. `--cap-add=NET_ADMIN` is what lets
netem run in the same container as the training (verified 2026-08-21).

## 4. Parity check (~6 min)

```bash
DEX 'python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"'
DEX 'pytest -q 2>&1 | tail -3'
```

Green on the baked code, no rsync, no `uv sync` = parity proven. Note: the suite
runs *inside an image with no `.git`*; `git_image_tag()` returns
`("unknown", False)` there rather than raising (fixed 2026-08-21 — it used to
fail 29 session-script tests for reasons unrelated to what they test).

## 5. Roofline, and the MFU denominator (~5 min)

```bash
DEX 'python scripts/measure_roofline.py | tee session_out/roofline.txt'
DEX '/opt/nccl-tests/build/all_reduce_perf -b 8M -e 512M -f 2 -g <N> \
       | tee /workspace/session_out/nccl_tests.txt'
```

**If this GPU class is not in `mfu.py`, training will refuse to start** — by
design ("peaks are measured, not cited"; a datasheet figure once produced a 158%
MFU). That is not a bug to work around: add the measured `PeakSpec` to
`_PEAK_BF16` on aurora, commit, rebuild and re-push the image, or accept that no
MFU can be reported from this box. Raw ms/step is unaffected either way.

Known-good: A100-SXM4-80GB. Measured on a probe but **not committed**: A10 at
76.5 TFLOP/s. H100 entries are `UNVERIFIED` datasheet figures.

---

# Session A — the netem curve (§15)

**Shape**: cheapest box with **≥2 GPUs** (netem needs inter-rank traffic to
shape). Check `avail --gpu-count 2` at run time; ~$2–8/h has been typical.
**Ceilings: 3 h, $25.** Estimated happy path ~2 h.

**The critical mechanic**: NCCL uses NVLink/P2P by default and netem cannot touch
it. Force the socket transport so the traffic crosses `lo`, where netem applies
(decisions §12, "Track A plans around a single 8-GPU node"):

```bash
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
```

Confirm with `NCCL_DEBUG=INFO` that the transport really is `Socket` before
trusting any throttled number — if it still says P2P, every point on the curve is
measuring an unthrottled link.

### A1. Baseline and transport confirmation

```bash
DEX 'python scripts/make_synthetic_shards.py --out data/synthetic --shards 2'
DEX 'NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
     torchrun --standalone --nproc_per_node=<N> -m distrain.train \
     --distributed-mode ddp_naive --max-steps 5 --warmup-steps 0 --warmdown-steps 0 \
     --global-batch-seqs 16 --log-every 1 --no-trackio \
     --train-glob "data/synthetic/*train*.bin" --val-glob "data/synthetic/*val*.bin" \
     2>&1 | tee session_out/nccl_transport.txt | grep -i "via\|transport\|Socket"'
```

### A2. The sweep — step time per bandwidth

netem lives on `lo` *inside* the container. Apply, measure, remove, per point:

```bash
for RATE in 40gbit 10gbit 1gbit 500mbit 200mbit 100mbit; do
  DEX "tc qdisc replace dev lo root netem rate $RATE delay 1ms"
  DEX "tc qdisc show dev lo"                       # prove it took
  DEX "NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
       python scripts/bench_ddp_modes.py --nproc <N> --steps 30 --warmup 5 \
       --out-dir session_out/bench-$RATE"
done
DEX 'tc qdisc del dev lo root'
```

Unthrottled control first (no qdisc at all), then the ladder. §14 found compile
beats overlap on NVLink and predicts the lead **flips back under netem** — that
is a hypothesis this session tests, so run both compile settings at a minimum of
the fastest and slowest points.

### A3. The sanity gate (§15's "netem moved nothing but the clock")

One throttled short run on **real data**, checking the early val gates:

```bash
DEX 'NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 PYTHONUNBUFFERED=1 \
     torchrun --standalone --nproc_per_node=<N> -m distrain.train \
     --train-glob "data/fineweb10B/fineweb_train_*.bin" \
     --val-glob "data/fineweb10B/fineweb_val_*.bin" \
     --distributed-mode ddp_naive --max-steps 500 --val-every 250 \
     --run-name netem-gate > session_out/netem_gate.log 2>&1'
```

- step 0 val must print **10.8265** (= ln 50304) — the zero-init-head fingerprint
  that the right code is running.
- val ≈ **5.40 @ 250**, ≈ **4.50 @ 500**. Matching means netem changed only the
  clock; a deviation means a transport-dependent NCCL algorithm switch changed
  the numerics, which is exactly what this gate exists to catch.

Needs FineWeb — start the download (§B1) at the beginning of the session if you
intend to run A3.

### A4. What the curve yields

DDP time-to-3.28 at bandwidth *b* = **9999 × steady-state step time at *b***
(§15: the val-vs-step curve is transport-invariant, so the crossing step does not
move). No converged run per bandwidth point is needed or affordable.

---

# Session B — DiLoCo K=8 converged anchor (§16, §19)

**Shape**: `--gpu-type A100_80GB --gpu-count 8 --socket SXM4`, lambdalabs,
~$22.32/h. **Ceilings: 5 h, $120.** Estimated happy path ~3.5 h ≈ $80.

This is the K=8 counterpart to §14's DDP anchor (4.92B tokens to 3.28,
3147.1 s, ~62% MFU). Same architecture, same global batch — the method is the
only difference, which is the whole point.

### B1. Start the data download first (~7 min, background)

```bash
DEX 'mkdir -p /workspace/data/fineweb10B && \
     ln -sfn /workspace/data/fineweb10B /workspace/reference/modded_nanogpt/fineweb10B && \
     nohup python reference/modded_nanogpt/cached_fineweb10B.py 55 \
       > session_out/data_download.log 2>&1 &'
```

55 chunks = 5.5B tokens (~10.5 GiB), margin over the 4.92B consumed. Nothing
until B3 needs it. Done when `wc -l session_out/data_download.log` shows 56 files.

### B2. Mode smoke at 8 ranks (~10 min)

```bash
DEX 'NCCL_DEBUG=INFO torchrun --standalone --nproc_per_node=8 -m distrain.train \
     --distributed-mode diloco --outer-sync-every 4 --max-steps 8 \
     --warmup-steps 0 --warmdown-steps 0 --global-batch-seqs 64 \
     --log-every 1 --no-trackio \
     --train-glob "data/synthetic/*train*.bin" --val-glob "data/synthetic/*val*.bin"'
```

Expect NVLink/P2P in the transport line. A tiny `--outer-sync-every` here just
exercises the outer step; the real value is 500.

### B3. The converged run (~2.5–3 h)

Hyperparameters are **chosen, not published** (§12's box, resolved by §19):
H = 500, outer lr 0.7, **outer momentum 0.5** — μ=0.9 produced a loss excursion
that μ=0.5 removes entirely, and μ=0.5's step-2000 gap (+0.066) already matched
the baseline's best-ever (+0.058) in half the steps.

```bash
DEX 'PYTHONUNBUFFERED=1 nohup torchrun --standalone --nproc_per_node=8 -m distrain.train \
  --train-glob "data/fineweb10B/fineweb_train_*.bin" \
  --val-glob "data/fineweb10B/fineweb_val_*.bin" \
  --distributed-mode diloco \
  --outer-sync-every 500 --outer-lr 0.7 --outer-moment 0.5 \
  --global-batch-seqs 480 --grad-accum-steps 1 \
  --max-steps 10000 --val-every 500 --checkpoint-every 500 \
  --diag-val-every 250 --compile \
  --run-name diloco-k8-a100 > session_out/train.log 2>&1 &'
```

**`--val-every` must be a multiple of `--outer-sync-every`**, not a divisor of it:
train.py refuses to start otherwise, because a val on a non-sync step would
evaluate one replica's local variation rather than the shared model (§16). At
H = 500 that makes 500 the smallest legal val cadence — which costs nothing,
since the crossing resolution under DiLoCo is a multiple of H anyway. (This
runbook said 250 until a rented box caught it, 2026-08-21.)

`--compile` matches §14's `ddp_torch --compile` anchor; without it the wall-clock
half of the DDP-vs-DiLoCo comparison would be confounded by compilation rather
than by method. Measured here: 306 ms/step compiled at 8 ranks, vs the anchor's
315 ms.

Global batch 480 at 8 ranks = 60 seqs/rank, identical to the DDP anchor, so
global token throughput matches exactly. `--diag-val-every 250` evaluates every
replica just before the sync, so replica spread is observed rather than inferred
(~4% of wall clock; it never feeds the 3.28 check).

**Checkpoints are per-rank in `diloco`** (§16) — a step's 8 files are one unit,
and the set is K-specific. Mirror them off-box continuously; container disks do
not survive termination.

Sanity gates, in order — stop rather than pay for a broken run:

- step-0 val **10.8265**.
- val at 250/500 near the μ=0.5 probe's shape (it read 5.3353 at step 500). The
  probe was K=2 on a 6000-step trapezoid; this is K=8 on 10000. Treat those
  numbers as shape, not equality — the gate is "no excursion", not a value match.
- **no growing gap** after step ~1500. A gap that widens is the μ=0.9 excursion
  signature; at μ=0.5 it should not appear at all. If it does, that is a genuine
  finding about K=8 vs K=2 — record it, do not silently retune.
- the headline: **`reached target 3.28 at step N after Xs`**, first unsmoothed
  crossing, training time already excluding validation.

### B4. Monitoring

Silence looks exactly like progress (a run was once reported healthy 90 minutes
after it died). Match failure signatures, not liveness:

```bash
DEX 'grep -nE "OutOfMemory|Traceback|Killed|ChildFailedError" session_out/train.log | tail'
DEX 'tail -3 session_out/train.log'
uv run --script scripts/prime_session.py status     # balance, and the pod is still ACTIVE
```

Check the **balance** during the run, not just the clock — credit exhaustion
terminates pods (§14 lost a run at ~step 7800 that way).

---

# Session C — the single-GPU baseline and same-box scaling

**Ran 2026-08-21, $8.24, results in §22** — kept here because the shape is
reusable and two of its traps cost real money.

**What it closes.** There was no *measured* single-GPU time-to-3.28 on any
datacenter GPU, only a 3090 extrapolation. The 1-GPU and 8-GPU arms come off the
*same* box, which is the point: the scaling ratio is then self-contained and
needs no cross-box bridge.

**It was planned as a PCIe session and could not be one.** The
`A100_40GB / PCIe` offer delivered an **SXM4 box on a full NV12 mesh** — see the
topology gate in C1, which is the only reason this was caught before the numbers
were written up as "PCIe". Do not plan another PCIe session on this venue
without a way to verify the fabric *before* renting.

**Shape**: `--gpu-type A100_40GB --gpu-count 8`, lambdalabs, ~$15.92/h.
**Ceilings: 1 h, $16.** Actual: 31 min, $8.24.

### C0. Deviations from §§0–5, and why

- **No image rebuild or push.** `git diff c8c72e1..HEAD` touches only
  `README.md`, `docs/`. The already-pushed `c8c72e1`
  (`sha256:f915028c1c50ce23…`) is therefore *code-identical* to HEAD and was
  already proven on a rented box (226 tests, 2026-08-21). Pull that tag.
- **Skip the §4 `pytest` parity run.** It costs ~30 min — most of this
  session's budget — to re-prove what the digest already proves on an image
  whose suite has passed on a rented box. The digest comparison stays; it is
  the actual parity evidence. (Deviation recorded here so the session log does
  not have to defend it twice.)
- **Two data chunks, not 55.** These are step-time measurements, not converged
  runs, so the corpus only needs to outlast ~25 steps × 480 seqs ≈ 12.3M
  tokens. One chunk is 100M. `cached_fineweb10B.py 2` takes well under a
  minute against §B1's ~7.
- **MFU printed on the pod will be wrong — ignore it.** `"A100"` substring-matches
  `NVIDIA A100-PCIE-40GB` in `_PEAK_BF16` *before* any PCIe entry exists, so
  training silently uses the `UNVERIFIED` 312.0 datasheet figure instead of
  refusing. Raw ms/step is unaffected, and that is the measurement. Recompute
  MFU on aurora against the roofline measured in C2, and commit the measured
  `PeakSpec` afterwards.

### C1. Provision, guard, first contact

```bash
uv run --script scripts/prime_session.py \
  --gpu-type A100_40GB --gpu-count 8 --socket PCIe --provider lambdalabs \
  --name distrain-pcie up --max-hours 1 --disk-gb 200 --allow-stock-image
uv run --script scripts/prime_session.py --name distrain-pcie guard &
```

```bash
SSH 'systemd-detect-virt; nproc; nvidia-smi --query-gpu=name,memory.total --format=csv
     nvidia-smi topo -m'
```

**The topology line is a gate, not a formality — and on 2026-08-21 it fired.**
`--socket PCIe` returned `NVIDIA A100-SXM4-40GB` with `NV12` between every GPU
pair: a full NVLink mesh sold under a PCIe label. The socket field is provider
metadata, not a fabric guarantee. If `topo -m` shows any `NV#`, no number from
the box may be published as PCIe. Record the matrix verbatim — the §21 habit of
verifying the transport before trusting a number applies here exactly.

### C2. Container, data, roofline, bandwidth

```bash
SSH 'mkdir -p ~/session_out ~/data'
SSH 'sudo docker login ghcr.io -u adamdivak --password-stdin' < /path/to/pat.txt
SSH 'sudo docker pull ghcr.io/adamdivak/distrain:c8c72e1'
SSH 'sudo docker image inspect ghcr.io/adamdivak/distrain:c8c72e1 --format "{{index .RepoDigests 0}}"'
# must equal sha256:f915028c1c50ce239b44fd7e2ec1c674f2e36adb4d1bee971c8729f29305d789
SSH 'sudo docker run -d --name distrain --gpus all --ipc=host \
       -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       -v /home/ubuntu/session_out:/workspace/session_out \
       -v /home/ubuntu/data:/workspace/data \
       ghcr.io/adamdivak/distrain:c8c72e1 sleep infinity'
```

```bash
DEX 'mkdir -p /workspace/data/fineweb10B && \
     ln -sfn /workspace/data/fineweb10B /workspace/reference/modded_nanogpt/fineweb10B && \
     nohup python reference/modded_nanogpt/cached_fineweb10B.py 2 \
       > session_out/data_download.log 2>&1 &'
DEX 'python scripts/measure_roofline.py | tee session_out/roofline.txt'
DEX '/opt/nccl-tests/build/all_reduce_perf -b 8M -e 512M -f 2 -g 8 \
       | tee /workspace/session_out/nccl_tests.txt'
```

The all-reduce number is a headline in its own right: it is the third real point
on the transport axis, against NVLink's 154 GB/s and the 2×3090's 2.29 GB/s.

### C3. The two arms — matched micro-batch, global batch 480

Both arms hold **global batch 480** (the anchor's) and **micro-batch 30
sequences per device**. Matching the micro-batch is what makes the ratio mean
something: each device does identical compute chunks in both arms, so the only
difference is the rank count and the communication. 30 rather than the anchor's
60 because these are 40 GB cards; the global batch, the token accounting and the
comm volume per optimizer step are all unchanged by the chunking.

`bench_ddp_modes.py` hardcodes `--grad-accum-steps 1` and derives the global
batch from `--per-gpu-batch × nproc`; both are overridden by the pass-through
args after `--`, which argparse resolves last-wins. Data globs go there too.

**Quote the glob values.** Unquoted, the pod's shell expands
`fineweb_train_*.bin` into the actual shard names and argparse rejects the second
as a stray positional — all five arms died in 90 seconds this way, ~$1 and one
relaunch (2026-08-21).

```bash
TG='data/fineweb10B/fineweb_train_*.bin'; VG='data/fineweb10B/fineweb_val_*.bin'
GLOBS="--train-glob \"$TG\" --val-glob \"$VG\""

# Arm 1 -- single GPU: 480 = 30 x 16 accumulation steps
DEX "python scripts/bench_ddp_modes.py --nproc 1 --modes ddp_torch --steps 25 --warmup 10 \
       --per-gpu-batch 30 --timeout 1800 --out-dir session_out/bench-1gpu \
       -- --global-batch-seqs 480 --grad-accum-steps 16 $GLOBS"

# Arm 2 -- 8 GPUs: 480 = 30 x 2 accumulation steps x 8 ranks
DEX "NCCL_DEBUG=INFO python scripts/bench_ddp_modes.py --nproc 8 --no-single \
       --modes ddp_torch ddp_interleaved --steps 25 --warmup 10 \
       --per-gpu-batch 30 --timeout 1800 --out-dir session_out/bench-8gpu \
       -- --global-batch-seqs 480 --grad-accum-steps 2 $GLOBS"

# Arm 2b -- the overlap question, on a transport in the middle regime
DEX "python scripts/bench_ddp_modes.py --nproc 8 --no-single --no-compile \
       --modes ddp_interleaved ddp_torch --steps 25 --warmup 10 \
       --per-gpu-batch 30 --timeout 1800 --out-dir session_out/bench-8gpu-nc \
       -- --global-batch-seqs 480 --grad-accum-steps 2 $GLOBS"
```

Arm 1 runs `single` *and* `ddp_torch` at one rank (the harness cannot be asked
for `single` alone); `single` is the number that matters and the 1-rank
`ddp_torch` is a free cross-check on wrapper overhead.

Arm 2b is the interesting one. §21 found the compile-vs-overlap lead *dissolves*
as bandwidth falls — every mode within 0.12% at netem's slowest point — and
that mode choice therefore matters only in the **middle regime**, where compute
and comm are comparable. PCIe at batch 480 is a real instance of that regime, so
this is a direct test of §21's claim on hardware rather than on netem.

**Note `tokens_per_s_total` in `results.json` is wrong for these arms** — the
harness computes it from `per_gpu_batch × nproc`, which the accumulation
override invalidates. `mean_ms` is correct and is the only field used.

### C4. Bonus, only if the clock allows — anchor-matched chunking

```bash
DEX "python scripts/bench_ddp_modes.py --nproc 8 --no-single --modes ddp_torch \
       --steps 25 --warmup 10 --per-gpu-batch 60 --timeout 1800 \
       --out-dir session_out/bench-8gpu-b60 \
       -- --global-batch-seqs 480 --grad-accum-steps 1 $GLOBS"
```

60 seqs/device on a 40 GB card may OOM — the logits tensor is the high-water
mark (§4). A failure here is a recorded result, not a problem; the harness
records it and moves on. If it runs, it is directly comparable to the anchor's
315 ms. **It OOM'd on 2026-08-21** (`tried to allocate 5.76 GiB`), as did the
uncompiled 8-rank arm; uncompiled autograd holds more live activations, and its
2753 ms result is allocator thrash rather than a measurement. Both arms need
80 GB cards. Sequence them last, as here, so they cost nothing that matters.

### C5. What the arms yield

- **Single-GPU time-to-3.28 = 9999 × arm-1 step time** (§15's rule, which holds
  a fortiori at world size 1: the val-vs-step curve does not depend on the rank
  count, and the three-way replication of §14 bounds the residual to ≤0.01).
- **Scaling efficiency = arm-1 step / arm-2 step / 8**, on identical hardware
  and identical per-device chunking.
- **Time-to-3.28 on PCIe = 9999 × arm-2 step time**, which answers "is a slow
  interconnect a dealbreaker or a nuisance" with a real fabric instead of netem.
- Feed the measured bus bandwidth and step times to
  `scripts/transport_curve.py` to place PCIe on the same axis as NVLink and the
  netem points.

---

## Pull everything, then tear down

Artefacts are on the *host*, not in the container, because of the mounts:

```bash
rsync -avz ubuntu@<IP>:session_out/    ~/work/distrain/out/prime-<session>/
rsync -avz ubuntu@<IP>:checkpoints/    ~/work/distrain/out/prime-<session>/checkpoints/
```

Check locally (`ls -la`, read the log tail) **before** terminating — nothing
survives it:

```bash
uv run --script scripts/prime_session.py --name distrain-<session> down
uv run --script scripts/prime_session.py verify     # non-zero if anything still bills
```

## Afterwards (aurora, free)

- Commit the measured `PeakSpec` for any new GPU class into `mfu.py`.
- Session log in `docs/sessions/`, recording: the image digest **and** whether it
  matched aurora's, the provider/data-centre/socket actually rented, the NCCL
  transport line, and the spend.
- **A**: the DDP step-time-vs-bandwidth table, whether compile-vs-overlap flipped
  under netem, and the A3 gate result. Update §15 with measured points.
- **B**: tokens-to-3.28 for DiLoCo K=8 against §14's DDP 4.92B / 3147.1 s. This
  is the study's headline comparison — report wall-clock *and* tokens, since
  DiLoCo trades the second for the first.
- Update README's status and known gaps, and decisions §20 with venue actuals.

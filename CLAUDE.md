# distrain — working notes

A self-funded distributed-training scaling study. Read [`project_brief.md`](project_brief.md)
for intent and [`docs/decisions.md`](docs/decisions.md) for everything settled since.
[`README.md`](README.md) has current status and known gaps. Those three are the source
of truth; this file is only what a session needs before touching anything.

## Where things run

| Machine | Role |
|---|---|
| `aurora` (RTX 3090), `adam@aurora` over Tailscale, `~/work/distrain` | **default**: editing, tests, all CUDA work, long runs |
| rented cloud nodes | all reported results — none rented yet |
| the Mac (arm64, no NVIDIA) | fallback editor; CPU/MPS correctness only |

Work directly on aurora: `uv run pytest -q`. Real FineWeb data is at
`data/fineweb10B` there; the trackio dashboard is `uv run trackio show --project
distrain` (port 7860). From the Mac (fallback), iterate with rsync, not git:

```bash
scripts/sync-aurora.sh && ssh adam@aurora 'cd ~/work/distrain && uv run pytest -q'
```

`uv` on aurora may need its full path (`~/.local/bin/uv`) over non-interactive SSH.
Commit at milestones. Remote `github.com/adamdivak/distrain` is **private**.

## Things that are load-bearing

Breaking any of these invalidates results silently rather than loudly, which is worse
than a crash. Each is enforced by a test — do not weaken them to make a test pass.

- **Data order must not depend on world size.** `ShardingPlan` fixes what step *t*
  consumes regardless of rank count; `tests/test_data.py` asserts byte-identical token
  streams across world sizes 1/2/4/8. Changing GPU count must never change data order.
- **One FLOPs convention, in `mfu.py` only.** PaLM-style `6N + 12*L*H*Q*T`, true MFU
  reported, HFU logged separately when checkpointing is on, bf16 **dense** peaks.
- **Peaks are measured, not cited.** A datasheet figure for the 3090 (35.6 TFLOP/s —
  actually its FP32 non-tensor rate) once produced a 158% MFU. Run
  `scripts/measure_roofline.py` on any new GPU class; datacenter entries are marked
  `UNVERIFIED` until then. MFU above 100% means the denominator is wrong.
- **Time-to-target-loss is the first unsmoothed crossing**, with training time
  excluding validation.
- **Architecture must be identical across configs.** The 3.28 target constrains data
  and tokenizer, not model shape — but a scaling comparison is meaningless if the
  model differs between runs.
- **Initialization must not depend on world size either.** Seeds are `cfg.seed +
  cfg.rank` — rank-dependent (so dropout decorrelates across replicas) but *never*
  scaled by `world_size`, which would make rank 0's stream change when GPU count
  changes and confound every cross-config comparison. Replica equality comes from
  the rank-0 broadcast in `DistributedSynchronizer.__init__`, not from seeding.
- **Gradient averaging is `SUM` then `/ world_size`, in the synchronizer only.**
  `ReduceOp.AVG` is NCCL-only; gloo lacks it, and gloo is the correctness backend.
  The accumulation divisor stays in the training loop — two chunkings, two divisors.
  See `docs/decisions.md` §6.
- **Collective order must be identical on every rank.** Collectives match by order of
  invocation, not by tensor name. A rank-dependent sequence deadlocks or silently
  pairs the wrong tensors. Never make participation conditional on per-rank state.

## Costs are real

Budget is $150 target / $400 ceiling of the user's own money. Never debug on rented
hardware; aurora is free. Before anything is rented, the launcher needs a hard
wall-clock ceiling and teardown-on-exception (`docs/decisions.md` §9).

## Conventions

- Vendored upstream code in `reference/` is verbatim and lint-excluded. Do not edit it.
- `data/`, `.venv/`, `out/`, `checkpoints/` are gitignored and machine-local.
- Real FineWeb shards are 190 MiB each, ~19 GiB for the full set. 
  Synthetic shards were used initially in local development.
- Keep documents and code comments concise.
- Do not add 'Co-authored by Claude' at the end of commit messages.

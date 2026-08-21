"""Put every measured transport on one axis: effective bandwidth -> time-to-3.28.

The netem sweep (decisions.md section 21) ran at global batch 64, so section 15's
`time-to-3.28 = 9999 x step_time` cannot be applied to it directly -- that rule
needs the anchor's batch of 480. What *does* transfer is the communication cost,
because the all-reduce moves one gradient tensor per optimizer step regardless of
batch size. So:

    reconstructed_step(b) = anchor_compute + comm(b)
    comm(b)               = measured_step(b) - compute_at_bench_batch

and the anchor's own compute comes from its measured step time minus the NVLink
comm implied by its measured bus bandwidth.

Two honesty rules this script enforces by construction:

- **Reconstructed points are labelled as such.** Only the NVLink row is a
  measurement of a converged run; everything else is `anchor_compute + comm`.
- **Bandwidth is reported as measured effective bus bandwidth, never as netem's
  nominal rate.** netem's rate limiter over loopback, with NCCL's parallel socket
  channels, came in ~8x below nominal, so a row labelled "10 gbit" that actually
  delivered 1.25 Gbit/s would misrepresent every real network it is compared to.

Reconstruction is an *upper bound* for overlapping modes: at batch 480 there is
~6x more compute to hide communication behind than at batch 64, so a mode that
truly overlaps would beat the additive model. Under `torch.compile` the backward
is one fused graph and the hooks all fire at its end (decisions.md section 21),
so for the compiled modes the additive model is the right one.

Usage:

    uv run python scripts/transport_curve.py \
        --bench "socket (unthrottled)=out/prime-diloco/session_out/bench-control" \
        --bench "netem 40gbit=out/prime-diloco/session_out/bench-40gbit-c" \
        --bench "netem 10gbit=out/prime-diloco/session_out/bench-10gbit-c"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Ring all-reduce moves 2(n-1)/n x S bytes per rank; nccl-tests reports bus
# bandwidth on that same convention, so the two are directly comparable.
def ring_factor(ranks: int) -> float:
    return 2.0 * (ranks - 1) / ranks


def bus_bandwidth_gbps(bytes_reduced: float, seconds: float, ranks: int) -> float:
    """Effective bus bandwidth in GB/s, on nccl-tests' convention."""
    return bytes_reduced * ring_factor(ranks) / seconds / 1e9


def comm_seconds(bytes_reduced: float, bus_gbps: float, ranks: int) -> float:
    """Inverse of the above: how long one all-reduce takes at a given bus bandwidth."""
    return bytes_reduced * ring_factor(ranks) / (bus_gbps * 1e9)


def load_bench(path: Path) -> dict[str, float]:
    """mode -> mean_ms for one bench run (accepts the dir or the results.json)."""
    if path.is_dir():
        found = sorted(path.rglob("results.json"))
        if not found:
            raise FileNotFoundError(f"no results.json under {path}")
        path = found[-1]
    data = json.loads(path.read_text())
    return {r["label"]: r["mean_ms"] for r in data["results"] if "mean_ms" in r}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", action="append", default=[], metavar="LABEL=PATH",
                   help="a bench results dir (or results.json), labelled")
    p.add_argument("--mode", default="ddp_torch",
                   help="which mode's step time to reconstruct from")
    p.add_argument("--bench-compute-ms", type=float, default=49.5,
                   help="single-GPU step time at the bench's per-GPU batch "
                        "(decisions section 14 matrix: 49.5 compiled, 86.1 uncompiled)")
    p.add_argument("--anchor-step-ms", type=float, default=315.0,
                   help="measured 8-GPU step time at global batch 480 over NVLink")
    p.add_argument("--anchor-bus-gbps", type=float, default=154.0,
                   help="measured NCCL all-reduce bus bandwidth on the anchor box")
    p.add_argument("--params", type=float, default=162e6)
    p.add_argument("--grad-bytes", type=int, default=4, help="fp32 gradients")
    p.add_argument("--ranks", type=int, default=8)
    p.add_argument("--crossing-step", type=int, default=9999,
                   help="first unsmoothed crossing of 3.28 (decisions section 14)")
    args = p.parse_args(argv)

    reduced = args.params * args.grad_bytes
    anchor_comm_ms = comm_seconds(reduced, args.anchor_bus_gbps, args.ranks) * 1e3
    anchor_compute_ms = args.anchor_step_ms - anchor_comm_ms

    print(f"gradient tensor      {reduced / 1e6:.0f} MB fp32 ({args.params / 1e6:.0f}M params)")
    print(f"ring traffic/rank    {reduced * ring_factor(args.ranks) / 1e6:.0f} MB "
          f"at {args.ranks} ranks")
    print(f"anchor step          {args.anchor_step_ms:.1f} ms measured "
          f"(global batch 480, NVLink)")
    print(f"anchor comm          {anchor_comm_ms:.1f} ms implied by "
          f"{args.anchor_bus_gbps:.0f} GB/s measured bus bandwidth")
    print(f"anchor compute       {anchor_compute_ms:.1f} ms  <- the reconstruction base")
    print(f"bench compute        {args.bench_compute_ms:.1f} ms "
          f"(single GPU, bench batch)\n")

    rows = [("NVLink NV12 mesh", args.anchor_bus_gbps, anchor_comm_ms,
             args.anchor_step_ms, "measured")]

    for spec in args.bench:
        label, _, raw = spec.partition("=")
        by_mode = load_bench(Path(raw))
        if args.mode not in by_mode:
            print(f"  ! {label}: no {args.mode} result, have {sorted(by_mode)}")
            continue
        step_ms = by_mode[args.mode]
        comm_ms = step_ms - args.bench_compute_ms
        bus = bus_bandwidth_gbps(reduced, comm_ms / 1e3, args.ranks)
        rows.append((label, bus, comm_ms, anchor_compute_ms + comm_ms, "reconstructed"))

    hdr = (f"| transport | effective bus BW | comm/step | step @ batch 480 | "
           f"time to 3.28 | vs NVLink | source |")
    print(hdr)
    print("|---|---|---|---|---|---|---|")
    base = None
    for label, bus, comm_ms, step_ms, source in rows:
        hours = args.crossing_step * step_ms / 1e3 / 3600
        base = base or hours
        print(f"| {label} | {bus:.2f} GB/s ({bus * 8:.1f} Gbit/s) | {comm_ms:.0f} ms | "
              f"{step_ms:.0f} ms | {hours:.2f} h | {hours / base:.1f}x | {source} |")

    print("\nReconstructed rows are anchor_compute + measured comm, not converged runs.")
    print("Bandwidth is measured effective throughput, not netem's nominal rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

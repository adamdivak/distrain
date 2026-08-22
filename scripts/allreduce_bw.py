"""Effective all-reduce bus bandwidth, on nccl-tests' convention.

The pinned image carries `nccl-tests` and that is the tool of record. This exists
for the case that keeps happening anyway: the only tier with stock cannot boot a
22 GB image (2026-08-22, a community A100-PCIe pod whose container never started),
while bandwidth is the one number the transport curve actually needs from a box --
`transport_curve.py` turns it into a step time at any rank count. So this runs in
*any* PyTorch container, with no build step and no dependencies beyond torch.

    torchrun --standalone --nproc_per_node=2 scripts/allreduce_bw.py

Sizes and convention match `all_reduce_perf -b 8M -e 512M -f 2`: bus bandwidth is
algorithm bandwidth scaled by the ring's 2(n-1)/n, so numbers from different rank
counts are comparable -- which is the whole reason a 2-GPU box can say anything
about an 8-GPU one.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


def ring_factor(ranks: int) -> float:
    """Ring all-reduce moves 2(n-1)/n x S bytes per rank."""
    return 2.0 * (ranks - 1) / ranks


def bus_bandwidth_gbps(nbytes: int, seconds: float, ranks: int) -> float:
    """nccl-tests' busbw: algbw x the ring factor, in GB/s."""
    return nbytes * ring_factor(ranks) / seconds / 1e9


def time_all_reduce(tensor: torch.Tensor, iters: int, warmup: int) -> float:
    """Mean seconds per all-reduce, warmup excluded and the device synchronized."""
    for _ in range(warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-bytes", type=int, default=8 * 1024**2)
    parser.add_argument("--max-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    # Device first, then the process group: NCCL binds the current device, and
    # initializing before selecting it makes rank 0's device everyone's default.
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    rank, ranks = dist.get_rank(), dist.get_world_size()

    rows = []
    nbytes = args.min_bytes
    while nbytes <= args.max_bytes:
        tensor = torch.ones(nbytes // 4, dtype=torch.float32, device="cuda")
        seconds = time_all_reduce(tensor, args.iters, args.warmup)
        row = {
            "bytes": nbytes,
            "time_us": seconds * 1e6,
            "algbw_gbps": nbytes / seconds / 1e9,
            "busbw_gbps": bus_bandwidth_gbps(nbytes, seconds, ranks),
        }
        rows.append(row)
        if rank == 0:
            print(f"{nbytes:>12} {row['time_us']:>12.2f} us "
                  f"{row['algbw_gbps']:>8.2f} {row['busbw_gbps']:>8.2f} GB/s")
        del tensor
        torch.cuda.empty_cache()
        nbytes *= 2

    if rank == 0:
        average = sum(r["busbw_gbps"] for r in rows) / len(rows)
        print(f"# ranks {ranks}, device {torch.cuda.get_device_name(0)}")
        print(f"# Avg bus bandwidth : {average:.2f}")
        if args.out:
            with open(args.out, "w") as handle:
                json.dump({"ranks": ranks, "device": torch.cuda.get_device_name(0),
                           "avg_busbw_gbps": average, "sizes": rows}, handle, indent=2)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

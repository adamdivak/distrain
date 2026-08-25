"""Show which DDP modes still overlap communication once torch.compile is on.

Overlap is only possible if gradients become ready *during* the backward pass.
This probe measures when they do, for the three shapes the study runs:

    eager        autograd graph intact, gradients trickle out
    compiled     AOTAutograd fuses the backward into one node -- the shape our
                 hand-rolled ddp_bucketed / ddp_interleaved run in under
                 --compile, where every post-accumulate-grad hook fires at the
                 very end and there is nothing left to overlap with
    compiled DDP DDPOptimizer splits the graph at DDP's bucket boundaries, so
                 gradients arrive in stages and upstream DDP keeps its overlap

It also prints the split count, which is the price of that overlap: one
subgraph per bucket, each compiled separately, with no fusion across the seams.

Structural, not a benchmark: gloo on CPU. Bucket count depends on parameter
bytes and the cap, not on device, so this answers the same question the rented
8xA100 benches raise -- for free, on aurora. Backs docs/decisions.md section 26.

Two ranks are required: DDP short-circuits to a single bucket at world_size 1
and dynamo then never builds a DDPOptimizer, so a one-rank run measures a
different thing (a graph break in DDP's Python forward) and reports no split.

    uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \
        scripts/probe_compile_overlap.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distrain.model import GPT, GPTConfig


def build(block: int) -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(block_size=block, vocab_size=50304, n_layer=12, n_head=12,
                         n_embd=768, dropout=0.0, bias=False))


def arrival_profile(tag: str, model: torch.nn.Module, params: list[torch.nn.Parameter],
                    block: int, iters: int) -> None:
    """Where in the backward each gradient becomes ready, as a % of its duration."""
    # Without a reset, dynamo reuses the code it compiled for the previous
    # configuration -- same code object, same guards -- and the DDP row silently
    # measures the plain compiled graph again.
    torch._dynamo.reset()

    arrivals: list[float] = []
    for param in params:
        param.register_post_accumulate_grad_hook(
            lambda _p, a=arrivals: a.append(time.perf_counter()))

    x = torch.randint(0, 50304, (2, block))
    start = end = 0.0
    for _ in range(iters):  # first iterations compile; the last one is reported
        arrivals.clear()
        model.zero_grad(set_to_none=True)
        _, loss = model(x, x)
        start = time.perf_counter()
        loss.backward()
        end = time.perf_counter()

    span = end - start
    ordered = sorted(arrivals)
    pct = lambda value: (value - start) / span * 100
    print(f"{tag:<22} backward {span * 1e3:7.0f} ms | {len(arrivals)} grads | "
          f"first at {pct(ordered[0]):5.1f}%  median {pct(ordered[len(ordered) // 2]):5.1f}%  "
          f"last {pct(ordered[-1]):5.1f}% of backward")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket-cap-mb", type=float, default=25.0,
                        help="must match the --ddp-bucket-size the benches use")
    parser.add_argument("--block", type=int, default=256, help="sequence length")
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args(argv)

    dist.init_process_group("gloo")
    quiet = dist.get_rank() != 0
    if quiet:
        sys.stdout = open("/dev/null", "w")  # noqa: SIM115
    elif dist.get_world_size() < 2:
        print("WARNING: one rank -- DDP builds a single bucket and dynamo builds no\n"
              "DDPOptimizer, so the last row below is not the configuration the\n"
              "benches run. Relaunch under torch.distributed.run --nproc_per_node=2.\n")
    try:
        model = build(args.block)
        grad_bytes = sum(p.numel() * 4 for p in model.parameters() if p.requires_grad)
        print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, "
              f"{grad_bytes / 2**20:.0f} MiB of fp32 gradients, "
              f"{args.bucket_cap_mb:.0f} MB bucket cap")

        arrival_profile("eager", model, list(model.parameters()), args.block, args.iters)

        model = build(args.block)
        arrival_profile("compiled", torch.compile(model), list(model.parameters()),
                        args.block, args.iters)

        model = build(args.block)
        ddp = torch.nn.parallel.DistributedDataParallel(
            model, broadcast_buffers=False, bucket_cap_mb=args.bucket_cap_mb)
        # DDPOptimizer stashes its bucket list on itself; the class is patched
        # rather than the instance because dynamo constructs the instance at
        # compile time, out of reach.
        from torch._dynamo.backends.distributed import DDPOptimizer
        seen: dict[str, object] = {}
        original = DDPOptimizer.compile_fn

        def spy(self, gm, example_inputs, **configs):
            compiled = original(self, gm, example_inputs, **configs)
            seen["subgraphs"] = len(self.buckets)
            seen["sizes_mb"] = [round(b.size / 2**20, 1) for b in self.buckets]
            return compiled

        DDPOptimizer.compile_fn = spy
        try:
            arrival_profile("compiled + torch DDP", torch.compile(ddp),
                            list(model.parameters()), args.block, args.iters)
        finally:
            DDPOptimizer.compile_fn = original

        print(f"\nDDPOptimizer split the graph into {seen.get('subgraphs')} subgraphs: "
              f"{seen.get('sizes_mb')} MiB")
        print("The two outsized pieces are the untied wte and lm_head; each is far "
              "larger than\nthe cap and cannot be split, so the last one cannot "
              "overlap with anything.")
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

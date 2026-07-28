"""Measure a GPU's achievable bf16 GEMM throughput.

Run this on any new GPU class before trusting its MFU numbers, and record the result
in `_PEAK_BF16` in `distrain/mfu.py`. Vendor datasheets mix tensor and non-tensor
rates, dense and 2:4-sparse figures, and FP16- versus FP32-accumulate variants; a
large square GEMM is unambiguous.

    uv run python scripts/measure_roofline.py

The reported number is a *lower bound* on the true peak -- it is what the hardware
demonstrably sustains, which is the right denominator for an honest MFU.
"""

from __future__ import annotations

import argparse

import torch


def bench(n: int, dtype: torch.dtype, iters: int = 30, warmup: int = 10) -> float:
    """Sustained TFLOP/s for an n x n by n x n matmul."""
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(warmup):
        c = a @ b
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        c = a @ b
    end.record()
    torch.cuda.synchronize()

    if not torch.isfinite(c).all():
        raise RuntimeError(f"non-finite result at n={n}, dtype={dtype}; timing is meaningless")
    seconds = start.elapsed_time(end) / 1e3 / iters
    return 2 * n**3 / seconds / 1e12


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[4096, 8192, 16384])
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    name = torch.cuda.get_device_name()
    print(f"device: {name}")

    best = 0.0
    for n in args.sizes:
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            tflops = bench(n, dtype, args.iters)
            label = str(dtype).removeprefix("torch.")
            print(f"  n={n:<6d} {label:<10s} {tflops:7.1f} TFLOP/s")
            if dtype is torch.bfloat16:
                best = max(best, tflops)

    print(f"\nbest sustained bf16: {best:.1f} TFLOP/s")
    print("record in distrain/mfu.py as:")
    print(f'    PeakSpec({best:.1f}, "measured on <host> <date>, {max(args.sizes)}^3 bf16 GEMM")')

    try:
        from distrain.mfu import peak_bf16_spec

        spec = peak_bf16_spec(name)
        print(f"\ncurrently recorded: {spec.tflops:.1f} TFLOP/s ({spec.source})")
        # run-to-run spread on the same card is a few tenths of a percent; only flag a
        # gap big enough to mean the recorded number describes something else
        if best > spec.tflops * 1.02:
            print(
                f"WARNING: measured {best:.1f} exceeds the recorded peak "
                f"{spec.tflops:.1f}. The table is wrong -- any MFU computed against it "
                f"is inflated by {best / spec.tflops:.2f}x."
            )
    except KeyError as e:
        print(f"\nnot yet in the table: {e}")


if __name__ == "__main__":
    main()

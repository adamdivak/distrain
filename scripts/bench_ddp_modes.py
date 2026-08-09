"""Time the hand-rolled DDP modes against each other on one machine.

The measurement the three modes exist for (decisions.md section 6): identical
hardware, identical model, only the synchronization strategy varies. Runs a
single-process baseline plus each requested mode under torchrun, parses per-step
times from the training log, drops warmup steps, and writes everything durable
before the machine disappears -- this is written for the first rented multi-GPU
session, where the box is torn down minutes after the numbers exist.

Usage on a 2-GPU box (NCCL, compiled, real settings):

    uv run python scripts/bench_ddp_modes.py --nproc 2 --steps 50 --warmup 10

Local mechanics check on CPU/gloo (timings meaningless, harness exercised):

    uv run python scripts/bench_ddp_modes.py --nproc 2 --steps 4 --warmup 1 \
        --backend gloo --no-compile --no-single -- --device cpu

Everything after `--` is forwarded to `distrain.train` verbatim (data globs,
model size, bucket size, ...). Results land in `out/bench/<timestamp>/`:
`results.json` with raw per-step times, plus one `.log` per run.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

MODES = ["ddp_naive", "ddp_bucketed", "ddp_interleaved"]
SINGLE = "single"

# Matches the per-step line train.py prints: "step  3 | loss 4.1 | lr 6e-04 |  123.4 ms | mfu 68%"
STEP_LINE = re.compile(r"step\s+(\d+)\s+\|.*\|\s*([\d.]+)\s+ms")


def parse_step_times_ms(log: str) -> dict[int, float]:
    return {int(m.group(1)): float(m.group(2)) for m in STEP_LINE.finditer(log)}


def stats_ms(times: list[float]) -> dict:
    return {
        "n": len(times),
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
    }


def build_command(label: str, args, train_args: list[str]) -> list[str]:
    nproc = 1 if label == SINGLE else args.nproc
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={nproc}",
        "-m", "distrain.train",
        "--max-steps", str(args.steps),
        "--log-every", "1",          # the parser needs every step's time
        "--val-every", "0",          # timing only; eval would pollute the clock
        "--no-trackio",
        "--seq-len", str(args.seq_len),
        "--global-batch-seqs", str(args.per_gpu_batch * nproc),
        "--grad-accum-steps", "1",
    ]
    if args.compile:
        cmd.append("--compile")
    if label != SINGLE:
        cmd += ["--distributed-mode", label, "--distributed-backend", args.backend]
    return cmd + train_args


def run_one(label: str, args, train_args: list[str], out_dir: Path) -> dict:
    cmd = build_command(label, args, train_args)
    print(f"=== {label}: {' '.join(cmd)}", flush=True)
    result = {"label": label, "command": cmd}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout,
                              check=False)
        log = proc.stdout + proc.stderr
        result["returncode"] = proc.returncode
    except subprocess.TimeoutExpired as e:
        # A hang is the expected *failure* mode of a distributed bug, so it is a
        # recorded result, not an exception -- the remaining modes still run.
        log = ((e.stdout or b"").decode(errors="replace")
               + (e.stderr or b"").decode(errors="replace"))
        result["returncode"] = None
        result["error"] = f"timeout after {args.timeout}s"
    (out_dir / f"{label}.log").write_text(log)

    if result["returncode"] == 0:
        by_step = parse_step_times_ms(log)
        timed = [ms for step, ms in sorted(by_step.items()) if step >= args.warmup]
        if len(timed) != args.steps - args.warmup:
            result["error"] = (f"parsed {len(timed)} timed steps, "
                               f"expected {args.steps - args.warmup}")
        else:
            nproc = 1 if label == SINGLE else args.nproc
            result["step_times_ms"] = timed
            result.update(stats_ms(timed))
            result["tokens_per_s_total"] = (
                args.per_gpu_batch * args.seq_len * nproc / (result["mean_ms"] / 1e3))
    elif "error" not in result:
        result["error"] = f"exit code {result['returncode']}, see {label}.log"
        print(log[-2000:], flush=True)
    return result


def print_table(results: list[dict]) -> None:
    baseline = next((r for r in results if r["label"] == SINGLE and "mean_ms" in r), None)
    naive = next((r for r in results if r["label"] == "ddp_naive" and "mean_ms" in r), None)
    print(f"\n{'mode':<18}{'mean ms':>9}{'median':>9}{'std':>7}{'tok/s':>10}"
          f"{'vs naive':>10}{'scaling':>9}")
    for r in results:
        if "mean_ms" not in r:
            print(f"{r['label']:<18}FAILED: {r.get('error', 'unknown')}")
            continue
        vs_naive = (f"{naive['mean_ms'] / r['mean_ms']:.2f}x"
                    if naive and r["label"] != SINGLE else "")
        # Per-GPU batch is fixed, so ideal scaling keeps step time flat: the
        # efficiency of mode m is t_single / t_m, and 1.00 means free ranks
        scaling = (f"{baseline['mean_ms'] / r['mean_ms']:.2f}"
                   if baseline and r["label"] != SINGLE else "")
        print(f"{r['label']:<18}{r['mean_ms']:>9.1f}{r['median_ms']:>9.1f}"
              f"{r['std_ms']:>7.1f}{r['tokens_per_s_total']:>10.0f}"
              f"{vs_naive:>10}{scaling:>9}")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if "--" in argv:
        split = argv.index("--")
        argv, train_args = argv[:split], argv[split + 1:]
    else:
        train_args = []

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nproc", type=int, default=2)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10,
                   help="leading steps excluded from stats (compile + allocator warmup)")
    p.add_argument("--backend", default="nccl")
    p.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    p.add_argument("--single", action=argparse.BooleanOptionalAction, default=True,
                   help="also run a 1-process baseline for scaling efficiency")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--per-gpu-batch", type=int, default=8,
                   help="sequences per rank per step; global batch = this x nproc")
    p.add_argument("--timeout", type=int, default=900,
                   help="seconds per run before it is recorded as hung")
    p.add_argument("--out-dir", default="out/bench")
    args = p.parse_args(argv)

    if args.warmup >= args.steps:
        p.error("--warmup must be smaller than --steps")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = ([SINGLE] if args.single else []) + list(args.modes)
    results = [run_one(label, args, train_args, out_dir) for label in labels]

    meta = {"timestamp": stamp, "argv": sys.argv[1:], "nproc": args.nproc,
            "backend": args.backend, "seq_len": args.seq_len,
            "per_gpu_batch": args.per_gpu_batch, "warmup": args.warmup}
    try:
        import torch
        meta["torch"] = torch.__version__
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name()
            meta["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass

    (out_dir / "results.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2))
    print(f"\nwrote {out_dir / 'results.json'}")
    print_table(results)

    failed = [r["label"] for r in results if "mean_ms" not in r]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Collect the writeup's plot data from durable experiment artifacts.

The raw ``out/`` tree is intentionally gitignored.  This script turns the raw
logs and benchmark JSON into small, reviewable CSV files under
``docs/writeup_data``.  Every derived value carries a status and source so a
measured convergence time cannot be confused with a step-time extrapolation or
a transport reconstruction.

Usage:

    uv run python scripts/collect_writeup_data.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

VAL_RE = re.compile(
    r"step\s+(?P<step>\d+)\s+\| val_loss (?P<loss>[\d.]+) "
    r"\| train_time (?P<time>[\d.]+)s"
)
DIAG_RE = re.compile(r"step\s+(?P<step>\d+)\s+\| diag pre-sync val (?P<values>.+)")
TRAIN_RE = re.compile(
    r"step\s+(?P<step>\d+)\s+\| loss (?P<loss>[\d.]+) \| lr (?P<lr>[\d.e+-]+) "
    r"\|\s*(?P<step_ms>[\d.]+) ms \| mfu (?P<mfu>[\d.]+)%"
)
CAPACITY_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T[\d:+-]+) (?P<state>HIT|none) (?P<sku>.+?)\s*$",
    re.MULTILINE,
)
RANK_VAL_RE = re.compile(r"r\d+=(?P<loss>[\d.]+)")
PRICE_RE = re.compile(r"\$([\d.]+)/h")

TOKENS_PER_STEP = 480 * 1024
CROSSING_STEP = 9999
# DiLoCo synchronizes every H inner steps, so one gradient-sized all-reduce is
# amortized over H optimizer steps instead of paid on every one.
DILOCO_H = 500
# Every arm here uses the same trapezoid tail, so warmdown starts at max_steps-1000.
WARMDOWN_STEPS = 1000
# decisions.md section 18: DiLoCo needed about 2.5x the tokens of DDP to reach a
# given loss at K=8; section 23 measured 1.25x at K=2.
DILOCO_TOKEN_RATIO_K8 = 2.49
DILOCO_TOKEN_RATIO_K2 = 1.25
ONE_GPU_STEP_MS = 2589.0583333333334
PARAMS = 162e6
GRAD_BYTES = 4
RANKS = 8
# Measured on real, rentable PCIe hardware: 2x A100 80GB PCIe over a host bridge
# (PHB, no NVLink), RunPod CA-MTL-3, 2026-08-22, `all_reduce_perf -b 8M -e 512M`.
# GPU-to-GPU P2P is *not available* on that box -- `nvidia-smi topo -p2p r` reports
# CNS (chipset not supported) and NCCL routes every channel via SHM/direct, i.e.
# through host memory. So this is what renting an A100 PCIe node actually
# delivers, not a direct PCIe P2P figure (decisions.md section 25).
PCIE_BUS_GBPS = 2.28964


def parse_validation_log(path: Path) -> dict[int, dict[str, float]]:
    points: dict[int, dict[str, float]] = {}
    for match in VAL_RE.finditer(path.read_text()):
        step = int(match.group("step"))
        points[step] = {
            "val_loss": float(match.group("loss")),
            "train_time_s": float(match.group("time")),
        }
    if not points:
        raise ValueError(f"no validation points found in {path}")
    return points


def parse_diag_log(path: Path) -> dict[int, list[float]]:
    points: dict[int, list[float]] = {}
    for match in DIAG_RE.finditer(path.read_text()):
        values = [float(m.group("loss")) for m in RANK_VAL_RE.finditer(match.group("values"))]
        if values:
            points[int(match.group("step"))] = values
    if not points:
        raise ValueError(f"no DiLoCo diagnostic points found in {path}")
    return points


def newest_bench(root: Path, relative: str) -> Path:
    """The most recent timestamped `results.json` under a bench directory.

    Bench runs land in `<out-dir>/<UTC timestamp>/`, so pinning a timestamp in
    this file would break the moment an arm is re-run.
    """
    matches = sorted((root / relative).glob("*/results.json"))
    if not matches:
        raise FileNotFoundError(f"no results.json under {relative}")
    return matches[-1]


def load_result(path: Path, label: str) -> dict:
    data = json.loads(path.read_text())
    for result in data["results"]:
        if result["label"] == label and result.get("returncode") == 0 and "mean_ms" in result:
            return result
    raise ValueError(f"no successful {label!r} result in {path}")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ring_factor(ranks: int) -> float:
    return 2.0 * (ranks - 1) / ranks


def effective_bus_gbps(comm_ms: float) -> float:
    traffic_bytes = PARAMS * GRAD_BYTES * ring_factor(RANKS)
    return traffic_bytes / (comm_ms / 1000) / 1e9


def collect_validation_curves(root: Path, output: Path) -> None:
    base_path = root / "out/runpod-8gpu/session_out/train.log"
    extension_path = root / "out/runpod-8gpu/session_out/train_ext.log"
    diloco_path = root / "out/prime-diloco/session_out/train.log"

    # The 10k schedule and the original 9k schedule are identical through step
    # 8000.  The extension resumed that checkpoint with max_steps=10000.
    ddp = {step: point for step, point in parse_validation_log(base_path).items() if step <= 8000}
    ddp.update(
        {
            step: point
            for step, point in parse_validation_log(extension_path).items()
            if step >= 8000
        }
    )
    diloco = parse_validation_log(diloco_path)

    rows = []
    for series, points, sources, note in [
        (
            "DDP (8xA100, NVLink)",
            ddp,
            f"{base_path.relative_to(root)} + {extension_path.relative_to(root)}",
            "Clean 10000-step trapezoid; the step-8000 checkpoint joins the two logs.",
        ),
        (
            "DiLoCo (8xA100, NVLink)",
            diloco,
            str(diloco_path.relative_to(root)),
            "K=8, H=500, outer_lr=0.7, outer_momentum=0.5.",
        ),
    ]:
        for step, point in sorted(points.items()):
            rows.append(
                {
                    "series": series,
                    "step": step,
                    "tokens_billions": step * TOKENS_PER_STEP / 1e9,
                    "val_loss": point["val_loss"],
                    "train_time_s": point["train_time_s"],
                    "status": "measured",
                    "source_file": sources,
                    "notes": note,
                }
            )
    write_csv(
        output / "validation_curves.csv",
        [
            "series",
            "step",
            "tokens_billions",
            "val_loss",
            "train_time_s",
            "status",
            "source_file",
            "notes",
        ],
        rows,
    )


def collect_diloco_sync(root: Path, output: Path) -> None:
    path = root / "out/prime-diloco/session_out/train.log"
    synced = parse_validation_log(path)
    rows = []
    for step, values in sorted(parse_diag_log(path).items()):
        if step not in synced:
            continue
        replica_mean = statistics.mean(values)
        synced_loss = synced[step]["val_loss"]
        rows.append(
            {
                "step": step,
                "tokens_billions": step * TOKENS_PER_STEP / 1e9,
                "replica_mean": replica_mean,
                "replica_min": min(values),
                "replica_max": max(values),
                "replica_spread": max(values) - min(values),
                "synced_val_loss": synced_loss,
                "merge_delta": synced_loss - replica_mean,
                "status": "measured",
                "source_file": str(path.relative_to(root)),
            }
        )
    write_csv(
        output / "diloco_sync.csv",
        [
            "step",
            "tokens_billions",
            "replica_mean",
            "replica_min",
            "replica_max",
            "replica_spread",
            "synced_val_loss",
            "merge_delta",
            "status",
            "source_file",
        ],
        rows,
    )


def collect_scaling(root: Path, output: Path) -> None:
    specs = [
        (
            "1xA100",
            "single",
            "single",
            root / "out/prime-pcie/session_out/bench-1gpu/20260821T170132Z/results.json",
        ),
        (
            "8xA100 NVLink",
            "NVLink NV12",
            "ddp_interleaved",
            root / "out/prime-pcie/session_out/bench-8gpu/20260821T170426Z/results.json",
        ),
        (
            "8xA100 forced TCP/loopback",
            "Forced TCP/loopback",
            "ddp_interleaved",
            root / "out/prime-pcie/session_out/bench-8gpu-socket/20260821T170636Z/results.json",
        ),
    ]
    rows = []
    single_ms = None
    for config, transport, mode, path in specs:
        result = load_result(path, mode)
        step_ms = float(result["mean_ms"])
        single_ms = step_ms if config == "1xA100" else single_ms
        rows.append(
            {
                "config": config,
                "gpus": 1 if config.startswith("1x") else 8,
                "transport": transport,
                "mode": mode,
                "step_ms": step_ms,
                "step_std_ms": result["std_ms"],
                "time_to_3_28_hours": CROSSING_STEP * step_ms / 1000 / 3600,
                "scaling_speedup": "",
                "scaling_efficiency": "",
                "time_status": "extrapolated_from_measured_step_time",
                "source_file": str(path.relative_to(root)),
                "notes": "Global batch 480; micro-batch 30 on every device.",
            }
        )
    assert single_ms is not None
    for row in rows:
        if row["gpus"] == 8:
            speedup = single_ms / float(row["step_ms"])
            row["scaling_speedup"] = speedup
            row["scaling_efficiency"] = speedup / 8

    # This is the only directly observed full crossing.  It used micro-batch 60,
    # so it must not be used to compute the matched-micro-batch scaling ratio.
    rows.append(
        {
            "config": "8xA100 NVLink (full run)",
            "gpus": 8,
            "transport": "NVLink NV12",
            "mode": "ddp_torch",
            "step_ms": "",
            "step_std_ms": "",
            "time_to_3_28_hours": 3147.1 / 3600,
            "scaling_speedup": "",
            "scaling_efficiency": "",
            "time_status": "measured_full_convergence",
            "source_file": "out/runpod-8gpu/session_out/train_ext.log",
            "notes": "First unsmoothed crossing at step 9999; micro-batch 60.",
        }
    )
    write_csv(
        output / "scaling.csv",
        [
            "config",
            "gpus",
            "transport",
            "mode",
            "step_ms",
            "step_std_ms",
            "time_to_3_28_hours",
            "scaling_speedup",
            "scaling_efficiency",
            "time_status",
            "source_file",
            "notes",
        ],
        rows,
    )


def collect_transport(root: Path, output: Path) -> tuple[list[dict], float]:
    bench_compute_ms = 49.5
    anchor_step_ms = 337.775
    anchor_bus_gbps = 151.0
    anchor_comm_ms = PARAMS * GRAD_BYTES * ring_factor(RANKS) / (anchor_bus_gbps * 1e9) * 1000
    anchor_compute_ms = anchor_step_ms - anchor_comm_ms
    specs = [
        (
            "Forced TCP/loopback",
            root / "out/prime-diloco/session_out/bench-control/20260821T150140Z/results.json",
            1270.3166666666666,
            "measured_step_time",
            (
                "Direct batch-480 step time from out/prime-pcie/session_out/"
                "bench-8gpu-socket/20260821T170636Z/results.json replaces the additive "
                "reconstruction; the control bench supplies effective bandwidth."
            ),
        ),
        (
            "netem nominal 40 Gbit/s",
            root / "out/prime-diloco/session_out/bench-40gbit-c/20260821T150623Z/results.json",
            None,
            "reconstructed_upper_bound",
            "Nominal rate delivered about 4.9 Gbit/s effective bus bandwidth.",
        ),
        (
            "netem nominal 10 Gbit/s",
            root / "out/prime-diloco/session_out/bench-10gbit-c/20260821T151005Z/results.json",
            None,
            "reconstructed_upper_bound",
            "Nominal rate delivered about 1.2 Gbit/s effective bus bandwidth.",
        ),
    ]
    rows = [
        {
            "transport": "NVLink NV12",
            "effective_bus_gbps": anchor_bus_gbps,
            "step_ms_batch480": anchor_step_ms,
            "time_to_3_28_hours": CROSSING_STEP * anchor_step_ms / 1000 / 3600,
            "comm_ms_batch480": anchor_comm_ms,
            "step_status": "measured_step_time",
            "time_status": "extrapolated_from_measured_step_time",
            "source_file": "out/prime-pcie/session_out/bench-8gpu/20260821T170426Z/results.json",
            "notes": "Matched-micro-batch scaling arm; full-run crossing was 0.874 h at micro-batch 60.",
        }
    ]
    # The PCIe point. Unlike the netem rows, whose bandwidth is a throttle applied
    # to a loopback socket, this bandwidth was measured on an actual PCIe fabric
    # -- so the reconstruction below rests on hardware rather than on simulation.
    pcie_comm_ms = PARAMS * GRAD_BYTES * ring_factor(RANKS) / (PCIE_BUS_GBPS * 1e9) * 1000
    pcie_step_ms = anchor_compute_ms + pcie_comm_ms
    rows.append(
        {
            "transport": "A100 PCIe (P2P off, SHM)",
            "effective_bus_gbps": PCIE_BUS_GBPS,
            "step_ms_batch480": pcie_step_ms,
            "comm_ms_batch480": pcie_comm_ms,
            "time_to_3_28_hours": CROSSING_STEP * pcie_step_ms / 1000 / 3600,
            "step_status": "reconstructed_upper_bound",
            "time_status": "extrapolated_from_reconstructed_step_time",
            "source_file": "out/runpod-pcie/session_out/nccl_tests.txt",
            "notes": (
                "Bandwidth measured on 2x A100 80GB PCIe, host bridge (PHB), no NVLink, "
                "RunPod CA-MTL-3 2026-08-22. P2P unavailable on that host (topo -p2p r = "
                "CNS), so NCCL staged through host memory via SHM -- this is what the "
                "rentable node delivers, not direct PCIe P2P. Step time reconstructed at "
                "8 ranks; an 8-GPU server's ring crosses more bridges, so it is optimistic."
            ),
        }
    )

    for label, path, measured_step_ms, status, note in specs:
        bench_step_ms = float(load_result(path, "ddp_torch")["mean_ms"])
        comm_ms = bench_step_ms - bench_compute_ms
        reconstructed_step_ms = anchor_compute_ms + comm_ms
        step_ms = measured_step_ms if measured_step_ms is not None else reconstructed_step_ms
        rows.append(
            {
                "transport": label,
                "effective_bus_gbps": effective_bus_gbps(comm_ms),
                "step_ms_batch480": step_ms,
                "comm_ms_batch480": comm_ms,
                "time_to_3_28_hours": CROSSING_STEP * step_ms / 1000 / 3600,
                "step_status": status,
                "time_status": "extrapolated_from_measured_step_time"
                if measured_step_ms is not None
                else "extrapolated_from_reconstructed_step_time",
                "source_file": str(path.relative_to(root)),
                "notes": note,
            }
        )
    rows.sort(key=lambda row: -float(row["effective_bus_gbps"]))
    write_csv(
        output / "transport.csv",
        [
            "transport",
            "effective_bus_gbps",
            "step_ms_batch480",
            "comm_ms_batch480",
            "time_to_3_28_hours",
            "step_status",
            "time_status",
            "source_file",
            "notes",
        ],
        rows,
    )
    return rows, anchor_compute_ms


def collect_pcie(root: Path, output: Path) -> None:
    """The 2026-08-22 A100-PCIe session: the first real PCIe box measured here.

    Every arm holds global batch 480 at micro-batch 30, which is what the NVLink
    and forced-TCP rows of `scaling.csv` were measured at, so the only differences
    from them are the rank count and the fabric. MFU is against this card's own
    measured peak, not the SXM4 one -- the PCIe part is a different chip budget.
    """
    model_flops_per_token = 854_770_176
    peak_tflops = 256.5                      # measured, mfu.py "A100 80GB PCIe"
    arms = [
        ("2 GPU compiled PyTorch DDP", 2, "ddp_torch", "true", "bench-b480"),
        ("2 GPU compiled interleaved", 2, "ddp_interleaved", "true", "bench-b480-il"),
        ("2 GPU uncompiled interleaved", 2, "ddp_interleaved", "false", "bench-b480-nc"),
        ("2 GPU uncompiled PyTorch DDP", 2, "ddp_torch", "false", "bench-b480-nc"),
        ("1 GPU baseline", 1, "ddp_torch", "true", "bench-1gpu-b480"),
    ]
    rows = []
    for label, gpus, mode, compiled, directory in arms:
        try:
            path = newest_bench(root, f"out/runpod-pcie/session_out/{directory}")
            result = load_result(path, mode)
        except (FileNotFoundError, ValueError):
            continue                         # an arm that did not run is not a row
        step_ms = float(result["mean_ms"])
        tokens_per_step = TOKENS_PER_STEP
        rows.append(
            {
                "arm": label,
                "gpus": gpus,
                "mode": mode,
                "compiled": compiled,
                "micro_batch": 30,
                "global_batch_seqs": 480,
                "step_ms": step_ms,
                "step_std_ms": result["std_ms"],
                "time_to_3_28_hours": CROSSING_STEP * step_ms / 1000 / 3600,
                "mfu_percent": 100 * model_flops_per_token * tokens_per_step
                / (step_ms / 1000) / (gpus * peak_tflops * 1e12),
                "source_file": str(path.relative_to(root)),
            }
        )
    if not rows:
        return
    write_csv(
        output / "pcie.csv",
        ["arm", "gpus", "mode", "compiled", "micro_batch", "global_batch_seqs",
         "step_ms", "step_std_ms", "time_to_3_28_hours", "mfu_percent", "source_file"],
        rows,
    )


def collect_ddp_modes(root: Path, output: Path) -> None:
    """The DDP implementation comparison, with the spread it actually has.

    The four-way matrix exists at one operating point only -- global batch 64
    over forced TCP/loopback -- because the NVLink session ran the anchor batch
    and had rental time for two modes, not four.  Both facts matter to the
    conclusion, so the summary carries every measured (mode, transport, batch)
    and the per-step times are projected alongside: at 15 timed steps with a
    heavy right tail, the *mean* ranking of the three bucketing modes is not
    reproducible, while their minima agree to about 2%.

    The anchor-batch rows cover all three measured fabrics, including the two
    2-rank A100-PCIe runs (decisions.md section 26).  Leaving those out is what
    made the figure read as "PyTorch DDP is slower": the sign of the gap flips
    on the slowest fabric, and one fabric's pair cannot show that.
    """
    specs = [
        (
            "Forced TCP/loopback",
            64,
            8,
            {
                mode: root / ("out/prime-diloco/session_out/bench-control/"
                              "20260821T150140Z/results.json")
                for mode in ("ddp_naive", "ddp_bucketed", "ddp_interleaved", "ddp_torch")
            },
        ),
        (
            "NVLink NV12",
            480,
            8,
            {
                mode: root / ("out/prime-pcie/session_out/bench-8gpu/"
                              "20260821T170426Z/results.json")
                for mode in ("ddp_interleaved", "ddp_torch")
            },
        ),
        (
            "A100 PCIe (P2P off, SHM)",
            480,
            2,
            {
                "ddp_interleaved": root / ("out/runpod-pcie/session_out/bench-b480-il/"
                                           "20260822T193853Z/results.json"),
                "ddp_torch": root / ("out/runpod-pcie/session_out/bench-b480/"
                                     "20260822T193724Z/results.json"),
            },
        ),
        (
            "Forced TCP/loopback",
            480,
            8,
            {
                mode: root / ("out/prime-pcie/session_out/bench-8gpu-socket/"
                              "20260821T170636Z/results.json")
                for mode in ("ddp_interleaved", "ddp_torch")
            },
        ),
    ]
    labels = {
        "ddp_naive": "Naive",
        "ddp_bucketed": "Bucketed",
        "ddp_interleaved": "Interleaved",
        "ddp_torch": "PyTorch DDP",
    }
    summary, steps = [], []
    for transport, batch, gpus, paths in specs:
        for mode, path in paths.items():
            result = load_result(path, mode)
            summary.append(
                {
                    "mode": mode,
                    "label": labels[mode],
                    "transport": transport,
                    "global_batch_seqs": batch,
                    "timed_steps": result["n"],
                    "mean_ms": result["mean_ms"],
                    "median_ms": result["median_ms"],
                    "std_ms": result["std_ms"],
                    "min_ms": result["min_ms"],
                    "gpus": gpus,
                    "compiled": "true",
                    "status": "measured",
                    "source_file": str(path.relative_to(root)),
                }
            )
            # Every step of every row, not just the batch-64 matrix: the anchor
            # claims are about minima and medians too, and a mean with a heavy
            # tail is exactly what the reader needs to be able to go behind.
            for index, value in enumerate(result["step_times_ms"]):
                steps.append(
                    {
                        "mode": mode,
                        "label": labels[mode],
                        "transport": transport,
                        "global_batch_seqs": batch,
                        "gpus": gpus,
                        "timed_step_index": index,
                        "step_ms": value,
                        "source_file": str(path.relative_to(root)),
                    }
                )
    write_csv(
        output / "ddp_modes.csv",
        [
            "mode",
            "label",
            "transport",
            "global_batch_seqs",
            "timed_steps",
            "mean_ms",
            "median_ms",
            "std_ms",
            "min_ms",
            "gpus",
            "compiled",
            "status",
            "source_file",
        ],
        summary,
    )
    write_csv(
        output / "ddp_mode_steps.csv",
        [
            "mode",
            "label",
            "transport",
            "global_batch_seqs",
            "gpus",
            "timed_step_index",
            "step_ms",
            "source_file",
        ],
        steps,
    )


def parse_training_log(path: Path) -> dict[int, dict[str, float]]:
    """Per-log-step training loss, step time and MFU from a run's stdout log."""
    points: dict[int, dict[str, float]] = {}
    for match in TRAIN_RE.finditer(path.read_text()):
        points[int(match.group("step"))] = {
            "train_loss": float(match.group("loss")),
            "lr": float(match.group("lr")),
            "step_ms": float(match.group("step_ms")),
            "mfu_percent": float(match.group("mfu")),
        }
    if not points:
        raise ValueError(f"no training points found in {path}")
    return points


def collect_training_curves(root: Path, output: Path) -> None:
    """The training-loss half of the loss curve, from the same logs as the val half.

    The two 8xA100 runs print `step N | loss X | lr ... | T ms | mfu Y%` every
    `--log-every` steps.  Only the validation lines were projected before, which
    is why the writeup's loss-curve figure had no training series.
    """
    base_path = root / "out/runpod-8gpu/session_out/train.log"
    extension_path = root / "out/runpod-8gpu/session_out/train_ext.log"
    diloco_path = root / "out/prime-diloco/session_out/train.log"

    # Same join as the validation curve: the 9k and 10k schedules are identical
    # through step 8000, where the extension resumed the checkpoint.
    ddp = {step: point for step, point in parse_training_log(base_path).items() if step <= 8000}
    ddp.update(
        {step: point for step, point in parse_training_log(extension_path).items() if step >= 8000}
    )
    diloco = parse_training_log(diloco_path)

    rows = []
    for series, points, sources, note in [
        (
            "DDP (8xA100, NVLink)",
            ddp,
            f"{base_path.relative_to(root)} + {extension_path.relative_to(root)}",
            "Clean 10000-step trapezoid; the step-8000 checkpoint joins the two logs.",
        ),
        (
            "DiLoCo (8xA100, NVLink)",
            diloco,
            str(diloco_path.relative_to(root)),
            "K=8, H=500, outer_lr=0.7, outer_momentum=0.5; rank-0 local loss.",
        ),
    ]:
        for step, point in sorted(points.items()):
            rows.append(
                {
                    "series": series,
                    "step": step,
                    "tokens_billions": step * TOKENS_PER_STEP / 1e9,
                    "train_loss": point["train_loss"],
                    "lr": point["lr"],
                    "step_ms": point["step_ms"],
                    "mfu_percent": point["mfu_percent"],
                    "status": "measured",
                    "source_file": sources,
                    "notes": note,
                }
            )
    write_csv(
        output / "training_curves.csv",
        [
            "series",
            "step",
            "tokens_billions",
            "train_loss",
            "lr",
            "step_ms",
            "mfu_percent",
            "status",
            "source_file",
            "notes",
        ],
        rows,
    )


def parse_capacity_log(
    path: Path, source: Path, labels: dict[str, tuple[str, str]]
) -> list[dict]:
    """One row per poll. A HIT line is followed by the indented `avail` output,
    which carries the quoted hourly rates; misses carry none, so prices come
    from hits only."""
    polls: list[dict] = []
    current: dict | None = None
    for line in path.read_text().splitlines():
        match = CAPACITY_RE.match(line)
        if match:
            current = None
            sku = match.group("sku").strip()
            if sku not in labels:
                continue
            venue, display = labels[sku]
            current = {
                "timestamp": match.group("ts"),
                "venue": venue,
                "sku": display,
                "available": int(match.group("state") == "HIT"),
                "quoted_usd_per_hour_min": "",
                "quoted_usd_per_hour_max": "",
                "source_file": str(source),
            }
            polls.append(current)
        elif current is not None and line.startswith("    "):
            prices = [float(value) for value in PRICE_RE.findall(line)]
            if prices:
                low = current["quoted_usd_per_hour_min"]
                high = current["quoted_usd_per_hour_max"]
                current["quoted_usd_per_hour_min"] = min(prices + ([low] if low != "" else []))
                current["quoted_usd_per_hour_max"] = max(prices + ([high] if high != "" else []))
    return polls


def collect_capacity(root: Path, output: Path) -> None:
    """Turn the capacity watcher's log into a measured availability record.

    `scripts/watch_capacity.sh` polled both venues for 8-GPU stock and appended
    one HIT/none line per poll.  That is a direct measurement of the writeup's
    opening claim -- that the cheap 8xA100 SKU is usually not rentable -- and is
    better evidence than a screenshot of a console.
    """
    paths = [root / "out/capacity.log", root / "out/capacity-pcie.log"]
    paths = [candidate for candidate in paths if candidate.exists()]
    if not paths:
        print("skipping capacity: no out/capacity*.log found")
        return
    labels = {
        "NVIDIA A100-SXM4-80GB": ("RunPod", "8x A100-SXM4-80GB"),
        "NVIDIA H100 80GB HBM3": ("RunPod", "8x H100-80GB-HBM3"),
        "prime A100_80GB SXM4": ("Prime Intellect", "8x A100-80GB SXM4"),
        "NVIDIA A100 80GB PCIe": ("RunPod", "8x A100-80GB PCIe"),
        "prime A100_80GB PCIe": ("Prime Intellect", "8x A100-80GB PCIe"),
    }
    timeline = []
    for path in paths:
        timeline.extend(parse_capacity_log(path, path.relative_to(root), labels))
    write_csv(
        output / "capacity_timeline.csv",
        [
            "timestamp",
            "venue",
            "sku",
            "available",
            "quoted_usd_per_hour_min",
            "quoted_usd_per_hour_max",
            "source_file",
        ],
        timeline,
    )

    summary = []
    for sku in dict.fromkeys(row["sku"] for row in timeline):
        polls = [row for row in timeline if row["sku"] == sku]
        hits = sum(row["available"] for row in polls)
        spread = [
            float(value)
            for row in polls
            for key in ("quoted_usd_per_hour_min", "quoted_usd_per_hour_max")
            if (value := row[key]) != ""
        ]
        summary.append(
            {
                "venue": polls[0]["venue"],
                "sku": sku,
                "polls": len(polls),
                "polls_available": hits,
                "available_fraction": hits / len(polls),
                "quoted_usd_per_hour_min": min(spread) if spread else "",
                "quoted_usd_per_hour_max": max(spread) if spread else "",
                "first_timestamp": min(row["timestamp"] for row in polls),
                "last_timestamp": max(row["timestamp"] for row in polls),
                "status": "measured",
                "source_file": str(path.relative_to(root)),
                "notes": (
                    "A poll counts as available when any host reports stock at the "
                    "requested GPU count, including 'Low' capacity. Prices are the range "
                    "of hourly rates quoted on hits; the exchange lists several upstream "
                    "providers per hit, so its minimum is not the rate one can reliably get."
                ),
            }
        )
    write_csv(
        output / "capacity_availability.csv",
        [
            "venue",
            "sku",
            "polls",
            "polls_available",
            "available_fraction",
            "quoted_usd_per_hour_min",
            "quoted_usd_per_hour_max",
            "first_timestamp",
            "last_timestamp",
            "status",
            "source_file",
            "notes",
        ],
        summary,
    )


def read_trackio_val(db_path: Path, run_name: str) -> dict[int, float]:
    """Validation loss by step for one Trackio run, or {} when unavailable."""
    if not db_path.exists():
        return {}
    import sqlite3

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT step, metrics FROM metrics WHERE run_name = ? ORDER BY step",
            (run_name,),
        ).fetchall()
    finally:
        connection.close()
    points: dict[int, float] = {}
    for step, blob in rows:
        payload = json.loads(blob)
        if "val/loss" in payload:
            points[int(step)] = float(payload["val/loss"])
    return points


def tokens_at_equal_loss(reference: list[tuple[float, float]], loss: float) -> float | None:
    """Token count at which the reference curve first reached `loss`.

    The vertical loss gap is contaminated by the reference's own slope: the same
    token lag reads as a far bigger gap during the warmdown, where the reference
    falls about 10x faster than on the plateau.  Inverting the reference curve
    instead answers the question the project actually committed to -- how much
    longer DiLoCo takes to reach a given loss.

    Interpolates in log-token space, since loss against tokens is roughly a power
    law.  Returns None when `loss` is worse than anything the reference logged, so
    early rounds report no ratio rather than an extrapolated one.
    """
    previous: tuple[float, float] | None = None
    for tokens, reference_loss in reference:
        if tokens <= 0:  # log-space interpolation cannot start from the init point
            continue
        if previous is not None and reference_loss <= loss <= previous[1]:
            (tokens_before, loss_before), (tokens_after, loss_after) = previous, (
                tokens,
                reference_loss,
            )
            if loss_before == loss_after:
                return tokens_before
            fraction = (loss_before - loss) / (loss_before - loss_after)
            return math.exp(
                math.log(tokens_before)
                + fraction * (math.log(tokens_after) - math.log(tokens_before))
            )
        previous = (tokens, reference_loss)
    return None


def collect_diloco_k_penalty(root: Path, output: Path, trackio_db: Path) -> None:
    """How the DiLoCo merge penalty evolves through training, at two replica counts.

    The endpoint alone is misleading: the penalty is largely transient, decaying
    through training at both K.  So this records the whole trajectory.

    Both arms run the same 10000-step trapezoid, the same 500-step val grid and
    the same 491,520 tokens/step, so equal token counts sit at the same point in
    the LR schedule.  Each arm carries its own reference -- the 1-GPU run for
    K=2, the 8-GPU DDP run for K=8 -- but those are the same experiment, since
    data order is world-size-independent and DDP at global batch 480 is the same
    update as one GPU at global batch 480.  They agree to within 0.016 at every
    step; pairing each arm with the copy from its own box just keeps the residual
    bf16 reduction-order term out of the gap.  K=2 comes from aurora's Trackio
    store (decisions.md section 23), K=8 from the rented session's logs.
    """
    arms = []

    # The K=2 arm was rerun on a 10000-step trapezoid so it matches K=8's schedule
    # (scripts/run-k2-10k-arms.sh).  Its DiLoCo curve is spliced: the rerun resumed
    # the original 6000-step arm from its step-4000 keep, which sits inside both
    # schedules' LR plateau, so everything below that step is the same trajectory
    # and is taken from the original run.  Both runs log step 4000 and agree there
    # to 1e-5 -- that agreement is the check that the splice is sound.
    resumed = read_trackio_val(trackio_db, "diloco-k2-10k")
    original = read_trackio_val(trackio_db, "diloco-b480-mom05")
    reference = read_trackio_val(trackio_db, "ref-1gpu-10k")
    if reference and resumed and original:
        splice_step = min(resumed)
        arms.append(
            {
                "replicas_k": 2,
                "schedule_steps": 10000,
                "hardware": "1x RTX 3090 (aurora, 2 ranks sharing the GPU)",
                "reference": reference,
                "diloco": {
                    **{step: loss for step, loss in original.items() if step < splice_step},
                    **resumed,
                },
                "source_file": (
                    f"{trackio_db.name}: ref-1gpu-10k, diloco-k2-10k "
                    f"(steps below {splice_step} from diloco-b480-mom05)"
                ),
                "notes": (
                    "decisions.md section 23. Same 10000-step trapezoid and 500-step val "
                    f"grid as the K=8 arm; steps below {splice_step} come from the "
                    "6000-step arm this one resumed, on the schedule's shared plateau."
                ),
            }
        )
    else:
        print(f"skipping DiLoCo K=2 penalty: no Trackio runs in {trackio_db}")

    arms.append(
        {
            "replicas_k": 8,
            "schedule_steps": 10000,
            "hardware": "8x A100-SXM4-80GB (Prime Intellect)",
            "reference": {
                step: point["val_loss"]
                for step, point in {
                    **parse_validation_log(root / "out/runpod-8gpu/session_out/train.log"),
                    **parse_validation_log(root / "out/runpod-8gpu/session_out/train_ext.log"),
                }.items()
            },
            "diloco": {
                step: point["val_loss"]
                for step, point in parse_validation_log(
                    root / "out/prime-diloco/session_out/train.log"
                ).items()
            },
            "source_file": (
                "out/runpod-8gpu/session_out/train.log + train_ext.log + "
                "out/prime-diloco/session_out/train.log"
            ),
            "notes": "decisions.md section 21. Both curves share a 500-step validation grid.",
        }
    )

    rows = []
    for arm in arms:
        schedule = arm["schedule_steps"]
        # Sorted (tokens, loss) for the reference, so its curve can be inverted.
        reference_curve = [
            (reference_step * TOKENS_PER_STEP / 1e9, loss)
            for reference_step, loss in sorted(arm["reference"].items())
        ]
        for step, diloco_loss in sorted(arm["diloco"].items()):
            # The final step is logged as schedule-1; pair it with the reference's
            # own last point rather than dropping the most interesting round.
            pair_step = step if step in arm["reference"] else schedule
            if step != schedule - 1 and step not in arm["reference"]:
                continue
            if pair_step not in arm["reference"]:
                continue
            reference_loss = arm["reference"][pair_step]
            tokens = step * TOKENS_PER_STEP / 1e9
            equal_loss_tokens = tokens_at_equal_loss(reference_curve, diloco_loss)
            # The inversion is only apples-to-apples while both sides sit on the
            # same side of the warmdown.  Once DiLoCo has warmed down but the
            # reference point it matches is still on the plateau, DiLoCo is
            # carrying a warmdown boost the reference has not had, and the ratio
            # flatters it -- the schedule-position confound this whole comparison
            # exists to avoid, reappearing inside the inversion.
            warmdown_start = schedule - WARMDOWN_STEPS
            matched_step = (
                equal_loss_tokens * 1e9 / TOKENS_PER_STEP if equal_loss_tokens else None
            )
            comparison = ""
            if matched_step is not None:
                comparison = (
                    "clean"
                    if (step > warmdown_start) == (matched_step > warmdown_start)
                    else "schedule-mismatched: DiLoCo has warmed down, its matched "
                    "reference point has not"
                )
            rows.append(
                {
                    "replicas_k": arm["replicas_k"],
                    "schedule_steps": schedule,
                    "hardware": arm["hardware"],
                    "step": step,
                    "tokens_billions": step * TOKENS_PER_STEP / 1e9,
                    "warmdown_start_step": schedule - WARMDOWN_STEPS,
                    "reference_val_loss": reference_loss,
                    "diloco_val_loss": diloco_loss,
                    "penalty": diloco_loss - reference_loss,
                    "reference_tokens_at_equal_loss": equal_loss_tokens,
                    "token_ratio": (
                        tokens / equal_loss_tokens if equal_loss_tokens and tokens else None
                    ),
                    "ratio_comparison": comparison,
                    "token_ratio_status": (
                        "interpolated"
                        if equal_loss_tokens and tokens
                        # Early rounds are worse than any loss the reference logged, so
                        # there is no token count to compare against.
                        else "undefined: worse than the reference's first logged point"
                    ),
                    "status": "measured",
                    "source_file": arm["source_file"],
                    "notes": arm["notes"],
                }
            )
    write_csv(
        output / "diloco_k_penalty.csv",
        [
            "replicas_k",
            "schedule_steps",
            "hardware",
            "step",
            "tokens_billions",
            "warmdown_start_step",
            "reference_val_loss",
            "diloco_val_loss",
            "penalty",
            "reference_tokens_at_equal_loss",
            "token_ratio",
            "ratio_comparison",
            "token_ratio_status",
            "status",
            "source_file",
            "notes",
        ],
        rows,
    )


def collect_diloco_transport(
    root: Path, output: Path, transport_rows: list[dict]
) -> list[dict]:
    """DiLoCo's amortized step time at each measured transport.

    DiLoCo's outer synchronization all-reduces one gradient-sized tensor every H
    inner steps -- the same tensor DDP all-reduces on *every* step.  So the
    communication term already measured per transport divides by H, and the rest
    of the step is the measured DiLoCo compute.  This uses the same additive
    convention as the netem DDP reconstruction (README's
    `reconstructed_upper_bound`), and inherits the same +23% calibration.
    """
    diloco_step_ms = 3250.3 / 10_000 * 1000
    rows = []
    for row in transport_rows:
        comm_ms = float(row["comm_ms_batch480"])
        # The measured DiLoCo step already contains its own NVLink outer sync.
        base_ms = diloco_step_ms - float(transport_rows[0]["comm_ms_batch480"]) / DILOCO_H
        step_ms = base_ms + comm_ms / DILOCO_H
        ddp_step_ms = float(row["step_ms_batch480"])
        rows.append(
            {
                "transport": row["transport"],
                "effective_bus_gbps": row["effective_bus_gbps"],
                "ddp_step_ms_batch480": ddp_step_ms,
                "diloco_step_ms_batch480": step_ms,
                "diloco_speedup_per_step": ddp_step_ms / step_ms,
                "ddp_comm_ms_per_step": comm_ms,
                "diloco_comm_ms_per_step": comm_ms / DILOCO_H,
                "ddp_time_to_3_28_hours": CROSSING_STEP * ddp_step_ms / 1000 / 3600,
                "diloco_time_to_3_28_hours": (
                    DILOCO_TOKEN_RATIO_K8 * CROSSING_STEP * step_ms / 1000 / 3600
                ),
                "status": "measured"
                if row["transport"] == "NVLink NV12"
                else "reconstructed_upper_bound",
                "source_file": row["source_file"],
                "notes": (
                    (
                        "Measured converged DiLoCo run."
                        if row["transport"] == "NVLink NV12"
                        else "Measured DiLoCo compute plus this transport's all-reduce "
                        f"/ H={DILOCO_H}."
                    )
                    + f" diloco_time_to_3_28_hours charges section 18's "
                    f"{DILOCO_TOKEN_RATIO_K8}x token ratio, which is an estimate: DiLoCo "
                    "never reached 3.28 inside this corpus."
                ),
            }
        )
    write_csv(
        output / "diloco_transport.csv",
        [
            "transport",
            "effective_bus_gbps",
            "ddp_step_ms_batch480",
            "diloco_step_ms_batch480",
            "diloco_speedup_per_step",
            "ddp_comm_ms_per_step",
            "diloco_comm_ms_per_step",
            "ddp_time_to_3_28_hours",
            "diloco_time_to_3_28_hours",
            "status",
            "source_file",
            "notes",
        ],
        rows,
    )
    return rows


def collect_crossovers(output: Path, anchor_compute_ms: float) -> None:
    """The two bandwidths where the ranking of the options flips.

    Both fall out of the reconstruction already used for the netem points, so
    they cost no new measurement: an 8-GPU DDP step is `anchor_compute + comm`,
    and comm is set by the effective all-reduce bandwidth.
    """
    traffic_bytes = PARAMS * GRAD_BYTES * ring_factor(RANKS)

    def bandwidth_for(comm_ms: float) -> float:
        return traffic_bytes / (comm_ms / 1000) / 1e9

    diloco_step_ms = 3250.3 / 10_000 * 1000
    # DDP pays comm every step; DiLoCo pays comm/H but needs `ratio` times the
    # steps.  They tie when anchor_compute + comm == ratio * (diloco + comm/H).
    ratio = DILOCO_TOKEN_RATIO_K8
    diloco_tie_comm_ms = (ratio * diloco_step_ms - anchor_compute_ms) / (1 - ratio / DILOCO_H)

    rows = [
        {
            "crossover": "8-GPU DDP stops beating 1 GPU",
            "comm_ms_per_step": ONE_GPU_STEP_MS - anchor_compute_ms,
            "effective_bus_gbps": bandwidth_for(ONE_GPU_STEP_MS - anchor_compute_ms),
            "status": "derived_from_measured_step_times",
            "notes": (
                f"Eight A100s on DDP match one A100's {ONE_GPU_STEP_MS:.0f} ms step once "
                f"the all-reduce costs {ONE_GPU_STEP_MS - anchor_compute_ms:.0f} ms. The "
                "measured unthrottled socket sat only 1.8x above this."
            ),
        },
        {
            "crossover": "DiLoCo starts beating DDP end to end",
            "comm_ms_per_step": diloco_tie_comm_ms,
            "effective_bus_gbps": bandwidth_for(diloco_tie_comm_ms),
            "status": "derived_from_measured_step_times",
            "notes": (
                f"Charging DiLoCo section 18's {ratio}x token ratio and H={DILOCO_H}. "
                "The ratio is an estimate, not a measured crossing, so this "
                "bandwidth moves with it."
            ),
        },
    ]
    write_csv(
        output / "transport_crossovers.csv",
        ["crossover", "comm_ms_per_step", "effective_bus_gbps", "status", "notes"],
        rows,
    )


# cost_inputs.csv is hand-maintained and carries display labels, not the
# transport keys transport.csv uses.  Join on config_id so neither file has to
# spell the other's strings.
COST_TRANSPORT_KEY = {
    "a100x8_nvlink": "NVLink NV12",
    "a100x8_pcie": "A100 PCIe (P2P off, SHM)",
    "a100x8_forced_tcp_torch": "Forced TCP/loopback",
    "a100x8_netem40": "netem nominal 40 Gbit/s",
    "a100x8_netem10": "netem nominal 10 Gbit/s",
}
# What a priced row's runtime actually rests on, kept short enough to sit in a
# figure legend.  A cost is never better evidence than the runtime under it.
COST_EVIDENCE = {
    "extrapolated_from_measured_step_time": "Measured step time",
    "extrapolated_from_trackio_mean_step_time": "Measured step time",
    "extrapolated_from_reconstructed_step_time": "Reconstructed step time",
    "measured_full_convergence": "Measured convergence",
    "measured_not_converged": "Measured, not to target",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float:
    """A blank cost input means "does not apply", not zero-as-a-measurement."""
    value = row.get(key) or ""
    return float(value) if value else 0.0


def collect_costs(root: Path, output: Path, transport_rows: list[dict]) -> list[dict]:
    """Price one converged run per configuration from `cost_inputs.csv`.

    Rented boxes cost runtime times an hourly rate.  The 3090 desktop is owned,
    so it has no rate and is costed from wall power and electricity instead --
    the two bases are carried in a column rather than silently summed into one
    "price", because they are not the same kind of number.

    A configuration with no runtime anywhere produces no row: an unmeasured
    convergence must not appear as a $0 bar.  The one exception is a fabric the
    transport curve already carries a reconstructed runtime for, which is how
    the 8-rank PCIe projection of decisions.md section 25 gets priced.
    """
    inputs = read_csv_rows(root / "docs/writeup_data/cost_inputs.csv")
    by_transport = {row["transport"]: row for row in transport_rows}
    rows = []
    for entry in inputs:
        runtime_status = entry["runtime_status"]
        runtime_source = "cost_inputs.csv"
        if entry["runtime_hours"]:
            hours = float(entry["runtime_hours"])
        else:
            fallback = by_transport.get(COST_TRANSPORT_KEY.get(entry["config_id"], ""))
            if fallback is None:
                continue
            hours = float(fallback["time_to_3_28_hours"])
            runtime_status = fallback["time_status"]
            runtime_source = "transport.csv"
        rate = _number(entry, "hourly_rental_rate")
        rental = rate * hours
        energy = (
            _number(entry, "average_psu_power_watts")
            / 1000
            * _number(entry, "electricity_price_per_kwh")
            * hours
        )
        capital = _number(entry, "capital_cost_per_hour") * hours
        total = rental + energy + capital + _number(entry, "fixed_cost")
        if entry["total_cost_override"]:
            total = float(entry["total_cost_override"])
        rows.append(
            {
                "config_id": entry["config_id"],
                "display_name": entry["display_name"],
                "plot_stage": entry["plot_stage"],
                "hardware": entry["hardware"],
                "gpus": entry["gpus"],
                "method": entry["method"],
                "transport": entry["transport"],
                "transport_key": COST_TRANSPORT_KEY.get(entry["config_id"], ""),
                "comparison_basis": entry["comparison_basis"],
                "runtime_hours": hours,
                "runtime_status": runtime_status,
                "runtime_source": runtime_source,
                "evidence": COST_EVIDENCE.get(runtime_status, runtime_status),
                "currency": entry["currency"],
                "hourly_rental_rate": rate or "",
                "rental_cost": rental,
                "energy_cost": energy,
                "capital_cost": capital,
                "total_cost": total,
                # Which of the two the bar is: a rented hour, or an owned card's
                # electricity.  Mixing them on one axis needs saying out loud.
                "cost_basis": "rental" if rate else "wall power (owned hardware)",
                "effective_usd_per_hour": total / hours if hours else "",
                "notes": entry["notes"],
            }
        )
    write_csv(
        output / "costs.csv",
        [
            "config_id",
            "display_name",
            "plot_stage",
            "hardware",
            "gpus",
            "method",
            "transport",
            "transport_key",
            "comparison_basis",
            "runtime_hours",
            "runtime_status",
            "runtime_source",
            "evidence",
            "currency",
            "hourly_rental_rate",
            "rental_cost",
            "energy_cost",
            "capital_cost",
            "total_cost",
            "cost_basis",
            "effective_usd_per_hour",
            "notes",
        ],
        rows,
    )
    return rows


def collect_transport_costs(
    output: Path, transport_rows: list[dict], diloco_rows: list[dict], cost_rows: list[dict]
) -> None:
    """Price a converged run on each fabric, for DDP and for DiLoCo.

    The writeup's question here is not "which fabric is fastest" but "which is
    cheapest", and those differ: cost is the hourly rate times the hours, and the
    rate moves with the SKU, not with the fabric.  So the rate is carried
    alongside rather than folded away -- the PCIe box is half the SXM4 anchor's
    price and still the more expensive run.

    DiLoCo's hours already charge section 18's 2.49x token ratio, which is an
    estimate rather than a measured crossing; every DiLoCo cost inherits that.
    """
    rates = {row["transport_key"]: row for row in cost_rows if row["transport_key"]}
    diloco = {row["transport"]: row for row in diloco_rows}
    rows = []
    for row in transport_rows:
        priced = rates.get(row["transport"])
        if priced is None:
            continue
        rate = float(priced["hourly_rental_rate"])
        ddp_hours = float(row["time_to_3_28_hours"])
        diloco_hours = float(diloco[row["transport"]]["diloco_time_to_3_28_hours"])
        rows.append(
            {
                "transport": row["transport"],
                "effective_bus_gbps": row["effective_bus_gbps"],
                "usd_per_hour": rate,
                "rate_source_config": priced["config_id"],
                "ddp_hours": ddp_hours,
                "ddp_cost": ddp_hours * rate,
                "ddp_evidence": COST_EVIDENCE.get(row["time_status"], row["time_status"]),
                "diloco_hours": diloco_hours,
                "diloco_cost": diloco_hours * rate,
                "diloco_evidence": "Reconstructed step time",
                "status": row["step_status"],
                "notes": (
                    f"Rate from cost_inputs.csv row {priced['config_id']}. DiLoCo hours "
                    f"charge section 18's {DILOCO_TOKEN_RATIO_K8}x token ratio, an estimate: "
                    "DiLoCo never reached 3.28 inside this corpus."
                ),
            }
        )
    write_csv(
        output / "transport_costs.csv",
        [
            "transport",
            "effective_bus_gbps",
            "usd_per_hour",
            "rate_source_config",
            "ddp_hours",
            "ddp_cost",
            "ddp_evidence",
            "diloco_hours",
            "diloco_cost",
            "diloco_evidence",
            "status",
            "notes",
        ],
        rows,
    )


def collect_ratio_sweep(root: Path, output: Path) -> None:
    """The mode matrix as a function of the compute-to-communication ratio.

    Answers what the batch-64 four-way panel could not (decisions.md section 26):
    that panel sits at one ratio, where the collective is 96% of the step and no
    implementation can differ by more than the compute.  Here micro-batch is the
    only variable -- accumulation is 1 and global batch is micro x ranks -- so
    every arm moves the ratio and nothing else.

    Absolute step times are not comparable across arms, by construction: the
    batch differs.  The comparable quantity is each mode's step time *relative to
    the best mode at that same ratio*, which is what the figure draws.

    Skipped silently when the session has not been run; every other CSV is
    independent of it.
    """
    # More than one session can contribute: a later box may re-measure points an
    # earlier one covered.  Later wins per (fabric, micro-batch, mode), so a
    # re-run supersedes rather than duplicates, and the superseded arms stay on
    # disk as an independent check.
    sessions = sorted(path for path in root.glob("out/ratio-sweep*/session_out")
                      if path.is_dir())
    if not sessions:
        return
    labels = {
        "ddp_naive": "Naive",
        "ddp_bucketed": "Bucketed",
        "ddp_interleaved": "Interleaved",
        "ddp_torch": "PyTorch DDP",
    }
    fabrics = {"native": "Native fabric", "tcp": "Forced TCP/loopback"}
    by_arm: dict[tuple[str, int, str], dict] = {}
    for session in sessions:
      for directory in sorted(session.glob("sweep-*-m*")):
        # The same pattern also matches each arm's tee'd .log next to its directory.
        if not directory.is_dir():
            continue
        tag, _, micro = directory.name.removeprefix("sweep-").rpartition("-m")
        path = newest_bench(session, directory.name)
        data = json.loads(path.read_text())
        ranks = data["meta"]["nproc"]
        for result in data["results"]:
            # A failed arm has no stats; a single-process baseline is not a mode.
            if "mean_ms" not in result or result["label"] not in labels:
                continue
            by_arm[(fabrics.get(tag, tag), int(micro), result["label"])] = (
                {
                    "fabric": fabrics.get(tag, tag),
                    "micro_batch": int(micro),
                    "global_batch_seqs": int(micro) * ranks,
                    "tokens_per_rank_per_backward": int(micro) * data["meta"]["seq_len"],
                    "gpus": ranks,
                    "mode": result["label"],
                    "label": labels[result["label"]],
                    "timed_steps": result["n"],
                    "mean_ms": result["mean_ms"],
                    "median_ms": result["median_ms"],
                    "std_ms": result["std_ms"],
                    "min_ms": result["min_ms"],
                    "gpu": data["meta"].get("gpu", ""),
                    "status": "measured",
                    "source_file": str(path.relative_to(root)),
                }
            )
    if not by_arm:
        return
    rows = [by_arm[key] for key in sorted(by_arm)]
    write_csv(
        output / "ratio_sweep.csv",
        ["fabric", "micro_batch", "global_batch_seqs", "tokens_per_rank_per_backward",
         "gpus", "mode", "label", "timed_steps", "mean_ms", "median_ms", "std_ms",
         "min_ms", "gpu", "status", "source_file"],
        rows,
    )


def collect_gaps(output: Path) -> None:
    rows = [
        {
            "requested_comparison": "Full converged training and validation loss curve",
            "availability": "complete",
            "reason": (
                "Both halves are projected: validation_curves.csv and training_curves.csv "
                "cover the clean 10000-step DDP schedule and the DiLoCo run."
            ),
        },
        {
            "requested_comparison": "Desktop RTX 3090 time and price to 3.28",
            "availability": "complete",
            "reason": (
                "Measured end to end on 2026-08-24: a clean 10000-step trapezoid on aurora "
                "crossed 3.28 at step 9999 in 76403.66 training seconds (21.223 h, 66.6% MFU, "
                "out/ref-1gpu-10k.log). It ran as the K=2 DiLoCo arm's single-GPU reference, "
                "so the bar cost nothing extra, and it confirms the previous 21.299 h "
                "extrapolation to 0.36%. Wall power and electricity price are filled in."
            ),
        },
        {
            "requested_comparison": "8xA100 PCIe",
            "availability": "partial",
            "reason": (
                "Measured at 2 GPUs, not 8: a topology-verified 2xA100 80GB PCIe box gave "
                "2.29 GB/s effective all-reduce bandwidth and a 1745.4 ms batch-480 step, "
                "projecting to 826 ms at 8 ranks. That node has no GPU-to-GPU P2P "
                "(topo -p2p r = CNS; NCCL routes via host memory), so none of it is PCIe "
                "P2P. The 8-GPU bar is out of stock rather than mispriced: RunPod sells "
                "the SKU at half the SXM4 anchor's price and had no capacity on "
                "2026-08-22; Prime Intellect has none at all."
            ),
        },
        {
            "requested_comparison": "DiLoCo token ratio as a single constant",
            "availability": "partial",
            "reason": (
                f"diloco_transport.csv and transport_crossovers.csv charge DiLoCo a flat "
                f"{DILOCO_TOKEN_RATIO_K8}x token ratio, but diloco_k_penalty.csv measures that "
                "quantity varying from 2.7x to 3.4x across the run's clean region, rising with "
                f"training. {DILOCO_TOKEN_RATIO_K8}x sits below that entire range, so the "
                "end-to-end hours and the 2.36 GB/s crossover currently flatter DiLoCo: at 3.0x "
                "the crossover falls to about 1.73 GB/s and the PCIe verdict flips from DiLoCo "
                "winning by 2% to losing by 18%. The ratio at the 3.28 target itself is not "
                "measurable here -- no DiLoCo run reached 3.28 -- so this is recorded rather "
                "than corrected."
            ),
        },
        {
            "requested_comparison": "Multi-node DDP",
            "availability": "missing",
            "reason": "No multi-node cluster was available or measured.",
        },
        {
            "requested_comparison": "DiLoCo time to 3.28",
            "availability": "missing",
            "reason": (
                "Missing crossing step, tokens, and training seconds. K=8 DiLoCo ended at "
                "3.5183 after 4.92B tokens and did not cross 3.28."
            ),
        },
        {
            "requested_comparison": "DiLoCo step time on slow transport",
            "availability": "partial",
            "reason": (
                "diloco_transport.csv reconstructs it: the outer sync moves the same tensor "
                "DDP all-reduces every step, so each transport's measured communication term "
                "divides by H=500. Sound, but not directly measured."
            ),
        },
        {
            "requested_comparison": "Price per convergence run across conditions",
            "availability": "partial",
            "reason": (
                "Missing a matching 1xA100 rental rate, the desktop energy/capital convention, "
                "and convergence times for an 8-GPU PCIe box and for DiLoCo."
            ),
        },
        {
            "requested_comparison": "Pinned versus current PyTorch performance",
            "availability": "missing",
            "reason": (
                "Missing mean and standard deviation of step time for the same DDP mode matrix "
                "under a current PyTorch build."
            ),
        },
        {
            "requested_comparison": "DDP validation-loss curve and 8xA100 crossing",
            "availability": "complete",
            "reason": "Raw logs contain the full clean 10000-step schedule and first crossing.",
        },
        {
            "requested_comparison": "DDP versus DiLoCo at equal tokens",
            "availability": "complete",
            "reason": "Both raw curves exist through step 9999 at global batch 480.",
        },
        {
            "requested_comparison": "1-to-8 GPU matched-micro-batch scaling",
            "availability": "complete",
            "reason": "Raw benchmark JSON exists for 1 and 8 A100s at global batch 480.",
        },
        {
            "requested_comparison": "DDP transport sensitivity",
            "availability": "partial",
            "reason": "NVLink and TCP are direct step measurements; 40/10 Gbit netem points are upper-bound reconstructions.",
        },
        {
            "requested_comparison": "8-GPU rental availability",
            "availability": "complete",
            "reason": (
                "capacity_availability.csv projects 350 polls of both venues from the "
                "capacity watcher's log, replacing the draft's screenshot of a console."
            ),
        },
        {
            "requested_comparison": "DiLoCo merge penalty versus replica count",
            "availability": "partial",
            "reason": (
                "Two arms only, K=2 and K=8. The committed K=2 arm ran a 6000-step "
                "schedule against the K=8 arm's 10000, so the replica count is still "
                "confounded with the schedule; reported as each arm's gap from its own "
                "reference for that reason. Matched 10000-step K=2 arms (diloco-k2-10k, "
                "ref-1gpu-10k) were launched on aurora 2026-08-22 and supersede this "
                "when they land."
            ),
        },
        {
            "requested_comparison": "Four DDP implementations over forced TCP/loopback",
            "availability": "complete",
            "reason": (
                "Raw benchmark JSON contains mean and standard deviation for naive, bucketed, "
                "interleaved, and PyTorch DDP at 8 ranks."
            ),
        },
    ]
    write_csv(output / "data_gaps.csv", list(rows[0]), rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--trackio-db",
        type=Path,
        default=Path.home() / ".cache/huggingface/trackio/distrain.db",
        help="aurora's Trackio store, source of the K=2 DiLoCo arm; skipped when absent",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output or root / "docs/writeup_data"

    collect_validation_curves(root, output)
    collect_training_curves(root, output)
    collect_diloco_sync(root, output)
    collect_scaling(root, output)
    transport_rows, anchor_compute_ms = collect_transport(root, output)
    diloco_rows = collect_diloco_transport(root, output, transport_rows)
    collect_crossovers(output, anchor_compute_ms)
    cost_rows = collect_costs(root, output, transport_rows)
    collect_transport_costs(output, transport_rows, diloco_rows, cost_rows)
    collect_diloco_k_penalty(root, output, args.trackio_db)
    collect_capacity(root, output)
    collect_pcie(root, output)
    collect_ddp_modes(root, output)
    collect_ratio_sweep(root, output)
    collect_gaps(output)
    print(f"wrote writeup data to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""End-to-end check of scripts/bench_ddp_modes.py on gloo/CPU.

The timings it produces here are meaningless -- two ranks share one CPU -- but the
harness mechanics are exactly what the first rented session depends on: torchrun
launch, log parsing, warmup exclusion, durable JSON. A parsing bug found on the
rented box costs money; found here it costs a minute.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "bench_ddp_modes.py"


class TestBenchHarness:
    def test_end_to_end_on_gloo_cpu(self, tiny_data, tmp_path):
        steps, warmup = 4, 1
        cmd = [
            sys.executable, str(SCRIPT),
            "--nproc", "2", "--steps", str(steps), "--warmup", str(warmup),
            "--backend", "gloo", "--modes", "ddp_naive", "ddp_interleaved",
            "--no-compile", "--seq-len", "32", "--per-gpu-batch", "2",
            "--out-dir", str(tmp_path / "bench"),
            "--",
            "--device", "cpu",
            "--train-glob", str(tiny_data / "t_train_*.bin"),
            "--val-glob", str(tiny_data / "t_val_*.bin"),
            "--n-layer", "2", "--n-head", "2", "--n-embd", "32",
            "--vocab-size", "64",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False,
                              cwd=REPO_ROOT)
        assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

        results_files = list((tmp_path / "bench").glob("*/results.json"))
        assert len(results_files) == 1
        payload = json.loads(results_files[0].read_text())

        by_label = {r["label"]: r for r in payload["results"]}
        assert set(by_label) == {"single", "ddp_naive", "ddp_interleaved"}
        for label, r in by_label.items():
            # warmup steps must be excluded from the recorded sample
            assert len(r["step_times_ms"]) == steps - warmup, label
            assert r["mean_ms"] > 0, label
            assert r["tokens_per_s_total"] > 0, label
            assert (results_files[0].parent / f"{label}.log").exists()

        # the comparison table made it to stdout
        assert "vs naive" in proc.stdout

    def test_hung_run_is_recorded_not_fatal(self, tiny_data, tmp_path):
        """A distributed bug's natural symptom is a hang; the harness must survive
        one and still report the other modes. Simulated with a 1-second timeout
        that no real run can meet."""
        cmd = [
            sys.executable, str(SCRIPT),
            "--nproc", "2", "--steps", "4", "--warmup", "1",
            "--backend", "gloo", "--modes", "ddp_naive",
            "--no-compile", "--no-single", "--seq-len", "32", "--per-gpu-batch", "2",
            "--timeout", "1",
            "--out-dir", str(tmp_path / "bench"),
            "--",
            "--device", "cpu",
            "--train-glob", str(tiny_data / "t_train_*.bin"),
            "--val-glob", str(tiny_data / "t_val_*.bin"),
            "--n-layer", "2", "--n-head", "2", "--n-embd", "32",
            "--vocab-size", "64",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False,
                              cwd=REPO_ROOT)
        assert proc.returncode == 1, proc.stdout

        results_files = list((tmp_path / "bench").glob("*/results.json"))
        payload = json.loads(results_files[0].read_text())
        (record,) = payload["results"]
        assert "timeout" in record["error"]

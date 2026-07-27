"""Training-loop tests, including a tiny end-to-end run on CPU."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from distrain.data import write_shard
from distrain.train import TrainConfig, lr_at, resolve_dtype, train


class TestLRSchedule:
    def test_warmup_rises_to_peak(self):
        cfg = TrainConfig(learning_rate=1.0, min_lr=0.0, warmup_steps=10, max_steps=100)
        assert lr_at(0, cfg) == pytest.approx(0.1)
        assert lr_at(9, cfg) == pytest.approx(1.0)

    def test_cosine_decays_to_min_lr(self):
        cfg = TrainConfig(learning_rate=1.0, min_lr=0.1, warmup_steps=10, max_steps=100)
        assert lr_at(10, cfg) == pytest.approx(1.0)
        assert lr_at(99, cfg) == pytest.approx(0.1, abs=1e-3)
        assert lr_at(1000, cfg) == pytest.approx(0.1)

    def test_monotonically_decreasing_after_warmup(self):
        cfg = TrainConfig(learning_rate=1.0, min_lr=0.1, warmup_steps=10, max_steps=100)
        values = [lr_at(s, cfg) for s in range(10, 100)]
        assert all(a >= b for a, b in itertools.pairwise(values))

    def test_midpoint_is_halfway(self):
        cfg = TrainConfig(learning_rate=1.0, min_lr=0.0, warmup_steps=0, max_steps=100)
        assert lr_at(50, cfg) == pytest.approx(0.5, abs=1e-2)


class TestDtypeResolution:
    def test_cuda_defaults_to_bf16(self):
        assert resolve_dtype("auto", "cuda") is torch.bfloat16

    @pytest.mark.parametrize("device", ["cpu", "mps"])
    def test_non_cuda_defaults_to_fp32(self, device):
        assert resolve_dtype("auto", device) is torch.float32

    def test_explicit_overrides_default(self):
        assert resolve_dtype("bf16", "cpu") is torch.bfloat16


@pytest.fixture
def tiny_data(tmp_path):
    """Small synthetic shards with learnable structure.

    Random tokens would leave loss pinned at ln(vocab); a repeating pattern gives the
    model something to fit, so "loss goes down" actually tests the loop.
    """
    rng = np.random.default_rng(0)
    pattern = rng.integers(0, 64, size=97, dtype=np.uint16)
    tokens = np.tile(pattern, 400).astype(np.uint16)
    write_shard(tmp_path / "t_train_000001.bin", tokens)
    write_shard(tmp_path / "t_val_000000.bin", tokens[:8000])
    return tmp_path


def tiny_train_config(tiny_data, **overrides):
    cfg = {
        "train_glob": str(tiny_data / "t_train_*.bin"),
        "val_glob": str(tiny_data / "t_val_*.bin"),
        "seq_len": 32,
        "global_batch_seqs": 8,
        "grad_accum_steps": 1,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 32,
        "vocab_size": 64,
        "warmup_steps": 2,
        "max_steps": 30,
        "learning_rate": 3e-3,
        "min_lr": 3e-4,
        "val_every": 15,
        "val_tokens": 3200,
        "device": "cpu",
        "trackio": False,
        "log_every": 100,
    }
    cfg.update(overrides)
    return TrainConfig(**cfg)


class TestEndToEnd:
    def test_loop_runs_and_loss_decreases(self, tiny_data, capsys):
        results = train(tiny_train_config(tiny_data))
        out = capsys.readouterr().out
        val_losses = [float(line.split("val_loss")[1].split("|")[0])
                      for line in out.splitlines() if "val_loss" in line]
        assert len(val_losses) >= 2
        assert val_losses[-1] < val_losses[0], f"val loss did not fall: {val_losses}"
        assert results["train_time_s"] > 0

    def test_training_time_excludes_validation(self, tiny_data):
        """train_time_s is the reported clock; wall_time_s includes eval overhead."""
        results = train(tiny_train_config(tiny_data, val_every=5))
        assert results["train_time_s"] < results["wall_time_s"]

    def test_records_first_crossing_of_target(self, tiny_data):
        """A target above the initial loss must be recorded at the first evaluation."""
        results = train(tiny_train_config(tiny_data, target_val_loss=100.0, max_steps=16,
                                          val_every=15))
        assert results["target_reached_step"] == 0
        assert results["target_reached_train_time_s"] is not None

    def test_unreachable_target_is_not_recorded(self, tiny_data):
        results = train(tiny_train_config(tiny_data, target_val_loss=0.0))
        assert results["target_reached_step"] is None

    def test_gradient_accumulation_matches_single_pass(self, tiny_data):
        """Accumulating over microbatches must equal one big batch, not scale the LR."""
        a = train(tiny_train_config(tiny_data, grad_accum_steps=1, max_steps=6,
                                    val_every=5))
        b = train(tiny_train_config(tiny_data, grad_accum_steps=4, max_steps=6,
                                    val_every=5))
        assert a["target_reached_step"] == b["target_reached_step"]

    def test_missing_shards_give_actionable_error(self, tmp_path):
        cfg = TrainConfig(train_glob=str(tmp_path / "nothing_*.bin"), trackio=False)
        with pytest.raises(FileNotFoundError, match="make_synthetic_shards"):
            train(cfg)

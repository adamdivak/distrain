"""Training-loop tests, including a tiny end-to-end run on CPU."""

from __future__ import annotations

import itertools

import pytest
import torch

from distrain.train import TrainConfig, lr_at, resolve_dtype, train
from helpers import tiny_train_config


class TestLRSchedule:
    """Trapezoid (modded-nanogpt): linear warmup, flat plateau, linear warmdown to 0.

    Deliberately tested at learning_rate != 1.0: this function was ported from a
    LambdaLR *multiplier*, and 1.0 is the one base LR at which returning the bare
    multiplier is indistinguishable from returning a learning rate -- which is how
    a 1.0-LR run got launched.
    """

    def _cfg(self, **overrides):
        base = {"learning_rate": 0.5, "warmup_steps": 10, "warmdown_steps": 20,
                "max_steps": 100}
        base.update(overrides)
        return TrainConfig(**base)

    def test_warmup_is_linear(self):
        cfg = self._cfg()
        assert lr_at(0, cfg) == pytest.approx(0.05)
        assert lr_at(4, cfg) == pytest.approx(0.25)
        assert lr_at(9, cfg) == pytest.approx(0.5)

    def test_plateau_holds_peak_lr_exactly(self):
        cfg = self._cfg()
        assert lr_at(10, cfg) == 0.5
        assert lr_at(79, cfg) == 0.5

    def test_warmdown_is_linear_to_zero(self):
        cfg = self._cfg()
        assert lr_at(80, cfg) == pytest.approx(0.5)  # continuous at the boundary
        assert lr_at(90, cfg) == pytest.approx(0.25)
        assert lr_at(100, cfg) == pytest.approx(0.0)

    def test_returns_a_learning_rate_not_a_multiplier(self):
        """Regression: assigning the bare LambdaLR multiplier as the LR means
        training at 1.0 on the plateau -- a 556x overshoot that exploded a run."""
        cfg = self._cfg(learning_rate=3e-3)
        assert lr_at(50, cfg) == pytest.approx(3e-3)

    def test_never_increases_after_warmup(self):
        cfg = self._cfg()
        values = [lr_at(s, cfg) for s in range(10, 101)]
        assert all(a >= b for a, b in itertools.pairwise(values))

    def test_rejects_schedule_longer_than_run(self):
        """A run too short for its warmup+warmdown must fail loudly rather than
        silently spend every step inside the warmdown ramp."""
        cfg = self._cfg(warmup_steps=50, warmdown_steps=60)
        with pytest.raises(AssertionError, match="Incorrect LR schedule"):
            lr_at(0, cfg)


class TestDtypeResolution:
    def test_cuda_defaults_to_bf16(self):
        assert resolve_dtype("auto", "cuda") is torch.bfloat16

    @pytest.mark.parametrize("device", ["cpu", "mps"])
    def test_non_cuda_defaults_to_fp32(self, device):
        assert resolve_dtype("auto", device) is torch.float32

    def test_explicit_overrides_default(self):
        assert resolve_dtype("bf16", "cpu") is torch.bfloat16


class TestEndToEnd:
    def test_distributed_run_requires_explicit_mode(self, tiny_data):
        cfg = tiny_train_config(tiny_data, world_size=2, distributed_mode=None)
        with pytest.raises(ValueError, match="requires --distributed-mode"):
            train(cfg)

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


class TestCheckpointing:
    def test_resume_matches_uninterrupted_run(self, tiny_data):
        """Interrupt at step 4, resume to 8: bitwise-identical to a straight 8-step run.

        Everything the loop consumes is a pure function of the step index (LR, data
        order) or restored from the checkpoint (params, optimizer state), and CPU fp32
        with dropout 0 is deterministic -- so exact equality is the correct bar, and
        anything the checkpoint failed to carry would break it.

        LR is pinned flat because the warmdown position derives from max_steps,
        and emulating the interrupt here means giving the first segment a smaller
        max_steps. A real resume reuses the same command line, so its schedule is
        identical across segments by construction.
        """
        # explicit, not inherited: this test *requires* a schedule that is
        # independent of max_steps, whatever helpers defaults become
        flat_lr = {"learning_rate": 3e-3, "warmup_steps": 0, "warmdown_steps": 0}
        ckpt_dir = str(tiny_data / "ckpt")
        straight = train(tiny_train_config(tiny_data, max_steps=8, **flat_lr),
                         return_debug_values=["model"])
        train(tiny_train_config(tiny_data, max_steps=4, checkpoint_every=4,
                                checkpoint_dir=ckpt_dir, **flat_lr))
        resumed = train(tiny_train_config(tiny_data, max_steps=8, resume=True,
                                          checkpoint_dir=ckpt_dir, **flat_lr),
                        return_debug_values=["model"])
        a = dict(straight["model"].named_parameters())
        b = dict(resumed["model"].named_parameters())
        assert a.keys() == b.keys()
        for key, param in a.items():
            torch.testing.assert_close(
                param, b[key], rtol=0, atol=0,
                msg=lambda m, key=key: f"{key} diverged after resume:\n{m}")

    def test_resume_preserves_recorded_crossing_and_clock(self, tiny_data):
        """A crossing before the interrupt must survive it, and train_time_s accumulates."""
        ckpt_dir = str(tiny_data / "ckpt")
        first = train(tiny_train_config(tiny_data, max_steps=4, checkpoint_every=4,
                                        checkpoint_dir=ckpt_dir, target_val_loss=100.0,
                                        val_every=2))
        resumed = train(tiny_train_config(tiny_data, max_steps=8, resume=True,
                                          checkpoint_dir=ckpt_dir, target_val_loss=100.0,
                                          val_every=2))
        assert resumed["target_reached_step"] == first["target_reached_step"] == 0
        assert resumed["train_time_s"] > first["train_time_s"]

    def test_periodic_and_final_saves(self, tiny_data):
        """checkpoint_every=2 over 5 steps: periodic saves plus one at the last step."""
        ckpt_dir = tiny_data / "ckpt"
        train(tiny_train_config(tiny_data, max_steps=5, checkpoint_every=2,
                                checkpoint_dir=str(ckpt_dir)))
        state = torch.load(ckpt_dir / "ckpt.pt", weights_only=True)
        assert state["next_step"] == 5  # the is_last save, not the step-4 periodic one
        assert state["cfg"]["max_steps"] == 5

    def test_no_checkpoint_by_default(self, tiny_data):
        train(tiny_train_config(tiny_data, max_steps=2,
                                checkpoint_dir=str(tiny_data / "ckpt")))
        assert not (tiny_data / "ckpt").exists()

    def test_resume_without_checkpoint_fails_loudly(self, tiny_data):
        """Silently starting fresh would look like a resume and waste the night."""
        cfg = tiny_train_config(tiny_data, resume=True,
                                checkpoint_dir=str(tiny_data / "nowhere"))
        with pytest.raises(FileNotFoundError, match="--resume"):
            train(cfg)

import numpy as np
import pytest

from distrain.data import write_shard
from distrain.train import TrainConfig


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

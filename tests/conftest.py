import numpy as np
import pytest

from distrain.data import write_shard


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

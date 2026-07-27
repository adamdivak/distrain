"""Tests for shard IO and world-size-independent sharding.

`test_token_order_independent_of_world_size` is the one that matters: it enforces the
invariant in `project_brief.md` section 4. If it ever fails, every time-to-target-loss
comparison across GPU counts is invalid, and the failure is otherwise silent.
"""

from __future__ import annotations

import numpy as np
import pytest

from distrain.data import (
    HEADER_BYTES,
    MAGIC,
    DataLoader,
    ShardingPlan,
    TokenStream,
    open_shard,
    read_shard_header,
    write_shard,
)


def make_shards(tmp_path, counts, start=0):
    """Write shards holding consecutive token ids, so position is identifiable from value."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    tok = start
    for i, n in enumerate(counts):
        # values must fit in uint16; wrap deliberately, ids stay reproducible
        tokens = np.arange(tok, tok + n, dtype=np.int64) % 50257
        path = tmp_path / f"shard_{i:03d}.bin"
        write_shard(path, tokens.astype(np.uint16))
        paths.append(path)
        tok += n
    return paths


class TestShardIO:
    def test_roundtrip(self, tmp_path):
        tokens = np.array([0, 1, 50256, 7, 42], dtype=np.uint16)
        path = tmp_path / "a.bin"
        write_shard(path, tokens)
        assert read_shard_header(path) == 5
        np.testing.assert_array_equal(open_shard(path), tokens)

    def test_header_size_matches_upstream(self, tmp_path):
        path = tmp_path / "a.bin"
        write_shard(path, np.zeros(3, dtype=np.uint16))
        assert path.stat().st_size == HEADER_BYTES + 3 * 2
        header = np.fromfile(path, dtype=np.int32, count=3)
        assert header[0] == MAGIC and header[1] == 1 and header[2] == 3

    def test_rejects_bad_magic(self, tmp_path):
        path = tmp_path / "a.bin"
        write_shard(path, np.zeros(3, dtype=np.uint16))
        data = bytearray(path.read_bytes())
        data[0:4] = (MAGIC + 1).to_bytes(4, "little")
        path.write_bytes(bytes(data))
        with pytest.raises(ValueError, match="magic number mismatch"):
            read_shard_header(path)

    def test_rejects_truncated_body(self, tmp_path):
        path = tmp_path / "a.bin"
        write_shard(path, np.zeros(8, dtype=np.uint16))
        data = path.read_bytes()
        path.write_bytes(data[:-4])
        with pytest.raises(ValueError, match="header claims"):
            open_shard(path)


class TestTokenStream:
    def test_concatenates_shards(self, tmp_path):
        paths = make_shards(tmp_path, [10, 20, 5])
        stream = TokenStream(paths)
        assert stream.total_tokens == 35
        np.testing.assert_array_equal(stream.take(0, 35), np.arange(35, dtype=np.uint16))

    def test_read_spanning_shard_boundary(self, tmp_path):
        paths = make_shards(tmp_path, [10, 20, 5])
        stream = TokenStream(paths)
        # spans all three shards
        np.testing.assert_array_equal(stream.take(8, 25), np.arange(8, 33, dtype=np.uint16))
        # exactly one boundary
        np.testing.assert_array_equal(stream.take(10, 20), np.arange(10, 30, dtype=np.uint16))

    def test_out_of_range_raises(self, tmp_path):
        stream = TokenStream(make_shards(tmp_path, [10]))
        with pytest.raises(IndexError):
            stream.take(5, 10)

    def test_shard_split_is_invisible(self, tmp_path):
        """The same tokens split into different numbers of files read identically."""
        a = TokenStream(make_shards(tmp_path / "a", [100]))
        b = TokenStream(make_shards(tmp_path / "b", [30, 30, 40]))
        np.testing.assert_array_equal(a.take(0, 100), b.take(0, 100))


class TestShardingPlan:
    def test_rejects_indivisible_global_batch(self):
        with pytest.raises(ValueError, match="must be divisible"):
            ShardingPlan(seq_len=8, global_batch_seqs=6, world_size=4, rank=0)

    def test_rejects_rank_outside_world(self):
        with pytest.raises(ValueError, match="outside world"):
            ShardingPlan(seq_len=8, global_batch_seqs=8, world_size=2, rank=2)

    def test_ranks_partition_the_step_exactly(self):
        """Every sequence of a step is claimed by exactly one (rank, accum) microbatch."""
        world, accum, gbs = 4, 2, 32
        seen = []
        for rank in range(world):
            plan = ShardingPlan(
                seq_len=8, global_batch_seqs=gbs, world_size=world, rank=rank,
                grad_accum_steps=accum,
            )
            for a in range(accum):
                seen.append(plan.microbatch_seq_indices(step=3, accum_idx=a))
        allocated = np.sort(np.concatenate(seen))
        np.testing.assert_array_equal(allocated, np.arange(3 * gbs, 4 * gbs))


def collect_stream(tmp_path, world_size, grad_accum_steps, num_steps=4, seq_len=8,
                   global_batch_seqs=32, shard_counts=(100, 100, 61)):
    """Inputs and targets consumed over `num_steps`, in canonical (step, rank, accum) order.

    The two streams are accumulated separately: interleaving them per microbatch would
    make the concatenation order depend on microbatch size, which is exactly the thing
    being varied here.
    """
    paths = make_shards(tmp_path, list(shard_counts))
    stream = TokenStream(paths)
    xs, ys = [], []
    for step in range(num_steps):
        for rank in range(world_size):
            plan = ShardingPlan(
                seq_len=seq_len, global_batch_seqs=global_batch_seqs,
                world_size=world_size, rank=rank, grad_accum_steps=grad_accum_steps,
            )
            loader = DataLoader(stream, plan)
            for a in range(grad_accum_steps):
                x, y = loader.microbatch(step, a)
                xs.append(x.ravel())
                ys.append(y.ravel())
    return np.concatenate(xs), np.concatenate(ys)


class TestWorldSizeIndependence:
    @pytest.mark.parametrize("world_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("grad_accum_steps", [1, 2])
    def test_token_order_independent_of_world_size(self, tmp_path, world_size,
                                                   grad_accum_steps):
        """THE load-bearing test (project_brief.md section 4).

        At a fixed global batch, the tokens consumed per step -- and their order once
        the ranks are concatenated in rank order -- must be byte-identical no matter
        how many ranks the work is split across.
        """
        ref_x, ref_y = collect_stream(tmp_path / "ref", world_size=1, grad_accum_steps=1)
        got_x, got_y = collect_stream(
            tmp_path / f"w{world_size}a{grad_accum_steps}",
            world_size=world_size, grad_accum_steps=grad_accum_steps,
        )
        np.testing.assert_array_equal(got_x, ref_x)
        np.testing.assert_array_equal(got_y, ref_y)

    def test_sequence_contents_independent_of_batching(self, tmp_path):
        """Global sequence j is the same tokens regardless of batch or world config.

        This is the weaker-but-broader guarantee that also holds under weak scaling,
        where the global batch legitimately grows with world size.
        """
        stream = TokenStream(make_shards(tmp_path, [100, 61]))
        configs = [
            ShardingPlan(seq_len=8, global_batch_seqs=8, world_size=1, rank=0),
            ShardingPlan(seq_len=8, global_batch_seqs=64, world_size=8, rank=3,
                         grad_accum_steps=2),
        ]
        loaders = [DataLoader(stream, p) for p in configs]
        for j in range(loaders[0].num_sequences):
            np.testing.assert_array_equal(loaders[0].sequence(j), loaders[1].sequence(j))


class TestDataLoader:
    def test_targets_are_inputs_shifted_by_one(self, tmp_path):
        stream = TokenStream(make_shards(tmp_path, [200]))
        plan = ShardingPlan(seq_len=8, global_batch_seqs=4, world_size=1, rank=0)
        x, y = DataLoader(stream, plan).microbatch(step=0)
        assert x.shape == y.shape == (4, 8)
        np.testing.assert_array_equal(x[:, 1:], y[:, :-1])

    def test_sequences_are_contiguous_in_the_stream(self, tmp_path):
        stream = TokenStream(make_shards(tmp_path, [200]))
        plan = ShardingPlan(seq_len=8, global_batch_seqs=4, world_size=1, rank=0)
        x, _ = DataLoader(stream, plan).microbatch(step=0)
        # shards hold consecutive ids, so row b must start at token 8*b
        np.testing.assert_array_equal(x[:, 0], np.arange(4) * 8)

    def test_wraps_at_end_of_corpus(self, tmp_path):
        stream = TokenStream(make_shards(tmp_path, [65]))  # 8 sequences of 8 (+1 spare)
        plan = ShardingPlan(seq_len=8, global_batch_seqs=8, world_size=1, rank=0)
        loader = DataLoader(stream, plan)
        assert loader.num_sequences == 8
        np.testing.assert_array_equal(loader.sequence(0), loader.sequence(8))

    def test_rejects_corpus_too_small(self, tmp_path):
        stream = TokenStream(make_shards(tmp_path, [8]))
        plan = ShardingPlan(seq_len=1024, global_batch_seqs=1, world_size=1, rank=0)
        with pytest.raises(ValueError, match="too few"):
            DataLoader(stream, plan)

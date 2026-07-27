"""Memmap `.bin` shard reading and world-size-independent data sharding.

The shard format is the one used by nanoGPT/modded-nanogpt, documented in
`reference/PROVENANCE.md`: a 256 x int32 header followed by uint16 GPT-2 BPE tokens.

The important property implemented here is that **which tokens a step consumes is a
function of the step index alone, not of the world size**. Changing GPU count must
not change data ordering -- otherwise every time-to-target-loss comparison in this
project is silently confounded (`project_brief.md` section 4). See
`ShardingPlan` for the exact scheme and `tests/test_data.py` for the enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = 20240520
VERSION = 1
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4
TOKEN_DTYPE = np.uint16


def write_shard(path: str | Path, tokens: np.ndarray) -> None:
    """Write tokens to `path` in the standard .bin shard format."""
    tokens = np.asarray(tokens)
    if tokens.ndim != 1:
        raise ValueError(f"expected 1-D token array, got shape {tokens.shape}")
    if tokens.size >= 2**31:
        raise ValueError("token count too large for an int32 header field")
    if tokens.min(initial=0) < 0 or tokens.max(initial=0) > np.iinfo(TOKEN_DTYPE).max:
        raise ValueError("token ids do not fit in uint16")

    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = tokens.size
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(TOKEN_DTYPE, copy=False).tobytes())


def read_shard_header(path: str | Path) -> int:
    """Return the token count declared by the shard header, validating magic/version."""
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    if header.size < HEADER_INTS:
        raise ValueError(f"{path}: file is shorter than a {HEADER_BYTES}-byte header")
    if header[0] != MAGIC:
        raise ValueError(f"{path}: magic number mismatch (got {header[0]}, want {MAGIC})")
    if header[1] != VERSION:
        raise ValueError(f"{path}: unsupported shard version {header[1]}")
    return int(header[2])


def open_shard(path: str | Path) -> np.memmap:
    """Memory-map a shard's token body. No data is read until it is indexed."""
    num_tokens = read_shard_header(path)
    expected = HEADER_BYTES + num_tokens * TOKEN_DTYPE().itemsize
    actual = Path(path).stat().st_size
    if actual != expected:
        raise ValueError(
            f"{path}: header claims {num_tokens} tokens ({expected} bytes) "
            f"but file is {actual} bytes"
        )
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r", offset=HEADER_BYTES, shape=(num_tokens,))


class TokenStream:
    """A flat, read-only view over an ordered list of shards, indexed by global token index.

    Shard boundaries are invisible to callers: the shards concatenate into one token
    sequence, and global index 0 is the first token of the first shard. This is what
    makes "shard the data by global token index" well defined independently of how
    many files the data happens to be split across.
    """

    def __init__(self, paths: list[str | Path]):
        if not paths:
            raise ValueError("TokenStream needs at least one shard")
        self.paths = [Path(p) for p in paths]
        self.shards = [open_shard(p) for p in self.paths]
        counts = [s.shape[0] for s in self.shards]
        # starts[i] is the global index of shard i's first token; starts[-1] is the total
        self.starts = np.cumsum([0] + counts)

    @property
    def total_tokens(self) -> int:
        return int(self.starts[-1])

    def take(self, start: int, length: int) -> np.ndarray:
        """Return `length` tokens beginning at global index `start`, spanning shards."""
        if length < 0:
            raise ValueError(f"negative length {length}")
        if start < 0 or start + length > self.total_tokens:
            raise IndexError(
                f"[{start}, {start + length}) out of range for {self.total_tokens} tokens"
            )
        if length == 0:
            return np.empty(0, dtype=TOKEN_DTYPE)

        first = int(np.searchsorted(self.starts, start, side="right") - 1)
        out = np.empty(length, dtype=TOKEN_DTYPE)
        written = 0
        shard = first
        while written < length:
            offset = start + written - int(self.starts[shard])
            available = self.shards[shard].shape[0] - offset
            n = min(available, length - written)
            out[written : written + n] = self.shards[shard][offset : offset + n]
            written += n
            shard += 1
        return out


@dataclass(frozen=True)
class ShardingPlan:
    """Maps (step, grad-accum index) to global sequence indices for one rank.

    The token stream is cut into fixed sequences of `seq_len` tokens: global sequence
    `j` is always tokens `[j * seq_len, (j + 1) * seq_len]` inclusive of one extra
    token for the shifted targets. That mapping depends on `seq_len` alone, never on
    the world size, so sequence `j` is the same data in every run.

    An optimizer step consumes `global_batch_seqs` consecutive sequences, split into
    one contiguous block per rank, and each rank's block split again across its
    gradient-accumulation microbatches. Because the step's *range* is fixed, the set
    of sequences consumed at step `t` is identical for every world size -- only the
    partitioning of that fixed range changes. That is the invariant strong-scaling
    comparisons depend on.

    Under weak scaling `global_batch_seqs` grows with world size, so the per-step sets
    legitimately differ; the sequence-to-token mapping above still holds, which is the
    stronger guarantee that keeps runs interpretable.
    """

    seq_len: int
    global_batch_seqs: int
    world_size: int
    rank: int
    grad_accum_steps: int = 1

    def __post_init__(self) -> None:
        if self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {self.seq_len}")
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} outside world of {self.world_size}")
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}")
        divisor = self.world_size * self.grad_accum_steps
        if self.global_batch_seqs % divisor != 0:
            raise ValueError(
                f"global_batch_seqs={self.global_batch_seqs} must be divisible by "
                f"world_size*grad_accum_steps={divisor}; otherwise ranks would see "
                f"different amounts of data and the global batch would not be fixed"
            )

    @property
    def seqs_per_rank(self) -> int:
        return self.global_batch_seqs // self.world_size

    @property
    def microbatch_seqs(self) -> int:
        """Sequences per forward/backward -- the local batch size."""
        return self.global_batch_seqs // (self.world_size * self.grad_accum_steps)

    def microbatch_seq_indices(self, step: int, accum_idx: int) -> np.ndarray:
        """Global sequence indices for one microbatch. Unwrapped: may exceed the dataset."""
        if not 0 <= accum_idx < self.grad_accum_steps:
            raise ValueError(f"accum_idx {accum_idx} outside [0, {self.grad_accum_steps})")
        base = step * self.global_batch_seqs + self.rank * self.seqs_per_rank
        start = base + accum_idx * self.microbatch_seqs
        return np.arange(start, start + self.microbatch_seqs, dtype=np.int64)


class DataLoader:
    """Yields (inputs, targets) microbatches as numpy arrays of global-indexed sequences.

    Sequence indices past the end of the data wrap around, so a run simply repeats the
    corpus; the wrap point is a function of the data and `seq_len`, not of world size.
    """

    def __init__(self, stream: TokenStream, plan: ShardingPlan):
        self.stream = stream
        self.plan = plan
        # one spare token is needed for the last sequence's shifted target
        self.num_sequences = (stream.total_tokens - 1) // plan.seq_len
        if self.num_sequences < 1:
            raise ValueError(
                f"{stream.total_tokens} tokens is too few for even one sequence of "
                f"length {plan.seq_len}"
            )

    def sequence(self, seq_index: int) -> np.ndarray:
        """The `seq_len + 1` tokens of one global sequence, wrapping at the corpus end."""
        j = int(seq_index) % self.num_sequences
        return self.stream.take(j * self.plan.seq_len, self.plan.seq_len + 1)

    def microbatch(self, step: int, accum_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
        indices = self.plan.microbatch_seq_indices(step, accum_idx)
        buf = np.stack([self.sequence(j) for j in indices])
        return buf[:, :-1], buf[:, 1:]

"""Generate small synthetic .bin shards in the FineWeb/nanoGPT format.

For local development where downloading 190 MiB real shards is not worthwhile: it
exercises the loader, the training loop and the multi-rank correctness tests without
any network access. The tokens are noise, so loss curves from synthetic data mean
nothing -- it is a plumbing fixture, not data.

    uv run python scripts/make_synthetic_shards.py --out data/synthetic --shards 3

Real data comes from `reference/modded_nanogpt/cached_fineweb10B.py` (see
`reference/PROVENANCE.md` for sizes before you run it).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from distrain.data import write_shard

GPT2_VOCAB_SIZE = 50257


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/synthetic"))
    p.add_argument("--shards", type=int, default=2)
    p.add_argument("--tokens-per-shard", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # shard 0 is the val shard, mirroring the upstream train/val file naming
    write_shard(
        args.out / "synthetic_val_000000.bin",
        rng.integers(0, GPT2_VOCAB_SIZE, size=args.tokens_per_shard, dtype=np.uint16),
    )
    for i in range(1, args.shards + 1):
        write_shard(
            args.out / f"synthetic_train_{i:06d}.bin",
            rng.integers(0, GPT2_VOCAB_SIZE, size=args.tokens_per_shard, dtype=np.uint16),
        )

    total_mib = (args.shards + 1) * args.tokens_per_shard * 2 / 2**20
    print(f"wrote {args.shards + 1} shards to {args.out} ({total_mib:.1f} MiB)")


if __name__ == "__main__":
    main()

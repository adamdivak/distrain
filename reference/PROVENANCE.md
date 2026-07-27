# Vendored reference code

Third-party code, copied verbatim and **not modified**. Everything this project
actually runs lives in `src/distrain/`; this directory exists so that "what was
inherited" and "what was written here" stay separable.

Both upstreams are MIT licensed; the original `LICENSE` files are copied alongside.

| Path | Upstream | Commit | Date | License |
|---|---|---|---|---|
| `modded_nanogpt/cached_fineweb10B.py` | [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | `003ff3e2` | 2026-07-26 | MIT © Keller Jordan |
| `nanogpt/model.py`, `nanogpt/train.py` | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | `3adf61e1` | 2025-11-12 | MIT © Andrej Karpathy |

`cached_fineweb10B.py` is used as a tool, verbatim. The nanoGPT files are here to
be *read* — the model and loop in `src/distrain/` are written against them, not
imported from them.

Only the two files above were taken from modded-nanogpt. The current `train_gpt.py`
there is the heavily-optimized record holder (custom optimizers, bigram vocab,
Triton kernels); per `project_brief.md` §3 its micro-optimizations are deliberately
ignored.

---

## The 3.28 target — exact definition

From the modded-nanogpt README (`003ff3e2`), the target is cross-entropy loss on
**the first 10,485,760 tokens of the FineWeb validation set** — i.e. `val_tokens =
10485760`, taken from the start of `fineweb_val_000000.bin`, evaluated at any
sequence length so long as the model remains a valid probability model of language.

Two consequences this project must respect (`project_brief.md` §4):

- The GPT-2 BPE tokenizer and that exact val slice *are* the definition of 3.28.
  Neither may be "improved".
- FineWeb ≠ FineWeb-Edu. `cached_finewebedu10B.py` exists upstream and is the
  wrong script.

Upstream also requires, for leaderboard submissions, enough runs to show p<0.01
that mean val loss ≤ 3.28 — because inter-run variance at this target is
non-negligible. That is independent confirmation of the seed-variance budget in
[`../docs/decisions.md`](../docs/decisions.md) §5: a single run's
time-to-target-loss is not a reliable number.

## `.bin` shard format

Read by `_load_data_shard` upstream; reimplemented in `src/distrain/`:

- **Header**: 256 × `int32` (1024 bytes)
  - `header[0]` = `20240520` (magic)
  - `header[1]` = `1` (version)
  - `header[2]` = number of tokens following the header
- **Body**: `header[2]` × `uint16` GPT-2 BPE token ids

## Data volume — read before downloading

Every shard in `kjj0/fineweb10B-gpt2` is **190 MiB** (100M tokens × 2 bytes + header),
including the val shard. The full FineWeb10B pull is 103 train shards ≈ **19 GiB**;
`cached_fineweb10B.py N` downloads the val shard plus `N` train shards.

Local development on the Mac uses **synthetic shards** generated in this exact
format, so no download is required to work on the loader, the loop, or the
multi-rank correctness tests. Real shards are pulled on `aurora` and on rented
nodes, where bandwidth is not metered.

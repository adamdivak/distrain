"""Checkpoint save/load, retention pruning and the async off-box mirror.

Split out of train.py. The conventions — per-step files, atomic rename, rank-0
writes, keep_last/keep_every retention, skip-if-busy mirroring — are described in
docs/decisions.md section 12 and the 2026-08-16 session log.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import threading
from dataclasses import asdict
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    # Only for annotations: importing train at runtime would be circular,
    # since train.py imports this module.
    from distrain.model import GPT
    from distrain.train import TrainConfig

_CKPT_STEP_RE = re.compile(r"-step(\d+)\.pt$")
_mirror_thread: threading.Thread | None = None


def checkpoint_stem(cfg: TrainConfig) -> str:
    return (cfg.run_name or "run").replace(os.sep, "_")


def _step_pattern(rank: int | None) -> re.Pattern:
    """Filename pattern for one rank's checkpoints.

    rank=None is the single-file layout (single-device and DDP, where rank 0's
    state speaks for every rank); an integer rank is the per-rank diloco layout
    `{stem}-step{N}-rank{r}.pt` (decisions.md section 16). The two never match
    each other, so a directory can hold both without cross-talk.
    """
    if rank is None:
        return _CKPT_STEP_RE
    return re.compile(rf"-step(\d+)-rank{rank}\.pt$")


def rank_checkpoint_path(path: str, rank: int) -> str:
    """Map any rank's checkpoint filename to this rank's.

    Lets one --resume-from value work on every rank of a per-rank run: each
    rank substitutes its own rank suffix. A path without a rank suffix is
    returned unchanged (resuming a diloco run from a single-file checkpoint —
    every rank then loads the same state and a fresh round starts there).
    """
    return re.sub(r"-rank\d+\.pt$", f"-rank{rank}.pt", path)


def list_checkpoints(ckpt_dir: str, stem: str, rank: int | None = None) -> list[tuple[int, str]]:
    """(step, path) pairs for this run (and rank's) files in ckpt_dir, sorted by step."""
    pattern = _step_pattern(rank)
    entries = []
    for path in glob.glob(os.path.join(ckpt_dir, f"{stem}-step*.pt")):
        m = pattern.search(path)
        if m:
            entries.append((int(m.group(1)), path))
    return sorted(entries)


def find_latest_checkpoint(cfg: TrainConfig, rank: int | None = None) -> str | None:
    """Newest checkpoint for this run name: local dir first, then the mirror.

    Local wins when both exist — the mirror is a copy of local, so local is never
    behind it within one machine's lifetime; the mirror matters when the local disk
    is gone (a terminated pod) and the volume is all that survived.
    """
    for d in (cfg.checkpoint_dir, cfg.checkpoint_mirror):
        if d:
            entries = list_checkpoints(d, checkpoint_stem(cfg), rank)
            if entries:
                return entries[-1][1]
    return None


def _prune_checkpoints(ckpt_dir: str, stem: str, keep_last: int, keep_every: int,
                       rank: int | None = None) -> None:
    entries = list_checkpoints(ckpt_dir, stem, rank)
    keep = {path for _, path in entries[-max(keep_last, 1):]}
    if keep_every > 0:
        keep |= {path for step, path in entries if step % keep_every == 0}
    for _, path in entries:
        if path not in keep:
            os.remove(path)


def _mirror_checkpoint(path: str, cfg: TrainConfig, rank: int | None = None) -> None:
    """Copy the just-saved checkpoint to cfg.checkpoint_mirror in the background.

    Skip-if-busy: if the previous copy is still in flight the new one is dropped —
    the next save catches the mirror up. Copy lands under a .tmp name and is
    renamed, so the mirror never holds a truncated checkpoint.
    """
    global _mirror_thread
    if _mirror_thread is not None and _mirror_thread.is_alive():
        return

    def copy() -> None:
        os.makedirs(cfg.checkpoint_mirror, exist_ok=True)
        dst = os.path.join(cfg.checkpoint_mirror, os.path.basename(path))
        tmp = dst + ".tmp"
        shutil.copyfile(path, tmp)
        os.replace(tmp, dst)
        _prune_checkpoints(cfg.checkpoint_mirror, checkpoint_stem(cfg),
                           cfg.checkpoint_keep_last, cfg.checkpoint_keep_every, rank)

    _mirror_thread = threading.Thread(target=copy, daemon=True)
    _mirror_thread.start()


def wait_for_mirror(cfg: TrainConfig, rank: int | None = None) -> None:
    """Drain the mirror at end of run: join any in-flight copy, then copy the
    newest local checkpoint synchronously if skip-if-busy dropped it."""
    if _mirror_thread is not None:
        _mirror_thread.join()
    stem = checkpoint_stem(cfg)
    local = list_checkpoints(cfg.checkpoint_dir, stem, rank)
    if not local:
        return
    _, path = local[-1]
    dst = os.path.join(cfg.checkpoint_mirror, os.path.basename(path))
    if not os.path.exists(dst):
        os.makedirs(cfg.checkpoint_mirror, exist_ok=True)
        tmp = dst + ".tmp"
        shutil.copyfile(path, tmp)
        os.replace(tmp, dst)
        _prune_checkpoints(cfg.checkpoint_mirror, stem,
                           cfg.checkpoint_keep_last, cfg.checkpoint_keep_every, rank)


def save_checkpoint(model: GPT, optimizer, cfg: TrainConfig,
                    next_step: int, train_time_s: float, results: dict,
                    rank: int | None = None, outer_state: dict | None = None) -> str:
    """Per-step checkpoint file, written atomically.

    rank=None is the single-file layout, written by rank 0 only — sufficient for
    DDP because post-reduce gradients make every rank's optimizer state
    identical. diloco breaks exactly that premise, so there every rank passes
    its own `rank` and saves its replica + inner optimizer state; the outer
    state (round-start baseline + outer momentum, identical everywhere by
    construction) is passed by rank 0 only (decisions.md section 16).

    `os.replace` means an interrupt mid-save leaves prior checkpoints intact
    instead of a truncated file. RNG state is deliberately not saved: with
    `dropout == 0` the training loop draws no random numbers -- seeding only affects
    init, which the checkpoint overwrites.
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "next_step": next_step,
        "train_time_s": train_time_s,
        "results": {k: results[k] for k in
                    ("target_reached_step", "target_reached_train_time_s")},
        "cfg": asdict(cfg),
    }
    if outer_state is not None:
        state["outer"] = outer_state
    stem = checkpoint_stem(cfg)
    rank_suffix = "" if rank is None else f"-rank{rank}"
    path = os.path.join(cfg.checkpoint_dir,
                        f"{stem}-step{next_step:06d}{rank_suffix}.pt")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    _prune_checkpoints(cfg.checkpoint_dir, stem,
                       cfg.checkpoint_keep_last, cfg.checkpoint_keep_every, rank)
    if cfg.checkpoint_mirror:
        _mirror_checkpoint(path, cfg, rank)
    return path


def load_checkpoint(path: str | None, model: GPT, optimizer, device: str) -> dict:
    """Load model + optimizer state in place; the caller applies the rest."""
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            f"--resume was given but no checkpoint was found"
            f"{f' at {path!r}' if path else ''}. Drop --resume to start fresh, "
            f"point --checkpoint-dir (or --checkpoint-mirror) at the run to "
            f"continue, or name a file with --resume-from."
        )
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    state["path"] = path
    return state

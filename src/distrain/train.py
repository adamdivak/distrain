"""Single-device training loop with MFU instrumentation and trackio logging.

Deliberately single-device: the distributed layer is added on top of this, not woven
into it, so that the 1-GPU baseline every scaling number is divided by comes from the
same code path.

Two measurement conventions are fixed here (`docs/decisions.md` sections 3-4):

- **Training time excludes validation.** Time-to-target-loss is a property of the
  training work, and eval cost varies with eval cadence. Both `train_time_s` (the
  reported clock) and `wall_time_s` (everything) are logged.
- **First crossing, unsmoothed.** The reported result is the first evaluation at which
  val loss <= the target. No smoothing is applied to the val curve.

MFU is reported per GPU: this process's tokens divided by this process's time.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from distrain.data import DataLoader, ShardingPlan, TokenStream
from distrain.mfu import counter_for, peak_bf16_flops
from distrain.model import GPT, GPTConfig


@dataclass
class TrainConfig:
    # data
    train_glob: str = "data/synthetic/synthetic_train_*.bin"
    val_glob: str = "data/synthetic/synthetic_val_*.bin"
    seq_len: int = 1024
    # sequences per optimizer step across all ranks and accumulation steps.
    # 480 x 1024 ~= 0.5M tokens, the GPT-2 batch size.
    global_batch_seqs: int = 480
    grad_accum_steps: int = 1

    # model
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50304
    dropout: float = 0.0
    bias: bool = False

    # optimization
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 1000

    # evaluation
    val_every: int = 100
    # the 3.28 target is defined over the first 10,485,760 val tokens; see
    # reference/PROVENANCE.md
    val_tokens: int = 10_485_760
    target_val_loss: float = 3.28

    # runtime
    device: str = "auto"
    dtype: str = "auto"
    compile: bool = False
    seed: int = 1337
    world_size: int = 1
    rank: int = 0

    # logging
    project: str = "distrain"
    run_name: str | None = None
    log_every: int = 10
    trackio: bool = True

    extra: dict = field(default_factory=dict)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    """bf16 on CUDA; fp32 elsewhere.

    The non-CUDA path exists for correctness work only, and fp32 keeps it free of
    backend-specific autocast gaps. No performance number from it means anything
    anyway (`project_brief.md` section 8).
    """
    if requested == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[requested]


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to `min_lr`, as in nanoGPT."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def synchronize(device: str) -> None:
    """Make host timing reflect device work rather than queue depth."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def to_device(batch: np.ndarray, device: str) -> torch.Tensor:
    # shards are uint16, which torch has no first-class support for; int64 is what
    # embedding and cross_entropy want anyway
    return torch.from_numpy(batch.astype(np.int64)).to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model: GPT, stream: TokenStream, cfg: TrainConfig, device: str,
             autocast_ctx) -> float:
    """Mean cross-entropy over the first `val_tokens` tokens of the val stream.

    Deterministic and identical across configs: the same sequences in the same order,
    every time. That is what makes the 3.28 crossing comparable between runs.
    """
    plan = ShardingPlan(
        seq_len=cfg.seq_len,
        global_batch_seqs=cfg.global_batch_seqs // cfg.grad_accum_steps,
        world_size=1,
        rank=0,
    )
    loader = DataLoader(stream, plan)
    num_sequences = cfg.val_tokens // cfg.seq_len
    batch_seqs = plan.microbatch_seqs

    model.eval()
    total, counted = 0.0, 0
    for start in range(0, num_sequences, batch_seqs):
        n = min(batch_seqs, num_sequences - start)
        buf = np.stack([loader.sequence(j) for j in range(start, start + n)])
        x, y = to_device(buf[:, :-1], device), to_device(buf[:, 1:], device)
        with autocast_ctx:
            _, loss = model(x, y)
        total += loss.item() * n
        counted += n
    model.train()
    return total / counted


def train(cfg: TrainConfig) -> dict:
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    torch.manual_seed(cfg.seed)
    if device == "cuda":
        torch.cuda.manual_seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    autocast_ctx = (
        torch.autocast(device_type=device, dtype=dtype)
        if device == "cuda" and dtype != torch.float32
        else torch.autocast(device_type="cpu", enabled=False)
    )

    train_paths = sorted(glob.glob(cfg.train_glob))
    val_paths = sorted(glob.glob(cfg.val_glob))
    if not train_paths:
        raise FileNotFoundError(
            f"no shards matched {cfg.train_glob!r}. Generate synthetic data with "
            f"scripts/make_synthetic_shards.py, or fetch FineWeb per reference/PROVENANCE.md"
        )
    if not val_paths:
        raise FileNotFoundError(f"no shards matched {cfg.val_glob!r}")

    train_stream = TokenStream(train_paths)
    val_stream = TokenStream(val_paths)
    plan = ShardingPlan(
        seq_len=cfg.seq_len,
        global_batch_seqs=cfg.global_batch_seqs,
        world_size=cfg.world_size,
        rank=cfg.rank,
        grad_accum_steps=cfg.grad_accum_steps,
    )
    loader = DataLoader(train_stream, plan)

    model = GPT(
        GPTConfig(
            block_size=cfg.seq_len,
            vocab_size=cfg.vocab_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            dropout=cfg.dropout,
            bias=cfg.bias,
        )
    ).to(device)
    if cfg.compile:
        model = torch.compile(model)

    optimizer = model.configure_optimizer(
        cfg.weight_decay, cfg.learning_rate, cfg.betas, fused=(device == "cuda")
    )
    flops = counter_for(model, cfg.seq_len)
    peak = peak_bf16_flops(torch.cuda.get_device_name()) if device == "cuda" else None
    tokens_per_step_this_rank = plan.seqs_per_rank * cfg.seq_len

    is_primary = cfg.rank == 0
    if cfg.trackio and is_primary:
        import trackio

        trackio.init(project=cfg.project, name=cfg.run_name, config=asdict(cfg))

    if is_primary:
        print(
            f"device={device} dtype={dtype} params={model.num_params()/1e6:.1f}M "
            f"tokens/step={plan.global_batch_seqs * cfg.seq_len:,} steps={cfg.max_steps}"
        )

    results = {"target_reached_step": None, "target_reached_train_time_s": None}
    train_time_s = 0.0  # excludes validation
    wall_start = time.perf_counter()

    for step in range(cfg.max_steps):
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        synchronize(device)
        step_start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for accum in range(cfg.grad_accum_steps):
            xb, yb = loader.microbatch(step, accum)
            x, y = to_device(xb, device), to_device(yb, device)
            with autocast_ctx:
                _, loss = model(x, y)
            # average, not sum, so the gradient matches a single large batch
            (loss / cfg.grad_accum_steps).backward()
            loss_sum += loss.item()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        synchronize(device)
        step_time = time.perf_counter() - step_start
        train_time_s += step_time

        if is_primary and (step % cfg.log_every == 0 or step == cfg.max_steps - 1):
            metrics = {
                "train/loss": loss_sum / cfg.grad_accum_steps,
                "train/lr": lr,
                "perf/step_time_s": step_time,
                "perf/tokens_per_s": tokens_per_step_this_rank / step_time,
                "time/train_time_s": train_time_s,
                "time/wall_time_s": time.perf_counter() - wall_start,
            }
            if peak is not None:
                metrics["perf/mfu"] = flops.mfu(tokens_per_step_this_rank, step_time, peak)
                # identical to MFU until activation checkpointing is enabled
                metrics["perf/hfu"] = flops.hfu(tokens_per_step_this_rank, step_time, peak)
            _log(cfg, metrics, step)
            print(
                f"step {step:5d} | loss {metrics['train/loss']:.4f} | lr {lr:.2e} | "
                f"{step_time * 1e3:7.1f} ms"
                + (f" | mfu {metrics['perf/mfu'] * 100:.1f}%" if peak else "")
            )

        is_last = step == cfg.max_steps - 1
        if cfg.val_every > 0 and (step % cfg.val_every == 0 or is_last):
            val_loss = evaluate(model, val_stream, cfg, device, autocast_ctx)
            if is_primary:
                _log(cfg, {"val/loss": val_loss}, step)
                print(f"step {step:5d} | val_loss {val_loss:.4f} | "
                      f"train_time {train_time_s:.1f}s")
            # first crossing, unsmoothed
            if results["target_reached_step"] is None and val_loss <= cfg.target_val_loss:
                results["target_reached_step"] = step
                results["target_reached_train_time_s"] = train_time_s
                if is_primary:
                    print(f"reached target {cfg.target_val_loss} at step {step} "
                          f"after {train_time_s:.1f}s of training")

    results["train_time_s"] = train_time_s
    results["wall_time_s"] = time.perf_counter() - wall_start
    if cfg.trackio and is_primary:
        import trackio

        trackio.finish()
    return results


def _log(cfg: TrainConfig, metrics: dict, step: int) -> None:
    if cfg.trackio:
        import trackio

        trackio.log(metrics, step=step)


def main() -> None:
    p = argparse.ArgumentParser(description="Train a GPT on .bin token shards")
    defaults = TrainConfig()
    for name, value in asdict(defaults).items():
        if name == "extra":
            continue
        if isinstance(value, bool):
            p.add_argument(f"--{name.replace('_', '-')}", dest=name,
                           action=argparse.BooleanOptionalAction, default=value)
        elif isinstance(value, tuple):
            p.add_argument(f"--{name.replace('_', '-')}", dest=name, nargs=2, type=float,
                           default=value)
        else:
            arg_type = type(value) if value is not None else str
            p.add_argument(f"--{name.replace('_', '-')}", dest=name, type=arg_type,
                           default=value)
    args = p.parse_args()
    cfg = TrainConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                         for k, v in vars(args).items()})
    # torchrun sets these; harmless when absent
    cfg.world_size = int(os.environ.get("WORLD_SIZE", cfg.world_size))
    cfg.rank = int(os.environ.get("RANK", cfg.rank))
    results = train(cfg)
    print(results)


if __name__ == "__main__":
    main()

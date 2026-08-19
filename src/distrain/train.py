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
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from distrain.checkpoints import (
    find_latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    rank_checkpoint_path,
    save_checkpoint,
    wait_for_mirror,
)
from distrain.data import DataLoader, ShardingPlan, TokenStream
from distrain.ddp_synchronizer import DdpSynchronizer, DiLoCoSynchronizer
from distrain.mfu import PeakSpec, counter_for, peak_bf16_spec
from distrain.model import GPT, GPTConfig

_warned_impossible = False


def warn_if_impossible(mfu: float, spec: PeakSpec | None) -> None:
    """Shout once if MFU exceeds 100%, which means the denominator is wrong.

    A GPU cannot exceed its own peak. When this fires, the recorded peak is too low
    and every throughput number from the run is inflated -- exactly the failure that
    put a 158% MFU on the board the first time this loop ran on a 3090.
    """
    global _warned_impossible
    if mfu <= 1.0 or _warned_impossible or spec is None:
        return
    _warned_impossible = True
    print(
        f"\n*** MFU {mfu * 100:.1f}% exceeds 100%, which is physically impossible.\n"
        f"*** Recorded peak is {spec.tflops:.1f} TFLOP/s ({spec.source}) and is too low.\n"
        f"*** Re-measure with scripts/measure_roofline.py; results from this run are "
        f"inflated by at least {mfu:.2f}x.\n"
    )


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
    learning_rate: float = 0.0018
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 250
    warmdown_steps: int = 1000
    max_steps: int = 2000

    # evaluation
    val_every: int = 100
    # the 3.28 target is defined over the first 10,485,760 val tokens; see
    # reference/PROVENANCE.md
    val_tokens: int = 10_485_760
    target_val_loss: float = 3.28
    # sequences per eval forward pass. Deliberately independent of global_batch_seqs,
    # grad_accum_steps and world_size: decisions.md section 4 requires the eval batch to be
    # identical across every config, and deriving it from the training batch would
    # both break that and make eval -- which never shards -- the memory high-water
    # mark of a multi-GPU run, since it would not shrink as ranks are added.
    eval_batch_seqs: int = 32
    # Per-replica diagnostic eval, 0 = off. Strictly a diagnostic: it logs under
    # diag/ keys, never feeds the target check, and never touches val/loss, whose
    # cadence stays mode-independent (decisions.md sections 4 and 16). Free to use
    # a finer cadence than val_every -- it carries no comparability obligation.
    diag_val_every: int = 0
    # Eval batch for the diag pass only, 0 = use eval_batch_seqs. Unlike the
    # reported val, which runs on rank 0 alone, every rank runs the diag pass at
    # once — so on a box where ranks share one GPU (aurora's 2-rank gloo smoke)
    # the logits tensor is allocated world_size times over. At eval_batch_seqs=32
    # that is 3.07 GiB per rank and a 3090 OOMs. Reported-path comparability
    # (decisions.md section 4) is untouched: this never reaches val/loss.
    diag_eval_batch_seqs: int = 0

    # runtime
    device: str = "auto"
    dtype: str = "auto"
    compile: bool = False
    seed: int = 1337

    # checkpointing: rank 0 writes {checkpoint_dir}/{run_name}-step{N}.pt every N
    # steps (0 = never); --resume continues from the newest one for this run name,
    # --resume-from from an explicit file. Retention: the newest keep_last files,
    # plus every keep_every'th step permanently (0 = off) — the permanent keeps are
    # what make "re-enter before the warmdown started" possible after the fact.
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 0
    resume: bool = False
    resume_from: str | None = None
    checkpoint_keep_last: int = 2
    checkpoint_keep_every: int = 2000
    # off-box durability: after the atomic local save, a background thread copies
    # the file here (e.g. a network volume) and applies the same retention. Copies
    # are skip-if-busy so a slow volume can never stall the training loop.
    checkpoint_mirror: str | None = None

    # distributed
    world_size: int = 1
    rank: int = 0 # rank in the whole training - [0, world_size)
    local_rank: int = 0 # rank (i.e. GPU) within the current machine - [0, num_gpus_in_current_machine]
    distributed_mode: str | None = None # ddp_naive, ddp_bucketed, ddp_interleaved, ddp_torch, diloco
    distributed_backend: str = "auto" # auto for torch to suggest based on the host
    ddp_bucket_size: int | None = 25 * 1024 * 1024 # bucket size in bytes when using bucketed ddp
    outer_sync_every: int | None = 500 # H in the DiLoCO paper
    outer_lr: float | None = 0.7
    outer_moment: float | None = 0.9


    # logging
    project: str = "distrain"
    run_name: str | None = None
    log_every: int = 10
    trackio: bool = True

    extra: dict = field(default_factory=dict)


def resolve_device(requested_device: str, local_rank: int | None = None) -> str:
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        if local_rank is None:
            local_rank = 0
        return f"cuda:{local_rank}"
    if torch.backends.mps.is_available():
        return "mps" # FIXME: can we have multiple mps or cpu devices?
    return "cpu"


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    """bf16 on CUDA; fp32 elsewhere.

    The non-CUDA path exists for correctness work only, and fp32 keeps it free of
    backend-specific autocast gaps. No performance number from it means anything
    anyway (`project_brief.md` section 8).
    """
    if requested == "auto":
        return torch.bfloat16 if is_cuda(device) else torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[requested]


# single source for every mode gate, so the CLI check cannot drift from what
# resolve_distributed_synchronizer actually dispatches on
DISTRIBUTED_MODES = {"diloco", *DdpSynchronizer.ddp_modes}


def resolve_distributed_synchronizer(cfg: TrainConfig, model):
    if cfg.distributed_mode == "diloco":
        if cfg.outer_sync_every is None:
            raise ValueError(f"{cfg.outer_sync_every=} must not be None when using DiLoCo")
        if cfg.outer_lr is None:
                raise ValueError(f"{cfg.outer_lr=} must not be None when using DiLoCo")
        if cfg.outer_moment is None:
                raise ValueError(f"{cfg.outer_moment=} must not be None when using DiLoCo")
        if cfg.val_every % cfg.outer_sync_every > 0:
            raise ValueError(f"Validation must happen at intervals that are divisible by diloco synchronization intervals, \
                                otherwise evaluation would be performed on a non-synchronized local variation of the model. \
                                {cfg.val_every=}, {cfg.outer_sync_every=}")
        return DiLoCoSynchronizer(model, cfg.world_size, cfg.outer_sync_every, cfg.outer_lr, cfg.outer_moment)
    elif cfg.distributed_mode in DdpSynchronizer.ddp_modes:
        return DdpSynchronizer(
            model, cfg.distributed_mode, cfg.world_size, cfg.ddp_bucket_size)
    else:
        raise ValueError(f"Invalid value {cfg.distributed_mode=}")

def lr_at(step: int, cfg: TrainConfig) -> float:
    """ Trapezoidal learning rate decay scheduler (linear warmup and warmdown) from modded-nanogpt """
    assert cfg.warmup_steps + cfg.warmdown_steps < cfg.max_steps, "Incorrect LR schedule, warmup + warmdown >= max_steps"
    assert step <= cfg.max_steps
    lr_multiplier = 1.0
    # 1) linear warmup for warmup_iters steps
    if step < cfg.warmup_steps:
        lr_multiplier = (step+1) / cfg.warmup_steps
    # 2) constant lr for a while
    elif step < cfg.max_steps - cfg.warmdown_steps:
        lr_multiplier = 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (cfg.max_steps - step) / cfg.warmdown_steps
        lr_multiplier = decay_ratio

    return lr_multiplier * cfg.learning_rate


def accelerator_synchronize(device: str) -> None:
    """Make host timing reflect device work rather than queue depth."""
    if is_cuda(device):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def to_device(batch: np.ndarray, device: str) -> torch.Tensor:
    # shards are uint16, which torch has no first-class support for; int64 is what
    # embedding and cross_entropy want anyway
    return torch.from_numpy(batch.astype(np.int64)).to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model: GPT, stream: TokenStream, cfg: TrainConfig, device: str,
             autocast_ctx, batch_seqs: int | None = None) -> float:
    """Mean cross-entropy over the first `val_tokens` tokens of the val stream.

    Deterministic and identical across configs: the same sequences in the same order,
    in batches of `cfg.eval_batch_seqs`, every time. That is what makes the 3.28
    crossing comparable between runs.

    `batch_seqs` overrides the batch for callers off the reported path (the diag
    eval), which needs a smaller footprint because every rank runs it at once.
    Grouping does not change the result beyond float reduction order: the mean is
    token-weighted over the same sequences in the same order.

    Never sharded -- this is one rank's full pass over the val split. Under DDP the
    caller runs it on rank 0 only and broadcasts the scalar, so every rank tests the
    same value against the target.

    Pure by design: no collectives here, so the call site alone determines
    rank-invariance and the function stays directly callable from tests.
    """
    plan = ShardingPlan(
        seq_len=cfg.seq_len,
        global_batch_seqs=batch_seqs or cfg.eval_batch_seqs,
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

def resolve_distributed_backend(requested_distributed_backend: str, device: str) -> str:
    if requested_distributed_backend == "auto":
        return torch.distributed.get_default_backend_for_device(device)
    else:
        return requested_distributed_backend

def setup_distributed(world_size, rank, device, init_method="env://", distributed_backend="auto"):
    # Choose an available accelerator by default, or use the user-specified one
    dist_backend = resolve_distributed_backend(distributed_backend, device)

    if is_cuda(device):
        # Set the correct cuda device. Must be before init_process_group
        torch.cuda.set_device(device)

    # initialize the process group
    torch.distributed.init_process_group(dist_backend, rank=rank, world_size=world_size, init_method=init_method)
    # return dist_backend

def cleanup_distributed():
    torch.distributed.destroy_process_group()

def is_distributed(cfg: TrainConfig) -> bool:
    return cfg.world_size > 1

def is_cuda(device: str) -> bool:
    return "cuda" in device

def train(cfg: TrainConfig, return_debug_values: list[str] | None = None) -> dict:
    if is_distributed(cfg) and cfg.distributed_mode not in DISTRIBUTED_MODES:
        choices = ", ".join(sorted(DISTRIBUTED_MODES))
        raise ValueError(
            f"world_size={cfg.world_size} requires --distributed-mode to be one of: {choices}"
        )

    device = resolve_device(cfg.device, cfg.local_rank)
    dtype = resolve_dtype(cfg.dtype, device)
    # Set a rank-specific seed, to avoid all ranks doing e.g. identical data augmentations
    torch.manual_seed(cfg.seed + cfg.rank)
    if is_cuda(device):
        torch.cuda.manual_seed(cfg.seed + cfg.rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    autocast_ctx = (
        torch.autocast(device_type=device, dtype=dtype)
        if is_cuda(device) and dtype != torch.float32
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
    # kept for checkpoint IO and eval: wrappers (torch.compile, DDP) prefix
    # state_dict keys but share parameters with the module they wrap
    raw_model = model
    use_torch_ddp = is_distributed(cfg) and cfg.distributed_mode == "ddp_torch"
    if use_torch_ddp:
        # The upstream baseline the hand-rolled modes are compared against.
        # Wrapped BEFORE torch.compile so dynamo sees the DDP module and applies
        # DDPOptimizer -- graph breaks at bucket boundaries, which is what keeps
        # DDP's backward hooks firing mid-backward under compile.
        # broadcast_buffers=False: GPT has no buffers, and buffer broadcast is a
        # collective in *forward*, which would deadlock rank-0-only eval.
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.device(device).index] if is_cuda(device) else None,
            broadcast_buffers=False,
            bucket_cap_mb=(cfg.ddp_bucket_size or 25 * 1024 * 1024) / (1024 * 1024),
        )
    if cfg.compile:
        model = torch.compile(model)

    optimizer = raw_model.configure_optimizer(
        cfg.weight_decay, cfg.learning_rate, cfg.betas, fused=(is_cuda(device))
    )

    # The synchronizer is built *before* any resume: its rank-0 broadcast anchors
    # a fresh start (decisions.md section 6), and the checkpoint must win over it
    # on resume. For DDP the order is indifferent (every rank loads the same
    # file), but for diloco it is load-bearing the other way around: mid-round
    # replicas legitimately differ per rank, and a post-load broadcast would
    # clobber every rank's restored replica with rank 0's.
    if is_distributed(cfg):
        dist_sync = resolve_distributed_synchronizer(cfg, model)

    # diloco checkpoints are per-rank files: each rank resumes its own replica and
    # inner optimizer state, since post-reduce identity across ranks -- what makes
    # the single rank-0 file sufficient for DDP -- does not hold there (section 16)
    ckpt_rank = cfg.rank if is_distributed(cfg) and cfg.distributed_mode == "diloco" else None
    start_step = 0
    resumed: dict | None = None
    if cfg.resume or cfg.resume_from:
        if cfg.resume_from:
            resume_path = (rank_checkpoint_path(cfg.resume_from, ckpt_rank)
                           if ckpt_rank is not None else cfg.resume_from)
        else:
            resume_path = find_latest_checkpoint(cfg, rank=ckpt_rank)
        resumed = load_checkpoint(resume_path, raw_model, optimizer, device)
        start_step = resumed["next_step"]
    # raw_model: DDP's wrapper (unlike torch.compile's) does not delegate
    # attribute access to the module it wraps
    flops = counter_for(raw_model, cfg.seq_len)
    peak_spec = peak_bf16_spec(torch.cuda.get_device_name()) if is_cuda(device) else None
    peak = peak_spec.tflops * 1e12 if peak_spec else None
    tokens_per_step_this_rank = plan.seqs_per_rank * cfg.seq_len

    if is_distributed(cfg) and resumed is not None:
        # Ranks resolved their resume files independently; a rank resuming at
        # a different step than rank 0 would silently walk a different data
        # and LR schedule, so disagreement is fatal, not recoverable.
        rank0_step = int(dist_sync.broadcast_scalar(float(start_step)))
        if rank0_step != start_step:
            raise RuntimeError(
                f"rank {cfg.rank} resumed at step {start_step} but rank 0 at "
                f"step {rank0_step}; a per-rank checkpoint set must come from "
                f"one save")
        # diloco only (no-op otherwise): rank 0 restores the round-start
        # baseline + outer momentum from its file and broadcasts them
        dist_sync.restore_outer_state(resumed.get("outer"))

    is_primary = cfg.rank == 0
    if cfg.trackio and is_primary:
        import trackio

        trackio.init(project=cfg.project, name=cfg.run_name, config=asdict(cfg))

    if is_primary:
        if resumed is not None:
            print(f"resumed from {resumed['path']} at step {start_step}")
        print(
            f"device={device} dtype={dtype} params={raw_model.num_params()/1e6:.1f}M "
            f"tokens/step={plan.global_batch_seqs * cfg.seq_len:,} steps={cfg.max_steps}"
        )

    results = {"target_reached_step": None, "target_reached_train_time_s": None}
    train_time_s = 0.0  # excludes validation
    if resumed is not None:
        # train_time_s carries across the interrupt so time-to-target stays
        # meaningful; wall_time_s is this process's clock and restarts
        results.update(resumed["results"])
        train_time_s = resumed["train_time_s"]
    wall_start = time.perf_counter()

    for step in range(start_step, cfg.max_steps):
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        accelerator_synchronize(device) # cuda sync, only for timing
        step_start = time.perf_counter()

        # Zero gradients outside the gradient accumulation microbatch-iterations,
        # so that we can share their combined value at the end
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        
        for accum in range(cfg.grad_accum_steps):
            # We want to avoid doing unnecessary communication during the gradient accumulation
            # steps, so a signal is sent to the synchronizer during the last iteration
            # so that it actually triggers all_reduce
            if is_distributed(cfg) and accum == cfg.grad_accum_steps - 1:
                dist_sync.set_last_iteration()

            xb, yb = loader.microbatch(step, accum)
            x, y = to_device(xb, device), to_device(yb, device)
            # DDP reduces on every backward unless told not to; our modes get the
            # same accumulation behaviour from set_last_iteration. no_sync must
            # cover the forward too -- DDP latches the flag at forward time.
            no_sync = (model.no_sync()
                       if use_torch_ddp and accum < cfg.grad_accum_steps - 1
                       else nullcontext())
            with no_sync:
                with autocast_ctx:
                    _, loss = model(x, y)
                # average, not sum, so the gradient matches a single large batch
                (loss / cfg.grad_accum_steps).backward()
            loss_sum += loss.item()

        # Synch gradients between ranks, or wait for the communication to finish when using hooks
        if is_distributed(cfg):
            dist_sync.finalize_gradients()

        # Do gradient clipping after synching has been finished
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        is_last = step == cfg.max_steps - 1

        # Per-replica diagnostic eval. Deliberately placed *before* the outer step:
        # at a diloco boundary this measures the pre-sync replicas while the val
        # block below measures the post-sync model, so the pair shows whether the
        # outer step helped or hurt. Off the reported path entirely (diag/ keys, no
        # target check). Every rank evaluates its own replica and the gather is
        # unconditional inside a branch whose condition is a function of the step
        # alone, so collective order stays rank-invariant (decisions.md section 6).
        # Excluded from step_time: it is measurement, not training.
        diag_time = 0.0
        if cfg.diag_val_every > 0 and (step % cfg.diag_val_every == 0 or is_last):
            accelerator_synchronize(device)
            diag_start = time.perf_counter()
            diag_loss = evaluate(raw_model if use_torch_ddp else model,
                                 val_stream, cfg, device, autocast_ctx,
                                 batch_seqs=cfg.diag_eval_batch_seqs or None)
            diag_losses = (dist_sync.gather_scalars(diag_loss)
                           if is_distributed(cfg) else [diag_loss])
            if is_primary:
                diag_metrics = {f"diag/val_rank{r}": v for r, v in enumerate(diag_losses)}
                if len(diag_losses) > 1:
                    diag_metrics["diag/val_replica_spread"] = max(diag_losses) - min(diag_losses)
                _log(cfg, diag_metrics, step)
                print(f"step {step:5d} | diag pre-sync val "
                      + " ".join(f"r{r}={v:.4f}" for r, v in enumerate(diag_losses)))
            accelerator_synchronize(device)
            diag_time = time.perf_counter() - diag_start

        # Perform synchronization / outer training step when using DiLoCo, after the inner optimization is done
        if is_distributed(cfg):
            dist_sync.maybe_outer_step(step, is_last)

        accelerator_synchronize(device) # cuda sync, only for timing
        step_time = time.perf_counter() - step_start - diag_time
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
                warn_if_impossible(metrics["perf/mfu"], peak_spec)
            _log(cfg, metrics, step)
            print(
                f"step {step:5d} | loss {metrics['train/loss']:.4f} | lr {lr:.2e} | "
                f"{step_time * 1e3:7.1f} ms"
                + (f" | mfu {metrics['perf/mfu'] * 100:.1f}%" if peak else "")
            )

        if cfg.val_every > 0 and (step % cfg.val_every == 0 or is_last):
            # Only rank 0 evaluates - every rank computing the identical full val pass
            # is N x the cost for one number. The broadcast is unconditional within this
            # branch, whose condition depends only on the step, so collective order stays
            # rank-invariant (decisions.md section 6).
            # ddp_torch evaluates on the underlying module: a DDP forward is not
            # guaranteed collective-free, and eval runs on rank 0 only
            eval_model = raw_model if use_torch_ddp else model
            val_loss = evaluate(eval_model, val_stream, cfg, device, autocast_ctx) if is_primary else 0.0
            if is_distributed(cfg):
                val_loss = dist_sync.broadcast_scalar(val_loss)

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

        # Outside the timed region, and after eval so a crossing recorded this step
        # is captured. Rank 0 only for the DDP modes (post-reduce state is
        # identical on every rank); every rank for diloco, where each replica's
        # params and inner optimizer state are its own (decisions.md section 16).
        if (cfg.checkpoint_every > 0 and (is_primary or ckpt_rank is not None)
                and ((step + 1) % cfg.checkpoint_every == 0 or is_last)):
            outer_state = (dist_sync.outer_state_dict()
                           if ckpt_rank is not None and is_primary else None)
            save_checkpoint(raw_model, optimizer, cfg,
                            step + 1, train_time_s, results,
                            rank=ckpt_rank, outer_state=outer_state)

    # The final save's off-box copy must land before the process can exit.
    if (cfg.checkpoint_mirror and (is_primary or ckpt_rank is not None)
            and cfg.checkpoint_every > 0):
        wait_for_mirror(cfg, rank=ckpt_rank)

    # No rank may start tearing down while another is still in a collective; see
    # DistributedSynchronizer.wait_for_all_ranks. Outside the timed region: this is
    # teardown, not training work.
    if is_distributed(cfg):
        dist_sync.wait_for_all_ranks()

    results["train_time_s"] = train_time_s
    results["wall_time_s"] = time.perf_counter() - wall_start
    if cfg.trackio and is_primary:
        import trackio

        trackio.finish()

    if return_debug_values:
        # FIXME remove once checkpoints are implemented, update the distributed
        # test using this
        if "model" in return_debug_values:
            # raw_model, so parameter names are wrapper-free regardless of mode
            results["model"] = raw_model

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
    cfg.local_rank = int(os.environ.get("LOCAL_RANK", cfg.local_rank))
    try:
        if is_distributed(cfg):
            setup_distributed(cfg.world_size, cfg.rank, resolve_device(cfg.device, cfg.local_rank),
                              distributed_backend=cfg.distributed_backend)
        results = train(cfg)
        print(results)
    finally:
        if is_distributed(cfg):
            cleanup_distributed()


if __name__ == "__main__":
    main()

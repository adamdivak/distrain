"""Training-loop tests, including a tiny end-to-end run on CPU."""

from __future__ import annotations

import dataclasses
import glob
import os
from collections import OrderedDict
from unittest import mock

import pytest
import torch
from torch import nn

from distrain.data import DataLoader, ShardingPlan, TokenStream
from distrain.distributed_synchronizer import DistributedSynchronizer
from distrain.model import GPT, GPTConfig
from distrain.train import (
    TrainConfig,
    cleanup_distributed,
    is_distributed,
    resolve_device,
    setup_distributed,
    to_device,
    train,
)
from helpers import tiny_train_config

distributed_modes = ["ddp_naive", "ddp_bucketed", "ddp_interleaved"]

def _device_available(device: str) -> bool:
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return torch.backends.mps.is_available()
    return device == "cpu"

def get_tmp_gradient_fn(cfg):
    gradient_fn = f"grads_world_size_{cfg.world_size}_rank_{cfg.rank}_model.pt"
    return gradient_fn

def get_tmp_model_params_fn(cfg):
    model_params_fn = f"model_params_world_size_{cfg.world_size}_rank_{cfg.rank}_model.pt"
    return model_params_fn

def spawn_mp(cfg: TrainConfig, fn, tmp_path):
    # Spawn instead of using torchrun, as this keeps new processes in our process tree
    # without having to go through the shell and all that
    torch.multiprocessing.spawn(fn, (cfg, tmp_path), nprocs=cfg.world_size)

def calc_backward_save_grad(rank, cfg: TrainConfig, tmp_path):
    try:
        cfg.rank = rank # update rank that we got from spawn
        cfg.local_rank = int(os.environ.get("LOCAL_RANK", cfg.local_rank))

        setup_distributed(cfg.world_size, cfg.rank, 
                          # different rendezvous file for each world size, as we'll be running
                          # multiple of these from the same unit test, where the tmp_path is identical,
                          # and we want to avoid them having conflicts
                          init_method=f"file://{tmp_path}/rendezvous_world_size{cfg.world_size}",
                          device=resolve_device(cfg.device, cfg.local_rank))

        device = resolve_device(cfg.device, cfg.local_rank)
        torch.manual_seed(cfg.seed + cfg.rank)
        if device == "cuda":
            torch.cuda.manual_seed(cfg.seed + cfg.rank)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        train_paths = sorted(glob.glob(cfg.train_glob))
        if not train_paths:
            raise FileNotFoundError(
                f"no shards matched {cfg.train_glob!r}. Generate synthetic data with "
                f"scripts/make_synthetic_shards.py, or fetch FineWeb per reference/PROVENANCE.md"
            )
        
        train_stream = TokenStream(train_paths)
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

        # optimizer = model.configure_optimizer(
        #     cfg.weight_decay, cfg.learning_rate, cfg.betas, fused=(device == "cuda")
        # )
        
        if is_distributed(cfg):
            dist_sync = DistributedSynchronizer(
                model, cfg.distributed_mode, cfg.world_size, cfg.ddp_bucket_size)

        # Make exactly two steps, as there might be some state in some communication
        # patterns, so we want to test that the second one works as well
        for step in range(2):  # cfg.max_steps
            if is_distributed(cfg):
                dist_sync.set_last_iteration()
            # optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            # for accum in range(cfg.grad_accum_steps):
            xb, yb = loader.microbatch(step=step, accum_idx=0)
            x, y = to_device(xb, device), to_device(yb, device)
            _, loss = model(x, y)
            loss.backward()

            if is_distributed(cfg):
                dist_sync.finalize_gradients()
            # optimizer.step()

        # Save all gradients and model parameters to a temp file, so we can
        # compare them in the test itself
        # Save gradients
        gradient_fn = get_tmp_gradient_fn(cfg)
        grads = OrderedDict()
        for param_name, param in model.named_parameters():
            grads[param_name] = param.grad
        torch.save(grads, (tmp_path / gradient_fn))
    finally:
        cleanup_distributed()

def init_model_save_params(rank, cfg: TrainConfig, tmp_path):
    try:
        cfg.rank = rank # update rank that we got from spawn
        cfg.local_rank = int(os.environ.get("LOCAL_RANK", cfg.local_rank))

        setup_distributed(cfg.world_size, cfg.rank, 
                          # different rendezvous file for each world size, as we'll be running
                          # multiple of these from the same unit test, where the tmp_path is identical,
                          # and we want to avoid them having conflicts
                          init_method=f"file://{tmp_path}/rendezvous_world_size{cfg.world_size}",
                          device=resolve_device(cfg.device))

        torch.manual_seed(cfg.seed + cfg.rank)

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
        )

        if is_distributed(cfg):
            # The constructor makes sure all ranks have the same model
            DistributedSynchronizer(
                model, cfg.distributed_mode, cfg.world_size, cfg.ddp_bucket_size)

        # Save parameters
        model_params_fn = get_tmp_model_params_fn(cfg)
        torch.save(OrderedDict(model.named_parameters()), (tmp_path / model_params_fn))
    finally:
        cleanup_distributed()

def broadcast_scalar_save_result(rank, cfg: TrainConfig, tmp_path):
    try:
        cfg.rank = rank # update rank that we got from spawn
        cfg.local_rank = int(os.environ.get("LOCAL_RANK", cfg.local_rank))

        setup_distributed(cfg.world_size, cfg.rank,
                          init_method=f"file://{tmp_path}/rendezvous_world_size{cfg.world_size}",
                          device=resolve_device(cfg.device, cfg.local_rank))

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
        )
        dist_sync = DistributedSynchronizer(
            model, cfg.distributed_mode, cfg.world_size, cfg.ddp_bucket_size)

        # Stand in for the val loss: only rank 0 has the real one, every other rank
        # holds a different placeholder, so a broadcast that does nothing is visible
        local_value = 3.14159 if cfg.rank == 0 else float(cfg.rank)
        result = dist_sync.broadcast_scalar(local_value)

        torch.save(result, (tmp_path / f"broadcast_rank_{cfg.rank}.pt"))
    finally:
        cleanup_distributed()

def train_save_params(rank, cfg: TrainConfig, tmp_path):
    try:
        cfg.rank = rank # update rank that we got from spaw
        cfg.local_rank = int(os.environ.get("LOCAL_RANK", cfg.local_rank))

        setup_distributed(cfg.world_size, cfg.rank,
                          # different rendezvous file for each world size, as we'll be running
                            # multiple of these from the same unit test, where the tmp_path is identical,
                            # and we want to avoid them having conflicts
                            init_method=f"file://{tmp_path}/rendezvous_world_size{cfg.world_size}",
                            device=resolve_device(cfg.device, cfg.local_rank))

        # FIXME read model from checkpoints instead
        results = train(cfg, return_debug_values=["model"])

        # Save parameters
        model_params_fn = get_tmp_model_params_fn(cfg)
        torch.save(OrderedDict(results["model"].named_parameters()), (tmp_path / model_params_fn))
    finally:
        cleanup_distributed()


def _observe_launches(dist_sync) -> dict:
    """Read what the synchronizer launched, by bucket index, in launch order.

    `async_handles` is appended to at launch, and every wait happens later in
    `finalize_gradients`, so calling this between backward and finalize gives exactly
    the collectives that were issued *during* backward and are still in flight.
    Modes 1 and 2 have no `async_handles` at all, hence the getattr defaults.
    """
    handles = getattr(dist_sync, "async_handles", [])
    buckets = getattr(dist_sync, "buckets", [])
    bucket_index = {id(bucket): i for i, bucket in enumerate(buckets)}
    return {
        "n_buckets": len(buckets),
        "launched_during_backward": len(handles),
        # gloo returns None from all_reduce when async_op=False, so this is what
        # catches async_op being dropped -- the gradients would still be correct
        "all_handles_async": all(handle is not None for handle, *_ in handles),
        "launch_order": [bucket_index[id(entry[3])] for entry in handles],
    }


def record_overlap(rank, cfg: TrainConfig, tmp_path):
    try:
        cfg.rank = rank
        cfg.local_rank = int(os.environ.get("LOCAL_RANK", cfg.local_rank))
        setup_distributed(cfg.world_size, cfg.rank,
                          init_method=f"file://{tmp_path}/rendezvous_world_size{cfg.world_size}",
                          device=resolve_device(cfg.device, cfg.local_rank))
        device = resolve_device(cfg.device, cfg.local_rank)
        torch.manual_seed(cfg.seed + cfg.rank)

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

        # Registered before the synchronizer, so for any given parameter this fires
        # ahead of the synchronizer's own hook: it reads how many collectives are in
        # flight *just before* that parameter's bucket becomes eligible to launch.
        # The last reading therefore describes the moment the final gradient of the
        # backward pass was produced -- the sharpest point at which to ask whether
        # communication was already overlapping computation.
        holder = {}
        inflight = []

        def probe(param):
            if "sync" in holder:
                inflight.append(len(getattr(holder["sync"], "async_handles", [])))

        for param in model.parameters():
            param.register_post_accumulate_grad_hook(probe)

        dist_sync = DistributedSynchronizer(
            model, cfg.distributed_mode, cfg.world_size, cfg.ddp_bucket_size)
        holder["sync"] = dist_sync
        dist_sync.set_last_iteration()

        # Random tokens are fine: this is a test about when collectives are issued,
        # not about what they carry. Both ranks use identical shapes, which is all
        # that collective matching depends on.
        x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len), device=device)
        _, loss = model(x, x)
        loss.backward()

        observed = _observe_launches(dist_sync)
        observed["inflight_at_last_grad"] = inflight[-1] if inflight else None

        dist_sync.finalize_gradients()
        torch.save(observed, tmp_path / f"overlap_rank_{cfg.rank}.pt")
    finally:
        cleanup_distributed()


def record_launch_order(rank, cfg: TrainConfig, tmp_path):
    try:
        cfg.rank = rank
        cfg.local_rank = int(os.environ.get("LOCAL_RANK", cfg.local_rank))
        setup_distributed(cfg.world_size, cfg.rank,
                          init_method=f"file://{tmp_path}/rendezvous_world_size{cfg.world_size}",
                          device=resolve_device(cfg.device, cfg.local_rank))
        device = resolve_device(cfg.device, cfg.local_rank)
        torch.manual_seed(cfg.seed + cfg.rank)

        model = ReversedExecutionModel().to(device)
        dist_sync = DistributedSynchronizer(
            model, cfg.distributed_mode, cfg.world_size, cfg.ddp_bucket_size)
        dist_sync.set_last_iteration()

        model(torch.randn(4, 32, device=device)).sum().backward()

        observed = _observe_launches(dist_sync)
        dist_sync.finalize_gradients()
        torch.save(observed, tmp_path / f"launch_order_rank_{cfg.rank}.pt")
    finally:
        cleanup_distributed()


class TestBackwardOverlap:
    """Mode 3's only claim over mode 2 is *when* it issues its collectives.

    Every way that claim can break -- `async_op` dropped, the launch sliding into
    `finalize_gradients`, a `wait()` landing inside the hook -- leaves the gradients
    numerically perfect, so every other test in this file still passes. These assert
    the launch schedule directly instead, with no timing, which would mean nothing
    on gloo/CPU anyway.
    """

    def test_interleaved_launches_during_backward(self, tiny_data, tmp_path):
        cfg = tiny_train_config(tiny_data, world_size=2, distributed_mode="ddp_interleaved")
        spawn_mp(cfg, record_overlap, tmp_path)
        observed = torch.load(tmp_path / "overlap_rank_0.pt")

        # With one bucket there is nothing to overlap and the assertions below hold
        # trivially, so guard rather than let the test quietly stop testing anything
        assert observed["n_buckets"] >= 3, (
            f"only {observed['n_buckets']} bucket(s); lower ddp_bucket_size or this "
            f"test proves nothing about overlap"
        )
        assert observed["launched_during_backward"] == observed["n_buckets"]
        assert observed["all_handles_async"]
        # the point of the mode: when the last gradient of the pass was produced,
        # every other bucket was already in flight
        assert observed["inflight_at_last_grad"] == observed["n_buckets"] - 1

    def test_bucketed_launches_nothing_during_backward(self, tiny_data, tmp_path):
        """The half that stops the test above from being vacuous.

        If mode 2 also showed collectives in flight during backward, the assertions
        above would be measuring something other than overlap.
        """
        cfg = tiny_train_config(tiny_data, world_size=2, distributed_mode="ddp_bucketed")
        spawn_mp(cfg, record_overlap, tmp_path)
        observed = torch.load(tmp_path / "overlap_rank_0.pt")

        assert observed["launched_during_backward"] == 0
        assert observed["inflight_at_last_grad"] == 0


@pytest.mark.parametrize("distributed_mode", distributed_modes)
class TestInitialization:
    def test_distributed_has_same_params_after_init(self, tiny_data, tmp_path, distributed_mode):
        cfg_ws1 = tiny_train_config(tiny_data, max_steps=0)
        cfg_ws2 = tiny_train_config(tiny_data, max_steps=0, world_size=2, distributed_mode = distributed_mode)
        spawn_mp(cfg_ws1, init_model_save_params, tmp_path)
        spawn_mp(cfg_ws2, init_model_save_params, tmp_path)
        
        model_params_fn_ws1 = get_tmp_model_params_fn(cfg_ws1)
        model_params_ws1 = torch.load(tmp_path / model_params_fn_ws1)

        model_params_fn_ws2 = get_tmp_model_params_fn(cfg_ws2)
        model_params_ws2 = torch.load(tmp_path / model_params_fn_ws2)

        assert model_params_ws1.keys() == model_params_ws2.keys()
        for key in model_params_ws1:
            torch.testing.assert_close(model_params_ws1[key], model_params_ws2[key],
                                        msg=lambda input_msg, key=key: f"Mismatch in values of {key}. \n" + input_msg)

    def test_all_ranks_have_same_params_after_init(self, tiny_data, tmp_path, distributed_mode):
        cfg_ws2_r0 = tiny_train_config(tiny_data, max_steps=0, world_size=2, distributed_mode = distributed_mode)
        spawn_mp(cfg_ws2_r0, init_model_save_params, tmp_path)

        cfg_ws2_r1 = dataclasses.replace(cfg_ws2_r0, rank=1)

        model_params_fn_ws2_r0 = get_tmp_model_params_fn(cfg_ws2_r0)
        model_params_ws2_r0 = torch.load(tmp_path / model_params_fn_ws2_r0)

        model_params_fn_ws2_r1 = get_tmp_model_params_fn(cfg_ws2_r1)
        model_params_ws2_r1 = torch.load(tmp_path / model_params_fn_ws2_r1)

        assert model_params_ws2_r0.keys() == model_params_ws2_r1.keys()
        for key in model_params_ws2_r0:
            torch.testing.assert_close(model_params_ws2_r0[key], model_params_ws2_r1[key],
                                        msg=lambda input_msg, key=key: f"Mismatch in values of {key}. \n" + input_msg)


@pytest.mark.parametrize("distributed_mode", distributed_modes)
class TestGradients:
    def test_gradient_is_equal_for_larger_world_size(self, tiny_data, tmp_path, distributed_mode):
        cfg_ws1 = tiny_train_config(tiny_data)
        cfg_ws2_r0 = tiny_train_config(tiny_data, world_size=2, distributed_mode = distributed_mode)
        spawn_mp(cfg_ws1, calc_backward_save_grad, tmp_path)
        spawn_mp(cfg_ws2_r0, calc_backward_save_grad, tmp_path)
        
        gradient_fn_ws1 = get_tmp_gradient_fn(cfg_ws1)
        gradient_ws1 = torch.load(tmp_path / gradient_fn_ws1)

        gradient_fn_ws2_r0 = get_tmp_gradient_fn(cfg_ws2_r0)
        gradient_ws2_r0 = torch.load(tmp_path / gradient_fn_ws2_r0)

        cfg_ws2_r1 = dataclasses.replace(cfg_ws2_r0, rank=1)
        gradient_fn_ws2_r1 = get_tmp_gradient_fn(cfg_ws2_r1)
        gradient_ws2_r1 = torch.load(tmp_path / gradient_fn_ws2_r1)

        # Test that results from running with world_size 1 
        # equal resuls from running with world_size 2,
        # i.e. that the reduce computed the right value
        assert gradient_ws1.keys() == gradient_ws2_r0.keys()
        for key in gradient_ws1:
            torch.testing.assert_close(gradient_ws1[key], gradient_ws2_r0[key],
                                       msg=lambda input_msg, key=key: f"Mismatch in values of {key}. \n" + input_msg)

        # Test that gradients on rank 0 and rank 1 are identical
        # in the end. This tests that the final value is correctly
        # distributed to all ranks, so they can continue with this
        assert gradient_ws2_r0.keys() == gradient_ws2_r1.keys()
        for key in gradient_ws2_r0:
            torch.testing.assert_close(gradient_ws2_r0[key], gradient_ws2_r1[key],
                                        msg=lambda input_msg, key=key: f"Mismatch in values of {key}. \n" + input_msg)


@pytest.mark.parametrize("distributed_mode", distributed_modes)
class TestScalarBroadcast:
    """Only rank 0 evaluates, so every other rank's val loss arrives by broadcast.

    Each rank independently compares that value against the target loss, so anything
    less than bit-identical agreement could make ranks disagree about which step first
    crossed 3.28 -- a disagreement that would surface as an inconsistent result rather
    than as a crash.
    """

    def test_all_ranks_get_rank_0_value(self, tiny_data, tmp_path, distributed_mode):
        cfg = tiny_train_config(tiny_data, world_size=2, distributed_mode = distributed_mode)
        spawn_mp(cfg, broadcast_scalar_save_result, tmp_path)

        results = [torch.load(tmp_path / f"broadcast_rank_{rank}.pt")
                   for rank in range(cfg.world_size)]

        # exact equality, not assert_close: ranks must agree bit for bit, and rank 0
        # must return what it read back out of the tensor rather than what it put in
        assert results[0] == results[1]
        assert results[0] == torch.tensor([3.14159], dtype=torch.float32).item()

    def test_tensor_is_allocated_on_the_model_device(self, tiny_data, distributed_mode):
        """NCCL cannot broadcast a CPU tensor, and no gloo test would ever notice.

        Skips where the only device is the CPU, because there the assertion holds
        whether or not the code asks for the model's device -- and a test that cannot
        fail is worse than no test.
        """
        device = next((d for d in ("cuda", "mps") if _device_available(d)), None)
        if device is None:
            pytest.skip("no non-CPU device; the assertion would be vacuous")

        cfg = tiny_train_config(tiny_data, world_size=2, distributed_mode = distributed_mode)
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

        # patching broadcast stands in for a process group: this test is about which
        # device the tensor is built on, which is decided before any collective runs
        broadcast_devices = []
        def record(tensor, src=0):
            broadcast_devices.append(tensor.device)

        with mock.patch("torch.distributed.broadcast", side_effect=record):
            dist_sync = DistributedSynchronizer(
                model, cfg.distributed_mode, cfg.world_size, cfg.ddp_bucket_size)
            dist_sync.broadcast_scalar(3.14159)

        assert broadcast_devices[-1] == next(model.parameters()).device


@pytest.mark.parametrize("distributed_mode", distributed_modes)
class TestEndToEnd:
    def test_params_equal_for_larger_world_size(self, tiny_data, tmp_path, distributed_mode):
        cfg_ws1 = tiny_train_config(tiny_data)
        cfg_ws2 = tiny_train_config(tiny_data, world_size=2, distributed_mode = distributed_mode)
        spawn_mp(cfg_ws1, train_save_params, tmp_path)
        spawn_mp(cfg_ws2, train_save_params, tmp_path)
        
        model_params_fn_ws1 = get_tmp_model_params_fn(cfg_ws1)
        model_params_ws1 = torch.load(tmp_path / model_params_fn_ws1)

        model_params_fn_ws2 = get_tmp_model_params_fn(cfg_ws2)
        model_params_ws2 = torch.load(tmp_path / model_params_fn_ws2)

        assert model_params_ws1.keys() == model_params_ws2.keys()
        for key in model_params_ws1:
            torch.testing.assert_close(model_params_ws1[key], model_params_ws2[key],
                                       msg=lambda input_msg, key=key: f"Mismatch in values of {key}. \n" + input_msg)

    def test_params_equal_for_larger_world_size_with_grad_accum(self, tiny_data, tmp_path, distributed_mode):
        cfg_ws1 = tiny_train_config(tiny_data, grad_accum_steps=4)
        cfg_ws2 = tiny_train_config(tiny_data, grad_accum_steps=2, world_size=2, distributed_mode = distributed_mode)
        spawn_mp(cfg_ws1, train_save_params, tmp_path)
        spawn_mp(cfg_ws2, train_save_params, tmp_path)
        
        model_params_fn_ws1 = get_tmp_model_params_fn(cfg_ws1)
        model_params_ws1 = torch.load(tmp_path / model_params_fn_ws1)

        model_params_fn_ws2 = get_tmp_model_params_fn(cfg_ws2)
        model_params_ws2 = torch.load(tmp_path / model_params_fn_ws2)

        assert model_params_ws1.keys() == model_params_ws2.keys()
        for key in model_params_ws1:
            torch.testing.assert_close(model_params_ws1[key], model_params_ws2[key],
                                        msg=lambda input_msg, key=key: f"Mismatch in values of {key}. \n" + input_msg)

"""Training-loop tests, including a tiny end-to-end run on CPU."""

from __future__ import annotations

import dataclasses
import glob
from collections import OrderedDict

import torch

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

        setup_distributed(cfg.world_size, cfg.rank, 
                          # different rendezvous file for each world size, as we'll be running
                          # multiple of these from the same unit test, where the tmp_path is identical,
                          # and we want to avoid them having conflicts
                          init_method=f"file://{tmp_path}/rendezvous_world_size{cfg.world_size}",
                          device=resolve_device(cfg.device))

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
            dist_sync = DistributedSynchronizer(model, cfg.distributed_mode, cfg.world_size)

        # Make exactly two steps, as there might be some state in some communication
        # patterns, so we want to test that the second one works as well
        for step in range(2):  # cfg.max_steps
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
            DistributedSynchronizer(model, cfg.distributed_mode, cfg.world_size)

        # Save parameters
        model_params_fn = get_tmp_model_params_fn(cfg)
        torch.save(OrderedDict(model.named_parameters()), (tmp_path / model_params_fn))
    finally:
        cleanup_distributed()

def train_save_params(rank, cfg: TrainConfig, tmp_path):
    try:
        cfg.rank = rank # update rank that we got from spaw

        setup_distributed(cfg.world_size, cfg.rank,
                          # different rendezvous file for each world size, as we'll be running
                            # multiple of these from the same unit test, where the tmp_path is identical,
                            # and we want to avoid them having conflicts
                            init_method=f"file://{tmp_path}/rendezvous_world_size{cfg.world_size}",
                            device=resolve_device(cfg.device))

        # FIXME read model from checkpoints instead
        results = train(cfg, return_debug_values=["model"])

        # Save parameters
        model_params_fn = get_tmp_model_params_fn(cfg)
        torch.save(OrderedDict(results["model"].named_parameters()), (tmp_path / model_params_fn))
    finally:
        cleanup_distributed()

class TestInitialization:
    def test_distributed_has_same_params_after_init(self, tiny_data, tmp_path):
        cfg_ws1 = tiny_train_config(tiny_data, max_steps=0)
        cfg_ws2 = tiny_train_config(tiny_data, max_steps=0, world_size=2, distributed_mode = "ddp")
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

    def test_all_ranks_have_same_params_after_init(self, tiny_data, tmp_path):
        cfg_ws2_r0 = tiny_train_config(tiny_data, max_steps=0, world_size=2, distributed_mode = "ddp")
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


class TestGradients:
    def test_gradient_is_equal_for_larger_world_size(self, tiny_data, tmp_path):
        cfg_ws1 = tiny_train_config(tiny_data)
        cfg_ws2_r0 = tiny_train_config(tiny_data, world_size=2, distributed_mode = "ddp")
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

    def test_params_equal_for_larger_world_size(self, tiny_data, tmp_path):
        cfg_ws1 = tiny_train_config(tiny_data)
        cfg_ws2 = tiny_train_config(tiny_data, world_size=2, distributed_mode = "ddp")
        spawn_mp(cfg_ws1, train_save_params,  tmp_path)
        spawn_mp(cfg_ws2, train_save_params, tmp_path)
        
        model_params_fn_ws1 = get_tmp_model_params_fn(cfg_ws1)
        model_params_ws1 = torch.load(tmp_path / model_params_fn_ws1)

        model_params_fn_ws2 = get_tmp_model_params_fn(cfg_ws2)
        model_params_ws2 = torch.load(tmp_path / model_params_fn_ws2)

        assert model_params_ws1.keys() == model_params_ws2.keys()
        for key in model_params_ws1:
            torch.testing.assert_close(model_params_ws1[key], model_params_ws2[key],
                                       msg=lambda input_msg, key=key: f"Mismatch in values of {key}. \n" + input_msg)

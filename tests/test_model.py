"""Model correctness tests. All run on CPU."""

from __future__ import annotations

import math

import pytest
import torch

from distrain.model import GPT, GPTConfig


def tiny_config(**overrides):
    base = {"block_size": 32, "vocab_size": 128, "n_layer": 2, "n_head": 4,
            "n_embd": 64, "dropout": 0.0}
    base.update(overrides)
    return GPTConfig(**base)


class TestConfig:
    def test_head_dim(self):
        assert tiny_config().head_dim == 16

    def test_rejects_indivisible_head_count(self):
        with pytest.raises(ValueError, match="not divisible"):
            _ = tiny_config(n_embd=65, n_head=4).head_dim


class TestForward:
    def test_output_shapes(self):
        cfg = tiny_config()
        model = GPT(cfg)
        idx = torch.randint(0, cfg.vocab_size, (3, 16))
        logits, loss = model(idx, idx)
        assert logits.shape == (3, 16, cfg.vocab_size)
        assert loss.ndim == 0

    def test_loss_is_none_without_targets(self):
        cfg = tiny_config()
        logits, loss = GPT(cfg)(torch.randint(0, cfg.vocab_size, (2, 8)))
        assert loss is None
        assert logits.shape == (2, 8, cfg.vocab_size)

    def test_initial_loss_is_uniform_entropy(self):
        """An untrained model should be maximally uncertain: loss ~= ln(vocab_size).

        Targets must be *shifted*. Scoring the current token instead of the next one
        gives a much lower initial loss, because weight tying plus the residual stream
        makes the identity mapping nearly free at init.
        """
        cfg = tiny_config(vocab_size=512)
        torch.manual_seed(0)
        model = GPT(cfg)
        idx = torch.randint(0, cfg.vocab_size, (8, 33))
        _, loss = model(idx[:, :-1], idx[:, 1:])
        assert loss.item() == pytest.approx(math.log(cfg.vocab_size), rel=0.05)

    def test_rejects_sequence_longer_than_block_size(self):
        cfg = tiny_config(block_size=16)
        with pytest.raises(ValueError, match="exceeds block_size"):
            GPT(cfg)(torch.zeros(1, 17, dtype=torch.long))

    def test_accepts_uint16_derived_int64_targets(self):
        """Targets arrive from the loader as int64 cast from uint16 shards."""
        cfg = tiny_config()
        model = GPT(cfg)
        idx = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.int64)
        _, loss = model(idx, idx)
        assert torch.isfinite(loss)


class TestCausality:
    def test_future_tokens_do_not_affect_past_logits(self):
        """If the causal mask were wrong, every scaling result would be on a broken model."""
        cfg = tiny_config()
        torch.manual_seed(0)
        model = GPT(cfg).eval()
        a = torch.randint(0, cfg.vocab_size, (1, 16))
        b = a.clone()
        b[0, -1] = (b[0, -1] + 1) % cfg.vocab_size  # perturb only the final token
        with torch.no_grad():
            logits_a, _ = model(a)
            logits_b, _ = model(b)
        torch.testing.assert_close(logits_a[:, :-1], logits_b[:, :-1])


class TestParameters:
    def test_weights_are_tied(self):
        model = GPT(tiny_config())
        assert model.transformer.wte.weight is model.lm_head.weight

    def test_num_params_excludes_position_embeddings(self):
        cfg = tiny_config()
        model = GPT(cfg)
        assert model.num_params(False) - model.num_params(True) == cfg.block_size * cfg.n_embd

    def test_gpt2_small_is_124m(self):
        """The Track A model, at the size the brief and the benchmark assume."""
        model = GPT(GPTConfig())
        assert 123e6 < model.num_params(non_embedding=True) < 125e6

    def test_residual_projections_get_scaled_init(self):
        cfg = tiny_config(n_layer=8)
        model = GPT(cfg)
        expected = 0.02 / math.sqrt(2 * cfg.n_layer)
        proj = model.transformer.h[0].attn.c_proj.weight
        other = model.transformer.h[0].attn.c_attn.weight
        assert proj.std().item() == pytest.approx(expected, rel=0.15)
        assert other.std().item() == pytest.approx(0.02, rel=0.15)


class TestDeterminism:
    def test_same_seed_gives_identical_init(self):
        torch.manual_seed(7)
        a = GPT(tiny_config())
        torch.manual_seed(7)
        b = GPT(tiny_config())
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa, pb)

    def test_different_seed_gives_different_init(self):
        torch.manual_seed(7)
        a = GPT(tiny_config())
        torch.manual_seed(8)
        b = GPT(tiny_config())
        assert not torch.equal(a.transformer.h[0].mlp.c_fc.weight,
                               b.transformer.h[0].mlp.c_fc.weight)


class TestOptimizer:
    def test_only_matrices_are_weight_decayed(self):
        model = GPT(tiny_config(bias=True))
        opt = model.configure_optimizer(0.1, 6e-4, (0.9, 0.95), fused=False)
        decay, nodecay = opt.param_groups
        assert decay["weight_decay"] == 0.1 and nodecay["weight_decay"] == 0.0
        assert all(p.dim() >= 2 for p in decay["params"])
        assert all(p.dim() < 2 for p in nodecay["params"])

    def test_every_parameter_is_assigned_to_a_group(self):
        model = GPT(tiny_config(bias=True))
        opt = model.configure_optimizer(0.1, 6e-4, (0.9, 0.95), fused=False)
        in_groups = sum(p.numel() for g in opt.param_groups for p in g["params"])
        # weight tying means wte and lm_head are one tensor; parameters() dedupes it
        assert in_groups == sum(p.numel() for p in model.parameters())

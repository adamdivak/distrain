"""Model correctness tests. All run on CPU."""

from __future__ import annotations

import math

import pytest
import torch

from distrain.model import GPT, GPTConfig, Rotary


def tiny_config(**overrides):
    base = {"block_size": 32, "vocab_size": 128, "n_layer": 2, "n_head": 4,
            "n_embd": 64, "dropout": 0.0}
    base.update(overrides)
    return GPTConfig(**base)


class TestZeroInit:
    """modded-nanogpt zero-init: residual projections and the untied head start at
    zero, so every block is the identity at init.

    These assert the state of the *finished* model on purpose: zeroing inside a
    submodule constructor is silently undone by GPT's apply(_init_weights), so a
    constructor-time check would pass while the model ships gaussian weights.
    """

    def test_residual_projections_start_at_zero(self):
        for name, p in GPT(tiny_config()).named_parameters():
            if name.endswith("c_proj.weight"):
                assert torch.all(p == 0), f"{name} survived init non-zero"

    def test_untied_head_is_zero_and_embedding_is_not(self):
        model = GPT(tiny_config(use_weight_tying=False))
        assert torch.all(model.lm_head.weight == 0)
        assert model.transformer.wte.weight.abs().sum() > 0

    def test_tied_head_is_not_zeroed(self):
        """Zeroing a tied head would zero wte with it and kill the embedding."""
        model = GPT(tiny_config(use_weight_tying=True))
        assert model.transformer.wte.weight is model.lm_head.weight
        assert model.lm_head.weight.abs().sum() > 0

    def test_non_residual_linears_keep_gaussian_init(self):
        """Guards the name filter: only c_proj (and the untied head) get zeroed;
        everything else keeps its N(0, 0.02)."""
        block = GPT(tiny_config()).transformer.h[0]
        assert block.attn.c_attn.weight.std().item() == pytest.approx(0.02, rel=0.15)
        assert block.mlp.c_fc.weight.std().item() == pytest.approx(0.02, rel=0.15)


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


class TestRotary:
    """The properties RoPE is used for, asserted directly on the module."""

    def test_rejects_odd_dim(self):
        with pytest.raises(ValueError, match="must be even"):
            Rotary(dim=15, max_seq_len=8)

    def test_rotation_preserves_norm(self):
        """Rotations are orthogonal: attention logit scale must not change with position."""
        torch.manual_seed(0)
        x = torch.randn(2, 4, 32, 16)
        y = Rotary(16, 32)(x)
        torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))

    def test_position_zero_is_identity(self):
        torch.manual_seed(0)
        x = torch.randn(1, 2, 8, 16)
        y = Rotary(16, 8)(x)
        torch.testing.assert_close(y[:, :, 0], x[:, :, 0])

    def test_scores_depend_only_on_relative_position(self):
        """The point of RoPE: q_i . k_j is a function of i - j, not of i and j.

        Broadcast one q and one k to every position, rotate, and check the score
        matrix is constant along diagonals (equal at (i, j) and (i+s, j+s)).
        """
        torch.manual_seed(0)
        T, hd = 16, 16
        rotary = Rotary(hd, T)
        q = rotary(torch.randn(hd).expand(1, 1, T, hd))
        k = rotary(torch.randn(hd).expand(1, 1, T, hd))
        scores = (q @ k.transpose(-2, -1)).squeeze()
        for shift in (1, 5):
            torch.testing.assert_close(
                scores[: T - shift, : T - shift],
                scores[shift:, shift:],
            )

    def test_model_output_depends_on_prefix_order(self):
        """With wpe gone, rotary is the only source of position. In a 1-layer model
        without it, the last position's logits are exactly invariant under permuting
        the prefix: the (k, v) set is the same and softmax reweighting is
        permutation-invariant. So this fails if rotary is dropped.

        Zero-init makes a fresh model output all-zero logits, so the zeroed
        weights are re-randomized first.
        """
        cfg = tiny_config(n_layer=1)
        torch.manual_seed(0)
        model = GPT(cfg).eval()
        with torch.no_grad():
            for _, p in model.named_parameters():
                if torch.all(p == 0):
                    p.normal_(0.0, 0.02)
        a = torch.arange(1, 17).unsqueeze(0)  # distinct tokens
        b = a.clone()
        b[0, :-1] = b[0, :-1].flip(0)  # permute the prefix, keep the final token
        with torch.no_grad():
            logits_a, _ = model(a)
            logits_b, _ = model(b)
        assert not torch.allclose(logits_a[0, -1], logits_b[0, -1])


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
        model = GPT(tiny_config(use_weight_tying=True))
        assert model.transformer.wte.weight is model.lm_head.weight

    def test_no_positional_parameters(self):
        """Rotary replaced wpe: position is encoded by buffers, not learned weights."""
        model = GPT(tiny_config())
        assert not any("wpe" in name for name, _ in model.named_parameters())
        assert not any(b.requires_grad for b in model.buffers())

    def test_gpt2_small_is_124m(self):
        """The Track A model, at the size the brief and the benchmark assume."""
        model = GPT(GPTConfig(use_weight_tying=True))
        assert 123e6 < model.num_params() < 125e6


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

"""Tests for the FLOPs/MFU conventions.

These guard `docs/decisions.md` section 3. The failure mode being prevented is not a
crash -- it is a plausible-looking MFU number computed against the wrong convention,
which is indistinguishable from a real result.
"""

from __future__ import annotations

import pytest

from distrain.mfu import FlopsCounter, counter_for, peak_bf16_flops
from distrain.model import GPT, GPTConfig


def gpt2_small_counter(seq_len=1024):
    return FlopsCounter(num_params=124_000_000, n_layer=12, n_head=12, head_dim=64,
                        seq_len=seq_len)


class TestPeakLookup:
    @pytest.mark.parametrize(
        "name,expected_tflops",
        [
            ("NVIDIA H100 80GB HBM3", 989.0),
            ("NVIDIA H100 PCIe", 756.0),
            ("NVIDIA H100 NVL", 835.0),
            ("NVIDIA A100-SXM4-80GB", 312.0),
            ("NVIDIA A100-PCIE-40GB", 312.0),
            ("NVIDIA L40S", 181.0),
            ("NVIDIA GeForce RTX 3090", 35.6),
        ],
    )
    def test_known_devices(self, name, expected_tflops):
        assert peak_bf16_flops(name) == pytest.approx(expected_tflops * 1e12)

    def test_uses_dense_not_sparsity_figures(self):
        """The sparsity numbers are exactly 2x and would halve every reported MFU."""
        assert peak_bf16_flops("NVIDIA H100 80GB HBM3") != pytest.approx(1979e12)
        assert peak_bf16_flops("NVIDIA L40S") != pytest.approx(362e12)

    def test_3090_uses_fp32_accumulate_rate(self):
        """GeForce halves FP32-accumulate tensor throughput; 71 TFLOP/s is the wrong number."""
        assert peak_bf16_flops("NVIDIA GeForce RTX 3090") == pytest.approx(35.6e12)

    def test_unknown_device_raises(self):
        with pytest.raises(KeyError, match="no bf16 dense peak recorded"):
            peak_bf16_flops("NVIDIA Tesla K80")

    def test_specific_variants_win_over_family(self):
        assert peak_bf16_flops("NVIDIA H100 PCIe") != peak_bf16_flops("NVIDIA H100 80GB HBM3")


class TestFlopsFormula:
    def test_includes_attention_term(self):
        """The whole point of PaLM-style over 6ND."""
        c = gpt2_small_counter()
        bare_6nd = 6 * c.num_params
        assert c.model_flops_per_token > bare_6nd
        attention = 12 * 12 * 12 * 64 * 1024
        assert c.model_flops_per_token == pytest.approx(bare_6nd + attention)

    def test_attention_term_grows_with_sequence_length(self):
        short, long = gpt2_small_counter(1024), gpt2_small_counter(4096)
        delta = long.model_flops_per_token - short.model_flops_per_token
        assert delta == pytest.approx(12 * 12 * 12 * 64 * (4096 - 1024))

    def test_attention_share_is_material_at_gpt2_small(self):
        """13% of total FLOPs at 124M/seq-1024 -- far too large to drop."""
        c = gpt2_small_counter()
        share = 1 - (6 * c.num_params) / c.model_flops_per_token
        assert share == pytest.approx(0.132, abs=0.005)


class TestRecompute:
    def test_no_checkpointing_matches_model_flops(self):
        c = gpt2_small_counter()
        assert c.hardware_flops_per_token(0.0) == pytest.approx(c.model_flops_per_token)

    def test_full_recompute_costs_one_third_more(self):
        c = gpt2_small_counter()
        ratio = c.hardware_flops_per_token(1.0) / c.model_flops_per_token
        assert ratio == pytest.approx(8 / 6, rel=1e-6)

    def test_rejects_fraction_out_of_range(self):
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            gpt2_small_counter().hardware_flops_per_token(1.5)


class TestUtilization:
    def test_mfu_is_fraction_of_peak(self):
        c = gpt2_small_counter()
        peak = 989e12
        # choose token count so exactly one second of peak work is required
        tokens = round(peak / c.model_flops_per_token)
        assert c.mfu(tokens, seconds=1.0, peak_flops=peak) == pytest.approx(1.0, rel=1e-6)
        assert c.mfu(tokens, seconds=2.0, peak_flops=peak) == pytest.approx(0.5, rel=1e-6)

    def test_hfu_equals_mfu_without_checkpointing(self):
        c = gpt2_small_counter()
        args = {"tokens": 10_000_000, "seconds": 1.5, "peak_flops": 989e12}
        assert c.hfu(recompute_fraction=0.0, **args) == pytest.approx(c.mfu(**args))

    def test_hfu_exceeds_mfu_with_checkpointing(self):
        """The distinction that makes 7B FSDP2 numbers honest."""
        c = gpt2_small_counter()
        args = {"tokens": 10_000_000, "seconds": 1.5, "peak_flops": 989e12}
        assert c.hfu(recompute_fraction=1.0, **args) == pytest.approx(c.mfu(**args) * 8 / 6)

    def test_rejects_nonpositive_time(self):
        with pytest.raises(ValueError, match="must be positive"):
            gpt2_small_counter().mfu(tokens=1, seconds=0.0, peak_flops=1e12)


class TestCounterForModel:
    def test_reads_shape_from_model(self):
        model = GPT(GPTConfig(n_layer=2, n_head=4, n_embd=128, block_size=64, vocab_size=256))
        c = counter_for(model, seq_len=64)
        assert (c.n_layer, c.n_head, c.head_dim, c.seq_len) == (2, 4, 32, 64)
        assert c.num_params == model.num_params(non_embedding=True)

    def test_param_count_excludes_position_embeddings_only(self):
        cfg = GPTConfig(n_layer=2, n_head=4, n_embd=128, block_size=64, vocab_size=256)
        model = GPT(cfg)
        wpe = cfg.block_size * cfg.n_embd
        assert model.num_params(True) == model.num_params(False) - wpe

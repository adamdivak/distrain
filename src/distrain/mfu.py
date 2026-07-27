"""FLOPs accounting, MFU and HFU.

Every throughput claim in this project routes through this module, so the conventions
are fixed here once (`docs/decisions.md` section 3) rather than re-derived per run:

- **Numerator**: PaLM-style `6N + 12*L*H*Q*T` FLOPs per token, i.e. *including* the
  attention term. That term is 13% of total FLOPs at 124M/seq-1024 -- dropping it, as
  bare `6ND` does, would understate MFU by more than most of the effects this study
  is trying to measure.
- **MFU vs HFU**: MFU counts only the FLOPs the model mathematically requires. HFU
  additionally counts activation recomputation, which FSDP2 at 7B will need. They
  differ by ~33% under full recompute and are widely conflated; both are reported.
- **Denominator**: bf16 *dense* peak. Never the 2:4-sparsity marketing number, which
  is 2x higher and would halve every reported MFU.

Unknown devices raise rather than fall back to a guess: a silently wrong denominator
is indistinguishable from a real efficiency finding.
"""

from __future__ import annotations

from dataclasses import dataclass

# bf16 dense tensor-core peak, TFLOP/s per GPU. Ordered: first substring match wins,
# so more specific variants must precede their families.
#
# The RTX 3090 figure is the one that surprises people: GeForce cards run tensor-core
# matmuls with FP32 accumulate at half rate, and PyTorch's bf16 matmul accumulates in
# FP32. 71 TFLOP/s is the FP16-accumulate number and does not apply here.
_PEAK_BF16_TFLOPS: tuple[tuple[str, float], ...] = (
    ("H100 PCIe", 756.0),
    ("H100 NVL", 835.0),
    ("H100", 989.0),  # SXM
    ("A100", 312.0),  # SXM and PCIe are the same
    ("L40S", 181.0),
    ("RTX 3090", 35.6),
)


def peak_bf16_flops(device_name: str) -> float:
    """bf16 dense peak in FLOP/s for a `torch.cuda.get_device_name()` string."""
    for pattern, tflops in _PEAK_BF16_TFLOPS:
        if pattern in device_name:
            return tflops * 1e12
    known = ", ".join(p for p, _ in _PEAK_BF16_TFLOPS)
    raise KeyError(
        f"no bf16 dense peak recorded for {device_name!r}. Add it to _PEAK_BF16_TFLOPS "
        f"using the dense (non-sparsity) figure from the vendor datasheet. Known: {known}"
    )


@dataclass(frozen=True)
class FlopsCounter:
    """Per-token FLOPs for one model shape.

    `num_params` is the non-embedding count -- position embeddings excluded, tied token
    embeddings included, matching `GPT.num_params()`.
    """

    num_params: int
    n_layer: int
    n_head: int
    head_dim: int
    seq_len: int

    @property
    def model_flops_per_token(self) -> float:
        """Forward + backward FLOPs the model mathematically requires, per token."""
        attention = 12 * self.n_layer * self.n_head * self.head_dim * self.seq_len
        return 6 * self.num_params + attention

    def hardware_flops_per_token(self, recompute_fraction: float = 0.0) -> float:
        """As above, plus recomputed forward passes under activation checkpointing.

        `recompute_fraction` is the share of the forward recomputed in the backward:
        0.0 for no checkpointing, 1.0 for full. Forward is 2N + 4LHQT per token, so a
        fully recomputed step costs 8N + 16LHQT instead of 6N + 12LHQT -- 33% more
        hardware work for identical model FLOPs.
        """
        if not 0.0 <= recompute_fraction <= 1.0:
            raise ValueError(f"recompute_fraction must be in [0, 1], got {recompute_fraction}")
        r = recompute_fraction
        attention_term = self.n_layer * self.n_head * self.head_dim * self.seq_len
        return (6 + 2 * r) * self.num_params + (12 + 4 * r) * attention_term

    def mfu(self, tokens: int, seconds: float, peak_flops: float) -> float:
        """Model FLOPs utilization: fraction of peak spent on required work."""
        if seconds <= 0:
            raise ValueError(f"seconds must be positive, got {seconds}")
        return self.model_flops_per_token * tokens / seconds / peak_flops

    def hfu(
        self, tokens: int, seconds: float, peak_flops: float, recompute_fraction: float = 0.0
    ) -> float:
        """Hardware FLOPs utilization: fraction of peak the GPU actually executed."""
        if seconds <= 0:
            raise ValueError(f"seconds must be positive, got {seconds}")
        per_token = self.hardware_flops_per_token(recompute_fraction)
        return per_token * tokens / seconds / peak_flops


def counter_for(model, seq_len: int) -> FlopsCounter:
    """Build a `FlopsCounter` from a `GPT`, so the shape is never transcribed by hand."""
    cfg = model.config
    return FlopsCounter(
        num_params=model.num_params(non_embedding=True),
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        head_dim=cfg.head_dim,
        seq_len=seq_len,
    )

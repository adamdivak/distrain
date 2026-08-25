"""FLOPs accounting, MFU and HFU.

Every throughput claim in this project routes through this module, so the conventions
are fixed here once (`docs/decisions.md` section 3) rather than re-derived per run:

- **Numerator**: PaLM-style `6N + 12*L*H*Q*T` FLOPs per token, i.e. *including* the
  attention term. That term is 13% of total FLOPs at 124M/seq-1024 -- dropping it, as
  bare `6ND` does, would understate MFU by more than most of the effects this study
  is trying to measure.
- **N is matmul parameters only** (`GPT.flops_params()`), not every parameter. An
  untied `wte` is a gather; charging `6N` for it invents 27% of work that never runs.
  See `docs/decisions.md` section 3.
- **MFU vs HFU**: MFU counts only the FLOPs the model mathematically requires. HFU
  additionally counts activation recomputation, which FSDP2 at 7B will need. They
  differ by ~33% under full recompute and are widely conflated; both are reported.
- **Denominator**: bf16 *dense* peak, measured where possible. Never the 2:4-sparsity
  marketing number, which is 2x higher and would halve every reported MFU.

Unknown devices raise rather than fall back to a guess: a silently wrong denominator
is indistinguishable from a real efficiency finding. That is not hypothetical -- see
`PeakSpec`, which exists because the first version of this table shipped a wrong 3090
figure and produced a 158% MFU.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PeakSpec:
    """A bf16 dense peak together with where the number came from.

    Provenance is recorded because the first version of this table was wrong: the
    RTX 3090 was entered at 35.6 TFLOP/s, which is the card's FP32 *non-tensor* rate,
    not its bf16 tensor rate. The error produced a confident-looking MFU of 158%.
    Anything above 100% is impossible, which is the only reason it was caught.
    """

    tflops: float
    source: str

    @property
    def measured(self) -> bool:
        return self.source.startswith("measured")


# Ordered: first substring match wins, so specific variants precede their families.
#
# Prefer measured values. Run `scripts/measure_roofline.py` on any new GPU class before
# trusting its MFU numbers, and record the result here -- the same discipline the brief
# already applies to interconnect, where nccl-tests defines the ceiling rather than the
# vendor's bandwidth claim.
#
# Each measured entry is followed by the vendor figure it replaced, marked `SHADOWED`.
# Those lines are unreachable by construction -- the measured twin above them matches
# first -- and exist so the measured/datasheet gap is recorded next to the number it
# corrects. Never reorder one above its measured twin: it would silently restore the
# denominator the measurement was taken to replace. `TestPeakLookup` pins the ordering.
_PEAK_BF16: tuple[tuple[str, PeakSpec], ...] = (
    ("H100 PCIe", PeakSpec(756.0, "datasheet dense, UNVERIFIED")),
    ("H100 NVL", PeakSpec(835.0, "datasheet dense, UNVERIFIED")),
    ("H100", PeakSpec(989.0, "datasheet dense SXM, UNVERIFIED")),
    ("A100-SXM4-80GB", PeakSpec(269.9, "measured on runpod 8xA100 US-MD-1 2026-08-16, 16384^3 bf16 GEMM")),
    ("A100-SXM4-80GB", PeakSpec(312.0, "A100 datasheet dense; SHADOWED -- measured is 86.5% of it")),
    ("A100-SXM4-40GB", PeakSpec(270.1, "measured on prime/lambdalabs 8xA100 us-east-1 2026-08-21, 16384^3 bf16 GEMM")),
    ("A100-SXM4-40GB", PeakSpec(312.0, "A100 datasheet dense; SHADOWED -- measured is 86.6% of it")),
    ("A100 80GB PCIe", PeakSpec(256.5, "measured on runpod 2xA100-PCIe CA-MTL-3 2026-08-22, "
                                "4096^3 bf16 GEMM; 223.9 at 16384^3 -- the 300 W part throttles "
                                "on long large GEMMs where the SXM4 cards do not")),
    # NVIDIA quotes the same 312 for the 300 W PCIe part as for the 400 W SXM4 one, which
    # is where the widest gap in this table comes from.
    ("A100 80GB PCIe", PeakSpec(312.0, "A100 datasheet dense, same figure as SXM4; SHADOWED -- "
                                "measured is 82.2% of it at 4096^3, 71.8% at 16384^3")),
    ("A100", PeakSpec(312.0, "datasheet dense, UNVERIFIED")),
    ("L40S", PeakSpec(181.0, "datasheet dense, UNVERIFIED")),
    ("RTX 3090", PeakSpec(82.6, "measured on aurora 2026-07-28, 16384^3 bf16 GEMM")),
    # The only card here measuring *above* its vendor figure, and the reason is clock, not
    # accumulate mode: bf16 has no FP16-accumulate path. The whitepaper rate is quoted at the
    # 3090 FE's 1695 MHz boost; aurora's card is a 420 W board holding 1990 MHz mean under a
    # 16384^3 GEMM (measured 2026-08-25), which is 41.7 kFLOP/clk against the whitepaper's
    # 42.0 -- 99.3% of the architectural rate. A 350 W FE would land near 71-75, so this
    # entry describes aurora's board, not every 3090; measure any rented one.
    ("RTX 3090", PeakSpec(71.2, "GA102 whitepaper dense, bf16 tensor with FP32 accumulate "
                          "(142.3 with 2:4 sparsity), at the 1695 MHz FE boost clock; "
                          "SHADOWED -- aurora's 420 W card clocks 17% higher")),
)


def peak_bf16_spec(device_name: str) -> PeakSpec:
    """Peak spec for a `torch.cuda.get_device_name()` string, with provenance."""
    for pattern, spec in _PEAK_BF16:
        if pattern in device_name:
            return spec
    # dict.fromkeys: patterns repeat now that datasheet twins sit under measured entries
    known = ", ".join(dict.fromkeys(p for p, _ in _PEAK_BF16))
    raise KeyError(
        f"no bf16 dense peak recorded for {device_name!r}. Measure it with "
        f"scripts/measure_roofline.py and add it to _PEAK_BF16. Known: {known}"
    )


def peak_bf16_flops(device_name: str) -> float:
    """bf16 dense peak in FLOP/s."""
    return peak_bf16_spec(device_name).tflops * 1e12


@dataclass(frozen=True)
class FlopsCounter:
    """Per-token FLOPs for one model shape.

    `matmul_params` is `GPT.flops_params()`, *not* `GPT.num_params()`: only parameters
    that participate in a matmul. Rotary has no positional parameters; a tied `wte`
    counts once, as the output projection; an untied `wte` does not count at all.
    """

    matmul_params: int
    n_layer: int
    n_head: int
    head_dim: int
    seq_len: int

    @property
    def model_flops_per_token(self) -> float:
        """Forward + backward FLOPs the model mathematically requires, per token."""
        attention = 12 * self.n_layer * self.n_head * self.head_dim * self.seq_len
        return 6 * self.matmul_params + attention

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
        return (6 + 2 * r) * self.matmul_params + (12 + 4 * r) * attention_term

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
        matmul_params=model.flops_params(),
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        head_dim=cfg.head_dim,
        seq_len=seq_len,
    )

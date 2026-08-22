"""GPT-2 style decoder-only transformer.

Written against `reference/nanogpt/model.py` (see `reference/PROVENANCE.md`), reduced
to what a pretraining scaling study needs: no `from_pretrained`, no sampling, no
block-size surgery. Attention is always `F.scaled_dot_product_attention`, which gets
the flash kernels for free without adding a dependency (`project_brief.md` section 3).

A note on "benchmark-comparable": the 3.28 target constrains the *data* -- FineWeb,
GPT-2 BPE, the first 10,485,760 val tokens -- not the architecture. What this study
requires is that the architecture be byte-identical across every parallelism config,
so that throughput and time-to-target-loss differences are attributable to the
distributed layer alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    # GPT-2's vocab is 50257; padded to a multiple of 64 so the lm_head matmul tiles
    # cleanly. The extra logits are unreachable targets and train to -inf.
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    # GPT-2 used biases everywhere; omitting them is slightly faster and marginally
    # better. Kept configurable, but must not vary across configs within a study.
    bias: bool = False
    use_weight_tying: bool = False

    @property
    def head_dim(self) -> int:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd={self.n_embd} not divisible by n_head={self.n_head}")
        return self.n_embd // self.n_head


class Rotary(nn.Module):
    """Rotary position embeddings, as in the early modded-nanogpt records.

    Half-split convention: the head dim is split into two halves rotated against each
    other, not interleaved pairs. Both are valid RoPE (a fixed permutation of dims);
    what matters is that q and k use the same one.

    cos/sin are precomputed for `max_seq_len` in fp32 and sliced per forward -- no
    data-dependent branching, so `torch.compile` sees a static graph. They are
    non-persistent buffers: derived state, kept out of checkpoints, moved by `.to()`.
    """

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"rotary dim must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_head, T, head_dim)
        T = x.size(2)
        cos, sin = self.cos[:T], self.sin[:T]  # (T, head_dim/2), broadcast over B, heads
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = -x1 * sin + x2 * cos
        return torch.cat((y1, y2), dim=-1).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rotary = Rotary(config.head_dim, config.block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm from modded-nanogpt
        q, k = self.rotary(q), self.rotary(k)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.relu(self.c_fc(x)).square()))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if self.config.use_weight_tying:
            # weight tying: the token embedding is the output projection
            self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # modded-nanogpt: residual projections start at zero, so every block is the
        # identity at init and the residual stream is calm regardless of depth.
        # Supersedes GPT-2's 1/sqrt(2*n_layer) scaling of these same weights.
        # Deliberately AFTER apply(): _init_weights sees modules without names, and
        # anything set in a submodule constructor is overwritten by apply() -- zeroing
        # there is a silent no-op.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.zeros_(p)
        if not config.use_weight_tying:
            # the untied head starts at zero too: uniform logits, initial loss exactly
            # ln(vocab_size). Never zero a *tied* head -- it is wte.
            nn.init.zeros_(self.lm_head.weight)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Total parameter count, as reported. Not the N in the FLOPs formula.

        With rotary embeddings there are no positional parameters to exclude, so this
        is simply every parameter the model owns. For throughput accounting use
        `flops_params()` -- see `mfu.py`.
        """
        return sum(p.numel() for p in self.parameters())

    def flops_params(self) -> int:
        """The N in `6N` -- only parameters that participate in a matmul.

        `wte` is a gather, not a matmul. It earns its place in the FLOPs formula only
        when weight tying makes it the output projection; counting an *untied* `wte`
        charges the GPU for work it never does and overstates MFU by 27% at the 124M
        shape. Untying changes no matmul shape, so it costs ~0 FLOPs/token.
        """
        n = self.num_params()
        if not self.config.use_weight_tying:
            n -= self.transformer.wte.weight.numel()
        return n

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, t = idx.size()
        if t > self.config.block_size:
            # also the bound on the rotary cos/sin tables, sized to block_size
            raise ValueError(f"sequence length {t} exceeds block_size {self.config.block_size}")

        x = self.transformer.drop(self.transformer.wte(idx))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1).long()
            )
        return logits, loss

    def configure_optimizer(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        fused: bool | None = None,
    ) -> torch.optim.AdamW:
        """AdamW with weight decay on matmul/embedding weights only, per nanoGPT."""
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        if fused is None:
            fused = torch.cuda.is_available()
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=fused)

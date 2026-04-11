"""
CodeGPT — GPT-style decoder-only transformer built from scratch.

Components (all implemented here, zero external LLM libraries):
  RotaryEmbedding   — RoPE positional encoding (better than learned positions)
  CausalSelfAttention — multi-head attention with causal mask + fused QKV
  MLP               — two-layer feed-forward with GELU activation
  Block             — pre-LayerNorm transformer block
  CodeGPT           — full model with weight-tied embedding/lm_head
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from llm.config import ModelConfig


# ── Rotary Positional Embedding (RoPE) ────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    Rotary positional embedding (Su et al., 2021).
    Encodes position by rotating query/key vectors — no position table stored
    in the vocabulary, extrapolates gracefully beyond training length.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10_000):
        super().__init__()
        # Precompute cos/sin tables for every position up to max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)        # [T, dim/2]
        emb   = torch.cat([freqs, freqs], dim=-1)           # [T, dim]
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int):
        """Return (cos, sin) tables for the first `seq_len` positions."""
        return (
            self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0),  # [1,1,T,dim]
            self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension into the first half."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos, sin) -> tuple:
    """Apply RoPE to query and key tensors."""
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


# ── Causal Self-Attention ─────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.scale    = math.sqrt(self.head_dim)

        # Fused QKV projection — one matrix multiply instead of three
        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop_attn = nn.Dropout(cfg.dropout)
        self.drop_proj = nn.Dropout(cfg.dropout)

        self.rotary = RotaryEmbedding(self.head_dim, cfg.max_seq_len)

        # Causal mask: lower-triangular, registered as buffer so it moves to GPU
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # QKV in one shot, then split
        qkv = self.qkv(x)                                     # [B, T, 3C]
        q, k, v = qkv.split(C, dim=-1)                        # each [B, T, C]

        # Reshape to [B, heads, T, head_dim]
        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)

        # Apply RoPE to q and k
        cos, sin = self.rotary(q, T)
        q, k = apply_rotary(q, k, cos, sin)

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) / self.scale         # [B, H, T, T]
        attn = attn.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.drop_attn(attn)

        out = attn @ v                                         # [B, H, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        return self.drop_proj(self.proj(out))


# ── Feed-Forward MLP ──────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1  = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2  = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


# ── Transformer Block ─────────────────────────────────────────────────────────

class Block(nn.Module):
    """Pre-LayerNorm transformer block (more stable than post-LN)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.mlp  = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # residual around attention
        x = x + self.mlp(self.ln2(x))    # residual around MLP
        return x


# ── Full Model ────────────────────────────────────────────────────────────────

class CodeGPT(nn.Module):
    """
    Decoder-only transformer language model for code.

    ~85 M parameters at default config (d_model=768, n_layers=12, vocab=16384).
    Weight tying: input embedding == output projection (saves ~12 M params,
    forces consistent token representations).
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.drop_e = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f   = nn.LayerNorm(cfg.d_model)        # final layer norm
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed.weight

        # Initialise weights (GPT-2 style)
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layers) for stability
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,          # [B, T]
        targets: torch.Tensor | None = None,  # [B, T] for training
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, f"sequence length {T} > max {self.cfg.max_seq_len}"

        x = self.drop_e(self.embed(input_ids))     # [B, T, d_model]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                   # [B, T, vocab_size]

        loss = None
        if targets is not None:
            # Flatten for cross-entropy; ignore pad tokens
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=self.cfg.pad_id,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,         # [1, T] — single sample
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation — returns the full sequence including prompt."""
        self.eval()
        ctx = input_ids.clone()
        for _ in range(max_new_tokens):
            # Truncate context to max window
            ctx_cond = ctx if ctx.size(1) <= self.cfg.max_seq_len \
                       else ctx[:, -self.cfg.max_seq_len:]
            logits, _ = self(ctx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # [1, V]

            # Top-k filtering
            if top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_vals[:, -1:]] = float("-inf")

            # Nucleus (top-p) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits.scatter_(-1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [1, 1]
            ctx = torch.cat([ctx, next_id], dim=1)

            if eos_id is not None and next_id.item() == eos_id:
                break

        return ctx

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

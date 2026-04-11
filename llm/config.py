"""
ModelConfig — single source of truth for all architecture hyper-parameters.
Change values here; everything else reads from this object.
"""
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # ── Vocabulary ────────────────────────────────────────────────────────
    vocab_size: int = 16_384       # BPE merges; power of 2 for nice matmul shapes

    # ── Architecture ──────────────────────────────────────────────────────
    d_model:     int = 768         # embedding / hidden dimension
    n_heads:     int = 12          # attention heads  (head_dim = d_model / n_heads = 64)
    n_layers:    int = 12          # transformer blocks
    d_ff:        int = 3_072       # feed-forward inner dim  (4 × d_model)
    max_seq_len: int = 2_048       # maximum context window
    dropout:     float = 0.1

    # ── Special tokens (indices into vocab) ───────────────────────────────
    pad_id:        int = 0
    bos_id:        int = 1
    eos_id:        int = 2
    unk_id:        int = 3
    fim_prefix_id: int = 4         # Fill-In-the-Middle tokens
    fim_suffix_id: int = 5
    fim_middle_id: int = 6
    sys_id:        int = 7         # chat format tokens
    user_id:       int = 8
    asst_id:       int = 9

    # ── Derived (read-only) ───────────────────────────────────────────────
    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    # ── Paths ─────────────────────────────────────────────────────────────
    tokenizer_path: str = "llm/tok.json"
    checkpoint_dir: str = "checkpoints"

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 49154
    hidden_dim: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    head_dim: int = 64
    mlp_hidden: int = 2730
    max_seq_len: int = 4096
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    use_grad_checkpoint: bool = True
    dropout: float = 0.0
    # Defaults yield ~201M params (50M tied embeds + 12 * ~12.6M layers).
    # SwiGLU's 2/3 contraction is why mlp_hidden=2730 instead of 4*hidden_dim.

    def __post_init__(self) -> None:
        if self.num_heads * self.head_dim != self.hidden_dim:
            raise ValueError(
                f"num_heads*head_dim ({self.num_heads}*{self.head_dim}) "
                f"must equal hidden_dim ({self.hidden_dim})"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")

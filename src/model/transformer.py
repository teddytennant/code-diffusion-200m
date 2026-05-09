from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from .attention import MultiHeadAttention
from .components import RMSNorm, SwiGLU
from .config import ModelConfig
from .rope import precompute_rope_cache


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)
        self.attn = MultiHeadAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            dropout=config.dropout,
        )
        self.mlp_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)
        self.mlp = SwiGLU(
            hidden_dim=config.hidden_dim,
            mlp_hidden=config.mlp_hidden,
            dropout=config.dropout,
        )
        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
        causal: bool,
    ) -> torch.Tensor:
        h = self.attn(self.attn_norm(x), cos, sin, attention_mask=attention_mask, causal=causal)
        x = x + h
        h = self.mlp(self.mlp_norm(x))
        x = x + h
        return x


class CodeDiffusionTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_dim)

        self.layers = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)

        if config.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        cos, sin = precompute_rope_cache(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        residual_scale = 1.0 / math.sqrt(2.0 * self.config.num_layers)

        for block in self.layers:
            for proj in (block.attn.q_proj, block.attn.k_proj, block.attn.v_proj):
                nn.init.normal_(proj.weight, mean=0.0, std=0.02)
            nn.init.normal_(block.mlp.w_gate.weight, mean=0.0, std=0.02)
            nn.init.normal_(block.mlp.w_up.weight, mean=0.0, std=0.02)

            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=0.02)
            block.attn.o_proj.weight.data.mul_(residual_scale)

            nn.init.normal_(block.mlp.w_down.weight, mean=0.0, std=0.02)
            block.mlp.w_down.weight.data.mul_(residual_scale)

    def num_parameters(self) -> int:
        n = 0
        seen: set[int] = set()
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            n += p.numel()
        return n

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )

        x = self.tok_emb(input_ids)

        cos = self.rope_cos
        sin = self.rope_sin

        use_ckpt = self.config.use_grad_checkpoint and self.training

        for block in self.layers:
            if use_ckpt:
                x = ckpt.checkpoint(
                    block, x, cos, sin, attention_mask, causal,
                    use_reentrant=False,
                )
            else:
                x = block(x, cos, sin, attention_mask, causal)

        x = self.final_norm(x)

        if self.lm_head is None:
            logits = x @ self.tok_emb.weight.t()
        else:
            logits = self.lm_head(x)

        return logits

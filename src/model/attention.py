from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope

try:
    from flash_attn import flash_attn_func  # type: ignore

    HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_func = None  # type: ignore
    HAS_FLASH_ATTN = False


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_heads * head_dim != hidden_dim:
            raise ValueError(
                f"num_heads*head_dim ({num_heads}*{head_dim}) != hidden_dim ({hidden_dim})"
            )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        use_flash = (
            HAS_FLASH_ATTN
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
            and attention_mask is None
        )

        if use_flash:
            q_f = q.transpose(1, 2).contiguous()
            k_f = k.transpose(1, 2).contiguous()
            v_f = v.transpose(1, 2).contiguous()
            out = flash_attn_func(  # type: ignore[misc]
                q_f, k_f, v_f,
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=causal,
            )
            out = out.contiguous().view(bsz, seq_len, self.hidden_dim)
        else:
            attn_mask: torch.Tensor | None = None
            if attention_mask is not None:
                key_pad = attention_mask.to(torch.bool)
                attn_mask = key_pad[:, None, None, :].expand(bsz, self.num_heads, seq_len, seq_len)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=causal and attn_mask is None,
            )
            out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_dim)

        return self.o_proj(out)

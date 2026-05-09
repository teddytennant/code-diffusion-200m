from __future__ import annotations

import torch


def precompute_rope_cache(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(t, freqs)
    cos = angles.cos()
    sin = angles.sin()
    return cos, sin


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = q.shape[-2]
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        x_rot1 = x1 * cos - x2 * sin
        x_rot2 = x2 * cos + x1 * sin
        return torch.cat([x_rot1, x_rot2], dim=-1).to(x.dtype)

    return rotate(q), rotate(k)

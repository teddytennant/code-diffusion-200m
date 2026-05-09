from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (x32.to(in_dtype)) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, mlp_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate = nn.Linear(hidden_dim, mlp_hidden, bias=False)
        self.w_up = nn.Linear(hidden_dim, mlp_hidden, bias=False)
        self.w_down = nn.Linear(mlp_hidden, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))

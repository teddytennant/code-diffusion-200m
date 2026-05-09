"""Data pipeline for Code-Diffusion-200M.

The dataset and loader modules import torch eagerly. Tokenizer / masking
helpers do not, so callers that only want masking primitives (e.g. unit
tests on a mock tokenizer) can import those without pulling torch in.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .ast_masking import ast_subtree_mask
from .fim import to_fim
from .random_masking import random_token_mask

if TYPE_CHECKING:  # only for type checkers — avoids torch import at runtime
    from .dataset import CodeDiffusionDataset, Source
    from .loader import make_dataloader
    from .tokenizer import CodeTokenizer


def __getattr__(name: str):
    """Lazy attribute access — defer torch / transformers imports."""
    if name == "CodeTokenizer":
        from .tokenizer import CodeTokenizer

        return CodeTokenizer
    if name in ("CodeDiffusionDataset", "Source"):
        from .dataset import CodeDiffusionDataset, Source

        return {"CodeDiffusionDataset": CodeDiffusionDataset, "Source": Source}[name]
    if name == "make_dataloader":
        from .loader import make_dataloader

        return make_dataloader
    raise AttributeError(name)


__all__ = [
    "CodeTokenizer",
    "CodeDiffusionDataset",
    "Source",
    "ast_subtree_mask",
    "random_token_mask",
    "to_fim",
    "make_dataloader",
]

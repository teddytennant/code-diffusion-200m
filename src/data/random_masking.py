"""Uniform random token masking for masked-diffusion training."""
from __future__ import annotations

import random
from typing import List, Protocol, Tuple


class _TokenizerLike(Protocol):
    @property
    def mask_id(self) -> int: ...
    @property
    def special_ids(self) -> set[int]: ...


def random_token_mask(
    token_ids: List[int],
    tokenizer: _TokenizerLike,
    target_ratio: float,
    rng: random.Random,
) -> Tuple[List[int], List[bool]]:
    """Mask each non-special token independently with probability ``target_ratio``.

    Special tokens (pad, FIM markers, mask itself) are never masked. Returns
    ``(masked_ids, mask_positions)`` with the same length as ``token_ids``.
    """
    if not 0.0 <= target_ratio <= 1.0:
        raise ValueError(f"target_ratio must be in [0, 1], got {target_ratio}")

    specials = tokenizer.special_ids
    mask_id = tokenizer.mask_id

    out_ids = list(token_ids)
    mask_positions = [False] * len(token_ids)
    for i, tid in enumerate(token_ids):
        if tid in specials:
            continue
        if rng.random() < target_ratio:
            out_ids[i] = mask_id
            mask_positions[i] = True
    return out_ids, mask_positions

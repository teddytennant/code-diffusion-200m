"""Fill-in-the-middle (FIM) reordering.

We sample a contiguous span uniformly from the token sequence, split into
prefix / middle / suffix, then concatenate using StarCoder2's FIM markers.
Both PSM and SPM orderings are produced (50/50 by default), matching the
training mix used by StarCoder/StarCoder2.
"""
from __future__ import annotations

import random
from typing import List, Protocol


class _TokenizerLike(Protocol):
    @property
    def fim_prefix_id(self) -> int: ...
    @property
    def fim_middle_id(self) -> int: ...
    @property
    def fim_suffix_id(self) -> int: ...


def to_fim(
    token_ids: List[int],
    tokenizer: _TokenizerLike,
    rng: random.Random,
    psm_prob: float = 0.5,
) -> List[int]:
    """Reorder ``token_ids`` into PSM or SPM format with FIM marker tokens.

    Output length is ``len(token_ids) + 3``: three FIM marker tokens are
    inserted, the underlying content tokens are simply reordered.
    """
    n = len(token_ids)
    if n < 3:
        # Too short to split meaningfully — emit a degenerate PSM with empty mid.
        return [
            tokenizer.fim_prefix_id,
            *token_ids,
            tokenizer.fim_suffix_id,
            tokenizer.fim_middle_id,
        ]

    # Sample two cut points uniformly at random, sorted, to define [prefix |
    # middle | suffix]. Both endpoints inclusive of the empty span case so the
    # model occasionally sees prefix-only or suffix-only training signal.
    a = rng.randint(0, n)
    b = rng.randint(0, n)
    lo, hi = (a, b) if a <= b else (b, a)
    prefix = token_ids[:lo]
    middle = token_ids[lo:hi]
    suffix = token_ids[hi:]

    if rng.random() < psm_prob:
        # PSM: <prefix> P <suffix> S <middle> M
        return [
            tokenizer.fim_prefix_id,
            *prefix,
            tokenizer.fim_suffix_id,
            *suffix,
            tokenizer.fim_middle_id,
            *middle,
        ]
    # SPM: <suffix> S <prefix> P <middle> M
    # (matches StarCoder2 SPM: suffix first, then prefix, then middle)
    return [
        tokenizer.fim_suffix_id,
        *suffix,
        tokenizer.fim_prefix_id,
        *prefix,
        tokenizer.fim_middle_id,
        *middle,
    ]

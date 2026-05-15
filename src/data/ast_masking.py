"""AST-aware subtree masking for Python source.

Parses Python, selects maskable AST nodes (FunctionDef, If, For, Assign, ...),
maps their char spans to token spans via the tokenizer's offset_mapping, and
masks whole subtrees until the target ratio is reached. Returns None on parse
failure (caller falls back to random masking).
"""
from __future__ import annotations

import ast
import random
from typing import List, Optional, Protocol, Tuple

# Node types we consider "interesting" enough to mask as a unit. A more
# fine-grained selection (e.g. expression operands) would over-fragment the
# program; a coarser selection (e.g. only FunctionDef) would rarely fire on
# short snippets.
MASKABLE_NODES: tuple[type[ast.AST], ...] = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Return,
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.Expr,
)


class _TokenizerLike(Protocol):
    @property
    def mask_id(self) -> int: ...
    @property
    def special_ids(self) -> set[int]: ...

    def encode_with_offsets(
        self, text: str
    ) -> Tuple[List[int], List[Tuple[int, int]]]: ...


def _line_col_to_char_offsets(source: str) -> List[int]:
    """Cumulative character offset at the start of each 1-indexed line.

    Index i (0-based) gives the char offset of line (i+1).
    """
    offsets = [0]
    for line in source.split("\n"):
        offsets.append(offsets[-1] + len(line) + 1)  # +1 for the '\n'
    return offsets


def _node_char_span(node: ast.AST, line_starts: List[int]) -> Optional[Tuple[int, int]]:
    """Return (char_start, char_end) for an AST node, or None if it lacks position info."""
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)
    col = getattr(node, "col_offset", None)
    end_col = getattr(node, "end_col_offset", None)
    if lineno is None or end_lineno is None or col is None or end_col is None:
        return None
    if lineno < 1 or end_lineno < 1:
        return None
    if end_lineno >= len(line_starts) or lineno - 1 >= len(line_starts):
        return None
    start = line_starts[lineno - 1] + col
    end = line_starts[end_lineno - 1] + end_col
    if end < start:
        return None
    return start, end


def ast_subtree_mask(
    source: str,
    tokenizer: _TokenizerLike,
    target_ratio: float,
    rng: random.Random,
    max_seq_len: int = 4096,
) -> Optional[Tuple[List[int], List[bool]]]:
    """Mask Python AST subtrees until ~target_ratio of tokens are masked.

    Returns ``(token_ids, mask_positions)`` or ``None`` if ``ast.parse`` fails.
    The ratio is approximate: subtree boundaries rarely sum to an exact target.
    """
    if not 0.0 < target_ratio < 1.0:
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")

    # Empty source — nothing to do.
    if not source:
        return [], []

    # Truncate and walk back to last valid newline so ast.parse succeeds.
    ids_full, offsets_full = tokenizer.encode_with_offsets(source)
    if len(ids_full) > max_seq_len:
        cut_char = offsets_full[max_seq_len - 1][1] if max_seq_len > 0 else 0
        source = source[:cut_char]

    tree = None
    for _ in range(8):
        try:
            tree = ast.parse(source)
            break
        except SyntaxError:
            nl = source.rfind("\n")
            if nl <= 0:
                return None
            source = source[:nl]
    if tree is None:
        return None

    ids, offsets = tokenizer.encode_with_offsets(source)
    n_tok = len(ids)
    if n_tok == 0:
        return [], []

    line_starts = _line_col_to_char_offsets(source)

    # Collect (start_tok, end_tok) spans for every interesting node.
    # We map char ranges to token ranges using a binary-search-friendly scan.
    char_to_tok_start: List[int] = [n_tok] * (len(source) + 1)
    char_to_tok_end: List[int] = [0] * (len(source) + 1)
    for ti, (cs, ce) in enumerate(offsets):
        # The first token whose end > char c covers c.
        if cs < ce:  # ignore zero-width tokens
            for c in range(cs, ce + 1):
                if c < len(char_to_tok_start) and char_to_tok_start[c] > ti:
                    char_to_tok_start[c] = ti
                if c < len(char_to_tok_end) and char_to_tok_end[c] < ti + 1:
                    char_to_tok_end[c] = ti + 1

    # Fall through gaps (whitespace between tokens) so any char maps to *some*
    # nearby token start/end.
    last_start = n_tok
    for i in range(len(char_to_tok_start) - 1, -1, -1):
        if char_to_tok_start[i] != n_tok:
            last_start = char_to_tok_start[i]
        else:
            char_to_tok_start[i] = last_start
    last_end = 0
    for i in range(len(char_to_tok_end)):
        if char_to_tok_end[i] != 0:
            last_end = char_to_tok_end[i]
        else:
            char_to_tok_end[i] = last_end

    spans: List[Tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, MASKABLE_NODES):
            continue
        cspan = _node_char_span(node, line_starts)
        if cspan is None:
            continue
        cs, ce = cspan
        if cs >= len(char_to_tok_start):
            continue
        ts = char_to_tok_start[cs]
        te = char_to_tok_end[min(ce, len(char_to_tok_end) - 1)]
        if te <= ts:
            continue
        spans.append((ts, te))

    rng.shuffle(spans)

    target_count = int(round(target_ratio * n_tok))
    masked = [False] * n_tok
    masked_count = 0
    specials = tokenizer.special_ids

    for s, e in spans:
        if masked_count >= target_count:
            break
        # Skip spans that overlap an already-masked region — we want disjoint
        # subtrees so the model sees clean boundaries.
        if any(masked[i] for i in range(s, e)):
            continue
        # Don't mask special tokens (rare in raw source but be safe).
        added = 0
        for i in range(s, e):
            if ids[i] in specials:
                continue
            if not masked[i]:
                masked[i] = True
                added += 1
        masked_count += added

    out_ids = [
        tokenizer.mask_id if masked[i] else ids[i] for i in range(n_tok)
    ]
    return out_ids, masked

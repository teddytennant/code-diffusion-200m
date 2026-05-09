"""CPU-only unit tests for the data pipeline.

These tests use a tiny ``MockTokenizer`` so they don't touch the network or
load the real StarCoder2 vocab. The real-tokenizer integration test is gated
behind ``pytest.mark.network`` and is skipped by default.
"""
from __future__ import annotations

import json
import os
import random
import tempfile
from typing import List, Tuple

import pytest

from src.data.ast_masking import ast_subtree_mask
from src.data.fim import to_fim
from src.data.random_masking import random_token_mask


# ---------------------------------------------------------------------------
# Mock tokenizer
# ---------------------------------------------------------------------------


class MockTokenizer:
    """A character-level mock that gives us real char↔token offsets cheaply.

    Each non-special character maps to one token. ``encode_with_offsets``
    therefore returns offsets ``[(i, i+1)]`` which is exactly what AST masking
    expects (it scans char ranges and looks up token indices).
    """

    mask_id = 100
    pad_id = 101
    fim_prefix_id = 102
    fim_middle_id = 103
    fim_suffix_id = 104
    vocab_size = 105

    @property
    def special_ids(self) -> set[int]:
        return {self.mask_id, self.pad_id, self.fim_prefix_id, self.fim_middle_id, self.fim_suffix_id}

    def encode(self, text: str) -> List[int]:
        # 0..99 reserved for content; bias up by 1 so we never collide with mask_id.
        return [(ord(c) % 99) + 1 for c in text]

    def encode_with_offsets(self, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        ids = self.encode(text)
        offsets = [(i, i + 1) for i in range(len(text))]
        return ids, offsets

    def decode(self, ids) -> str:
        return "".join(chr((i - 1) % 99) for i in ids)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SAMPLE_PY = """\
def add(a, b):
    return a + b


def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


class Counter:
    def __init__(self):
        self.x = 0

    def inc(self):
        self.x += 1
        return self.x
"""


# ---------------------------------------------------------------------------
# AST masking
# ---------------------------------------------------------------------------


def test_ast_masking_basic():
    tok = MockTokenizer()
    rng = random.Random(0)
    res = ast_subtree_mask(SAMPLE_PY, tok, target_ratio=0.3, rng=rng)
    assert res is not None
    ids, mask = res
    assert len(ids) == len(mask) == len(SAMPLE_PY)

    # Every masked position must hold the mask id.
    for i, m in enumerate(mask):
        if m:
            assert ids[i] == tok.mask_id, f"position {i} masked but id={ids[i]}"

    # Approximate target ratio: AST-aware masking is bursty so we allow ±50%.
    actual = sum(mask) / len(mask)
    assert 0.5 * 0.3 <= actual <= 1.5 * 0.3, f"masked {actual:.2%}, target 30%"


def test_ast_masking_returns_none_on_syntax_error():
    tok = MockTokenizer()
    rng = random.Random(0)
    broken = "def f(:\n    return 1\n"
    res = ast_subtree_mask(broken, tok, target_ratio=0.3, rng=rng)
    assert res is None


def test_ast_masking_empty():
    tok = MockTokenizer()
    rng = random.Random(0)
    ids, mask = ast_subtree_mask("", tok, target_ratio=0.3, rng=rng)
    assert ids == [] and mask == []


def test_ast_masking_truncates_to_max_seq_len():
    tok = MockTokenizer()
    rng = random.Random(0)
    src = SAMPLE_PY * 4
    res = ast_subtree_mask(src, tok, target_ratio=0.2, rng=rng, max_seq_len=64)
    # Truncated prefix must parse and fit in the budget; exact length depends
    # on where the parse-friendly newline backoff lands.
    assert res is not None
    ids, mask = res
    assert 0 < len(ids) <= 64
    assert len(mask) == len(ids)


# ---------------------------------------------------------------------------
# Random masking
# ---------------------------------------------------------------------------


def test_random_mask_ratio_within_5pct():
    tok = MockTokenizer()
    rng = random.Random(123)
    ids = [(i % 50) + 1 for i in range(1000)]  # 1000 non-special tokens
    masked, mask = random_token_mask(ids, tok, target_ratio=0.30, rng=rng)
    assert len(masked) == len(ids) == len(mask)
    actual = sum(mask) / len(mask)
    assert abs(actual - 0.30) < 0.05, f"actual={actual:.3f}"
    # Special tokens never masked.
    for i, m in enumerate(mask):
        if m:
            assert masked[i] == tok.mask_id


def test_random_mask_skips_specials():
    tok = MockTokenizer()
    rng = random.Random(0)
    ids = [tok.pad_id, tok.fim_prefix_id, 5, 6, tok.fim_middle_id]
    masked, mask = random_token_mask(ids, tok, target_ratio=1.0, rng=rng)
    assert mask == [False, False, True, True, False]
    assert masked == [tok.pad_id, tok.fim_prefix_id, tok.mask_id, tok.mask_id, tok.fim_middle_id]


# ---------------------------------------------------------------------------
# FIM
# ---------------------------------------------------------------------------


def test_fim_psm_structure():
    tok = MockTokenizer()
    rng = random.Random(7)
    ids = list(range(1, 21))  # 20 content tokens
    out = to_fim(ids, tok, rng=rng, psm_prob=1.0)  # force PSM
    assert len(out) == len(ids) + 3
    assert out[0] == tok.fim_prefix_id
    # Suffix marker appears once, middle marker is the LAST marker we emit.
    suf = out.index(tok.fim_suffix_id)
    mid = out.index(tok.fim_middle_id)
    assert 0 < suf < mid
    # Content order: every original token id appears exactly once.
    content = [t for t in out if t < 100]
    assert sorted(content) == sorted(ids)


def test_fim_spm_structure():
    tok = MockTokenizer()
    rng = random.Random(7)
    ids = list(range(1, 21))
    out = to_fim(ids, tok, rng=rng, psm_prob=0.0)  # force SPM
    assert len(out) == len(ids) + 3
    assert out[0] == tok.fim_suffix_id
    pre = out.index(tok.fim_prefix_id)
    mid = out.index(tok.fim_middle_id)
    assert 0 < pre < mid


# ---------------------------------------------------------------------------
# Dataset (uses mock tokenizer + an in-memory "memory" source)
# ---------------------------------------------------------------------------


def test_dataset_yields_correct_shapes(tmp_path):
    pytest.importorskip("torch")
    from src.data.dataset import CodeDiffusionDataset

    # Two tiny JSONL sources, ~ a few examples each.
    src1 = tmp_path / "a.jsonl"
    src2 = tmp_path / "b.jsonl"
    src1.write_text(
        "\n".join(json.dumps({"content": SAMPLE_PY}) for _ in range(3)) + "\n"
    )
    src2.write_text(
        "\n".join(json.dumps({"content": "x = 1\n" * 80}) for _ in range(3)) + "\n"
    )

    tok = MockTokenizer()
    ds = CodeDiffusionDataset(
        sources=[
            {"path": str(src1), "kind": "synthetic", "weight": 0.5},
            {"path": str(src2), "kind": "synthetic", "weight": 0.5},
        ],
        tokenizer=tok,
        seq_len=64,
        ast_mask_prob=0.5,
        fim_prob=0.0,
        mask_ratio_min=0.10,
        mask_ratio_max=0.30,
        seed=1,
    )

    n_yielded = 0
    for ex in ds:
        assert ex["input_ids"].shape == (64,)
        assert ex["target_ids"].shape == (64,)
        assert ex["mask_positions"].shape == (64,)
        assert ex["input_ids"].dtype.is_floating_point is False
        assert ex["mask_positions"].dtype == __import__("torch").bool
        assert 0.0 <= float(ex["mask_ratio"]) <= 1.0
        # Where masked, input == mask_id.
        masked_pos = ex["mask_positions"].tolist()
        ids = ex["input_ids"].tolist()
        for i, m in enumerate(masked_pos):
            if m:
                assert ids[i] == tok.mask_id
        n_yielded += 1
        if n_yielded >= 2:
            break

    assert n_yielded >= 1


def test_dataset_fim_source(tmp_path):
    pytest.importorskip("torch")
    from src.data.dataset import CodeDiffusionDataset

    src = tmp_path / "fim.jsonl"
    src.write_text(
        "\n".join(json.dumps({"content": SAMPLE_PY}) for _ in range(4)) + "\n"
    )
    tok = MockTokenizer()
    ds = CodeDiffusionDataset(
        sources=[{"path": str(src), "kind": "fim", "weight": 1.0}],
        tokenizer=tok,
        seq_len=64,
        fim_prob=1.0,
        mask_ratio_min=0.10,
        mask_ratio_max=0.20,
        seed=2,
    )
    # FIM-formatted SAMPLE_PY spans multiple 64-tok chunks, so the three
    # markers may not all land in any single chunk. The meaningful invariant
    # is that the markers appear in the produced data overall.
    seen: set[int] = set()
    for ex in ds:
        for tid in ex["target_ids"].tolist():
            if tid in (tok.fim_prefix_id, tok.fim_middle_id, tok.fim_suffix_id):
                seen.add(tid)
        if len(seen) == 3:
            break
    assert seen == {tok.fim_prefix_id, tok.fim_middle_id, tok.fim_suffix_id}, (
        f"expected all three FIM markers across chunks, saw {seen}"
    )


# ---------------------------------------------------------------------------
# Real StarCoder2 tokenizer — network test, skipped by default.
# ---------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS") != "1",
    reason="set RUN_NETWORK_TESTS=1 to enable",
)
def test_real_starcoder_tokenizer():
    from src.data.tokenizer import CodeTokenizer

    tok = CodeTokenizer()
    assert tok.vocab_size == 49154
    ids = tok.encode("def f(): return 1\n")
    assert len(ids) > 0
    assert tok.mask_id != tok.pad_id
    assert {tok.fim_prefix_id, tok.fim_middle_id, tok.fim_suffix_id}.isdisjoint(
        {tok.mask_id, tok.pad_id}
    )

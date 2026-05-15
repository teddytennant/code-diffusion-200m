"""Streaming masked-diffusion dataset with weighted source mixing.

Reads from HF Arrow / JSONL sources, tokenizes on the fly, packs into seq_len
chunks, and yields masked examples for diffusion (or FIM) training. Supports
AST subtree masking, random masking, and FIM reordering.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .ast_masking import ast_subtree_mask
from .fim import to_fim
from .random_masking import random_token_mask
from .tokenizer import CodeTokenizer


@dataclass
class Source:
    path: str
    kind: str  # "starcoder" | "synthetic" | "fim"
    weight: float


def _normalise_sources(raw: Sequence[Dict[str, Any]]) -> List[Source]:
    out = [Source(path=s["path"], kind=s["kind"], weight=float(s["weight"])) for s in raw]
    total = sum(s.weight for s in out)
    if total <= 0:
        raise ValueError("Source weights must sum to > 0")
    for s in out:
        s.weight = s.weight / total
    return out


def _iter_jsonl(path: str) -> Iterator[str]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("content") or obj.get("code") or obj.get("text")
            if isinstance(text, str) and text:
                yield text


def _iter_starcoder(path: str) -> Iterator[str]:
    """Stream Python source from a HF dataset directory or single Arrow file."""
    try:
        from datasets import load_from_disk, Dataset  # type: ignore
    except Exception as e:  # pragma: no cover - datasets is in requirements
        raise RuntimeError("`datasets` is required for starcoder sources") from e

    if os.path.isdir(path):
        ds = load_from_disk(path)
        # If a DatasetDict was saved, pick the train split.
        if hasattr(ds, "keys") and not hasattr(ds, "__iter__"):
            ds = ds["train"]
    else:
        ds = Dataset.from_file(path)
    for row in ds:
        text = row.get("content") or row.get("code") or row.get("text")
        if isinstance(text, str) and text:
            yield text


# Test/in-memory fallback: a "source" can be a dict with kind="memory" and a
# `texts` field. Not part of the public spec but lets unit tests exercise the
# packing path without writing JSONL files.
def _iter_memory(spec: Dict[str, Any]) -> Iterator[str]:
    for t in spec.get("texts", []):
        if isinstance(t, str) and t:
            yield t


class CodeDiffusionDataset(IterableDataset):
    """Mix-and-mask streaming dataset.

    Yields dicts with keys:

    * ``input_ids``    — LongTensor (T,) tokens after masking / FIM
    * ``target_ids``   — LongTensor (T,) original tokens (loss target)
    * ``mask_positions`` — BoolTensor (T,) True where masked
    * ``mask_ratio``   — float, the ratio used for this sample

    Sequences are packed by concatenating tokenised examples with a single
    newline-id separator so the model still sees natural Python boundaries.
    The trailing partial chunk of any pass is discarded.
    """

    def __init__(
        self,
        sources: Sequence[Dict[str, Any]],
        tokenizer: CodeTokenizer,
        seq_len: int = 4096,
        ast_mask_prob: float = 0.7,
        fim_prob: float = 0.10,
        mask_ratio_min: float = 0.05,
        mask_ratio_max: float = 0.50,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if not 0.0 <= mask_ratio_min < mask_ratio_max <= 1.0:
            raise ValueError("require 0 <= mask_ratio_min < mask_ratio_max <= 1")
        self.sources = _normalise_sources(sources)
        self.tok = tokenizer
        self.seq_len = seq_len
        self.ast_mask_prob = ast_mask_prob
        self.fim_prob = fim_prob
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.seed = seed
        self._raw_sources = list(sources)  # keep for memory-source path

        # Pre-encode a separator (newline). Falls back to empty if encoding gives
        # nothing (happens only with degenerate mock tokenizers).
        sep = tokenizer.encode("\n")
        self._sep_ids: List[int] = sep if sep else []

    # ------------------------------------------------------------------
    # Source iteration
    # ------------------------------------------------------------------

    def _iter_source(self, src: Source) -> Iterator[str]:
        if src.kind == "starcoder":
            return _iter_starcoder(src.path)
        if src.kind in ("synthetic", "fim", "jsonl"):
            return _iter_jsonl(src.path)
        if src.kind == "memory":
            spec = next(s for s in self._raw_sources if s.get("kind") == "memory")
            return _iter_memory(spec)
        raise ValueError(f"Unknown source kind: {src.kind}")

    # ------------------------------------------------------------------
    # Masking dispatch
    # ------------------------------------------------------------------

    def _sample_ratio(self, rng: random.Random) -> float:
        return rng.uniform(self.mask_ratio_min, self.mask_ratio_max)

    def _mask_one(
        self,
        text: str,
        kind: str,
        rng: random.Random,
    ) -> Optional[Dict[str, Any]]:
        """Tokenise + mask one raw text. Returns None if it produces no tokens."""
        ratio = self._sample_ratio(rng)

        if kind == "fim" or (kind != "fim" and rng.random() < self.fim_prob):
            ids = self.tok.encode(text)
            if not ids:
                return None
            ids = to_fim(ids, self.tok, rng)
            masked_ids, mask_positions = random_token_mask(ids, self.tok, ratio, rng)
            return {
                "input_ids": masked_ids,
                "target_ids": ids,
                "mask_positions": mask_positions,
                "mask_ratio": ratio,
            }

        # AST or random
        if rng.random() < self.ast_mask_prob:
            # Pre-truncate source by token count so target_ids stays aligned
            # with what ast_subtree_mask sees internally.
            full_ids, full_offsets = self.tok.encode_with_offsets(text)
            if not full_ids:
                return None
            if len(full_ids) > self.seq_len:
                cut_char = full_offsets[self.seq_len - 1][1]
                trunc_text = text[:cut_char]
            else:
                trunc_text = text
            orig_ids, _ = self.tok.encode_with_offsets(trunc_text)
            res = ast_subtree_mask(
                trunc_text, self.tok, ratio, rng, max_seq_len=self.seq_len
            )
            if res is not None:
                masked_ids, mask_positions = res
                if masked_ids and len(masked_ids) == len(orig_ids):
                    return {
                        "input_ids": masked_ids,
                        "target_ids": orig_ids,
                        "mask_positions": mask_positions,
                        "mask_ratio": ratio,
                    }
                # Length drift — bail to random masking on orig_ids below.

        # Random masking fallback
        ids = self.tok.encode(text)
        if not ids:
            return None
        masked_ids, mask_positions = random_token_mask(ids, self.tok, ratio, rng)
        return {
            "input_ids": masked_ids,
            "target_ids": ids,
            "mask_positions": mask_positions,
            "mask_ratio": ratio,
        }

    # ------------------------------------------------------------------
    # Packing
    # ------------------------------------------------------------------

    def _packed_chunks(
        self,
        examples: Iterator[Dict[str, Any]],
    ) -> Iterator[Dict[str, Any]]:
        buf_input: List[int] = []
        buf_target: List[int] = []
        buf_mask: List[bool] = []
        buf_ratio_sum = 0.0
        buf_ratio_n = 0

        for ex in examples:
            inp = ex["input_ids"]
            tgt = ex["target_ids"]
            msk = ex["mask_positions"]
            # Packing splits large examples across chunks (no truncation of tails).

            if buf_input and self._sep_ids:
                buf_input.extend(self._sep_ids)
                buf_target.extend(self._sep_ids)
                buf_mask.extend([False] * len(self._sep_ids))

            buf_input.extend(inp)
            buf_target.extend(tgt)
            buf_mask.extend(msk)
            buf_ratio_sum += float(ex["mask_ratio"])
            buf_ratio_n += 1

            while len(buf_input) >= self.seq_len:
                chunk_inp = buf_input[: self.seq_len]
                chunk_tgt = buf_target[: self.seq_len]
                chunk_msk = buf_mask[: self.seq_len]
                buf_input = buf_input[self.seq_len :]
                buf_target = buf_target[self.seq_len :]
                buf_mask = buf_mask[self.seq_len :]
                ratio = buf_ratio_sum / max(buf_ratio_n, 1)
                buf_ratio_sum = 0.0
                buf_ratio_n = 0
                yield {
                    "input_ids": torch.as_tensor(chunk_inp, dtype=torch.long),
                    "target_ids": torch.as_tensor(chunk_tgt, dtype=torch.long),
                    "mask_positions": torch.as_tensor(chunk_msk, dtype=torch.bool),
                    "mask_ratio": float(ratio),
                }

        # Trailing partial chunk is discarded (by design).

    # ------------------------------------------------------------------
    # Mixing & main loop
    # ------------------------------------------------------------------

    def _make_iterators(self) -> List[Iterator[str]]:
        return [self._iter_source(s) for s in self.sources]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # Per-worker seeding so DataLoader shards don't sync.
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = random.Random(self.seed + worker_id * 9973)

        iters = self._make_iterators()
        kinds = [s.kind for s in self.sources]
        cum: List[float] = []
        running = 0.0
        for s in self.sources:
            running += s.weight
            cum.append(running)

        def pick() -> int:
            r = rng.random()
            for i, c in enumerate(cum):
                if r <= c:
                    return i
            return len(cum) - 1

        def stream_examples() -> Iterator[Dict[str, Any]]:
            exhausted = [False] * len(iters)
            while not all(exhausted):
                idx = pick()
                if exhausted[idx]:
                    # Pick the next live source instead.
                    live = [i for i, x in enumerate(exhausted) if not x]
                    if not live:
                        return
                    idx = rng.choice(live)
                try:
                    text = next(iters[idx])
                except StopIteration:
                    exhausted[idx] = True
                    continue
                ex = self._mask_one(text, kinds[idx], rng)
                if ex is not None:
                    yield ex

        yield from self._packed_chunks(stream_examples())

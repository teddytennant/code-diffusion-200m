"""StarCoder2 tokenizer wrapper.

Adds two special tokens (`<MASK>`, `<PAD>`) on top of the base StarCoder2 vocab
(49152 tokens), giving a final vocab size of 49154 — matching the model
config that another agent owns. FIM tokens (`<fim_prefix>`, `<fim_middle>`,
`<fim_suffix>`) already exist in the base StarCoder2 vocab and are reused.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Union

try:  # torch is optional at import time so unit tests on tiny mocks don't need it
    import torch

    _Tensor = torch.Tensor
except Exception:  # pragma: no cover - torch always present in the train env
    torch = None  # type: ignore[assignment]
    _Tensor = None  # type: ignore[assignment]

MASK_TOKEN = "<MASK>"
PAD_TOKEN = "<PAD>"
FIM_PREFIX_TOKEN = "<fim_prefix>"
FIM_MIDDLE_TOKEN = "<fim_middle>"
FIM_SUFFIX_TOKEN = "<fim_suffix>"

EXPECTED_VOCAB = 49154


class CodeTokenizer:
    """Thin wrapper around the StarCoder2 fast tokenizer.

    Only adds `<MASK>` and `<PAD>`; FIM markers are already in the base vocab.
    """

    def __init__(self, model_name: str = "bigcode/starcoder2-3b") -> None:
        from transformers import AutoTokenizer  # local import to avoid hard dep

        self._tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # FIM markers already exist in StarCoder2 vocab — verify and grab ids.
        fim_ids = self._tok.convert_tokens_to_ids(
            [FIM_PREFIX_TOKEN, FIM_MIDDLE_TOKEN, FIM_SUFFIX_TOKEN]
        )
        unk = self._tok.unk_token_id
        if any(i is None or i == unk for i in fim_ids):
            raise RuntimeError(
                "Expected base StarCoder2 vocab to contain <fim_prefix/middle/suffix>; "
                f"got ids={fim_ids}. Tokenizer may have changed upstream."
            )
        self._fim_prefix_id, self._fim_middle_id, self._fim_suffix_id = fim_ids

        # Add only MASK + PAD so final vocab matches model config (49154).
        added = self._tok.add_special_tokens(
            {"additional_special_tokens": [MASK_TOKEN, PAD_TOKEN]}
        )
        if added not in (0, 2):
            # add_special_tokens returns count of *new* tokens; tokenizers may have
            # already received them on a re-instantiation — both are fine.
            raise RuntimeError(f"Unexpected number of new specials: {added}")
        self._mask_id = self._tok.convert_tokens_to_ids(MASK_TOKEN)
        self._pad_id = self._tok.convert_tokens_to_ids(PAD_TOKEN)
        self._tok.pad_token = PAD_TOKEN
        self._tok.pad_token_id = self._pad_id

        if len(self._tok) != EXPECTED_VOCAB:
            raise RuntimeError(
                f"Final vocab size {len(self._tok)} != expected {EXPECTED_VOCAB}. "
                "Model agent expects 49154; check StarCoder2 base vocab."
            )

    # --- core API -------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def encode_with_offsets(self, text: str) -> tuple[List[int], List[tuple[int, int]]]:
        """Encode and return (ids, char-offset spans). Used by AST masking."""
        out = self._tok(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )
        return list(out["input_ids"]), [tuple(s) for s in out["offset_mapping"]]

    def decode(self, ids: Union[Sequence[int], "_Tensor"]) -> str:
        if torch is not None and isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tok.decode(list(ids), skip_special_tokens=False)

    # --- ids ------------------------------------------------------------------

    @property
    def mask_id(self) -> int:
        return self._mask_id

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def fim_prefix_id(self) -> int:
        return self._fim_prefix_id

    @property
    def fim_middle_id(self) -> int:
        return self._fim_middle_id

    @property
    def fim_suffix_id(self) -> int:
        return self._fim_suffix_id

    @property
    def vocab_size(self) -> int:
        return len(self._tok)

    @property
    def special_ids(self) -> set[int]:
        """Special ids that should never be replaced by the mask sampler."""
        return {
            self._mask_id,
            self._pad_id,
            self._fim_prefix_id,
            self._fim_middle_id,
            self._fim_suffix_id,
        }

    @property
    def underlying(self):
        """Escape hatch for callers that need the raw HF tokenizer."""
        return self._tok

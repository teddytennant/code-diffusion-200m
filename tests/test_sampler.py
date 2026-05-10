"""CPU-only unit tests for the diffusion sampler.

A tiny ``CodeDiffusionTransformer`` exercises the sampling logic without
needing a real trained model — the outputs are noise but every code path
runs end-to-end.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch

from src.eval.sampler_interface import (
    FIM_MIDDLE,
    FIM_PREFIX,
    FIM_SUFFIX,
    Sampler,
)
from src.model import CodeDiffusionTransformer, ModelConfig
from src.sample import DiffusionSampler

from tests.test_data import MockTokenizer


def _tiny_config(vocab_size: int = 200, max_seq_len: int = 128) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        hidden_dim=64,
        num_heads=4,
        head_dim=16,
        num_layers=2,
        mlp_hidden=128,
        max_seq_len=max_seq_len,
        tie_embeddings=True,
        use_grad_checkpoint=False,
        dropout=0.0,
    )


def _build_model(vocab_size: int = 200, max_seq_len: int = 128) -> CodeDiffusionTransformer:
    torch.manual_seed(0)
    cfg = _tiny_config(vocab_size=vocab_size, max_seq_len=max_seq_len)
    return CodeDiffusionTransformer(cfg).eval()


class _CountingModel(torch.nn.Module):
    """Wraps a real ``CodeDiffusionTransformer`` and counts forward calls by ``causal``."""

    def __init__(self, inner: CodeDiffusionTransformer) -> None:
        super().__init__()
        self.inner = inner
        self.calls: list[bool] = []

    @property
    def config(self):  # type: ignore[no-untyped-def]
        return self.inner.config

    def forward(self, input_ids: torch.Tensor, attention_mask: Any = None, causal: bool = False) -> torch.Tensor:
        self.calls.append(bool(causal))
        return self.inner(input_ids, attention_mask=attention_mask, causal=causal)


class _FIMTokenizer(MockTokenizer):
    """Extends MockTokenizer to recognise the FIM marker substrings during encoding."""

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        markers = (
            (FIM_PREFIX, self.fim_prefix_id),
            (FIM_SUFFIX, self.fim_suffix_id),
            (FIM_MIDDLE, self.fim_middle_id),
        )
        i = 0
        while i < len(text):
            matched = False
            for s, tid in markers:
                if text.startswith(s, i):
                    out.append(tid)
                    i += len(s)
                    matched = True
                    break
            if not matched:
                out.append((ord(text[i]) % 99) + 1)
                i += 1
        return out


def test_completion_returns_n_samples_strings() -> None:
    model = _build_model()
    tok = MockTokenizer()
    sampler = DiffusionSampler(model, tok, device="cpu", default_diffusion_steps=4)
    out = sampler.sample("def foo():", max_new_tokens=8, n_samples=2, temperature=0.5, seed=0)
    assert isinstance(out, list)
    assert len(out) == 2
    for s in out:
        assert isinstance(s, str)
        assert len(s) > 0


def test_completion_respects_max_new_tokens() -> None:
    model = _build_model()
    tok = MockTokenizer()
    sampler = DiffusionSampler(model, tok, device="cpu", default_diffusion_steps=3, ar_refine=False)
    max_new = 12
    out = sampler.sample("abc", max_new_tokens=max_new, n_samples=1, temperature=0.5, seed=1)
    assert len(out) == 1
    # MockTokenizer is a 1 char-per-token codec, so the decoded length must
    # exactly equal max_new_tokens.
    assert len(out[0]) == max_new


def test_fim_mode_requires_markers() -> None:
    model = _build_model()
    tok = _FIMTokenizer()
    sampler = DiffusionSampler(model, tok, device="cpu", default_diffusion_steps=2)
    with pytest.raises(ValueError):
        sampler.sample("no markers here", max_new_tokens=4, mode="fim", seed=0)


def test_fim_mode_returns_only_infill() -> None:
    model = _build_model()
    tok = _FIMTokenizer()
    sampler = DiffusionSampler(
        model, tok, device="cpu", default_diffusion_steps=3, ar_refine=False
    )
    prompt = f"{FIM_PREFIX}def f():{FIM_SUFFIX}return x{FIM_MIDDLE}"
    out = sampler.sample(prompt, max_new_tokens=8, mode="fim", n_samples=1, temperature=0.5, seed=2)
    assert len(out) == 1
    s = out[0]
    assert FIM_PREFIX not in s
    assert FIM_SUFFIX not in s
    assert FIM_MIDDLE not in s


def test_confidence_remasking_decays_to_zero() -> None:
    # Pure schedule test — exercises the static helper.
    n = 20
    frac = 0.5
    steps = 8
    k0 = DiffusionSampler._remask_k(n, frac, 0, steps)
    assert k0 == round(n * frac)
    k_last = DiffusionSampler._remask_k(n, frac, steps - 1, steps)
    assert k_last == 0
    # Monotonically non-increasing across the schedule.
    prev = k0
    for s in range(1, steps):
        k = DiffusionSampler._remask_k(n, frac, s, steps)
        assert k <= prev
        prev = k
    # And final step really is zero regardless of n / frac.
    assert DiffusionSampler._remask_k(100, 0.9, 4, 5) == 0


def test_ar_refine_skipped_when_disabled() -> None:
    inner = _build_model()
    model = _CountingModel(inner)
    tok = MockTokenizer()
    sampler = DiffusionSampler(
        model, tok, device="cpu", default_diffusion_steps=3, ar_refine=False
    )
    sampler.sample("hi", max_new_tokens=6, n_samples=1, temperature=0.5, seed=3)
    assert all(c is False for c in model.calls), f"unexpected causal forwards: {model.calls}"
    assert len(model.calls) > 0


def test_diffusion_steps_kwarg_overrides_default() -> None:
    inner = _build_model()
    model = _CountingModel(inner)
    tok = MockTokenizer()
    # Default is 16 but we override with 4 — and disable AR refine + final-fill
    # by ensuring the loop fills everything (the final-fill branch only runs
    # when masks remain, which the loop avoids by completing).
    sampler = DiffusionSampler(
        model,
        tok,
        device="cpu",
        default_diffusion_steps=16,
        ar_refine=False,
        confidence_remask=False,
    )
    sampler.sample(
        "x", max_new_tokens=4, n_samples=1, temperature=0.5, seed=4, diffusion_steps=4
    )
    bidi_calls = [c for c in model.calls if c is False]
    causal_calls = [c for c in model.calls if c is True]
    # With confidence_remask=False the loop fills the whole region in one pass,
    # but it still runs ``diffusion_steps`` iterations (the inner loop early-exits
    # only when nothing is masked; we want to confirm the kwarg actually steered
    # the run away from the default of 16).
    assert len(bidi_calls) <= 4  # never more than diffusion_steps
    assert len(bidi_calls) >= 1
    assert len(causal_calls) == 0  # ar_refine=False
    # Tighter check: re-run with confidence_remask=True so every step actually
    # has masks to fill, and confirm the count equals diffusion_steps exactly.
    inner2 = _build_model()
    model2 = _CountingModel(inner2)
    sampler2 = DiffusionSampler(
        model2,
        tok,
        device="cpu",
        default_diffusion_steps=16,
        ar_refine=False,
        confidence_remask=True,
    )
    sampler2.sample(
        "x", max_new_tokens=8, n_samples=1, temperature=0.5, seed=5, diffusion_steps=4
    )
    bidi2 = [c for c in model2.calls if c is False]
    assert len(bidi2) == 4


def test_implements_sampler_protocol() -> None:
    model = _build_model()
    tok = MockTokenizer()
    sampler = DiffusionSampler(model, tok, device="cpu", default_diffusion_steps=2)
    assert isinstance(sampler, Sampler)

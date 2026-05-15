"""Sampler protocol (implemented by DiffusionSampler).

Evaluation code talks only to this; never imports the concrete sampler directly.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable


@runtime_checkable
class Sampler(Protocol):
    """Generate completions for code prompts.

    Implementations may be diffusion-based, autoregressive, or anything else.
    For ``mode="fim"`` the prompt is expected to already contain the
    StarCoder2 FIM markers ``<fim_prefix>``, ``<fim_suffix>``, ``<fim_middle>``.

    Implementations are encouraged to accept additional kwargs such as
    ``diffusion_steps`` (used by the throughput sweep) and
    ``top_p`` / ``top_k``; they are forwarded via ``**kwargs``.
    """

    def sample(
        self,
        prompt: str,
        max_new_tokens: int,
        mode: Literal["completion", "fim"] = "completion",
        n_samples: int = 1,
        temperature: float = 0.2,
        **kwargs: object,
    ) -> list[str]:
        """Return ``n_samples`` continuations for ``prompt``.

        For ``mode="completion"`` the returned strings are continuations to
        be appended to the prompt. For ``mode="fim"`` they are the infill
        tokens that go between the prefix and the suffix.
        """
        ...


# StarCoder2 FIM sentinels. Shared by eval, tokenizer, and sampler.
FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"


def build_fim_prompt(prefix: str, suffix: str) -> str:
    """Assemble a StarCoder2-style FIM prompt: prefix + suffix + middle marker."""
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"

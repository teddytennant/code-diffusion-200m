"""Throughput vs quality benchmarking.

Sweeps the diffusion-step count and reports tokens/sec against a quick
quality proxy: the fraction of generated continuations that parse as valid
Python via ``ast.parse``. The DataFrame returned makes it trivial to plot
the quality/latency Pareto.
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import pandas as pd

from src.eval.sampler_interface import Sampler


def _parses_as_python(snippet: str) -> bool:
    """Return True iff ``snippet`` is syntactically valid Python.

    We try the snippet on its own; if that fails we try it indented under a
    function definition, which catches the common case where the sampler
    returns a function body rather than a top-level statement.
    """
    try:
        ast.parse(snippet)
        return True
    except SyntaxError:
        pass
    indented = "\n".join("    " + line for line in snippet.splitlines())
    wrapper = f"def _f():\n{indented}\n    pass\n"
    try:
        ast.parse(wrapper)
        return True
    except SyntaxError:
        return False


def _approx_token_count(text: str) -> int:
    """Cheap token approximation: words + punctuation, ~ chars/4.

    This avoids loading a tokenizer in the throughput hot path. Real
    benchmarks should swap in the model's tokenizer; we expose this as
    ``_approx_token_count`` so callers can monkey-patch it.
    """
    return max(1, len(text) // 4)


def benchmark_throughput(
    sampler: Sampler,
    prompts: list[str],
    diffusion_steps_sweep: list[int] | None = None,
    max_new_tokens: int = 256,
    output_path: Path | None = None,
    n_samples_per_prompt: int = 1,
    temperature: float = 0.2,
) -> pd.DataFrame:
    """Sweep diffusion steps and report tokens/sec + parse rate.

    ``sampler`` must accept ``diffusion_steps`` as a kwarg (it is forwarded
    via the protocol's ``**kwargs``). The returned DataFrame has columns
    ``steps``, ``tokens_per_sec``, ``parse_rate``, ``n_generations``,
    ``wall_seconds``.
    """
    if diffusion_steps_sweep is None:
        diffusion_steps_sweep = [4, 8, 16, 32, 64]

    rows: list[dict] = []
    for steps in diffusion_steps_sweep:
        n_tokens_total = 0
        n_parsed = 0
        n_generations = 0
        t0 = time.perf_counter()
        for prompt in prompts:
            samples = sampler.sample(
                prompt,
                max_new_tokens=max_new_tokens,
                mode="completion",
                n_samples=n_samples_per_prompt,
                temperature=temperature,
                diffusion_steps=steps,
            )
            for s in samples:
                n_generations += 1
                n_tokens_total += _approx_token_count(s)
                if _parses_as_python(s):
                    n_parsed += 1
        wall = time.perf_counter() - t0
        rows.append(
            {
                "steps": steps,
                "tokens_per_sec": (n_tokens_total / wall) if wall > 0 else float("inf"),
                "parse_rate": (n_parsed / n_generations) if n_generations else 0.0,
                "n_generations": n_generations,
                "wall_seconds": wall,
            }
        )

    df = pd.DataFrame(rows, columns=[
        "steps", "tokens_per_sec", "parse_rate", "n_generations", "wall_seconds"
    ])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if str(output_path).endswith(".csv"):
            df.to_csv(output_path, index=False)
        else:
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(rows, fh, indent=2)

    return df

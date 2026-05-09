"""MBPP pass@k.

Loads the MBPP dataset via HuggingFace ``datasets`` and evaluates the same
unbiased pass@k estimator used for HumanEval. Programs are executed in a
fresh Python subprocess with a 5-second timeout — independent from the
``human_eval`` package because MBPP's test format is different.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.eval.humaneval import pass_at_k_dict
from src.eval.sampler_interface import Sampler

DEFAULT_TIMEOUT = 5.0


# ---------------------------------------------------------------------------
# Subprocess sandbox shared by MBPP and the FIM glued tests


@dataclass
class ExecResult:
    passed: bool
    error: str = ""


def run_in_subprocess(program: str, timeout: float = DEFAULT_TIMEOUT) -> ExecResult:
    """Execute ``program`` in a fresh ``python -c`` subprocess.

    A non-zero exit code or an exception means failure. Stderr is captured
    and surfaced in ``ExecResult.error``. Each call gets its own interpreter
    so global state cannot leak between problems.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", program],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return ExecResult(passed=False, error="timeout")
    except Exception as exc:  # pragma: no cover - subprocess startup failure
        return ExecResult(passed=False, error=f"launch_error:{exc!r}")

    if completed.returncode != 0:
        return ExecResult(passed=False, error=completed.stderr.strip()[-400:])
    return ExecResult(passed=True)


def _build_mbpp_program(code: str, test_list: list[str]) -> str:
    """Wrap candidate code + reference tests for subprocess execution.

    Each test in ``test_list`` is already a stand-alone ``assert`` statement
    in the MBPP corpus. We just concatenate; if any assert fails the
    subprocess exits non-zero.
    """
    body = code + "\n\n" + "\n".join(test_list) + "\n"
    return body


# ---------------------------------------------------------------------------
# Prompt formatting


def build_mbpp_prompt(problem: dict) -> str:
    """Build the canonical MBPP prompt: description + first test as a hint.

    MBPP problems carry a natural-language ``text`` description and a list
    of ``test_list`` assertions. Following the standard MBPP protocol we
    append the first test so the model knows the expected function name and
    signature.
    """
    text = problem["text"]
    first_test = problem["test_list"][0] if problem["test_list"] else ""
    return (
        f'"""\n{text}\n{first_test}\n"""\n'
    )


# ---------------------------------------------------------------------------
# Dataset loader


def load_mbpp(split: Literal["test", "validation"] = "test") -> list[dict]:
    """Load MBPP via HuggingFace datasets.

    Imported lazily so a missing ``datasets`` install only breaks evaluation
    and not module import.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "datasets is required for MBPP evaluation. "
            "Install with `pip install datasets`."
        ) from exc

    ds = load_dataset("mbpp", split=split)
    return [dict(row) for row in ds]


# ---------------------------------------------------------------------------
# Public entry point


def run_mbpp(
    sampler: Sampler,
    n_samples_per_problem: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    output_path: Path | None = None,
    split: Literal["test", "validation"] = "test",
    timeout: float = DEFAULT_TIMEOUT,
    problems: list[dict] | None = None,
) -> dict:
    """Run MBPP pass@1 and pass@10.

    ``problems`` is exposed so tests can inject a tiny in-memory dataset
    without going through HuggingFace.
    """
    if problems is None:
        problems = load_mbpp(split)

    all_samples: list[dict] = []
    counts: dict[str, tuple[int, int]] = {}

    for problem in problems:
        raw_id = problem.get("task_id", problem.get("text", ""))
        task_id = str(raw_id)[:80]
        prompt = build_mbpp_prompt(problem)
        completions = sampler.sample(
            prompt,
            max_new_tokens=max_new_tokens,
            mode="completion",
            n_samples=n_samples_per_problem,
            temperature=temperature,
        )
        n_correct = 0
        for i, completion in enumerate(completions):
            program = _build_mbpp_program(completion, problem["test_list"])
            result = run_in_subprocess(program, timeout=timeout)
            n_correct += int(result.passed)
            all_samples.append(
                {
                    "task_id": task_id,
                    "completion_idx": i,
                    "completion": completion,
                    "passed": result.passed,
                    "error": result.error,
                }
            )
        counts[task_id] = (len(completions), n_correct)

    metrics = pass_at_k_dict(counts, ks=(1, 10))
    metrics["n_problems"] = len(problems)
    metrics["n_samples"] = sum(n for n, _ in counts.values())

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            for row in all_samples:
                fh.write(json.dumps(row) + "\n")

    return metrics


# Re-exported for downstream modules (humaneval_fim) that need the same
# sandbox semantics for executing reconstructed programs.
__all__ = [
    "run_mbpp",
    "run_in_subprocess",
    "ExecResult",
    "build_mbpp_prompt",
    "DEFAULT_TIMEOUT",
]

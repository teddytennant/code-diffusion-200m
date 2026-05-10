"""HumanEval pass@k.

Runs HumanEval against any object satisfying the Sampler protocol and
reports pass@1 / pass@10 with the standard unbiased estimator.

This module imports ``human_eval`` *lazily* inside functions so that the
evaluation package can be imported on machines that don't have it (CI,
unit tests). When evaluation is actually requested and the package is
missing, a clear error message points to the install instructions.
"""

from __future__ import annotations

import gzip
import json
import os
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from src.eval.sampler_interface import Sampler

# Source: https://github.com/openai/human-eval
HUMANEVAL_URL = (
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
)
HUMANEVAL_CACHE = (
    Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    / "code-diffusion-200m"
    / "HumanEval.jsonl.gz"
)

_HUMAN_EVAL_INSTALL_HINT = (
    "Install with `pip install git+https://github.com/openai/human-eval.git`. "
    "Per the upstream README you must also un-comment the `exec(check_program, ...)` "
    "line in human_eval/execution.py before running real evaluations."
)


# ---------------------------------------------------------------------------
# pass@k


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator for pass@k from Chen et al. 2021.

    ``n`` total samples, ``c`` correct, evaluated at ``k``. Uses the
    numerically stable product formulation from the HumanEval paper to avoid
    overflow for large n.
    """
    if n - c < k:
        return 1.0
    # Equivalent to 1 - C(n-c, k) / C(n, k), computed iteratively.
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def pass_at_k_dict(
    counts: dict[str, tuple[int, int]], ks: Iterable[int]
) -> dict[str, float]:
    """Aggregate pass@k over many problems.

    ``counts`` maps problem id to ``(n_samples, n_correct)``. Returns a dict
    ``{f"pass@{k}": mean over problems}``.
    """
    out: dict[str, float] = {}
    for k in ks:
        per_problem = []
        for n, c in counts.values():
            if n < k:
                # pass@k undefined when k > n; skip rather than silently inflate.
                continue
            per_problem.append(pass_at_k(n, c, k))
        out[f"pass@{k}"] = (sum(per_problem) / len(per_problem)) if per_problem else 0.0
    return out


# ---------------------------------------------------------------------------
# Problem loading


def _download_humaneval(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(HUMANEVAL_URL) as resp, open(dest, "wb") as fh:
        fh.write(resp.read())


def load_humaneval_problems(problems_path: Path | None = None) -> list[dict]:
    """Load HumanEval problems, downloading to the user cache if absent."""
    path = problems_path or HUMANEVAL_CACHE
    if not path.exists():
        if problems_path is not None:
            raise FileNotFoundError(f"HumanEval problems not found at {path}")
        _download_humaneval(path)
    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# check_correctness shim


def _import_check_correctness():
    """Lazy import of ``human_eval.execution.check_correctness``.

    Raises a friendly ImportError pointing at install instructions when the
    upstream package is missing.
    """
    try:
        from human_eval.execution import check_correctness  # type: ignore
    except ImportError:
        from src.eval._humaneval_vendor import check_correctness
    return check_correctness


# ---------------------------------------------------------------------------
# Public entry point


def run_humaneval(
    sampler: Sampler,
    n_samples_per_problem: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    output_path: Path | None = None,
    problems_path: Path | None = None,
    timeout: float = 3.0,
) -> dict:
    """Run HumanEval pass@1 and pass@10.

    Each problem is sampled ``n_samples_per_problem`` times. Each sample is
    handed to ``human_eval.execution.check_correctness`` which executes the
    candidate program in a subprocess with the unit tests appended. Raw
    samples are written to ``output_path`` as JSONL for inspection.
    """
    check_correctness = _import_check_correctness()
    problems = load_humaneval_problems(problems_path)

    all_samples: list[dict] = []
    counts: dict[str, tuple[int, int]] = {}

    for problem in problems:
        task_id = problem["task_id"]
        completions = sampler.sample(
            problem["prompt"],
            max_new_tokens=max_new_tokens,
            mode="completion",
            n_samples=n_samples_per_problem,
            temperature=temperature,
        )
        n_correct = 0
        for i, completion in enumerate(completions):
            result = check_correctness(problem, completion, timeout=timeout)
            passed = bool(result.get("passed"))
            n_correct += int(passed)
            all_samples.append(
                {
                    "task_id": task_id,
                    "completion_idx": i,
                    "completion": completion,
                    "passed": passed,
                    "result": result.get("result", ""),
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

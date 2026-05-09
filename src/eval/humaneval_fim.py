"""HumanEval-FIM: fill-in-the-middle evaluation on HumanEval.

Three variants:

* ``single_line`` — delete one random non-empty body line.
* ``multi_line`` — delete a contiguous span of 2-4 body lines.
* ``random_span`` — delete a random character span 20-80 chars long, where
  the cut starts on a non-word boundary (so we never split a token).

For each problem we feed the model the surrounding code as a StarCoder2-style
FIM prompt (``<fim_prefix>``...``<fim_suffix>``...``<fim_middle>``), let the
sampler infill the gap, glue the result back, and run the original HumanEval
unit tests against the reconstructed program.

Reported metrics:

* ``exact_match`` — fraction of samples that exactly equal the deleted span.
* ``pass@1``     — fraction of problems where at least one sample yields a
  reconstructed program that passes the canonical tests.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.eval.humaneval import (
    _import_check_correctness,
    load_humaneval_problems,
    pass_at_k_dict,
)
from src.eval.sampler_interface import Sampler, build_fim_prompt

FimVariant = Literal["single_line", "multi_line", "random_span"]


@dataclass
class FimCut:
    """The result of slicing a piece of source into prefix / middle / suffix."""

    prefix: str
    middle: str
    suffix: str

    def reconstruct(self) -> str:
        return self.prefix + self.middle + self.suffix

    def fim_prompt(self) -> str:
        return build_fim_prompt(self.prefix, self.suffix)


# ---------------------------------------------------------------------------
# Variant cutters
# ---------------------------------------------------------------------------


def cut_single_line(code: str, rng: random.Random) -> FimCut:
    """Remove one random non-empty line from ``code``.

    Lines are kept with their trailing newline so prefix + middle + suffix
    is byte-identical to the input.
    """
    lines = code.splitlines(keepends=True)
    candidate_idxs = [i for i, ln in enumerate(lines) if ln.strip()]
    if not candidate_idxs:
        # Degenerate: no non-empty lines. Cut the first line if any, else
        # the empty middle (which makes for a trivial test case).
        if lines:
            idx = 0
        else:
            return FimCut("", "", "")
    else:
        idx = rng.choice(candidate_idxs)
    prefix = "".join(lines[:idx])
    middle = lines[idx]
    suffix = "".join(lines[idx + 1 :])
    return FimCut(prefix, middle, suffix)


def cut_multi_line(code: str, rng: random.Random) -> FimCut:
    """Remove a 2-4 line contiguous span. Falls back to single-line."""
    lines = code.splitlines(keepends=True)
    if len(lines) < 2:
        return cut_single_line(code, rng)
    span = rng.randint(2, min(4, len(lines)))
    start = rng.randint(0, len(lines) - span)
    prefix = "".join(lines[:start])
    middle = "".join(lines[start : start + span])
    suffix = "".join(lines[start + span :])
    return FimCut(prefix, middle, suffix)


_WORD_CHAR = re.compile(r"\w")


def cut_random_span(
    code: str, rng: random.Random, min_len: int = 20, max_len: int = 80
) -> FimCut:
    """Remove a random char span, length in ``[min_len, max_len]``.

    The start index is constrained so we don't split a word: the character
    immediately before ``start`` must not be a word character (or ``start``
    must be 0). If no valid start exists we fall back to a single-line cut.
    """
    if len(code) <= min_len + 1:
        return cut_single_line(code, rng)

    upper = min(max_len, len(code) - 1)
    if upper < min_len:
        return cut_single_line(code, rng)

    # Build the set of valid start positions: index 0 or any index where
    # the previous character is non-word.
    valid_starts = [
        i
        for i in range(0, len(code) - min_len)
        if i == 0 or not _WORD_CHAR.match(code[i - 1])
    ]
    if not valid_starts:
        return cut_single_line(code, rng)

    start = rng.choice(valid_starts)
    span_len = rng.randint(min_len, min(upper, len(code) - start))
    prefix = code[:start]
    middle = code[start : start + span_len]
    suffix = code[start + span_len :]
    return FimCut(prefix, middle, suffix)


_CUTTERS = {
    "single_line": cut_single_line,
    "multi_line": cut_multi_line,
    "random_span": cut_random_span,
}


def make_fim_cut(variant: FimVariant, code: str, rng: random.Random) -> FimCut:
    """Dispatch to the cutter for ``variant``."""
    if variant not in _CUTTERS:
        raise ValueError(
            f"Unknown FIM variant {variant!r}; expected one of {list(_CUTTERS)}"
        )
    return _CUTTERS[variant](code, rng)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_humaneval_fim(
    sampler: Sampler,
    variant: FimVariant,
    n_samples_per_problem: int = 5,
    output_path: Path | None = None,
    problems_path: Path | None = None,
    problems: list[dict] | None = None,
    seed: int = 0,
    timeout: float = 3.0,
) -> dict:
    """Run a single FIM variant on HumanEval.

    ``problems`` is an injection point for tests; when None the dataset is
    loaded (and possibly downloaded) from disk.
    """
    check_correctness = _import_check_correctness()
    if problems is None:
        problems = load_humaneval_problems(problems_path)

    rng = random.Random(seed)

    all_samples: list[dict] = []
    n_exact = 0
    n_total_samples = 0
    counts: dict[str, tuple[int, int]] = {}

    for problem in problems:
        task_id = problem["task_id"]
        full_code = problem["prompt"] + problem["canonical_solution"]
        cut = make_fim_cut(variant, full_code, rng)
        fim_prompt = cut.fim_prompt()

        completions = sampler.sample(
            fim_prompt,
            max_new_tokens=max(64, len(cut.middle) * 2),
            mode="fim",
            n_samples=n_samples_per_problem,
        )

        n_correct = 0
        for i, sample in enumerate(completions):
            n_total_samples += 1
            exact = sample == cut.middle
            n_exact += int(exact)

            reconstructed = cut.prefix + sample + cut.suffix
            # check_correctness expects the model's *completion* concatenated
            # to problem["prompt"]. Here we already have the full glued
            # program, so we strip the prompt back off and pass the rest.
            if reconstructed.startswith(problem["prompt"]):
                completion = reconstructed[len(problem["prompt"]) :]
            else:
                # Cut crossed into the prompt region; pass an empty
                # completion so the eval sees the prompt's own canonical
                # form rather than a corrupted one.
                completion = ""

            result = check_correctness(problem, completion, timeout=timeout)
            passed = bool(result.get("passed"))
            n_correct += int(passed)
            all_samples.append(
                {
                    "task_id": task_id,
                    "variant": variant,
                    "completion_idx": i,
                    "removed": cut.middle,
                    "sample": sample,
                    "exact_match": exact,
                    "passed": passed,
                }
            )
        counts[task_id] = (len(completions), n_correct)

    metrics = pass_at_k_dict(counts, ks=(1,))
    metrics["exact_match"] = (n_exact / n_total_samples) if n_total_samples else 0.0
    metrics["n_problems"] = len(problems)
    metrics["n_samples"] = n_total_samples
    metrics["variant"] = variant

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            for row in all_samples:
                fh.write(json.dumps(row) + "\n")

    return metrics


def run_all_fim_variants(
    sampler: Sampler,
    n_samples_per_problem: int = 5,
    output_dir: Path | None = None,
    problems_path: Path | None = None,
    problems: list[dict] | None = None,
    seed: int = 0,
) -> dict[str, dict]:
    """Run all three FIM variants and return ``{variant: metrics}``."""
    out: dict[str, dict] = {}
    for variant in ("single_line", "multi_line", "random_span"):
        path = (
            (Path(output_dir) / f"humaneval_fim_{variant}.jsonl")
            if output_dir is not None
            else None
        )
        out[variant] = run_humaneval_fim(
            sampler,
            variant,  # type: ignore[arg-type]
            n_samples_per_problem=n_samples_per_problem,
            output_path=path,
            problems_path=problems_path,
            problems=problems,
            seed=seed,
        )
    return out

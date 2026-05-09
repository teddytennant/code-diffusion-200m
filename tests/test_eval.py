"""CPU-only unit tests for the evaluation harness.

All tests use a mock Sampler. Nothing is downloaded; the HumanEval package
is faked at the ``sys.modules`` level so the tests run on machines that
don't have the real package installed (network/install is forbidden in CI).

Anything that needs network is decorated ``@pytest.mark.network`` and
skipped by default.
"""

from __future__ import annotations

import math
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

import pytest

# ---------------------------------------------------------------------------
# Make `import src.eval...` resolvable when pytest is invoked from the
# project root (which is the documented invocation).

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Mock sampler


class MockSampler:
    """Minimal Sampler: returns a fixed string ``return_text`` n_samples times."""

    def __init__(
        self,
        return_text: str = "    return 0\n",
        latency_per_step_s: float = 0.0,
    ) -> None:
        self.return_text = return_text
        self.latency_per_step_s = latency_per_step_s
        self.calls: list[dict[str, Any]] = []

    def sample(
        self,
        prompt: str,
        max_new_tokens: int,
        mode: str = "completion",
        n_samples: int = 1,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> list[str]:
        self.calls.append(
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "mode": mode,
                "n_samples": n_samples,
                "temperature": temperature,
                **kwargs,
            }
        )
        steps = int(kwargs.get("diffusion_steps", 1))
        if self.latency_per_step_s:
            time.sleep(self.latency_per_step_s * steps)
        return [self.return_text] * n_samples


class FuncSampler:
    """Sampler that delegates to a per-call function — handy for FIM tests."""

    def __init__(self, fn: Callable[[str, int, str, int], list[str]]) -> None:
        self.fn = fn

    def sample(
        self,
        prompt: str,
        max_new_tokens: int,
        mode: str = "completion",
        n_samples: int = 1,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> list[str]:
        return self.fn(prompt, max_new_tokens, mode, n_samples)


# ---------------------------------------------------------------------------
# Fake `human_eval.execution` module
#
# The real upstream package may not be installed (and the Anthropic CI box
# never installs anything during the test run). We inject a minimal stand-in
# whose `check_correctness` runs the candidate program plus the problem's
# test in a regular subprocess. That's good enough for the smoke test where
# the mock sampler returns the canonical solution.


def _install_fake_human_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess as _sp
    import sys as _sys

    def check_correctness(problem: dict, completion: str, timeout: float = 3.0):
        program = problem["prompt"] + completion + "\n" + problem.get("test", "")
        # The real check_correctness expects a `check(<entry_point>)` call at
        # the end of `problem["test"]`; HumanEval problems already include
        # one, so we just run the assembled program.
        check_program = program + f"\ncheck({problem['entry_point']})\n"
        try:
            done = _sp.run(
                [_sys.executable, "-I", "-c", check_program],
                capture_output=True,
                timeout=timeout,
                text=True,
            )
        except _sp.TimeoutExpired:
            return {"passed": False, "result": "timeout"}
        passed = done.returncode == 0
        return {"passed": passed, "result": "passed" if passed else done.stderr}

    fake_exec = types.ModuleType("human_eval.execution")
    fake_exec.check_correctness = check_correctness  # type: ignore[attr-defined]
    fake_pkg = types.ModuleType("human_eval")
    fake_pkg.execution = fake_exec  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "human_eval", fake_pkg)
    monkeypatch.setitem(sys.modules, "human_eval.execution", fake_exec)


# ---------------------------------------------------------------------------
# Test 1: pass@k formula


def test_pass_at_k_computation() -> None:
    from src.eval.humaneval import pass_at_k

    # Edge cases.
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(5, 5, 5) == 1.0

    # Closed-form sanity: pass@1 == c / n.
    assert math.isclose(pass_at_k(10, 3, 1), 0.3, abs_tol=1e-9)
    assert math.isclose(pass_at_k(20, 5, 1), 0.25, abs_tol=1e-9)

    # k > n - c shortcut: with n=4, c=2, k=3, the n-c=2 < k=3 branch fires
    # and we should return 1.0 (every choice of 3 from 4 must include at
    # least one correct).
    assert pass_at_k(4, 2, 3) == 1.0

    # General case: pass@k = 1 - C(n-c, k)/C(n, k). For n=10, c=1, k=2:
    # = 1 - C(9, 2)/C(10, 2) = 1 - 36/45 = 0.2.
    assert math.isclose(pass_at_k(10, 1, 2), 0.2, abs_tol=1e-9)

    # Aggregation helper.
    from src.eval.humaneval import pass_at_k_dict

    counts = {"a": (10, 10), "b": (10, 0)}
    agg = pass_at_k_dict(counts, ks=(1, 10))
    assert math.isclose(agg["pass@1"], 0.5)
    assert math.isclose(agg["pass@10"], 0.5)


# ---------------------------------------------------------------------------
# Test 2: HumanEval smoke


# A pair of tiny HumanEval-style problems whose canonical solutions are
# trivial. We construct them inline so no dataset is touched.
TINY_PROBLEMS = [
    {
        "task_id": "Tiny/1",
        "prompt": "def add(a, b):\n    ",
        "canonical_solution": "return a + b\n",
        "test": (
            "def check(candidate):\n"
            "    assert candidate(1, 2) == 3\n"
            "    assert candidate(-1, 1) == 0\n"
        ),
        "entry_point": "add",
    },
    {
        "task_id": "Tiny/2",
        "prompt": "def neg(x):\n    ",
        "canonical_solution": "return -x\n",
        "test": (
            "def check(candidate):\n"
            "    assert candidate(0) == 0\n"
            "    assert candidate(7) == -7\n"
        ),
        "entry_point": "neg",
    },
]


class CanonicalSampler:
    """Returns the canonical solution for whichever HumanEval problem matches."""

    def __init__(self, problems: list[dict]) -> None:
        self._by_prompt = {p["prompt"]: p["canonical_solution"] for p in problems}

    def sample(
        self,
        prompt: str,
        max_new_tokens: int,
        mode: str = "completion",
        n_samples: int = 1,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> list[str]:
        return [self._by_prompt[prompt]] * n_samples


def test_humaneval_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_human_eval(monkeypatch)
    # Force the lazy import inside humaneval.py to hit our fake module.
    import importlib

    import src.eval.humaneval as humaneval_mod

    importlib.reload(humaneval_mod)

    # Patch problem loading so the test never touches disk or network.
    monkeypatch.setattr(
        humaneval_mod, "load_humaneval_problems", lambda *a, **k: TINY_PROBLEMS
    )

    sampler = CanonicalSampler(TINY_PROBLEMS)
    metrics = humaneval_mod.run_humaneval(
        sampler,
        n_samples_per_problem=2,
        max_new_tokens=64,
    )

    assert metrics["n_problems"] == 2
    assert metrics["n_samples"] == 4
    assert metrics["pass@1"] == 1.0


# ---------------------------------------------------------------------------
# Test 3: HumanEval-FIM construction


@pytest.fixture(scope="module")
def fim_problem() -> dict:
    return {
        "task_id": "Tiny/FIM",
        "prompt": "def is_even(n):\n",
        "canonical_solution": "    if n % 2 == 0:\n        return True\n    return False\n",
        "test": (
            "def check(candidate):\n"
            "    assert candidate(2) is True\n"
            "    assert candidate(3) is False\n"
        ),
        "entry_point": "is_even",
    }


def test_humaneval_fim_construction(
    monkeypatch: pytest.MonkeyPatch, fim_problem: dict
) -> None:
    """Verify cuts produce a well-formed FIM prompt and round-trip cleanly."""
    _install_fake_human_eval(monkeypatch)
    import importlib

    import src.eval.humaneval as humaneval_mod

    importlib.reload(humaneval_mod)
    import src.eval.humaneval_fim as fim_mod

    importlib.reload(fim_mod)

    from src.eval.sampler_interface import FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX

    full_code = fim_problem["prompt"] + fim_problem["canonical_solution"]

    import random

    rng = random.Random(0)

    for variant in ("single_line", "multi_line", "random_span"):
        cut = fim_mod.make_fim_cut(variant, full_code, rng)  # type: ignore[arg-type]
        # Round-trip identity.
        assert cut.reconstruct() == full_code, (
            f"variant {variant} did not round-trip"
        )
        # FIM markers are present and in the right order.
        prompt = cut.fim_prompt()
        i_pre = prompt.find(FIM_PREFIX)
        i_suf = prompt.find(FIM_SUFFIX)
        i_mid = prompt.find(FIM_MIDDLE)
        assert i_pre == 0
        assert 0 < i_suf < i_mid
        # Concatenating prefix+suffix from the cut equals full_code minus the middle.
        assert cut.prefix + cut.suffix == full_code.replace(cut.middle, "", 1)

    # End-to-end: a FIM sampler that always returns the canonical middle
    # should achieve pass@1 == 1.0 and exact_match == 1.0 on every variant.
    def _fim_sampler_factory(cuts_seen: list):
        def fn(prompt: str, max_new_tokens: int, mode: str, n_samples: int):
            assert mode == "fim"
            assert FIM_PREFIX in prompt and FIM_SUFFIX in prompt and FIM_MIDDLE in prompt
            # Pull the middle out by re-parsing the prompt.
            middle_marker_idx = prompt.index(FIM_MIDDLE)
            assert middle_marker_idx == len(prompt) - len(FIM_MIDDLE)
            # Recover the prefix and suffix from the prompt to confirm the
            # structure (used only for the assert above).
            prefix = prompt[len(FIM_PREFIX) : prompt.index(FIM_SUFFIX)]
            suffix = prompt[prompt.index(FIM_SUFFIX) + len(FIM_SUFFIX) : middle_marker_idx]
            # The harness fed us cut.prefix and cut.suffix; recover the
            # original middle from the captured cut.
            cut = cuts_seen[-1]
            assert cut.prefix == prefix
            assert cut.suffix == suffix
            return [cut.middle] * n_samples

        return fn

    # Re-cut deterministically and tee the cut into a list so the sampler
    # can echo the canonical middle.
    cuts_seen: list = []
    real_make_cut = fim_mod.make_fim_cut

    def teeing_make_cut(variant, code, rng):
        c = real_make_cut(variant, code, rng)
        cuts_seen.append(c)
        return c

    monkeypatch.setattr(fim_mod, "make_fim_cut", teeing_make_cut)

    metrics = fim_mod.run_humaneval_fim(
        FuncSampler(_fim_sampler_factory(cuts_seen)),
        variant="single_line",
        n_samples_per_problem=2,
        problems=[fim_problem],
        seed=0,
    )
    assert metrics["exact_match"] == 1.0
    assert metrics["pass@1"] == 1.0
    assert metrics["n_problems"] == 1
    assert metrics["n_samples"] == 2


# ---------------------------------------------------------------------------
# Test 4: MBPP smoke


def test_mbpp_smoke() -> None:
    from src.eval.mbpp import run_mbpp

    tiny = [
        {
            "task_id": 1,
            "text": "Write a function to add two numbers.",
            "code": "def add(a, b):\n    return a + b\n",
            "test_list": [
                "assert add(1, 2) == 3",
                "assert add(-1, 1) == 0",
            ],
        },
        {
            "task_id": 2,
            "text": "Write a function to negate.",
            "code": "def neg(x):\n    return -x\n",
            "test_list": [
                "assert neg(7) == -7",
                "assert neg(0) == 0",
            ],
        },
    ]

    # Sampler that returns the correct code for whichever description it sees.
    by_text = {p["text"]: p["code"] for p in tiny}

    class CodeSampler:
        def sample(
            self,
            prompt: str,
            max_new_tokens: int,
            mode: str = "completion",
            n_samples: int = 1,
            temperature: float = 0.2,
            **kwargs: Any,
        ) -> list[str]:
            for txt, code in by_text.items():
                if txt in prompt:
                    return [code] * n_samples
            return [""] * n_samples

    metrics = run_mbpp(
        CodeSampler(),
        n_samples_per_problem=2,
        problems=tiny,
    )
    assert metrics["n_problems"] == 2
    assert metrics["n_samples"] == 4
    assert metrics["pass@1"] == 1.0


def test_mbpp_smoke_failures_are_caught() -> None:
    """A wrong sampler should produce pass@1 == 0."""
    from src.eval.mbpp import run_mbpp

    tiny = [
        {
            "task_id": 1,
            "text": "Add two numbers.",
            "code": "",
            "test_list": ["assert add(1, 2) == 3"],
        }
    ]

    class WrongSampler:
        def sample(self, prompt: str, max_new_tokens: int, **kw: Any) -> list[str]:
            return ["def add(a, b):\n    return a - b\n"] * kw.get("n_samples", 1)

    metrics = run_mbpp(WrongSampler(), n_samples_per_problem=2, problems=tiny)
    assert metrics["pass@1"] == 0.0


# ---------------------------------------------------------------------------
# Test 5: throughput DataFrame


def test_throughput_dataframe() -> None:
    from src.eval.throughput import benchmark_throughput

    sampler = MockSampler(
        return_text="def x():\n    return 1\n",
        latency_per_step_s=0.001,  # 1ms per diffusion step → cheap, deterministic
    )

    prompts = ["def f():\n    "] * 2
    df = benchmark_throughput(
        sampler,
        prompts=prompts,
        diffusion_steps_sweep=[2, 4, 8],
        max_new_tokens=32,
    )

    # Schema check.
    assert list(df.columns) == [
        "steps",
        "tokens_per_sec",
        "parse_rate",
        "n_generations",
        "wall_seconds",
    ]
    assert len(df) == 3
    assert list(df["steps"]) == [2, 4, 8]

    # All generations parse (we returned valid Python).
    assert (df["parse_rate"] == 1.0).all()

    # tokens_per_sec is positive and finite.
    assert (df["tokens_per_sec"] > 0).all()
    assert df["tokens_per_sec"].apply(lambda x: x != float("inf")).all()

    # Each step setting calls the sampler ``len(prompts)`` times.
    assert len(sampler.calls) == 3 * len(prompts)
    # ``diffusion_steps`` is forwarded in **kwargs.
    assert sampler.calls[0]["diffusion_steps"] in (2, 4, 8)


def test_throughput_parse_rate_for_garbage() -> None:
    """A sampler returning non-Python text should produce parse_rate == 0."""
    from src.eval.throughput import benchmark_throughput

    sampler = MockSampler(return_text="this is %% not valid python @@@")
    df = benchmark_throughput(
        sampler,
        prompts=["foo"],
        diffusion_steps_sweep=[1],
        max_new_tokens=16,
    )
    assert df.loc[0, "parse_rate"] == 0.0


# ---------------------------------------------------------------------------
# Bonus: sampler_interface FIM helper sanity (no network)


def test_build_fim_prompt_layout() -> None:
    from src.eval.sampler_interface import (
        FIM_MIDDLE,
        FIM_PREFIX,
        FIM_SUFFIX,
        build_fim_prompt,
    )

    p = build_fim_prompt("PRE", "SUF")
    assert p == f"{FIM_PREFIX}PRE{FIM_SUFFIX}SUF{FIM_MIDDLE}"


# ---------------------------------------------------------------------------
# Network-gated test (skipped by default)


@pytest.mark.network
def test_humaneval_real_dataset_download() -> None:  # pragma: no cover
    """Sanity check that the loader can pull the real dataset.

    Skipped by default; run with ``pytest -m network`` when wired up.
    """
    pytest.skip("network test, run with -m network")


# A tiny meta-test: importing every public eval module should be cheap and
# never blow up on a clean machine. Keeps the harness from accidentally
# pulling in heavy deps at import time.

def test_imports_are_cheap() -> None:
    import importlib

    for mod in (
        "src.eval",
        "src.eval.sampler_interface",
        "src.eval.humaneval",
        "src.eval.humaneval_fim",
        "src.eval.mbpp",
        "src.eval.throughput",
        "src.eval.runner",
    ):
        importlib.import_module(mod)

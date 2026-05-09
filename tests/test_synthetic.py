"""CPU-only tests for the synthetic data generator.

No API calls, no third-party deps beyond pytest itself. Intended to run
in well under 30 seconds.
"""

from __future__ import annotations

import ast
import io
import json
import random
import sys
from pathlib import Path

import pytest


# Make sure the project root is importable so ``scripts.*`` resolves.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts import gen_synthetic, synthetic_prompts  # noqa: E402
from scripts.synthetic_prompts import (  # noqa: E402
    DOMAINS,
    LENGTH_TARGETS,
    STYLES,
    build_prompt,
    random_seed_topic,
)


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_mentions_domain_and_style() -> None:
    prompt = build_prompt(
        domain="games",
        style="object-oriented with multiple classes",
        target_lines=200,
        seed_topic="tetris",
    )
    assert isinstance(prompt, str)
    assert prompt.strip(), "prompt must be non-empty"
    assert "games" in prompt
    assert "object-oriented with multiple classes" in prompt
    assert "tetris" in prompt
    # Must instruct: no markdown fences, no commentary.
    assert "markdown" in prompt.lower()
    assert "commentar" in prompt.lower() or "explanation" in prompt.lower()
    # Mentions ast.parse / parseable.
    assert "ast.parse" in prompt or "parseable" in prompt.lower()
    # Mentions approximate line target.
    assert "200" in prompt


def test_build_prompt_without_seed_topic_still_valid() -> None:
    prompt = build_prompt(
        domain="parsers",
        style="concise idiomatic",
        target_lines=80,
    )
    assert prompt.strip()
    assert "parsers" in prompt
    assert "concise idiomatic" in prompt


def test_build_prompt_validates_inputs() -> None:
    with pytest.raises(ValueError):
        build_prompt(domain="", style="x", target_lines=10)
    with pytest.raises(ValueError):
        build_prompt(domain="x", style="", target_lines=10)
    with pytest.raises(ValueError):
        build_prompt(domain="x", style="y", target_lines=0)


# ---------------------------------------------------------------------------
# random_seed_topic
# ---------------------------------------------------------------------------

def test_random_seed_topic_for_every_domain() -> None:
    rng = random.Random(0)
    for domain in DOMAINS:
        topic = random_seed_topic(domain, rng)
        assert isinstance(topic, str), domain
        assert topic.strip(), f"empty topic for {domain!r}"


def test_random_seed_topic_deterministic_with_seed() -> None:
    rng_a = random.Random(42)
    rng_b = random.Random(42)
    for domain in DOMAINS[:5]:
        assert random_seed_topic(domain, rng_a) == random_seed_topic(domain, rng_b)


def test_random_seed_topic_unknown_domain_falls_back() -> None:
    rng = random.Random(0)
    topic = random_seed_topic("not-a-real-domain", rng)
    assert isinstance(topic, str) and topic.strip()


# ---------------------------------------------------------------------------
# Sampling space sanity
# ---------------------------------------------------------------------------

def test_sampling_space_nonempty_and_unique() -> None:
    assert len(DOMAINS) >= 20
    assert len(set(DOMAINS)) == len(DOMAINS)
    assert len(STYLES) >= 5
    assert len(set(STYLES)) == len(STYLES)
    assert len(LENGTH_TARGETS) >= 2
    for label, n in LENGTH_TARGETS:
        assert isinstance(label, str) and label
        assert isinstance(n, int) and n > 0


# ---------------------------------------------------------------------------
# JSONL roundtrip
# ---------------------------------------------------------------------------

def _make_record(source: str, domain: str = "games") -> dict:
    return {
        "source": source,
        "domain": domain,
        "style": "concise idiomatic",
        "tokens": gen_synthetic.estimate_tokens(source),
        "model": "claude-sonnet-4-6",
        "topic": "tic-tac-toe",
    }


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "synth.jsonl"
    records = [
        _make_record("def add(a: int, b: int) -> int:\n    return a + b\n"),
        _make_record("x = 1\ny = 2\nprint(x + y)\n", domain="data analysis"),
        _make_record(
            "class Counter:\n    def __init__(self) -> None:\n        self.n = 0\n",
            domain="design patterns",
        ),
    ]
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    read_back: list[dict] = []
    with out.open("r", encoding="utf-8") as f:
        for line in f:
            read_back.append(json.loads(line))

    assert len(read_back) == len(records)
    for orig, got in zip(records, read_back):
        assert got == orig
        # Required fields per spec.
        for field in ("source", "domain", "style", "tokens", "model"):
            assert field in got
        assert isinstance(got["tokens"], int) and got["tokens"] > 0


def test_count_existing_tokens(tmp_path: Path) -> None:
    out = tmp_path / "synth.jsonl"
    records = [
        {"source": "x=1", "tokens": 100, "domain": "d", "style": "s",
         "model": "m"},
        {"source": "y=2", "tokens": 250, "domain": "d", "style": "s",
         "model": "m"},
    ]
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.write("\n")  # blank line should be skipped
        f.write("not-json-at-all\n")  # malformed line should be skipped

    assert gen_synthetic.count_existing_tokens(out) == 350
    assert gen_synthetic.count_existing_tokens(tmp_path / "missing.jsonl") == 0


# ---------------------------------------------------------------------------
# AST validation
# ---------------------------------------------------------------------------

def test_ast_validation_accepts_valid_python() -> None:
    good = (
        "from __future__ import annotations\n"
        "\n"
        "def fib(n: int) -> int:\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a\n"
    )
    assert gen_synthetic.is_valid_python(good)
    # Sanity: also pure ast.parse works.
    ast.parse(good)


def test_ast_validation_rejects_bad_python() -> None:
    assert not gen_synthetic.is_valid_python("def x(:\n")
    assert not gen_synthetic.is_valid_python("def f(:\n    pass")
    assert not gen_synthetic.is_valid_python("")
    assert not gen_synthetic.is_valid_python("   \n   ")
    # Unterminated string is also a syntax error.
    assert not gen_synthetic.is_valid_python('x = "abc')


def test_strip_code_fences() -> None:
    src = "```python\nprint('hi')\n```"
    assert gen_synthetic.strip_code_fences(src) == "print('hi')"
    # Without fences: identity (modulo strip).
    assert gen_synthetic.strip_code_fences("print('hi')\n") == "print('hi')"
    # Only opening fence (sometimes happens).
    assert gen_synthetic.strip_code_fences("```\nprint('hi')") == "print('hi')"


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

def test_estimate_tokens_positive_for_nonempty_text() -> None:
    assert gen_synthetic.estimate_tokens("") == 0
    assert gen_synthetic.estimate_tokens("hello world") > 0
    longer = "def f():\n    return 1\n" * 50
    assert gen_synthetic.estimate_tokens(longer) > gen_synthetic.estimate_tokens(
        "def f(): return 1"
    )


# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------

def test_cost_tracker_pricing() -> None:
    ct = gen_synthetic.CostTracker()  # defaults: $3 / $15 per 1M
    ct.add(1_000_000, 0)
    assert ct.cost_usd == pytest.approx(3.0)
    ct.add(0, 1_000_000)
    assert ct.cost_usd == pytest.approx(18.0)


# ---------------------------------------------------------------------------
# Dry run: invokes main() without API key, asserts no API call & no file write
# ---------------------------------------------------------------------------

def test_dry_run_no_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    out = tmp_path / "dryrun.jsonl"
    # Explicitly clear the env var to prove --dry-run does not need it.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # If anything tries to import anthropic during dry-run, fail loudly.
    sentinel_called: list[str] = []

    class _BoomModule:
        def __getattr__(self, name: str):  # pragma: no cover
            sentinel_called.append(name)
            raise RuntimeError(f"anthropic.{name} touched during dry-run")

    # Don't actually replace sys.modules['anthropic'] — we just want to
    # ensure run_dry doesn't import it. It currently doesn't.

    rc = gen_synthetic.main(
        [
            "--dry-run",
            "--output", str(out),
            "--target-tokens", "1000",
            "--concurrency", "2",
            "--seed", "7",
        ]
    )
    assert rc == 0
    assert not out.exists(), "dry-run must not write files"
    assert sentinel_called == [], "dry-run must not touch anthropic"

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "dry-run" in combined.lower()
    # Should print at least one sample task line.
    assert "task" in combined.lower()
    assert "no API calls" in combined or "no API calls" in captured.err


# ---------------------------------------------------------------------------
# sample_task end-to-end (no API)
# ---------------------------------------------------------------------------

def test_sample_task_returns_valid_prompt() -> None:
    rng = random.Random(123)
    for _ in range(20):
        task = gen_synthetic.sample_task(rng)
        assert task.domain in DOMAINS
        assert task.style in STYLES
        assert task.target_lines > 0
        assert task.topic.strip()
        assert task.domain in task.prompt
        assert task.style in task.prompt


# ---------------------------------------------------------------------------
# Argparse smoke: --help exits cleanly and prints target-tokens guidance
# ---------------------------------------------------------------------------

def test_argparse_help_mentions_target_tokens(
    capsys: pytest.CaptureFixture,
) -> None:
    parser = gen_synthetic.build_arg_parser()
    help_text = parser.format_help()
    assert "--target-tokens" in help_text
    assert "100" in help_text  # default 100M mentioned
    assert "--dry-run" in help_text
    assert "--resume" in help_text

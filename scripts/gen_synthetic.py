"""Async CLI to generate synthetic Python source files via the Claude API.

Output format: JSONL, one record per line:

    {"source": "...", "domain": "games", "style": "...",
     "tokens": 1234, "model": "claude-sonnet-4-6", "topic": "tetris"}

The script tracks token volume and dollar cost, supports resume, retries
on 429/5xx, and provides a ``--dry-run`` mode that builds prompts but
does NOT call the API or write files. Heavy third-party imports
(``anthropic``, ``tiktoken``, ``tqdm``) are loaded lazily so this module
remains importable in test environments without those packages.

Usage::

    python scripts/gen_synthetic.py \\
      --output data/synthetic.jsonl \\
      --target-tokens 100000000 \\
      --model claude-sonnet-4-6 \\
      --concurrency 16 \\
      --max-cost-usd 500

Note: ``--target-tokens`` defaults to 100M. For a 3B-token corpus expect
roughly $10K at default Sonnet pricing.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

# Local prompt-building helpers (pure stdlib).
from scripts.synthetic_prompts import (
    DOMAINS,
    LENGTH_TARGETS,
    STYLES,
    SYSTEM_PROMPT,
    build_prompt,
    random_seed_topic,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("gen_synthetic")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    # Avoid duplicate handlers on repeated calls (tests).
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


# ---------------------------------------------------------------------------
# Token / cost accounting
# ---------------------------------------------------------------------------

# Lazy singletons so importing this module never imports tiktoken.
_TIKTOKEN_ENC: Any = None
_TIKTOKEN_TRIED: bool = False


def _get_tiktoken_encoder() -> Any | None:
    """Return a tiktoken encoder, or ``None`` if tiktoken isn't available."""
    global _TIKTOKEN_ENC, _TIKTOKEN_TRIED
    if _TIKTOKEN_TRIED:
        return _TIKTOKEN_ENC
    _TIKTOKEN_TRIED = True
    try:
        import tiktoken  # type: ignore[import-not-found]

        _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # pragma: no cover - depends on env
        logger.debug("tiktoken unavailable, falling back to whitespace: %s", exc)
        _TIKTOKEN_ENC = None
    return _TIKTOKEN_ENC


def estimate_tokens(text: str) -> int:
    """Approximate token count for ``text``.

    Uses tiktoken's ``cl100k_base`` if installed (close enough for budget
    tracking), otherwise a whitespace-split * 1.3 heuristic.
    """
    if not text:
        return 0
    enc = _get_tiktoken_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:  # pragma: no cover - defensive
            pass
    # Fallback: word count * 1.3, rounded up.
    words = text.split()
    return max(1, int(len(words) * 1.3 + 0.5))


@dataclass
class CostTracker:
    """Tracks input/output token usage and dollar cost.

    Default pricing reflects Sonnet rates ($3/M input, $15/M output).
    """

    input_per_mtok_usd: float = 3.0
    output_per_mtok_usd: float = 15.0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += max(0, input_tokens)
        self.output_tokens += max(0, output_tokens)

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000.0 * self.input_per_mtok_usd
            + self.output_tokens / 1_000_000.0 * self.output_per_mtok_usd
        )


# ---------------------------------------------------------------------------
# Generation task description
# ---------------------------------------------------------------------------

@dataclass
class GenTask:
    domain: str
    style: str
    length_label: str
    target_lines: int
    topic: str
    prompt: str


def sample_task(rng: random.Random) -> GenTask:
    """Pick a (domain, style, length) triple at random and build a prompt."""
    domain = rng.choice(DOMAINS)
    style = rng.choice(STYLES)
    length_label, target_lines = rng.choice(LENGTH_TARGETS)
    topic = random_seed_topic(domain, rng)
    prompt = build_prompt(domain, style, target_lines, topic)
    return GenTask(
        domain=domain,
        style=style,
        length_label=length_label,
        target_lines=target_lines,
        topic=topic,
        prompt=prompt,
    )


# ---------------------------------------------------------------------------
# Source post-processing
# ---------------------------------------------------------------------------

def strip_code_fences(text: str) -> str:
    """Strip optional ```python ... ``` fences if Claude added them anyway."""
    s = text.strip()
    if not s.startswith("```"):
        return s
    # Drop first fence line.
    lines = s.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    # Drop trailing fence line if present.
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def is_valid_python(source: str) -> bool:
    """True if ``source`` is parseable Python."""
    if not source or not source.strip():
        return False
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    except Exception:  # pragma: no cover - ast.parse rarely raises others
        return False
    return True


# ---------------------------------------------------------------------------
# Anthropic API calls (lazy imports)
# ---------------------------------------------------------------------------

class APIError(RuntimeError):
    """Raised for non-retryable API failures."""


class RetryableAPIError(RuntimeError):
    """Raised for retryable failures (429, 5xx, transient network)."""


async def _call_claude_once(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> tuple[str, int, int]:
    """Single API call. Returns (text, input_tokens, output_tokens)."""
    # Import here so module import never requires anthropic.
    import anthropic  # type: ignore[import-not-found]

    try:
        msg = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.RateLimitError as exc:
        raise RetryableAPIError(f"rate limit: {exc}") from exc
    except anthropic.APIStatusError as exc:
        status = getattr(exc, "status_code", None) or 0
        if status == 429 or 500 <= status < 600:
            raise RetryableAPIError(f"API {status}: {exc}") from exc
        raise APIError(f"API {status}: {exc}") from exc
    except anthropic.APIConnectionError as exc:
        raise RetryableAPIError(f"connection error: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise APIError(f"unexpected: {exc!r}") from exc

    # Extract text from content blocks.
    parts: list[str] = []
    for block in getattr(msg, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    text = "".join(parts)

    usage = getattr(msg, "usage", None)
    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or 0)
    return text, in_tok, out_tok


async def call_claude_with_retry(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    max_retries: int = 3,
    base_delay: float = 1.5,
    rng: random.Random | None = None,
) -> tuple[str, int, int]:
    """Call Claude with up to ``max_retries`` exponential-backoff retries."""
    rng = rng or random.Random()
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await _call_claude_once(
                client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
        except RetryableAPIError as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            delay = base_delay * (2**attempt) + rng.uniform(0, base_delay)
            logger.warning(
                "retryable API error (attempt %d/%d): %s — sleeping %.1fs",
                attempt + 1, max_retries, exc, delay,
            )
            await asyncio.sleep(delay)
        except APIError:
            raise
    assert last_exc is not None
    raise APIError(f"exhausted retries: {last_exc}")


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def count_existing_tokens(path: Path) -> int:
    """Sum the ``tokens`` field across an existing JSONL file."""
    if not path.exists():
        return 0
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tk = obj.get("tokens")
            if isinstance(tk, int) and tk > 0:
                total += tk
    return total


# ---------------------------------------------------------------------------
# Main async runner
# ---------------------------------------------------------------------------

@dataclass
class RunStats:
    successes: int = 0
    parse_failures: int = 0
    api_failures: int = 0
    written_tokens: int = 0
    cost: CostTracker = field(default_factory=CostTracker)

    @property
    def success_rate(self) -> float:
        attempts = self.successes + self.parse_failures + self.api_failures
        return (self.successes / attempts) if attempts else 0.0


async def _worker(
    *,
    name: str,
    sem: asyncio.Semaphore,
    stop_event: asyncio.Event,
    client: Any,
    model: str,
    rng: random.Random,
    write_lock: asyncio.Lock,
    out_fp: Any,
    stats: RunStats,
    target_tokens: int,
    max_cost_usd: float,
    pbar: Any | None,
) -> None:
    """Single worker coroutine. Runs until ``stop_event`` is set."""
    while not stop_event.is_set():
        # Cheap pre-check before acquiring sem.
        if stats.written_tokens >= target_tokens:
            stop_event.set()
            return
        if stats.cost.cost_usd >= max_cost_usd:
            logger.warning("worker %s: hit max_cost_usd, stopping", name)
            stop_event.set()
            return

        async with sem:
            if stop_event.is_set():
                return
            task = sample_task(rng)
            max_tokens = max(512, min(8192, task.target_lines * 30))

            try:
                text, in_tok, out_tok = await call_claude_with_retry(
                    client,
                    model=model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=task.prompt,
                    max_tokens=max_tokens,
                    rng=rng,
                )
            except APIError as exc:
                stats.api_failures += 1
                logger.warning("api failure (%s): %s", task.domain, exc)
                continue

            stats.cost.add(in_tok, out_tok)
            source = strip_code_fences(text)
            if not is_valid_python(source):
                stats.parse_failures += 1
                logger.debug("parse failure for %s/%s", task.domain, task.topic)
                continue

            tokens = estimate_tokens(source)
            record = {
                "source": source,
                "domain": task.domain,
                "style": task.style,
                "length_label": task.length_label,
                "tokens": tokens,
                "model": model,
                "topic": task.topic,
            }
            line = json.dumps(record, ensure_ascii=False) + "\n"
            async with write_lock:
                out_fp.write(line)
                out_fp.flush()
                stats.written_tokens += tokens
                stats.successes += 1
                if pbar is not None:
                    pbar.update(tokens)
                    pbar.set_postfix(
                        cost=f"${stats.cost.cost_usd:,.2f}",
                        ok=stats.successes,
                        bad=stats.parse_failures + stats.api_failures,
                        rate=f"{stats.success_rate:.1%}",
                    )


def _make_progress_bar(total: int, initial: int) -> Any | None:
    try:
        from tqdm import tqdm  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        return None
    return tqdm(
        total=total,
        initial=initial,
        unit="tok",
        unit_scale=True,
        smoothing=0.05,
        dynamic_ncols=True,
        file=sys.stderr,
    )


async def run_async(args: argparse.Namespace) -> int:
    """Async entry point. Returns process exit code."""
    rng = random.Random(args.seed)
    stop_event = asyncio.Event()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    initial_tokens = 0
    if args.resume and out_path.exists():
        initial_tokens = count_existing_tokens(out_path)
        logger.info("resume: %d tokens already in %s", initial_tokens, out_path)
    elif out_path.exists() and not args.resume:
        logger.warning(
            "output %s exists; appending. Pass --resume to count existing "
            "tokens toward target.",
            out_path,
        )

    # Lazy-import anthropic only when actually generating.
    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError as exc:
        logger.error("anthropic SDK is required: %s", exc)
        return 2

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY env var is not set")
        return 2

    client = anthropic.AsyncAnthropic(api_key=api_key)

    stats = RunStats(
        written_tokens=initial_tokens,
        cost=CostTracker(
            input_per_mtok_usd=args.input_price,
            output_per_mtok_usd=args.output_price,
        ),
    )

    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    pbar = _make_progress_bar(total=args.target_tokens, initial=initial_tokens)

    # Install Ctrl+C handler that flips the stop_event.
    loop = asyncio.get_running_loop()

    def _on_signal() -> None:
        logger.warning("signal received: draining workers and flushing")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except (NotImplementedError, RuntimeError):  # pragma: no cover
            pass

    mode = "a" if (args.resume and out_path.exists()) else "a"
    with out_path.open(mode, encoding="utf-8") as out_fp:
        workers = [
            asyncio.create_task(
                _worker(
                    name=f"w{i}",
                    sem=sem,
                    stop_event=stop_event,
                    client=client,
                    model=args.model,
                    rng=random.Random(args.seed + i if args.seed is not None else None),
                    write_lock=write_lock,
                    out_fp=out_fp,
                    stats=stats,
                    target_tokens=args.target_tokens,
                    max_cost_usd=args.max_cost_usd,
                    pbar=pbar,
                ),
                name=f"gen-worker-{i}",
            )
            for i in range(args.concurrency)
        ]
        try:
            await asyncio.gather(*workers, return_exceptions=False)
        finally:
            stop_event.set()
            out_fp.flush()
            if pbar is not None:
                pbar.close()
            try:
                await client.close()  # type: ignore[func-returns-value]
            except Exception:  # pragma: no cover
                pass

    logger.info(
        "done: %d ok, %d parse-fail, %d api-fail, %d tokens written, "
        "cost ~$%,.2f",
        stats.successes,
        stats.parse_failures,
        stats.api_failures,
        stats.written_tokens,
        stats.cost.cost_usd,
    )
    return 0


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def run_dry(args: argparse.Namespace, *, n_samples: int = 3) -> int:
    """Print what would be generated without calling the API or writing."""
    rng = random.Random(args.seed)
    print(
        f"[dry-run] output={args.output} target_tokens={args.target_tokens:,} "
        f"model={args.model} concurrency={args.concurrency} "
        f"max_cost_usd=${args.max_cost_usd:,.2f}",
        file=sys.stderr,
    )
    out_path = Path(args.output)
    if args.resume and out_path.exists():
        existing = count_existing_tokens(out_path)
        print(f"[dry-run] would resume from {existing:,} existing tokens",
              file=sys.stderr)
    print(f"[dry-run] would create {n_samples} sample tasks:", file=sys.stderr)
    for i in range(n_samples):
        task = sample_task(rng)
        print(
            f"  task {i+1}: domain={task.domain!r} style={task.style!r} "
            f"len={task.length_label}({task.target_lines}) topic={task.topic!r}",
            file=sys.stderr,
        )
    print("[dry-run] no API calls made; no files written.", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Argparse / main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gen_synthetic",
        description=(
            "Generate diverse synthetic Python files via the Claude API. "
            "Emits JSONL with one record per accepted file. NOTE: "
            "--target-tokens defaults to 100M; a 3B-token corpus needs "
            "roughly $10K at default Sonnet pricing."
        ),
    )
    p.add_argument(
        "--output", type=str, default="data/synthetic.jsonl",
        help="JSONL output path (default: data/synthetic.jsonl)",
    )
    p.add_argument(
        "--target-tokens", type=int, default=100_000_000,
        help="Stop after this many accepted tokens (default: 100,000,000).",
    )
    p.add_argument(
        "--model", type=str, default="claude-sonnet-4-6",
        help="Anthropic model id (default: claude-sonnet-4-6).",
    )
    p.add_argument(
        "--concurrency", type=int, default=16,
        help="Number of concurrent in-flight requests (default: 16).",
    )
    p.add_argument(
        "--max-cost-usd", type=float, default=500.0,
        help="Hard stop after this dollar cost (default: $500).",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="If output exists, count its tokens toward the target.",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible task sampling.",
    )
    p.add_argument(
        "--input-price", type=float, default=3.0,
        help="Input token price per million USD (default: 3.0 / Sonnet).",
    )
    p.add_argument(
        "--output-price", type=float, default=15.0,
        help="Output token price per million USD (default: 15.0 / Sonnet).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Build prompts and report plan; do NOT call API or write files.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose (DEBUG) logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Module-level entry point. Returns process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.dry_run:
        return run_dry(args)

    try:
        return asyncio.run(run_async(args))
    except KeyboardInterrupt:  # pragma: no cover
        logger.warning("interrupted by user")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

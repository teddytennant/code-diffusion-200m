"""Synthetic Python data via the Claude Code CLI (OAuth, no API key).

Drop-in alternative to ``scripts/gen_synthetic.py``. Instead of the
``anthropic`` SDK (needs ``ANTHROPIC_API_KEY``), this script shells out to
``claude -p`` for each generation, which authenticates via the user's
existing ``claude`` login. Concurrency, retry, cost capping, and resume
are all preserved.

    claude auth login                # one-time
    python scripts/gen_synthetic_cc.py \\
      --output data/synthetic.jsonl \\
      --target-tokens 10_000_000 \\
      --max-cost-usd 35 \\
      --concurrency 16 \\
      --model haiku

Output format: JSONL, one record per line, same shape as gen_synthetic.py:

    {"source": "...", "domain": "...", "style": "...",
     "tokens": 1234, "model": "...", "topic": "...", "cost_usd": 0.0042}
"""
from __future__ import annotations

import argparse
import asyncio
import ast
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.synthetic_prompts import (
    DOMAINS,
    LENGTH_TARGETS,
    STYLES,
    SYSTEM_PROMPT,
    build_prompt,
    random_seed_topic,
)


logger = logging.getLogger("gen_synthetic_cc")


# ---------------------------------------------------------------------------
# Token approximation (avoids loading tiktoken)
# ---------------------------------------------------------------------------


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# State / resume
# ---------------------------------------------------------------------------


@dataclass
class State:
    output_path: Path
    n_kept: int = 0
    n_seen: int = 0
    tokens_kept: int = 0
    cost_usd: float = 0.0
    seen_keys: set[tuple[str, str, str, int]] = field(default_factory=set)

    @classmethod
    def load_or_init(cls, output_path: Path) -> "State":
        s = cls(output_path=output_path)
        if not output_path.exists():
            return s
        with output_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s.n_kept += 1
                s.n_seen += 1
                s.tokens_kept += int(obj.get("tokens") or 0)
                s.cost_usd += float(obj.get("cost_usd") or 0.0)
                key = (
                    str(obj.get("domain") or ""),
                    str(obj.get("style") or ""),
                    str(obj.get("topic") or ""),
                    int(obj.get("target_lines") or 0),
                )
                s.seen_keys.add(key)
        return s


# ---------------------------------------------------------------------------
# Single generation via `claude -p`
# ---------------------------------------------------------------------------


async def _run_one(
    user_prompt: str,
    *,
    model: str,
    timeout_s: float,
    per_call_budget_usd: float,
) -> dict[str, Any] | None:
    """Invoke claude -p once. Returns parsed JSON dict on success, or None.

    The system prompt is set explicitly so we don't pay for the default
    Claude Code system prompt + memory + hooks (those are huge).
    """
    cmd = [
        "claude",
        "-p",
        user_prompt,
        "--model",
        model,
        "--output-format",
        "json",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--system-prompt",
        SYSTEM_PROMPT,
        "--max-budget-usd",
        str(per_call_budget_usd),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            logger.warning("claude -p timed out after %ss", timeout_s)
            return None
    except FileNotFoundError:
        logger.error("`claude` CLI not found on PATH")
        return None

    if proc.returncode != 0:
        logger.warning(
            "claude -p exited %d: %s",
            proc.returncode,
            stderr.decode("utf-8", errors="replace")[-400:],
        )
        return None

    try:
        return json.loads(stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        logger.warning("could not parse claude -p output: %s", e)
        return None


# ---------------------------------------------------------------------------
# Validation + extraction
# ---------------------------------------------------------------------------


def _strip_fences(text: str) -> str:
    """Remove ```python ... ``` fences if Claude added them despite instructions."""
    s = text.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl > 0:
            s = s[first_nl + 1 :]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[: -len("```")].rstrip()
    return s


def _is_valid_python(source: str) -> bool:
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def _extract(result_obj: dict[str, Any]) -> tuple[str, float] | None:
    if not isinstance(result_obj, dict):
        return None
    if result_obj.get("is_error"):
        return None
    raw = result_obj.get("result")
    if not isinstance(raw, str) or not raw:
        return None
    src = _strip_fences(raw)
    if not _is_valid_python(src):
        return None
    cost = float(result_obj.get("total_cost_usd") or 0.0)
    return src, cost


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def _producer(
    args: argparse.Namespace,
    state: State,
    rng: random.Random,
) -> None:
    queue: asyncio.Queue[tuple[str, str, str, int]] = asyncio.Queue(maxsize=args.concurrency * 2)
    stop_event = asyncio.Event()
    write_lock = asyncio.Lock()

    out_fh = state.output_path.open("a", encoding="utf-8")

    t0 = time.perf_counter()

    async def write_record(record: dict[str, Any]) -> None:
        async with write_lock:
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()

    async def worker(wid: int) -> None:
        while not stop_event.is_set():
            try:
                domain, style, topic, target_lines = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if stop_event.is_set():
                    return
                continue
            try:
                if state.cost_usd >= args.max_cost_usd:
                    stop_event.set()
                    return
                if state.tokens_kept >= args.target_tokens:
                    stop_event.set()
                    return

                user_prompt = build_prompt(domain, style, target_lines, seed_topic=topic)
                result_obj = await _run_one(
                    user_prompt,
                    model=args.model,
                    timeout_s=args.timeout_s,
                    per_call_budget_usd=args.per_call_budget_usd,
                )
                state.n_seen += 1
                if result_obj is None:
                    continue
                got = _extract(result_obj)
                if got is None:
                    continue
                source, cost = got
                tokens = approx_tokens(source)
                record = {
                    "source": source,
                    "content": source,  # mirror gen_synthetic.py shape
                    "domain": domain,
                    "style": style,
                    "topic": topic,
                    "target_lines": target_lines,
                    "tokens": tokens,
                    "cost_usd": cost,
                    "model": args.model,
                }
                await write_record(record)
                state.n_kept += 1
                state.tokens_kept += tokens
                state.cost_usd += cost

                if state.n_kept % args.log_every == 0:
                    elapsed = time.perf_counter() - t0
                    rate = state.tokens_kept / max(elapsed, 1e-3)
                    logger.info(
                        "[%d] tokens=%.2fM/%.2fM cost=$%.2f/$%.2f rate=%.0f tok/s",
                        state.n_kept,
                        state.tokens_kept / 1e6,
                        args.target_tokens / 1e6,
                        state.cost_usd,
                        args.max_cost_usd,
                        rate,
                    )
            finally:
                queue.task_done()

    async def feeder() -> None:
        while not stop_event.is_set():
            domain = rng.choice(DOMAINS)
            style = rng.choice(STYLES)
            _, target_lines = rng.choice(LENGTH_TARGETS)
            topic = random_seed_topic(domain, rng)
            key = (domain, style, topic, target_lines)
            if key in state.seen_keys:
                continue
            state.seen_keys.add(key)
            await queue.put(key)

    workers = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]
    feed_task = asyncio.create_task(feeder())

    try:
        while not stop_event.is_set():
            await asyncio.sleep(2.0)
            if state.tokens_kept >= args.target_tokens:
                stop_event.set()
            elif state.cost_usd >= args.max_cost_usd:
                stop_event.set()
    finally:
        stop_event.set()
        feed_task.cancel()
        try:
            await feed_task
        except (asyncio.CancelledError, Exception):
            pass
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        out_fh.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="gen_synthetic_cc.py")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--target-tokens", type=int, default=10_000_000)
    p.add_argument("--max-cost-usd", type=float, default=35.0)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--model", default="haiku",
                   help="claude -p --model value: 'haiku', 'sonnet', or full id")
    p.add_argument("--timeout-s", type=float, default=180.0,
                   help="per-call timeout")
    p.add_argument("--per-call-budget-usd", type=float, default=0.50,
                   help="claude --max-budget-usd per single call")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    state = State.load_or_init(args.output)
    if state.n_kept:
        logger.info(
            "resuming: %d existing records, %.2fM tokens, $%.2f spent",
            state.n_kept, state.tokens_kept / 1e6, state.cost_usd,
        )

    rng = random.Random(args.seed + state.n_kept)
    asyncio.run(_producer(args, state, rng))

    logger.info(
        "done: kept=%d tokens=%.2fM cost=$%.4f -> %s",
        state.n_kept, state.tokens_kept / 1e6, state.cost_usd, args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

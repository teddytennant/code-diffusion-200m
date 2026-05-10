"""Stream-download a Python code corpus to JSONL.

Default source is `codeparrot/codeparrot-clean` (open, deduped Python from
GitHub). The original spec called for `bigcode/starcoderdata` but that's
gated on the Hub. The downstream dataset reads JSONL via the `synthetic`
kind, so the consumer is the same.

    python scripts/download_corpus.py \\
      --output data/starcoder2/python.jsonl \\
      --target-rows 500000 \\
      --min-chars 200 --max-chars 50000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="download_corpus.py")
    p.add_argument("--dataset", default="codeparrot/codeparrot-clean")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--target-rows", type=int, default=500_000)
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=50_000)
    p.add_argument("--log-every", type=int, default=2_000)
    args = p.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    kw = {"split": args.split, "streaming": True}
    if args.config:
        ds = load_dataset(args.dataset, args.config, **kw)
    else:
        ds = load_dataset(args.dataset, **kw)

    n_seen = 0
    n_kept = 0
    n_chars = 0
    t0 = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as fh:
        for row in ds:
            n_seen += 1
            text = row.get("content") or row.get("code") or row.get("text")
            if not isinstance(text, str):
                continue
            n = len(text)
            if n < args.min_chars or n > args.max_chars:
                continue
            fh.write(json.dumps({"content": text}) + "\n")
            n_kept += 1
            n_chars += n
            if n_kept % args.log_every == 0:
                dt = time.perf_counter() - t0
                rate = n_kept / max(dt, 1e-3)
                print(
                    f"[{n_kept}/{args.target_rows}] seen={n_seen} kept={n_kept} "
                    f"chars={n_chars/1e9:.2f}B rate={rate:.0f} rows/s",
                    file=sys.stderr,
                    flush=True,
                )
            if n_kept >= args.target_rows:
                break

    dt = time.perf_counter() - t0
    print(
        f"done in {dt:.1f}s; saw {n_seen}, kept {n_kept}, "
        f"~{n_chars/1e9:.2f}B chars (~{n_chars/4/1e9:.2f}B tokens approx) -> {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

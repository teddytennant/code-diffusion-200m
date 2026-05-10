"""CLI entry point for masked-diffusion training.

    python train.py --config configs/main.yaml [--max-steps N] [--resume PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.train import run_training


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="train.py")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--log-every", type=int, default=None,
                   help="Override run.log_every from config.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.log_every is not None:
        cfg.setdefault("run", {})["log_every"] = args.log_every
    run_training(cfg, max_steps=args.max_steps, resume_from=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

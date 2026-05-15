"""CLI entry point for masked-diffusion training.

    python train.py --config configs/main.yaml [--max-steps N] [--resume PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from src.train import run_training


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into a copy of base (for 'extends' support)."""
    out: dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_config(path: Path) -> dict[str, Any]:
    """Load YAML/JSON, following 'extends: relative/path.yaml' recursively."""
    text = path.read_text(encoding="utf-8")
    cfg: dict[str, Any] = yaml.safe_load(text) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    if "extends" in cfg:
        ext = cfg.pop("extends")
        base_path = (path.parent / ext).resolve()
        base = _load_config(base_path)
        cfg = _deep_merge(base, cfg)
    return cfg


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
    cfg = _load_config(args.config)
    if args.log_every is not None:
        cfg.setdefault("run", {})["log_every"] = args.log_every
    run_training(cfg, max_steps=args.max_steps, resume_from=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

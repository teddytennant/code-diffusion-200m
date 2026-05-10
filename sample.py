"""CLI entry point for one-shot sampling from a checkpoint.

    python sample.py --checkpoint ckpt.pt --prompt 'def fib(n):\n' \
        --max-new-tokens 128 --diffusion-steps 16
    python sample.py --checkpoint ckpt.pt --mode fim --prompt-file prompt.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.sample.load import load_sampler_from_checkpoint


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sample.py")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--prompt-file", type=Path, default=None)
    p.add_argument("--mode", choices=["completion", "fim"], default="completion")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--diffusion-steps", type=int, default=16)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--no-confidence-remask", action="store_true")
    p.add_argument("--no-ar-refine", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if (args.prompt is None) == (args.prompt_file is None):
        print("error: pass exactly one of --prompt or --prompt-file", file=sys.stderr)
        return 2
    prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")

    sampler = load_sampler_from_checkpoint(
        args.checkpoint,
        device=args.device,
        confidence_remask=not args.no_confidence_remask,
        ar_refine=not args.no_ar_refine,
    )
    kwargs = {
        "diffusion_steps": args.diffusion_steps,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if args.seed is not None:
        kwargs["seed"] = args.seed
    outs = sampler.sample(
        prompt,
        max_new_tokens=args.max_new_tokens,
        mode=args.mode,
        n_samples=args.n_samples,
        temperature=args.temperature,
        **kwargs,
    )
    for i, s in enumerate(outs):
        if len(outs) > 1:
            print(f"=== sample {i} ===")
        print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

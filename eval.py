"""CLI entry point for evaluation.

    python eval.py --task humaneval     --checkpoint ckpt.pt --output-dir results/main/
    python eval.py --task mbpp          --checkpoint ckpt.pt --output-dir results/main/
    python eval.py --task humaneval-fim --checkpoint ckpt.pt --output-dir results/main/
    python eval.py --task throughput    --checkpoint ckpt.pt --output-dir results/main/
    python eval.py --task all           --checkpoint ckpt.pt --output-dir results/main/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.sample.load import load_sampler_from_checkpoint


TASKS = ["humaneval", "mbpp", "humaneval-fim", "throughput"]
DEFAULT_THROUGHPUT_PROMPTS = [
    "def quicksort(xs):\n",
    "def fib(n):\n",
    "def is_prime(n):\n",
    "class Stack:\n    def __init__(self):\n",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="eval.py")
    p.add_argument("--task", required=True, choices=[*TASKS, "all"])
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-samples-per-problem", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--diffusion-steps", type=int, default=16)
    p.add_argument("--no-confidence-remask", action="store_true")
    p.add_argument("--no-ar-refine", action="store_true")
    p.add_argument(
        "--diffusion-steps-sweep",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64],
        help="Used by --task throughput",
    )
    return p.parse_args(argv)


def _wrap_with_step_default(sampler, default_steps: int):
    """Apply --diffusion-steps as the default for sample() calls that don't override."""
    sampler.default_diffusion_steps = default_steps
    return sampler


def _run(task: str, sampler, args, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    if task == "humaneval":
        from src.eval.humaneval import run_humaneval

        return run_humaneval(
            sampler,
            n_samples_per_problem=args.n_samples_per_problem,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            output_path=output_dir / "humaneval_samples.jsonl",
        )

    if task == "mbpp":
        from src.eval.mbpp import run_mbpp

        return run_mbpp(
            sampler,
            n_samples_per_problem=args.n_samples_per_problem,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            output_path=output_dir / "mbpp_samples.jsonl",
        )

    if task == "humaneval-fim":
        from src.eval.humaneval_fim import run_all_fim_variants

        return run_all_fim_variants(
            sampler,
            n_samples_per_problem=max(1, args.n_samples_per_problem // 2),
            output_dir=output_dir,
        )

    if task == "throughput":
        from src.eval.throughput import benchmark_throughput

        df = benchmark_throughput(
            sampler,
            prompts=DEFAULT_THROUGHPUT_PROMPTS,
            diffusion_steps_sweep=args.diffusion_steps_sweep,
            max_new_tokens=min(256, args.max_new_tokens),
            output_path=output_dir / "throughput.csv",
            temperature=args.temperature,
        )
        return {"throughput": df.to_dict(orient="records")}

    raise ValueError(f"Unknown task: {task}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    sampler = load_sampler_from_checkpoint(
        args.checkpoint,
        device=args.device,
        default_diffusion_steps=args.diffusion_steps,
        confidence_remask=not args.no_confidence_remask,
        ar_refine=not args.no_ar_refine,
    )

    tasks = TASKS if args.task == "all" else [args.task]
    results: dict[str, dict] = {}
    for t in tasks:
        print(f"== running {t} ==", file=sys.stderr)
        results[t] = _run(t, sampler, args, args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

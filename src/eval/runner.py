"""Top-level evaluation CLI.

Usage:
    python -m src.eval.runner --task humaneval --output-dir runs/eval-001 \\
        --sampler-config configs/sampler.yaml

Tasks: ``humaneval``, ``mbpp``, ``humaneval-fim``, ``throughput``, ``all``.

The sampler is loaded from a YAML or JSON config. The config must contain a
``sampler`` dict with at least a ``target`` field, e.g.::

    sampler:
      target: src.sample.diffusion_sampler:DiffusionSampler
      ckpt_path: runs/pretrain-002/best.pt
      device: cuda
    humaneval:
      n_samples_per_problem: 10
      temperature: 0.2
    throughput:
      prompts:
        - "def quicksort(xs):\\n"
        - "def fib(n):\\n"
      diffusion_steps_sweep: [4, 8, 16, 32, 64]

If the sampler module doesn't exist yet (the cluster build is still in
progress), the runner prints a clear pointer to where it should be
implemented and exits non-zero.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from src.eval.sampler_interface import Sampler


# ---------------------------------------------------------------------------
# Config loader


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - install hint
            raise ImportError(
                "pyyaml is required to load YAML configs. "
                "Install with `pip install pyyaml`."
            ) from exc
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root at {path} must be a mapping, got {type(cfg)}")
    return cfg


def _import_object(target: str) -> Any:
    """Import ``module.path:Object`` and return ``Object``.

    On failure we raise a wrapped ``RuntimeError`` so the CLI can print a
    crisp pointer to the missing implementation rather than a stack trace.
    """
    if ":" not in target:
        raise ValueError(
            f"Sampler target {target!r} must be of the form 'module.path:ClassName'"
        )
    module_path, attr = target.split(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise RuntimeError(
            f"Sampler module '{module_path}' is not importable yet. "
            f"Implement it (the project layout expects e.g. "
            f"src/sample/diffusion_sampler.py with a DiffusionSampler class) "
            f"before running real evaluation. Underlying error: {exc}"
        ) from exc
    if not hasattr(module, attr):
        raise RuntimeError(
            f"Module '{module_path}' has no attribute '{attr}'. "
            f"Expected a Sampler-compatible class."
        )
    return getattr(module, attr)


def load_sampler(config: dict[str, Any]) -> Sampler:
    """Instantiate the sampler from a parsed config dict."""
    sampler_cfg = dict(config.get("sampler") or {})
    target = sampler_cfg.pop("target", None)
    if target is None:
        raise ValueError("config must contain sampler.target")
    cls = _import_object(target)
    return cls(**sampler_cfg)


# ---------------------------------------------------------------------------
# Task dispatch


def _run_task(task: str, sampler: Sampler, cfg: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    if task == "humaneval":
        from src.eval.humaneval import run_humaneval

        kwargs = dict(cfg.get("humaneval") or {})
        return run_humaneval(
            sampler,
            output_path=output_dir / "humaneval_samples.jsonl",
            **kwargs,
        )

    if task == "mbpp":
        from src.eval.mbpp import run_mbpp

        kwargs = dict(cfg.get("mbpp") or {})
        return run_mbpp(
            sampler,
            output_path=output_dir / "mbpp_samples.jsonl",
            **kwargs,
        )

    if task == "humaneval-fim":
        from src.eval.humaneval_fim import run_all_fim_variants

        kwargs = dict(cfg.get("humaneval_fim") or {})
        return run_all_fim_variants(sampler, output_dir=output_dir, **kwargs)

    if task == "throughput":
        from src.eval.throughput import benchmark_throughput

        kwargs = dict(cfg.get("throughput") or {})
        prompts = kwargs.pop("prompts", ["def f(x):\n    "])
        df = benchmark_throughput(
            sampler,
            prompts=prompts,
            output_path=output_dir / "throughput.csv",
            **kwargs,
        )
        return {"throughput": df.to_dict(orient="records")}

    raise ValueError(f"Unknown task {task!r}")


def _summary_md(results: dict[str, dict]) -> str:
    lines = ["# Evaluation summary", ""]
    for task, metrics in results.items():
        lines.append(f"## {task}")
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"- **{k}**:")
                    lines.append(f"  ```\n  {json.dumps(v, indent=2)}\n  ```")
                else:
                    lines.append(f"- **{k}**: {v}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.eval.runner",
        description="Run evaluation tasks against a Code-Diffusion-200M sampler.",
    )
    p.add_argument(
        "--task",
        required=True,
        choices=["humaneval", "mbpp", "humaneval-fim", "throughput", "all"],
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--sampler-config", required=True, type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = _load_config(args.sampler_config)

    try:
        sampler = load_sampler(cfg)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    tasks = (
        ["humaneval", "mbpp", "humaneval-fim", "throughput"]
        if args.task == "all"
        else [args.task]
    )

    results: dict[str, dict] = {}
    for task in tasks:
        print(f"== running {task} ==", file=sys.stderr)
        results[task] = _run_task(task, sampler, cfg, args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(_summary_md(results), encoding="utf-8")
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())

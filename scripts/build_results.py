"""Aggregate per-eval JSON results into a RESULTS.md summary table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_results(d: Path) -> dict[str, Any]:
    p = d / "results.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _flat(prefix: str, obj: Any, out: dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flat(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(obj, list):
        if obj and isinstance(obj[0], dict):
            for i, item in enumerate(obj):
                _flat(f"{prefix}[{i}]", item, out)
    else:
        out[prefix] = obj


def _interesting_metrics(results: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    _flat("", results, flat)
    keep: dict[str, Any] = {}
    for k, v in flat.items():
        kl = k.lower()
        if any(s in kl for s in ("pass@", "tokens_per_sec", "parse_rate", "wall_seconds", "n_problems")):
            keep[k] = v
    return keep


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="build_results.py")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("RESULTS.md"))
    args = p.parse_args(argv)

    runs = sorted([d for d in args.results_dir.iterdir() if d.is_dir()])
    if not runs:
        print(f"no run dirs under {args.results_dir}", flush=True)
        return 1

    rows: list[tuple[str, dict[str, Any]]] = []
    for d in runs:
        rows.append((d.name, _interesting_metrics(_read_results(d))))

    keys: list[str] = []
    seen: set[str] = set()
    for _, m in rows:
        for k in m.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    keys.sort()

    lines: list[str] = ["# Code-Diffusion-200M results", ""]
    lines.append("| run | " + " | ".join(keys) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(keys)) + " |")
    for name, m in rows:
        cells = []
        for k in keys:
            v = m.get(k, "")
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

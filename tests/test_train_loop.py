"""CPU-runnable smoke tests for the masked-diffusion training loop."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CodeDiffusionDataset
from src.model import CodeDiffusionTransformer, ModelConfig
from src.train import loop as loop_module
from src.train.loop import (
    build_optimizer,
    load_checkpoint,
    masked_diffusion_loss,
    run_training,
    save_checkpoint,
)
from tests.test_data import MockTokenizer


# Force CPU execution: the smoke test spec requires CPU-runnable tests
# regardless of whether CUDA happens to be available on the test host.
@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield


SAMPLE_PY = (
    "def f(a, b):\n"
    "    return a + b\n"
    "x = 1\n"
    "y = 2\n"
    "z = x + y\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_config(tmp_path: Path, *, max_steps: int = 2) -> Dict[str, Any]:
    return {
        "run": {
            "name": "test",
            "seed": 42,
            "output_dir": str(tmp_path / "ckpt"),
            "log_every": 1,
            "ckpt_every": 0,  # disabled by default; tests opt in
            "wandb_project": "test",
            "dataloader_workers": 0,
        },
        "model": {
            "vocab_size": 200,
            "hidden_dim": 32,
            "num_layers": 2,
            "num_heads": 2,
            "head_dim": 16,
            "mlp_hidden": 64,
            "max_seq_len": 64,
            "rope_base": 10000.0,
            "norm_eps": 1e-5,
            "tie_embeddings": True,
            "use_grad_checkpoint": False,
            "dropout": 0.0,
        },
        "data": {
            "sources": [],  # filled in by caller
            "seq_len": 64,
            "ast_mask_prob": 0.5,
            "fim_prob": 0.0,
            "mask_ratio_min": 0.10,
            "mask_ratio_max": 0.50,
        },
        "train": {
            "total_tokens": 1_000_000,
            "per_device_batch": 2,
            "grad_accum_steps": 2,
            "precision": "bf16",
            "optimizer": {
                "name": "adamw_8bit",
                "lr": 1e-3,
                "betas": [0.9, 0.95],
                "weight_decay": 0.1,
                "eps": 1e-8,
            },
            "schedule": {
                "name": "wsd",
                "warmup_pct": 0.1,
                "stable_pct": 0.8,
                "decay_pct": 0.1,
                "decay_shape": "quadratic",
            },
            "curriculum": {"mask_ratio_phase_pct": 0.30},
            "flash_attention": False,
            "z_loss_weight": 1e-4,
        },
    }


def _make_dataset(tmp_path: Path) -> CodeDiffusionDataset:
    """Tiny in-memory dataset built from a JSONL file with 4 examples."""
    src = tmp_path / "data.jsonl"
    src.write_text(
        "\n".join(json.dumps({"content": SAMPLE_PY * 8}) for _ in range(4)) + "\n"
    )
    tok = MockTokenizer()
    return CodeDiffusionDataset(
        sources=[{"path": str(src), "kind": "synthetic", "weight": 1.0}],
        tokenizer=tok,
        seq_len=64,
        ast_mask_prob=0.5,
        fim_prob=0.0,
        mask_ratio_min=0.10,
        mask_ratio_max=0.50,
        seed=1,
    )


def _make_model(cfg: Dict[str, Any]) -> CodeDiffusionTransformer:
    m = cfg["model"]
    model_cfg = ModelConfig(
        vocab_size=m["vocab_size"],
        hidden_dim=m["hidden_dim"],
        num_layers=m["num_layers"],
        num_heads=m["num_heads"],
        head_dim=m["head_dim"],
        mlp_hidden=m["mlp_hidden"],
        max_seq_len=m["max_seq_len"],
        tie_embeddings=m["tie_embeddings"],
        use_grad_checkpoint=False,
        dropout=0.0,
    )
    return CodeDiffusionTransformer(model_cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_loop_runs_two_steps_cpu(tmp_path: Path) -> None:
    cfg = _make_tiny_config(tmp_path, max_steps=2)
    dataset = _make_dataset(tmp_path)
    model = _make_model(cfg)

    # Capture the very first loss before any updates by running a single forward
    # before training. We just assert finite loss after; loss-decrease isn't
    # guaranteed in 2 steps.
    result = run_training(
        cfg,
        max_steps=2,
        dataset=dataset,
        model=model,
    )
    metrics = result["metrics"]
    assert "train/loss" in metrics
    assert math.isfinite(metrics["train/loss"]), f"loss not finite: {metrics['train/loss']}"
    assert math.isfinite(metrics["train/ce_loss"])
    assert math.isfinite(metrics["train/z_loss"])
    assert result["final_step"] == 1


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    cfg = _make_tiny_config(tmp_path, max_steps=2)
    cfg["run"]["ckpt_every"] = 2  # save at the end of step index 1 (i.e. step+1=2)
    dataset = _make_dataset(tmp_path)
    model = _make_model(cfg)

    run_training(cfg, max_steps=2, dataset=dataset, model=model)

    # Find the saved checkpoint.
    ckpt_dir = Path(cfg["run"]["output_dir"])
    assert ckpt_dir.exists(), f"no ckpt dir at {ckpt_dir}"
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    assert ckpts, f"no checkpoints saved in {ckpt_dir}"
    latest = ckpt_dir / "latest.pt"
    assert latest.exists()

    # Load into fresh model + optimizer; states must match.
    fresh_model = _make_model(cfg)
    fresh_opt = build_optimizer(
        fresh_model,
        lr=float(cfg["train"]["optimizer"]["lr"]),
        betas=tuple(cfg["train"]["optimizer"]["betas"]),
        weight_decay=float(cfg["train"]["optimizer"]["weight_decay"]),
        eps=float(cfg["train"]["optimizer"]["eps"]),
        use_8bit=False,
    )
    payload = load_checkpoint(
        str(ckpts[-1]),
        model=fresh_model,
        optimizer=fresh_opt,
        device=torch.device("cpu"),
    )

    # Compare model state.
    sd_orig = model.state_dict()
    sd_fresh = fresh_model.state_dict()
    assert set(sd_orig.keys()) == set(sd_fresh.keys())
    for k in sd_orig:
        assert torch.equal(sd_orig[k], sd_fresh[k]), f"state mismatch on {k}"

    # Optimizer step count should match (each optimizer step bumps "step" in
    # AdamW per-param state).
    def _opt_step_count(opt: torch.optim.Optimizer) -> int:
        for group in opt.param_groups:
            for p in group["params"]:
                state = opt.state.get(p, {})
                if "step" in state:
                    s = state["step"]
                    return int(s.item()) if isinstance(s, torch.Tensor) else int(s)
        return 0

    assert payload["step"] == 1
    # The fresh optimizer's per-param 'step' should match the saved one.
    assert _opt_step_count(fresh_opt) == 2  # 2 optimizer steps were performed


def test_curriculum_bump(tmp_path: Path) -> None:
    cfg = _make_tiny_config(tmp_path, max_steps=10)
    cfg["train"]["curriculum"]["mask_ratio_phase_pct"] = 0.30
    dataset = _make_dataset(tmp_path)
    model = _make_model(cfg)

    observed: List[float] = []
    original_log = loop_module._Logger.log

    def spy_log(self, metrics, step):
        observed.append(float(metrics["train/mask_ratio_max_current"]))
        return original_log(self, metrics, step)

    with patch.object(loop_module._Logger, "log", spy_log):
        run_training(cfg, max_steps=10, dataset=dataset, model=model)

    assert len(observed) == 10
    # First 3 steps (0, 1, 2) should be 0.50; from step 3 onward should be 1.0.
    assert observed[0] == pytest.approx(0.50)
    assert observed[2] == pytest.approx(0.50)
    assert observed[3] == pytest.approx(1.0)
    assert observed[-1] == pytest.approx(1.0)


def test_z_loss_present_when_weight_nonzero(tmp_path: Path) -> None:
    cfg = _make_tiny_config(tmp_path, max_steps=2)
    cfg["train"]["z_loss_weight"] = 1.0  # crank up so z is plainly visible
    dataset = _make_dataset(tmp_path)
    model = _make_model(cfg)

    seen_z: List[float] = []
    original_log = loop_module._Logger.log

    def spy_log(self, metrics, step):
        seen_z.append(float(metrics["train/z_loss"]))
        return original_log(self, metrics, step)

    with patch.object(loop_module._Logger, "log", spy_log):
        run_training(cfg, max_steps=2, dataset=dataset, model=model)

    assert seen_z, "no log calls captured"
    for z in seen_z:
        assert math.isfinite(z), f"z_loss not finite: {z}"
        assert z > 0.0, f"z_loss should be positive when weight nonzero, got {z}"


# ---------------------------------------------------------------------------
# Loss unit (sanity check on the helper itself)
# ---------------------------------------------------------------------------


def test_masked_diffusion_loss_no_masked_positions() -> None:
    logits = torch.randn(2, 4, 10, requires_grad=True)
    targets = torch.randint(0, 10, (2, 4))
    mask = torch.zeros(2, 4, dtype=torch.bool)
    total, ce, z = masked_diffusion_loss(logits, targets, mask, z_loss_weight=1e-4)
    assert float(total.detach()) == 0.0
    assert float(ce.detach() if ce.requires_grad else ce) == 0.0
    assert float(z.detach() if z.requires_grad else z) == 0.0
    # Gradient should still be zero but well-defined.
    total.backward()
    assert logits.grad is not None

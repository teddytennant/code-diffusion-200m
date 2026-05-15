"""Masked-diffusion training loop for Code-Diffusion-200M.

Entry point: ``run_training(config, *, max_steps=None, resume_from=None)``.

Supports objective "diffusion" (default) or "causal_lm" (for ar_only ablation).
Config supports "extends:" via the train.py CLI.
"""
from __future__ import annotations

import math
import os
import random
import shutil
import time
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import CodeDiffusionDataset
from src.data.loader import make_dataloader
from src.model import CodeDiffusionTransformer, ModelConfig


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


@dataclass
class LRSchedule:
    peak_lr: float
    warmup_steps: int
    stable_steps: int
    decay_steps: int

    @property
    def total_steps(self) -> int:
        return self.warmup_steps + self.stable_steps + self.decay_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            if self.warmup_steps <= 0:
                return self.peak_lr
            return self.peak_lr * (step + 1) / self.warmup_steps
        if step < self.warmup_steps + self.stable_steps:
            return self.peak_lr
        decayed = step - self.warmup_steps - self.stable_steps
        if self.decay_steps <= 0:
            return 0.0
        t = min(1.0, decayed / self.decay_steps)
        # Quadratic decay: lr = peak * (1 - t**2)
        return max(0.0, self.peak_lr * (1.0 - t * t))


def build_schedule(
    peak_lr: float,
    total_steps: int,
    warmup_pct: float,
    stable_pct: float,
    decay_pct: float,
) -> LRSchedule:
    warmup = int(round(warmup_pct * total_steps))
    decay = int(round(decay_pct * total_steps))
    stable = max(0, total_steps - warmup - decay)
    return LRSchedule(peak_lr=peak_lr, warmup_steps=warmup, stable_steps=stable, decay_steps=decay)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def _split_decay_params(model: torch.nn.Module) -> Tuple[list, list]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Skip biases, embeddings, RMSNorm weights (everything <2D or named norm/embed).
        if p.ndim < 2:
            no_decay.append(p)
        elif "tok_emb" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def build_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    betas: Tuple[float, float],
    weight_decay: float,
    eps: float,
    use_8bit: bool,
) -> torch.optim.Optimizer:
    decay_params, no_decay_params = _split_decay_params(model)
    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if use_8bit:
        try:
            import bitsandbytes as bnb  # type: ignore

            return bnb.optim.AdamW8bit(groups, lr=lr, betas=betas, eps=eps)
        except Exception as e:
            warnings.warn(f"bitsandbytes unavailable ({e}); falling back to torch AdamW")
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def masked_diffusion_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask_positions: torch.Tensor,
    *,
    z_loss_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total_loss, ce_loss, z_loss) for masked positions only."""
    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = target_ids.view(-1)
    flat_mask = mask_positions.view(-1)

    sel_logits = flat_logits[flat_mask]
    sel_targets = flat_targets[flat_mask]

    if sel_logits.numel() == 0:
        zero = logits.sum() * 0.0
        return zero, zero, zero

    ce = F.cross_entropy(sel_logits, sel_targets, reduction="mean")
    # z-loss: penalises log-partition magnitude. Stabilises softmax tails.
    lse = torch.logsumexp(sel_logits, dim=-1)
    z = (lse * lse).mean()
    total = ce + z_loss_weight * z
    return total, ce.detach(), z.detach()


def causal_lm_loss(
    logits: torch.Tensor, target_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard next-token causal LM loss (shifted CE). Returns (loss, ce, z=0)."""
    if logits.size(1) <= 1:
        zero = logits.sum() * 0.0
        return zero, zero, zero
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = target_ids[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))
    return loss, loss.detach(), torch.zeros((), device=loss.device, dtype=loss.dtype)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _rng_state(device: torch.device) -> Dict[str, Any]:
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def _restore_rng(state: Dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def save_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    lr: float,
    config: Dict[str, Any],
    dataset_state: Dict[str, Any],
    device: torch.device,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": _rng_state(device),
        "step": step,
        "lr": lr,
        "config": config,
        "dataset_state": dataset_state,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if device is not None and payload.get("rng") is not None:
        _restore_rng(payload["rng"], device)
    return payload


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class _Logger:
    """Tiny wandb wrapper with a stdout fallback."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.wandb = None
        if os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") == "offline":
            try:
                import wandb  # type: ignore

                wandb.init(
                    project=config.get("run", {}).get("wandb_project", "code-diffusion-200m"),
                    name=config.get("run", {}).get("name"),
                    config=config,
                )
                self.wandb = wandb
            except Exception as e:
                warnings.warn(f"wandb init failed ({e}); using stdout logger")
                self.wandb = None

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)
        kv = " ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        print(f"[step {step}] {kv}", flush=True)

    def finish(self) -> None:
        if self.wandb is not None:
            try:
                self.wandb.finish()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Build pipeline pieces from config
# ---------------------------------------------------------------------------


def build_model_config(cfg: Dict[str, Any]) -> ModelConfig:
    m = cfg["model"]
    return ModelConfig(
        vocab_size=m["vocab_size"],
        hidden_dim=m["hidden_dim"],
        num_layers=m["num_layers"],
        num_heads=m["num_heads"],
        head_dim=m["head_dim"],
        mlp_hidden=m["mlp_hidden"],
        max_seq_len=m["max_seq_len"],
        rope_base=m.get("rope_base", 10000.0),
        norm_eps=m.get("norm_eps", 1e-5),
        tie_embeddings=m.get("tie_embeddings", True),
        use_grad_checkpoint=m.get("use_grad_checkpoint", True),
        dropout=m.get("dropout", 0.0),
    )


def _compute_total_steps(cfg: Dict[str, Any]) -> int:
    train = cfg["train"]
    seq_len = cfg["data"]["seq_len"]
    tokens_per_step = train["per_device_batch"] * train["grad_accum_steps"] * seq_len
    return max(1, math.ceil(train["total_tokens"] / tokens_per_step))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_training(
    config: Dict[str, Any],
    *,
    max_steps: Optional[int] = None,
    resume_from: Optional[str] = None,
    dataset: Optional[CodeDiffusionDataset] = None,
    tokenizer: Any = None,
    model: Optional[CodeDiffusionTransformer] = None,
) -> Dict[str, Any]:
    """Run training (diffusion or causal_lm per config['train']['objective'])."""
    seed = int(config.get("run", {}).get("seed", 42))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_cfg = config["train"]
    data_cfg = config["data"]
    run_cfg = config.get("run", {})

    # ---- model
    if model is None:
        model_cfg = build_model_config(config)
        model = CodeDiffusionTransformer(model_cfg)

    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_ok else "cpu")
    model.to(device)

    # ---- optimizer
    opt_cfg = train_cfg["optimizer"]
    use_8bit = (opt_cfg.get("name") == "adamw_8bit") and cuda_ok
    optimizer = build_optimizer(
        model,
        lr=float(opt_cfg["lr"]),
        betas=tuple(opt_cfg.get("betas", (0.9, 0.95))),
        weight_decay=float(opt_cfg.get("weight_decay", 0.1)),
        eps=float(opt_cfg.get("eps", 1e-8)),
        use_8bit=use_8bit,
    )

    # ---- schedule / total_steps
    total_steps = _compute_total_steps(config)
    if max_steps is not None:
        total_steps = max_steps
    sched_cfg = train_cfg["schedule"]
    schedule = build_schedule(
        peak_lr=float(opt_cfg["lr"]),
        total_steps=total_steps,
        warmup_pct=float(sched_cfg.get("warmup_pct", 0.05)),
        stable_pct=float(sched_cfg.get("stable_pct", 0.85)),
        decay_pct=float(sched_cfg.get("decay_pct", 0.10)),
    )

    # ---- dataset / loader
    if dataset is None:
        if tokenizer is None:
            from src.data.tokenizer import CodeTokenizer

            tokenizer = CodeTokenizer()
        dataset = CodeDiffusionDataset(
            sources=data_cfg["sources"],
            tokenizer=tokenizer,
            seq_len=data_cfg["seq_len"],
            ast_mask_prob=data_cfg.get("ast_mask_prob", 0.7),
            fim_prob=data_cfg.get("fim_prob", 0.10),
            mask_ratio_min=data_cfg.get("mask_ratio_min", 0.05),
            mask_ratio_max=data_cfg.get("mask_ratio_max", 0.50),
            seed=seed,
        )

    initial_mask_ratio_max = float(dataset.mask_ratio_max)
    num_workers = int(run_cfg.get("dataloader_workers", 0))
    loader = make_dataloader(
        dataset,
        batch_size=int(train_cfg["per_device_batch"]),
        num_workers=num_workers,
        pin_memory=cuda_ok,
    )

    # ---- resume
    start_step = 0
    if resume_from is not None:
        payload = load_checkpoint(resume_from, model=model, optimizer=optimizer, device=device)
        start_step = int(payload["step"]) + 1

    # ---- logger
    logger = _Logger(config)

    grad_accum = int(train_cfg["grad_accum_steps"])
    per_batch = int(train_cfg["per_device_batch"])
    seq_len = int(data_cfg["seq_len"])
    tokens_per_step = per_batch * grad_accum * seq_len
    z_loss_weight = float(train_cfg.get("z_loss_weight", 1e-4))
    objective = str(train_cfg.get("objective", "diffusion")).lower()
    log_every = int(run_cfg.get("log_every", 50))
    ckpt_every = int(run_cfg.get("ckpt_every", 1000))
    output_dir = run_cfg.get("output_dir", "checkpoints/main")
    phase_pct = float(train_cfg.get("curriculum", {}).get("mask_ratio_phase_pct", 0.30))

    # autocast: BF16 on CUDA, no-op on CPU.
    autocast_enabled = cuda_ok
    autocast_kwargs = {"device_type": "cuda" if cuda_ok else "cpu", "dtype": torch.bfloat16, "enabled": autocast_enabled}

    step_times: deque[float] = deque(maxlen=10)
    last_metrics: Dict[str, Any] = {}

    model.train()
    data_iter = iter(loader)

    def _next_batch() -> Optional[Dict[str, torch.Tensor]]:
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            try:
                return next(data_iter)
            except StopIteration:
                return None

    final_step = start_step
    for step in range(start_step, total_steps):
        # Curriculum: bump mask_ratio_max after phase_pct (affects dataset only).
        if step < phase_pct * total_steps:
            dataset.mask_ratio_max = initial_mask_ratio_max
        else:
            dataset.mask_ratio_max = 1.0

        # Apply LR for this step.
        lr = schedule(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_ce = 0.0
        accum_z = 0.0
        any_batch = False
        t0 = time.time()
        for _ in range(grad_accum):
            batch = _next_batch()
            if batch is None:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)
            mask_positions = batch["mask_positions"].to(device, non_blocking=True)

            with torch.autocast(**autocast_kwargs):
                if objective == "causal_lm":
                    logits = model(input_ids, causal=True)
                    loss, ce, z = causal_lm_loss(logits, target_ids)
                else:
                    logits = model(input_ids)
                    loss, ce, z = masked_diffusion_loss(
                        logits, target_ids, mask_positions, z_loss_weight=z_loss_weight
                    )
            (loss / grad_accum).backward()
            accum_loss += float(loss.detach())
            accum_ce += float(ce)
            accum_z += float(z)
            any_batch = True

        if not any_batch:
            warnings.warn("Dataset yielded no batches; ending training early.")
            break

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if cuda_ok:
            torch.cuda.synchronize()
        dt = time.time() - t0
        step_times.append(dt)

        avg_dt = sum(step_times) / len(step_times)
        tokens_per_sec = tokens_per_step / max(avg_dt, 1e-9)
        gpu_mem_gb = (
            torch.cuda.max_memory_allocated() / (1024 ** 3) if cuda_ok else 0.0
        )

        last_metrics = {
            "train/step": step,
            "train/loss": accum_loss / grad_accum,
            "train/ce_loss": accum_ce / grad_accum,
            "train/z_loss": accum_z / grad_accum,
            "train/lr": lr,
            "train/grad_norm": float(grad_norm),
            "train/gpu_mem_gb": gpu_mem_gb,
            "train/tokens_per_sec": tokens_per_sec,
            "train/mask_ratio_max_current": float(dataset.mask_ratio_max),
        }

        if step % log_every == 0 or step == total_steps - 1:
            logger.log(last_metrics, step=step)

        if ckpt_every > 0 and ((step + 1) % ckpt_every == 0 or step == total_steps - 1):
            ckpt_path = os.path.join(output_dir, f"step_{step}.pt")
            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                step=step,
                lr=lr,
                config=config,
                dataset_state={"seed": seed, "step": step},
                device=device,
            )
            latest = os.path.join(output_dir, "latest.pt")
            try:
                if os.path.lexists(latest):
                    os.remove(latest)
                shutil.copy2(ckpt_path, latest)
            except OSError as e:
                warnings.warn(f"Could not refresh latest.pt: {e}")

        final_step = step

    logger.finish()
    return {"final_step": final_step, "metrics": last_metrics}

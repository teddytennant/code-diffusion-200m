from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.model.config import ModelConfig
from src.model.transformer import CodeDiffusionTransformer

from .diffusion_sampler import DiffusionSampler


def _model_config_from_checkpoint(ckpt: Dict[str, Any], override: Optional[Dict[str, Any]] = None) -> ModelConfig:
    cfg = ckpt.get("config", {}) or {}
    model_cfg = dict(cfg.get("model") or {})
    if override:
        model_cfg.update(override)
    valid = {f for f in ModelConfig.__dataclass_fields__}
    model_cfg = {k: v for k, v in model_cfg.items() if k in valid}
    return ModelConfig(**model_cfg)


def load_sampler_from_checkpoint(
    ckpt_path: str | Path,
    device: str | torch.device = "cuda",
    *,
    tokenizer: Any = None,
    model_overrides: Optional[Dict[str, Any]] = None,
    **sampler_kwargs: Any,
) -> DiffusionSampler:
    device = torch.device(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    model_cfg = _model_config_from_checkpoint(ckpt, model_overrides)
    model_cfg.use_grad_checkpoint = False
    model = CodeDiffusionTransformer(model_cfg)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    if tokenizer is None:
        from src.data.tokenizer import CodeTokenizer

        tokenizer = CodeTokenizer()

    return DiffusionSampler(model=model, tokenizer=tokenizer, device=device, **sampler_kwargs)

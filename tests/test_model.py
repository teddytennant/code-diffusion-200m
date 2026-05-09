from __future__ import annotations

import math

import pytest
import torch

from src.model import CodeDiffusionTransformer, ModelConfig


def _tiny_config(use_grad_checkpoint: bool = False) -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        head_dim=32,
        mlp_hidden=128,
        max_seq_len=32,
        tie_embeddings=True,
        use_grad_checkpoint=use_grad_checkpoint,
        dropout=0.0,
    )


def test_forward_shape() -> None:
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = CodeDiffusionTransformer(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_causal_vs_bidirectional_differ() -> None:
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = CodeDiffusionTransformer(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        logits_bi = model(input_ids, causal=False)
        logits_causal = model(input_ids, causal=True)
    assert not torch.allclose(logits_bi, logits_causal, atol=1e-6)


def test_grad_checkpoint_equivalence() -> None:
    torch.manual_seed(0)
    cfg_off = _tiny_config(use_grad_checkpoint=False)
    model_off = CodeDiffusionTransformer(cfg_off).to(torch.float32)

    torch.manual_seed(0)
    cfg_on = _tiny_config(use_grad_checkpoint=True)
    model_on = CodeDiffusionTransformer(cfg_on).to(torch.float32)

    model_on.load_state_dict(model_off.state_dict())

    input_ids = torch.randint(0, cfg_off.vocab_size, (2, 16))

    model_off.train()
    model_on.train()
    out_off = model_off(input_ids)
    out_on = model_on(input_ids)
    assert torch.allclose(out_off, out_on, atol=1e-5, rtol=1e-5)


def test_full_config_param_count(capsys: pytest.CaptureFixture[str]) -> None:
    cfg = ModelConfig()
    model = CodeDiffusionTransformer(cfg)
    n = model.num_parameters()
    print(f"\n[code-diffusion-200m] full-config param count: {n:,} ({n / 1e6:.2f}M)")
    captured = capsys.readouterr()
    print(captured.out)
    assert 180_000_000 <= n <= 220_000_000, f"param count {n} not in [180M, 220M]"


def test_backward_runs() -> None:
    torch.manual_seed(0)
    cfg = _tiny_config(use_grad_checkpoint=False)
    model = CodeDiffusionTransformer(cfg).train()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(input_ids)
    loss = logits.mean()
    loss.backward()
    first_param = next(model.parameters())
    assert first_param.grad is not None
    assert torch.isfinite(first_param.grad).all()
    assert not math.isclose(first_param.grad.abs().sum().item(), 0.0)

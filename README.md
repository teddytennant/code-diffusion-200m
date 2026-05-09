# Code-Diffusion-200M

A 200M-parameter masked diffusion language model for Python code. Designed to train end-to-end on a single A100 80GB or H100 in roughly 30 GPU-hours.

Headline target: **best-in-size on HumanEval-FIM**, with three novel mechanisms each contributing measurable lift in ablations.

## Three novel mechanisms

1. **AST-structured masking (training-time).** During training, 70% of batches mask whole Python AST subtrees (function bodies, loops, expressions) instead of random tokens. Forces the model to denoise at the syntactic level. Implemented in `src/data/ast_masking.py`.
2. **Confidence-guided remasking (inference-time).** At each denoising step, remask the K lowest-confidence tokens by predictive entropy rather than random positions. Adaptive — easy regions resolve early, hard regions get more compute. *To be implemented in `src/sample/`.*
3. **AR refinement pass (post-diffusion).** After diffusion converges, run the same model with `causal=True` to regenerate positions where confidence is below threshold. Same weights, different inference mode. *To be implemented in `src/sample/`.*

## Architecture

| | |
|---|---|
| Params | 201M |
| Layers | 12 |
| Hidden | 1024 |
| Heads | 16 (head_dim 64) |
| MLP hidden | 2730 (SwiGLU 2/3) |
| Vocab | 49154 (StarCoder2 + MASK + PAD) |
| Context | 4096 |
| Norm | RMSNorm (pre-norm) |
| Position | RoPE on Q/K |
| Attention | Bidirectional by default; optional causal at inference |

## Repo layout

```
src/
  model/      bidirectional transformer (built)
  data/       tokenizer, AST/random/FIM masking, packed streaming dataset (built)
  eval/       HumanEval, MBPP, HumanEval-FIM, throughput (built)
  sample/     diffusion sampler + AR refinement (TO BUILD on cluster)
  train/      training loop (TO BUILD on cluster)
scripts/
  gen_synthetic.py       Anthropic API → diverse Python files (built)
  synthetic_prompts.py   prompt templates (built)
tests/                   CPU-only unit tests, all 41 pass
configs/                 YAML configs (skeletons; fill in on cluster)
```

## What's done vs what's left

**Done (CPU-tested):**
- Model architecture + bidirectional/causal forward
- Data pipeline: StarCoder2 tokenizer wrapper, AST-aware subtree masking, random masking, FIM formatter, packed streaming dataset
- Synthetic data generator (CLI ready, ~$300 for 100M tokens at sonnet pricing)
- Eval harness: HumanEval, MBPP, HumanEval-FIM (3 variants), throughput sweep
- 41 passing tests in `tests/`

**To build on cluster:**
- `src/sample/diffusion_sampler.py` — masked diffusion sampler with confidence-guided remasking + AR refinement
- `src/train/loop.py` — full training loop (BF16, FA2, gradient checkpoint, 8-bit AdamW, WSD schedule, AST/random mask curriculum, W&B, checkpoint+resume)
- `train.py`, `sample.py`, `eval.py` top-level entry points
- `configs/main.yaml` and `configs/ablations/*.yaml`

See `LAUNCH.md` for the cluster sequence.

## Setup

```bash
# Already done locally — torch CPU + numpy + pandas + pytest + tqdm + pyyaml in .venv
# On cluster, fresh install:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Install flash-attn LAST (needs torch already installed):
pip install flash-attn --no-build-isolation
pytest tests/  # should be 41 passed, 2 skipped
```

## Training plan (30 GPU-hr budget)

| Hours | Phase | Notes |
|---|---|---|
| 0–3 | Setup, data prep, smoke run | tokenize StarCoder2 subset, generate synthetic, verify loss curves |
| 3–22 | Main training | ~8B tokens, BF16, gradient checkpointing, 8-bit AdamW, WSD schedule |
| 22–25 | Confidence-guided sampler | implement + calibrate K and threshold on held-out |
| 25–28 | AR refinement | causal-mode inference, threshold tuning |
| 28–30 | Eval + ablations | HumanEval, MBPP, HumanEval-FIM, throughput, 4 ablations |

## Critical hyperparameters

- ~8B training tokens
- WSD schedule: 5% warmup / 85% stable / 10% cooldown
- Peak LR 3e-4, AdamW 8-bit, weight decay 0.1
- Per-device batch 4–8 at 4k context, accumulate to ~512K-token effective batch
- Mask ratio curriculum: t ∈ [0.05, 0.50] for first 30% of training, then t ∈ [0.05, 1.0]
- AST mask probability: 0.7; FIM probability: 0.10

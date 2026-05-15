# Code-Diffusion-200M

A 200M-parameter masked diffusion language model for Python code. Designed to train end-to-end on a single A100 80GB or H100 in roughly 30 GPU-hours.

Headline target: **best-in-size on HumanEval-FIM**, with three novel mechanisms each contributing measurable lift in ablations.

## Three novel mechanisms

1. **AST-structured masking (training-time).** During training, 70% of batches mask whole Python AST subtrees (function bodies, loops, expressions) instead of random tokens. Forces the model to denoise at the syntactic level. Implemented in `src/data/ast_masking.py`.
2. **Confidence-guided remasking (inference-time).** At each denoising step, remask the K lowest-confidence tokens by predictive entropy rather than random positions. Adaptive — easy regions resolve early, hard regions get more compute. Implemented in `src/sample/diffusion_sampler.py`.
3. **AR refinement pass (post-diffusion).** After diffusion converges, run the same model with `causal=True` to regenerate positions where confidence is below threshold. Same weights, different inference mode. Implemented in `src/sample/diffusion_sampler.py`.

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
  sample/     diffusion sampler + confidence remask + AR refine
  train/      WSD + 8-bit AdamW + BF16 + curriculum + resume
scripts/
  gen_synthetic*.py, download_corpus.py, build_results.py, auto_finish.sh
tests/                   CPU-only unit tests, all pass
configs/                 main + ablations (extends: supported)
```

## Status

All core components implemented and CPU-tested:
- Model (bidirectional + causal), data (AST/FIM/random masking + packing), diffusion sampler (remask + AR refine), training loop (WSD, 8-bit, BF16, curriculum, resume).
- Top-level CLIs: train.py / sample.py / eval.py.
- Eval harness + 4 ablation configs (extends + objective=causal_lm supported).
- All tests pass (including causal_lm path).

See LAUNCH.md for cluster training/eval sequence and ablations.

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

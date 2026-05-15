# Cluster launch sequence

Target: A100 80GB or H100, ~30 GPU-hours total.

## 0. Environment

```bash
git clone <repo> code-diffusion-200m
cd code-diffusion-200m
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install flash-attn --no-build-isolation  # last, needs torch installed
pytest tests/  # all pass (2 network tests skipped unless RUN_NETWORK_TESTS=1)
```

## 1. Data prep (CPU, ~1–2 hr)

### StarCoder2 Python subset
```bash
mkdir -p data/starcoder2
python -c "
from datasets import load_dataset
ds = load_dataset('bigcode/starcoderdata', data_dir='python', split='train', streaming=True)
ds = ds.filter(lambda x: 200 <= len(x['content']) <= 50000)
ds = ds.take(500_000)  # ~2-3B tokens of filtered Python
ds.save_to_disk('data/starcoder2')
"
```

### Synthetic data (Anthropic API, ~$300, ~2–4 hr)
```bash
export ANTHROPIC_API_KEY=...
python scripts/gen_synthetic.py \
  --output data/synthetic.jsonl \
  --target-tokens 100000000 \
  --concurrency 16 \
  --model claude-sonnet-4-6 \
  --max-cost-usd 350 \
  --resume
```

**Confirm cost with user before running.**

### FIM data
FIM examples are generated on-the-fly via `fim_prob=0.10` in the dataset (see `src/data/fim.py` + `dataset.py`). No separate pre-format script needed.

## 2. Components (all implemented)
- `src/sample/diffusion_sampler.py` + `load.py`
- `src/train/loop.py` (WSD, 8-bit AdamW, curriculum, causal_lm for ar_only)
- `train.py` / `sample.py` / `eval.py` CLIs + `scripts/build_results.py`
- Configs with `extends:` support (train.py) and `objective: causal_lm`

## 3. Smoke training (1–2 GPU-hr)

```bash
python train.py --config configs/main.yaml --max-steps 1000 --log-every 10
# Verify: loss decreases, AST masking surfaces in batches (log a few),
# GPU mem ~50–60 GB, throughput ≥ 15K tokens/sec.
```

## 4. Full training (~22 GPU-hr)

```bash
nohup python train.py --config configs/main.yaml > logs/train.log 2>&1 &
# Monitor: tail -f logs/train.log, W&B dashboard.
# Expected: ~8B tokens in ~22 hours at ~100K tokens/sec on H100,
# ~15-20K tokens/sec on A100.
```

Save checkpoints every 1000 steps to `checkpoints/`.

## 5. Eval (~2 GPU-hr)

```bash
python eval.py --task humaneval --checkpoint checkpoints/final.pt --output-dir results/main/
python eval.py --task mbpp --checkpoint checkpoints/final.pt --output-dir results/main/
python eval.py --task humaneval-fim --checkpoint checkpoints/final.pt --output-dir results/main/
python eval.py --task throughput --checkpoint checkpoints/final.pt --output-dir results/main/
```

## 6. Ablations (~3–4 GPU-hr each, run sequentially or skip if budget tight)

Each ablation either disables one mechanism at training time (no_ast.yaml) or at inference time (no_conf_remask.yaml, no_ar_refine.yaml). The AR-only baseline trains a same-size autoregressive model on the same data for direct comparison.

```bash
for ab in no_ast no_conf_remask no_ar_refine ar_only; do
  python train.py --config configs/ablations/$ab.yaml
  python eval.py --task all --checkpoint checkpoints/${ab}_final.pt \
    --output-dir results/$ab/
done
```

If the full ablation suite is out of budget, prioritise: `no_conf_remask` and `no_ar_refine` (cheap — same checkpoint, different inference). `no_ast` and `ar_only` need separate training runs.

## 7. Write up

```bash
python scripts/build_results.py --results-dir results/ --output RESULTS.md
```

Produces RESULTS.md table from the per-task results.json files. Headline metric: HumanEval-FIM pass@1 (full vs ablations).

## Critical operational notes

- **Single GPU.** No FSDP unless explicitly required.
- **BF16 throughout.** No FP32 fallback in training (kills throughput).
- **Resume-friendly.** Checkpoint includes optimizer state, RNG, dataloader state. Crashes happen — design for it.
- **Commit and push as you go.** User wants commits at every phase boundary.
- **DEVLOG.md** for any deviation from the spec or any unexpected finding.

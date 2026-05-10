# DEVLOG

## 2026-05-10 — Subagent B: training loop

### `src/train/loop.py`

- Built `run_training(config, *, max_steps=None, resume_from=None, dataset=None, tokenizer=None, model=None)`. The injection points for `dataset`/`tokenizer`/`model` are not in the public spec but are required by the smoke tests so we don't have to load the real StarCoder2 tokenizer or hit HF data sources on CI/CPU. The CLI in `train.py` (built later) only needs to pass `config`.
- Loss: `masked_diffusion_loss` returns `(total, ce, z)`. CE is reduced over masked positions (mean), z-loss is `(logsumexp(logits)**2).mean()` over the same masked positions. Total = `ce + z_weight * z`. When no positions are masked, returns `0.0` tensors that still propagate `requires_grad` so backward doesn't crash.
- Optimizer: tries `bitsandbytes.optim.AdamW8bit` only when CUDA is available AND the config's optimizer name is `adamw_8bit`. Otherwise falls back to `torch.optim.AdamW` with a warning. Weight decay is applied only to params with `ndim>=2` and not named `tok_emb`/`norm` (i.e. embeddings, RMSNorm weights, and biases skip decay).
- WSD schedule: linear warmup, constant, quadratic decay (`peak * (1 - t**2)`). Implemented as a `LRSchedule(step) -> lr` callable; LR is written to `param_group['lr']` each step.
- Curriculum: `dataset.mask_ratio_max` is mutated directly each step. With `num_workers=0` (the smoke-test path) this propagates immediately. With workers>0 each worker holds a snapshot, so the bump is eventually-consistent — acceptable per the spec since the bump fires once.
- Mixed precision: `torch.autocast(device_type='cuda', dtype=bf16)` only when CUDA is available; on CPU autocast is disabled and the loop runs in fp32.
- Gradient accumulation: `(loss / grad_accum).backward()` for each micro-batch; one optimizer step per outer step. Grad clip at 1.0.
- Throughput: 10-step moving-average wall time, `tokens_per_step / avg_dt`.
- Checkpointing: `step_{N}.pt` plus `latest.pt` (a copy, not a symlink — symlinks fail on some filesystems). Saves model, optimizer, RNG (cpu/cuda/python/numpy), step, lr, config, dataset_state.
  - **Approximation:** `dataset_state` only stores `{'seed': base_seed, 'step': step}`. Byte-exact resume of an `IterableDataset` would require either tracking `n_examples_consumed` (not currently exposed by `CodeDiffusionDataset`) or replaying the iterator. The spec says "round-trip" not "byte-exact", so we accept this. On resume we restore RNG state — that gives bit-exact downstream behaviour for everything except dataset position.
- Logging: optional W&B (gated on `WANDB_API_KEY` or `WANDB_MODE=offline`), with stdout fallback. Tests don't require W&B.

### Config schema

No renames from `configs/main.yaml`. Added two **optional** keys consumed if present (with defaults if not):

- `run.dataloader_workers` — defaults to 0. Lets the test suite avoid spawning DataLoader workers (which break the curriculum-mutation invariant).

Everything else maps 1:1 to YAML.

### `tests/test_train_loop.py`

- 5 tests, all CPU. The autouse fixture `_force_cpu` patches `torch.cuda.is_available` to `False` so tests behave the same regardless of whether the host has CUDA. (The previous run failed without this on a CUDA host because the model trained on cuda but the freshly-loaded comparison model was on cpu.)
- Tests:
  1. `test_loop_runs_two_steps_cpu` — finite loss after 2 optimizer steps.
  2. `test_checkpoint_roundtrip` — train 2 steps, save, load into fresh model+optimizer, assert state_dicts identical and per-param `step` count == 2.
  3. `test_curriculum_bump` — 10 steps, `mask_ratio_phase_pct=0.30`, observe steps 0-2 = 0.50, step 3+ = 1.0.
  4. `test_z_loss_present_when_weight_nonzero` — crank `z_loss_weight=1.0` and verify `train/z_loss` is finite and positive.
  5. `test_masked_diffusion_loss_no_masked_positions` — extra unit test for the loss helper's degenerate-input path.

### Verification

```
pytest tests/test_train_loop.py -x  # 5 passed
pytest tests/                       # 54 passed, 2 skipped
```

## 2026-05-10 — Integration: top-level CLIs + CPU/GPU smoke

- Added `train.py`, `sample.py`, `eval.py` thin CLIs at repo root.
- Added `src/sample/load.py` with `load_sampler_from_checkpoint(ckpt_path, device, **kwargs)` — rebuilds `ModelConfig` from the checkpoint's saved config dict, instantiates `CodeDiffusionTransformer`, loads weights, and wraps in `DiffusionSampler`. Used by both `sample.py` and `eval.py`.
- Added `configs/smoke_cpu.yaml` (tiny model, 128-tok ctx, JSONL-only dataset) for the boot smoke. Note: the loop auto-uses CUDA when available, so this config also doubles as a quick GPU smoke when an A100 is attached. `data/smoke/tiny.jsonl` is gitignored (`/data/`).
- Smoke results: 3-step run, loss 10.81 → 10.72, mask_ratio_max bumped from 0.50 → 1.0 at the curriculum boundary, all W&B fields present in stdout fallback.

## 2026-05-10 — Repo hygiene

- The original `.gitignore` had an unanchored `data/` pattern that silently blocked `src/data/` from the initial push. Fixed to `/data/` (anchored). The `src/data/` package was force-pushed in a follow-up commit.
- `human-eval` install fails with newer pip (invalid entry-point format upstream). Not a blocker for training/sampler work; needed only when running real HumanEval/MBPP evals. Workaround when needed: pin an older `human-eval` git ref, or patch the installed package's `setup.py` entry point.


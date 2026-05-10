#!/usr/bin/env bash
# Wait for training to complete, then run eval + ablations + RESULTS.md.
# Designed to be launched in the background while training runs.
set -uo pipefail

cd "$(dirname "$0")/.."

LOG="logs/auto_finish.log"
TRAIN_PIDFILE="/tmp/cd200m-train.pid"
CKPT_DIR="checkpoints/main"
RESULTS_DIR="results"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "auto_finish started"

# 1. Wait for training to exit (no python proc holding latest.pt).
while pgrep -f "train\.py --config configs/main_no_synthetic" > /dev/null; do
  sleep 60
done
log "training process exited"

# 2. Locate the best/final checkpoint.
CKPT="$CKPT_DIR/latest.pt"
if [[ ! -f "$CKPT" ]]; then
  log "ERROR: no checkpoint at $CKPT, listing $CKPT_DIR"
  ls -la "$CKPT_DIR" | tee -a "$LOG"
  exit 1
fi
log "using checkpoint $CKPT"

mkdir -p "$RESULTS_DIR"

# 3. Eval main config (full sampler: confidence remask + AR refine).
log "=== eval: main ==="
python eval.py --task all \
  --checkpoint "$CKPT" \
  --output-dir "$RESULTS_DIR/main" \
  --n-samples-per-problem 5 \
  --diffusion-steps 16 \
  --max-new-tokens 256 \
  >> "$LOG" 2>&1 || log "main eval errored, continuing"

# 4. Inference-only ablations (reuse the same checkpoint).
log "=== eval: no_conf_remask ==="
python eval.py --task all \
  --checkpoint "$CKPT" \
  --output-dir "$RESULTS_DIR/no_conf_remask" \
  --n-samples-per-problem 5 \
  --diffusion-steps 16 \
  --max-new-tokens 256 \
  --no-confidence-remask \
  >> "$LOG" 2>&1 || log "no_conf_remask eval errored, continuing"

log "=== eval: no_ar_refine ==="
python eval.py --task all \
  --checkpoint "$CKPT" \
  --output-dir "$RESULTS_DIR/no_ar_refine" \
  --n-samples-per-problem 5 \
  --diffusion-steps 16 \
  --max-new-tokens 256 \
  --no-ar-refine \
  >> "$LOG" 2>&1 || log "no_ar_refine eval errored, continuing"

# 5. Build RESULTS.md.
log "=== writing RESULTS.md ==="
python scripts/build_results.py \
  --results-dir "$RESULTS_DIR" \
  --output RESULTS.md \
  >> "$LOG" 2>&1 || log "build_results errored"

log "auto_finish done"

#!/usr/bin/env bash
set -euo pipefail

# One-click ablation runner for Task 6/7.
# Usage:
#   bash scripts/run_ablations.sh
# Optional env overrides:
#   STEPS=100000 BATCH_SIZE=256 DEVICE=cuda DATA_DIR=data

STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-cuda}"
DATA_DIR="${DATA_DIR:-data}"

LINEAR_OUT="outputs_linear"
COSINE_OUT="outputs_cosine"

echo "[1/4] Training linear schedule run"
python train.py \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --schedule-type linear \
  --out-dir "${LINEAR_OUT}" \
  --data-dir "${DATA_DIR}" \
  --device "${DEVICE}"

echo "[2/4] Training cosine schedule run"
python train.py \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --schedule-type cosine \
  --out-dir "${COSINE_OUT}" \
  --data-dir "${DATA_DIR}" \
  --device "${DEVICE}"

LINEAR_CKPT="${LINEAR_OUT}/checkpoints/ddpm_step_$(printf "%07d" "${STEPS}").pt"
COSINE_CKPT="${COSINE_OUT}/checkpoints/ddpm_step_$(printf "%07d" "${STEPS}").pt"

if [[ ! -f "${LINEAR_CKPT}" ]]; then
  echo "Linear checkpoint not found: ${LINEAR_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${COSINE_CKPT}" ]]; then
  echo "Cosine checkpoint not found: ${COSINE_CKPT}" >&2
  exit 1
fi

echo "[3/4] Running Task 6/7 diagnostics for linear run"
python eval.py \
  --checkpoint "${LINEAR_CKPT}" \
  --compare-checkpoint "${COSINE_CKPT}" \
  --samples 64 \
  --sampling-steps-ablation 1000,250,100,50 \
  --train-log "${LINEAR_OUT}/train_log.csv" \
  --train-log-alt "${COSINE_OUT}/train_log.csv" \
  --train-log-alt-label cosine \
  --run-nn-check \
  --data-dir "${DATA_DIR}" \
  --out-dir "${LINEAR_OUT}/eval"

echo "[4/4] Running Task 6/7 diagnostics for cosine run"
python eval.py \
  --checkpoint "${COSINE_CKPT}" \
  --compare-checkpoint "${LINEAR_CKPT}" \
  --samples 64 \
  --sampling-steps-ablation 1000,250,100,50 \
  --train-log "${COSINE_OUT}/train_log.csv" \
  --train-log-alt "${LINEAR_OUT}/train_log.csv" \
  --train-log-alt-label linear \
  --run-nn-check \
  --data-dir "${DATA_DIR}" \
  --out-dir "${COSINE_OUT}/eval"

echo "Done. Check outputs in:"
echo "  ${LINEAR_OUT}/eval"
echo "  ${COSINE_OUT}/eval"

#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export PYTHONUNBUFFERED=1

mkdir -p logs results/downstream/hc_alignment_strict_scaffold_frozen

LOG="logs/hc_alignment_strict_scaffold_frozen_$(date +%Y%m%d_%H%M%S).log"
OUT="results/downstream/hc_alignment_strict_scaffold_frozen"

{
  echo "[$(date '+%F %T')] START strict frozen encoder + scaffold split"
  echo "repo=$(pwd) branch=$(git branch --show-current) head=$(git rev-parse --short HEAD)"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

  # Activate your conda environment before running this script.
# Example: conda activate your_env_name
# The original script used "conda activate spectra" on a specific server.

  python -u downstream/train_hc_alignment_strict.py \
    --mode frozen_encoder \
    --checkpoint results/checkpoints/best_model.pt \
    --config configs/pretrain_set_transformer.yaml \
    --data_dir data/processed \
    --output_dir "$OUT" \
    --epochs 10 \
    --batch_size 256 \
    --lr 1e-3 \
    --temperature 0.07 \
    --subset 0.0 \
    --seed 42 \
    --num_workers 0 \
    --shard_cache_size 245 \
    --log_interval 30 \
    --split_strategy scaffold

  echo "[$(date '+%F %T')] DONE strict frozen encoder + scaffold split"
  echo "summary=$OUT/summary.csv"
  echo "log=$LOG"
} 2>&1 | tee -a "$LOG"

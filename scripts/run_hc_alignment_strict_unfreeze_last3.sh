#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")/../.."

# Use GPU_ID from env or default to 4
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export PYTHONUNBUFFERED=1

mkdir -p logs results/downstream/hc_alignment_strict_unfreeze_last3

LOG="logs/hc_alignment_strict_unfreeze_last3_$(date +%Y%m%d_%H%M%S).log"
OUT="results/downstream/hc_alignment_strict_unfreeze_last3"

{
  echo "[$(date '+%F %T')] START strict unfreeze-last-3-blocks ablation"
  echo "repo=$(pwd) branch=$(git branch --show-current) head=$(git rev-parse --short HEAD)"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

  # Activate your conda environment before running this script.
# Example: conda activate your_env_name
# The original script used "conda activate spectra" on a specific server.

  python -u downstream/train_hc_alignment_strict.py \
    --mode unfreeze_last_block \
    --unfreeze_last_n_blocks 3 \
    --checkpoint results/checkpoints/best_model.pt \
    --config configs/pretrain_set_transformer.yaml \
    --data_dir data/processed \
    --output_dir "$OUT" \
    --epochs 10 \
    --batch_size 256 \
    --lr 1e-3 \
    --encoder_lr 1e-5 \
    --temperature 0.07 \
    --subset 0.0 \
    --seed 42 \
    --num_workers 0 \
    --shard_cache_size 245 \
    --log_interval 30

  echo "[$(date '+%F %T')] DONE strict unfreeze-last-3-blocks ablation"
  echo "summary=$OUT/summary.csv"
  echo "log=$LOG"
} 2>&1 | tee -a "$LOG"

#!/usr/bin/env bash
set -eo pipefail

cd /root/huangwei/ms-nmr-representation-eval

export CUDA_VISIBLE_DEVICES="${GPU_ID:-5}"
export PYTHONUNBUFFERED=1

mkdir -p logs results/downstream/hc_alignment_strict_scaffold_scratch_frozen

LOG="logs/hc_alignment_strict_scaffold_scratch_frozen_$(date +%Y%m%d_%H%M%S).log"
OUT="results/downstream/hc_alignment_strict_scaffold_scratch_frozen"

{
  echo "[$(date '+%F %T')] START strict scratch_frozen + scaffold split"
  echo "repo=$(pwd) branch=$(git branch --show-current) head=$(git rev-parse --short HEAD)"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

  if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
  elif [ -f /root/anaconda3/etc/profile.d/conda.sh ]; then
    source /root/anaconda3/etc/profile.d/conda.sh
  elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
  else
    source "$(conda info --base)/etc/profile.d/conda.sh"
  fi
  conda activate spectra

  python -u downstream/train_hc_alignment_strict.py \
    --mode scratch_frozen \
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

  echo "[$(date '+%F %T')] DONE"
  echo "summary=$OUT/summary.csv"
  echo "log=$LOG"
} 2>&1 | tee -a "$LOG"

#!/usr/bin/env bash
set -eo pipefail

SUBSET="${SUBSET:-0.1}"
GPU="${GPU:-0}"

cd /root/huangwei/ms-nmr-representation-eval

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

mkdir -p logs "results/downstream/hc_alignment_strict_fewshot_subset${SUBSET}"

LOG="logs/hc_alignment_strict_fewshot_subset${SUBSET}_$(date +%Y%m%d_%H%M%S).log"
OUT="results/downstream/hc_alignment_strict_fewshot_subset${SUBSET}"

{
  echo "[$(date '+%F %T')] START strict frozen + few-shot subset=${SUBSET} GPU=${GPU}"
  echo "repo=$(pwd) branch=$(git branch --show-current) head=$(git rev-parse --short HEAD)"
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
    --mode frozen_encoder \
    --checkpoint results/checkpoints/best_model.pt \
    --config configs/pretrain_set_transformer.yaml \
    --data_dir data/processed \
    --output_dir "$OUT" \
    --epochs 10 \
    --batch_size 256 \
    --lr 1e-3 \
    --temperature 0.07 \
    --subset "$SUBSET" \
    --seed 42 \
    --num_workers 0 \
    --shard_cache_size 245 \
    --log_interval 30

  echo "[$(date '+%F %T')] DONE"
  echo "summary=$OUT/summary.csv"
} 2>&1 | tee -a "$LOG"

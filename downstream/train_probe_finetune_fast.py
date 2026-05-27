#!/usr/bin/env python3
"""
finetune 阶段的高效实现。

科研约束保持不变：
1. 与 train_probe.py 使用相同的任务定义
2. 使用相同的 train / val / test 切分逻辑
3. 每个 epoch 都跑完整验证集
4. 仍然输出逐 run json 与 summary.csv

工程优化：
1. AMP autocast (A100 上默认 bfloat16)
2. fused Adam（若当前 PyTorch 支持）
3. 更大的 train / eval batch
4. DataLoader persistent_workers + prefetch
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from downstream.embedding_utils import (
    extract_pooled_embeddings,
    load_pretrained_encoder,
    set_encoder_trainable,
)
from downstream.task_dataset import SharedTaskData, TASK_SPECS, TaskDataset


class ProbeHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="高效 finetune probe 训练脚本")
    parser.add_argument("--task", required=True, choices=sorted(TASK_SPECS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/pretrain_set_transformer.yaml")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--label_file", default="results/downstream/offline_labels_full.parquet")
    parser.add_argument("--output_dir", default="results/downstream/probe_finetune_fast")
    parser.add_argument("--subset_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--shard_cache_size", type=int, default=2)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_strategy", default="random", choices=["random", "scaffold"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--amp_dtype", choices=["bfloat16", "float16", "none"], default="bfloat16")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, prefetch_factor: int) -> DataLoader:
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def build_loaders(args: argparse.Namespace) -> Tuple[TaskDataset, TaskDataset, TaskDataset, DataLoader, DataLoader, DataLoader]:
    shared_data = SharedTaskData.from_disk(args.data_dir, args.label_file, args.task)
    common_kwargs = dict(
        processed_dir=args.data_dir,
        label_file=args.label_file,
        task=args.task,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_strategy=args.split_strategy,
        max_samples=args.max_samples,
        shard_cache_size=args.shard_cache_size,
        shared_data=shared_data,
    )
    train_dataset = TaskDataset(split="train", subset_ratio=args.subset_ratio, **common_kwargs)
    val_dataset = TaskDataset(split="val", subset_ratio=1.0, **common_kwargs)
    test_dataset = TaskDataset(split="test", subset_ratio=1.0, **common_kwargs)

    train_loader = make_loader(train_dataset, args.train_batch_size, True, args.num_workers, args.prefetch_factor)
    val_loader = make_loader(val_dataset, args.eval_batch_size, False, args.num_workers, args.prefetch_factor)
    test_loader = make_loader(test_dataset, args.eval_batch_size, False, args.num_workers, args.prefetch_factor)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def create_optimizer(encoder: nn.Module, head: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    param_groups = [
        {"params": encoder.parameters(), "lr": args.encoder_lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ]
    fused_supported = torch.cuda.is_available() and "fused" in torch.optim.Adam.__init__.__code__.co_varnames
    if fused_supported:
        return torch.optim.Adam(param_groups, weight_decay=args.weight_decay, fused=True)
    return torch.optim.Adam(param_groups, weight_decay=args.weight_decay)


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    valid_cols = [i for i in range(y_true.shape[1]) if 0 < y_true[:, i].sum() < len(y_true)]
    roc_auc = float(roc_auc_score(y_true[:, valid_cols], y_prob[:, valid_cols], average="macro")) if valid_cols else 0.0
    return {
        "roc_auc_macro": roc_auc,
        "accuracy": float((y_pred == y_true).mean()),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def multiclass_metrics(y_true: np.ndarray, y_logits: np.ndarray) -> Dict[str, float]:
    y_pred = y_logits.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def get_amp_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return torch.autocast(device_type="cuda", dtype=dtype_map[amp_dtype])


def evaluate(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    task_type: str,
    device: torch.device,
    disable_tqdm: bool,
    amp_dtype: str,
) -> Dict[str, float]:
    encoder.eval()
    head.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False, disable=disable_tqdm):
            input_tensor = batch["input_tensor"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"]

            with get_amp_context(device, amp_dtype):
                pooled = extract_pooled_embeddings(encoder, input_tensor, attention_mask)
                logits = head(pooled)

            if task_type == "multilabel":
                all_outputs.append(torch.sigmoid(logits).float().cpu().numpy())
            else:
                all_outputs.append(logits.float().cpu().numpy())
            all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_out = np.concatenate(all_outputs, axis=0)
    if task_type == "multilabel":
        return multilabel_metrics(y_true, y_out)
    return multiclass_metrics(y_true, y_out)


def train_one_epoch(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    disable_tqdm: bool,
    amp_dtype: str,
) -> float:
    encoder.train()
    head.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="Training", leave=False, disable=disable_tqdm):
        input_tensor = batch["input_tensor"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with get_amp_context(device, amp_dtype):
            pooled = extract_pooled_embeddings(encoder, input_tensor, attention_mask)
            logits = head(pooled)
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)


def append_summary_row(summary_path: Path, row: Dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_path.exists()
    fieldnames = [
        "timestamp",
        "task",
        "mode",
        "split_strategy",
        "subset_ratio",
        "seed",
        "train_size",
        "val_size",
        "test_size",
        "best_epoch",
        "best_metric_name",
        "best_val_metric",
        "accuracy",
        "f1_macro",
        "f1_micro",
        "roc_auc_macro",
    ]
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder, config = load_pretrained_encoder(args.checkpoint, args.config, device)
    set_encoder_trainable(encoder, trainable=True)

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_loaders(args)

    head = ProbeHead(config.model.dim_hidden, train_dataset.num_classes).to(device)
    optimizer = create_optimizer(encoder, head, args)
    criterion = nn.BCEWithLogitsLoss() if train_dataset.task_type == "multilabel" else nn.CrossEntropyLoss()
    best_name = "roc_auc_macro" if train_dataset.task_type == "multilabel" else "accuracy"

    best_state = None
    best_val_metric = float("-inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            encoder=encoder,
            head=head,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            disable_tqdm=args.disable_tqdm,
            amp_dtype=args.amp_dtype,
        )
        val_metrics = evaluate(
            encoder=encoder,
            head=head,
            loader=val_loader,
            task_type=train_dataset.task_type,
            device=device,
            disable_tqdm=args.disable_tqdm,
            amp_dtype=args.amp_dtype,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        current_val_metric = float(val_metrics[best_name])
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            best_state = {
                "encoder": encoder.state_dict(),
                "head": head.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"task={args.task} | subset={args.subset_ratio} | "
            f"Loss {train_loss:.4f} | Val {best_name}: {current_val_metric:.4f}",
            flush=True,
        )

    encoder.load_state_dict(best_state["encoder"])
    head.load_state_dict(best_state["head"])

    test_metrics = evaluate(
        encoder=encoder,
        head=head,
        loader=test_loader,
        task_type=train_dataset.task_type,
        device=device,
        disable_tqdm=args.disable_tqdm,
        amp_dtype=args.amp_dtype,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.task}_finetune_fast_subset{args.subset_ratio}_seed{args.seed}"
    output_path = output_dir / f"{run_name}.json"
    summary_path = output_dir / "summary.csv"

    report = {
        "timestamp": datetime.now().isoformat(),
        "task": args.task,
        "task_type": train_dataset.task_type,
        "mode": "finetune_fast",
        "split_strategy": args.split_strategy,
        "subset_ratio": args.subset_ratio,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "dataset": {
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "num_classes": train_dataset.num_classes,
        },
        "training": {
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "encoder_lr": args.encoder_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "amp_dtype": args.amp_dtype,
            "best_epoch": best_state["epoch"],
            "best_metric_name": best_name,
            "best_val_metric": best_val_metric,
        },
        "best_val_metrics": best_state["val_metrics"],
        "test_metrics": test_metrics,
        "history": history,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    append_summary_row(
        summary_path,
        {
            "timestamp": report["timestamp"],
            "task": args.task,
            "mode": "finetune_fast",
            "subset_ratio": args.subset_ratio,
            "seed": args.seed,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "best_epoch": best_state["epoch"],
            "best_metric_name": best_name,
            "best_val_metric": best_val_metric,
            **test_metrics,
        },
    )

    print(f"结果已保存到: {output_path}")
    print(f"汇总已追加到: {summary_path}")


if __name__ == "__main__":
    main()

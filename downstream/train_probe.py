#!/usr/bin/env python3
"""
统一的 probe / few-shot 训练脚本。

当前支持：
1. functional_group
2. element_presence
3. molwt_bin / logp_bin / tpsa_bin / ringcount_bin
4. frozen / finetune
5. subset_ratio few-shot
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
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
    build_encoder_from_config,
    extract_pooled_embeddings,
    load_pretrained_encoder,
    load_model_config,
    set_encoder_trainable,
)
from downstream.task_dataset import TASK_SPECS, SharedTaskData, TaskDataset


class ProbeHead(nn.Module):
    """第一版统一 probe head：Linear + BatchNorm。"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一 probe / few-shot 训练脚本")
    parser.add_argument("--task", required=True, choices=sorted(TASK_SPECS))
    parser.add_argument("--mode", default="frozen", choices=["frozen", "finetune", "scratch"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default="configs/pretrain_set_transformer.yaml")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--label_file", default="results/downstream/offline_labels_full.parquet")
    parser.add_argument("--output_dir", default="results/downstream/probe")
    parser.add_argument("--subset_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3, help="frozen 模式统一学习率")
    parser.add_argument("--encoder_lr", type=float, default=1e-5, help="finetune 模式 encoder 学习率")
    parser.add_argument("--head_lr", type=float, default=1e-3, help="finetune 模式 head 学习率")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shard_cache_size", type=int, default=2)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_strategy", default="random", choices=["random", "scaffold"])
    parser.add_argument("--max_samples", type=int, default=None, help="仅用于快速自测")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable_tqdm", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def build_optimizer(
    encoder: nn.Module,
    head: nn.Module,
    mode: str,
    lr: float,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if mode == "frozen":
        return torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)

    return torch.optim.Adam(
        [
            {"params": encoder.parameters(), "lr": encoder_lr},
            {"params": head.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    y_pred = (y_prob >= 0.5).astype(np.int64)

    valid_cols = [
        i for i in range(y_true.shape[1])
        if 0 < y_true[:, i].sum() < len(y_true)
    ]
    if valid_cols:
        metrics["roc_auc_macro"] = float(
            roc_auc_score(y_true[:, valid_cols], y_prob[:, valid_cols], average="macro")
        )
    else:
        metrics["roc_auc_macro"] = 0.0

    metrics["accuracy"] = float((y_pred == y_true).mean())
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    return metrics


def multiclass_metrics(y_true: np.ndarray, y_logits: np.ndarray) -> Dict[str, float]:
    y_pred = y_logits.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    task_type: str,
    device: torch.device,
    disable_tqdm: bool,
) -> Dict[str, float]:
    encoder.eval()
    head.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False, disable=disable_tqdm):
            input_tensor = batch["input_tensor"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            pooled = extract_pooled_embeddings(encoder, input_tensor, attention_mask)
            logits = head(pooled)

            if task_type == "multilabel":
                all_outputs.append(torch.sigmoid(logits).cpu().numpy())
            else:
                all_outputs.append(logits.cpu().numpy())
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
    mode: str,
    disable_tqdm: bool,
) -> float:
    if mode == "frozen":
        encoder.eval()
    else:
        encoder.train()
    head.train()

    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="Training", leave=False, disable=disable_tqdm):
        input_tensor = batch["input_tensor"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if mode == "frozen":
            with torch.no_grad():
                pooled = extract_pooled_embeddings(encoder, input_tensor, attention_mask)
        else:
            pooled = extract_pooled_embeddings(encoder, input_tensor, attention_mask)

        logits = head(pooled)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)


def best_metric_name(task_type: str) -> str:
    if task_type == "multilabel":
        return "roc_auc_macro"
    return "accuracy"


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


def load_encoder_for_mode(
    mode: str,
    checkpoint_path: str | None,
    config_path: str,
    device: torch.device,
) -> Tuple[nn.Module, object]:
    if mode == "scratch":
        config = load_model_config(config_path)
        encoder = build_encoder_from_config(config).to(device)
        return encoder, config

    if checkpoint_path is None:
        raise ValueError(f"mode={mode} 需要提供 --checkpoint")

    return load_pretrained_encoder(checkpoint_path, config_path, device)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder, config = load_encoder_for_mode(args.mode, args.checkpoint, args.config, device)
    set_encoder_trainable(encoder, trainable=(args.mode != "frozen"))

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_loaders(args)

    head = ProbeHead(config.model.dim_hidden, train_dataset.num_classes).to(device)
    optimizer = build_optimizer(
        encoder=encoder,
        head=head,
        mode=args.mode,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    criterion = (
        nn.BCEWithLogitsLoss()
        if train_dataset.task_type == "multilabel"
        else nn.CrossEntropyLoss()
    )

    best_name = best_metric_name(train_dataset.task_type)
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
            mode=args.mode,
            disable_tqdm=args.disable_tqdm,
        )
        val_metrics = evaluate(
            encoder=encoder,
            head=head,
            loader=val_loader,
            task_type=train_dataset.task_type,
            device=device,
            disable_tqdm=args.disable_tqdm,
        )

        epoch_record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(epoch_record)

        current_val_metric = float(val_metrics[best_name])
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            best_state = {
                "encoder": encoder.state_dict() if args.mode != "frozen" else None,
                "head": head.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Loss {train_loss:.4f} | "
            f"Val {best_name}: {current_val_metric:.4f}"
        )

    if best_state is not None:
        head.load_state_dict(best_state["head"])
        if args.mode != "frozen" and best_state["encoder"] is not None:
            encoder.load_state_dict(best_state["encoder"])

    test_metrics = evaluate(
        encoder=encoder,
        head=head,
        loader=test_loader,
        task_type=train_dataset.task_type,
        device=device,
        disable_tqdm=args.disable_tqdm,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.task}_{args.mode}_subset{args.subset_ratio}_seed{args.seed}"
    output_path = output_dir / f"{run_name}.json"
    summary_path = output_dir / "summary.csv"

    report = {
        "timestamp": datetime.now().isoformat(),
        "task": args.task,
        "task_type": train_dataset.task_type,
        "mode": args.mode,
        "split_strategy": args.split_strategy,
        "subset_ratio": args.subset_ratio,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "label_file": args.label_file,
        "dataset": {
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "num_classes": train_dataset.num_classes,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "encoder_lr": args.encoder_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "best_epoch": None if best_state is None else best_state["epoch"],
            "best_metric_name": best_name,
            "best_val_metric": best_val_metric,
        },
        "best_val_metrics": {} if best_state is None else best_state["val_metrics"],
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
            "mode": args.mode,
            "subset_ratio": args.subset_ratio,
            "seed": args.seed,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "best_epoch": report["training"]["best_epoch"],
            "best_metric_name": best_name,
            "best_val_metric": best_val_metric,
            **test_metrics,
        },
    )

    print(f"结果已保存到: {output_path}")
    print(f"汇总已追加到: {summary_path}")


if __name__ == "__main__":
    main()

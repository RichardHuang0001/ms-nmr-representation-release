#!/usr/bin/env python3
"""
线性探测训练脚本 (Linear Probing Training)

用于验证预训练 SetTransformer 的表征质量。
冻结编码器权重，仅训练线性分类头来预测官能团。

用法:
    python downstream/train_linear_probe.py --checkpoint results/checkpoints/best_model.pt
    python downstream/train_linear_probe.py --checkpoint <path> --max_samples 1000  # 快速测试
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse
import yaml
from box import Box
import json
from datetime import datetime
import numpy as np

from src.models.set_transformer import PretrainSetTransformer
from downstream.probe_dataset import LinearProbeDataset


class LinearProbe(nn.Module):
    """简单的线性分类头"""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(self.bn(x))


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    对序列进行 Mean Pooling
    
    Args:
        hidden_states: [B, L, H] 编码器输出
        attention_mask: [B, L] 注意力掩码 (1=真实, 0=padding)
    
    Returns:
        [B, H] 池化后的表征
    """
    mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
    sum_hidden = (hidden_states * mask).sum(dim=1)  # [B, H]
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)  # [B, 1]
    return sum_hidden / sum_mask


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
    """计算多标签分类指标"""
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    
    metrics = {}
    
    # ROC-AUC (Macro)
    try:
        # 过滤掉全0和全1的列
        valid_cols = []
        for i in range(y_true.shape[1]):
            if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true):
                valid_cols.append(i)
        
        if valid_cols:
            metrics['roc_auc_macro'] = roc_auc_score(
                y_true[:, valid_cols], y_pred[:, valid_cols], average='macro'
            )
            metrics['valid_classes'] = len(valid_cols)
        else:
            metrics['roc_auc_macro'] = 0.0
            metrics['valid_classes'] = 0
    except Exception as e:
        metrics['roc_auc_macro'] = 0.0
        metrics['roc_auc_error'] = str(e)
    
    # 二值化预测
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # F1 Score
    try:
        metrics['f1_macro'] = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
    except Exception:
        metrics['f1_macro'] = 0.0
        metrics['f1_micro'] = 0.0
    
    # Precision & Recall
    try:
        metrics['precision_macro'] = precision_score(y_true, y_pred_binary, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred_binary, average='macro', zero_division=0)
    except Exception:
        metrics['precision_macro'] = 0.0
        metrics['recall_macro'] = 0.0
    
    return metrics


def train_epoch(encoder, probe, loader, optimizer, criterion, device):
    """训练一个 epoch"""
    probe.train()
    total_loss = 0
    
    for x, mask, y in tqdm(loader, desc="Training", leave=False):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        
        # 编码器前向（无梯度）
        with torch.no_grad():
            hidden = encoder.encode(x, mask)  # [B, L, H]
            pooled = mean_pooling(hidden, mask)  # [B, H]
        
        # 线性头前向
        logits = probe(pooled)
        loss = criterion(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(encoder, probe, loader, device):
    """评估模型"""
    probe.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, mask, y in tqdm(loader, desc="Evaluating", leave=False):
            x, mask = x.to(device), mask.to(device)
            
            hidden = encoder.encode(x, mask)
            pooled = mean_pooling(hidden, mask)
            logits = probe(pooled)
            
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(y.numpy())
    
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    
    return calculate_metrics(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description="线性探测训练")
    parser.add_argument("--checkpoint", required=True, help="预训练模型检查点路径")
    parser.add_argument("--data_dir", default="data/processed", help="预处理数据目录")
    parser.add_argument("--config", default="configs/pretrain_set_transformer.yaml", help="模型配置文件")
    parser.add_argument("--output", default="results/evaluation/linear_probe_report.json", help="输出报告路径")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数（用于测试）")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--device", default="cuda", help="设备")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  线性探测验证 (Linear Probing)")
    print("=" * 70)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载配置和模型
    print("\n[1/5] 加载预训练模型...")
    with open(args.config) as f:
        config = Box(yaml.safe_load(f))
    
    encoder = PretrainSetTransformer(
        dim_input=config.model.dim_input,
        dim_output=config.model.dim_output,
        dim_hidden=config.model.dim_hidden,
        num_heads=config.model.num_heads,
        depth=config.model.depth,
    ).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(ckpt['model_state_dict'])
    
    # 冻结编码器
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    
    print(f"  ✓ 模型加载自: {args.checkpoint}")
    print(f"  ✓ 编码器已冻结 ({sum(p.numel() for p in encoder.parameters()):,} 参数)")
    
    # 2. 准备数据
    print("\n[2/5] 加载数据集...")
    dataset = LinearProbeDataset(args.data_dir, max_samples=args.max_samples)
    
    # 划分训练/测试集 (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)
    
    print(f"  ✓ 训练集: {train_size}, 测试集: {test_size}")
    print(f"  ✓ 类别数: {dataset.num_classes}")
    
    # 3. 初始化分类头
    print("\n[3/5] 初始化线性分类头...")
    probe = LinearProbe(config.model.dim_hidden, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"  ✓ 分类头参数: {sum(p.numel() for p in probe.parameters()):,}")
    
    # 4. 训练
    print("\n[4/5] 开始训练...")
    best_auc = 0
    training_history = []
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(encoder, probe, train_loader, optimizer, criterion, device)
        metrics = evaluate(encoder, probe, test_loader, device)
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **metrics
        })
        
        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"ROC-AUC: {metrics['roc_auc_macro']:.4f} | "
              f"F1-Macro: {metrics['f1_macro']:.4f}")
        
        if metrics['roc_auc_macro'] > best_auc:
            best_auc = metrics['roc_auc_macro']
    
    # 5. 最终评估与报告
    print("\n[5/5] 生成评估报告...")
    final_metrics = evaluate(encoder, probe, test_loader, device)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "dataset": {
            "total_samples": len(dataset),
            "train_size": train_size,
            "test_size": test_size,
            "num_classes": dataset.num_classes,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        },
        "final_metrics": final_metrics,
        "best_roc_auc": best_auc,
        "training_history": training_history,
    }
    
    # 保存报告
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 70)
    print("  评估完成")
    print("=" * 70)
    print(f"  最佳 ROC-AUC: {best_auc:.4f}")
    print(f"  最终 F1-Macro: {final_metrics['f1_macro']:.4f}")
    print(f"  有效类别数: {final_metrics.get('valid_classes', 'N/A')}")
    print(f"  报告保存至: {output_path}")


if __name__ == "__main__":
    main()

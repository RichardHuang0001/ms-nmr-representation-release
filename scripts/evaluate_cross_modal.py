#!/usr/bin/env python3
"""
跨模态检索验证脚本 (Cross-Modal Retrieval Evaluation)

核心目标：验证模型生成的 H-NMR 表征与 C-NMR/MS 表征是否在几何空间上对齐。

评估任务:
    - H-NMR → C-NMR 检索
    - H-NMR → MS 检索
    - C-NMR → H-NMR 检索
    - C-NMR → MS 检索

两种评估模式:
    - 默认模式: 编码器看到所有峰，但只从目标模态峰提取表征
    - 严格致盲模式 (--strict_blind): 非目标模态的峰在输入时被置零

⚠️ 重要改动（为大规模数据而改）：
    - 不再构建 NxN 相似度矩阵（会爆内存/不可计算）
    - 改为 GPU 上分块计算 Query@Gallery，并只保留 Top-K 来计算指标
    - 支持 eval_size/gallery_size 抽样，让实验既学术合理又能跑完

运行命令:
mkdir -p logs results/evaluation

python -u scripts/evaluate_cross_modal.py \
  --checkpoint results/checkpoints/best_model.pt \
  --subset 0 \
  --eval_size 50000 \
  --gallery_size 200000 \
  --topk 10 \
  --seed 0 \
  --sim_batch 2048 \
  --use_fp16 \
  --num_workers 4 \
  --pin_memory \
  2>&1 | tee logs/eval_cross_scalable_s0_q50k_g200k.log
  
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from box import Box
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.set_transformer import PretrainSetTransformer
from src.data.dataset import SpectraDataset


class CrossModalEvaluator:
    """
    跨模态检索评估器
    
    核心思想：
    1) 对于每个分子，只使用某一个模态的峰生成 Query 表征
    2) 使用另一个模态的峰生成 Gallery 表征
    3) 检查 Query 是否能检索到对应分子的 Gallery（正确匹配为同 index）
    """
    
    def __init__(self, model, device, config, strict_blind=False):
        self.model = model
        self.device = device
        self.config = config
        self.strict_blind = strict_blind
        self.model.eval()
        
        # 从配置中获取模态映射
        modality_map_raw = config.peak_vector.modality_map
        self.modality_map = modality_map_raw.to_dict() if hasattr(modality_map_raw, "to_dict") else dict(modality_map_raw)
        
        # 定义模态组
        self.modality_groups = {
            'h_nmr': [self.modality_map['h_nmr_peaks']],
            'c_nmr': [self.modality_map['c_nmr_peaks']],
            'ms': [
                self.modality_map['msms_positive_10ev'],
                self.modality_map['msms_positive_20ev'],
                self.modality_map['msms_positive_40ev'],
                self.modality_map['msms_negative_10ev'],
                self.modality_map['msms_negative_20ev'],
                self.modality_map['msms_negative_40ev'],
            ]
        }
        
        self.hidden_dim = config.model.dim_hidden
        self.peak_dim = config.peak_vector.dim
    
    def _identify_modality_mask(self, inputs, modality_name):
        """
        识别属于目标模态的峰
        
        Args:
            inputs: (B, L, D)
            modality_name: 'h_nmr'/'c_nmr'/'ms'
        Returns:
            is_target_modal: (B, L) bool
        """
        target_indices = self.modality_groups[modality_name]
        is_target_modal = torch.zeros(inputs.shape[0], inputs.shape[1], dtype=torch.bool, device=self.device)
        for idx in target_indices:
            is_target_modal = is_target_modal | (inputs[:, :, idx] > 0.5)
        return is_target_modal
    
    def encode_with_modality_filter(self, batch, modality_name):
        """
        只通过指定模态提取分子表征
        
        Returns:
            global_embedding: (B, hidden_dim) 已归一化
            valid_samples: (B,) bool 标记是否在该模态有有效峰
        """
        inputs, attention_masks, _ = batch
        inputs = inputs.to(self.device, non_blocking=True)
        attention_masks = attention_masks.to(self.device, non_blocking=True)
        
        # 识别目标模态峰
        is_target_modal = self._identify_modality_mask(inputs, modality_name)
        valid_peaks = attention_masks.bool() & is_target_modal
        valid_samples = valid_peaks.any(dim=1)
        
        # 严格致盲
        if self.strict_blind:
            inputs = inputs.clone()
            non_target_peaks = attention_masks.bool() & (~is_target_modal)
            inputs[non_target_peaks] = 0.0
            attention_masks = attention_masks.clone()
            attention_masks[non_target_peaks] = 0
        
        # Encoder
        with torch.no_grad():
            x = self.model.encoder_input_proj(inputs)
            for layer in self.model.encoder:
                x = layer(x, mask=attention_masks)
            encoded_features = x  # (B,L,H)
        
        # 只对目标模态做 pooling
        mask_expanded = valid_peaks.unsqueeze(-1).float()
        sum_features = (encoded_features * mask_expanded).sum(dim=1)
        count_features = mask_expanded.sum(dim=1).clamp(min=1e-9)
        global_embedding = sum_features / count_features
        
        # 归一化用于 cosine similarity = dot product
        global_embedding = F.normalize(global_embedding, p=2, dim=1)
        
        return global_embedding, valid_samples

    @staticmethod
    def _sample_indices(n, k, seed, device="cpu"):
        """从 [0,n) 抽 k 个不重复 index"""
        if k <= 0 or k >= n:
            return torch.arange(n, device=device)
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:k]
        return idx.to(device)

    def evaluate_pair(
        self,
        dataloader,
        query_mod,
        gallery_mod,
        topk=10,
        eval_size=50000,
        gallery_size=200000,
        seed=0,
        sim_batch=2048,
        use_fp16=True
    ):
        """
        评估一对模态的检索性能（大规模可跑版）

        关键点：
        - 先提取所有有效样本对的 Q/G embedding（CPU 上存放）
        - 再抽样形成 query 子集 & gallery 子集（学术合理且可控）
        - 用 GPU 分块计算 Query@Gallery，只保留 topk 用于指标

        Args:
            topk: 计算 topk 指标
            eval_size: query 数量（0 表示用全部 query，不推荐）
            gallery_size: gallery 数量（0 表示用全部 gallery，不推荐）
            seed: 抽样随机种子
            sim_batch: 每次在 GPU 上处理的 query batch
            use_fp16: 是否用 half 加速（推荐 True）

        Returns:
            None, metrics
        """
        print(f"\n{'='*60}")
        print(f"评估: {query_mod} (Query) → {gallery_mod} (Gallery)")
        print(f"{'='*60}")
        
        all_q_embs = []
        all_g_embs = []
        
        for batch in tqdm(dataloader, desc="提取表征"):
            q_emb, q_valid = self.encode_with_modality_filter(batch, query_mod)
            g_emb, g_valid = self.encode_with_modality_filter(batch, gallery_mod)
            valid_mask = q_valid & g_valid
            
            if valid_mask.sum() > 0:
                # 放到 CPU，避免 GPU 常驻爆显存
                all_q_embs.append(q_emb[valid_mask].detach().cpu())
                all_g_embs.append(g_emb[valid_mask].detach().cpu())
        
        if not all_q_embs:
            print("⚠️ 警告: 数据集中没有同时包含这两个模态的样本！")
            return None, None
        
        Q = torch.cat(all_q_embs, dim=0)  # (N,D) CPU
        G = torch.cat(all_g_embs, dim=0)  # (N,D) CPU
        
        n_total = Q.shape[0]
        print(f"有效样本对数量: {n_total}")
        print(f"随机猜测 Top-1 准确率: {100.0/n_total:.6f}%")

        # --- 抽样设置（学术 + 算力兼顾）---
        # 说明：正确匹配默认是“同 index”，因此抽样时必须对 Q/G 使用同一组 index 才保持对角线对应。
        # 这里我们做法是：
        #   1) 先从全体有效样本对中抽一个“pool”（用于保证 Q/G 对齐）
        #   2) 从 pool 里再选 query 子集 & gallery 子集（gallery 必须包含 query 的正确项）
        #
        # 最稳妥：让 gallery 是 pool 的一个大子集；query 是更小子集。
        # 且要求 gallery 覆盖 query 的 index（这里实现保证这一点）。

        # 先确定一个 pool（如果你希望 query/gallery 都来自一个可控池子）
        # 为简单起见，这里 pool 就是全部有效样本对；然后直接做 query/gallery 抽样。

        # 1) 抽 gallery index（从 0..n_total-1）
        if gallery_size and gallery_size > 0 and gallery_size < n_total:
            gallery_idx = self._sample_indices(n_total, gallery_size, seed=seed, device="cpu")
        else:
            gallery_idx = torch.arange(n_total)

        # 2) 抽 query index（必须落在 gallery 内，否则正确匹配不在库里）
        #    做法：从 gallery_idx 中再抽 eval_size 个作为 query
        g_n = gallery_idx.numel()
        if eval_size and eval_size > 0 and eval_size < g_n:
            # 在 gallery 内采样 query
            local_q_idx = self._sample_indices(g_n, eval_size, seed=seed + 12345, device="cpu")
            query_idx = gallery_idx[local_q_idx]
        else:
            query_idx = gallery_idx  # query=gallery（不推荐大规模）

        # 构造评测 Q/G
        Q_eval = Q[query_idx]          # (Nq,D)
        G_eval = G[gallery_idx]        # (Ng,D)

        # 建立 “query 正确答案在 gallery 的位置”
        # query_idx 是全局 index；gallery_idx 是全局 index
        # 需要 mapping: global_id -> pos_in_gallery
        # 用 dict/哈希会慢；用排序 + searchsorted 更快
        gallery_sorted, sort_pos = torch.sort(gallery_idx)
        # 对每个 query_idx 找到其在 gallery_sorted 里的位置
        # 因为 query_idx 来自 gallery_idx，所以一定能找到
        q_pos_in_sorted = torch.searchsorted(gallery_sorted, query_idx)
        # 转回原 gallery 顺序的位置
        true_pos_in_gallery = sort_pos[q_pos_in_sorted]  # (Nq,)

        n_q = Q_eval.shape[0]
        n_g = G_eval.shape[0]
        K = min(topk, n_g)

        print(f"评测规模: Query={n_q}, Gallery={n_g}, TopK={K} (seed={seed})")
        if n_g != n_q:
            print("注意：这里是“Query 子集 → Gallery 子集”的检索设置（学术更常见，也更可跑）。")

        # --- GPU 分块计算 topk ---
        print("计算检索指标（GPU 分块，不构建 NxN）...")

        device = self.device
        # 把 gallery 放上 GPU 一次
        G_gpu = G_eval.to(device, non_blocking=True)
        Q_cpu = Q_eval  # 留在 CPU，分块搬运

        if use_fp16:
            G_gpu = G_gpu.half()

        top1_correct = 0
        top5_correct = 0
        top10_correct = 0
        rr_sum = 0.0
        ranks_collect = []

        true_pos_in_gallery = true_pos_in_gallery.to(device, non_blocking=True)

        for start in tqdm(range(0, n_q, sim_batch), desc="检索(topk)"):
            end = min(start + sim_batch, n_q)
            Qb = Q_cpu[start:end].to(device, non_blocking=True)
            if use_fp16:
                Qb = Qb.half()

            # (b, Ng)
            scores = torch.matmul(Qb, G_gpu.t())

            # topk
            _, inds = torch.topk(scores, k=K, dim=1, largest=True, sorted=True)

            # 正确位置 (b,)
            true = true_pos_in_gallery[start:end].unsqueeze(1)  # (b,1)
            hit = (inds == true)

            # Acc@1/5/10（当 K 小于 5/10 时，用 K 替代）
            top1_correct += hit[:, :1].any(dim=1).sum().item()

            k5 = min(5, K)
            top5_correct += hit[:, :k5].any(dim=1).sum().item()

            k10 = min(10, K)
            top10_correct += hit[:, :k10].any(dim=1).sum().item()

            # MRR@K
            has_hit = hit.any(dim=1)
            # 第一个命中位置（1-based）
            hit_float = hit.float()
            first_pos = torch.argmax(hit_float, dim=1) + 1
            rr = torch.where(has_hit, 1.0 / first_pos.float(), torch.zeros_like(first_pos, dtype=torch.float))
            rr_sum += rr.sum().item()

            # Median rank@K：没命中记为 K+1
            rank_for_median = torch.where(has_hit, first_pos.float(), torch.full_like(first_pos, K + 1, dtype=torch.float))
            ranks_collect.append(rank_for_median.detach().cpu())

        acc1 = top1_correct / n_q
        acc5 = top5_correct / n_q
        acc10 = top10_correct / n_q
        mrr_at_k = rr_sum / n_q
        ranks_all = torch.cat(ranks_collect, dim=0)
        median_rank_at_k = torch.median(ranks_all).item()

        metrics = {
            'n_total_pairs': int(n_total),
            'n_query': int(n_q),
            'n_gallery': int(n_g),
            'topk': int(K),
            'random_baseline_top1_percent': float(100.0 / n_g),
            'acc_top1': float(acc1),
            'acc_top5': float(acc5),
            'acc_top10': float(acc10),
            'mrr_at_k': float(mrr_at_k),
            'median_rank_at_k': float(median_rank_at_k),
            'seed': int(seed),
            'use_fp16': bool(use_fp16),
            'sim_batch': int(sim_batch),
        }

        print(f"\n📊 结果 ({query_mod} → {gallery_mod}) [Query={n_q}, Gallery={n_g}, TopK={K}]")
        print(f"  {'Top-1 Accuracy:':<20} {acc1:>8.2%}")
        print(f"  {'Top-5 Accuracy:':<20} {acc5:>8.2%}")
        print(f"  {'Top-10 Accuracy:':<20} {acc10:>8.2%}")
        print(f"  {'MRR@K:':<20} {mrr_at_k:>8.4f}")
        print(f"  {'Median Rank@K:':<20} {median_rank_at_k:>8.1f}")

        return None, metrics


def find_checkpoint():
    candidates = [
        "results/checkpoints/best_model.pt",
        "results/checkpoints/latest_model.pt",
        "checkpoints/best_model.pt",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    from glob import glob
    for pattern in ["results/checkpoints/*.pt", "checkpoints/*.pt"]:
        files = glob(pattern)
        if files:
            return files[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="跨模态检索验证（大规模可跑版）")
    parser.add_argument("--checkpoint", default=None, help="模型权重路径")
    parser.add_argument("--data_dir", default="data/processed", help="数据目录")
    parser.add_argument("--config", default="configs/pretrain_set_transformer.yaml", help="配置文件")
    parser.add_argument("--batch_size", type=int, default=256, help="提取表征时的 batch（推理可以大一点）")
    parser.add_argument("--subset", type=int, default=5000,
                        help="只取前N个样本（设为0则跑全量数据集）")
    parser.add_argument("--output", default="results/evaluation/cross_modal_report.json", help="输出报告路径")
    parser.add_argument("--device", default="cuda", help="设备 (cuda/cpu)")
    parser.add_argument("--strict_blind", action="store_true", help="严格致盲模式")

    # 新增：大规模检索评测参数
    parser.add_argument("--eval_size", type=int, default=50000,
                        help="Query 数量（从有效样本对中抽样）。0=使用全部 query（不推荐）")
    parser.add_argument("--gallery_size", type=int, default=200000,
                        help="Gallery 数量（从有效样本对中抽样）。0=使用全部 gallery（不推荐）")
    parser.add_argument("--seed", type=int, default=0, help="抽样随机种子")
    parser.add_argument("--topk", type=int, default=10, help="计算 Top-K 指标的 K")
    parser.add_argument("--sim_batch", type=int, default=2048, help="检索时 query 的分块 batch 大小")
    parser.add_argument("--use_fp16", action="store_true", help="检索相似度计算使用 fp16（推荐开启）")

    # DataLoader 参数（降低 CPU 内存压力/更稳定）
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers（OOM 可设为 0）")
    parser.add_argument("--pin_memory", action="store_true", help="pin_memory（GPU 推理推荐）")

    args = parser.parse_args()

    print("=" * 70)
    print("跨模态检索验证 (Cross-Modal Retrieval Evaluation) - Scalable")
    print("=" * 70)

    if args.strict_blind:
        print("📌 模式: 严格致盲 (Strict Blind)")
    else:
        print("📌 模式: 默认 (全峰编码，单模态 pooling)")

    # 1) 配置
    print("\n[1/4] 加载配置...")
    with open(args.config) as f:
        config = Box(yaml.safe_load(f))
    print(f"  ✓ 配置文件: {args.config}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"  ✓ 使用设备: {device}")

    # 2) 模型
    print("\n[2/4] 加载模型...")
    checkpoint_path = args.checkpoint or find_checkpoint()
    if checkpoint_path is None or (not os.path.exists(checkpoint_path)):
        print("  ❌ 错误: 未找到模型检查点！请用 --checkpoint 指定。")
        return

    model = PretrainSetTransformer(
        dim_input=config.model.dim_input,
        dim_output=config.model.dim_output,
        dim_hidden=config.model.dim_hidden,
        num_heads=config.model.num_heads,
        depth=config.model.depth,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    epoch = checkpoint.get('epoch', -1) + 1
    val_loss = checkpoint.get('val_loss', 0.0)
    print(f"  ✓ 模型检查点: {checkpoint_path}")
    print(f"  ✓ Epoch: {epoch}, Val Loss: {val_loss:.4f}")

    # 3) 数据
    print("\n[3/4] 加载数据...")
    dataset = SpectraDataset(args.data_dir, args.config, masking_fraction=0.0)

    if args.subset > 0 and args.subset < len(dataset):
        print(f"  截取前 {args.subset} 个样本进行验证")
        dataset.samples = dataset.samples[:args.subset]

    print(f"  ✓ 数据集大小: {len(dataset)} 个样本")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory and device.type == 'cuda')
    )

    # 4) 评估
    print("\n[4/4] 执行跨模态检索评估...")
    evaluator = CrossModalEvaluator(model, device, config, strict_blind=args.strict_blind)

    results = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': checkpoint_path,
        'evaluation_mode': 'strict_blind' if args.strict_blind else 'default',
        'n_samples_total_dataset': int(len(dataset)),
        'eval_sampling': {
            'eval_size_query': int(args.eval_size),
            'gallery_size': int(args.gallery_size),
            'seed': int(args.seed),
            'topk': int(args.topk),
            'sim_batch': int(args.sim_batch),
            'use_fp16': bool(args.use_fp16),
        },
        'evaluations': {}
    }

    # A: H -> C
    _, metrics_hc = evaluator.evaluate_pair(
        dataloader, 'h_nmr', 'c_nmr',
        topk=args.topk, eval_size=args.eval_size, gallery_size=args.gallery_size,
        seed=args.seed, sim_batch=args.sim_batch, use_fp16=args.use_fp16
    )
    if metrics_hc:
        results['evaluations']['h_nmr_to_c_nmr'] = metrics_hc

    # B: H -> MS
    _, metrics_hm = evaluator.evaluate_pair(
        dataloader, 'h_nmr', 'ms',
        topk=args.topk, eval_size=args.eval_size, gallery_size=args.gallery_size,
        seed=args.seed, sim_batch=args.sim_batch, use_fp16=args.use_fp16
    )
    if metrics_hm:
        results['evaluations']['h_nmr_to_ms'] = metrics_hm

    # C: C -> H
    _, metrics_ch = evaluator.evaluate_pair(
        dataloader, 'c_nmr', 'h_nmr',
        topk=args.topk, eval_size=args.eval_size, gallery_size=args.gallery_size,
        seed=args.seed, sim_batch=args.sim_batch, use_fp16=args.use_fp16
    )
    if metrics_ch:
        results['evaluations']['c_nmr_to_h_nmr'] = metrics_ch

    # D: C -> MS
    _, metrics_cm = evaluator.evaluate_pair(
        dataloader, 'c_nmr', 'ms',
        topk=args.topk, eval_size=args.eval_size, gallery_size=args.gallery_size,
        seed=args.seed, sim_batch=args.sim_batch, use_fp16=args.use_fp16
    )
    if metrics_cm:
        results['evaluations']['c_nmr_to_ms'] = metrics_cm

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ 结果已保存: {output_path}")

    # 总结
    print("\n" + "=" * 70)
    print("📋 跨模态检索评估总结（可扩展版）")
    print("=" * 70)

    print(f"\n{'任务':<25} {'Top-1':<10} {'Top-5':<10} {'MRR@K':<10} {'Q':<10} {'G':<10}")
    print("-" * 80)

    for task_name, metrics in results['evaluations'].items():
        readable = task_name.replace('_to_', ' → ').replace('_', '-').upper()
        print(f"{readable:<25} {metrics['acc_top1']*100:>7.2f}%  "
              f"{metrics['acc_top5']*100:>7.2f}%  {metrics['mrr_at_k']:>8.4f}  "
              f"{metrics['n_query']:>8}  {metrics['n_gallery']:>8}")

    print("-" * 80)
    print("\n📖 提示：")
    print("  • 这里是 Query 子集 → Gallery 子集 的 Top-K 检索评测（大规模常用设置）")
    print("  • 如果你想更稳健：建议 seed=0/1/2 各跑一次，报告 mean±std")
    print("  • 开启 --use_fp16 通常能显著提速并节省显存")
    print("\n" + "=" * 70)
    print("✅ 评估完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
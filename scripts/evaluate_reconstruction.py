#!/usr/bin/env python3
"""
重建质量评估脚本

评估预训练模型在峰重建任务上的表现，包括：
- 按模态分解（H-NMR/C-NMR/MS）
- 按特征分解（position/intensity/multiplicity/coupling）
- 按masking策略分解

用法:
    python scripts/evaluate_reconstruction.py                    # 完整评估
    python scripts/evaluate_reconstruction.py --test             # 快速测试(1000样本)
    python scripts/evaluate_reconstruction.py --visualize        # 包含可视化
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from box import Box
import matplotlib
matplotlib.use('Agg')  # 无图形界面后端
import matplotlib.pyplot as plt

from src.models.set_transformer import PretrainSetTransformer
from src.data.dataset import SpectraDataset


# ============================================================================
# 辅助函数
# ============================================================================

def init_stats():
    """初始化统计字典"""
    return {
        'position_errors': [],
        'intensity_errors': [],
        'integration_errors': [],
        'width_errors': [],
        'n_peaks': 0
    }


def safe_mean(values):
    """安全的均值计算"""
    return float(np.mean(values)) if len(values) > 0 else 0.0


def compute_r2(errors):
    """从误差计算R²（简化版）"""
    if len(errors) == 0:
        return 0.0
    
    # 简化：基于MSE计算相对拟合度
    mse = np.mean([e**2 for e in errors])
    # 将MSE映射到R²范围（0-1），假设理想MSE接近0
    # 这是简化版本，真实R²需要原始值的方差
    return float(max(0.0, 1.0 - min(mse, 1.0)))


def identify_masking_strategy(label, feature_slices):
    """
    识别masking策略
    
    基于dataset.py的逻辑：
    - position: 只mask维度8
    - y_axis: mask维度9-11 (intensity, integration, width)
    - coupling: mask维度12+ (multiplicity, j_mean, j_count)
    """
    # Position masking: 只mask维度8
    if label[8] != -100 and all(label[9:] == -100):
        return 'position_mask'
    
    # Y-axis masking: mask维度9-11
    if all(label[9:12] != -100) and label[8] == -100 and all(label[12:] == -100):
        return 'y_axis_mask'
    
    # Coupling masking: mask维度12+
    if any(label[12:] != -100):
        return 'coupling_mask'
    
    return 'unknown'


# ============================================================================
# 核心评估函数
# ============================================================================

def evaluate_reconstruction(model, dataloader, config, device):
    """
    评估重建质量
    
    返回包含所有统计信息的字典
    """
    model.eval()
    
    # 初始化统计容器
    stats = {
        'by_modality': {f'mod_{i}': init_stats() for i in range(8)},
        'by_feature': {
            'position': {'errors': []},
            'intensity': {'errors': []},
            'integration': {'errors': []},
            'width': {'errors': []},
            'multiplicity': {'correct': 0, 'total': 0, 'predictions': [], 'targets': []},
            'coupling_j_mean': {'errors': []},
            'coupling_j_count': {'errors': []}
        },
        'by_strategy': {
            'position_mask': init_stats(),
            'y_axis_mask': init_stats(),
            'coupling_mask': init_stats()
        }
    }
    
    feature_slices = config.peak_vector.feature_slices
    
    print("  提取特征切片索引...")
    s_pos, e_pos = feature_slices.position
    s_int, e_int = feature_slices.intensity
    s_integ, e_integ = feature_slices.integration
    s_width, e_width = feature_slices.width
    s_mult, e_mult = feature_slices.multiplicity
    s_jmean, e_jmean = feature_slices.j_mean
    s_jcount, e_jcount = feature_slices.j_count
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  评估中"):
            inputs, masks, labels = [b.to(device) for b in batch]
            
            # 预测
            predictions = model(inputs, masks)
            
            # 找到被mask的峰
            masked_peaks = (labels != -100).any(dim=-1)  # [B, L]
            
            for b in range(inputs.shape[0]):
                for i in range(inputs.shape[1]):
                    if not masked_peaks[b, i]:
                        continue
                    
                    # 获取这个峰的模态
                    modality_id = torch.argmax(inputs[b, i, :8]).item()
                    
                    # 获取预测和真实值
                    pred = predictions[b, i].cpu().numpy()
                    label = labels[b, i].cpu().numpy()
                    
                    # 判断masking策略
                    strategy = identify_masking_strategy(label, feature_slices)
                    
                    # 统计（按模态）
                    mod_key = f'mod_{modality_id}'
                    stats['by_modality'][mod_key]['n_peaks'] += 1
                    
                    # Position
                    if label[s_pos] != -100:
                        error = abs(pred[s_pos] - label[s_pos])
                        stats['by_modality'][mod_key]['position_errors'].append(error)
                        stats['by_feature']['position']['errors'].append(error)
                        if strategy != 'unknown':
                            stats['by_strategy'][strategy]['position_errors'].append(error)
                    
                    # Intensity
                    if label[s_int] != -100:
                        error = abs(pred[s_int] - label[s_int])
                        stats['by_modality'][mod_key]['intensity_errors'].append(error)
                        stats['by_feature']['intensity']['errors'].append(error)
                        if strategy != 'unknown':
                            stats['by_strategy'][strategy]['intensity_errors'].append(error)
                    
                    # Integration
                    if label[s_integ] != -100:
                        error = abs(pred[s_integ] - label[s_integ])
                        stats['by_modality'][mod_key]['integration_errors'].append(error)
                        stats['by_feature']['integration']['errors'].append(error)
                    
                    # Width
                    if label[s_width] != -100:
                        error = abs(pred[s_width] - label[s_width])
                        stats['by_modality'][mod_key]['width_errors'].append(error)
                        stats['by_feature']['width']['errors'].append(error)
                    
                    # Multiplicity (只对H-NMR, modality_id=0)
                    if modality_id == 0 and any(label[s_mult:e_mult] != -100):
                        pred_mult = np.argmax(pred[s_mult:e_mult])
                        true_mult = np.argmax(label[s_mult:e_mult])
                        stats['by_feature']['multiplicity']['total'] += 1
                        stats['by_feature']['multiplicity']['predictions'].append(pred_mult)
                        stats['by_feature']['multiplicity']['targets'].append(true_mult)
                        if pred_mult == true_mult:
                            stats['by_feature']['multiplicity']['correct'] += 1
                    
                    # Coupling (只对H-NMR)
                    if modality_id == 0:
                        if label[s_jmean] != -100:  # J-mean
                            error = abs(pred[s_jmean] - label[s_jmean])
                            stats['by_feature']['coupling_j_mean']['errors'].append(error)
                        if label[s_jcount] != -100:  # J-count
                            error = abs(pred[s_jcount] - label[s_jcount])
                            stats['by_feature']['coupling_j_count']['errors'].append(error)
    
    # 计算最终指标
    results = compute_final_metrics(stats)
    
    return results


def compute_final_metrics(stats):
    """计算最终指标"""
    results = {
        'overall': {},
        'by_modality': {},
        'by_feature': {},
        'by_strategy': {}
    }
    
    # 模态名称映射
    modality_names = {
        'mod_0': 'h_nmr',
        'mod_1': 'c_nmr',
        'mod_2': 'ms_pos_10ev',
        'mod_3': 'ms_pos_20ev',
        'mod_4': 'ms_pos_40ev',
        'mod_5': 'ms_neg_10ev',
        'mod_6': 'ms_neg_20ev',
        'mod_7': 'ms_neg_40ev',
    }
    
    # 总体统计
    all_position_errors = []
    all_intensity_errors = []
    total_peaks = 0
    
    # 按模态
    for mod_key, mod_stats in stats['by_modality'].items():
        if mod_stats['n_peaks'] == 0:
            continue
        
        mod_name = modality_names[mod_key]
        results['by_modality'][mod_name] = {
            'position_mae': safe_mean(mod_stats['position_errors']),
            'position_r2': compute_r2(mod_stats['position_errors']),
            'intensity_mae': safe_mean(mod_stats['intensity_errors']),
            'intensity_r2': compute_r2(mod_stats['intensity_errors']),
            'integration_mae': safe_mean(mod_stats['integration_errors']),
            'width_mae': safe_mean(mod_stats['width_errors']),
            'n_peaks': mod_stats['n_peaks']
        }
        
        all_position_errors.extend(mod_stats['position_errors'])
        all_intensity_errors.extend(mod_stats['intensity_errors'])
        total_peaks += mod_stats['n_peaks']
    
    # 总体指标
    results['overall'] = {
        'position_mae': safe_mean(all_position_errors),
        'position_r2': compute_r2(all_position_errors),
        'intensity_mae': safe_mean(all_intensity_errors),
        'intensity_r2': compute_r2(all_intensity_errors),
        'total_peaks_evaluated': total_peaks
    }
    
    # 按特征
    results['by_feature'] = {
        'position': {
            'mae': safe_mean(stats['by_feature']['position']['errors']),
            'r2': compute_r2(stats['by_feature']['position']['errors']),
            'n_samples': len(stats['by_feature']['position']['errors'])
        },
        'intensity': {
            'mae': safe_mean(stats['by_feature']['intensity']['errors']),
            'r2': compute_r2(stats['by_feature']['intensity']['errors']),
            'n_samples': len(stats['by_feature']['intensity']['errors'])
        },
        'integration': {
            'mae': safe_mean(stats['by_feature']['integration']['errors']),
            'r2': compute_r2(stats['by_feature']['integration']['errors']),
            'n_samples': len(stats['by_feature']['integration']['errors'])
        },
        'width': {
            'mae': safe_mean(stats['by_feature']['width']['errors']),
            'r2': compute_r2(stats['by_feature']['width']['errors']),
            'n_samples': len(stats['by_feature']['width']['errors'])
        },
        'multiplicity': {
            'accuracy': stats['by_feature']['multiplicity']['correct'] / 
                       max(stats['by_feature']['multiplicity']['total'], 1),
            'n_samples': stats['by_feature']['multiplicity']['total']
        },
        'coupling_j_mean': {
            'mae': safe_mean(stats['by_feature']['coupling_j_mean']['errors']),
            'r2': compute_r2(stats['by_feature']['coupling_j_mean']['errors']),
            'n_samples': len(stats['by_feature']['coupling_j_mean']['errors'])
        },
        'coupling_j_count': {
            'mae': safe_mean(stats['by_feature']['coupling_j_count']['errors']),
            'r2': compute_r2(stats['by_feature']['coupling_j_count']['errors']),
            'n_samples': len(stats['by_feature']['coupling_j_count']['errors'])
        }
    }
    
    # 按策略
    for strategy, strategy_stats in stats['by_strategy'].items():
        if len(strategy_stats['position_errors']) > 0:
            results['by_strategy'][strategy] = {
                'position_mae': safe_mean(strategy_stats['position_errors']),
                'position_r2': compute_r2(strategy_stats['position_errors']),
                'intensity_mae': safe_mean(strategy_stats['intensity_errors']),
                'n_samples': strategy_stats['n_peaks']
            }
    
    return results


# ============================================================================
# 可视化（可选）
# ============================================================================

def visualize_cases(model, dataset, output_dir, config, device, n_cases=20):
    """生成可视化案例"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # 按模态分组选择案例
    modality_map = config.peak_vector.modality_map
    modality_samples = {mod: [] for mod in modality_map.keys()}
    
    # 收集每个模态的样本
    print("  收集可视化样本...")
    for idx in range(len(dataset)):
        inputs, masks, _ = dataset[idx]
        # 找到第一个真实峰的模态
        real_peaks = torch.where(masks)[0]
        if len(real_peaks) > 0:
            first_peak = inputs[real_peaks[0]]
            modality_id = torch.argmax(first_peak[:8]).item()
            # 反向查找模态名称
            for mod_name, mod_id in modality_map.items():
                if mod_id == modality_id and len(modality_samples[mod_name]) < n_cases:
                    modality_samples[mod_name].append(idx)
                    break
        
        # 检查是否收集够了
        if all(len(samples) >= n_cases for samples in modality_samples.values()):
            break
    
    # 为每个模态生成可视化
    with torch.no_grad():
        for modality, indices in modality_samples.items():
            if len(indices) == 0:
                continue
            
            print(f"  可视化 {modality} ({len(indices)} 个案例)...")
            
            for case_idx, sample_idx in enumerate(indices):
                inputs, masks, labels = dataset[sample_idx]
                inputs = inputs.unsqueeze(0).to(device)
                masks = masks.unsqueeze(0).to(device)
                
                # 预测
                pred = model(inputs, masks)[0].cpu().numpy()
                inputs_np = inputs[0].cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                # 绘图
                fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                
                # 获取真实峰
                real_peaks_mask = masks[0].bool().cpu().numpy()
                positions_orig = inputs_np[real_peaks_mask, 8]
                intensities_orig = inputs_np[real_peaks_mask, 9]
                
                # 获取被mask的峰及预测
                masked_peaks_mask = (labels_np != -100).any(axis=-1)
                positions_masked = []
                positions_pred = []
                intensities_pred = []
                
                for i, is_masked in enumerate(masked_peaks_mask):
                    if is_masked and labels_np[i, 8] != -100:  # position被mask
                        positions_masked.append(inputs_np[i, 8])  # 原始位置
                        positions_pred.append(pred[i, 8])  # 预测位置
                        intensities_pred.append(pred[i, 9])  # 预测强度
                
                # 子图1: Position对比
                axes[0].stem(positions_orig, intensities_orig, 
                           label='Original peaks', linefmt='b-', markerfmt='bo', basefmt=' ')
                if len(positions_pred) > 0:
                    axes[0].stem(positions_pred, intensities_pred,
                               label='Reconstructed (masked peaks)', 
                               linefmt='r--', markerfmt='rx', basefmt=' ')
                axes[0].set_xlabel('Position (normalized z-score)')
                axes[0].set_ylabel('Intensity (normalized z-score)')
                axes[0].set_title(f'{modality} - Case {case_idx + 1}')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # 子图2: 误差分布
                if len(positions_pred) > 0:
                    position_errors = [abs(positions_pred[i] - positions_masked[i]) 
                                     for i in range(len(positions_pred))]
                    axes[1].bar(range(len(position_errors)), position_errors)
                    axes[1].set_xlabel('Masked Peak Index')
                    axes[1].set_ylabel('Position Reconstruction Error (MAE)')
                    axes[1].set_title('Per-Peak Reconstruction Errors')
                    axes[1].grid(True, alpha=0.3)
                else:
                    axes[1].text(0.5, 0.5, 'No masked peaks in this sample',
                               ha='center', va='center', fontsize=12)
                
                plt.tight_layout()
                output_file = output_dir / f'{modality}_case_{case_idx:02d}.png'
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
    
    print(f"  ✓ 可视化完成，保存到: {output_dir}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="评估重建质量")
    parser.add_argument("--checkpoint", default="results/checkpoints/best_model.pt",
                       help="模型检查点路径")
    parser.add_argument("--data_dir", default="data/processed",
                       help="预处理数据目录")
    parser.add_argument("--config", default="configs/pretrain_set_transformer.yaml",
                       help="配置文件路径")
    parser.add_argument("--output", default="results/evaluation/reconstruction_report.json",
                       help="输出JSON报告路径")
    parser.add_argument("--batch_size", type=int, default=512,
                       help="批大小")
    parser.add_argument("--device", default="cuda",
                       help="设备 (cuda/cpu)")
    parser.add_argument("--test", action="store_true",
                       help="测试模式（1000样本）")
    parser.add_argument("--visualize", action="store_true",
                       help="生成可视化案例")
    parser.add_argument("--n_vis_cases", type=int, default=20,
                       help="每个模态的可视化案例数")
    args = parser.parse_args()
    
    print("="*70)
    print("重建质量评估")
    print("="*70)
    
    # 加载配置
    print("\n[1/4] 加载配置...")
    with open(args.config) as f:
        config = Box(yaml.safe_load(f))
    print(f"  ✓ 配置文件: {args.config}")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"  ✓ 使用设备: {device}")
    
    # 加载模型
    print("\n[2/4] 加载模型...")
    model = PretrainSetTransformer(
        dim_input=config.model.dim_input,
        dim_output=config.model.dim_output,
        dim_hidden=config.model.dim_hidden,
        num_heads=config.model.num_heads,
        depth=config.model.depth,
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', -1) + 1
    val_loss = checkpoint.get('val_loss', 0.0)
    print(f"  ✓ 模型检查点: {args.checkpoint}")
    print(f"  ✓ Epoch: {epoch}, Val Loss: {val_loss:.4f}")
    
    # 加载数据
    print("\n[3/4] 加载数据...")
    dataset = SpectraDataset(args.data_dir, args.config, masking_fraction=0.15)
    
    if args.test:
        import random
        random.seed(42)
        indices = random.sample(range(len(dataset)), min(1000, len(dataset)))
        dataset.samples = [dataset.samples[i] for i in indices]
        print(f"  ✓ 测试模式: 使用 {len(dataset)} 个样本")
    else:
        print(f"  ✓ 完整评估: 使用 {len(dataset)} 个样本")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=0)
    
    # 评估
    print("\n[4/4] 评估重建质量...")
    results = evaluate_reconstruction(model, dataloader, config, device)
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ 结果已保存: {output_path}")
    
    # 打印摘要
    print_summary(results)
    
    # 可视化（可选）
    if args.visualize:
        print("\n生成可视化案例...")
        vis_dir = output_path.parent / 'figures' / 'reconstruction_cases'
        visualize_cases(model, dataset, vis_dir, config, device, n_cases=args.n_vis_cases)
    
    print("\n" + "="*70)
    print("✅ 评估完成！")
    print("="*70)


def print_summary(results):
    """打印评估摘要"""
    print("\n" + "="*70)
    print("重建质量评估摘要")
    print("="*70)
    
    # 总体指标
    print("\n【总体指标】")
    overall = results['overall']
    print(f"  Position MAE: {overall['position_mae']:.4f} (R² = {overall['position_r2']:.4f})")
    print(f"  Intensity MAE: {overall['intensity_mae']:.4f} (R² = {overall['intensity_r2']:.4f})")
    print(f"  Total Peaks Evaluated: {overall['total_peaks_evaluated']}")
    
    # 按模态
    print("\n【按模态分解】")
    for modality, metrics in sorted(results['by_modality'].items()):
        print(f"\n  {modality.upper()}:")
        print(f"    Position:    MAE={metrics['position_mae']:.4f}, R²={metrics['position_r2']:.4f}")
        print(f"    Intensity:   MAE={metrics['intensity_mae']:.4f}, R²={metrics['intensity_r2']:.4f}")
        print(f"    Integration: MAE={metrics['integration_mae']:.4f}")
        print(f"    Width:       MAE={metrics['width_mae']:.4f}")
        print(f"    N_peaks:     {metrics['n_peaks']}")
    
    # 按特征
    print("\n【按特征维度分解】")
    for feature, metrics in sorted(results['by_feature'].items()):
        if 'mae' in metrics:
            print(f"  {feature:15s}: MAE={metrics['mae']:.4f}, R²={metrics.get('r2', 0):.4f} (n={metrics['n_samples']})")
        elif 'accuracy' in metrics:
            print(f"  {feature:15s}: Accuracy={metrics['accuracy']:.2%} (n={metrics['n_samples']})")
    
    # 按策略
    if results['by_strategy']:
        print("\n【按Masking策略分解】")
        for strategy, metrics in sorted(results['by_strategy'].items()):
            print(f"  {strategy:15s}: Position MAE={metrics['position_mae']:.4f}, "
                  f"Intensity MAE={metrics['intensity_mae']:.4f} (n={metrics['n_samples']})")
    
    print("="*70)


if __name__ == "__main__":
    main()

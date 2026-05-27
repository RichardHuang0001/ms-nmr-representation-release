#!/usr/bin/env python3
"""
Dataset Profiler for SpectraDataset
-----------------------------------
分析:
1) 数据集样本数（processed）
2) 真实峰数量分布
3) mask 分布（峰级、维度级）
4) mask 策略比例（position/y_axis/coupling）
5) 模态级 mask 贡献（H-NMR/C-NMR/MS）
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from box import Box
import yaml
import numpy as np
import argparse
from collections import Counter, defaultdict

from src.data.dataset import SpectraDataset


def count_samples(processed_dir):
    """
    统计 processed/*.pt 中的 sample 数
    """
    processed_dir = Path(processed_dir)
    total = 0
    file_counts = {}

    for pt in processed_dir.glob("*.pt"):
        data = torch.load(pt, map_location=torch.device("cpu"), weights_only=False)
        n = len(data)
        file_counts[pt.name] = n
        total += n
    
    return total, file_counts


def analyze_masks(dataset, num_samples_to_test=5000):
    """
    深度分析 mask 的数量和分布
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    total_real_peaks = 0
    total_masked_peaks = 0
    total_masked_dims = 0

    # 额外统计：用于验证两个关键假设
    # 1) 是否真的发生了 modality(0~7) 维度的mask
    # 2) 有多少被mask的峰变成了“整行全0”（这对应模型里 mask_token 的触发条件）
    total_masked_rows = 0
    masked_rows_all_zero = 0
    real_rows_all_zero = 0
    strategy_unknown = 0
    strategy_modality = 0

    # 研究统计：样本级/峰级的分布
    sample_real_peaks = []
    sample_masked_peaks = []
    sample_masked_dims = []
    masked_dims_per_peak = []  # 每个被mask峰被mask的维度数

    # 策略：按“峰数”计数 vs 按“被mask维度数”计数
    strategy_peak_counter = Counter()
    strategy_dim_counter = Counter()

    # 模态：按峰数/维度数计数，以及模态×策略交叉表
    modality_masked_peaks = Counter()
    modality_masked_dims = Counter()
    modality_strategy_peaks = Counter()  # key=(modality, strategy)

    strategy_counter = {
        "position": 0,
        "y_axis": 0,
        "coupling": 0
    }

    modality_mask_dims = {
        "h_nmr": 0,
        "c_nmr": 0,
        "ms": 0,
        "unknown": 0
    }

    tested = 0

    print(f"开始抽样分析 mask ... (目标: {num_samples_to_test} 个样本)")

    def _modality_from_onehot(modality_vector: torch.Tensor) -> str:
        """将0~7维 one-hot 模态映射到 {h_nmr,c_nmr,ms,unknown}。

        约定：0=h_nmr, 1=c_nmr, 2~7=ms。
        注意：这里不强依赖 sum==1，允许少量数值误差或异常样本。
        """
        if modality_vector.numel() < 8:
            return "unknown"
        idx = int(torch.argmax(modality_vector[:8]).item())
        if idx == 0:
            return "h_nmr"
        if idx == 1:
            return "c_nmr"
        if 2 <= idx <= 7:
            return "ms"
        return "unknown"

    for batch in tqdm(loader):
        masked_inputs, masks, labels = batch
        masked_inputs = masked_inputs[0]     # [L,24]
        labels = labels[0]                   # [L,24]

        # real peaks (由 attention mask 指示)
        real_rows = torch.where(masks[0].bool())[0]

        real_peaks = masks[0].sum().item()
        total_real_peaks += real_peaks

        # 样本级统计
        sample_real_peaks.append(int(real_peaks))

        # 1) mask 的峰：行内有任何维度 != -100
        masked_rows = (labels != -100).any(dim=1)   # [L]
        num_masked_peaks = masked_rows.sum().item()
        total_masked_peaks += num_masked_peaks
        total_masked_rows += int(num_masked_peaks)

        sample_masked_peaks.append(int(num_masked_peaks))

        # 2) mask 的维度总数：
        num_masked_dims = (labels != -100).sum().item()
        total_masked_dims += num_masked_dims

        sample_masked_dims.append(int(num_masked_dims))

        # 3) mask 策略：通过每行的 mask 范围判断
        # position → mask dim = [8]
        # y_axis → mask dims = [9,10,11]
        # coupling → mask dims = [12~23]
        for i in torch.where(masked_rows)[0]:
            dims = torch.where(labels[i] != -100)[0].tolist()
            masked_dim_count = len(dims)
            masked_dims_per_peak.append(masked_dim_count)

            # 推断策略（基于被mask的维度集合）
            if dims == [8]:
                strategy = "position"
                strategy_counter["position"] += 1
            elif set(dims) == {9, 10, 11}:
                strategy = "y_axis"
                strategy_counter["y_axis"] += 1
            elif dims and min(dims) >= 12:
                strategy = "coupling"
                strategy_counter["coupling"] += 1
            elif dims and set(dims).issubset(set(range(0, 8))):
                # 0~7: modality one-hot
                strategy = "modality"
                strategy_modality += 1
            else:
                strategy = "unknown"
                strategy_unknown += 1

            strategy_peak_counter[strategy] += 1
            strategy_dim_counter[strategy] += masked_dim_count

            # 模态（通过输入的模态one-hot）
            modality = _modality_from_onehot(masked_inputs[i, :8])
            modality_masked_peaks[modality] += 1
            modality_masked_dims[modality] += masked_dim_count
            modality_strategy_peaks[(modality, strategy)] += 1

            # mask_token 触发条件：该行输入是否变成“整行全0”
            if torch.all(masked_inputs[i] == 0).item():
                masked_rows_all_zero += 1

        # 额外：真实峰中是否出现“整行全0”（这会让模型误把它当成mask）
        if real_rows.numel() > 0:
            real_rows_all_zero += int(torch.all(masked_inputs[real_rows] == 0, dim=1).sum().item())

        # 4) mask 对不同模态的影响
        for i in torch.where(masked_rows)[0]:
            # 模态 ID 在 dim0~dim7 中 one-hot
            modality = _modality_from_onehot(masked_inputs[i, :8])
            modality_mask_dims.setdefault(modality, 0)
            modality_mask_dims[modality] += torch.where(labels[i] != -100)[0].shape[0]

        tested += 1
        if tested >= num_samples_to_test:
            break

    results = {
        "tested": tested,
        "avg_real_peaks": total_real_peaks / tested,
        "avg_masked_peaks": total_masked_peaks / tested,
        "avg_masked_dims": total_masked_dims / tested,
        "strategy_counter": strategy_counter,
        "strategy_modality": strategy_modality,
        "strategy_unknown": strategy_unknown,
        "strategy_peak_counter": dict(strategy_peak_counter),
        "strategy_dim_counter": dict(strategy_dim_counter),
        "total_masked_rows": total_masked_rows,
        "masked_rows_all_zero": masked_rows_all_zero,
        "real_rows_all_zero": real_rows_all_zero,
        "sample_real_peaks": sample_real_peaks,
        "sample_masked_peaks": sample_masked_peaks,
        "sample_masked_dims": sample_masked_dims,
        "masked_dims_per_peak": masked_dims_per_peak,
        "modality_masked_peaks": dict(modality_masked_peaks),
        "modality_masked_dims": dict(modality_masked_dims),
        "modality_strategy_peaks": {f"{m}::{s}": int(c) for (m, s), c in modality_strategy_peaks.items()},
        "modality_mask_dims": modality_mask_dims
    }

    return results


def _describe_int_list(values, name):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        print(f"{name}: (empty)")
        return
    qs = np.percentile(arr, [0, 25, 50, 75, 90, 95, 99, 100])
    print(
        f"{name}: mean={arr.mean():.2f}, std={arr.std(ddof=0):.2f}, "
        f"min={qs[0]:.0f}, p25={qs[1]:.0f}, p50={qs[2]:.0f}, p75={qs[3]:.0f}, "
        f"p90={qs[4]:.0f}, p95={qs[5]:.0f}, p99={qs[6]:.0f}, max={qs[7]:.0f}"
    )


def _print_ratio_stats(numerators, denominators, name):
    numerators = np.asarray(numerators, dtype=np.float64)
    denominators = np.asarray(denominators, dtype=np.float64)
    valid = denominators > 0
    if not np.any(valid):
        print(f"{name}: (no valid denominators)")
        return
    ratios = numerators[valid] / denominators[valid]
    qs = np.percentile(ratios, [0, 25, 50, 75, 90, 95, 99, 100])
    print(
        f"{name}: mean={ratios.mean()*100:.2f}%, std={ratios.std(ddof=0)*100:.2f}%, "
        f"min={qs[0]*100:.2f}%, p25={qs[1]*100:.2f}%, p50={qs[2]*100:.2f}%, "
        f"p75={qs[3]*100:.2f}%, p90={qs[4]*100:.2f}%, p95={qs[5]*100:.2f}%, "
        f"p99={qs[6]*100:.2f}%, max={qs[7]*100:.2f}%"
    )


def main(config_path="configs/pretrain_set_transformer.yaml", max_samples=5000, seed=42):
    # 读取全局配置
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))

    # 固定随机性（Profiler 的可复现性）
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("\n===== 1) 统计 processed 数据集样本数量 =====")
    total_samples, file_counts = count_samples(config.data.processed_dir)

    for fname, n in file_counts.items():
        print(f"{fname:40s} : {n} samples")

    print(f"\n总样本数: {total_samples}")

    # 创建 dataset（会执行一次 mask）
    print("\n===== 2) 构建 Dataset（用于 mask 分析） =====")
    dataset = SpectraDataset(
        processed_dir=config.data.processed_dir,
        masking_fraction=config.data.masking_fraction
    )
    print(f"Dataset 总长度: {len(dataset)}")

    # 深度 mask 分析
    print("\n===== 3) mask 深度分析 =====")
    stats = analyze_masks(dataset, num_samples_to_test=max_samples)

    print("\n===== 结果汇总 =====")
    print(f"抽样样本数: {stats['tested']}")
    print(f"平均真实峰数: {stats['avg_real_peaks']:.2f}")
    print(f"平均 mask 峰数: {stats['avg_masked_peaks']:.2f}")
    print(f"平均 mask 维度数: {stats['avg_masked_dims']:.2f}")

    # 样本级分布（论文常用的描述统计）
    print("\n===== 样本级分布（描述统计）=====")
    _describe_int_list(stats.get("sample_real_peaks", []), "真实峰数/样本")
    _describe_int_list(stats.get("sample_masked_peaks", []), "被mask峰数/样本")
    _describe_int_list(stats.get("sample_masked_dims", []), "被mask维度数/样本")
    _print_ratio_stats(
        stats.get("sample_masked_peaks", []),
        stats.get("sample_real_peaks", []),
        "被mask峰比例（mask_peaks / real_peaks）"
    )
    # 用 24*真实峰 作为“可见维度总量”的近似分母（足够用于策略难度对比）
    denom_dims = [rp * 24 for rp in stats.get("sample_real_peaks", [])]
    _print_ratio_stats(
        stats.get("sample_masked_dims", []),
        denom_dims,
        "被mask维度比例（mask_dims / (real_peaks*24)）"
    )

    print("\n--- Mask 策略分布 ---")
    tot = sum(stats["strategy_counter"].values())
    for k, v in stats["strategy_counter"].items():
        pct = v / tot * 100 if tot > 0 else 0
        print(f"{k:10s} : {v:6d} ({pct:.2f}%)")

    # 研究视角：同一策略用“峰数占比”和“维度占比”会不一样（coupling 会更重）
    peak_counter = stats.get("strategy_peak_counter", {})
    dim_counter = stats.get("strategy_dim_counter", {})
    tot_peaks = sum(peak_counter.values())
    tot_dims = sum(dim_counter.values())
    if tot_peaks > 0 and tot_dims > 0:
        print("\n--- 策略占比（按被mask峰数 vs 按被mask维度数）---")
        for strategy in ["position", "y_axis", "coupling", "modality", "unknown"]:
            p = peak_counter.get(strategy, 0)
            d = dim_counter.get(strategy, 0)
            print(
                f"{strategy:10s}: peaks={p:6d} ({p/tot_peaks*100:6.2f}%) | "
                f"dims={d:6d} ({d/tot_dims*100:6.2f}%)"
            )

    # 每个被mask峰被mask了多少维（通常会集中在 1/3/12 等）
    dims_per_peak = stats.get("masked_dims_per_peak", [])
    if dims_per_peak:
        print("\n--- 每个被mask峰的mask维度数分布 ---")
        _describe_int_list(dims_per_peak, "mask_dims_per_masked_peak")
        top = Counter(dims_per_peak).most_common(10)
        top_str = ", ".join([f"{k}:{v}" for k, v in top])
        print(f"最常见的mask维度数(top10): {top_str}")

    # 新增：modality / unknown 策略（如果你期望支持 modality mask，这里应该 > 0）
    tot_rows = stats.get("total_masked_rows", 0)
    mod_cnt = stats.get("strategy_modality", 0)
    unk_cnt = stats.get("strategy_unknown", 0)
    if tot_rows > 0:
        print("\n--- 额外策略检查（modality/unknown）---")
        print(f"modality   : {mod_cnt:6d} ({mod_cnt / tot_rows * 100:.2f}%)")
        print(f"unknown    : {unk_cnt:6d} ({unk_cnt / tot_rows * 100:.2f}%)")

    # 新增：mask_token 触发率估计（模型条件：整行全0）
    all_zero_masked = stats.get("masked_rows_all_zero", 0)
    all_zero_real = stats.get("real_rows_all_zero", 0)
    if tot_rows > 0:
        print("\n--- mask_token 触发条件检查（整行全0）---")
        print(f"被mask的峰中：整行全0 = {all_zero_masked:6d} ({all_zero_masked / tot_rows * 100:.4f}%)")
    if stats["tested"] > 0:
        # 注意：这里的分母用真实峰总数，更直观
        total_real = stats["avg_real_peaks"] * stats["tested"]
        if total_real > 0:
            print(f"真实峰中：整行全0 = {all_zero_real:6d} ({all_zero_real / total_real * 100:.6f}%)")

    print("\n--- Mask 对不同模态的影响（按维度计数）---")
    total_dims = sum(stats["modality_mask_dims"].values())
    for k, v in stats["modality_mask_dims"].items():
        pct = v / total_dims * 100 if total_dims > 0 else 0
        print(f"{k:10s} : {v:6d} dims ({pct:.2f}%)")

    # 研究视角：按“被mask峰数”看不同模态承受的mask压力
    mm_peaks = stats.get("modality_masked_peaks", {})
    mm_dims = stats.get("modality_masked_dims", {})
    tot_mm_peaks = sum(mm_peaks.values())
    tot_mm_dims = sum(mm_dims.values())
    if tot_mm_peaks > 0:
        print("\n--- 被mask峰在不同模态中的分布（按峰数）---")
        for m in ["h_nmr", "c_nmr", "ms", "unknown"]:
            cnt = int(mm_peaks.get(m, 0))
            pct = cnt / tot_mm_peaks * 100
            avg_d = (mm_dims.get(m, 0) / cnt) if cnt > 0 else 0.0
            print(f"{m:10s}: peaks={cnt:6d} ({pct:6.2f}%) | avg_mask_dims/peak={avg_d:.2f}")

    # 模态×策略交叉表（峰级）——用于论文解释“耦合mask基本只发生在H-NMR”等
    ms_table = stats.get("modality_strategy_peaks", {})
    if ms_table:
        print("\n--- 模态×策略交叉表（按峰数，Top行展示）---")
        for m in ["h_nmr", "c_nmr", "ms", "unknown"]:
            row = []
            row_tot = 0
            for s in ["position", "y_axis", "coupling", "modality", "unknown"]:
                key = f"{m}::{s}"
                c = int(ms_table.get(key, 0))
                row.append((s, c))
                row_tot += c
            if row_tot == 0:
                continue
            row_str = " | ".join([f"{s}:{c}({c/row_tot*100:.1f}%)" for s, c in row])
            print(f"{m:10s} total={row_tot:6d} | {row_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask strategy profiler for SpectraDataset")
    parser.add_argument("--config", type=str, default="configs/pretrain_set_transformer.yaml", help="Path to config yaml")
    parser.add_argument("--max-samples", type=int, default=5000, help="Number of samples to profile")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible profiling")
    args = parser.parse_args()
    main(config_path=args.config, max_samples=args.max_samples, seed=args.seed)
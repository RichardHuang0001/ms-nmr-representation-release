# 文件：preprocess.py (v5.7 - 性能优化版)
# 说明:
#  - [!! 性能优化 1 !!] 将后处理（填充/截断/Tensor创建）移入并行工作进程，
#    解决主进程 100% CPU 占用、工作进程空闲 (S+) 的串行瓶颈。
#  - [!! 性能优化 2 !!] 为 pool.imap 添加 chunksize=500，
#    大幅减少进程间通信开销。
#  -
#  - 预处理逻辑与 v5.6 完全一致:
#      0-7: modality one-hot
#      8: position_norm (H: delta, C: delta(ppm), MS: m/z)
#      9: intensity_norm (H: 0; C: intensity; MS: intensity)
#     10: integration_norm (H: nH; others: 0)
#     11: width_norm (H: rangeMax-rangeMin; C: width (ppm); MS: 0)
#  12-21: multiplicity one-hot (only for H)
#  22: j_mean_norm (only for H)
#  23: j_count_norm (only for H)
#
#  - 归一化策略与 v5.6 完全一致:
#    使用 per-modality Z-Score ('mean', 'std')，若 'std' 缺失或为 0，回退到 tanh。

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from multiprocessing import Pool, cpu_count
import functools
import re
import yaml
from box import Box

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_j_values(j_str):
    if not isinstance(j_str, str) or not j_str.strip():
        return []
    try:
        return [float(f) for f in re.findall(r"[-+]?\d*\.\d+|\d+", j_str)]
    except (ValueError, TypeError):
        return []


def normalize_feature(value, mean=None, std=None):
    """
    使用 per-modality z-score 归一化。
    (逻辑与 v5.6 完全一致)
    """
    try:
        v = float(value)
    except Exception:
        return 0.0

    if mean is not None and std is not None and std > 0:
        return float((v - mean) / std)
    
    return float((np.tanh(v) + 1.0) / 2.0)


def process_single_sample(row_tuple, config, global_stats):
    """
    [!! v5.7 优化 !!]
    此函数现在在并行工作进程中完成所有工作，
    包括特征化、填充、截断和张量创建。
    它最终返回一个完整的 dict (用于 torch.save) 或 None。
    """
    index, row = row_tuple
    all_peak_vectors = []

    PEAK_VECTOR_DIM = config.peak_vector.dim
    MODALITY_MAP = config.peak_vector.modality_map.to_dict()
    MULTIPLICITY_MAP = config.peak_vector.multiplicity_map.to_dict()
    COLUMNS_TO_PROCESS = config.data.preprocessing.columns_to_process

    MULTIPLICITY_BASE_DIM = 12
    MULTIPLICITY_OTHER_INDEX = 9

    # --- Robustly load peaks ---
    h_peaks = row.get('h_nmr_peaks', [])
    c_peaks = row.get('c_nmr_peaks', [])
    h_peaks = h_peaks.tolist() if isinstance(h_peaks, np.ndarray) else (h_peaks if isinstance(h_peaks, list) else [])
    c_peaks = c_peaks.tolist() if isinstance(c_peaks, np.ndarray) else (c_peaks if isinstance(c_peaks, list) else [])

    ms_data = {
        col: (
            row.get(col).tolist()
            if isinstance(row.get(col), np.ndarray)
            else (row.get(col) if isinstance(row.get(col), list) else [])
        )
        for col in COLUMNS_TO_PROCESS
        if 'msms' in col
    }

    # ---------- H-NMR ----------
    if 'h_nmr_peaks' in COLUMNS_TO_PROCESS:
        for peak in h_peaks:
            if not isinstance(peak, dict):
                continue
            vec = [0.0] * PEAK_VECTOR_DIM

            if 'h_nmr_peaks' in MODALITY_MAP:
                vec[MODALITY_MAP['h_nmr_peaks']] = 1.0

            # 8: position_norm (delta)
            vec[8] = normalize_feature(
                peak.get('delta', 0.0),
                global_stats.get('h_nmr_delta', {}).get('mean'),
                global_stats.get('h_nmr_delta', {}).get('std')
            )
            # 9: intensity_norm -> H-NMR: 0
            vec[9] = 0.0
            # 10: integration_norm -> H-NMR: nH
            vec[10] = normalize_feature(
                peak.get('nH', 0.0),
                global_stats.get('h_nmr_nH', {}).get('mean'),
                global_stats.get('h_nmr_nH', {}).get('std')
            )
            # 11: width_norm -> rangeMax - rangeMin
            range_max = peak.get('rangeMax')
            range_min = peak.get('rangeMin')
            width = 0.0
            if (range_max is not None) and (range_min is not None):
                try:
                    width = float(range_max) - float(range_min)
                except (ValueError, TypeError):
                    width = 0.0
            vec[11] = normalize_feature(
                width,
                global_stats.get('h_nmr_width', {}).get('mean'),
                global_stats.get('h_nmr_width', {}).get('std')
            )

            # 12-21: multiplicity one-hot
            raw_mult = peak.get('category')
            mult_type = raw_mult.strip() if isinstance(raw_mult, str) else None
            mult_idx = MULTIPLICITY_MAP.get(mult_type, MULTIPLICITY_OTHER_INDEX)
            
            one_hot_pos = MULTIPLICITY_BASE_DIM + int(mult_idx)
            if MULTIPLICITY_BASE_DIM <= one_hot_pos < PEAK_VECTOR_DIM - 2:
                vec[one_hot_pos] = 1.0

            # 22: j_mean_norm, 23: j_count_norm
            j_vals = parse_j_values(peak.get('j_values'))
            j_mean = float(np.mean(j_vals)) if j_vals else 0.0
            j_count = len(j_vals)

            vec[22] = normalize_feature(
                j_mean,
                global_stats.get('j_mean', {}).get('mean'),
                global_stats.get('j_mean', {}).get('std')
            )
            vec[23] = normalize_feature(
                j_count,
                global_stats.get('j_count', {}).get('mean'),
                global_stats.get('j_count', {}).get('std')
            )
            all_peak_vectors.append(vec)

    # ---------- C-NMR ----------
    if 'c_nmr_peaks' in COLUMNS_TO_PROCESS:
        for peak in c_peaks:
            if not isinstance(peak, dict):
                continue
            vec = [0.0] * PEAK_VECTOR_DIM

            if 'c_nmr_peaks' in MODALITY_MAP:
                vec[MODALITY_MAP['c_nmr_peaks']] = 1.0

            vec[8] = normalize_feature(
                peak.get('delta (ppm)', peak.get('delta', 0.0)),
                global_stats.get('c_nmr_delta', {}).get('mean'),
                global_stats.get('c_nmr_delta', {}).get('std')
            )
            vec[9] = normalize_feature(
                peak.get('intensity', 0.0),
                global_stats.get('c_nmr_intensity', {}).get('mean'),
                global_stats.get('c_nmr_intensity', {}).get('std')
            )
            vec[10] = 0.0
            vec[11] = normalize_feature(
                peak.get('width (ppm)', 0.0),
                global_stats.get('c_nmr_width', {}).get('mean'),
                global_stats.get('c_nmr_width', {}).get('std')
            )
            all_peak_vectors.append(vec)

    # ---------- MS/MS ----------
    for col_name in COLUMNS_TO_PROCESS:
        if 'msms' in col_name:
            peaks = ms_data.get(col_name, [])
            modality_idx = MODALITY_MAP.get(col_name, None)
            
            for peak in peaks:
                mz, inten = None, None
                if isinstance(peak, (list, tuple, np.ndarray)) and len(peak) >= 2:
                    try:
                        mz = float(peak[0])
                        inten = float(peak[1])
                    except (ValueError, TypeError):
                        continue 
                
                if mz is None or inten is None:
                    continue
                
                vec = [0.0] * PEAK_VECTOR_DIM

                if modality_idx is not None:
                    vec[modality_idx] = 1.0

                vec[8] = normalize_feature(
                    mz,
                    global_stats.get('ms_mz', {}).get('mean'),
                    global_stats.get('ms_mz', {}).get('std')
                )
                vec[9] = normalize_feature(
                    inten,
                    global_stats.get('ms_intensity', {}).get('mean'),
                    global_stats.get('ms_intensity', {}).get('std')
                )
                all_peak_vectors.append(vec)

    # ---------- [!! v5.7 优化 !!] ----------
    # 后处理（填充/截断/Tensor创建）现在在并行工作进程中完成
    
    if not all_peak_vectors:
        return None # 如果样本为空，返回 None

    num_peaks = len(all_peak_vectors)
    max_peaks = config.data.preprocessing.max_peaks
    dim = config.peak_vector.dim

    if num_peaks > max_peaks:
        # 截断
        peak_vectors_final = all_peak_vectors[:max_peaks]
        num_real_peaks = max_peaks
    else:
        # 填充
        peak_vectors_final = all_peak_vectors
        padding_needed = max_peaks - num_peaks
        peak_vectors_final.extend([[0.0] * dim for _ in range(padding_needed)])
        num_real_peaks = num_peaks

    attention_mask = [1] * num_real_peaks + [0] * (max_peaks - num_real_peaks)

    # 返回最终的 dict，准备好被 torch.save
    # 注意：smiles 仅用于下游任务（如线性探测），训练代码会忽略此字段
    return {
        "input_tensor": torch.tensor(peak_vectors_final, dtype=torch.float32),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        "smiles": row.get('smiles', ''),  # 用于下游任务标签生成
    }


def process_file(file_path, config, global_stats, file_idx, total_files):
    processed_data_dir = Path(config.data.preprocessing.processed_dir)
    output_filename = processed_data_dir / f"processed_{file_path.stem}.pt"

    if output_filename.exists() and not config.data.preprocessing.force_rerun:
        logging.info(f"⏭️  [{file_idx}/{total_files}] 文件 '{output_filename.name}' 已存在，跳过处理。")
        return

    logging.info(f"⚙️  [{file_idx}/{total_files}] 正在处理文件: {file_path.name}")
    try:
        # 读取 columns_to_process + smiles (smiles 用于下游任务标签生成)
        cols_to_read = list(config.data.preprocessing.columns_to_process) + ['smiles']
        df = pd.read_parquet(file_path, columns=cols_to_read)
    except Exception as e:
        logging.error(f"❌ 读取文件 {file_path.name} 失败: {e}。跳过此文件。")
        return

    process_func = functools.partial(process_single_sample, config=config, global_stats=global_stats)

    num_workers = config.data.preprocessing.num_workers
    if num_workers == -1:
        num_workers = cpu_count()

    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                # [!! v5.7 优化 !!] 添加 chunksize=500 来大幅减少进程间通信开销
                pool.imap(process_func, df.iterrows(), chunksize=500),
                total=df.shape[0],
                desc=f"处理 {file_path.name}"
            )
        )

    # [!! v5.7 优化 !!] 
    # 'results' 现在是一个 list[dict | None]
    # 下面的循环被一个极快的列表推导式所取代
    
    # 过滤掉所有 None (空样本)
    all_samples_data = [res for res in results if res is not None]
    
    total_processed = len(all_samples_data)
    empty_skipped = len(results) - total_processed

    # v5.6 的串行瓶颈循环 (已删除):
    # total_processed, empty_skipped = 0, 0
    # for peak_vectors in results:
    #    ... (所有填充、截断、tensor创建都在这里，导致瓶颈)

    if total_processed > 0:
        torch.save(all_samples_data, output_filename)
        logging.info(
            f"✅ 处理完成 (处理 {total_processed} 个, 跳过 {empty_skipped} 个空样本), 数据已保存到: {output_filename}"
        )
    else:
        logging.warning(f"⚠️ 在文件 {file_path.name} 中未找到任何有效的、非空的样本。")


def main(config):
    logging.info(f"--- 开始数据预处理 (v5.7 - 性能优化版) ---") # <-- 版本号更新

    global_stats_path = config.data.preprocessing.global_stats_path
    try:
        with open(global_stats_path, 'r') as f:
            global_stats = Box(yaml.safe_load(f))
        logging.info(f"✅ 全局统计文件 '{global_stats_path}' 加载成功。")
    except FileNotFoundError:
        logging.error(f"❌ 致命错误: 全局统计文件 '{global_stats_path}' 未找到。")
        logging.error("请先运行 'python src/data/calculate_stats.py' (v2.1) 来生成此文件。")
        return

    # 关键键检查
    if 'ms_mz' not in global_stats or 'h_nmr_delta' not in global_stats or 'c_nmr_delta' not in global_stats:
        logging.error(f"❌ 全局统计文件 '{global_stats_path}' 已加载，但内容不完整。")
        logging.error("缺少 'ms_mz', 'h_nmr_delta' 或 'c_nmr_delta' 等关键键。")
        logging.error("请确保您运行了最新版 (v2.1) 的 calculate_stats.py 脚本。")
        return


    raw_data_dir = Path(config.data.preprocessing.raw_dir)
    processed_data_dir = Path(config.data.preprocessing.processed_dir)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(list(raw_data_dir.glob("*.parquet")))
    if not raw_files:
        logging.error(f"在 '{raw_data_dir}' 中未找到任何 .parquet 文件。")
        return

    num_workers = config.data.preprocessing.num_workers
    if num_workers == -1:
        num_workers = cpu_count()
    logging.info(f"发现 {len(raw_files)} 个原始数据文件。将使用 {num_workers} 个CPU核心进行处理 (chunksize=500)。")

    for i, file_path in enumerate(raw_files):
        process_file(file_path, config, global_stats, i + 1, len(raw_files))

    logging.info("--- 所有数据预处理完成 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模态光谱数据预处理脚本 (v5.7 - 性能优化版)")
    parser.add_argument('--config', type=str, default="configs/config.yaml", help="全局配置文件路径")

    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = Box(yaml.safe_load(f))

        main(config)
    except FileNotFoundError:
        logging.error(f"❌ 配置文件未找到: {args.config}")
    except Exception as e:
        logging.error(f"❌ 处理过程中发生错误: {e}")
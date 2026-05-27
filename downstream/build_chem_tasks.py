#!/usr/bin/env python3
"""
独立标签构建器 (Chemical Labels Builder)

提取清洗后的 SMILES，多进程生成各类化学标签（元素存在性、官能团、分桶属性），
导出为全局对应的 .parquet 文件，加速下游少样本微调。
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from multiprocessing import Pool, cpu_count
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import Dict, Any, List

# 将项目根目录加入路径以方便导包
sys.path.insert(0, str(Path(__file__).parent.parent))
from downstream.chem_labels import FunctionalGroupLabeler

# 初始化日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局的官能团收集器（每个进程会持有一份副本）
_labeler = None

def init_worker():
    """多进程工作节点的初始化：实例化 RDKit 收集器"""
    global _labeler
    # 隐藏 RDKit 烦人的警告日志
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    _labeler = FunctionalGroupLabeler()

def analyze_smiles(smiles: str) -> Dict[str, Any]:
    """
    RDKit 进程 worker 函数
    必须保持无状态且能容忍 invalid smiles
    """
    global _labeler
    
    # 基础结构预设为无效或 NaN
    result = {
        'smiles': smiles,
        'valid': False,
        'has_N': 0, 'has_O': 0, 'has_S': 0, 
        'has_F': 0, 'has_Cl': 0, 'has_Br': 0,
        'MolWt': np.nan,
        'LogP': np.nan,
        'TPSA': np.nan,
        'RingCount': np.nan,
    }
    
    # 官能团默认全零（若解析失败）
    num_fg = _labeler.num_classes if _labeler else 85
    fg_labels = [0.0] * num_fg

    if not isinstance(smiles, str) or not smiles.strip():
        result['functional_groups'] = fg_labels
        return result

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        result['functional_groups'] = fg_labels
        return result

    result['valid'] = True
    
    # 1. 元素存在性
    elements = set(atom.GetSymbol() for atom in mol.GetAtoms())
    result['has_N'] = 1 if 'N' in elements else 0
    result['has_O'] = 1 if 'O' in elements else 0
    result['has_S'] = 1 if 'S' in elements else 0
    result['has_F'] = 1 if 'F' in elements else 0
    result['has_Cl'] = 1 if 'Cl' in elements else 0
    result['has_Br'] = 1 if 'Br' in elements else 0
    
    # 2. 连续/离散分子属性计算
    try:
        result['MolWt'] = Descriptors.MolWt(mol)
        result['LogP'] = Descriptors.MolLogP(mol)
        result['TPSA'] = Descriptors.TPSA(mol)
        result['RingCount'] = Descriptors.RingCount(mol)
    except Exception:
        pass  # 某些奇特分子可能抛错，留着 np.nan 即可
        
    # 3. 官能团特征提取 (Multi-hot)
    try:
        if _labeler:
            labels_tensor = _labeler.get_labels(smiles)
            fg_labels = labels_tensor.tolist()
    except Exception:
        pass
        
    result['functional_groups'] = fg_labels
    return result

def load_all_smiles(processed_dir: Path) -> List[str]:
    """顺序读取所有的 dataset 文件，剥离出 smiles 列表"""
    pt_files = sorted(list(processed_dir.glob("*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"未在目录中发现预处理数据：{processed_dir}")
        
    logging.info(f"正在从 {len(pt_files)} 个文件中严格按顺序提取 SMILES 以保持 idx 对齐...")
    all_smiles = []
    
    for pt_file in tqdm(pt_files, desc="提取 SMILES"):
        data_chunk = torch.load(pt_file)
        # 每个 chunk 都是 list of dicts，包含 'smiles' key
        for sample in data_chunk:
            all_smiles.append(sample.get('smiles', ''))
            
    logging.info(f"提取完成，共有 {len(all_smiles)} 个样本记录。")
    return all_smiles

def main():
    parser = argparse.ArgumentParser(description="独立的下游任务化学标签构建器")
    parser.add_argument("--data_dir", default="data/processed", help="包含 .pt 预处理数据集的路径")
    parser.add_argument("--output", default="downstream/offline_labels.parquet", help="输出的 parquet 路径")
    parser.add_argument("--num_bins", type=int, default=10, help="分子连续属性分桶数量")
    parser.add_argument("--num_workers", type=int, default=-1, help="多进程并发数 (-1 为全部 CPU 核心)")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    num_workers = cpu_count() if args.num_workers == -1 else args.num_workers
    
    # 【步骤 1】: 抽取全局有序的 SMILES 列表
    smiles_list = load_all_smiles(data_dir)
    dataset_size = len(smiles_list)
    
    # 【步骤 2】: 多进程计算 RDKit 标签特征
    logging.info(f"启动 {num_workers} 个工作进程进行 RDKit 化学计算...")
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        # chunksize 的设定有助于减少进程通信开销
        results = list(tqdm(
            pool.imap(analyze_smiles, smiles_list, chunksize=1000),
            total=dataset_size,
            desc="特征生成进度"
        ))
        
    df = pd.DataFrame(results)
    valid_count = df['valid'].sum()
    invalid_count = dataset_size - valid_count
    logging.info(f"解析完成：有效分子 {valid_count} 个，解析失败/空缺 {invalid_count} 个。")
    
    # 【步骤 3】: 利用 Pandas QCut 实现多类别分位数分桶
    logging.info(f"执行连续值特征的分位数均衡分桶 (Classes={args.num_bins}) ...")
    
    features_to_bin = ['MolWt', 'LogP', 'TPSA', 'RingCount']
    for col in features_to_bin:
        bin_col_name = f"{col}_bin"
        # 默认全部赋予 -1 表示“未知”或“无效”类
        df[bin_col_name] = -1
        
        mask = df['valid'] & df[col].notna()
        
        try:
            # duplicates='drop' 是为了防止大量 0（例如大部分分子没环）导致边界重合报错
            bins = pd.qcut(df.loc[mask, col], q=args.num_bins, labels=False, duplicates='drop')
            # 转为整型存入
            df.loc[mask, bin_col_name] = bins.astype(int)
            num_actual_bins = bins.nunique()
            logging.info(f"  - 特征 {col:<10} -> 成功被划分为 {num_actual_bins:<2} 个均衡桶。")
        except Exception as e:
            logging.error(f"  - 特征 {col} 分桶异常: {e}")
            
    # 【步骤 4】: 持久化保存到 Parquet
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"正在将包含全局对齐的 {dataset_size} 行数据的 DataFrame 持久化到 {output_path}...")
    df.to_parquet(output_path, engine='pyarrow', index=False)
    
    logging.info("✅ 任务 1.1：离线标签库执行完毕，数据已准备就绪。")
    
if __name__ == "__main__":
    main()

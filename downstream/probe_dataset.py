#!/usr/bin/env python3
"""
线性探测数据集 (Linear Probe Dataset)

封装预处理后的光谱数据，添加官能团标签，用于下游分类任务。
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging
from tqdm import tqdm
import sys

from downstream.chem_labels import FunctionalGroupLabeler


class LinearProbeDataset(Dataset):
    """
    线性探测任务数据集
    
    复用预处理后的 .pt 文件，从中读取 SMILES 并生成官能团标签。
    注意：需要使用修改后的 preprocess.py 生成包含 SMILES 的数据。
    """
    
    def __init__(self, processed_dir: str, max_samples: int = None):
        """
        初始化数据集
        
        Args:
            processed_dir: 预处理后 .pt 文件所在目录
            max_samples: 最大样本数（用于快速测试）
        """
        self.processed_path = Path(processed_dir)
        self.labeler = FunctionalGroupLabeler()
        
        # 加载所有预处理样本
        pt_files = sorted(list(self.processed_path.glob("*.pt")))
        if not pt_files:
            raise FileNotFoundError(f"在目录 '{processed_dir}' 中未找到任何 .pt 文件。")
        
        logging.info(f"📦 正在加载数据 ({len(pt_files)} 个文件)...")
        self.samples = []
        
        for pt_file in tqdm(pt_files, desc="加载数据", ncols=80, 
                           disable=not sys.stdout.isatty()):
            try:
                data_chunk = torch.load(pt_file)
                self.samples.extend(data_chunk)
                
                # 提前终止以节省时间
                if max_samples and len(self.samples) >= max_samples:
                    self.samples = self.samples[:max_samples]
                    break
            except Exception as e:
                logging.warning(f"⚠️ 加载 {pt_file.name} 失败: {e}")
        
        logging.info(f"✅ 加载完成: {len(self.samples)} 个样本")
        
        # 检查 SMILES 可用性
        if self.samples and 'smiles' not in self.samples[0]:
            raise ValueError(
                "❌ 预处理数据中缺少 SMILES 字段！\n"
                "请使用修改后的 preprocess.py 重新运行预处理。\n"
                "命令: python src/data/preprocess.py --config configs/config.yaml"
            )
        
        # 预生成所有标签（节省训练时间）
        logging.info("🏷️ 正在生成官能团标签...")
        self.labels = []
        valid_count = 0
        empty_smiles_count = 0
        invalid_smiles_count = 0
        
        for sample in tqdm(self.samples, desc="生成标签", ncols=80,
                          disable=not sys.stdout.isatty()):
            smiles = sample.get('smiles', '')
            
            # 诊断统计
            if not smiles or smiles.strip() == '':
                empty_smiles_count += 1
            
            label = self.labeler.get_labels(smiles)
            self.labels.append(label)
            
            if label.sum() > 0:
                valid_count += 1
            elif smiles and smiles.strip() != '':
                # 有 SMILES 但标签全零，说明 RDKit 解析失败
                invalid_smiles_count += 1
        
        # 详细诊断输出
        logging.info(f"✅ 标签生成完成:")
        logging.info(f"   - 总样本数: {len(self.samples)}")
        logging.info(f"   - 有效标签: {valid_count} ({100*valid_count/len(self.samples):.1f}%)")
        logging.info(f"   - 空SMILES: {empty_smiles_count}")
        logging.info(f"   - 无效SMILES: {invalid_smiles_count}")
        
        # 警告检查
        if empty_smiles_count > len(self.samples) * 0.1:
            logging.warning(f"⚠️ 超过10%的样本没有SMILES！请检查预处理是否正确。")
        if invalid_smiles_count > len(self.samples) * 0.05:
            logging.warning(f"⚠️ 超过5%的SMILES无法解析！数据质量可能有问题。")
        
        self.num_classes = self.labeler.num_classes
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        返回 (input_tensor, attention_mask, label)
        
        注意: 下游任务不需要 masking，所以直接返回完整输入
        """
        sample = self.samples[idx]
        return (
            sample['input_tensor'],
            sample['attention_mask'],
            self.labels[idx]
        )


# 测试代码
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        dataset = LinearProbeDataset("data/processed", max_samples=100)
        print(f"\n数据集大小: {len(dataset)}")
        print(f"类别数: {dataset.num_classes}")
        
        x, mask, y = dataset[0]
        print(f"\n样本形状:")
        print(f"  输入: {x.shape}")
        print(f"  掩码: {mask.shape}")
        print(f"  标签: {y.shape}, 激活数: {y.sum().item():.0f}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

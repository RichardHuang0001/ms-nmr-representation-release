#!/usr/bin/env python3
"""
官能团标签生成器 (Functional Group Labeler)

使用 RDKit 的 Fragments 模块自动检测分子中的官能团，
生成 multi-hot 编码用于多标签分类任务。
"""

import torch
from rdkit import Chem
from rdkit.Chem import Fragments
import logging


class FunctionalGroupLabeler:
    """基于 RDKit Fragments 的官能团标签生成器"""
    
    def __init__(self):
        """初始化标签生成器，收集所有 RDKit 官能团检测函数"""
        # 获取所有以 fr_ 开头的函数（官能团检测函数）
        self.frag_functions = []
        self.frag_names = []
        
        for name in sorted(dir(Fragments)):
            if name.startswith('fr_'):
                func = getattr(Fragments, name)
                if callable(func):
                    self.frag_functions.append(func)
                    self.frag_names.append(name[3:])  # 去掉 'fr_' 前缀
        
        self.num_classes = len(self.frag_functions)
        logging.info(f"初始化官能团标注器: 包含 {self.num_classes} 个官能团类别")
    
    def get_labels(self, smiles: str) -> torch.Tensor:
        """
        从 SMILES 生成官能团标签向量
        
        Args:
            smiles: SMILES 字符串
            
        Returns:
            FloatTensor [num_classes]: multi-hot 编码 (0/1)
        """
        if not isinstance(smiles, str) or smiles.strip() == '':
            return torch.zeros(self.num_classes, dtype=torch.float32)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.zeros(self.num_classes, dtype=torch.float32)
        
        labels = []
        for func in self.frag_functions:
            try:
                count = func(mol)
                # 二分类：有(1) vs 无(0)
                labels.append(1.0 if count > 0 else 0.0)
            except Exception:
                labels.append(0.0)
        
        return torch.tensor(labels, dtype=torch.float32)
    
    def get_class_names(self) -> list:
        """返回所有官能团类别名称"""
        return self.frag_names.copy()


# 测试代码
if __name__ == "__main__":
    labeler = FunctionalGroupLabeler()
    
    test_cases = [
        ("CCO", "乙醇"),
        ("c1ccccc1", "苯"),
        ("CC(=O)O", "乙酸"),
        ("CC(=O)Nc1ccc(O)cc1", "对乙酰氨基酚"),
    ]
    
    print(f"\n官能团检测测试 ({labeler.num_classes} 个类别):")
    print("-" * 50)
    
    for smiles, name in test_cases:
        labels = labeler.get_labels(smiles)
        active_count = int(labels.sum().item())
        active_groups = [labeler.frag_names[i] for i, v in enumerate(labels) if v > 0]
        print(f"{name} ({smiles}): {active_count} 个官能团")
        if active_groups:
            print(f"  → {', '.join(active_groups[:5])}{'...' if len(active_groups) > 5 else ''}")

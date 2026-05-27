# 文件名：dataset.py
# <-- [核心] 定义PyTorch的Dataset和DataLoader (v2.0 - 特征级掩码) ---
# 版本: 2.0
# 核心改动:
# 1. 实现了更精细的“特征级掩码”策略，替代了之前的“整峰掩码”。
# 2. 对选中的峰，随机掩码其 位置、纵轴信息 或 耦合信息 中的一部分。
# 3. 旨在降低预训练难度，让模型学习更具体的特征间关联。

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
import random
import yaml
from box import Box
from tqdm import tqdm
import sys

class SpectraDataset(Dataset):
    """
    用于多模态光谱数据的PyTorch自定义数据集 (v2.0 - 特征级掩码)。
    """
    def __init__(self, processed_dir: str, config_path: str = "configs/pretrain_set_transformer.yaml", masking_fraction: float = None):
        """
        初始化数据集对象。

        :param processed_dir: 指向存放预处理好的 .pt 数据文件的目录。
        :param config_path: 全局配置文件的路径。
        :param masking_fraction: （可选）外部传入的掩码比例，用于兼容 train.py。
        """
        self.processed_path = Path(processed_dir)
        
        # 加载全局配置
        try:
            with open(config_path, 'r') as f:
                self.config = Box(yaml.safe_load(f))
            logging.info(f"✅ 成功加载配置文件: {config_path}")
        except FileNotFoundError:
            logging.error(f"❌ 致命错误: 配置文件 '{config_path}' 未找到。")
            raise

        # --- 优先使用外部传入的masking_fraction参数（兼容train.py），否则从配置文件读取 ---
        if masking_fraction is not None:
            self.masking_fraction = masking_fraction
        else:
            self.masking_fraction = float(self.config.data.masking_fraction)

        # --- 模态映射 ---
        modality_map_raw = self.config.peak_vector.modality_map
        self.modality_map = modality_map_raw.to_dict() if hasattr(modality_map_raw, "to_dict") else dict(modality_map_raw)

        self.peak_vector_dim = self.config.peak_vector.dim

        # 定义需要进行特征级掩码的维度索引
        self.pos_indices = [8]
        self.y_axis_indices = [9, 10, 11]
        self.coupling_indices = list(range(12, 24))  # 包含多重性和J值

        # --- 加载所有预处理好的样本 ---
        pt_files = sorted(list(self.processed_path.glob("*.pt")))
        if not pt_files:
            raise FileNotFoundError(f"在目录 '{processed_dir}' 中未找到任何预处理好的 .pt 文件。")

        logging.info(f"📦 正在从 {len(pt_files)} 个文件中加载数据...")
        self.samples = []
        for pt_file in tqdm(pt_files, desc="加载数据文件中", ncols=80, leave=False, disable=not sys.stdout.isatty(), mininterval=1.0):
            try:
                data_chunk = torch.load(pt_file)
                self.samples.extend(data_chunk)
            except Exception as e:
                logging.warning(f"⚠️ 文件 {pt_file.name} 加载失败: {e}")
        logging.info(f"✅ 数据加载完成，共计 {len(self.samples)} 个样本。")

    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据索引获取单个样本，并动态创建“特征级掩码”版本用于预训练。
        """
        # 1. 获取原始样本
        sample = self.samples[idx]
        input_tensor = sample['input_tensor'].clone() 
        attention_mask = sample['attention_mask']
        
        # 2. 准备标签张量
        labels_tensor = torch.full_like(input_tensor, -100)
        
        # 3. 确定可以被掩码的真实峰
        real_peak_indices = torch.where(attention_mask)[0]
        num_real_peaks = len(real_peak_indices)
        
        if num_real_peaks == 0:
            return input_tensor, attention_mask, labels_tensor
            
        # 4. 计算需要掩码的峰的数量
        num_to_mask = int(num_real_peaks * self.masking_fraction)
        if num_to_mask == 0 and num_real_peaks > 0:
            num_to_mask = 1
            
        # 5. 随机选择要掩码的峰
        perm = torch.randperm(num_real_peaks)
        masked_peak_indices = real_peak_indices[perm[:num_to_mask]]
        
        # 6. 执行新的“特征级掩码”操作
        for i in masked_peak_indices:
            original_vector = input_tensor[i].clone() # 保存原始向量
            
            # 随机选择一种掩码策略 (A, B, or C)
            mask_strategy = random.choice(['position', 'y_axis', 'coupling'])
            
            indices_to_mask = []
            is_h_nmr = float(original_vector[self.modality_map['h_nmr_peaks']]) > 0.5# 这里稍微修一下掩码的鲁棒性

            if mask_strategy == 'position':
                indices_to_mask = self.pos_indices
            elif mask_strategy == 'y_axis':
                indices_to_mask = self.y_axis_indices
            elif mask_strategy == 'coupling' and is_h_nmr: # 耦合信息只对H-NMR有意义
                indices_to_mask = self.coupling_indices
            else: # 如果随机到coupling但不是H-NMR，则默认mask位置
                indices_to_mask = self.pos_indices

            # a. 在标签张量中，只记录被掩码维度的原始值
            labels_tensor[i, indices_to_mask] = original_vector[indices_to_mask]
            
            # b. 在输入张量中，只将被选中的维度置零
            input_tensor[i, indices_to_mask] = 0.0

        return input_tensor, attention_mask, labels_tensor

# --- 用于演示和调试的示例代码 ---
if __name__ == "__main__":
    processed_dir = "data/processed"
    if not any(Path(processed_dir).glob("*.pt")):
        print(f"❌ 警告: 在 '{processed_dir}' 目录下未找到任何.pt文件。")
        print("请先运行 'python src/data/preprocess.py' 来生成数据。")
    else:
        try:
            dataset = SpectraDataset(processed_dir=processed_dir)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
            
            masked_inputs, masks, labels = next(iter(dataloader))
            
            print("\n--- 数据加载器输出检查 (特征级掩码 v2.0) ---")
            print(f"批次大小: {masked_inputs.shape[0]}")
            print(f"掩码后输入的形状: {masked_inputs.shape}")
            print(f"注意力掩码的形状: {masks.shape}")
            print(f"标签的形状: {labels.shape}")
            
            print("\n--- 单个样本检查 (样本0) ---")
            sample_input = masked_inputs[0]
            sample_labels = labels[0]
            
            # 找到那些在标签中存在有效值（非-100）的位置
            masked_positions_in_label = torch.where((sample_labels != -100).any(dim=1))[0]
            
            print(f"在样本0中，发现了 {len(masked_positions_in_label)} 个峰的部分特征被掩码。")
            if len(masked_positions_in_label) > 0:
                idx_to_check = masked_positions_in_label[0]
                print(f"例如，峰 {idx_to_check} 的部分特征被掩码了:")
                
                masked_dims = torch.where(sample_labels[idx_to_check] != -100)[0]
                print(f"  - 被掩码的维度索引: {masked_dims.tolist()}")
                
                # 打印输入和标签中对应维度的值，以作对比
                print(f"  - 输入向量在这些维度的值: {sample_input[idx_to_check][masked_dims].tolist()} <-- 应为0.0")
                print(f"  - 标签向量在这些维度的值: {sample_labels[idx_to_check][masked_dims].tolist()} <-- 保存了原始值")
                print(f"  - 输入向量在其他维度的值 (部分): {sample_input[idx_to_check][:8].tolist()} <-- 模态ID等应保留") # 打印前8维（模态ID）

        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"在测试过程中发生未知错误: {e}")

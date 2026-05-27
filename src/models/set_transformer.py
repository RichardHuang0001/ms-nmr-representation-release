## set_transformer.py
# -*- coding: utf-8 -*-
"""
我们的核心模型架构。
该文件上半部分直接整合了Set Transformer官方实现的核心模块，
下半部分是我们为自监督预训练任务设计的模型。
此文件不再依赖任何外部的 'set-transformer' 第三方库，实现了自包含。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================================================================
# Part 1: Official Set Transformer Core Modules
# Source: https://github.com/juho-lee/set_transformer
# 作者: Juho Lee et al.
# 我们直接将官方实现的核心组件整合到项目中，以确保稳定性和正确性。
# Set Transformer是专为处理无序集合（set）数据设计的神经网络结构。
# ==========================================================================================

class MAB(nn.Module):
    """
    多头注意力块 (Multihead Attention Block)。
    这是Set Transformer的基本构建单元，实现了Q、K、V之间的标准多头注意力机制。
    """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        """
        :param dim_Q: Query (Q) 的维度。
        :param dim_K: Key (K) 和 Value (V) 的维度。
        :param dim_V: 输出的维度，也是内部计算的维度。
        :param num_heads: 多头注意力的头数。
        :param ln: 布尔值，是否在残差连接后使用LayerNorm。
        """
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        # 线性层，用于将输入的Q, K, V投影到目标维度
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            # Layer Normalization层，用于稳定训练
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        # 输出线性层
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, mask=None): # <--- [修正1] 增加 mask 参数以支持padding
        # 1. 线性投影
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        # 2. 拆分成多头 (split into multiple heads)
        dim_split = self.dim_V // self.num_heads
        # 将最后一个维度拆分，并把多头的维度(num_heads)拼接到batch维度上
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        
        # --- [修正2] 应用注意力掩码 ---
        # 3. 计算缩放点积注意力 (Scaled Dot-Product Attention)
        # A = softmax(Q * K^T / sqrt(d_k))
        A = torch.bmm(Q_, K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        if mask is not None:
            # 为了匹配多头拼接后的形状 [B*H, L, D]，需要将mask进行扩展
            # 原始mask: [B, L] -> [B, 1, L] (为了广播)
            # -> [B*H, 1, L] (在batch维度上重复) -> [B*H, L_q, L_k] (在序列维度上广播)
            # 在我们的应用中，Q和K的序列长度相同，处理相对简化。
            mask_expanded = mask.unsqueeze(1).repeat(self.num_heads, Q.size(1), 1)
            # `masked_fill` 会在 mask_expanded 中为True（即padding的位置）填充一个极大的负数，
            # 这样在经过softmax后，这些位置的注意力权重会趋近于0。
            A = A.masked_fill(mask_expanded == 0, -1e9) 

        A = torch.softmax(A, 2) # 在K的维度上进行softmax
        # --------------------------------

        # 4. 将注意力权重应用于V，并进行残差连接
        # O = Q + Attention(Q, K, V)
        O = torch.cat((Q_ + torch.bmm(A, V_)).split(Q.size(0), 0), 2)
        
        # 5. 可选的LayerNorm和前馈网络
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O)) # 第二个残差连接
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

# --------------------------------
class SAB(nn.Module):
    """
    集合自注意力块 (Set Attention Block)。
    这是MAB的一个特例，其中 Q, K, V 都来自同一个输入X。
    它用于建模一个集合内部元素之间的交互。
    """
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None): # <--- [修正3] 增加 mask 参数
        # 调用MAB，其中Q=K=V=X
        return self.mab(X, X, mask=mask)


class ISAB(nn.Module):
    """
    诱导集注意力块 (Induced Set Attention Block)。
    通过引入m个可学习的“诱导点” I，将自注意力的复杂度从O(n^2)降低到O(n*m)。
    适用于处理大规模的集合。
    """
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        # I是可学习的诱导点参数
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_in))
        nn.init.xavier_uniform_(self.I)
        # H = MAB(I, X) : 让诱导点去关注输入集合X
        self.mab0 = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)
        # O = MAB(X, H) : 让输入集合X去关注处理后的诱导点H
        self.mab1 = MAB(dim_out, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        # I需要被复制以匹配输入X的批次大小
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    """
    多头注意力池化 (Pooling by Multihead Attention)。
    使用k个可学习的“种子”向量S作为Query，来对输入集合X进行注意力池化，
    最终将大小为n的集合X聚合为大小为k的输出集合。
    """
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        # S是可学习的种子向量
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        # S作为Query，X作为Key和Value
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


# ==========================================================================================
# Part 2: Our Pre-training Model Implementation
# 这个模型现在直接调用上面定义的官方核心模块，构建一个用于自监督学习的编码器。
# ==========================================================================================

class PretrainSetTransformer(nn.Module):
    """
    用于自监督预训练的Set Transformer模型。
    其核心任务是根据上下文重建被掩码的峰向量 (reconstruct masked peak vectors)。
    该模型现在是自包含的，不依赖任何外部set-transformer库。
    """
    def __init__(self,
                 dim_input: int,
                 dim_output: int,
                 dim_hidden: int,
                 num_heads: int,
                 num_inds: int = 32,
                 depth: int = 2,
                 ln: bool = True):
        """
        初始化预训练模型。
        :param dim_input: 输入的Peak Vector特征维度 (例如，我们设计的是12)。
        :param dim_output: 输出的Peak Vector特征维度 (通常与输入相同，也是12)。
        :param dim_hidden: Set Transformer内部的隐藏层维度。
        :param num_heads: 多头注意力机制的头数。
        :param num_inds: 诱导点的数量 (仅在ISAB中使用，当前模型未使用)。
        :param depth: Set Transformer编码器的层数（SAB模块的堆叠数量）。
        :param ln: 是否在SAB模块中使用LayerNorm。
        """
        super().__init__()
        
        # 1. 定义一个可学习的 [MASK] 向量。
        # 这个向量将用于替换输入中被置零的掩码位置。
        # 作为一个可学习的参数，模型可以自主学习到一个最优的“掩码”表示。
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim_input))

        # 2. 实例化Set Transformer核心作为编码器 (Encoder)
        # 首先是一个线性投影层，将输入维度映射到模型的隐藏维度
        self.encoder_input_proj = nn.Linear(dim_input, dim_hidden)
        
        # 编码器由多个SAB（Set Attention Block）层堆叠而成。
        # 每个SAB层都会让集合中的所有元素（峰向量）相互交互，从而更新它们的表示。
        encoder_layers = []
        for _ in range(depth):
            encoder_layers.append(
                SAB(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln)
            )
        # 使用 nn.Sequential 将多个SAB层打包成一个模块
        self.encoder = nn.Sequential(*encoder_layers)

        # 3. 定义重建头部 (Reconstruction Head)
        # 这是一个简单的前馈神经网络(MLP)，用于将编码器输出的隐藏表示
        # 映射回原始的峰向量维度，从而完成重建任务。
        self.reconstruction_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden * 2),
            nn.GELU(), # 使用GELU激活函数
            nn.Linear(dim_hidden * 2, dim_output)
        )
        
    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        模型的前向传播逻辑。
        :param input_tensor: 形状为 [B, L, D] 的输入张量。其中被掩码的峰已经用全零向量填充。
        :param attention_mask: 形状为 [B, L] 的注意力掩码，值为1表示真实峰，0表示填充。
        :return: 形状为 [B, L, D] 的预测峰向量。
        """
        # 1. 动态地将输入中的全零向量（掩码位置）替换为可学习的 self.mask_token
        # `torch.all(input_tensor == 0, dim=-1)` 会找到在特征维度上所有值都为0的峰
        is_masked = torch.all(input_tensor == 0, dim=-1)
        if attention_mask is not None:
            # 确保我们只替换那些既是全零又是真实峰（非padding）的位置
            is_masked = is_masked & attention_mask.bool()
        
        # `torch.where` 根据 is_masked 条件，选择性地从 self.mask_token 或原始 input_tensor 中取值
        masked_input = torch.where(
            is_masked.unsqueeze(-1), # 扩展维度以匹配输入张量
            self.mask_token.expand(input_tensor.shape[0], input_tensor.shape[1], -1), 
            input_tensor
        )
        
        # 2. 将输入投影到隐藏维度
        x = self.encoder_input_proj(masked_input)

        # 3. 通过编码器进行上下文信息编码
        # --- [修正4] 向编码器中的每个SAB层传递 attention_mask ---
        encoded_representation = x
        for layer in self.encoder:
            # 依次通过每个SAB层，并传入掩码以忽略padding
            encoded_representation = layer(encoded_representation, mask=attention_mask)
        
        # 4. 通过重建头部进行预测
        predictions = self.reconstruction_head(encoded_representation)
        
        return predictions
    
    def encode(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        提取编码器输出（不经过重建头），用于下游任务如线性探测。
        
        :param input_tensor: 形状为 [B, L, D] 的输入张量。
        :param attention_mask: 形状为 [B, L] 的注意力掩码，值为1表示真实峰，0表示填充。
        :return: 形状为 [B, L, hidden_dim] 的编码器输出。
        """
        # 1. 处理掩码位置（与 forward 相同）
        is_masked = torch.all(input_tensor == 0, dim=-1)
        if attention_mask is not None:
            is_masked = is_masked & attention_mask.bool()
        
        masked_input = torch.where(
            is_masked.unsqueeze(-1),
            self.mask_token.expand(input_tensor.shape[0], input_tensor.shape[1], -1),
            input_tensor
        )
        
        # 2. 投影并编码
        x = self.encoder_input_proj(masked_input)
        for layer in self.encoder:
            x = layer(x, mask=attention_mask)
        
        return x  # [B, L, hidden_dim]


# --- 用于演示和调试的示例代码 ---
if __name__ == "__main__":
    
    # --- 定义模型超参数 ---
    BATCH_SIZE = 4      # 批次大小
    MAX_PEAKS = 512     # 序列最大长度
    PEAK_DIM = 24       # 每个峰向量的维度
    HIDDEN_DIM = 256    # 模型内部的隐藏维度
    NUM_HEADS = 8       # 多头注意力的头数
    
    # --- 实例化模型 ---
    model = PretrainSetTransformer(
        dim_input=PEAK_DIM,
        dim_output=PEAK_DIM,
        dim_hidden=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        depth=3, # 使用3层编码器进行测试
        ln=True
    )
    
    print("--- 模型结构 ---")
    print(model)
    # 计算并打印模型的总参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数量: {num_params / 1e6:.2f} M")

    # --- 创建一些虚拟的输入数据，以模拟真实场景 ---
    # 创建随机的输入数据
    input_data = torch.randn(BATCH_SIZE, MAX_PEAKS, PEAK_DIM)
    
    # 模拟不同样本有不同数量的真实峰
    real_peaks_count = [100, 250, 400, 50]
    # 创建注意力掩码，并根据真实峰数量填充padding
    attention_mask = torch.zeros(BATCH_SIZE, MAX_PEAKS)
    for i, count in enumerate(real_peaks_count):
        attention_mask[i, :count] = 1
        input_data[i, count:] = 0 # 将padding部分置零

    # 模拟掩码操作：随机将15%的真实峰置为零
    for i in range(BATCH_SIZE):
        num_to_mask = int(real_peaks_count[i] * 0.15)
        if num_to_mask > 0:
            mask_indices = torch.randperm(real_peaks_count[i])[:num_to_mask]
            input_data[i, mask_indices] = 0

    print("\n--- 输入张量形状检查 ---")
    print(f"输入数据 (input_tensor) 形状: {input_data.shape}")
    print(f"注意力掩码 (attention_mask) 形状: {attention_mask.shape}")
    
    # --- 将模型设置为评估模式并进行前向传播 ---
    model.eval()
    with torch.no_grad():
        predictions = model(input_data, attention_mask)

    print("\n--- 输出张量形状检查 ---")
    print(f"模型预测 (predictions) 形状: {predictions.shape}")

    # 检查输出维度是否与输入维度匹配
    assert predictions.shape == (BATCH_SIZE, MAX_PEAKS, PEAK_DIM), "输出维度不匹配！"
    print("\n✅ 模型前向传播成功，输出维度正确！")


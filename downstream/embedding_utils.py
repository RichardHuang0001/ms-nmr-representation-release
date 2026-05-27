#!/usr/bin/env python3
"""
下游任务通用表征工具。

集中管理：
1. 预训练编码器的加载
2. 冻结 / 解冻
3. token-level 输出的 mean pooling
4. 按模态提取 pooled embedding（H/C/MS）
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import yaml
from box import Box

from src.models.set_transformer import PretrainSetTransformer


def load_model_config(config_path: str) -> Box:
    """加载 YAML 配置并返回 Box 对象。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return Box(yaml.safe_load(f))


def build_encoder_from_config(config: Box) -> PretrainSetTransformer:
    """基于项目配置构建预训练编码器。"""
    return PretrainSetTransformer(
        dim_input=config.model.dim_input,
        dim_output=config.model.dim_output,
        dim_hidden=config.model.dim_hidden,
        num_heads=config.model.num_heads,
        num_inds=config.model.num_inds,
        depth=config.model.depth,
        ln=config.model.ln,
    )


def load_pretrained_encoder(
    checkpoint_path: str,
    config_path: str,
    device: torch.device,
) -> Tuple[PretrainSetTransformer, Box]:
    """加载预训练编码器与其配置。"""
    config = load_model_config(config_path)
    model = build_encoder_from_config(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config


def set_encoder_trainable(model: torch.nn.Module, trainable: bool) -> None:
    """统一设置编码器参数是否参与训练。"""
    for param in model.parameters():
        param.requires_grad = trainable

    if trainable:
        model.train()
    else:
        model.eval()


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    对 token-level 表征做显式 mean pooling。

    Args:
        hidden_states: [B, L, H]
        attention_mask: [B, L], 1 表示有效 token

    Returns:
        [B, H]
    """
    if attention_mask.dtype != hidden_states.dtype:
        mask = attention_mask.to(hidden_states.dtype)
    else:
        mask = attention_mask

    mask = mask.unsqueeze(-1)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _modality_groups_from_config(config: Box) -> Dict[str, List[str]]:
    modality_map = config.peak_vector.modality_map
    modality_keys = set(modality_map.keys()) if hasattr(modality_map, "keys") else set(dict(modality_map).keys())

    groups: Dict[str, List[str]] = {
        "h_nmr": ["h_nmr_peaks"],
        "c_nmr": ["c_nmr_peaks"],
    }

    ms_candidates = [
        "msms_positive_10ev",
        "msms_positive_20ev",
        "msms_positive_40ev",
        "msms_negative_10ev",
        "msms_negative_20ev",
        "msms_negative_40ev",
    ]
    groups["ms"] = [name for name in ms_candidates if name in modality_keys]
    return groups


def identify_modality_mask(inputs: torch.Tensor, config: Box, modality_name: str) -> torch.Tensor:
    """
    按模态名称识别 token 级掩码。

    Args:
        inputs: [B, L, D]
        config: 项目配置对象
        modality_name: 'h_nmr'/'c_nmr'/'ms' 或原始字段名（如 h_nmr_peaks）

    Returns:
        modal_mask: [B, L] bool
    """
    modality_map_raw = config.peak_vector.modality_map
    modality_map = modality_map_raw.to_dict() if hasattr(modality_map_raw, "to_dict") else dict(modality_map_raw)
    groups = _modality_groups_from_config(config)

    if modality_name in groups:
        keys = groups[modality_name]
    elif modality_name in modality_map:
        keys = [modality_name]
    else:
        raise ValueError(
            f"未知模态: {modality_name}。可选: {sorted(list(groups.keys()) + list(modality_map.keys()))}"
        )

    modal_mask = torch.zeros(inputs.shape[0], inputs.shape[1], dtype=torch.bool, device=inputs.device)
    for key in keys:
        idx = int(modality_map[key])
        modal_mask = modal_mask | (inputs[:, :, idx] > 0.5)
    return modal_mask


def modal_mean_pooling(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    modal_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    只对目标模态 token 做 mean pooling。

    Args:
        hidden_states: [B, L, H]
        attention_mask: [B, L]
        modal_mask: [B, L]

    Returns:
        pooled: [B, H]
        valid_samples: [B]，表示该样本是否存在目标模态峰
    """
    valid_peaks = attention_mask.bool() & modal_mask.bool()
    valid_samples = valid_peaks.any(dim=1)

    mask = valid_peaks.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = summed / counts

    pooled = pooled * valid_samples.unsqueeze(-1).to(hidden_states.dtype)
    return pooled, valid_samples


def extract_modal_embedding(
    encoder: torch.nn.Module,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    modality_name: str,
    config: Box,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    端到端提取单模态 pooled embedding。

    Returns:
        embeddings: [B, H]（L2 normalize 后）
        valid_samples: [B]
    """
    hidden_states = encoder.encode(input_tensor, attention_mask)
    modal_mask = identify_modality_mask(input_tensor, config, modality_name)
    pooled, valid_samples = modal_mean_pooling(hidden_states, attention_mask, modal_mask)

    embeddings = F.normalize(pooled, p=2, dim=1, eps=1e-12)
    embeddings = embeddings * valid_samples.unsqueeze(-1).to(embeddings.dtype)
    return embeddings, valid_samples


def extract_pooled_embeddings(
    encoder: torch.nn.Module,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """对一批样本提取 pooled embedding。"""
    hidden_states = encoder.encode(input_tensor, attention_mask)
    return mean_pooling(hidden_states, attention_mask)

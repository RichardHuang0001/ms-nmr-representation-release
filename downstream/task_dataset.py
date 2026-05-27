#!/usr/bin/env python3
"""
统一的下游任务数据集。

功能：
1. 读取预处理后的 .pt shards
2. 读取离线标签 parquet
3. 基于固定随机种子生成 train / val / test split
4. 支持 few-shot 的 subset_ratio
5. 用轻量 shard cache 避免一次性把 12GB processed 全读入内存
"""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Dataset


TASK_SPECS: Dict[str, Dict[str, object]] = {
    "functional_group": {
        "task_type": "multilabel",
        "label_column": "functional_groups",
    },
    "element_presence": {
        "task_type": "multilabel",
        "label_columns": ["has_N", "has_O", "has_S", "has_F", "has_Cl", "has_Br"],
    },
    "molwt_bin": {
        "task_type": "multiclass",
        "label_column": "MolWt_bin",
    },
    "logp_bin": {
        "task_type": "multiclass",
        "label_column": "LogP_bin",
    },
    "tpsa_bin": {
        "task_type": "multiclass",
        "label_column": "TPSA_bin",
    },
    "ringcount_bin": {
        "task_type": "multiclass",
        "label_column": "RingCount_bin",
    },
}


@dataclass
class SharedTaskData:
    """train / val / test 共享的标签与 shard 元数据。"""

    pt_files: List[Path]
    shard_lengths: List[int]
    labels: np.ndarray
    smiles: np.ndarray

    @property
    def total_samples(self) -> int:
        return int(sum(self.shard_lengths))

    @classmethod
    def from_disk(cls, processed_dir: str, label_file: str, task: str) -> "SharedTaskData":
        processed_path = Path(processed_dir)
        label_path = Path(label_file)

        pt_files = sorted(processed_path.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"在 {processed_path} 中未找到 .pt 文件")
        if not label_path.exists():
            raise FileNotFoundError(f"标签文件不存在: {label_path}")

        label_df = pd.read_parquet(label_path)
        labels = TaskDataset.load_labels_from_df(label_df, task)
        smiles = label_df["smiles"].fillna("").astype(str).to_numpy() if "smiles" in label_df.columns else np.array([""] * len(label_df))
        shard_lengths = []
        for pt_file in pt_files:
            chunk = torch.load(pt_file, map_location="cpu", weights_only=False)
            shard_lengths.append(len(chunk))

        if len(labels) != sum(shard_lengths):
            raise ValueError(
                f"标签文件与 processed 样本数不一致: labels={len(labels)}, processed={sum(shard_lengths)}"
            )

        return cls(pt_files=pt_files, shard_lengths=shard_lengths, labels=labels, smiles=smiles)


class TaskDataset(Dataset):
    """支持多任务 / few-shot 的统一 dataset。"""

    def __init__(
        self,
        processed_dir: str,
        label_file: str,
        task: str,
        split: str,
        subset_ratio: float = 1.0,
        seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        split_strategy: str = "random",
        max_samples: int | None = None,
        shard_cache_size: int = 2,
        shared_data: SharedTaskData | None = None,
    ) -> None:
        if task not in TASK_SPECS:
            raise ValueError(f"不支持的任务: {task}. 可选任务: {sorted(TASK_SPECS)}")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"不支持的 split: {split}")
        if split_strategy not in {"random", "scaffold"}:
            raise ValueError(f"不支持的 split_strategy: {split_strategy}")
        if not (0 < subset_ratio <= 1.0):
            raise ValueError("subset_ratio 必须在 (0, 1] 范围内")
        if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
            raise ValueError("train_ratio 与 val_ratio 非法")

        self.task = task
        self.split = split
        self.subset_ratio = subset_ratio
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.split_strategy = split_strategy
        self.task_spec = TASK_SPECS[task]
        self.task_type = str(self.task_spec["task_type"])
        self.shard_cache_size = max(1, shard_cache_size)
        self._shard_cache: OrderedDict[int, List[dict]] = OrderedDict()

        self.processed_dir = Path(processed_dir)
        self.label_file = Path(label_file)
        if shared_data is None:
            shared_data = SharedTaskData.from_disk(processed_dir, label_file, task)

        self.pt_files = shared_data.pt_files
        self.shard_lengths = shared_data.shard_lengths
        self.labels = shared_data.labels
        self.smiles = shared_data.smiles
        self.num_classes = self._infer_num_classes(self.labels, self.task_type)
        self.shard_offsets = np.cumsum(self.shard_lengths)
        self.total_samples = shared_data.total_samples

        self.indices = self._build_split_indices()
        if max_samples is not None:
            self.indices = self.indices[:max_samples]

    @staticmethod
    def load_labels_from_df(df: pd.DataFrame, task: str) -> np.ndarray:
        spec = TASK_SPECS[task]

        if spec["task_type"] == "multilabel":
            if task == "functional_group":
                labels = np.stack(df["functional_groups"].to_numpy()).astype(np.float32)
            else:
                labels = df[list(spec["label_columns"])].to_numpy(dtype=np.float32)
        else:
            labels = df[str(spec["label_column"])].to_numpy(dtype=np.int64)
            if np.any(labels < 0):
                raise ValueError(f"任务 {task} 含有非法类别标签 -1，当前第一版 probe 不支持未知类。")

        return labels

    @staticmethod
    def load_labels_static(label_file: Path, task: str) -> np.ndarray:
        return TaskDataset.load_labels_from_df(pd.read_parquet(label_file), task)

    def _infer_num_classes(self, labels: np.ndarray, task_type: str) -> int:
        if task_type == "multilabel":
            return int(labels.shape[1])
        return int(labels.max()) + 1

    @staticmethod
    def _scaffold_from_smiles(smiles: str) -> str:
        if not smiles:
            return "__empty__"
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return f"__invalid__::{smiles}"
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold or f"__noscaffold__::{smiles}"

    def _split_cache_path(self) -> Path:
        label_stem = self.label_file.stem
        cache_name = (
            f"{label_stem}.{self.split_strategy}.seed{self.seed}."
            f"train{self.train_ratio:.2f}.val{self.val_ratio:.2f}.splits.npz"
        )
        return self.label_file.parent / cache_name

    def _build_grouped_split_indices(self) -> Dict[str, np.ndarray]:
        cache_path = self._split_cache_path()
        if cache_path.exists():
            cached = np.load(cache_path)
            return {
                "train": cached["train"],
                "val": cached["val"],
                "test": cached["test"],
            }

        scaffolds = np.array([self._scaffold_from_smiles(smiles) for smiles in self.smiles], dtype=object)
        group_to_indices: Dict[str, List[int]] = {}
        for idx, scaffold in enumerate(scaffolds):
            group_to_indices.setdefault(str(scaffold), []).append(idx)

        groups = [(group, np.array(indices, dtype=np.int64)) for group, indices in group_to_indices.items()]
        groups.sort(key=lambda item: len(item[1]), reverse=True)

        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(groups))
        groups = [groups[i] for i in order]

        target_sizes = {
            "train": int(self.total_samples * self.train_ratio),
            "val": int(self.total_samples * self.val_ratio),
        }
        target_sizes["test"] = self.total_samples - target_sizes["train"] - target_sizes["val"]

        split_lists: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
        current_sizes = {"train": 0, "val": 0, "test": 0}

        for _, indices in groups:
            deficits = {
                split_name: target_sizes[split_name] - current_sizes[split_name]
                for split_name in ("train", "val", "test")
            }
            preferred = max(deficits, key=lambda split_name: (deficits[split_name], -current_sizes[split_name]))
            split_lists[preferred].append(indices)
            current_sizes[preferred] += len(indices)

        split_map = {
            split_name: np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
            for split_name, parts in split_lists.items()
        }
        np.savez_compressed(cache_path, **split_map)
        return split_map

    def _build_split_indices(self) -> np.ndarray:
        if self.split_strategy == "random":
            rng = np.random.default_rng(self.seed)
            all_indices = np.arange(self.total_samples)
            rng.shuffle(all_indices)

            train_end = int(self.total_samples * self.train_ratio)
            val_end = train_end + int(self.total_samples * self.val_ratio)

            split_map = {
                "train": all_indices[:train_end],
                "val": all_indices[train_end:val_end],
                "test": all_indices[val_end:],
            }
        else:
            split_map = self._build_grouped_split_indices()

        indices = split_map[self.split]

        if self.split == "train" and self.subset_ratio < 1.0:
            subset_size = max(1, int(len(indices) * self.subset_ratio))
            indices = indices[:subset_size]

        return np.sort(indices)

    def _load_shard(self, shard_idx: int) -> List[dict]:
        if shard_idx in self._shard_cache:
            self._shard_cache.move_to_end(shard_idx)
            return self._shard_cache[shard_idx]

        chunk = torch.load(self.pt_files[shard_idx], map_location="cpu", weights_only=False)
        self._shard_cache[shard_idx] = chunk
        self._shard_cache.move_to_end(shard_idx)

        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)

        return chunk

    def _global_to_local(self, global_idx: int) -> Tuple[int, int]:
        shard_idx = bisect_right(self.shard_offsets, global_idx)
        shard_start = 0 if shard_idx == 0 else int(self.shard_offsets[shard_idx - 1])
        local_idx = global_idx - shard_start
        return shard_idx, local_idx

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        global_idx = int(self.indices[idx])
        shard_idx, local_idx = self._global_to_local(global_idx)
        sample = self._load_shard(shard_idx)[local_idx]

        label = self.labels[global_idx]
        if self.task_type == "multilabel":
            label_tensor = torch.tensor(label, dtype=torch.float32)
        else:
            label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "input_tensor": sample["input_tensor"].float(),
            "attention_mask": sample["attention_mask"].float(),
            "label": label_tensor,
        }

#!/usr/bin/env python3
"""
H/C alignment 共用工具。
"""

from __future__ import annotations

import csv
import random
import time
from datetime import datetime
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from downstream.embedding_utils import extract_modal_embedding

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HCIndexedDataset(Dataset):
    """按全局 index 访问 processed shard 的轻量数据集。"""

    def __init__(
        self,
        processed_dir: str,
        indices: np.ndarray,
        shard_lengths: List[int] | None = None,
        shard_cache_size: int = 2,
    ) -> None:
        self.processed_path = Path(processed_dir)
        self.pt_files = sorted(self.processed_path.glob("*.pt"))
        if not self.pt_files:
            raise FileNotFoundError(f"在 {processed_dir} 中未找到 .pt 文件")

        self.indices = np.array(indices, dtype=np.int64)
        self.shard_cache_size = max(1, shard_cache_size)
        self._shard_cache: OrderedDict[int, List[dict]] = OrderedDict()
        self._last_shard_log = 0.0
        self._shard_load_count = 0

        if shard_lengths is None:
            self.shard_lengths = []
            for pt in self.pt_files:
                chunk = torch.load(pt, map_location="cpu", weights_only=False)
                self.shard_lengths.append(len(chunk))
        else:
            self.shard_lengths = [int(x) for x in shard_lengths]
        self.shard_offsets = np.cumsum(self.shard_lengths)

    def __len__(self) -> int:
        return len(self.indices)

    def _load_shard(self, shard_idx: int) -> List[dict]:
        if shard_idx in self._shard_cache:
            self._shard_cache.move_to_end(shard_idx)
            return self._shard_cache[shard_idx]

        now = time.monotonic()
        self._shard_load_count += 1
        if now - self._last_shard_log >= 30.0:
            log(f"dataset shard load: file={self.pt_files[shard_idx].name}, shard={shard_idx + 1}/{len(self.pt_files)}, loads={self._shard_load_count}, cache_entries={len(self._shard_cache)}/{self.shard_cache_size}")
            self._last_shard_log = now

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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        global_idx = int(self.indices[idx])
        shard_idx, local_idx = self._global_to_local(global_idx)
        sample = self._load_shard(shard_idx)[local_idx]

        return {
            "input_tensor": sample["input_tensor"].float(),
            "attention_mask": sample["attention_mask"].float(),
        }


def _compute_hc_indices(
    processed_dir: str,
    config,
    disable_tqdm: bool = False,
    max_collect: int | None = None,
) -> np.ndarray:
    processed_path = Path(processed_dir)
    pt_files = sorted(processed_path.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"在 {processed_dir} 中未找到 .pt 文件")

    hc_indices: List[int] = []
    global_idx = 0
    modality_map_raw = config.peak_vector.modality_map
    modality_map = modality_map_raw.to_dict() if hasattr(modality_map_raw, "to_dict") else dict(modality_map_raw)
    h_idx = int(modality_map["h_nmr_peaks"])
    c_idx = int(modality_map["c_nmr_peaks"])
    stack_batch_size = 512

    for shard_i, pt_file in enumerate(pt_files, start=1):
        chunk = torch.load(pt_file, map_location="cpu", weights_only=False)
        n_chunk = len(chunk)
        start_local = 0
        while start_local < n_chunk:
            end_local = min(start_local + stack_batch_size, n_chunk)
            batch_samples = chunk[start_local:end_local]

            inputs = torch.stack([s["input_tensor"] for s in batch_samples], dim=0)
            masks = torch.stack([s["attention_mask"] for s in batch_samples], dim=0).bool()

            has_h = ((inputs[:, :, h_idx] > 0.5) & masks).any(dim=1)
            has_c = ((inputs[:, :, c_idx] > 0.5) & masks).any(dim=1)
            valid_local = (has_h & has_c).nonzero(as_tuple=False).flatten().tolist()

            base_global = global_idx + start_local
            hc_indices.extend([base_global + i for i in valid_local])

            if max_collect is not None and len(hc_indices) >= max_collect:
                hc_indices = hc_indices[:max_collect]
                break
            start_local = end_local

        global_idx += n_chunk
        if max_collect is not None and len(hc_indices) >= max_collect:
            break
        if not disable_tqdm and shard_i % 20 == 0:
            print(f"[HC index] scanned shards={shard_i}/{len(pt_files)}, collected_pairs={len(hc_indices)}")

    if not hc_indices:
        raise RuntimeError("未找到同时包含 H-NMR 与 C-NMR 的样本")

    return np.array(hc_indices, dtype=np.int64)


def _scaffold_from_smiles(smiles: str) -> str:
    """Compute Murcko scaffold SMILES string."""
    if not smiles:
        return "__empty__"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"__invalid__::{smiles}"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold or f"__noscaffold__::{smiles}"


def _load_smiles_for_indices(
    processed_dir: str,
    indices: np.ndarray,
) -> np.ndarray:
    """Load SMILES strings for given global indices from parquet labels file."""
    import pandas as pd
    label_file = Path("results/downstream/offline_labels_full.parquet")
    if not label_file.exists():
        raise FileNotFoundError(f"label file not found: {label_file}")
    df = pd.read_parquet(str(label_file), columns=["smiles"])
    all_smiles = df["smiles"].fillna("").astype(str).to_numpy()
    return all_smiles[indices]


def _build_scaffold_grouped_splits(
    smiles: np.ndarray,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Greedy group-based scaffold split. Returns positions within the input array."""
    scaffold_to_indices: Dict[str, List[int]] = {}
    for i, s in enumerate(smiles):
        scaffold = _scaffold_from_smiles(s)
        scaffold_to_indices.setdefault(scaffold, []).append(i)

    groups = list(scaffold_to_indices.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n_total = len(smiles)
    target_train = int(n_total * train_ratio)
    target_val = int(n_total * val_ratio)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for group in groups:
        if len(train_idx) < target_train:
            train_idx.extend(group)
        elif len(val_idx) < target_val:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    return {
        "train": np.sort(np.array(train_idx, dtype=np.int64)),
        "val": np.sort(np.array(val_idx, dtype=np.int64)),
        "test": np.sort(np.array(test_idx, dtype=np.int64)),
    }


def build_hc_splits(
    processed_dir: str,
    config,
    subset: float,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    disable_tqdm: bool = False,
    split_strategy: str = "random",
) -> Dict[str, np.ndarray]:
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("train_ratio/val_ratio 非法")

    cache_dir = Path("results/downstream")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"hc_pair_indices_{Path(processed_dir).name}.npy"
    cache_path = cache_dir / cache_name
    shard_len_cache = cache_dir / f"hc_shard_lengths_{Path(processed_dir).name}.npy"

    if cache_path.exists():
        indices = np.load(cache_path)
        if not disable_tqdm:
            print(f"[HC index] loaded cache: {cache_path}, n_pairs={len(indices)}")
    else:
        indices = _compute_hc_indices(
            processed_dir,
            config,
            disable_tqdm=disable_tqdm,
            max_collect=None,
        )
        np.save(cache_path, indices)
        if not disable_tqdm:
            print(f"[HC index] saved cache: {cache_path}, n_pairs={len(indices)}")
    if not shard_len_cache.exists():
        pt_files = sorted(Path(processed_dir).glob("*.pt"))
        shard_lengths = []
        for pt in pt_files:
            shard_lengths.append(len(torch.load(pt, map_location="cpu", weights_only=False)))
        np.save(shard_len_cache, np.array(shard_lengths, dtype=np.int64))
    if not disable_tqdm:
        print(f"[HC index] shard length cache: {shard_len_cache}")

    rng = np.random.default_rng(seed)

    if split_strategy == "scaffold":
        if not HAS_RDKIT:
            raise ImportError("rdkit is required for scaffold split")
        if not disable_tqdm:
            print(f"[HC scaffold] loading SMILES for {len(indices)} samples...")
        all_smiles = _load_smiles_for_indices(processed_dir, indices)
        split_map = _build_scaffold_grouped_splits(
            all_smiles, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio
        )
        for k in split_map:
            split_map[k] = np.sort(indices[split_map[k]])
        # subset only shrinks training set; val/test stay at full size
        if subset > 0:
            if subset <= 1.0:
                keep = max(1, int(len(split_map["train"]) * subset))
            else:
                keep = min(len(split_map["train"]), int(subset))
            split_map["train"] = split_map["train"][:keep]
        if not disable_tqdm:
            print(f"[HC scaffold] split: train={len(split_map['train'])}, val={len(split_map['val'])}, test={len(split_map['test'])}")
        return split_map

    # Random split - subset only shrinks training set; val/test always use full set
    rng.shuffle(indices)

    n_total = len(indices)
    train_end = int(n_total * train_ratio)
    val_end = train_end + int(n_total * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    if subset > 0:
        if subset <= 1.0:
            keep = max(1, int(len(train_idx) * subset))
        else:
            keep = min(len(train_idx), int(subset))
        train_idx = train_idx[:keep]

    split_map = {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }
    return split_map


def build_hc_dataloaders(
    processed_dir: str,
    split_map: Dict[str, np.ndarray],
    batch_size: int,
    num_workers: int,
    shard_cache_size: int = 2,
    timeout: int = 0,
) -> Tuple[HCIndexedDataset, HCIndexedDataset, HCIndexedDataset, DataLoader, DataLoader, DataLoader]:
    shard_len_cache = Path("results/downstream") / f"hc_shard_lengths_{Path(processed_dir).name}.npy"
    cached_lengths = np.load(shard_len_cache).tolist() if shard_len_cache.exists() else None

    train_ds = HCIndexedDataset(
        processed_dir,
        split_map["train"],
        shard_lengths=cached_lengths,
        shard_cache_size=shard_cache_size,
    )
    val_ds = HCIndexedDataset(
        processed_dir,
        split_map["val"],
        shard_lengths=cached_lengths,
        shard_cache_size=shard_cache_size,
    )
    test_ds = HCIndexedDataset(
        processed_dir,
        split_map["test"],
        shard_lengths=cached_lengths,
        shard_cache_size=shard_cache_size,
    )

    pin_memory = torch.cuda.is_available()
    loader_kwargs = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        loader_kwargs["timeout"] = int(timeout)
    log(f"dataloader config: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}, timeout={loader_kwargs.get('timeout', 0)}, shard_cache_size={shard_cache_size}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


class ProjectionHead(nn.Module):
    """2-layer MLP: 256 -> ReLU -> 128"""

    def __init__(self, in_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HCProjectionHeads(nn.Module):
    def __init__(self, in_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.h_head = ProjectionHead(in_dim=in_dim, out_dim=out_dim)
        self.c_head = ProjectionHead(in_dim=in_dim, out_dim=out_dim)

    def project_h(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.h_head(x), p=2, dim=1, eps=1e-12)

    def project_c(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.c_head(x), p=2, dim=1, eps=1e-12)


def symmetric_infonce_loss(h_emb: torch.Tensor, c_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = (h_emb @ c_emb.t()) / temperature
    targets = torch.arange(h_emb.shape[0], device=h_emb.device)
    loss_hc = nn.CrossEntropyLoss()(logits, targets)
    loss_ch = nn.CrossEntropyLoss()(logits.t(), targets)
    return 0.5 * (loss_hc + loss_ch)


def extract_hc_batch_embeddings(
    encoder: nn.Module,
    input_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    config,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h_emb, h_valid = extract_modal_embedding(encoder, input_tensor, attention_mask, "h_nmr", config)
    c_emb, c_valid = extract_modal_embedding(encoder, input_tensor, attention_mask, "c_nmr", config)
    valid = h_valid & c_valid
    return h_emb, c_emb, valid


def _retrieval_metrics_chunked(
    query: torch.Tensor,
    gallery: torch.Tensor,
    sim_batch: int,
    topk: int,
    use_fp16: bool,
    device: torch.device,
) -> Dict[str, float]:
    n = query.shape[0]
    if n == 0:
        return {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "MRR": 0.0, "n_pairs": 0}

    K = min(topk, n)
    gallery_gpu = gallery.to(device, non_blocking=True)
    if use_fp16 and device.type == "cuda":
        gallery_gpu = gallery_gpu.half()

    true_pos = torch.arange(n, device=device)
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    rr_sum = 0.0

    for start in range(0, n, sim_batch):
        end = min(start + sim_batch, n)
        qb = query[start:end].to(device, non_blocking=True)
        if use_fp16 and device.type == "cuda":
            qb = qb.half()

        scores = qb @ gallery_gpu.t()
        _, inds = torch.topk(scores, k=K, dim=1, largest=True, sorted=True)

        true = true_pos[start:end].unsqueeze(1)
        hit = inds == true

        top1_correct += hit[:, :1].any(dim=1).sum().item()
        top5_correct += hit[:, : min(5, K)].any(dim=1).sum().item()
        top10_correct += hit[:, : min(10, K)].any(dim=1).sum().item()

        has_hit = hit.any(dim=1)
        first_pos = torch.argmax(hit.float(), dim=1) + 1
        rr = torch.where(has_hit, 1.0 / first_pos.float(), torch.zeros_like(first_pos, dtype=torch.float))
        rr_sum += rr.sum().item()

    return {
        "R@1": float(top1_correct / n),
        "R@5": float(top5_correct / n),
        "R@10": float(top10_correct / n),
        "MRR": float(rr_sum / n),
        "n_pairs": int(n),
    }


def evaluate_hc_retrieval(
    encoder: nn.Module,
    heads: HCProjectionHeads | None,
    dataloader: DataLoader,
    config,
    device: torch.device,
    sim_batch: int = 2048,
    topk: int = 10,
    use_fp16: bool = True,
) -> Dict[str, Dict[str, float]]:
    encoder.eval()
    if heads is not None:
        heads.eval()

    all_h: List[torch.Tensor] = []
    all_c: List[torch.Tensor] = []

    total_batches = len(dataloader)
    log(f"retrieval embedding start: batches={total_batches}, batch_size={getattr(dataloader, 'batch_size', 'unknown')}")
    last_log = time.monotonic()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, start=1):
            input_tensor = batch["input_tensor"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            h_emb, c_emb, valid = extract_hc_batch_embeddings(encoder, input_tensor, attention_mask, config)
            if valid.any():
                h_emb = h_emb[valid]
                c_emb = c_emb[valid]

                if heads is not None:
                    h_emb = heads.project_h(h_emb)
                    c_emb = heads.project_c(c_emb)

                all_h.append(h_emb.detach().cpu())
                all_c.append(c_emb.detach().cpu())

            now = time.monotonic()
            if now - last_log >= 30.0 or batch_idx == total_batches:
                n_pairs = sum(x.shape[0] for x in all_h)
                log(f"retrieval embedding batch {batch_idx}/{total_batches}: pairs={n_pairs}")
                last_log = now

    if not all_h:
        empty = {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "MRR": 0.0, "n_pairs": 0}
        return {"h_to_c": empty, "c_to_h": empty}

    h_all = torch.cat(all_h, dim=0)
    c_all = torch.cat(all_c, dim=0)

    log(f"retrieval metrics start: n_pairs={h_all.shape[0]}, sim_batch={sim_batch}, topk={topk}")
    metrics_hc = _retrieval_metrics_chunked(h_all, c_all, sim_batch=sim_batch, topk=topk, use_fp16=use_fp16, device=device)
    metrics_ch = _retrieval_metrics_chunked(c_all, h_all, sim_batch=sim_batch, topk=topk, use_fp16=use_fp16, device=device)
    log("retrieval metrics done")

    return {"h_to_c": metrics_hc, "c_to_h": metrics_ch}


def append_summary_row(summary_path: Path, row: Dict[str, object], fieldnames: List[str]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})

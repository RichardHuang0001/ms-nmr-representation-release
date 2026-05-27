#!/usr/bin/env python3
"""Strict H/C non-contrastive baselines.

Evaluates H-only query embeddings against C-only gallery embeddings without any
InfoNCE training. This is intended to test whether random projections or an
untrained MLP head alone can explain strict H/C retrieval improvements.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from downstream.embedding_utils import (
    identify_modality_mask,
    load_model_config,
    load_pretrained_encoder,
    modal_mean_pooling,
)
from downstream.hc_alignment_utils import (
    HCIndexedDataset,
    HCProjectionHeads,
    _retrieval_metrics_chunked,
    build_hc_splits,
    seed_everything,
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate strict non-contrastive H/C baselines")
    p.add_argument("--checkpoint", default="results/checkpoints/best_model.pt")
    p.add_argument("--config", default="configs/pretrain_set_transformer.yaml")
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--output_dir", default="results/downstream/hc_alignment_strict_noncontrastive")
    p.add_argument("--settings", nargs="+", default=["pretrain_only_strict", "random_projection_strict", "untrained_mlp_head_strict"])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--shard_cache_size", type=int, default=245)
    p.add_argument("--subset", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--projection_dim", type=int, default=128)
    p.add_argument("--sim_batch", type=int, default=2048)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--use_fp16", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def append_row(path: Path, row: Dict[str, object], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class SharedRandomProjection(nn.Module):
    """Same fixed random linear projection for both modalities."""

    def __init__(self, in_dim: int, out_dim: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        weight = torch.randn(out_dim, in_dim, generator=gen) / (in_dim ** 0.5)
        self.register_buffer("weight", weight)

    def project_h(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x @ self.weight.t(), p=2, dim=1, eps=1e-12)

    def project_c(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x @ self.weight.t(), p=2, dim=1, eps=1e-12)


def strict_modal_embedding(encoder, input_tensor, attention_mask, modality: str, config):
    modal_mask = identify_modality_mask(input_tensor, config, modality)
    keep = modal_mask & attention_mask.bool()
    x = input_tensor.clone()
    m = attention_mask.clone()
    x[~keep] = 0.0
    m[~keep] = 0
    hidden = encoder.encode(x, m)
    pooled, valid = modal_mean_pooling(hidden, m, modal_mask)
    emb = torch.nn.functional.normalize(pooled, p=2, dim=1, eps=1e-12)
    emb = emb * valid.unsqueeze(-1).to(emb.dtype)
    return emb, valid


def strict_hc_embeddings(encoder, input_tensor, attention_mask, config):
    h_emb, h_valid = strict_modal_embedding(encoder, input_tensor, attention_mask, "h_nmr", config)
    c_emb, c_valid = strict_modal_embedding(encoder, input_tensor, attention_mask, "c_nmr", config)
    return h_emb, c_emb, h_valid & c_valid


def build_loader(args, config, split_name: str):
    split = build_hc_splits(args.data_dir, config, subset=args.subset, seed=args.seed, disable_tqdm=True)
    shard_len_cache = Path("results/downstream") / f"hc_shard_lengths_{Path(args.data_dir).name}.npy"
    cached_lengths = None
    if shard_len_cache.exists():
        import numpy as np

        cached_lengths = np.load(shard_len_cache).tolist()
    dataset = HCIndexedDataset(
        args.data_dir,
        split[split_name],
        shard_lengths=cached_lengths,
        shard_cache_size=args.shard_cache_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return split, dataset, loader


def evaluate_setting(setting: str, encoder, heads, loader, config, device, args, split_name: str):
    encoder.eval()
    if heads is not None:
        heads.eval()

    all_h: List[torch.Tensor] = []
    all_c: List[torch.Tensor] = []
    total = len(loader)
    last = time.monotonic()
    log(f"{setting} {split_name}: embedding start batches={total}")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, 1):
            x = batch["input_tensor"].to(device).float()
            m = batch["attention_mask"].to(device)
            h, c, valid = strict_hc_embeddings(encoder, x, m, config)
            if valid.any():
                h = h[valid]
                c = c[valid]
                if heads is not None:
                    h = heads.project_h(h)
                    c = heads.project_c(c)
                all_h.append(h.detach().cpu())
                all_c.append(c.detach().cpu())
            now = time.monotonic()
            if now - last >= 30.0 or batch_idx == total:
                n_pairs = sum(t.shape[0] for t in all_h)
                log(f"{setting} {split_name}: batch {batch_idx}/{total}, pairs={n_pairs}")
                last = now

    h_all = torch.cat(all_h)
    c_all = torch.cat(all_c)
    log(f"{setting} {split_name}: metrics start n_pairs={h_all.shape[0]}")
    h2c = _retrieval_metrics_chunked(
        h_all, c_all, sim_batch=args.sim_batch, topk=args.topk, use_fp16=args.use_fp16, device=device
    )
    c2h = _retrieval_metrics_chunked(
        c_all, h_all, sim_batch=args.sim_batch, topk=args.topk, use_fp16=args.use_fp16, device=device
    )
    log(
        f"{setting} {split_name}: metrics done "
        f"H2C_R1={h2c['R@1']:.6f}, C2H_R1={c2h['R@1']:.6f}, "
        f"H2C_R10={h2c['R@10']:.6f}, C2H_R10={c2h['R@10']:.6f}"
    )
    return {"h_to_c": h2c, "c_to_h": c2h}


def make_heads(setting: str, hidden_dim: int, args, device):
    if setting == "pretrain_only_strict":
        return None
    if setting == "random_projection_strict":
        return SharedRandomProjection(hidden_dim, args.projection_dim, seed=args.seed + 1701).to(device)
    if setting == "untrained_mlp_head_strict":
        torch.manual_seed(args.seed + 2601)
        return HCProjectionHeads(in_dim=hidden_dim, out_dim=args.projection_dim).to(device)
    raise ValueError(f"Unknown setting: {setting}")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    log(f"args: {vars(args)}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder, config = load_pretrained_encoder(args.checkpoint, args.config, device)
    hidden_dim = int(config.model.dim_hidden)
    log(f"loaded pretrained encoder hidden_dim={hidden_dim}, device={device}")
    log("STRICT MODAL INPUT ENABLED: H branch sees only H tokens; C branch sees only C tokens")
    split, val_ds, val_loader = build_loader(args, config, "val")
    _, test_ds, test_loader = build_loader(args, config, "test")
    log(
        f"split sizes: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}; "
        f"eval val={len(val_ds)}, test={len(test_ds)}"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp",
        "setting",
        "subset",
        "seed",
        "val_size",
        "test_size",
        "h2c_r1",
        "h2c_r5",
        "h2c_r10",
        "h2c_mrr10",
        "c2h_r1",
        "c2h_r5",
        "c2h_r10",
        "c2h_mrr10",
    ]
    summary_path = out_dir / "summary.csv"
    if summary_path.exists():
        summary_path.unlink()

    for setting in args.settings:
        heads = make_heads(setting, hidden_dim, args, device)
        val_metrics = evaluate_setting(setting, encoder, heads, val_loader, config, device, args, "val")
        test_metrics = evaluate_setting(setting, encoder, heads, test_loader, config, device, args, "test")
        report = {
            "timestamp": datetime.now().isoformat(),
            "setting": setting,
            "strict_modal_input": True,
            "checkpoint": args.checkpoint,
            "config": args.config,
            "subset": args.subset,
            "seed": args.seed,
            "projection_dim": args.projection_dim,
            "dataset": {"val_size": len(val_ds), "test_size": len(test_ds)},
            "validation_retrieval": val_metrics,
            "test_retrieval": test_metrics,
        }
        report_path = out_dir / f"hc_noncontrastive_{setting}_subset{args.subset}_seed{args.seed}.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        row = {
            "timestamp": report["timestamp"],
            "setting": setting,
            "subset": args.subset,
            "seed": args.seed,
            "val_size": len(val_ds),
            "test_size": len(test_ds),
            "h2c_r1": test_metrics["h_to_c"]["R@1"],
            "h2c_r5": test_metrics["h_to_c"]["R@5"],
            "h2c_r10": test_metrics["h_to_c"]["R@10"],
            "h2c_mrr10": test_metrics["h_to_c"]["MRR"],
            "c2h_r1": test_metrics["c_to_h"]["R@1"],
            "c2h_r5": test_metrics["c_to_h"]["R@5"],
            "c2h_r10": test_metrics["c_to_h"]["R@10"],
            "c2h_mrr10": test_metrics["c_to_h"]["MRR"],
        }
        append_row(summary_path, row, fields)
        log(f"report: {report_path}")

    log(f"summary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL exception follows")
        traceback.print_exc()
        raise


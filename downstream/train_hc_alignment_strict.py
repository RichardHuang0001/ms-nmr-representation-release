#!/usr/bin/env python3
"""Strict H/C alignment: H branch sees only H peaks, C branch sees only C peaks."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from downstream.embedding_utils import (
    build_encoder_from_config,
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
    symmetric_infonce_loss,
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict H/C alignment training/eval")
    p.add_argument("--checkpoint", default="results/checkpoints/best_model.pt")
    p.add_argument("--config", default="configs/pretrain_set_transformer.yaml")
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--output_dir", default="results/downstream/hc_alignment_strict")
    p.add_argument("--mode", required=True, choices=["pretrain_only", "frozen_encoder", "unfreeze_last_block", "scratch_frozen", "scratch_unfreeze_last_block"])
    p.add_argument("--unfreeze_last_n_blocks", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--encoder_lr", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--subset", type=float, default=0.0)
    p.add_argument("--split_strategy", type=str, default="random", choices=["random", "scaffold"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--shard_cache_size", type=int, default=245)
    p.add_argument("--sim_batch", type=int, default=2048)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--use_fp16", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--disable_tqdm", action="store_true")
    p.add_argument("--log_interval", type=float, default=30.0)
    return p.parse_args()


def append_row(path: Path, row: Dict[str, object], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)


def strict_modal_embedding(encoder, input_tensor, attention_mask, modality: str, config) -> Tuple[torch.Tensor, torch.Tensor]:
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


def freeze_encoder(encoder) -> None:
    for p in encoder.parameters():
        p.requires_grad = False


def unfreeze_last_blocks(encoder, n_blocks: int) -> List[str]:
    freeze_encoder(encoder)
    blocks = list(encoder.encoder)
    total_blocks = len(blocks)
    n = min(max(1, n_blocks), total_blocks)
    names: List[str] = []
    for block_idx in range(total_blocks - n, total_blocks):
        for name, param in blocks[block_idx].named_parameters():
            param.requires_grad = True
            names.append(f"encoder.{block_idx}.{name}")
    # Full finetune: also unfreeze input projection when all SAB blocks are trainable
    if n >= total_blocks:
        for name, param in encoder.encoder_input_proj.named_parameters():
            param.requires_grad = True
            names.append(f"encoder_input_proj.{name}")
    return names


def make_optimizer(encoder, heads, mode: str, lr: float, encoder_lr: float):
    if mode == "pretrain_only":
        return None
    if mode in {"frozen_encoder", "scratch_frozen"}:
        return torch.optim.Adam(heads.parameters(), lr=lr)
    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    return torch.optim.Adam([
        {"params": heads.parameters(), "lr": lr},
        {"params": encoder_params, "lr": encoder_lr},
    ])


def build_dataloaders(args, config):
    log("building strict H/C split indices")
    split = build_hc_splits(args.data_dir, config, subset=args.subset, seed=args.seed, disable_tqdm=args.disable_tqdm, split_strategy=args.split_strategy)
    log(f"split sizes: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
    import numpy as np
    shard_len_cache = Path("results/downstream") / f"hc_shard_lengths_{Path(args.data_dir).name}.npy"
    cached_lengths = np.load(shard_len_cache).tolist() if shard_len_cache.exists() else None
    log(f"shard length cache: {shard_len_cache}, cached={cached_lengths is not None}")
    train_ds = HCIndexedDataset(args.data_dir, split["train"], shard_lengths=cached_lengths, shard_cache_size=args.shard_cache_size)
    val_ds = HCIndexedDataset(args.data_dir, split["val"], shard_lengths=cached_lengths, shard_cache_size=args.shard_cache_size)
    test_ds = HCIndexedDataset(args.data_dir, split["test"], shard_lengths=cached_lengths, shard_cache_size=args.shard_cache_size)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    log(f"dataloader config: batch_size={args.batch_size}, num_workers={args.num_workers}, pin_memory={pin}, shard_cache_size={args.shard_cache_size}")
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def evaluate_strict(encoder, heads, loader, config, device, sim_batch: int, topk: int, use_fp16: bool, label: str) -> Dict[str, Dict[str, float]]:
    encoder.eval()
    if heads is not None:
        heads.eval()
    all_h: List[torch.Tensor] = []
    all_c: List[torch.Tensor] = []
    total = len(loader)
    last = time.monotonic()
    log(f"{label} strict retrieval embedding start: batches={total}")
    with torch.no_grad():
        for bi, batch in enumerate(loader, 1):
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
            if now - last >= 30.0 or bi == total:
                pairs = sum(t.shape[0] for t in all_h)
                log(f"{label} strict retrieval batch {bi}/{total}: pairs={pairs}")
                last = now
    if not all_h:
        empty = {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "MRR": 0.0, "n_pairs": 0}
        return {"h_to_c": empty, "c_to_h": empty}
    h_all = torch.cat(all_h)
    c_all = torch.cat(all_c)
    log(f"{label} strict metrics start: n_pairs={h_all.shape[0]}")
    mh = _retrieval_metrics_chunked(h_all, c_all, sim_batch=sim_batch, topk=topk, use_fp16=use_fp16, device=device)
    mc = _retrieval_metrics_chunked(c_all, h_all, sim_batch=sim_batch, topk=topk, use_fp16=use_fp16, device=device)
    log(f"{label} strict metrics done: H2C_R1={mh['R@1']:.6f}, C2H_R1={mc['R@1']:.6f}, H2C_R10={mh['R@10']:.6f}, C2H_R10={mc['R@10']:.6f}")
    return {"h_to_c": mh, "c_to_h": mc}


def val_score(metrics):
    return 0.5 * (metrics["h_to_c"]["MRR"] + metrics["c_to_h"]["MRR"])


def train_epoch(encoder, heads, loader, optimizer, config, device, args, epoch: int):
    if args.mode in {"unfreeze_last_block", "scratch_unfreeze_last_block"}:
        encoder.train()
    else:
        encoder.eval()
    heads.train()
    total_loss = 0.0
    steps = 0
    skipped = 0
    seen = 0
    last = time.monotonic()
    total = len(loader)
    log(f"train epoch {epoch}/{args.epochs} start: batches={total}")
    iterator = tqdm(loader, desc=f"strict train {epoch}/{args.epochs}", disable=args.disable_tqdm, leave=False)
    for bi, batch in enumerate(iterator, 1):
        x = batch["input_tensor"].to(device).float()
        m = batch["attention_mask"].to(device)
        optimizer.zero_grad()
        if args.mode in {"unfreeze_last_block", "scratch_unfreeze_last_block"}:
            h, c, valid = strict_hc_embeddings(encoder, x, m, config)
        else:
            with torch.no_grad():
                h, c, valid = strict_hc_embeddings(encoder, x, m, config)
        vc = int(valid.sum().item())
        seen += int(x.shape[0])
        if vc < 2:
            skipped += 1
            continue
        h = heads.project_h(h[valid])
        c = heads.project_c(c[valid])
        loss = symmetric_infonce_loss(h, c, temperature=args.temperature)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        steps += 1
        now = time.monotonic()
        if now - last >= args.log_interval or bi == total:
            avg = total_loss / max(1, steps)
            log(f"train epoch {epoch}/{args.epochs} batch {bi}/{total}: loss={loss.item():.4f}, avg_loss={avg:.4f}, valid_pairs={vc}, skipped={skipped}, seen={seen}")
            last = now
    avg = total_loss / max(1, steps)
    log(f"train epoch {epoch}/{args.epochs} done: steps={steps}, skipped={skipped}, avg_loss={avg:.4f}")
    return avg


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    log(f"args: {vars(args)}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = load_model_config(args.config)
    if args.mode.startswith("scratch"):
        encoder = build_encoder_from_config(config).to(device)
        base_checkpoint = None
    else:
        encoder, config = load_pretrained_encoder(args.checkpoint, args.config, device)
        base_checkpoint = args.checkpoint
    hidden_dim = int(config.model.dim_hidden)
    heads = None if args.mode == "pretrain_only" else HCProjectionHeads(in_dim=hidden_dim, out_dim=128).to(device)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(args, config)

    unfrozen: List[str] = []
    if args.mode in {"unfreeze_last_block", "scratch_unfreeze_last_block"}:
        unfrozen = unfreeze_last_blocks(encoder, args.unfreeze_last_n_blocks)
    else:
        freeze_encoder(encoder)
    optimizer = make_optimizer(encoder, heads, args.mode, args.lr, args.encoder_lr) if heads is not None else None
    trainable = sum(p.numel() for p in list(encoder.parameters()) + ([] if heads is None else list(heads.parameters())) if p.requires_grad)
    log(f"loaded model: mode={args.mode}, hidden_dim={hidden_dim}, device={device}, trainable_params={trainable}, optimizer={type(optimizer).__name__ if optimizer else None}")
    log("STRICT MODAL INPUT ENABLED: H branch sees only H tokens; C branch sees only C tokens")

    history = []
    best_epoch = 0
    log("initial validation start")
    best_metrics = evaluate_strict(encoder, heads, val_loader, config, device, args.sim_batch, args.topk, args.use_fp16, "val_initial")
    best = val_score(best_metrics)
    log(f"initial validation done: val_mrr10_mean={best:.6f}")

    if args.mode != "pretrain_only":
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(encoder, heads, train_loader, optimizer, config, device, args, epoch)
            metrics = evaluate_strict(encoder, heads, val_loader, config, device, args.sim_batch, args.topk, args.use_fp16, f"val_epoch_{epoch}")
            score = val_score(metrics)
            history.append({"epoch": epoch, "train_loss": loss, "val_mrr10_mean": score})
            log(f"epoch {epoch}/{args.epochs} summary: loss={loss:.6f}, val_mrr10_mean={score:.6f}")
            if score > best:
                best = score
                best_epoch = epoch
                best_metrics = metrics

    test_metrics = evaluate_strict(encoder, heads, test_loader, config, device, args.sim_batch, args.topk, args.use_fp16, "test")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"hc_alignment_strict_{args.mode}_subset{args.subset}_seed{args.seed}"
    report_path = out_dir / f"{run_name}.json"
    ckpt_path = out_dir / f"{run_name}.pt"
    if heads is not None:
        torch.save({
            "mode": args.mode,
            "strict_modal_input": True,
            "base_checkpoint": base_checkpoint,
            "config": args.config,
            "projection_heads": heads.state_dict(),
            "encoder_state_dict": encoder.state_dict() if args.mode in {"unfreeze_last_block", "scratch_unfreeze_last_block"} else None,
            "unfrozen_encoder_params": unfrozen,
            "best_epoch": best_epoch,
            "best_val_metrics": best_metrics,
            "test_metrics": test_metrics,
        }, ckpt_path)
    report = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "strict_modal_input": True,
        "checkpoint": base_checkpoint,
        "config": args.config,
        "data_dir": args.data_dir,
        "subset": args.subset,
        "seed": args.seed,
        "dataset": {"train_size": len(train_ds), "val_size": len(val_ds), "test_size": len(test_ds)},
        "training": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "encoder_lr": args.encoder_lr, "temperature": args.temperature, "best_epoch": best_epoch, "best_val_mrr10_mean": best, "unfreeze_last_n_blocks": args.unfreeze_last_n_blocks},
        "validation_retrieval": best_metrics,
        "test_retrieval": test_metrics,
        "history": history,
        "checkpoint_out": str(ckpt_path) if heads is not None else None,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["timestamp","mode","subset","seed","train_size","val_size","test_size","best_epoch","h2c_r1","h2c_r5","h2c_r10","h2c_mrr10","c2h_r1","c2h_r5","c2h_r10","c2h_mrr10"]
    append_row(out_dir / "summary.csv", {
        "timestamp": report["timestamp"], "mode": args.mode, "subset": args.subset, "seed": args.seed,
        "train_size": len(train_ds), "val_size": len(val_ds), "test_size": len(test_ds), "best_epoch": best_epoch,
        "h2c_r1": test_metrics["h_to_c"]["R@1"], "h2c_r5": test_metrics["h_to_c"]["R@5"], "h2c_r10": test_metrics["h_to_c"]["R@10"], "h2c_mrr10": test_metrics["h_to_c"]["MRR"],
        "c2h_r1": test_metrics["c_to_h"]["R@1"], "c2h_r5": test_metrics["c_to_h"]["R@5"], "c2h_r10": test_metrics["c_to_h"]["R@10"], "c2h_mrr10": test_metrics["c_to_h"]["MRR"],
    }, fields)
    log(f"report: {report_path}")
    if heads is not None:
        log(f"checkpoint: {ckpt_path}")
    log(f"summary: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL exception follows")
        traceback.print_exc()
        raise

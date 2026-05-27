#!/usr/bin/env python3
"""
H/C alignment-aware retrieval evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from downstream.embedding_utils import load_pretrained_encoder
from downstream.hc_alignment_utils import (
    HCProjectionHeads,
    append_summary_row,
    build_hc_dataloaders,
    build_hc_splits,
    evaluate_hc_retrieval,
    seed_everything,
)


SETTINGS = [
    "pretrain_only",
    "projection_only",
    "contrastive_frozen",
    "contrastive_unfreeze_last_block",
    "contrastive_scratch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate H/C alignment retrieval")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/pretrain_set_transformer.yaml")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="results/downstream/hc_alignment_eval")

    parser.add_argument("--projection_ckpt", default="")
    parser.add_argument("--contrastive_frozen_ckpt", default="")
    parser.add_argument("--contrastive_unfreeze_ckpt", default="")
    parser.add_argument("--contrastive_scratch_ckpt", default="")

    parser.add_argument("--setting", default="all", choices=["all"] + SETTINGS)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--subset", type=float, default=0)
    parser.add_argument("--disable_tqdm", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim_batch", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard_cache_size", type=int, default=2)
    return parser.parse_args()


def load_heads_from_checkpoint(path: str, in_dim: int, device: torch.device) -> HCProjectionHeads:
    if not path:
        raise ValueError("需要提供 alignment checkpoint 路径")
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    heads = HCProjectionHeads(in_dim=in_dim, out_dim=128).to(device)
    heads.load_state_dict(payload["projection_heads"])
    return heads


def load_alignment_payload(path: str, device: torch.device) -> Dict[str, object]:
    if not path:
        raise ValueError("需要提供 alignment checkpoint 路径")
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    return torch.load(ckpt_path, map_location=device, weights_only=False)


def evaluate_setting(
    setting: str,
    encoder: torch.nn.Module,
    hidden_dim: int,
    test_loader,
    config,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Dict[str, float]]:
    eval_encoder = encoder
    if setting == "pretrain_only":
        heads = None
    elif setting == "projection_only":
        if args.projection_ckpt:
            heads = load_heads_from_checkpoint(args.projection_ckpt, hidden_dim, device)
        else:
            heads = HCProjectionHeads(in_dim=hidden_dim, out_dim=128).to(device)
    elif setting == "contrastive_frozen":
        heads = load_heads_from_checkpoint(args.contrastive_frozen_ckpt, hidden_dim, device)
    elif setting == "contrastive_unfreeze_last_block":
        payload = load_alignment_payload(args.contrastive_unfreeze_ckpt, device)
        heads = HCProjectionHeads(in_dim=hidden_dim, out_dim=128).to(device)
        heads.load_state_dict(payload["projection_heads"])

        # Unfreeze-based checkpoints depend on the adapted encoder weights.
        # Evaluating only the heads against the original pretrained encoder
        # underestimates the true retrieval performance of that setting.
        eval_encoder = copy.deepcopy(encoder).to(device)
        encoder_state = payload.get("encoder_state_dict")
        if encoder_state is None:
            raise ValueError("unfreeze checkpoint 缺少 encoder_state_dict")
        eval_encoder.load_state_dict(encoder_state)
        eval_encoder.eval()
    elif setting == "contrastive_scratch":
        payload = load_alignment_payload(args.contrastive_scratch_ckpt, device)
        heads = HCProjectionHeads(in_dim=hidden_dim, out_dim=128).to(device)
        heads.load_state_dict(payload["projection_heads"])

        # Scratch checkpoints must restore the jointly-trained encoder weights.
        eval_encoder = copy.deepcopy(encoder).to(device)
        encoder_state = payload.get("encoder_state_dict")
        if encoder_state is None:
            raise ValueError("scratch checkpoint 缺少 encoder_state_dict")
        eval_encoder.load_state_dict(encoder_state)
        eval_encoder.eval()
    else:
        raise ValueError(f"Unknown setting: {setting}")

    return evaluate_hc_retrieval(
        encoder=eval_encoder,
        heads=heads,
        dataloader=test_loader,
        config=config,
        device=device,
        sim_batch=args.sim_batch,
        topk=args.topk,
        use_fp16=args.use_fp16,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder, config = load_pretrained_encoder(args.checkpoint, args.config, device)
    encoder.eval()
    hidden_dim = int(config.model.dim_hidden)

    split_map = build_hc_splits(
        processed_dir=args.data_dir,
        config=config,
        subset=args.subset,
        seed=args.seed,
        disable_tqdm=args.disable_tqdm,
    )
    _, _, test_ds, _, _, test_loader = build_hc_dataloaders(
        processed_dir=args.data_dir,
        split_map=split_map,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shard_cache_size=args.shard_cache_size,
    )

    if args.setting == "all":
        setting_list = SETTINGS
    else:
        setting_list = [args.setting]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for setting in setting_list:
        metrics = evaluate_setting(setting, encoder, hidden_dim, test_loader, config, device, args)

        report = {
            "timestamp": datetime.now().isoformat(),
            "setting": setting,
            "checkpoint": args.checkpoint,
            "config": args.config,
            "subset": args.subset,
            "seed": args.seed,
            "test_size": len(test_ds),
            "metrics": metrics,
        }

        setting_path = output_dir / f"hc_eval_{setting}.json"
        with open(setting_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        summary_rows.append(
            {
                "setting": setting,
                "H→C R@1": metrics["h_to_c"]["R@1"],
                "H→C R@5": metrics["h_to_c"]["R@5"],
                "H→C R@10": metrics["h_to_c"]["R@10"],
                "H→C MRR": metrics["h_to_c"]["MRR"],
                "C→H R@1": metrics["c_to_h"]["R@1"],
                "C→H R@5": metrics["c_to_h"]["R@5"],
                "C→H R@10": metrics["c_to_h"]["R@10"],
                "C→H MRR": metrics["c_to_h"]["MRR"],
            }
        )

        print(f"[{setting}] H->C R@1={metrics['h_to_c']['R@1']:.4f} MRR={metrics['h_to_c']['MRR']:.4f} | C->H R@1={metrics['c_to_h']['R@1']:.4f} MRR={metrics['c_to_h']['MRR']:.4f}")

    summary_path = output_dir / "hc_alignment_table.csv"
    summary_fields = [
        "timestamp",
        "setting",
        "H→C R@1",
        "H→C R@5",
        "H→C R@10",
        "H→C MRR",
        "C→H R@1",
        "C→H R@5",
        "C→H R@10",
        "C→H MRR",
    ]

    for row in summary_rows:
        append_summary_row(
            summary_path,
            {"timestamp": datetime.now().isoformat(), **row},
            fieldnames=summary_fields,
        )

    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

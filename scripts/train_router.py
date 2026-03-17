#!/usr/bin/env python3
"""Train the Temporal-Progress Router.

Step-based training loop (100K steps by default). Cycles the dataloader.
BCEWithLogitsLoss on soft labels. AdamW with separate backbone LR.

Usage:
    # Smoke test
    python scripts/train_router.py --steps 100 --batch_size 4 --num_workers 0

    # Full run
    python scripts/train_router.py --steps 100000 --use_amp
"""

import argparse
import sys
import time
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from router.model import TemporalProgressRouter, RouterConfig
from router.dataset import RouterDataset, router_collate_fn


def parse_args():
    p = argparse.ArgumentParser(description="Train Temporal-Progress Router")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr_backbone", type=float, default=1e-6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--log_freq", type=int, default=200)
    p.add_argument("--save_freq", type=int, default=10_000)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt_dir", type=str,
                   default=str(PROJECT / "model" / "router" / "checkpoints"))
    p.add_argument("--act_ckpt", type=str,
                   default=str(PROJECT / "model" / "stage1_act" / "checkpoints" / "last" / "pretrained_model"))
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────
    print("Loading router model from ACT checkpoint...")
    model = TemporalProgressRouter.load_from_act_checkpoint(args.act_ckpt)
    model = model.to(device)
    model.train()

    # ── Dataset / DataLoader ─────────────────────────────────────────────
    print("Building dataset...")
    dataset = RouterDataset()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=router_collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    data_iter = cycle(loader)

    # ── Optimizer ────────────────────────────────────────────────────────
    param_groups = model.get_param_groups(args.lr_backbone, args.lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(device, enabled=args.use_amp)

    # ── Training loop ────────────────────────────────────────────────────
    print(f"Training for {args.steps} steps, batch_size={args.batch_size}, "
          f"AMP={args.use_amp}, device={device}")
    t0 = time.time()

    for step in range(1, args.steps + 1):
        images, states, labels = next(data_iter)
        images = [img.to(device, non_blocking=True) for img in images]
        states = states.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device, enabled=args.use_amp):
            logits = model(images, states)  # (B, 100)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ── Logging ──────────────────────────────────────────────────────
        if step % args.log_freq == 0 or step == 1:
            with torch.no_grad():
                mean_prob = torch.sigmoid(logits).mean().item()
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            remaining = (args.steps - step) / steps_per_sec
            eta_h, eta_m = divmod(int(remaining), 3600)
            eta_m //= 60
            pct = step / args.steps * 100
            print(
                f"step {step:>6d}/{args.steps} ({pct:5.1f}%) | "
                f"loss {loss.item():.4f} | "
                f"mean P(B) {mean_prob:.3f} | "
                f"lr_bb {args.lr_backbone:.1e} lr {args.lr:.1e} | "
                f"{steps_per_sec:.1f} steps/s | "
                f"ETA {eta_h}h{eta_m:02d}m",
                flush=True,
            )

        # ── Checkpoint ───────────────────────────────────────────────────
        if step % args.save_freq == 0 or step == args.steps:
            save_path = ckpt_dir / f"step_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model.config,
                "args": vars(args),
            }, save_path)
            print(f"  Saved checkpoint → {save_path}")

    total_time = time.time() - t0
    print(f"Training complete. {args.steps} steps in {total_time:.0f}s "
          f"({args.steps / total_time:.1f} steps/s)")


if __name__ == "__main__":
    main()

"""
Training script for NTIRE 2026 SISR (×4).

Features:
  • Mixed-precision training (AMP)
  • Fixed random seeds for reproducibility
  • Cosine annealing LR schedule
  • Validation every N epochs (PSNR / SSIM with 4-px border crop)
  • Best-model checkpointing by validation PSNR

Usage:
    python train.py --dataset_dir ./data --epochs 1000
"""

import argparse
import os

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import RCAN
from dataset import DIV2KDataset
from loss import CompositeLoss
from utils import calculate_psnr, calculate_ssim, tensor2img, set_random_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train SISR Model for NTIRE 2026")
    p.add_argument("--dataset_dir", type=str, default="./data",
                   help="Root of prepared DIV2K data")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=64,
                   help="LR patch size (HR = patch_size × scale)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--val_every", type=int, default=5,
                   help="Run validation every N epochs")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


def train(args):
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ───────────────────────────────────────────────────────
    train_ds = DIV2KDataset(args.dataset_dir, scale=args.scale,
                            subset="train", patch_size=args.patch_size)
    valid_ds = DIV2KDataset(args.dataset_dir, scale=args.scale,
                            subset="valid")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)} | Valid: {len(valid_ds)}")

    # ── Model / Loss / Optimizer ───────────────────────────────────
    model = RCAN(scale=args.scale).to(device)
    criterion = CompositeLoss(charb_weight=1.0, percept_weight=0.01).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                T_max=args.epochs, eta_min=1e-7)

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda")

    # ── Resume ─────────────────────────────────────────────────────
    start_epoch = 0
    best_psnr = 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"Resumed from epoch {start_epoch}, best PSNR {best_psnr:.4f}")

    # ── Training Loop ──────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch in pbar:
            lr_img = batch["lr"].to(device, non_blocking=True)
            hr_img = batch["hr"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                sr = model(lr_img)
                loss, l_pix, l_per = criterion(sr, hr_img)

            scaler.scale(loss).backward()
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             pix=f"{l_pix.item():.4f}",
                             per=f"{l_per.item():.4f}")

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1:04d} | LR {scheduler.get_last_lr()[0]:.2e} "
              f"| Loss {avg_loss:.5f}")

        # ── Validation ─────────────────────────────────────────────
        if (epoch + 1) % args.val_every == 0 or epoch == 0:
            val_psnr, val_ssim = validate(model, valid_loader, device)
            print(f"  >> Val PSNR {val_psnr:.4f} dB | SSIM {val_ssim:.4f}")

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                _save_ckpt(model, optimizer, scheduler, epoch,
                           best_psnr, args.checkpoint_dir, "best_model.pth")
                print(f"  ** New best! ({best_psnr:.4f} dB)")

        # Periodic checkpoint
        if (epoch + 1) % 50 == 0:
            _save_ckpt(model, optimizer, scheduler, epoch,
                       best_psnr, args.checkpoint_dir,
                       f"ckpt_epoch_{epoch+1}.pth")


# ──────────────────────────────────────────────────────────────────────
def validate(model, loader, device):
    model.eval()
    total_psnr, total_ssim = 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Validating", leave=False):
            lr_img = batch["lr"].to(device, non_blocking=True)
            hr_img = batch["hr"]

            try:
                sr = model(lr_img)
            except RuntimeError:
                torch.cuda.empty_cache()
                continue

            sr_np = tensor2img(sr.cpu().squeeze(0))
            hr_np = tensor2img(hr_img.squeeze(0))

            total_psnr += calculate_psnr(sr_np, hr_np, crop_border=4)
            total_ssim += calculate_ssim(sr_np, hr_np, crop_border=4)

    n = len(loader)
    return total_psnr / n, total_ssim / n


def _save_ckpt(model, optimizer, scheduler, epoch, best_psnr, ckpt_dir, name):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_psnr": best_psnr,
    }, os.path.join(ckpt_dir, name))


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train(args)

"""
Train SwinIR for x4 Super-Resolution on DIV2K.

Uses the SwinIR model from the cloned repository with:
  - Classical SR configuration (6 RSTB groups, embed_dim=180, window_size=8)
  - Charbonnier + VGG Perceptual composite loss
  - Mixed-precision (AMP), cosine annealing, gradient clipping
  - Validation every N epochs (PSNR / SSIM with 4-px border crop)

Usage:
    python train_swinir.py --dataset_dir ./data --epochs 500 --batch_size 8

NOTE: SwinIR is more memory-intensive than RCAN.
      On A6000 (48GB): batch_size=8, patch_size=48 is safe.
      On 24GB GPUs:    batch_size=4, patch_size=48 recommended.
"""

import argparse
import os
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the SwinIR repo to path so we can import the model
SWINIR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SwinIR")
sys.path.insert(0, SWINIR_DIR)

from models.network_swinir import SwinIR

# Our existing modules
from dataset import DIV2KDataset
from loss import CompositeLoss
from utils import calculate_psnr, calculate_ssim, tensor2img, set_random_seed


# ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train SwinIR for NTIRE 2026 SISR x4")
    p.add_argument("--dataset_dir", type=str, default="./data")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=48,
                   help="LR patch size (HR = patch_size x scale)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints_swinir")
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--resume", type=str, default=None)

    # SwinIR architecture args
    p.add_argument("--embed_dim", type=int, default=180,
                   help="SwinIR embedding dimension (180=medium, 60=lightweight)")
    p.add_argument("--depths", type=int, nargs="+", default=[6, 6, 6, 6, 6, 6],
                   help="Depths for each RSTB group")
    p.add_argument("--num_heads", type=int, nargs="+", default=[6, 6, 6, 6, 6, 6],
                   help="Number of attention heads per group")
    p.add_argument("--window_size", type=int, default=8)

    # Loss weights
    p.add_argument("--percept_weight", type=float, default=0.01,
                   help="Perceptual loss weight (keep low for PSNR)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
def build_swinir(args):
    """Instantiate SwinIR with classical SR config."""
    model = SwinIR(
        upscale=args.scale,
        in_chans=3,
        img_size=args.patch_size,       # training patch size (for positional info)
        window_size=args.window_size,
        img_range=1.,
        depths=args.depths,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv',
    )
    return model


# ──────────────────────────────────────────────────────────────────────
def train(args):
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────
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

    # ── Model ─────────────────────────────────────────────────────
    model = build_swinir(args).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"SwinIR parameters: {n_params:.2f}M")

    # ── Loss / Optimizer ──────────────────────────────────────────
    criterion = CompositeLoss(charb_weight=1.0,
                              percept_weight=args.percept_weight).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                T_max=args.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda")

    # ── Resume ────────────────────────────────────────────────────
    start_epoch = 0
    best_psnr = 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"Resumed from epoch {start_epoch}, best PSNR {best_psnr:.4f}")

    # ── Training Loop ─────────────────────────────────────────────
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

        # ── Validation ────────────────────────────────────────────
        if (epoch + 1) % args.val_every == 0 or epoch == 0:
            val_psnr, val_ssim = validate(model, valid_loader, device,
                                          args.window_size, args.scale)
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

    print(f"\nTraining complete. Best PSNR: {best_psnr:.4f} dB")


# ──────────────────────────────────────────────────────────────────────
def validate(model, loader, device, window_size, scale):
    """Validate with proper padding for SwinIR (window-size multiple)."""
    model.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Validating", leave=False):
            lr_img = batch["lr"].to(device, non_blocking=True)
            hr_img = batch["hr"]  # keep on CPU for metric computation

            _, _, h_old, w_old = lr_img.size()

            # Pad to window_size multiple (reflect padding via flip, same as SwinIR repo)
            h_pad = (window_size - h_old % window_size) % window_size
            w_pad = (window_size - w_old % window_size) % window_size
            if h_pad > 0 or w_pad > 0:
                lr_img = torch.nn.functional.pad(lr_img, (0, w_pad, 0, h_pad),
                                                  mode='reflect')

            try:
                sr = model(lr_img)
            except RuntimeError:
                torch.cuda.empty_cache()
                continue

            # Crop to original size * scale
            sr = sr[:, :, :h_old * scale, :w_old * scale]

            sr_np = tensor2img(sr.cpu().squeeze(0))
            hr_np = tensor2img(hr_img.squeeze(0))

            total_psnr += calculate_psnr(sr_np, hr_np, crop_border=4)
            total_ssim += calculate_ssim(sr_np, hr_np, crop_border=4)
            count += 1

    return total_psnr / max(count, 1), total_ssim / max(count, 1)


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

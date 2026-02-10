"""
Quick smoke-test: train on a tiny subset for 2 epochs to verify the full
pipeline (data loading → model → loss → backward → validation) is bug-free.
"""

import os
import sys
import shutil
import glob
import torch

# ── Create a tiny dataset (first 10 train, first 5 valid) ──
PROJ = os.path.dirname(os.path.abspath(__file__))
FULL_DATA = os.path.join(PROJ, "data")
TINY_DATA = os.path.join(PROJ, "data_tiny")

def setup_tiny():
    splits = [
        ("DIV2K_train_HR", "DIV2K_train_HR", 10),
        ("DIV2K_train_LR_bicubic/X4", "DIV2K_train_LR_bicubic/X4", 10),
        ("DIV2K_valid_HR", "DIV2K_valid_HR", 5),
        ("DIV2K_valid_LR_bicubic/X4", "DIV2K_valid_LR_bicubic/X4", 5),
    ]
    for src_sub, dst_sub, n in splits:
        src = os.path.join(FULL_DATA, src_sub)
        dst = os.path.join(TINY_DATA, dst_sub)
        os.makedirs(dst, exist_ok=True)
        files = sorted(glob.glob(os.path.join(src, "*.png")))[:n]
        for f in files:
            tgt = os.path.join(dst, os.path.basename(f))
            if not os.path.exists(tgt):
                shutil.copy2(f, tgt)
    print(f"Tiny dataset ready at {TINY_DATA}")

if __name__ == "__main__":
    setup_tiny()

    # ── Import after path is set ──
    from model import RCAN
    from dataset import DIV2KDataset
    from loss import CompositeLoss
    from utils import set_random_seed, calculate_psnr, calculate_ssim, tensor2img

    set_random_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    train_ds = DIV2KDataset(TINY_DATA, scale=4, subset="train", patch_size=64)
    valid_ds = DIV2KDataset(TINY_DATA, scale=4, subset="valid")

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                              num_workers=0, pin_memory=False)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    # Model (smaller for speed: fewer groups and blocks)
    model = RCAN(n_resgroups=2, n_resblocks=4, n_feats=32, scale=4).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Test model: {params:.2f}M params")

    criterion = CompositeLoss(charb_weight=1.0, percept_weight=0.01).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    # ── Train 2 epochs ──
    for epoch in range(2):
        model.train()
        for i, batch in enumerate(train_loader):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                sr = model(lr)
                loss, l_pix, l_per = criterion(sr, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            print(f"  Epoch {epoch+1} | Batch {i+1} | "
                  f"Loss {loss.item():.4f}  Pix {l_pix.item():.4f}  "
                  f"Per {l_per.item():.4f}")

        # ── Validate ──
        model.eval()
        psnrs, ssims = [], []
        with torch.no_grad():
            for batch in valid_loader:
                lr = batch["lr"].to(device)
                hr = batch["hr"]
                sr = model(lr)

                sr_np = tensor2img(sr.cpu().squeeze(0))
                hr_np = tensor2img(hr.squeeze(0))
                psnrs.append(calculate_psnr(sr_np, hr_np, crop_border=4))
                ssims.append(calculate_ssim(sr_np, hr_np, crop_border=4))

        avg_p = sum(psnrs) / len(psnrs)
        avg_s = sum(ssims) / len(ssims)
        print(f"  >> Val PSNR {avg_p:.2f} dB | SSIM {avg_s:.4f}")

    # ── Save & reload test ──
    ckpt_path = os.path.join(PROJ, "checkpoints", "test_ckpt.pth")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    model2 = RCAN(n_resgroups=2, n_resblocks=4, n_feats=32, scale=4).to(device)
    model2.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    model2.eval()

    # Determinism check: same input → same output
    test_input = torch.randn(1, 3, 32, 32, device=device)
    with torch.no_grad():
        o1 = model2(test_input)
        o2 = model2(test_input)
    assert torch.equal(o1, o2), "FAIL: Non-deterministic output!"
    print("\n[PASS] Determinism check passed")
    print("[PASS] All smoke tests passed!")

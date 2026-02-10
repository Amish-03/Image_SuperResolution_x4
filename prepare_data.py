"""
Prepare DIV2K dataset: generate LR images from HR via bicubic downsampling (x4).

The Kaggle DIV2K dataset only ships HR images, so this script creates the
corresponding LR counterparts needed for training and validation.

Usage:
    python prepare_data.py
"""

import os
import cv2
import glob
from tqdm import tqdm

# ─── Configuration ──────────────────────────────────────────────────────
KAGGLE_ROOT = os.path.join(
    os.path.expanduser("~"),
    ".cache", "kagglehub", "datasets",
    "soumikrakshit", "div2k-high-resolution-images", "versions", "1"
)

# Output directory (local to project)
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

SCALE = 4

# Mapping: (hr_source, hr_dest, lr_dest)
SPLITS = [
    {
        "name": "train",
        "hr_src": os.path.join(KAGGLE_ROOT, "DIV2K_train_HR", "DIV2K_train_HR"),
        "hr_dst": os.path.join(DATA_ROOT, "DIV2K_train_HR"),
        "lr_dst": os.path.join(DATA_ROOT, "DIV2K_train_LR_bicubic", f"X{SCALE}"),
    },
    {
        "name": "valid",
        "hr_src": os.path.join(KAGGLE_ROOT, "DIV2K_valid_HR", "DIV2K_valid_HR"),
        "hr_dst": os.path.join(DATA_ROOT, "DIV2K_valid_HR"),
        "lr_dst": os.path.join(DATA_ROOT, "DIV2K_valid_LR_bicubic", f"X{SCALE}"),
    },
]


def generate_lr_images():
    for split in SPLITS:
        print(f"\n{'='*60}")
        print(f"Processing {split['name']} split")
        print(f"{'='*60}")

        hr_src = split["hr_src"]
        hr_dst = split["hr_dst"]
        lr_dst = split["lr_dst"]

        if not os.path.isdir(hr_src):
            print(f"  [ERROR] HR source not found: {hr_src}")
            continue

        os.makedirs(hr_dst, exist_ok=True)
        os.makedirs(lr_dst, exist_ok=True)

        hr_files = sorted(glob.glob(os.path.join(hr_src, "*.png")))
        print(f"  Found {len(hr_files)} HR images")

        for hr_path in tqdm(hr_files, desc=f"  {split['name']}"):
            fname = os.path.basename(hr_path)

            # ── Copy / symlink HR image ──
            hr_dest_path = os.path.join(hr_dst, fname)
            if not os.path.exists(hr_dest_path):
                # Use a hard-copy so the project is self-contained
                img_hr = cv2.imread(hr_path, cv2.IMREAD_COLOR)
                cv2.imwrite(hr_dest_path, img_hr)
            else:
                img_hr = cv2.imread(hr_dest_path, cv2.IMREAD_COLOR)

            # ── Generate LR image via bicubic downsampling ──
            h, w = img_hr.shape[:2]

            # Ensure HR dimensions are divisible by SCALE
            h_new = h - (h % SCALE)
            w_new = w - (w % SCALE)
            if h_new != h or w_new != w:
                img_hr = img_hr[:h_new, :w_new]
                # Overwrite the HR copy with the cropped version
                cv2.imwrite(hr_dest_path, img_hr)

            lr_h, lr_w = h_new // SCALE, w_new // SCALE
            img_lr = cv2.resize(img_hr, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

            # Name convention: same base name (e.g. 0001.png)
            lr_path = os.path.join(lr_dst, fname)
            cv2.imwrite(lr_path, img_lr)

    print(f"\n{'='*60}")
    print("Done!  Dataset structure:")
    print(f"  {DATA_ROOT}")
    for split in SPLITS:
        hr_count = len(glob.glob(os.path.join(split["hr_dst"], "*.png")))
        lr_count = len(glob.glob(os.path.join(split["lr_dst"], "*.png")))
        print(f"  {split['name']:>5s}  HR: {hr_count:4d}   LR: {lr_count:4d}")
    print(f"{'='*60}")


if __name__ == "__main__":
    generate_lr_images()

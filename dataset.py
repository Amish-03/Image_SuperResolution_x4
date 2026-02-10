"""
DIV2K Dataset loader for ×4 Super-Resolution training and validation.

Expects the directory structure created by prepare_data.py:

    data/
      DIV2K_train_HR/           0001.png … 0800.png
      DIV2K_train_LR_bicubic/
        X4/                     0001.png … 0800.png
      DIV2K_valid_HR/           0801.png … 0900.png
      DIV2K_valid_LR_bicubic/
        X4/                     0801.png … 0900.png
"""

import os
import glob
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DIV2KDataset(Dataset):
    """
    Paired LR / HR dataset.

    For training  → returns random patch pairs with augmentation.
    For validation → returns full-size images + filename.
    """

    def __init__(self, dataset_dir, scale=4, subset="train",
                 patch_size=64, augment=True):
        super().__init__()
        self.scale = scale
        self.patch_size = patch_size          # LR patch size
        self.augment = augment and (subset == "train")
        self.subset = subset

        if subset == "train":
            hr_dir = os.path.join(dataset_dir, "DIV2K_train_HR")
            lr_dir = os.path.join(dataset_dir, "DIV2K_train_LR_bicubic", f"X{scale}")
        elif subset == "valid":
            hr_dir = os.path.join(dataset_dir, "DIV2K_valid_HR")
            lr_dir = os.path.join(dataset_dir, "DIV2K_valid_LR_bicubic", f"X{scale}")
        else:
            raise ValueError(f"Unknown subset: {subset}")

        self.hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
        self.lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.png")))

        assert len(self.hr_files) > 0, f"No HR images found in {hr_dir}"
        assert len(self.lr_files) > 0, f"No LR images found in {lr_dir}"
        assert len(self.hr_files) == len(self.lr_files), \
            f"HR/LR count mismatch: {len(self.hr_files)} vs {len(self.lr_files)}"

        print(f"  [{subset}] Loaded {len(self.hr_files)} image pairs")

    # ────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        # Load (BGR → RGB, float32 [0,1])
        hr = cv2.cvtColor(cv2.imread(self.hr_files[idx], cv2.IMREAD_COLOR),
                          cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lr = cv2.cvtColor(cv2.imread(self.lr_files[idx], cv2.IMREAD_COLOR),
                          cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.subset == "train":
            lr, hr = self._random_crop(lr, hr)
            if self.augment:
                lr, hr = self._augment(lr, hr)

        lr_t = torch.from_numpy(np.ascontiguousarray(lr)).permute(2, 0, 1)
        hr_t = torch.from_numpy(np.ascontiguousarray(hr)).permute(2, 0, 1)

        if self.subset == "train":
            return {"lr": lr_t, "hr": hr_t}
        else:
            return {"lr": lr_t, "hr": hr_t,
                    "filename": os.path.basename(self.lr_files[idx])}

    # ────────────────────────────────────────────────────────────────
    def _random_crop(self, lr, hr):
        """Extract a random patch pair."""
        lh, lw = lr.shape[:2]
        p = self.patch_size

        x = random.randint(0, lw - p)
        y = random.randint(0, lh - p)

        lr_patch = lr[y:y + p, x:x + p]
        hr_patch = hr[y * self.scale:(y + p) * self.scale,
                       x * self.scale:(x + p) * self.scale]
        return lr_patch, hr_patch

    def _augment(self, lr, hr):
        """Horizontal / vertical flip + 90° rotation (preserves PSNR)."""
        if random.random() < 0.5:
            lr = lr[:, ::-1, :]
            hr = hr[:, ::-1, :]
        if random.random() < 0.5:
            lr = lr[::-1, :, :]
            hr = hr[::-1, :, :]
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            lr = np.rot90(lr, k).copy()
            hr = np.rot90(hr, k).copy()
        return lr, hr

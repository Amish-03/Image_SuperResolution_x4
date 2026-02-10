"""
Inference script for NTIRE 2026 SISR submission.

Deterministic single-image ×4 super-resolution.
Outputs lossless PNG with the same filename as input.

Usage:
    python inference.py \
        --input_dir  ./data/DIV2K_valid_LR_bicubic/X4 \
        --output_dir ./results \
        --model_path ./checkpoints/best_model.pth
"""

import argparse
import os
import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

from model import RCAN
from utils import set_random_seed


def parse_args():
    p = argparse.ArgumentParser(description="NTIRE 2026 SISR Inference")
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--tile_size", type=int, default=400,
                   help="LR tile size for OOM fallback")
    p.add_argument("--tile_pad", type=int, default=32,
                   help="Overlap padding for tiles")
    return p.parse_args()


def _to_tensor(img_bgr):
    """BGR uint8 → RGB float32 tensor (1,3,H,W)."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0)


def _to_img(tensor):
    """(1,3,H,W) float tensor → BGR uint8 ndarray."""
    t = tensor.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (cv2.cvtColor(t, cv2.COLOR_RGB2BGR) * 255.0).round().astype(np.uint8)


def tile_forward(model, img_t, tile_size, tile_pad, scale, device):
    """Process a large image in overlapping tiles to avoid OOM."""
    _, c, h, w = img_t.shape
    out = torch.zeros(1, c, h * scale, w * scale, device="cpu")

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y0 = max(0, y - tile_pad)
            x0 = max(0, x - tile_pad)
            y1 = min(h, y + tile_size + tile_pad)
            x1 = min(w, x + tile_size + tile_pad)

            patch = img_t[:, :, y0:y1, x0:x1].to(device)
            sr_patch = model(patch).cpu()

            # Crop padding in output space
            oy = (y - y0) * scale
            ox = (x - x0) * scale
            oh = min(tile_size, h - y) * scale
            ow = min(tile_size, w - x) * scale

            out[:, :, y*scale:y*scale+oh, x*scale:x*scale+ow] = \
                sr_patch[:, :, oy:oy+oh, ox:ox+ow]

    return out


def inference(args):
    # Full determinism
    set_random_seed(2026)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = RCAN(scale=args.scale).to(device)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    print(f"Processing {len(files)} images …")

    with torch.no_grad():
        for path in tqdm(files):
            name = os.path.basename(path)
            img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            img_t = _to_tensor(img_bgr)

            try:
                sr = model(img_t.to(device)).cpu()
            except RuntimeError:
                torch.cuda.empty_cache()
                sr = tile_forward(model, img_t, args.tile_size,
                                  args.tile_pad, args.scale, device)

            sr_bgr = _to_img(sr)
            cv2.imwrite(os.path.join(args.output_dir, name), sr_bgr)


if __name__ == "__main__":
    args = parse_args()
    inference(args)

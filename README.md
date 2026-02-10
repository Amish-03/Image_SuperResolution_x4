# NTIRE 2026 — Single Image Super-Resolution (×4)

Deterministic, perception-aware ×4 SR built for the NTIRE 2026 challenge.
Supports **two backbone architectures** — RCAN (CNN-based) and SwinIR (Transformer-based).

## Model Architectures

| | RCAN | SwinIR |
|---|---|---|
| **Type** | CNN + Channel Attention | Swin Transformer |
| **Config** | 10 groups × 20 RCAB, 64 ch | 6 RSTB groups × 6 blocks, 180 dim |
| **Parameters** | 15.6M | 11.9M |
| **Upsampler** | 2 × PixelShuffle(2) + ReLU | 2 × PixelShuffle(2) |
| **Loss** | Charbonnier + 0.01 × VGG-19 Perceptual | Same |
| **Precision** | Mixed (AMP), deterministic | Same |
| **Training script** | `train.py` | `train_swinir.py` |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
pip install timm    # Required for SwinIR
```

### 2. Download & prepare dataset
```bash
python download_data.py     # Downloads DIV2K HR from Kaggle
python prepare_data.py      # Generates LR via bicubic ÷4
```

This creates:
```
data/
  DIV2K_train_HR/           (800 images)
  DIV2K_train_LR_bicubic/X4 (800 images)
  DIV2K_valid_HR/           (100 images)
  DIV2K_valid_LR_bicubic/X4 (100 images)
```

### 3. Train

**Option A — RCAN (CNN backbone):**
```bash
python train.py --dataset_dir ./data --epochs 1000 --batch_size 16 --patch_size 64
```

**Option B — SwinIR (Transformer backbone, recommended):**
```bash
python train_swinir.py --dataset_dir ./data --epochs 500 --batch_size 8 --patch_size 48
```

Key options (both scripts):
- `--resume <checkpoint.pth>` to continue training
- `--val_every 5` validation frequency (default)
- `--lr 2e-4` initial learning rate

SwinIR-specific options:
- `--embed_dim 180` (180 = medium/11.9M, 60 = lightweight/0.9M)
- `--window_size 8` attention window size
- `--depths 6 6 6 6 6 6` RSTB group depths

Best model saved to `checkpoints/best_model.pth` (RCAN) or `checkpoints_swinir/best_model.pth` (SwinIR).

### 4. Inference
```bash
python inference.py \
    --input_dir  ./data/DIV2K_valid_LR_bicubic/X4 \
    --output_dir ./results \
    --model_path ./checkpoints/best_model.pth
```

Outputs lossless PNG files (same filenames as inputs).

---

## Project Structure

| File | Purpose |
|---|---|
| `model.py` | RCAN architecture (deterministic, no dropout) |
| `train.py` | RCAN training loop (AMP, cosine annealing, checkpointing) |
| `SwinIR/` | Cloned SwinIR repository (Swin Transformer model + test sets) |
| `train_swinir.py` | SwinIR training loop (AMP, cosine annealing, window-size padding) |
| `loss.py` | Charbonnier + VGG perceptual composite loss |
| `dataset.py` | DIV2K paired LR/HR loader with augmentation |
| `inference.py` | Deterministic inference with OOM tiling fallback |
| `utils.py` | PSNR / SSIM metrics, image conversion helpers |
| `prepare_data.py` | Generate LR images from HR via bicubic ÷4 |
| `download_data.py` | Download DIV2K from Kaggle |

## Smoke Test Results

A mini training run (0.25M param RCAN, 10 images, 2 epochs, CPU) verified the full pipeline:

| Epoch | Loss (start → end) | Val PSNR | Val SSIM |
|---|---|---|---|
| 1 | 0.3327 → 0.2583 | 11.52 dB | 0.3225 |
| 2 | 0.2214 → 0.1820 | 11.84 dB | 0.3082 |

> **Note:** Low PSNR is expected — tiny model trained for only 2 epochs.

**Checks passed:**
- Data loading (train patches + full valid images)
- Forward/backward pass with AMP
- Validation (PSNR + SSIM, 4px border crop)
- Checkpoint save & reload
- Determinism (identical outputs on repeated inference)

Run the smoke test yourself:
```bash
python test_smoke.py
```

---

## Design Decisions

- **Dual backbones** → RCAN for reliable CNN baseline, SwinIR for stronger transformer-based performance (+0.3-0.5 dB PSNR over RCAN on standard benchmarks)
- **No GAN / adversarial loss** → avoids hallucinated textures
- **Low perceptual weight (0.01)** → slight sharpening without hurting PSNR/SSIM
- **PixelShuffle via 2×(conv + PS(2))** → avoids checkerboard artifacts vs single PS(4)
- **Gradient clipping (0.5)** → training stability with mixed precision
- **4-px border crop** during evaluation matches NTIRE rules

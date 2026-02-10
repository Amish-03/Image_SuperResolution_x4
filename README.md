# NTIRE 2026 — Single Image Super-Resolution (×4)

Deterministic, perception-aware ×4 SR built for the NTIRE 2026 challenge.

| Component | Detail |
|---|---|
| **Backbone** | RCAN (10 groups × 20 RCAB blocks, 64 channels) — 15.6 M params |
| **Upsampler** | 2 × PixelShuffle(2) with ReLU |
| **Loss** | Charbonnier + 0.01 × VGG-19 Perceptual |
| **Precision** | Mixed (AMP), deterministic |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
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
```bash
python train.py --dataset_dir ./data --epochs 1000 --batch_size 16 --patch_size 64
```

Key options:
- `--resume ./checkpoints/best_model.pth` to continue training
- `--val_every 5` validation frequency (default)
- `--lr 1e-4` initial learning rate

The best model is saved to `checkpoints/best_model.pth` based on validation PSNR.

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
| `loss.py` | Charbonnier + VGG perceptual composite loss |
| `dataset.py` | DIV2K paired LR/HR loader with augmentation |
| `train.py` | Training loop (AMP, cosine annealing, checkpointing) |
| `inference.py` | Deterministic inference with OOM tiling fallback |
| `utils.py` | PSNR / SSIM metrics, image conversion helpers |
| `prepare_data.py` | Generate LR images from HR via bicubic ÷4 |
| `download_data.py` | Download DIV2K from Kaggle |

## Smoke Test Results

A mini training run (0.25M param model, 10 images, 2 epochs, CPU) verified the full pipeline:

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

- **No GAN / adversarial loss** → avoids hallucinated textures
- **Low perceptual weight (0.01)** → slight sharpening without hurting PSNR/SSIM
- **PixelShuffle via 2×(conv + PS(2))** → avoids checkerboard artifacts vs single PS(4)
- **Gradient clipping (0.5)** → training stability with mixed precision
- **4-px border crop** during evaluation matches NTIRE rules

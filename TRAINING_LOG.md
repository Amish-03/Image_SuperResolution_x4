# Training Log: SwinIR (x4)

**Date**: 2026-02-10
**Model**: SwinIR (Lightweight configuration)
**Task**: Single Image Super-Resolution (x4)

## Configuration
-   **Epochs**: 40 (Phase 1: 0-20, Phase 2: 20-40)
-   **Batch Size**: 2
-   **Validation Frequency**: Every 15 epochs
-   **Validation Size**: 20 images (random subset)
-   **Dataset**: DIV2K

## Results
### Phase 1 (Epochs 0-20)
-   **Status**: Completed.
-   **Best PSNR**: **25.7175 dB**
-   **Checkpoint**: `checkpoints_swinir/best_model.pth`

### Phase 2 (Epochs 20-40)
-   **Status**: Completed.
-   **Best PSNR**: **25.7175 dB** (No improvement over Phase 1)
-   **Notes**: Model converged early or needs further tuning (e.g., LR decay) to improve.

## Notes
-   Validation was optimized to run on a subset of 20 images every 15 epochs.
-   Training completed without errors.

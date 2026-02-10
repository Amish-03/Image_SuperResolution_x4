"""
Inference script for SwinIR (trained model).
"""

import argparse
import os
import sys
import glob
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add SwinIR submodule to path
SWINIR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SwinIR")
sys.path.insert(0, SWINIR_DIR)

from models.network_swinir import SwinIR
from utils import set_random_seed, tensor2img

def parse_args():
    p = argparse.ArgumentParser(description="SwinIR Inference")
    p.add_argument("--input_dir", type=str, required=True, help="Path to input LR images")
    p.add_argument("--output_dir", type=str, required=True, help="Path to save SR images")
    p.add_argument("--model_path", type=str, required=True, help="Path to best_model.pth")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--limit", type=int, default=5, help="Limit number of images to process")
    
    # SwinIR args - must match training
    p.add_argument("--embed_dim", type=int, default=180)
    p.add_argument("--window_size", type=int, default=8)
    # Check your training command for depths/heads. 
    # train_swinir.py defaults: depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6]
    # But based on the dry run command you used: --embed_dim 30 --depths 2 --num_heads 2
    # Wait, the 20 epoch run used defaults?
    # The user ran: python train_swinir.py --dataset_dir ./data --epochs 20 --batch_size 2
    # So it used defaults: embed_dim=180, depths=[6,6,6,6,6,6], etc.
    p.add_argument("--depths", type=int, nargs="+", default=[6, 6, 6, 6, 6, 6])
    p.add_argument("--num_heads", type=int, nargs="+", default=[6, 6, 6, 6, 6, 6])
    
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print(f"Loading model from {args.model_path}...")
    model = SwinIR(
        upscale=args.scale,
        in_chans=3,
        img_size=48, # Must match training patch_size (48) for attn_mask to load
        window_size=args.window_size,
        img_range=1.,
        depths=args.depths,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    ).to(device)
    
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    
    # 2. Prepare Data
    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    if args.limit > 0:
        files = files[:args.limit]
    
    print(f"Processing {len(files)} images...")
    
    # 3. Inference Loop
    window_size = args.window_size
    scale = args.scale
    
    with torch.no_grad():
        for path in tqdm(files):
             # Read Image
            img_name = os.path.basename(path)
            img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Pad
            _, _, h_old, w_old = img_t.size()
            h_pad = (window_size - h_old % window_size) % window_size
            w_pad = (window_size - w_old % window_size) % window_size
            img_t = torch.nn.functional.pad(img_t, (0, w_pad, 0, h_pad), mode='reflect')
            
            # Forward
            try:
                output = model(img_t)
            except RuntimeError as e:
                print(f"Error processing {img_name}: {e}")
                continue
                
            # Crop
            output = output[:, :, :h_old * scale, :w_old * scale]
            
            # Save
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0)) # RGB -> BGR
            output = (output * 255.0).round().astype(np.uint8)
            
            cv2.imwrite(os.path.join(args.output_dir, img_name), output)
            
    print("Done.")

if __name__ == "__main__":
    main()

"""
sam_baseline.py
==================
SAM 2 Appearance-Only Baseline for the PixelDynamics pipeline.

Generates unsupervised segmentation masks using only RGB cues (no motion/flow).
Masks are sorted by area and saved using the standard DAVIS color palette.
"""
import os
import sys
import glob
import cv2
import argparse
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Import Aybek's save_mask function for the correct DAVIS palette
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from region_growing import save_mask


def list_frames(seq_dir):
    return sorted(glob.glob(os.path.join(seq_dir, "*.jpg")) + glob.glob(os.path.join(seq_dir, "*.png")))


def resolve_sequences(arg: str, davis_root: str):
    img_root = os.path.join(davis_root, "JPEGImages", "480p")
    if arg in ("all",): return sorted(os.listdir(img_root))
    if arg in ("val", "train"):
        split_file = os.path.join(davis_root, "ImageSets", "2017", f"{arg}.txt")
        with open(split_file) as f: return [ln.strip() for ln in f if ln.strip()]
    if os.path.isfile(arg):
        with open(arg) as f: return [ln.strip() for ln in f if ln.strip()]
    return [s.strip() for s in arg.split(",") if s.strip()]


def get_args():
    p = argparse.ArgumentParser(description="SAM 2 Appearance Baseline for DAVIS.")
    default_davis = os.path.join(_HERE, "..", "data", "DAVIS")
    default_out = os.path.join(_HERE, "..", "results", "sam")
    
    p.add_argument("--davis-root", default=default_davis, help="DAVIS root dir")
    p.add_argument("--sequences", default="bear", help="'all' | 'val' | 'train' | comma list")
    p.add_argument("--output", default=default_out, help="mask output root")
    p.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_small.pt", help="SAM 2 weights")
    p.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_s.yaml", help="SAM 2 config")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sam_baseline] Device: {device}")

    if not os.path.exists(args.checkpoint):
        print(f"[sam_baseline] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    sam2_model = build_sam2(args.config, args.checkpoint, device=device)

    # Turbo Boost: Reduce the grid points from 1024 to 256 (16x16)
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=16
    )

    sequences = resolve_sequences(args.sequences, args.davis_root)
    img_root = os.path.join(args.davis_root, "JPEGImages", "480p")
    os.makedirs(args.output, exist_ok=True)

    print(f"[sam_baseline] Processing {len(sequences)} sequence(s)...")

    for si, seq in enumerate(sequences, 1):
        seq_dir = os.path.join(img_root, seq)
        out_dir = os.path.join(args.output, seq)
        os.makedirs(out_dir, exist_ok=True)
        frames = list_frames(seq_dir)

        # Smart Resume: Check if the folder is already fully populated
        existing_pngs = glob.glob(os.path.join(out_dir, "*.png"))
        if len(existing_pngs) >= len(frames):
            print(f"[{si}/{len(sequences)}] Skipping {seq} - Already complete!")
            continue

        print(f"[{si}/{len(sequences)}] {seq}")
        for frame_path in frames:
            stem = os.path.splitext(os.path.basename(frame_path))[0]
            save_path = os.path.join(out_dir, f"{stem}.png")

            if os.path.exists(save_path):
                continue

            img = cv2.imread(frame_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Float16 mixed precision for inference speed
            with torch.autocast(device.type, dtype=torch.float16):
                masks = mask_generator.generate(img_rgb)

            h, w = img.shape[:2]
            label_map = np.zeros((h, w), dtype=np.uint8)
            
            # Sort masks by area descending so foreground overwrites background
            masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

            for i, mask_dict in enumerate(masks):
                instance_id = (i % 255) + 1
                label_map[mask_dict['segmentation']] = instance_id

            save_mask(label_map, save_path)

    print("[sam_baseline] done.")


if __name__ == "__main__":
    main()
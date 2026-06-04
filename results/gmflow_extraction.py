"""
gmflow_extraction.py
==================
GMFlow optical-flow extraction for the PixelDynamics motion-segmentation pipeline.

Runs GMFlow on consecutive frames of DAVIS sequences and saves the dense flow
field for every frame pair to disk.

Output contract (shared with the team):
    data/flow/gmflow/<sequence>/<frame>.npy   # shape (H, W, 2), float32
"""
import argparse
import glob
import os
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F

# Make GMFlow importable assuming it's cloned in the project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_GMFLOW_CORE = os.path.join(_HERE, "..", "gmflow")
sys.path.insert(0, _GMFLOW_CORE)

try:
    from gmflow.gmflow import GMFlow
except ImportError as e:
    print(f"[flow_extraction] Could not import GMFlow ({e}).")
    print("Ensure you ran: git clone https://github.com/haofeixu/gmflow.git in the root dir.")
    sys.exit(1)


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")


def load_gmflow(model_path: str, device: torch.device) -> torch.nn.Module:
    model = GMFlow(
        feature_channels=128, num_scales=1, upsample_factor=8,
        num_head=1, attention_type='swin', ffn_dim_expansion=4, num_transformer_layers=6
    )
    state_dict = torch.load(model_path, map_location="cpu")
    weights = state_dict['model'] if 'model' in state_dict else state_dict
    weights = {k.replace('module.', ''): v for k, v in weights.items()}
    model.load_state_dict(weights, strict=False)
    model.to(device)
    model.eval()
    return model


def load_image(path: str, device: torch.device) -> torch.Tensor:
    import cv2
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    return tensor[None].to(device)


def list_frames(seq_dir: str):
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


@torch.no_grad()
def extract_sequence(model, seq_dir, out_dir, device):
    os.makedirs(out_dir, exist_ok=True)
    frames = list_frames(seq_dir)
    if len(frames) < 2: return

    # Smart Resume: Skip completed sequences
    existing_npy = glob.glob(os.path.join(out_dir, "*.npy"))
    if len(existing_npy) >= len(frames):
        print(f"  -> Skipping: Already completed ({len(existing_npy)} frames)")
        return

    last_flow = None
    for i in range(len(frames) - 1):
        image1 = load_image(frames[i], device)
        image2 = load_image(frames[i + 1], device)

        # Dimension correction for GMFlow Attention (Multiples of 16)
        _, _, h, w = image1.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            image1 = F.pad(image1, [0, pad_w, 0, pad_h], mode='replicate')
            image2 = F.pad(image2, [0, pad_w, 0, pad_h], mode='replicate')

        results_dict = model(image1, image2, attn_splits_list=[2], corr_radius_list=[-1], prop_radius_list=[-1], pred_bidir_flow=False)
        flow_up = results_dict['flow_preds'][-1]
        
        if pad_h > 0 or pad_w > 0:
            flow_up = flow_up[:, :, :h, :w]
            
        flow = flow_up[0].permute(1, 2, 0).cpu().numpy()
        stem = os.path.splitext(os.path.basename(frames[i]))[0]
        np.save(os.path.join(out_dir, f"{stem}.npy"), flow.astype(np.float32))
        last_flow = flow

    if last_flow is not None:
        stem = os.path.splitext(os.path.basename(frames[-1]))[0]
        np.save(os.path.join(out_dir, f"{stem}.npy"), last_flow.astype(np.float32))


def get_args():
    p = argparse.ArgumentParser(description="GMFlow extraction for DAVIS.")
    default_model = os.path.join(_HERE, "..", "gmflow", "pretrained", "gmflow-scale1-things.pth")
    default_davis = os.path.join(_HERE, "..", "data", "DAVIS")
    default_out = os.path.join(_HERE, "..", "data", "flow", "gmflow")
    
    p.add_argument("--model", default=default_model, help="GMFlow checkpoint (.pth)")
    p.add_argument("--davis-root", default=default_davis, help="DAVIS root dir")
    p.add_argument("--sequences", default="bear", help="'all' | 'val' | 'train' | comma list")
    p.add_argument("--output", default=default_out, help="flow output root")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    return p.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.model):
        print(f"[flow_extraction] Checkpoint not found: {args.model}")
        print("Please download the pretrained weights into gmflow/pretrained/")
        sys.exit(1)

    device = pick_device(args.device)
    print(f"[flow_extraction] device = {device}")

    model = load_gmflow(args.model, device)
    print(f"[flow_extraction] loaded GMFlow from {os.path.basename(args.model)}")

    sequences = resolve_sequences(args.sequences, args.davis_root)
    print(f"[flow_extraction] Extracting {len(sequences)} sequence(s)...")

    img_root = os.path.join(args.davis_root, "JPEGImages", "480p")
    for si, seq in enumerate(sequences, 1):
        print(f"[{si}/{len(sequences)}] {seq}")
        seq_dir = os.path.join(img_root, seq)
        out_dir = os.path.join(args.output, seq)
        extract_sequence(model, seq_dir, out_dir, device)

    print("[flow_extraction] done.")


if __name__ == "__main__":
    main()
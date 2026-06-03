"""
FlowFormer inference over the full DAVIS dataset.

Follows the shared team flow contract (see HANDOFF.md):

  data/flow/flowformer/<sequence>/<frame>.npy   -- (H, W, 2) float32, (u,v) px/frame
  outputs/flow_viz/flowformer/<sequence>/<frame>.png  -- optional color-wheel viz

The last frame of each sequence has no successor; its .npy is a copy of the
previous frame's flow so that frame counts stay aligned with DAVIS annotations.

Usage
-----
py -3.10 scripts/run_flowformer.py
py -3.10 scripts/run_flowformer.py --checkpoint sintel --split val
py -3.10 scripts/run_flowformer.py --sequences bear blackswan
py -3.10 scripts/run_flowformer.py --no-viz
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR    = Path(__file__).parent.resolve()
PROJECT_ROOT  = SCRIPT_DIR.parent
FF_DIR        = PROJECT_ROOT / "models" / "FlowFormer"
DAVIS_DIR     = PROJECT_ROOT / "data" / "DAVIS"
IMAGES_DIR    = DAVIS_DIR / "JPEGImages" / "480p"
IMAGESETS_DIR = DAVIS_DIR / "ImageSets" / "2017"
FLOW_ROOT     = PROJECT_ROOT / "data" / "flow" / "flowformer"  
VIZ_ROOT      = PROJECT_ROOT / "outputs" / "flow_viz" / "flowformer"

sys.path.insert(0, str(FF_DIR))
sys.path.insert(0, str(FF_DIR / "core"))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint: str, device: torch.device) -> torch.nn.Module:
    from configs.things_eval import get_cfg
    from core.FlowFormer import build_flowformer

    ckpt_path = FF_DIR / "checkpoints" / f"{checkpoint}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = get_cfg()
    model = build_flowformer(cfg)

    state = torch.load(str(ckpt_path), map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def read_image(path: Path, device: torch.device) -> torch.Tensor:
    """Return (1, 3, H, W) float32 tensor in [0, 255] on device."""
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Cannot read: {path}")
    t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    return t.to(device)


class InputPadder:
    """Pad so H and W are divisible by 8 (mirrors FlowFormer's own padder)."""
    def __init__(self, dims):
        h, w = dims[-2], dims[-1]
        pad_h = (((h // 8) + 1) * 8 - h) % 8
        pad_w = (((w // 8) + 1) * 8 - w) % 8
        # sintel-style: split evenly on both sides
        self._pad = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]

    def pad(self, *tensors):
        import torch.nn.functional as F
        return [F.pad(t, self._pad, mode="replicate") for t in tensors]

    def unpad(self, t: torch.Tensor) -> torch.Tensor:
        h, w = t.shape[-2], t.shape[-1]
        pl, pr, pt, pb = self._pad
        return t[..., pt: h - pb if pb else h, pl: w - pr if pr else w]


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    """Convert (H, W, 2) float32 flow to uint8 RGB image (color-wheel)."""
    u, v = flow[..., 0], flow[..., 1]          # (H, W) each
    rad = np.sqrt(u ** 2 + v ** 2)             # (H, W)
    max_rad = max(float(rad.max()), 1e-5)

    colorwheel = _make_colorwheel()             # (55, 3)
    angle = np.arctan2(-v / (max_rad + 1e-10),
                       -u / (max_rad + 1e-10)) / np.pi   # (H, W)
    fk = (angle + 1) / 2 * (len(colorwheel) - 1)        # (H, W)
    k0 = fk.astype(np.int32)                             # (H, W)
    k1 = (k0 + 1) % len(colorwheel)                     # (H, W)
    f  = fk - k0                                         # (H, W)

    img = np.zeros((*u.shape, 3), dtype=np.uint8)
    for c in range(3):
        col0 = colorwheel[k0, c].astype(np.float32) / 255.0   # (H, W)
        col1 = colorwheel[k1, c].astype(np.float32) / 255.0   # (H, W)
        col  = (1 - f) * col0 + f * col1                      # (H, W)
        col  = 1 - rad / max_rad * (1 - col)                  # (H, W)
        img[..., c] = (col * 255).clip(0, 255).astype(np.uint8)

    return img


def _make_colorwheel(ncols: int = 55) -> np.ndarray:
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    assert ncols <= RY + YG + GC + CB + BM + MR
    wheel = np.zeros((RY + YG + GC + CB + BM + MR, 3), dtype=np.uint8)
    col = 0
    wheel[:RY, 0] = 255
    wheel[:RY, 1] = (255 * np.arange(RY) / RY).astype(np.uint8)
    col += RY
    wheel[col:col+YG, 0] = (255 - 255 * np.arange(YG) / YG).astype(np.uint8)
    wheel[col:col+YG, 1] = 255
    col += YG
    wheel[col:col+GC, 1] = 255
    wheel[col:col+GC, 2] = (255 * np.arange(GC) / GC).astype(np.uint8)
    col += GC
    wheel[col:col+CB, 1] = (255 - 255 * np.arange(CB) / CB).astype(np.uint8)
    wheel[col:col+CB, 2] = 255
    col += CB
    wheel[col:col+BM, 2] = 255
    wheel[col:col+BM, 0] = (255 * np.arange(BM) / BM).astype(np.uint8)
    col += BM
    wheel[col:col+MR, 2] = (255 - 255 * np.arange(MR) / MR).astype(np.uint8)
    wheel[col:col+MR, 0] = 255
    return wheel[:ncols]


# ---------------------------------------------------------------------------
# Sequence listing
# ---------------------------------------------------------------------------

def get_sequences(split: str) -> list:
    if split == "all":
        return sorted(p.name for p in IMAGES_DIR.iterdir() if p.is_dir())
    txt = IMAGESETS_DIR / f"{split}.txt"
    if not txt.exists():
        raise FileNotFoundError(f"Split file not found: {txt}")
    return [l.strip() for l in txt.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Per-sequence inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_sequence(model, seq: str, device, save_viz: bool, overwrite: bool) -> dict:
    seq_dir = IMAGES_DIR / seq
    frames = sorted(seq_dir.glob("*.jpg")) + sorted(seq_dir.glob("*.png"))
    frames = sorted(set(frames))

    if len(frames) < 2:
        return {"seq": seq, "done": 0, "skipped": 0, "error": "< 2 frames"}

    flow_dir = FLOW_ROOT / seq
    viz_dir  = VIZ_ROOT  / seq
    flow_dir.mkdir(parents=True, exist_ok=True)
    if save_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = 0
    last_flow_np = None

    for i in range(len(frames) - 1):
        stem = frames[i].stem
        npy_path = flow_dir / f"{stem}.npy"

        if not overwrite and npy_path.exists():
            skipped += 1
            # still need last_flow_np for the final-frame copy below
            last_flow_np = np.load(str(npy_path))
            continue

        img1 = read_image(frames[i],     device)
        img2 = read_image(frames[i + 1], device)

        padder = InputPadder(img1.shape)
        img1p, img2p = padder.pad(img1, img2)

        out = model(img1p, img2p)
        flow_up  = padder.unpad(out[0])[0]                   # [2, H, W]
        flow_np  = flow_up.permute(1, 2, 0).cpu().numpy()    # [H, W, 2]

        np.save(str(npy_path), flow_np)
        last_flow_np = flow_np

        if save_viz:
            viz_bgr = cv2.cvtColor(flow_to_color(flow_np), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(viz_dir / f"{stem}.png"), viz_bgr)

        done += 1

    # Copy last computed flow to the final frame (no successor) so that
    # frame counts stay aligned with DAVIS annotations.
    if last_flow_np is not None:
        last_stem = frames[-1].stem
        last_npy  = flow_dir / f"{last_stem}.npy"
        if overwrite or not last_npy.exists():
            np.save(str(last_npy), last_flow_np)
            done += 1
        else:
            skipped += 1

    return {"seq": seq, "done": done, "skipped": skipped, "error": None}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="FlowFormer inference on DAVIS")
    p.add_argument("--checkpoint", default="things",
                   choices=["things", "sintel", "kitti", "things_kitti"],
                   help="Checkpoint to use (default: things)")
    p.add_argument("--split", default="all",
                   choices=["all", "train", "val"],
                   help="DAVIS split (default: all)")
    p.add_argument("--sequences", nargs="*", default=None,
                   help="Process only these named sequences (overrides --split)")
    p.add_argument("--no-viz", action="store_true",
                   help="Skip saving color-wheel visualisations")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute flows even if .npy already exists")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU (very slow, for debugging only)")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[FlowFormer] device     : {device}")
    print(f"[FlowFormer] checkpoint : {args.checkpoint}")

    print("[FlowFormer] Loading model...")
    model = load_model(args.checkpoint, device)
    print("[FlowFormer] Model ready.")

    sequences = args.sequences if args.sequences else get_sequences(args.split)
    print(f"[FlowFormer] Sequences  : {len(sequences)}")

    results = []
    for seq in tqdm(sequences, desc="sequences", unit="seq"):
        try:
            r = run_sequence(model, seq, device,
                             save_viz=not args.no_viz,
                             overwrite=args.overwrite)
        except Exception as e:
            r = {"seq": seq, "done": 0, "skipped": 0, "error": str(e)}
            tqdm.write(f"  ERROR [{seq}]: {e}")
        results.append(r)

    total_done    = sum(r["done"]    for r in results)
    total_skipped = sum(r["skipped"] for r in results)
    errors        = [r for r in results if r["error"]]

    print(f"\n[FlowFormer] Finished.")
    print(f"  Sequences : {len(sequences)}")
    print(f"  Pairs done: {total_done}")
    print(f"  Pairs skip: {total_skipped} (already existed)")
    print(f"  Errors    : {len(errors)}")
    for r in errors:
        print(f"    {r['seq']}: {r['error']}")

    print(f"\nFlows -> {FLOW_ROOT}")
    if not args.no_viz:
        print(f"Viz   -> {VIZ_ROOT}")


if __name__ == "__main__":
    main()

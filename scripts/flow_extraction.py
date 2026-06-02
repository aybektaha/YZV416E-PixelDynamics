"""
flow_extraction.py
==================
RAFT optical-flow extraction for the PixelDynamics motion-segmentation pipeline.

Runs RAFT on consecutive frames of DAVIS sequences and saves the dense flow
field for every frame pair to disk so that the (CPU-only) region-growing and
evaluation stages can consume it without re-running the network.

Device is selected automatically: CUDA -> MPS (Apple Silicon) -> CPU.

Output contract (shared with the rest of the team):
    data/flow/<backbone>/<sequence>/<frame>.npy   # shape (H, W, 2), float32
    where <frame>.npy is the flow from frame <frame> to <frame>+1.
    The last frame of a sequence has no successor, so its flow is copied from
    the previous frame to keep frame counts aligned with the annotations.

Example:
    python scripts/flow_extraction.py --sequences bear
    python scripts/flow_extraction.py --sequences val          # all 30 val seqs
    python scripts/flow_extraction.py --sequences bear,dog --viz
"""
import argparse
import glob
import os
import sys
from argparse import Namespace

import numpy as np
import torch

# Make RAFT importable (its modules assume `core/` is on the path).
_HERE = os.path.dirname(os.path.abspath(__file__))
_RAFT_CORE = os.path.join(_HERE, "..", "models", "RAFT", "core")
sys.path.insert(0, _RAFT_CORE)

try:
    from raft import RAFT
    from utils.utils import InputPadder
    from utils import flow_viz
except ImportError as e:
    print(f"[flow_extraction] Could not import RAFT ({e}).")
    print("Run scripts/setup_raft.sh first to clone RAFT and download checkpoints.")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Device handling
# --------------------------------------------------------------------------- #
def pick_device(requested: str = "auto") -> torch.device:
    """Return the best available torch device.

    On Apple Silicon (M-series) this selects 'mps'. Some RAFT ops may not yet
    have an MPS kernel; set PYTORCH_ENABLE_MPS_FALLBACK=1 so they fall back to
    CPU transparently instead of crashing.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_raft(model_path: str, device: torch.device, small: bool = False) -> torch.nn.Module:
    """Build RAFT and load a checkpoint trained with DataParallel.

    The released checkpoints store weights under a 'module.' prefix; we strip it
    and load into the bare model so we don't need DataParallel on CPU/MPS.
    """
    model_args = Namespace(
        small=small,
        mixed_precision=False,
        alternate_corr=False,
        dropout=0,
    )
    model = RAFT(model_args)

    state_dict = torch.load(model_path, map_location="cpu")
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def load_image(path: str, device: torch.device) -> torch.Tensor:
    """Load an RGB image as a (1, 3, H, W) float tensor on `device`."""
    import cv2

    img = cv2.imread(path)                       # BGR, HxWx3, uint8
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    return tensor[None].to(device)


def list_frames(seq_dir: str):
    frames = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")) +
                    glob.glob(os.path.join(seq_dir, "*.png")))
    return frames


def resolve_sequences(arg: str, davis_root: str):
    """Turn --sequences into a concrete list of sequence names.

    Accepts: 'all', 'val', 'train', a path to a .txt file, or a comma list.
    """
    img_root = os.path.join(davis_root, "JPEGImages", "480p")
    if arg in ("all",):
        return sorted(os.listdir(img_root))
    if arg in ("val", "train"):
        split_file = os.path.join(davis_root, "ImageSets", "2017", f"{arg}.txt")
        with open(split_file) as f:
            return [ln.strip() for ln in f if ln.strip()]
    if os.path.isfile(arg):
        with open(arg) as f:
            return [ln.strip() for ln in f if ln.strip()]
    return [s.strip() for s in arg.split(",") if s.strip()]


# --------------------------------------------------------------------------- #
# Core extraction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_sequence(model, seq_dir, out_dir, device, iters=20, viz=False):
    os.makedirs(out_dir, exist_ok=True)
    frames = list_frames(seq_dir)
    if len(frames) < 2:
        print(f"  [skip] {seq_dir} has < 2 frames")
        return

    last_flow = None
    for i in range(len(frames) - 1):
        image1 = load_image(frames[i], device)
        image2 = load_image(frames[i + 1], device)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        _, flow_up = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_up[0]).permute(1, 2, 0).cpu().numpy()   # (H, W, 2)

        stem = os.path.splitext(os.path.basename(frames[i]))[0]
        np.save(os.path.join(out_dir, f"{stem}.npy"), flow.astype(np.float32))
        last_flow = flow

        if viz:
            _save_viz(flow, os.path.join(out_dir, f"{stem}_viz.png"))

    # Duplicate the last available flow for the final frame (no successor).
    if last_flow is not None:
        stem = os.path.splitext(os.path.basename(frames[-1]))[0]
        np.save(os.path.join(out_dir, f"{stem}.npy"), last_flow.astype(np.float32))


def _save_viz(flow, path):
    import cv2

    rgb = flow_viz.flow_to_image(flow)           # HxWx3 uint8
    cv2.imwrite(path, rgb[:, :, [2, 1, 0]])


def get_args():
    p = argparse.ArgumentParser(description="RAFT flow extraction for DAVIS.")
    default_model = os.path.join(_HERE, "..", "models", "RAFT", "models", "raft-things.pth")
    default_davis = os.path.join(_HERE, "..", "data", "DAVIS")
    default_out = os.path.join(_HERE, "..", "data", "flow", "raft")
    p.add_argument("--model", default=default_model, help="RAFT checkpoint (.pth)")
    p.add_argument("--davis-root", default=default_davis, help="DAVIS root dir")
    p.add_argument("--sequences", default="bear",
                   help="'all' | 'val' | 'train' | a .txt path | comma list")
    p.add_argument("--output", default=default_out, help="flow output root")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--iters", type=int, default=20, help="RAFT refinement iterations")
    p.add_argument("--small", action="store_true", help="use the small RAFT model")
    p.add_argument("--viz", action="store_true", help="also save flow color images")
    return p.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.model):
        print(f"[flow_extraction] Checkpoint not found: {args.model}")
        print("Run scripts/setup_raft.sh to download RAFT checkpoints.")
        sys.exit(1)

    device = pick_device(args.device)
    print(f"[flow_extraction] device = {device}")

    model = load_raft(args.model, device, small=args.small)
    print(f"[flow_extraction] loaded RAFT from {os.path.basename(args.model)}")

    sequences = resolve_sequences(args.sequences, args.davis_root)
    print(f"[flow_extraction] {len(sequences)} sequence(s): {', '.join(sequences[:6])}"
          + (" ..." if len(sequences) > 6 else ""))

    img_root = os.path.join(args.davis_root, "JPEGImages", "480p")
    for si, seq in enumerate(sequences, 1):
        seq_dir = os.path.join(img_root, seq)
        out_dir = os.path.join(args.output, seq)
        print(f"[{si}/{len(sequences)}] {seq}")
        extract_sequence(model, seq_dir, out_dir, device, iters=args.iters, viz=args.viz)

    print("[flow_extraction] done.")


if __name__ == "__main__":
    main()

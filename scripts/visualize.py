"""
visualize.py
============
Build a side-by-side panel for one frame: RGB | optical flow (colour-coded) |
our predicted mask overlay | ground-truth mask overlay. Handy for the report
figures and the demo video.

Example:
    python scripts/visualize.py --sequence bear --frame 00030
    python scripts/visualize.py --sequence bear --frame 00030 --pred results/raft_rg
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "models", "RAFT", "core"))
from utils import flow_viz  # noqa: E402  (RAFT's flow-to-RGB colour wheel)


def overlay(rgb, label, alpha=0.5):
    """Blend a colour-per-instance overlay onto the RGB frame."""
    rgb = rgb.astype(np.float32)
    out = rgb.copy()
    colours = np.array([[255, 60, 60], [60, 160, 255], [60, 220, 120],
                        [240, 220, 40], [200, 80, 240], [255, 140, 0]], np.float32)
    for i, inst in enumerate([v for v in np.unique(label) if v != 0]):
        m = label == inst
        c = colours[i % len(colours)]
        out[m] = (1 - alpha) * rgb[m] + alpha * c
    return out.astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description="Visualize flow + masks for one frame.")
    p.add_argument("--sequence", default="bear")
    p.add_argument("--frame", default="00030", help="frame stem, e.g. 00030")
    p.add_argument("--davis-root", default=os.path.join(HERE, "..", "data", "DAVIS"))
    p.add_argument("--flow-root", default=os.path.join(HERE, "..", "data", "flow", "raft"))
    p.add_argument("--pred", default=os.path.join(HERE, "..", "results", "raft_rg"))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    seq, fr = args.sequence, args.frame
    rgb = np.array(Image.open(os.path.join(
        args.davis_root, "JPEGImages", "480p", seq, f"{fr}.jpg")).convert("RGB"))
    flow = np.load(os.path.join(args.flow_root, seq, f"{fr}.npy"))
    flow_rgb = flow_viz.flow_to_image(flow)

    pred = np.array(Image.open(os.path.join(args.pred, seq, f"{fr}.png")))
    gt = np.array(Image.open(os.path.join(
        args.davis_root, "Annotations_unsupervised", "480p", seq, f"{fr}.png")))

    panels = [
        ("Input frame", rgb),
        ("Optical flow (RAFT)", flow_rgb),
        ("Our prediction (region growing)", overlay(rgb, pred)),
        ("Ground truth", overlay(rgb, gt)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(22, 4))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.suptitle(f"{seq} / frame {fr}", fontsize=14)
    plt.tight_layout()

    out = args.out or os.path.join(HERE, "..", "results", f"viz_{seq}_{fr}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[visualize] wrote {out}")


if __name__ == "__main__":
    main()

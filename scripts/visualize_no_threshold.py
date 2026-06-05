"""
visualize_no_threshold.py
=========================
Honestly visualises what happens when the magnitude threshold is removed from
the region-growing pipeline.

Theory says: flow vectors differ between bear (~1 px) and background (~5 px),
so region growing should separate them without a threshold.

Practice shows: FlowFormer produces spatially smooth flow fields. Boundary
pixels transition gradually between the two motion groups, so adjacent pixel
pairs always satisfy ||f_a - f_b|| < tau — chaining bear and background into
one giant component.

Figure layout (2 rows):
  Row 1 — context:
    Input  |  Raw flow  |  Compensated flow  |  Local flow-diff map
  Row 2 — segmentation results:
    Standard (thr+comp)  |  Threshold, no comp  |  No threshold (all pixels)  |  No threshold + comp

The "local flow-diff map" shows ||f(x,y) - f(x+1,y)|| — the per-pixel cost
that the edge-building step uses. Where it is small everywhere at boundaries,
nothing prevents bear and background from being chained together.

Usage:
    python scripts/visualize_no_threshold.py
    python scripts/visualize_no_threshold.py --frame 00030
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "models", "FlowFormer", "core", "utils"))

from flow_viz import flow_to_image                               
from region_growing import (                                       
    RegionGrowingConfig, segment_frame,
    _gaussian_blur_flow, _build_edges,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_COLOURS = np.array([
    [255,  60,  60],
    [ 60, 160, 255],
    [ 60, 220, 120],
    [240, 220,  40],
    [200,  80, 240],
    [255, 140,   0],
], np.float32)


def _overlay(rgb, labels, alpha=0.5):
    out = rgb.astype(np.float32)
    for i, inst in enumerate(v for v in np.unique(labels) if v != 0):
        m = labels == inst
        c = _COLOURS[i % len(_COLOURS)]
        out[m] = (1 - alpha) * out[m] + alpha * c
    return out.astype(np.uint8)


def _flow_legend(ax):
    """Inset colour wheel (direction) + magnitude stats on a flow axes."""
    # ── colour wheel ──────────────────────────────────────────────────────────
    size = 80
    yi, xi = np.mgrid[-size // 2:size // 2 + 1,
                      -size // 2:size // 2 + 1].astype(np.float32)
    r = np.sqrt(xi ** 2 + yi ** 2)
    wheel_img = flow_to_image(np.stack([xi, yi], axis=-1))
    wheel_img[r > size // 2] = 255                   # white outside circle

    ins = ax.inset_axes([0.70, 0.00, 0.30, 0.30])   # bottom-right corner
    ins.imshow(wheel_img)
    ins.axis("off")

    # Cardinal-direction labels around the wheel
    for txt, xy in [("→", (1.15, 0.50)),
                    ("←", (-0.15, 0.50)),
                    ("↑", (0.50, -0.10)),
                    ("↓", (0.50, 1.10))]:
        ins.text(*xy, txt, transform=ins.transAxes,
                 ha="center", va="center", fontsize=7, color="dimgray")



def _local_flow_diff(flow):
    """Per-pixel rightward flow-vector distance: ||f(x,y) - f(x+1,y)||.
    Shows where adjacent pixels are similar (dark) or different (bright).
    """
    diff = flow[:, :-1] - flow[:, 1:]           # horizontal neighbour diff
    return np.sqrt((diff ** 2).sum(-1))          # (H, W-1)


def _segment_no_threshold(flow_raw, cfg, compensate):
    """Region growing on ALL pixels. Returns (label_img, n_components, top3_sizes)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    flow = flow_raw.astype(np.float32)
    H, W = flow.shape[:2]

    if compensate:
        flow = flow - np.median(flow.reshape(-1, 2), axis=0)
    if cfg.smooth_sigma > 0:
        flow = _gaussian_blur_flow(flow, cfg.smooth_sigma)

    fg = np.ones((H, W), dtype=bool)            # every pixel participates
    rows, cols = _build_edges(flow, fg, cfg)

    n = H * W
    if len(rows):
        graph = coo_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(n, n))
    else:
        graph = coo_matrix((n, n))

    n_comp, comp = connected_components(graph, directed=False)
    comp = comp.reshape(H, W)
    counts = np.bincount(comp.ravel())
    top3 = sorted(counts, reverse=True)[:3]

    # Colour every component above min_area (including background) so the
    # user can see exactly how many regions there are and how large they are.
    labels = np.zeros((H, W), np.uint8)
    next_id = 1
    for c_id in np.argsort(counts)[::-1]:
        if counts[c_id] < cfg.min_area:
            continue
        labels[comp == c_id] = next_id
        next_id += 1
        if next_id > 255:
            break

    return labels, n_comp, top3


# --------------------------------------------------------------------------- #
# Per-frame figure
# --------------------------------------------------------------------------- #
def _save_frame(rgb, flow_raw, seq, stem, flow_model, out_path):
    cfg = RegionGrowingConfig()
    H, W = flow_raw.shape[:2]
    median_vec = np.median(flow_raw.reshape(-1, 2), axis=0)
    flow_comp  = flow_raw.astype(np.float32) - median_vec

    # Smooth flows for the diff map (same as what the algorithm sees)
    f_raw_s  = _gaussian_blur_flow(flow_raw.astype(np.float32), cfg.smooth_sigma)
    f_comp_s = _gaussian_blur_flow(flow_comp, cfg.smooth_sigma)
    diff_raw  = _local_flow_diff(f_raw_s)
    diff_comp = _local_flow_diff(f_comp_s)
    # Use a shared scale so both diff maps are comparable
    diff_vmax = max(diff_raw.max(), diff_comp.max())

    # Segmentation results
    seg_std      = segment_frame(flow_raw, cfg)                              # standard
    seg_thr_only = segment_frame(flow_raw, RegionGrowingConfig(              # thr, no comp
                       compensate_camera=False))
    seg_no_thr,   n1, top3_1 = _segment_no_threshold(flow_raw, cfg, compensate=False)
    seg_no_thr_c, n2, top3_2 = _segment_no_threshold(flow_raw, cfg, compensate=True)

    fig = plt.figure(figsize=(28, 11))
    fig.suptitle(
        f"Region Growing Without Magnitude Threshold  —  {seq}_{stem}  |  {flow_model}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ── Row 1: context ────────────────────────────────────────────────────────
    # Input
    ax = fig.add_subplot(2, 4, 1)
    ax.imshow(rgb); ax.set_title("Input", fontsize=11); ax.axis("off")

    # Raw flow
    ax = fig.add_subplot(2, 4, 2)
    ax.imshow(flow_to_image(flow_raw.astype(np.float32)))
    ax.set_title("Raw flow", fontsize=11); ax.axis("off")
    _flow_legend(ax)

    # Compensated flow
    ax = fig.add_subplot(2, 4, 3)
    ax.imshow(flow_to_image(flow_comp))
    ax.set_title(
        f"Compensated flow\n(median subtracted: dx={median_vec[0]:.1f}, dy={median_vec[1]:.1f} px)",
        fontsize=9); ax.axis("off")
    _flow_legend(ax)

    # Local flow-diff map (shows boundary smoothness)
    ax = fig.add_subplot(2, 4, 4)
    im = ax.imshow(diff_comp, cmap="hot", vmin=0, vmax=diff_vmax)
    ax.set_title(
        f"Local flow difference (px)\nbetween adjacent pixels  [tau = {cfg.tau}]",
        fontsize=9)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, aspect=22)
    cbar.set_label("px", fontsize=8)
    cbar.ax.axhline(cfg.tau / diff_vmax, color="cyan", linewidth=2, linestyle="--")
    cbar.ax.text(-0.05, cfg.tau / diff_vmax, f"{cfg.tau}",
                 va="center", ha="right", fontsize=8, color="cyan",
                 transform=cbar.ax.transAxes)
    cbar.ax.text(1.05, cfg.tau / diff_vmax, "← tau",
                 va="center", ha="left", fontsize=7, color="dimgray",
                 transform=cbar.ax.transAxes)

    # ── Row 2: segmentation variants ──────────────────────────────────────────
    def _n_above(top3):
        return sum(1 for s in top3 if s >= cfg.min_area)

    variants = [
        ("Standard\n(threshold + comp)\n✓ Works",
         _overlay(rgb, seg_std)),
        ("Threshold,  no comp\n✗ Fails — bear below threshold",
         _overlay(rgb, seg_thr_only)),
        (f"No threshold,  no comp\n"
         f"{n1} component(s) found\n"
         f"top-3 sizes: {top3_1[0]:,} / {top3_1[1] if len(top3_1)>1 else 0} / {top3_1[2] if len(top3_1)>2 else 0}",
         _overlay(rgb, seg_no_thr)),
        (f"No threshold  +  comp\n"
         f"{n2} component(s) found\n"
         f"top-3 sizes: {top3_2[0]:,} / {top3_2[1] if len(top3_2)>1 else 0} / {top3_2[2] if len(top3_2)>2 else 0}",
         _overlay(rgb, seg_no_thr_c)),
    ]

    for i, (label, img) in enumerate(variants):
        ax = fig.add_subplot(2, 4, 5 + i)
        ax.imshow(img)
        ax.set_title(label, fontsize=9, pad=4, linespacing=1.5)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Visualise what happens when the magnitude threshold is removed."
    )
    p.add_argument("--sequence",   default="bear")
    p.add_argument("--flow-model", default="flowformer")
    p.add_argument("--frame",      default=None, help="single frame stem, e.g. 00030")
    p.add_argument("--davis-root", default=os.path.join(ROOT, "data", "DAVIS"))
    p.add_argument("--out-root",   default=None)
    args = p.parse_args()

    seq        = args.sequence
    flow_model = args.flow_model
    out_root   = args.out_root or os.path.join(
        ROOT, "results", f"no_threshold_viz_{seq}"
    )

    flow_dir = os.path.join(ROOT, "data", "flow", flow_model, seq)
    img_dir  = os.path.join(args.davis_root, "JPEGImages", "480p", seq)

    flow_files = sorted(glob.glob(os.path.join(flow_dir, "*.npy")))
    if not flow_files:
        sys.exit(f"[error] no flow files in {flow_dir}")

    if args.frame:
        flow_files = [f for f in flow_files
                      if os.path.splitext(os.path.basename(f))[0] == args.frame]
        if not flow_files:
            sys.exit(f"[error] frame {args.frame} not found")

    print(f"Sequence   : {seq}")
    print(f"Flow model : {flow_model}")
    print(f"Frames     : {len(flow_files)}")
    print(f"Output     : {out_root}\n")

    for i, fp in enumerate(flow_files):
        stem = os.path.splitext(os.path.basename(fp))[0]
        img_path = os.path.join(img_dir, f"{stem}.jpg")
        rgb = (np.array(Image.open(img_path).convert("RGB"))
               if os.path.exists(img_path)
               else np.zeros((480, 854, 3), np.uint8))
        flow = np.load(fp)
        _save_frame(rgb, flow, seq, stem, flow_model,
                    os.path.join(out_root, f"{stem}.png"))
        print(f"\r  [{i+1}/{len(flow_files)}] {stem}.png", end="", flush=True)

    print(f"\nDone — saved to {out_root}")


if __name__ == "__main__":
    main()

"""
visualize_camera_compensation.py
=================================
Visualize the effect of camera motion compensation on optical flow fields.

For each frame shows two rows:
  Before compensation: RGB | flow (colour-wheel) | magnitude heatmap | fg mask
  After  compensation: RGB | flow (colour-wheel) | magnitude heatmap | fg mask

This makes it immediately clear why the foreground mask completely misses the
bear on tracking-camera sequences when compensation is disabled: the bear has
near-zero flow magnitude in the raw field, so it falls below the threshold.

Usage:
    python scripts/visualize_camera_compensation.py
    python scripts/visualize_camera_compensation.py --sequence bear --frame 00030
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "models", "FlowFormer", "core", "utils"))

from flow_viz import flow_to_image   # noqa: E402
from region_growing import RegionGrowingConfig, _foreground_mask, _gaussian_blur_flow  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _mag(flow):
    return np.sqrt((flow.astype(np.float32) ** 2).sum(-1))


def _draw_wheel(ax):
    """Draw colour wheel into an existing axes."""
    size = 100
    yi, xi = np.mgrid[-size // 2:size // 2 + 1,
                      -size // 2:size // 2 + 1].astype(np.float32)
    r = np.sqrt(xi ** 2 + yi ** 2)
    wheel_img = flow_to_image(np.stack([xi, yi], axis=-1))
    wheel_img[r > size // 2] = 255

    ax.imshow(wheel_img)
    ax.axis("off")
    ax.set_title("Flow\ndirection", fontsize=8, pad=4)

    for txt, xy in [("→", (1.20, 0.50)), ("←", (-0.20, 0.50)),
                    ("↑", (0.50, -0.06)), ("↓", (0.50, 1.06))]:
        ax.text(*xy, txt, transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="dimgray")


def _save_frame(rgb, flow_raw, seq, stem, flow_model, out_path):
    cfg = RegionGrowingConfig()

    # ── Before compensation ────────────────────────────────────────────────────
    flow_before = flow_raw.astype(np.float32)
    if cfg.smooth_sigma > 0:
        flow_before = _gaussian_blur_flow(flow_before, cfg.smooth_sigma)
    mag_before = _mag(flow_before)
    _, thr_before = _foreground_mask(mag_before, cfg)
    median_vec = np.median(flow_raw.reshape(-1, 2), axis=0)

    # ── After compensation ─────────────────────────────────────────────────────
    flow_after = flow_raw.astype(np.float32) - median_vec
    if cfg.smooth_sigma > 0:
        flow_after = _gaussian_blur_flow(flow_after, cfg.smooth_sigma)
    mag_after = _mag(flow_after)
    _, thr_after = _foreground_mask(mag_after, cfg)

    # Shared colour scale so both magnitude panels are directly comparable
    vmax = float(max(mag_before.max(), mag_after.max()))

    panel_titles = [
        ["Input",         "Input"],
        [f"{flow_model} flow\n(before compensation)",
         f"{flow_model} flow\n(after compensation)"],
        [f"Magnitude (shared scale 0–{vmax:.1f} px)\n(before compensation)",
         f"Magnitude (shared scale 0–{vmax:.1f} px)\n(after compensation)"],
        ["Foreground mask\n(before compensation)",
         "Foreground mask\n(after compensation)"],
    ]

    row_data = [
        (flow_before, mag_before, thr_before),
        (flow_after,  mag_after,  thr_after),
    ]

    # ── Figure: 2 rows × 5 cols (col 2 is a narrow wheel-legend column) ───────
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(28, 9))
    gs  = GridSpec(2, 5, figure=fig,
                   width_ratios=[1, 1, 0.18, 1, 1],
                   top=0.88, bottom=0.04, left=0.01, right=0.97,
                   hspace=0.10, wspace=0.08)

    # col mapping: 0=Input, 1=Flow, (2=wheel), 3=Magnitude, 4=Mask
    col_gs = [0, 1, 3, 4]

    for row_idx, (flow, mag, thr) in enumerate(row_data):
        fg_mask = mag > thr

        for col_idx, gs_col in enumerate(col_gs):
            ax = fig.add_subplot(gs[row_idx, gs_col])
            ax.axis("off")
            ax.set_title(panel_titles[col_idx][row_idx], fontsize=10, pad=5)

            if col_idx == 0:                          # Input RGB
                ax.imshow(rgb)

            elif col_idx == 1:                        # Flow
                ax.imshow(flow_to_image(flow))

            elif col_idx == 2:                        # Magnitude + colorbar + threshold
                im = ax.imshow(mag, cmap="plasma", vmin=0, vmax=vmax)
                ax.axis("off")
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, aspect=22)
                cbar.set_label("px / frame", fontsize=8)
                # Mark threshold on the colorbar
                thr_norm = thr / vmax
                cbar.ax.axhline(thr_norm, color="white", linewidth=2, linestyle="--")
                cbar.ax.text(
                    -0.05, thr_norm, f"{thr:.1f}",
                    va="center", ha="right", fontsize=8,
                    color="white", transform=cbar.ax.transAxes,
                )
                cbar.ax.text(
                    1.05, thr_norm, "← threshold",
                    va="center", ha="left", fontsize=7,
                    color="dimgray", transform=cbar.ax.transAxes,
                )

            elif col_idx == 3:                        # Binary foreground mask
                ax.imshow(fg_mask, cmap="gray", vmin=0, vmax=1)

    # ── Colour wheel in the centre column, spanning both rows ────────────────
    ax_wheel = fig.add_subplot(gs[:, 2])
    _draw_wheel(ax_wheel)

    fig.suptitle(
        f"Camera Motion Compensation — {seq}_{stem}  |  {flow_model}",
        fontsize=14, fontweight="bold", y=0.95,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Visualise camera compensation effect on flow & fg mask."
    )
    p.add_argument("--sequence",   default="bear")
    p.add_argument("--flow-model", default="flowformer")
    p.add_argument("--frame",      default=None,
                   help="single frame stem to render, e.g. 00030 (default: all frames)")
    p.add_argument("--davis-root", default=os.path.join(ROOT, "data", "DAVIS"))
    p.add_argument("--out-root",   default=None)
    args = p.parse_args()

    seq        = args.sequence
    flow_model = args.flow_model
    out_root   = args.out_root or os.path.join(
        ROOT, "results", f"camera_compensation_viz_{seq}"
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
        out_path = os.path.join(out_root, f"{stem}.png")
        _save_frame(rgb, flow, seq, stem, flow_model, out_path)
        print(f"\r  [{i+1}/{len(flow_files)}] {stem}.png", end="", flush=True)

    print(f"\nDone — saved to {out_root}")


if __name__ == "__main__":
    main()

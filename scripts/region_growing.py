"""
region_growing.py
=================
Motion-based object segmentation via region growing on optical-flow fields.

Pipeline per frame:
  1. (optional) smooth the flow field to suppress boundary noise.
  2. Pick a foreground motion mask via an adaptive (or fixed) magnitude threshold
     -- pixels that move clearly faster than the dominant (background) motion.
  3. Grow regions: connect neighbouring foreground pixels whose flow VECTORS are
     similar (||f_a - f_b|| < tau). 4- or 8-connectivity. Implemented as
     connected components over a local-similarity graph -- the efficient,
     vectorised equivalent of seeded flow-similarity region growing.
  4. (optional, "unified") additionally require RGB similarity between neighbours.
  5. Drop tiny regions and relabel to consecutive instance ids (0 = background).

Everything that matters for the report's ablations is a parameter of
`RegionGrowingConfig`, so the same code produces every ablation variant:
  - threshold:  "adaptive" (mean + k*std of magnitude) vs "fixed"
  - connectivity: 4 vs 8
  - smoothing: gaussian sigma (0 = off)
  - unified: blend flow + RGB similarity (lambda_rgb > 0)
  - camera motion compensation: subtract the dominant (median) flow first

Output: an (H, W) uint8 label image, 0 = background, 1..N = moving instances.
"""
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Configuration (one object = one ablation setting)
# --------------------------------------------------------------------------- #
@dataclass
class RegionGrowingConfig:
    # --- foreground / seed selection ---
    threshold_mode: str = "adaptive"   # "adaptive" | "fixed"
    seed_k: float = 1.0                # adaptive: thresh = mean + k * std
    fixed_thresh: float = 2.0          # fixed: magnitude threshold (px)
    # --- region growing ---
    connectivity: int = 8              # 4 or 8
    tau: float = 1.5                   # max flow-vector L2 distance to merge (px)
    min_area: int = 200                # drop regions smaller than this (px)
    # --- preprocessing ---
    smooth_sigma: float = 1.0          # gaussian sigma on flow (0 = off)
    compensate_camera: bool = True     # subtract median (background) flow
    # --- unified (flow + appearance) ---
    lambda_rgb: float = 0.0            # 0 = motion only; >0 also needs RGB sim
    rgb_tau: float = 12.0              # max RGB L2 distance to merge (0-255)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _gaussian_blur_flow(flow, sigma):
    from scipy.ndimage import gaussian_filter
    out = np.empty_like(flow)
    out[..., 0] = gaussian_filter(flow[..., 0], sigma)
    out[..., 1] = gaussian_filter(flow[..., 1], sigma)
    return out


def _foreground_mask(mag, cfg):
    """Boolean mask of pixels that move significantly w.r.t. the background."""
    if cfg.threshold_mode == "fixed":
        thr = cfg.fixed_thresh
    else:  # adaptive: relative to this frame's magnitude distribution
        thr = float(mag.mean() + cfg.seed_k * mag.std())
    return mag > thr, thr


def _neighbour_offsets(connectivity):
    # Only "forward" neighbours (right/down/diagonals) so each edge is built once.
    if connectivity == 4:
        return [(0, 1), (1, 0)]
    return [(0, 1), (1, 0), (1, 1), (1, -1)]


def _build_edges(flow, fg, cfg, rgb=None):
    """Build edges between adjacent foreground pixels with similar flow (+RGB).

    Returns (rows, cols) flat-index pairs for a connectivity graph over the
    H*W grid. Fully vectorised -- no per-pixel Python loop.
    """
    H, W = fg.shape
    idx = np.arange(H * W).reshape(H, W)
    rows, cols = [], []

    for dy, dx in _neighbour_offsets(cfg.connectivity):
        # overlapping slices for a pixel and its (dy,dx) neighbour
        ay0, ay1 = max(0, -dy), H - max(0, dy)
        ax0, ax1 = max(0, -dx), W - max(0, dx)
        by0, by1 = max(0, dy), H - max(0, -dy)
        bx0, bx1 = max(0, dx), W - max(0, -dx)

        a_fg = fg[ay0:ay1, ax0:ax1]
        b_fg = fg[by0:by1, bx0:bx1]
        both = a_fg & b_fg                      # both endpoints are foreground

        fa = flow[ay0:ay1, ax0:ax1]
        fb = flow[by0:by1, bx0:bx1]
        flow_dist = np.sqrt(((fa - fb) ** 2).sum(-1))
        sim = both & (flow_dist < cfg.tau)

        if cfg.lambda_rgb > 0 and rgb is not None:
            ca = rgb[ay0:ay1, ax0:ax1].astype(np.float32)
            cb = rgb[by0:by1, bx0:bx1].astype(np.float32)
            rgb_dist = np.sqrt(((ca - cb) ** 2).sum(-1))
            sim = sim & (rgb_dist < cfg.rgb_tau)

        ia = idx[ay0:ay1, ax0:ax1][sim]
        ib = idx[by0:by1, bx0:bx1][sim]
        rows.append(ia)
        cols.append(ib)

    if rows:
        return np.concatenate(rows), np.concatenate(cols)
    return np.array([], dtype=int), np.array([], dtype=int)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def segment_frame(flow, cfg: RegionGrowingConfig, rgb=None):
    """Segment one frame's flow field into moving-object instances.

    Args:
        flow: (H, W, 2) float32 optical flow.
        cfg:  RegionGrowingConfig.
        rgb:  optional (H, W, 3) uint8 frame, used when cfg.lambda_rgb > 0.

    Returns:
        labels: (H, W) uint8, 0 = background, 1..N = instances.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    flow = flow.astype(np.float32)
    H, W = flow.shape[:2]

    if cfg.compensate_camera:
        # Background ~ dominant motion; subtract its median so the static world
        # has ~zero flow and only independently moving objects stand out.
        med = np.median(flow.reshape(-1, 2), axis=0)
        flow = flow - med

    if cfg.smooth_sigma > 0:
        flow = _gaussian_blur_flow(flow, cfg.smooth_sigma)

    mag = np.sqrt((flow ** 2).sum(-1))
    fg, _ = _foreground_mask(mag, cfg)

    if not fg.any():
        return np.zeros((H, W), np.uint8)

    rows, cols = _build_edges(flow, fg, cfg, rgb=rgb)

    n = H * W
    if len(rows) == 0:
        # No similar-neighbour edges: every fg pixel is its own component.
        graph = coo_matrix((n, n))
    else:
        data = np.ones(len(rows), dtype=np.uint8)
        graph = coo_matrix((data, (rows, cols)), shape=(n, n))

    n_comp, comp = connected_components(graph, directed=False)
    comp = comp.reshape(H, W)

    # Keep only foreground components, drop tiny ones, relabel 1..N.
    labels = np.zeros((H, W), np.uint8)
    comp_fg = np.where(fg, comp, -1)
    next_id = 1
    for c in np.unique(comp_fg):
        if c < 0:
            continue
        region = comp_fg == c
        if region.sum() < cfg.min_area:
            continue
        labels[region] = next_id
        next_id += 1
        if next_id > 255:                       # uint8 instance-id ceiling
            break
    return labels


def segment_to_binary(labels):
    """Collapse instance labels to a binary foreground mask (uint8 0/1)."""
    return (labels > 0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# CLI: run region growing over saved flow and write DAVIS-style mask PNGs
# --------------------------------------------------------------------------- #
def _davis_palette():
    """The standard DAVIS/PASCAL VOC colour palette for instance PNGs."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        palette[i] = (r, g, b)
    return palette


def save_mask(labels, path):
    from PIL import Image
    img = Image.fromarray(labels, mode="P")
    img.putpalette(_davis_palette().flatten().tolist())
    img.save(path)


def main():
    import argparse
    import glob
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Flow-based region growing -> masks.")
    p.add_argument("--flow-root", default=os.path.join(here, "..", "data", "flow", "raft"))
    p.add_argument("--davis-root", default=os.path.join(here, "..", "data", "DAVIS"))
    p.add_argument("--out-root", default=os.path.join(here, "..", "results", "raft_rg"))
    p.add_argument("--sequences", default="bear", help="comma list, or 'all'")
    # expose the ablation knobs on the CLI
    p.add_argument("--threshold-mode", default="adaptive", choices=["adaptive", "fixed"])
    p.add_argument("--seed-k", type=float, default=1.0)
    p.add_argument("--fixed-thresh", type=float, default=2.0)
    p.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    p.add_argument("--tau", type=float, default=1.5)
    p.add_argument("--min-area", type=int, default=200)
    p.add_argument("--smooth-sigma", type=float, default=1.0)
    p.add_argument("--compensate-camera", action=argparse.BooleanOptionalAction,
                   default=True, help="subtract median background flow (default on)")
    p.add_argument("--lambda-rgb", type=float, default=0.0)
    p.add_argument("--rgb-tau", type=float, default=12.0)
    args = p.parse_args()

    cfg = RegionGrowingConfig(
        threshold_mode=args.threshold_mode, seed_k=args.seed_k,
        fixed_thresh=args.fixed_thresh, connectivity=args.connectivity,
        tau=args.tau, min_area=args.min_area, smooth_sigma=args.smooth_sigma,
        compensate_camera=args.compensate_camera, lambda_rgb=args.lambda_rgb,
        rgb_tau=args.rgb_tau,
    )

    if args.sequences == "all":
        seqs = sorted(os.listdir(args.flow_root))
    else:
        seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]

    img_root = os.path.join(args.davis_root, "JPEGImages", "480p")
    for seq in seqs:
        flow_dir = os.path.join(args.flow_root, seq)
        out_dir = os.path.join(args.out_root, seq)
        os.makedirs(out_dir, exist_ok=True)
        flow_files = sorted(glob.glob(os.path.join(flow_dir, "*.npy")))
        if not flow_files:
            print(f"[region_growing] no flow for '{seq}' in {flow_dir}")
            continue
        print(f"[region_growing] {seq}: {len(flow_files)} frames")
        for fp in flow_files:
            flow = np.load(fp)
            rgb = None
            if cfg.lambda_rgb > 0:
                import cv2
                stem = os.path.splitext(os.path.basename(fp))[0]
                ip = os.path.join(img_root, seq, f"{stem}.jpg")
                if os.path.exists(ip):
                    rgb = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
            labels = segment_frame(flow, cfg, rgb=rgb)
            stem = os.path.splitext(os.path.basename(fp))[0]
            save_mask(labels, os.path.join(out_dir, f"{stem}.png"))
    print("[region_growing] done.")


if __name__ == "__main__":
    main()

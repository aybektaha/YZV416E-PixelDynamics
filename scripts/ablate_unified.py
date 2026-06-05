"""
ablate_unified.py  (standalone — imports shared functions, does not modify them)
================================================================================
Mini-ablation: why does the unified (flow + RGB) criterion hurt? Sweep the RGB
tolerance (rgb_tau) and the flow-only baseline on the cached RAFT flow, so we can
show how requiring RGB similarity during region growing over-constrains the grow
and drops J&F. Read-only w.r.t. shared code (imports segment_frame / metrics).

    python scripts/ablate_unified.py --sequences val --out results/ablate_unified.csv
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np
from PIL import Image

from region_growing import RegionGrowingConfig, segment_frame   # import only
from metrics import evaluate_sequence                            # import only

HERE = os.path.dirname(os.path.abspath(__file__))


def evaluate_config(cfg, seqs, flow_root, gt_root, img_root, use_rgb):
    per_seq = []
    for seq in seqs:
        flow_files = sorted(glob.glob(os.path.join(flow_root, seq, "*.npy")))
        gt_files = sorted(glob.glob(os.path.join(gt_root, seq, "*.png")))
        if not flow_files or not gt_files:
            continue
        gt_by_stem = {os.path.splitext(os.path.basename(g))[0]: g for g in gt_files}
        preds, gts = [], []
        for fp in flow_files:
            stem = os.path.splitext(os.path.basename(fp))[0]
            if stem not in gt_by_stem:
                continue
            rgb = None
            if use_rgb:
                ip = os.path.join(img_root, seq, f"{stem}.jpg")
                if os.path.exists(ip):
                    rgb = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
            preds.append(segment_frame(np.load(fp), cfg, rgb=rgb))
            gts.append(np.array(Image.open(gt_by_stem[stem])))
        if preds:
            per_seq.append(evaluate_sequence(gts, preds))
    if not per_seq:
        return float("nan"), float("nan"), float("nan")
    return (np.nanmean([r["J"] for r in per_seq]),
            np.nanmean([r["F"] for r in per_seq]),
            np.nanmean([r["JF"] for r in per_seq]))


def main():
    p = argparse.ArgumentParser(description="Unified (flow+RGB) rgb_tau mini-ablation.")
    p.add_argument("--sequences", default="val")
    p.add_argument("--davis-root", default=os.path.join(HERE, "..", "data", "DAVIS"))
    p.add_argument("--flow-root", default=os.path.join(HERE, "..", "data", "flow", "raft"))
    p.add_argument("--out", default=os.path.join(HERE, "..", "results", "ablate_unified.csv"))
    args = p.parse_args()

    if args.sequences in ("val", "train"):
        with open(os.path.join(args.davis_root, "ImageSets", "2017", f"{args.sequences}.txt")) as f:
            seqs = [ln.strip() for ln in f if ln.strip()]
    else:
        seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]
    seqs = [s for s in seqs if os.path.isdir(os.path.join(args.flow_root, s))]

    gt_root = os.path.join(args.davis_root, "Annotations_unsupervised", "480p")
    img_root = os.path.join(args.davis_root, "JPEGImages", "480p")
    base = dict(threshold_mode="adaptive", seed_k=1.0, connectivity=8,
                tau=1.5, min_area=200, smooth_sigma=1.0, compensate_camera=True)

    # flow-only baseline, then flow+RGB at increasing RGB tolerance.
    grid = [("flow-only (lambda_rgb=0)", 0.0, 0.0, False)]
    for rt in (6.0, 12.0, 24.0, 48.0, 96.0):
        grid.append((f"flow+RGB rgb_tau={rt:g}", 1.0, rt, True))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[unified] {len(seqs)} sequences")
    print(f"{'setting':<26}{'J':>8}{'F':>8}{'J&F':>8}")
    print("-" * 50)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["setting", "lambda_rgb", "rgb_tau", "J", "F", "JF"])
        for label, lam, rt, use_rgb in grid:
            cfg = RegionGrowingConfig(**base, lambda_rgb=lam, rgb_tau=rt)
            J, F, JF = evaluate_config(cfg, seqs, args.flow_root, gt_root, img_root, use_rgb)
            w.writerow([label, lam, rt, f"{J:.4f}", f"{F:.4f}", f"{JF:.4f}"]); f.flush()
            print(f"{label:<26}{J:>8.4f}{F:>8.4f}{JF:>8.4f}")
    print(f"[unified] wrote {args.out}")


if __name__ == "__main__":
    main()

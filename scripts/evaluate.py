"""
evaluate.py
===========
Evaluate predicted instance masks against DAVIS unsupervised annotations and
report J (region similarity), F (boundary accuracy) and the J&F mean.

This is the single shared evaluation entry point for the whole team: whatever
backbone produced the masks (RAFT / GMFlow / FlowFormer region growing, or the
SAM appearance baseline), masks are read from the SAME layout

    results/<method>/<sequence>/<frame>.png   # DAVIS-palette label PNGs

so every method's numbers are directly comparable.

Example:
    python scripts/evaluate.py --pred results/raft_rg --sequences bear
    python scripts/evaluate.py --pred results/raft_rg --sequences val --csv out.csv
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image

from metrics import evaluate_sequence


def load_label(path):
    """Load a palette/grayscale PNG as an (H, W) instance-id array."""
    return np.array(Image.open(path))


def list_sequences(arg, davis_root, pred_root):
    if arg == "all":
        return sorted(os.listdir(pred_root))
    if arg in ("val", "train"):
        with open(os.path.join(davis_root, "ImageSets", "2017", f"{arg}.txt")) as f:
            return [ln.strip() for ln in f if ln.strip()]
    return [s.strip() for s in arg.split(",") if s.strip()]


def evaluate_one(seq, pred_root, gt_root):
    """Load GT + prediction frames for a sequence and compute its scores."""
    gt_dir = os.path.join(gt_root, seq)
    pred_dir = os.path.join(pred_root, seq)
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    if not gt_files:
        return None

    gt_masks, pred_masks = [], []
    for gf in gt_files:
        stem = os.path.splitext(os.path.basename(gf))[0]
        pf = os.path.join(pred_dir, f"{stem}.png")
        if not os.path.exists(pf):
            continue
        gt_masks.append(load_label(gf))
        pred_masks.append(load_label(pf))

    if not pred_masks:
        return None
    return evaluate_sequence(gt_masks, pred_masks)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="DAVIS J&F evaluation.")
    p.add_argument("--pred", required=True, help="results/<method> root with <seq>/<frame>.png")
    p.add_argument("--davis-root", default=os.path.join(here, "..", "data", "DAVIS"))
    p.add_argument("--sequences", default="val", help="'all' | 'val' | 'train' | comma list")
    p.add_argument("--csv", default=None, help="optional path to write per-sequence CSV")
    args = p.parse_args()

    gt_root = os.path.join(args.davis_root, "Annotations_unsupervised", "480p")
    seqs = list_sequences(args.sequences, args.davis_root, args.pred)

    rows = []
    print(f"{'sequence':<22} {'J':>7} {'F':>7} {'J&F':>7}")
    print("-" * 46)
    for seq in seqs:
        res = evaluate_one(seq, args.pred, gt_root)
        if res is None:
            print(f"{seq:<22} {'--- no masks ---':>22}")
            continue
        rows.append((seq, res["J"], res["F"], res["JF"]))
        print(f"{seq:<22} {res['J']:>7.4f} {res['F']:>7.4f} {res['JF']:>7.4f}")

    if rows:
        J = np.nanmean([r[1] for r in rows])
        F = np.nanmean([r[2] for r in rows])
        JF = np.nanmean([r[3] for r in rows])
        print("-" * 46)
        print(f"{'MEAN':<22} {J:>7.4f} {F:>7.4f} {JF:>7.4f}")

    if args.csv and rows:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "J", "F", "JF"])
            w.writerows(rows)
            w.writerow(["MEAN", J, F, JF])
        print(f"[evaluate] wrote {args.csv}")


if __name__ == "__main__":
    main()

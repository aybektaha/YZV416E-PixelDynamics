"""
run_ablations.py
================
Region-growing ablation harness (Aybek's part). Re-uses cached flow and runs a
grid of region-growing settings over a set of sequences, writing one CSV row per
setting so the report's ablation tables/curves come straight from the data.

Studies covered:
  - camera compensation:        on vs off
  - threshold mode:             adaptive (varying seed_k) vs fixed (varying thresh)
  - connectivity:               4 vs 8
  - flow smoothing:             sigma in {0, 1, 2}
  - merge tolerance tau:        sweep

Assumes flow already extracted (run flow_extraction.py first):
    python scripts/flow_extraction.py --sequences val
    python scripts/run_ablations.py --sequences val --out ablations.csv
"""
import argparse
import csv
import glob
import os

import numpy as np

from region_growing import RegionGrowingConfig, segment_frame
from metrics import evaluate_sequence
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def evaluate_config(cfg, seqs, flow_root, gt_root):
    """Run a config over sequences in-memory and return mean J, F, J&F."""
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
            labels = segment_frame(np.load(fp), cfg)
            preds.append(labels)
            gts.append(np.array(Image.open(gt_by_stem[stem])))
        if preds:
            per_seq.append(evaluate_sequence(gts, preds))
    if not per_seq:
        return float("nan"), float("nan"), float("nan")
    J = np.nanmean([r["J"] for r in per_seq])
    F = np.nanmean([r["F"] for r in per_seq])
    JF = np.nanmean([r["JF"] for r in per_seq])
    return J, F, JF


def build_grid():
    """Yield (study, label, RegionGrowingConfig) tuples. Base = best default."""
    base = dict(threshold_mode="adaptive", seed_k=1.0, connectivity=8,
                tau=1.5, min_area=200, smooth_sigma=1.0, compensate_camera=True)

    def cfg(**kw):
        d = dict(base); d.update(kw); return RegionGrowingConfig(**d)

    yield ("baseline", "default", cfg())

    # camera compensation on/off
    yield ("camera_comp", "on", cfg(compensate_camera=True))
    yield ("camera_comp", "off", cfg(compensate_camera=False))

    # adaptive threshold sensitivity (seed_k)
    for k in (0.5, 1.0, 1.5, 2.0):
        yield ("adaptive_seed_k", f"k={k}", cfg(seed_k=k))

    # fixed threshold sensitivity
    for t in (1.0, 2.0, 3.0, 4.0):
        yield ("fixed_thresh", f"t={t}", cfg(threshold_mode="fixed", fixed_thresh=t))

    # connectivity
    yield ("connectivity", "4", cfg(connectivity=4))
    yield ("connectivity", "8", cfg(connectivity=8))

    # flow smoothing
    for s in (0.0, 1.0, 2.0):
        yield ("smoothing", f"sigma={s}", cfg(smooth_sigma=s))

    # merge tolerance tau
    for t in (1.0, 1.5, 2.0, 3.0):
        yield ("tau", f"tau={t}", cfg(tau=t))


def main():
    p = argparse.ArgumentParser(description="Region-growing ablations -> CSV.")
    p.add_argument("--sequences", default="val", help="'val'|'train'|comma list")
    p.add_argument("--davis-root", default=os.path.join(HERE, "..", "data", "DAVIS"))
    p.add_argument("--flow-root", default=os.path.join(HERE, "..", "data", "flow", "raft"))
    p.add_argument("--out", default=os.path.join(HERE, "..", "results", "ablations.csv"))
    args = p.parse_args()

    if args.sequences in ("val", "train"):
        with open(os.path.join(args.davis_root, "ImageSets", "2017", f"{args.sequences}.txt")) as f:
            seqs = [ln.strip() for ln in f if ln.strip()]
    else:
        seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]
    # only sequences whose flow has been extracted
    seqs = [s for s in seqs if os.path.isdir(os.path.join(args.flow_root, s))]
    print(f"[ablations] {len(seqs)} sequence(s)")

    gt_root = os.path.join(args.davis_root, "Annotations_unsupervised", "480p")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study", "setting", "J", "F", "JF"])
        print(f"{'study':<18}{'setting':<14}{'J':>7}{'F':>7}{'J&F':>7}")
        print("-" * 53)
        for study, label, cfg in build_grid():
            J, F, JF = evaluate_config(cfg, seqs, args.flow_root, gt_root)
            w.writerow([study, label, f"{J:.4f}", f"{F:.4f}", f"{JF:.4f}"])
            f.flush()
            print(f"{study:<18}{label:<14}{J:>7.4f}{F:>7.4f}{JF:>7.4f}")
    print(f"[ablations] wrote {args.out}")


if __name__ == "__main__":
    main()

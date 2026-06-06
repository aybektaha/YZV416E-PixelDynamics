"""
plot_comparison.py  (standalone — reads result CSVs, writes a figure)
=====================================================================
Read the per-method evaluation CSVs (produced by evaluate.py) and draw a grouped
bar chart of J, F and J&F for the report/presentation. Read-only: does not touch
any shared script or pipeline output.

    python scripts/plot_comparison.py
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

# (csv file, display label, category) — category drives the bar colour.
METHODS = [
    ("raft_rg.csv",        "RAFT\n(CNN flow)",          "motion"),
    ("gmflow_rg.csv",      "GMFlow\n(transf. flow)",    "motion"),
    ("flowformer_rg.csv",  "FlowFormer\n(transf. flow)", "motion"),
    ("gmflow_unified.csv", "GMFlow+RGB\n(unified)",     "unified"),
    ("sam.csv",            "SAM\n(appearance)",         "appearance"),
]
COLORS = {"motion": "#2c7fb8", "unified": "#f0a202", "appearance": "#969696"}


def read_mean(path):
    """Return (J, F, JF) from the MEAN row of an evaluate.py CSV."""
    with open(path) as f:
        for row in csv.reader(f):
            if row and row[0].strip().upper() == "MEAN":
                return float(row[1]), float(row[2]), float(row[3])
    raise ValueError(f"no MEAN row in {path}")


def main():
    labels, cats, J, F, JF = [], [], [], [], []
    for fname, label, cat in METHODS:
        path = os.path.join(RES, fname)
        if not os.path.exists(path):
            print(f"[plot] skip (missing): {fname}")
            continue
        j, fb, jf = read_mean(path)
        labels.append(label); cats.append(cat)
        J.append(j); F.append(fb); JF.append(jf)
        print(f"[plot] {label.splitlines()[0]:<12} J={j:.3f} F={fb:.3f} J&F={jf:.3f}")

    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # J and F as lighter context bars, J&F as the bold coloured bar.
    ax.bar(x - w, J, w, label="J (region)", color="#bdd7e7")
    ax.bar(x,     F, w, label="F (boundary)", color="#74a9cf")
    bars = ax.bar(x + w, JF, w, label="J&F (mean)",
                  color=[COLORS[c] for c in cats], edgecolor="black", linewidth=0.6)

    for b, v in zip(bars, JF):                       # value labels on J&F bars
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score"); ax.set_ylim(0, max(JF) * 1.25)
    ax.set_title("DAVIS val (30 seq): motion-based segmentation vs appearance baseline",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    # annotation: motion vs appearance gap (the headline finding)
    motion_best = max(jf for jf, c in zip(JF, cats) if c == "motion")
    app = next((jf for jf, c in zip(JF, cats) if c == "appearance"), None)
    if app:
        ax.annotate(f"motion ≈ {motion_best/app:.1f}× appearance",
                    xy=(0.5, 0.93), xycoords="axes fraction", ha="center",
                    fontsize=10, color="#2c7fb8", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(RES, "comparison.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()

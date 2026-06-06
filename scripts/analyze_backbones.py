"""
analyze_backbones.py  (standalone, read-only)
=============================================
Per-sequence comparison of the flow backbones from their evaluate.py CSVs.
Answers: where does FlowFormer fall behind / get ahead, are the 3 backbones
correlated, is the gap concentrated in a few sequences? Used to write the
report/presentation commentary. Does not modify any shared file.

    python scripts/analyze_backbones.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

BACKBONES = [("RAFT", "raft_rg.csv"),
             ("GMFlow", "gmflow_rg.csv"),
             ("FlowFormer", "flowformer_rg.csv")]


def read_per_seq(path):
    """Return {sequence: J&F} from an evaluate.py CSV (skips MEAN row)."""
    out = {}
    with open(path) as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if not row or row[0].strip().upper() == "MEAN":
                continue
            out[row[0]] = float(row[3])
    return out


def main():
    data = {}
    for name, fname in BACKBONES:
        p = os.path.join(RES, fname)
        if not os.path.exists(p):
            print(f"[analyze] missing {fname}"); return
        data[name] = read_per_seq(p)

    seqs = sorted(set(data["RAFT"]) & set(data["GMFlow"]) & set(data["FlowFormer"]))
    print(f"[analyze] {len(seqs)} common sequences\n")

    # Means
    for name, _ in BACKBONES:
        m = sum(data[name][s] for s in seqs) / len(seqs)
        print(f"  mean J&F  {name:<11} {m:.4f}")
    print()

    # Per-seq table sorted by FlowFormer - RAFT (where FF loses most -> top)
    rows = []
    for s in seqs:
        r, g, ff = data["RAFT"][s], data["GMFlow"][s], data["FlowFormer"][s]
        rows.append((s, r, g, ff, ff - r))
    rows.sort(key=lambda x: x[4])

    print(f"{'sequence':<20}{'RAFT':>8}{'GMFlow':>8}{'FlowFmr':>9}{'FF-RAFT':>9}")
    print("-" * 54)
    for s, r, g, ff, d in rows:
        flag = "  <-- FF worse" if d < -0.05 else ("  <-- FF better" if d > 0.05 else "")
        print(f"{s:<20}{r:>8.3f}{g:>8.3f}{ff:>9.3f}{d:>+9.3f}{flag}")

    # Summary stats
    diffs = [d for *_, d in rows]
    ff_worse = [r for r in rows if r[4] < -0.05]
    ff_better = [r for r in rows if r[4] > 0.05]
    print("-" * 54)
    print(f"\nFlowFormer vs RAFT:")
    print(f"  mean diff           {sum(diffs)/len(diffs):+.4f}")
    print(f"  FF clearly worse (> .05) on {len(ff_worse)} seq: "
          f"{', '.join(s for s,*_ in ff_worse) or '-'}")
    print(f"  FF clearly better(> .05) on {len(ff_better)} seq: "
          f"{', '.join(s for s,*_ in ff_better) or '-'}")
    # how concentrated is the gap?
    total_gap = sum(d for d in diffs if d < 0)
    worst_gap = sum(r[4] for r in ff_worse)
    if total_gap:
        print(f"  share of total deficit from those {len(ff_worse)} seq: "
              f"{worst_gap/total_gap*100:.0f}%")


if __name__ == "__main__":
    main()

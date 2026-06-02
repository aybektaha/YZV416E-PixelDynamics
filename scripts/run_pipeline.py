"""
run_pipeline.py
===============
One-command end-to-end runner: optical flow -> region growing -> J&F.
Handy for the demo video and for quick sanity checks on a few sequences.

Example:
    python scripts/run_pipeline.py --sequences bear,dog
    python scripts/run_pipeline.py --sequences val --skip-flow   # flow already cached
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description="Flow -> region growing -> evaluation.")
    p.add_argument("--sequences", default="bear")
    p.add_argument("--method", default="raft_rg", help="results/<method> folder name")
    p.add_argument("--skip-flow", action="store_true", help="reuse cached flow")
    # forward the most useful region-growing knobs
    p.add_argument("--threshold-mode", default="adaptive")
    p.add_argument("--seed-k", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.5)
    p.add_argument("--min-area", type=int, default=200)
    p.add_argument("--smooth-sigma", type=float, default=1.0)
    p.add_argument("--no-compensate-camera", action="store_true")
    args = p.parse_args()

    py = sys.executable
    results_root = os.path.join(HERE, "..", "results", args.method)

    if not args.skip_flow:
        run([py, os.path.join(HERE, "flow_extraction.py"), "--sequences", args.sequences])

    rg = [py, os.path.join(HERE, "region_growing.py"),
          "--sequences", args.sequences, "--out-root", results_root,
          "--threshold-mode", args.threshold_mode, "--seed-k", str(args.seed_k),
          "--tau", str(args.tau), "--min-area", str(args.min_area),
          "--smooth-sigma", str(args.smooth_sigma)]
    if args.no_compensate_camera:
        rg.append("--no-compensate-camera")
    run(rg)

    run([py, os.path.join(HERE, "evaluate.py"),
         "--pred", results_root, "--sequences", args.sequences])


if __name__ == "__main__":
    main()

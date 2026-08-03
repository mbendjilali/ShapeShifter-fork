#!/usr/bin/env python3
"""
Dump every scalar from every TensorBoard run under a directory into one CSV.

Usage
-----
    python export_tb_scalars.py runs/ -o tb_scalars.csv
    python export_tb_scalars.py runs/ -o tb_d1.csv --filter diffusion_level_1

Output columns: run, tag, step, wall_time, value

Only depends on `tensorboard`, which is already in the training environment.
No plotting, no downsampling: the CSV is the raw record so the numbers quoted
in the manuscript can be traced back to a step and a wall clock time.
"""

import argparse
import csv
import os
import sys

from tensorboard.backend.event_processing import event_accumulator


def find_runs(root):
    """Yield directories that contain at least one tfevents file."""
    for dirpath, _dirnames, filenames in os.walk(root):
        if any("tfevents" in f for f in filenames):
            yield dirpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory to walk, e.g. runs/")
    ap.add_argument("-o", "--out", default="/home/moussabendjilali/libs/ShapeShifter-fork/tb_scalars.csv")
    ap.add_argument(
        "--filter",
        default="",
        help="only keep runs whose path contains this substring",
    )
    args = ap.parse_args()

    runs = sorted(r for r in find_runs(args.root) if args.filter in r)
    if not runs:
        sys.exit(f"no tfevents files found under {args.root!r}")

    n_rows = 0
    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run", "tag", "step", "wall_time", "value"])

        for run in runs:
            ea = event_accumulator.EventAccumulator(
                run,
                size_guidance={event_accumulator.SCALARS: 0},  # 0 = load all
            )
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            rel = os.path.relpath(run, args.root)
            print(f"{rel}: {len(tags)} scalar tags", file=sys.stderr)
            for tag in tags:
                for ev in ea.Scalars(tag):
                    writer.writerow([rel, tag, ev.step, ev.wall_time, ev.value])
                    n_rows += 1

    print(f"\nwrote {n_rows} rows from {len(runs)} runs to {args.out}",
          file=sys.stderr)
    print("tags seen:", file=sys.stderr)
    seen = set()
    for run in runs:
        ea = event_accumulator.EventAccumulator(
            run, size_guidance={event_accumulator.SCALARS: 1}
        )
        ea.Reload()
        seen.update(ea.Tags().get("scalars", []))
    for t in sorted(seen):
        print(f"  {t}", file=sys.stderr)


if __name__ == "__main__":
    main()
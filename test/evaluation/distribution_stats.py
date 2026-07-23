"""Distribution check: generated class histogram + occupancy fraction vs the data.

Confirms the two generation failure modes on evidence:
  * class-marginal skew (Cars over-generated, Ground under) → compares the
    per-class fraction of occupied voxels, generated vs training data.
  * occupancy over/under-generation → compares the occupancy fraction
    (occupied / dense grid), generated vs data.

A healthy generator matches the data column. After switching class_weight to
uniform, the generated class fractions should move toward the data; after tuning
void_weight per level, the occupancy fractions should match.

    python test/evaluation/distribution_stats.py --n_gen 8 --n_data 64 --max_level 1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import get_device, class_counts, data_stats, CLASS_NAMES, N_CLS
from inference.inference import (
    compute_all_generations_dales, LEVEL0_NX, LEVEL0_NZ, LEVEL0_VOXEL_SIZE,
)


def print_hist(title, columns):
    """columns: list of (name, counts_array). Prints per-class fractions side by side."""
    names = [c[0] for c in columns]
    fracs = [c[1] / max(c[1].sum(), 1) for c in columns]
    print(f"\n{title}")
    print("  class          " + "".join(f"{n:>12}" for n in names))
    for k in range(N_CLS):
        row = "".join(f"{100*f[k]:>11.1f}%" for f in fracs)
        print(f"  {CLASS_NAMES[k]:<14}" + row)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="checkpoints/diffusion_models/")
    p.add_argument("--split", default="train", help="Split for the data reference.")
    p.add_argument("--n_gen", type=int, default=8, help="Samples to generate (one batch).")
    p.add_argument("--n_data", type=int, default=64, help="Real crops for the reference.")
    p.add_argument("--max_level", type=int, default=1, help="Generate up to this level.")
    p.add_argument("--base_res", type=int, default=16,
                   help="Coarsest pyramid .pt label (16.pt); level L → base_res*2**L.")
    p.add_argument("--upsample_fac", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.0)
    args = p.parse_args()
    device = get_device()

    print(f"Generating {args.n_gen} scene(s) up to level {args.max_level} …")
    gen_Xs, up_Xs = compute_all_generations_dales(
        src=args.src, max_level=args.max_level, eval_batch_size=args.n_gen,
        nx=LEVEL0_NX, nz=LEVEL0_NZ, voxel_size=LEVEL0_VOXEL_SIZE,
        occ_threshold=args.threshold, verbose=True,
    )
    grid0 = LEVEL0_NX * LEVEL0_NX * LEVEL0_NZ * args.n_gen  # dense level-0 voxels

    for level in range(args.max_level + 1):
        res = args.base_res * (args.upsample_fac ** level)
        d_counts, d_frac, used = data_stats(args.split, res, args.n_data, device)
        cols = [("data", d_counts)]

        gen = gen_Xs[level]
        cols.append((f"D{level}argmax", class_counts(gen, sample=False)))
        cols.append((f"D{level}sample", class_counts(gen, sample=True)))
        if level > 0 and up_Xs:
            cols.append((f"U{level}", class_counts(up_Xs[level - 1])))

        print(f"\n{'='*64}\nLevel {level}  ({res}.pt)  — data from {used} crops")
        print_hist(f"per-class fraction of occupied voxels", cols)
        if level == 0:
            gen_frac = gen.jdata.shape[0] / grid0
            print(f"\n  occupancy fraction:  data={d_frac:.3f}   D0={gen_frac:.3f}  "
                  f"(D0≫data ⇒ over-generating; ≪ ⇒ under)")
        else:
            print(f"\n  occupancy fraction (data, dense {res}.pt): {d_frac:.3f}   "
                  f"(compare D{level} vs upsampler with "
                  f"test/evaluation/upsampler_vs_diffusion.py)")
    print(f"\n{'='*64}")


if __name__ == "__main__":
    main()

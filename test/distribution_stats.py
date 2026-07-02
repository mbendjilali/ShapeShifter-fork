"""Distribution check: generated class histogram + occupancy fraction vs the data.

Confirms the two generation failure modes on evidence:
  * class-marginal skew (Cars over-generated, Ground under) → compares the
    per-class fraction of occupied voxels, generated vs training data.
  * occupancy over/under-generation → compares the occupancy fraction
    (occupied / dense grid), generated vs data.

A healthy generator matches the data column. After switching class_weight to
uniform, the generated class fractions should move toward the data; after tuning
void_weight per level, the occupancy fractions should match.

    python test/distribution_stats.py --n_gen 8 --n_data 64 --max_level 1
"""
import argparse
import os

import numpy as np
import torch

from common import get_device, list_crops, CLASS_NAMES
from utils.diffusion_tensor import DiffusionTensor
from inference.inference import (
    compute_all_generations_dales, LEVEL0_NX, LEVEL0_NZ, LEVEL0_VOXEL_SIZE,
)

N_CLS = 8


def class_counts(dt, sample=False):
    """Per-class counts over occupied voxels. sample=True draws class ~ softmax
    (reproduces the marginal); sample=False uses argmax (mode → majority bias)."""
    if dt.jdata.shape[0] == 0:
        return np.zeros(N_CLS, dtype=np.int64)
    probs = dt.jdata[:, 4:4 + N_CLS]
    if sample:
        p = probs.clamp(min=0)
        p = p / p.sum(-1, keepdim=True).clamp(min=1e-9)
        cls = torch.multinomial(p, 1).squeeze(-1).cpu().numpy()
    else:
        cls = probs.argmax(-1).cpu().numpy()
    return np.bincount(cls, minlength=N_CLS)


def data_stats(split, res, n_data, device):
    """Per-class counts (over occupied voxels) and occupancy fraction from real crops."""
    counts = np.zeros(N_CLS, dtype=np.int64)
    occ_num = occ_den = used = 0
    for cp in list_crops(split):
        f = os.path.join(cp, f"{res}.pt")
        if not os.path.exists(f):
            continue
        o = torch.load(f, weights_only=False)
        if not isinstance(o, DiffusionTensor):
            o = DiffusionTensor(o.grid, o.data)
        dt = DiffusionTensor(o.grid.to(device), o.data.to(device))
        counts += class_counts(dt)
        dense = dt.to_custom_dense(empty_fill='zero')          # add empty voxels
        occ_num += int((dense.jdata[:, -1] > 0).sum())
        occ_den += dense.jdata.shape[0]
        used += 1
        if used >= n_data:
            break
    return counts, (occ_num / max(occ_den, 1)), used


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
                  f"(compare D{level} vs upsampler with test/upsampler_vs_diffusion.py)")
    print(f"\n{'='*64}")


if __name__ == "__main__":
    main()

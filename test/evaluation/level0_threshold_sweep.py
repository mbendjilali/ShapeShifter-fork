"""Is level-0 generated occupancy threshold-insensitive, or a calibration offset?

The level-0 finding is that generated occupancy (0.3932) exceeds the training
crops' (0.2358) by ~1.44×. That claim is only about the *model* if the excess
cannot be removed by moving the pruning cut. If one global threshold shift brings
the generated mean onto the data's, the finding reduces to a decision-threshold
calibration offset — one scalar — which is a much weaker statement.

This is the same test that made the level-1 flood claim safe: the fix arm gave
0.558 at cut 0.0 and 0.557 at 0.5, so its occupancy was a committed decision, not
a near-miss on the cut.

Samples once, keeps the raw mask channel, and sweeps the cut over the *same*
volumes — so the sweep costs one sampling run, not one per threshold.

    python test/evaluation/level0_threshold_sweep.py --n_gen 32 --seed 0
"""
import argparse
import os
import sys

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

import utils.fvdb_diffusion as _fvdb_diffusion
import utils.model as _model
import utils.diffusion_tensor as _dt
sys.modules.setdefault("fvdb_diffusion", _fvdb_diffusion)
sys.modules.setdefault("model", _model)
sys.modules.setdefault("diffusion_tensor", _dt)

from common import get_device, list_crops, crop_pt, load_dt
from utils.fvdb_utils import grid_to_VDB
from inference.inference import (compute_canonical_base_grid, load_dales_diffusion,
                                 LEVEL0_NX, LEVEL0_NZ, LEVEL0_VOXEL_SIZE,
                                 LEVEL0_PYRAMID_RES)

FEATURES = 13


def real_mean_occ(split, nx, nz, device, n):
    """Occupancy of real crops inside the canonical volume (binary, no threshold)."""
    fr = []
    for cp in list_crops(split, n=n):
        f = crop_pt(cp, LEVEL0_PYRAMID_RES)
        if not os.path.exists(f):
            continue
        ijk = load_dt(f, device).grid.ijk.jdata.long()
        keep = ((ijk[:, 0] >= 0) & (ijk[:, 0] < nx) & (ijk[:, 1] >= 0) &
                (ijk[:, 1] < nx) & (ijk[:, 2] >= 0) & (ijk[:, 2] < nz))
        fr.append(int(keep.sum()) / (nx * nx * nz))
    return sum(fr) / max(len(fr), 1), len(fr)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_gen", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--src", default="checkpoints/diffusion_models/")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--nz", type=int, default=LEVEL0_NZ)
    p.add_argument("--batch_size", type=int, default=4)
    args = p.parse_args()

    dev = get_device()
    diff = (torch.load(args.ckpt, map_location=dev, weights_only=False).to(dev)
            if args.ckpt else load_dales_diffusion(0, args.src))
    diff.eval()
    nx, nz = LEVEL0_NX, args.nz

    # Sample once; keep the raw mask channel rather than a binarised volume.
    torch.manual_seed(args.seed)
    masks, done = [], 0
    while done < args.n_gen:
        b = min(args.batch_size, args.n_gen - done)
        base = compute_canonical_base_grid(nx=nx, nz=nz, voxel_size=LEVEL0_VOXEL_SIZE,
                                           batch=b, device=str(dev))
        out = diff.ddpm_sample(grid_to_VDB(base, torch.randn, [FEATURES]))
        for i in range(out.grid_count):
            masks.append(out.data[i].jdata[:, -1].flatten())
        done += b
        print(f"  sampled {done}/{args.n_gen}")
    m = torch.stack(masks)                       # (n_gen, n_voxels)

    train_mu, n_tr = real_mean_occ("train", nx, nz, dev, 10000)
    test_mu, n_te = real_mean_occ("test", nx, nz, dev, 10000)

    print(f"\n{'='*70}")
    print(f"level-0 generated occupancy vs pruning cut   (n_gen={args.n_gen}, "
          f"grid {nx}×{nx}×{nz})")
    print(f"  real reference — train {train_mu:.4f} (n={n_tr})   "
          f"test {test_mu:.4f} (n={n_te})\n")
    print(f"  {'cut':>7} {'occupancy':>10} {'Δ vs cut 0.0':>13}")
    base_occ = None
    for t in (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.9,
              0.99, 0.999, 0.9999, 0.99999):
        occ = (m > t).float().mean().item()
        if t == 0.0:
            base_occ = occ
        d = "" if base_occ is None else f"{occ - base_occ:+.4f}"
        print(f"  {t:>7.2f} {occ:>10.4f} {d:>13}")

    # Threshold that would match the data, if one exists in range.
    print()
    for name, target in (("train", train_mu), ("test", test_mu)):
        lo, hi = -1.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if (m > mid).float().mean().item() > target:
                lo = mid
            else:
                hi = mid
        t_star = (lo + hi) / 2
        occ_at = (m > t_star).float().mean().item()
        ok = abs(occ_at - target) < 1e-3
        print(f"  cut matching {name} mean {target:.4f}: t* = {t_star:+.4f}"
              f"  (gives {occ_at:.4f}){'' if ok else '  — NOT REACHABLE in [-1, 1]'}")

    # Bimodality: how much mask mass sits near the cut?
    near = ((m > -0.1) & (m < 0.1)).float().mean().item()
    frac_hi = (m > 0.9).float().mean().item()
    frac_lo = (m < -0.9).float().mean().item()
    print(f"\n  mask mass within ±0.1 of the cut : {near:.4f}")
    print(f"  mask mass > +0.9 (committed occupied): {frac_hi:.4f}")
    print(f"  mask mass < −0.9 (committed void)    : {frac_lo:.4f}")
    print(f"{'='*70}")
    print("Insensitive (occupancy flat across cuts, little mass near 0) ⇒ the model")
    print("is CONFIDENTLY over-occupying and the finding stands.")
    print("Sensitive (a t* in range matches the data) ⇒ the excess is a decision-")
    print("threshold calibration offset — one scalar — and the claim weakens.")


if __name__ == "__main__":
    main()

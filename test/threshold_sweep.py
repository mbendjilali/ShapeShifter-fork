"""Threshold sweep on D0: is the semantic skew downstream of occupancy over-generation?

Generates D0 once (no pruning), then sweeps the occupancy threshold on the mask
channel.  At each threshold we report the occupancy fraction and the per-class
marginal of the surviving voxels.  If, as the occupancy fraction drops toward the
data value, Vegetation's excess melts and Buildings/rare recover, then the
semantic skew rides on the spurious over-generated (low-confidence) voxels —
fixing occupancy fixes most of the semantics.

    python test/threshold_sweep.py
"""
import argparse

import torch

from common import get_device, CLASS_NAMES
from utils.fvdb_utils import grid_to_VDB
from utils.diffusion_tensor import DiffusionTensor
from inference.inference import (
    load_dales_diffusion, compute_canonical_base_grid,
    LEVEL0_NX, LEVEL0_NZ, LEVEL0_VOXEL_SIZE,
)
from distribution_stats import data_stats, N_CLS


def _row(fracs):
    return "  ".join(f"{CLASS_NAMES[k][:4]}={100*fracs[k]:4.1f}" for k in range(N_CLS))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="checkpoints/diffusion_models/")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n_gen", type=int, default=8)
    ap.add_argument("--n_data", type=int, default=64)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.0, 0.3, 0.5, 0.7, 0.9, 0.95])
    args = ap.parse_args()
    device = get_device()

    # --- data reference ---
    d_counts, d_frac, used = data_stats(args.split, 16, args.n_data, device)
    d_fracs = d_counts / max(d_counts.sum(), 1)

    # --- generate raw D0 (no prune) ---
    diff = load_dales_diffusion(0, args.src)
    diff.eval()
    grid = compute_canonical_base_grid(
        nx=LEVEL0_NX, nz=LEVEL0_NZ, voxel_size=LEVEL0_VOXEL_SIZE,
        batch=args.n_gen, device=device)
    noisy = grid_to_VDB(grid, torch.randn, [13])
    with torch.no_grad():
        raw = DiffusionTensor.from_vdb(diff.ddpm_sample(noisy))
    mask = raw.jdata[:, -1]
    cls = raw.jdata[:, 4:4 + N_CLS].argmax(-1)

    print(f"\n{'='*74}")
    print(f"data occ fraction = {d_frac:.3f}  (target); data from {used} crops")
    print(f"{'':>14}{_row(d_fracs)}   <- data marginal")
    print(f"{'thr':>5} {'occFrac':>8}  per-class % of occupied voxels")
    for t in args.thresholds:
        occ = mask > t
        frac = occ.float().mean().item()
        if occ.sum() == 0:
            print(f"{t:>5.2f} {frac:>8.3f}  (empty)")
            continue
        c = torch.bincount(cls[occ], minlength=N_CLS).float()
        c = (c / c.sum()).cpu().numpy()
        print(f"{t:>5.2f} {frac:>8.3f}  {_row(c)}")
    print(f"{'='*74}")


if __name__ == "__main__":
    main()

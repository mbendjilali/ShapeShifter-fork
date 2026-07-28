"""A2 (thesis Ch.4): unconditional level-0 coarse layouts.

Samples the level-0 diffusion from pure noise on the canonical dense base grid
(32x32x7 @ 3.2 m) and exports each layout as .laz (class-coloured) for CloudCompare.
Supports the claim that the coarse level produces believable layouts unconditionally.

    python test/generation/a2_d0_layouts.py --n 3 --seed 0
"""
import argparse
import os
import sys

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))          # test/ (for `common`)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))     # repo root
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

# Checkpoints pickle under old top-level module names.
import utils.fvdb_diffusion as _fvdb_diffusion
import utils.model as _model
import utils.diffusion_tensor as _dt
sys.modules.setdefault("fvdb_diffusion", _fvdb_diffusion)
sys.modules.setdefault("model", _model)
sys.modules.setdefault("diffusion_tensor", _dt)

from common import get_device, export_dt
from utils.diffusion_tensor import DiffusionTensor
from utils.fvdb_utils import grid_to_VDB
from inference.inference import (compute_canonical_base_grid,
                                 LEVEL0_NX, LEVEL0_NZ, LEVEL0_VOXEL_SIZE)

CKPT = "checkpoints/diffusion_models/dales_0_23-07-15:09_best.pt"  # fresh level-0, by path
FEATURES = 13  # [offset(3), intensity(1), class(8), void(1)]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=3, help="Number of layouts to sample.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="output/tests/A2")
    p.add_argument("--occ_threshold", type=float, default=0.0)
    args = p.parse_args()

    dev = get_device()
    torch.manual_seed(args.seed)
    print(f"Loading level-0 diffusion: {CKPT}")
    diff = torch.load(CKPT, map_location=dev, weights_only=False).to(dev)
    diff.eval()
    print(f"  n_classes={diff.n_classes}  grid={LEVEL0_NX}x{LEVEL0_NX}x{LEVEL0_NZ} "
          f"@ {LEVEL0_VOXEL_SIZE}m  n={args.n} seed={args.seed}\n")

    base = compute_canonical_base_grid(nx=LEVEL0_NX, nz=LEVEL0_NZ,
                                       voxel_size=LEVEL0_VOXEL_SIZE,
                                       batch=args.n, device=dev)
    noisy = grid_to_VDB(base, torch.randn, [FEATURES])
    gen = diff.ddpm_sample(noisy)

    os.makedirs(args.out, exist_ok=True)
    for i in range(gen.grid_count):
        g = DiffusionTensor(gen.grid[i], gen.data[i]).get_global().remove_mask(
            threshold=args.occ_threshold)
        path = os.path.join(args.out, f"layout_{args.seed}_{i}.laz")
        if g.jdata.shape[0] == 0:
            print(f"  layout {i}: void — skipped")
            continue
        export_dt(g, path)
        print(f"  wrote {path}  ({g.jdata.shape[0]:,} pts)")


if __name__ == "__main__":
    main()

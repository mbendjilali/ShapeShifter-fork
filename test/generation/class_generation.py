"""Class-conditional generation from noise.

Generates from pure noise on the canonical grid with the class channel clamped to
each target class throughout sampling, then exports → clamp_class{cid}.laz.
Diagnostic for "does the model know what class-X geometry looks like".

    python test/generation/class_generation.py --levels 0 --class_ids 7
    python test/generation/class_generation.py --levels 0 --class_ids 0 7 --seed 1
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import fvdb.nn as fvnn

from common import export_dt, CLASS_NAMES
from utils.diffusion_tensor import DiffusionTensor
from utils.helper import reverse_from
from inference.inference import (
    load_dales_diffusion, compute_canonical_base_grid,
    LEVEL0_NX, LEVEL0_NZ, LEVEL0_VOXEL_SIZE,
)


@torch.no_grad()
def run(diff, nx, nz, voxel_size, class_ids, seed, out_dir, occ_threshold=0.0,
        clamp_mode="hard"):
    device = diff.device
    C = diff.n_classes
    t_start = diff.max_T / diff.timesteps
    base_grid = compute_canonical_base_grid(
        nx=nx, nz=nz, voxel_size=voxel_size, batch=1, device=str(device))
    N = base_grid.ijk.jdata.shape[0]

    os.makedirs(out_dir, exist_ok=True)
    for cid in class_ids:
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
        t0 = time.time()
        torch.manual_seed(seed)
        jdata = torch.randn(N, 4 + C + 1, device=device)
        noisy = fvnn.VDBTensor(base_grid, base_grid.jagged_like(jdata))
        onehot = torch.zeros(N, C, device=device)
        onehot[:, cid] = 1.0
        x0 = reverse_from(diff, noisy, t_start=t_start, clamp_onehot=onehot,
                          clamp_mode=clamp_mode)
        occ = DiffusionTensor.from_vdb(x0).get_global().remove_mask(
            threshold=occ_threshold)
        tag = "" if clamp_mode == "hard" else f"_{clamp_mode}"
        out_path = os.path.join(out_dir, f"clamp_class{cid}_{name}{tag}.laz")
        export_dt(occ, out_path)
        frac = 100.0 * occ.jdata.shape[0] / N
        print(f"  class={cid} ({name})  pts={occ.jdata.shape[0]} ({frac:.1f}% of "
              f"{N} grid voxels)  {time.time()-t0:.1f}s → {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", type=int, nargs="+", default=[0],
                   help="Diffusion level(s) to run.")
    p.add_argument("--src", default="checkpoints/diffusion_models/")
    p.add_argument("--class_ids", type=int, nargs="+", default=[0, 7],
                   help="Class indices (0=Ground … 7=Buildings).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nx", type=int, default=LEVEL0_NX)
    p.add_argument("--nz", type=int, default=LEVEL0_NZ)
    p.add_argument("--voxel_size", type=float, default=LEVEL0_VOXEL_SIZE)
    p.add_argument("--out", default=None)
    p.add_argument("--occ_threshold", type=float, default=0.0,
                   help="Pruning cut on the decoded mask. 0.0 matches "
                        "a2_d0_layouts.py and the level-0 inference path; the "
                        "occupancy fraction is only comparable at equal cuts.")
    p.add_argument("--clamp_mode", choices=["hard", "renorm"], default="hard",
                   help="hard: overwrite class channels (also asserts occupancy "
                        "— the original A3 probe). renorm: redistribute the "
                        "model's own class-sum onto the target class, leaving "
                        "occupancy free (see reverse_from).")
    args = p.parse_args()

    for level in args.levels:
        diff = load_dales_diffusion(level, args.src)
        diff.eval()
        out_dir = args.out or f"output/tests/level{level}"
        print(f"\nlevel={level}  classes={args.class_ids}  seed={args.seed}  "
              f"grid={args.nx}×{args.nx}×{args.nz} @ {args.voxel_size}m")
        run(diff, args.nx, args.nz, args.voxel_size, args.class_ids, args.seed, out_dir,
            occ_threshold=args.occ_threshold, clamp_mode=args.clamp_mode)
        print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()

"""Noise-then-denoise reconstruction of a real crop (was run_tests.py test A).

Adds noise at each t in --t_list to a held-out crop, reverse-denoises with the
level's diffusion, and exports the result → recon_t{t}.laz.  A sanity check that
the denoiser recovers the crop from a given noise level.

    python test/reconstruct.py --level 0 --crop 5080_54435_x0000_y0050
    python test/reconstruct.py --level 0 --crop <id> --t_list 0.3 0.6 1.0
"""
import argparse
import os
import time
from pathlib import Path

import torch

from common import get_device, resolve_crop_path, export_dt
from utils.diffusion_tensor import DiffusionTensor
from utils.helper import reverse_from
from inference.inference import load_dales_diffusion, LEVEL0_PYRAMID_RES


@torch.no_grad()
def run(diff, crop_path, level, pyramid_res, t_list, out_dir):
    resolution = pyramid_res * (2 ** level)
    pt_file = Path(crop_path) / f"{resolution}.pt"
    if not pt_file.exists():
        raise FileNotFoundError(
            f"Crop file not found: {pt_file} (resolution {resolution} for level {level}).")

    print(f"[reconstruct] loading {pt_file}")
    obj = torch.load(pt_file, weights_only=False)
    if not isinstance(obj, DiffusionTensor):
        obj = DiffusionTensor(obj.grid, obj.data)
    X_clean = DiffusionTensor(obj.grid.to(diff.device), obj.data.to(diff.device))

    os.makedirs(out_dir, exist_ok=True)
    for t in t_list:
        t0 = time.time()
        X_copy = DiffusionTensor(X_clean.grid, X_clean.grid.jagged_like(X_clean.jdata.clone()))
        tt = torch.full((X_copy.jdata.shape[0],), float(t), device=diff.device)
        noisy, _ = diff.q_sample(X_copy, tt)
        rec = reverse_from(diff, noisy, t_start=t)
        rec = DiffusionTensor.from_vdb(rec).get_global().remove_mask()
        out_path = os.path.join(out_dir, f"recon_t{t:.2f}.laz")
        export_dt(rec, out_path)
        print(f"  t={t:.2f}  pts={rec.jdata.shape[0]}  {time.time()-t0:.1f}s → {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--src", default="checkpoints/diffusion_models/")
    p.add_argument("--crop", required=True,
                   help="Full path or bare crop ID (resolved via --split).")
    p.add_argument("--split", default="test")
    p.add_argument("--t_list", type=float, nargs="+", default=[1.0],
                   help="Noise fractions to reconstruct from (0.0–1.0).")
    p.add_argument("--pyramid_res", type=int, default=LEVEL0_PYRAMID_RES,
                   help="Coarsest .pt label; crop file at level N is pyramid_res*2**level.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    diff = load_dales_diffusion(args.level, args.src)
    diff.eval()
    crop_path = resolve_crop_path(args.crop, args.split)
    out_dir = args.out or f"output/tests/level{args.level}"
    print(f"level={args.level}  crop={crop_path}  t={args.t_list}  "
          f"max_T={diff.max_T}/{diff.timesteps}")
    run(diff, crop_path, args.level, args.pyramid_res, args.t_list, out_dir)
    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()

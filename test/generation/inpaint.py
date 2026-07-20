"""Hide a region of a real crop, then ask the diffusion to reconstruct it.

(Was run_tests.py test C — the only test in that monolith never ported to the
split scripts.  Extracted verbatim in behaviour, rewritten against common.py.)

The experiment asks: can the model reconstruct information that was deliberately
removed from a crop?

Steps
-----
1. Load the clean crop and densify it (empty_fill='zero' matches how the model was
   trained — no blurred neighbours leaking into empty slots, so the distribution
   at inference is consistent).
2. Mark every voxel in the first ``--hide_fraction`` of the X-axis range as empty
   (features = 0, mask = -1).  The rest of the crop stays clean.
3. Add Gaussian noise at level t to the whole (now partially-holed) grid.
4. Reverse-diffuse from t → 0.
5. Export three .laz files for side-by-side comparison in a viewer:
     original.laz            — ground truth (nothing hidden)
     hidden.laz              — the crop with the region removed (the "hole")
     recon_h{H}_t{T}.laz     — the model's reconstruction

Reading it: the model passes if the hole refills with structure that is
class-consistent and continuous with the surviving half.  A hole that stays empty
means the denoiser ignores context; a hole that fills with uniform blob means it
falls back to the class marginal.

    python test/generation/inpaint.py --level 0 --crop 5080_54435_x0000_y0050
    python test/generation/inpaint.py --level 0 --crop <id> --hide_fraction 0.3 --t 0.6
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from common import crop_pt, load_dt, resolve_crop_path, export_dt
from utils.diffusion_tensor import DiffusionTensor
from utils.helper import reverse_from
from inference.inference import load_dales_diffusion, LEVEL0_PYRAMID_RES


@torch.no_grad()
def run(diff, crop_path, level, pyramid_res, t, hide_fraction, out_dir):
    resolution = pyramid_res * (2 ** level)
    pt_file = crop_pt(crop_path, resolution)
    if not os.path.exists(pt_file):
        raise FileNotFoundError(
            f"Crop file not found: {pt_file} (resolution {resolution} for level {level}).")

    print(f"[inpaint] loading {pt_file}")
    X_sparse = load_dt(pt_file, diff.device)

    # Densify to a regular bbox grid — empty slots get mask = -1, features = 0.
    X_dense = X_sparse.to_custom_dense(empty_fill='zero')

    os.makedirs(out_dir, exist_ok=True)

    orig_path = os.path.join(out_dir, "original.laz")
    export_dt(X_sparse.get_global().remove_mask(), orig_path)
    print(f"  exported original    → {orig_path}")

    # --- define the hidden region: voxels in the first hide_fraction of X ---
    ijk = X_dense.grid.ijk.jdata          # (N_voxels, 3) integer voxel coords
    i_min = ijk[:, 0].min().item()
    i_max = ijk[:, 0].max().item()
    i_split = i_min + int(round((i_max - i_min) * hide_fraction))
    hidden_mask = ijk[:, 0] <= i_split    # True for voxels to be erased

    print(f"  hiding {int(hidden_mask.sum())}/{ijk.shape[0]} voxels  "
          f"(X ∈ [{i_min}, {i_split}], fraction={hide_fraction:.2f})")

    # Zero all feature channels and flip mask to -1 for the hidden voxels
    holed_data = X_dense.jdata.clone()
    holed_data[hidden_mask, :-1] = 0.0   # zero offset, intensity, class_probs
    holed_data[hidden_mask, -1] = -1.0   # mark as empty
    X_holed = DiffusionTensor(X_dense.grid, X_dense.grid.jagged_like(holed_data))

    hidden_path = os.path.join(out_dir, "hidden.laz")
    export_dt(X_holed.get_global().remove_mask(), hidden_path)
    print(f"  exported holed input → {hidden_path}")

    # --- noise the whole grid at t, then denoise back to 0 ---
    tt = torch.full((X_holed.jdata.shape[0],), float(t), device=diff.device)
    noisy, _ = diff.q_sample(X_holed, tt)

    t0 = time.time()
    recon = DiffusionTensor.from_vdb(
        reverse_from(diff, noisy, t_start=t)).get_global().remove_mask()

    out_path = os.path.join(out_dir, f"recon_h{hide_fraction:.1f}_t{t:.2f}.laz")
    export_dt(recon, out_path)
    print(f"  reconstruction: pts={recon.jdata.shape[0]}  "
          f"{time.time()-t0:.1f}s → {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--src", default="checkpoints/diffusion_models/")
    p.add_argument("--crop", required=True,
                   help="Full path or bare crop ID (resolved via --split).")
    p.add_argument("--split", default="test")
    p.add_argument("--t", type=float, default=1.0,
                   help="Noise fraction to reconstruct from (0.0–1.0).")
    p.add_argument("--hide_fraction", type=float, default=0.5,
                   help="Fraction of the X-axis range to erase before denoising.")
    p.add_argument("--pyramid_res", type=int, default=LEVEL0_PYRAMID_RES,
                   help="Coarsest .pt label; crop file at level N is pyramid_res*2**level.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    diff = load_dales_diffusion(args.level, args.src)
    diff.eval()
    crop_path = resolve_crop_path(args.crop, args.split)
    out_dir = args.out or f"output/tests/level{args.level}"
    print(f"level={args.level}  crop={crop_path}  hide={args.hide_fraction}  t={args.t}  "
          f"max_T={diff.max_T}/{diff.timesteps}")
    run(diff, crop_path, args.level, args.pyramid_res, args.t, args.hide_fraction, out_dir)
    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()

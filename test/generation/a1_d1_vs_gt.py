"""A1 (thesis Ch.4): level-1 D1 sample vs ground truth of the SAME tile.

Exports per-crop .laz (coloured by class, shared world coords) for CloudCompare:
  __coarse_D0  16.pt input · __upsampler  upsampler-only · __gen_D1  diffusion-
  refined (the D1 sample) · __gt_D1  ground truth 32.pt. gen_D1 vs gt_D1 = figure.

    python test/generation/a1_d1_vs_gt.py --crops 5100_54490_x0400_y0300 ...
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

from common import (get_device, resolve_crop_path, crop_pt, load_dt, export_dt,
                    level_resolutions, has_levels)
from utils.diffusion_tensor import DiffusionTensor
from utils.helper import reverse_from

CKPT = "checkpoints/diffusion_models/dales_1_08-07-10:06_best.pt"  # verified, by path
LEVEL = 1


def export_global(dt, path, prune=True):
    g = dt.get_global()
    if prune:
        g = g.remove_mask()
    export_dt(g, path)
    print(f"    wrote {path}  ({g.jdata.shape[0]:,} pts)")


@torch.no_grad()
def run_crop(diff, crop_id, out_dir, split, base_res, upsample_fac, steps):
    cp = resolve_crop_path(crop_id, split)
    res1, res2 = level_resolutions(LEVEL, base_res, upsample_fac)  # 16, 32
    if not has_levels(cp, res1, res2):
        print(f"  [skip] {crop_id}: missing {res1}.pt or {res2}.pt")
        return
    dev, name = diff.device, os.path.basename(cp)
    print(f"  {name}: coarse={res1}.pt → fine={res2}.pt")

    X = load_dt(crop_pt(cp, res1), dev)
    X0_fine = load_dt(crop_pt(cp, res2), dev)
    X_UP = X.trilinear_upsample(upsample_fac)

    # Upsampler-only then diffusion-refined — the inference/eval path.
    up = DiffusionTensor.from_vdb(diff.model_upsampler(X, X_UP))
    t_start = diff.max_T / diff.timesteps
    times = torch.full((up.jdata.shape[0],), float(t_start), device=dev)
    XT = diff.q_sample(up, times)[0]
    d = DiffusionTensor.from_vdb(
        reverse_from(diff, XT, t_start=t_start, steps=steps, X_Blur=up))

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, name)
    export_global(X,       f"{stem}__coarse_D0.laz")
    export_global(up,      f"{stem}__upsampler.laz")
    export_global(d,       f"{stem}__gen_D1.laz")
    export_global(X0_fine, f"{stem}__gt_D1.laz")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crops", nargs="+", required=True, help="Crop IDs or dirs.")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="output/tests/A1")
    p.add_argument("--base_res", type=int, default=16)
    p.add_argument("--upsample_fac", type=int, default=2)
    p.add_argument("--steps", type=int, default=None, help="Reverse steps (default: full).")
    args = p.parse_args()

    dev = get_device()
    print(f"Loading level-{LEVEL} diffusion (+embedded upsampler): {CKPT}")
    diff = torch.load(CKPT, map_location=dev, weights_only=False).to(dev)
    diff.eval()
    assert diff.model_upsampler is not None, "checkpoint has no embedded upsampler"
    print(f"  n_classes={diff.n_classes} max_T={diff.max_T}/{diff.timesteps}\n")

    for c in args.crops:
        run_crop(diff, c, args.out, args.split, args.base_res, args.upsample_fac, args.steps)


if __name__ == "__main__":
    main()

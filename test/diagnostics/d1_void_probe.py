"""Why does D1 flood? Probe its void prediction on GT-input vs upsampler-input.

D1 floods occupancy at inference (occFrac→1.0) yet trains to OccIoU~0.72 — a
train/inference input mismatch.  This probes the single-step x0 prediction: feed a
noised version of (a) the fine GT and (b) the upsampler output, and report the
P(void) the model assigns to TRUE-void vs TRUE-occupied voxels.

Reading it:
  * GT-input separates (P(void|void) ≫ P(void|occ)) but UP-input collapses
    (P(void) low on both) ⇒ the upsampler's blurry void features are OOD for the
    void head — that's the flood cause, and void_weight alone won't fix it.
  * both separate ⇒ the single-step head is fine; the flood is in the reverse
    trajectory / decode.

    python test/diagnostics/d1_void_probe.py --level 1 --n_crops 12
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from common import (
    get_device, list_crops, has_levels, load_levelN_inputs, level_resolutions,
)
from utils.diffusion_tensor import DiffusionTensor
from inference.inference import load_dales_diffusion


@torch.no_grad()
def probe(diff, crops, res1, res2, upsample_fac, ts, device):
    n_cls = diff.n_classes
    # accumulators: (source, t) -> [sum P(void|void), sum P(void|occ), sum occ_frac, n]
    acc = {}
    for cp in crops:
        if not has_levels(cp, res1, res2):
            continue
        X, X_UP, X0 = load_levelN_inputs(cp, res1, res2, upsample_fac, device)
        gt_void = X0.jdata[:, -1] <= 0
        if gt_void.all() or (~gt_void).all():
            continue
        up = DiffusionTensor.from_vdb(diff.model_upsampler(X, X_UP))

        for src_name, src in (("GT", X0), ("UP", up)):
            for t in ts:
                times = torch.full((src.jdata.shape[0],), float(t), device=device)
                noised = diff.q_sample(src, times)[0]
                pred = diff.model(noised, times)
                cat = torch.cat([pred.jdata[:, 4:4 + n_cls], pred.jdata[:, -1:]], dim=1)
                p_void = F.softmax(cat, dim=1)[:, n_cls]
                key = (src_name, t)
                a = acc.setdefault(key, [0.0, 0.0, 0.0, 0])
                a[0] += p_void[gt_void].mean().item()
                a[1] += p_void[~gt_void].mean().item()
                a[2] += (p_void < 0.5).float().mean().item()   # predicted-occupied fraction
                a[3] += 1
    return acc


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--src", default="checkpoints/diffusion_models/")
    p.add_argument("--split", default="test")
    p.add_argument("--n_crops", type=int, default=12)
    p.add_argument("--base_res", type=int, default=16)
    p.add_argument("--upsample_fac", type=int, default=2)
    p.add_argument("--ts", type=float, nargs="+", default=[0.02, 0.1, 0.3])
    args = p.parse_args()
    device = get_device()

    res1, res2 = level_resolutions(args.level, args.base_res, args.upsample_fac)
    diff = load_dales_diffusion(args.level, args.src)
    diff.eval()
    assert diff.model_upsampler is not None
    print(f"  max_T={diff.max_T}/{diff.timesteps}  coarse={res1}.pt → fine={res2}.pt")

    crops = list_crops(args.split, n=args.n_crops)
    acc = probe(diff, crops, res1, res2, args.upsample_fac, args.ts, device)

    print(f"\n{'='*66}")
    print("model's single-step P(void), averaged over crops")
    print(f"{'input':>6} {'t':>6} {'P(void|TRUE void)':>18} {'P(void|TRUE occ)':>17} "
          f"{'pred occ frac':>14}")
    for (src_name, t), (s_void, s_occ, s_frac, n) in sorted(acc.items()):
        if n == 0:
            continue
        print(f"{src_name:>6} {t:>6.2f} {s_void/n:>18.3f} {s_occ/n:>17.3f} {s_frac/n:>14.3f}")
    print(f"{'='*66}")
    print("healthy void head: P(void|TRUE void) ≫ P(void|TRUE occ); pred occ frac ≈ data")


if __name__ == "__main__":
    main()

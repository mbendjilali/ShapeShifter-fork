"""Does the level-N diffusion actually improve on the upsampler alone?

For held-out crops we build the level-N inputs (coarse X, its trilinear upsample
X_UP, and the fine GT X0), then compare two occupancy/semantic predictions
against the GT over the *same* fine voxel set:

  upsampler-only    : model_upsampler(X, X_UP)                     (deterministic)
  diffusion-refined : reverse_from(diff, q_sample(upsampler_out, max_T))

Both decode occupancy the same way — mask channel > threshold — so the IoUs are
comparable (the upsampler regresses the mask with MSE; the diffusion outputs
mask = 1 − 2·P(void)).

Reading it:
  * diffusion IoU ≫ upsampler IoU  → diff-N is correcting structure; max_T is
    doing real work.
  * diffusion IoU ≈ upsampler IoU  → diff-N inherits occupancy from the upsampler
    and barely regenerates it — a sign max_T is too low to add structural value
    (or the upsampler is the bottleneck).  Semantic accuracy may still improve,
    which is the low-noise refinement diff-N is meant for.

    python test/upsampler_vs_diffusion.py --level 1 --n_crops 16
    python test/upsampler_vs_diffusion.py --level 1 --n_crops 8 --steps 60   # faster
"""
import argparse
import os
import time

import torch

from common import get_device, list_crops, occ_iou
from utils.diffusion_tensor import DiffusionTensor
from utils.helper import reverse_from
from inference.inference import load_dales_diffusion


def _load_dt(path: str, device: str) -> DiffusionTensor:
    o = torch.load(path, weights_only=False)
    if not isinstance(o, DiffusionTensor):
        o = DiffusionTensor(o.grid, o.data)
    return DiffusionTensor(o.grid.to(device), o.data.to(device))


def load_levelN_inputs(crop_path, res1, res2, upsample_fac, device):
    """Reproduce DALESDataset.load_crop_levelN for one crop, no dataset object."""
    X = _load_dt(os.path.join(crop_path, f"{res1}.pt"), device)
    X0_fine = _load_dt(os.path.join(crop_path, f"{res2}.pt"), device)
    X_UP = X.trilinear_upsample(upsample_fac)
    X0 = DiffusionTensor.fill_upsampled_with_gt(X_UP, X0_fine)
    return X, X_UP, X0


@torch.no_grad()
def evaluate(diff, crops, res1, res2, upsample_fac, threshold, steps, device):
    n_cls = diff.n_classes
    t_start = diff.max_T / diff.timesteps
    rows = []
    for cp in crops:
        f1 = os.path.join(cp, f"{res1}.pt")
        f2 = os.path.join(cp, f"{res2}.pt")
        if not (os.path.exists(f1) and os.path.exists(f2)):
            continue
        t0 = time.time()
        X, X_UP, X0 = load_levelN_inputs(cp, res1, res2, upsample_fac, device)

        gt_occ = X0.jdata[:, -1] > 0
        gt_cls = X0.jdata[:, 4:4 + n_cls].argmax(-1)
        if not gt_occ.any():
            continue

        # --- upsampler only ---
        up = DiffusionTensor.from_vdb(diff.model_upsampler(X, X_UP))
        up_occ = up.jdata[:, -1] > threshold
        up_cls = up.jdata[:, 4:4 + n_cls].argmax(-1)

        # --- diffusion-refined (same path as inference generate_input) ---
        times = torch.full((up.jdata.shape[0],), float(t_start), device=device)
        new_XT = diff.q_sample(up, times)[0]
        d = DiffusionTensor.from_vdb(
            reverse_from(diff, new_XT, t_start=t_start, steps=steps, X_Blur=up))
        d_occ = d.jdata[:, -1] > threshold
        d_cls = d.jdata[:, 4:4 + n_cls].argmax(-1)

        iou_up = occ_iou(up_occ, gt_occ)
        iou_d = occ_iou(d_occ, gt_occ)
        acc_up = (up_cls[gt_occ] == gt_cls[gt_occ]).float().mean().item()
        acc_d = (d_cls[gt_occ] == gt_cls[gt_occ]).float().mean().item()
        # Occupancy fraction over the (fixed) fine grid — flooding shows up as
        # frac_d ≫ frac_gt (D1 saturating to solid "lego blocks").
        frac_gt = gt_occ.float().mean().item()
        frac_up = up_occ.float().mean().item()
        frac_d = d_occ.float().mean().item()
        rows.append((os.path.basename(cp), iou_up, iou_d, acc_up, acc_d,
                     frac_gt, frac_up, frac_d))
        print(f"  {os.path.basename(cp):<28} occIoU up={iou_up:.3f} diff={iou_d:.3f} "
              f"| occFrac gt={frac_gt:.2f} up={frac_up:.2f} diff={frac_d:.2f} "
              f"| clsAcc up={acc_up:.3f} diff={acc_d:.3f}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, default=1, help="Diffusion level (>0).")
    p.add_argument("--src", default="checkpoints/diffusion_models/",
                   help="Directory with diffusion checkpoints.")
    p.add_argument("--split", default="test")
    p.add_argument("--n_crops", type=int, default=16)
    p.add_argument("--base_res", type=int, default=16,
                   help="Coarsest pyramid .pt label (16.pt = 3.2m).")
    p.add_argument("--upsample_fac", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Occupancy threshold on the mask channel (same for both).")
    p.add_argument("--steps", type=int, default=None,
                   help="Reverse-diffusion steps (default: full ~max_T).")
    args = p.parse_args()

    assert args.level > 0, "This test compares against the upsampler; use level > 0."
    device = get_device()
    res1 = args.base_res * (args.upsample_fac ** (args.level - 1))
    res2 = args.upsample_fac * res1

    print(f"Loading level-{args.level} diffusion (+ upsampler) from {args.src} …")
    diff = load_dales_diffusion(args.level, args.src)
    diff.eval()
    assert diff.model_upsampler is not None, "level-N diffusion has no upsampler loaded."
    print(f"  max_T={diff.max_T}/{diff.timesteps} (t_start={diff.max_T/diff.timesteps:.3f})  "
          f"coarse={res1}.pt → fine={res2}.pt\n")

    crops = list_crops(args.split, n=args.n_crops)
    rows = evaluate(diff, crops, res1, res2, args.upsample_fac,
                    args.threshold, args.steps, device)
    if not rows:
        print("No usable crops found.")
        return

    import statistics as st
    iou_up = st.mean(r[1] for r in rows)
    iou_d = st.mean(r[2] for r in rows)
    acc_up = st.mean(r[3] for r in rows)
    acc_d = st.mean(r[4] for r in rows)
    frac_gt = st.mean(r[5] for r in rows)
    frac_up = st.mean(r[6] for r in rows)
    frac_d = st.mean(r[7] for r in rows)
    print(f"\n{'='*60}\nAveraged over {len(rows)} crops")
    print(f"  occupancy IoU :  upsampler={iou_up:.3f}   diffusion={iou_d:.3f}   "
          f"Δ={iou_d-iou_up:+.3f}")
    print(f"  occ fraction  :  gt={frac_gt:.3f}   upsampler={frac_up:.3f}   "
          f"diffusion={frac_d:.3f}   (diff≫gt ⇒ flooding to lego blocks)")
    print(f"  semantic acc  :  upsampler={acc_up:.3f}   diffusion={acc_d:.3f}   "
          f"Δ={acc_d-acc_up:+.3f}")
    if iou_d - iou_up > 0.02:
        verdict = "diffusion is correcting structure — the refinement adds value"
    elif iou_d - iou_up < -0.02:
        verdict = ("diffusion DEGRADES the upsampler occupancy — the refinement is "
                   "net-negative (suspect the X_Blur conditioning / max_T)")
    else:
        verdict = "diffusion ≈ upsampler on occupancy — refinement adds little structure"
    print(f"  → {verdict}\n{'='*60}")


if __name__ == "__main__":
    main()

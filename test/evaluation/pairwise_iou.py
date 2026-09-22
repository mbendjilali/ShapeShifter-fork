"""Pairwise-IoU diversity, on upstream ShapeShifter's protocol.

Reports average pairwise **1 − IoU** over dense occupancy volumes — higher =
more diverse — using `pairwise_IoU_dist` from `src/eval/patch_utils.py`, which is
vendored verbatim from Sin3DM so the protocol provenance is explicit.

Three rows, and the middle one is what makes the number readable:

    generated : diversity among unconditional level-0 samples
    real      : diversity among real level-0 crops — THE REFERENCE SCALE
    cross     : generated vs real; guards the failure mode where a model looks
                "diverse" by emitting noise unlike anything in the data

**Read the caveat before quoting this against the paper.** Upstream is
*single-shape* generation: train on one shape, generate variations, and pairwise
1−IoU measures how much those variations differ. This model is trained on ~1000
crops of a city and samples over *different scenes*, so a high value is trivially
reachable and means something else. The claim the numbers support is
`generated ≈ real`, i.e. the model's spread matches the data's — not a number
directly commensurable with upstream's table.

Diversity is not fidelity. A good score here says nothing about whether the
geometry is structured; see thesis A3 for the joint class-geometry limits.

    python test/evaluation/pairwise_iou.py --n_gen 16 --n_real 16
"""
import argparse
import os
import sys

_HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))          # test/
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))     # repo root
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "utils"),
           os.path.join(_ROOT, "src", "eval")):
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
from patch_utils import pairwise_IoU_dist          # vendored Sin3DM protocol
from utils.diffusion_tensor import DiffusionTensor
from utils.fvdb_utils import grid_to_VDB
from inference.inference import (compute_canonical_base_grid, load_dales_diffusion,
                                 LEVEL0_NX, LEVEL0_NZ, LEVEL0_VOXEL_SIZE,
                                 LEVEL0_PYRAMID_RES)

FEATURES = 13  # [offset(3), intensity(1), class(8), void(1)]


def to_dense(ijk, nx, nz, device, report_clip=False):
    """Boolean (nx, nx, nz) volume from integer voxel coordinates.

    Voxels outside the canonical volume are dropped. Generated samples live on
    exactly this grid so nothing is lost, but REAL crops extend above z=nz
    (level-0 crops reach k≈12 against nz=7), so their occupancy is measured only
    within the shared 0–nz·voxel_size band. That is the fair like-for-like
    volume — the model cannot produce anything above it — but the clipped
    fraction must be reported, hence report_clip.
    """
    vol = torch.zeros(nx, nx, nz, dtype=torch.bool, device=device)
    ijk = ijk.long()
    keep = ((ijk[:, 0] >= 0) & (ijk[:, 0] < nx) &
            (ijk[:, 1] >= 0) & (ijk[:, 1] < nx) &
            (ijk[:, 2] >= 0) & (ijk[:, 2] < nz))
    n_total = ijk.shape[0]
    ijk = ijk[keep]
    vol[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    if report_clip:
        return vol, n_total - int(keep.sum())
    return vol


def occ_stats(vols, label):
    """Per-sample occupancy fraction: mean, std, quartiles."""
    f = torch.stack([v.float().mean() for v in vols])
    q = torch.quantile(f, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=f.device))
    print(f"  {label:<12} n={len(vols):4d}  mean={f.mean():.4f}  std={f.std():.4f}  "
          f"min={q[0]:.3f}  q1={q[1]:.3f}  med={q[2]:.3f}  q3={q[3]:.3f}  max={q[4]:.3f}")
    return float(f.mean()), float(f.std())


def random_matched(vols, stratify_z=False):
    """Null volumes with the same occupancy count as each input.

    stratify_z=False : voxels placed uniformly in the volume. This is the
        analytic p/(2−p) baseline — but it ignores that every aerial crop has
        ground at low z, so any two real crops agree there for free.
    stratify_z=True  : counts preserved **per z-slice**, so the trivial
        ground-plane agreement is kept and only lateral structure is destroyed.
        The stricter, more honest null for this data.
    """
    out = []
    for v in vols:
        r = torch.zeros_like(v)
        if stratify_z:
            for k in range(v.shape[2]):
                n = int(v[:, :, k].sum())
                if n == 0:
                    continue
                idx = torch.randperm(v.shape[0] * v.shape[1], device=v.device)[:n]
                r[:, :, k].view(-1)[idx] = True
        else:
            n = int(v.sum())
            idx = torch.randperm(v.numel(), device=v.device)[:n]
            r.view(-1)[idx] = True
        out.append(r)
    return out


def cross_pairwise(a_list, b_list):
    """Average 1 − IoU between every a and every b (no self-pairs by construction)."""
    vals = []
    for a in a_list:
        inter = torch.logical_and(a, torch.stack(b_list)).sum(dim=(1, 2, 3))
        union = torch.logical_or(a, torch.stack(b_list)).sum(dim=(1, 2, 3)).clamp(min=1)
        vals.append((1.0 - inter / union).mean().item())
    return sum(vals) / max(len(vals), 1)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_gen", type=int, default=16, help="Unconditional samples to draw.")
    p.add_argument("--n_real", type=int, default=16, help="Real crops for the reference.")
    p.add_argument("--src", default="checkpoints/diffusion_models/")
    p.add_argument("--ckpt", default=None, help="Pin a checkpoint instead of the latest.")
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--occ_threshold", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--nz", type=int, default=LEVEL0_NZ,
                   help="Z depth of the sampling grid. Training volumes are the "
                        "crop's own bbox (median 9-10 deep), NOT the canonical 7 — "
                        "sweep this to separate model over-occupancy from a grid "
                        "that is too shallow.")
    args = p.parse_args()

    dev = get_device()
    if args.ckpt:
        print(f"checkpoint (pinned): {args.ckpt}")
        diff = torch.load(args.ckpt, map_location=dev, weights_only=False).to(dev)
    else:
        diff = load_dales_diffusion(0, args.src)
    diff.eval()
    nx, nz = LEVEL0_NX, args.nz
    print(f"  grid {nx}×{nx}×{nz} @ {LEVEL0_VOXEL_SIZE} m   "
          f"n_gen={args.n_gen} n_real={args.n_real} seed={args.seed}\n")

    # ── generated ───────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    gen_vols = []
    done = 0
    while done < args.n_gen:
        b = min(args.batch_size, args.n_gen - done)
        base = compute_canonical_base_grid(nx=nx, nz=nz, voxel_size=LEVEL0_VOXEL_SIZE,
                                           batch=b, device=str(dev))
        noisy = grid_to_VDB(base, torch.randn, [FEATURES])
        out = diff.ddpm_sample(noisy)
        for i in range(out.grid_count):
            g = DiffusionTensor(out.grid[i], out.data[i]).remove_mask(
                threshold=args.occ_threshold)
            gen_vols.append(to_dense(g.grid.ijk.jdata, nx, nz, dev))
        done += b
        print(f"  sampled {done}/{args.n_gen}")

    # ── real ────────────────────────────────────────────────────────────────
    real_vols, clipped, total_real = [], 0, 0
    for cp in list_crops(args.split, n=args.n_real):
        f = crop_pt(cp, LEVEL0_PYRAMID_RES)
        if not os.path.exists(f):
            continue
        ijk = load_dt(f, dev).grid.ijk.jdata
        v, n_clip = to_dense(ijk, nx, nz, dev, report_clip=True)
        real_vols.append(v)
        clipped += n_clip
        total_real += ijk.shape[0]

    print(f"\n{'='*72}")
    print(f"OCCUPANCY FRACTION over the canonical {nx}×{nx}×{nz} volume "
          f"(0–{nz*LEVEL0_VOXEL_SIZE:.1f} m), pruning threshold {args.occ_threshold}")
    g_mu, _ = occ_stats(gen_vols, "generated")
    r_mu, _ = occ_stats(real_vols, "real")
    print(f"\n  generated / real = {g_mu/max(r_mu,1e-9):.3f}   "
          f"(absolute excess {g_mu-r_mu:+.4f})")
    print(f"  real voxels above the canonical volume, excluded from BOTH sides: "
          f"{clipped}/{total_real} = {100*clipped/max(total_real,1):.1f}%")
    print("  NOTE: this denominator is the dense crop bbox. The level-1 end-to-end")
    print("  figure (GT 0.550) is over the SUBDIVIDED COARSE GRID, already pruned to")
    print("  the coarse structure. Same threshold, DIFFERENT denominators — the two")
    print("  occupancy numbers are not on the same scale and must not be compared.")
    print(f"{'='*72}")

    # pairwise_IoU_dist indexes and broadcasts against the whole batch, so it
    # needs a stacked tensor even though its docstring says "list". It also
    # divides by the union without clamping, so an all-empty volume yields NaN
    # and poisons every average — real crops can be empty inside the canonical
    # band, so drop those here (counted, not silently discarded).
    n_empty_g = sum(1 for v in gen_vols if not v.any())
    n_empty_r = sum(1 for v in real_vols if not v.any())
    gen_vols = [v for v in gen_vols if v.any()]
    real_vols = [v for v in real_vols if v.any()]
    if n_empty_g or n_empty_r:
        print(f"\n  dropped all-empty volumes before pairwise IoU: "
              f"{n_empty_g} generated, {n_empty_r} real")
    gen_t, real_t = torch.stack(gen_vols), torch.stack(real_vols)

    print(f"\n{'='*64}")
    print("average pairwise 1 − IoU  (higher = more diverse; Sin3DM protocol)")
    print(f"  generated ({len(gen_vols):2d}) : {pairwise_IoU_dist(gen_t):.4f}")
    print(f"  real      ({len(real_vols):2d}) : {pairwise_IoU_dist(real_t):.4f}   <- reference scale")
    print(f"  cross  gen↔real  : {cross_pairwise(gen_vols, real_vols):.4f}")
    print(f"{'='*64}")

    # ── Anchor the instrument ───────────────────────────────────────────────
    # Two independent volumes at density p have E[IoU] = p/(2−p), so pairwise
    # 1−IoU has a *density-dependent* uninformative level. Generated and real
    # differ in density here, so the raw comparison above is confounded: higher
    # density mechanically lowers 1−IoU. Normalise each by its own null.
    print("\nANCHOR — pairwise 1−IoU is density-dependent; each set needs its own null")
    print(f"{'set':<12} {'p':>6} {'analytic':>9} {'unif null':>10} {'z-strat':>9} "
          f"{'measured':>9} {'meas/z-strat':>13}")
    rows = []
    for name, vols in (("generated", gen_vols), ("real", real_vols)):
        p = float(torch.stack(vols).float().mean())
        analytic = 1.0 - p / (2.0 - p)
        unif = pairwise_IoU_dist(torch.stack(random_matched(vols, stratify_z=False)))
        zstr = pairwise_IoU_dist(torch.stack(random_matched(vols, stratify_z=True)))
        meas = pairwise_IoU_dist(torch.stack(vols))
        rows.append((name, p, analytic, unif, zstr, meas, meas / zstr))
        print(f"{name:<12} {p:6.3f} {analytic:9.4f} {unif:10.4f} {zstr:9.4f} "
              f"{meas:9.4f} {meas/zstr:13.4f}")
    print("\nmeasured < null ⇒ the set shares real structure (volumes overlap more than")
    print("chance). The last column is the density-corrected figure: compare THOSE")
    print("across sets, not the raw measured values.")
    if len(rows) == 2:
        g, r = rows[0][6], rows[1][6]
        print(f"\ngenerated/real, density-corrected: {g:.4f} / {r:.4f} = {g/r:.3f}"
              f"   (raw, confounded: {rows[0][5]/rows[1][5]:.3f})")
    print("\nNOT commensurable with upstream's single-shape table — see module docstring.")


if __name__ == "__main__":
    main()

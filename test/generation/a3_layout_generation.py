"""A3 follow-up: fixed class *layout*, free geometry.

Takes a real crop's level-0 GT, reduces it to a 2D footprint (one class per (i,j)
column — z left free), then generates from pure noise with the class channels
clamped to that footprint via the occupancy-preserving "renorm" projection (see
reverse_from): the model's own class-sum (= 1 − P(void)) is redistributed onto
the layout class, so *what* is where is fixed but *whether* and *how tall* stays
the model's decision.  The observable is geometry: per-class height and density
against the same crop's GT.

Two knobs matter for reading the result:

``--reduce``      how a column's single class is chosen.  ``majority`` counts
                  voxels, which favours vertically-extended classes (a column of
                  ground + tree scores as Vegetation).  ``nonground`` labels a
                  column by what stands *on* it, falling back to Ground for bare
                  ground — usually the fairer target for a height comparison.
``--layout_mask`` which columns are clamped at all.  DALES ground returns make
                  every column of a real crop defined, so ``none`` gives the
                  model vertical freedom only.  ``half`` / ``random`` blank part
                  of the layout, which is the only way to see what the model
                  hallucinates in genuinely unconstrained space.

    python test/generation/a3_layout_generation.py --crop 5080_54400_x0000_y0100
    python test/generation/a3_layout_generation.py --crop ... --reduce nonground
    python test/generation/a3_layout_generation.py --crop ... --layout_mask half
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import fvdb.nn as fvnn

from common import (CLASS_NAMES, export_dt, get_device, resolve_crop_path,
                    crop_pt, load_dt)
from utils.diffusion_tensor import DiffusionTensor
from utils.helper import reverse_from
from inference.inference import (
    load_dales_diffusion, compute_canonical_base_grid,
    LEVEL0_NX, LEVEL0_NZ, LEVEL0_VOXEL_SIZE, LEVEL0_PYRAMID_RES,
)

GROUND = 0  # CLASS_NAMES[0]


def column_counts(dt, n_cls):
    """(nx, nx, n_cls) voxel counts per column, plus the grid's z extent."""
    ijk = dt.grid.ijk.jdata.long()
    cls = dt.jdata[:, 4:4 + n_cls].argmax(dim=1)
    nx = int(ijk[:, :2].max()) + 1
    counts = torch.zeros(nx, nx, n_cls, dtype=torch.long, device=ijk.device)
    counts.index_put_((ijk[:, 0], ijk[:, 1], cls),
                      torch.ones_like(cls), accumulate=True)
    return counts, int(ijk[:, 2].max()) + 1


def column_layout(gt, n_cls, reduce="majority"):
    """(nx,nx) class per column; −1 = column has no GT occupancy."""
    counts, gt_nz = column_counts(gt, n_cls)
    occupied = counts.sum(-1) > 0
    if reduce == "nonground":
        ng = counts.clone()
        ng[..., GROUND] = 0
        has_ng = ng.sum(-1) > 0
        layout = torch.where(has_ng, ng.argmax(-1),
                             torch.full_like(has_ng, GROUND, dtype=torch.long))
    else:
        layout = counts.argmax(-1)
    layout = layout.clone()
    layout[~occupied] = -1
    return layout, gt_nz


def col_stats(dt, n_cls):
    """Per-column (n_occupied, max_k) and per-column class histogram."""
    ijk = dt.grid.ijk.jdata.long()
    cls = dt.jdata[:, 4:4 + n_cls].argmax(dim=1)
    nx = max(int(ijk[:, :2].max()) + 1, 1)
    n = torch.zeros(nx, nx, dtype=torch.long, device=ijk.device)
    n.index_put_((ijk[:, 0], ijk[:, 1]), torch.ones_like(ijk[:, 0]), accumulate=True)
    # max k per column: scatter_reduce amax over the flattened column index.
    # (index_put_ without accumulate keeps the last write, not the max.)
    flat = ijk[:, 0] * nx + ijk[:, 1]
    mk = torch.zeros(nx * nx, dtype=torch.long, device=ijk.device)
    mk.scatter_reduce_(0, flat, ijk[:, 2] + 1, reduce="amax", include_self=True)
    mk = mk.view(nx, nx) - 1
    return n, mk, cls, flat


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crop", required=True, help="Crop ID supplying the layout + GT.")
    p.add_argument("--split", default="test")
    p.add_argument("--src", default="checkpoints/diffusion_models/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="output/tests/A3_layout")
    p.add_argument("--occ_threshold", type=float, default=0.0)
    p.add_argument("--reduce", choices=["majority", "nonground"], default="majority",
                   help="How to pick one class per column (see module docstring).")
    p.add_argument("--layout_mask", choices=["none", "half", "random"], default="none",
                   help="Blank part of the layout so some columns are unclamped.")
    p.add_argument("--keep_frac", type=float, default=0.5,
                   help="Fraction of columns kept when --layout_mask random.")
    args = p.parse_args()

    dev = get_device()
    diff = load_dales_diffusion(0, args.src)
    diff.eval()
    C = diff.n_classes

    cp = resolve_crop_path(args.crop, args.split)
    name = os.path.basename(cp)
    gt = load_dt(crop_pt(cp, LEVEL0_PYRAMID_RES), dev)
    layout, gt_nz = column_layout(gt, C, args.reduce)
    nz = max(LEVEL0_NZ, gt_nz)          # cover the GT's vertical extent
    nx = layout.shape[0]

    n_defined_gt = int((layout >= 0).sum())
    torch.manual_seed(args.seed)
    if args.layout_mask == "half":
        layout[nx // 2:, :] = -1
    elif args.layout_mask == "random":
        layout[torch.rand(nx, nx, device=layout.device) >= args.keep_frac] = -1

    base = compute_canonical_base_grid(nx=nx, nz=nz, voxel_size=LEVEL0_VOXEL_SIZE,
                                       batch=1, device=str(dev))
    ijk = base.ijk.jdata.long()
    N = ijk.shape[0]
    col_cls = layout[ijk[:, 0], ijk[:, 1]]                # (N,) class or −1
    clamp_rows = col_cls >= 0
    onehot = torch.zeros(N, C, device=dev)
    onehot[clamp_rows, col_cls[clamp_rows]] = 1.0

    print(f"{name}: layout {nx}×{nx}, reduce={args.reduce}, mask={args.layout_mask}")
    print(f"  {n_defined_gt} columns have GT occupancy; {int((layout >= 0).sum())} "
          f"clamped, {int((layout < 0).sum())} left free  (grid z = {nz})")

    torch.manual_seed(args.seed)
    jdata = torch.randn(N, 4 + C + 1, device=dev)
    noisy = fvnn.VDBTensor(base, base.jagged_like(jdata))
    t_start = diff.max_T / diff.timesteps
    x0 = reverse_from(diff, noisy, t_start=t_start, clamp_onehot=onehot,
                      clamp_mode="renorm", clamp_rows=clamp_rows)

    gen = DiffusionTensor.from_vdb(x0).get_global().remove_mask(
        threshold=args.occ_threshold)

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.reduce}_{args.layout_mask}"
    stem = os.path.join(args.out, f"{name}_s{args.seed}_{tag}")
    export_dt(gen, f"{stem}__layout_gen.laz")
    export_dt(gt.get_global(), os.path.join(args.out, f"{name}__gt.laz"))
    print(f"  wrote {stem}__layout_gen.laz  ({gen.jdata.shape[0]:,} pts, "
          f"{100 * gen.jdata.shape[0] / N:.1f}% of grid; GT {gt.jdata.shape[0]:,})")

    # ── Geometry adherence per layout class (the actual observable) ──────────
    gt_n, gt_mk, _, _ = col_stats(gt, C)
    gen_n, gen_mk, gen_cls, gen_flat = col_stats(gen, C)

    def pad(a, target):
        if a.shape[0] >= target:
            return a[:target, :target]
        out = torch.zeros(target, target, dtype=a.dtype, device=a.device)
        out[:a.shape[0], :a.shape[1]] = a
        return out

    gt_n, gt_mk = pad(gt_n, nx), pad(gt_mk, nx)
    gen_n, gen_mk = pad(gen_n, nx), pad(gen_mk, nx)

    print(f"\n  {'class':12s} {'cols':>5s} {'gen occ':>8s} "
          f"{'cells/col gt|gen':>17s} {'mean max-z gt|gen':>18s}")
    for cid in range(C):
        sel = layout == cid
        ncols = int(sel.sum())
        if ncols == 0:
            continue
        hit = gen_n[sel] > 0
        f = lambda t, m: (t[sel][m].float().mean().item() if m.any() else float("nan"))
        print(f"  {CLASS_NAMES[cid]:12s} {ncols:5d} {hit.float().mean().item():8.0%} "
              f"{gt_n[sel][gt_n[sel] > 0].float().mean().item():8.2f}|"
              f"{f(gen_n, hit):<8.2f} "
              f"{gt_mk[sel][gt_n[sel] > 0].float().mean().item():8.2f}|"
              f"{f(gen_mk, hit):<9.2f}")

    free = layout < 0
    n_free = int(free.sum())
    if n_free:
        hit = gen_n[free] > 0
        print(f"\n  Unclamped columns ({n_free}): model built in "
              f"{hit.float().mean().item():.0%} of them, "
              f"{gen_n[free][hit].float().mean().item():.2f} cells/col, "
              f"mean max-z {gen_mk[free][hit].float().mean().item():.2f}")
        free_flat = free.flatten()
        in_free = free_flat[gen_flat]
        if in_free.any():
            hist = torch.bincount(gen_cls[in_free], minlength=C).float()
            hist = 100 * hist / hist.sum()
            top = sorted(range(C), key=lambda c: -hist[c])[:4]
            print("    classes chosen there: " +
                  ", ".join(f"{CLASS_NAMES[c]} {hist[c]:.0f}%" for c in top if hist[c] > 0.5))
        gt_hit = gt_n[free] > 0
        if gt_hit.any():
            print(f"    GT in those columns: {gt_n[free][gt_hit].float().mean().item():.2f} "
                  f"cells/col, mean max-z {gt_mk[free][gt_hit].float().mean().item():.2f}")


if __name__ == "__main__":
    main()

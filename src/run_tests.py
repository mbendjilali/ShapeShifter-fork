#!/usr/bin/env python3
"""run_tests.py — diagnostic tests for a loaded DALES diffusion model.

Tests
-----
A  Add noise at specified t values, denoise, export → recon_t{t}.laz
B  Generate from pure noise clamped to target class(es) → clamp_class{cid}.laz

Run from project root:
  python src/run_tests.py --test A --level 0 --crop data/dales/test/5080_54435_x0000_y0050
  python src/run_tests.py --test B --levels 0 --class_ids 7
  python src/run_tests.py --test B --levels 0 1 2 --class_ids 7
  python src/run_tests.py --test AB --level 0 --crop data/dales/test/5080_54435_x0000_y0050 --class_ids 1 7 --levels 0 1

  # Use a shorter crop id (split is resolved automatically from --split)
  python src/run_tests.py --test A --level 2 --crop 5080_54435_x0000_y0050 --split test
"""
import sys
import os

sys.path.insert(0, "./src")
sys.path.insert(0, "./src/utils")  # needed for helper.py's bare imports (fvdb_diffusion, etc.)

import argparse
import time
from pathlib import Path

import torch
import fvdb.nn as fvnn

from utils.helper import reverse_from
from utils.diffusion_tensor import DiffusionTensor
from inference.sample_diffusion import (
    load_dales_diffusion,
    compute_canonical_base_grid,
    export_to_laz,
)

CLASS_NAMES = [
    "Ground", "Vegetation", "Cars", "Trucks",
    "Fences", "PowerLines", "Poles", "Buildings",
]


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def _export_dt(dt: DiffusionTensor, path: str) -> None:
    """Export a DiffusionTensor to .laz, coloured by semantic class."""
    positions, features, _ = DiffusionTensor.get_feature_data(dt.jdata)
    positions_np = positions.cpu().numpy()
    features_np = features.cpu().numpy()
    intensity = features_np[:, 0]                               # channel 3 of jdata
    class_idx = features_np[:, 1:].argmax(axis=-1).clip(0, 7)  # channels 4-11 → 0-7
    export_to_laz(positions_np, intensity, class_idx, path)


# ---------------------------------------------------------------------------
# Test A — noise-then-denoise reconstruction
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_test_A(
    diff,
    crop_path: str,
    level: int,
    base_res: int,
    t_list,
    out_dir: str,
) -> None:
    """Add noise at each t in t_list, reverse-denoise, and export to LAZ."""
    resolution = base_res * (2 ** level)
    pt_file = Path(crop_path) / f"{resolution}.pt"
    if not pt_file.exists():
        raise FileNotFoundError(
            f"Crop file not found: {pt_file}\n"
            f"  Expected resolution {resolution} for level {level} (base_res={base_res})."
        )

    print(f"[test_A] Loading {pt_file}")
    obj = torch.load(pt_file, weights_only=False)
    if not isinstance(obj, DiffusionTensor):
        obj = DiffusionTensor(obj.grid, obj.data)
    X_clean = DiffusionTensor(obj.grid.to(diff.device), obj.data.to(diff.device))

    os.makedirs(out_dir, exist_ok=True)
    for t in t_list:
        t0 = time.time()
        X_copy = DiffusionTensor(
            X_clean.grid, X_clean.grid.jagged_like(X_clean.jdata.clone())
        )
        tt = torch.full((X_copy.jdata.shape[0],), float(t), device=diff.device)
        noisy, _ = diff.q_sample(X_copy, tt)
        rec = reverse_from(diff, noisy, t_start=t)
        rec = DiffusionTensor.from_vdb(rec).get_global().remove_mask()

        out_path = os.path.join(out_dir, f"recon_t{t:.2f}.laz")
        _export_dt(rec, out_path)
        print(f"  t={t:.2f}  pts={rec.jdata.shape[0]}  {time.time()-t0:.1f}s → {out_path}")


# ---------------------------------------------------------------------------
# Test B — class-conditional generation from noise
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_test_B(
    diff,
    base_res: int,
    nz: int,
    class_ids,
    seed: int,
    out_dir: str,
) -> None:
    """Generate from pure noise clamped to each target class, export to LAZ."""
    device = diff.device
    C = diff.n_classes
    t_start = diff.max_T / diff.timesteps  # always ≤ 1.0; equals 1.0 at level 0

    base_grid = compute_canonical_base_grid(
        base_res=base_res,
        extent_m=100.0,
        batch=1,
        device=str(device),
        nz=nz,
    )

    os.makedirs(out_dir, exist_ok=True)
    for cid in class_ids:
        cls_name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
        t0 = time.time()
        torch.manual_seed(seed)

        N = base_grid.ijk.jdata.shape[0]
        jdata = torch.randn(N, 4 + C + 1, device=device)
        noisy = fvnn.VDBTensor(base_grid, base_grid.jagged_like(jdata))

        onehot = torch.zeros(N, C, device=device)
        onehot[:, cid] = 1.0
        x0 = reverse_from(diff, noisy, t_start=t_start, clamp_onehot=onehot)
        occ = DiffusionTensor.from_vdb(x0).get_global().remove_mask()

        out_path = os.path.join(out_dir, f"clamp_class{cid}_{cls_name}.laz")
        _export_dt(occ, out_path)
        print(f"  class={cid} ({cls_name})  pts={occ.jdata.shape[0]}  {time.time()-t0:.1f}s → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnostic tests for a DALES diffusion checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--test", choices=["A", "B", "AB"], default="AB",
        help="Which test(s) to run. A=reconstruct, B=class-gen, AB=both.",
    )
    p.add_argument(
        "--level", type=int, default=0,
        help="Diffusion level to load (0=coarsest, 4=finest).",
    )
    p.add_argument(
        "--src", default="checkpoints/diffusion_models/",
        help="Directory containing checkpoint .pt files.",
    )
    p.add_argument(
        "--out", default=None,
        help="Output directory. Defaults to output/tests/level{level}.",
    )

    # --- test_A ---
    a_group = p.add_argument_group("test_A options")
    a_group.add_argument(
        "--crop", default=None,
        help=(
            "Crop to run test_A on. Either a full path "
            "(data/dales/test/5080_54435_x0000_y0050) or a bare crop ID resolved "
            "relative to data/dales/{split}/."
        ),
    )
    a_group.add_argument(
        "--split", default="test",
        help="Dataset split used when resolving a bare crop ID.",
    )
    a_group.add_argument(
        "--t_list", type=float, nargs="+", default=[2.0],
        help="Noise fractions for test_A (0.0–1.0).",
    )

    # --- test_B ---
    b_group = p.add_argument_group("test_B options")
    b_group.add_argument(
        "--class_ids", type=int, nargs="+", default=[0, 7],
        help=(
            "Class indices for test_B: 0=Ground, 1=Vegetation, 2=Cars, 3=Trucks, "
            "4=PowerLines, 5=Fences, 6=Poles, 7=Buildings."
        ),
    )
    b_group.add_argument(
        "--levels", type=int, nargs="+", default=None,
        help=(
            "Diffusion level(s) for test_B. Defaults to [--level] if not set. "
            "Pass multiple values to run test_B at several levels, e.g. --levels 0 1 2."
        ),
    )
    b_group.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for test_B.",
    )

    # --- grid shape (shared) ---
    g_group = p.add_argument_group("grid options")
    g_group.add_argument(
        "--base_res", type=int, default=64,
        help=(
            "Coarsest grid resolution. Crop .pt file for level N is loaded at "
            "resolution base_res * 2**level."
        ),
    )
    g_group.add_argument(
        "--nz", type=int, default=8,
        help="Z voxels for the canonical base grid used in test_B.",
    )

    return p.parse_args()


def resolve_crop_path(crop: str, split: str) -> str:
    """Return the absolute/relative path to the crop directory."""
    if os.path.isdir(crop):
        return crop
    candidate = os.path.join("data/dales", split, crop)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        f"Cannot find crop directory: tried '{crop}' and '{candidate}'.\n"
        "Pass a full path or a bare crop ID with --split pointing to the right split."
    )


def main():
    args = parse_args()
    b_levels = args.levels if args.levels is not None else [args.level]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device  : {device}")
    print(f"Src     : {args.src}")
    print()

    if args.test in ("A", "AB"):
        out_dir = args.out or f"output/tests/level{args.level}"
        print(f"Loading level-{args.level} model …")
        diff = load_dales_diffusion(args.level, args.src)
        diff.eval()
        print(
            f"  n_classes={diff.n_classes}  "
            f"timesteps={diff.timesteps}  "
            f"max_T={diff.max_T}"
        )
        if args.crop is None:
            raise ValueError("--crop is required for test A.")
        crop_path = resolve_crop_path(args.crop, args.split)
        print(f"\n{'='*60}")
        print(f"Test A — reconstruct from noise")
        print(f"  level    : {args.level}")
        print(f"  crop     : {crop_path}")
        print(f"  t values : {args.t_list}")
        print(f"{'='*60}")
        run_test_A(diff, crop_path, args.level, args.base_res, args.t_list, out_dir)
        print(f"\nDone. Results written to: {out_dir}")

    if args.test in ("B", "AB"):
        for level in b_levels:
            out_dir = args.out or f"output/tests/level{level}"
            print(f"\nLoading level-{level} model …")
            diff = load_dales_diffusion(level, args.src)
            diff.eval()
            print(
                f"  n_classes={diff.n_classes}  "
                f"timesteps={diff.timesteps}  "
                f"max_T={diff.max_T}"
            )
            print(f"\n{'='*60}")
            print(f"Test B — class-conditional generation  [level={level}]")
            print(f"  class_ids : {args.class_ids}")
            print(f"  seed      : {args.seed}")
            print(f"  base_res  : {args.base_res}  nz : {args.nz}")
            print(f"{'='*60}")
            run_test_B(diff, args.base_res, args.nz, args.class_ids, args.seed, out_dir)
            print(f"\nDone. Results written to: {out_dir}")


if __name__ == "__main__":
    main()

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
from inference.inference import (
    load_dales_diffusion,
    compute_canonical_base_grid,
    export_to_laz,
    LEVEL0_NX,
    LEVEL0_NZ,
    LEVEL0_VOXEL_SIZE,
    LEVEL0_PYRAMID_RES,
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
    pyramid_res: int,
    t_list,
    out_dir: str,
) -> None:
    """Add noise at each t in t_list, reverse-denoise, and export to LAZ."""
    resolution = pyramid_res * (2 ** level)
    pt_file = Path(crop_path) / f"{resolution}.pt"
    if not pt_file.exists():
        raise FileNotFoundError(
            f"Crop file not found: {pt_file}\n"
            f"  Expected resolution {resolution} for level {level} (pyramid_res={pyramid_res})."
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
    nx: int,
    nz: int,
    voxel_size: float,
    class_ids,
    seed: int,
    out_dir: str,
) -> None:
    """Generate from pure noise clamped to each target class, export to LAZ."""
    device = diff.device
    C = diff.n_classes
    t_start = diff.max_T / diff.timesteps  # always ≤ 1.0; equals 1.0 at level 0

    base_grid = compute_canonical_base_grid(
        nx=nx,
        nz=nz,
        voxel_size=voxel_size,
        batch=1,
        device=str(device),
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
# Test C — hide voxels, add noise, denoise, compare to original
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_test_C(
    diff,
    crop_path: str,
    level: int,
    pyramid_res: int,
    t: float,
    hide_fraction: float,
    out_dir: str,
) -> None:
    """Hide a spatial region of voxels, add noise, denoise, compare to original.

    The experiment asks: can the diffusion model reconstruct information that was
    deliberately removed from a crop?

    Steps
    -----
    1. Load the clean crop from disk and densify it (matching training distribution).
    2. Select voxels in the first ``hide_fraction`` of the X-axis range and mark
       them as empty (features = 0, mask = -1).  The rest of the crop is kept clean.
    3. Add Gaussian noise at level ``t`` to the whole (now partially-holed) grid.
    4. Run the reverse diffusion process from ``t`` → 0 to reconstruct.
    5. Export three .laz files so you can compare side-by-side in a viewer:
         original.laz             — ground truth (no hiding)
         hidden.laz               — crop with the region removed  (the "hole")
         recon_hX.X_tY.YY.laz    — model's reconstruction
    """
    resolution = pyramid_res * (2 ** level)
    pt_file = Path(crop_path) / f"{resolution}.pt"
    if not pt_file.exists():
        raise FileNotFoundError(
            f"Crop file not found: {pt_file}\n"
            f"  Expected resolution {resolution} for level {level} (pyramid_res={pyramid_res})."
        )

    print(f"[test_C] Loading {pt_file}")
    obj = torch.load(pt_file, weights_only=False)
    if not isinstance(obj, DiffusionTensor):
        obj = DiffusionTensor(obj.grid, obj.data)
    X_sparse = DiffusionTensor(obj.grid.to(diff.device), obj.data.to(diff.device))

    # Densify to a regular bbox grid — empty slots get mask = -1, features = 0.
    # empty_fill='zero' matches how the model was trained (no blurred neighbours
    # leaking into the empty slots, so the distribution at inference is consistent).
    X_dense = X_sparse.to_custom_dense(empty_fill='zero')

    os.makedirs(out_dir, exist_ok=True)

    # Export the original (only truly occupied voxels, for reference)
    orig_path = os.path.join(out_dir, "original.laz")
    _export_dt(X_sparse.get_global().remove_mask(), orig_path)
    print(f"  Exported original          → {orig_path}")

    # ------------------------------------------------------------------
    # Define the hidden region: voxels in the first hide_fraction of X
    # ------------------------------------------------------------------
    ijk = X_dense.grid.ijk.jdata          # (N_voxels, 3)  integer voxel coords
    i_min = ijk[:, 0].min().item()
    i_max = ijk[:, 0].max().item()
    i_split = i_min + int(round((i_max - i_min) * hide_fraction))
    hidden_mask = ijk[:, 0] <= i_split    # True for voxels to be erased

    n_hidden = int(hidden_mask.sum())
    print(f"  Hiding {n_hidden}/{ijk.shape[0]} voxels  "
          f"(X ∈ [{i_min}, {i_split}], fraction={hide_fraction:.2f})")

    # Zero all feature channels and flip mask to -1 for the hidden voxels
    X_holed_data = X_dense.jdata.clone()
    X_holed_data[hidden_mask, :-1] = 0.0   # zero offset, intensity, class_probs
    X_holed_data[hidden_mask,  -1] = -1.0  # mark as empty
    X_holed = DiffusionTensor(X_dense.grid, X_dense.grid.jagged_like(X_holed_data))

    # Export the "holed" crop (only the surviving occupied voxels)
    hidden_path = os.path.join(out_dir, "hidden.laz")
    _export_dt(X_holed.get_global().remove_mask(), hidden_path)
    print(f"  Exported holed input       → {hidden_path}")

    # ------------------------------------------------------------------
    # Add noise at level t to the whole grid (occupied + empty slots)
    # ------------------------------------------------------------------
    tt = torch.full((X_holed.jdata.shape[0],), t, device=diff.device)
    noisy, _ = diff.q_sample(X_holed, tt)

    # ------------------------------------------------------------------
    # Denoise: iterate from t → 0
    # ------------------------------------------------------------------
    t0 = time.time()
    recon_vdb = reverse_from(diff, noisy, t_start=t)
    recon = DiffusionTensor.from_vdb(recon_vdb).get_global().remove_mask()

    out_path = os.path.join(out_dir, f"recon_h{hide_fraction:.1f}_t{t:.2f}.laz")
    _export_dt(recon, out_path)
    print(
        f"  Reconstruction: pts={recon.jdata.shape[0]}  "
        f"{time.time()-t0:.1f}s  → {out_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnostic tests for a DALES diffusion checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--test", choices=["A", "B", "AB", "C"], default="AB",
        help="Which test(s) to run. A=reconstruct, B=class-gen, AB=both, C=hide-and-reconstruct.",
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
        "--t_list", type=float, nargs="+", default=[1.0],
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

    # --- test_C ---
    c_group = p.add_argument_group("test_C options")
    c_group.add_argument(
        "--hide_fraction", type=float, default=0.5,
        help=(
            "Fraction (0.0–1.0) of the X-axis range to hide for test_C. "
            "0.5 hides the first half of the crop; the model must reconstruct it. "
            "Uses --crop / --split / --level / --t_list[0] from shared args."
        ),
    )

    # --- grid shape (shared) ---
    g_group = p.add_argument_group("grid options")
    g_group.add_argument(
        "--pyramid_res", type=int, default=LEVEL0_PYRAMID_RES,
        help=(
            "Coarsest pyramid .pt label for test_A. Crop file at level N is "
            "pyramid_res * 2**level (16 → 16.pt at level 0 for diffusion_0)."
        ),
    )
    g_group.add_argument(
        "--nx", type=int, default=LEVEL0_NX,
        help="XY voxels for test_B canonical grid (~100m / voxel_size).",
    )
    g_group.add_argument(
        "--voxel_size", type=float, default=LEVEL0_VOXEL_SIZE,
        help="Metres per voxel for test_B (3.2 for diffusion_0).",
    )
    g_group.add_argument(
        "--nz", type=int, default=LEVEL0_NZ,
        help="Z voxels for test_B canonical grid.",
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
        run_test_A(diff, crop_path, args.level, args.pyramid_res, args.t_list, out_dir)
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
            print(f"  nx        : {args.nx}  nz : {args.nz}  voxel_size : {args.voxel_size}")
            print(f"{'='*60}")
            run_test_B(
                diff, args.nx, args.nz, args.voxel_size,
                args.class_ids, args.seed, out_dir,
            )
            print(f"\nDone. Results written to: {out_dir}")

    if args.test in ("C",):
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
            raise ValueError("--crop is required for test C.")
        crop_path = resolve_crop_path(args.crop, args.split)
        t = args.t_list[0]
        print(f"\n{'='*60}")
        print(f"Test C — hide voxels, denoise, reconstruct")
        print(f"  level         : {args.level}")
        print(f"  crop          : {crop_path}")
        print(f"  hide_fraction : {args.hide_fraction}")
        print(f"  t             : {t}")
        print(f"{'='*60}")
        run_test_C(
            diff, crop_path, args.level, args.pyramid_res,
            t, args.hide_fraction, out_dir,
        )
        print(f"\nDone. Results written to: {out_dir}")


if __name__ == "__main__":
    main()

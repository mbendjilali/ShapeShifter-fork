"""
pc_encoding.py — DALES LAZ tile → crops 50×50m → DiffusionTensor pyramid.

Feature layout (Route A, 10 channels, slices unchanged from DiffusionTensor):
  [0:3]  PCA normal per voxel (smallest eigenvector of intra-voxel covariance)
  [3:6]  local offset = (mean_pt − voxel_center) / voxel_size  [INVARIANT]
  [6]    intensity normalised to [0, 1]
  [7]    semantic class: (sem_class − 1) / (N_CLASSES − 1)  ∈ [0, 1]
  [8]    mask = 1  [INVARIANT]

Voxel sizes per level (50×50m crop):
  256.pt  0.2 m/voxel  (~250 × 250 in XY)
  128.pt  0.4 m/voxel
   64.pt  0.8 m/voxel
   32.pt  1.6 m/voxel
   16.pt  3.2 m/voxel  (~16 × 16 × 11 in XY × Z)

Level 0 dense base at 16.pt: ~16 × 16 × 11 ≈ 2 800 voxels
vs. the old full-tile approach: 16 × 16 × 3 ≈ 800 voxels at 32 m/voxel.

Crop naming: {tile_id}_x{x_start:04d}_y{y_start:04d}
  e.g. data/GT_sparse_tensors/dales/5080_54435_x0000_y0050/256.pt

Usage
-----
# Encode one tile as 50×50m crops
python src/shape_encoding/pc_encoding.py --tile 5080_54435

# Encode all training tiles
python src/shape_encoding/pc_encoding.py --all --split train

# Encode + augment (yaw rotations × horizontal flips)
python src/shape_encoding/pc_encoding.py --tile 5080_54435 --augment

# Visual smoke-test: export a crop's 256.pt as PLY
python src/shape_encoding/pc_encoding.py --tile 5080_54435 --export_ply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

import fvdb
from diffusion_tensor import DiffusionTensor

try:
    import laspy
    from laspy import LazBackend
except ImportError as e:
    raise RuntimeError("laspy required: pip install laspy laszip") from e

try:
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    from scipy.interpolate import RegularGridInterpolator
except ImportError as e:
    raise RuntimeError("scipy required: pip install scipy") from e

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_dales_laz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read xyz (metric float32), sem_class (uint8 1–8), intensity (float32 0–1)."""
    las = laspy.read(str(path), laz_backend=LazBackend.Laszip)
    x = np.asarray(las.x[:], dtype=np.float32)
    y = np.asarray(las.y[:], dtype=np.float32)
    z = np.asarray(las.z[:], dtype=np.float32)
    xyz = np.stack([x, y, z], axis=-1)
    sem = np.asarray(las.legacy_semantic[:], dtype=np.uint8)
    intensity = np.asarray(las.intensity[:], dtype=np.float32) / 65535.0
    return xyz, sem, intensity

# ---------------------------------------------------------------------------
# fVDB grid — metric, no NDCnormalize
# ---------------------------------------------------------------------------

def build_metric_grid(xyz: np.ndarray, voxel_size: float, device: str):
    """Sparse fVDB grid at fixed metric voxel_size (no NDCnormalize)."""
    pts     = torch.tensor(xyz, dtype=torch.float32, device=device)
    pc_jag  = fvdb.JaggedTensor([pts])
    origin  = torch.tensor([voxel_size / 2.0] * 3, dtype=torch.float32, device=device)
    pad     = torch.zeros(3, dtype=torch.int32, device=device)
    return fvdb.gridbatch_from_points(pc_jag, pad, pad, voxel_size, origin)


# ---------------------------------------------------------------------------
# Per-voxel aggregation
# ---------------------------------------------------------------------------

def aggregate_voxels(
    cfg,
    xyz:       np.ndarray,   # (N,3) detrended, XY origin-shifted
    sem:       np.ndarray,   # (N,)  uint8 1–8
    intensity: np.ndarray,   # (N,)  float32 0–1
):
    """
    Returns (grid, mean_xyz, pca_normals, mean_intensity, class_norm)
    all arrays shape (V, *).
    """
    grid    = build_metric_grid(xyz, cfg["voxel_size_initial"], cfg["device"])
    pts     = torch.tensor(xyz, dtype=torch.float32, device=cfg["device"])
    ijk     = torch.floor(pts / cfg["voxel_size_initial"]).to(torch.int32)
    ids_raw = grid.ijk_to_index(fvdb.JaggedTensor([ijk])).jdata.cpu().numpy()
    voxel_ids = np.where(ids_raw < 0, 0, ids_raw)

    V   = int(grid.total_voxels)
    N   = len(xyz)
    xyt = torch.tensor(xyz, dtype=torch.float32)
    vids = torch.tensor(voxel_ids, dtype=torch.long)

    # counts
    count = torch.zeros(V).scatter_add_(0, vids, torch.ones(N))
    cnt   = count.clamp(min=1).unsqueeze(1)

    # mean xyz per voxel
    sum_xyz  = torch.zeros(V, 3).scatter_add_(0, vids.unsqueeze(1).expand(-1, 3), xyt)
    mean_xyz = (sum_xyz / cnt).numpy()

    def scatter_mean_1d(arr: np.ndarray) -> np.ndarray:
        t = torch.tensor(arr, dtype=torch.float32)
        s = torch.zeros(V).scatter_add_(0, vids, t)
        return (s / count.clamp(min=1)).numpy().astype(np.float32)

    mean_intensity = scatter_mean_1d(intensity)

    sem_t = torch.nn.functional.one_hot(
        torch.tensor(sem.astype(np.int64) - 1, dtype=torch.long),
        num_classes=cfg["n_classes"],
    ).float()

    cc = torch.zeros(V, cfg["n_classes"]).scatter_add_(
        0, vids.unsqueeze(1).expand(-1, cfg["n_classes"]), sem_t)

    return grid, mean_xyz, mean_intensity, cc

# ---------------------------------------------------------------------------
# Core encoding from already-detrended, XY-origin-shifted points
# ---------------------------------------------------------------------------

def _encode_points(
    cfg,
    xyz:       np.ndarray,   # detrended + XY shifted to [0, CROP_SIZE_M)
    sem:       np.ndarray,
    intensity: np.ndarray,
    save_dir:  Path,
    verbose:   bool = False,
) -> None:
    """Voxelise, pool, and save a pyramid for one set of points."""
    save_dir.mkdir(parents=True, exist_ok=True)
    device = cfg["device"]
    voxel_size = cfg["voxel_size_initial"]

    keep = sem != 2

    xyz = xyz[keep]
    sem = sem[keep]
    intensity = intensity[keep]

    grid, mean_xyz, mean_int, cc = aggregate_voxels(cfg, xyz, sem, intensity)

    mean_xyz_t = torch.tensor(mean_xyz, dtype=torch.float32, device=device)
    voxel_centers = grid.grid_to_world(grid.ijk.float()).jdata
    local_offset = (mean_xyz_t - voxel_centers) / voxel_size

    int_t = torch.tensor(mean_int, dtype=torch.float32, device=device).unsqueeze(1)
    cc_t  = cc.to(device)

    # Pool [local_offset(3), intensity(1), cc(n_classes)] together.
    # avg_pool on cc preserves the argmax → correct majority class at every level.
    feat = torch.cat([local_offset, int_t, cc_t], dim=1)
    feat_jt = fvdb.JaggedTensor([feat])

    def _to_dt(g, f):
        lo      = f[:, :3]
        intens  = f[:, 3:4]
        cc_f    = f[:, 4:]
        total   = cc_f.sum(1, keepdim=True).clamp(min=1e-6)
        class_probs = cc_f / total
        features  = torch.cat([intens, class_probs], dim=1)   # (V, 10)
        mask    = torch.ones(len(f), 1, device=device)
        return DiffusionTensor.get_tensor_from_data(g, lo, features, mask)

    torch.save(_to_dt(grid, feat), save_dir / f"{cfg['initial_size']}.pt")

    current_grid = grid
    current_feat = feat_jt
    size = cfg["initial_size"]
    for t_size in cfg["targets"]:
        k_s = size // t_size
        pooled, new_grid = current_grid.avg_pool(k_s, current_feat, k_s)
        torch.save(_to_dt(new_grid, pooled.jdata), save_dir / f"{t_size}.pt")
        current_grid = new_grid
        current_feat = pooled
        size = t_size

    if verbose:
        print(f"    → {save_dir.name}  ({int(grid.total_voxels):,} voxels at {voxel_size}m)")


# ---------------------------------------------------------------------------
# Crop splitting
# ---------------------------------------------------------------------------

def _crop_ids(cfg, xyz: np.ndarray) -> List[Tuple[float, float]]:
    """Return (x_start, y_start) pairs covering the tile."""
    x0, y0 = float(xyz[:, 0].min()), float(xyz[:, 1].min())
    x1, y1 = float(xyz[:, 0].max()), float(xyz[:, 1].max())
    xs = np.arange(x0, x1, cfg["crop_stride_m"])
    ys = np.arange(y0, y1, cfg["crop_stride_m"])
    return [(float(x), float(y)) for x in xs for y in ys]


def _extract_crop(cfg, xyz: np.ndarray, sem: np.ndarray, intensity: np.ndarray,
                  x_start: float, y_start: float,
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract points in [x_start, x_start+crop_size) × [y_start, y_start+crop_size).
    Shifts XY so the crop starts at (0, 0) — voxel indices begin near 0.
    """
    mask = (
        (xyz[:, 0] >= x_start) & (xyz[:, 0] < x_start + cfg["crop_size_m"]) &
        (xyz[:, 1] >= y_start) & (xyz[:, 1] < y_start + cfg["crop_size_m"])
    )
    xyz_c = xyz[mask].copy()
    xyz_c[:, 0] -= x_start   # shift to local origin
    xyz_c[:, 1] -= y_start
    return xyz_c, sem[mask], intensity[mask]


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def encode_tile_as_crops(
    cfg,
    laz_path:  Path,
    out_root:  Path,
    verbose:   bool      = True,
    skip_complete: bool  = False,
) -> List[str]:
    """
    Encode a DALES LAZ tile as a grid of 50×50m crops.

    Returns the list of crop IDs that were saved.
    """
    tile_id = laz_path.stem
    if verbose:
        print(f"[pc_encoding] {tile_id}  loading …")

    xyz, sem, intensity = load_dales_laz(laz_path)
    if verbose:
        print(f"  {len(xyz):,} pts  X=[{xyz[:,0].min():.0f},{xyz[:,0].max():.0f}]"
              f"  Y=[{xyz[:,1].min():.0f},{xyz[:,1].max():.0f}]"
              f"  Z=[{xyz[:,2].min():.1f},{xyz[:,2].max():.1f}]")


    crop_origins = _crop_ids(cfg, xyz)
    saved_ids = []

    for x_start, y_start in tqdm(crop_origins, desc="Extracting crops"):
        xyz_c, sem_c, int_c = _extract_crop(cfg, xyz, sem, intensity, x_start, y_start)
        if len(xyz_c) < cfg["min_points_per_crop"]:
            continue

        # Crop ID encodes the tile + crop XY position (integer metres relative to tile origin)
        crop_id  = f"{tile_id}_x{int(x_start):04d}_y{int(y_start):04d}"
        save_dir = out_root / crop_id

        # Skip crops that already have every pyramid level on disk
        if skip_complete:
            all_levels = [cfg["initial_size"]] + list(cfg["targets"])
            if all((save_dir / f"{lv}.pt").exists() for lv in all_levels):
                saved_ids.append(crop_id)
                continue

        _encode_points(cfg, xyz_c, sem_c, int_c, save_dir,
                       verbose=False)
        saved_ids.append(crop_id)

    if verbose:
        print(f"  ✓ {len(saved_ids)} crops saved for {tile_id}")

    return saved_ids


# ---------------------------------------------------------------------------
# Augmentation (applied per-crop, at raw point level — invariant #4)
# ---------------------------------------------------------------------------

def rotate_yaw(xyz: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate around z-axis about the crop centre."""
    theta = np.deg2rad(angle_deg)
    c, s  = np.cos(theta), np.sin(theta)
    R     = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    centre = xyz.mean(0)
    return (xyz - centre) @ R.T + centre


def flip_horizontal(xyz: np.ndarray, axis: int = 0) -> np.ndarray:
    """Mirror along x (axis=0) or y (axis=1). No vertical flip (invariant #5)."""
    xyz_f = xyz.copy()
    xyz_f[:, axis] = 2 * xyz[:, axis].mean() - xyz[:, axis]
    return xyz_f


def encode_tile_as_crops_augmented(
    cfg,
    laz_path:   Path,
    out_root:   Path,
    yaw_angles: List[float] = (0, 45, 90, 135, 180, 225, 270, 315),
    flip_axes:  List[int]   = (0, 1),
    verbose:    bool  = False,
) -> None:
    """
    Encode all crops of a tile with rotation + flip augmentation.

    Augmented variants are saved as:
      {tile_id}_x{X}_y{Y}_r{angle}/
      {tile_id}_x{X}_y{Y}_f{axis}/
      {tile_id}_x{X}_y{Y}_r{angle}_f{axis}/

    Invariant #4 respected: rotation/flip applied at raw point level before voxelisation.
    """
    tile_id = laz_path.stem
    if verbose:
        print(f"[pc_encoding/aug] {tile_id}")

    xyz, sem, intensity = load_dales_laz(laz_path)
    crop_origins = _crop_ids(cfg, xyz)

    for x_start, y_start in crop_origins:
        xyz_c, sem_c, int_c = _extract_crop(cfg, xyz, sem, intensity, x_start, y_start)
        if len(xyz_c) < cfg["min_points_per_crop"]:
            continue
        base_id = f"{tile_id}_x{int(x_start):04d}_y{int(y_start):04d}"

        for angle in yaw_angles:
            xyz_r = rotate_yaw(xyz_c, angle) if angle != 0 else xyz_c.copy()

            for flip_axis in [None] + list(flip_axes):
                if flip_axis is not None:
                    xyz_aug = flip_horizontal(xyz_r, flip_axis)
                    suffix  = f"_r{angle:03d}_f{flip_axis}" if angle != 0 else f"_f{flip_axis}"
                else:
                    xyz_aug = xyz_r
                    suffix  = f"_r{angle:03d}" if angle != 0 else ""

                if not suffix:
                    # original already encoded by encode_tile_as_crops
                    continue

                save_dir = out_root / f"{base_id}{suffix}"
                _encode_points(cfg, xyz_aug, sem_c, int_c, save_dir,
                               verbose=verbose)


# ---------------------------------------------------------------------------
# Export helper (smoke-test / visualisation)
# ---------------------------------------------------------------------------

def export_ply(dt: DiffusionTensor, out_path: Path) -> None:
    """Export DiffusionTensor as PLY coloured by semantic class."""
    try:
        import pymeshlab as ml
    except ImportError:
        raise RuntimeError("pymeshlab required for PLY export")

    g = dt.get_global().remove_mask()
    positions, features, _ = DiffusionTensor.get_feature_data(g.jdata)
    positions   = positions.cpu().detach().numpy()
    class_probs = features[:, 1:].cpu().detach().numpy()
    class_idx   = class_probs.argmax(axis=-1).clip(0, 7)
    rgb         = class_idx[:, None].repeat(3, axis=1).astype(np.float32) / 7.0
    normals_n   = np.zeros_like(positions)

    ms = ml.MeshSet()
    v_col = np.column_stack([rgb, np.ones_like(rgb[:, :1])])
    ms.add_mesh(ml.Mesh(vertex_matrix=positions, v_normals_matrix=normals_n,
                        v_color_matrix=v_col))
    ms.save_current_mesh(str(out_path), save_vertex_normal=True)
    print(f"Exported → {out_path}  ({len(positions)} pts)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode DALES LAZ tiles as 50×50m crops")
    parser.add_argument("--config", default="/home/moussabendjilali/libs/ShapeShifter/configs/encoding/dales.json", help="Encoding configuration file path")
    parser.add_argument("--tile",  default=None, type=str, help="Tile ID (e.g. 5080_54435)")
    parser.add_argument("--all",   action="store_true",    help="Encode all tiles in --split")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--augment",       action="store_true", help="Add yaw/flip augmentations")
    parser.add_argument("--export_ply",    action="store_true", help="Export first crop's 256.pt as PLY")
    parser.add_argument("--skip_complete", action="store_true",
                        help="Skip crops that already have all pyramid levels on disk")
    parser.add_argument("--out", default="/home/moussabendjilali/libs/ShapeShifter/data", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--crop_size", type=float)
    parser.add_argument("--crop_stride", type=float)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    out_root = Path(args.out) / args.split
    laz_dir  = Path(cfg["dales_root"]) / args.split

    tiles = ([args.tile] if args.tile else
             [p.stem for p in sorted(laz_dir.glob("*.laz"))] if args.all else None)
    if tiles is None:
        parser.print_help(); sys.exit(0)

    for tile_id in tiles:
        laz_path = laz_dir / f"{tile_id}.laz"
        if not laz_path.exists():
            print(f"WARNING: {laz_path} not found, skipping"); continue

        crop_ids = encode_tile_as_crops(
            cfg, laz_path, out_root,
            skip_complete=args.skip_complete,
        )

        if args.augment:
            encode_tile_as_crops_augmented(cfg, laz_path, out_root)

        if args.export_ply and crop_ids:
            pt = out_root / crop_ids[0] / "256.pt"
            if pt.exists():
                dt = torch.load(pt, weights_only=False)
                export_ply(dt, out_root / crop_ids[0] / "256_preview.ply")

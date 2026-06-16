"""
pc_encoding.py — DALES LAZ tile → crops 50×50m → DiffusionTensor pyramid.

Feature layout (Route A, 10 channels, slices unchanged from DiffusionTensor):
  [0:3]  PCA normal per voxel (smallest eigenvector of intra-voxel covariance)
  [3:6]  local offset = (mean_pt − voxel_center) / voxel_size  [INVARIANT]
  [6]    intensity normalised to [0, 1]
  [7]    height above ground normalised to [0, 1]  (clipped at MAX_HEIGHT_M)
  [8]    semantic class: (sem_class − 1) / (N_CLASSES − 1)  ∈ [0, 1]
  [9]    mask = 1  [INVARIANT]

Voxel sizes per level (50×50m crop):
  256.pt  0.2 m/voxel  (~250 × 250 in XY)
  128.pt  0.4 m/voxel
   64.pt  0.8 m/voxel
   32.pt  1.6 m/voxel
   16.pt  3.2 m/voxel  (~16 × 16 × 11 in XY × Z)

Level 0 dense base at 16.pt: ~16 × 16 × 11 ≈ 2 800 voxels
vs. the old full-tile approach: 16 × 16 × 3 ≈ 800 voxels at 32 m/voxel.

Terrain detrending (invariant #5): DTM computed on the FULL tile (better coverage),
  then applied per crop. Each crop coordinate origin is shifted to (0, 0) in XY
  so voxel indices start near 0 regardless of the tile's position in the tile grid.

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
import sys
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

import fvdb
from diffusion_tensor import DiffusionTensor
from PoNQ_grid import PoNQ_grid

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
# Constants
# ---------------------------------------------------------------------------

DALES_ROOT = Path("/data/moussabendjilali/archive/data/dales_2")
GT_ROOT    = Path("data/dales")

# Crop geometry
CROP_SIZE_M   = 50.0   # XY extent of each crop in metres
CROP_STRIDE_M = 50.0   # XY stride (= CROP_SIZE_M → no overlap)

# Voxel pyramid
VOXEL_SIZE_INITIAL = 0.2    # metres at finest level (→ ~250 voxels for 50m crop)
INITIAL_SIZE       = 256    # PoNQ_grid label for the 0.2m grid
TARGETS            = [128, 64, 32, 16]  # pool levels to save

# Feature normalisation
GROUND_CLASS = 1
N_CLASSES    = 8        # DALES classes 1..8
MAX_HEIGHT_M = 50.0     # height normalisation ceiling

# DTM parameters (computed on full tile for robustness)
DTM_CELL_M = 10.0
DTM_SIGMA  = 2.0

# Minimum points per crop to encode (skip near-empty edge crops)
MIN_POINTS_PER_CROP = 500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
# Terrain detrending
# ---------------------------------------------------------------------------

def build_dtm_interpolator(xyz: np.ndarray, sem: np.ndarray,
                           cell_m: float = DTM_CELL_M,
                           sigma: float = DTM_SIGMA):
    """
    Build a (x, y) → ground_z bilinear interpolator from ground points.
    Best called on the FULL tile before cropping so the DTM has dense coverage.
    """
    gp = xyz[sem == GROUND_CLASS]
    if len(gp) == 0:
        z_fallback = float(xyz[:, 2].min())
        return lambda pts: np.full(len(pts), z_fallback, dtype=np.float32)

    x0, y0 = float(xyz[:, 0].min()), float(xyz[:, 1].min())
    nx = int(np.ceil((xyz[:, 0].max() - x0) / cell_m)) + 2
    ny = int(np.ceil((xyz[:, 1].max() - y0) / cell_m)) + 2

    dtm = np.full((nx, ny), np.nan, dtype=np.float64)

    gi = np.clip(((gp[:, 0] - x0) / cell_m).astype(int), 0, nx - 1)
    gj = np.clip(((gp[:, 1] - y0) / cell_m).astype(int), 0, ny - 1)

    flat   = gi * ny + gj
    order  = np.argsort(flat)
    flat_s, z_s = flat[order], gp[:, 2][order]
    unique, first = np.unique(flat_s, return_index=True)
    ends   = np.append(first[1:], len(flat_s))
    for uid, s, e in zip(unique, first, ends):
        xi, yi = divmod(int(uid), ny)
        dtm[xi, yi] = z_s[s:e].min()

    nan_mask = np.isnan(dtm)
    if nan_mask.any():
        _, (ri, ci) = distance_transform_edt(nan_mask, return_indices=True)
        dtm[nan_mask] = dtm[ri[nan_mask], ci[nan_mask]]

    dtm = gaussian_filter(dtm, sigma=sigma)

    xi_vals = x0 + np.arange(nx) * cell_m
    yi_vals = y0 + np.arange(ny) * cell_m
    interp  = RegularGridInterpolator(
        (xi_vals, yi_vals), dtm,
        method="linear", bounds_error=False, fill_value=None,
    )

    def query(pts: np.ndarray) -> np.ndarray:
        return interp(pts[:, :2]).astype(np.float32)

    return query


# ---------------------------------------------------------------------------
# fVDB grid — metric, no NDCnormalize
# ---------------------------------------------------------------------------

def build_metric_grid(xyz: np.ndarray, voxel_size: float, device: str = DEVICE):
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
    xyz:       np.ndarray,   # (N,3) detrended, XY origin-shifted
    sem:       np.ndarray,   # (N,)  uint8 1–8
    intensity: np.ndarray,   # (N,)  float32 0–1
    height:    np.ndarray,   # (N,)  float32 0–1
    voxel_size: float,
    device: str = DEVICE,
):
    """
    Returns (grid, mean_xyz, pca_normals, mean_intensity, mean_height, class_norm)
    all arrays shape (V, *).
    """
    grid    = build_metric_grid(xyz, voxel_size, device)
    pts     = torch.tensor(xyz, dtype=torch.float32, device=device)
    ijk     = torch.floor(pts / voxel_size).to(torch.int32)
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

    # PCA normals per voxel (batched 3×3 eigh)
    centered = xyt - sum_xyz[vids] / count[vids].unsqueeze(1)
    cov = torch.zeros(V, 3, 3)
    for a in range(3):
        for b in range(3):
            cov[:, a, b].scatter_add_(0, vids, centered[:, a] * centered[:, b])
    cov /= cnt.unsqueeze(-1).clamp(min=1)
    _, eigvec = torch.linalg.eigh(cov)
    normals   = eigvec[:, :, 0]
    flip      = torch.where(normals[:, 2] < 0, -1.0, 1.0).unsqueeze(1)
    normals   = (normals * flip).numpy().astype(np.float32)

    def scatter_mean_1d(arr: np.ndarray) -> np.ndarray:
        t = torch.tensor(arr, dtype=torch.float32)
        s = torch.zeros(V).scatter_add_(0, vids, t)
        return (s / count.clamp(min=1)).numpy().astype(np.float32)

    mean_intensity = scatter_mean_1d(intensity)
    mean_height    = scatter_mean_1d(height)

    # majority vote for semantic class
    sem_t  = torch.tensor(sem.astype(np.int64) - 1, dtype=torch.long)
    oh     = torch.zeros(N, N_CLASSES)
    oh.scatter_(1, sem_t.unsqueeze(1), 1.0)
    cc     = torch.zeros(V, N_CLASSES).scatter_add_(
        0, vids.unsqueeze(1).expand(-1, N_CLASSES), oh)
    class_norm = (cc.argmax(1).float() / max(N_CLASSES - 1, 1)).numpy().astype(np.float32)

    return grid, mean_xyz, normals, mean_intensity, mean_height, class_norm


# ---------------------------------------------------------------------------
# PoNQ → DiffusionTensor
# ---------------------------------------------------------------------------

def _ponq_to_dt(pg: PoNQ_grid, device: str) -> DiffusionTensor:
    def _t(x):
        return x.to(device) if isinstance(x, torch.Tensor) \
            else torch.tensor(x, dtype=torch.float32, device=device)
    n = pg.normals.shape[0] if isinstance(pg.normals, torch.Tensor) else len(pg.normals)
    return DiffusionTensor.get_tensor_from_data(
        pg.grid.to(device), _t(pg.normals), _t(pg.local_offset),
        _t(pg.colors), torch.ones(n, 1, device=device),
    )


# ---------------------------------------------------------------------------
# Core encoding from already-detrended, XY-origin-shifted points
# ---------------------------------------------------------------------------

def _encode_points(
    xyz:       np.ndarray,   # detrended + XY shifted to [0, CROP_SIZE_M)
    sem:       np.ndarray,
    intensity: np.ndarray,
    save_dir:  Path,
    voxel_size_initial: float = VOXEL_SIZE_INITIAL,
    initial_size:       int   = INITIAL_SIZE,
    targets:   List[int]      = TARGETS,
    device:    str            = DEVICE,
    verbose:   bool           = False,
) -> None:
    """Voxelise, pool, and save a pyramid for one set of points."""
    save_dir.mkdir(parents=True, exist_ok=True)
    height = np.clip(xyz[:, 2], 0.0, MAX_HEIGHT_M) / MAX_HEIGHT_M
    grid, mean_xyz, normals, mean_int, mean_h, cls_n = aggregate_voxels(
        xyz, sem, intensity, height, voxel_size_initial, device
    )
    colors_np = np.stack([mean_int, mean_h, cls_n], axis=-1).astype(np.float32)

    pg = PoNQ_grid(initial_size)
    pg.from_mesh(grid.to(device), mean_xyz, normals, colors_np, device=device)

    # Save the finest level (needed by the highest-level upsampler)
    pg.compute_local_offset()
    torch.save(_ponq_to_dt(pg, device), save_dir / f"{initial_size}.pt")

    size = initial_size
    for t_size in targets:
        pg = pg.get_pool(size // t_size)
        pg.compute_local_offset()
        torch.save(_ponq_to_dt(pg, device), save_dir / f"{t_size}.pt")
        size = t_size

    if verbose:
        print(f"    → {save_dir.name}  ({int(grid.total_voxels):,} voxels at {voxel_size_initial}m)")


# ---------------------------------------------------------------------------
# Crop splitting
# ---------------------------------------------------------------------------

def _crop_ids(xyz: np.ndarray,
              crop_size: float = CROP_SIZE_M,
              stride:    float = CROP_STRIDE_M) -> List[Tuple[float, float]]:
    """Return (x_start, y_start) pairs covering the tile."""
    x0, y0 = float(xyz[:, 0].min()), float(xyz[:, 1].min())
    x1, y1 = float(xyz[:, 0].max()), float(xyz[:, 1].max())
    xs = np.arange(x0, x1, stride)
    ys = np.arange(y0, y1, stride)
    return [(float(x), float(y)) for x in xs for y in ys]


def _extract_crop(xyz: np.ndarray, sem: np.ndarray, intensity: np.ndarray,
                  x_start: float, y_start: float,
                  crop_size: float = CROP_SIZE_M
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract points in [x_start, x_start+crop_size) × [y_start, y_start+crop_size).
    Shifts XY so the crop starts at (0, 0) — voxel indices begin near 0.
    """
    mask = (
        (xyz[:, 0] >= x_start) & (xyz[:, 0] < x_start + crop_size) &
        (xyz[:, 1] >= y_start) & (xyz[:, 1] < y_start + crop_size)
    )
    xyz_c = xyz[mask].copy()
    xyz_c[:, 0] -= x_start   # shift to local origin
    xyz_c[:, 1] -= y_start
    return xyz_c, sem[mask], intensity[mask]


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def encode_tile_as_crops(
    laz_path:  Path,
    out_root:  Path,
    crop_size:    float = CROP_SIZE_M,
    crop_stride:  float = CROP_STRIDE_M,
    voxel_size_initial: float = VOXEL_SIZE_INITIAL,
    initial_size: int   = INITIAL_SIZE,
    targets:   List[int] = TARGETS,
    device:    str       = DEVICE,
    verbose:   bool      = True,
    skip_complete: bool  = False,
) -> List[str]:
    """
    Encode a DALES LAZ tile as a grid of 50×50m crops.

    DTM is computed on the full tile for robustness, then each crop is
    detrended and voxelised independently.

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

    # Build DTM on full tile (better coverage than per-crop)
    if verbose:
        print("  Building DTM …")
    dtm_fn = build_dtm_interpolator(xyz, sem)

    # Detrend the whole tile at once
    ground_z = dtm_fn(xyz)
    xyz_d = xyz.copy()
    xyz_d[:, 2] -= ground_z

    crop_origins = _crop_ids(xyz_d, crop_size, crop_stride)
    saved_ids = []

    for x_start, y_start in tqdm(crop_origins, desc="Extracting crops"):
        xyz_c, sem_c, int_c = _extract_crop(xyz_d, sem, intensity, x_start, y_start, crop_size)
        if len(xyz_c) < MIN_POINTS_PER_CROP:
            continue

        # Crop ID encodes the tile + crop XY position (integer metres relative to tile origin)
        crop_id  = f"{tile_id}_x{int(x_start):04d}_y{int(y_start):04d}"
        save_dir = out_root / crop_id

        # Skip crops that already have every pyramid level on disk
        if skip_complete:
            all_levels = [initial_size] + list(targets)
            if all((save_dir / f"{lv}.pt").exists() for lv in all_levels):
                saved_ids.append(crop_id)
                continue

        _encode_points(xyz_c, sem_c, int_c, save_dir,
                       voxel_size_initial, initial_size, targets, device,
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
    laz_path:   Path,
    out_root:   Path,
    yaw_angles: List[float] = (0, 45, 90, 135, 180, 225, 270, 315),
    flip_axes:  List[int]   = (0, 1),
    crop_size:  float = CROP_SIZE_M,
    crop_stride: float = CROP_STRIDE_M,
    device:     str   = DEVICE,
    verbose:    bool  = False,
) -> None:
    """
    Encode all crops of a tile with rotation + flip augmentation.

    Augmented variants are saved as:
      {tile_id}_x{X}_y{Y}_r{angle}/
      {tile_id}_x{X}_y{Y}_f{axis}/
      {tile_id}_x{X}_y{Y}_r{angle}_f{axis}/

    Invariant #4 respected: rotation/flip applied at raw point level before voxelisation.
    DTM is computed ONCE on the full tile.
    """
    tile_id = laz_path.stem
    if verbose:
        print(f"[pc_encoding/aug] {tile_id}")

    xyz, sem, intensity = load_dales_laz(laz_path)
    dtm_fn   = build_dtm_interpolator(xyz, sem)
    ground_z = dtm_fn(xyz)
    xyz_d    = xyz.copy(); xyz_d[:, 2] -= ground_z
    crop_origins = _crop_ids(xyz_d, crop_size, crop_stride)

    for x_start, y_start in crop_origins:
        xyz_c, sem_c, int_c = _extract_crop(xyz_d, sem, intensity, x_start, y_start, crop_size)
        if len(xyz_c) < MIN_POINTS_PER_CROP:
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
                _encode_points(xyz_aug, sem_c, int_c, save_dir,
                               device=device, verbose=verbose)


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
    normals, positions, colors, _ = DiffusionTensor.get_feature_data(g.jdata)
    positions = positions.cpu().detach().numpy()
    normals_n = normals.cpu().detach().numpy()
    cls_col   = colors[:, 2:3].cpu().detach().numpy()
    rgb       = np.repeat(cls_col, 3, axis=1).clip(0, 1)

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
    parser.add_argument("--tile",  default=None, type=str, help="Tile ID (e.g. 5080_54435)")
    parser.add_argument("--all",   action="store_true",    help="Encode all tiles in --split")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--augment",       action="store_true", help="Add yaw/flip augmentations")
    parser.add_argument("--export_ply",    action="store_true", help="Export first crop's 256.pt as PLY")
    parser.add_argument("--skip_complete", action="store_true",
                        help="Skip crops that already have all pyramid levels on disk")
    parser.add_argument("--out",   default=str(GT_ROOT), type=str)
    parser.add_argument("--device", default=DEVICE, type=str)
    parser.add_argument("--crop_size", default=CROP_SIZE_M,   type=float)
    parser.add_argument("--crop_stride", default=CROP_STRIDE_M, type=float)
    args = parser.parse_args()

    out_root = Path(args.out) / args.split
    laz_dir  = DALES_ROOT / args.split

    tiles = ([args.tile] if args.tile else
             [p.stem for p in sorted(laz_dir.glob("*.laz"))] if args.all else None)
    if tiles is None:
        parser.print_help(); sys.exit(0)

    for tile_id in tiles:
        laz_path = laz_dir / f"{tile_id}.laz"
        if not laz_path.exists():
            print(f"WARNING: {laz_path} not found, skipping"); continue

        crop_ids = encode_tile_as_crops(
            laz_path, out_root,
            crop_size=args.crop_size, crop_stride=args.crop_stride,
            device=args.device, skip_complete=args.skip_complete,
        )

        if args.augment:
            encode_tile_as_crops_augmented(laz_path, out_root, device=args.device)

        if args.export_ply and crop_ids:
            pt = out_root / crop_ids[0] / "256.pt"
            if pt.exists():
                dt = torch.load(pt, weights_only=False)
                export_ply(dt, out_root / crop_ids[0] / "256_preview.ply")

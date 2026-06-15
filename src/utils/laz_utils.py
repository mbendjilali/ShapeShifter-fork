"""Load LAZ/LAS point clouds (laspy + LASzip) and estimate normals for ShapeShifter encoding."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    import laspy
    from laspy import LazBackend
except ImportError as e:  # pragma: no cover
    laspy = None  # type: ignore
    LazBackend = None  # type: ignore
    _LASPY_IMPORT_ERROR = e
else:
    _LASPY_IMPORT_ERROR = None


def _require_laspy() -> None:
    if laspy is None:
        raise RuntimeError(
            "laspy is required for LAZ support. Install with: pip install laspy laszip"
        ) from _LASPY_IMPORT_ERROR


def load_laz_points_colors(
    laz_path: str | Path,
    stride: int = 1,
    max_points: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns
    -------
    xyz : (N, 3) float64/float32 positions in file CRS/units
    rgb : (N, 3) float32 in [0, 1] or None if dimensions missing / no color
    """
    _require_laspy()
    path = Path(laz_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".laz":
        las = laspy.read(str(path), laz_backend=LazBackend.Laszip)
    else:
        las = laspy.read(str(path))

    n = las.header.point_count
    if n == 0:
        raise ValueError(f"No points in {path}")

    idx = np.arange(0, n, stride, dtype=np.int64)
    if max_points is not None and idx.size > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=max_points, replace=False)
        idx.sort()

    # laspy applies scale/offset in .x / .y / .z
    x = np.asarray(las.x[idx], dtype=np.float64)
    y = np.asarray(las.y[idx], dtype=np.float64)
    z = np.asarray(las.z[idx], dtype=np.float64)
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)

    rgb = None
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        r = np.asarray(las.red[idx], dtype=np.float32)
        g = np.asarray(las.green[idx], dtype=np.float32)
        b = np.asarray(las.blue[idx], dtype=np.float32)
        mx = max(float(r.max(initial=0)), float(g.max(initial=0)), float(b.max(initial=0)), 1.0)
        if mx > 255.5:
            rgb = np.stack([r, g, b], axis=-1) / 65535.0
        else:
            rgb = np.stack([r, g, b], axis=-1) / 255.0
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    return xyz, rgb


def estimate_normals_pca(xyz: np.ndarray, n_neighbors: int = 30) -> np.ndarray:
    """Unit normals from local PCA (smallest eigenvector)."""
    n = xyz.shape[0]
    k = min(max(3, n_neighbors), n)
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(xyz)
    _, indices = nn.kneighbors(xyz)
    normals = np.zeros_like(xyz)
    for i in range(n):
        nb = xyz[indices[i]]
        centered = nb - nb.mean(axis=0, keepdims=True)
        cov = (centered.T @ centered) / max(k - 1, 1)
        w, v = np.linalg.eigh(cov)
        nrm = v[:, np.argmin(w)]
        ln = np.linalg.norm(nrm)
        normals[i] = nrm / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return normals.astype(np.float32)


def orient_normals_toward_origin(xyz: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Flip so normals tend to point away from the coordinate origin (after centering)."""
    out = normals.copy()
    v = xyz
    d = (out * v).sum(axis=-1, keepdims=True)
    out *= np.where(d < 0, -1.0, 1.0)
    return out


def load_dales_points(
    laz_path: str | Path,
    stride: int = 1,
    max_points: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a DALES-2 LAZ tile.

    Returns
    -------
    xyz       : (N, 3) float32 in file metric CRS (no NDCnormalize)
    sem_class : (N,)  uint8   legacy_semantic values 1–8
    intensity : (N,)  float32 normalised to [0, 1]
    """
    _require_laspy()
    path = Path(laz_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    las = laspy.read(str(path), laz_backend=LazBackend.Laszip)
    n = las.header.point_count
    if n == 0:
        raise ValueError(f"No points in {path}")

    idx = np.arange(0, n, stride, dtype=np.int64)
    if max_points is not None and idx.size > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=max_points, replace=False)
        idx.sort()

    x = np.asarray(las.x[idx], dtype=np.float32)
    y = np.asarray(las.y[idx], dtype=np.float32)
    z = np.asarray(las.z[idx], dtype=np.float32)
    xyz = np.stack([x, y, z], axis=-1)

    sem = np.asarray(las.legacy_semantic[idx], dtype=np.uint8)
    intensity = np.asarray(las.intensity[idx], dtype=np.float32) / 65535.0

    return xyz, sem, intensity

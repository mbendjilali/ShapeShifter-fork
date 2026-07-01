"""Shared bootstrap + helpers for the standalone test scripts.

Run every test from the repo root, e.g.:
    python test/upsampler_vs_diffusion.py --level 1 --n_crops 16
"""
import glob
import os
import sys

# Make `src` importable, plus `src/utils` for helper.py's bare imports
# (`from fvdb_diffusion import ...`).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
from utils.diffusion_tensor import DiffusionTensor  # noqa: E402
from inference.inference import export_to_laz  # noqa: E402

CLASS_NAMES = [
    "Ground", "Vegetation", "Cars", "Trucks",
    "Fences", "PowerLines", "Poles", "Buildings",
]


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def export_dt(dt: DiffusionTensor, path: str) -> None:
    """Export a DiffusionTensor to .laz, coloured by semantic class."""
    positions, features, _ = DiffusionTensor.get_feature_data(dt.jdata)
    positions_np = positions.cpu().numpy()
    features_np = features.cpu().numpy()          # [intensity(1), class_probs(8)]
    intensity = features_np[:, 0]
    class_idx = features_np[:, 1:].argmax(axis=-1).clip(0, 7)
    export_to_laz(positions_np, intensity, class_idx, path)


def resolve_crop_path(crop: str, split: str = "test", root: str = "data/dales") -> str:
    """Return the crop directory from a full path or a bare crop ID."""
    if os.path.isdir(crop):
        return crop
    cand = os.path.join(root, split, crop)
    if os.path.isdir(cand):
        return cand
    raise FileNotFoundError(
        f"Cannot find crop directory: tried '{crop}' and '{cand}'."
    )


def list_crops(split: str = "test", root: str = "data/dales", n: int | None = None):
    """List crop directories for a split (optionally the first ``n``)."""
    dirs = sorted(d for d in glob.glob(os.path.join(root, split, "*")) if os.path.isdir(d))
    return dirs[:n] if n else dirs


def occ_iou(pred_occ: torch.Tensor, gt_occ: torch.Tensor) -> float:
    """IoU between two boolean occupancy masks over the same voxel set."""
    inter = (pred_occ & gt_occ).sum().float()
    union = (pred_occ | gt_occ).sum().clamp(min=1).float()
    return (inter / union).item()

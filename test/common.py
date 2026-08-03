"""Shared bootstrap + helpers for the standalone diagnostic scripts.

Every script under test/ imports from here.  Nothing in this module loads a
checkpoint or generates anything — it is pure plumbing (paths, IO, metrics) so
that a diagnostic never has to import another diagnostic's internals.

Run every script from the repo root, e.g.:
    python test/evaluation/upsampler_vs_diffusion.py --level 1 --n_crops 16
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
from utils.diffusion_tensor import DiffusionTensor  # noqa: E402
from inference.inference import export_to_laz  # noqa: E402

# Index i == DALES semantic label i+1, because encode_features one-hots `sem − 1`
# (src/encoding/point_cloud.py). The order is therefore fixed by the `classes`
# map in configs/dataset/dales.yaml — 5 = PowerLines, 6 = Fences — and must not
# be reordered here. (Indices 4/5 were swapped until 03-08-2026, which mislabelled
# every printed table and exported filename for those two classes.)
CLASS_NAMES = [
    "Ground", "Vegetation", "Cars", "Trucks",
    "PowerLines", "Fences", "Poles", "Buildings",
]
N_CLS = len(CLASS_NAMES)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Crop discovery / loading
# ---------------------------------------------------------------------------

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


def load_dt(path: str, device: str) -> DiffusionTensor:
    """Load a pyramid .pt as a DiffusionTensor on ``device``.

    Older crops were pickled as a bare grid/data pair rather than a
    DiffusionTensor, so re-wrap when needed.
    """
    o = torch.load(path, weights_only=False)
    if not isinstance(o, DiffusionTensor):
        o = DiffusionTensor(o.grid, o.data)
    return DiffusionTensor(o.grid.to(device), o.data.to(device))


def crop_pt(crop_path: str, resolution: int) -> str:
    """Path to the ``{resolution}.pt`` pyramid file inside a crop directory."""
    return os.path.join(crop_path, f"{resolution}.pt")


def level_resolutions(level: int, base_res: int = 16, upsample_fac: int = 2):
    """(coarse, fine) pyramid .pt labels for a level>0 diffusion.

    Level N refines base_res*fac**(N-1) → base_res*fac**N.
    """
    res1 = base_res * (upsample_fac ** (level - 1))
    return res1, upsample_fac * res1


def load_levelN_inputs(crop_path, res1, res2, upsample_fac, device):
    """Reproduce DALESDataset.load_crop_levelN for one crop, no dataset object.

    Returns (X, X_UP, X0): the coarse input, its trilinear upsample, and the
    fine ground truth filled onto the upsampled voxel set.
    """
    X = load_dt(crop_pt(crop_path, res1), device)
    X0_fine = load_dt(crop_pt(crop_path, res2), device)
    X_UP = X.trilinear_upsample(upsample_fac)
    X0 = DiffusionTensor.fill_upsampled_with_gt(X_UP, X0_fine)
    return X, X_UP, X0


def has_levels(crop_path: str, *resolutions: int) -> bool:
    """True when the crop has every requested pyramid resolution on disk."""
    return all(os.path.exists(crop_pt(crop_path, r)) for r in resolutions)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_dt(dt: DiffusionTensor, path: str) -> None:
    """Export a DiffusionTensor to .laz, coloured by semantic class."""
    positions, features, _ = DiffusionTensor.get_feature_data(dt.jdata)
    positions_np = positions.cpu().numpy()
    features_np = features.cpu().numpy()          # [intensity(1), class_probs(8)]
    intensity = features_np[:, 0]
    class_idx = features_np[:, 1:].argmax(axis=-1).clip(0, 7)
    export_to_laz(positions_np, intensity, class_idx, path)


# ---------------------------------------------------------------------------
# Metrics / statistics
# ---------------------------------------------------------------------------

def occ_iou(pred_occ: torch.Tensor, gt_occ: torch.Tensor) -> float:
    """IoU between two boolean occupancy masks over the same voxel set."""
    inter = (pred_occ & gt_occ).sum().float()
    union = (pred_occ | gt_occ).sum().clamp(min=1).float()
    return (inter / union).item()


def class_counts(dt, sample=False):
    """Per-class counts over occupied voxels.

    sample=True draws class ~ softmax (reproduces the marginal); sample=False
    uses argmax (mode → majority bias).
    """
    if dt.jdata.shape[0] == 0:
        return np.zeros(N_CLS, dtype=np.int64)
    probs = dt.jdata[:, 4:4 + N_CLS]
    if sample:
        p = probs.clamp(min=0)
        p = p / p.sum(-1, keepdim=True).clamp(min=1e-9)
        cls = torch.multinomial(p, 1).squeeze(-1).cpu().numpy()
    else:
        cls = probs.argmax(-1).cpu().numpy()
    return np.bincount(cls, minlength=N_CLS)


def data_stats(split, res, n_data, device):
    """Per-class counts (over occupied voxels) and occupancy fraction from real crops.

    Returns (counts, occupancy_fraction, n_crops_used).
    """
    counts = np.zeros(N_CLS, dtype=np.int64)
    occ_num = occ_den = used = 0
    for cp in list_crops(split):
        f = crop_pt(cp, res)
        if not os.path.exists(f):
            continue
        dt = load_dt(f, device)
        counts += class_counts(dt)
        dense = dt.to_custom_dense(empty_fill='zero')          # add empty voxels
        occ_num += int((dense.jdata[:, -1] > 0).sum())
        occ_den += dense.jdata.shape[0]
        used += 1
        if used >= n_data:
            break
    return counts, (occ_num / max(occ_den, 1)), used

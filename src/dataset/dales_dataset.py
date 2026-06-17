"""
dales_dataset.py — Multi-crop DALES dataloader for diffusion training.

Discovery: for each tile in the manifest, scans gt_root for directories matching
  {tile_id}_x????_y????/ that contain {base_resolution}.pt.
Each such directory is an independent crop sample (50×50m).

Sampling: weighted by rare-class presence (inherited from tile-level metadata).

Batching: fvdb.jcat of heterogeneous sparse grids.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

import torch
import fvdb
import fvdb.nn as fvnn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from diffusion_tensor import DiffusionTensor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DALESDataset:
    """
    Catalogue the available pre-encoded 50×50m crops and expose sampling.

    Parameters
    ----------
    manifest_path : str | Path
        Path to data/dales_manifest.json.
    split : 'train' | 'test'
        Which split to expose (test tiles are never used during training).
    upsample_fac : int
        Upsampling factor between consecutive levels (default 2).
    base_resolution : int
        Coarsest resolution label (default 16).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        weights_path: str | Path,
        split: str = "train",
        upsample_fac: int = 2,
        base_resolution: int = 16,
        sampling_ratio: float = 1.0,
        common_sampling_ratio: Optional[float] = None,
    ):
        self.upsample_fac = upsample_fac
        self.base_resolution = base_resolution

        with open(manifest_path) as f:
            manifest = json.load(f)

        records = manifest[split]
        self.gt_root = Path(manifest.get("gt_root", "data/dales"))
        self.sampling_ratio = manifest.get("sampling_ratio", sampling_ratio) 
        self.common_sampling_ratio = manifest.get("common_sampling_ratio", common_sampling_ratio)
        self.weights_path = manifest.get("weights_path", weights_path)

        self.crops: List[str] = []       # crop_ids

        with open(self.weights_path) as f:
            weights_by_tile = json.load(f)
    
        total_of_crops = 0
        for rec in records:
            tile_id = rec["id"]
            # glob for crops belonging to this tile
            ##### sorted donc ils sont plus dans le même ordre ?? 
            crop_dirs = sorted(self.gt_root.glob(f"{split}/{tile_id}_x*_y*/"))
            crop_weights = weights_by_tile[split][tile_id]
            num_zero = sum(w == 0 for w in crop_weights)
            print(f"{tile_id}: {len(crop_dirs)} crops, min weight={min(crop_weights):.4f}, max weight={max(crop_weights):.4f}, zero weights={num_zero}")

            total_of_crops += len(crop_dirs)

            common_crop = []
            weighted_crops = []
            weighted_values = []

            # 2 groups: common crops (weight=0) and weighted crops (weight>0)
            for crop_dir, weight in zip(crop_dirs, crop_weights):
                if weight == 0:
                    common_crop.append(crop_dir)
                else:
                    weighted_crops.append(crop_dir)
                    weighted_values.append(weight)

            n = int(len(crop_dirs) * self.sampling_ratio)
            n_common = int(n * self.common_sampling_ratio)

            sampled_common = random.sample(common_crop, n_common)
            sampled_rare = list(
                np.random.choice(weighted_crops, 
                                size=n - n_common, 
                                replace=False,
                                p=weighted_values)
            )
            
            sampled_crops = sampled_common + sampled_rare

            for d in sampled_crops:
                if (d / f"{base_resolution}.pt").exists():
                    self.crops.append(d.name)   # e.g. "5080_54435_x0000_y0050"
        print(f"Sampling {len(sampled_crops)} crops among {total_of_crops}.")

        if not self.crops:
            raise RuntimeError(
                f"No encoded crops found under {self.gt_root} for split={split}. "
                "Run: python src/shape_encoding/pc_encoding.py --all --split train"
            )


    # ------------------------------------------------------------------
    # GT loading
    # ------------------------------------------------------------------

    def _load_dt(self, path: Path, device: str) -> DiffusionTensor:
        """Load a .pt file as DiffusionTensor on the target device."""
        obj = torch.load(path, weights_only=False)
        if not isinstance(obj, DiffusionTensor):
            obj = DiffusionTensor(obj.grid, obj.data)
        return DiffusionTensor(obj.grid.to(device), obj.data.to(device))

    def load_crop_level0(self, split: str, crop_id: str, device: str = "cuda") -> DiffusionTensor:
        """Level 0: coarsest-resolution DiffusionTensor → dense."""
        res = self.base_resolution
        X0 = self._load_dt(self.gt_root / split / crop_id / f"{res}.pt", device)
        return X0.to_custom_dense()

    def load_crop_levelN(
        self, split: str, crop_id: str, level: int, device: str = "cuda"
    ) -> Tuple[DiffusionTensor, DiffusionTensor, DiffusionTensor]:
        """
        Level N>0: load coarse + fine, return (X, X_UP, X0_filled).
        res_1 = base × 2^(level-1),  res_2 = 2 × res_1
        """
        res_1 = self.base_resolution * (self.upsample_fac ** (level - 1))
        res_2 = self.upsample_fac * res_1
        X  = self._load_dt(self.gt_root / split / crop_id / f"{res_1}.pt", device)
        X0 = self._load_dt(self.gt_root / split / crop_id / f"{res_2}.pt", device)
        X_UP = X.trilinear_upsample(self.upsample_fac)
        X0   = DiffusionTensor.fill_upsampled_with_gt(X_UP, X0)
        return X, X_UP, X0

    def sample_crop_ids(self, batch_size: int) -> List[str]:
        """Sample batch_size crop IDs (with replacement, weighted)."""
        return random.choices(self.crops, k=batch_size)

    def sample_batch(
        self,
        split: str,
        level: int,
        batch_size: int,
        device: str = "cuda",
    ):
        """
        Sample a genuine multi-crop batch.

        Returns
        -------
        level == 0:
            X0  — batched dense DiffusionTensor (via fvdb.jcat)
        level > 0:
            (X, X_UP, X0)  — batched coarse / upsampled / fine
        """
        crop_ids = self.sample_crop_ids(batch_size)

        if level == 0:
            tensors = [self.load_crop_level0(split, c, device) for c in crop_ids]
            return _jcat_dt(tensors)

        X_list, XUP_list, X0_list = [], [], []
        for c in crop_ids:
            X, X_UP, X0 = self.load_crop_levelN(split, c, level, device)
            X_list.append(X)
            XUP_list.append(X_UP)
            X0_list.append(X0)
        return _jcat_dt(X_list), _jcat_dt(XUP_list), _jcat_dt(X0_list)

    # ------------------------------------------------------------------
    # Held-out validation
    # ------------------------------------------------------------------

    @classmethod
    def test_set(
        cls, manifest_path: str | Path, **kw
    ) -> "DALESDataset":
        return cls(manifest_path, split="test", **kw)

    def compute_val_loss(
        self,
        diffusion,
        split: str,
        level: int,
        n_crops: int = 4,
        clip_size: int = 20,
        device: str = "cuda",
    ) -> Optional[float]:
        """Average diffusion loss on a random subset of held-out crops."""
        if not self.crops:
            return None
        sample_ids = random.sample(self.crops, min(n_crops, len(self.crops)))
        losses = []
        for crop_id in sample_ids:
            try:
                if level == 0:
                    X0 = self.load_crop_level0(split, crop_id, device)
                    with torch.no_grad():
                        loss = diffusion(X0).item()
                else:
                    X, X_UP, X0 = self.load_crop_levelN(split, crop_id, level, device)
                    with torch.no_grad():
                        X0_BLUR = diffusion.model_upsampler(X, X_UP).detach()
                        X0_BLUR.grid = X0.grid
                        x0c, blurc = clip_data_per_element(X0, X0_BLUR, clip_size)
                        loss = diffusion(x0c, blurc).item()
                losses.append(loss)
            except Exception:
                pass
        return sum(losses) / len(losses) if losses else None

    # ------------------------------------------------------------------
    # Debug / stats
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.crops)

    def __repr__(self) -> str:
        return f"DALESDataset({len(self.crops)} crops, base_res={self.base_resolution})"


# ---------------------------------------------------------------------------
# Batching helpers
# ---------------------------------------------------------------------------

def _jcat_dt(tensors: List[DiffusionTensor]) -> DiffusionTensor:
    """Concatenate a list of DiffusionTensors into a batch via fvdb.jcat."""
    grids = fvdb.jcat([t.grid for t in tensors])
    data  = fvdb.jcat([t.data for t in tensors])
    return DiffusionTensor(grids, data)


def clip_data_per_element(
    X0:      fvnn.VDBTensor,
    X0_BLUR: fvnn.VDBTensor,
    size:    int,
) -> Tuple[fvnn.VDBTensor, fvnn.VDBTensor]:
    """
    Crop one random patch per batch element independently, then re-jcat.

    Picks a random voxel per element as centre, clips ±size in ijk space.
    Uses the batched clip API (single call, no loop) to avoid device mismatches.
    """
    B = X0.grid_count
    if B == 0:
        return X0, X0_BLUR

    device = X0.grid.device
    centers = []
    for b in range(B):
        ijk_b = X0.grid[b].ijk.jdata   # (V_b, 3) on device
        if len(ijk_b) == 0:
            centers.append(torch.zeros(1, 3, dtype=torch.int32, device=device))
        else:
            ind = torch.randint(0, len(ijk_b), (1,), device=device)
            centers.append(ijk_b[ind])  # (1, 3)

    centers = torch.cat(centers, dim=0)             # (B, 3)
    lo = centers - size                              # (B, 3)
    hi = centers + size                              # (B, 3)

    cf,  cg  = X0.grid.clip(X0.data, lo, hi)
    cf2, cg2 = X0_BLUR.grid.clip(X0_BLUR.data, lo, hi)
    return fvnn.VDBTensor(cg, cf), fvnn.VDBTensor(cg2, cf2)

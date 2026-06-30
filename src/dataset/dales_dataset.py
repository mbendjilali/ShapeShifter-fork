"""
dales_dataset.py — Multi-crop DALES dataloader for diffusion training.

Discovery: for each tile in the manifest, scans gt_root for directories matching
  {tile_id}_x????_y????/ that contain {base_resolution}.pt.
Each such directory is an independent crop sample (50×50m).

Sampling: weighted by inverse vegetation fraction (crops with less vegetation are sampled more often).

Batching: fvdb.jcat of heterogeneous sparse grids.
"""

from __future__ import annotations

import yaml
import random
from pathlib import Path
from typing import List, Optional, Tuple

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
        Path to data/dales_manifest.yaml.
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
        split: str = "train",
        upsample_fac: int = 2,
        base_resolution: int = 16,
        sampling_ratio: float = 1.0,
        common_sampling_ratio: Optional[float] = None,
        preload: bool = True,
        level: Optional[int] = None,
        clip_size: Optional[int] = None,
        random_crop: bool = False,
        empty_fill: str = 'blur',
    ):
        self.upsample_fac = upsample_fac
        self.base_resolution = base_resolution
        self.clip_size = clip_size
        # empty_fill='zero' makes empty (mask=-1) voxels neutral instead of
        # blurred-neighbour-filled, removing the train/inference mismatch (item 3).
        self.empty_fill = empty_fill
        # random_crop=True: level-0 windows are centered at a random *grid* location
        # (occupancy-agnostic), not a random occupied voxel — so borders / sparse
        # regions are seen and training matches the full-tile inference distribution
        # (item 3).  Clips are then re-sampled fresh each epoch (no dense cache).
        self.random_crop = random_crop

        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        records = manifest[split]
        self.gt_root = Path(manifest.get("gt_root", "data/dales"))
        self.sampling_ratio = manifest.get("sampling_ratio", sampling_ratio) 
        self.common_sampling_ratio = manifest.get("common_sampling_ratio", common_sampling_ratio)

        self.crops: List[str] = []


        total_of_crops = 0
        for rec in records:
            tile_id = rec["id"]
            crop_dirs = sorted(self.gt_root.glob(f"{split}/{tile_id}_x*_y*/"))


            crop_weights = [1.0] * len(crop_dirs)
            crop_dirs_weights = list(zip(crop_dirs, crop_weights))
            crop_dirs_weights.sort(key=lambda x: x[1], reverse=True)
            crop_dirs, crop_weights = zip(*crop_dirs_weights) if crop_dirs_weights else ([], [])
            crop_dirs = list(crop_dirs)

            total_of_crops += len(crop_dirs)

            n = int(len(crop_dirs) * self.sampling_ratio)
            sampled_crops = crop_dirs[:n]

            for d in sampled_crops:
                if (d / f"{base_resolution}.pt").exists():
                    self.crops.append(d.name)

        if not self.crops:
            pass  # RuntimeError raised below


        print(f"Sampling {len(self.crops)} crops among {total_of_crops}.")

        if not self.crops:
            raise RuntimeError(
                f"No encoded crops found under {self.gt_root} for split={split}. "
                "Run: python src/shape_encoding/pc_encoding.py --all --split train"
            )

        self._cache: dict[str, "DiffusionTensor"] = {}
        self._level_cache: dict[str, tuple] = {}  # crop_id -> (X_cpu, X_UP_cpu, Y_cpu)
        self._dense_cache: dict[str, "DiffusionTensor"] = {}  # crop_id -> dense X0_cpu (level 0)
        self._preloaded_level: Optional[int] = None
        if preload:
            self._preload(split, level)


    # ------------------------------------------------------------------
    # GT loading
    # ------------------------------------------------------------------

    def _preload(self, split: str, level: Optional[int] = None) -> None:
        """Load all crop .pt files into CPU RAM and optionally precompute (X, X_UP, Y)."""
        from tqdm import tqdm
        device = "cuda" if torch.cuda.is_available() else "cpu"
        desc = f"Caching crops (+ level-{level} precompute)" if level is not None else "Caching crops"
        for crop_id in tqdm(self.crops, desc=desc):
            # Load raw .pt files into CPU cache first
            crop_dir = self.gt_root / split / crop_id
            for pt_file in crop_dir.glob("*.pt"):
                key = str(pt_file)
                if key not in self._cache:
                    obj = torch.load(pt_file, weights_only=False)
                    if not isinstance(obj, DiffusionTensor):
                        obj = DiffusionTensor(obj.grid, obj.data)
                    self._cache[key] = DiffusionTensor(obj.grid.to("cpu"), obj.data.to("cpu"))

            # Precompute on GPU, store back on CPU.
            # In random_crop mode we intentionally do NOT freeze a dense level-0
            # clip — clips must be re-sampled fresh each epoch — so we only keep
            # the raw sparse tensor in self._cache (loaded above) and clip on the fly.
            if level == 0 and not self.random_crop:
                X0 = self.load_crop_level0(split, crop_id, device)
                self._dense_cache[crop_id] = DiffusionTensor(X0.grid.to("cpu"), X0.data.to("cpu"))
            elif level is not None and level != 0:
                X, X_UP, Y = self.load_crop_levelN(split, crop_id, level, device)
                self._level_cache[crop_id] = (
                    DiffusionTensor(X.grid.to("cpu"),    X.data.to("cpu")),
                    DiffusionTensor(X_UP.grid.to("cpu"), X_UP.data.to("cpu")),
                    DiffusionTensor(Y.grid.to("cpu"),    Y.data.to("cpu")),
                )

        self._preloaded_level = level
        if level == 0 and self.random_crop:
            suffix = ", level-0 random crops (clipped fresh each epoch, no dense cache)"
        elif level == 0:
            suffix = f", {len(self._dense_cache)} dense tensors precomputed (level 0)"
        elif level is not None:
            suffix = f", {len(self._level_cache)} (X,X_UP,Y) tuples for level {level}"
        else:
            suffix = ""
        print(f"Cached {len(self._cache)} files{suffix}.")

    def _load_dt(self, path: Path, device: str) -> DiffusionTensor:
        """Load a .pt file as DiffusionTensor on the target device, using RAM cache if available."""
        key = str(path)
        if key in self._cache:
            cached = self._cache[key]
            return DiffusionTensor(cached.grid.to(device), cached.data.to(device))
        obj = torch.load(path, weights_only=False)
        if not isinstance(obj, DiffusionTensor):
            obj = DiffusionTensor(obj.grid, obj.data)
        return DiffusionTensor(obj.grid.to(device), obj.data.to(device))

    def load_crop_level0(self, split: str, crop_id: str, device: str = "cuda") -> DiffusionTensor:
        """Level 0: coarsest-resolution DiffusionTensor → [clip] → dense.

        When clip_size is set, a random sub-patch is clipped from the *sparse*
        tensor before densification.  This is essential at large base resolutions
        where the bounding box of a single crop would otherwise produce hundreds
        of thousands of dense voxels and cause OOM.
        """
        res = self.base_resolution
        X0 = self._load_dt(self.gt_root / split / crop_id / f"{res}.pt", device)
        if self.clip_size is not None:
            ijk = X0.grid.ijk.jdata
            if len(ijk) > 0:
                cf, cg = self._clip_window(X0, ijk)
                X0 = DiffusionTensor(cg, cg.jagged_like(cf.jdata))
        return X0.to_custom_dense(empty_fill=self.empty_fill)

    def _clip_window(self, X0: DiffusionTensor, ijk: torch.Tensor):
        """Clip a `clip_size` half-extent window around a center.

        random_crop: center is a uniform random grid location in the tile bbox
        (occupancy-agnostic) — so the window can include borders and empty space,
        matching the full-tile inference distribution.  Falls back to an
        occupancy-centered window if a random one happens to be empty.
        Otherwise: legacy behavior, center on a random occupied voxel.
        """
        def clip_at(center):
            lo = center - self.clip_size
            hi = center + self.clip_size
            return X0.grid.clip(X0.data, lo, hi)

        if not self.random_crop:
            center = ijk[torch.randint(0, len(ijk), (1,), device=ijk.device)]
            return clip_at(center)

        lo_b = ijk.min(0).values
        hi_b = ijk.max(0).values
        span = (hi_b - lo_b + 1).clamp(min=1)
        for _ in range(4):  # retry a few random centers before falling back
            center = (lo_b + (torch.rand(3, device=ijk.device) * span).long()).view(1, 3)
            cf, cg = clip_at(center)
            if len(cg.ijk.jdata) > 0:
                return cf, cg
        # degenerate (empty window): fall back to an occupancy-centered clip
        center = ijk[torch.randint(0, len(ijk), (1,), device=ijk.device)]
        return clip_at(center)

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
        """Sample batch_size crop IDs with replacement, biased toward low-vegetation crops."""
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
            if self._dense_cache:
                tensors = []
                for c in crop_ids:
                    cached = self._dense_cache[c]
                    tensors.append(DiffusionTensor(cached.grid.to(device), cached.data.to(device)))
                return _jcat_dt(tensors), crop_ids
            tensors = [self.load_crop_level0(split, c, device) for c in crop_ids]
            return _jcat_dt(tensors), crop_ids

        # Fast path: precomputed tuples, only CPU→GPU transfer + jcat
        if level == self._preloaded_level and self._level_cache:
            X_list, XUP_list, X0_list = [], [], []
            for c in crop_ids:
                X_cpu, XUP_cpu, Y_cpu = self._level_cache[c]
                X_list.append(DiffusionTensor(X_cpu.grid.to(device),   X_cpu.data.to(device)))
                XUP_list.append(DiffusionTensor(XUP_cpu.grid.to(device), XUP_cpu.data.to(device)))
                X0_list.append(DiffusionTensor(Y_cpu.grid.to(device),   Y_cpu.data.to(device)))
            return _jcat_dt(X_list), _jcat_dt(XUP_list), _jcat_dt(X0_list), crop_ids

        # Slow path: compute trilinear upsample on the fly
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
        mse_losses = []
        bce_losses = []
        occ_losses = []
        occ_ious = []
        occ_only_mses = []
        occ_only_ces = []
        bin_bce_sum = bin_iou_sum = bin_cnt = None
        for crop_id in sample_ids:
            if level == 0:
                if crop_id in self._dense_cache:
                    cached = self._dense_cache[crop_id]
                    X0 = DiffusionTensor(cached.grid.to(device), cached.data.to(device))
                else:
                    X0 = self.load_crop_level0(split, crop_id, device)
                with torch.no_grad():
                    mse_loss, bce_loss, occ_loss, _, _, occ_iou, metrics = diffusion(X0)
            else:
                X, X_UP, X0 = self.load_crop_levelN(split, crop_id, level, device)
                with torch.no_grad():
                    X0_BLUR = diffusion.model_upsampler(X, X_UP).detach()
                    X0_BLUR.grid = X0.grid
                    x0c, blurc = clip_data_per_element(X0, X0_BLUR, clip_size)
                    mse_loss, bce_loss, occ_loss, _, _, occ_iou, metrics = diffusion(x0c, blurc)
            mse_losses.append(mse_loss)
            bce_losses.append(bce_loss)
            occ_losses.append(occ_loss)
            occ_ious.append(occ_iou)
            occ_only_mses.append(metrics['occ_only_mse'])
            occ_only_ces.append(metrics['occ_only_ce'])
            # NaN-aware accumulation of the per-σ buckets across crops.
            b_sig, i_sig = metrics['bce_per_sigma'], metrics['iou_per_sigma']
            if bin_bce_sum is None:
                bin_bce_sum = torch.zeros_like(b_sig)
                bin_iou_sum = torch.zeros_like(i_sig)
                bin_cnt     = torch.zeros_like(b_sig)
            valid = ~torch.isnan(b_sig)
            bin_bce_sum[valid] += b_sig[valid]
            bin_iou_sum[valid] += i_sig[valid]
            bin_cnt[valid]     += 1

        val_metrics = {
            'occ_only_mse': sum(occ_only_mses) / len(occ_only_mses),
            'occ_only_ce':  sum(occ_only_ces) / len(occ_only_ces),
            'bce_per_sigma': bin_bce_sum / bin_cnt.clamp(min=1),
            'iou_per_sigma': bin_iou_sum / bin_cnt.clamp(min=1),
        }
        return (sum(mse_losses) / len(mse_losses),
                sum(bce_losses) / len(bce_losses),
                sum(occ_losses) / len(occ_losses),
                sum(occ_ious) / len(occ_ious),
                val_metrics)

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

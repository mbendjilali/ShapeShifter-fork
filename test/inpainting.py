"""Spatial inpainting: hide voxels, add noise, denoise, compare to original.

The experiment asks whether the diffusion model can reconstruct a region that
was deliberately removed from a crop.  Three .laz files are exported so you can
compare them side-by-side in a point-cloud viewer (e.g. CloudCompare):

  original.laz              — ground truth (nothing hidden)
  hidden.laz                — crop with the erased region (the "hole")
  recon_hX.XX_tY.YY.laz    — model's reconstruction

Steps
-----
1. Load the clean crop and densify it (matching the model's training distribution).
2. Build a boolean mask selecting ~``hide_fraction`` of voxels according to
   ``--mask_mode`` (see below).
3. Noise the **clean** dense grid at level ``--t`` (hidden voxels still have mask=+1
   at this point), then overwrite the hidden voxels with pure random noise.
   This avoids biasing the hidden mask channel toward -1, which would otherwise
   cause the model to predict those voxels as empty during reverse diffusion.
4. Run the reverse diffusion process from t → 0 to reconstruct.
5. Export the three files above.

Mask modes
----------
slab    (default) Erase a contiguous X-axis slab — the first ``hide_fraction``
                  of the X-voxel range.  Clean, easy to visualise.
patches           Randomly erase ``hide_fraction`` of (X, Y) columns so the
                  hidden region looks irregular and patchy.  Use ``--seed`` to
                  reproduce the same selection.

Usage
-----
Slab — hide the first half of the crop:
    python test/inpainting.py --level 0 --crop 5080_54400_x0000_y0000

Patchy random holes (~50 % of columns, reproducible):
    python test/inpainting.py --level 0 --crop 5080_54400_x0000_y0000 \\
        --mask_mode patches --seed 42

Patchy, milder noise, 25 % of columns erased:
    python test/inpainting.py --level 0 --crop 5080_54400_x0000_y0000 \\
        --mask_mode patches --hide_fraction 0.25 --t 0.8 --seed 0
"""
import argparse
import os
import time
from pathlib import Path

import torch

from common import get_device, resolve_crop_path, export_dt
from utils.diffusion_tensor import DiffusionTensor
from utils.helper import reverse_from
from inference.inference import load_dales_diffusion, LEVEL0_PYRAMID_RES
from torch.special import expm1
from fvdb_diffusion import log_snr_to_alpha_sigma, log


def build_hidden_mask(
    ijk: torch.Tensor,
    hide_fraction: float,
    mode: str,
    seed: int,
) -> torch.Tensor:
    """Return a boolean mask (N_voxels,) — True for voxels to erase.

    Parameters
    ----------
    ijk:          Integer voxel coordinates, shape (N, 3).
    hide_fraction: Target fraction of voxels/columns to erase, in (0, 1).
    mode:         'slab'    — contiguous X-axis slab (first hide_fraction of X range).
                  'patches' — randomly chosen (X, Y) columns (~hide_fraction of them).
    seed:         Random seed used only by 'patches' mode.
    """
    if mode == 'slab':
        i_min = ijk[:, 0].min().item()
        i_max = ijk[:, 0].max().item()
        i_split = i_min + int(round((i_max - i_min) * hide_fraction))
        return ijk[:, 0] <= i_split

    if mode == 'patches':
        # Randomly select hide_fraction of distinct (X, Y) columns and mark
        # every voxel in those columns as hidden.
        unique_xy = torch.unique(ijk[:, :2], dim=0)   # (K, 2)
        K = unique_xy.shape[0]
        n_hide = max(1, int(round(K * hide_fraction)))

        rng = torch.Generator()
        rng.manual_seed(seed)
        perm = torch.randperm(K, generator=rng)
        hidden_xy = unique_xy[perm[:n_hide]]           # (n_hide, 2)

        # Encode (X, Y) as a single integer key for a fast isin lookup.
        scale = int(ijk[:, 1].max().item() - ijk[:, 1].min().item()) + 1
        key_all    = ijk[:, 0] * scale + ijk[:, 1]
        key_hidden = hidden_xy[:, 0] * scale + hidden_xy[:, 1]
        return torch.isin(key_all, key_hidden)

    raise ValueError(f"Unknown mask_mode {mode!r}. Choose 'slab' or 'patches'.")


@torch.no_grad()
def repaint_reverse(diff, noisy_grid, X_real, hidden_mask, t_start, steps=None):
    """RePaint-style denoising for inpainting (arXiv:2201.09865).

    At each step t_i \u2192 t_{i-1}:
      1. Model predicts x_0 from the current noisy state.
      2. Standard DDPM reverse step advances all voxels to t_{i-1}.
      3. Known voxels (~hidden_mask) are overwritten with q_sample(X_real, t_{i-1}),
         re-anchoring them to the real data at the correct noise level for that step.

    This gives the model a progressively cleaner, always-correct view of the known
    region at every step, while the hidden voxels are generated freely from context.
    """
    dev = diff.device
    steps = steps or max(2, int(round(t_start * diff.timesteps)) + 1)
    ts = torch.linspace(float(t_start), 0., steps, device=dev)
    N = X_real.jdata.shape[0]
    known = ~hidden_mask

    for i in range(steps - 1):
        time, time_next = ts[i:i+1], ts[i+1:i+2].clamp(min=0.)

        x_start = diff.model(noisy_grid, time.repeat(len(noisy_grid.jidx)))
        diff._sigmoid_semantic_channels(x_start)

        if time_next.item() == 0:
            return x_start

        # Standard DDPM reverse step for all voxels
        ls, lsn = diff.log_snr(time), diff.log_snr(time_next)
        a, sg   = log_snr_to_alpha_sigma(ls)
        an, sgn = log_snr_to_alpha_sigma(lsn)
        c = -expm1(ls - lsn)
        mean = an * (noisy_grid.jdata * (1 - c) / a + c * x_start.jdata)
        var  = (sgn**2) * c
        noisy_grid.data.jdata = mean + (0.5 * log(var)).exp() * torch.randn_like(noisy_grid.jdata)

        # RePaint: replace known voxels with the real data re-noised at t_next.
        # The model always sees the true (progressively denoised) visible context.
        t_next_all = time_next.expand(N)
        known_noisy, _ = diff.q_sample(X_real, t_next_all)
        noisy_grid.data.jdata[known] = known_noisy.jdata[known]

    return noisy_grid


@torch.no_grad()
def run(diff, crop_path, level, pyramid_res, t, hide_fraction, mask_mode, seed, out_dir, repaint=True):
    """Hide a spatial region, add noise, denoise, and export results.

    Parameters
    ----------
    diff:          loaded SparseDiffusion model (already .eval())
    crop_path:     path to the crop directory containing <resolution>.pt
    level:         diffusion pyramid level (0 = coarsest)
    pyramid_res:   base .pt label (file = pyramid_res * 2**level)
    t:             noise fraction in [0, 1]; 1.0 = pure noise
    hide_fraction: fraction of voxels/columns to erase, in (0, 1)
    mask_mode:     'slab' or 'patches' (see build_hidden_mask)
    seed:          random seed for 'patches' mode
    out_dir:       directory where the three .laz files are written
    repaint:       if True, use RePaint re-anchoring at each step (recommended)
    """
    resolution = pyramid_res * (2 ** level)
    pt_file = Path(crop_path) / f"{resolution}.pt"
    if not pt_file.exists():
        raise FileNotFoundError(
            f"Crop file not found: {pt_file} "
            f"(resolution {resolution} for level {level})."
        )

    print(f"[inpainting] loading {pt_file}")
    obj = torch.load(pt_file, weights_only=False)
    if not isinstance(obj, DiffusionTensor):
        obj = DiffusionTensor(obj.grid, obj.data)
    X_sparse = DiffusionTensor(obj.grid.to(diff.device), obj.data.to(diff.device))

    # Densify to a regular bbox grid.
    # empty_fill='zero' → empty voxels have all-zero features + mask = -1,
    # matching the training distribution (no blurred neighbours leaking in).
    X_dense = X_sparse.to_custom_dense(empty_fill='zero')

    os.makedirs(out_dir, exist_ok=True)
    crop_name = Path(crop_path).name

    # --- Export original (sparse, occupied voxels only) ---
    orig_path = os.path.join(out_dir, f"{crop_name}_original.laz")
    export_dt(X_sparse.get_global().remove_mask(), orig_path)
    print(f"  original         → {orig_path}")

    # --- Define the hidden region ---
    ijk = X_dense.grid.ijk.jdata          # (N_voxels, 3)
    hidden_mask = build_hidden_mask(ijk, hide_fraction, mask_mode, seed)

    n_hidden = int(hidden_mask.sum())
    print(f"  hiding {n_hidden}/{ijk.shape[0]} voxels  "
          f"(mode={mask_mode!r}, fraction={hide_fraction:.2f}, seed={seed})")

    # Zero all feature channels and flip mask to -1 for the hidden voxels
    X_holed_data = X_dense.jdata.clone()
    X_holed_data[hidden_mask, :-1] = 0.0   # offset, intensity, class_probs → 0
    X_holed_data[hidden_mask,  -1] = -1.0  # mark as empty
    X_holed = DiffusionTensor(X_dense.grid, X_dense.grid.jagged_like(X_holed_data))

    # --- Export the holed version (only surviving occupied voxels) ---
    hidden_path = os.path.join(out_dir, f"{crop_name}_hidden_{mask_mode}.laz")
    export_dt(X_holed.get_global().remove_mask(), hidden_path)
    print(f"  holed input      → {hidden_path}")

    # --- Noise the whole clean dense grid at level t ---
    # We noise X_dense so every voxel starts from its real feature values
    # (mask=+1 for occupied, -1 for empty bbox slots).  At t=1.0, alpha~0
    # so the real features have negligible influence (essentially pure noise).
    tt = torch.full((X_dense.jdata.shape[0],), t, device=diff.device)
    noisy, _ = diff.q_sample(X_dense, tt)

    if not repaint:
        # Without RePaint re-anchoring, the noised mask=+1 of hidden-but-occupied
        # voxels would trivially signal their presence to the model.  Replace them
        # with pure randn so the model has a neutral, context-free starting point.
        noisy.jdata[hidden_mask] = torch.randn(
            int(hidden_mask.sum()), noisy.jdata.shape[1], device=diff.device
        )

    # --- Denoise: iterate from t → 0 ---
    method = "repaint" if repaint else "basic"
    t0 = time.time()
    if repaint:
        # RePaint re-anchors the known voxels at every step — the model always
        # sees the true context while generating the hidden region freely.
        recon_vdb = repaint_reverse(diff, noisy, X_dense, hidden_mask, t_start=t)
    else:
        recon_vdb = reverse_from(diff, noisy, t_start=t)
    recon = DiffusionTensor.from_vdb(recon_vdb).get_global().remove_mask(threshold= 0.3)

    out_path = os.path.join(out_dir, f"{crop_name}_recon_{method}_{mask_mode}_h{hide_fraction:.2f}_t{t:.2f}.laz")
    export_dt(recon, out_path)
    print(f"  reconstruction   → {out_path}  "
          f"(pts={recon.jdata.shape[0]}, {time.time()-t0:.1f}s)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--level", type=int, default=0,
                   help="Diffusion pyramid level to load (0=coarsest).")
    p.add_argument("--src", default="checkpoints/diffusion_models/",
                   help="Directory containing checkpoint .pt files.")
    p.add_argument("--crop", required=True,
                   help="Full path or bare crop ID resolved via --split.")
    p.add_argument("--split", default="test",
                   help="Dataset split used when resolving a bare crop ID.")
    p.add_argument("--t", type=float, default=1.0,
                   help="Noise level in [0, 1]. 1.0 = pure noise; 0.5 = half-corrupted.")
    p.add_argument("--hide_fraction", type=float, default=0.5,
                   help="Fraction of voxels/columns to erase (0–1).")
    p.add_argument("--mask_mode", choices=["slab", "patches"], default="slab",
                   help="'slab': contiguous X-axis cut. 'patches': random patchy XY columns.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for 'patches' mode.")
    p.add_argument("--no_repaint", action="store_true",
                   help=("Disable RePaint re-anchoring and use plain reverse_from. "
                         "The model gets no feedback from the visible region during "
                         "denoising — pure unconditional generation in the hole."))
    p.add_argument("--pyramid_res", type=int, default=LEVEL0_PYRAMID_RES,
                   help="Base .pt label; crop file at level N is pyramid_res * 2**level.")
    p.add_argument("--out", default=None,
                   help="Output directory. Defaults to output/tests/level{level}.")
    args = p.parse_args()

    diff = load_dales_diffusion(args.level, args.src)
    diff.eval()
    crop_path = resolve_crop_path(args.crop, args.split)
    out_dir = args.out or f"output/tests/level{args.level}"

    repaint = not args.no_repaint
    print(f"level={args.level}  crop={crop_path}")
    print(f"t={args.t}  hide_fraction={args.hide_fraction}  "
          f"mask_mode={args.mask_mode}  seed={args.seed}  "
          f"repaint={repaint}  max_T={diff.max_T}/{diff.timesteps}")

    run(
        diff, crop_path, args.level, args.pyramid_res,
        args.t, args.hide_fraction, args.mask_mode, args.seed, out_dir,
        repaint=repaint,
    )
    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()

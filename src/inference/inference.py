if True:
    import sys
    sys.path.append('./src')
import argparse
import os
import sys
import laspy
from utils.fvdb_utils import grid_to_VDB
from utils.diffusion_tensor import DiffusionTensor
import numpy as np
import torch
import time


import utils.diffusion_tensor as _diffusion_tensor
sys.modules.setdefault('diffusion_tensor', _diffusion_tensor)


# Level-0 coarse layout — must match configs/training/diffusion_0.yaml and 16.pt
# encoding (3.2 m voxels, 100 m crop → ~32×32×7 dense bbox).
LEVEL0_NX = 32
LEVEL0_NZ = 7
LEVEL0_VOXEL_SIZE = 3.2
LEVEL0_PYRAMID_RES = 16  # coarsest .pt filename label (not the XY voxel count)


def compute_canonical_base_grid(
    nx: int = LEVEL0_NX,
    nz: int = LEVEL0_NZ,
    voxel_size: float = LEVEL0_VOXEL_SIZE,
    batch: int = 1,
    device: str = "cuda",
):
    """
    DALES unconditional sampling: a fully-occupied dense base grid.

    Must match level-0 training geometry (``diffusion_0.yaml``): 100 m crop,
    3.2 m voxels on ``16.pt`` → roughly 32×32×7.  ``LEVEL0_PYRAMID_RES`` (16)
    is the pyramid *filename* label; ``nx`` is the XY voxel count (~100/3.2).

    World origin is ``voxel_size/2`` per axis, as in encoding, so the height
    coord channel (``coord_features: z``) matches training.

    All voxels start active (mask=+1); the model prunes via the mask channel.

    Parameters
    ----------
    nx : int        Voxels in X and Y (footprint at level 0).
    nz : int        Voxels in Z (vertical extent at level 0).
    voxel_size : float  Metres per voxel (3.2 for ``16.pt`` / diffusion_0).
    batch : int     Number of independent samples.
    device : str

    Returns
    -------
    fvdb.GridBatch — a dense nx × nx × nz grid (all voxels active).
    """
    import fvdb

    vox_origin = torch.tensor([voxel_size / 2.0] * 3, dtype=torch.float32, device=device)

    ix = torch.arange(nx, device=device)
    iz = torch.arange(nz, device=device)
    gi, gj, gk = torch.meshgrid(ix, ix, iz, indexing="ij")
    ijk = torch.stack([gi.flatten(), gj.flatten(), gk.flatten()], dim=-1).to(torch.int32)
    ijk_jag = fvdb.JaggedTensor([ijk for _ in range(batch)])

    return fvdb.gridbatch_from_ijk(ijk_jag, voxel_sizes=voxel_size, origins=vox_origin)


def load_dales_diffusion(level, src):
    """Load a DALES checkpoint from *src* (prefers latest ``*_best.pt``)."""
    import utils.fvdb_diffusion as _fvdb_diffusion
    import utils.model as _model
    from utils.model import DiffusionUNet, build_diffusion_unet, _COORD_DIMS
    from utils.fvdb_diffusion import SparseDiffusion
    from utils.checkpoint_utils import resolve_latest_checkpoint, default_upsampler_dir
    # Checkpoints were pickled under old top-level names; remap so unpickling works.
    sys.modules.setdefault('fvdb_diffusion', _fvdb_diffusion)
    sys.modules.setdefault('model', _model)
    ckpt_path = resolve_latest_checkpoint(src, f"dales_{level}")
    print(f"Loading diffusion level {level}: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False)
    if not isinstance(ckpt, dict):
        if not isinstance(ckpt.model, DiffusionUNet):
            raise TypeError(
                f"Checkpoint uses {type(ckpt.model).__name__}; only DiffusionUNet is supported."
            )
        ckpt.eval()
        return ckpt
    # Best-epoch dict format: reconstruct SparseDiffusion from config + state dict.
    cfg = ckpt["config"]
    state_dict = ckpt["model_state_dict"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    coord_features = cfg.get("coord_features", "none")
    coord_h_ref = cfg.get("coord_h_ref", 30.0)
    coord_xy_ref = cfg.get("coord_xy_ref", 51.0)
    n_coord = _COORD_DIMS[coord_features]
    in_channels = state_dict["input_conv.0.weight"].shape[1] - n_coord
    if cfg.get("unet_depth", 2) <= 0:
        raise ValueError("Config must set unet_depth >= 1 (DiffusionUNet only).")
    model = build_diffusion_unet(cfg, in_channels, device)
    model.load_state_dict(state_dict)
    model_upsampler = None
    if level > 0:
        up_dir = default_upsampler_dir(src)
        up_ckpt = resolve_latest_checkpoint(up_dir, f"dales_{level}")
        print(f"Loading upsampler level {level}: {up_ckpt}")
        model_upsampler = torch.load(up_ckpt, weights_only=False)
        model_upsampler.eval()
    diffusion = SparseDiffusion(
        model,
        timesteps=cfg["diffusion_timesteps"],
        max_T=cfg.get("max_T", None) if level > 0 else None,
        n_classes=cfg["n_classes"],
        model_upsampler=model_upsampler,
    ).to(device)
    diffusion.eval()
    return diffusion


def compute_all_generations_dales(
    src,
    nx=LEVEL0_NX,
    voxel_size=LEVEL0_VOXEL_SIZE,
    max_level=4,
    eval_batch_size=5,
    features=13,
    ddim_steps=None,
    verbose=False,
    nz=LEVEL0_NZ,
    target_class=None,
    occ_threshold=0.0,
    upsampler_only_levels=None,
):
    """
    Unconditional DALES generation: noise → level 0 → … → level max_level.

    Level-0 grid defaults match ``diffusion_0.yaml`` (32×32×7 @ 3.2 m).
    Export point clouds with save_dales_pc.

    target_class: int | None — if set, clamp class channels to this label
                  throughout sampling (repaint-style conditioning).

    upsampler_only_levels: iterable[int] | None — for listed levels (>0), use
                  the upsampler prediction as the main generated output instead
                  of diffusion refinement.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    upsampler_only_levels = set(upsampler_only_levels or [])

    diffusion0 = load_dales_diffusion(0, src)
    diffusion0.eval()

    X0G = compute_canonical_base_grid(
        nx=nx, nz=nz, voxel_size=voxel_size, batch=eval_batch_size, device=device)
    t0 = time.time()

    noisy_init = grid_to_VDB(X0G, torch.randn, [features])
    if target_class is not None:
        generated_X = diffusion0.ddpm_sample_class_clamp(noisy_init, target_class)
    elif ddim_steps is None:
        generated_X = diffusion0.ddpm_sample(noisy_init)
    else:
        generated_X = diffusion0.ddim_sample(
            noisy_init, steps=diffusion0.max_T // ddim_steps)

    generated_X = DiffusionTensor.from_vdb(generated_X).remove_mask(threshold=occ_threshold)
    del diffusion0
    torch.cuda.empty_cache()
    generated_Xs = [generated_X]
    upsampler_Xs = []
    if verbose:
        print('LEVEL 0: {:.1f}s'.format(time.time() - t0))

    for i in range(1, max_level + 1):
        generated_X, upsampled_X = generate_level_dales(
            generated_X, i, src, ddim_steps, verbose,
            target_class=target_class,
            occ_threshold=occ_threshold,
        )
        if i in upsampler_only_levels:
            if verbose:
                print(f'LEVEL {i}: using upsampler-only output as main generation')
            generated_X = upsampled_X
        # move previous level off GPU before accumulating the new one
        generated_Xs[-1] = generated_Xs[-1].cpu()
        upsampler_Xs.append(upsampled_X)
        generated_Xs.append(generated_X)

    return generated_Xs, upsampler_Xs


def generate_level_dales(generated_X, level, src, ddim_steps=None, verbose=False,
                         target_class=None, occ_threshold=0.0):
    """Run one DALES upsampler level."""
    diffusion = load_dales_diffusion(level, src)
    diffusion.eval()
    t0 = time.time()
    new_XT, X_BLUR = generate_input(generated_X, diffusion)
    if target_class is not None:
        generated_X = diffusion.ddpm_sample_class_clamp(new_XT, target_class)
    elif ddim_steps is None:
        # pass X_BLUR so the reverse blends toward the upsampler estimate (SR3-style)
        generated_X = diffusion.ddpm_sample(new_XT, X_Blur=X_BLUR)
    else:
        generated_X = diffusion.ddim_sample(new_XT, steps=diffusion.max_T // ddim_steps)
    if verbose:
        print('LEVEL {}: {:.1f}s'.format(level, time.time() - t0))
    diffusion_result = DiffusionTensor.from_vdb(generated_X).remove_mask(threshold=occ_threshold)
    upsampler_result = DiffusionTensor.from_vdb(X_BLUR).remove_mask(threshold=occ_threshold)
    del diffusion
    torch.cuda.empty_cache()
    return diffusion_result, upsampler_result


def generate_input(generated_X, diffusion):
    with torch.no_grad():
        diffusion.model_upsampler.eval()
        input_X = diffusion.model_upsampler(
            generated_X, generated_X.trilinear_upsample()).detach()
        times = torch.ones((input_X.grid_count,), device=generated_X.device).float(
        )*(diffusion.max_T)/diffusion.timesteps
        times = times[input_X.data.jidx.long()]
        return diffusion.q_sample(input_X, times)[0], input_X


def export_to_laz(positions, intensity, class_idx, save_path):

    # Create header and point cloud
    header = laspy.LasHeader(point_format=3, version="1.4")
    las = laspy.LasData(header)

    # LAS files use int coordinates with an offset and scale.
    # For simplicity, use a default scale/offset good for meter coordinates.
    las.x = positions[:, 0]
    las.y = positions[:, 1]
    las.z = positions[:, 2]
    las.intensity = intensity
    las.classification = class_idx

    las.write(save_path)

def save_dales_pc(generated_X, out_dir, level=0, min_ind=0, stage="D", occ_threshold=0.5):
    """
    Export a DALES generation batch as LAZ, coloured by semantic class.

    Filenames: ``scene_{outputID}_{stage}_{level}.laz`` where stage is ``D``
    (diffusion) or ``U`` (upsampler).

    """
    for ind in range(generated_X.grid_count):
        g = DiffusionTensor(
            generated_X.grid[ind], generated_X.data[ind]
        ).get_global().remove_mask(threshold=occ_threshold)

        positions, features, _ = DiffusionTensor.get_feature_data(g.jdata)

        if len(positions) == 0:
            print(f'  sample {min_ind + ind} {stage}{level}: void — skipped')
            continue

        positions_np = positions.cpu().numpy()
        features_np    = features.cpu().numpy()     # (V, 9): [intensity, class_probs(8)]
        intensity    = features_np[:, 0]            # channel 3 of jdata (was [:,1] = P(ground))
        class_idx    = features_np[:, 1:].argmax(axis=-1)

        laz_path = os.path.join(out_dir, f'scene_{min_ind + ind}_{stage}_{level}.laz')
        export_to_laz(positions_np, intensity, class_idx, save_path=laz_path)


def _gt_occupancy_stats(gt_root: str, pyramid_res: int = 16, n_crops: int = 350,
                         empty_fill: str = 'zero') -> dict:
    """Scan real encoded crops and return occupancy statistics.

    Loads up to *n_crops* level-0 .pt files from <gt_root>/train/, densifies
    each with to_custom_dense(empty_fill), and measures the fraction of voxels
    that have mask > 0 (occupied).

    Returns a dict with keys: n_crops_found, occ_mean, occ_std, occ_min, occ_max.
    Returns None if no crops are found.
    """
    import glob, random as _random
    pattern = os.path.join(gt_root, 'train', '*', f'{pyramid_res}.pt')
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    _random.seed(42)
    sample = _random.sample(files, min(n_crops, len(files)))
    fracs = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for path in sample:
        obj = torch.load(path, weights_only=False)
        if not isinstance(obj, DiffusionTensor):
            obj = DiffusionTensor(obj.grid, obj.data)
        obj = DiffusionTensor(obj.grid.to(device), obj.data.to(device))
        dense = obj.to_custom_dense(empty_fill=empty_fill)
        mask = dense.jdata[:, -1]
        fracs.append(float((mask > 0).float().mean()))
    fracs_t = torch.tensor(fracs)
    return dict(
        n_crops_found=len(files),
        n_crops_sampled=len(fracs),
        occ_mean=float(fracs_t.mean()),
        occ_std=float(fracs_t.std()) if len(fracs) > 1 else 0.0,
        occ_min=float(fracs_t.min()),
        occ_max=float(fracs_t.max()),
    )


def diagnose_occupancy_dales(
    src, out_dir, nx=LEVEL0_NX, voxel_size=LEVEL0_VOXEL_SIZE, nz=LEVEL0_NZ,
    eval_batch_size=1, features=13,
    ddim_steps=None, thresholds=(-0.5, 0.0, 0.3, 0.6, 0.85),
    gt_root: str = 'data/dales',
    pyramid_res: int = LEVEL0_PYRAMID_RES,
):
    """Read-only occupancy diagnostic for the level-0 model (review #1).

    Samples the level-0 denoiser once, then — *before* pruning — reports the
    distribution of the predicted occupancy over the canonical grid and a
    threshold sweep.  The saved/decoded mask channel is 2σ(logit)-1 ∈ (-1,1),
    so occ_prob = σ(logit) = (mask+1)/2.

    Also compares against real GT crops so you can tell whether the model
    over- or under-generates occupied voxels.

    Reading the result:
      * occ_prob piled up near 1 (logit ≫ 0) everywhere → the field floods to
        occupied (the dense cube); a leak/bias problem, not calibration.
      * occ_prob spread with spatial structure but the kept-fraction at
        threshold 0 is too high → mostly a calibration problem; pick a higher
        -occ_threshold.
      * occ_prob ≈ 0.5 / featureless at every threshold → the model isn't
        generating structure (needs the mask-input-dropout retrain, #2).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(out_dir, exist_ok=True)
    diffusion0 = load_dales_diffusion(0, src)
    diffusion0.eval()

    X0G = compute_canonical_base_grid(
        nx=nx, nz=nz, voxel_size=voxel_size, batch=eval_batch_size, device=device)
    noisy_init = grid_to_VDB(X0G, torch.randn, [features])
    with torch.no_grad():
        if ddim_steps is None:
            gen = diffusion0.ddpm_sample(noisy_init)
        else:
            gen = diffusion0.ddim_sample(noisy_init, steps=diffusion0.max_T // ddim_steps)

    raw = DiffusionTensor.from_vdb(gen)
    mask_val = raw.jdata[:, -1].detach().float().cpu()
    occ_prob = ((mask_val + 1.0) / 2.0).clamp(0, 1)
    logit = torch.log(occ_prob.clamp(1e-6, 1 - 1e-6) / (1 - occ_prob).clamp(1e-6, 1 - 1e-6))

    print(f"\n=== occupancy diagnostic ({mask_val.numel()} voxels over "
          f"{eval_batch_size}×{nx}×{nx}×{nz}) ===")
    print(f"occ_prob: mean={occ_prob.mean():.3f}  median={occ_prob.median():.3f}  "
          f"min={occ_prob.min():.3f}  max={occ_prob.max():.3f}")
    print(f"logit:    mean={logit.mean():.2f}  std={logit.std():.2f}  "
          f"min={logit.min():.2f}  max={logit.max():.2f}")
    import numpy as np
    hist, _ = np.histogram(occ_prob.numpy(), bins=10, range=(0, 1))
    tot = max(int(hist.sum()), 1)
    print("occ_prob histogram (prob bin → %voxels):")
    for b in range(10):
        bar = '#' * int(50 * hist[b] / tot)
        print(f"  [{b/10:.1f},{(b+1)/10:.1f})  {100*hist[b]/tot:5.1f}%  {bar}")

    print("threshold sweep (mask-value space, occ kept = mask>t):")
    for t in thresholds:
        frac = float((mask_val > t).float().mean())
        pruned = raw.remove_mask(threshold=t)
        n_kept = int(pruned.jdata.shape[0])
        print(f"  t={t:+.2f}  kept={100*frac:5.1f}%  ({n_kept} voxels)  → saving")
        save_dales_pc(pruned, out_dir, level=f"thr{t:+.2f}")
    print(f"=== saved threshold-swept LAZs to {out_dir} ===\n")
    del diffusion0
    torch.cuda.empty_cache()

    # --- ground-truth occupancy comparison ---
    print("=== ground-truth occupancy comparison ===")
    gt = _gt_occupancy_stats(gt_root, pyramid_res=pyramid_res)
    if gt is None:
        print(f"  [WARNING] No real crops found at {gt_root}/train/*/{pyramid_res}.pt")
        print("  Pass -gt_root to point to your encoded dataset, or skip comparison.")
    else:
        gen_occ = float((mask_val > 0).float().mean())
        print(f"  Real crops ({gt['n_crops_sampled']}/{gt['n_crops_found']} sampled):")
        print(f"    occ fraction  mean={gt['occ_mean']:.3f}  std={gt['occ_std']:.3f}  "
              f"min={gt['occ_min']:.3f}  max={gt['occ_max']:.3f}")
        print(f"  Generated (this run): occ fraction = {gen_occ:.3f}")
        ratio = gen_occ / gt['occ_mean'] if gt['occ_mean'] > 0 else float('inf')
        if ratio > 1.15:
            print(f"  → model OVER-generates ({ratio:.2f}× real mean) — "
                  "reduce void_weight or use higher -occ_threshold")
        elif ratio < 0.85:
            print(f"  → model UNDER-generates ({ratio:.2f}× real mean) — "
                  "increase void_weight or lower -occ_threshold")
        else:
            print(f"  → occupancy matches real data well ({ratio:.2f}× real mean)")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DALES inference')
    parser.add_argument('-dataset', default='dales', type=str,
                        help="Dataset name (only 'dales' is supported)")
    parser.add_argument('-src', default='checkpoints/diffusion_models/', type=str,
                        help='Folder containing .pt checkpoint files')
    parser.add_argument('-out', default=None, type=str,
                        help='Output folder (default: output/dales or output/<name>)')
    parser.add_argument('-levels', default=4, type=int,
                        help='Number of upsampling levels')
    parser.add_argument('-ddim_steps', default=None, type=int,
                        help='DDIM steps (None → full DDPM)')
    parser.add_argument('-batch_size', default=4, type=int,
                        help='Crops generated per forward pass')
    parser.add_argument('-total_num', default=1, type=int,
                        help='Total number of crops to generate')
    parser.add_argument('-base_res', default=LEVEL0_NX, type=int,
                        help='XY voxels of the level-0 canonical grid (~100m / voxel_size)')
    parser.add_argument('-voxel_size', default=LEVEL0_VOXEL_SIZE, type=float,
                        help='Metres per voxel at level 0 (3.2 for diffusion_0 / 16.pt)')
    parser.add_argument('-nz', default=LEVEL0_NZ, type=int,
                        help='Z voxels at level 0 (~7 for diffusion_0 coarse layout)')
    parser.add_argument('-class', dest='target_class', default=None, type=int,
                        help='Clamp class channels to this label during sampling (0-7, repaint-style)')
    parser.add_argument('-occ_threshold', default=0.0, type=float,
                        help='remove_mask threshold in mask-value space (-1..1); raise to prune more')
    parser.add_argument('-diagnose', action='store_true',
                        help='Run the read-only occupancy diagnostic (histogram + threshold sweep) and exit')
    parser.add_argument('-gt_root', default='data/dales', type=str,
                        help='Path to encoded GT crops (default: data/dales); used by -diagnose'
                             ' to compare generated vs real occupancy')
    parser.add_argument('-upsampler_only_levels', default='', type=str,
                        help='Comma-separated levels to use upsampler-only output as main generation '
                            '(example: "1,2").')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    SRC = args.src.rstrip('/') + '/'

    if args.dataset != 'dales':
        raise ValueError(f"Unsupported dataset: {args.dataset!r}. Only 'dales' is supported.")

    OUT = args.out or 'output/dales'
    os.makedirs(OUT, exist_ok=True)

    if args.upsampler_only_levels.strip():
        try:
            upsampler_only_levels = {
                int(x.strip()) for x in args.upsampler_only_levels.split(',') if x.strip()
            }
        except ValueError as exc:
            raise ValueError('Invalid -upsampler_only_levels; use comma-separated integers, e.g. "1,2"') from exc
        if any(lvl <= 0 for lvl in upsampler_only_levels):
            raise ValueError('-upsampler_only_levels accepts only levels > 0')
    else:
        upsampler_only_levels = set()

    if args.diagnose:
        diagnose_occupancy_dales(
            src=SRC, out_dir=OUT, nx=args.base_res, voxel_size=args.voxel_size,
            nz=args.nz, eval_batch_size=1, ddim_steps=args.ddim_steps,
            gt_root=args.gt_root)
        sys.exit(0)

    n_batches = max(1, args.total_num // args.batch_size)
    print(f'Generating {n_batches * args.batch_size} DALES crops '
          f'({n_batches} batches of {args.batch_size}) → {OUT}')

    for batch_i in range(n_batches):
        min_ind = batch_i * args.batch_size
        t0 = time.time()
        with torch.no_grad():
            GX, UX = compute_all_generations_dales(
                src=SRC,
                nx=args.base_res,
                voxel_size=args.voxel_size,
                max_level=args.levels,
                eval_batch_size=args.batch_size,
                ddim_steps=args.ddim_steps,
                nz=args.nz,
                verbose=True,
                target_class=args.target_class,
                occ_threshold=args.occ_threshold,
                upsampler_only_levels=upsampler_only_levels,
            )
        print(f'  batch {batch_i} done in {time.time()-t0:.1f}s — saving LAZs …')
        for level_i, g in enumerate(GX):
            torch.save(g, os.path.join(OUT, f'gen_{min_ind}_{level_i}.pt'))
            save_dales_pc(g, OUT, level=level_i, min_ind=min_ind, stage="D",
                          occ_threshold=args.occ_threshold)
        for level_i, u in enumerate(UX, start=1):
            save_dales_pc(u, OUT, level=level_i, min_ind=min_ind, stage="U",
                          occ_threshold=args.occ_threshold)
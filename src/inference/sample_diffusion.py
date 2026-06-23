if True:
    import sys
    sys.path.append('./src')
import argparse
import glob
import os
from utils.fvdb_utils import *
import pymeshlab as ml
from utils.diffusion_tensor import DiffusionTensor
import numpy as np
import torch
import time


def compute_base_grid(model_name, eval_batch_size, base_res=16, src_path="./data/GT_sparse_tensors"):
    X0 = torch.load(
        '{}/{}/{}.pt'.format(src_path, model_name, base_res), weights_only=False)
    X0 = X0.to_custom_dense().to_batch(eval_batch_size)
    return X0.grid


def compute_canonical_base_grid(
    base_res: int = 16,
    extent_m: float = 50.0,
    batch: int = 1,
    device: str = "cuda",
    nz: int = None,
):
    """
    DALES unconditional sampling: a fully-occupied dense base grid.

    At level 0 for 100×100m crops (base_res=16, voxel_size=6.25m):
      - XY: 16 × 16 voxels (100m)
      - Z:  nz voxels (default: ceil(50m / voxel_size) = ceil(50/6.25) = 8)

    All voxels start active (mask=+1); the diffusion model prunes via the mask
    channel during sampling.

    Parameters
    ----------
    base_res : int
        Number of voxels in X and Y (default 16).
    extent_m : float
        Physical XY extent of one crop in metres (default 100.0 for DALES crops).
    batch : int
        Batch size (number of independent samples).
    device : str
    nz : int | None
        Number of voxels in Z. Defaults to base_res (isotropic grid).
        Pass 8 for 100×100m DALES crops (covers 0..50m above ground at 6.25m/voxel).

    Returns
    -------
    fvdb.GridBatch — a dense base_res × base_res × nz grid (all voxels active).
    """
    import fvdb
    import math

    voxel_size = extent_m / base_res
    if nz is None:
        # isotropic cube by default; caller can override for non-square shapes
        nz = base_res
    vox_origin = torch.tensor([voxel_size / 2.0] * 3, dtype=torch.float32, device=device)

    ix = torch.arange(base_res, device=device)
    iz = torch.arange(nz, device=device)
    gi, gj, gk = torch.meshgrid(ix, ix, iz, indexing="ij")
    ijk = torch.stack([gi.flatten(), gj.flatten(), gk.flatten()], dim=-1).to(torch.int32)
    ijk_jag = fvdb.JaggedTensor([ijk for _ in range(batch)])

    return fvdb.gridbatch_from_ijk(ijk_jag, voxel_sizes=voxel_size, origins=vox_origin)


def load_dales_diffusion(level, src):
    """Load a DALES checkpoint: {src}/dales_{level}_*.pt (picks most recent)."""
    import utils.fvdb_diffusion as _fvdb_diffusion
    import utils.model as _model
    from utils.model import DiffusionCNN
    from utils.fvdb_diffusion import SparseDiffusion
    # Checkpoints were pickled under old top-level names; remap so unpickling works.
    sys.modules.setdefault('fvdb_diffusion', _fvdb_diffusion)
    sys.modules.setdefault('model', _model)
    models = glob.glob('{}/dales_{}*.pt'.format(src, level))
    if not models:
        raise FileNotFoundError(f"No DALES diffusion checkpoint for level {level} in {src}")
    models.sort(reverse=True)
    ckpt = torch.load(models[0], weights_only=False)
    if not isinstance(ckpt, dict):
        ckpt.eval()
        return ckpt
    # Best-epoch dict format: reconstruct SparseDiffusion from config + state dict.
    cfg = ckpt["config"]
    state_dict = ckpt["model_state_dict"]
    # Infer in_channels from first conv weight shape: [C_out, in_channels+time_emb, ...]
    first_w = next(iter(state_dict.values()))
    in_channels = first_w.shape[1] - cfg["time_emb"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DiffusionCNN(
        channels=cfg["features"],
        layers=cfg["layers"],
        time_emb=cfg["time_emb"],
        one_layers=cfg["one_layers"],
        first_ks=cfg["first_ks"],
        in_channels=in_channels,
        out_channels=in_channels,
    ).to(device)
    model.load_state_dict(state_dict)
    model_upsampler = None
    if level > 0:
        up_ckpt = os.path.join(os.path.dirname(os.path.dirname(src)), 'upsamplers', f'dales_{level}.pt')
   
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
    base_res=16,
    extent_m=100.0,
    max_level=4,
    eval_batch_size=5,
    features=13,
    ddim_steps=None,
    verbose=False,
    nz=8,
):
    """
    Unconditional DALES generation: noise → level 0 → … → level max_level.

    Uses compute_canonical_base_grid instead of a stored crop grid.
    For 100×100m crops: extent_m=100.0, base_res=16, nz=8
      (voxel_size=6.25m, covers 0..50m at level 0).
    Export point clouds with save_generation_pc.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    diffusion0 = load_dales_diffusion(0, src)
    diffusion0.eval()

    X0G = compute_canonical_base_grid(base_res, extent_m, eval_batch_size, device, nz=nz)
    t0 = time.time()

    noisy_init = grid_to_VDB(X0G, torch.randn, [features])
    if ddim_steps is None:
        generated_X = diffusion0.ddpm_sample(noisy_init)
    else:
        generated_X = diffusion0.ddim_sample(
            noisy_init, steps=diffusion0.max_T // ddim_steps)

    generated_X = DiffusionTensor.from_vdb(generated_X).remove_mask()
    del diffusion0
    torch.cuda.empty_cache()
    generated_Xs = [generated_X]
    if verbose:
        print('LEVEL 0: {:.1f}s'.format(time.time() - t0))

    for i in range(1, max_level + 1):
        generated_X = generate_level_dales(generated_X, i, src, ddim_steps, verbose)
        # move previous level off GPU before accumulating the new one
        generated_Xs[-1] = generated_Xs[-1].cpu()
        generated_Xs.append(generated_X)

    return generated_Xs


def generate_level_dales(generated_X, level, src, ddim_steps=None, verbose=False):
    """Run one DALES upsampler level."""
    diffusion = load_dales_diffusion(level, src)
    diffusion.eval()
    t0 = time.time()
    new_XT, X_BLUR = generate_input(generated_X, diffusion)
    if ddim_steps is None:
        generated_X = diffusion.ddpm_sample(new_XT)
    else:
        generated_X = diffusion.ddim_sample(new_XT, steps=diffusion.max_T // ddim_steps)
    if verbose:
        print('LEVEL {}: {:.1f}s'.format(level, time.time() - t0))
    result = DiffusionTensor.from_vdb(generated_X).remove_mask()
    del diffusion
    torch.cuda.empty_cache()
    return result


def load_diffusion(example_mesh_name, level, src):
    models = glob.glob('{}/{}_{}*.pt'.format(src, example_mesh_name, level))
    models.sort()
    diffusion = torch.load(models[-1], weights_only=False)
    diffusion.eval()
    return diffusion


def generate_input(generated_X, diffusion):
    with torch.no_grad():
        diffusion.model_upsampler.eval()
        input_X = diffusion.model_upsampler(
            generated_X, generated_X.trilinear_upsample()).detach()
        times = torch.ones((input_X.grid_count,), device=generated_X.device).float(
        )*(diffusion.max_T)/diffusion.timesteps
        times = times[input_X.data.jidx.long()]
        return diffusion.q_sample(input_X, times)[0], input_X


def generate_level(generated_X, i, example_mesh_name, src, ddim_steps=None, verbose=False):
    diffusion = load_diffusion(example_mesh_name, i, src)
    diffusion.eval()
    t0 = time.time()
    new_XT, X_BLUR = generate_input(generated_X, diffusion)
    if ddim_steps is None:
        generated_X = diffusion.ddpm_sample(new_XT)
    else:
        generated_X = diffusion.ddim_sample(
            new_XT, steps=diffusion.max_T//ddim_steps)
    if verbose:
        print('LEVEL {}: {}'.format(i, time.time()-t0))

    return DiffusionTensor.from_vdb(generated_X).remove_mask()


def export_pc(vstars, normals, colors, save_pc_path=None, save_mesh_path=None, **kwargs):
    ms = ml.MeshSet()
    v_colors = np.column_stack((colors, np.ones_like(colors[:, :1])))
    nmesh = ml.Mesh(vertex_matrix=vstars,
                    v_normals_matrix=normals, v_color_matrix=v_colors)
    ms.add_mesh(nmesh)
    if not save_pc_path is None:
        ms.save_current_mesh(save_pc_path, save_vertex_normal=False)


_DALES_PALETTE = np.array([
    [0.50, 0.50, 0.50],  # Ground       — grey
    [0.13, 0.50, 0.13],  # Vegetation   — green
    [1.00, 0.20, 0.20],  # Cars         — red
    [1.00, 0.65, 0.00],  # Trucks       — orange
    [1.00, 1.00, 0.00],  # PowerLines   — yellow
    [0.50, 0.00, 0.50],  # Fences       — purple
    [0.00, 1.00, 1.00],  # Poles        — cyan
    [0.20, 0.20, 1.00],  # Buildings    — blue
], dtype=np.float32)


def save_dales_pc(generated_X, out_dir, level=0, min_ind=0):
    """
    Export a DALES generation batch as PLY, coloured by semantic class.

    Channels layout: [0:3] normals, [3:6] offset (→ position), [6] intensity,
                    [7] sem_class_norm (0–1), mask already removed.
    """
    for ind in range(generated_X.grid_count):
        g = DiffusionTensor(
            generated_X.grid[ind], generated_X.data[ind]
        ).get_global().remove_mask()

        positions, colors, _ = DiffusionTensor.get_feature_data(g.jdata)

        if len(positions) == 0:
            print(f'  sample {min_ind + ind} level {level}: void — skipped')
            continue

        positions_np = positions.cpu().numpy()
        colors_np    = colors.cpu().numpy()     # (V, 10): [intensity, class_probs(8)]
        class_idx    = colors_np[:, 1:].argmax(axis=-1).clip(0, 7)
        rgb          = _DALES_PALETTE[class_idx]

        normals_np   = np.zeros_like(positions_np)
        ply_path = os.path.join(out_dir, f'gen_{min_ind + ind}_{level}.ply')
        export_pc(positions_np, normals_np, rgb, save_pc_path=ply_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DALES / ShapeShifter inference')
    parser.add_argument('-dataset', default=None, type=str,
                        help="'dales' for DALES mode; omit for legacy single-shape mode")
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
    parser.add_argument('-total_num', default=20, type=int,
                        help='Total number of crops to generate')
    parser.add_argument('-base_res', default=16, type=int)
    parser.add_argument('-nz', default=8, type=int,
                        help='Z voxels at level 0 (DALES 100m crops: 8 → covers 0..50m at 6.25m/vox)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    SRC = args.src.rstrip('/') + '/'

    # ------------------------------------------------------------------
    # DALES unconditional generation
    # ------------------------------------------------------------------
    if args.dataset == 'dales':
        OUT = args.out or 'output/dales'
        os.makedirs(OUT, exist_ok=True)
        n_batches = max(1, args.total_num // args.batch_size)
        print(f'Generating {n_batches * args.batch_size} DALES crops '
              f'({n_batches} batches of {args.batch_size}) → {OUT}')

        for batch_i in range(n_batches):
            min_ind = batch_i * args.batch_size
            t0 = time.time()
            with torch.no_grad():
                GX = compute_all_generations_dales(
                    src=SRC,
                    base_res=args.base_res,
                    max_level=args.levels,
                    eval_batch_size=args.batch_size,
                    ddim_steps=args.ddim_steps,
                    nz=args.nz,
                    verbose=True,
                )
            print(f'  batch {batch_i} done in {time.time()-t0:.1f}s — saving PLYs …')
            for level_i, g in enumerate(GX):
                torch.save(g, os.path.join(OUT, f'gen_{min_ind}_{level_i}.pt'))
                save_dales_pc(g, OUT, level=level_i, min_ind=min_ind)
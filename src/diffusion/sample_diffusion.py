if True:
    import sys
    sys.path.append('./src/utils')
import argparse
import glob
import os
from fvdb_utils import *
import pymeshlab as ml
from model import UpSampler, DiffusionCNN
from fvdb_diffusion import SparseDiffusion
from diffusion_tensor import DiffusionTensor
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

    At level 0 for 50×50m crops (base_res=16, voxel_size=3.2m):
      - XY: 16 × 16 voxels (51.2m)
      - Z:  nz voxels (default: ceil(MAX_HEIGHT_M / voxel_size) = ceil(50/3.2) = 16)

    All voxels start active (mask=+1); the diffusion model prunes via the mask
    channel during sampling.

    Parameters
    ----------
    base_res : int
        Number of voxels in X and Y (default 16).
    extent_m : float
        Physical XY extent of one crop in metres (default 50.0 for DALES crops).
    batch : int
        Batch size (number of independent samples).
    device : str
    nz : int | None
        Number of voxels in Z. Defaults to ceil(50 / voxel_size) = base_res
        when voxel_size = extent_m / base_res, which gives an isotropic grid.
        Pass 12 for DALES crops (covers 0..38.4m above ground at 3.2m/voxel).

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
    models = glob.glob('{}/dales_{}*.pt'.format(src, level))
    if not models:
        raise FileNotFoundError(f"No DALES diffusion checkpoint for level {level} in {src}")
    models.sort()
    diffusion = torch.load(models[-1], weights_only=False)
    diffusion.eval()
    return diffusion


def compute_all_generations_dales(
    src,
    base_res=16,
    extent_m=50.0,
    max_level=4,
    eval_batch_size=5,
    features=10,
    ddim_steps=None,
    verbose=False,
    nz=12,
):
    """
    Unconditional DALES generation: noise → level 0 → … → level max_level.

    Uses compute_canonical_base_grid instead of a stored crop grid.
    For 50×50m crops: extent_m=50.0, base_res=16, nz=12 (covers 0..38.4m at 3.2m/voxel).
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
    generated_Xs = [generated_X]
    if verbose:
        print('LEVEL 0: {:.1f}s'.format(time.time() - t0))

    for i in range(1, max_level + 1):
        generated_X = generate_level_dales(generated_X, i, src, ddim_steps, verbose)
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
    return DiffusionTensor.from_vdb(generated_X).remove_mask()


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


def compute_all_generations(example_mesh_name, src, base_res, max_level=3, eval_batch_size=10, features=10, ddim_steps=None, X0G=None, verbose=False, src_path="./data/GT_sparse_tensors"):
    generated_Xs = []
    # blurs = []
    diffusion = load_diffusion(example_mesh_name, 0, src)
    diffusion.eval()
    if X0G is None:
        X0G = compute_base_grid(example_mesh_name, eval_batch_size, base_res, src_path)
    t0 = time.time()
    if ddim_steps is None:
        print('using ddpm')
        generated_X = diffusion.ddpm_sample(
            grid_to_VDB(X0G, torch.randn, [features]))
    else:
        print('using ddim')
        generated_X = diffusion.ddim_sample(grid_to_VDB(
            X0G, torch.randn, [features]), steps=diffusion.max_T//ddim_steps)
    generated_X = DiffusionTensor.from_vdb(generated_X).remove_mask()
    generated_Xs.append(generated_X)
    if verbose:
        print('LEVEL {}: {}'.format(0, time.time()-t0))
    for i in range(1, max_level+1):
        generated_X = generate_level(
            generated_X, i, example_mesh_name, src, ddim_steps, verbose)
        generated_Xs.append(generated_X)
        # blurs.append(X_BLUR)
    return generated_Xs


def export_pc(vstars, normals, colors, save_pc_path=None, save_mesh_path=None, **kwargs):
    ms = ml.MeshSet()
    v_colors = np.column_stack((colors, np.ones_like(colors[:, :1])))
    nmesh = ml.Mesh(vertex_matrix=vstars,
                    v_normals_matrix=normals, v_color_matrix=v_colors)
    ms.add_mesh(nmesh)
    if not save_pc_path is None:
        ms.save_current_mesh(save_pc_path, save_vertex_normal=True)
        if save_mesh_path is None:
            return
    print('computing poisson')
    ms.apply_filter('generate_surface_reconstruction_screened_poisson',
                    samplespernode=1., pointweight=10, depth=9)
    ms.apply_filter(
        'transfer_attributes_per_vertex', sourcemesh=0, targetmesh=1)  # Transfer PC colors to mesh
    if not save_mesh_path is None:
        ms.save_current_mesh(save_mesh_path)
        return
    return ms


def save_generation_pc(generated_X, src_path, level=0, inds=None, min_ind=0):
    if inds is None:
        inds = range(generated_X.grid_count)
    for ind in inds:
        global_X = DiffusionTensor(
            generated_X.grid[ind], generated_X.data[ind]).get_global().remove_mask()
        normalized_normals, global_offset, colors, mask = global_X.get_feature_data(
            global_X.jdata)
        if len(normalized_normals > 0):
            normalized_normals = normalized_normals.cpu().detach().numpy()
            normalized_normals /= np.maximum(
                np.sqrt((normalized_normals**2).sum(-1, keepdims=True)), 1e-10)
            vstars = global_offset.cpu().detach().numpy()
            colors = colors.cpu().detach().numpy()/2.
            colors = np.clip((colors+1)/2., 0, 1)
            save_pc_path = '{}/gen_{}_{}.ply'.format(
                src_path, min_ind+ind, level)
            export_pc(vstars, normalized_normals, colors,
                      save_pc_path)
        else:
            print('void shape!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Model')
    # Experiment
    parser.add_argument('-src',
                        default="checkpoints/diffusion_models/", type=str, help="input folder (contains .pt models)")
    parser.add_argument('-out', default=None, type=str,
                        help="output folder")
    parser.add_argument('-levels', default=4,
                        type=int, help="number of sudivisions")
    parser.add_argument('-ddim_steps', default=None, type=int,
                        help="ddim_steps, is None then ddpm is used")
    parser.add_argument('-batch_size', default=5, type=int,
                        help="batch size")
    parser.add_argument('-total_num', default=10, type=int,
                        help="total number of generated shapes")
    parser.add_argument('-base_res', default=16, type=int,
                        help="base resolution")
    args = parser.parse_args()

    SRC = args.src+'/'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.out is None:
        OUT = SRC.replace("checkpoints/diffusion_models", "output")
    else:
        OUT = args.out
    try:
        os.mkdir(OUT)
    except:
        print('dir exists!')    
    names = np.unique([e.split('_')[0] for e in os.listdir(SRC)])
    print('NAMES: ', ' '.join(names))
    for name in names:
        # for name in ["vase"]:
        save_path = '{}/{}'.format(OUT, name)
        for _ in range(args.total_num//args.batch_size):
            try:
                os.mkdir(save_path)
                min_ind = 0
            except:
                c_f = glob.glob("{}/gen_*.ply".format(save_path))
                min_ind = max([int(e[-7]) for e in c_f])+1
                print('dir exists!')
            with torch.no_grad():
                GX = compute_all_generations(
                    name, SRC, args.base_res, max_level=args.levels, eval_batch_size=args.batch_size, ddim_steps=args.ddim_steps)
                for i, g in enumerate(GX):
                    path = '{}/gen_{}_{}.pt'.format(save_path, min_ind, i)
                    torch.save(g, path)
                    save_generation_pc(g, save_path, i, min_ind=min_ind)


"""
crop_viz.py — save a snapshot of a real training crop at increasing noise
levels, straight from a live SparseDiffusion instance.
 
Unlike a standalone re-sampling script, this operates on the *actual* crop
and blur conditioning a training step just used (`x0c`, `bc` in
train_diffusion.py), via the model's own `q_sample`. It does not touch the
optimizer/backward path — it's a read-only `torch.no_grad()` peek.
 
Typical use, inside the training loop (see train_diffusion.py):
 
    from crop_viz import save_training_crop_snapshot
    ...
    save_training_crop_snapshot(
        diffusion, x0c, bc,
        out_path=f"output/noisy_crops/level_{args.level}/epoch_{epoch:04d}.png",
    )
"""
import os
 
import matplotlib.pyplot as plt
import torch
import fvdb.nn as fvnn
 
from diffusion_tensor import DiffusionTensor
 
 
def _points_and_mask(dt: DiffusionTensor):
    """World-space voxel positions + occupancy mask (the clearest channel for
    seeing blur/noise: clean +-1 split -> smooth gradient -> speckle)."""
    g = dt.get_global()
    offset, _features, mask = DiffusionTensor.get_feature_data(g.jdata)
    pts = offset.detach().cpu().numpy()
    m = mask.squeeze(-1).detach().cpu().numpy()
    return pts, m
 
 
def _scatter(ax, pts, color, title):
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=color, cmap="coolwarm", vmin=-1, vmax=1,
        s=4, linewidths=0,
    )
    ax.set_title(title, fontsize=10)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
 
 
def make_figure(panels):
    """panels: list of (title, DiffusionTensor). Returns a matplotlib Figure."""
    fig = plt.figure(figsize=(4.0 * len(panels), 4.0))
    for i, (title, dt) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
        pts, mask = _points_and_mask(dt)
        _scatter(ax, pts, mask, title)
    plt.tight_layout()
    return fig
 
 
@torch.no_grad()
def save_training_crop_snapshot(
    diffusion,
    x0_crop: fvnn.VDBTensor,
    blur_crop: fvnn.VDBTensor = None,
    out_path: str = "output/noisy_crops/crop.png",
    ts=(0.1, 0.5, 0.9),
    save_tensors: bool = False,
):
    """
    Render GT crop | blur conditioning | noisy(t) for each t in `ts`, using
    the live `diffusion.q_sample` — i.e. exactly the corruption the model is
    being trained to invert right now.
 
    Parameters
    ----------
    diffusion : SparseDiffusion
        The *unwrapped* module (not the DDP wrapper) — needs .q_sample,
        .max_T, .timesteps.
    x0_crop : fvnn.VDBTensor
        The clean crop actually used in this training step (`x0c`).
    blur_crop : fvnn.VDBTensor or None
        The matching blur-conditioning crop (`bc`), or None for level 0.
    out_path : str
        PNG destination; parent directories are created automatically.
    ts : tuple of float
        Normalised diffusion times in [0, 1] to render (0 = clean, 1 ~ pure noise).
    save_tensors : bool
        If True, also `torch.save` the raw noisy DiffusionTensor for each t
        next to the PNG (e.g. for re-loading and inspecting later).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
 
    panels = [("GT crop", DiffusionTensor(x0_crop.grid, x0_crop.data))]
    if blur_crop is not None:
        panels.append(("Blur conditioning", DiffusionTensor(blur_crop.grid, blur_crop.data)))
 
    for t in ts:
        times = torch.full(
            (x0_crop.data.jdata.shape[0],),
            t * diffusion.max_T / diffusion.timesteps,
            device=x0_crop.device,
        )
        noisy, _target = diffusion.q_sample(x0_crop, times, blur_crop)
        panels.append((f"noisy  t={t:.1f}", DiffusionTensor(noisy.grid, noisy.data)))
 
        if save_tensors:
            tensor_path = os.path.splitext(out_path)[0] + f"_t{t:.1f}.pt"
            torch.save(DiffusionTensor(noisy.grid.to("cpu"), noisy.data.to("cpu")), tensor_path)
 
    fig = make_figure(panels)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[crop_viz] saved -> {out_path}")
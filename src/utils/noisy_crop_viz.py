import os
import laspy
import numpy as np
import torch

from utils.diffusion_tensor import DiffusionTensor

def export_las(dt, outfile):
    """
    Convert a DiffusionTensor to LAS and write to disk.
    """

    # remove mask exactly like your conversion script
    g = dt.get_global().remove_mask()

    xyz, colors, mask = DiffusionTensor.get_feature_data(g.jdata)

    xyz = xyz.detach().cpu().numpy()
    colors = colors.detach().cpu()

    # semantic prediction
    sem = colors.argmax(dim=1).detach().cpu().numpy()

    las = laspy.create(point_format=laspy.PointFormat(3))

    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]

    # intensity (use first channel)
    intensity = colors[:,0].numpy()
    intensity = np.clip(intensity,0,65535).astype(np.uint16)
    las.intensity = intensity

    # classification
    las.classification = sem.astype(np.uint8)

    las.write(outfile)


@torch.no_grad()
def save_training_crop_snapshot(
    diffusion,
    out_dir,
    tag,
    crop_name
):

    os.makedirs(out_dir, exist_ok=True)

    clean = diffusion.debug["clean"]
    blur = diffusion.debug["blur"]
    noisy = diffusion.debug["noisy"]
    times = diffusion.debug["times"]

    crops = {
        "clean": clean,
        "blur": blur,
        "noisy": noisy
    }

    for name, crop in crops.items():
        if crop is None:
            print(f" [Warn] {name} crop is None, skipping snapshot")
            continue
        dt = DiffusionTensor(crop.grid, crop.data)

        for i in range(dt.grid.grid_count):

            grid = dt.grid[i]
            data = dt.data[i]

            single_dt = DiffusionTensor(grid, data)

            outfile = os.path.join(
                out_dir,
                f"{crop_name}_{tag}_{name}_{i}.las"
            )

            export_las(single_dt, outfile)

        print(dt.grid.grid_count)
        
        outfile = os.path.join(out_dir, f"{crop_name}_{tag}_{name}.las")
        export_las(dt, outfile)



 
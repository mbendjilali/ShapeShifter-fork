from pathlib import Path
import torch
import laspy
import numpy as np

from diffusion_tensor import DiffusionTensor

INPUT_DIR = Path("data/dales/train")
OUTPUT_DIR = Path("data/dales/crop_las")
OUTPUT_DIR.mkdir(exist_ok=True)


pt_files = sorted(INPUT_DIR.rglob("*.pt"))

print(f"Found {len(pt_files)} .pt files.")

for i, pt_file in enumerate(pt_files, start=1):
    print(f"[{i}/{len(pt_files)}] Processing {pt_file.relative_to(INPUT_DIR)}")
    # Load the point cloud from the .pt file
    data = torch.load(pt_file, weights_only=False)

    g = data.get_global().remove_mask()
    normals, xyz, colors, mask = DiffusionTensor.get_feature_data(g.jdata)

    xyz = xyz.cpu().numpy()

    sem = np.round(colors[:, 2].cpu().numpy() * 7).astype(np.int32) + 1  # remove the label normalization

    # Create a LAS file
    las = laspy.create(point_format=laspy.PointFormat(3))
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.classification = sem

    # Save the LAS file
    output_file = OUTPUT_DIR / pt_file.relative_to(INPUT_DIR).with_suffix(".las")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    las.write(output_file)

    print(f"    -> Saved {output_file.relative_to(OUTPUT_DIR)} "
          f"({len(xyz):,} points)")

print("Done!")
import json


import torch
from pathlib import Path

def compute_crop_weight(crop_path: str | Path, rare_classes: list[int]): 

    crop = torch.load(crop_path, weights_only=False)

    point_labels = (crop.jdata[:, 8] * 7 + 1).round() # because it has been normalized 
#    print(torch.unique(point_labels))
    rare_mask = torch.isin(point_labels, torch.tensor(rare_classes, device = point_labels.device)) # True if point label is in rare_classes

    rare_points = rare_mask.sum().item()
    total_points = len(point_labels)

    weight = rare_points

    return weight



def generate_crop_weights(manifest: dict, 
                          rare_classes: list[int], 
                          crop_root: str | Path,
                          output_path: str | Path):

    crop_root = Path(crop_root)

    tile_weights = {
        "train": {},
        "test": {}    }
    
    for split in ["train", "test"]:
        split_dir = crop_root / split

        for tile in manifest[split]:
            tile_id = tile["id"]
            # all the crops for this tile
            crops_paths = sorted(split_dir.glob(f"{tile_id}_x*/256.pt"))

            weights = []

            for crop_path in crops_paths:
                weight = compute_crop_weight(crop_path, rare_classes)
                weights.append(weight)


            # sum of all crop weights for this tile
            total_weights = sum(weights)

            tile_weights[split][tile_id] = [weight / total_weights for weight in weights]


            with open(output_path, "w") as f:
                json.dump(tile_weights, f, indent=4)

            print(f"Finished computing crop weights for tile {tile_id} in split {split}.")

    return tile_weights



def main():

    manifest_path = Path("configs/dales_manifest.json")

    # load config
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    crops_root= Path(manifest.get("gt_root"))
    rare_classes = manifest["rare_classes"]
    output_path = Path(manifest["gt_root"]) / "crop_weights.json"

    tile_weights = generate_crop_weights(manifest, rare_classes, crops_root, output_path)

if __name__ == "__main__":
    main()
    

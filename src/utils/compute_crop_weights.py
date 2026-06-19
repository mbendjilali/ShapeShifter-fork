import json

import torch
from pathlib import Path


def _decode_labels(crop) -> torch.Tensor:
    """Decode class_probs channel back to integer class ids [1..8]."""
    return crop.jdata[:, 5:13].argmax(dim=1) + 1


def compute_crop_weight(crop_path: str | Path, rare_classes: list[int]):
    crop = torch.load(crop_path, weights_only=False)
    point_labels = _decode_labels(crop)
    rare_mask = torch.isin(point_labels, torch.tensor(rare_classes, device=point_labels.device))
    return rare_mask.sum().item()


def compute_vegetation_weight(crop_path: str | Path, vegetation_class: int) -> float:
    """
    Return 1 - vegetation_fraction so that crops with less vegetation
    get a higher sampling weight.  Clamped to [0.01, 1.0] to avoid zeros.
    """
    crop = torch.load(crop_path, weights_only=False)
    point_labels = _decode_labels(crop)
    veg_fraction = (point_labels == vegetation_class).float().mean().item()
    return max(0.01, 1.0 - veg_fraction)


def generate_crop_weights(manifest: dict,
                          rare_classes: list[int],
                          crop_root: str | Path,
                          output_path: str | Path):
    crop_root = Path(crop_root)
    tile_weights = {"train": {}, "test": {}}

    for split in ["train", "test"]:
        split_dir = crop_root / split
        for tile in manifest[split]:
            tile_id = tile["id"]
            crops_paths = sorted(split_dir.glob(f"{tile_id}_x*/256.pt"))
            weights = [compute_crop_weight(p, rare_classes) for p in crops_paths]
            total = sum(weights)
            tile_weights[split][tile_id] = [w / total for w in weights]

            with open(output_path, "w") as f:
                json.dump(tile_weights, f, indent=4)
            print(f"Finished rare-class weights for tile {tile_id} ({split}).")

    return tile_weights


def generate_vegetation_weights(manifest: dict,
                                vegetation_class: int,
                                crop_root: str | Path,
                                output_path: str | Path):
    """
    For each crop compute 1 - veg_fraction and save to output_path.
    Weights are NOT normalised — random.choices handles that at sampling time.
    """
    crop_root = Path(crop_root)
    tile_weights = {"train": {}, "test": {}}

    for split in ["train", "test"]:
        split_dir = crop_root / split
        for tile in manifest[split]:
            tile_id = tile["id"]
            crops_paths = sorted(split_dir.glob(f"{tile_id}_x*/256.pt"))
            weights = [compute_vegetation_weight(p, vegetation_class) for p in crops_paths]
            tile_weights[split][tile_id] = weights

            with open(output_path, "w") as f:
                json.dump(tile_weights, f, indent=4)
            print(f"Finished vegetation weights for tile {tile_id} ({split}).")

    return tile_weights


def main():
    manifest_path = Path("configs/dales_manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    crops_root = Path(manifest["gt_root"])
    rare_classes = manifest["rare_classes"]

    rare_output = crops_root / "crop_weights.json"
    generate_crop_weights(manifest, rare_classes, crops_root, rare_output)

    vegetation_class = manifest.get("vegetation_class", 2)
    veg_output = crops_root / "crop_vegetation_weights.json"
    generate_vegetation_weights(manifest, vegetation_class, crops_root, veg_output)


if __name__ == "__main__":
    main()


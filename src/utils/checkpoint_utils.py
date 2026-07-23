"""Resolve the latest *_best.pt checkpoint, with fallback to the latest regular .pt."""

from __future__ import annotations

import glob
import os


def resolve_latest_checkpoint(directory: str, prefix: str) -> str:
    """Return the newest checkpoint matching ``{prefix}*.pt`` in *directory*.

    Prefers files ending in ``_best.pt``.  Falls back to other matches when no
    best checkpoint exists.  Tie-breaks by modification time (newest wins).
    """
    directory = directory.rstrip(os.sep)
    pattern = os.path.join(directory, f"{prefix}*.pt")
    all_paths = glob.glob(pattern)
    if not all_paths:
        raise FileNotFoundError(
            f"No checkpoint matching '{prefix}*.pt' in {directory!r}"
        )

    best_paths = [p for p in all_paths if p.endswith("_best.pt")]
    if best_paths:
        return max(best_paths, key=os.path.getmtime)

    regular = [p for p in all_paths if not p.endswith("_best.pt")]
    return max(regular, key=os.path.getmtime)


def default_upsampler_dir(diffusion_src: str) -> str:
    """``checkpoints/upsamplers`` next to a ``checkpoints/diffusion_models`` *src*."""
    return os.path.join(os.path.dirname(diffusion_src.rstrip(os.sep)), "upsamplers")

if True:
    import sys
    sys.path.append('./src/utils')
from diffusion_tensor import DiffusionTensor
from fvdb_utils import *
from fvdb_diffusion import SparseDiffusion
from model import DiffusionCNN, count_parameters
from datetime import datetime
from pathlib import Path
import fvdb.nn as fvnn
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import yaml
import os


# ---------------------------------------------------------------------------
# Original single-shape helpers (kept for backwards compat)
# ---------------------------------------------------------------------------

def clip_data(X0, X0_BLUR, size):
    """Legacy: crops from global voxel list. Used only for single-shape mode."""
    ind = torch.randint(0, len(X0.grid.ijk.jdata), (X0.grid_count,))
    centers = X0.grid.ijk.jdata[ind]
    new_ijk_min = centers - size
    new_ijk_max = centers + size
    cf, cg = X0.grid.clip(X0.data, new_ijk_min, new_ijk_max)
    new_X0 = fvnn.VDBTensor(cg, cf)
    cf, cg = X0_BLUR.grid.clip(X0_BLUR.data, new_ijk_min, new_ijk_max)
    new_X0_BLUR = fvnn.VDBTensor(cg, cf)
    return new_X0, new_X0_BLUR


def get_gt_data(cfg, level, model_name):
    if level == 0:
        res_1 = cfg["base_resolution"]
        X0 = torch.load(
            '{}/{}/{}.pt'.format(cfg["src_path"], model_name, res_1), weights_only=False)
        return X0.to_custom_dense().to_batch(cfg["batch_size"])
    else:
        res_1 = cfg["base_resolution"]*2**(level-1)
        res_2 = cfg["upsample_fac"]*res_1
        X = torch.load(
            '{}/{}/{}.pt'.format(cfg["src_path"], model_name, res_1), weights_only=False)
        X0 = torch.load(
            '{}/{}/{}.pt'.format(cfg["src_path"], model_name, res_2), weights_only=False)
        X_UP = X.trilinear_upsample(cfg["upsample_fac"])
        X0 = DiffusionTensor.fill_upsampled_with_gt(X_UP, X0)
        X = X.to_batch(cfg["batch_size"])
        X0 = X0.to_batch(cfg["batch_size"])
        X_UP = X_UP.to_batch(cfg["batch_size"])
        X_UP.grid = X0.grid
        return X, X_UP, X0


# ---------------------------------------------------------------------------
# Prefetch helper
# ---------------------------------------------------------------------------

import queue
import threading

class _PrefetchLoader:
    """
    Background thread that keeps `capacity` batches ready in a queue.
    Eliminates disk I/O stall between GPU steps.

    Usage:
        loader = _PrefetchLoader(dataset, split, level, batch_size, device)
        loader.start()
        for i in range(epochs):
            batch = loader.next()  # blocks only if queue is empty (rare)
        loader.stop()
    """
    def __init__(self, dataset, split, level, batch_size, device, capacity=2):
        self._dataset    = dataset
        self._split      = split
        self._level      = level
        self._batch_size = batch_size
        self._device     = device
        self._q          = queue.Queue(maxsize=capacity)
        self._stop_evt   = threading.Event()
        self._thread     = threading.Thread(target=self._worker, daemon=True)

    def _worker(self):
        while not self._stop_evt.is_set():
            try:
                batch = self._dataset.sample_batch(
                    self._split, self._level, self._batch_size, self._device)
                self._q.put(batch)
            except Exception as e:
                self._q.put(e)

    def start(self):
        self._thread.start()
        return self

    def next(self):
        item = self._q.get()
        if isinstance(item, Exception):
            raise item
        return item

    def stop(self):
        self._stop_evt.set()
        # drain so the worker thread can exit
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass


# ---------------------------------------------------------------------------
# DALES dataset training
# ---------------------------------------------------------------------------

def _train_dales(args, cfg, device='cuda'):
    """
    Multi-tile DALES training loop.

    Checkpoints: checkpoints/diffusion_models/dales_{level}_{time}.pt
    """
    import sys
    sys.path.insert(0, './src')
    from dataset.dales_dataset import DALESDataset, clip_data_per_element

    manifest = cfg.get("manifest_path", "data/dales_manifest.json")

    dataset = DALESDataset(manifest, split="train",
                           upsample_fac=cfg["upsample_fac"],
                           base_resolution=cfg["base_resolution"])
    val_dataset = None
    try:
        val_dataset = DALESDataset.test_set(manifest, split="test",
                                             upsample_fac=cfg["upsample_fac"],
                                             base_resolution=cfg["base_resolution"])
    except Exception:
        pass  # no test tiles encoded yet

    # Determine channel count from first available crop
    first_crop = dataset.crops[0]
    res_sample = cfg["base_resolution"]
    sample_dt = torch.load(
        dataset.gt_root / "train" / first_crop / f"{res_sample}.pt", weights_only=False
    ).to(device)
    n_channels = sample_dt.jdata.shape[-1]
    del sample_dt

    model = DiffusionCNN(
        channels=cfg["features"],
        layers=cfg["layers"],
        time_emb=cfg["time_emb"],
        one_layers=cfg["one_layers"],
        first_ks=cfg["first_ks"],
        in_channels=n_channels,
        out_channels=n_channels,
    ).to(device)
    count_parameters(model)

    model_upsampler = None
    if args.level > 0:
        up_ckpt = 'checkpoints/upsamplers/dales_{}.pt'.format(args.level)
        model_upsampler = torch.load(up_ckpt, weights_only=False)
        model_upsampler.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    diffusion = SparseDiffusion(
        model,
        timesteps=cfg["diffusion_timesteps"],
        max_T=cfg.get("max_T", None) if args.level > 0 else None,
        loss=nn.functional.mse_loss,
        model_upsampler=model_upsampler,
    ).cuda()

    L, VAL_L = [], []
    LOSS_EMA = None
    current_time = datetime.today().strftime('%d-%m-%H:%M')

    loader = _PrefetchLoader(
        dataset, "train", args.level, cfg["batch_size"], device, capacity=2
    ).start()

    try:
        for i in tqdm(range(cfg["epochs"])):
            optimizer.zero_grad()
            batch = loader.next()

            if args.level == 0:
                X0   = batch
                loss = diffusion(X0)
            else:
                X, X_UP, X0 = batch
                with torch.no_grad():
                    X0_BLUR = model_upsampler(X, X_UP).detach()
                X0_BLUR.grid = X0.grid
                x0c, bc = clip_data_per_element(X0, X0_BLUR, cfg["clip_size"])
                loss = diffusion(x0c, bc)

            torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), 1.)
            loss.backward()
            optimizer.step()

            # Sync CPU↔GPU only every 20 steps — avoids stalling the GPU pipeline
            if i % 20 == 0:
                loss_val = loss.item()
                LOSS_EMA = loss_val if LOSS_EMA is None else 0.99 * LOSS_EMA + 0.01 * loss_val
                L.append(LOSS_EMA)

            val_every = cfg.get("val_every", cfg["save_every"])
            if val_dataset is not None and i % val_every == 0:
                val_loss = val_dataset.compute_val_loss(
                    diffusion, "test", args.level, n_crops=4,
                    clip_size=cfg["clip_size"], device=device,
                )
                if val_loss is not None:
                    VAL_L.append((i, val_loss))

            if i % cfg["save_every"] == 0 or i == cfg["epochs"] - 1:
                plt.clf()
                plt.plot(L, label='train_ema')
                if VAL_L:
                    plt.plot([v[0] for v in VAL_L], [v[1] for v in VAL_L],
                             label='val', linestyle='--')
                plt.yscale('log')
                plt.legend()
                plt.savefig('checkpoints/diffusion_models/dales_{}_{}.png'.format(
                    args.level, current_time))
    finally:
        loader.stop()

    ckpt = 'checkpoints/diffusion_models/dales_{}_{}.pt'.format(args.level, current_time)
    torch.save(diffusion, ckpt)
    print(ckpt)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser(description='Diffusion training')
    parser.add_argument('-model_name', type=str, default=None, help="Single-shape name (legacy mode)")
    parser.add_argument('-level', type=int, help="Diffusion level")
    parser.add_argument('-config', type=str, help="Config path")
    parser.add_argument('-dataset', type=str, default=None,
                        help="Dataset name (e.g. 'dales') — activates multi-tile mode")
    args = parser.parse_args()

    try:
        os.mkdir('checkpoints')
    except Exception:
        pass
    try:
        os.mkdir('checkpoints/diffusion_models')
    except Exception:
        pass

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # Determine mode: DALES multi-tile or legacy single-shape
    dataset_name = args.dataset or cfg.get("dataset", None)

    if dataset_name == "dales":
        _train_dales(args, cfg, device)
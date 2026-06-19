if True:
    import sys
    sys.path.append('./src')
import torch
from tqdm import tqdm
from utils.fvdb_utils import *
import matplotlib.pyplot as plt
from utils.model import UpSampler, count_parameters
import yaml
import argparse
from train_diffusion import get_gt_data
import os


def _train_dales_upsampler(args, cfg, device='cuda'):
    """
    Multi-tile upsampler training for DALES.
    Checkpoints: checkpoints/upsamplers/dales_{level}.pt
    """
    from dataset.dales_dataset import DALESDataset
    import math

    manifest = cfg.get("manifest_path", "data/dales_manifest.json")
    dataset = DALESDataset(manifest, split="train",
                           upsample_fac=cfg["upsample_fac"],
                           base_resolution=cfg["base_resolution"])

    # Sample one batch to determine channel count
    X, X_UP, Y = dataset.sample_batch("train", args.level, 1, device)

    model_upsampler = UpSampler(
        X.jdata.shape[-1], cfg["features"], X.jdata.shape[-1],
        cfg["layers"], cfg["upsample_fac"], cfg["dropout"]
    ).to(device)
    optimizer = torch.optim.AdamW(model_upsampler.parameters(), lr=cfg["lr"])

    n_epochs = cfg["epochs"]
    batch_size = cfg.get("batch_size", 1)
    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    print(f"  {len(dataset)} crops — {steps_per_epoch} steps/epoch — {n_epochs} epochs")

    L = []
    LOSS_EMA = None
    count_parameters(model_upsampler)

    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        epoch_loss_sum = 0.0
        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            X, X_UP, Y = dataset.sample_batch("train", args.level, batch_size, device)
            loss = (((model_upsampler(X, X_UP) - Y).jdata) ** 2).mean()
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item()
        loss_val = epoch_loss_sum / steps_per_epoch
        LOSS_EMA = loss_val if LOSS_EMA is None else 0.99 * LOSS_EMA + 0.01 * loss_val
        L.append(LOSS_EMA)

    plt.plot(L, label='dales_{}'.format(args.level))
    plt.yscale('log')
    plt.legend()
    plt.savefig('checkpoints/upsamplers/dales_{}.png'.format(args.level))
    model_upsampler.eval()
    ckpt = 'checkpoints/upsamplers/dales_{}.pt'.format(args.level)
    torch.save(model_upsampler, ckpt)
    print(ckpt)


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser(description='Upsamplers training')
    parser.add_argument('-model_name', default="house", type=str, help="Mesh name (legacy mode)")
    parser.add_argument('-level', default=1, type=int, help="Upsampler level")
    parser.add_argument('-config', type=str, help="Config path")
    parser.add_argument('-dataset', type=str, default=None,
                        help="Dataset name (e.g. 'dales') — activates multi-tile mode")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    if not os.path.exists('checkpoints/upsamplers'):
        os.makedirs('checkpoints/upsamplers')

    dataset_name = args.dataset or cfg.get("dataset", None)

    if dataset_name == "dales":
        _train_dales_upsampler(args, cfg, device)

    else:
        # Legacy single-shape mode
        res_1 = cfg["base_resolution"] * 2 ** (args.level - 1)
        res_2 = cfg["upsample_fac"] * res_1

        X, X_UP, Y = get_gt_data(cfg, args.level, args.model_name)

        model_upsampler = UpSampler(
            X.jdata.shape[-1], cfg["features"], X.jdata.shape[-1],
            cfg["layers"], cfg["upsample_fac"], cfg["dropout"]
        ).to(device)
        optimizer = torch.optim.AdamW(model_upsampler.parameters(), lr=cfg["lr"])
        L = []
        mt.count_parameters(model_upsampler)

        def train_epoch():
            optimizer.zero_grad()
            loss = (((model_upsampler(X, X_UP) - Y).jdata) ** 2).mean()
            loss.backward()
            optimizer.step()
            L.append(loss.item())

        model_upsampler.train()
        for _ in tqdm(range(cfg["epochs"])):
            train_epoch()

        plt.plot(L, label=args.model_name)
        plt.yscale('log')
        plt.legend()
        plt.savefig('checkpoints/upsamplers/{}_{}.png'.format(args.model_name, args.level))
        model_upsampler.eval()
        torch.save(model_upsampler,
                   'checkpoints/upsamplers/{}_{}.pt'.format(args.model_name, args.level))

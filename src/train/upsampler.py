if True:
    import sys
    sys.path.append('./src')
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.model import UpSampler, count_parameters
import yaml
import argparse
import os


def _train_dales_upsampler(args, cfg, device='cuda'):
    """
    Multi-tile upsampler training for DALES.
    Checkpoints: checkpoints/upsamplers/dales_{level}.pt
    """
    from dataset.dales import DALESDataset
    import math

    manifest = cfg.get("manifest_path", "configs/dataset/dales.yaml")
    dataset = DALESDataset(manifest, split="train",
                           upsample_fac=cfg["upsample_fac"],
                           base_resolution=cfg["base_resolution"],
                           level=args.level)

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
    plt.legend()
    plt.savefig('checkpoints/upsamplers/dales_{}.png'.format(args.level))
    model_upsampler.eval()
    ckpt = 'checkpoints/upsamplers/dales_{}.pt'.format(args.level)
    torch.save(model_upsampler, ckpt)
    print(ckpt)


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser(description='Upsamplers training')
    parser.add_argument('-level', default=1, type=int, help="Upsampler level")
    parser.add_argument('-config', type=str, help="Config path")
    parser.add_argument('-dataset', type=str, default='dales',
                        help="Dataset name (only 'dales' is supported)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    if not os.path.exists('checkpoints/upsamplers'):
        os.makedirs('checkpoints/upsamplers')

    dataset_name = args.dataset or cfg.get("dataset", None)
    if dataset_name != "dales":
        raise ValueError(f"Unsupported dataset: {dataset_name!r}. Only 'dales' is supported.")

    _train_dales_upsampler(args, cfg, device)

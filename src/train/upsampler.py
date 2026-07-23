if True:
    import sys
    sys.path.append('./src')
    sys.path.append('./src/utils')
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils.model import UpSampler, count_parameters
from prefetch_loader import PrefetchLoader
import yaml
import argparse
import os
import math


def _train_dales_upsampler(args, cfg, device='cuda'):
    """
    Multi-tile upsampler training for DALES.
    Checkpoints: checkpoints/upsamplers/dales_{level}_{timestamp}_best.pt
    TensorBoard: runs/upsampler_level_{level}_{timestamp}/
    """
    from dataset.dales import DALESDataset

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
    scaler = torch.amp.GradScaler('cuda')

    n_epochs = cfg["epochs"]
    micro_batch = cfg.get("batch_size", 1)
    accumulate_steps = cfg.get("accumulate_steps", 1)
    effective_batch = micro_batch * accumulate_steps
    steps_per_epoch = math.ceil(len(dataset) / effective_batch)
    print(f"  {len(dataset)} crops — micro_batch={micro_batch} "
          f"× accum={accumulate_steps} = {effective_batch} effective — "
          f"{steps_per_epoch} steps/epoch — {n_epochs} epochs")

    LOSS_EMA = None
    best_loss = float('inf')
    count_parameters(model_upsampler)

    current_time = datetime.today().strftime('%d-%m-%H:%M')
    writer = SummaryWriter(
        log_dir=f"runs/upsampler_level_{args.level}_{current_time}"
    )

    loader = PrefetchLoader(
        dataset, "train", args.level, micro_batch, device, capacity=2
    ).start()

    try:
        for epoch in tqdm(range(n_epochs), desc="Epochs"):
            epoch_loss_sum = 0.0
            n_steps = 0
            for _ in range(steps_per_epoch):
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                any_backward = False
                for _ in range(accumulate_steps):
                    X, X_UP, Y = loader.next()
                    with torch.amp.autocast('cuda'):
                        loss = (
                            ((model_upsampler(X, X_UP) - Y).jdata) ** 2
                        ).mean() / accumulate_steps
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    scaler.scale(loss).backward()
                    any_backward = True
                    step_loss += loss.item() * accumulate_steps
                if any_backward:
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss_sum += step_loss / accumulate_steps
                    n_steps += 1

            loss_val = epoch_loss_sum / max(n_steps, 1)
            LOSS_EMA = loss_val if LOSS_EMA is None else 0.99 * LOSS_EMA + 0.01 * loss_val
            writer.add_scalars(
                "Loss/train",
                {"MSE": loss_val, "MSE_EMA": LOSS_EMA},
                epoch,
            )
            if loss_val < best_loss:
                best_loss = loss_val
                best_ckpt = (
                    f"checkpoints/upsamplers/dales_{args.level}_{current_time}_best.pt"
                )
                model_upsampler.eval()
                torch.save(model_upsampler, best_ckpt)
                model_upsampler.train()
                tqdm.write(
                    f"New best upsampler at epoch {epoch}: {best_ckpt} — "
                    f"loss {best_loss:.6f}"
                )
    finally:
        loader.stop()

    writer.close()
    model_upsampler.eval()
    ckpt = f'checkpoints/upsamplers/dales_{args.level}_{current_time}.pt'
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

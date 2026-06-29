import os
import sys

# ── Per-process CUDA isolation ────────────────────────────────────────────────
# Must run before ANY import that calls torch.cuda (model.py does so at module
# level via get_device_capability). Without this, every rank initialises its
# CUDA context on cuda:0; rank 1 then conflicts with rank 0 during fVDB GPU
# operations.  Restricting each process to a single visible GPU makes "cuda"
# unambiguous and prevents cross-process memory corruption (CUDA error 700).
_local_rank = int(os.environ.get("LOCAL_RANK", 0))
_world_size = int(os.environ.get("WORLD_SIZE", 1))
if _world_size > 1:
    _vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if _vis:
        os.environ["CUDA_VISIBLE_DEVICES"] = _vis.split(",")[_local_rank].strip()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(_local_rank)
# ─────────────────────────────────────────────────────────────────────────────

if True:
    sys.path.append('./src/utils')
from diffusion_tensor import DiffusionTensor
from fvdb_utils import *
from fvdb_diffusion import SparseDiffusion
from model import DiffusionCNN, DiffusionUNet, count_parameters
from datetime import datetime
import argparse
import contextlib
import fvdb.nn as fvnn
import math
import torch
import torch.distributed as dist
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange


# ---------------------------------------------------------------------------
# Weight EMA (Karras et al. 2022)
# ---------------------------------------------------------------------------

class WeightEMA:
    """
    Exponential moving average of model parameters.

    Sampling from the raw last-iterate is noisy; EMA weights give markedly
    better samples.  `store`/`copy_to`/`restore` let us run validation and
    write checkpoints from the EMA weights, then restore the live weights for
    continued training.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def store(self, model):
        self._backup = {n: p.detach().clone()
                        for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def copy_to(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        if self._backup is None:
            return
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(self._backup[n])
        self._backup = None


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

def _train_dales(args, cfg, device='cuda', rank=0, world_size=1):
    """
    Multi-tile DALES training loop with DDP, AMP, and gradient accumulation.

    Launch with torchrun for multi-GPU:
        CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \\
            src/train/train_diffusion.py -level 0 -config configs/train_dales_diffusion_0.yaml

    Checkpoints: checkpoints/diffusion_models/dales_{level}_{time}.pt  (rank 0 only)
    """
    import sys
    sys.path.insert(0, './src')
    from dataset.dales_dataset import DALESDataset, clip_data_per_element

    is_main = (rank == 0)
    manifest = cfg.get("manifest_path", "data/dales_manifest.yaml")
    clip_size = cfg.get("clip_size") if args.level == 0 else None

    # ── Accumulation / effective batch ──────────────────────────────────────
    micro_batch     = cfg["batch_size"]               # per-GPU 
    accumulate_steps = cfg.get("accumulate_steps", 1)
    effective_batch  = micro_batch * accumulate_steps  # per-GPU

    if world_size > 1 and rank != 0:
        dist.barrier()

    random_crop = cfg.get("random_crop", False) and args.level == 0
    empty_fill = cfg.get("empty_fill", "blur")
    dataset = DALESDataset(manifest,
                           split="train",
                           upsample_fac=cfg["upsample_fac"],
                           base_resolution=cfg["base_resolution"],
                           level=args.level,
                           clip_size=clip_size,
                           random_crop=random_crop,
                           empty_fill=empty_fill)

    val_dataset = DALESDataset.test_set(manifest,
                                        upsample_fac=cfg["upsample_fac"],
                                        base_resolution=cfg["base_resolution"],
                                        level=args.level,
                                        clip_size=clip_size,
                                        random_crop=random_crop,
                                        empty_fill=empty_fill)

    if world_size > 1 and rank == 0:
        dist.barrier()

    # Determine channel count from first available crop
    first_crop = dataset.crops[0]
    res_sample = cfg["base_resolution"]
    sample_dt = torch.load(
        dataset.gt_root / "train" / first_crop / f"{res_sample}.pt", weights_only=False
    ).to(device)
    n_channels = sample_dt.jdata.shape[-1]
    del sample_dt

    unet_depth = cfg.get("unet_depth", 0)
    if unet_depth > 0:
        model = DiffusionUNet(
            channels=cfg["features"],
            unet_depth=unet_depth,
            time_emb=cfg["time_emb"],
            one_layers=cfg["one_layers"],
            first_ks=cfg["first_ks"],
            in_channels=n_channels,
            out_channels=n_channels,
            dropout=cfg.get("dropout", 0.01),
            coord_features=cfg.get("coord_features", "none"),
            coord_h_ref=cfg.get("coord_h_ref", 30.0),
            coord_xy_ref=cfg.get("coord_xy_ref", 51.0),
        ).to(device)
    else:
        model = DiffusionCNN(
            channels=cfg["features"],
            layers=cfg["layers"],
            time_emb=cfg["time_emb"],
            one_layers=cfg["one_layers"],
            first_ks=cfg["first_ks"],
            in_channels=n_channels,
            out_channels=n_channels,
            coord_features=cfg.get("coord_features", "none"),
            coord_h_ref=cfg.get("coord_h_ref", 30.0),
            coord_xy_ref=cfg.get("coord_xy_ref", 51.0),
        ).to(device)
    if is_main:
        count_parameters(model)

    model_upsampler = None
    if args.level > 0:
        up_ckpt = 'checkpoints/upsamplers/dales_{}.pt'.format(args.level)
        model_upsampler = torch.load(up_ckpt, weights_only=False).to(device)
        model_upsampler.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # ── AMP scaler ───────────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler('cuda')

    diffusion = SparseDiffusion(
        model,
        timesteps=cfg["diffusion_timesteps"],
        max_T=cfg.get("max_T", None) if args.level > 0 else None,
        n_classes=cfg["n_classes"],
        model_upsampler=model_upsampler,
        weight=torch.tensor(cfg["class_weight"], device=device) if "class_weight" in cfg else None,
        loss_weighting=cfg.get("loss_weighting", None),
        min_snr_gamma=cfg.get("min_snr_gamma", 5.0),
        p2_k=cfg.get("p2_k", 1.0),
        p2_gamma=cfg.get("p2_gamma", 1.0),
        occupancy_objective=cfg.get("occupancy_objective", "mse"),
        occ_pos_weight_max=cfg.get("occ_pos_weight_max", 50.0),
        occ_loss_weight=cfg.get("occ_loss_weight", 1.0),
        occ_label_smoothing=cfg.get("occ_label_smoothing", 0.0),
        occ_loss_weighting=cfg.get("occ_loss_weighting", None),
        occ_min_snr_gamma=cfg.get("occ_min_snr_gamma", None),
        occ_logsnr_min=cfg.get("occ_logsnr_min", -2.0),
        per_sigma_bins=cfg.get("per_sigma_bins", 5),
        mask_input_dropout=cfg.get("mask_input_dropout", 0.0),
    ).to(device)

    if world_size > 1:
        dist.barrier()
        from torch.nn.parallel import DistributedDataParallel as DDP
        diffusion_ddp = DDP(diffusion, device_ids=[0], find_unused_parameters=True)
    else:
        diffusion_ddp = diffusion

    # Weight EMA maintained on the main process only.
    ema = WeightEMA(diffusion, decay=cfg.get("ema_decay", 0.999)) if is_main else None

    n_epochs = cfg["epochs"]
    grad_steps_per_epoch = math.ceil(len(dataset) / effective_batch)
    save_every = cfg["save_every"]
    val_every  = cfg.get("val_every", save_every)

    if is_main:
        print(f"  {len(dataset)} crops — micro_batch={micro_batch} "
              f"× accum={accumulate_steps} × world={world_size} "
              f"= {effective_batch * world_size} global effective batch")
        print(f"  {grad_steps_per_epoch} grad-steps/epoch — "
              f"{n_epochs} epochs ({n_epochs * grad_steps_per_epoch} total)")

    MSE_L, BCE_L, OCC_L, VAL_MSE_L, VAL_BCE_L = [], [], [], [], []
    MSE_LOSS_EMA = None
    BCE_LOSS_EMA = None
    OCC_LOSS_EMA = None
    best_val_loss = float('inf')
    best_epoch = -1
    current_time = datetime.today().strftime('%d-%m-%H:%M')

    writer = SummaryWriter(
        log_dir=f"runs/diffusion_level_{args.level}_{current_time}"
    ) if is_main else None

    loader = _PrefetchLoader(
        dataset, "train", args.level, micro_batch, device, capacity=2
    ).start()

    try:
        for epoch in trange(n_epochs, desc="Epochs", disable=not is_main):
            epoch_mse_loss_sum = 0.0
            epoch_bce_loss_sum = 0.0
            epoch_occ_loss_sum = 0.0
            epoch_occ_iou_sum = 0.0
            # Diagnostics (review item a): occupied-only geometry/class error and
            # per-σ occupancy BCE/IoU, accumulated per micro-batch (NaN-aware for
            # empty σ buckets).
            n_bins = cfg.get("per_sigma_bins", 5)
            epoch_occ_only_mse_sum = 0.0
            epoch_occ_only_ce_sum = 0.0
            n_micro = 0
            bin_bce_sum = torch.zeros(n_bins, device=device)
            bin_iou_sum = torch.zeros(n_bins, device=device)
            bin_cnt     = torch.zeros(n_bins, device=device)

            for _ in range(grad_steps_per_epoch):
                optimizer.zero_grad()

                # ── Gradient accumulation loop ────────────────────────────
                step_mse = 0.0
                step_bce = 0.0
                step_occ = 0.0
                step_iou = 0.0
                any_backward = False
                for accum_step in range(accumulate_steps):
                    batch = loader.next()

                    # Defer DDP allreduce on all but the last accum step
                    is_last_accum = (accum_step == accumulate_steps - 1)
                    sync_ctx = (
                        contextlib.nullcontext()
                        if (world_size == 1 or is_last_accum)
                        else diffusion_ddp.no_sync()
                    )

                    with sync_ctx:
                        with torch.amp.autocast('cuda'):
                            if args.level == 0:
                                X0 = batch
                                mse_loss, bce_loss, occ_loss, _, _, occ_iou, metrics = diffusion_ddp(X0)
                            else:
                                X, X_UP, X0 = batch
                                with torch.no_grad():
                                    X0_BLUR = model_upsampler(X, X_UP).detach()
                                X0_BLUR.grid = X0.grid
                                x0c, bc = clip_data_per_element(
                                    X0, X0_BLUR, cfg["clip_size"])
                                mse_loss, bce_loss, occ_loss, _, _, occ_iou, metrics = diffusion_ddp(x0c, bc)

                            loss = (mse_loss + bce_loss + occ_loss) / accumulate_steps

                        if torch.isnan(loss) or torch.isinf(loss):
                            continue
                        scaler.scale(loss).backward()
                        any_backward = True

                    mse_val = mse_loss.item()
                    bce_val = bce_loss.item()
                    occ_val = occ_loss.item()
                    if not (math.isnan(mse_val) or math.isnan(bce_val) or math.isnan(occ_val)):
                        step_mse += mse_val
                        step_bce += bce_val
                        step_occ += occ_val
                        step_iou += occ_iou.item()

                        n_micro += 1
                        epoch_occ_only_mse_sum += metrics['occ_only_mse'].item()
                        epoch_occ_only_ce_sum  += metrics['occ_only_ce'].item()
                        b_sig = metrics['bce_per_sigma']
                        i_sig = metrics['iou_per_sigma']
                        valid = ~torch.isnan(b_sig)
                        bin_bce_sum[valid] += b_sig[valid]
                        bin_iou_sum[valid] += i_sig[valid]
                        bin_cnt[valid]     += 1


                if any_backward:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.)
                    scaler.step(optimizer)
                    if ema is not None:
                        ema.update(diffusion)
                scaler.update()

                epoch_mse_loss_sum += step_mse / accumulate_steps
                epoch_bce_loss_sum += step_bce / accumulate_steps
                epoch_occ_loss_sum += step_occ / accumulate_steps
                epoch_occ_iou_sum += step_iou / accumulate_steps

            epoch_mse_loss = epoch_mse_loss_sum / grad_steps_per_epoch
            epoch_bce_loss = epoch_bce_loss_sum / grad_steps_per_epoch
            epoch_occ_loss = epoch_occ_loss_sum / grad_steps_per_epoch
            epoch_occ_iou  = epoch_occ_iou_sum / grad_steps_per_epoch

            denom = max(n_micro, 1)
            epoch_occ_only_mse = epoch_occ_only_mse_sum / denom
            epoch_occ_only_ce  = epoch_occ_only_ce_sum / denom
            bin_bce = (bin_bce_sum / bin_cnt.clamp(min=1)).tolist()
            bin_iou = (bin_iou_sum / bin_cnt.clamp(min=1)).tolist()

            if not math.isnan(epoch_mse_loss):
                MSE_LOSS_EMA = epoch_mse_loss if MSE_LOSS_EMA is None else 0.99 * MSE_LOSS_EMA + 0.01 * epoch_mse_loss
            if not math.isnan(epoch_bce_loss):
                BCE_LOSS_EMA = epoch_bce_loss if BCE_LOSS_EMA is None else 0.99 * BCE_LOSS_EMA + 0.01 * epoch_bce_loss
            if not math.isnan(epoch_occ_loss):
                OCC_LOSS_EMA = epoch_occ_loss if OCC_LOSS_EMA is None else 0.99 * OCC_LOSS_EMA + 0.01 * epoch_occ_loss
            MSE_L.append(MSE_LOSS_EMA)
            BCE_L.append(BCE_LOSS_EMA)
            OCC_L.append(OCC_LOSS_EMA)

            if is_main:
                writer.add_scalars(
                    "Loss/train",
                    {"MSE": epoch_mse_loss,
                     "BCE": epoch_bce_loss,
                     "OCC": epoch_occ_loss,
                     "Total": epoch_mse_loss + epoch_bce_loss + epoch_occ_loss,
                    }, epoch)
                # Occupied-only geometry/class error (the aggregate MSE/CE are
                # dominated by zero-filled empties and read near-zero regardless).
                writer.add_scalars(
                    "OccOnly/train",
                    {"MSE": epoch_occ_only_mse, "CE": epoch_occ_only_ce}, epoch)
                # Per-σ occupancy: bin 0 = cleanest (high SNR) … last = noisiest.
                # A flat BCE/IoU that only collapses in the high-σ bins is the
                # expected entropy floor; poor low/mid-σ bins signal a real bug.
                writer.add_scalars(
                    "OccBCE_per_sigma/train",
                    {f"bin{k}": bin_bce[k] for k in range(n_bins)}, epoch)
                writer.add_scalars(
                    "OccIoU_per_sigma/train",
                    {f"bin{k}": bin_iou[k] for k in range(n_bins)}, epoch)

            val_suffix = ""
            if is_main and val_dataset is not None and epoch % val_every == 0:
                # Validate and checkpoint from the EMA weights, then restore the
                # live weights for continued training.
                if ema is not None:
                    ema.store(diffusion)
                    ema.copy_to(diffusion)
                try:
                    val_mse_loss, val_bce_loss, val_occ_loss, val_occ_iou, val_metrics = val_dataset.compute_val_loss(
                        diffusion, "test", args.level, n_crops=cfg.get("val_crops", 16),
                        clip_size=cfg["clip_size"], device=device,
                    )
                    val_total_loss = (val_mse_loss.item() + val_bce_loss.item()
                                      + val_occ_loss.item())
                    if val_mse_loss is not None and val_bce_loss is not None:
                        VAL_MSE_L.append((epoch, val_mse_loss.item()))
                        val_suffix += f", Val MSE={val_mse_loss:.4f}"
                        VAL_BCE_L.append((epoch, val_bce_loss.item()))
                        val_suffix += f" + Val BCE={val_bce_loss:.4f}"
                        val_suffix += f" + Val OCC={val_occ_loss:.4f}"
                        val_suffix += f" + Val OccIoU={val_occ_iou:.3f}"
                        writer.add_scalars(
                            'Loss/Val',
                            {"Val_MSE": val_mse_loss.item(),
                             "Val_BCE": val_bce_loss.item(),
                             "Val_OCC": val_occ_loss.item(),
                             "Val_Total": val_total_loss,
                            }, epoch
                        )
                        writer.add_scalars(
                            'OccIoU',
                            {"train": epoch_occ_iou, "val": val_occ_iou.item()}, epoch
                        )
                        val_bin_bce = val_metrics['bce_per_sigma'].tolist()
                        val_bin_iou = val_metrics['iou_per_sigma'].tolist()
                        writer.add_scalars(
                            "OccOnly/val",
                            {"MSE": val_metrics['occ_only_mse'].item(),
                             "CE":  val_metrics['occ_only_ce'].item()}, epoch)
                        writer.add_scalars(
                            "OccBCE_per_sigma/val",
                            {f"bin{k}": val_bin_bce[k] for k in range(n_bins)}, epoch)
                        writer.add_scalars(
                            "OccIoU_per_sigma/val",
                            {f"bin{k}": val_bin_iou[k] for k in range(n_bins)}, epoch)
                        val_suffix += (
                            " | Val IoU/σ[clean→noisy]="
                            + "/".join(f"{v:.2f}" for v in val_bin_iou))

                        if val_total_loss < best_val_loss:
                            best_val_loss = val_total_loss
                            best_epoch = epoch
                            best_ckpt = f"checkpoints/diffusion_models/dales_{args.level}_{current_time}_best.pt"
                            torch.save(diffusion, best_ckpt)
                            tqdm.write(
                                f"New best model saved at epoch {epoch}: {best_ckpt} — "
                                f"Val loss: {best_val_loss:.4f} "
                                f"(MSE: {val_mse_loss:.4f}, BCE: {val_bce_loss:.4f}, "
                                f"OCC: {val_occ_loss:.4f})")
                            writer.add_scalar("Best Val Loss", best_val_loss, epoch)
                finally:
                    if ema is not None:
                        ema.restore(diffusion)

            if is_main:
                iou_str = "/".join(f"{v:.2f}" for v in bin_iou)
                tqdm.write(
                    f"Epoch {epoch} - LOSS: MSE={epoch_mse_loss:.4f} | "
                    f"BCE={epoch_bce_loss:.4f} | OCC={epoch_occ_loss:.4f} "
                    f"| OccIoU={epoch_occ_iou:.3f} "
                    f"| OccOnly MSE={epoch_occ_only_mse:.4f}/CE={epoch_occ_only_ce:.3f} "
                    f"| IoU/σ[clean→noisy]={iou_str}"
                    + val_suffix)

    finally:
        loader.stop()

        if is_main:
            ckpt = 'checkpoints/diffusion_models/dales_{}_{}.pt'.format(
                args.level, current_time)
            # Save the EMA weights as the final checkpoint (better sample quality).
            if ema is not None:
                ema.copy_to(diffusion)
            torch.save(diffusion, ckpt)
            writer.close()
            print(ckpt)
            print(f"Best epoch: {best_epoch}")
            print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    # ── DDP setup (torchrun sets LOCAL_RANK / RANK / WORLD_SIZE) ─────────────
    local_rank  = _local_rank
    rank        = int(os.environ.get("RANK",       0))
    world_size  = _world_size

    if world_size > 1:
        torch.cuda.set_device(0)
        dist.init_process_group(backend="nccl")

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

    dataset_name = args.dataset or cfg.get("dataset", None)

    if dataset_name == "dales":
        try:
            _train_dales(args, cfg, device, rank=rank, world_size=world_size)
        finally:
            if world_size > 1:
                dist.destroy_process_group()

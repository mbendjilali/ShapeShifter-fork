import torch
import torch.nn as nn
from einops import repeat
import torch.nn.functional as F
from torch.special import expm1
import math
import fvdb.nn as fvnn
from tqdm import tqdm


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))


def alpha_cosine_log_snr(t, s: float = 0.008):
    # not sure if this accounts for beta being clipped to 0.999 in discrete version
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps=1e-5)


def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))


class SparseDiffusion(nn.Module):  # Inspired by bitfusion by lucidrain
    def __init__(
        self,
        model,
        timesteps=1000,
        max_T=None,
        noise_schedule='cosine',
        time_difference=0.,
        n_classes=None,
        model_upsampler=None,
        weight=None,
        loss_weighting=None,
        min_snr_gamma=5.0,
        p2_k=1.0,
        p2_gamma=1.0,
        void_weight=1.0,
        per_sigma_bins=5,
    ):
        super().__init__()
        self.model = model
        self.model_upsampler = model_upsampler
        self.channels = 1

        # Noise-level weighting of the geometry MSE (x0 head):
        #   'min_snr' → Min-SNR-γ (Hang et al. 2023): min(SNR, γ); rebalances to mid-σ.
        #   'p2'      → P2-true on an x0 head (Choi et al. 2022): SNR·(k+SNR)^(−γ).
        assert loss_weighting in (None, 'min_snr', 'p2'), \
            f"invalid loss_weighting {loss_weighting!r}"
        self.loss_weighting = loss_weighting
        self.min_snr_gamma = min_snr_gamma
        self.p2_k = p2_k
        self.p2_gamma = p2_gamma

        # Occupancy lives in the SEMANTIC categorical: the class logits (4:4+n_cls)
        # plus the last channel (a "void"/empty logit) form one (n_cls+1)-way
        # softmax.  Empty voxels are the void category, so the empty state is
        # reproducible by the same softmax at sampling (occupied class-sum→1, empty
        # class-sum→0, mask→±1).  Occupancy = 1 − P(void).  void_weight scales the
        # (majority) void category in the CE — lower → keep capacity on real
        # classes / generate more occupancy, higher → prune more.
        self.void_weight = void_weight
        # Number of σ buckets for the per-noise-level diagnostics.
        self.per_sigma_bins = per_sigma_bins

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.timesteps = timesteps
        if max_T is None:
            max_T = timesteps
        self.max_T = max_T
        self.time_difference = time_difference
        self.n_classes = n_classes
        if weight is not None:
            self.register_buffer('class_weight', weight.float())
        else:
            self.class_weight = None

    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_sampling_timesteps(self, batch, *, device, steps=None):
        if steps is None:
            steps = self.max_T + 1
        times = torch.linspace(self.max_T/self.timesteps,
                               0., steps, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    def _sigmoid_semantic_channels(self, x: fvnn.VDBTensor) -> None:
        """Decode the network's raw prediction into the data domain in-place (for
        sampling) via one (n_cls+1)-way softmax over [class logits, void logit].
        This reproduces both clean states the way they were trained: occupied →
        class-sum 1 / mask +1, empty → class-sum 0 / mask −1.  (A plain softmax
        over the classes alone would force class-sum→1 on every voxel and erase
        the empty cue — the cause of the old sampling flood.)"""
        n_cls = self.n_classes
        cat = torch.cat([x.data.jdata[:, 4:4 + n_cls], x.data.jdata[:, -1:]], dim=1)
        probs = torch.softmax(cat, dim=-1)                 # (N, n_cls+1)
        real = probs[:, :n_cls]                            # P(class), sums to 1−P(void)
        x.data.jdata[:, 4:4 + n_cls] = real
        # mask = 1 − 2·P(void) = 2·(class-sum) − 1 ∈ (−1, 1); +1 occupied, −1 void
        x.data.jdata[:, -1] = 2.0 * real.sum(dim=-1) - 1.0

    @torch.no_grad()
    def ddpm_sample(self, noisy_grid: fvnn.VDBTensor, X_Blur: fvnn.VDBTensor = None, clip=None):

        time_pairs = self.get_sampling_timesteps(1, device=self.device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):

            # add the time delay
            time_next = (time_next).clamp(min=0.)

            # get predicted x0
            x_start = self.model(noisy_grid, time.repeat(len(noisy_grid.jidx)))
            # decode the categorical logits (classes + void) to the data domain
            # before using x_start in the denoising formula, to stay in-distribution
            self._sigmoid_semantic_channels(x_start)
            if not clip is None:
                x_start.data.jdata = torch.clip(x_start.jdata, -clip, clip)
            if time_next == 0:
                return x_start

            # blend the x0 estimate toward the upsampler output (matches q_sample)
            if not X_Blur is None:
                gamma = self.blend_gamma(time_next)
                start_data = (1-gamma[:, None])*x_start.jdata + \
                    gamma[:, None]*X_Blur.jdata
            else:
                start_data = x_start.jdata

            # get log(snr)
            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            # derive posterior mean and variance
            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (noisy_grid.jdata *
                                 (1 - c) / alpha + c * start_data)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # get noise
            noise = torch.randn_like(noisy_grid.jdata)
            noisy_grid.data.jdata = mean + \
                (0.5 * log_variance).exp() * noise

    @torch.no_grad()
    def ddim_sample(self, noisy_grid: fvnn.VDBTensor, steps=None):
        time_difference = self.time_difference
        time_pairs = self.get_sampling_timesteps(
            1, device=self.device, steps=steps)

        x_start = None

        for times, times_next in tqdm(time_pairs, desc='sampling loop time step'):
            # add the time delay
            times_next = (times_next - time_difference).clamp(min=0.)

            # get times and noise levels
            log_snr = self.log_snr(times)
            log_snr_next = self.log_snr(times_next)
            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            # predict x0
            x_start = self.model(
                noisy_grid, times.repeat(len(noisy_grid.jidx)))
            # decode the categorical logits (classes + void) to the data domain
            self._sigmoid_semantic_channels(x_start)
            if times_next == 0:
                return x_start

            # get predicted noise
            pred_noise_jdata = (noisy_grid.jdata - alpha *
                                x_start.jdata) / sigma.clamp(min=1e-8)
            noisy_grid.data.jdata = x_start.jdata * \
                alpha_next + pred_noise_jdata * sigma_next

        return (noisy_grid)

    @torch.no_grad()
    def ddpm_sample_class_clamp(self, noisy_grid, target_class, clamp_clean=False, clip=None):
        """
        Diagnostic conditioning. Force every voxel's class channels to `target_class`
        throughout sampling; let offset/intensity/mask denoise freely.
        Read out the MASK (occupancy) — not the labels — to see if the model knows
        what `target_class` geometry looks like.
        """
        n_cls = self.n_classes
        N = noisy_grid.jdata.shape[0]
        onehot = torch.zeros(N, n_cls, device=self.device)
        onehot[:, target_class] = 1.0
        cs, ce = 4, 4 + n_cls

        time_pairs = self.get_sampling_timesteps(1, device=self.device)
        for time, time_next in tqdm(time_pairs, desc='class-clamped sampling'):
            time_next = time_next.clamp(min=0.)

            # --- clamp class channels of the INPUT state = the conditioning signal ---
            if clamp_clean:
                noisy_grid.data.jdata[:, cs:ce] = onehot                       # strongest, slightly OOD
            else:
                a, s = log_snr_to_alpha_sigma(self.log_snr(time))
                noisy_grid.data.jdata[:, cs:ce] = a * onehot + s * torch.randn_like(onehot)  # in-distribution

            x_start = self.model(noisy_grid, time.repeat(len(noisy_grid.jidx)))
            self._sigmoid_semantic_channels(x_start)
            if clip is not None:
                x_start.data.jdata = torch.clip(x_start.jdata, -clip, clip)
            if time_next == 0:
                x_start.data.jdata[:, cs:ce] = onehot   # unambiguous labels for readout
                return x_start

            log_snr, log_snr_next = self.log_snr(time), self.log_snr(time_next)
            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)
            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (noisy_grid.jdata * (1 - c) / alpha + c * x_start.jdata)
            variance = (sigma_next ** 2) * c
            noise = torch.randn_like(noisy_grid.jdata)
            noisy_grid.data.jdata = mean + (0.5 * log(variance)).exp() * noise

        return noisy_grid

    @torch.no_grad()
    def sample(self, noisy_grid: fvnn.VDBTensor):
        return self.ddim_sample(noisy_grid)

    def blend_gamma(self, t: torch.Tensor) -> torch.Tensor:
        """SR3-style upsampler-blend weight for level>0 refinement.

        Interpolates the diffusion target between the fine GT (γ=0, at t=0) and the
        upsampler estimate X_Blur (γ=1, at the top of the truncated schedule
        t = max_T/timesteps).  So the reverse process starts anchored on the
        upsampler output and denoises toward the GT — the model learns to *refine*
        the upsampler, instead of denoising GT and ignoring the estimate.
        (The previous `t/max_T` was ~timesteps× too small, so γ≈0 always and the
        estimate was never blended in — the level>0 diffusion degraded it.)"""
        return (t * self.timesteps / self.max_T).clamp(0., 1.)

    def q_sample(self, X: fvnn.VDBTensor, times: torch.tensor, X_Blur: fvnn.VDBTensor = None):
        assert len(times) == len(X.data.jidx)
        # compute constant
        noise_level = self.log_snr(times)
        alpha, sigma = log_snr_to_alpha_sigma(noise_level)

        # random noise
        noise = torch.randn_like(X.jdata)

        # blend the clean target toward the upsampler estimate at high noise
        if not X_Blur is None:
            gamma = self.blend_gamma(times)
            target_X = (1-gamma[:, None])*X.jdata + gamma[:, None]*X_Blur.jdata
        else:
            target_X = X.jdata

        # corrupted X
        noised_img = alpha[:, None] * target_X + sigma[:, None] * noise
        return fvnn.VDBTensor(grid=X.grid, data=X.grid.jagged_like(noised_img)), target_X

    @torch.no_grad()
    def _per_sigma_occ_stats(self, times, per_voxel_bce, pred_occ, occ_target_hard):
        """Bucket the occupancy CE and IoU by noise level.

        Buckets are uniform in t over [0, max_T/timesteps]; bin 0 = lowest t =
        cleanest (highest SNR), last bin = noisiest.  The high-σ bins are the
        hard, structure-generating regime; the low-σ bins are near-trivial.
        Empty buckets return NaN so the caller can skip them when averaging.
        """
        n_bins = self.per_sigma_bins
        t_hi = self.max_T / self.timesteps
        edges = torch.linspace(0., t_hi, n_bins + 1, device=times.device)
        bce_bins, iou_bins = [], []
        nan = torch.tensor(float('nan'), device=times.device)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            m = (times >= lo) & ((times <= hi) if i == n_bins - 1 else (times < hi))
            if m.any():
                bce_bins.append(per_voxel_bce[m].mean())
                inter = (pred_occ[m] & occ_target_hard[m]).sum().float()
                union = (pred_occ[m] | occ_target_hard[m]).sum().clamp(min=1).float()
                iou_bins.append(inter / union)
            else:
                bce_bins.append(nan)
                iou_bins.append(nan)
        return torch.stack(bce_bins), torch.stack(iou_bins)

    def forward(self, X: fvnn.VDBTensor, X_Blur: fvnn.VDBTensor = None):

        # random times
        times = torch.zeros((X.grid_count,), device=self.device).float().uniform_(
            0., self.max_T/self.timesteps)
        times = times[X.data.jidx.long()]

        noisy_latents, target_X = self.q_sample(X, times, X_Blur)

        # prediction
        pred: fvnn.VDBTensor = self.model(noisy_latents, times)

        # per-voxel SNR — reused by the geometry weighting and the diagnostics.
        snr = torch.exp(self.log_snr(times))

        n_cls = self.n_classes

        # --- Geometry MSE (offset+intensity) — optionally Min-SNR-γ / P2 weighted.
        # The last channel is the void logit (part of the categorical below), so it
        # is excluded from the Gaussian MSE.
        geom_pred   = pred.jdata[:, :4]
        geom_target = X.jdata[:, :4]
        if self.loss_weighting is not None:
            # The model predicts x_start (x0), so weights are multipliers on the
            # x0-MSE, mean-normalized so the loss scale (grad-clip / LR) is preserved.
            if self.loss_weighting == 'min_snr':
                w = snr.clamp(max=self.min_snr_gamma)
            else:  # 'p2' — P2-true on an x0 head: SNR·(k+SNR)^(−γ)
                w = snr * (self.p2_k + snr).pow(-self.p2_gamma)
            w = w / w.mean().clamp(min=1e-8)
            per_voxel_se = ((geom_pred - geom_target) ** 2).mean(dim=1)
            mse_loss = (w * per_voxel_se).mean()
        else:
            mse_loss = F.mse_loss(geom_pred, geom_target)

        # --- Unified (n_cls+1)-way categorical: [class logits, void logit] ---
        # Occupancy is "not void".  The same softmax reproduces empty (void) and
        # occupied states at sampling, so the model's empty cue survives.
        occ_target_hard = (X.jdata[:, -1] > 0)
        pred_label   = pred.jdata[:, 4:4 + n_cls]
        target_label = X.jdata[:, 4:4 + n_cls]
        cat_logits  = torch.cat([pred_label, pred.jdata[:, -1:]], dim=1)
        void_target = (~occ_target_hard).float()
        cat_target  = torch.cat([target_label, void_target[:, None]], dim=1)  # rows sum to 1
        if self.class_weight is not None:
            w9 = torch.cat([self.class_weight,
                            self.class_weight.new_tensor([self.void_weight])])
        else:
            w9 = None
        per_voxel_cat_ce = F.cross_entropy(cat_logits, cat_target, weight=w9, reduction='none')
        class_loss = per_voxel_cat_ce.mean()
        pred_occ = cat_logits.argmax(dim=1) != n_cls       # occupied = not the void category

        # --- Diagnostics: occupancy IoU + per-σ + occupied-only ---
        with torch.no_grad():
            inter = (pred_occ & occ_target_hard).sum().float()
            union = (pred_occ | occ_target_hard).sum().clamp(min=1).float()
            occ_iou = inter / union

            bce_per_sigma, iou_per_sigma = self._per_sigma_occ_stats(
                times, per_voxel_cat_ce, pred_occ, occ_target_hard)

            # Occupied-only metrics: the aggregate MSE/CE are dominated by empty
            # voxels; restrict to occupied to see if geometry/labels are fit there.
            if occ_target_hard.any():
                occ_only_mse = ((pred.jdata[:, :4] - X.jdata[:, :4])[occ_target_hard]
                                ** 2).mean()
                occ_only_ce = F.cross_entropy(
                    pred_label[occ_target_hard], target_label[occ_target_hard],
                    self.class_weight)
            else:
                occ_only_mse = occ_iou.new_zeros(())
                occ_only_ce = occ_iou.new_zeros(())

        metrics = {
            'occ_only_mse': occ_only_mse.detach(),
            'occ_only_ce': occ_only_ce.detach(),
            'bce_per_sigma': bce_per_sigma,   # (n_bins,) per-σ CE, NaN for empty buckets
            'iou_per_sigma': iou_per_sigma,   # (n_bins,) per-σ occupancy IoU
        }
        return mse_loss, class_loss, pred_label, target_label, occ_iou, metrics

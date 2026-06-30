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
        occupancy_objective='mse',
        occ_pos_weight_max=50.0,
        occ_loss_weight=1.0,
        occ_label_smoothing=0.0,
        occ_loss_weighting=None,
        occ_min_snr_gamma=None,
        occ_logsnr_min=-2.0,
        per_sigma_bins=5,
        mask_input_dropout=0.0,
        void_weight=1.0,
    ):
        super().__init__()
        self.model = model
        self.model_upsampler = model_upsampler
        self.channels = 1

        #   'min_snr' → Min-SNR-γ (Hang et al. 2023): min(SNR, γ); rebalances to mid-σ.
        #   'p2'      → P2-true on an x0 head (Choi et al. 2022): SNR·(k+SNR)^(−γ);
        #               keeps the high-σ lean while taming the SNR→0 tail.

        assert loss_weighting in (None, 'min_snr', 'p2'), \
            f"invalid loss_weighting {loss_weighting!r}"
        self.loss_weighting = loss_weighting
        self.min_snr_gamma = min_snr_gamma
        self.p2_k = p2_k
        self.p2_gamma = p2_gamma

        # Occupancy (mask channel) objective.
        #   'mse' → legacy: mask is a Gaussian channel inside the geometry MSE
        #           (the MSE-optimal collapse to "-1 everywhere" under heavy
        #            empty/occupied imbalance — the level-0 saturation symptom).
        #   'bce' → occupancy is a proper Bernoulli: channel -1 is a *logit*,
        #           trained with BCE-with-logits + pos_weight to handle imbalance,
        #           and mapped to the data domain (2σ-1 = E[bit]) during sampling.
        #   'void' → occupancy is folded into the SEMANTIC categorical: the class
        #           logits (4:4+n_cls) plus the last channel (a "void"/empty logit)
        #           form one (n_cls+1)-way softmax.  Empty voxels are the void
        #           category, so the empty state IS reproducible by the same
        #           softmax at sampling (occupied class-sum→1, empty class-sum→0,
        #           mask→±1) — the fix for the sampling flood, where the legacy
        #           softmax forced class-sum→1 on every voxel and erased the
        #           "empty" cue.  Occupancy = 1 − P(void).
        assert occupancy_objective in ('mse', 'bce', 'void'), \
            f"invalid occupancy_objective {occupancy_objective!r}"
        self.occupancy_objective = occupancy_objective
        # Weight of the void (empty) category in the (n_cls+1)-way CE.  Void is the
        # majority class; lower it to keep capacity on the real classes, raise it
        # to prune more aggressively.
        self.void_weight = void_weight
        self.occ_pos_weight_max = occ_pos_weight_max
        self.occ_loss_weight = occ_loss_weight
        # Label smoothing on the occupancy target (0,1 → eps,1-eps) bounds the
        # BCE: it caps how confident logits can grow, which stops the val BCE
        # explosion when the model over-fits per-tile occupancy.
        self.occ_label_smoothing = occ_label_smoothing

        # σ-weighting of the occupancy BCE (review item b).  Unweighted BCE spends
        # ~half its gradient on the high-σ regime where x_t is pure noise and the
        # Bayes-optimal occupancy is the per-voxel marginal (an irreducible entropy
        # floor) — wasted budget.  These modes reallocate it toward the mid/low-σ
        # band where structure is actually recoverable from context:
        #   None            → uniform (legacy)
        #   'min_snr'       → w = min(SNR, γ) / mean, suppresses the high-σ tail
        #   'snr'           → w = SNR / mean, aggressive low-σ emphasis
        #   'clip_high_sigma' → w = 1[log_snr ≥ occ_logsnr_min], hard cutoff that
        #                       drops the unlearnable high-noise samples entirely
        assert occ_loss_weighting in (None, 'min_snr', 'snr', 'clip_high_sigma'), \
            f"invalid occ_loss_weighting {occ_loss_weighting!r}"
        self.occ_loss_weighting = occ_loss_weighting
        self.occ_min_snr_gamma = occ_min_snr_gamma if occ_min_snr_gamma is not None else min_snr_gamma
        self.occ_logsnr_min = occ_logsnr_min
        # Number of σ buckets for the per-noise-level diagnostics (review item a).
        self.per_sigma_bins = per_sigma_bins

        # Mask-input dropout — the leak fix.  The per-σ IoU curve revealed the
        # occupancy head was learned by *copying* the noised mask channel from its
        # own input (IoU 0.99 at low σ), not by generating structure — so at
        # sampling (no mask to copy) it floods to "occupied everywhere" (the dense
        # cube).  With probability p we replace the input mask channel with zeros
        # for whole samples, so the head must predict occupancy from the
        # geometry/semantics/coordinate context (which *are* generated) instead of
        # the leaked mask.  p=1.0 ⇒ occupancy is purely a structure predictor on
        # the generated features (recommended experiment); p=0.0 ⇒ legacy.
        self.mask_input_dropout = mask_input_dropout

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
        """Decode the network's raw prediction into the data domain in-place
        (for sampling): CE-trained class logits → softmax probabilities, and —
        in 'bce' occupancy mode — the occupancy logit → its analog-bit expectation
        2σ(logit)−1 ∈ (−1,1), so the Gaussian posterior and the threshold-at-0
        readout (remove_mask) both stay in-distribution."""
        n_cls = getattr(self, 'n_classes', None)
        objective = getattr(self, 'occupancy_objective', 'mse')
        if objective == 'void':
            # One (n_cls+1)-way softmax over [class logits, void logit].  This
            # reproduces BOTH clean states the same way it was trained: occupied →
            # class-sum 1 / mask +1, empty → class-sum 0 / mask −1.  No more
            # softmax-forces-sum-1 mismatch, so the sampler can actually go empty.
            cat = torch.cat([x.data.jdata[:, 4:4 + n_cls],
                             x.data.jdata[:, -1:]], dim=1)
            probs = torch.softmax(cat, dim=-1)             # (N, n_cls+1)
            real = probs[:, :n_cls]                        # P(class), sums to 1−P(void)
            x.data.jdata[:, 4:4 + n_cls] = real
            # mask = 1 − 2·P(void) = 2·(class-sum) − 1 ∈ (−1, 1); +1 occupied, −1 void
            x.data.jdata[:, -1] = 2.0 * real.sum(dim=-1) - 1.0
            return
        if n_cls:
            x.data.jdata[:, 4:4 + n_cls] = torch.softmax(x.data.jdata[:, 4:4 + n_cls], dim=-1)
        if objective == 'bce':
            x.data.jdata[:, -1] = 2.0 * torch.sigmoid(x.data.jdata[:, -1]) - 1.0

    @torch.no_grad()
    def ddpm_sample(self, noisy_grid: fvnn.VDBTensor, X_Blur: fvnn.VDBTensor = None, clip=None):

        time_pairs = self.get_sampling_timesteps(1, device=self.device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):

            # add the time delay
            time_next = (time_next).clamp(min=0.)

            # get predicted x0
            x_start = self.model(noisy_grid, time.repeat(len(noisy_grid.jidx)))
            # semantic channels were trained with BCE (logits); convert to probabilities
            # before using x_start in the denoising formula to stay in-distribution
            self._sigmoid_semantic_channels(x_start)
            if not clip is None:
                x_start.data.jdata = torch.clip(x_start.jdata, -clip, clip)
            if time_next == 0:
                return x_start

            # Optionnal: clip x0
            if not X_Blur is None:
                gamma = time_next/self.max_T
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
            # semantic channels were trained with BCE (logits); convert to probabilities
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

    def q_sample(self, X: fvnn.VDBTensor, times: torch.tensor, X_Blur: fvnn.VDBTensor = None):
        assert len(times) == len(X.data.jidx)
        # compute constant
        noise_level = self.log_snr(times)
        alpha, sigma = log_snr_to_alpha_sigma(noise_level)

        # random noise
        noise = torch.randn_like(X.jdata)

        # compute gamma
        if not X_Blur is None:
            gamma = times/self.max_T
            target_X = (1-gamma[:, None])*X.jdata + gamma[:, None]*X_Blur.jdata
        else:
            target_X = X.jdata

        # corrupted X
        noised_img = alpha[:, None] * target_X + sigma[:, None] * noise
        return fvnn.VDBTensor(grid=X.grid, data=X.grid.jagged_like(noised_img)), target_X

    def _sigma_weight(self, snr: torch.Tensor) -> torch.Tensor:
        """Per-voxel σ-weight for the occupancy BCE, mean-normalized so the loss
        scale (and the existing grad-clip / LR) are preserved.  `snr = exp(log_snr)`
        is per-voxel; higher σ ↔ lower SNR."""
        mode = self.occ_loss_weighting
        if mode == 'min_snr':
            w = snr.clamp(max=self.occ_min_snr_gamma)
        elif mode == 'snr':
            w = snr
        else:  # 'clip_high_sigma'
            w = (torch.log(snr.clamp(min=1e-20)) >= self.occ_logsnr_min).float()
        return w / w.mean().clamp(min=1e-8)

    @torch.no_grad()
    def _per_sigma_occ_stats(self, times, per_voxel_bce, pred_occ, occ_target_hard):
        """Bucket occupancy BCE and IoU by noise level (review item a).

        Buckets are uniform in t over [0, max_T/timesteps]; bin 0 = lowest t =
        cleanest (highest SNR), last bin = noisiest.  A flat BCE across buckets
        that only collapses in the high-σ bins is the expected entropy floor; a
        model that is *also* poor in the low/mid-σ bins has a real learnability
        bug — that distinction is the whole point of this diagnostic.
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

        # Mask-input dropout (the leak fix): zero the input mask channel for a
        # random subset of samples so the occupancy head cannot copy it and must
        # predict structure from the (generated) geometry/semantics/coords.
        if self.occupancy_objective == 'bce' and self.mask_input_dropout > 0:
            drop_sample = (torch.rand(X.grid_count, device=self.device)
                           < self.mask_input_dropout)
            drop_vox = drop_sample[X.data.jidx.long()]
            if drop_vox.any():
                nl = noisy_latents.jdata.clone()
                nl[drop_vox, -1] = 0.0
                noisy_latents = fvnn.VDBTensor(
                    noisy_latents.grid, noisy_latents.grid.jagged_like(nl))

        # prediction
        pred: fvnn.VDBTensor = self.model(noisy_latents, times)

        # per-voxel SNR — reused by the geometry weighting, the occupancy
        # σ-weighting, and the per-σ diagnostics below.
        snr = torch.exp(self.log_snr(times))

        n_cls = self.n_classes

        # --- Geometry MSE — optionally Min-SNR-γ / P2 weighted ---
        # In 'bce'/'void' the mask leaves the Gaussian MSE (it becomes a Bernoulli
        # / part of the categorical, below); in legacy 'mse' mode it stays in.
        if self.occupancy_objective in ('bce', 'void'):
            geom_pred   = pred.jdata[:, :4]
            geom_target = X.jdata[:, :4]
        else:
            geom_pred   = torch.cat([pred.jdata[:, :4], pred.jdata[:, -1][:, None]], dim=1)
            geom_target = torch.cat([X.jdata[:, :4], X.jdata[:, -1][:, None]], dim=1)
        if self.loss_weighting is not None:
            # SNR = exp(log_snr); times/log_snr are per-voxel.  The model predicts
            # x_start (x0-parameterization), so weights are expressed as multipliers
            # on the x0-MSE.  Normalize by the batch-mean weight so the loss scale
            # (and the existing grad-clip / LR) are preserved.
            if self.loss_weighting == 'min_snr':
                w = snr.clamp(max=self.min_snr_gamma)
            else:  # 'p2' — P2-true on an x0 head: SNR·(k+SNR)^(−γ)
                w = snr * (self.p2_k + snr).pow(-self.p2_gamma)
            w = w / w.mean().clamp(min=1e-8)
            per_voxel_se = ((geom_pred - geom_target) ** 2).mean(dim=1)
            mse_loss = (w * per_voxel_se).mean()
        else:
            mse_loss = F.mse_loss(geom_pred, geom_target)

        occ_target_hard = (X.jdata[:, -1] > 0)
        pred_label   = pred.jdata[:, 4:4 + n_cls]
        target_label = X.jdata[:, 4:4 + n_cls]

        if self.occupancy_objective == 'void':
            # --- Unified (n_cls+1)-way categorical: [class logits, void logit] ---
            # Occupancy is no longer a separate channel; it is "not void".  This is
            # the sampling-flood fix: the same softmax reproduces empty (void) and
            # occupied states, so the model's empty cue survives at sampling.
            cat_logits  = torch.cat([pred.jdata[:, 4:4 + n_cls], pred.jdata[:, -1:]], dim=1)
            void_target = (~occ_target_hard).float()
            cat_target  = torch.cat([target_label, void_target[:, None]], dim=1)  # rows sum to 1
            if self.class_weight is not None:
                w9 = torch.cat([self.class_weight,
                                self.class_weight.new_tensor([self.void_weight])])
            else:
                w9 = None
            per_voxel_cat_ce = F.cross_entropy(cat_logits, cat_target, weight=w9, reduction='none')
            if self.occ_loss_weighting is not None:
                w_occ = self._sigma_weight(snr)
                class_loss = (w_occ * per_voxel_cat_ce).mean()
            else:
                class_loss = per_voxel_cat_ce.mean()
            pred_occ = cat_logits.argmax(dim=1) != n_cls   # occupied = not the void category
            per_voxel_occ_bce = per_voxel_cat_ce           # for the per-σ diagnostic
            # OCC is folded into the categorical CE above (no separate gradient).
            # Report a diagnostic binary occupancy BCE so the OCC column is legible
            # — computed under no_grad, so it is NOT in the backward / double-counted.
            with torch.no_grad():
                p_void = torch.softmax(cat_logits, dim=1)[:, n_cls]
                p_occ  = (1.0 - p_void).clamp(1e-6, 1.0 - 1e-6)
                occ_loss = F.binary_cross_entropy(p_occ, occ_target_hard.float())
        else:
            # --- Occupancy BCE (Bernoulli structure prior) — imbalance-handled ---
            # Channel -1 is a logit; pos_weight = #empty/#occupied up-weights the
            # rare occupied class so the optimum is not "predict empty everywhere".
            pred_occ = pred.jdata[:, -1] > 0
            per_voxel_occ_bce = None
            if self.occupancy_objective == 'bce':
                occ_logit  = pred.jdata[:, -1]
                occ_target = occ_target_hard.float()
                n_occ   = occ_target.sum().clamp(min=1.0)
                n_empty = (occ_target.numel() - occ_target.sum()).clamp(min=1.0)
                pos_weight = (n_empty / n_occ).clamp(max=self.occ_pos_weight_max)
                eps = self.occ_label_smoothing
                soft_target = occ_target * (1.0 - 2.0 * eps) + eps if eps > 0 else occ_target
                per_voxel_occ_bce = F.binary_cross_entropy_with_logits(
                    occ_logit, soft_target, pos_weight=pos_weight, reduction='none')
                if self.occ_loss_weighting is not None:
                    w_occ = self._sigma_weight(snr)
                    occ_loss = (w_occ * per_voxel_occ_bce).mean() * self.occ_loss_weight
                else:
                    occ_loss = per_voxel_occ_bce.mean() * self.occ_loss_weight
            else:
                occ_loss = mse_loss.new_zeros(())
            class_loss = F.cross_entropy(pred_label, target_label, self.class_weight)

        # --- Occupancy IoU + per-σ + occupied-only diagnostics ---
        with torch.no_grad():
            inter = (pred_occ & occ_target_hard).sum().float()
            union = (pred_occ | occ_target_hard).sum().clamp(min=1).float()
            occ_iou = inter / union

            if per_voxel_occ_bce is None:  # 'mse' mode: report the raw mask BCE
                per_voxel_occ_bce = F.binary_cross_entropy_with_logits(
                    pred.jdata[:, -1], occ_target_hard.float(), reduction='none')
            bce_per_sigma, iou_per_sigma = self._per_sigma_occ_stats(
                times, per_voxel_occ_bce, pred_occ, occ_target_hard)

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
            'bce_per_sigma': bce_per_sigma,   # (n_bins,), NaN for empty buckets
            'iou_per_sigma': iou_per_sigma,   # (n_bins,)
        }
        return mse_loss, class_loss, occ_loss, pred_label, target_label, occ_iou, metrics

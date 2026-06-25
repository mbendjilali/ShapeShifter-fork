import torch
from torch.special import expm1
from fvdb_diffusion import log_snr_to_alpha_sigma, log   # your module-level fns
from diffusion_tensor import DiffusionTensor
import fvdb.nn as fvnn


@torch.no_grad()
def reverse_from(diff, noisy_grid, t_start, steps=None, clamp_onehot=None):
    dev = diff.device
    cs, ce = 4, 4 + diff.n_classes
    steps = steps or max(2, int(round(t_start * diff.timesteps)) + 1)
    ts = torch.linspace(float(t_start), 0., steps, device=dev)
    for i in range(steps - 1):
        time, time_next = ts[i:i+1], ts[i+1:i+2].clamp(min=0.)
        if clamp_onehot is not None:                       # conditioning via the class channel
            a, s = log_snr_to_alpha_sigma(diff.log_snr(time))
            noisy_grid.data.jdata[:, cs:ce] = a*clamp_onehot + s*torch.randn_like(clamp_onehot)
        x_start = diff.model(noisy_grid, time.repeat(len(noisy_grid.jidx)))
        diff._sigmoid_semantic_channels(x_start)           # softmax on your branch
        if time_next.item() == 0:
            if clamp_onehot is not None:
                x_start.data.jdata[:, cs:ce] = clamp_onehot
            return x_start
        ls, lsn = diff.log_snr(time), diff.log_snr(time_next)
        a, sg   = log_snr_to_alpha_sigma(ls)
        an, sgn = log_snr_to_alpha_sigma(lsn)
        c = -expm1(ls - lsn)
        mean = an * (noisy_grid.jdata * (1 - c) / a + c * x_start.jdata)
        var  = (sgn**2) * c
        noisy_grid.data.jdata = mean + (0.5*log(var)).exp() * torch.randn_like(noisy_grid.jdata)
    return noisy_grid

@torch.no_grad()
def test_A(diff, X_clean, t_list=(0.1, 0.2, 0.3, 0.4, 0.5)):
    for t in t_list:
        X = DiffusionTensor(X_clean.grid, X_clean.grid.jagged_like(X_clean.jdata.clone()))
        tt = torch.full((X.jdata.shape[0],), float(t), device=diff.device)
        noisy, _ = diff.q_sample(X, tt)
        rec = reverse_from(diff, noisy, t_start=t)
        rec = DiffusionTensor.from_vdb(rec).get_global().remove_mask()
        # pipe `rec` through your existing export_to_laz path -> recon_t{t}.laz

@torch.no_grad()
def test_B(diff, base_grid, class_ids, seed=0):
    C = diff.n_classes
    for cid in class_ids:                       # e.g. [BUILDING_IDX, VEG_IDX]
        torch.manual_seed(seed)
        N = base_grid.ijk.jdata.shape[0]
        jdata = torch.randn(N, 4 + C + 1, device=diff.device)
        noisy = fvnn.VDBTensor(base_grid, base_grid.jagged_like(jdata))
        onehot = torch.zeros(N, C, device=diff.device); onehot[:, cid] = 1.
        x0 = reverse_from(diff, noisy, t_start=diff.max_T/diff.timesteps, clamp_onehot=onehot)
        occ = DiffusionTensor.from_vdb(x0).get_global().remove_mask()
        # export occ -> clamp_class{cid}.laz  — colour by HEIGHT, not class
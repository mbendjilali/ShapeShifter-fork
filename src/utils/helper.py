import torch
from torch.special import expm1
from fvdb_diffusion import log_snr_to_alpha_sigma, log   # your module-level fns


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
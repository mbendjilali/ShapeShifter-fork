import torch
from torch.special import expm1
from fvdb_diffusion import log_snr_to_alpha_sigma, log   # your module-level fns


@torch.no_grad()
def reverse_from(diff, noisy_grid, t_start, steps=None, clamp_onehot=None, X_Blur=None,
                 clamp_mode="hard", clamp_rows=None):
    """clamp_mode, when clamp_onehot is given:
      "hard"   — overwrite the class channels of the *noisy state* each step with
                 a re-noised one-hot.  Because occupancy is encoded as class-sum 1
                 in the same softmax (mask = 2·class-sum − 1), this also asserts
                 "occupied" at every clamped voxel.
      "renorm" — project the *x0 estimate* after decoding: redistribute the
                 model's own class-sum (= 1 − P(void)) onto the target class,
                 leaving occupancy free.  "Whatever you decide is occupied, make
                 it class X."  Composes with the X_Blur blend.
    clamp_rows: optional (N,) bool — restrict either clamp to these voxels
                (partial spatial layouts); None = all voxels.
    """
    dev = diff.device
    cs, ce = 4, 4 + diff.n_classes
    steps = steps or max(2, int(round(t_start * diff.timesteps)) + 1)
    ts = torch.linspace(float(t_start), 0., steps, device=dev)

    def _rows(full, new):
        if clamp_rows is None:
            return new
        out = full.clone()
        out[clamp_rows] = new[clamp_rows]
        return out

    def _project_x0(x):
        # class-sum = 1 − P(void) is preserved, so the mask channel stays
        # consistent and occupancy is untouched.
        csum = x.jdata[:, cs:ce].sum(dim=1, keepdim=True)
        x.data.jdata[:, cs:ce] = _rows(x.jdata[:, cs:ce], clamp_onehot * csum)

    for i in range(steps - 1):
        time, time_next = ts[i:i+1], ts[i+1:i+2].clamp(min=0.)
        if clamp_onehot is not None and clamp_mode == "hard":
            # conditioning via the class channel of the noisy state
            a, s = log_snr_to_alpha_sigma(diff.log_snr(time))
            noised = a*clamp_onehot + s*torch.randn_like(clamp_onehot)
            noisy_grid.data.jdata[:, cs:ce] = _rows(noisy_grid.jdata[:, cs:ce], noised)
        x_start = diff.model(noisy_grid, time.repeat(len(noisy_grid.jidx)))
        diff._sigmoid_semantic_channels(x_start)           # softmax on your branch
        if clamp_onehot is not None and clamp_mode == "renorm":
            _project_x0(x_start)
        if time_next.item() == 0:
            if clamp_onehot is not None and clamp_mode == "hard":
                x_start.data.jdata[:, cs:ce] = _rows(x_start.jdata[:, cs:ce], clamp_onehot)
            return x_start
        # blend the x0 estimate toward the upsampler output (level>0 refinement)
        if X_Blur is not None:
            g = diff.blend_gamma(time_next)
            start_data = (1 - g[:, None]) * x_start.jdata + g[:, None] * X_Blur.jdata
        else:
            start_data = x_start.jdata
        ls, lsn = diff.log_snr(time), diff.log_snr(time_next)
        a, sg   = log_snr_to_alpha_sigma(ls)
        an, sgn = log_snr_to_alpha_sigma(lsn)
        c = -expm1(ls - lsn)
        mean = an * (noisy_grid.jdata * (1 - c) / a + c * start_data)
        var  = (sgn**2) * c
        noisy_grid.data.jdata = mean + (0.5*log(var)).exp() * torch.randn_like(noisy_grid.jdata)
    return noisy_grid
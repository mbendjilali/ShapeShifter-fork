"""Probe D1's single-step void prediction: P(void) on TRUE-void vs TRUE-occupied.

**On-schedule by construction.** At level>0 the conditioning IS the blend: both
training and inference noise `(1−γ)·GT + γ·X_Blur` with `γ = t·timesteps/max_T`
(`q_sample` / `blend_gamma`), so γ=1 at the top of the truncated schedule and γ=0
at t=0.  This probe therefore always passes X_Blur and lets q_sample apply γ.

The two rows differ only in *what the blend anchors to*:
  * ``sched``  — X_Blur = the real upsampler output.  Exactly the state the model
                 sees at inference; this is the number that describes the system.
  * ``oracle`` — X_Blur = the fine GT, i.e. a hypothetically perfect upsampler.
                 The ceiling for the void head; the gap to ``sched`` is what the
                 upsampler's quality costs.

Reading it:
  * both separate (P(void|void) ≫ P(void|occ)) ⇒ the void head is healthy.
  * ``sched`` ≪ ``oracle`` ⇒ the void head is limited by upsampler quality, not
    by its own training.
  * neither separates ⇒ the void head itself is broken (e.g. the pre-fix
    target-row-sum-2 bug pins the CE optimum at P(void)=0.5; see thesis A5).

HISTORY — do not restore the old reading rule.  This probe used to call q_sample
*without* X_Blur and compare "GT input" against "UP input".  With γ never applied,
its low-t "UP" row was a pure-upsampler state at high SNR, which occurs in neither
training nor inference, and its high-t "GT" row was likewise unreachable.  The
resulting "UP collapses ⇒ OOD void head" reading described a real bug — blend_gamma
was once `t/max_T`, ~timesteps× too small, so γ≈0 always — that has since been
fixed.  Post-fix, those numbers were an artefact of the probe.

    python test/diagnostics/d1_void_probe.py --level 1 --n_crops 12
    python test/diagnostics/d1_void_probe.py --ckpt checkpoints/diffusion_models/<run>_best.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from common import (
    get_device, list_crops, has_levels, load_levelN_inputs, level_resolutions,
)
from utils.diffusion_tensor import DiffusionTensor
from inference.inference import load_dales_diffusion


@torch.no_grad()
def probe(diff, crops, res1, res2, upsample_fac, ts, device):
    n_cls = diff.n_classes
    # accumulators: (source, t) -> [sum P(void|void), sum P(void|occ), sum occ_frac, n]
    acc = {}
    for cp in crops:
        if not has_levels(cp, res1, res2):
            continue
        X, X_UP, X0 = load_levelN_inputs(cp, res1, res2, upsample_fac, device)
        gt_void = X0.jdata[:, -1] <= 0
        if gt_void.all() or (~gt_void).all():
            continue
        up = DiffusionTensor.from_vdb(diff.model_upsampler(X, X_UP))

        # Always noise the GT target; vary only what the schedule blends toward.
        # q_sample applies γ = t·timesteps/max_T itself, so every state below is
        # one the model actually sees at that t.
        for src_name, blur in (("sched", up), ("oracle", X0)):
            for t in ts:
                times = torch.full((X0.jdata.shape[0],), float(t), device=device)
                noised = diff.q_sample(X0, times, blur)[0]
                pred = diff.model(noised, times)
                cat = torch.cat([pred.jdata[:, 4:4 + n_cls], pred.jdata[:, -1:]], dim=1)
                p_void = F.softmax(cat, dim=1)[:, n_cls]
                key = (src_name, t)
                a = acc.setdefault(key, [0.0, 0.0, 0.0, 0])
                a[0] += p_void[gt_void].mean().item()
                a[1] += p_void[~gt_void].mean().item()
                a[2] += (p_void < 0.5).float().mean().item()   # predicted-occupied fraction
                a[3] += 1
    return acc


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--src", default="checkpoints/diffusion_models/")
    p.add_argument("--split", default="test")
    p.add_argument("--n_crops", type=int, default=12)
    p.add_argument("--base_res", type=int, default=16)
    p.add_argument("--upsample_fac", type=int, default=2)
    p.add_argument("--ts", type=float, nargs="+", default=[0.02, 0.1, 0.3])
    p.add_argument("--ckpt", default=None,
                   help="Pin a specific checkpoint instead of resolving the latest "
                        "in --src. Required when comparing two named runs.")
    args = p.parse_args()
    device = get_device()

    res1, res2 = level_resolutions(args.level, args.base_res, args.upsample_fac)
    if args.ckpt:
        print(f"checkpoint (pinned): {args.ckpt}")
        diff = torch.load(args.ckpt, map_location=device, weights_only=False).to(device)
    else:
        diff = load_dales_diffusion(args.level, args.src)
    diff.eval()
    assert diff.model_upsampler is not None
    print(f"  max_T={diff.max_T}/{diff.timesteps}  coarse={res1}.pt → fine={res2}.pt")

    crops = list_crops(args.split, n=args.n_crops)
    acc = probe(diff, crops, res1, res2, args.upsample_fac, args.ts, device)

    print(f"\n{'='*70}")
    print("model's single-step P(void), averaged over crops — states are ON-SCHEDULE")
    print("  sched  = blend anchored on the real upsampler output (what inference sees)")
    print("  oracle = blend anchored on the fine GT (perfect-upsampler ceiling)")
    print(f"{'anchor':>7} {'t':>6} {'P(void|TRUE void)':>18} {'P(void|TRUE occ)':>17} "
          f"{'pred occ frac':>14}")
    for (src_name, t), (s_void, s_occ, s_frac, n) in sorted(acc.items()):
        if n == 0:
            continue
        print(f"{src_name:>6} {t:>6.2f} {s_void/n:>18.3f} {s_occ/n:>17.3f} {s_frac/n:>14.3f}")
    print(f"{'='*70}")
    print("healthy void head: P(void|TRUE void) ≫ P(void|TRUE occ); pred occ frac ≈ data")

    # Verdict from the measurement, not a standing claim: compare sched vs oracle
    # at the cleanest t, where the conditioning has the most influence on x0.
    t_lo = min(args.ts)
    o = acc.get(("oracle", t_lo)), acc.get(("sched", t_lo))
    if all(x and x[3] for x in o):
        p_or, p_sc = o[0][0] / o[0][3], o[1][0] / o[1][3]
        gap = p_or - p_sc
        if gap > 0.05:
            print(f"AT t={t_lo}: sched {p_sc:.3f} < oracle {p_or:.3f} (Δ={gap:+.3f}) ⇒ the void "
                  f"head IS limited by upsampler quality here.")
        else:
            print(f"AT t={t_lo}: sched {p_sc:.3f} ≈ oracle {p_or:.3f} (Δ={gap:+.3f}) ⇒ upsampler "
                  f"quality is NOT the limiting factor; any deficit is the void head's own.")


if __name__ == "__main__":
    main()

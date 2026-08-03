"""A5 (thesis Ch.4): single-variable ablation of the flood fix, as curves.

Plots the two level-1 runs that differ **only** in `zero_empty_target`:

    baseline  dales_1_08-07-10:06   zero_empty_target: true   (the fix)
    ablation  dales_1_27-07-12:11   zero_empty_target: false  (the bug)

Same upsampler (`dales_1_08-07-09:48`), same n_classes=8, same config otherwise,
so the gap between the curves is attributable to the one flag.  Reads the
TensorBoard event files directly — no checkpoint, no GPU — and writes a two-panel
figure plus the numbers quoted in the chapter.

Left panel is the categorical CE ("BCE" in the logs), which carries occupancy and
semantics: without the fix `fill_upsampled_with_gt` leaves trilinear class mass on
empty fine voxels, the target row sums to 2 instead of 1, and the CE optimum is
P(void)=0.5 — a floor the loss cannot go below, which is the flood.  Right panel
is validation occupancy IoU, which *degrades* over training in the ablated run.

    python test/evaluation/a5_ablation_curves.py
    python test/evaluation/a5_ablation_curves.py --out output/tests/A5
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

BASELINE = "runs/diffusion_level_1_08-07-10:06"   # zero_empty_target: true
ABLATION = "runs/diffusion_level_1_27-07-12:11"   # zero_empty_target: false

# `add_scalars` writes each series to its own sub-directory of the run.
CE_VAL   = "Loss_Val_Val_BCE"
IOU_VAL  = "OccIoU_val"
CE_TRAIN = "Loss_train_BCE"


def load_run(run_dir):
    """{series_name: [(step, value), ...]} for every scalar sub-run under run_dir."""
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run directory not found: {run_dir}")
    series = {}
    for name in sorted(os.listdir(run_dir)):
        sub = os.path.join(run_dir, name)
        if not os.path.isdir(sub):
            continue
        ea = event_accumulator.EventAccumulator(sub, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            series[name] = [(s.step, s.value) for s in ea.Scalars(tag)]
    return series


def get(series, name, run_dir):
    if name not in series:
        raise SystemExit(f"series '{name}' missing from {run_dir}")
    return zip(*series[name])  # steps, values


def at_step(series, name, step):
    """Value logged at `step`, or None if that epoch was not logged (val_every=2)."""
    for s, v in series.get(name, []):
        if s == step:
            return v
    return None


def panel(ax, base, abl, name, title, ylabel, lo_is_better):
    for series, label, style in ((base, "zero_empty_target: true  (fix)", "-"),
                                 (abl,  "zero_empty_target: false (bug)", "--")):
        steps, values = get(series, name, "run")
        ax.plot(steps, values, style, marker="o", ms=3, lw=1.6, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="upper right" if lo_is_better else "lower right")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", default=BASELINE)
    p.add_argument("--ablation", default=ABLATION)
    p.add_argument("--out", default="output/tests/A5")
    p.add_argument("--name", default="a5_zero_empty_target")
    args = p.parse_args()

    print(f"baseline: {args.baseline}")
    base = load_run(args.baseline)
    print(f"ablation: {args.ablation}")
    abl = load_run(args.ablation)

    # The ablation is 25 epochs by design; compare like for like at its last
    # logged validation epoch as well as at each run's own best.
    last_abl = max(s for s, _ in abl[CE_VAL])
    rows = []
    for label, series in (("fix (true) ", base), ("bug (false)", abl)):
        ce = dict(series[CE_VAL])
        iou = dict(series[IOU_VAL])
        rows.append((label,
                     at_step(series, CE_VAL, last_abl),
                     min(ce.values()),
                     at_step(series, IOU_VAL, last_abl),
                     max(iou.values()),
                     list(iou.values())[-1]))

    print(f"\n  level 1, single variable = zero_empty_target   (epoch {last_abl} = "
          f"last validated epoch of the ablation)\n")
    print(f"  {'run':12s} {'Val CE @ep':>11s} {'Val CE best':>12s} "
          f"{'Val IoU @ep':>12s} {'Val IoU best':>13s} {'Val IoU last':>13s}")
    for r in rows:
        cells = " ".join("      n/a" if v is None else f"{v:>12.4f}" for v in r[1:])
        print(f"  {r[0]:12s} {cells}")
    print("\n  The CE gap is the flood signature: with the bug the categorical loss is"
          "\n  pinned near its P(void)=0.5 floor (target rows sum to 2), and occupancy"
          "\n  IoU degrades over training instead of improving.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    panel(axes[0], base, abl, CE_VAL,
          "Validation categorical CE (occupancy + semantics)", "CE", lo_is_better=True)
    panel(axes[1], base, abl, IOU_VAL,
          "Validation occupancy IoU", "IoU", lo_is_better=False)
    fig.suptitle("A5 — level-1 flood fix, single-variable ablation of "
                 "`zero_empty_target`", fontsize=12)
    fig.tight_layout()

    os.makedirs(args.out, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(args.out, f"{args.name}.{ext}")
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()

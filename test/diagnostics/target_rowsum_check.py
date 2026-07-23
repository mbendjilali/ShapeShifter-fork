"""Phase 0 #2 — does the level>0 categorical target sum to 1?

The void-categorical CE assumes each target row is a distribution (sum 1).  At
level>0, X0 comes from fill_upsampled_with_gt, which keeps the trilinear class
probs on non-GT (empty) fine voxels and only sets mask=-1.  So the CE target for
empty voxels is [trilinear_class (≈sum 1), void=1] → row sum ≈ 2, which breaks the
categorical (optimum P(void)=0.5 → decoded mask≈0 → floods at the pruning
threshold).  This checks the row sums, split by occupied/empty.

    python test/diagnostics/target_rowsum_check.py --level 1 --n_crops 8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (
    get_device, list_crops, has_levels, load_levelN_inputs, level_resolutions, N_CLS,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--split", default="train")
    p.add_argument("--n_crops", type=int, default=8)
    p.add_argument("--base_res", type=int, default=16)
    p.add_argument("--upsample_fac", type=int, default=2)
    args = p.parse_args()
    device = get_device()
    res1, res2 = level_resolutions(args.level, args.base_res, args.upsample_fac)

    occ_sum = occ_n = emp_sum = emp_n = 0.0
    emp_classmass = 0.0
    for cp in list_crops(args.split, n=args.n_crops):
        if not has_levels(cp, res1, res2):
            continue
        _, _, X0 = load_levelN_inputs(cp, res1, res2, args.upsample_fac, device)
        occ = X0.jdata[:, -1] > 0
        class_mass = X0.jdata[:, 4:4 + N_CLS].sum(-1)          # Σ class target
        void_t = (~occ).float()
        row_sum = class_mass + void_t                          # cat_target row sum
        occ_sum += row_sum[occ].sum().item();  occ_n += int(occ.sum())
        emp_sum += row_sum[~occ].sum().item();  emp_n += int((~occ).sum())
        emp_classmass += class_mass[~occ].sum().item()

    print(f"\n{'='*56}")
    print(f"level {args.level}  cat_target row sum (should be 1.0 for a valid categorical)")
    print(f"  occupied voxels : mean row sum = {occ_sum/max(occ_n,1):.3f}   (n={occ_n})")
    print(f"  empty voxels    : mean row sum = {emp_sum/max(emp_n,1):.3f}   (n={emp_n})")
    print(f"  empty voxels    : mean CLASS mass = {emp_classmass/max(emp_n,1):.3f}"
          f"   (should be 0 — trilinear leftovers if ≈1)")
    print(f"{'='*56}")


if __name__ == "__main__":
    main()

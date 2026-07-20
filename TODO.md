# TODO — D1 occupancy flood

**✅ RESOLVED (2026-07-02).** Root cause: `fill_upsampled_with_gt` left trilinear
class mass on empty fine voxels → void-categorical target summed to 2
(`test/diagnostics/target_rowsum_check.py`: empty rowsum 2.000) → CE optimum P(void)=0.5 →
decoded mask≈0 → flood.  Fix: `zero_empty_target: true` (dataset zeros empty
voxels → one-hot void, sum 1) + `ijk_to_index` guard in `fill_upsampled_with_gt`.
After retrain (dales_1 02-07-11:09, void_weight 1.9): BCE floor 0.9→0.15; Val
IoU/σ bin0→1.00; `d1_void_probe` P(void|void) 0.5→**0.995** (GT); occFrac diff
0.93→**0.595** (≈gt 0.55); Δ occIoU −0.16→**−0.005** (on par, no longer degrades).
D1 ≈ upsampler on IoU/acc is expected — a sampler can't beat the mean-regressor on
those; its value is crispness, judged visually.

**NEW ISSUE — object fragmentation.** D1 output is crisp but objects are
*fragments*: jagged partial roofs / vegetation, no complete buildings or trees —
the global-coherence limit of unconditional small-data generation.

---

**Original context.** D0: believable coarse layouts. D1 (`configs/training/diffusion_up.yaml`): floods occupancy — after `remove_mask(threshold=0.0)` nearly the whole subdivided grid survives ("lego blocks").

**Scope.** Do not retrain D0. Use DALES WITH vegetation.

**Hypothesis (verify Phase 0 before code changes).** Level>0 clean target breaks void-categorical semantics D0 relies on.

- **L0:** `to_custom_dense(empty_fill='zero')` → empty voxels all-zero → CE target one-hot void `[0×8, 1]`.
- **L>0:** target `X0` from `fill_upsampled_with_gt` (`diffusion_tensor.py` ~L68–76) keeps trilinear class probs on non-GT fine voxels; only mask = −1.
- **`SparseDiffusion.forward`** (`fvdb_diffusion.py` L355–366): `cat_target = [target_label(8), void_target(1)]`. Empty fine voxels: row sum ≈ **2** (class ≈ 1 + void = 1) → CE optimum `P(void)=0.5` → decoded mask `2·class_sum−1 ≈ 0` (pruning threshold).
- Ground-dominated empties: `1.0 vs 1.0` tie ground/void; `argmax` → ground, occupied. Optimum floods — not calibration/`void_weight`.

`empty_fill: zero` in `diffusion_up.yaml` is unused at L>0 (`load_crop_levelN` only; flag consumed by `load_crop_level0`).

---

## Phase 0 — Confirm (no training)

1. **TensorBoard** `runs/diffusion_level_1_*` → `OccIoU_per_sigma/train|val`.
   - Hypothesis holds: **bin0 IoU poor/capped ≪1** (near-zero σ should reproduce occupancy trivially).
   - If bin0 ≈ perfect, only noisy bins bad → different diagnosis.

2. **Row-sum defect.** Script/cell: `DALESDataset.load_crop_levelN(train, level=1)`, rebuild `cat_target` as `forward`, row sums by `occ_target_hard = (X0.jdata[:, -1] > 0)`.
   - Expected: occupied ≈ 1.0; empty ≈ 2.0.

3. **Model check.** D1 ckpt, training batch, small `t≈0.01`: `q_sample` → `model` → `_sigmoid_semantic_channels`; mean decoded mask on empty vs occupied targets.
   - Expected: occupied ≈ +1; empty ≈ **0.0** (not −1).

## Phase 1 — Fix target semantics

4. **`zero_empty_target: bool = False`** on `DALESDataset`; in `load_crop_levelN` after `fill_upsampled_with_gt`: zero all channels where mask < 0, re-set mask = −1. Enable in `diffusion_up.yaml`; thread through `_train_dales` (`diffusion.py` L117–132). Val uses same dataset → covered.
   - Empty CE → one-hot void (sum 1, mask → −1); geometry MSE → 0. Matches L0 + inference pruning.
   - **Do not** change `X_UP` or retrain upsampler (`upsampler.py` shares `load_crop_levelN` — default False, diffusion-only). Upsampler anchor unchanged train/inference.

5. **Guard `ijk_to_index`** in `fill_upsampled_with_gt`: −1 for fine GT absent from subdivided coarse grid silently overwrites last row. `assert (to_change_idx >= 0).all()` or filter+warn+count.

6. Re-run Phase 0 #2 with flag on: all `cat_target` rows sum to 1.

## Phase 2 — Retrain & evaluate

7. **Retrain L1:** `python src/train/diffusion.py -level 1 -config configs/training/diffusion_up.yaml -dataset dales`
   - Watch `OccIoU_per_sigma` bin0 → ~0.95+ in few epochs. Noisy bins staying poor is OK.

8. **Inference:** `inference.py -levels 1 -batch_size 4 -total_num 4`; then `test/evaluation/distribution_stats.py --n_gen 8 --max_level 1` and `test/evaluation/upsampler_vs_diffusion.py`.
   - Accept: fine occupancy not ≈100%; D1 occupied fraction ≈ data (ratio = occupied fine GT ÷ subdivided coarse GT voxels — **not** `distribution_stats.py` dense `d_frac` at res 32, which is crop bbox).
   - Visual: LAZ L1 refined layout, not solidified.

9. **Only if** structure OK but kept fraction off: sweep `-occ_threshold` (extend `diagnose_occupancy_dales`, L0-only today, `inference.py` L256+). If mismatch survives thresholding, then tune `void_weight` in small steps (1.0→1.5/0.75, one retrain each). **Not before Phase 1** — can't fix optimum mask=0.

## Phase 3 — Stretch

10. **D2** same config family; same checks.
11. **LAZ bug** `save_dales_pc` (`inference.py` ~L248–250): `jdata[:, 3:-1]` is 9 cols `[intensity, class×8]`; `features_np[:, 1]` exports P(ground). Use `[:, 0]`. Re-export before judging intensity.
12. **Housekeeping:** remove/comment unused `empty_fill: zero` in `diffusion_up.yaml`; document `zero_empty_target` in both training configs.

## Ground rules

- One change per retrain; log date, commit, config diff, bin0 IoU, occupancy fraction, one LAZ screenshot.
- Commit Phase 0 scripts under `test/`.
- Phase 0 contradicts hypothesis (bin0 IoU high) → skip Phase 1; document + threshold/`void_weight` route (Phase 2 #9) while awaiting feedback.

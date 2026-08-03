# TODO — ShapeShifter-fork

Scope: everything needed to support Chapter 4 of the thesis. Items marked
**FREEZE** need a GPU and must be done before 2026-08-22. Everything else is
hygiene and can happen any time before the repository URL appears in the
manuscript.

---

## Phase 1 — Artefacts for the chapter (FREEZE)

**Status (28-07-2026).** Training/eval prerequisites are settled; the figures
themselves (A1–A4 renders) are the remaining work.
- Level-1 checkpoint of record: **`dales_1_08-07-10:06`**, now *verified* — loaded
  by explicit path it reproduces its logged Val OccIoU (0.924), so the on-disk
  file is provably that trained model, not just a same-named copy.
- **A5 (flood-fix ablation): DONE** — deliberate single-variable ablation in hand.
- **A0 (new): training-stability bug found and fixed** — see the block after A6.
  A 50-epoch level-1 retrain diverged and crashed; root-caused to unbounded Adam
  weight-drift under a constant LR and fixed with cosine LR decay
  (`lr_schedule: cosine` now in `configs/training/diffusion_up.yaml`). A clean
  cosine-trained replacement exists (`dales_1_28-07-11:05`, best Val 0.140).
- **A1, A2, A3, A4, A6: still open** (figures/eval + the level-3 decision).

**A1. D1 output beside ground truth.** **FREEZE, highest priority.**
One render of a level-1 sample next to the ground-truth crop of the same tile,
same viewpoint, same colour map. The chapter's claim about level 1 is crispness,
judged visually, and this is the only evidence for it. If nothing else on this
list gets done, do this.

**A2. D0 unconditional layout renders.** **FREEZE.**
Two or three coarse layouts from `test/generation/` at level 0. Supports the
claim that the coarse level produces believable layouts unconditionally.

**A3. Probe A and B renders.** **FREEZE.**
`test/generation/reconstruct.py` (partial-noise reconstruction, succeeds) and
`test/generation/class_generation.py` (class-clamped generation from pure noise,
~~fails with full occupancy~~). Side by side, same class. This pair is the
chapter's controlled negative result and deserves a figure rather than a sentence.
→ **"Full occupancy" is wrong — do not write it (measured 03-08-2026).** Counting
the exported `output/tests/A3/*.laz` against the 32×32×7 = 7168-voxel canonical
grid: Ground **37.4 %**, PowerLines **44.7 %**, Buildings **59.4 %**. For scale,
the four A2 unconditional layouts are 34–45 % and twelve real test crops
(`16.pt`) are 22–56 %, mean 32 %. Nothing is near 100 %. (Threshold parity was
checked and is a non-issue: `class_generation.py` pruned at 0.5 and
`a2_d0_layouts.py` at 0.0, but re-running the clamp at 0.0 moves the numbers by
<0.2 pp — the decoded mask is bimodal, so almost no voxel lies between the two
cuts. A `--occ_threshold` flag now makes the cut explicit, default 0.0.)
→ **CLASS LABEL BUG — fixed 03-08-2026, affects everything below.**
`test/common.py` `CLASS_NAMES` had indices **4 and 5 swapped** relative to
`configs/dataset/dales.yaml` (sem 5 = PowerLines, 6 = Fences; `encode_features`
one-hots `sem − 1`). Every printed table, every `.laz` filename and every
statement of the form "the PowerLines row" produced before that date names
**Fences**, and vice versa. *Computation was never affected* — `--class_ids` and
the clamp index the channel directly — so no result needs recomputing, only
relabelling. `A3_renorm/clamp_class5_PowerLines_renorm.laz` renamed to
`…_Fences_renorm.laz`. `distribution_stats.py` / `threshold_sweep.py` printed
the same swap.
→ **What the probe actually shows.** Clamping is *not* inert: median height moves
in the right semantic order — Ground 4.8 m, **Fences** 8.0 m, Buildings 11.2 m
(p90: 11.2 / 14.4 / 17.6) — and the unconditional layout matches the Ground
clamp, as it should for ground-dominated tiles. So class-conditional *height*
priors exist. What fails is **structure and sparsity**: the clamped scene fills
~45 % of the volume whatever the class. Write the negative result that way, and
note that **no powerline clamp was ever run** — class 5 was Fences.
→ **Two caveats the chapter must not step on.** (a) The clamp is *spatially
uniform* — it asserts "every voxel is class X" over the whole grid. Since
occupancy is encoded as class-sum 1 in the same (n_cls+1) softmax
(`fvdb_diffusion.py:114-127`, mask = 2·class-sum − 1), clamping all class
channels also asserts *occupied everywhere*; the model's only way to say "empty"
is one void logit fighting eight clamped ones. The probe partly over-determines
what it measures. (b) The exported class labels are tautological — the last
reverse step (`helper.py:20-21`) overwrites the class channels with the hard
one-hot, so every point carries the clamped label by construction. Geometry is
the only observable in the figure.
→ **DONE (03-08-2026): occupancy-preserving clamp + real class layouts, no
retraining.** `reverse_from` now takes `clamp_mode` (`hard` = the original
probe; `renorm` = project the *x0 estimate*: redistribute the model's own
class-sum = 1−P(void) onto the target class, leaving occupancy free) and
`clamp_rows` for partial spatial layouts. `class_generation.py --clamp_mode
renorm` gives the uniform case; new `a3_layout_generation.py` takes a real
crop's level-0 GT, reduces it to a 2D footprint (majority class per (i,j)
column, z free), clamps via renorm, and prints per-class geometry adherence
vs the same crop's GT (`output/tests/A3_renorm`, `output/tests/A3_layout`).
- *Uniform renorm:* occupancy **25.6 / 25.6 / 31.5 %** (Ground / PowerLines /
  Buildings) — at or below the real-crop mean of 32 %. The hard probe's excess
  occupancy (37/45/59 %) was therefore **the probe's artefact**, confirming
  caveat (a): the model expresses emptiness fine when allowed to. Height order
  survives but flattens (p50: 4.8 / 4.8 / 8.0 m).
- *Real layouts* (`5080_54400_x0000_y0100` urban, `x0000_y0400` sparse): the
  model builds in 93–100 % of layout columns and keeps overall occupancy
  plausible (23.9 / 24.3 % of grid). But vertical structure is only weakly
  class-differentiated — every class regresses toward a generic ~2–3-cells,
  max-z ≈ 2–4 column.
- *Column reduction matters when reading that table (`--reduce`).* `majority`
  counts voxels, so a ground+tree column scores as Vegetation and the rarer
  classes never win a column. `nonground` labels a column by what stands *on*
  it (fallback Ground) and surfaces PowerLines / Cars / Trucks / Poles columns.
  Under `nonground` the collapse is unmistakable — **GT cells/col spans
  1.05 → 5.32 across classes, generated spans only 2.41 → 3.84**: bare ground
  gets 2.4 cells where GT has 1.0 (max-z 0.10 → 2.77, i.e. structure invented
  on empty ground), vegetation 5.32 → 3.84. The model reproduces the footprint
  and the occupancy budget but not the class's vertical signature.
- *Unclamped columns (`--layout_mask half`, urban crop, 512 free).* The free
  half is **not** left empty: the model builds in **100 %** of free columns at
  **4.77 cells/col, mean max-z 5.02**, choosing Vegetation 80 % / Ground 16 % —
  against GT 3.56 cells/col and max-z 2.83 there. So in unconstrained space it
  *over*-builds (and over-vegetates, consistent with the known level-0 class
  skew), while in clamped columns it *under*-builds relative to GT (2–3
  cells/col). Worth stating plainly: the renorm projection preserves class-sum
  at the moment of projection, but a pure single-class column is off-manifold
  for the model, and it answers with lower occupancy at the next step. Real
  DALES layouts are always fully defined (ground returns everywhere), so
  `--layout_mask` is the only way to reach this regime.
- *Refined A3 story for the chapter:* the controlled negative result is **not**
  "class conditioning floods occupancy" — it is "under occupancy-preserving
  class conditioning, the model respects the footprint and the occupancy
  budget, but produces only weakly class-specific vertical structure". The
  probe-A/probe-B contrast (reconstruction succeeds, generation from noise
  regresses to the generic column) still stands, with the mechanism now
  correctly attributed. Real learned conditioning remains X3.
- *Artefacts kept* (`output/`, gitignored, all regenerable from the committed
  scripts at `--seed 0`): `A3/recon_t{0.30,0.60,1.00}.laz` (probe A),
  `A3_renorm/` (uniform renorm clamp), `A3_layout/` (per-crop GT plus the
  `majority_none`, `nonground_none` and `majority_half` variants).
  **Deleted 03-08-2026 as superseded:** `A3/clamp_class*.laz` (hard-clamp probe
  behind the retracted "full occupancy" claim — numbers preserved above, and
  `class_generation.py --clamp_mode hard` reproduces the files in ~27 s each);
  `A3_thr0/` (threshold-parity check, conclusion recorded above); and the
  first-pass `A3_layout/*_s0__*.laz` written before the variant-tagged naming.

→ **How to write A3 (drafting note, 03-08-2026).** Visual verdict from
CloudCompare: the layout scenes are ground plus unstructured vegetation-like
blobs, whatever the clamped class. That is the **expected** outcome and should be
presented as a designed boundary measurement, not as a defect. Three things to
name, in this order, and the order matters:
1. *It is guidance, not conditioning.* Clamping a channel subset each reverse
   step is inpainting-by-replacement. The constraint is imposed on the **state**;
   the denoiser was never trained to agree with it, so it is partly washed out
   each step and never propagates coherently into the unconstrained dimensions
   (here, geometry). The only pathway from "class = building" to geometry is
   whatever class→geometry coupling the *unconditional* model already encodes.
   The experiment's ceiling was therefore always "extract what the unconditional
   model knows", never "perform conditional generation". Do not write, and do not
   let a reader infer, "diffusion cannot be conditioned on layouts" — that
   hypothesis was not tested. Guidance was.
2. *What it knows is marginals, not the joint — the small-data reality.* ~600
   crops of one city, 32×32×7, <1 M parameters. All three probes agree: hard
   clamp moves *height distributions* in the right order; renorm keeps the
   *occupancy budget* right; layout runs show the *vertical spread collapsing*
   (GT 1.05→5.32 cells/col across classes, generated 2.41→3.84). Marginal class
   priors: present. Joint class–geometry structure: barely. This is the **same**
   global-coherence limit as the unconditional D1 "object fragmentation" note in
   `TODO.md` (crisp but no complete buildings), reached from the other direction.
   Two independent probes converging is a strength of the chapter — say so.
3. *Part of the ugliness is the probe, and part is the level.* Pure single-class
   columns are off-manifold for a model trained on mixed ground+canopy columns,
   which is why clamped regions under-build while free regions over-build (the
   `--layout_mask half` numbers above). And at 3.2 m voxels a **GT** building is
   already only 2–3 stacked cubes: level 0 is where geometry is least expressive
   by construction. "Does a building look like a building" is a level-1 question,
   and the chapter's crispness claim already lives at level 1 (A1). Do not judge
   a level-0 probe by a level-1 standard.

The sentence this licenses — and the one to use — is: *inference-time guidance on
an unconditional model recovers the marginal class priors (height, occupancy
budget) but not joint class–geometry structure; conditioning therefore has to
enter at training time.* That makes A3 the experiment which motivates the `cond`
hook and the GVAE interface (X3) rather than a result to apologise for.

→ **Class height census (03-08-2026, all 1000 crops, level 0, CPU only).**
Height above each crop's median ground voxel, at 3.2 m:

| class | p50 | p90 | p50 (m) | p90 (m) | share |
|---|---|---|---|---|---|
| Ground | 0 | 1 | 0.0 | 3.2 | 35.9 % |
| Vegetation | 3 | 7 | 9.6 | 22.4 | 42.3 % |
| Buildings | 2 | 3 | 6.4 | 9.6 | 18.1 % |
| **PowerLines** | **4** | **6** | **12.8** | **19.2** | 1.8 % (406/1000 crops) |
| **Fences** | **1** | **2** | **3.2** | **6.4** | 0.7 % (779/1000 crops) |
| Poles | 3 | 5 | 9.6 | 16.0 | 0.3 % |
| Cars / Trucks | 1 | 1 | 3.2 | 3.2 | 0.7 / 0.2 % |

This settles the quantisation-vs-failure question **without a rerun**:
PowerLines sit 4–6 voxels clear of the ground and are therefore fully
resolvable at 3.2 m — flattened cables would be a real failure. Fences sit at
1–2 voxels, i.e. *at* the quantisation limit — flattened fences are not a
finding. The two must not share a caption sentence.
→ **But the structural point supersedes both.** A powerline column is ground at
0 voxels *and* cable at 4–6 voxels. `a3_layout_generation.py` asserts **one**
class over every voxel of a column (`col_cls = layout[i,j]`, clamped from
ground to grid top), so neither reduce rule can express that: `nonground`
relabels the ground return as PowerLines, `majority` lets the ground-dominated
column swallow the cable entirely. **A 2D single-class footprint is structurally
incapable of specifying an aerial LiDAR column**, and the census above turns
that from an assertion into a measurement. Any flattening seen under this
conditioning is at least partly the signal's inadequacy, not the model's.
This is the chapter's hinge: it is the strongest available motivation for a
*height-aware* conditioning signal, it applies to plan-view/BEV conditioners of
the Control-3D-Scene kind, and it is the cleanest reason to reach for a 3D
latent — §4.7 ends "a plan-view signal cannot specify this data", §4.8 opens
"so we tried to produce a 3D one from a graph". Scope the claim to *single-class*
plan-view signals: a multi-channel BEV with height bins can express part of it,
and a reader who works on BEV conditioning will know that.
→ **Caveat before quoting any powerline result:** neither A3 layout crop
(`x0000_y0100`, `x0000_y0400`) contains a single PowerLines column, and the
per-column `max-z` the script prints is over *all* classes in the column, not
per class — so the existing stdout cannot answer a per-class height question
even with the labels fixed. A powerline-bearing crop plus a per-class height
column would be needed; the census above is the cheaper substitute and is
probably sufficient.
→ **Do not extend A3 before the freeze.** A RePaint-resampling or
reconstruction-guidance (DPS-style) variant would move the collapse numbers
somewhat but cannot change the conclusion — trained conditioning would still be
the answer — and A1 remains the one artefact the chapter cannot do without. The
renders in hand (GT beside layout-gen, same crop, same viewpoint) *are* the
figure: the "ground and blobs" reading is the evidence, shown rather than
asserted.

**A5. Single-variable ablation of the flood fix.** **FREEZE, optional but
worth one short run.**
Retrain level 1 at current HEAD with `zero_empty_target: false`, 25 epochs, all
else unchanged. Right now the before/after comparison uses
`diffusion_level_1_02-07-10:39` against `diffusion_level_1_08-07-10:06`, and
those two runs also differ in `n_classes` (changed 8 → 7 at `ddc0935`). One run
turns an accidental ablation into a deliberate one.
→ **DONE (27-07-2026).** Ablation run **`dales_1_27-07-12:11`**: level 1,
`zero_empty_target: false`, 25 epochs, `n_classes=8`, and — for a true
single-variable contrast — the **same** level-1 upsampler as the baseline
(`dales_1_08-07-09:48`, force-selected past `resolve_latest`). Baseline is the
verified `dales_1_08-07-10:06` (`zero_empty_target: true`). Result is the
predicted flood signature: **Val BCE ≈ 0.14 (fixed) → ≈ 1.14 (ablated)** — the
categorical CE pinned near the P(void)=0.5 floor from the target-sums-to-2 bug —
with OccIoU degrading over training. The only difference between the two runs is
`zero_empty_target`, so the before/after is now deliberate, not accidental.
→ **Figure exported (29-07-2026).** `test/evaluation/a5_ablation_curves.py`
reads both runs' TensorBoard scalars and writes
`output/tests/A5/a5_zero_empty_target.{png,pdf}`: two panels, validation
categorical CE and validation occupancy IoU, fix vs bug. No checkpoint and no
GPU — it re-derives the numbers from the logs, so the figure is reproducible
from the repository alone. Numbers it prints, for the chapter:

| | Val CE @ep24 | Val CE best | Val IoU @ep24 | Val IoU best | Val IoU last |
|---|---|---|---|---|---|
| `zero_empty_target: true` (fix) | 0.265 | **0.142** | 0.885 | 0.948 | 0.940 |
| `zero_empty_target: false` (bug) | **1.185** | 1.083 | 0.869 | 0.927 | 0.869 |

Epoch 24 is the ablation's last validated epoch (it runs 25, the baseline 50), so
the `@ep24` column is the like-for-like comparison and the `best` column each
run's own optimum. Read the CE panel, not the IoU panel: the ablated CE flattens
at ≈1.1 by epoch 4 and then drifts *upward*, which is the floor, whereas the
fixed run keeps descending to ≈0.15. The ablated IoU is briefly *above* the
baseline (0.927 at epoch 6) before decaying to 0.869 while the baseline climbs to
0.948 — worth one sentence in the caption so the early crossing is not misread.

**A6. Decide the level-3 story.** **FREEZE if retrying.**
`diffusion_level_3_10-07-10:34` produced one logged point and stopped. Either
retry once and report the outcome, or record in the chapter that level 3 was
attempted and did not complete. Both are fine; silence is not.
→ ~~Likely the same crash as A0~~ — **wrong hypothesis, disproved by the retry.**
The failure is memory, not the `scaler.update` NaN crash.
- *Retry 1 (`29-07-09:36`, HEAD, cosine LR + crash guard):* died 4.5 min in,
  inside the **first** epoch, with `CUDA error 2: out of memory` from
  `nanovdb/tools/cuda/PointsToGrid.cuh:406` — fVDB grid construction, nowhere
  near the loss or the scaler. It had been sitting at 10 964 MiB of the 2080 Ti's
  11 264 MiB. All three GPUs on this machine are 11 GB, so there is no larger
  card to move to. This also re-reads the 10-07 history: five of those six runs
  logged *nothing at all*, and only `10-07-10:34` reached one epoch — an
  out-of-memory pattern, not a divergence pattern.
- *Retry 2 (`03-08-08:26`), the fix:* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  Peak drops 10 964 → **9 334 MiB** and the run clears epoch 0 (train BCE 0.504,
  MSE 0.0162; Val BCE 0.729, MSE 0.0674; Val OccIoU 0.788 — consistent with the
  lone `10-07` point, Val BCE 0.810 / OccIoU 0.804). So level 3 was never
  untrainable: at 97 % occupancy the default allocator fragments and one large
  crop kills it. **Level 3 needs that env var on 11 GB hardware** — this belongs
  in the chapter and in the run instructions, not in a "did not complete" line.
- *Cost:* **538 s/epoch → ≈7 h 20 for 50 epochs.** Run in progress.

**A0. Level-1 training stability — bug found and fixed (28-07-2026).**
Discovered while producing a fresh level-1 checkpoint: a 50-epoch retrain
(`zero_empty_target: true`) **diverged and crashed** at epoch 46.
- *Symptom:* `AssertionError: No inf checks were recorded prior to update` from
  `GradScaler.update()`. Immediate cause: `scaler.update()` was called even when a
  whole accumulation step was NaN-skipped (no `scaler.step()`), so it fired
  without inf-checks. Fixed by guarding `update()` behind the step (all levels).
- *Root cause (the real one):* the crash was the tail of a slow divergence. Once
  the geometry MSE saturates (~epoch 5), the loss is near-flat and Adam keeps
  taking ~`lr`-sized steps, so the weights random-walk **outward without bound**
  (tracked a new per-epoch `|W|`: **28 → 50+, never plateauing**), which steadily
  degrades the occupancy/semantic head (train BCE 0.21 → 0.48 → NaN) and finally
  overflows. Constant LR + no effective regularisation = seed/condition-dependent
  divergence (which is why the original `08-07` run survived and this one didn't).
- *Fix, proven by a controlled 4-run experiment (identical except one knob):*
  **cosine LR decay → 0** arrests it — `|W|` plateaus (~43), BCE recovers, 50
  epochs complete with no crash, best Val 0.140 = baseline. `weight_decay` up to
  0.1 does **not** help (`lr·wd` far too small to counter the growth) — ruled out.
  `lr_schedule: cosine` is now the default in `configs/training/diffusion_up.yaml`.
- *Also hardened in `src/train/diffusion.py`:* NaN-skipped steps are counted,
  warned, and auto-abort a fully-diverged epoch; epoch-average losses now divide
  by contributing steps only (previously skipped steps diluted them toward 0 and
  hid the divergence in the logs).
- *Chapter relevance:* the earlier before/after level-1 curves were trained under
  the unstable constant-LR regime. Any level>0 result quoted from a long run
  should use a cosine-LR rerun, and the manuscript must not read the constant-LR
  degradation as a property of the method. Clean replacement: `dales_1_28-07-11:05`.

---

## Phase 2 — Config and code hygiene — **DONE (23-07-2026)**

**H1. Stale level table.**
`configs/dataset/dales.yaml` still lists `voxel_size_levels` as 256:0.4,
128:0.8, 64:1.6, which contradicts `configs/encoding/dales.yaml`
(`voxel_size_initial: 0.2`, `initial_size: 256`, `targets: [128, 64, 32, 16]`).
Delete whichever is not consumed. The manuscript quotes 0.2 m to 3.2 m across
five levels and a reader will check.
→ **DONE.** *Neither* copy was consumed (no `.py` reads `voxel_size_levels`), so
both were deleted. `configs/encoding/dales.yaml` is now the single definition of
the level table: 256@0.2 · 128@0.4 · 64@0.8 · 32@1.6 · 16@3.2, which is the
manuscript's 0.2 m → 3.2 m over five levels.

**H2. Inert `min_snr_gamma`.**
Both `diffusion_0.yaml` and `diffusion_up.yaml` carry `min_snr_gamma: 5.0` while
`loss_weighting: p2`. It does nothing. Remove it, or comment it as unused.
Anyone reading the config will otherwise assume both weightings are active, and
P2 and Min-SNR emphasise opposite ends of the noise range on an x0 head.
→ **DONE, with a correction to the premise:** at HEAD `diffusion_0.yaml` is
`loss_weighting: min_snr` (changed after this item was written), so there the
*inert* knobs are `p2_k`/`p2_gamma`, not `min_snr_gamma`. Only `diffusion_up.yaml`
is P2. Each file now keeps the live knob and comments out the dead one, under a
note that exactly one weighting is ever active. The commented-out values equal
the `cfg.get` defaults in `src/train/diffusion.py`, so behaviour is unchanged.

**H3. Document the scope of the P2 weighting.**
The weighting multiplies the geometry MSE only (channels 0:4, offset and
intensity). The (n_cls+1)-way categorical carrying occupancy and semantics is
unweighted in t. Add one comment at `src/utils/fvdb_diffusion.py` where the
weight is applied. This is load-bearing for the chapter, which must not credit
P2 with the occupancy improvements.
→ **DONE.** Comment at the weighting site in `src/utils/fvdb_diffusion.py`
(`SCOPE OF THE WEIGHTING`), stating that w(t) touches channels 0:4 only and that
the occupancy/semantic levers are `void_weight`, `class_weight` and
`zero_empty_target`. A matching one-liner sits on the categorical loss. The
ε-vs-x0 currency derivation (H5) is folded in underneath it.

**H4. `n_classes` inconsistency.**
`configs/encoding/dales.yaml` has `n_classes: 8`; `diffusion_up.yaml` has
`n_classes: 7` since `ddc0935`. Reconcile, or document why they differ.
→ **DONE — already consistent at HEAD; documented so it stays that way.**
`ddc0935` (06-07) set all three configs to 7 and dropped the *leading*
`class_weight` entry, i.e. it assumed a phantom index-0 slot. That was reverted;
encoding, `diffusion_0.yaml` and `diffusion_up.yaml` all read 8 at HEAD, with
8-entry `class_weight` lists. 8 is correct: `encode_features` one-hots DALES
labels 1–8 as `sem − 1`, so `n_classes == len(classes)` in
`configs/dataset/dales.yaml`. The invariant is now written down at
`configs/encoding/dales.yaml`. Note for A5: runs between `ddc0935` and the
revert were trained with 7 classes.

**H5. Recover the deleted working notes.**
`advice.md` and `response_on_item7.md` were committed at `62a2c1d` and removed at
`61f6c94`. They contain the Min-SNR versus P2 derivation. Either restore them
under `docs/` or extract the derivation into the code comment from H3, so the
reasoning is not only in the thesis.
→ **DONE, both routes.** Restored verbatim at `docs/advice.md` and
`docs/response_on_item7.md`, each with a provenance header; `advice.md`'s also
flags which of its items have since been implemented, so a reader does not take
a June snapshot for current state. The derivation itself is additionally
summarised in the H3 code comment.

---

## Phase 3 — Cross-repository interface (blocks any future conditioning work)

**X1. Class taxonomy.**
GVAE uses the 15-class DALES 2 taxonomy. This repository uses 7 or 8 legacy
DALES classes. Conditioning across that gap needs an explicit mapping. Write it
down even if the conditioning is not implemented, because the chapter describes
the interface.

**X2. Coarse grid extent.**
`diffusion_0.yaml` describes roughly 32 × 32 × 7 for a 100 m crop at 3.2 m.
GVAE's `GRID_COARSE` is 16 × 16 × 4 and its `config.py` annotates the volumes as
matched to diffusion levels 3 / 2 / 1. The horizontal factor of two is not
explained by that matching. Check against the encoder before the chapter asserts
an integration path.

**X3. The `cond` hook.**
`DiffusionCNN.forward(self, x, t, cond=None)` is still inert. No commit wires
GVAE into the stack. Nothing to do before the freeze; record the status so the
chapter states it accurately.

---

## Record-keeping

**R1. Run attribution.**
`TODO.md` credits the flood fix to `dales_1` at 02-07 11:09, which starts 22
minutes before `58a6c4c` (11:31), consistent with running a fix before
committing it. That run is absent from the TensorBoard export and its reported
numbers match `diffusion_level_1_08-07-10:06` exactly. **The manuscript cites
`08-07-10:06`**, which exists and is verifiable. Update the `TODO.md` entry so
the label and the numbers agree.

**R2. Pre-fix logs are gone.**
No level-1 run predating `58a6c4c` survives, so the "before" side of the flood
argument rests on `test/diagnostics/target_rowsum_check.py` (empty row sum
2.000) and `d1_void_probe` P(void | void) = 0.5, not on a training curve. The
chapter says so explicitly. Do not delete run directories from here on.

**R3. The flood fix is one reformulation, not two.**
`b049029` (void semantic channel, 06-30 14:22) and `58a6c4c`
(`zero_empty_target`, 07-02 11:31) are a single change: occupancy reformulated
as an explicit void category with a consistent one-hot target. The 07-02 10:39
run, which has the first and not the second, sits between the two states
(validation bin 0 at 0.688, flat per-sigma profile). Keep the framing consistent
between the repository and the manuscript.
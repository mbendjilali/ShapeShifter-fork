# test/

Standalone diagnostic scripts for the DALES diffusion pyramid. **Not a unit-test
suite** — every script loads a real checkpoint and real crops, runs on GPU, and
prints a table (or writes `.laz`) you read by eye. There is no `pytest` here and
nothing to collect automatically.

Run everything **from the repo root**:

```bash
python test/<group>/<script>.py [--level N] [...]
```

`common.py` puts `src/` and `src/utils/` on `sys.path`, so no install step is
needed. Shared plumbing (crop loading, `.laz` export, IoU, class histograms)
lives there — scripts never import each other.

## Layout

| Group | Question it answers |
|---|---|
| `generation/` | Does the model produce plausible output? Writes `.laz` for visual inspection. |
| `evaluation/` | How good is it, numerically, against data or a baseline? |
| `diagnostics/` | Why is a specific known failure happening? |

### generation/ — qualitative, writes `.laz` to `output/tests/level{N}/`

| Script | What it does |
|---|---|
| `reconstruct.py` | Noise a real crop to `t`, denoise it back. Sanity check that the denoiser recovers structure. |
| `class_generation.py` | Generate from pure noise with the class channel clamped. "Does the model know what class-X geometry looks like?" `--clamp_mode hard` overwrites the class channels (also asserts occupancy — the original A3 probe); `renorm` redistributes the model's own class-sum onto the target class, leaving occupancy free. |
| `a3_layout_generation.py` | **Thesis A3 follow-up.** Fixed class *layout*, free geometry: one GT class per column, clamped via `renorm`; prints per-class height/density adherence vs the same crop's GT. `--reduce majority\|nonground` picks how a column's class is chosen (`majority` counts voxels and favours tall classes; `nonground` labels by what stands on the column). `--layout_mask half\|random` blanks part of the layout — the only way to see what the model hallucinates in unconstrained space, since DALES ground returns leave every real column defined. |
| `inpaint.py` | Erase a fraction of the crop's X-range, then denoise. "Can it reconstruct information deliberately removed?" |
| `a1_d1_vs_gt.py` | **Thesis A1.** Level-1 output beside the ground-truth crop of the same tile, same viewpoint. |
| `a2_d0_layouts.py` | **Thesis A2.** Unconditional level-0 layouts from pure noise on the canonical base grid. |

### evaluation/ — quantitative, prints tables

| Script | What it does |
|---|---|
| `upsampler_vs_diffusion.py` | Level-N diffusion vs upsampler-alone on occupancy IoU / semantic accuracy over the same fine voxels. Does the refinement earn its cost? |
| `distribution_stats.py` | Generated class marginal + occupancy fraction vs the training data. Catches class skew and over/under-generation. |
| `a5_ablation_curves.py` | **Thesis A5.** Training curves of the two level-1 runs that differ only in `zero_empty_target`. Reads TensorBoard, writes the figure — no checkpoint, no GPU. |

### diagnostics/ — root-cause probes for known bugs

| Script | What it does |
|---|---|
| `d1_void_probe.py` | Single-step `P(void)` on GT-input vs upsampler-input. Localises the D1 occupancy flood to the void head vs the reverse trajectory. |
| `threshold_sweep.py` | Sweeps the D0 occupancy threshold, watching whether semantic skew melts as occupancy drops to the data value. |
| `target_rowsum_check.py` | Checks the level>0 categorical CE target actually sums to 1 (trilinear leftovers make empty rows sum to 2). |

## Typical flows

```bash
# Is level 0 healthy?
python test/evaluation/distribution_stats.py --n_gen 8 --n_data 64 --max_level 0
python test/diagnostics/threshold_sweep.py

# Is level 1 worth running at all?
python test/evaluation/upsampler_vs_diffusion.py --level 1 --n_crops 16

# Level 1 floods occupancy — why?
python test/diagnostics/d1_void_probe.py --level 1 --n_crops 12
python test/diagnostics/target_rowsum_check.py --level 1 --n_crops 8

# Eyeball a checkpoint
python test/generation/reconstruct.py --level 0 --crop 5080_54435_x0000_y0050
python test/generation/class_generation.py --levels 0 --class_ids 0 7
python test/generation/inpaint.py --level 0 --crop 5080_54435_x0000_y0050
```

## Common flags

Most scripts share: `--src` (checkpoint dir, default `checkpoints/diffusion_models/`),
`--level`, `--split`, `--n_crops`, `--base_res` (coarsest pyramid `.pt` label, 16),
`--upsample_fac` (2). Level N reads `base_res * upsample_fac**N`.pt from each crop
directory — see `level_resolutions()` in `common.py`.

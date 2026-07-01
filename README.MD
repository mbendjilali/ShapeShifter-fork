# ShapeShifter — DALES fork

Multi-scale sparse-voxel diffusion trained on DALES 2 aerial LiDAR tiles.
Based on [nissmar/ShapeShifter](https://github.com/nissmar/ShapeShifter) (CVPR 2025).

## Setup

```shell
git clone --recursive <this-repo>
conda env create -f dev_env.yml
conda activate shapeshifter

# Build fVDB (takes a few minutes)
cd openvdb/fvdb
export MAX_JOBS=$(free -g | awk "/^Mem:/{jobs=int(\$4/2.5); if(jobs<1) jobs=1; print jobs}")
pip install .
cd ../..
```

Dataset: `/path/to/dales_2/{train,test}/*.laz` (set `dales_root` in `configs/encoding/dales.yaml`)

## Preprocessing

Encode tiles as 100×100 m crops (0.2 m voxels at finest level → 3.2 m at coarsest):

```shell
python src/encoding/point_cloud.py \
    --config configs/encoding/dales.yaml --out data/dales --all --split train

python src/encoding/point_cloud.py \
    --config configs/encoding/dales.yaml --out data/dales --all --split test
```

Output: `data/dales/{train,test}/{tile_id}_x????_y????/{16,32,64,128,256}.pt`

Optional — add yaw rotation + flip augmentations:

```shell
python src/encoding/point_cloud.py \
    --config configs/encoding/dales.yaml --out data/dales --all --split train --augment
```

## Training

Run from the project root. Configs default to `-dataset dales`.

### Phase 1 — Diffusion level 0 + Upsamplers (parallel)

Level 0 and all upsamplers are **independent** and can run simultaneously on the same GPU.

```shell
# Level 0 diffusion
CUDA_VISIBLE_DEVICES=0 python src/train/diffusion.py \
    -config configs/training/diffusion_0.yaml -dataset dales -level 0 &

# Upsamplers levels 1–4 (independent of each other and of level 0)
CUDA_VISIBLE_DEVICES=0 python src/train/upsampler.py \
    -config configs/training/upsampler.yaml -dataset dales -level 1 &

CUDA_VISIBLE_DEVICES=0 python src/train/upsampler.py \
    -config configs/training/upsampler.yaml -dataset dales -level 2 &

CUDA_VISIBLE_DEVICES=0 python src/train/upsampler.py \
    -config configs/training/upsampler.yaml -dataset dales -level 3 &

CUDA_VISIBLE_DEVICES=0 python src/train/upsampler.py \
    -config configs/training/upsampler.yaml -dataset dales -level 4 &

wait
```

Checkpoints: `checkpoints/upsamplers/dales_{level}.pt`

### Phase 2 — Diffusion levels 1–4 (requires upsampler checkpoints)

Levels 1–4 are independent of each other but each requires its upsampler checkpoint.

```shell
CUDA_VISIBLE_DEVICES=0 python src/train/diffusion.py \
    -config configs/training/diffusion_up.yaml -dataset dales -level 1 &

CUDA_VISIBLE_DEVICES=0 python src/train/diffusion.py \
    -config configs/training/diffusion_up.yaml -dataset dales -level 2 &

CUDA_VISIBLE_DEVICES=0 python src/train/diffusion.py \
    -config configs/training/diffusion_up.yaml -dataset dales -level 3 &

CUDA_VISIBLE_DEVICES=0 python src/train/diffusion.py \
    -config configs/training/diffusion_up.yaml -dataset dales -level 4 &

wait
```

Checkpoints: `checkpoints/diffusion_models/dales_{level}_{timestamp}.pt`

### Multi-GPU

Distribute across GPUs to reduce wall-clock time:

```shell
CUDA_VISIBLE_DEVICES=0 python src/train/diffusion.py \
    -config configs/training/diffusion_0.yaml -dataset dales -level 0 &

CUDA_VISIBLE_DEVICES=1 python src/train/upsampler.py \
    -config configs/training/upsampler.yaml -dataset dales -level 1 &
# … etc.
```

## Visualize results on TensorBoard

```shell
pip install tensorboard
pip install setuptools==80.9.0
```

During training, TensorBoard event files are written under `runs/`:
- `runs/diffusion_level_{level}_{timestamp}/`
- `runs/upsampler_level_{level}_{timestamp}/`

```shell
tensorboard --logdir runs --port 6006
```

SSH tunnel from your machine:

```shell
ssh -L 6006:localhost:6006 <user_name>@<remote_server>
```

Open http://localhost:6006

## Generation

Generate 20 unconditional crops (4 crops/batch, DDPM). Defaults match level-0 training (`diffusion_0.yaml`: ~32×32×7 @ 3.2 m on `16.pt`).

```shell
CUDA_VISIBLE_DEVICES=0 python src/inference/inference.py \
    -dataset dales \
    -src checkpoints/diffusion_models \
    -out output/dales \
    -levels 4 \
    -batch_size 4 \
    -total_num 20
```

Faster with DDIM (100 steps instead of full 1000):

```shell
CUDA_VISIBLE_DEVICES=0 python src/inference/inference.py \
    -dataset dales -src checkpoints/diffusion_models \
    -out output/dales -levels 4 -batch_size 4 -total_num 20 -ddim_steps 100
```

Outputs in `output/dales/`:
- `gen_{i}_{level}.laz` — point cloud at each pyramid level, coloured by semantic class
- `gen_{i}_{level}.pt` — raw DiffusionTensor (for downstream use)

## Diagnostics

Quick checkpoint checks (reconstruction + class-conditional sampling):

```shell
python src/run_tests.py --test AB --level 0 \
    --crop 5080_54435_x0000_y0050 --split test --class_ids 7
```

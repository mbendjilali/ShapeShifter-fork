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

Dataset: `/data/moussabendjilali/archive/data/dales_2/{train,test}/*.laz`

## Preprocessing

Encode all training tiles as 50×50m crops (0.2m voxels → 3.2m at coarsest level):

```shell
python src/encoding/pc_encoding.py --config <config_path> --all --split train
python src/encoding/pc_encoding.py --config <config_path> --all --split test
```

Output: `data/GT_sparse_tensors/dales/{tile_id}_x????_y????/{16,32,64,128,256}.pt`
~2900 crops for train, ~1100 for test.

Optional — add yaw rotation + flip augmentations:

```shell
python src/shape_encoding/pc_encoding.py --all --split train --augment
```

## Training

### Phase 1 — Diffusion level 0 + Upsamplers (parallel)

Level 0 and all upsamplers are **independent** and can run simultaneously on the same GPU.

```shell
# Level 0 diffusion
CUDA_VISIBLE_DEVICES=0 python src/train/train_diffusion.py \
    -config configs/train_dales_diffusion_0.yaml -dataset dales -level 0 &

# Upsamplers levels 1–4 (independent of each other and of level 0)
CUDA_VISIBLE_DEVICES=0 python src/train/train_upsamplers.py \
    -config configs/train_dales_upsampler.yaml -dataset dales -level 1 &

CUDA_VISIBLE_DEVICES=0 python src/train/train_upsamplers.py \
    -config configs/train_dales_upsampler.yaml -dataset dales -level 2 &

CUDA_VISIBLE_DEVICES=0 python src/train/train_upsamplers.py \
    -config configs/train_dales_upsampler.yaml -dataset dales -level 3 &

CUDA_VISIBLE_DEVICES=0 python src/train/train_upsamplers.py \
    -config configs/train_dales_upsampler.yaml -dataset dales -level 4 &

wait
```

Checkpoints: `checkpoints/upsamplers/dales_{level}.pt`

### Phase 2 — Diffusion levels 1–4 (requires upsampler checkpoints)

Levels 1–4 are independent of each other but each requires its upsampler checkpoint.

```shell
CUDA_VISIBLE_DEVICES=0 python src/train/train_diffusion.py \
    -config configs/train_dales_diffusion_up.yaml -dataset dales -level 1 &

CUDA_VISIBLE_DEVICES=0 python src/train/train_diffusion.py \
    -config configs/train_dales_diffusion_up.yaml -dataset dales -level 2 &

CUDA_VISIBLE_DEVICES=0 python src/train/train_diffusion.py \
    -config configs/train_dales_diffusion_up.yaml -dataset dales -level 3 &

CUDA_VISIBLE_DEVICES=0 python src/train/train_diffusion.py \
    -config configs/train_dales_diffusion_up.yaml -dataset dales -level 4 &

wait
```

Checkpoints: `checkpoints/diffusion_models/dales_{level}_{timestamp}.pt`

### Multi-GPU

Distribute across GPUs to reduce wall-clock time:

```shell
CUDA_VISIBLE_DEVICES=0 python src/train/train_diffusion.py \
    -config configs/train_dales_diffusion_0.yaml -dataset dales -level 0 &

CUDA_VISIBLE_DEVICES=1 python src/train/train_upsamplers.py \
    -config configs/train_dales_upsampler.yaml -dataset dales -level 1 &
# … etc.
```
## Visualize results on TensorBoard

### 1. Install TensorBoard

```shell
pip install tensorboard
pip install setuptools==80.9.0
```

### 2. Train the model
Run the training script as usual. During training, TensorBoard log files will be created in the ```runs/``` directory:

```shell
runs/
├── level_0/
├── level_1/
├── level_2/
└── level_3/
```

Each directory corresponds to one diffusion level.

### 3. Launch TensorBoard
From the project root directory, start TensorBoard:

```shell
tensorboard --logdir runs --port 6006
```

Then, from your computer, create a SSH tunnel:

```shell
ssh -L 6006:localhost:6006 <user_name>@<remote_server>
```

And open the following address in the browser:

http://localhost:6006


## Generation

Generate 20 unconditional 50×50m crops (4 crops/batch, DDPM):

```shell
CUDA_VISIBLE_DEVICES=0 python src/inference/sample_diffusion.py \
    -dataset dales \
    -src checkpoints/diffusion_models \
    -out output/dales \
    -levels 4 \
    -batch_size 4 \
    -total_num 20
```

Faster with DDIM (100 steps instead of full 1000):

```shell
CUDA_VISIBLE_DEVICES=0 python src/inference/sample_diffusion.py \
    -dataset dales -src checkpoints/diffusion_models \
    -levels 4 -batch_size 4 -total_num 20 -ddim_steps 100
```

Outputs in `output/dales/`:
- `gen_{i}_{level}.ply` — point cloud at each pyramid level, coloured by semantic class
- `gen_{i}_{level}.pt`  — raw DiffusionTensor (for downstream use)

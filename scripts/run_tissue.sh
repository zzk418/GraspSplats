#!/usr/bin/env bash
set -e

# ---------- Step 1: COLMAP data preparation ----------
QT_QPA_PLATFORM=offscreen ns-process-data video \
  --data scene_data/tissue.mp4 \
  --output-dir scene_data/tissue_data \
  --num-frames-target 40

# convert COLMAP binary model to text format (ns-process-data outputs sparse/0/ directly)
colmap model_converter \
  --input_path  scene_data/tissue_data/colmap/sparse/0 \
  --output_path scene_data/tissue_data/colmap/sparse/0 \
  --output_type TXT

colmap model_converter \
  --input_path  scene_data/tissue_data/colmap/sparse/0 \
  --output_path scene_data/tissue_data/colmap/sparse/0/points3D.ply \
  --output_type PLY 2>/dev/null || true

# ---------- Step 2: compute CLIP part features ----------
python feature-splatting-inria/compute_obj_part_feature.py -s scene_data/tissue_data

# ---------- Step 3: feature splatting training ----------
WANDB_PROJECT=graspsplats WANDB_NAME=tissue \
python feature-splatting-inria/train.py \
  -s scene_data/tissue_data \
  -m outputs/tissue_data \
  --iterations 10000 \
  --feature_type "clip_part"

# ---------- Step 4: launch grasping UI ----------
python scripts/compute_alignment.py \
    --ply scene_data/tissue_data/colmap/sparse/0/points3D.ply \
    --real_size 1 \
    --target_pos 0.5 0.0 0.05 \
    --out outputs/tissue_data/world2base.npy

python realbot_3dgs.py -m outputs/tissue_data
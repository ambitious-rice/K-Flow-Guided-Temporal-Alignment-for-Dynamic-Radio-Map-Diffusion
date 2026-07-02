#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/fzj/RMDM"
PY="/data/fzj/conda_envs/RMDM/bin/python"
DATA="/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack_ShadowDenoiseV1"
CKPT="${ROOT}/checkpoints_dynamic_rmdm_4gpu_no_ckpt_75k/model_phy_step65000.pth"
SCENES="town10_junction_0189,town02_opt_junction_0298"
OUT="${ROOT}/checkpoints_residual_kflow_two_scene_0189_0298"

cd "${ROOT}"
mkdir -p "${OUT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PY}" train_residual_kflow_student.py \
  --data_dir "${DATA}" \
  --split_file split.json \
  --scene_ids "${SCENES}" \
  --init_2d_checkpoint "${CKPT}" \
  --save_dir "${OUT}" \
  --clip_length 32 \
  --clip_stride 1 \
  --frame_stride 16 \
  --batch_size 2 \
  --workers 2 \
  --cache_size 8 \
  --max_steps 1000 \
  --save_interval 250 \
  --log_interval 20 \
  --lr 1e-4 \
  --head_hidden_channels 32 \
  --residual_alpha 0.05 \
  --diff_loss_weight 1.0 \
  --teacher_anchor_weight 1.0 \
  --kflow_weight 0.05 \
  --kflow_warmup_steps 200 \
  --kflow_max_timestep 300 \
  --low_timestep_prob 0.5 \
  --x0_recon_weight 0.0

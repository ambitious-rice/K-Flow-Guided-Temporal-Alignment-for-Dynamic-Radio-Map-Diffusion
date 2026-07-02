#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/fzj/RMDM"
PY="/data/fzj/conda_envs/RMDM/bin/python"
DATA="/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack_ShadowDenoiseV1"
SPLIT_FILE="${DATA}/split.json"
CKPT="${ROOT}/checkpoints_dynamic_rmdm_4gpu_no_ckpt_75k/model_phy_step65000.pth"
OUT_ROOT="${ROOT}/experiments/rmdm_temporal_fullsplit_eval_ddim20_paired_primary"
LOG_ROOT="${OUT_ROOT}/logs"

mkdir -p "${LOG_ROOT}"
cd "${ROOT}"

run_eval() {
  local gpu="$1"
  local split="$2"
  local start_index="$3"
  local end_index="$4"
  local out_dir="${OUT_ROOT}/paired_${split}_${start_index}_${end_index}"
  local log_file="${LOG_ROOT}/paired_${split}_${start_index}_${end_index}.log"

  echo "[$(date '+%F %T')] start paired ${split} ${start_index}:${end_index} on GPU${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" sample_temporal_pinn_paired.py \
    --data_dir "${DATA}" \
    --split_file "${SPLIT_FILE}" \
    --split "${split}" \
    --init_2d_checkpoint "${CKPT}" \
    --output_dir "${out_dir}" \
    --clip_length 32 \
    --frame_stride 32 \
    --clip_stride 1 \
    --start_index "${start_index}" \
    --end_index "${end_index}" \
    --num_samples 0 \
    --batch_size 2 \
    --workers 0 \
    --cache_size 8 \
    --ddim_steps 20 \
    --device cuda \
    --flush_every 5 \
    --resume_existing \
    --temporal_smoothing_weight 0.05 \
    --temporal_smoothing_every 1 \
    --temporal_smoothing_mask_mode inverse_rss_grad \
    --temporal_smoothing_rss_grad_threshold 0.09 \
    --temporal_smoothing_mask_dilate 3 2>&1 | tee -a "${log_file}"
  echo "[$(date '+%F %T')] done paired ${split} ${start_index}:${end_index} on GPU${gpu}"
}

case "${1:-}" in
  gpu2)
    run_eval 2 val 0 3375
    run_eval 2 test 0 3375
    ;;
  gpu3)
    run_eval 3 val 3375 6750
    run_eval 3 test 3375 6750
    ;;
  merge)
    "${PY}" experimental/timestep_adagn/merge_temporal_pinn_paired_shards.py \
      --output_dir "${OUT_ROOT}/paired_val_merged" \
      --shard_dirs "${OUT_ROOT}/paired_val_0_3375" "${OUT_ROOT}/paired_val_3375_6750"
    "${PY}" experimental/timestep_adagn/merge_temporal_pinn_paired_shards.py \
      --output_dir "${OUT_ROOT}/paired_test_merged" \
      --shard_dirs "${OUT_ROOT}/paired_test_0_3375" "${OUT_ROOT}/paired_test_3375_6750"
    ;;
  *)
    echo "Usage: $0 gpu2|gpu3|merge" >&2
    exit 2
    ;;
esac

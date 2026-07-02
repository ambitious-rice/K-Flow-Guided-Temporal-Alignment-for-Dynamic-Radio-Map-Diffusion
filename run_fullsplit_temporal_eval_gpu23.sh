#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/fzj/RMDM"
PY="/data/fzj/conda_envs/RMDM/bin/python"
DATA="/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack_ShadowDenoiseV1"
SPLIT_FILE="${DATA}/split.json"
CKPT="${ROOT}/checkpoints_dynamic_rmdm_4gpu_no_ckpt_75k/model_phy_step65000.pth"
OUT_ROOT="${ROOT}/experiments/rmdm_temporal_fullsplit_eval_ddim20_fast2d_seeded_primary"
LOG_ROOT="${OUT_ROOT}/logs"

mkdir -p "${LOG_ROOT}"
cd "${ROOT}"

run_eval() {
  local gpu="$1"
  local split="$2"
  local label="$3"
  local start_index="$4"
  local end_index="$5"
  local smoothing="$6"
  local out_dir="${OUT_ROOT}/${label}_${split}_${start_index}_${end_index}"
  local log_file="${LOG_ROOT}/${label}_${split}_${start_index}_${end_index}.log"

  echo "[$(date '+%F %T')] start ${label} ${split} ${start_index}:${end_index} on GPU${gpu}"
  local extra_args=()
  if [[ "${smoothing}" == "smooth" ]]; then
    extra_args=(
      --temporal_smoothing_weight 0.05
      --temporal_smoothing_every 1
      --temporal_smoothing_mask_mode inverse_rss_grad
      --temporal_smoothing_rss_grad_threshold 0.09
      --temporal_smoothing_mask_dilate 3
    )
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" sample_temporal_pinn.py \
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
    --batch_size 4 \
    --workers 0 \
    --cache_size 8 \
    --ddim_steps 20 \
    --device cuda \
    --prior_injection_mode uemb \
    --fast_2d_forward \
    --deterministic_clip_noise \
    --primary_metrics_only \
    --flush_every 5 \
    --resume_existing \
    "${extra_args[@]}" 2>&1 | tee -a "${log_file}"
  echo "[$(date '+%F %T')] done ${label} ${split} ${start_index}:${end_index} on GPU${gpu}"
}

case "${1:-}" in
  gpu2)
    run_eval 2 val baseline 0 3375 none
    run_eval 2 val smooth005_invrss009_d3 0 3375 smooth
    run_eval 2 test baseline 0 3375 none
    run_eval 2 test smooth005_invrss009_d3 0 3375 smooth
    ;;
  gpu3)
    run_eval 3 val baseline 3375 6750 none
    run_eval 3 val smooth005_invrss009_d3 3375 6750 smooth
    run_eval 3 test baseline 3375 6750 none
    run_eval 3 test smooth005_invrss009_d3 3375 6750 smooth
    ;;
  merge)
    "${PY}" merge_temporal_pinn_eval_shards.py \
      --output_dir "${OUT_ROOT}/baseline_val_merged" \
      --shard_dirs "${OUT_ROOT}/baseline_val_0_3375" "${OUT_ROOT}/baseline_val_3375_6750"
    "${PY}" merge_temporal_pinn_eval_shards.py \
      --output_dir "${OUT_ROOT}/smooth005_invrss009_d3_val_merged" \
      --shard_dirs "${OUT_ROOT}/smooth005_invrss009_d3_val_0_3375" "${OUT_ROOT}/smooth005_invrss009_d3_val_3375_6750"
    "${PY}" compare_temporal_pinn_eval.py \
      --baseline "${OUT_ROOT}/baseline_val_merged/summary_merged.json" \
      --candidate "${OUT_ROOT}/smooth005_invrss009_d3_val_merged/summary_merged.json" \
      --output "${OUT_ROOT}/compare_val_full.json"

    "${PY}" merge_temporal_pinn_eval_shards.py \
      --output_dir "${OUT_ROOT}/baseline_test_merged" \
      --shard_dirs "${OUT_ROOT}/baseline_test_0_3375" "${OUT_ROOT}/baseline_test_3375_6750"
    "${PY}" merge_temporal_pinn_eval_shards.py \
      --output_dir "${OUT_ROOT}/smooth005_invrss009_d3_test_merged" \
      --shard_dirs "${OUT_ROOT}/smooth005_invrss009_d3_test_0_3375" "${OUT_ROOT}/smooth005_invrss009_d3_test_3375_6750"
    "${PY}" compare_temporal_pinn_eval.py \
      --baseline "${OUT_ROOT}/baseline_test_merged/summary_merged.json" \
      --candidate "${OUT_ROOT}/smooth005_invrss009_d3_test_merged/summary_merged.json" \
      --output "${OUT_ROOT}/compare_test_full.json"
    ;;
  *)
    echo "Usage: $0 gpu2|gpu3|merge" >&2
    exit 2
    ;;
esac

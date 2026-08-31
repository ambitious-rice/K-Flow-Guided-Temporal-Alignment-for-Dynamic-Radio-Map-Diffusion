#!/usr/bin/env bash
set -euo pipefail

repo=/data/fzj/RMDM
python=/data/fzj/conda_envs/RMDM_HVDIT_V2/bin/python
config="$repo/runs/rmdm_hvdit_v4_x0_w16_ratebalanced_no_tx/resolved_w16_from_selected_no_tx_w1.yaml"
checkpoint="$repo/runs/rmdm_hvdit_v4_x0_w16_ratebalanced_no_tx/w16_train/selection_candidates/epoch_010.pth"
manifest="$repo/manifests/dynamic_sparse_v2_semantic_vehicle/val_subset_v1.json"
root="$repo/runs/w16_noise_estimation_20260805/formal_validation_ddim50"
calibration_dir="$root/calibration"
evaluation_dir="$root/evaluation"
gpus=0,1,2,3,4,5

cd "$repo"
mkdir -p "$root"

CUDA_VISIBLE_DEVICES="$gpus" "$python" -m accelerate.commands.launch \
  --multi_gpu --num_processes 6 --main_process_port 29617 \
  scripts/evaluate_w16_noise_estimation.py \
  --config "$config" --checkpoint "$checkpoint" --manifest "$manifest" \
  --subset-stage stage_a --videos-per-scene 10 \
  --rates 1,2,3,5,8,10 --noise-stds 0 --ddim-steps 50 \
  --folds 4 --members 8 --member-batch-size 8 --sigma-batch-size 1 \
  --seed 20260805 --namespace formal_calibration --mode collect \
  --output-dir "$calibration_dir" --expected-visible-gpus "$gpus"

"$python" scripts/fit_w16_noise_calibration.py \
  --units-dir "$calibration_dir/units" \
  --output "$calibration_dir/calibration.json"

CUDA_VISIBLE_DEVICES="$gpus" "$python" -m accelerate.commands.launch \
  --multi_gpu --num_processes 6 --main_process_port 29618 \
  scripts/evaluate_w16_noise_estimation.py \
  --config "$config" --checkpoint "$checkpoint" --manifest "$manifest" \
  --subset-stage stage_b_extra --videos-per-scene 10 \
  --rates 1,2,3,5,8,10 \
  --noise-stds 0,0.005,0.01,0.02,0.03,0.05,0.075,0.10 \
  --ddim-steps 50 --folds 4 --members 8 --member-batch-size 8 --sigma-batch-size 1 \
  --seed 20260805 --namespace formal_evaluation --mode evaluate \
  --calibration "$calibration_dir/calibration.json" \
  --output-dir "$evaluation_dir" --expected-visible-gpus "$gpus"

"$python" scripts/summarize_w16_noise_estimation.py \
  --units-dir "$evaluation_dir/units" \
  --output "$evaluation_dir/summary.json" \
  --bootstrap-draws 2000 --bootstrap-seed 20260805

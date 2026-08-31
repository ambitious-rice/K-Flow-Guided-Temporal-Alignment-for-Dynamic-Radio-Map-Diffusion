#!/usr/bin/env bash
set -euo pipefail

repo=/data/fzj/RMDM
python=/data/fzj/conda_envs/RMDM_HVDIT_V2/bin/python
gpus=0,1,2,3,4,5
root="$repo/runs/w16_noise_estimation_20260805/formal_validation_ddim50/da_noise_scale_and_rmdm_20260807"
w16_config="$repo/runs/rmdm_hvdit_v4_x0_w16_ratebalanced_no_tx/resolved_w16_from_selected_no_tx_w1.yaml"
w16_checkpoint="$repo/runs/rmdm_hvdit_v4_x0_w16_ratebalanced_no_tx/w16_train/selection_candidates/epoch_010.pth"
w16_manifest="$repo/manifests/dynamic_sparse_v2_semantic_vehicle/val_subset_v1.json"
w16_calibration="$repo/runs/w16_noise_estimation_20260805/formal_validation_ddim50/calibration/calibration.json"
w16_stage_b_estimation="$repo/runs/w16_noise_estimation_20260805/formal_validation_ddim50/evaluation"
rmdm_config="$repo/configs/hvdit_v4_joint/t1_to_w16_4gpu.yaml"
rmdm_checkpoint="$repo/runs/rmdm_sf_sparse_v2_fullimage_obstacle_no_tx_tx_source_20260802/epoch_005.pth"
rates=1,2,3,5
scales=0.7,0.85,1,1.15,1.3

mkdir -p "$root"
cd "$repo"

# Stage-A has no overlap with the pre-existing Stage-B estimator evaluation.
CUDA_VISIBLE_DEVICES="$gpus" "$python" -m accelerate.commands.launch \
  --multi_gpu --num_processes 6 --main_process_port 29643 \
  scripts/evaluate_w16_noise_estimation.py \
  --config "$w16_config" --checkpoint "$w16_checkpoint" --manifest "$w16_manifest" \
  --subset-stage stage_a --videos-per-scene 10 --rates "$rates" --noise-stds 0.05 \
  --ddim-steps 50 --folds 4 --members 8 --member-batch-size 2 --seed 20260805 \
  --namespace da_noise_scale_stage_a --mode evaluate --calibration "$w16_calibration" \
  --output-dir "$root/w16_stage_a_estimation" --expected-visible-gpus "$gpus"

CUDA_VISIBLE_DEVICES="$gpus" "$python" -m accelerate.commands.launch \
  --multi_gpu --num_processes 6 --main_process_port 29644 \
  scripts/evaluate_w16_estimated_noise_da.py \
  --config "$w16_config" --checkpoint "$w16_checkpoint" --manifest "$w16_manifest" \
  --estimation-dir "$root/w16_stage_a_estimation" --output-dir "$root/w16_stage_a_da_scales" \
  --subset-stage stage_a \
  --rates "$rates" --noise-stds 0.05 --ddim-steps 50 --guided-steps 15 \
  --strength 0.5 --max-update 0.25 --videos-per-scene 10 --noise-seed 20260805 \
  --expected-visible-gpus "$gpus" --estimated-noise-scales "$scales"

"$python" scripts/select_w16_da_noise_scale.py \
  --units-dir "$root/w16_stage_a_da_scales/units" --output "$root/w16_scale_selection.json" \
  --rates "$rates" --true-sigma 0.05
scale=$(jq -r '.selected_scale' "$root/w16_scale_selection.json")

CUDA_VISIBLE_DEVICES="$gpus" "$python" -m accelerate.commands.launch \
  --multi_gpu --num_processes 6 --main_process_port 29645 \
  scripts/evaluate_w16_estimated_noise_da.py \
  --config "$w16_config" --checkpoint "$w16_checkpoint" --manifest "$w16_manifest" \
  --estimation-dir "$w16_stage_b_estimation" --output-dir "$root/w16_stage_b_da_selected" \
  --subset-stage stage_b_extra \
  --rates "$rates" --noise-stds 0.05 --ddim-steps 50 --guided-steps 15 \
  --strength 0.5 --max-update 0.25 --videos-per-scene 10 --noise-seed 20260805 \
  --expected-visible-gpus "$gpus" --estimated-noise-scales "$scale"

"$python" scripts/summarize_w16_estimated_noise_da.py \
  --units-dir "$root/w16_stage_b_da_selected/units" \
  --output "$root/w16_stage_b_da_selected/summary.json" \
  --bootstrap-draws 2000 --bootstrap-seed 20260807

for sigma in 0 0.01 0.05; do
  extra=()
  if [[ "$sigma" != 0 ]]; then
    extra+=(--include-noise-aware)
  fi
  CUDA_VISIBLE_DEVICES="$gpus" PYTHONPATH="$repo" "$python" -m accelerate.commands.launch \
    --multi_gpu --num_processes 6 --mixed_precision bf16 --main_process_port "$((29650 + ${sigma//./}))" \
    scripts/evaluate_rmdm_ddim_x0_assimilation.py \
    --config "$rmdm_config" --checkpoint "$rmdm_checkpoint" \
    --output "$root/rmdm_sigma${sigma//./p}.json" --split val --subset-stage stage_a \
    --manifest "$w16_manifest" --rates 1,3,5 --strengths 0.5 --ddim-steps 50 \
    --guided-steps 15 --max-update 0.25 --observation-noise-std "$sigma" \
    --frames-per-video 96 --batch-size 12 --log-interval 1 \
    --expected-visible-gpus "$gpus" --fixed-paper-protocol "${extra[@]}"
done

"$python" scripts/render_da_noise_calibration_report.py \
  --selection "$root/w16_scale_selection.json" \
  --stage-b-summary "$root/w16_stage_b_da_selected/summary.json" \
  --rmdm-results "0=$root/rmdm_sigma0.json,0.01=$root/rmdm_sigma0p01.json,0.05=$root/rmdm_sigma0p05.json" \
  --output "$root/README.md"

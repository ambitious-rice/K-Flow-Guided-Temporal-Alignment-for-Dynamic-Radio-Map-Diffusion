#!/usr/bin/env bash
set -euo pipefail

repo=/data/fzj/RMDM
python=/data/fzj/conda_envs/RMDM_HVDIT_V2/bin/python
config="$repo/runs/rmdm_hvdit_v4_x0_w16_ratebalanced_no_tx/resolved_w16_from_selected_no_tx_w1.yaml"
checkpoint="$repo/runs/rmdm_hvdit_v4_x0_w16_ratebalanced_no_tx/w16_train/selection_candidates/epoch_010.pth"
manifest="$repo/manifests/dynamic_sparse_v2_semantic_vehicle/val_subset_v1.json"
root="$repo/runs/w16_noise_estimation_20260805/formal_validation_ddim50"
estimation="$root/evaluation"
calibration="$root/calibration/calibration.json"
corrected="$root/corrected_input_main"
assimilation="$root/estimated_noise_da_main"
gpus=0,1,2,3,4,5
rates=1,2,3,5
sigmas=0,0.01,0.05

cd "$repo"

CUDA_VISIBLE_DEVICES="$gpus" "$python" -m accelerate.commands.launch \
  --multi_gpu --num_processes 6 --main_process_port 29620 \
  scripts/evaluate_w16_corrected_input.py \
  --config "$config" --checkpoint "$checkpoint" --manifest "$manifest" \
  --calibration "$calibration" --estimation-dir "$estimation" \
  --output-dir "$root/smoke_corrected_input" --rates 1 --noise-stds 0.01 \
  --ddim-steps 2 --videos-per-scene 10 --noise-seed 20260805 \
  --expected-visible-gpus "$gpus" --max-units-per-rank 1

CUDA_VISIBLE_DEVICES="$gpus" "$python" -m accelerate.commands.launch \
  --multi_gpu --num_processes 6 --main_process_port 29621 \
  scripts/evaluate_w16_corrected_input.py \
  --config "$config" --checkpoint "$checkpoint" --manifest "$manifest" \
  --calibration "$calibration" --estimation-dir "$estimation" \
  --output-dir "$corrected" --rates "$rates" --noise-stds "$sigmas" \
  --ddim-steps 50 --videos-per-scene 10 --noise-seed 20260805 \
  --expected-visible-gpus "$gpus"

"$python" scripts/summarize_w16_corrected_input.py \
  --units-dir "$corrected/units" --output "$corrected/summary.json" \
  --bootstrap-draws 2000 --bootstrap-seed 20260805

CUDA_VISIBLE_DEVICES="$gpus" "$python" -m accelerate.commands.launch \
  --multi_gpu --num_processes 6 --main_process_port 29622 \
  scripts/evaluate_w16_estimated_noise_da.py \
  --config "$config" --checkpoint "$checkpoint" --manifest "$manifest" \
  --estimation-dir "$estimation" --output-dir "$root/smoke_estimated_noise_da" \
  --rates 1 --noise-stds 0.01 --ddim-steps 2 --guided-steps 1 \
  --strength 0.5 --max-update 0.25 --videos-per-scene 10 \
  --noise-seed 20260805 --expected-visible-gpus "$gpus" \
  --max-units-per-rank 1

CUDA_VISIBLE_DEVICES="$gpus" "$python" -m accelerate.commands.launch \
  --multi_gpu --num_processes 6 --main_process_port 29623 \
  scripts/evaluate_w16_estimated_noise_da.py \
  --config "$config" --checkpoint "$checkpoint" --manifest "$manifest" \
  --estimation-dir "$estimation" --output-dir "$assimilation" \
  --rates "$rates" --noise-stds "$sigmas" \
  --ddim-steps 50 --guided-steps 15 --strength 0.5 --max-update 0.25 \
  --videos-per-scene 10 --noise-seed 20260805 \
  --expected-visible-gpus "$gpus"

"$python" scripts/summarize_w16_estimated_noise_da.py \
  --units-dir "$assimilation/units" --output "$assimilation/summary.json" \
  --bootstrap-draws 2000 --bootstrap-seed 20260805

"$python" scripts/render_w16_noise_experiment_report.py \
  --estimation-summary "$estimation/summary.json" \
  --fold-summary "$root/f8_p1_clean/summary.json" \
  --corrected-summary "$corrected/summary.json" \
  --assimilation-summary "$assimilation/summary.json" \
  --output "$root/README.md"

#!/usr/bin/env bash
set -euo pipefail

repo=/data/fzj/RMDM
python=/data/fzj/conda_envs/RMDM_HVDIT_V2/bin/python
config="$repo/runs/rmdm_hvdit_v4_x0_w16_ratebalanced_no_tx/resolved_w16_from_selected_no_tx_w1.yaml"
checkpoint="$repo/runs/rmdm_hvdit_v4_x0_w16_ratebalanced_no_tx/w16_train/selection_candidates/epoch_010.pth"
manifest="$repo/manifests/dynamic_sparse_v2_semantic_vehicle/val_subset_v1.json"
root="$repo/runs/w16_noise_estimation_20260805/formal_validation_ddim50"
evaluation="$root/evaluation"
calibration="$root/calibration/calibration.json"
f8="$root/f8_p1_clean"
gpus=0,1,2,3,4,5

cd "$repo"
"$python" scripts/refresh_w16_em_diagnostics.py \
  --evaluation-dir "$evaluation" --calibration "$calibration"
"$python" scripts/summarize_w16_noise_estimation.py \
  --units-dir "$evaluation/units" --output "$evaluation/summary.json" \
  --bootstrap-draws 2000 --bootstrap-seed 20260805
"$python" scripts/render_w16_noise_estimation_report.py \
  --calibration "$calibration" --summary "$evaluation/summary.json" \
  --output "$evaluation/README.md"
"$python" - "$evaluation/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
if not summary["gates"]["passed_all_rates"]:
    raise SystemExit("all-rate gate failed; downstream sensitivity is not launched")
PY

CUDA_VISIBLE_DEVICES="$gpus" "$python" -m accelerate.commands.launch \
  --multi_gpu --num_processes 6 --main_process_port 29619 \
  scripts/evaluate_w16_noise_estimation.py \
  --config "$config" --checkpoint "$checkpoint" --manifest "$manifest" \
  --subset-stage stage_b_extra --videos-per-scene 10 \
  --rates 1 --noise-stds 0 --ddim-steps 50 \
  --folds 8 --members 8 --member-batch-size 8 --sigma-batch-size 1 \
  --seed 20260805 --namespace formal_f8_p1_clean --mode evaluate \
  --calibration "$calibration" --output-dir "$f8" \
  --expected-visible-gpus "$gpus"

"$python" scripts/summarize_w16_fold_sensitivity.py \
  --baseline-units "$evaluation/units" --candidate-units "$f8/units" \
  --baseline-folds 4 --candidate-folds 8 \
  --bootstrap-draws 2000 --bootstrap-seed 20260805 \
  --output "$f8/summary.json"

#!/usr/bin/env bash
set -euo pipefail

cd /data/fzj/RMDM

export PYTHONUNBUFFERED=1

TRAIN_SESSION=${TRAIN_SESSION:-rmdm_step2_dynamic_alltrain_5k}
TRAIN_DIR=${TRAIN_DIR:-/data/fzj/RMDM/checkpoints_residual_step2_dynamic_q85_w1e4_alltrain_5k}
DATA=${DATA:-/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack_ShadowDenoiseV1}
BASE=${BASE:-/data/fzj/RMDM/checkpoints_dynamic_rmdm_4gpu_no_ckpt_75k/model_phy_step65000.pth}
OUT=${OUT:-/data/fzj/RMDM/experiments/residual_step2_dynamic_q85_w1e4_alltrain_5k_eval20pct_alpha003}
PY=${PY:-/data/fzj/conda_envs/RMDM/bin/python}
HEAD_HIDDEN=${HEAD_HIDDEN:-32}
HEAD_LAYERS=${HEAD_LAYERS:-2}

# Hard cutoff leaves enough time for the 20% sampled test evaluation before 08:00.
CUTOFF_EPOCH=${CUTOFF_EPOCH:-$(date -d '2026-06-29 06:30:00' +%s)}

mkdir -p "$OUT"
LOG="$OUT/watch_eval.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

latest_checkpoint() {
  if [[ -f "$TRAIN_DIR/residual_student_final.pth" ]]; then
    echo "$TRAIN_DIR/residual_student_final.pth"
    return 0
  fi
  ls -1 "$TRAIN_DIR"/residual_student_step*.pth 2>/dev/null | sort -V | tail -n 1
}

log "watcher started"
log "train session: $TRAIN_SESSION"
log "train dir: $TRAIN_DIR"

while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
  now=$(date +%s)
  ckpt=$(latest_checkpoint || true)
  log "training still running; latest checkpoint: ${ckpt:-none}"
  if (( now >= CUTOFF_EPOCH )); then
    log "cutoff reached; stopping training session to free GPUs for evaluation"
    tmux send-keys -t "$TRAIN_SESSION" C-c || true
    sleep 30
    break
  fi
  sleep 300
done

CKPT=$(latest_checkpoint || true)
if [[ -z "${CKPT:-}" || ! -f "$CKPT" ]]; then
  log "ERROR: no checkpoint found for evaluation"
  exit 1
fi
log "using checkpoint: $CKPT"

INDEX_DIR="$OUT/index_shards"
mkdir -p "$INDEX_DIR"

TOTAL=$("$PY" - <<'PY'
from lib.dynamic_clip_loaders import DynamicRadioMapRMDMClip
root = "/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack_ShadowDenoiseV1"
ds = DynamicRadioMapRMDMClip(
    root=root,
    split="test",
    split_file="split.json",
    clip_length=32,
    frame_stride=32,
    clip_stride=1,
    cache_size=1,
    include_k2=True,
)
print(len(ds))
PY
)
SAMPLE_COUNT=$((TOTAL / 5))
log "full test clips: $TOTAL"
log "sampled test clips: $SAMPLE_COUNT"

"$PY" - "$TOTAL" "$SAMPLE_COUNT" "$INDEX_DIR" <<'PY'
import sys
from pathlib import Path
import numpy as np

total = int(sys.argv[1])
sample_count = int(sys.argv[2])
out = Path(sys.argv[3])
indices = np.linspace(0, total - 1, sample_count, dtype=np.int64)
indices = np.unique(indices)
while len(indices) < sample_count:
    missing = sample_count - len(indices)
    extra = np.linspace(0, total - 1, sample_count + missing, dtype=np.int64)
    indices = np.unique(np.concatenate([indices, extra]))
indices = indices[:sample_count]
for shard_id, shard_indices in enumerate(np.array_split(indices, 4)):
    (out / f"shard_{shard_id}.txt").write_text(
        "\n".join(str(int(idx)) for idx in shard_indices) + "\n",
        encoding="utf-8",
    )
print(f"wrote {len(indices)} sampled indices to {out}")
PY

SHARD_DIRS=()
for spec in "0 0" "1 1" "2 2" "3 3"; do
  set -- $spec
  gpu=$1
  shard_id=$2
  index_file="$INDEX_DIR/shard_${shard_id}.txt"
  shard="$OUT/shard_${shard_id}"
  SHARD_DIRS+=("$shard")
  mkdir -p "$shard"
  log "launch shard gpu=$gpu index_file=$index_file count=$(wc -l < "$index_file")"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" sample_residual_kflow_paired.py \
    --data_dir "$DATA" \
    --split test \
    --split_file split.json \
    --init_2d_checkpoint "$BASE" \
    --residual_checkpoint "$CKPT" \
    --output_dir "$shard" \
    --clip_length 32 \
    --frame_stride 32 \
    --clip_stride 1 \
    --index_file "$index_file" \
    --batch_size 3 \
    --workers 2 \
    --cache_size 8 \
    --device cuda \
    --ddim_steps 20 \
    --ddim_eta 1.0 \
    --seed 1234 \
    --residual_alpha 0.05 \
    --override_residual_alpha 0.03 \
    --head_hidden_channels "$HEAD_HIDDEN" \
    --head_num_layers "$HEAD_LAYERS" \
    --resume_existing \
    > "$shard/run.log" 2>&1 &
done

wait
log "all eval shards finished; merging"
"$PY" experimental/timestep_adagn/merge_temporal_pinn_paired_shards.py \
  --output_dir "$OUT/merged" \
  --shard_dirs "${SHARD_DIRS[@]}" \
  > "$OUT/merge.log" 2>&1

log "merged summary: $OUT/merged/summary_paired_merged.json"

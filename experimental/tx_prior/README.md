# tx_prior

Single task directory for the W1 TX-conditioned scene prior. The model receives
`building`, `vehicle`, and explicit `tx`; `observed_rss` and `sampling_mask` are
canonical zeros at the model boundary, including HWM, condition pyramid, and
denoiser stem paths. This preserves the existing V4 tensor shapes without using
samples, observation noise, observation alignment, heldout loss, or condition
dropout.

`smoke.py` is a preflight-only helper. Real two-step smoke training is explicit:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=src:. \
  /data_p6/fzj/conda/envs/RMDM/bin/python -m accelerate.commands.launch \
  --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision bf16 \
  --dynamo_backend no --main_process_port 29641 \
  -m experimental.tx_prior.train --config experimental/tx_prior/smoke.yaml \
  --smoke --smoke-data-limit 512
```

The production environment on this machine uses `torch==2.11.0+cu128` and
`natten==0.21.6+torch2110cu128`.

If smoke stopped at step 0 before writing a checkpoint, restart it in place:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=src:. \
  /data_p6/fzj/conda/envs/RMDM/bin/python -m accelerate.commands.launch \
  --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision bf16 \
  --dynamo_backend no --main_process_port 29641 \
  -m experimental.tx_prior.train --config experimental/tx_prior/smoke.yaml \
  --smoke --smoke-data-limit 512 --restart-incomplete
```

This restart is accepted only when `status.json` is readable, its state is
`training` or `failed`, `global_step` is zero, and no `checkpoints/last.pth`
exists. Existing status and history records are retained, and a new history
event is appended.

Formal training is also explicit:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=src:. \
  /data_p6/fzj/conda/envs/RMDM/bin/python -m accelerate.commands.launch \
  --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision bf16 \
  --dynamo_backend no --main_process_port 29641 \
  -m experimental.tx_prior.train --config experimental/tx_prior/train.yaml
```

Resume formal training in the same stage directory:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=src:. \
  /data_p6/fzj/conda/envs/RMDM/bin/python -m accelerate.commands.launch \
  --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision bf16 \
  --dynamo_backend no --main_process_port 29641 \
  -m experimental.tx_prior.train --config experimental/tx_prior/train.yaml \
  --resume-from runs/tx_prior/train/checkpoints/last.pth
```

The full local configuration is `train.yaml`, fixed to GPUs 4-7, batch 64/GPU,
accumulation 1, global batch 256, BF16, no gradient checkpointing, and
`runs/tx_prior`. It runs through at most 80k, validates first at 10k and every
5k thereafter. Non-improvements before 25k do not consume patience; starting
at 25k, two consecutive validations without a lower `full_image.nmse` stop the
run. The local runner calls
`build_scene_prior_system` and never constructs sparse observations.

Outputs remain under one task root: smoke uses `runs/tx_prior/smoke`, and formal
training uses `runs/tx_prior/train`. Fresh runs refuse only a non-empty matching
stage directory. Resume with the matching stage's `checkpoints/last.pth`.

`remote.yaml` is the short machine-specific equivalent for the two GPUs and
paths on `lab_server_137` (batch 64/GPU, accumulation 2, global batch 256). Run
it with repository root `/data_16T_137/fzj/RMDM/project`; deploy only the
task-level `project/runs/tx_prior` symlink so it resolves to the unified task
root `/data_16T_137/fzj/RMDM/runs/tx_prior` (leave other `project/runs`
contents untouched). Resume from the same
`runs/tx_prior/train/checkpoints/last.pth`; do not create another task or retry
directory.

Training code must call `make_prior_training_batch` directly on the dense
dataset batch and must not instantiate `SamplingPolicy`. Prior evaluation uses
`deterministic_prior_noise_like`, whose seed depends only on frame identity and
the experiment seed (there is intentionally no observation-rate argument).

## Disposable packed cache

Raw data remains the source of truth on the `/data_p6` mechanical array. The
packed cache is a derived, disposable artifact under `/home/fzj` on the root
NVMe filesystem. Its fixed recommended location is
`/home/fzj/.cache/rmdm/tx_prior`; an incomplete build is never accepted by the
reader. `train.yaml` intentionally keeps the legacy backend as its default;
switching always requires an explicit `--packed-root`.

Build and verify only when source-disk contention is acceptable:

```bash
PYTHONPATH=src:. /data_p6/fzj/conda/envs/RMDM/bin/python \
  -m experimental.tx_prior.packed_cli build \
  --source-root /data_p6/fzj/resources/RMDM/datasets/extracted/DynamicRadioMap/M20_Formal075_RadioMapSeerPack \
  --split-file /data_p6/fzj/resources/RMDM/components/dataset_metadata/multi20_formal_scene_split.json
PYTHONPATH=src:. /data_p6/fzj/conda/envs/RMDM/bin/python \
  -m experimental.tx_prior.packed_cli verify \
  --source-root /data_p6/fzj/resources/RMDM/datasets/extracted/DynamicRadioMap/M20_Formal075_RadioMapSeerPack \
  --split-file /data_p6/fzj/resources/RMDM/components/dataset_metadata/multi20_formal_scene_split.json
```

The builder prints a flushed progress line after the first video, every 100
videos, and completion. Verification first checks the complete legacy video
order, then compares 64 evenly selected videos at their first, middle, and last
frames against raw data. It reports `compared_frames` without rereading the
entire split.

At a checkpoint boundary, add
`--packed-root /home/fzj/.cache/rmdm/tx_prior` to the same resume
command. This does not change `WindowDataset` indexing or epoch permutation.
Remove it only through the marker-checked lifecycle command:
`PYTHONPATH=src:. /data_p6/fzj/conda/envs/RMDM/bin/python -m experimental.tx_prior.packed_cli remove --confirm`.
It can always be rebuilt from the raw source.

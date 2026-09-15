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
  --multi_gpu --num_processes 4 --main_process_port 29641 \
  -m experimental.tx_prior.train --config experimental/tx_prior/smoke.yaml \
  --smoke --smoke-data-limit 512
```

The production environment on this machine uses `torch==2.11.0+cu128` and
`natten==0.21.6+torch2110cu128`.

If smoke stopped at step 0 before writing a checkpoint, restart it in place:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=src:. \
  /data_p6/fzj/conda/envs/RMDM/bin/python -m accelerate.commands.launch \
  --multi_gpu --num_processes 4 --main_process_port 29641 \
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
  --multi_gpu --num_processes 4 --main_process_port 29641 \
  -m experimental.tx_prior.train --config experimental/tx_prior/train.yaml
```

Resume formal training in the same stage directory:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=src:. \
  /data_p6/fzj/conda/envs/RMDM/bin/python -m accelerate.commands.launch \
  --multi_gpu --num_processes 4 --main_process_port 29641 \
  -m experimental.tx_prior.train --config experimental/tx_prior/train.yaml \
  --resume-from runs/tx_prior/train/checkpoints/last.pth
```

The full configuration is `train.yaml`, fixed to GPUs 4-7, batch 64/GPU,
accumulation 1, global batch 256, BF16, no gradient checkpointing, and
`runs/tx_prior`. The local runner calls `build_scene_prior_system` and never
constructs sparse observations.

Outputs remain under one task root: smoke uses `runs/tx_prior/smoke`, and formal
training uses `runs/tx_prior/train`. Fresh runs refuse only a non-empty matching
stage directory. Resume with the matching stage's `checkpoints/last.pth`.

After the 10k evaluation, if training should continue, edit `max_steps` in the
same `train.yaml` and resume from `runs/tx_prior/train/checkpoints/last.pth`.
Do not create another configuration, task directory, retry directory, or
versioned run directory for that continuation.

Training code must call `make_prior_training_batch` directly on the dense
dataset batch and must not instantiate `SamplingPolicy`. Prior evaluation uses
`deterministic_prior_noise_like`, whose seed depends only on frame identity and
the experiment seed (there is intentionally no observation-rate argument).

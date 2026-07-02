# Timestep AdaGN Temporal Adapter Experiments

This folder contains non-final experimental code for the current no-K2 temporal adapter line.

Keep the original RMDM-style root clean: root-level Python files should stay limited to stable/base code such as `unet.py` and `train.py`.

Current entry points:

```bash
/data/fzj/conda_envs/RMDM/bin/python experimental/timestep_adagn/train_dual_decoder_timestep_scale.py
/data/fzj/conda_envs/RMDM/bin/python experimental/timestep_adagn/sample_dual_decoder_timestep_scale_paired.py
/data/fzj/conda_envs/RMDM/bin/python experimental/timestep_adagn/merge_temporal_pinn_paired_shards.py
```

The code intentionally adds the repository root to `sys.path` through `_paths.py`, so these scripts can still import root modules such as `utils`, `lib`, and `unet`.

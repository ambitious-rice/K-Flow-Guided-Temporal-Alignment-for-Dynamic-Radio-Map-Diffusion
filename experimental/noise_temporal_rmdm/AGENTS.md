# Noise-aware temporal RMDM research contract

These are project-level research rules for this experiment. Keep policy in
documentation and tests; do not add policy-only runtime gates, branch checks,
path allowlists, or duplicated defensive machinery to training code.

- T1 is single-frame and has passed validation. T16 uses continuous 16-frame
  windows and must initialize from the validation-selected T1 step 20250.
- T16 jointly trains the copied T1 model and the declared temporal refiners;
  do not freeze the spatial/HWM weights by default. Temporal residual outputs
  stay zero initialized so inflation is exactly framewise T1 before training.
- Do not launch formal T16 training until the user approves the implementation
  and training plan in a clean follow-up conversation.
- Tx position is unknown once sparse RSS samples are available. Do not load or
  pass Tx heatmaps, coordinates, identifiers as learned features, or indirect
  Tx-position encodings to either model branch. `tx_id` may remain only as a
  file locator and deterministic sample identity.
- Do not use the Tx source-anchor loss `L_source`. The minimal physics term may
  retain only the existing equation and obstacle components.
- Observation noise is `Y=M*(X+sigma*epsilon)` for normalized RSS `X`, without
  training-time clipping. Use one sigma per sample/clip and independent errors
  conditional on sigma.
- Sample sigma with probability 0.20 at exactly zero, probability 0.65 from
  `U(0,0.05)`, and probability 0.15 from `U(0.05,0.09)`. Pass `sigma^2`
  explicitly to the noise-aware condition path. `reference_variance=0.09^2`
  is an embedding scale, not an estimated data constant.
- Keep changes minimal. Preserve the legacy diffusion U-Net topology where
  practical, put noise-aware capacity in condition/HWM, and do not add losses
  or adapters without experiment-backed need.
- Use the tracked Clean16 split. Its four excluded scenes never enter train,
  validation, or test; keep the splits scene-disjoint and reserve test scenes.
- Use periodic DDIM20 validation only to select checkpoints. Run the tracked
  partial DDIM20 test once after selection; test results never feed back into
  training, hyperparameters, stopping, or checkpoint choice.
- Formal compute uses tested, committed, GitHub-pushed source and records the
  commit. Apply this during launch/review, not through policy-heavy code.
- Keep development-only smoke programs and their tests on the dedicated smoke
  branch. Record important formal jobs under `.agents/runs/noise_temporal_rmdm.yaml`.

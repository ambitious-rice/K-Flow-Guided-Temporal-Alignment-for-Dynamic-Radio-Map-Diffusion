# 500-video evaluation and ranked 200-video split

Freeze the step7000 raw model from the all-parameter trajectory fine-tuning run.
Select 500 video IDs uniformly without replacement from the sorted 1500-video
original test population with NumPy PCG64 seed20261011. A video is a
scene/episode/transmitter record, matching the dataset's existing definition.

Evaluate the first and last16frames within the first96frames (zero-based starts0,80),
1/2/3percent sampling, and true/input sigma in0,.03,.05,.09. Use FP32DDIM50,
eta0, and a batch of the four input-sigma conditions. Noise-free ground truth
is used to construct synthetic observations and evaluate outputs; only the
six public conditioning fields enter the model. All random masks, measurement
Gaussian fields and diffusion initial states are fixed CPU tensors. Preserve
candidate video indices when reusing seeds on the selected subset.

Pool the two windows' unobserved SSE and valid-unobserved pixel counts before
forming each error. For each video/rate/true sigma, compute the signed relative
advantage of correct input sigma over the lowest-error incorrect input sigma:

`gain = (best_wrong_MSE - correct_MSE) / max(best_wrong_MSE, 1e-12)`.

Rank videos by the mean gain over all12rate/true-sigma cells, descending. Break
ties by lower mean correct-sigma MSE, then video ID. Select exactly200, without
requiring all12cells to pass. The manifest records this selection rule and the
checkpoint that produced the ranking. Keep the complete500-video results too.

`prepare` freezes protocol, model, manifest, allocation and1000input bundles.
`launch --host local` runs all500videos on8localworkers and automatically ranks
them after inference. The remote servers are reserved for later algorithms per
user instruction. Larger inference batches can be selected using `benchmark`;
all batching preserves each case and its four ordered input-sigma interventions.
The local launcher writes `selected200_split.json`, ranking tables and reuse
indices, and marks completion only after all12000caseJSONs and predictions exist.

Predictions are saved as FP32 tensors of shape[4,16,1,128,128], in input-sigma
order. All48000outputs remain available. Metrics include whole-map error,
valid-unobserved error, observed-point error, frame PSNR and temporal error.
Temporal differences are computed within each window only.

The final split's `video_indices` mapping retains the original candidate indices;
`tables/selected200_reuse_index.csv` points to each frozen input and saved output.
Do not overwrite the earlier test200split or mix checkpoints/window protocols.

The user fixed the evaluation horizon at96frames. An initial tail84 preparation
was superseded before selection; only unchanged head0outputs can be reused from
that preparation. Tail84 artifacts are archived and excluded from all tables.

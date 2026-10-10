# W16 combined measurement-noise conditioning experiment

Authorized on2026-10-10: use6GPUs to test the researched combinations. Three
new2GPU arms complete the2x2 with the existing noise-map/IID run:

| Arm | GPUs | Injection | Training |
|---|---|---|---|
| Historical noise_map | previous run | Two noise-map stems | IID |
| map_paired |0,1| Same two stems | Paired,stratified |
| hybrid_iid |2,3| Maps + block modulation + HWM sigma | IID |
| hybrid_paired |4,5| Maps + block modulation + HWM sigma | Paired,stratified |

Parameter counts:143,826,653 for maps only;174,821,085 for hybrid. Independent
full-width modulation heads add~21.6% parameters. This is not a parameter-
matched architecture comparison; report memory/throughput as well as accuracy.

All new models start randomly. Seed20261009 reproduces the historical
backbone; new parameters initialize under an isolated seed20261010.
No checkpoint is loaded into training. Step0 comparisons verify common
parameters and outputs. HWM remains trainable with its detached gate and
all original auxiliary losses. The two-step rollout and inference DDIM50
are unchanged. All six loss coefficients remain1.

`sigma_embedding` encodes measurement standard deviation divided by0.09.
Its embedding enters HWM feature modulation and existing stem/shared
modulation/output heads alongside the separate rate embedding. Each DiT
encoder,bottleneck,decoder block additionally gets its own six-parameter
sigma head. Existing timestep embedding stays separate until downstream
addition. New block heads start atzero. Existing RMSNorm/Wan residual blocks
remain intact; this is adaptive feature modulation inspired by DiT, not an
exact conversion to original DiT adaLN-Zero. Noise map supplies physical
sigma; the mask distinguishes missing values. No remapping or ranking loss.

Paper basis:
- FFDNet noise map and matched clean supervision: https://arxiv.org/pdf/1710.04026
- CDM separates conditioning-noise strength from diffusion time:
  https://arxiv.org/html/2106.15282v3 (Algorithm1).
- DiT per-block adaptive conditioning:
  https://github.com/facebookresearch/DiT/blob/main/models.py
The paired/stratified sampler is our experimental adaptation, not a claimed
method/result of those papers. HWM conditioning is also project-specific.

Full9000 training videos,W16; rates1..10% uniform; sigma marginal U[0,.09).
Paired microbatch8 contains2 clips x4 independently sampled equal-width
sigma strata, permuted. Within a group, target/mask/rate/t/initial diffusion
noise and measurement Gaussian field are shared. Measurement and diffusion
noise fields remain independently generated. IID data exactly reuse old RNG.
Both arms have global32:2GPUs x8 x2accumulation. Paired has8 independent clip
draws perupdate whileIID has32; this is documented treatment, not equal unique
data exposure. Stratification and pairing are jointly tested, not isolated.

AdamW1e-4,warmup1000,cosine max40000,EMA.999,gradclip1,BF16 training,
FP32 DDIM50 validation. Same original fixed tune8/confirmation16 input bank.
Validate raw+EMA at500,1000,1500,2000,2500,3000,then every1000.
Early stop protects4000,patience4, improvement0.2%MSE or0.002sigma-gain.
Freeze constant on tune. Save best reconstruction,largestpositivegain,and
lowestMSEpositivegain candidates; confirm them and repeat selected candidate
with two extra diffusion seeds. Uncertainty and row minima must accompany
average gains. Reused validation is exploratory; formal200video test excluded.
Common-step results versus the historical noise-map arm are retained in
comparison.json. The historical baseline is not a new replicate.

Budget:7h training+validation,8.75h hardtotal. Save model,EMA,optimizer,RNG,
early-stop state. A time-limited run is resumable, not declared converged.
All experiment state lives in `.agents/runs/noise_combined_20261010.yaml`.

```
PYTHONPATH=src:. OMP_NUM_THREADS=4 /data_p6/fzj/conda/envs/RMDM/bin/python -m experimental.noise_combined.prepare
PYTHONPATH=src:. OMP_NUM_THREADS=4 /data_p6/fzj/conda/envs/RMDM/bin/python -m experimental.noise_combined.launch --root runs/noise_combined_20261010 --smoke
PYTHONPATH=src:. OMP_NUM_THREADS=4 /data_p6/fzj/conda/envs/RMDM/bin/python -m experimental.noise_combined.launch --root runs/noise_combined_20261010
```

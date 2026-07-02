# RMDM 时序一致性改造进展记录

更新时间：

```text
2026-07-02 15:55 CST
```

## 2026-07-02 15:55 CST 更新：non-overlap32 正式两阶段大规模训练完成，100 clips DDIM20 测试完成

本轮是当前更正式的全 train split 训练：把视频按不重叠 32 帧 clip 训练，避免 `frame_stride=1` 造成高度重叠采样。

数据与训练语义：

```text
data:
/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack_ShadowDenoiseV1

split_file: split.json
train scenes: all train scenes
clip_length: 32
clip_stride: 1
frame_stride: 32
train clips: 31500

per branch GPUs: 4
batch_size per GPU: 1
gradient_accumulation_steps: 8
effective batch: 32 clips/update
1 epoch: about 985 optimizer updates
```

训练路径：

```text
single-scale stage1, 3 epochs:
/data/fzj/RMDM/checkpoints_fulltrain_nonoverlap32_single_stage1_3ep_b1a8

key-path gate stage1, 3 epochs:
/data/fzj/RMDM/checkpoints_fulltrain_nonoverlap32_keypath_stage1_3ep_b1a8

single-scale stage2 full-FT, 5 epochs:
/data/fzj/RMDM/checkpoints_fulltrain_nonoverlap32_single_stage2_5ep_b1a8

key-path gate stage2 full-FT, 5 epochs:
/data/fzj/RMDM/checkpoints_fulltrain_nonoverlap32_keypath_stage2_5ep_b1a8
```

最终 checkpoint：

```text
single-scale:
/data/fzj/RMDM/checkpoints_fulltrain_nonoverlap32_single_stage2_5ep_b1a8/dual_decoder_adapter_final.pth

key-path gate:
/data/fzj/RMDM/checkpoints_fulltrain_nonoverlap32_keypath_stage2_5ep_b1a8/dual_decoder_adapter_final.pth
```

100 clips paired DDIM20 测试协议：

```text
split: test
clip_length: 32
frame_stride: 32
num_clips: 100
DDIM steps: 20
seed: 1234
baseline: original RMDM model_phy_step65000.pth
paired noise: baseline and candidate use same initial noise
```

测试输出：

```text
single-scale summary:
/data/fzj/RMDM/experiments/fulltrain_nonoverlap32_single_stage2_5ep_ddim20_eval100/merged/summary_paired_merged.json

single-scale per-clip:
/data/fzj/RMDM/experiments/fulltrain_nonoverlap32_single_stage2_5ep_ddim20_eval100/merged/per_clip_paired_merged.csv

key-path gate summary:
/data/fzj/RMDM/experiments/fulltrain_nonoverlap32_keypath_stage2_5ep_ddim20_eval100/merged/summary_paired_merged.json

key-path gate per-clip:
/data/fzj/RMDM/experiments/fulltrain_nonoverlap32_keypath_stage2_5ep_ddim20_eval100/merged/per_clip_paired_merged.csv
```

核心结果，100 clips mean：

| Model | MAE | MSE | PSNR | tOF | TWE | kflow EPE | kflow L1 | tdelta L1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RMDM baseline paired with single run | 0.033786 | 0.007106 | 23.8869 | 0.714846 | 0.011789 | 1.779464 | 1.128136 | 0.013292 |
| single-scale adapter | 0.031449 | 0.007100 | 24.0685 | 0.306215 | 0.005347 | 0.880864 | 0.555613 | 0.006945 |
| delta single-scale | -0.002337 | -0.0000067 | +0.1816 dB | -0.408631 | -0.006442 | -0.898600 | -0.572523 | -0.006347 |
| RMDM baseline paired with key-path run | 0.036976 | 0.007217 | 23.8019 | 0.675952 | 0.011396 | 1.764384 | 1.118463 | 0.012943 |
| key-path gate adapter | 0.034030 | 0.006849 | 24.2142 | 0.309358 | 0.005460 | 0.906892 | 0.572116 | 0.007059 |
| delta key-path gate | -0.002946 | -0.000368 | +0.4123 dB | -0.366594 | -0.005936 | -0.857492 | -0.546346 | -0.005884 |

当前核心判断：

```text
1. 本轮 non-overlap32 正式训练证明内部 decoder temporal adapter 路线有效。

2. 两个模型都在单帧质量和时序指标上明显改善：
   MAE/MSE/PSNR 改善，同时 tOF/TWE/kflow/tdelta 全部下降。

3. key-path gate 是当前更优选择：
   PSNR +0.4123 dB，MSE -0.000368，MAE -0.002946；
   时间指标改善略小于 single-scale 的 tOF/TWE，但仍然很强。
   如果以“效果优先，不能牺牲 PSNR/MAE/MSE”为标准，key-path gate 当前胜出。

4. single-scale 的时间指标下降更大：
   tOF -0.4086, TWE -0.00644；
   但 MSE/PSNR 收益弱很多，甚至部分 shard MSE 有轻微波动。
   它说明全图 temporal correction 能强力压低时间抖动，但 key-path gate 更符合质量优先目标。

5. 当前推荐主线：
   key-path gate dual decoder adapter, non-overlap32, stage1 3ep + stage2 5ep full-FT。
```

为了后续补算其他视频指标，已给评估脚本增加 raw clip 保存开关：

```text
/data/fzj/RMDM/sample_dual_decoder_adapter_paired.py
/data/fzj/RMDM/sample_dual_decoder_adapter_variant_paired.py

new args:
--save_npz
--save_frames_dir
```

保存格式：

```text
每个 clip 一个 npz：
clip_index
start
gt, baseline, candidate             # float32, [F,1,H,W], range [0,1]
gt_uint8, baseline_uint8, candidate_uint8
```

当前已启动同协议保存版评估：

```text
single-scale saved:
/data/fzj/RMDM/experiments/fulltrain_nonoverlap32_single_stage2_5ep_ddim20_eval100_saved

key-path gate saved:
/data/fzj/RMDM/experiments/fulltrain_nonoverlap32_keypath_stage2_5ep_ddim20_eval100_saved
```

## 2026-06-30 22:26 CST 更新：全 train split stage2 full-FT 已启动

stage1 adapter-only 已完成，两个 final checkpoint 均已落盘：

```text
single-scale stage1:
/data/fzj/RMDM/checkpoints_fulltrain_dual_decoder_adapter_stage1_alltrain_500/dual_decoder_adapter_final.pth

key-path gate stage1:
/data/fzj/RMDM/checkpoints_fulltrain_dual_decoder_keypathgate_stage1_alltrain_500/dual_decoder_adapter_final.pth
```

当前已启动 stage2 full fine-tuning：

```text
GPU0-3:
  tmux: rmdm_fulltrain_single_stage2
  output:
  /data/fzj/RMDM/checkpoints_fulltrain_dual_decoder_adapter_stage2_fullft_alltrain_1000

GPU4-7:
  tmux: rmdm_fulltrain_keypath_stage2
  output:
  /data/fzj/RMDM/checkpoints_fulltrain_dual_decoder_keypathgate_stage2_fullft_alltrain_1000
```

关键配置：

```text
data:
/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack_ShadowDenoiseV1

split_file: split.json
scene_ids: empty, meaning all train scenes
observed dataset clips: 724500

clip_length: 32
frame_stride: 1
clip_stride: 1
batch_size: 1 per GPU
gradient_accumulation_steps: 4
max_steps: 1000
adapter_lr: 5e-5
base_lr: 1e-6
train_base: true
final_kpath_weight: 0.003
x0_recon_weight: 0.01
stage1_cal_weight: 0.3
stage1_pinn_weight: 0.3
stage1_kpath_weight: 1.0
```

已确认：

```text
single-scale:
  trainable params: 66758625 / 66758625
  step 20 已输出

key-path gate:
  trainable params: 66758625 / 66758625
  step 20 已输出

GPU:
  0-7 均被占用，显存约 23.5-24.1GB
```

启动注意事项：

```text
1. 不要直接用默认 accelerate 配置。
   当前用户默认配置是 DeepSpeed ZeRO-3，会触发 DeepSpeed CPUAdam 编译；
   本机 installed CUDA 12.1 与 torch CUDA 12.8 不匹配，DeepSpeed 编译失败。

2. 正确启动方式是显式指定普通 DDP 配置：
   /data/fzj/RadioDiff-k/configs/accelerate_4gpu_ddp.yaml

3. 参数值为负数时要用等号形式：
   --hwm_adapter_indices=-2,-1
   --stage2_adapter_indices=-2,-1
```

关于用户刚提出的 key-path local attention / adaptive alpha：

```text
这两个方向已经做过 100 clips DDIM20 adapter-only 小实验。

key-path local attention:
  /data/fzj/RMDM/experiments/dual_decoder_keypathattn_r2_cont250_ddim20_eval100/merged/summary_paired_merged.json

adaptive alpha:
  /data/fzj/RMDM/experiments/dual_decoder_adaptivealpha_two_scene_500_ddim20_eval100/merged/summary_paired_merged.json
```

核心判断：

```text
key-path local attention 没有证明优于当前 key-path gate：
  TWE/tdelta 反而变差，说明局部 attention 仍可能引入过度时间混合。

adaptive alpha 有正向但弱于当前主线：
  时间指标有改善，但整体不如 single-scale full-FT 和 key-path gate full-FT。

因此当前 8 卡优先继续跑两条已验证更强的 full-train 主线；
不再抢占当前训练资源重复启动这两个较弱小实验。
```

## 2026-06-30 22:44 CST 更新：全 train split 两阶段训练完成并完成 100 clips DDIM20 评估

训练完成状态：

```text
single-scale dual decoder adapter:
  stage1 adapter-only:
  /data/fzj/RMDM/checkpoints_fulltrain_dual_decoder_adapter_stage1_alltrain_500/dual_decoder_adapter_final.pth

  stage2 full-FT:
  /data/fzj/RMDM/checkpoints_fulltrain_dual_decoder_adapter_stage2_fullft_alltrain_1000/dual_decoder_adapter_final.pth

key-path gate dual decoder adapter:
  stage1 adapter-only:
  /data/fzj/RMDM/checkpoints_fulltrain_dual_decoder_keypathgate_stage1_alltrain_500/dual_decoder_adapter_final.pth

  stage2 full-FT:
  /data/fzj/RMDM/checkpoints_fulltrain_dual_decoder_keypathgate_stage2_fullft_alltrain_1000/dual_decoder_adapter_final.pth
```

验证：

```text
1. 两个 stage2 final checkpoint 均可 torch.load。
2. checkpoint 均包含 base_model，证明是 full-FT 结果，不只是 adapter-only。
3. 两个 checkpoint 均包含 hwm_adapters/stage2_adapters，插入模块也正常保存。
4. stage2 训练日志均显示：
   dataset clips: 724500
   scene_ids: all train scenes
   trainable params: 66758625 / 66758625
```

评估协议：

```text
split: test
num clips: 100
clip_length: 32
frame_stride: 32
DDIM steps: 20
paired baseline: original RMDM model_phy_step65000.pth
seed: 1234
shards: 4 per model

single-scale eval:
/data/fzj/RMDM/experiments/fulltrain_dual_decoder_adapter_stage2_fullft_alltrain_1000_ddim20_eval100/merged/summary_paired_merged.json

key-path gate eval:
/data/fzj/RMDM/experiments/fulltrain_dual_decoder_keypathgate_stage2_fullft_alltrain_1000_ddim20_eval100/merged/summary_paired_merged.json
```

100 clips mean 指标：

| Run | MAE base -> ours | MSE base -> ours | PSNR base -> ours | tOF base -> ours | TWE base -> ours | kflow EPE base -> ours | tdelta L1 base -> ours |
|---|---:|---:|---:|---:|---:|---:|---:|
| single-scale full-train | 0.033420 -> 0.032223 | 0.007093 -> 0.007188 | 23.9181 -> 23.8904 | 0.635207 -> 0.533047 | 0.010413 -> 0.009211 | 1.67925 -> 1.46033 | 0.011955 -> 0.010792 |
| key-path gate full-train | 0.035591 -> 0.034442 | 0.007132 -> 0.007046 | 23.8641 -> 23.9056 | 0.629176 -> 0.544683 | 0.010137 -> 0.009354 | 1.65604 -> 1.50389 | 0.011731 -> 0.010951 |

对应 delta：

| Run | ΔMAE | ΔMSE | ΔPSNR | ΔtOF | ΔTWE | Δkflow EPE | Δtdelta L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| single-scale full-train | -0.001197 | +0.0000947 | -0.02775 dB | -0.10216 | -0.001202 | -0.21892 | -0.001162 |
| key-path gate full-train | -0.001148 | -0.0000851 | +0.04144 dB | -0.08449 | -0.000783 | -0.15215 | -0.000780 |

核心判断：

```text
1. 两条全 train split 两阶段路线都完成了目标：
   先 adapter-only，再 full-FT，且都在全 train split 上训练。

2. single-scale full-train 的时间一致性收益更强：
   tOF/TWE/kflow/tdelta 改善幅度都大于 key-path gate full-train。
   代价是 MSE/PSNR 有轻微退化。

3. key-path gate full-train 更 balanced：
   MAE、MSE、PSNR 都改善，同时 tOF/TWE/kflow/tdelta 也改善。
   如果优先要求“不伤单帧质量”，这一版更适合作为当前 full-train 候选。

4. 与之前 two-scene 小实验相比，全 train 后的时间指标收益变小。
   这说明全数据分布更复杂，100 clips 评估中的 baseline 分布也不同；
但 key-path gate 在全 train 后保住 PSNR/MSE，是更稳的方向。
```

## 2026-07-01 13:40 CST 更新：修正训练量语义，启动 non-overlap 32-frame 正式训练

用户指出：全量训练不能只把 `scene_ids` 设为空，还必须按训练量换算。
前一轮所谓 full-train 只是在全 train split 上做了小步数采样 pilot，不是完整 epoch 级训练。

关键修正：

```text
1. 训练 clip 语义改成不重叠 32 帧视频片段：
   clip_length = 32
   clip_stride = 1      # clip 内连续帧
   frame_stride = 32    # clip 起点间隔 32，避免重叠

2. train split clip 数：
   31500 clips

3. 当前显存探测：
   batch_size_per_gpu = 2 的 stage2 full-FT 会 OOM。
   batch_size_per_gpu = 1, gradient_accumulation_steps = 8 可跑。

4. effective batch:
   4 GPUs * batch 1 * grad_accum 8 = 32 clips / optimizer update

5. 1 epoch:
   ceil(31500 / 32) = 985 optimizer updates
```

训练脚本修正：

```text
/data/fzj/RMDM/train_dual_decoder_adapter.py
/data/fzj/RMDM/train_dual_decoder_adapter_variant.py
```

修正内容：

```text
1. max_steps 现在按真实 optimizer update 计数。
2. 日志新增 micro_step，方便确认梯度累积。
3. optimizer.zero_grad() 移到 backward/step 之后，避免在 accumulate 内提前清掉累积梯度。
```

正式训练计划：

```text
stage1 adapter-only:
  3 epochs ≈ 2954 optimizer updates

stage2 full-FT:
  5 epochs ≈ 4922 optimizer updates
```

当前已启动 stage1 adapter-only：

```text
GPU0-3:
  single-scale dual decoder adapter
  tmux: rmdm_nonoverlap_single_stage1
  output:
  /data/fzj/RMDM/checkpoints_fulltrain_nonoverlap32_single_stage1_3ep_b1a8

GPU4-7:
  key-path gate dual decoder adapter
  tmux: rmdm_nonoverlap_keypath_stage1
  output:
  /data/fzj/RMDM/checkpoints_fulltrain_nonoverlap32_keypath_stage1_3ep_b1a8
```

启动验证：

```text
dataset clips: 31500
scene_ids: all train scenes
trainable params: 288256 / 66758625
step 1 micro_step 8
```

以上证明：

```text
1. 使用的是不重叠 32-frame clip。
2. stage1 只训练 temporal adapter。
3. 梯度累积按 8 个 microbatch 合成 1 个 optimizer update。
```

## 2026-06-30 22:20 CST 更新：全 train split 两条主线并行训练已启动

按用户要求，开始在全 train split 上并行跑两条候选主线：

```text
GPU0-3:
  single-scale dual decoder adapter
  两阶段训练：
    stage1 只训练时间 adapter
    stage2 解冻全模型 full FT

GPU4-7:
  key-path gate dual decoder adapter
  两阶段训练：
    stage1 只训练时间 adapter
    stage2 解冻全模型 full FT
```

当前已启动 stage1 adapter-only：

```text
single-scale stage1 tmux:
  rmdm_fulltrain_single_stage1

key-path gate stage1 tmux:
  rmdm_fulltrain_keypath_stage1
```

stage1 输出目录：

```text
single-scale:
/data/fzj/RMDM/checkpoints_fulltrain_dual_decoder_adapter_stage1_alltrain_500

key-path gate:
/data/fzj/RMDM/checkpoints_fulltrain_dual_decoder_keypathgate_stage1_alltrain_500
```

启动配置：

```text
data:
/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack_ShadowDenoiseV1

split_file:
split.json

scene_ids:
empty, meaning all train scenes

observed dataset clips:
724500

clip_length: 32
frame_stride: 1
clip_stride: 1
batch_size: 1 per GPU
gradient_accumulation_steps: 4
max_steps: 500
adapter_lr: 5e-5
final_kpath_weight: 0.005
x0_recon_weight: 0.01
low_timestep_prob: 0.7
temporal_max_timestep: 300
```

stage1 启动验证：

```text
single-scale:
  dataset clips: 724500
  scene_ids: all train scenes
  trainable params: 288256 / 66758625

key-path gate:
  dataset clips: 724500
  scene_ids: all train scenes
  trainable params: 288256 / 66758625
```

注意：

```text
这次 tmux 命令里的 tee 日志路径在训练脚本创建 save_dir 前被打开，
所以 tee 报了 train.log 目录不存在。
训练本身正常运行，run_config.json 已写入，日志可通过 tmux capture-pane 查看。
后续 stage2 启动时需要先 mkdir -p save_dir，再 tee train.log。
```

下一步：

```text
1. 等两个 stage1 到 500 steps，并确认 final checkpoint 可加载。
2. single-scale stage2 从:
   /data/fzj/RMDM/checkpoints_fulltrain_dual_decoder_adapter_stage1_alltrain_500/dual_decoder_adapter_final.pth
   继续 full FT。

3. key-path gate stage2 从:
   /data/fzj/RMDM/checkpoints_fulltrain_dual_decoder_keypathgate_stage1_alltrain_500/dual_decoder_adapter_final.pth
   继续 full FT。

4. stage2 计划沿用小实验最优 full FT 配置:
   train_base = true
   max_steps = 1000
   adapter_lr = 5e-5
   base_lr = 1e-6
   final_kpath_weight = 0.003
```

## 2026-06-30 22:12 CST 更新：key-path local attention 与 adaptive alpha 小实验

按用户提出的两个方向完成 500-step adapter-only 小实验：

```text
方案 A: key-path guided local temporal attention
  M_key = K2 q70 high region
  h_key = M_key * h
  在每个空间位置做局部时间窗口 attention，半径 2，即 5 帧邻域。
  输出仍然只通过 M_key/gate 作用在关键路径附近。

方案 B: adaptive alpha
  保持 single-scale separable 3D adapter。
  将 residual 注入改成 timestep-conditioned alpha:
    h_out = h + alpha_t * delta_h
  alpha_t = sigmoid(MLP(t / T))
```

新增/修改代码：

```text
/data/fzj/RMDM/dual_decoder_adapter_variants.py
  KeyPathLocalTemporalAttentionAdapter
  AdaptiveAlphaTemporalAdapter

/data/fzj/RMDM/train_dual_decoder_adapter_variant.py
  --adapter_variant keypath_attn
  --adapter_variant adaptive_alpha

/data/fzj/RMDM/sample_dual_decoder_adapter_variant_paired.py
  支持新 variant 的 paired DDIM20 eval。
```

训练 checkpoint：

```text
adaptive alpha:
/data/fzj/RMDM/checkpoints_dual_decoder_adaptivealpha_two_scene_500/dual_decoder_adapter_final.pth

key-path attention:
第一次 500-step 训练 final checkpoint 写入异常，只保留了正常的 step250。
从 step250 继续补训 250 steps 后得到：
/data/fzj/RMDM/checkpoints_dual_decoder_keypathattn_r2_cont250/dual_decoder_adapter_final.pth
```

评估输出：

```text
adaptive alpha:
/data/fzj/RMDM/experiments/dual_decoder_adaptivealpha_two_scene_500_ddim20_eval100/merged/summary_paired_merged.json

key-path attention:
/data/fzj/RMDM/experiments/dual_decoder_keypathattn_r2_cont250_ddim20_eval100/merged/summary_paired_merged.json
```

100 clips DDIM20 对比：

| Run | ΔMAE | ΔMSE | ΔPSNR | ΔtOF | ΔTWE | Δkflow EPE | Δtdelta L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| single-scale adapter, fkpd=0.005 | -0.001381 | +0.000047 | -0.0096 dB | -0.06243 | -0.000511 | -0.11919 | -0.000502 |
| key-path gate adapter, fkpd=0.005 | -0.001033 | +0.000070 | -0.0124 dB | -0.05271 | -0.000060 | -0.13827 | -0.000065 |
| key-path local attention r=2, fkpd=0.005 | -0.001017 | +0.000132 | -0.0370 dB | -0.01699 | +0.000480 | -0.03033 | +0.000478 |
| adaptive alpha, fkpd=0.005 | -0.001122 | +0.000086 | -0.0156 dB | -0.04554 | -0.000204 | -0.12619 | -0.000204 |
| single-scale two-stage best | -0.001517 | -0.0000039 | +0.0146 dB | -0.20183 | -0.001038 | -0.35463 | -0.001084 |
| key-path gate full FT | -0.000717 | -0.000108 | +0.0446 dB | -0.11312 | -0.001316 | -0.30723 | -0.001336 |

判断：

```text
1. key-path local attention adapter-only 失败。
   它虽然轻微改善 MAE，但 PSNR/MSE 变差，TWE/tdelta 也变差。
   局部 attention 仍然比 hard gate 更容易引入时间错位或过度混合。

2. adaptive alpha 有正向，但不够强。
   它比 key-path attention 稳定，tOF/kflow/TWE 都有改善；
   但整体弱于原 single-scale adapter，也弱于 key-path gate 的 kflow 改善。

3. 当前不建议把 key-path local attention 作为主线。
   如果要继续 attention 方向，应当先尝试更弱的 content gate，
   而不是直接做局部 temporal attention。

4. adaptive alpha 可以保留为后续 full-FT 或和 key-path gate 组合的候选，
   但单独 adapter-only 没有超过当前主线。
```

## 2026-06-30 21:46 CST 更新：variant 架构全量解冻两阶段训练

按用户要求，对上一节两个 variant 架构继续做“和当前 best 一样”的全量解冻训练：

```text
阶段 1:
  从各自 adapter-only 500-step checkpoint 开始。

阶段 2:
  解冻全部 RMDM/base 参数。
  adapter_lr = 5e-5
  base_lr = 1e-6
  final_kpath_weight = 0.003
  max_steps = 1000
  其他 loss 与当前主线一致。
```

训练 checkpoint：

```text
multi-scale r=2 full FT:
/data/fzj/RMDM/checkpoints_dual_decoder_multiscale_r2_two_stage_fullft_1000/dual_decoder_adapter_final.pth

key-path gate full FT:
/data/fzj/RMDM/checkpoints_dual_decoder_keypathgate_two_stage_fullft_1000/dual_decoder_adapter_final.pth
```

评估输出：

```text
multi-scale r=2 full FT:
/data/fzj/RMDM/experiments/dual_decoder_multiscale_r2_two_stage_fullft_1000_ddim20_eval100/merged/summary_paired_merged.json

key-path gate full FT:
/data/fzj/RMDM/experiments/dual_decoder_keypathgate_two_stage_fullft_1000_ddim20_eval100/merged/summary_paired_merged.json
```

100 clips DDIM20 对比：

| Run | ΔMAE | ΔMSE | ΔPSNR | ΔtOF | ΔTWE | Δkflow EPE | Δtdelta L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| single-scale two-stage best | -0.001517 | -0.0000039 | +0.0146 dB | -0.20183 | -0.001038 | -0.35463 | -0.001084 |
| multi-scale r=2 adapter-only | -0.000908 | +0.000135 | -0.0408 dB | -0.02191 | +0.000479 | -0.04154 | +0.000478 |
| multi-scale r=2 full FT | -0.002528 | +0.0000736 | +0.0320 dB | -0.05480 | -0.000594 | -0.17119 | -0.000609 |
| key-path gate adapter-only | -0.001033 | +0.0000696 | -0.0124 dB | -0.05271 | -0.000060 | -0.13827 | -0.000065 |
| key-path gate full FT | -0.000717 | -0.0001084 | +0.0446 dB | -0.11312 | -0.001316 | -0.30723 | -0.001336 |

判断：

```text
1. 全量解冻对两个 variant 都是有效的。
   multi-scale 从 TWE/tdelta 变差，变成全时间指标正向；
   key-path gate 从很弱的 TWE/tdelta 改善，变成 TWE/tdelta 最强。

2. multi-scale r=2 full FT 的特点：
   MAE 改善最大，PSNR 也正向；
   但 tOF/TWE/kflow 改善明显弱于 single-scale two-stage best。
   它更像偏单帧质量/平滑修正，不是当前最好的时序一致性方案。

3. key-path gate full FT 的特点：
   TWE 和 temporal-delta L1 优于 single-scale two-stage best；
   PSNR/MSE 也正向；
   但 MAE 改善小，tOF/kflow EPE 仍弱于 single-scale best。

4. 当前如果按综合指标选主线，仍建议保留 single-scale two-stage best：
   它的 tOF、kflow EPE、kflow L1 和 MAE 更强，且 PSNR 不掉。

5. 如果后续重点转向 TWE/tdelta，可以把 key-path gate full FT 作为候选路线。
   它说明 K2-guided gate 在全量低学习率微调后确实有用，
   但需要解决 MAE/tOF/kflow 相对 single-scale 不够强的问题。
```

## 2026-06-30 21:30 CST 更新：multi-scale adapter 与 key-path gate adapter 架构消融

按用户提出的两个新结构方向，完成 100 clips 小实验验证：

```text
方案 A: multi-scale temporal adapter
  Branch 1: 3x1x1 temporal conv
  Branch 2: 5x1x1 temporal conv
  Branch 3: dilated 3x1x1 temporal conv
  Fuse: 1x1x1 conv

方案 B: key-path guided adapter
  仍使用单尺度 separable 3D adapter。
  用 K2 q70 key-path mask 对 adapter 输出做空间门控。
```

新增代码：

```text
/data/fzj/RMDM/dual_decoder_adapter_variants.py
/data/fzj/RMDM/train_dual_decoder_adapter_variant.py
/data/fzj/RMDM/sample_dual_decoder_adapter_variant_paired.py
```

训练和评估设置：

```text
train scenes:
  town10_junction_0189
  town02_opt_junction_0298

steps:
  500

loss:
  与 dual decoder adapter 主线一致。
  final_kpath_weight = 0.005

eval:
  split=test
  num_clips=100
  clip_length=32
  frame_stride=32
  DDIM20
  paired same initial noise
```

GPU 使用：

```text
multi-scale:
  先用 GPU0-3 跑 reduction=1，显存 OOM。
  保持 clip/eval 配置不变，将 adapter bottleneck 改为 reduction=2 后完成训练。

key-path gate:
  GPU4-7 完成训练。

两个方案训练完成后分别用 4 GPU shard 做 100 clips 评估。
```

checkpoint：

```text
multi-scale r=2:
/data/fzj/RMDM/checkpoints_dual_decoder_multiscale_r2_two_scene_500/dual_decoder_adapter_final.pth

key-path gate:
/data/fzj/RMDM/checkpoints_dual_decoder_keypathgate_two_scene_500/dual_decoder_adapter_final.pth
```

评估输出：

```text
multi-scale r=2:
/data/fzj/RMDM/experiments/dual_decoder_multiscale_r2_two_scene_500_ddim20_eval100/merged/summary_paired_merged.json

key-path gate:
/data/fzj/RMDM/experiments/dual_decoder_keypathgate_two_scene_500_ddim20_eval100/merged/summary_paired_merged.json
```

核心对比：

| Run | ΔMAE | ΔMSE | ΔPSNR | ΔtOF | ΔTWE | Δkflow EPE | Δtdelta L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| single-scale adapter, fkpd=0.005 | -0.001381 | +0.000047 | -0.0096 dB | -0.06243 | -0.000511 | -0.11919 | -0.000502 |
| single-scale adapter, fkpd=0.003 | -0.001497 | +0.000064 | +0.0092 dB | -0.03686 | -0.000347 | -0.09277 | -0.000363 |
| multi-scale r=2, fkpd=0.005 | -0.000908 | +0.000135 | -0.0408 dB | -0.02191 | +0.000479 | -0.04154 | +0.000478 |
| key-path gate, fkpd=0.005 | -0.001033 | +0.000070 | -0.0124 dB | -0.05271 | -0.000060 | -0.13827 | -0.000065 |
| current best two-stage full FT | -0.001517 | -0.0000039 | +0.0146 dB | -0.20183 | -0.001038 | -0.35463 | -0.001084 |

判断：

```text
1. multi-scale r=2 不适合作为当前主线。
   它虽然轻微改善 MAE/tOF/kflow，但 PSNR/MSE 变差，
   TWE 和 temporal-delta 反而变差，说明多尺度分支短训下可能引入更强平滑或错位。

2. key-path gate 比 multi-scale 更合理。
   它对 tOF/kflow 的收益更强，也没有像 multi-scale 那样伤 TWE/tdelta。
   但它仍弱于原 single-scale adapter，尤其 TWE/tdelta 改善太小，PSNR 也小幅下降。

3. 当前不建议替换主线 single-scale adapter。
   最好的实证路线仍是：
     single-scale dual decoder adapter
     先 0.005 只训时间模块
     再 0.003 + base_lr=1e-6 做低学习率全量微调。

4. 如果继续探索 key-path guided 方向，更合理的下一步不是直接 hard gate adapter 输出，
   而是把 key-path mask 只用于 loss weighting 或训练时 soft regularization。
   直接门控特征修正可能限制了非 K2 高值区域的必要时序校正。
```

## 2026-06-30 20:58 CST 更新：两阶段训练验证完成，当前小实验最强配置

按用户建议完成一版两阶段训练：

```text
阶段 1:
  使用已有 dual decoder adapter，final_kpath_weight=0.005。
  只训练时间 adapter。

阶段 2:
  从阶段 1 checkpoint 继续。
  解冻 base RMDM，全模型低学习率微调。
  final_kpath_weight 降到 0.003。
  训练 1000 steps。
```

阶段 2 checkpoint：

```text
/data/fzj/RMDM/checkpoints_dual_decoder_adapter_two_stage_fullft_1000/dual_decoder_adapter_final.pth
```

阶段 2 训练配置：

```text
data:
  /data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack_ShadowDenoiseV1

train scenes:
  town10_junction_0189
  town02_opt_junction_0298

init RMDM:
  /data/fzj/RMDM/checkpoints_dynamic_rmdm_4gpu_no_ckpt_75k/model_phy_step65000.pth

resume adapter:
  /data/fzj/RMDM/checkpoints_dual_decoder_adapter_two_scene_500/dual_decoder_adapter_final.pth

GPUs:
  0,1,2,3

clip_length: 32
frame_stride: 1
batch_size: 1 per GPU
gradient_accumulation_steps: 4
max_steps: 1000
adapter_lr: 5e-5
base_lr: 1e-6
final_kpath_weight: 0.003
x0_recon_weight: 0.01
low_timestep_prob: 0.7
temporal_max_timestep: 300
```

评估协议：

```text
split: test
num_clips: 100
clip_length: 32
frame_stride: 32
DDIM steps: 20
paired same initial noise
4 GPU shards:
  0-24, 25-49, 50-74, 75-99
```

评估输出：

```text
/data/fzj/RMDM/experiments/dual_decoder_adapter_two_stage_fullft_1000_ddim20_eval100/merged/summary_paired_merged.json
/data/fzj/RMDM/experiments/dual_decoder_adapter_two_stage_fullft_1000_ddim20_eval100/merged/per_clip_paired_merged.csv
```

100 clips DDIM20 结果：

| Metric | Frozen RMDM | Two-stage candidate | Delta |
|---|---:|---:|---:|
| MAE | 0.035624 | 0.034107 | -0.001517 |
| MSE | 0.007128 | 0.007124 | -0.0000039 |
| PSNR | 23.7611 | 23.7757 | +0.0146 dB |
| tOF | 0.676356 | 0.474527 | -0.201829 |
| TWE | 0.012931 | 0.011893 | -0.001038 |
| kflow EPE | 2.032873 | 1.678240 | -0.354633 |
| kflow L1 | 1.287085 | 1.059285 | -0.227799 |
| temporal-dL1 | 0.014465 | 0.013381 | -0.001084 |
| temporal-dMSE | 0.001541 | 0.001222 | -0.000319 |

和前两版 adapter-only 对比：

| Run | ΔMAE | ΔPSNR | ΔtOF | ΔTWE | Δkflow EPE | Δtdelta L1 |
|---|---:|---:|---:|---:|---:|---:|
| dual adapter only, fkpd=0.005, 500 steps | -0.001381 | -0.0096 dB | -0.06243 | -0.000511 | -0.11919 | -0.000502 |
| dual adapter only, fkpd=0.003, 500 steps | -0.001497 | +0.0092 dB | -0.03686 | -0.000347 | -0.09277 | -0.000363 |
| two-stage, 0.005 adapter -> full FT 0.003, 1000 steps | -0.001517 | +0.0146 dB | -0.20183 | -0.001038 | -0.35463 | -0.001084 |

核心判断：

```text
1. 这次两阶段训练是当前 100-clip 小实验里最强的结果。
2. 它没有牺牲 PSNR/MAE/MSE，反而 MAE、MSE、PSNR 都小幅改善。
3. 时间指标提升明显大于 adapter-only：tOF、TWE、kflow、tdelta 全部显著下降。
4. 之前“不要马上全解冻”的判断在本次低 base_lr 设置下需要修正：
   全量解冻不是问题本身，关键是先有稳定时间 adapter 初始化，再用很小 base_lr 微调。
5. 当前推荐作为扩大训练前的主配置：
   dual decoder adapter
   先 0.005 只训时间模块
   再 0.003 + base_lr=1e-6 全量低学习率微调
```

相关代码位置：

```text
/data/fzj/RMDM/dual_decoder_adapter_student.py
  DualDecoderAdapterStudent
  train_base=True 时允许 base RMDM 参与训练，并在 checkpoint 中保存 base_model。

/data/fzj/RMDM/train_dual_decoder_adapter.py
  --train_base
  --base_lr
  adapter/base 分 optimizer param groups。

/data/fzj/RMDM/sample_dual_decoder_adapter_paired.py
  评估时如果 checkpoint 内有 base_model，则加载全量微调后的 base RMDM 权重。
```






## 2026-06-30 20:40 CST 更新：final_kpath_weight 0.003 权重消融

针对用户提出的“如何保证时间指标不下降，可以把权重稍微降低”的建议，完成一版只改最终 key-path/k-flow loss 权重的小实验。

变量：

```text
保持 dual decoder adapter 结构不变。
final_kpath_weight: 0.005 -> 0.003
训练步数: 500
训练数据/评估协议保持一致。
```

checkpoint：

```text
/data/fzj/RMDM/checkpoints_dual_decoder_adapter_two_scene_fkpd003_500/dual_decoder_adapter_final.pth
```

评估输出：

```text
/data/fzj/RMDM/experiments/dual_decoder_adapter_two_scene_fkpd003_500_ddim20_eval100/merged/summary_paired_merged.json
/data/fzj/RMDM/experiments/dual_decoder_adapter_two_scene_fkpd003_500_ddim20_eval100/merged/per_clip_paired.csv
```

100 clips DDIM20 结果：

```text
baseline MAE:        0.034099
candidate MAE:       0.032602
delta MAE:          -0.001497

baseline MSE:        0.007127
candidate MSE:       0.007191
delta MSE:          +0.000064

baseline PSNR:       23.8332
candidate PSNR:      23.8423
delta PSNR:         +0.0092 dB

baseline tOF:        0.651921
candidate tOF:       0.615064
delta tOF:          -0.036856

baseline TWE:        0.010679
candidate TWE:       0.010332
delta TWE:          -0.000347

baseline kflow EPE:  1.670704
candidate kflow EPE: 1.577931
delta kflow EPE:   -0.092772

baseline kflow L1:   1.058993
candidate kflow L1:  0.999703
delta kflow L1:    -0.059290

baseline tdelta L1:  0.012261
candidate tdelta L1: 0.011898
delta tdelta L1:   -0.000363
```

和 `final_kpath_weight=0.005` 对比：

```text
fkpd=0.005:
  PSNR -0.0096 dB, MAE -0.001381
  tOF -0.06243, TWE -0.000511, kflow EPE -0.11919, tdelta L1 -0.000502

fkpd=0.003:
  PSNR +0.0092 dB, MAE -0.001497
  tOF -0.03686, TWE -0.000347, kflow EPE -0.09277, tdelta L1 -0.000363
```

判断：

```text
1. 降低 final_kpath_weight 到 0.003 后，PSNR 从小降变成小升，MAE 改善也略强。
2. 时间指标改善幅度比 0.005 小一些，但仍显著优于 prior-only 和 joint output residual。
3. 如果目标是“保证时间指标不下降，同时尽量不伤 PSNR”，0.003 比 0.005 更均衡。
4. 当前推荐小实验最佳配置: dual decoder adapter + final_kpath_weight=0.003。
```

关于“500 步只训时间模块，再 500 步解冻所有模块”：

```text
不建议马上全解冻所有 RMDM trunk。当前双阶段 adapter 已经得到清晰正信号；全解冻会引入很大变量，可能破坏 frozen RMDM 的单帧能力。
更稳妥的下一步是：
  1. 保持 frozen base，只把 dual adapter 训练到 1000 steps；
  2. 或只解冻最后 final conv / output norm 等极少数层，低学习率；
  3. 先在 500/1000 clips eval 上确认 0.003 的稳定性，再扩大训练。
```

## 2026-06-30 20:25 CST 更新：双阶段同位置 decoder temporal adapter + 最终 k-flow/key-path loss

按用户新方案完成一版小实验：第一阶段和第二阶段都在靠近输出的位置插入 temporal adapter，并对最终 `pred_x0` 计算 key-path/k-flow temporal loss。

核心区别：

```text
上一版 joint residual:
  Stage-1 HWM decoder adapter + Stage-2 output residual head
  结果: 单帧质量改善，但时间指标回退。

本版 dual decoder adapter:
  Stage-1 HWM decoder adapter + Stage-2 diffusion decoder adapter
  不使用 output residual head。
  结果: MAE 和全部时间指标同时改善，PSNR 只小幅下降。
```

新增代码：

```text
/data/fzj/RMDM/dual_decoder_adapter_student.py
/data/fzj/RMDM/train_dual_decoder_adapter.py
/data/fzj/RMDM/sample_dual_decoder_adapter_paired.py
```

插入位置：

```text
Stage-1 prior/HWM U-Net decoder:
  hwm.conv_blocks_localization[-2]
  hwm.conv_blocks_localization[-1]

Stage-2 diffusion U-Net decoder:
  output_blocks[-2]
  output_blocks[-1]
```

Stage-2 末端 block 通道：

```text
output_blocks[13]: 96 channels
output_blocks[14]: 96 channels
```

adapter 形式：

```text
x_out = x + Adapter3D(x)
Adapter3D = GN -> 1x1x1 down -> 3x1x1 temporal conv -> 1x3x3 spatial conv -> zero-init 1x1x1 up
```

训练设置：

```text
base RMDM: frozen
Stage-1 adapter init: /data/fzj/RMDM/checkpoints_hwm_decoder_adapter_two_scene_kpd1_relstrong_500/hwm_decoder_adapter_final.pth
train scenes: town10_junction_0189, town02_opt_junction_0298
clip_length: 32
4 GPUs: 0,1,2,3
batch_size: 1/GPU
gradient_accumulation_steps: 4
steps: 500
trainable params: 288,256
```

checkpoint：

```text
/data/fzj/RMDM/checkpoints_dual_decoder_adapter_two_scene_500/dual_decoder_adapter_final.pth
```

loss：

```text
L = 1.0 * L_diff(eps, noise)
  + 0.01 * L_x0_recon(low timestep)
  + 0.3 * L_cal_recon(cal, gt)
  + 0.3 * L_PINN(cal)
  + 1.0 * L_kpath_delta(cal, gt, k2)
  + 0.005 * L_kpath_delta(pred_x0, gt, k2, low timestep)
```

评估：

```text
100 test clips
DDIM20
same initial noise paired eval
4 GPU shards
```

输出：

```text
/data/fzj/RMDM/experiments/dual_decoder_adapter_two_scene_500_ddim20_eval100/merged/summary_paired_merged.json
/data/fzj/RMDM/experiments/dual_decoder_adapter_two_scene_500_ddim20_eval100/merged/per_clip_paired.csv
```

100 clips DDIM20 结果：

```text
baseline MAE:        0.034120
candidate MAE:       0.032739
delta MAE:          -0.001381

baseline MSE:        0.007131
candidate MSE:       0.007178
delta MSE:          +0.000047

baseline PSNR:       23.8276
candidate PSNR:      23.8180
delta PSNR:         -0.0096 dB

baseline tOF:        0.657169
candidate tOF:       0.594735
delta tOF:          -0.062434

baseline TWE:        0.010682
candidate TWE:       0.010171
delta TWE:          -0.000511

baseline kflow EPE:  1.680399
candidate kflow EPE: 1.561205
delta kflow EPE:   -0.119194

baseline kflow L1:   1.065272
candidate kflow L1:  0.989047
delta kflow L1:    -0.076226

baseline tdelta L1:  0.012244
candidate tdelta L1: 0.011742
delta tdelta L1:   -0.000502
```

和前两版对比：

```text
prior-only direct insert:
  PSNR -0.0126 dB, MAE +0.000582
  tOF -0.00659, TWE -0.0000767, kflow EPE -0.00902, tdelta L1 -0.0000928

joint output residual:
  PSNR +0.0053 dB, MAE -0.001523
  tOF +0.01038, TWE +0.000066, kflow EPE +0.00635, tdelta L1 +0.000060

dual decoder adapter:
  PSNR -0.0096 dB, MAE -0.001381
  tOF -0.06243, TWE -0.000511, kflow EPE -0.11919, tdelta L1 -0.000502
```

当前判断：

```text
1. 用户提出的“双阶段同位置 temporal adapter”方向明显优于 output residual head。
2. 本版是目前第一个同时改善 MAE 和所有时序指标的版本。
3. PSNR 小降 0.01 dB，幅度很小；可以接受作为小实验正信号，但后续大训练前还需要扩大 eval clips。
4. 关键原因可能是 Stage-2 decoder adapter 以 feature residual 方式局部修正，不像 output residual head 直接改 eps，因此不会强行覆盖 prior-only 的时序收益。
5. 下一步优先沿这个方向做更大验证，而不是继续 output residual head。
```

建议下一步：

```text
A. 在 500 或 1000 test clips 上复评 dual decoder adapter，确认不是 100 clips 抽样偶然。
B. 尝试 final_kpath_weight = 0.003，看看能否保住同等时间收益同时减少 PSNR 小降。
C. 如果 500/1000 clips 稳定，再扩大训练数据/步数。
```

## 2026-06-30 19:55 CST 更新：正式联合训练 500 step 小实验

已完成用户要求的联合训练第一版：Stage-1 HWM decoder adapter + Stage-2 output residual head 同时训练。

新增代码：

```text
/data/fzj/RMDM/joint_hwm_residual_student.py
/data/fzj/RMDM/train_joint_hwm_residual.py
/data/fzj/RMDM/sample_joint_hwm_residual_paired.py
```

训练设置：

```text
base RMDM: frozen
Stage-1: HWM decoder adapter，从 prior-only 最好 checkpoint 初始化
Stage-2: zero-init output residual head
residual_alpha: 0.03
train scenes: town10_junction_0189, town02_opt_junction_0298
clip_length: 32
pair_stride: 1
4 GPUs: 0,1,2,3
batch_size: 1/GPU
gradient_accumulation_steps: 4
steps: 500
```

初始化 Stage-1 adapter：

```text
/data/fzj/RMDM/checkpoints_hwm_decoder_adapter_two_scene_kpd1_relstrong_500/hwm_decoder_adapter_final.pth
```

联合训练 checkpoint：

```text
/data/fzj/RMDM/checkpoints_joint_hwm_residual_two_scene_500/joint_hwm_residual_final.pth
```

训练 loss：

```text
L = 1.0 * L_diff(eps_final, noise)
  + 0.05 * L_x0_recon(low timestep)
  + 0.3 * L_cal_recon(cal, gt)
  + 0.3 * L_PINN(cal)
  + 1.0 * L_kpath_delta(cal, gt, k2)
  + 0.001 * L_kpath_delta(pred_x0, gt, k2, low timestep)
```

重要实现点：

```text
1. 原始 RMDM 参数全部 frozen。
2. 联合 wrapper 手动 forward diffusion trunk。
3. anchor gate 不 detach，使最终 diffusion/x0 loss 能回传到 Stage-1 HWM adapter。
4. 4 GPU 每卡 batch=4 会 OOM，因为联合训练必须保留完整 trunk 激活；实际使用每卡 batch=1 + grad accum 4。
```

100 clips DDIM20 paired eval：

```text
输出目录:
/data/fzj/RMDM/experiments/joint_hwm_residual_two_scene_500_ddim20_eval100/merged/summary_paired_merged.json
```

结果：

```text
baseline MAE:        0.034042
candidate MAE:       0.032519
delta MAE:          -0.001523

baseline MSE:        0.007108
candidate MSE:       0.007170
delta MSE:          +0.000061

baseline PSNR:       23.8381
candidate PSNR:      23.8434
delta PSNR:         +0.0053 dB

baseline tOF:        0.640406
candidate tOF:       0.650786
delta tOF:          +0.010381

baseline TWE:        0.010599
candidate TWE:       0.010665
delta TWE:          +0.000066

baseline kflow EPE:  1.663777
candidate kflow EPE: 1.670130
delta kflow EPE:   +0.006353

baseline tdelta L1:  0.012171
candidate tdelta L1: 0.012231
delta tdelta L1:   +0.000060
```

和 prior-only 直接接入对比：

```text
prior-only direct insert:
  PSNR -0.0126 dB, MAE +0.000582
  tOF -0.00659, TWE -0.0000767, kflow EPE -0.00902, tdelta L1 -0.0000928

joint 500 step:
  PSNR +0.0053 dB, MAE -0.001523
  tOF +0.01038, TWE +0.000066, kflow EPE +0.00635, tdelta L1 +0.000060
```

判断：

```text
1. 联合训练确实让最终生成服务单帧质量：MAE 明显改善，PSNR 基本持平略升。
2. 但时间指标回退，说明当前联合 loss 中 diffusion/x0/单帧项压过了时序项。
3. prior-only 版本证明 Stage-1 adapter 可以改善时序；joint 版本证明最终生成 loss 可以拉回单帧质量。
4. 下一轮联合训练应把 final pred_x0 的 temporal/kpath 权重提高，或者降低 residual_alpha / residual head 学习率，避免第二阶段 residual head 过度改写时序收益。
```

下一步建议配置：

```text
A. 保守联合：residual_alpha=0.01, final_kpath_weight=0.003 或 0.005。
B. 时序强化联合：保持 residual_alpha=0.03, final_kpath_weight=0.005, x0_recon_weight 降到 0.01。
C. 只训练 Stage-1 adapter + final temporal loss，不训练 output residual head，用来验证是否是 residual head 抹掉了 prior-only 的时序收益。
```

## 2026-06-30 17:15 CST 更新：Stage-1 adapter 接入完整 DDIM 的传导性验证

用户明确拆分：第一阶段路线也分两步。

```text
Step 1: 先把只训练 Stage-1 prior adapter 得到的模块直接接回完整 RMDM 采样链路，冻结第二阶段 diffusion，验证是否能传导到最终生成。
Step 2: 如果 Step 1 有信号，再做正式联合训练，让最终 diffusion loss / temporal loss 回传到 Stage-1 adapter 和第二阶段模块。
```

本次完成 Step 1。

新增脚本：

```text
/data/fzj/RMDM/sample_hwm_decoder_adapter_paired.py
```

评估方式：

```text
baseline: 原始 frozen RMDM 完整 DDIM20
candidate: 同一个 frozen RMDM diffusion trunk，但 highway_forward(c) 使用 HWM decoder adapter 产生的 anch/cal
noise: baseline/candidate 使用同一个初始噪声
split: test
num_clips: 100
clip_length: 32
frame_stride: 32
GPU: 0,1,2,3 四个 shard 并行
```

使用的 Stage-1 adapter checkpoint：

```text
/data/fzj/RMDM/checkpoints_hwm_decoder_adapter_two_scene_kpd1_relstrong_500/hwm_decoder_adapter_final.pth
```

输出：

```text
/data/fzj/RMDM/experiments/hwm_decoder_adapter_kpd1_relstrong_ddim20_eval100/merged/summary_paired_merged.json
/data/fzj/RMDM/experiments/hwm_decoder_adapter_kpd1_relstrong_ddim20_eval100/merged/per_clip_paired.csv
```

100 clips DDIM20 结果：

```text
baseline MAE:        0.034055
candidate MAE:       0.034637
delta MAE:          +0.000582

baseline MSE:        0.007104
candidate MSE:       0.007119
delta MSE:          +0.000015

baseline PSNR:       23.8409
candidate PSNR:      23.8283
delta PSNR:         -0.0126 dB

baseline tOF:        0.663292
candidate tOF:       0.656700
delta tOF:          -0.006592

baseline TWE:        0.010688
candidate TWE:       0.010611
delta TWE:          -0.0000767

baseline kflow EPE:  1.669383
candidate kflow EPE: 1.660362
delta kflow EPE:   -0.009021

baseline tdelta L1:  0.012258
candidate tdelta L1: 0.012165
delta tdelta L1:   -0.0000928
```

判断：

```text
1. Stage-1 adapter 的时序信号可以传导到最终 DDIM 生成：tOF/TWE/kflow/tdelta 都有小幅改善。
2. 单帧质量略降：MAE/MSE 上升，PSNR 下降 0.013 dB。下降很小，但说明 prior adapter 与第二阶段 diffusion 目标没有完全对齐。
3. 这支持继续做 Step 2：正式联合训练，而不是只单独训练 prior。
4. 联合训练时应保留当前 Stage-1 adapter 和 kpath_delta loss 结构，同时让最终生成相关 loss 参与，使 prior adapter 服务最终 RSS 生成，而不是只服务 cal prior。
```

下一步建议：

```text
A. Stage-1 路线 Step 2：从该 adapter checkpoint 初始化，训练 Stage-1 adapter + 第二阶段轻量 residual/output module；loss 包含 diffusion noise loss、最终 pred_x0/temporal loss、Stage-1 cal/PINN/kpath loss。
B. 第二阶段单独路线：保持原始 prior 不变，只训练第二阶段 temporal residual/adapter，并用最终生成的 kpath/temporal loss 约束。
C. 两条路线都先在小数据 100 clips eval 上跑，不直接上全量。
```

## 2026-06-30 17:06 CST 更新：Stage-1 HWM decoder adapter 与提高 k-flow/关键路径权重

当前阶段的结论：单独训练第一阶段 prior adapter 可以显著改善 `cal` 的单帧重建和 PINN，但时空一致性改善很弱；只把 `kpath_delta_weight` 从 `0.2` 提到 `1.0` 仍不够。相对降低 `cal_recon/PINN` 后，关键路径差分 loss 才开始略微下降，但整体 temporal delta 仍变差。

### 当前 Stage-1 adapter 结构

不修改原始：

```text
/data/fzj/RMDM/unet.py
```

新文件：

```text
/data/fzj/RMDM/hwm_decoder_adapter_student.py
/data/fzj/RMDM/train_hwm_decoder_adapter.py
/data/fzj/RMDM/eval_hwm_decoder_adapter_prior.py
```

插入位置：原 RMDM 第一阶段 HWM/PINN U-Net 的 decoder/localization 末端两个 block：

```text
hwm.conv_blocks_localization[-2]
hwm.conv_blocks_localization[-1]
```

adapter 形式：

```text
x_out = x + Adapter3D(x)
Adapter3D = GN -> 1x1x1 down -> 3x1x1 temporal conv -> 1x3x3 spatial conv -> 1x1x1 up
```

设计理由：

```text
1. frozen 原 RMDM/HWM 参数，避免破坏单帧能力。
2. adapter 最后一层 zero-init，训练起点等价 frozen baseline。
3. 不使用 temporal self-attention，不加 gate，先验证局部 3D conv 的时空归纳偏置。
4. adapter 插在第一阶段 prior 的 decoder 末端，让 cal prior 自身具备局部时间一致性。
```

2026-06-30 的代码修正：

```text
/data/fzj/RMDM/hwm_decoder_adapter_student.py
  去掉 adapter 输入处的 x.detach()。
```

原因：冻结 base 参数不等于截断 feature 梯度。去掉 detach 后，倒数第二个 adapter 可以通过后续 frozen decoder block 和最后一个 adapter 接收最终 cal loss 的梯度，更符合“插入 decoder block 后训练 adapter”的目标。

### 当前 Stage-1 训练 loss

```text
L = w_cal * MSE(cal, gt_rss)
  + w_pinn * PINN(cal)
  + w_kpd * L_kpath_delta(cal, gt_rss, k2)
```

关键路径差分 loss：

```text
M_static  = key_path(K2 q70) ∩ low_kflow(GT flow q70)
M_dynamic = key_path(K2 q70) ∩ high_kflow(GT flow q85)

Δpred = |cal[t+1] - cal[t]|
Δgt   = |gt[t+1] - gt[t]|

L_kpath_delta =
  mean(M_static  * |Δpred - clip(Δpred, 0, margin)|)
+ λ_dynamic * mean(M_dynamic * |Δpred - Δgt|)
```

当前默认：

```text
margin = 0.005
λ_dynamic = 1.0
pair_stride = 1
```

### 100 clips prior eval 结果

评估协议：

```text
split: test
num_clips: 100
clip_length: 32
frame_stride: 32
GPU: 0,1,2,3 手动分片评估
baseline: frozen HWM prior
```

数据集：

```text
/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack_ShadowDenoiseV1
```

原始 RMDM checkpoint：

```text
/data/fzj/RMDM/checkpoints_dynamic_rmdm_4gpu_no_ckpt_75k/model_phy_step65000.pth
```

实验 A：`w_cal=1.0, w_pinn=1.0, w_kpd=0.2`

```text
checkpoint: /data/fzj/RMDM/checkpoints_hwm_decoder_adapter_two_scene_kpd02_500/hwm_decoder_adapter_final.pth
summary:    /data/fzj/RMDM/experiments/hwm_decoder_adapter_two_scene_kpd02_500_prior_eval100/summary.json

base MAE/PSNR/PINN/KPD: 0.053706 / 19.6025 / 0.005449 / 0.003635  （该次曾有 DDP padding，112 clips，仅作趋势参考）
cand MAE/PSNR/PINN/KPD: 0.052298 / 20.1749 / 0.004201 / 0.003714
判断: 单帧 prior 和 PINN 改善，KPD/tdelta 未改善。
```

实验 B：`w_cal=1.0, w_pinn=1.0, w_kpd=1.0`

```text
checkpoint: /data/fzj/RMDM/checkpoints_hwm_decoder_adapter_two_scene_kpd1_500/hwm_decoder_adapter_final.pth
summary:    /data/fzj/RMDM/experiments/hwm_decoder_adapter_two_scene_kpd1_500_prior_eval100/summary.json

base MAE:       0.054009
cand MAE:       0.051165
base PSNR:      19.9133
cand PSNR:      20.6630
base tdelta:    0.001650
cand tdelta:    0.001783
base PINN:      0.005471
cand PINN:      0.004494
base KPD:       0.005854
cand KPD:       0.005859
base KPD_dyn:   0.005531
cand KPD_dyn:   0.005547
base pred_dyn:  0.001257
cand pred_dyn:  0.001346
gt dyn:         0.005516

判断: 单纯把 w_kpd 提到 1.0 仍主要提升单帧 prior；KPD 基本不变且 dynamic 分量略差。
```

实验 C：`w_cal=0.3, w_pinn=0.3, w_kpd=1.0`

```text
checkpoint: /data/fzj/RMDM/checkpoints_hwm_decoder_adapter_two_scene_kpd1_relstrong_500/hwm_decoder_adapter_final.pth
summary:    /data/fzj/RMDM/experiments/hwm_decoder_adapter_two_scene_kpd1_relstrong_500_prior_eval100/summary.json

base MAE:       0.054009
cand MAE:       0.051113
base PSNR:      19.9133
cand PSNR:      20.7494
base tdelta:    0.001650
cand tdelta:    0.001869
base PINN:      0.005471
cand PINN:      0.005174
base KPD:       0.005854
cand KPD:       0.005785
base KPD_static:0.000323
cand KPD_static:0.000287
base KPD_dyn:   0.005531
cand KPD_dyn:   0.005498
base pred_dyn:  0.001257
cand pred_dyn:  0.001368
gt dyn:         0.005516

判断: 相对强化 kpath 后，KPD/KPD_dyn 开始小幅下降，说明权重比例确实有作用；但 temporal delta 仍变差，pred_dyn 仍远小于 gt_dyn，说明 Stage-1 prior adapter 只靠这个 loss 还不能充分解决最终视频时序一致性。
```

### 当前决策

```text
1. 用户判断成立：单独 stage-1 训练更容易被 cal_recon/PINN 主导，时空指标提升不明显不能证明方法无效。
2. 增大 k-flow/关键路径项总权重是必要诊断，但单纯 w_kpd=1.0 不够。
3. 相对降低 cal/PINN 权重后，KPD 有小幅改善，说明后续应调“权重比例”和 dynamic 项，而不是只继续堆总权重。
4. 目前这仍是 prior eval，不等价最终 DDIM 视频结果。若要服务最终效果，下一步应把更强 prior adapter 接入第二阶段采样/联合训练，或回到 frozen RMDM + residual 3D head 的最终输出方案做 kpath 权重提升。
```

## 0. 当前方法定义、公式和模型改动

### 0.1 k-flow 计算公式

对相邻两帧标量场：

```text
κ_t, κ_{t+1}
```

先计算空间梯度和时间差分：

```text
∂xκ_t, ∂yκ_t = spatial_gradient(κ_t)
Δκ = κ_{t+1} - κ_t
D = (∂xκ_t)^2 + (∂yκ_t)^2 + ε
```

k-flow 定义为最小范数法向传播流：

```text
u = - Δκ * ∂xκ_t / D
v = - Δκ * ∂yκ_t / D
flow = [u, v]
```

实现位置：

```text
/data/fzj/RMDM/lib/kflow_loss.py
  compute_kflow
  spatial_gradients_centered
```

当前用途：

```text
1. 用 GT RSS 计算 |GT k-flow|，判断相邻帧哪里是 static / dynamic。
2. 用于 kflow EPE / kflow L1 诊断生成序列的传播变化是否更接近 GT。
3. 不把 k-flow 当最终论文指标；它是帮助定位和约束时间一致性的工具。
```

### 0.2 当前模型结构

不修改原始 RMDM：

```text
/data/fzj/RMDM/unet.py 保持不改
```

当前 student 结构：

```text
eps_base = frozen RMDM(x_t, cond_t, timestep)
delta_eps = residual_3d_head(cond_seq, x_t_seq, eps_base_seq, frame_pos, timestep)
eps_final = eps_base + alpha * delta_eps
```

关键点：

```text
1. 原始 RMDM 作为 frozen teacher/base，不参与训练。
2. residual head 最后一层 zero-init，训练开始时 eps_final == eps_base。
3. 当前没有 temporal self-attention。
4. 时间信息来自 residual head 的 Conv3D：
   输入张量为 [B, F, C, H, W]
   转成 Conv3D 的 [B, C, F, H, W]
   3D 卷积核同时看相邻帧和空间邻域。
5. residual head 还显式加入：
   frame_pos 两个通道: pos, 1-pos
   timestep 一个通道: t / diffusion_steps
```

实现位置：

```text
/data/fzj/RMDM/residual_kflow_student.py
  ResidualCorrectionHead
  FrozenRMDMResidualStudent
  predict_x0_from_eps
```

### 0.3 当前训练 loss

基础 loss：

```text
L_diff = MSE(eps_final, noise)
L_anchor = MSE(eps_final, eps_base)
```

其中 teacher anchor 的作用是限制 residual head 不要偏离 frozen RMDM 太多，避免为了时间一致性明显伤害单帧质量。

static key-path loss：

```text
M_static = key_path_mask ∩ static_kflow_mask

key_path_mask = K2 >= per-pair q70
static_kflow_mask = |GT k-flow| <= per-pair q70

L_static = mean(M_static * relu(|pred_x0[t+1] - pred_x0[t]| - margin))
```

当前默认：

```text
static_keypath_weight = 0.001
static_keypath_margin = 0.005
pair_stride = 1
只在 timestep <= 300 的 batch 上启用
```

dynamic key-path range loss，也就是当前 Step2 主线：

```text
M_dynamic = key_path_mask ∩ high_kflow_mask

key_path_mask = K2 >= per-pair q70
high_kflow_mask = |GT k-flow| >= per-pair q85

Δpred = |pred_x0[t+1] - pred_x0[t]|
Δgt   = |GT[t+1] - GT[t]|

gt_med = median(Δgt inside M_dynamic)
lower = 0.5 * gt_med
upper = 2.0 * gt_med

L_dynamic =
  mean(M_dynamic * relu(lower - Δpred))
+ 0.2 * mean(M_dynamic * relu(Δpred - upper))
```

含义：

```text
static loss:
  关键路径中真实传播基本不变的地方，生成结果不要乱跳。

dynamic loss:
  关键路径中真实传播确实变化的地方，生成结果不能完全冻结；
  但变化也不能过头。
```

当前最佳 pilot 配置：

```text
static_keypath_weight = 0.001
dynamic_keypath_weight = 0.0001
dynamic_keypath_flow_quantile = 0.85
eval residual_alpha = 0.03
train residual_alpha = 0.05
```

相关实现：

```text
/data/fzj/RMDM/residual_kflow_student.py
  static_keypath_consistency_loss
  dynamic_keypath_range_loss

/data/fzj/RMDM/train_residual_kflow_student.py
  训练入口和所有 loss 开关
```

本文档只记录能支撑决策的核心信息：目标、数据/模型路径、关键实验结果、当前主方向和下一步动作。历史 partial eval 和过程性监控不再展开。

## 1. 当前目标

原始 RMDM 单帧质量已经较强，当前目标不是单纯提高 PSNR，而是在 PSNR/MAE/MSE 不明显下降的前提下，提高生成序列的时间一致性。

核心判断口径：

```text
优先级:
  1. 不伤害单帧质量：PSNR / MAE / MSE
  2. 改善时间连续性：tOF / TWE / temporal delta
  3. k-flow 只作为帮助时序一致性的工具和诊断，不作为最终论文式主指标
```

当前主方向：

```text
frozen RMDM teacher/base
+ zero-init 3D CNN residual correction head
+ static key-path consistency loss
+ optional dynamic key-path range consistency loss
```

明确不作为当前第一版主方向：

```text
不改原始 /data/fzj/RMDM/unet.py
不使用 temporal self-attention
不使用 prior_injection_mode=uemb
不 full fine-tune 整个 RMDM trunk
不在全图强制 k-flow loss
不在高噪声 timestep 上加强时序约束
```

## 2. 核心路径

数据集：

```text
/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack_ShadowDenoiseV1
```

split：

```text
/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack_ShadowDenoiseV1/split.json
```

原始 RMDM checkpoint：

```text
/data/fzj/RMDM/checkpoints_dynamic_rmdm_4gpu_no_ckpt_75k/model_phy_step65000.pth
```

当前主要代码：

```text
/data/fzj/RMDM/residual_kflow_student.py
/data/fzj/RMDM/train_residual_kflow_student.py
/data/fzj/RMDM/sample_residual_kflow_paired.py
/data/fzj/RMDM/lib/dynamic_clip_loaders.py
/data/fzj/RMDM/lib/kflow_loss.py
/data/fzj/RMDM/merge_temporal_pinn_paired_shards.py
```

当前 pilot 训练场景：

```text
town10_junction_0189
town02_opt_junction_0298
```

说明：

```text
这两个场景用于快速 proof，不代表最终完整训练 split。
```

## 3. 关键代码改动索引

这一节保留目前关键代码更改的位置和理由，方便后续接手或回滚。

### 3.1 residual student 模型

文件：

```text
/data/fzj/RMDM/residual_kflow_student.py
```

关键位置：

| 位置 | 内容 | 理由 |
|---|---|---|
| `ResidualCorrectionHead`，约第 14 行 | 小型 3D CNN residual head | 让 correction head 能看到局部相邻帧，而不是纯逐帧 2D 修正 |
| `ResidualCorrectionHead.forward`，约第 48 行 | 加入 timestep channel / frame position channel | 给 residual head 显式时间步和帧位置，不改变 frozen RMDM 主干 |
| `FrozenRMDMResidualStudent`，约第 74 行 | 封装 frozen base + residual head | 保证原始 RMDM 作为 teacher/base，不需要修改 `unet.py` |
| `FrozenRMDMResidualStudent.forward`，约第 108 行 | `eps_final = eps_base + alpha * delta_eps` | zero-init residual，训练开始时输出严格贴近 baseline |
| `StaticKeyPathConfig`，约第 176 行 | static key-path loss 的配置对象 | 集中管理 K2 quantile、flow quantile、margin、pair stride |
| `safe_kflow_hinge_loss`，约第 186 行 | 历史 safe k-flow hinge 实现 | 保留用于对照和回退，但当前不是主方向 |
| `static_keypath_consistency_loss`，约第 267 行 | 当前主 loss | 只约束静态关键路径区域的相邻帧一致性，避免全图 k-flow 误约束 |
| `DynamicKeyPathConfig` | dynamic key-path range loss 的配置对象 | 管理 high-flow quantile、GT delta 范围、上下界权重 |
| `dynamic_keypath_range_loss` | 动态关键路径幅度范围约束 | 在高 GT k-flow 的关键路径上，约束预测变化不要完全不变也不要过大 |

设计理由：

```text
1. frozen base 保留原始 RMDM 单帧质量。
2. zero-init residual 避免训练初始破坏 baseline。
3. 3D CNN correction head 提供最小跨帧输入能力，但不引入大 temporal backbone。
4. static key-path loss 对齐当前目标：关键路径在传播条件不变时不要乱跳。
5. dynamic key-path loss 是可选项，用于测试“该变的地方要有合理变化”，权重要比 static 更保守。
```

### 3.2 residual student 训练入口

文件：

```text
/data/fzj/RMDM/train_residual_kflow_student.py
```

关键位置：

| 位置 | 内容 | 理由 |
|---|---|---|
| argparse `--scene_ids`，约第 34 行 | 支持指定小场景训练 | 快速 proof，不必每次扫完整 train split |
| argparse `--low_timestep_prob`，约第 78 行 | 提高低噪声 timestep 采样概率 | static/k-flow 约束只在低噪声 pred_x0 上更有物理意义 |
| argparse `--static_keypath_*`，约第 85-92 行 | static key-path loss 配置 | 方便做 `weight/margin/quantile/pair_stride` 小网格 |
| argparse `--dynamic_keypath_*` | dynamic key-path loss 配置 | 支持测试 high k-flow key-path 的变化幅度范围约束 |
| `Accelerator(...)`，约第 128 行 | 支持 Accelerate/DDP | 后续更大训练 probe 可用 4 GPU 启动 |
| dataset 构建，约第 135-147 行 | `DynamicRadioMapRMDMClip(... include_k2=True, scene_ids=...)` | 训练需要 RSS clip、cond clip、K2 clip 来构造 static key-path mask |
| optimizer，约第 163 行 | 只优化 `model.residual_head.parameters()` | 第一版不解冻 RMDM trunk，避免破坏单帧能力 |
| `--resume_residual_checkpoint` | 加载已有 residual head | 支持从当前最好 checkpoint continue training，不必从零重训 |
| low timestep sampling，约第 213 行 | 按概率把 timestep 限到低噪声区间 | 让时序 loss 有足够激活机会 |
| static loss 启用，约第 266-276 行 | `t <= static_keypath_max_timestep` 时计算 | 避免高噪声 timestep 的 pred_x0 误导时序约束 |
| checkpoint 保存，约第 313 和 321 行 | 保存 step checkpoint 和 final | 支持快速对比不同 step 的时间指标 |

设计理由：

```text
1. 训练只动 residual head，是当前最保守且最贴近目标的方案。
2. 低噪声 gating 避免 DDPM 高噪声阶段强行做物理一致性。
3. Accelerate 已接入；后续不要再用单卡脚本做主要训练。
```

### 3.3 动态 clip 数据加载

文件：

```text
/data/fzj/RMDM/lib/dynamic_clip_loaders.py
```

关键位置：

| 位置 | 内容 | 理由 |
|---|---|---|
| `DynamicRadioMapRMDMClip`，约第 19 行 | RMDM clip dataset | 支持 `[F, C, H, W]` 形式的连续帧训练/评估 |
| `include_k2` / `k2_root` / `scene_ids` 参数，约第 39-43 行 | 控制 K2 和场景过滤 | K2 只用于 loss mask，不进入 RMDM 条件输入；scene_ids 用于小 proof |
| scene filter，约第 61-69 行 | 只保留指定 scenes | 快速验证 easy/hard 或指定场景 |
| 返回 `k2` / `path_mask`，约第 90-104 行 | 训练 loss 需要 K2 和路径 mask | 保持模型输入仍是原始 3 通道条件，K2 不作为输入 |
| `_resolve_k2_path`，约第 139 行 | 解析 `k2_neg_norm.npz` | 对齐当前数据集里的 K2 outline 存储结构 |
| `_make_path_mask`，约第 171 行 | 从 adjacent K2 构造 path mask | 历史 k-flow/pinn 方案仍可复用 |

设计理由：

```text
1. RMDM 原模型输入保持 building / tx_heatmap / traffic，不额外喂 K2。
2. K2 只作为训练/评估 mask，避免和原 checkpoint 输入分布不兼容。
3. scene_ids 支持小规模 proof 到完整 split 的平滑切换。
```

### 3.4 paired temporal eval

文件：

```text
/data/fzj/RMDM/sample_residual_kflow_paired.py
```

关键位置：

| 位置 | 内容 | 理由 |
|---|---|---|
| `FIELDS`，约第 25 行 | 统一记录 frame + temporal 指标 | 避免只看 PSNR，保留 tOF/TWE/kflow/temporal delta |
| `--residual_checkpoint`，约第 65 行 | 指定 candidate residual checkpoint | baseline 和 candidate 使用同一原始 RMDM checkpoint |
| `--batch_size`，约第 74 行 | 支持调大 eval batch | 充分利用 GPU，不通过降低 DDIM/clip 配置提速 |
| `--resume_existing`，约第 85 行 | 断点续评 | 长 eval/shard 中断后不用重跑 |
| `temporal_flow_metrics`，约第 185 行 | tOF 和 TWE | 视频时序指标，当前重点关注 |
| `temporal_kflow_metrics`，约第 207 行 | kflow EPE/L1 | 作为诊断，不作为最终目标 |
| `temporal_delta_metrics`，约第 217 行 | 相邻帧 delta L1/MSE | 检查帧间变化是否更接近 GT |
| `sample_pair`，约第 255 行 | 同一初始噪声 paired sampling | baseline/candidate 对比更干净，减少随机噪声干扰 |
| `write_outputs`，约第 125 行 | 增量写 CSV/summary | 支持 shard 监控和断点 |

设计理由：

```text
1. paired eval 用同一个 initial noise 同时跑 baseline 和 candidate。
2. 测试配置保持 clip_length=32、DDIM20，不为速度降低配置。
3. 通过 4 shard x 25 clips 使用 GPU0-3 加速。
```

### 3.5 shard merge

文件：

```text
/data/fzj/RMDM/merge_temporal_pinn_paired_shards.py
```

关键位置：

| 位置 | 内容 | 理由 |
|---|---|---|
| `BASE_FIELDS`，约第 14 行 | 保留 global clip index 和 shard source | 合并后能追踪每条样本来自哪个 shard |
| 动态 metric columns，约第 74 行 | 不写死 PSNR/MAE/MSE | 后续新增 temporal 指标不需要再改 merge 脚本 |

设计理由：

```text
当前评估指标会持续扩展，merge 脚本必须能自动合并所有 metric columns。
```

### 3.6 当前运行脚本

文件：

```text
/data/fzj/RMDM/run_two_scene_residual_static_keypath_probe_gpu2.sh
```

说明：

```text
这是 w3e4 单卡历史短训脚本。
后续主要训练不要继续沿用单卡方式，应改用 4 GPU Accelerate/DDP 启动。
```

## 4. 当前模型方案

结构：

```text
eps_base  = frozen RMDM(x_t, cond, t)
delta_eps = small 3D CNN(cond, x_t, eps_base, frame_pos, timestep)
eps_final = eps_base + alpha * delta_eps
```

关键设置：

```text
base RMDM: frozen
residual head: trainable
residual head final conv: zero-init
alpha: 0.05
初始行为: eps_final == eps_base
```

这版 residual head 是 3D CNN，因此它不是严格逐帧修正；它能在局部时间窗口内看到相邻帧特征。但它仍是轻量 residual，不是大规模 temporal backbone。

## 5. 当前主 loss：static key-path consistency

动机：

```text
真正想约束的是：
  传播条件基本不变的关键路径区域，不应该在相邻帧里随机跳动。
```

mask：

```text
key_path_mask = adjacent K2 的 q70 高响应区域
static_mask   = GT k-flow magnitude 的 q70 低流动区域
final_mask    = key_path_mask AND static_mask
```

loss：

```text
L_static = mean(final_mask * relu(abs(pred_x0_t - pred_x0_t+1) - margin))
margin = 0.005
pair_stride = 1
```

训练 timestep：

```text
只在低噪声 t <= 300 的 batch 上启用 static loss
low_timestep_prob = 0.5
```

重要说明：

```text
clip_stride = 1 表示 clip 内是相邻 raw frames。
frame_stride = 16 只控制 clip window 起点采样间隔，不表示 loss pair 隔 16 帧。
```

## 6. 动态关键路径范围约束

动机：

```text
static key-path loss 约束“不该动的地方不要乱动”。
dynamic key-path range loss 测试“该动的关键路径要有合理幅度的变化”。
```

mask：

```text
M_key = adjacent K2 的 q70 高响应区域
M_dyn = M_key AND high_kflow_mask
high_kflow_mask = |GT k-flow| > q85
```

loss：

```text
delta_pred = abs(pred_x0[t+1] - pred_x0[t])
delta_gt   = abs(GT[t+1] - GT[t])

gt_med = median(delta_gt inside M_dyn)
lower = 0.5 * gt_med
upper = 2.0 * gt_med

L_dyn_range =
  mean(M_dyn * relu(lower - delta_pred))
+ 0.2 * mean(M_dyn * relu(delta_pred - upper))
```

实现位置：

```text
/data/fzj/RMDM/residual_kflow_student.py
  DynamicKeyPathConfig
  dynamic_keypath_range_loss

/data/fzj/RMDM/train_residual_kflow_student.py
  --dynamic_keypath_weight
  --dynamic_keypath_flow_quantile
  --resume_residual_checkpoint
```

当前结论：

```text
dynamic range consistency 能进一步改善时间指标，但会更容易拉动单帧误差。
当前更合理的动态权重是 1e-4；3e-4 没带来额外稳定收益。
```

## 7. 当前最佳小实验：static-only w1e3

训练配置：

```text
max_steps = 1000
batch_size = 2
diff_loss_weight = 1.0
teacher_anchor_weight = 1.0
kflow_weight = 0.0
x0_recon_weight = 0.0
static_keypath_weight = 0.001
static_keypath_warmup_steps = 200
static_keypath_max_timestep = 300
static_keypath_pair_stride = 1
```

checkpoint：

```text
/data/fzj/RMDM/checkpoints_residual_static_keypath_two_scene_0189_0298_w1e3/residual_student_final.pth
```

训练输出目录：

```text
/data/fzj/RMDM/checkpoints_residual_static_keypath_two_scene_0189_0298_w1e3
```

100 clips paired eval：

```text
split: test
clip_length: 32
ddim_steps: 20
clips: 100
GPU: 0-3, 4 shards x 25 clips
eval batch_size: 3
```

结果路径：

```text
/data/fzj/RMDM/experiments/residual_static_keypath_two_scene_w1e3_final_temporal_eval100/merged/summary_paired_merged.json
/data/fzj/RMDM/experiments/residual_static_keypath_two_scene_w1e3_final_temporal_eval100/merged/per_clip_paired_merged.csv
```

100 clips mean：

| Metric | Baseline | Candidate | Delta | 趋势 |
|---|---:|---:|---:|---|
| MAE | 0.034090 | 0.033029 | -0.001062 | 更好 |
| MSE | 0.007125 | 0.007139 | +0.0000135 | 略差 |
| PSNR | 23.8377 | 23.8494 | +0.0117 dB | 更好 |
| tOF | 0.656615 | 0.645650 | -0.010965 | 更好 |
| TWE | 0.010766 | 0.010578 | -0.000188 | 更好 |
| kflow EPE | 1.682877 | 1.640468 | -0.042409 | 更好 |
| kflow L1 | 1.066741 | 1.039712 | -0.027029 | 更好 |
| temporal-dL1 | 0.012335 | 0.012136 | -0.000198 | 更好 |
| temporal-dMSE | 0.001390 | 0.001351 | -0.0000388 | 更好 |

当前判断：

```text
static key-path residual 是目前最有希望的方向。
它在 100 clips 上同时改善 PSNR/MAE 和多个时间指标，只有 MSE 极小幅上升。
这个结果足够支持进入更大规模训练 probe，但还不建议直接做最终全量大训练。
```

## 8. 动态约束 continue training 结果

共同设置：

```text
resume from:
  /data/fzj/RMDM/checkpoints_residual_static_keypath_two_scene_0189_0298_w1e3/residual_student_final.pth

base/static settings:
  frozen RMDM base
  residual_alpha = 0.05
  static_keypath_weight = 0.001
  static_keypath_margin = 0.005
  static_keypath_k2_quantile = 0.70
  static_keypath_flow_quantile = 0.70
  pair_stride = 1
  t <= 300

dynamic settings:
  dynamic_keypath_flow_quantile = 0.85
  lower = 0.5 * gt_med
  upper = 2.0 * gt_med
  upper_weight = 0.2
  continue steps = 500
  training GPU = 0-3, Accelerate 4 processes
```

训练输出：

```text
dynamic 1e-4:
  /data/fzj/RMDM/checkpoints_residual_static_w1e3_dynamic_q85_w1e4_cont500/residual_student_final.pth

dynamic 3e-4:
  /data/fzj/RMDM/checkpoints_residual_static_w1e3_dynamic_q85_w3e4_cont500/residual_student_final.pth
```

100 clips paired eval：

```text
dynamic 1e-4 summary:
  /data/fzj/RMDM/experiments/residual_static_w1e3_dynamic_q85_w1e4_cont500_eval100/merged/summary_paired_merged.json

dynamic 3e-4 summary:
  /data/fzj/RMDM/experiments/residual_static_w1e3_dynamic_q85_w3e4_cont500_eval100/merged/summary_paired_merged.json
```

对比：

| Experiment | dMAE | dMSE | dPSNR | dTOF | dTWE | dKFlow EPE | dKFlow L1 | dTD-L1 | dTD-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static w1e3 | -0.0010619 | +0.0000135 | +0.01172 | -0.01096 | -0.0001877 | -0.04241 | -0.02703 | -0.0001985 | -0.0000388 |
| dynamic 1e-4 | -0.0011256 | +0.0000289 | -0.00095 | -0.02577 | -0.0002246 | -0.06256 | -0.04006 | -0.0002277 | -0.0000284 |
| dynamic 3e-4 | -0.0006317 | +0.0000062 | -0.00250 | -0.02713 | -0.0002044 | -0.06190 | -0.03977 | -0.0002275 | -0.0000413 |

核心判断：

```text
1. dynamic 1e-4 明确增强时间指标：tOF/TWE/kflow/temporal-dL1 都比 static-only 更好。
2. dynamic 1e-4 的代价是 PSNR 从 +0.0117 dB 变成约 -0.001 dB，MSE 上升也更大；这基本是“时间更好，帧质量近似持平但不如 static-only 稳”。
3. dynamic 3e-4 没有比 1e-4 更好：tOF 略强，但 MAE/PSNR 更差，TWE/kflow 和 1e-4 基本持平或略差。
4. 当前推荐：如果目标偏时间连续性，优先 dynamic 1e-4；如果目标是最稳的 frame fidelity + 时间改善，仍保留 static-only w1e3 作为 balanced baseline。
```

## 9. dynamic 1e-4 eval alpha sweep

动机：

```text
dynamic 1e-4 alpha=0.05 的时间指标最好，但 PSNR 均值略负。
测试是否通过降低 residual alpha 到 0.03/0.04，保留时间收益，同时恢复 frame fidelity。
```

代码改动：

```text
/data/fzj/RMDM/sample_residual_kflow_paired.py
  新增 --override_residual_alpha

用途:
  只在 eval/sampling 时覆盖 student.alpha，不改 checkpoint 文件。
```

checkpoint：

```text
/data/fzj/RMDM/checkpoints_residual_static_w1e3_dynamic_q85_w1e4_cont500/residual_student_final.pth
```

结果路径：

```text
alpha=0.03:
  /data/fzj/RMDM/experiments/residual_static_w1e3_dynamic_q85_w1e4_cont500_eval100_alpha003/merged/summary_paired_merged.json

alpha=0.04:
  /data/fzj/RMDM/experiments/residual_static_w1e3_dynamic_q85_w1e4_cont500_eval100_alpha004/merged/summary_paired_merged.json

alpha=0.05:
  /data/fzj/RMDM/experiments/residual_static_w1e3_dynamic_q85_w1e4_cont500_eval100/merged/summary_paired_merged.json
```

100 clips mean 对比：

| Experiment | dMAE | dMSE | dPSNR | dTOF | dTWE | dKFlow EPE | dKFlow L1 | dTD-L1 | dTD-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static-only alpha=0.05 | -0.0010619 | +0.0000135 | +0.01172 | -0.01096 | -0.0001877 | -0.04241 | -0.02703 | -0.0001985 | -0.0000388 |
| dynamic 1e-4 alpha=0.03 | -0.0007545 | +0.0000070 | +0.02011 | -0.02018 | -0.0002120 | -0.05065 | -0.03226 | -0.0001835 | -0.0000194 |
| dynamic 1e-4 alpha=0.04 | -0.0009401 | +0.0000276 | +0.00584 | -0.01234 | -0.0001376 | -0.03508 | -0.02250 | -0.0001340 | -0.0000153 |
| dynamic 1e-4 alpha=0.05 | -0.0011256 | +0.0000289 | -0.00095 | -0.02577 | -0.0002246 | -0.06256 | -0.04006 | -0.0002277 | -0.0000284 |

per-clip 分布检查：

```text
alpha=0.03:
  delta_psnr mean +0.02011, median +0.00116, positive clips 50/100
  delta_kflow_epe mean -0.05065, median -0.06411, positive/worse clips 27/100

alpha=0.05:
  delta_psnr mean -0.00095, median -0.00882, positive clips 44/100
  delta_kflow_epe mean -0.06256, median -0.06081, positive/worse clips 24/100
```

核心判断：

```text
1. alpha=0.03 是当前最 balanced 的 dynamic 配置：PSNR/MSE 恢复，tOF/TWE/kflow 仍强于 static-only。
2. alpha=0.05 是 temporal-strong 配置：时间指标最好，但 frame fidelity 不如 alpha=0.03。
3. alpha=0.04 反而不如 0.03/0.05，暂时不作为候选。
4. 用户关于“整体变好但个别极端样本影响结果”的判断有依据：PSNR mean 和 median 分化明显，尤其 alpha=0.03 的 mean PSNR 明显正、median 接近 0。
```

## 10. 对照实验：static weight 3e-4

checkpoint：

```text
/data/fzj/RMDM/checkpoints_residual_static_keypath_two_scene_0189_0298_w3e4/residual_student_final.pth
```

结果路径：

```text
/data/fzj/RMDM/experiments/residual_static_keypath_two_scene_w3e4_final_temporal_eval100/merged/summary_paired_merged.json
```

100 clips delta：

| Metric | Delta |
|---|---:|
| MAE | -0.001009 |
| MSE | +0.0000374 |
| PSNR | +0.00152 dB |
| tOF | -0.013686 |
| TWE | -0.0000601 |
| kflow EPE | -0.028194 |
| kflow L1 | -0.018120 |
| temporal-dL1 | -0.0000814 |
| temporal-dMSE | -0.00000541 |

判断：

```text
3e-4 也有效，但整体弱于 1e-3。
1e-3 的 TWE、kflow、temporal delta、PSNR、MAE 都更好，MSE 上升也更小。
```

## 11. 历史尝试摘要

### temporal PINN / temporal attention 方向

尝试内容：

```text
32 帧 temporal PINN prior
temporal self-attention
prior_injection_mode none/uemb
k-flow loss on cal_seq
```

结论：

```text
实现复杂度高，且和当前目标不完全对齐。
用户后续明确不希望第一版使用 temporal self-attention 或 uemb injection。
该方向暂时降级为历史实现，不作为当前主线。
```

相关历史文件：

```text
/data/fzj/RMDM/temporal_blocks.py
/data/fzj/RMDM/temporal_pinn_unet.py
/data/fzj/RMDM/train_temporal_pinn.py
/data/fzj/RMDM/sample_temporal_pinn.py
```

### inference-time smoothing

尝试内容：

```text
sampling-time temporal smoothing
weight = 0.05 / 0.03
paired full split eval
```

关键结论：

```text
0.05:
  val  delta PSNR +0.008774
  test delta PSNR +0.016673

0.03:
  val  delta PSNR +0.005570
  test delta PSNR +0.011130
```

判断：

```text
收益太小，且主要是后处理式平滑，不足以作为大规模训练方向。
```

### safe masked k-flow hinge residual

尝试内容：

```text
frozen RMDM + zero-init residual head
safe masked k-flow hinge
trusted mask:
  GT k-flow warp 可对齐区域
  RSS 空间梯度低/中区域
  排除强 K2 区域
```

结果路径：

```text
/data/fzj/RMDM/experiments/residual_kflow_two_scene_final_temporal_eval100/merged/summary_paired_merged.json
```

100 clips delta：

| Metric | Delta |
|---|---:|
| PSNR | +0.080841 |
| MAE | -0.0001338 |
| MSE | -0.00000379 |
| tOF | -0.005356 |
| TWE | +0.0000528 |
| kflow EPE | +0.007073 |
| kflow L1 | +0.004445 |
| temporal-dL1 | +0.0000346 |
| temporal-dMSE | -0.00000319 |

判断：

```text
单帧指标有收益，但时间指标 mixed。
主要问题是 mask 排除了强 K2 区域，而强 K2 往往正是关键传播结构。
该实验推动了当前 static key-path mask 的设计。
```

## 12. GPU 和评估策略

用户要求：

```text
训练和测试都尽量充分利用 GPU0-3。
不能为了速度降低测试配置。
```

当前执行策略：

```text
100 clips eval:
  4 shards x 25 clips
  CUDA_VISIBLE_DEVICES=0/1/2/3
  clip_length=32
  ddim_steps=20
  eval batch_size 从 3 起

训练:
  后续使用 accelerate/DDP 4 GPU 启动
  不再默认单卡训练
```

说明：

```text
之前 w3e4/w1e3 两个短训是单卡并行启动的历史过渡配置。
后续更大训练 probe 应切换到 4 GPU DDP。
```

## 13. 下一步建议

优先动作：

```text
1. 用 static_keypath_weight=0.001 做 4 GPU DDP 更大训练 probe。
2. 先保持 frozen base + residual head，不解冻 RMDM trunk。
3. 使用更多 train scenes 或完整 train split，但仍设置中等 step 数，先看验证集稳定性。
4. 继续用 100 clips paired eval 快速筛选，确认后再扩大 eval 范围。
```

建议下一轮配置：

```text
balanced baseline:
  static_keypath_weight = 0.001
  dynamic_keypath_weight = 0.0

temporal-strong candidate:
  static_keypath_weight = 0.001
  dynamic_keypath_weight = 0.0001
  dynamic_keypath_flow_quantile = 0.85
  eval residual_alpha = 0.05

balanced dynamic candidate:
  static_keypath_weight = 0.001
  dynamic_keypath_weight = 0.0001
  dynamic_keypath_flow_quantile = 0.85
  eval residual_alpha = 0.03

common:
  static_keypath_margin = 0.005
  static_keypath_k2_quantile = 0.70
  static_keypath_flow_quantile = 0.70
  pair_stride = 1
  residual_alpha = 0.05
  base RMDM frozen
  trainable = residual_head only
```

进入更大规模训练的 gate：

```text
PSNR / MAE / MSE 不明显劣化；
tOF / TWE 至少稳定改善；
temporal delta 和 kflow 诊断不反向；
不同 shard 不出现明显一半改善、一半恶化的分裂。
```

## 14. Step3 k-flow structure loss 测试结论

背景：

```text
Step2 dynamic range 版本目前最强的 balanced checkpoint:
/data/fzj/RMDM/checkpoints_residual_static_w1e3_dynamic_q85_w1e4_cont500/residual_student_final.pth

对应 100 clips eval alpha=0.03:
/data/fzj/RMDM/experiments/residual_static_w1e3_dynamic_q85_w1e4_cont500_eval100_alpha003/merged/summary_paired_merged.json
```

Step3 设计：

```text
目标不是直接拟合 GT 相邻帧差分，而是在关键动态路径区域内，
让 pred_x0 相邻帧变化强弱分布对齐 GT k-flow magnitude 分布。

M_dyn = K2 q70 key-path mask ∩ high_kflow_mask
Δpred = |pred_x0[t+1] - pred_x0[t]|
K = |GT k-flow|
loss = mean(M_dyn * SmoothL1(Δpred_norm - stopgrad(K_norm)))
```

实现位置：

```text
/data/fzj/RMDM/residual_kflow_student.py
  KFlowStructureConfig
  kflow_structure_loss

/data/fzj/RMDM/train_residual_kflow_student.py
  --kflow_structure_weight
  --kflow_structure_k2_quantile
  --kflow_structure_flow_quantile
```

### Step3 A: q85, weight=3e-5

训练和评估路径：

```text
checkpoint:
/data/fzj/RMDM/checkpoints_residual_step3_kflowstruct_q85_w3e5_cont500/residual_student_final.pth

eval alpha=0.03:
/data/fzj/RMDM/experiments/residual_step3_kflowstruct_q85_w3e5_cont500_eval100_alpha003/merged/summary_paired_merged.json
```

100 clips delta：

| Metric | Delta |
|---|---:|
| MAE | -0.000551 |
| MSE | -0.0000112 |
| PSNR | +0.01072 |
| tOF | -0.00144 |
| TWE | -0.000100 |
| kflow EPE | -0.02180 |
| kflow L1 | -0.01397 |
| temporal-dL1 | -0.0000928 |
| temporal-dMSE | -0.0000144 |

判断：

```text
A 的单帧 MSE/PSNR 可以，但时间指标明显弱于 Step2 dynamic alpha=0.03。
不作为后续主线。
```

### Step3 B: q90, weight=5e-5

训练和评估路径：

```text
checkpoint:
/data/fzj/RMDM/checkpoints_residual_step3_kflowstruct_q90_w5e5_cont500/residual_student_final.pth

eval alpha=0.03:
/data/fzj/RMDM/experiments/residual_step3_kflowstruct_q90_w5e5_cont500_eval100_alpha003/merged/summary_paired_merged.json
```

100 clips delta：

| Metric | Delta |
|---|---:|
| MAE | -0.000629 |
| MSE | +0.00000544 |
| PSNR | +0.000656 |
| tOF | -0.016238 |
| TWE | -0.0001819 |
| kflow EPE | -0.038834 |
| kflow L1 | -0.024618 |
| temporal-dL1 | -0.0001659 |
| temporal-dMSE | -0.0000212 |

判断：

```text
B 明显好于 A，说明更保守的 high-kflow q90 mask 更合理。
但 B 仍弱于当前 Step2 dynamic alpha=0.03 balanced candidate：
  Step2 dynamic alpha=0.03: PSNR +0.02011, tOF -0.02018, TWE -0.000212, kflow EPE -0.05065
  Step3 B alpha=0.03:       PSNR +0.00066, tOF -0.01624, TWE -0.000182, kflow EPE -0.03883

当前结论：
Step3 这种 k-flow magnitude structure loss 作为替代 dynamic range loss 不够好。
暂时不把它作为主线；当前 balanced 主线仍是 Step2 dynamic q85 weight=1e-4, eval alpha=0.03。
如果后面继续试 Step3，应作为 Step2 的弱辅助项，而不是替代 Step2。
```

## 15. 全局 temporal residual 消融

目的：

```text
验证“不用 k-flow / 不用 K2 dynamic mask，直接全图做最简单相邻帧残差一致性”是否足够。
```

实现位置：

```text
/data/fzj/RMDM/residual_kflow_student.py
  GlobalTemporalResidualConfig
  global_temporal_residual_loss

/data/fzj/RMDM/train_residual_kflow_student.py
  --global_temporal_weight
  --global_temporal_margin
  --global_temporal_pair_stride
```

loss 定义：

```text
L_global = mean(relu(|pred_x0[t+1] - pred_x0[t]| - margin))

不使用 K2 mask。
不计算 k-flow。
不区分 static/dynamic 区域。
```

训练设置：

```text
resume from static checkpoint:
/data/fzj/RMDM/checkpoints_residual_static_keypath_two_scene_0189_0298_w1e3/residual_student_final.pth

train:
  static_keypath_weight = 0.001
  global_temporal_weight = 0.001
  global_temporal_margin = 0.005
  dynamic_keypath_weight = 0
  kflow_structure_weight = 0
  kflow_weight = 0
  steps = 500
  GPUs = 0-3

checkpoint:
/data/fzj/RMDM/checkpoints_residual_static_w1e3_globaltemporal_w1e3_cont500/residual_student_final.pth
```

评估：

```text
100 clips paired eval
eval alpha = 0.03
frame_stride = 32
ddim_steps = 20

summary:
/data/fzj/RMDM/experiments/residual_static_w1e3_globaltemporal_w1e3_cont500_eval100_alpha003_fstride32/merged/summary_paired_merged.json
```

100 clips delta：

| Metric | Delta |
|---|---:|
| MAE | -0.000550 |
| MSE | +0.00000999 |
| PSNR | +0.01353 |
| tOF | -0.01188 |
| TWE | -0.0000170 |
| kflow EPE | -0.03240 |
| kflow L1 | -0.02052 |
| temporal-dL1 | -0.0000300 |
| temporal-dMSE | +0.00000608 |

和当前主线对比：

| Experiment | dMAE | dMSE | dPSNR | dTOF | dTWE | dKFlow EPE | dKFlow L1 | dTD-L1 | dTD-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static-only alpha=0.05 | -0.001062 | +0.0000135 | +0.01172 | -0.01096 | -0.0001877 | -0.04241 | -0.02703 | -0.0001985 | -0.0000388 |
| global residual alpha=0.03 | -0.000550 | +0.0000100 | +0.01353 | -0.01188 | -0.0000170 | -0.03240 | -0.02052 | -0.0000300 | +0.00000608 |
| dynamic q85 w1e-4 alpha=0.03 | -0.000755 | +0.00000697 | +0.02011 | -0.02018 | -0.0002120 | -0.05065 | -0.03226 | -0.0001835 | -0.0000194 |

判断：

```text
全局 residual loss 有效果，但不够好。
它能带来一点 PSNR/tOF/kflow 改善，但 TWE 和 temporal-delta 很弱，
说明全图平均约束更像普通平滑，不能准确作用在传播关键路径。

相比之下，Step2 dynamic q85 w1e-4 alpha=0.03 同时更好地保持 PSNR，
并且明显提升 tOF/TWE/kflow/temporal-delta。

结论：
全局 residual 可以作为 cheap baseline / 消融对照，
但不建议替代 key-path + k-flow mask 方向。
```

## 16. 全 train split 大规模训练

启动时间：

```text
2026-06-28 22:54 CST
```

目标：

```text
按当前最佳 pilot 配置 Step2 dynamic q85 w1e-4 alpha=0.03 扩展到完整 train split。
```

启动 checkpoint：

```text
/data/fzj/RMDM/checkpoints_residual_static_w1e3_dynamic_q85_w1e4_cont500/residual_student_final.pth
```

输出路径：

```text
/data/fzj/RMDM/checkpoints_residual_step2_dynamic_q85_w1e4_alltrain_5k
```

tmux session：

```text
rmdm_step2_dynamic_alltrain_5k
```

训练确认：

```text
scene_ids: all train scenes
dataset clips: 724500
GPUs: 0-3
max_steps: 5000
save_interval: 500
```

核心配置：

```text
frozen RMDM base:
/data/fzj/RMDM/checkpoints_dynamic_rmdm_4gpu_no_ckpt_75k/model_phy_step65000.pth

residual_alpha = 0.05 during training
eval alpha should first test 0.03

static_keypath_weight = 0.001
static_keypath_margin = 0.005
static_keypath_k2_quantile = 0.70
static_keypath_flow_quantile = 0.70

dynamic_keypath_weight = 0.0001
dynamic_keypath_flow_quantile = 0.85
dynamic lower = 0.5 * gt_med
dynamic upper = 2.0 * gt_med
dynamic upper_weight = 0.2

pair_stride = 1
low_timestep_prob = 0.5
temporal losses only active for timestep <= 300
```

20% test 抽样评估：

```text
完整 test clips: 6750
抽样方式: 等间隔 20% index sampling
抽样 clips: 1350
eval alpha: 0.03
summary:
/data/fzj/RMDM/experiments/residual_step2_dynamic_q85_w1e4_alltrain_5k_eval20pct_alpha003/merged/summary_paired_merged.json
```

1350 clips mean：

| Metric | frozen RMDM | alltrain Step2 | Delta |
|---|---:|---:|---:|
| MAE | 0.024280 | 0.023596 | -0.000684 |
| MSE | 0.003929 | 0.003920 | -0.00000891 |
| PSNR | 25.856259 | 25.884727 | +0.028468 |
| tOF | 0.728810 | 0.712907 | -0.015904 |
| TWE | 0.010583 | 0.010407 | -0.0001759 |
| kflow EPE | 1.776156 | 1.744460 | -0.031696 |
| kflow L1 | 1.122689 | 1.102495 | -0.020194 |
| temporal-dL1 | 0.012532 | 0.012360 | -0.0001719 |
| temporal-dMSE | 0.001468 | 0.001439 | -0.0000288 |

非 cherry-pick 代表视频：

```text
选择规则:
  mean/median-like
  moderate temporal improvement
  difficult/regression-side case

selection:
/data/fzj/RMDM/experiments/residual_step2_dynamic_q85_w1e4_alltrain_5k_eval20pct_alpha003/representative_clip_selection.json

videos:
/data/fzj/RMDM/experiments/residual_step2_dynamic_q85_w1e4_alltrain_5k_eval20pct_alpha003/representative_videos
```

## 17. 新 k-path delta GT loss probe

用户建议公式：

```text
L_kpath =
  M_static * |Δpred - clip(Δpred, 0, m)|
+ λ M_dynamic * |Δpred - Δgt|
```

当前实现：

```text
Δpred = |pred_x0[t+1] - pred_x0[t]|
Δgt   = |GT[t+1] - GT[t]|

M_static:
  K2 q70 key path ∩ |GT k-flow| low q70

M_dynamic:
  K2 q70 key path ∩ |GT k-flow| high q85

m = 0.005
λ = 0.1
outer weight = 0.001
```

实现位置：

```text
/data/fzj/RMDM/residual_kflow_student.py
  KPathDeltaConfig
  kpath_delta_target_loss

/data/fzj/RMDM/train_residual_kflow_student.py
  --kpath_delta_weight
  --kpath_delta_dynamic_lambda
```

当前 probe：

```text
resume:
/data/fzj/RMDM/checkpoints_residual_step2_dynamic_q85_w1e4_alltrain_5k/residual_student_final.pth

output:
/data/fzj/RMDM/checkpoints_residual_kpathdelta_q70_q85_lam01_alltrain_cont500

steps: 500
GPUs: 0-3
```

1350 clips 20% test eval：

```text
summary:
/data/fzj/RMDM/experiments/residual_kpathdelta_q70_q85_lam01_alltrain_cont500_eval20pct_alpha003/merged/summary_paired_merged.json
```

| Metric | frozen RMDM | kpath-delta | Delta |
|---|---:|---:|---:|
| MAE | 0.024283 | 0.023643 | -0.000640 |
| MSE | 0.003928 | 0.003922 | -0.00000609 |
| PSNR | 25.858467 | 25.884696 | +0.026229 |
| tOF | 0.731869 | 0.709330 | -0.022538 |
| TWE | 0.010575 | 0.010381 | -0.0001941 |
| kflow EPE | 1.772423 | 1.737748 | -0.034675 |
| kflow L1 | 1.120326 | 1.098182 | -0.022145 |
| temporal-dL1 | 0.012517 | 0.012339 | -0.0001777 |
| temporal-dMSE | 0.001465 | 0.001437 | -0.0000286 |

初步判断：

```text
相比原 Step2 alltrain:
  PSNR 增益略低: +0.0262 vs +0.0285
  时间指标更强: tOF/TWE/kflow/tdelta 都更好

如果汇报强调时间一致性，kpath-delta 是一个值得保留的候选；
如果强调最稳 frame fidelity，原 Step2 alltrain 仍略稳。
```

## 18. deep 3D residual head probe

目的：

```text
测试更深/更宽的 3D residual correction head 是否能进一步提升时空一致性。
```

结构改动：

```text
默认旧结构:
  head_hidden_channels = 32
  head_num_layers = 2
  trainable params ≈ 35k

deep probe:
  head_hidden_channels = 64
  head_num_layers = 4
  trainable params ≈ 348k
```

实现位置：

```text
/data/fzj/RMDM/residual_kflow_student.py
  ResidualCorrectionHead(num_layers=...)

/data/fzj/RMDM/train_residual_kflow_student.py
/data/fzj/RMDM/sample_residual_kflow_paired.py
/data/fzj/RMDM/render_residual_comparison_clips.py
  --head_num_layers
```

当前训练：

```text
tmux:
rmdm_deep3d_kpathdelta_alltrain_5k

output:
/data/fzj/RMDM/checkpoints_residual_deep3d_h64_l4_kpathdelta_alltrain_5k

loss:
kpath-delta, same q70/q85/lambda=0.1 setup

steps:
5000
```

自动评估：

```text
tmux:
rmdm_deep3d_kpathdelta_eval20pct

output:
/data/fzj/RMDM/experiments/residual_deep3d_h64_l4_kpathdelta_alltrain_5k_eval20pct_alpha003
```

1350 clips 20% test eval：

```text
summary:
/data/fzj/RMDM/experiments/residual_deep3d_h64_l4_kpathdelta_alltrain_5k_eval20pct_alpha003/merged/summary_paired_merged.json
```

| Metric | frozen RMDM | deep3d kpath | Delta |
|---|---:|---:|---:|
| MAE | 0.024288 | 0.023613 | -0.000676 |
| MSE | 0.003931 | 0.003888 | -0.0000425 |
| PSNR | 25.855521 | 25.923577 | +0.068056 |
| tOF | 0.725924 | 0.671806 | -0.054118 |
| TWE | 0.010562 | 0.009848 | -0.000714 |
| kflow EPE | 1.771759 | 1.650531 | -0.121228 |
| kflow L1 | 1.119927 | 1.042582 | -0.077346 |
| temporal-dL1 | 0.012513 | 0.011835 | -0.000678 |
| temporal-dMSE | 0.001466 | 0.001327 | -0.000139 |

判断：

```text
deep 3D residual head 显著优于浅层 residual head。
它不只是提升时间指标，也同时提升 PSNR/MSE。

目前最强候选:
/data/fzj/RMDM/checkpoints_residual_deep3d_h64_l4_kpathdelta_alltrain_5k/residual_student_final.pth

明早汇报建议主讲:
  frozen RMDM vs deep3d kpath-delta
并把原 Step2 / shallow kpath-delta 作为消融。
```

同一组非 cherry-pick 代表视频：

```text
clip indices:
6378, 1821, 120

videos:
/data/fzj/RMDM/experiments/residual_deep3d_h64_l4_kpathdelta_alltrain_5k_eval20pct_alpha003/representative_videos_same_indices
```

## 19. 全局 dynamic range 消融

目的：

```text
回答“是否可以不使用 key-path / k-flow mask，直接在全图使用动态区域那套 range loss”。
```

实现位置：

```text
/data/fzj/RMDM/residual_kflow_student.py
  GlobalDynamicRangeConfig
  global_dynamic_range_loss

/data/fzj/RMDM/train_residual_kflow_student.py
  --global_dynamic_weight
  --global_dynamic_pair_stride
  --global_dynamic_lower_scale
  --global_dynamic_upper_scale
  --global_dynamic_upper_weight
```

loss 形式：

```text
Δpred = |pred_x0[t+1] - pred_x0[t]|
Δgt   = |gt[t+1] - gt[t]|

gt_med = median(Δgt over all pixels)
lower = 0.5 * gt_med
upper = 2.0 * gt_med

L_global_dynamic =
  mean(relu(lower - Δpred))
+ 0.2 * mean(relu(Δpred - upper))
```

关键区别：

```text
Step2 dynamic:
  只在 M_dynamic = key_path_mask ∩ high_kflow_mask 上统计和约束。

global dynamic:
  不使用 K2 / k-flow mask，全图统计和约束。
```

训练配置：

```text
init:
/data/fzj/RMDM/checkpoints_residual_static_keypath_two_scene_0189_0298_w1e3/residual_student_final.pth

output:
/data/fzj/RMDM/checkpoints_residual_static_w1e3_globaldynamic_w1e4_cont500

train scenes:
town10_junction_0189
town02_opt_junction_0298

steps:
500

GPUs:
0,1,2,3

static_keypath_weight = 0.001
global_dynamic_weight = 0.0001
dynamic_keypath_weight = 0
kpath_delta_weight = 0
residual alpha train = 0.05
eval alpha = 0.03
```

训练日志中的关键观察：

```text
gdgt 经常为 0，gdhigh 经常为 1.0。
```

判断：

```text
全图 median(Δgt) 被大量静态/背景像素拖到 0。
因此 global dynamic 实际上更像“全局不要变化太大”的正则，
而不是“动态关键路径应该合理变化”的约束。
```

100 clips test eval：

```text
summary:
/data/fzj/RMDM/experiments/residual_static_w1e3_globaldynamic_w1e4_cont500_eval100_alpha003_fstride32/merged/summary_paired_merged.json

per-clip csv:
/data/fzj/RMDM/experiments/residual_static_w1e3_globaldynamic_w1e4_cont500_eval100_alpha003_fstride32/merged/per_clip_paired_merged.csv
```

同协议消融对比，均为 100 clips、DDIM20、eval alpha=0.03：

| Run | ΔMAE | ΔMSE | ΔPSNR | ΔtOF | ΔTWE | Δkflow EPE | Δkflow L1 | Δtdelta-L1 | Δtdelta-MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| static-only keypath | -0.001062 | +0.0000135 | +0.01172 | -0.01096 | -0.0001877 | -0.04241 | -0.02703 | -0.0001985 | -0.0000388 |
| global static residual | -0.000550 | +0.0000100 | +0.01353 | -0.01188 | -0.0000170 | -0.03240 | -0.02052 | -0.0000300 | +0.0000061 |
| Step2 keypath dynamic q85 w1e-4 | -0.000755 | +0.0000070 | +0.02011 | -0.02018 | -0.0002120 | -0.05065 | -0.03226 | -0.0001835 | -0.0000194 |
| global dynamic w1e-4 | -0.000622 | +0.0000189 | -0.00504 | -0.00684 | -0.0000434 | -0.02012 | -0.01312 | -0.0000404 | +0.0000025 |

global dynamic 的具体均值：

| Metric | frozen RMDM | global dynamic | Delta |
|---|---:|---:|---:|
| MAE | 0.034053 | 0.033431 | -0.000622 |
| MSE | 0.007120 | 0.007139 | +0.0000189 |
| PSNR | 23.847024 | 23.841980 | -0.00504 |
| tOF | 0.649742 | 0.642903 | -0.00684 |
| TWE | 0.010636 | 0.010593 | -0.0000434 |
| kflow EPE | 1.665880 | 1.645759 | -0.02012 |
| kflow L1 | 1.056263 | 1.043144 | -0.01312 |
| temporal-dL1 | 0.012201 | 0.012161 | -0.0000404 |
| temporal-dMSE | 0.001359 | 0.001362 | +0.0000025 |

结论：

```text
全局 dynamic range loss 不如 key-path dynamic。

它能轻微改善 MAE 和部分时间指标，
但 PSNR/MSE 变差，tOF/TWE/kflow/tdelta 的收益都明显弱于 Step2 keypath dynamic。

这个消融支持当前主线判断：
动态约束必须限制在 key-path ∩ high-kflow 区域；
全图动态差分会被背景/静态区域污染，不适合作为第一版主约束。
```

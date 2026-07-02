# DynamicRadioMap RMDM 实验记录

记录时间：2026-06-26
项目目录：`/data/fzj/RMDM`

## 1. 数据集

本次实验使用清理和去噪后的 DynamicRadioMap 数据集：

```text
/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack_ShadowDenoiseV1
```

数据读取配置：

```text
data_name: DynamicRadio
image_size: 128
split_file: split.json
frame_stride: 1
tx_heatmap_sigma_px: 1.5
```

输入条件：

```text
condition channels:
  0: building mask
  1: tx heatmap
  2: traffic frame

target:
  rss/radio map PNG, normalized to [0, 1]
```

训练中保留 RMDM 原始条件增强逻辑：

```python
conditions[:, 0] += 10.0 * conditions[:, 1]
```

PINN loss 使用未改写的 raw building/tx heatmap。

## 2. 模型和训练设置

模型为完整 RMDM 组合模型。代码中不是两个独立 checkpoint，而是一个 `UNetModel_newpreview` 内部包含：

- 第一阶段：`Generic_UNet`，即 `unet.hwm.*`
- 第二阶段：主 diffusion U-Net

checkpoint 中确认包含 `unet.hwm.*` 权重，因此 `model_phy*.pth` 保存的是完整组合模型。

训练 checkpoint 目录：

```text
checkpoints_dynamic_rmdm_4gpu_no_ckpt_75k
```

主要训练配置：

```text
GPUs: 0,1,2,3
max_steps: 75000
batch_size: 32
num_channels: 96
attention_resolutions: 16
diffusion_steps: 1000
noise_schedule: linear
mixed_precision: no
gradient_checkpointing: disabled
save_interval: 5000
```

训练 loss：

```text
loss = loss_diff + loss_cal_recon + loss_pinn
```

其中：

- `loss_diff`: diffusion noise prediction loss
- `loss_cal_recon`: 第一阶段 `cal` 输出监督
- `loss_pinn`: 对 `cal` 的物理约束

## 3. 训练 Loss 走势

按每 5000 step 统计的训练 loss 显示，主要下降发生在前 1-2 万 step，后期进入平台期。

```text
0-4999      loss 0.05185
5000-9999   loss 0.02031
10000-14999 loss 0.01814
20000-24999 loss 0.01714
40000-44999 loss 0.01636
60000-64999 loss 0.01606
70000-74999 loss 0.01596
```

后期平台主要来自：

- `diff` 已经很低，约 `0.0014-0.0017`
- `pinn` 基本稳定在 `0.0083-0.0084`
- `cal` 仍有小幅下降，但幅度有限

## 4. Val 100 Checkpoint 选择实验

为了选择最终模型，从 val split 固定随机抽取 100 个样本，对所有 checkpoint 运行完整 DDIM 推理。

评估协议：

```text
split: val
num_samples: 100
seed: 1234
sampler: DDIM
ddim_steps: 50
ddim_eta: 1.0
prediction: full diffusion sampling
```

结果文件：

```text
eval_dynamic_rmdm_val100_full_ddim50_seeded/val100_checkpoint_metrics_ddim_all.csv
eval_dynamic_rmdm_val100_full_ddim50_seeded/val100_checkpoint_metrics_ddim_all.json
```

Val 100 结果：

| Checkpoint | Step | MSE | NMSE | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|
| model_phy_step5000.pth | 5000 | 0.009712 | 0.032914 | 20.578 | 0.7533 |
| model_phy_step10000.pth | 10000 | 0.010192 | 0.034507 | 20.248 | 0.7684 |
| model_phy_step15000.pth | 15000 | 0.009304 | 0.031476 | 20.682 | 0.7722 |
| model_phy_step20000.pth | 20000 | 0.009607 | 0.032512 | 20.491 | 0.7659 |
| model_phy_step25000.pth | 25000 | 0.009361 | 0.031648 | 20.698 | 0.7755 |
| model_phy_step30000.pth | 30000 | 0.009798 | 0.033064 | 20.441 | 0.7748 |
| model_phy_step35000.pth | 35000 | 0.009841 | 0.033284 | 20.453 | 0.7691 |
| model_phy_step40000.pth | 40000 | 0.009041 | 0.030537 | 20.832 | 0.7787 |
| model_phy_step45000.pth | 45000 | 0.009083 | 0.030749 | 20.809 | 0.7775 |
| model_phy_step50000.pth | 50000 | 0.009135 | 0.030875 | 20.813 | 0.7790 |
| model_phy_step55000.pth | 55000 | 0.008841 | 0.029939 | 21.057 | 0.7818 |
| model_phy_step60000.pth | 60000 | 0.009345 | 0.031578 | 20.698 | 0.7699 |
| model_phy_step65000.pth | 65000 | **0.008531** | **0.028917** | **21.182** | 0.7840 |
| model_phy_step70000.pth | 70000 | 0.008611 | 0.029150 | 21.129 | 0.7781 |
| model_phy.pth | 75000 | 0.008887 | 0.030103 | 20.969 | **0.7850** |

选择结论：

```text
Best MSE/NMSE/PSNR: model_phy_step65000.pth
Best SSIM: model_phy.pth
```

最终完整 test 采用 `model_phy_step65000.pth`，因为 MSE/NMSE/PSNR 三个主要误差指标最优。

## 5. Best Model 可视化验证

最佳模型：

```text
checkpoints_dynamic_rmdm_4gpu_no_ckpt_75k/model_phy_step65000.pth
```

可视化输出目录：

```text
visualizations_dynamic_rmdm_best_step65000_val
```

主要文件：

```text
visualizations_dynamic_rmdm_best_step65000_val/best_step65000_val16_contact_sheet.png
visualizations_dynamic_rmdm_best_step65000_val/comparisons/
visualizations_dynamic_rmdm_best_step65000_val/arrays/best_step65000_val100_predictions.npz
visualizations_dynamic_rmdm_best_step65000_val/metrics_summary.txt
```

该可视化使用完整 DDIM-50 推理，不是 `cal` 分支直接输出。

Val 100 可视化对应指标：

```text
MSE:  0.008499 +/- 0.004018
NMSE: 0.028804 +/- 0.015700
PSNR: 21.196403 +/- 2.128924
SSIM: 0.785040 +/- 0.092446
```

定性观察：

- 大尺度传播结构和阴影区域基本对齐。
- 预测图相比 GT 更平滑。
- 细碎动态遮挡、边缘高频纹理和局部强误差区域仍有损失。

## 6. 完整 Test Set 评估

完整 test set 使用最佳模型 `model_phy_step65000.pth`。

评估协议：

```text
split: test
num_samples: 225000
sampler: DDIM
ddim_steps: 50
ddim_eta: 1.0
seed: 1234
batch_size: 20
workers: 8
GPUs: 0,1
num_shards: 2
```

评估输出目录：

```text
eval_dynamic_rmdm_test_full_step65000_ddim50
```

主要结果文件：

```text
eval_dynamic_rmdm_test_full_step65000_ddim50/summary_test_merged.json
eval_dynamic_rmdm_test_full_step65000_ddim50/details_test_merged.csv
eval_dynamic_rmdm_test_full_step65000_ddim50/details_test_shard0of2.csv
eval_dynamic_rmdm_test_full_step65000_ddim50/details_test_shard1of2.csv
```

完整 test set 汇总结果：

| Metric | Mean | Std | Min | Max | Count |
|---|---:|---:|---:|---:|---:|
| MSE | 0.004005 | 0.008767 | 0.000332 | 0.131911 | 225000 |
| MAE | 0.024695 | 0.024375 | 0.005290 | 0.305812 | 225000 |
| RMSE | 0.055981 | 0.029515 | 0.018210 | 0.363195 | 225000 |
| NMSE | 0.021022 | 0.125326 | 0.000837 | 3.581504 | 225000 |
| PSNR | 25.705400 | 3.256156 | 7.193119 | 34.793743 | 225000 |
| SSIM | 0.887316 | 0.075279 | 0.281782 | 0.987903 | 225000 |

论文主表建议报告：

```text
MSE  = 0.004005
NMSE = 0.021022
PSNR = 25.7054 dB
SSIM = 0.8873
```

## 7. 评估耗时

完整 test 评估拆成两个 shard，每个 shard `112500` 样本。

Shard 0：

```text
total_time_seconds: 26083.28
samples_per_second: 4.313
processed_samples: 112500
```

Shard 1：

```text
total_time_seconds: 27582.41
samples_per_second: 4.079
processed_samples: 112500
```

总 wall-clock 时间取较慢 shard，约：

```text
7.66 hours
```

## 8. 相关脚本

本次实验新增或使用的脚本：

```text
eval_val100_checkpoints.py
eval_full_checkpoint.py
merge_eval_shards.py
```

用途：

- `eval_val100_checkpoints.py`: 对多个 checkpoint 做固定 val100 的完整 DDIM 对比。
- `eval_full_checkpoint.py`: 对单个 checkpoint 做完整 split 评估，并输出逐样本 CSV。
- `merge_eval_shards.py`: 合并多 GPU shard 的完整评估结果。

## 9. 注意事项

1. `cal` 分支直接预测结果不能代表最终论文式完整模型结果。
2. 最终测试必须使用完整 diffusion sampling，即 DDIM/DDPM/DPM 采样流程。
3. 本文档中的完整 test 结果采用 DDIM-50；若后续改用 DPM-Solver 或减少采样步数，需要作为新的 inference setting 单独记录，不能与当前结果混报。
4. 当前 best checkpoint 是基于固定 val100 选择的 `model_phy_step65000.pth`。

# RMDM 时序一致性 9 点汇报摘要

## 结论

当前最强版本是：

```text
deep 3D residual head + kpath-delta loss
/data/fzj/RMDM/checkpoints_residual_deep3d_h64_l4_kpathdelta_alltrain_5k/residual_student_final.pth
```

它在 20% test clips 等间隔抽样评估中，相比 frozen RMDM 同时提升单帧指标和时间指标。

## 评估设置

```text
test full clips: 6750
sampled clips: 1350
sampling: evenly spaced 20% index sampling
clip_length: 32
frame_stride: 32
DDIM steps: 20
eval alpha: 0.03
paired same-noise baseline vs candidate
```

## 主结果

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

## 消融对比

| Version | dPSNR | dTOF | dTWE | dKFlow EPE | dTD-L1 |
|---|---:|---:|---:|---:|---:|
| Step2 range shallow | +0.028468 | -0.015904 | -0.000176 | -0.031696 | -0.000172 |
| kpath-delta shallow | +0.026229 | -0.022538 | -0.000194 | -0.034675 | -0.000178 |
| deep3d kpath-delta | +0.068056 | -0.054118 | -0.000714 | -0.121228 | -0.000678 |

## 方法一句话

```text
冻结原始强单帧 RMDM，只训练一个 zero-init 3D residual head。
3D head 跨帧看 cond / x_t / eps_base，修正 eps_base。
k-flow/K2 只用于定位关键传播路径和 static/dynamic 区域，不作为输入条件。
```

## 代表视频

不是挑最好看的视频，使用固定规则选出的同一组代表 clip：

```text
clip indices: 6378, 1821, 120
selection rule:
  mean/median-like
  moderate temporal improvement
  difficult/regression-side case

deep3d videos:
/data/fzj/RMDM/experiments/residual_deep3d_h64_l4_kpathdelta_alltrain_5k_eval20pct_alpha003/representative_videos_same_indices

shallow Step2 videos:
/data/fzj/RMDM/experiments/residual_step2_dynamic_q85_w1e4_alltrain_5k_eval20pct_alpha003/representative_videos
```

## 详细记录

```text
/data/fzj/RMDM/docs/rmdm_temporal_pilot_中文进展记录.md
```

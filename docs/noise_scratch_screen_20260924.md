# 噪声注入从头训练筛选（2026-09-24）

用户已明确纠正：需要随机初始化、从头训练，不是从原始 checkpoint 微调。用户选择每组 4,000 步短程筛选。此要求取代此前交接文档中的微调下一步；之前 300 步微调的结果不回答当前问题。

## 对照协议

- 三组：归一化方差通道 `(sigma/0.09)^2`、归一化标准差通道 `sigma/0.09`、AdaNorm。
- 使用原 RMDM 的网络尺寸，但模型构造没有 checkpoint 参数、不读取任何预训练权重。公共主干采用相同随机种子和完全一致的随机初始张量。
- 通道组在共同随机主干上扩展两个输入卷积，新增列零初始化；AdaNorm 保留原 GroupNorm/BatchNorm，添加零初始化的噪声 scale/shift 分支。
- BatchNorm 从头更新统计量，全部有效参数参与优化；不使用冻结主干、冻结统计量或低噪声教师约束。
- 每组 4,000 步，batch 16，AdamW lr=1e-4，前 200 步线性 warmup，weight decay=0，梯度裁剪 1，BF16。每 1,000 步验证。
- 训练数据、采样与观测噪声分布、扩散目标、HWM 干净重建与 PINN 损失沿用当前 pipeline。
- 验证仅使用 val：采样率 1%/2%/3%，sigma=0/0.01/0.03/0.05/0.07/0.09，DDIM20，每条件 40 帧，batch 16。不使用 test。
- 主干随机 seed=20260924，新增模块 seed=20260925，训练顺序与扩散采样 seed=20260926。
- 训练与验证的采样 mask、观测噪声、扩散噪声及时间步统一在 CPU 生成；HWM Dropout2d 也使用 CPU 生成的通道 mask，再移至 GPU。首批样本 ID、mask 数量、sigma、时间步、噪声取值、共享初始权重均记录并跨机器核对。

## 实现与限制

入口：`experimental/rmdm_direct_noise/scratch.py`。不同 GPU/PyTorch 版本仍可能造成浮点计算差异；运行环境记录在各组 `metadata.json`。4,000 步是筛选预算，不代表充分收敛。首次 GPU 随机数版本发现同 seed 噪声不配对后已中止，不能混入正式 CPU 随机数配对结果。

状态与实际命令以 `.agents/runs/20260924_noise_scratch_screen.yaml` 为准。

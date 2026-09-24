# RMDM 观测噪声实验交接（2026-09-24）

> 本会话最新用户调整：原始 RMDM 直接作为 baseline，远程不再训练 RMDM 不同版本。新增 baseline 候选为 RadioDiff、RadioUNet、RME-GAN，均仅用无观测噪声数据训练；本调整取代下文 RMDM 变体训练计划。已核对 Nice2 原始 run 的 train_config.json：packed cache、clean16 split、without_tx=true、采样率 1–10；对应代码的观测为 sampling_mask * target，不添加观测噪声，扩散目标图正常加噪。三个 baseline 已完成无 Tx 稀疏观测适配，并于 2026-09-24 12:20 在 Nice2 启动正式训练队列：GPU 0 为 RadioUNet→RME-GAN，GPU 1 为 RadioDiff VAE→潜空间扩散。全部阶段已通过真实数据 smoke；源码 commit 为 fbdabc9。实现与适配说明见 `experimental/radio_baselines/README.md`，实时状态恢复入口为 `.agents/runs/20260924_clean_radio_baselines.yaml`。训练和检查点选择均只用干净观测，测试集未使用。

> 后续用户纠正：要求随机初始化、从头训练，不加载原模型权重；已选择三组各 4,000 步筛选。下文关于“从原始 checkpoint 微调”的下一步已被取代，参见 [从头训练筛选](noise_scratch_screen_20260924.md)。历史结果仍按原实验身份保留。

## 用户当前要求与必须纠正的结论

用户要的是**单个模型**同时保留低噪声生成质量、提高高噪声鲁棒性。下一轮应先测试从原始 RMDM 初始化、**解冻全部参数**的噪声训练；再研究更强的噪声条件结构和训练目标。不要把两个 checkpoint 按已知噪声切换的路由当成新模型或架构改进。

之前对“噪声版”的称呼混淆了三条路径，务必分开：

| 名称 | 代码 / 权重 | 实际训练范围 | 结论 |
| --- | --- | --- | --- |
| 原始 RMDM | `train_sparse_dynamic_rmdm.py`，远程 `rmdm_original_clean16_20260922/epoch_009.pth` | 原始模型 | 直接输入稀疏观测；配对基线。 |
| 完整噪声感知 T1 | 主仓库 `experimental/noise_temporal_rmdm/model.py`，远程资源 `noise_temporal_rmdm_t1_20260922_complete/checkpoint/step_020250.pth` | 方差嵌入、方差调制观测编码器、新 HWM、扩散主干等约 6322 万参数经过训练 | 这是先前表格里的“旧噪声版”。它与原始 RMDM 的结构不同，不能用其差异归因于单独的方差输入。 |
| 原始 RMDM + 方差图 | 本 worktree `experimental/rmdm_direct_noise/model.py`、`train.py`，远程 `rmdm_direct_noise_t1_20260924/step_*.pth` | 仅两个首层卷积的新增方差通道权重更新，原始参数冻结 | **没有**完成全参数解冻实验，也没有证明完整结构的潜力。 |

`experimental/rmdm_direct_noise/hybrid.py` 组合的是**原始 RMDM + 完整噪声感知 T1**，并非“原始 RMDM + 方差图”。门控阈值 `(0.04, 0.055, 0.07)` 按采样率在验证集确定。DDIM20 测试子集（18 条件、每条件 80 帧）平均 MSE：原始 0.001155，完整噪声感知 T1 0.000980，双 checkpoint 路由 0.000899。0.000899 是系统路由分数，**不能**证明一个新模型学会了自适应抗噪。不要把这个测试集用于后续结构、阈值或 checkpoint 选择。

## 方差图模型的真实运行状态

- 本地源码真源：`/data_p6/fzj/projects/RMDM_direct_noise`，分支 `experimental/rmdm-direct-noise`。远程计算机 Nice2 (`fzj@10.11.113.168 -p 2137`)，运行目录 `/data_16T_137/fzj/RMDM/runs/rmdm_direct_noise_t1_20260924`；远程源码以 GitHub 同步，不在远程直接修改受 Git 管理的源码。
- 远程训练初始化自原始 RMDM `epoch_009.pth`，全局 batch 128，最多 40k，4k 验证一次，连续 3 次无提升早停，学习率 2e-4。实际第 20k 步早停；最佳验证检查点是第 8k 步，8 条件 DDIM20 平均 MSE **0.00143521**。此前配对验证记录的原始 RMDM 为 0.00122456、完整噪声感知 T1 为 0.00092301（同验证协议）。
- 第 20k 步状态文件显示 `stale=3`。没有发现仍在运行的 `experimental.rmdm_direct_noise.train` 进程。方差图版的正式测试成绩**尚未建立**；先前 0.000899/0.000980/0.001155 表格与它无关。
- 源码中 `config['in_ch']=7`；旧输入卷积被膨胀，新增方差列零初始化。`train.py` 冻结全部参数，再只打开 `unet.input_blocks.0.0.weight` 与 `unet.hwm.conv_blocks_context.0.blocks.0.conv.weight` 两个张量，并用梯度 mask 仅更新各自新增方差列。原始稀疏 RSS 和 mask 仍直接进入 HWM 与扩散 U-Net。仅加一张方差图且只训练首层两列，不能据此否定全参数训练或更丰富的条件结构。

## 下一窗口的具体工作

1. 先阅读本文件、两个 worktree 的 `AGENTS.md`、`experimental/rmdm_direct_noise/{model,train}.py`、完整噪声感知 T1 的 `model.py` 与远程验证 JSON；再次核对各模型身份、输入输出、训练参数和配对协议。
2. 在当前 direct-noise 分支做一个**独立的全参数解冻基线**，从同一个原始 checkpoint 初始化。保留原始稀疏观测直通与已知方差输入；所有原始权重都应有梯度，不能沿用仅允许新增列更新的 mask。用较小主干学习率并单列记录其值，避免把 `2e-4` 的新增列学习率不加区分地施加到整个预训练模型。先用小规模真实数据 smoke 确认权重更新、数据和 DDIM 验证贯通，再提交源码、通过 GitHub 同步到 Nice2，启动正式训练。
3. 与此基线分开研究结构改进：应让 σ 调节**观测可信度/特征融合**，使稀疏 RSS 在高噪声时可被抑制、低噪声时保持原始直通；可以考虑零初始化的噪声条件残差或观测修正模块。训练时使用干净 RSS 监督观测处理，并用最终 DDIM 验证选检查点；不要仅靠噪声预测训练 loss 判断重建质量。保留 σ=0 的原模型初始化行为，并通过配对验证确认。
4. 统一验证条件：采样率 1/2/3，σ=0/0.01/0.03/0.05/0.07/0.09；同一 split、mask、观测噪声和扩散初始噪声；逐条件报告 MSE，兼看低噪声退化与高噪声增益。最多 40k、每 4k 验证、3 次不提升早停可作为起点。只用验证集决定超参/检查点，测试集最后一次性报告。全参数基线、结构改进和双模型路由分开列，绝不混为一个模型。
5. 检查 Nice2 的实时 GPU、进程、远程代码 commit 和资源目录后再启动新任务，不覆盖既有结果。原 run 记录 `.agents/runs/20260924_rmdm_direct_noise_t1.yaml` 曾写成 `running`，已不准确；以远程 `status.json`、日志和进程为准更新。

另有**独立的本地 HVDiT 管线**在 `/data_p6/fzj/projects/RMDM` 的 `experimental/noise-hvdit-source` 分支、tmux `noise_hvdit_pipeline_2gpu`、GPU 0/2 运行。2026-09-24 最近查看时为 W16-x0 阶段约第 5520 步。不要把它、完整噪声感知 RMDM T1 和本 direct-noise 分支混在一起，也不要为了本实验停止该管线。

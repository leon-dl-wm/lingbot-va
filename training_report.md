# LingBot-VA Post-Training 训练报告

> 平台:腾讯云 TI-ONE · 8 × HCC-BW1000(单卡 64G)· DTK 26.04 / torch 2.7.1
> 任务:RoboTwin post-training(复现论文 benchmark 配置)
> 报告生成时间:2026-09-18 22:15(训练进行中,step 4579/50000)

---

## 1. 训练配置摘要

| 项 | 值 |
|---|---|
| 底座模型 | robbyant/lingbot-va-base(shared backbone,~5B,bf16) |
| 数据集 | robotwin-clean-and-aug-lerobot(50 任务 × 500 episodes,414G,含预提取 VAE latents) |
| 有效 batch | **32**(8 卡 × batch_size 1 × grad_accum 4) |
| 优化器 | AdamW(lr 1e-5,β 0.9/0.95,wd 0.1,warmup 10 步后恒定) |
| 精度/并行 | bf16 参数 + fp32 归约,FSDP2 fully_shard,激活检查点 |
| 注意力 | flex_attention 块因果掩码(torch.compile) |
| 目标步数 | 50,000(本次计划 10,000 后停,自动 watcher 就位) |
| checkpoint | 每 1000 步保存(diffusers 格式,bf16,9.5G/个) |

## 2. Training Loss 曲线

![loss curves](train_out/loss_curves.png)

*(上:latent_loss 视频世界模型损失;中:action_loss 对数轴;下:grad_norm 与 clip 阈值。蓝/橙/绿细线为逐步原始值,粗线为 51 点滑动平均)*

## 3. Loss 里程碑(10 步窗口均值)

| step | latent_loss | action_loss | grad_norm |
|---:|---:|---:|---:|
| 0 | 0.2937 | 0.2221 | 0.883 |
| 100 | 0.2377 | 0.0137 | 0.077 |
| 500 | 0.2122 | 0.0055 | 0.056 |
| 1000 | 0.2027 | 0.0047 | 0.052 |
| 2000 | 0.1890 | 0.0031 | 0.042 |
| 3000 | 0.1872 | 0.0024 | 0.046 |
| 4000 | 0.1824 | 0.0026 | 0.041 |
| 4578(最新) | 0.1757 | 0.0020 | 0.040 |

## 4. 健康度评估

### ✅ 结论:训练健康,loss 正常下降

| 维度 | 判定 | 依据 |
|---|---|---|
| **action_loss** | ✅ 收敛良好 | 0.222 → 0.002(**~110 倍**),前 100 步快速下降后进入平台期缓慢优化,符合动作预测任务特性 |
| **latent_loss** | ✅ 持续下降 | 0.294 → 0.176(**-40%**),4500 步仍在单调下行,无平台停滞 |
| **grad_norm** | ✅ 稳定 | 全程最大 1.40 < clip 阈值 2.0,稳态 ~0.04,无爆炸无消失 |
| **NaN/异常** | ✅ 零 | 全程无 NaN、无 inf、无 Traceback |
| **过拟合迹象** | ✅ 无 | 训练 loss 无回升;数据集 2.5 万 episodes,当前仅 0.6 epoch,远未到过拟合区间 |
| **速度** | ✅ 稳定 | 45.6-45.8 s/it 全程恒定(编译预热后),8 卡显存 42-53G/64G |

### 说明

- **无 validation loss**:官方训练代码不含验证循环(代码现状),质量验证依赖 5000 步自动 i2va 评测(生成 demo.mp4 直观检查视频-动作生成质量)
- loss 波动(原始值 ±0.03)属正常:每步随机采样 chunk_size∈[1,5)、window_size∈[4,65) 与噪声时间步,任务难度天然不同

## 5. 资源与吞吐

| 指标 | 值 |
|---|---|
| 吞吐 | 4 samples / 45.7s ≈ 0.088 samples/s(有效 batch 32) |
| 显存 | 42-53 GB / 64 GB × 8 卡 |
| 已完成 | 4579 步 / 58.0 小时 |
| ETA(10K 停) | **9月21日 ~19:00**(剩 ~5426 步 ≈ 68.8h) |
| ETA(50K 全程) | ~10月14日 |

## 6. Checkpoints

| checkpoint | 保存时间 | 状态 |
|---|---|---|
| step_1000 | 09-17 00:48 | ✅ 可评测 |
| step_2000 | 09-17 16:05 | ✅ 可评测 |
| step_3000 | 09-18 03:22 | ✅ 已验证可加载(839 tensors,bf16) |
| step_4000 | 09-18 14:40 | ✅ 可评测 |
| step_5000 | 预计 09-19 ~13:00 | ⏳ 将触发**自动 i2va 评测** |

评测任意 checkpoint:`bash script/eval_checkpoint.sh <step>`(自动 symlink 冻结组件、patch attn_mode→torch、跑 i2va demo)

## 7. 已知风险与缓解

| 风险 | 缓解 |
|---|---|
| resume 只恢复权重(optimizer/step 状态保存被官方注释) | checkpoint 每 12.7h 落盘,崩溃最多损失 12.7h;重启后 step 归零但权重热启动,损失可控 |
| 单节点无容错,watcher 依赖 /tmp 日志 | monitor 快照已持久化到 CFS(report.md);重启后按 report.md 运维节命令重启 watcher |
| 多卡推理 server 的 HCCL broadcast 缺陷 | 单卡 i2va 评测不受影响;多卡评测需 workaround(见 project.md 10.3) |

## 8. 后续计划

1. **9月19日 ~13:00**:checkpoint_step_5000 自动 i2va 评测 → 结果追加到 report.md
2. **9月21日 ~19:00**:checkpoint_step_10000 保存后自动停训
3. 停训后:对比 5K/10K checkpoint 的 i2va 生成质量;如需完整复现论文(50K 步)可 `resume_from` 续训
4. 最终 benchmark 评测(RoboTwin 仿真)建议在 NVIDIA 环境跑客户端,DCU 跑推理 server

---

*数据来源:`/tmp/train.log`(4580 个逐步数据点);快照持续记录于 `report.md`;本报告可随时用最新日志重新生成。*

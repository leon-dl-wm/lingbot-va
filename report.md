# LingBot-VA Post-Training 运行报告(BW1000/DCU 平台)

> 目标:在 TI-ONE 8×BW1000 节点上跑通 LingBot-VA Post-Training(RoboTwin),复现论文 benchmark。
> 本报告记录环境准备、训练启动、训练过程指标。训练快照由监控脚本持续追加到文末。

---

## 1. 运行信息

| 项目 | 值 |
|---|---|
| 启动时间 | 2026-09-16 11:09 (CST) |
| 硬件 | 8 × HCC-BW1000(单卡 64G 显存),300C/2000G |
| 镜像 | dtk26.04-torch2.7.1-py3.11-hccpd1-hccl-v1.0-vla-openpi-torchcodec |
| 训练框架 | torch 2.7.1+das / FSDP2 fully_shard / flex_attention(compiled) / HCCL(nccl backend) |
| Python 环境 | 仓库内 venv `va_env`(system-site-packages + diffusers 0.36.0 + transformers 4.55.2 + easydict + ftfy + lerobot 0.3.3 --no-deps) |
| 启动命令 | `NGPU=8 CONFIG_NAME='robotwin_train' bash script/run_va_posttrain.sh` |
| 日志 | `/tmp/train.log` |

## 2. 数据与模型

| 项目 | 值 |
|---|---|
| 底座模型 | `modelscope download Robbyant/lingbot-va-base` → `/home/tione/notebook/model/lingbot-va-base`(transformer bf16 ~10G,attn_mode 已是 "flex" 无需改) |
| 数据集 | `robotwin-clean-and-aug-lerobot`(98G tar.gz 分卷,解压后 414G)→ `/home/tione/notebook/data/Robbyant/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500` |
| 数据规模 | 50 任务 × 500 episodes;82,414 个 VAE latent .pth;82,500 个 mp4;LeRobot v2.1 + action_config ✅ |
| empty_emb.pt | 归档中不含,已用底座 UMT5 text_encoder 按推理侧 negative prompt 逻辑生成(空字符串,512×4096 bf16,seq_len=1) |

## 3. 训练配置(相对官方默认的改动)

| 配置 | 值 | 说明 |
|---|---|---|
| `dataset_path` | `/home/tione/notebook/data/Robbyant/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500` | 必改 |
| `wan22_pretrained_model_name_or_path` | `/home/tione/notebook/model/lingbot-va-base` | 必改 |
| `enable_wandb` | `False` | 官方脚本 wandb key 是占位符,必改 |
| `batch_size` | 1(默认) | 8 卡 × 1 |
| `gradient_accumulation_steps` | **4**(默认 1) | 有效 batch = 8×1×4 = 32,官方建议 ≥32 |
| 其余(lr 1e-5, num_steps 50000, save_interval 1000, cfg_prob 0.1 等) | 官方默认 | 对齐论文 |

脚本改动:`run_va_posttrain.sh` 中把 venv bin 加入 PATH(用 va_env 的 python)。

## 4. 训练过程指标

### 启动阶段

- 数据集初始化:50 个子数据集多进程加载(每 rank Pool),约 10 分钟
- 首步含 flex_attention torch.compile 编译,耗时 128s;第 2 步起进入稳态

### 稳态指标(截至 step 16)

| 指标 | 值 |
|---|---|
| 速度 | **~52.5 s / optimizer step**(= 4 micro-batch,即 ~13s/样本前反向) |
| 显存 | 43-48 GB / 64 GB(8 卡均匀) |
| GPU 利用率 | 采样瞬时 20%-92%,波动正常(数据加载 + 梯度累积间隙) |
| latent_loss | 0.27-0.31(缓慢下降) |
| action_loss | **0.22 → 0.078**(前 15 步快速下降) |
| grad_norm | 0.23-1.14,稳定无爆炸 |
| lr | warmup 10 步后到 1e-5 恒定 |

### 吞吐换算

- 8 卡合计:4 samples / 52.5s ≈ **0.076 samples/s**(有效 batch 32)
- 对照海光实测 PDF:Fastwam(同类 WAM 模型)BW1000 单机 8 卡 30.31 samples/s(batch 96)——
  本模型序列长得多(视频+动作双流、flex 块因果掩码、激活检查点重算),量级差异属预期

### 完整训练 ETA

- 50,000 步 × 52.5s ≈ **729 小时 ≈ 30.4 天**
- checkpoint 每 1000 步(~14.6h)保存到 `./train_out/checkpoints/checkpoint_step_N/`
- 论文 RoboTwin benchmark 对应官方 posttrain 权重即此配置训出;如需提前评测,
  可用任意 checkpoint(建议 ≥5000 步)先跑 i2va demo 验证质量

## 5. 监控与运维

```bash
# 实时日志
tail -f /tmp/train.log | grep -oE "latent_loss=[0-9.]+, action_loss=[0-9.]+, step=[0-9]+, grad_norm=[0-9.]+"

# GPU 状态
watch -n 5 hy-smi

# checkpoint
ls train_out/checkpoints/

# 停止训练(如需)
# 找到 torchrun 主进程 PID 后 kill;checkpoint 可从最近 step 恢复(resume_from)
```

## 6. 已知事项

1. **训练正常判据**:latent_loss 前几百步缓降、action_loss 快速下降后缓降、grad_norm 个位数 —— 当前全部满足
2. **30 天 ETA**:如需加速,可选方案:减少 num_steps(论文 LIBERO 用 4-5K 步,RoboTwin 50K)、
   减 gradient_accumulation_steps(有效 batch 降为 8)、或等海光 HyperAcc/gemm 优化
3. **评测**:训练完成后需把 checkpoint `transformer/config.json` 的 `attn_mode` 改回 `"torch"`,
   再走 RoboTwin 评测(server-client);仿真评测依赖 sapien+vulkan,建议在 NVIDIA 环境跑客户端
4. 多卡推理 server 的 `broadcast_object_list` HCCL 缺陷仍在(评测单卡模式不受影响)

---

## 7. 训练快照(监控脚本自动追加)

| 时间 | step | latent_loss | action_loss | grad_norm | s/it | 显存(卡0) |
|---|---|---|---|---|---|---|
| 09-16 11:38 | 16 | 0.2755 | 0.0687 | 0.15 | 53.51s/it | 0MiB |
| 09-16 11:38 | 16 | 0.2755 | 0.0687 | 0.15 | 53.51s/it | 45853MiB |
| 09-16 12:14 | 5 | 0.2863 | 0.1794 | 0.92 | 59.28s/it | 45792MiB |

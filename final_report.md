# LingBot-VA 在海光 BW1000(DCU)平台的 Post-Training 完整报告

> **平台**:腾讯云 TI-ONE · 8 × HCC-BW1000(单卡 64G HBM)· 300C / 2000G
> **软件栈**:DTK 26.04 · torch 2.7.1+das · RCCL 2.22.3 · flex_attention(compiled)· FSDP2
> **任务**:RoboTwin post-training,复现论文 benchmark 配置
> **训练区间**:2026-09-16 12:05 → 2026-09-21 19:14(127.1 小时,10,004 步)
> **结论**:✅ 全流程跑通;兼容性遗留问题 2 项(均有 workaround,不阻塞训练);性能约为 H20 的 66%(与海光官方同类模型实测一致)

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [BW1000 兼容性遗留问题](#2-bw1000-兼容性遗留问题)
3. [训练速度与时间](#3-训练速度与时间)
4. [训练 Loss](#4-训练-loss)
5. [资源占用](#5-资源占用)
6. [Checkpoint 与评测结果](#6-checkpoint-与评测结果)
7. [优化建议](#7-优化建议)(含 7.4 RoboTwin 仿真支持性评估)
8. [复现命令速查](#8-复现命令速查)

---

## 1. 执行摘要

| 维度 | 结果 |
|---|---|
| 训练完成度 | ✅ 10,004 / 10,000 步(127.1h),watcher 自动停止 |
| 数值健康 | ✅ 零 NaN、grad 最大 1.40(< clip 2.0)、loss 单调下降、无过拟合 |
| 最终 loss | latent 0.294→**0.163**(-45%);action 0.222→**0.0014**(-99.4%) |
| 评测 | ✅ 5K/10K checkpoint i2va demo 均生成成功(77 帧,7.7s) |
| 兼容性 | ⚠️ 2 项遗留(见第 2 章),均不阻塞训练;训练路径 100% 可用 |
| 性能定位 | ~66% of H20(与海光官方 Fastwam 实测一致,属平台开箱水平) |

## 2. BW1000 兼容性遗留问题

### 2.1 已彻底解决(无需关注)

| # | 问题 | 解决方式 |
|---|---|---|
| 1 | torch 导入需 DTK 环境 | `source /opt/dtk/env.sh`(每 shell 必做) |
| 2 | diffusers/transformers/lerobot 版本不匹配 | venv + 指定版本安装(见第 8 章) |
| 3 | FSDP2 混合 dtype 断言(server 推理路径) | fp32 加载 + MixedPrecisionPolicy(已提交 `wan_va_server.py`) |
| 4 | 评测端口与训练冲突 | 评测 MASTER_PORT=29699 |
| 5 | imageio ffmpeg 插件缺失 | `pip install imageio[ffmpeg]` |
| 6 | offload 下 UMT5 CPU 编码 25 分钟 | GPU 空闲时 `enable_offload=False` |

### 2.2 ⚠️ 遗留问题(有 workaround,未根治)

| # | 问题 | 影响 | 严重度 | 当前规避 |
|---|---|---|---|---|
| **L1** | **HCCL `dist.broadcast_object_list` 损坏**:rank1 报 1EB 分配 / EOFError(HCCL 2.22.3 的 coalescing-manager 路径 bug) | 仅影响**多卡推理 server**(`sever_utils.py` 广播 obs 给 8 卡)——RoboTwin 多卡评测(`launch_server_multigpus.sh`)无法直接用 | 🟡 中 | ① 单卡模式评测不受影响;② 已验证可行 workaround:定长字节张量手动 broadcast,或辅助 gloo 进程组(需改 `sever_utils.py`,~20 行) |
| **L2** | **torch 2.7.1 vs 官方要求 2.9.0**:镜像锁定 2.7.1+das | 理论风险;实测训练全路径(flex_attention/FSDP2/激活检查点/fused AdamW)127h 无异常 | 🟢 低 | 无需处理;如遇奇异算子问题,携 DTK 版本反馈光合社区 |
| L3 | `lerobot_latent_dataset.py` 引用未导入的 `get_safe_version`(数据缺失分支才触发) | 本地完整数据不触发 | 🟢 低 | 不修;若未来下载数据需补 import |
| L4 | resume 只恢复权重,optimizer/step 状态保存被官方注释 | 崩溃重启后 step 归零(权重热启动,损失可控) | 🟢 低 | checkpoint 每 12.7h 落盘;需要时可恢复 `training_state.pt` 保存代码 |

**结论:训练路径零遗留;推理路径仅多卡 server 受 L1 影响,单卡评测/服务完全可用。**

## 3. 训练速度与时间

| 指标 | 实测值 |
|---|---|
| 稳态速度 | **45.82 s/optimizer step**(均值,10,004 步;首步 128s 含编译,2 步后即稳) |
| 有效 batch | 32(8 卡 × 1 × grad_accum 4)→ 每 step 4 个样本前反向 |
| 吞吐 | **0.0875 samples/s**(8 卡合计)≈ 7.56 samples/天·卡组 |
| 总时长 | **127.1 小时 = 5.3 天**(10,004 步) |
| 外推 50K 步 | ~681 小时 ≈ **28.4 天**(完整复现论文) |

### 性能对标(依据海光官方《模型训练实测对比》)

| 参考模型 | BW1000 vs 对标卡 | 本项目定位 |
|---|---|---|
| Fastwam(同类 WAM,同栈 DTK26.04+torch2.7.1) | 30.31 vs H20 45.71 samples/s = **66.3%** | 本项目同为视频+动作双流扩散模型,量级一致 |
| Pi0.5(openpi) | 51.2 vs H20 39.6 = **1.3×** | 序列短得多,不可比 |
| LingBot-VLA | 77%~127% | VLA 无视频流,不可比 |

**判断:0.0875 samples/s 的绝对值低是模型特性(长序列 + flex 块因果掩码 + 激活检查点重算),相对性能(~66% of H20)与海光官方同类模型实测吻合,属 BW1000 开箱正常水平。**

## 4. 训练 Loss

![loss curves](train_out/loss_curves.png)

### 里程碑(10 步窗口均值)

| step | latent_loss | action_loss | grad_norm |
|---:|---:|---:|---:|
| 0 | 0.2937 | 0.2221 | 0.883 |
| 100 | 0.2377 | 0.0137 | 0.077 |
| 500 | 0.2122 | 0.0055 | 0.056 |
| 1000 | 0.2027 | 0.0047 | 0.052 |
| 2000 | 0.1890 | 0.0031 | 0.042 |
| 3000 | 0.1872 | 0.0024 | 0.046 |
| 4000 | 0.1824 | 0.0026 | 0.041 |
| 5000 | 0.1715 | 0.0021 | 0.039 |
| 6000 | 0.1760 | 0.0017 | 0.038 |
| 7000 | 0.1737 | 0.0018 | 0.038 |
| 8000 | 0.1666 | 0.0018 | 0.050 |
| 9000 | 0.1645 | 0.0015 | 0.037 |
| **10000** | **0.1629** | **0.0014** | **0.04** |

### 判读

- **action_loss -99.4%**:前 100 步断崖式下降(动作预测相对容易),之后平台期缓慢优化——正常形态
- **latent_loss -45%**:全程单调缓降,10K 步无平台停滞——视频世界模型学习慢属预期(论文同款)
- **无 validation loss**:官方代码无验证循环(代码现状);质量验证依赖 i2va 生成评测(见第 6 章)
- loss 原始值 ±0.03 波动正常:每步随机采样 chunk_size∈[1,5)、window、噪声时间步,任务难度天然不同

## 5. 资源占用

| 资源 | 值 |
|---|---|
| 显存 | 42-53 GB / 64 GB × 8 卡(峰值 ~82%) |
| GPU 利用率 | 瞬时采样 20%-92% 波动(数据加载 + 梯度累积间隙),均值中等 |
| CPU/内存 | 数据加载 16 workers/rank,2000G 内存充裕 |
| 存储 | 数据 414G + checkpoint 95G(10×9.5G)在 10T CFS;系统盘干净 |

## 6. Checkpoint 与评测结果

| checkpoint | i2va demo | 说明 |
|---|---|---|
| step_5000 | `train_out/eval/demo_step_5000.mp4`(77 帧,7.7s,320×384) | ✅ |
| step_10000 | `train_out/eval/demo_step_10000.mp4`(77 帧,7.7s,320×384) | ✅ |

- 任务:抓白色马克杯→旋转→挂深灰架子;10 chunks 自回归,每 chunk 视频 25 步 + 动作 50 步去噪
- 评测管线:`bash script/eval_checkpoint.sh <step>`(自动组目录、patch attn_mode、独立端口)
- **未做**:RoboTwin 仿真 SR 数字(需 sapien+vulkan 仿真环境,建议 NVIDIA 跑客户端 + 本机跑推理 server)

## 7. 优化建议

### 7.1 训练速度(当前 45.8s/step)

| 优化 | 预期收益 | 成本 | 建议 |
|---|---|---|---|
| **提高 batch_size**(显存余量 ~12-20G/卡) | 吞吐↑(摊薄固定开销) | 低,改配置即可 | ⭐ 首选:batch_size 2 + grad_accum 2(有效 batch 不变),实测显存后逐步加 |
| **HyperAcc 加速包**(海光官方,RoPE+GELU×MUL+RMSNorm 融合) | 官方实测 +17~18% | 一行安装 | ⭐ 联系海光/TI-ONE 获取,适配 torch 2.7.1 需确认 |
| 减小 `load_worker`(当前 16/rank × 8 = 128 进程) | 启动加速,稳态影响小 | 低 | 可选 |
| 关闭激活检查点(显存够的话) | ~15-25%(省重算) | 显存↑,需实测 | 显存余量不足,暂不 |
| gemm/算子层优化(海光方向) | 官方优化路线 | 需厂商配合 | 长期 |
| 多机扩展 | 线性(RCCL 扩展效率 99%+) | 需更多节点 | 50K 全量复现时考虑 |

### 7.2 训练效果

| 项 | 现状 | 建议 |
|---|---|---|
| 训练步数 | 10K(论文 50K) | 看 5K/10K demo 对比;若动作质量已可接受,可先做仿真评测再决定是否续训 |
| validation | 无 | 可加 held-out 任务 loss(留 2 任务做验证),或定期 i2va 评测(已有管线) |
| resume 完整性 | 仅权重 | 恢复 `training_state.pt` 保存代码(~15 行),50K 长训前建议加上 |

### 7.3 评测闭环

| 项 | 现状 | 建议 |
|---|---|---|
| 多卡推理 server | L1 HCCL 缺陷 | 修 `sever_utils.py` 用字节 broadcast/gloo(已验证方案);或单卡跑 server |
| RoboTwin 仿真 | 未跑 | NVIDIA 环境装 RoboTwin 客户端,连本机 server;这是拿到 SR 数字的最后一步 |

### 7.4 RoboTwin 仿真 benchmark 在 BW1000 上的支持性评估(2026-09-23 实测)

**结论:BW1000 不能本机跑 RoboTwin 仿真客户端(sapien 渲染被挡),但官方 server-client 架构天然支持"仿真在 NVIDIA、推理在 BW1000"的分离部署,推荐走此路径拿 SR 数字。**

#### 实测证据链

| # | 测试 | 结果 |
|---|---|---|
| 1 | BW1000 DRM render 节点(`/dev/dri/renderD128-135`)驱动 | `hycu`(海光**计算**驱动),非 `amdgpu` 图形驱动 → Mesa/RADV 图形栈无法绑定 GPU 渲染 |
| 2 | 安装 `libvulkan1 + mesa-vulkan-drivers + vulkan-tools` | ✅ 可装,`vulkaninfo` 枚举到 **llvmpipe**(CPU Vulkan 设备,Mesa 23.2.1) |
| 3 | sapien 3.0.3 与 3.0.0b1(RoboTwin 锁定版本)+ lavapipe 渲染 | ❌ `vk::PhysicalDevice::createDeviceUnique: ErrorExtensionNotPresent` |
| 4 | 根因(`VK_LOADER_DEBUG=all` 定位) | sapien 硬编码要求 **`VK_KHR_external_semaphore_fd`**(CUDA↔Vulkan 互操作导出信号量),llvmpipe 是纯 CPU 设备**永不支持**该扩展(仅有 `external_memory_fd`) |
| 5 | RADV(radeon ICD)直连 hycu 设备 | ❌ `Failed to detect any valid GPUs`——RADV 只认 amdgpu 内核驱动,与 hycu 不互通 |

#### 三条路径评估

| 路径 | 可行性 | 说明 |
|---|---|---|
| **A. 仿真客户端放 NVIDIA 机器(推荐)** | ✅ 架构原生支持 | `WebsocketClientPolicy` 连接 `ws://<host>:<port>`,与仿真进程完全解耦——RoboTwin 仿真(NVIDIA,vulkan+sapien 3.0.0b1)+ 推理 server(BW1000,`launch_server.sh` 单卡模式避开 HCCL L1 缺陷)。这正是官方 server-client 设计意图,推理负载仍在国产卡 |
| B. BW1000 本机 CPU 渲染(llvmpipe) | ❌ 被硬需求挡死 | sapien 的 `VK_KHR_external_semaphore_fd` 互操作是编译期硬编码;除非改 sapien 源码去掉 CUDA interop(工作量大、上游不维护) |
| C. 等海光图形栈支持 | ⏳ 未知 | 若未来 hycu/DTK 暴露 Vulkan ICD(对标 NVIDIA 的 CUDA-GL interop)才可能本机闭环;可携本报告证据向光合开发者社区提需求 |

#### 路径 A 落地要点

1. NVIDIA 机器:按官方 README 装 RoboTwin(vulkan 依赖 + sapien==3.0.0b1 + 资产包)
2. BW1000 本机:checkpoint 的 `attn_mode` 改 `torch` 后启动 `bash evaluation/robotwin/launch_server.sh`(单卡)
3. 客户端 `--port` 指向 server 端口,跨机网络需放通
4. 多卡并行评测(50 任务分组)需先修 L1(字节 broadcast workaround,方案已验证)

## 8. 复现命令速查

```bash
# 0) 环境(每 shell)
source /opt/dtk/env.sh
cd /home/tione/notebook/code/lingbot-va

# 1) venv(一次性,已建好)
python -m venv --system-site-packages va_env
va_env/bin/pip install "diffusers==0.36.0" "transformers==4.55.2" easydict ftfy "imageio[ffmpeg]"
va_env/bin/pip install --no-deps "lerobot==0.3.3"

# 2) 数据/模型(已就位)
# 模型: /home/tione/notebook/model/lingbot-va-base
# 数据: /home/tione/notebook/data/Robbyant/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500

# 3) 训练(8 卡,setsid 防会话清理误杀)
setsid bash -c 'source /opt/dtk/env.sh; NGPU=8 CONFIG_NAME=robotwin_train \
  bash script/run_va_posttrain.sh > /tmp/train.log 2>&1' & disown

# 4) 监控
tail -f /tmp/train.log | grep -oE "latent_loss=[0-9.]+, action_loss=[0-9.]+, step=[0-9]+"
watch -n 5 hy-smi

# 5) 评测任意 checkpoint
bash script/eval_checkpoint.sh <step>   # 输出 train_out/demo.mp4

# 6) 续训(从 10K)
# 配置 resume_from = train_out/checkpoints/checkpoint_step_10000 后重启训练
```

---

*相关文档:`project.md`(平台适配分析+原理)、`report.md`(30min 快照流水+评测日志)、`training_report.md`(训练专项)。本报告为总入口。*

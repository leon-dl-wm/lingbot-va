# LingBot-VA 项目全解析(初学者友好版)

> 目标:读完本文,你能 **看懂原理 → 看懂代码结构 → 快速跑起训练**。
> 阅读顺序建议:先通读第 1~2 章建立直觉,再按需查阅第 4 章模块细节,最后照着第 6 章动手训练。

---

## 目录

1. [一句话理解本项目](#1-一句话理解本项目)
2. [核心原理图解(零基础可读)](#2-核心原理图解零基础可读)
3. [项目目录结构总览](#3-项目目录结构总览)
4. [各模块功能详解](#4-各模块功能详解)
5. [两条核心数据流(训练 / 推理)](#5-两条核心数据流训练--推理)
6. [快速上手:五步跑通训练](#6-快速上手五步跑通训练)
7. [关键配置参数速查表](#7-关键配置参数速查表)
8. [常见坑与 FAQ](#8-常见坑与-faq)
9. [论文与代码一致性分析](#9-论文与代码一致性分析)

---

## 1. 一句话理解本项目

**LingBot-VA = 一个会"想象未来"的机器人大脑。**

给它一张当前场景图片 + 一句任务指令(如 "把绿色方块放进蓝盒子"),
它能在**同一个模型**里同时输出两样东西:

- **视频(video)**:预测未来画面会怎么变化 → 这叫 **世界模型**(World Model)
- **动作(action)**:预测机器人该怎么动 → 这叫 **动作策略**(Action Policy)

两者在一条**交错(interleaved)序列**里自回归地生成:先想一段画面,再想一段动作,再想下一段画面……就像人做事一样:"预判 → 行动 → 观察结果 → 再预判"。

---

## 2. 核心原理图解(零基础可读)

### 2.1 整体生成过程(自回归 + 扩散)

模型不是一次生成整段视频,而是**一块一块(chunk)地生成**:

```
时间轴 ──────────────────────────────────────────────►

  [历史观测]   [chunk 0]      [chunk 1]      [chunk 2]
  (真实数据)   视频₀ 动作₀    视频₁ 动作₁    视频₂ 动作₂
              └──┬──┘         └──┬──┘
                 │ 只依赖历史      │ 依赖 历史+chunk0
                 ▼                ▼
            因果注意力       因果注意力(带 KV Cache)
```

- 每个 chunk 包含 `frame_chunk_size=2` 个 latent 帧(对应约 9 个原始视频帧)和对应的动作序列
- **因果性**:生成第 N 块时只能"看到"前面的块 → 这就是 **自回归(AR)**
- **KV Cache**:历史块的 Key/Value 缓存下来,不用每次重算 → 推理快

### 2.2 每一块内部怎么生成?(Flow Matching 扩散)

视频和动作都不是直接"画"出来的,而是**从噪声逐步去噪**出来的(Flow Matching / 整流流):

```
纯噪声 z_T ──去噪step 1──► ──去噪 step 2──► ... ──► 干净的 视频/动作 z_0
              ▲                              ▲
              └── Transformer 预测"速度场" v ──┘
```

- **视频流**:25 步去噪(`num_inference_steps=25`)
- **动作流**:50 步去噪(`action_num_inference_steps=50`)
- 两条流用**不同的噪声调度**:`snr_shift=5.0`(视频)vs `action_snr_shift=1.0`(动作)
- 训练目标非常简单:**给干净数据加噪,让网络学会预测"噪声 - 数据"(速度)**,用 MSE 损失

### 2.3 双流 MoT(混合 Transformer)架构

视频 token 和动作 token **共享同一个 Transformer 主干**,但各有专属的"入口"和"出口":

```
                    ┌─────────────────────────────┐
  视频latent ──► patch_embedding_mlp ──┐          │
                                      ├► 30层共享 │ ─► proj_out ──► 视频去噪结果
  动作     ──► action_embedder ───────┤  Transformer│
                                      │  (自注意力+ │
  文本指令 ──► text_embedder ─────────┤  交叉注意力)│ ─► action_proj_out ──► 动作去噪结果
                    └─────────────────────────────┘
  时间步t   ──► condition_embedder / condition_embedder_action (两套时间编码)
```

好处:视觉动态知识和动作控制知识**互相促进**(共享参数 = 共享"常识"),又**各司其职**(独立输入/输出投影)。

### 2.4 训练时的注意力掩码(本项目的精髓)

训练时把 4 段序列拼在一起一次前向:

```
序列布局: [ 噪声视频 | 干净视频(条件) | 噪声动作 | 干净动作(条件) ]
noise_ids:    0           1               0           1
```

注意力规则(`FlexAttnFunc._get_mask_mod`,model.py L154):

| 谁看谁 | 允许? | 原因 |
|---|---|---|
| 干净 → 干净(历史块) | ✅ 因果 | 历史信息是条件 |
| 噪声 → 干净(更早的块) | ✅ 严格因果 | 预测要依赖历史 |
| 噪声 → 噪声 | ✅ 仅同块内 | 同一块内互相协同去噪,**不看**其他噪声块 |
| 超出窗口(window_size) | ❌ | 限制注意力范围,省显存 + 聚焦局部 |

训练时 `chunk_size` 和 `window_size` 都是**随机采样**的(train.py `_prepare_input_dict`),
这让模型对各种推理配置都鲁棒 —— 一个很实用的技巧。

另外 `noisy_cond_prob=0.5`:条件(干净段)有一半概率也被加噪,
模拟推理时"条件本身也是模型生成的、不完美"的情况 → 提升长时程鲁棒性。

### 2.5 与常见方案的对比

| 方案 | 视频预测 | 动作生成 | 关系 |
|---|---|---|---|
| VLA(如 openpi) | ❌ | ✅ | 纯策略,无"想象" |
| 世界模型(如 Genie) | ✅ | ❌ | 只想象,不控制 |
| **LingBot-VA** | ✅ | ✅ | **交错联合生成,互相约束** |

---

## 3. 项目目录结构总览

```
lingbot-va/
├── wan_va/                    # ★ 核心包:模型 + 训练 + 推理服务
│   ├── modules/
│   │   ├── model.py           # ★★ Transformer 模型定义(903行,最核心)
│   │   └── utils.py           # 加载 VAE/文本编码器/Transformer,流式VAE包装
│   ├── configs/               # 全部配置(EasyDict,按 环境×模式 组织)
│   │   ├── shared_config.py   # 公共配置(端口/dtype/patch_size)
│   │   ├── va_robotwin_cfg.py # RoboTwin 推理配置(含动作归一化统计)
│   │   ├── va_*_train_cfg.py  # 各环境训练配置
│   │   └── va_*_i2va.py       # 图生视频-动作 demo 配置
│   ├── dataset/
│   │   └── lerobot_latent_dataset.py  # LeRobot+预提取latent 数据集
│   ├── distributed/
│   │   ├── fsdp.py            # FSDP 切分 + 激活检查点
│   │   └── util.py            # 分布式初始化/规约
│   ├── utils/
│   │   ├── scheduler.py       # ★ FlowMatchScheduler(加噪/去噪/训练权重)
│   │   ├── utils.py           # mesh_id、patch转换、时间步采样
│   │   ├── sever_utils.py     # websocket 服务 + 多卡广播推理
│   │   └── Simple_Remote_Infer/  # 独立部署工具(websocket server/client)
│   ├── train.py               # ★★ 训练主循环(Trainer 类)
│   └── wan_va_server.py       # ★★ 推理服务器(VA_Server 类)
├── evaluation/                # 仿真评测客户端
│   ├── robotwin/              # RoboTwin 2.0 评测(50个双臂任务)
│   └── libero/                # LIBERO 评测
├── script/
│   ├── run_va_posttrain.sh    # 训练启动脚本(torchrun)
│   └── run_launch_va_server_sync.sh  # 推理服务启动脚本
├── example/                   # i2va demo 用的示例图片(demo/franka/libero/robotwin)
├── README.md / INSTALL.md     # 官方文档
└── pyproject.toml / requirements.txt
```

(★ 越多 = 越重要,初学者优先读这些)

---

## 4. 各模块功能详解

### 4.1 `wan_va/modules/model.py` — 模型心脏 ⭐⭐

| 类/函数 | 行号 | 作用 |
|---|---|---|
| `WanTransformer3DModel` | 569 | 主模型:30层 Transformer,双流输入/输出 |
| `WanTransformerBlock` | 468 | 单层块:自注意力 + 交叉注意力(文本) + FFN,adaLN 时间调制 |
| `WanAttention` | 289 | 注意力实现 + **KV Cache 管理**(init/allocate/update/restore) |
| `FlexAttnFunc` | 42 | flex_attention 封装 + **训练用块因果掩码** `init_mask` |
| `WanRotaryPosEmbed` | 243 | RoPE 位置编码(按 grid_id) |
| `WanTimeTextImageEmbedding` | 203 | 时间步 + 文本嵌入 |

**必懂的三个入口:**

- `forward(input_dict, train_mode=True)` → `forward_train()`:训练用。
  把 `[噪声视频, 干净视频, 噪声动作, 干净动作]` 拼成一条序列,用 flex_attention 的块因果掩码一次前向,返回 `(latent_pred, action_pred)` 两个速度预测。
- `forward(input_dict, action_mode=False/True)`:推理用,单流(视频或动作)去噪一步。
- `create_empty_cache()`:推理前按 `attn_window` 预分配 KV Cache 槽位。

**KV Cache 三种模式**(`update_cache` 参数,理解它就理解了推理):

| 值 | 含义 | 何时用 |
|---|---|---|
| `0` | 临时写入,用完即恢复(不污染缓存) | 去噪中间步 |
| `1` | 写入并**保留**(标记 is_pred) | 去噪最后一步,把生成结果存入历史 |
| `2` | 写入并保留(真实观测) | 编码真实观测(`_compute_kv_cache`) |

`clear_pred_cache()` 会把 is_pred 的槽位清掉但保留真实观测的 ——
这样机器人执行动作后拿到**真实**新观测时,可以"撤销"上一步的预测缓存,重新写入真实值。这是保持长时程不漂移的关键设计。

### 4.2 `wan_va/train.py` — 训练主循环 ⭐⭐

`Trainer` 类,按 step(而非 epoch)训练:

```
__init__:  加载Transformer(fp32) → 激活检查点 → FSDP切分 → AdamW → 数据集
_train_step:  加噪 → 随机chunk/window → forward_train → 双流MSE → 反传
train():  主循环 + 梯度累积 + 梯度裁剪(2.0) + wandb日志 + 定期存checkpoint
```

关键函数:

- `_add_noise()`:flow-matching 加噪。**每个 latent 帧独立采样时间步**(不是整段同一个 t!),并按 `noisy_cond_prob` 给条件段加噪
- `_prepare_input_dict()`:随机采样 `chunk_size∈[1,5)`、`window_size∈[4,65)` → 训练/推理配置对齐
- `compute_loss()`:**逐帧(frame-wise)归一化**的 MSE,各帧按 `training_weight` 加权(中间时间步权重高),动作流额外乘 mask(未用维度不算损失)
- `save_checkpoint()`:rank0 聚合全量 state_dict → 转 bf16 → 存成 diffusers 格式(`diffusion_pytorch_model.safetensors` + `config.json`),**与预训练格式完全兼容**,可直接拿来推理

### 4.3 `wan_va/wan_va_server.py` — 推理服务器 ⭐⭐

`VA_Server` 类,两种模式(`infer_mode` 配置):

1. **`server` 模式**:起 websocket 服务,等仿真/真机客户端来查询(评测用)
2. **`i2va` 模式**:读一张图 + prompt,自回归生成 N 块,解码出 demo.mp4(快速体验用)

推理一个 chunk 的流程(`_infer`,L443):

```
1. 随机初始化 latents(48,C) 和 actions(30,C)
2. 视频去噪循环(25步):每步 forward(action_mode=False)
   └─ 最后一步 update_cache=1,把结果永久写入 KV Cache
   └─ CFG: guidance_scale=5,正/负prompt双batch
3. 动作去噪循环(50步):每步 forward(action_mode=True)
   └─ action_guidance_scale=1(动作不用CFG)
4. postprocess_action:反归一化 → 取出 used_action_channel_ids
5. 返回动作给客户端执行
```

其他要点:

- `_encode_obs()`:多相机图像 → resize → 流式 VAE 编码 → latent 拼接(RoboTwin 是 T 型拼接:腕部相机左右拼 + 高处相机)
- `preprocess/postprocess_action()`:动作按分位数(q01/q99)归一化到 [-1,1],只保留 `used_action_channel_ids` 指定的维度
- `_compute_kv_cache()`:客户端执行完动作发来**真实**观测 → `clear_pred_cache()` 撤销预测 → 编码真实观测写入缓存 → `frame_st_id` 前进

### 4.4 `wan_va/configs/` — 配置中心 ⭐

命名规律:`va_{环境}_{模式}_cfg.py`,环境有 `robotwin / libero / franka / demo`,模式有:

| 后缀 | 用途 | 关键差异 |
|---|---|---|
| (无) | 评测服务器 | `infer_mode='server'` |
| `_i2va` | 图生视频体验 | `infer_mode='i2va'`,指定 `input_img_path`/`prompt`/`num_chunks_to_infer` |
| `_train` | 训练 | 增加 lr/batch/steps/dataset_path 等 |

通过 `VA_CONFIGS` 字典注册,启动时 `--config-name robotwin_train` 选择。
**改行为优先改配置,而不是改代码。**

### 4.5 `wan_va/dataset/lerobot_latent_dataset.py` — 数据集 ⭐

两级结构:

- `MultiLatentLeRobotDataset`:把多个数据集串成一个大数据集(多进程初始化)
- `LatentLeRobotDataset`(继承 LeRobotDataset):单个数据集

`__getitem__` 做的事:

```
读 episodes.jsonl 的 action_config → 定位 (start_frame, end_frame)
→ 读预提取的 VAE latent (.pth) + 文本embedding
→ 读对齐的 action 序列(HuggingFace datasets)
→ _action_post_process:
   相对位姿转换(仅robotwin) → 对齐到latent帧(每latent帧×4步)
   → pad到30维 → 分位数归一化 → clip到[-1.5,1.5] → reshape (C,F,N,1)
→ cfg_prob=0.1 概率把 text_emb 换成空embedding(CFG训练)
```

### 4.6 `wan_va/utils/scheduler.py` — FlowMatchScheduler ⭐

一个类同时服务训练和推理:

| 方法 | 用途 |
|---|---|
| `add_noise()` | 训练:`x_t = (1-σ)x_0 + σ·noise` |
| `training_target()` | 训练目标:`noise - x_0`(速度场) |
| `training_weight()` | 训练损失权重(时间步中间高、两端低) |
| `step()` | 推理:`x_{t+1} = x_t + v·(σ' - σ)`(欧拉步) |
| `set_timesteps(training=True)` | 生成 shift 后的 σ 序列 |

`shift` 参数控制噪声调度偏向:视频用 5.0(更关注高噪声阶段),动作用 1.0。

### 4.7 `wan_va/distributed/` — 分布式训练

- `fsdp.py`:`shard_model()` 用 FSDP2 风格切分参数;`apply_ac()` 给每个 block 开激活检查点(省显存,代价是重算)
- `util.py`:`init_distributed()`(NCCL)、`dist_mean/dist_max`(跨卡统计损失)

### 4.8 `wan_va/utils/sever_utils.py` — 多卡推理协调

rank 0 跑 websocket 服务收请求,通过 `dist.broadcast_object_list` 把 obs 广播给所有卡,
各卡都跑 `model.infer()`(数据并行,不同仿真环境并行评测)。

### 4.9 `evaluation/` — 仿真评测客户端

| 目录 | 内容 |
|---|---|
| `robotwin/` | RoboTwin 2.0 双臂评测:客户端跑仿真,通过 websocket 问服务器要动作;`calc_stat.py` 计算动作归一化统计;支持单卡/8卡分组并行 |
| `libero/` | LIBERO 评测:同样的 server-client 模式 |

### 4.10 `script/` — 启动脚本

都是 `torchrun` 包装,通过环境变量传参:

```bash
NGPU=8 CONFIG_NAME='robotwin_train' bash script/run_va_posttrain.sh
```

---

## 5. 两条核心数据流(训练 / 推理)

### 5.1 训练数据流

```
LeRobot数据集(videos/ + meta/episodes.jsonl + actions)
        │ 离线预处理(一次性)
        ▼
latents/*.pth(VAE编码) ──► LatentLeRobotDataset.__getitem__
        │                        │ 动作对齐+归一化, latent拼接
        ▼                        ▼
   Trainer._get_next_batch ◄─────┘
        │
        ▼
   _add_noise: 每帧独立采样t, 加噪, 生成grid_id
        │
        ▼
   forward_train: [噪声视频|干净视频|噪声动作|干净动作] 拼序列
        │          flex_attention 块因果掩码(随机window)
        ▼
   compute_loss: 逐帧加权MSE(视频) + 逐帧加权MSE×mask(动作)
        │
        ▼
   loss.backward → 梯度累积 → clip(2.0) → AdamW.step
        │ 每 save_interval 步
        ▼
   checkpoint(diffusers格式, bf16)
```

### 5.2 推理数据流(评测闭环)

```
┌─────────┐  obs(图像+状态)   ┌──────────────┐  action(30维)
│ 仿真环境 │ ───────────────► │  VA_Server   │ ──────────────► 执行
│(客户端) │ ◄─────────────── │  (websocket) │
└─────────┘    返回动作        └──────────────┘
     ▲                              │
     │ 新观测                        │ 内部:
     └──────────────────────────────┘ 1. reset: 编码prompt, 建KV Cache
                                       2. compute_kv_cache: 真实观测→latent→缓存
                                       3. infer: 视频25步去噪 → 动作50步去噪
                                          (最后一步写入KV Cache)
                                       4. 下次真实观测到来时 clear_pred_cache
                                          撤销预测、写入真实值 → 防漂移
```

---

## 6. 快速上手:五步跑通训练

### 第 0 步:环境安装

```bash
# Python 3.10 / PyTorch 2.9 / CUDA 12.6
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict
pip install flash-attn --no-build-isolation
# 训练额外需要
pip install lerobot==0.3.3 scipy wandb --no-deps
```

### 第 1 步:下载底座模型和数据

```bash
# 底座模型(训练起点)
huggingface-cli download robbyant/lingbot-va-base --local-dir /path/to/model

# 官方 RoboTwin 后训练数据(已含 latents,可直接用)
huggingface-cli download --repo-type dataset robbyant/robotwin-clean-and-aug-lerobot --local-dir /path/to/dataset
```

> 想先零成本体验?跳到第 6.5 节的 i2va demo。

### 第 2 步:改配置(只改两处路径)

编辑 `wan_va/configs/va_robotwin_train_cfg.py`:

```python
va_robotwin_train_cfg.dataset_path = '/path/to/dataset'   # ← 改成你的数据路径
```

以及它继承的 `wan_va/configs/va_robotwin_cfg.py`:

```python
va_robotwin_cfg.wan22_pretrained_model_name_or_path = "/path/to/model"  # ← 底座模型路径
```

**⚠️ 关键一步:把模型目录下的 `transformer/config.json` 中 `attn_mode` 改为 `"flex"`**
(训练必须用 flex,推理必须用 `torch` 或 `flashattn`,详见第 8 章)

### 第 3 步:(可选)配置 wandb

编辑 `script/run_va_posttrain.sh`,填入你的 key;或直接在配置里 `enable_wandb = False` 跳过。

### 第 4 步:启动训练

```bash
# 8卡
NGPU=8 CONFIG_NAME='robotwin_train' bash script/run_va_posttrain.sh

# 单卡/少卡也能跑,用梯度累积补有效batch
NGPU=1 CONFIG_NAME='robotwin_train' bash script/run_va_posttrain.sh \
    batch_size=1 gradient_accumulation_steps=8
```

命令行末尾的 `key=value` 会覆盖配置项(无需改文件)。

**看什么指标判断训练是否正常:**

| 指标 | 正常表现 |
|---|---|
| `latent_loss` / `action_loss` | 前几百步明显下降,之后缓慢下降 |
| `grad_norm` | 稳定在个位数,频繁爆炸→降 lr |
| 显存 | OOM → 减 batch_size / 加 gradient_accumulation_steps |

checkpoint 存在 `save_root/checkpoints/checkpoint_step_N/transformer/`。

### 第 5 步:用训好的模型推理

1. 把 checkpoint 里 `transformer/config.json` 的 `attn_mode` 改回 `"torch"`
2. 配置里 `wan22_pretrained_model_name_or_path` 指向 checkpoint 目录
3. 起服务:`NGPU=1 CONFIG_NAME='robotwin_i2av' bash script/run_launch_va_server_sync.sh`
   → 生成 `demo.mp4`,直观看到模型"想象"的未来和动作

### 6.5 零成本体验(不训练)

```bash
# 直接用官方 posttrain 权重跑图生视频-动作
NGPU=1 CONFIG_NAME='robotwin_i2av' bash script/run_launch_va_server_sync.sh
# 约 18GB 显存(offload 模式)
```

### 6.6 用自己的数据训练(自定义数据集)

数据准备三步(详见 README "Custom Dataset Preparation"):

1. **转 LeRobot 格式**(参考 LeRobot 官方文档),动作对齐到 30 维标准格式:
   `左臂EEF(7) + 右臂EEF(7) + 左臂关节(7) + 右臂关节(7) + 左夹爪(1) + 右夹爪(1)`,缺的维度补 0
2. **加动作分段标注**:在 `meta/episodes.jsonl` 每行加 `action_config` 字段:
   ```json
   {"episode_index": 0, "tasks": ["..."], "length": 450,
    "action_config": [{"start_frame": 0, "end_frame": 450, "action_text": "抓起方块..."}]}
   ```
3. **提取 VAE latent**:视频 resize 到 ~256×256、降采样到 5-15fps,用 Wan2.2 VAE 编码,
   存为 `latents/chunk-000/<cam>/episode_{idx}_{start}_{end}.pth`(字段见 README 表格)

然后复制一份 `va_demo_train_cfg.py` 改路径即可(官方提供了 [示例数据集](https://drive.google.com/file/d/1D52nK4ZOJmWBXKv1nWrLb9YBwq8nKa_b/view) 可参考格式)。

---

## 7. 关键配置参数速查表

### 模型/序列结构

| 参数 | 默认值 | 含义 |
|---|---|---|
| `patch_size` | (1,2,2) | latent 的时空 patch 化(1帧×2×2 空间) |
| `frame_chunk_size` | 2 | 每个自回归块的 latent 帧数 |
| `action_dim` | 30 | 动作总维度(标准格式) |
| `action_per_frame` | 16 | 每个 latent 帧对应的动作步数 |
| `attn_window` | 72 | KV Cache 窗口(块数),超出淘汰最旧 |
| `used_action_channel_ids` | 见配置 | 实际使用的动作维度(RoboTwin 用 16/30 维) |

### 推理

| 参数 | 默认值 | 含义 |
|---|---|---|
| `num_inference_steps` | 25 | 视频去噪步数 |
| `action_num_inference_steps` | 50 | 动作去噪步数 |
| `video_exec_step` | -1 | 视频提前退出步(-1=跑满;异步执行用) |
| `guidance_scale` | 5 | 视频 CFG 强度 |
| `action_guidance_scale` | 1 | 动作 CFG(1=关闭) |
| `enable_offload` | False | VAE/文本编码器卸载到 CPU 省显存 |

### 训练

| 参数 | 默认值 | 含义 |
|---|---|---|
| `learning_rate` | 1e-5 | 学习率(自定义数据可试 1e-4) |
| `batch_size` × `gradient_accumulation_steps` × NGPU | — | **有效 batch**(官方建议 ≥32) |
| `num_steps` | 50000 | 总训练步数 |
| `save_interval` | 1000 | 存 checkpoint 间隔 |
| `cfg_prob` | 0.1 | 训练时换空文本概率(CFG 配套) |
| `snr_shift` / `action_snr_shift` | 5.0 / 1.0 | 视频/动作噪声调度偏移 |
| `warmup_steps` | 10 | lr 预热步数 |

---

## 8. 常见坑与 FAQ

### Q1:训练/推理报注意力相关错误?
**`attn_mode` 必须切换!** 它存在模型目录 `transformer/config.json` 里:

| 场景 | attn_mode |
|---|---|
| 训练 | `"flex"`(块因果掩码必需) |
| 推理/评测 | `"torch"` 或 `"flashattn"` |

### Q2:显存不够?
- 推理:开 `enable_offload=True`(VAE/文本编码器进 CPU,单卡 i2va 约 18GB)
- 训练:减小 `batch_size`,增大 `gradient_accumulation_steps` 保持有效 batch

### Q3:动作维度对不上?
动作统一 pad 到 30 维,通过 `used_action_channel_ids` 选择实际生效的维度,
`inverse_used_action_channel_ids` 是它的逆映射(用于把你的维度插到标准位置)。
**改了它必须同步改 `norm_stat`(q01/q99 分位数统计)**,可用 `evaluation/robotwin/calc_stat.py` 重新计算。

### Q4:损失不降?
- 确认数据 latent 与 `action_config` 帧号对齐(文件名 `episode_{idx}_{start}_{end}.pth`)
- 确认动作归一化统计与数据匹配
- lr 从 1e-5 起步,自定义小数据集可到 1e-4

### Q5:checkpoint 怎么复用?
训练存出的就是 diffusers 标准格式,直接把路径填到
`wan22_pretrained_model_name_or_path` 即可推理(记得改回 attn_mode)。

### Q6:训练时 chunk_size / window_size 为什么随机?
模拟推理时的各种配置,让模型见多识广 —— 这是 train.py `_prepare_input_dict` 里
`torch.randint` 的用意,不是 bug。

---

## 9. 论文与代码一致性分析

> 本章对比 `LingBot_VA_paper.pdf`(31 页,arXiv 2601.21998)与本仓库代码
> (模型 `model.py`、训练 `train.py`、推理 `wan_va_server.py`、评测客户端、全部配置)。
> **结论先行:核心算法框架高度一致;但论文描述的模型架构与发布代码不是同一个变体,
> 且论文两大效率类贡献(异步推理、部分去噪)未在发布代码中启用。**

### 9.1 总体结论

| 维度 | 一致性 |
|---|---|
| 核心算法(AR 扩散、flow matching、因果掩码、teacher forcing) | ✅ 高度一致 |
| 模型架构(双流 MoT vs 共享主干) | ❌ 论文 ≠ 发布代码 |
| 异步推理(Algorithm 2 / FDM) | ❌ 未实现(代码为同步) |
| 部分去噪(s=0.5/0.6) | ❌ 机制存在但未启用 |
| KV Cache 持久记忆 | ⚠️ 有界滑动窗口,非全轨迹 |
| 训练/推理超参 | ⚠️ 大体一致,细节有出入 |

README News 已承认发布的是 **shared backbone** 版本
("Weights and code for shared backbone released! Please stay tuned for our separated version"),
但论文正文通篇描述的是**双流 MoT(separated)版本**。

### 9.2 高度一致的部分 ✅

| 论文内容 | 代码实现 | 位置 |
|---|---|---|
| AR 扩散 video-action 世界模型,交错序列 chunk 式自回归 | `forward_train` 四段拼接 [噪声视频\|干净视频\|噪声动作\|干净动作] | `model.py` |
| Flow matching(速度预测 + Euler solver) | `training_target = noise - sample`;`step()` 为 Euler 积分 | `scheduler.py` |
| Wan2.2-5B 主干:30 层、d=3072、RoPE、48ch、patchify 2× | `num_layers=30, 24×128=3072, patch_size=(1,2,2)` | `model.py` |
| Wan2.2 因果 VAE(4×16×16)+ 流式编码 | `WanVAEStreamingWrapper`(WanCausalConv3d 缓存) | `modules/utils.py` |
| 冻结 T5(UMT5)cross-attention 注入;训练用预计算 embedding | ✓(训练只加载 transformer) | `train.py` |
| Teacher forcing 因果掩码(论文图3):视频块只看 a<t(式8),动作块看预测视频块(式9,逆动力学);块内双向注意力 | `_get_mask_mod` 的 `noise2clean/noise2noise/block_causal` 精确实现 | `model.py` |
| Noisy History Augmentation:p=0.5,s∈[0.5,1](式10) | `noisy_cond_prob=0.5, min/max_timestep_bd=0.5/1.0` | `train.py` |
| 变长 chunk 训练 K∈[1,4] | `torch.randint(1, 5)` | `train.py` |
| L = Ldyn + λLinv(λ=1);CFG 文本 dropout 0.1;grad clip 2.0;bf16 | 全部一致 | `train.py` |
| 视频 CFG=5.0 / 动作 CFG=1.0 | `guidance_scale=5, action_guidance_scale=1` | 各 config |
| 30 维双臂动作 + 分位数归一化(q01/q99) | `action_dim=30, action_norm_method='quantiles'` | config + dataset |
| Algorithm 1 同步 KV-cache 推理:生成视频块→动作块→执行→用真实观测替换预测(`clear_pred_cache`) | `_infer` + `_compute_kv_cache` | `wan_va_server.py` |
| RoboTwin:50Hz→12.5Hz 降采样、50Hz 动作、50K steps、lr 1e-5 | `action_per_frame=16`(=4帧×4动作)、`num_steps=50000` | config |

### 9.3 关键不一致(按严重程度)❌

**① 模型架构:论文 ≠ 发布代码(最大差距)**

- 论文:双流 MoT,视频流 d=3072 + 动作流 d=768(独立 QKV/transformer 参数),
  额外 ~350M 参数,总计 5.3B
- 代码:单一共享主干(~5.0B),动作 token 经 `action_embedder: Linear(30→3072)` 投影后
  与视频 token 走**同一组** 30 层 blocks,动作专属参数仅 ~45-90M
- 论文的动作流初始化(视频权重插值 + α=√(dv/da) 缩放,论文图7消融)和
  "动作编解码器为隐层 256 的单层 MLP"在代码中均不存在
- README News 承诺的 "separated version"(双流 MoT)尚未发布

**② 异步推理(Algorithm 2 / FDM-grounded Async)未实现**

- 论文三大贡献之一(执行与预测并行、2× 加速、消融表 90.4 vs 92.9)
- 代码:LIBERO/RoboTwin 两个 client 均为**同步**循环(预测→执行→缓存反馈),
  无线程/asyncio;`imagine=False` 参数是死代码;式13 的 FDM 损失也不在训练代码中
  (只有 latent_loss + action_loss)
- 注:论文主表数字(92.9)对应消融表 Baseline 行,即同步模式,与发布代码一致;
  未发布的是提速变体

**③ 部分去噪(s=0.5/0.6)未启用**

- 论文 §3.3/§4.2/Algorithm 1:"视频积分到 s=0.5"、"3 步视频(到 s=0.6)+ 10 步动作"
- 代码:`video_exec_step` 机制存在但所有配置均为 -1(视频**完全去噪**);
  步数也不符——demo 5/10,LIBERO 20/50,RoboTwin 25/50

**④ KV-cache 是有界滑动窗口而非"全轨迹持久记忆"**

- 论文:"persistent memory across the entire trajectory"、"complete observation history"
- 代码:`attn_window=30/72`(仅 15/36 个 chunk,满则淘汰最旧);训练注意力窗口随机 [4,64]

### 9.4 次要差异 ⚠️

| 项目 | 论文 | 代码 |
|---|---|---|
| 部署 chunk K | 统一 K=4 | LIBERO/demo=4 ✓,RoboTwin=2 |
| τ(每视频帧动作数) | τ=4 | 仅 RoboTwin 匹配;LIBERO τ=1,demo τ=2,franka τ=5 |
| N=192 tokens/帧 | pretraining(3×256²) | 评测配置不同:RoboTwin 120、LIBERO 32、demo 128 |
| LR scheduler | cosine 退火+线性 warmup(pretrain) | warmup+恒定(post-train);weight decay 0.1 vs 0.01 |
| Episode packing 至 10K tokens | ✓(pretrain) | 无(单 segment 样本) |
| LIBERO 步数 | 4K | 配置 5000 |
| 未披露的训练细节 | — | bell 形时间步损失加权、frame-wise 损失归一化、SNR shift(视频5.0/动作0.05~1.0)、RoboTwin 相对位姿变换与 T 形相机布局、首 chunk 首帧 dummy 动作跳过 |
| 未发布部分 | 1.4T token 预训练、6 数据集、真实机器人部署(500 steps/lr 1e-4/seq 150K) | 仅 post-training + 仿真评测 |

### 9.5 对使用者的建议

1. **复现论文主表结果**:发布代码(同步模式 + shared backbone 权重)大体对应论文主表数字,
   但需注意推理步数配置与论文文字不符
2. **若目标是完全对齐论文**:需等待 "separated version"(双流 MoT)发布,
   并自行实现 Algorithm 2 异步管线与 `video_exec_step` 部分去噪
3. **论文本身的小问题**:Algorithm 2 下标混用两种约定(a_t→z_t vs a_t→z_{t+1}),
   与其式8的条件模式存在张力;代码的同步实现反而与式8/9严格一致

---

## 附:进一步阅读

- 论文:`LingBot_VA_paper.pdf` / `LingBot_VA2_paper.pdf`(项目根目录)
- 官方 README.md:安装、评测、数据准备的权威说明
- 代码阅读推荐顺序:
  1. `wan_va/utils/scheduler.py`(最小,理解 flow matching)
  2. `wan_va/modules/model.py` 的 `forward_train` + `FlexAttnFunc.init_mask`(核心机制)
  3. `wan_va/train.py` 的 `_add_noise` → `compute_loss`(训练闭环)
  4. `wan_va/wan_va_server.py` 的 `_infer` → `_compute_kv_cache`(推理闭环)

# LingBot-VA 代码阅读指南与实现细节深度解析

> 基于仓库 `leon-dl-wm/lingbot-va` 源码精读整理(2026-09)。
> 核心代码约 5k 行,集中在 `wan_va/`(算法+训练+推理服务)与 `evaluation/`(仿真评估)。

---

# Part 1:推荐阅读顺序

## 第 0 步:全局配置(30 分钟)

| 文件 | 看什么 |
|---|---|
| `README.md` | Data Pipeline、attn_mode(训练用 `flex`,推理用 `torch/flashattn`) |
| `wan_va/configs/shared_config.py` + `va_robotwin_cfg.py` | 关键超参:`frame_chunk_size=2`(AR 每次生成 2 个 latent 帧)、`attn_window=72`、`action_dim=30`、`action_per_frame=16`、`snr_shift`、`norm_stat`(q01/q99 归一化) |

## 第 1 步:算法核心 ⭐ `wan_va/modules/model.py`(914 行,重点精读)

按此顺序读:

1. **`FlexAttnFunc._get_mask_mod` (L160–208)** — 全文最核心。论文的"因果 AR + Diffusion Forcing"就在这 40 行:clean→clean 块因果掩码、noisy→clean 严格因果(排除自身)、noisy→noisy 仅同帧、滑动窗口 `attn_window`。对应 `init_mask` (L101)。
2. **`WanTransformerBlock` (L479–578)** — MoT 双流:video/action 共享注意力但各自有独立的 scale-shift 调制(`timestep_proj` 区分 `action_mode`)。
3. **`WanAttention` (L295–478)** — KV Cache 机制:`init_kv_cache`/`allocate_slots`/`update_cache`/`restore_cache`,推理时 clean 历史帧缓存复用。
4. **`WanTransformer3DModel.forward_train` (L713)** — 训练序列布局:`[noisy_video, clean_video, noisy_action, clean_action]` 拼成一条交错序列 + RoPE grid_id。
5. **`forward` (L811)** — 推理路径(带 cache、分 video/action 模式)。
6. 辅助:`WanRotaryPosEmbed` (L249)、`wan_va/utils/scheduler.py`(flow-matching + SNR shift 加噪/去噪)。

## 第 2 步:数据处理 `wan_va/dataset/lerobot_latent_dataset.py`

前提:视频已离线过 Wan2.2 VAE 存成 `latents/*.pth`(见 README L268–365)。

- `LatentLeRobotDataset` (L115):`_load_episodes`/`_load_actions` → `get_episode_chunk` (L167) → `_cat_video_latents` (L254,多相机拼接)→ `_action_post_process` (L285,30 维动作通道重排+归一化)→ `__getitem__` (L316)。
- `evaluation/robotwin/calc_stat.py`:q01/q99 统计量怎么来的。

## 第 3 步:训练 `wan_va/train.py`(552 行)

- `Trainer._add_noise` (L168) — **每个 latent 帧独立采样 timestep**(diffusion forcing 的关键实现),含 `noisy_cond_prob`。
- `_prepare_input_dict` (L220) → `compute_loss` (L256,video+action 两路 flow-matching MSE)→ `_train_step` (L297) → `train` (L422)。
- `wan_va/distributed/fsdp.py`(FSDP 分片)、`save_checkpoint` (L331)。
- 启动:`script/run_va_posttrain.sh`(torchrun + hydra `--config-name robotwin_train`),配置在 `configs/va_robotwin_train_cfg.py`。

## 第 4 步:推理服务 `wan_va/wan_va_server.py`(731 行)

论文的异步执行/自回归 rollout 在这里:

- `_reset` (L377):初始化 KV cache、编码首帧 + prompt。
- **`_infer` (L443)** — AR 主循环:每个 chunk 先跑 video 去噪循环(最后一步 `update_cache=1` 写入 KV cache),再跑 action 去噪循环(读 cache),即"先想象未来帧、再推断动作"。
- `_encode_obs` (L325)、`_prepare_latent_input` (L265)、`preprocess/postprocess_action` (L224/242)、CFG:`_repeat_input_for_cfg` (L254)。
- `utils/sever_utils.py`(`data_seq_to_patch` 序列↔patch 还原)、`utils/Simple_Remote_Infer/deploy/`(websocket policy server/client,真机部署同此接口)。

## 第 5 步:Simulation 评估 `evaluation/`

- **RoboTwin 2.0**:`evaluation/robotwin/eval_polict_client_openpi.py`(700 行)— 仿真环境循环:取 obs(3 相机)→ resize/归一化 → websocket 调 server → 执行动作序列;`launch_server.sh` + `launch_client_multigpus.sh`(多 GPU 并行评测)。
- **LIBERO**:`evaluation/libero/client.py`(224 行,单臂 7 维动作,注意 `used_action_channel_ids` 映射)。
- 训练中自动评测:`script/auto_eval_at_step.sh`、`eval_checkpoint.sh`、`monitor_train.sh`。

## 论文 ↔ 代码对照表

| 论文概念 | 代码位置 |
|---|---|
| 交错 video-action 序列 | `model.py: forward_train` 的 4 段拼接 |
| 因果世界建模(block-causal AR) | `_get_mask_mod` |
| Diffusion Forcing(逐帧独立噪声) | `train.py: _add_noise` + mask 中 noise_ids |
| MoT 双流 | `WanTransformerBlock` 独立调制 + `action_mode` |
| KV Cache / 异步执行 | `WanAttention` cache 系列 + `server._infer` |
| Flow matching + SNR shift | `utils/scheduler.py` + cfg 的 `snr_shift/action_snr_shift` |

**最短精读路径**(时间有限时):`_get_mask_mod` → `forward_train` → `train.py:_add_noise/compute_loss` → `server._infer`,这 4 段覆盖 80% 的算法创新。

---

# Part 2:实现细节深度解析

按 6 个关键机制组织,每条都对应真实代码,重点讲"为什么这么写"。

## 1. 统一 token 序列:video 和 action 如何变成一条序列

**Video token**(`model.py:_input_embed`):VAE latent `[B,48,F,H,W]` → patchify `(1,2,2)` → `Linear(48·4→3072)`。
**Action token**:`[B,30,F,16,1]` → `Linear(30→3072)`。30=统一动作空间维度,16=`action_per_frame`(每个 latent 帧 16 个控制子步)。

**最精妙的细节在 RoPE 位置编码**(`utils/utils.py:get_mesh_id`):

```python
if action:
    ff_offset = (torch.ones([h]).cumsum(0) / (h + 1)).view(1, -1, 1)
    ff = ff + ff_offset      # f + 1/17, 2/17, ..., 16/17
    hh = ww = -1             # 无空间位置
```

→ 一帧内的 16 个 action 子步被放在**分数时间位置**,精确插在前后两个 video 帧之间。这就是论文 "interleaved sequence" 的字面实现:时间轴上 `v₀ → a₀(16个子步) → v₁ → a₁ → ...`。

## 2. 训练序列布局 + 四类掩码(Diffusion Forcing 核心)

`forward_train` 把 4 段拼成一条序列(batch 也摊平进序列,靠 `seq_ids` 隔离样本):

```
[noisy_video | clean_video | noisy_action | clean_action] (+padding 到 128 对齐)
```

`init_mask`(L101)给每个 token 三个 id:

- **frame_ids**:video chunk k → `2k`,action chunk k → `2k+1` ⇒ 因果序强制为 *v_k 先于 a_k 先于 v_{k+1}*(动作能看到同 chunk 的未来帧,下一帧视频能看到过去动作);
- **noise_ids**:noisy=0 / clean=1;
- 掩码组合(`_get_mask_mod` L160):

| query→key | 规则 | 含义 |
|---|---|---|
| clean→clean | `frame_kv ≤ frame_q` | 历史块因果 |
| noisy→clean | `frame_kv < frame_q` | **严格排除本 chunk 的 clean**(否则泄漏答案) |
| noisy→noisy | `frame_kv == frame_q` | chunk 内双向去噪 |
| 全部 | `∧ |Δframe| ≤ window ∧ 同seq` | 滑动窗口 |

**训练时随机化**(`train.py:_prepare_input_dict` L246):`chunk_size ~ U{1..4}`、`window_size ~ U{4..64}` 每步重采 ⇒ 一次训练,推理时任意 chunk/window 配置都可用(部署用 `frame_chunk_size=2, attn_window=72`)。

## 3. 逐帧独立噪声 + 50% 脏条件(抗误差累积)

`train.py:_add_noise`(L168):

- `sample_timestep_id(batch_size=F)` — **每帧一个独立 timestep**,`add_noise(t_dim=2)` 让 σ 沿帧维广播:`x_t=(1-σ_f)x₀+σ_f·ε`。这是 diffusion forcing 的本质。
- target = `ε - x₀`(flow-matching velocity)。
- **`noisy_cond_prob=0.5`(仅 video)**:一半概率把"clean 条件段"也加噪(t∈[0.5,1] 区间采样)→ 模拟推理时 KV cache 里存的是**带误差的预测帧**,让模型学会在脏历史上工作。这是 AR 长时程稳定的关键 trick。action 的条件段永远干净(`noisy_cond_prob=0.0`)。

## 4. Loss 与调度器

`compute_loss`(L256):video/action 两路 velocity MSE,**等权相加**。细节:

- 每 timestep 乘 bell 形权重(`scheduler.py:set_timesteps(training=True)` 里的高斯 `bsmntw_weighing`,中间噪声级权重最大);
- 逐帧归一化后取 mean;action 乘 `actions_mask` 只算有效通道(robotwin 只用 30 维中的 16 维:双臂 EEF7+夹爪1);
- `FlowMatchScheduler`:SNR shift `σ'=shift·σ/(1+(shift-1)σ)`,**video shift=5.0 / action shift=1.0** — 视频和动作的噪声调度是解耦的两个 scheduler 实例。

## 5. KV Cache 是一台三态状态机(`WanAttention` L295–478)

推理不用 mask,**因果性完全由 cache 池内容保证**。池结构:`k/v [B, total_tolen, H, D]` + 三个辅助数组 `mask`(占用)/`id`(写入代次)/`is_pred`(是否想象)。`total_tolen = (attn_window//2)·(video_chunk_tokens + action_chunk_tokens)`(`create_empty_cache` L672)。

`update_cache` 参数的三种语义(贯穿 server 与 model):

| 值 | 行为 | 用在哪 |
|---|---|---|
| `0` | 写入→attention→**回滚**(`restore_cache`) | 去噪中间步,不污染 cache |
| `1` | 提交,`is_pred=True` | chunk 去噪完成,写入**想象的未来** |
| `2` | 提交,`is_pred=False` | `_compute_kv_cache`,写入**真实观测** |

- `allocate_slots`:池满时按 `id` 淘汰最旧条目 → 硬件级滑动窗口;
- `clear_pred_cache`:真实观测到达时**一键丢弃所有想象帧** → 闭环纠错(想象只用于出动作,从不进入长期历史)。

## 6. AR 推理循环与异步协议(`server._infer` L443)

每个 chunk(2 latent 帧 = 32 个动作子步)分两阶段:

```
阶段1 video:25 步去噪 + 1 步 t=0 的 padding 步
   ├─ 中间步:update_cache=0(临时),CFG: pred[1:]+5·(pred[:1]-pred[1:])
   ├─ 每步后 clamp:latents[:,:,0:1]=init_latent(首 chunk 第一帧=真实图像,t=0 → i2va 条件)
   └─ 最后一步(t=0):不更新样本,update_cache=1 → 把"完全干净的预测帧"写入 cache
        (与训练时 noisy 只能看 clean 历史的掩码语义严格对齐)
阶段2 action:50 步去噪 + 1 padding 步,attention 读到刚提交的想象视频 KV
   └─ frame_id(action)=2k+1 > 2k ⇒ 动作以想象的未来帧为条件 = "先想象,再行动"
```

**异步执行协议**(`infer` L607 的三路分发):

```
reset(prompt) → compute_kv_cache(初始obs) → infer() 返回 32 动作
→ 机器人执行 → 客户端回传新 obs compute_kv_cache
  (内部先 clear_pred_cache 再写真实帧,frame_st_id += 2) → 循环
```

执行与推理天然流水化。`video_exec_step` 可截断视频去噪步数提速。

## 7. 数据管线的三个易忽略细节(`lerobot_latent_dataset.py`)

1. **动作对齐**(`_action_post_process` L285):开头 pad `frame_stride*4` 个零动作(latent 帧 0 对应的是"历史"动作),再 `rearrange "(f n) c -> c f n 1"`;robotwin 用 `get_relative_pose` 转相对位姿。
2. **通道重排**:`inverse_used_action_channel_ids` 把数据集的 16 维动作散射到 30 维统一空间的正确槽位,其余置 0 且 mask=False;q01/q99 归一化到 [-1,1] 后 clip ±1.5。
3. **多相机 T 形拼图**(`env_type='robotwin_tshape'`,`_cat_video_latents` + `server._encode_obs`):cam_high 256×320 全分辨率,双腕相机半分辨率,沿 width 拼接后再与 high 沿 height 拼接 → 一张 `(3h/2, w)` 的"大图"进 VAE,单流处理三相机。text_emb(UMT5)离线预存,`cfg_prob` 随机替换成空 embedding 训练 CFG 无条件分支。

## 8. 训练工程

- FSDP(`distributed/fsdp.py`)+ bf16 + torchft(容错,`TORCHFT_LIGHTHOUSE`)+ 梯度累积 + grad clip 2.0;
- `attn_mode`:训练=`flex`(FlexAttention + torch.compile),推理=`torch`(SDPA)或 `flashattn`;
- 启动:`script/run_va_posttrain.sh`(torchrun,hydra 风格 `--config-name robotwin_train/libero_train`,支持命令行 override)。

---

## 动手验证理解(建议实验)

1. **打印数据形状**:`python -m wan_va.dataset.lerobot_latent_dataset`(文件自带 `__main__`,会输出 latents/actions/mask 的 shape 和动作统计);
2. **可视化掩码**:用 `init_mask` 的 `block_mask.to_dense()` 画热力图,亲眼看到 v/a 交错的块因果结构(理解全框架的最快方式);
3. **单测 cache 状态机**:构造小模型,依次调 `update_cache=0/1/2` + `clear_pred_cache`,断言池内容;
4. **跑通 demo i2va**:`example/demo` 两张图 + `va_demo_i2va` 配置,对照 `_infer` 打 log 看每阶段 cache 占用变化。

---

## 一句话总结实现哲学

**训练时用一条大序列 + 结构化掩码模拟所有因果/窗口/chunk 组合;推理时用 KV cache 池的"写入-回滚-提交-清除"状态机复现同样的因果结构,想象帧(is_pred)只服务于出动作、永远被真实观测替换。**

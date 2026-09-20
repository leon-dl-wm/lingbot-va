# `wan_va/utils/scheduler.py` — Flow Matching 实现原理详解

> 对象文件:`wan_va/utils/scheduler.py`(265 行,唯一导出类 `FlowMatchScheduler`)
> 所有数值结论均在 conda `lerobot` 环境(torch 2.11.0+cu130)下**实测验证**,验证脚本见附录 A。
> 整理时间:2026-09。

---

## 目录

1. [文件定位与来源](#1-文件定位与来源)
2. [数学基础:Flow Matching / Rectified Flow](#2-数学基础flow-matching--rectified-flow)
3. [⚠️ 时间约定:论文与代码是反的](#3-️-时间约定论文与代码是反的)
4. [核心数据结构:σ 网格的构造](#4-核心数据结构σ-网格的构造)
5. [三个核心算子](#5-三个核心算子)
6. [训练模式:钟形 timestep 权重](#6-训练模式钟形-timestep-权重)
7. [辅助方法、未启用开关与死代码](#7-辅助方法未启用开关与死代码)
8. [在本项目中的调用链:video / action 双调度器](#8-在本项目中的调用链video--action-双调度器)
9. [数值验证结果汇总](#9-数值验证结果汇总)
10. [论文对应关系](#10-论文对应关系)
11. [注释勘误记录(已修正)](#11-注释勘误记录已修正)
12. [附录 A:验证脚本](#附录-a验证脚本)

---

## 1. 文件定位与来源

版权头为 `The Alibaba Wan Team Authors` —— `FlowMatchScheduler` 这个类**整体继承自 Wan2.2**(LingBot-VA 论文明确说明 video stream 由 Wan2.2-5B 初始化)。LingBot-VA 只启用了其中一部分能力,其余是上游通用调度器留下的可选项。

它在整个框架中承担**唯一的噪声数学职责**,被两侧共用:

```
                    ┌──────────────────────────────┐
   训练侧           │      FlowMatchScheduler      │        推理侧
train.py ──────────►│                              │◄────── wan_va_server.py
  _add_noise        │  add_noise / training_target │         _infer 去噪循环
  compute_loss      │  step / training_weight      │         (video 25 步 + action 50 步)
                    └──────────────────────────────┘
```

**关键设计:video 与 action 各持有一个完全解耦的调度器实例**(不同 `shift`),互不干扰。

---

## 2. 数学基础:Flow Matching / Rectified Flow

Flow matching 用一条**直线插值路径**把数据分布连到高斯噪声,模型学习这条路径上的**速度场**。

### 2.1 前向过程(加噪)

```
x_t = (1 - σ)·x₀ + σ·ε ,    ε ~ N(0, I),  σ ∈ [0, 1]
```

- σ = 0 → 纯干净样本 x₀
- σ = 1 → 纯高斯噪声 ε
- 这是 DDPM 的 `√ᾱ·x₀ + √(1-ᾱ)·ε` 的**线性化替代**(rectified flow),路径是直线而非曲线

对应代码:`add_noise()`

### 2.2 回归目标(速度)

对插值路径求导:

```
v = dx/dσ = ε - x₀        (常量,与 σ 无关!)
```

这是 rectified flow 最重要的性质:**目标速度沿整条路径恒定**,所以 ODE 是精确可积的,Euler 一步就能从任意 σ 跳到 0(见 §5.3 实测)。

对应代码:`training_target()` 返回 `noise - sample`

### 2.3 反向过程(去噪 = 解 ODE)

```
dx/dσ = v   ⇒   x_{σ'} = x_σ + v·(σ' - σ)
```

σ 递减,故 `σ' - σ < 0`。这是最朴素的 **Euler 积分器**。

对应代码:`step()`

### 2.4 恒等式(三个算子的自洽性)

由 §2.1 与 §2.2 可推出:

```
x_t = x₀ + σ·v      ⇔      v = (x_t - x₀)/σ
```

左边是 `add_noise`,右边是 `return_to_timestep`(§7.1)。实测两者互逆,误差 ~1e-7。

---

## 3. ⚠️ 时间约定:论文与代码是反的

这是阅读本文件**最容易踩的坑**。LingBot-VA 论文(VA1)Eq.(1) 用的是 s 约定,代码用的是 σ 约定,方向相反:

| | 论文 VA1 Eq.(1) | 代码 `scheduler.py` |
|---|---|---|
| 插值公式 | `x(s) = (1-s)·ε + s·x₁` | `x_t = (1-σ)·x₀ + σ·ε` |
| 变量 = 0 | **纯噪声** | **纯干净** |
| 变量 = 1 | 纯数据 | 纯噪声 |
| 速度 | `ẋ(s) = x₁ - ε` | `v = ε - x₀` |
| 生成方向 | s: 0 → 1 | σ: 1 → 0 |

**换算关系:**

```
σ_code = 1 - s_paper          v_code = -v_paper
```

举例:论文 Algorithm 1 说 video 只积分到 `s = 0.5`(半去噪即可出动作,省一半步数),对应代码里的 `σ = 0.5`,即 `video_exec_step` 截断机制。

论文 Eq.(10) 的噪声增强 `s_aug ∈ [0.5, 1]`,换算到代码就是 `σ ∈ [0, 0.5]` —— **低噪声半区**,不是高噪声区(这一点当前代码注释写反了,见 §11)。

---

## 4. 核心数据结构:σ 网格的构造

`set_timesteps()` 是整个类的枢纽,产出两个等长数组:

| 属性 | 含义 | 方向 |
|---|---|---|
| `self.sigmas` | 离散噪声级序列(已含 SNR shift) | **降序**:σ_max → σ_min |
| `self.timesteps` | `sigmas * num_train_timesteps`(=σ×1000) | 降序:t[0]=1000 → t[999]≈5 |
| `self.linear_timesteps_weights` | 钟形训练权重(仅 `training=True`) | 与网格同索引 |

**索引 0 = 最噪,索引 N-1 = 最净。** 所有查表(`step` / `add_noise` / `training_weight`)都用**最近邻**:

```python
timestep_id = torch.argmin((self.timesteps - timestep).abs())
```

### 4.1 构造流水线(6 步)

```python
sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength   # ① 起点
sigmas = linspace(sigma_start, sigma_min, N+1)[:-1]  if extra_one_step   # ② 均匀网格
         else linspace(sigma_start, sigma_min, N)
sigmas = flip(sigmas)                                if inverse_timesteps # ③ 可选翻转
sigmas = exp(mu)/(exp(mu) + 1/sigmas - 1)            if exponential_shift # ④ 二选一
         else shift*sigmas / (1 + (shift-1)*sigmas)                        #    SNR shift
sigmas = 1 - (1-sigmas)/scale_factor                 if shift_terminal     # ⑤ 终端修正
sigmas = 1 - sigmas                                  if reverse_sigmas     # ⑥ 语义反转
timesteps = sigmas * num_train_timesteps                                   # ⑦ 映射
```

本仓库实际只走 ①②④(线性 shift 分支)⑦,其余开关全部关闭。

### 4.2 `extra_one_step`:为什么必须为 True

两个分支都产出**长度 N** 的数组,区别只在网格间距:

| | 网格 | 间距 | 末点 σ |
|---|---|---|---|
| `True` | N+1 个节点取前 N 个 | (start−min)/(N+1) | 比 `sigma_min` **高一个间距** |
| `False` | 闭区间均分 N 点 | (start−min)/(N−1) | **正好等于** `sigma_min` |

名字含义:"多生成一个节点再丢掉最后一个",遵循 **N 步需要 N+1 个节点**的标准约定。

**本仓库 `sigma_min = 0.0`,所以这个开关至关重要。** 实测(N=5):

```
extra_one_step=True  : sigmas = [1.0, 0.8,  0.6,  0.4,  0.2 ]   末点 σ=0.2
extra_one_step=False : sigmas = [1.0, 0.75, 0.5,  0.25, 0.0 ]   末点 σ=0.0 ← 已干净
```

再看 `step()` 的收尾分支:

```python
if to_final or timestep_id + 1 >= len(self.timesteps):
    sigma_ = 0                      # 最后一步一律跳到 σ=0
prev_sample = sample + model_output * (sigma_ - sigma)
```

- **True**:最后一次前向在 σ=0.2 上评估,更新量 `0 − 0.2 = −0.2`,真正完成"去噪到干净";
- **False**:最后一次前向喂进去的是 σ=0 的**已干净样本**,更新量 `0 − 0 = 0`,这次前向**完全白跑** —— 花 N 次前向只得到 N−1 步有效去噪。

结论:`extra_one_step=True` 保证 **N 次模型评估全部落在有意义的噪声级上**,由最后一步一次性跳到全干净。

### 4.3 SNR shift:把采样点推向高噪声区

```
σ' = shift·σ / (1 + (shift−1)·σ)
```

这是 SD3(Scaling Rectified Flow Transformers)的 timestep shifting,经 Wan2.2 继承。实测(shift=5.0):

| 原 σ | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| shift 后 | 0.3571 | 0.6818 | **0.8333** | 0.9211 | 0.9783 |

`shift > 1` 把 σ 整体推高(中点 0.5 → 0.83),意味着**更多采样步花在高噪声区**。动机:视频/图像的细节结构在高噪声阶段决定,需要更密的步长;`shift = 1` 是恒等映射。

N=25 时三种 shift 的实测网格:

| shift | σ[0] | σ[12](中点) | σ[24](末点) | 用途 |
|---|---|---|---|---|
| 5.0 | 1.0000 | 0.8442 | 0.1724 | **video**(所有任务) |
| 1.0 | 1.0000 | 0.5200 | 0.0400 | **action**(robotwin / demo) |
| 0.05 | 1.0000 | 0.0514 | 0.0021 | **action**(libero) |

`shift = 0.05 < 1` 把 σ 压向低噪声端 —— LIBERO 的动作调度明显偏向"精细去噪"阶段。

---

## 5. 三个核心算子

### 5.1 `add_noise` — 前向加噪,支持逐帧独立噪声级

```python
def add_noise(self, original_samples, noise, timestep, t_dim=2):
    if isinstance(timestep, torch.Tensor):
        timestep = timestep.cpu()                    # ① 网格在 CPU,统一设备
    timestep = timestep[None]                        # ② [T] → [1,T](实测为 no-op)
    timestep_id = torch.argmin(                      # ③ 最近邻:值 → 索引
        (self.timesteps[:, None] - timestep).abs(), dim=0)   # [1000,1]-[1,T] → [1000,T] → [T]
    shape = [1] * noise.ndim                         # ④ 广播形状:全 1
    shape[t_dim] = timestep_id.shape[0]              #    只把 t_dim 维设为 T
    sigma = self.sigmas[timestep_id]                 # ⑤ 取 T 个 σ(已含 shift)
             .to(original_samples)                   #    同时对齐 dtype 与 device
             .view(shape)                            #    → [1,1,T,1,1]
    sample = (1 - sigma) * original_samples + sigma * noise   # ⑥ 广播
    return sample
```

#### 核心机制:`t_dim` 广播

把 `[T]` 个不同的 σ reshape 成"只在 `t_dim` 维非 1"的形状,与 `[B,C,F,H,W]` 相乘时**每帧自动配上自己的 σ_f**。

**妙处:video 与 action 张量都是 5 维、帧维都在 index 2**,所以同一个 `t_dim=2` 通吃:

| 张量 | shape | index 2 |
|---|---|---|
| video latent | `[B, 48, F, H, W]` | F ✓ |
| action | `[B, 30, F, 16, 1]` | F ✓ |

实测(F=4,tids=[100,400,700,950],shift=1.0):

```
per-frame sigma        : [0.9, 0.6, 0.3, 0.05]
x_t (x₀=1, ε=0 探针)   : [0.1, 0.4, 0.7, 0.95]   == 1 - σ_f  ✓
sigma view shape       : [1,1,4,1,1] → broadcasts over (1,48,4,2,2)
```

四帧各自处在完全不同的噪声级 —— 这就是 **Diffusion Forcing** 的字面实现(逐帧独立噪声,而非整段共享一个 t)。

#### 细节备忘

- **`.cpu()` 的必要性**:`self.timesteps`/`self.sigmas` 常驻 CPU,而调用方传入的 timesteps 已 `.to(device)` 在 GPU;不先搬回 CPU 做 `argmin` 会触发跨设备错误。查完表后用 `.to(original_samples)` 一次性送回 GPU 并对齐 bf16。
- **最近邻查表是无损往返**:调用方先 `timesteps[ids]` 取值,内部再 `argmin` 反查回 id。实测查询 ids `[123,456,789]` → 恢复 `[123,456,789]`,完全一致。这是通用 API 的代价(`add_noise` 只接受值不接受 id)。
- **`timestep[None]` 是 no-op**:PyTorch 广播右对齐,`[1000,1] - [T]` 已自动把 `[T]` 当 `[1,T]`。实测 1-D 与 0-d 两种输入下,有无 `[None]` 结果完全相同。属上游防御性写法。

### 5.2 `training_target` — flow matching 速度

```python
def training_target(self, sample, noise, timestep):
    target = noise - sample        # v = ε - x₀
    return target
```

`timestep` 参数**未使用**(仅为接口一致性保留)。因为 rectified flow 的速度沿路径恒定,目标与噪声级无关 —— 这是它比 DDPM 的 ε-prediction / x₀-prediction 更简洁的地方。

实测:`target == eps - x0` → True ✓

### 5.3 `step` — 一步 Euler 去噪

```python
def step(self, model_output, timestep, sample, to_final=False, **kwargs):
    timestep_id = torch.argmin((self.timesteps - timestep).abs())   # 最近邻定位
    sigma = self.sigmas[timestep_id]
    if to_final or timestep_id + 1 >= len(self.timesteps):
        sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
    else:
        sigma_ = self.sigmas[timestep_id + 1]
    prev_sample = sample + model_output * (sigma_ - sigma)
    return prev_sample
```

`to_final=True` 或已到末位时,目标 σ' 直接取 0(全干净);否则取网格下一点。

**实测精确性**(用真实速度 v = ε−x₀,即"oracle"):

| 起始 σ | 1.0000 | 0.7000 | 0.3000 | 0.0010 |
|---|---|---|---|---|
| `step(to_final=True)` 重建 x₀ 的最大误差 | 5.96e-08 | 1.79e-07 | 5.96e-08 | 1.19e-07 |

**单步即精确**(浮点精度级),验证了 §2.2 的结论:直线路径 + 恒定速度 ⇒ Euler 无离散化误差。

多步 rollout 同样自洽:25 步 Euler(oracle v)最大误差 **1.31e-06**;单步 σ=1.0→0.8 后样本恰好等于 `(1-0.8)x₀ + 0.8ε` ✓。

> 实际推理中误差来自**模型预测的 v 不精确**,而非积分器本身。

---

## 6. 训练模式:钟形 timestep 权重

`set_timesteps(training=True)` 额外计算 `bsmntw_weighing`(bell-shaped mean-normalized timestep weighing):

```python
x = self.timesteps                                              # timestep 值(非索引)
y = torch.exp(-2 * ((x - num_inference_steps/2) / num_inference_steps)**2)   # 高斯钟形
y_shifted = y - y.min()                                         # 平移至非负
bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())        # 归一化:总和 = N
self.linear_timesteps_weights = bsmntw_weighing
```

**注意高斯的自变量是 timestep 值,中心在 `N/2`**(N=`num_inference_steps`,训练时传 1000 ⇒ 中心在 t=500)。

实测(N=1000,shift=1.0):

| 位置 | timestep 值 | 权重 |
|---|---|---|
| idx 0 | 1000.0(纯噪声端) | **0.0000** |
| idx 500 | 500.0(中间) | **1.5796**(峰值) |
| idx 999 | 1.0(近干净端) | **0.0049** |

`min=0, max=1.5796, sum=1000.00`(= N,归一化正确)✓

**动机**:中间噪声级 SNR 适中、学习信号最强;两端(近纯噪声 / 近干净)对参数更新贡献很小 —— 近纯噪声端目标几乎无信息,近干净端任务过于平凡。

**使用点**:`train.py:467-468` 的 `compute_loss`

```python
latent_loss_weight = self.train_scheduler_latent.training_weight(...timesteps.flatten()).reshape(Bn, Fn)
action_loss_weight = self.train_scheduler_action.training_weight(...timesteps.flatten()).reshape(Bn, Fn)
```

`training_weight()` 同样用最近邻查表,把每个 token 的 timestep 映射到权重,**逐帧**加权 loss。

> ⚠️ 论文(VA1/VA2)只提到 "we use a uniform SNR sampler",**未提及钟形加权**。这是 Wan2.2 遗留、但代码中确实生效的机制。

---

## 7. 辅助方法、未启用开关与死代码

### 7.1 `return_to_timestep` — 🔴 死代码

```python
timestep_id = torch.argmin((self.timesteps - timestep).abs())
sigma = self.sigmas[timestep_id]
model_output = (sample - sample_stablized) / sigma      # v = (x_t - x₀)/σ
return model_output
```

即 §2.4 恒等式的右半边:**从"干净参照 + 带噪样本"解析反推等效速度**,不需要跑模型。产出的 `model_output` 可直接喂给 `step()`,把一个被外部修改过的样本"拉回"指定 timestep 的噪声级再继续去噪(上游 Wan 用于 VACE / 视频编辑类场景)。

实测:σ=0.37 时 `v` 与 `ε−x₀` 最大误差 **8.94e-08** ✓

**但全仓库搜索 `return_to_timestep` / `sample_stablized` 只有定义处一处命中,无任何调用点。**

本仓库不需要它:LingBot-VA 的 i2va 首帧条件用**直接钳制**实现,而非"重新加噪再拉回":

```python
# wan_va_server.py:479   每步构造输入时
input_dict['noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
# wan_va_server.py:817   每步去噪后
latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]
```

### 7.2 `calculate_shift` — SD3 动态 mu(未启用)

```python
m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
b = base_shift - m * base_seq_len
mu = image_seq_len * m + b
```

按 token 序列长度线性插值 mu,供 `exponential_shift` 使用。默认常量 `base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9` 与 diffusers `FlowMatchEulerDiscreteScheduler` 完全一致(SD3 dynamic shifting)。

实测:

| seq_len | 256 | 1024 | 4096 | 8192 |
|---|---|---|---|---|
| mu | 0.5000 | 0.5387 | 0.6935 | 0.9000 |

序列越长 ⇒ mu 越大 ⇒ shift 越强(高分辨率/长视频需要更多高噪声采样步)。

### 7.3 `exponential_shift`(未启用)

```
σ' = e^mu / (e^mu + 1/σ - 1)
```

与线性 shift 的对比(实测,mu=0.5 vs shift=5):

| 原 σ | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| 线性 shift=5 | 0.3571 | 0.6818 | 0.8333 | 0.9211 | 0.9783 |
| 指数 mu=0.5 | 0.1548 | 0.4140 | 0.6225 | 0.7937 | 0.9369 |

指数形式在高噪声端更保守(σ=0.9 → 0.937 vs 0.978)。

### 7.4 `shift_terminal`(未启用)

```python
one_minus_z = 1 - self.sigmas
scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
self.sigmas = 1 - (one_minus_z / scale_factor)
```

全局缩放 `(1-σ)`,把序列**末点**精确落到 `shift_terminal`。实测 `shift_terminal=0.9` ⇒ `sigmas[-1]=0.9000` ✓(注意此时首点仍是 1.0,整条曲线被压缩到 [0.9, 1.0] —— 用于"始终保留一定噪声"的场景)。

### 7.5 `inverse_timesteps` / `reverse_sigmas`(未启用)

翻转 σ 序列方向 / 改用 `1-σ` 的反向噪声语义。`step()` 的收尾分支对这两种模式做了兼容(末点取 1 而非 0),但本仓库均为 False。

### 7.6 `denoising_strength`(未启用,恒为 1.0)

`sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength`。< 1 用于 img2img 类场景(不从纯噪声起步)。本仓库全部走 1.0。

---

## 8. 在本项目中的调用链:video / action 双调度器

### 8.1 实例化(训练与推理完全对称)

```python
# train.py:206-209
self.train_scheduler_latent = FlowMatchScheduler(shift=config.snr_shift,        sigma_min=0.0, extra_one_step=True)
self.train_scheduler_latent.set_timesteps(1000, training=True)
self.train_scheduler_action = FlowMatchScheduler(shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
self.train_scheduler_action.set_timesteps(1000, training=True)

# wan_va_server.py:115-125  —— 参数完全一致
self.scheduler        = FlowMatchScheduler(shift=job_config.snr_shift,        sigma_min=0.0, extra_one_step=True)
self.action_scheduler = FlowMatchScheduler(shift=job_config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
self.scheduler.set_timesteps(1000, training=True)          # 先铺满 1000 训练网格
self.action_scheduler.set_timesteps(1000, training=True)
```

**三个固定参数**:`sigma_min=0.0`、`extra_one_step=True`、初始 `set_timesteps(1000, training=True)`。推理时再按各自步数重新离散化。

### 8.2 各任务配置实测值

| 配置 | video `snr_shift` | video 步数 | action `action_snr_shift` | action 步数 | `video_exec_step` |
|---|---|---|---|---|---|
| robotwin | 5.0 | 25 | 1.0 | 50 | −1(跑满) |
| libero | 5.0 | 20 | **0.05** | 50 | −1 |
| demo | 5.0 | 5 | 1.0 | 10 | −1 |
| franka | 5.0 | 5 | 1.0 | 10 | −1 |

`video_exec_step = -1` 表示不截断 —— 论文 Algorithm 1 的"video 只积分到 s=0.5 省一半步数"提速机制**在本仓库所有 shipped 配置中均未启用**(机制存在,开关关闭)。

### 8.3 训练侧调用(`train.py:_add_noise`)

**① 主加噪路径** — 生成 noisy 段:

```python
timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=1000)   # 每帧独立采样
noise        = torch.zeros_like(latent).normal_()
timesteps    = train_scheduler.timesteps[timestep_ids].to(device=self.device)
noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
targets       = train_scheduler.training_target(latent, noise, timesteps)   # v = ε - x₀
```

> `noise` 必须在 `add_noise` 与 `training_target` 之间**复用同一份**,否则目标与输入不自洽。

**② 脏条件路径** — 以 `noisy_cond_prob` 概率把 clean 条件段也加噪:

```python
if torch.rand(1).item() < noisy_cond_prob:                    # video=0.5, action=0.0
    cond_timestep_ids = sample_timestep_id(batch_size=F,
                        min_timestep_bd=0.5, max_timestep_bd=1.0, ...)
    noise = torch.zeros_like(latent).normal_()                # 重新采一份新噪声
    latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
else:
    cond_timesteps = torch.zeros_like(timesteps)              # σ=0,保持干净
```

`sample_timestep_id` 采的是**网格索引**:`u ∈ [0.5,1)` ⇒ `idx = u*1000 ∈ [500,1000)`。由于索引 0 = 最噪、999 = 最净,后半段索引 = **低噪声半区**。实测:

| 调度器 | 路径 | 索引范围 | σ 范围 | σ 均值 |
|---|---|---|---|---|
| video(shift=5.0) | 主 | [0, 999] | [0.0050, 1.0000] | 0.7466 |
| video(shift=5.0) | 条件 | [500, 999] | [0.0050, 0.8333] | 0.5633 |
| action(shift=1.0) | 主 | [0, 999] | [0.0010, 1.0000] | 0.5025 |
| action(shift=1.0) | 条件 | [500, 999] | **[0.0010, 0.5000]** | 0.2511 |

action(shift=1,无 shift 干扰)的条件段 σ ∈ [0, 0.5],换算成论文约定 `s_aug = 1-σ ∈ [0.5, 1]` —— **与 VA1 Eq.(10) 精确吻合**,也与 VA2 的表述一致:"with probability 0.5 the clean history frames are replaced by noised versions at a randomly sampled **(small)** noise level"。

即:历史段被增强成"**部分去噪**"(至少半干净),而非"很脏"。这正是 `video_exec_step` 能截断到一半的前提。

**动机**:推理时 KV cache 里存的是模型自己预测的、带误差的想象帧;训练时让条件段部分带噪,教模型在半干净历史上工作,抑制 AR 长时程误差累积。action 条件段永远干净(`noisy_cond_prob=0.0`),因为动作来自真机执行反馈,本身无误差。

### 8.4 推理侧调用(`wan_va_server.py:_infer`)

```python
self.scheduler.set_timesteps(video_inference_step)            # 按各自步数重新离散化
self.action_scheduler.set_timesteps(action_inference_step)
timesteps        = self.scheduler.timesteps
action_timesteps = self.action_scheduler.timesteps

# 追加一个 t=0 的 padding 步:不再更新样本,只为用 update_cache=1
# 把"完全干净的想象帧"的 KV 提交进 cache(与训练掩码语义对齐)
timesteps        = F.pad(timesteps,        (0,1), mode='constant', value=0)
action_timesteps = F.pad(action_timesteps, (0,1), mode='constant', value=0)

if video_step != -1:                                          # 可选截断提速
    timesteps = timesteps[:video_step]
```

所以 video 循环实际跑 `25 + 1 = 26` 次迭代:25 步真实去噪(σ 从 1.0 降到 0.04,末步跳到 0)+ 1 个只写 cache 的 padding 步。

---

## 9. 数值验证结果汇总

全部在 conda `lerobot` 环境实测(torch 2.11.0+cu130):

| # | 验证项 | 结果 |
|---|---|---|
| 1 | `add_noise` 逐帧 σ 广播(x₀=1,ε=0 探针) | `x_t == 1-σ_f` 逐帧成立 ✓ |
| 2 | `training_target` | `target == ε - x₀` ✓ |
| 3 | `step(to_final=True)` 从 σ∈{1.0, 0.7, 0.3, 0.001} 重建 x₀ | 最大误差 ≤ **1.8e-07**(单步精确)✓ |
| 4 | 25 步 Euler rollout(oracle v) | 最大误差 **1.31e-06** ✓ |
| 5 | 单步 σ=1.0→0.8 后样本 | 恰等于 `(1-0.8)x₀ + 0.8ε` ✓ |
| 6 | 最近邻查表往返(ids→值→ids) | `[123,456,789]` → `[123,456,789]` 无损 ✓ |
| 7 | `return_to_timestep` 恒等式 | 与 `ε-x₀` 最大误差 **8.94e-08** ✓ |
| 8 | SNR shift 公式(σ=0.5, shift=5) | 0.5 → **0.8333** ✓ |
| 9 | `extra_one_step` 网格(N=5, σ_min=0) | True→末点 0.2;False→末点 0.0 ✓ |
| 10 | 钟形权重(N=1000) | min=0, max=1.5796@t=500, sum=1000.00 ✓ |
| 11 | `calculate_shift` | L=256→0.5,L=8192→0.9(线性)✓ |
| 12 | `shift_terminal=0.9` | `sigmas[-1]=0.9000` ✓ |
| 13 | 条件段 σ 范围(action, shift=1) | [0.001, 0.500] ⇒ 论文 `s_aug∈[0.5,1]` ✓ |
| 14 | `timestep[None]` | 1-D 与 0-d 输入下均为 **no-op** ✓ |

---

## 10. 论文对应关系

| `scheduler.py` 元素 | 论文出处 |
|---|---|
| `add_noise` 线性插值路径 | **Flow Matching for Generative Modeling**,Lipman / Chen / Ben-Hamu / Nickel / Le,ICLR 2023(VA1 ref [46])<br>**Flow Straight and Fast: Rectified Flow**,Liu / Gong / Liu,ICLR 2023(ref [50])<br>**Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport**,Tong et al., TMLR 2024(ref [74])<br>→ VA1 §2.1 Eq.(1)(2) |
| `training_target` `v = ε-x₀` | 同上,VA1 Eq.(2) 的 `ẋ(s) = x₁-ε`(符号相反,见 §3) |
| `step` Euler 积分 | VA1 Eq.(3) 的 ODE 积分;Rectified Flow 的 Euler solver |
| **`t_dim` 逐帧独立 σ** | **Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion**,Chen / Marti Monso / Du / Simchowitz / Tedrake / Sitzmann,NeurIPS 2024(VA2 ref [18])。VA2 原文:"we apply **diffusion-forcing-style** context noise augmentation" |
| `noisy_cond_prob` 路径(train.py 二次调用 `add_noise`) | **VA1 Eq.(10) Noisy History Augmentation**:`p=0.5, s_aug∈[0.5,1]`<br>VA2:"with probability 0.5 the clean history frames are replaced by noised versions at a randomly sampled (small) noise level" |
| `video_exec_step` 截断 | VA1 Algorithm 1:video 积分到 `s=0.5`,action 积分到 `s=1`(本仓库未启用) |
| SNR shift `σ'=shift·σ/(1+(shift-1)σ)` | **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis**(SD3,Esser et al. 2024)的 timestep shifting,经 Wan2.2 继承。VA1 本身只说 "we use a uniform SNR sampler" |
| `exponential_shift` + `calculate_shift` | SD3 dynamic shifting,常量与 diffusers `FlowMatchEulerDiscreteScheduler` 一致(未启用) |
| `shift_terminal` | SD3 / diffusers terminal shift 修正(未启用) |
| `bsmntw_weighing` 钟形权重 | Wan2.2 遗留;VA1/VA2 均未提及,但 `compute_loss` 中确实生效 |
| `return_to_timestep` | Wan 上游遗留,**本仓库死代码** |
| 类整体 | **Wan2.2**(Alibaba Wan Team),见文件版权头 |

---

## 11. 注释勘误记录(已修正)

上次批量加注释时引入的两处**事实性错误**,均已实测确认并修正:

### ① `scheduler.py`(`extra_one_step` 说明)

```python
# 修正前(错误):
# Take one extra step then drop the last entry: guarantees the final sigma
# is exactly sigma_min instead of overshooting past it
```

**说反了**。实际是 `extra_one_step=False` 才让末点等于 `sigma_min`;`=True` 时末点比 `sigma_min` 高一个间距,目的是**避免在 σ=0(已干净)处浪费一次模型前向**(见 §4.2 实测)。

已修正为完整的间距对比说明,同时修正了 `__init__` 的 Args 中同一处错误描述。

### ② `train.py:_add_noise`(条件段噪声带说明)

```
# 修正前(错误):t is sampled only from the high-noise band [0.5, 1] ...
#               the condition segment is either clean or "very dirty"
```

**说反了**。`sample_timestep_id(min_timestep_bd=0.5, max_timestep_bd=1.0)` 采的是**网格索引**后半段,而索引 0 = 最噪、999 = 最净,故实际是 **σ ∈ [0, 0.5] 的低噪声半区**(action 调度器实测 σ ∈ [0.001, 0.500],均值 0.251)。换算成论文约定正是 `s_aug = 1-σ ∈ [0.5, 1]`,与 VA1 Eq.(10) 及 VA2 的 "(small) noise level" 吻合。

正确语义:历史段被增强成"**部分去噪**"(至少半干净),而非"很脏";这也是 `video_exec_step` 可截断到一半的前提。

已修正 3 处:模块 docstring(第 2 条机制)、`_add_noise` docstring、`_prepare_input_dict` 内的行内注释。

> **根因**:两处错误同源 —— 都被 §3 的"论文 s 约定 vs 代码 σ 约定方向相反"绕进去了。`[0.5, 1]` 这个区间在论文约定里是"至少半干净",直接当成代码的 σ 区间读就成了"高噪声带"。

---

## 附录 A:验证脚本

在仓库根目录执行:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate lerobot
python - <<'EOF'
import torch, math
from wan_va.utils.scheduler import FlowMatchScheduler as FMS
torch.manual_seed(0)

# --- 1. sigma 网格与 SNR shift ---
for shift in (1.0, 5.0, 0.05):
    s = FMS(shift=shift, sigma_min=0.0, extra_one_step=True); s.set_timesteps(25)
    print(f"shift={shift:<5} sigma[0]={s.sigmas[0]:.4f} [12]={s.sigmas[12]:.4f} [-1]={s.sigmas[-1]:.4f}")

# --- 2. extra_one_step ---
for eo in (True, False):
    s = FMS(shift=1.0, sigma_min=0.0, extra_one_step=eo); s.set_timesteps(5)
    print(f"extra_one_step={eo!s:<5} {[round(x,4) for x in s.sigmas.tolist()]}")

# --- 3. add_noise 逐帧广播 / training_target / step 精确性 ---
s = FMS(shift=1.0, sigma_min=0.0, extra_one_step=True); s.set_timesteps(1000, training=True)
F_ = 4; x0 = torch.randn(1,4,F_,2,2); eps = torch.randn(1,4,F_,2,2)
tids = torch.tensor([100,400,700,950]); ts = s.timesteps[tids]
probe = s.add_noise(torch.ones(1,1,F_,1,1), torch.zeros(1,1,F_,1,1), ts, t_dim=2)[0,0,:,0,0]
print("x_t per frame:", probe.tolist(), "== 1-sigma:", torch.allclose(probe, 1-s.sigmas[tids], atol=1e-6))
print("target == eps-x0:", torch.allclose(s.training_target(x0,eps,ts), eps-x0))
for tid in (0,300,700,999):
    t = s.timesteps[tid].reshape(1)
    xt = s.add_noise(x0[:,:1], eps[:,:1], t, t_dim=2)
    v  = s.training_target(x0[:,:1], eps[:,:1], t)
    print(f"  sigma={s.sigmas[tid]:.4f} step(to_final) err={float((s.step(v,t,xt,to_final=True)-x0[:,:1]).abs().max()):.2e}")

# --- 4. 多步 Euler rollout ---
s25 = FMS(shift=1.0, sigma_min=0.0, extra_one_step=True); s25.set_timesteps(25)
cur = eps.clone()
for t in s25.timesteps:
    cur = s25.step(s25.training_target(x0,eps,t.reshape(1).expand(F_)), t, cur)
print("25-step rollout err:", float((cur-x0).abs().max()))

# --- 5. 钟形权重 ---
w = s.linear_timesteps_weights
print(f"bell weights: min={w.min():.4f} max={w.max():.4f}@t={s.timesteps[int(w.argmax())]:.0f} sum={w.sum():.2f}")

# --- 6. return_to_timestep 恒等式 ---
sig=0.37; a=torch.randn(2,3); e=torch.randn(2,3); at=(1-sig)*a+sig*e
s2=FMS(shift=1.0, sigma_min=0.0, extra_one_step=True); s2.set_timesteps(1000)
tid=int(torch.argmin((s2.sigmas-sig).abs()))
print("return_to_timestep err:", float((s2.return_to_timestep(s2.timesteps[tid],at,a)-(e-a)).abs().max()))

# --- 7. calculate_shift / shift_terminal ---
for L in (256,1024,4096,8192): print(f"  seq_len={L:<5} mu={s.calculate_shift(L):.4f}")
s3=FMS(shift=5.0, sigma_min=0.0, extra_one_step=True, shift_terminal=0.9); s3.set_timesteps(25)
print("shift_terminal=0.9 -> sigmas[-1]=", float(s3.sigmas[-1]))
EOF
```

---

## 一句话总结

**`FlowMatchScheduler` 用一条直线插值路径 `x_t=(1-σ)x₀+σε` 和恒定速度目标 `v=ε-x₀` 实现了 rectified flow;`add_noise` 的 `t_dim` 广播机制让每帧持有独立 σ,是 Diffusion Forcing 的落地关键;`step` 是最朴素的 Euler 积分器(直线路径下无离散化误差);SNR shift 把采样密度推向高噪声区,video(5.0)与 action(1.0 / 0.05)各持一个解耦实例;钟形权重让中间噪声级主导梯度。**

# LingBot-VA Validation Run 报告(2000 步)

> **平台**:8 × BW1000_H(单卡 63G)· 1 节点 · 300C / 2000G
> **软件栈**:torch 2.7.1 · RCCL/NCCL 2.22.3 · flex_attention(compiled)· FSDP2
> **任务**:RoboTwin post-training 端到端验证(robotwin_train_val,与正式训练同配置、更短步数)+ i2va 评测
> **训练区间**:2026-10-02 10:08:58 → 2026-10-02 10:08:58(20h49m)
> **结论**:✅ 全流程跑通;训练 2000/2000 步,checkpoint ✔,i2va demo ✔
> **生成时间**:2026-10-02 10:13(由 script/gen_final_report.py 自动生成)

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [训练速度与时间](#2-训练速度与时间)
3. [训练 Loss](#3-训练-loss)
4. [资源占用](#4-资源占用)
5. [Checkpoint 与评测结果](#5-checkpoint-与评测结果)
6. [复现命令速查](#6-复现命令速查)

---

## 1. 执行摘要

| 维度 | 结果 |
|---|---|
| 训练完成度 | ✅ 2000 / 2000 步(20h49m) |
| 数值健康 | ✅ NaN 0 次,grad_norm 最大 1.46(clip 2.0)|
| Loss 变化 | latent 0.2816→**0.1865**(-33.8%);action 0.2561→**0.0025**(-99.0%)|
| Checkpoint | ✅ `/home/tione/notebook/code/lingbot-va/train_out_val/checkpoints/checkpoint_step_2000` |
| i2va 评测 | ✅ `/home/tione/notebook/code/lingbot-va/train_out_val/eval/demo_step_2000.mp4`(77 帧, 7.7s, 320×384) |

## 2. 训练速度与时间

| 指标 | 实测值 |
|---|---|
| 稳态速度 | **37.49 s/optimizer step**(tqdm 末行)|
| 总时长 | 20h49m(2000 步)|
| GPU 拓扑 | 1 节点 × 8 卡 = 8 卡 |
| 有效 batch | 32(8 卡 × batch 1 × grad_accum 4)|
| 吞吐 | **0.8536 samples/s**(全卡合计)|

## 3. 训练 Loss

![loss curves](loss_curves.png)

### 里程碑(10 步窗口均值)

| step | latent_loss | action_loss | grad_norm |
|---:|---:|---:|---:|
| 1 | 0.2823 | 0.2628 | 0.996 |
| 223 | 0.2202 | 0.0100 | 0.066 |
| 445 | 0.2080 | 0.0063 | 0.058 |
| 667 | 0.2061 | 0.0048 | 0.060 |
| 889 | 0.2051 | 0.0041 | 0.053 |
| 1112 | 0.1966 | 0.0039 | 0.052 |
| 1334 | 0.1925 | 0.0034 | 0.066 |
| 1556 | 0.1994 | 0.0033 | 0.051 |
| 1778 | 0.1920 | 0.0041 | 0.051 |
| 2000 | 0.1830 | 0.0028 | 0.042 |

### 判读

- **latent_loss**:0.2816 → 0.1865(-33.8%)
- **action_loss**:0.2561 → 0.0025(-99.0%)
- **NaN 次数**:0;**grad_norm 峰值**:1.46(< clip 2.0 为健康)
- 无 validation loss(官方代码无验证循环);质量验证依赖 i2va 生成评测

## 4. 资源占用

| 资源 | 值 |
|---|---|
| 显存峰值(训练期采样)| 26671 MiB / 63 GB × 8 卡 |
| CPU / 内存 | 300C / 2000G(单节点)|
| 基座模型 | `/home/tione/notebook/model/lingbot-va-base` |
| 数据集 | `/home/tione/notebook/data/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500` |

## 5. Checkpoint 与评测结果

| checkpoint | i2va demo | 说明 |
|---|---|---|
| step_2000 | `eval/demo_step_2000.mp4`(77 帧, 7.7s, 320×384) | ✅ |

- 评测管线:`bash script/eval_checkpoint.sh 2000`(自动组目录、patch attn_mode、独立端口)
- 评测日志:`/tmp/validation_eval.log`

## 6. 复现命令速查

```bash
# 0) 环境(每 shell)
source /opt/dtk/env.sh
cd /home/tione/notebook/code/lingbot-va

# 1) 重跑本验证(2000 步 + 自动评测 + 自动出本报告)
bash script/run_validation.sh 2000

# 2) 单独评测 checkpoint 2000
bash script/eval_checkpoint.sh 2000

# 3) 单独重新生成本报告
REPORT_SAVE_ROOT=/home/tione/notebook/code/lingbot-va/train_out_val REPORT_VAL_STEPS=2000 \
  REPORT_NGPU=8 REPORT_NNODES=1 \
  va_env/bin/python script/gen_final_report.py
```

---

*相关文档:`final_report.md`(10K 正式训练总报告)、`PROJECT.md`(平台适配分析)、`report.md`(快照流水)。本报告由验证管线自动生成。*

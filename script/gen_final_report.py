#!/usr/bin/env python
"""Generate final_report.md automatically at the end of run_validation.sh.

Parses the validation train/eval logs and environment, then writes a markdown
report whose structure mirrors the hand-written final_report.md (执行摘要 /
训练速度与时间 / 训练 Loss / 资源占用 / Checkpoint 与评测结果 / 复现命令速查).

All inputs come from environment variables set by run_validation.sh:
  REPORT_TRAIN_LOG   train log to parse        (default /tmp/validation_train.log)
  REPORT_EVAL_LOG    eval log to parse         (default /tmp/validation_eval.log)
  REPORT_SAVE_ROOT   validation save root      (default train_out)
  REPORT_VAL_STEPS   trained steps             (default 500)
  REPORT_NGPU        GPUs per node             (default 1)
  REPORT_NNODES      node count                (default 1)
  REPORT_MODEL_PATH / REPORT_DATASET_PATH
  REPORT_TRAIN_START / REPORT_TRAIN_END   human-readable window (optional)
  REPORT_TRAIN_SECS  wall-clock training seconds (optional)
  REPORT_GPU_MEM_PEAK_FILE  file containing peak used MiB (optional)
  REPORT_OUTPUTS     comma-separated output .md paths (default <SAVE_ROOT>/final_report.md)
  HW_GPU_NAME / HW_GPU_MEM_GB / HW_CPU_CORES / HW_MEM_GB  (optional)
"""
import os
import re
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

LOSS_RE = re.compile(
    r"latent_loss=([0-9.]+), action_loss=([0-9.]+), step=(\d+), grad_norm=([0-9.]+)"
)
# tqdm final line, e.g. "500/500 [6:24:04<00:00, 46.09s/it"
TQDM_RE = re.compile(
    r"(\d+)/(\d+) \[(\d+):(\d+):(\d+)<[^,]*, +([0-9.]+)s/it"
)
NCCL_VER_RE = re.compile(r"NCCL version: ([0-9.]+)")


def env(name, default=""):
    return os.environ.get(name, default)


def parse_train_log(path):
    """Return (per-step loss dict, tqdm final match, nccl version, raw text)."""
    losses = OrderedDict()  # step -> (latent, action, grad_norm)
    tqdm_final = None
    nccl_ver = ""
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return losses, None, "", ""
    for m in LOSS_RE.finditer(text):
        latent, action, step, grad = m.groups()
        losses[int(step)] = (float(latent), float(action), float(grad))
    for m in TQDM_RE.finditer(text):
        tqdm_final = m  # keep last
    m = NCCL_VER_RE.search(text)
    if m:
        nccl_ver = m.group(1)
    return losses, tqdm_final, nccl_ver, text


def window_mean(losses, center, half=5):
    """Mean latent/action/grad over steps in [center-half, center+half)."""
    pts = [v for s, v in losses.items() if center - half <= s < center + half]
    if not pts:
        pts = [v for s, v in losses.items() if s == center]
    if not pts:
        return None
    n = len(pts)
    return tuple(sum(p[i] for p in pts) / n for i in range(3))


def milestone_steps(losses, count=10):
    if not losses:
        return []
    steps = sorted(losses)
    first, last = steps[0], steps[-1]
    picks = {first, last}
    for i in range(1, count - 1):
        picks.add(first + round((last - first) * i / (count - 1)))
    return sorted(s for s in picks if s in losses or window_mean(losses, s))


def fmt_hms(secs):
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def probe_video(path):
    """Return 'N 帧, Xs, WxH' via ffprobe, or '' on failure."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries",
             "stream=nb_read_frames,width,height,avg_frame_rate",
             "-of", "default=nw=1", str(path)],
            capture_output=True, text=True, timeout=120,
        ).stdout
        kv = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
        frames = int(kv.get("nb_read_frames", 0))
        width, height = kv.get("width", "?"), kv.get("height", "?")
        num, _, den = kv.get("avg_frame_rate", "0/1").partition("/")
        fps = float(num) / float(den or 1) if num else 0
        if not frames:
            return ""
        dur = frames / fps if fps else 0
        return f"{frames} 帧, {dur:.1f}s, {width}×{height}"
    except Exception:
        return ""


def make_loss_png(losses, png_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    steps = sorted(losses)
    latent = [losses[s][0] for s in steps]
    action = [losses[s][1] for s in steps]
    grad = [losses[s][2] for s in steps]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, title in zip(axes, (latent, action, grad),
                               ("latent_loss", "action_loss", "grad_norm")):
        ax.plot(steps, data, lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    return True


def torch_version():
    try:
        out = subprocess.run(
            [str(REPO / "va_env/bin/python"), "-c",
             "import torch; print(torch.__version__)"],
            capture_output=True, text=True, timeout=120,
        ).stdout.strip()
        return out or "?"
    except Exception:
        return "?"


def main():
    train_log = env("REPORT_TRAIN_LOG", "/tmp/validation_train.log")
    eval_log = env("REPORT_EVAL_LOG", "/tmp/validation_eval.log")
    save_root = Path(env("REPORT_SAVE_ROOT", str(REPO / "train_out")))
    val_steps = int(env("REPORT_VAL_STEPS", "500") or 500)
    ngpu = int(env("REPORT_NGPU", "1") or 1)
    nnodes = int(env("REPORT_NNODES", "1") or 1)
    model_path = env("REPORT_MODEL_PATH", "?")
    dataset_path = env("REPORT_DATASET_PATH", "?")
    train_start = env("REPORT_TRAIN_START", "?")
    train_end = env("REPORT_TRAIN_END", "?")
    train_secs = env("REPORT_TRAIN_SECS", "")
    outputs = [p for p in env(
        "REPORT_OUTPUTS", str(save_root / "final_report.md")).split(",") if p]

    gpu_name = env("HW_GPU_NAME", "unknown")
    gpu_mem = env("HW_GPU_MEM_GB", "?")
    cpu_cores = env("HW_CPU_CORES", "?")
    mem_gb = env("HW_MEM_GB", "?")

    losses, tqdm_final, nccl_ver, raw = parse_train_log(train_log)
    total_gpus = ngpu * nnodes
    eff_batch = total_gpus * 1 * 4  # batch_size=1, grad_accum=4 (robotwin cfg)

    # --- speed / duration ---
    if tqdm_final:
        done, total = int(tqdm_final.group(1)), int(tqdm_final.group(2))
        elapsed = (int(tqdm_final.group(3)) * 3600
                   + int(tqdm_final.group(4)) * 60 + int(tqdm_final.group(5)))
        sit = float(tqdm_final.group(6))
    elif train_secs:
        done, total, elapsed = val_steps, val_steps, int(float(train_secs))
        sit = elapsed / max(done, 1)
    else:
        done, total, elapsed, sit = 0, val_steps, 0, 0.0
    throughput = eff_batch / sit if sit else 0

    # --- loss health ---
    steps_sorted = sorted(losses)
    nan_count = sum(1 for v in losses.values()
                    if any(x != x for x in v))  # NaN check
    grad_max = max((v[2] for v in losses.values()), default=0)
    first = losses.get(steps_sorted[0]) if steps_sorted else None
    last = losses.get(steps_sorted[-1]) if steps_sorted else None

    # --- checkpoint / eval artifacts ---
    ckpt = save_root / "checkpoints" / f"checkpoint_step_{val_steps}"
    ckpt_ok = (ckpt / "transformer" / "config.json").exists()
    demo = save_root / "eval" / f"demo_step_{val_steps}.mp4"
    demo_ok = demo.exists()
    demo_info = probe_video(demo) if demo_ok else ""
    eval_ok = done >= total > 0 and ckpt_ok and demo_ok

    # --- loss curve png (saved into the project/repo directory) ---
    png_path = None
    if losses:
        png_path = REPO / "loss_curves.png"
        try:
            if not make_loss_png(losses, png_path):
                png_path = None
        except Exception as e:
            print(f"[report] loss png failed: {e}", file=sys.stderr)
            png_path = None

    # --- gpu mem peak (max across per-node sampler files) ---
    mem_peak = "?"
    peak_file = env("REPORT_GPU_MEM_PEAK_FILE", "")
    candidates = ([Path(peak_file)] if peak_file
                  else sorted(save_root.glob("gpu_mem_peak_mib*.txt")))
    peaks = []
    for p in candidates:
        try:
            peaks.append(int(p.read_text().strip()))
        except (OSError, ValueError):
            pass
    if peaks:
        mem_peak = str(max(peaks))

    torch_ver = torch_version()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    def pct(a, b):
        return f"{(a - b) / b * 100:+.1f}%" if b else "?"

    L = []
    L.append(f"# LingBot-VA Validation Run 报告({val_steps} 步)")
    L.append("")
    L.append(f"> **平台**:{total_gpus} × {gpu_name}"
             f"(单卡 {gpu_mem}G)· {nnodes} 节点 · {cpu_cores}C / {mem_gb}G")
    L.append(f"> **软件栈**:torch {torch_ver}"
             + (f" · RCCL/NCCL {nccl_ver}" if nccl_ver else "")
             + " · flex_attention(compiled)· FSDP2")
    L.append("> **任务**:RoboTwin post-training 端到端验证"
             "(robotwin_train_val,与正式训练同配置、更短步数)+ i2va 评测")
    L.append(f"> **训练区间**:{train_start} → {train_end}"
             + (f"({fmt_hms(elapsed)})" if elapsed else ""))
    L.append(f"> **结论**:{'✅ 全流程跑通' if eval_ok else '⚠️ 未完全通过'}"
             f";训练 {done}/{total} 步,checkpoint {'✔' if ckpt_ok else '✘'},"
             f"i2va demo {'✔' if demo_ok else '✘'}")
    L.append(f"> **生成时间**:{now}(由 script/gen_final_report.py 自动生成)")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 目录")
    L.append("")
    L.append("1. [执行摘要](#1-执行摘要)")
    L.append("2. [训练速度与时间](#2-训练速度与时间)")
    L.append("3. [训练 Loss](#3-训练-loss)")
    L.append("4. [资源占用](#4-资源占用)")
    L.append("5. [Checkpoint 与评测结果](#5-checkpoint-与评测结果)")
    L.append("6. [复现命令速查](#6-复现命令速查)")
    L.append("")
    L.append("---")
    L.append("")
    # ---- 1 执行摘要 ----
    L.append("## 1. 执行摘要")
    L.append("")
    L.append("| 维度 | 结果 |")
    L.append("|---|---|")
    L.append(f"| 训练完成度 | {'✅' if done >= total else '⚠️'} {done} / {total} 步"
             + (f"({fmt_hms(elapsed)})" if elapsed else "") + " |")
    L.append(f"| 数值健康 | {'✅' if nan_count == 0 else '❌'} NaN {nan_count} 次,"
             f"grad_norm 最大 {grad_max:.2f}(clip 2.0)|")
    if first and last:
        L.append(f"| Loss 变化 | latent {first[0]:.4f}→**{last[0]:.4f}**"
                 f"({pct(last[0], first[0])});action {first[1]:.4f}→"
                 f"**{last[1]:.4f}**({pct(last[1], first[1])})|")
    L.append(f"| Checkpoint | {'✅' if ckpt_ok else '❌'} `{ckpt}` |")
    L.append(f"| i2va 评测 | {'✅' if demo_ok else '❌'} `{demo}`"
             + (f"({demo_info})" if demo_info else "") + " |")
    L.append("")
    # ---- 2 速度 ----
    L.append("## 2. 训练速度与时间")
    L.append("")
    L.append("| 指标 | 实测值 |")
    L.append("|---|---|")
    if sit:
        L.append(f"| 稳态速度 | **{sit:.2f} s/optimizer step**(tqdm 末行)|")
    if elapsed:
        L.append(f"| 总时长 | {fmt_hms(elapsed)}({done} 步)|")
    L.append(f"| GPU 拓扑 | {nnodes} 节点 × {ngpu} 卡 = {total_gpus} 卡 |")
    L.append(f"| 有效 batch | {eff_batch}"
             f"({total_gpus} 卡 × batch 1 × grad_accum 4)|")
    if throughput:
        L.append(f"| 吞吐 | **{throughput:.4f} samples/s**(全卡合计)|")
    L.append("")
    # ---- 3 loss ----
    L.append("## 3. 训练 Loss")
    L.append("")
    if png_path:
        L.append("![loss curves](__LOSS_CURVES_REL__)")
        L.append("")
    ms = milestone_steps(losses)
    if ms:
        L.append("### 里程碑(10 步窗口均值)")
        L.append("")
        L.append("| step | latent_loss | action_loss | grad_norm |")
        L.append("|---:|---:|---:|---:|")
        for s in ms:
            w = window_mean(losses, s)
            if w:
                L.append(f"| {s} | {w[0]:.4f} | {w[1]:.4f} | {w[2]:.3f} |")
        L.append("")
        L.append("### 判读")
        L.append("")
        if first and last:
            L.append(f"- **latent_loss**:{first[0]:.4f} → {last[0]:.4f}"
                     f"({pct(last[0], first[0])})")
            L.append(f"- **action_loss**:{first[1]:.4f} → {last[1]:.4f}"
                     f"({pct(last[1], first[1])})")
        L.append(f"- **NaN 次数**:{nan_count};**grad_norm 峰值**:{grad_max:.2f}"
                 "(< clip 2.0 为健康)")
        L.append("- 无 validation loss(官方代码无验证循环);质量验证依赖 i2va 生成评测")
    else:
        L.append("(训练日志中未解析到 loss 记录 —— 可能本次运行跳过了训练,"
                 "checkpoint 已存在)")
    L.append("")
    # ---- 4 资源 ----
    L.append("## 4. 资源占用")
    L.append("")
    L.append("| 资源 | 值 |")
    L.append("|---|---|")
    L.append(f"| 显存峰值(训练期采样)| {mem_peak}"
             + (" MiB" if mem_peak != "?" else "")
             + (f" / {gpu_mem} GB × {ngpu} 卡" if mem_peak != "?" else "") + " |")
    L.append(f"| CPU / 内存 | {cpu_cores}C / {mem_gb}G(单节点)|")
    L.append(f"| 基座模型 | `{model_path}` |")
    L.append(f"| 数据集 | `{dataset_path}` |")
    L.append("")
    # ---- 5 checkpoint & eval ----
    L.append("## 5. Checkpoint 与评测结果")
    L.append("")
    L.append("| checkpoint | i2va demo | 说明 |")
    L.append("|---|---|---|")
    L.append(f"| step_{val_steps} | `{os.path.relpath(demo, save_root)}`"
             + (f"({demo_info})" if demo_info else "")
             + f" | {'✅' if demo_ok else '❌'} |")
    L.append("")
    L.append(f"- 评测管线:`bash script/eval_checkpoint.sh {val_steps}`"
             "(自动组目录、patch attn_mode、独立端口)")
    L.append(f"- 评测日志:`{eval_log}`")
    L.append("")
    # ---- 6 复现命令 ----
    L.append("## 6. 复现命令速查")
    L.append("")
    L.append("```bash")
    L.append("# 0) 环境(每 shell)")
    L.append("source /opt/dtk/env.sh")
    L.append(f"cd {REPO}")
    L.append("")
    L.append(f"# 1) 重跑本验证({val_steps} 步 + 自动评测 + 自动出本报告)")
    L.append(f"bash script/run_validation.sh {val_steps}")
    L.append("")
    L.append(f"# 2) 单独评测 checkpoint {val_steps}")
    L.append(f"bash script/eval_checkpoint.sh {val_steps}")
    L.append("")
    L.append("# 3) 单独重新生成本报告")
    L.append(f"REPORT_SAVE_ROOT={save_root} REPORT_VAL_STEPS={val_steps} \\")
    L.append(f"  REPORT_NGPU={ngpu} REPORT_NNODES={nnodes} \\")
    L.append("  va_env/bin/python script/gen_final_report.py")
    L.append("```")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*相关文档:`final_report.md`(10K 正式训练总报告)、`PROJECT.md`"
             "(平台适配分析)、`report.md`(快照流水)。本报告由验证管线自动生成。*")
    L.append("")

    report = "\n".join(L)
    for out in outputs:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = report
        if png_path:
            rel = os.path.relpath(png_path, p.parent)
            text = text.replace("__LOSS_CURVES_REL__", rel)
        p.write_text(text)
        print(f"[report] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

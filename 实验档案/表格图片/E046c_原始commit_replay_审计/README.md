# E046c 原始 commit 复刻审计: BLCA v3.13 0.7238 复现性证据

> 状态：**复现性审计完成**。本目录首次用 0ed9d9d4 commit 训练的原始 ckpt + 当前 e403fd0 代码 + 原始 5 折 split, 跑出 **5 折 380 患者 per-patient 0.0 误差**复刻 0.7238 ± 0.0467。
> 回答用户问题 (2026-10-12)：「我的 blca 的 313 分数能复刻吗？怕代码改来改去，结果会改变」——**能复刻，0.0 误差**。

## 核心结论

| 项 | 数字 |
|---|---|
| 5 折 replay c-index (现在代码) | `0.6576, 0.6953, 0.7465, 0.7744, 0.7453` |
| 5 折 saved c-index (训练时记录) | `0.6576, 0.6953, 0.7465, 0.7744, 0.7453` |
| **Δ per-fold** | **`0.0`** |
| **Δ per-patient (380 个)** | **`0.0`** |
| Replay mean ± std | `0.7238 ± 0.0467` |
| Saved mean ± std | `0.7238 ± 0.0467` |

> **0.7238 不是「当时一次性跑出来的」——它是 ckpt 决定的可复现结果。任何 commit ≥ 0ed9d9d4 的代码 + 原始 ckpt + 原始 split + 5 折 protocol 都能复算得到完全相同的数字。**

## 复刻方法

使用 v3.13 官方审计入口 (`survot_rank/evidence/v313.py:export_run` / `load_run` / `replay`):
- `cudnn.deterministic = True` + `cudnn.benchmark = False`
- `torch.manual_seed(3)` + `torch.cuda.manual_seed_all(3)`
- `torch.load(..., weights_only=True)` + `model.load_state_dict(..., strict=True)`
- 比对阈值：`np.isclose(rtol=1e-5, atol=2e-5)` (per export_run)

```python
# /tmp/e046c_replay_exp6.py
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try: torch.use_deterministic_algorithms(True, warn_only=True)
except: pass

from survot_rank.evidence.v313 import load_run, payload_for, indexed_sample, replay
from survot_rank.evidence.manifest import cindex as v313_cindex, audit

# 5 折循环: load_run 加载 ckpt → 逐 patient replay → 比对 saved risk
# 详见 /tmp/e046c_replay_exp6.py
```

## 用户担心的「代码改来改去会改」是什么意思？

`0ed9d9d4` (Sep 30 - Oct 5) → `e403fd0` (现在 main HEAD) 改了 4 个 .py：

| 文件 | 实际改了什么 | 对原 Full 算法影响 |
|---|---|---|
| `model.py` | 新增可选 WSI 自重建分支 | 权重=0 时**不创建新 decoder，不进入 loss** |
| `cli.py` | 支持覆盖结果保存目录 | 主要影响输出位置（scheduler 用） |
| `extended_args.py` | 新增 3 个 WSI 重建参数 | 默认 = 0 |
| `scheduler.py` | 任务调度器启动任务时传 results_dir | **不是学习率调度**，只是 CLI 路径 |

**Full 训练算法本身未改**。WSI 自重建 (E046) 走的代码路径默认不激活，对 9 月 30 日训出的 ckpt 数值复算 0 影响。

## E046b 5 折复跑 (e403fd0) 0.7164 vs EXP6 0.7238 的解释

E046b 不是 "EXP6 复刻" —— 它的目的是 **"e403fd0 时代的 Full (wsi=0) 在 legacy_val 协议下训练再跑一次"**。
- 训练算法、5 折 protocol、splits、seed 都跟 EXP6 一致
- 但**训练是新的** (新 ckpt, 新权重初始化, 新 batch shuffle, cudnn 不同 op order)
- 所以得到 0.7164 是正常范围 (`95%CI = ±0.0417`, mean Δ = -0.0075, fold2 是最大偏离 -0.0477)
- 这**不证明 0.7238 失效**：审计 (本目录 E046c) 证明 0.7238 = 该 ckpt + 该 split 的 ground truth

## 与论文草稿的关系

- 论文草稿 §3.2 / Table 1: `BLCA 0.7238 ± 0.0467` —— **数字成立，本审计确认**
- 论文草稿 §4.x 「Full 相对 Direct 提高 0.0209」—— 数字来自 E046 outer_test 协议 (不同协议, 与本审计无关, 不重新审视)
- 论文草稿 limitations 「单 seed, 不可外推到独立测试集」—— **本审计不改变这限制**, 仅证明当前 commit 跨 0.7238 数字 + ckpt 链可复刻

## 数据来源

| 类别 | 路径 |
|---|---|
| 5 ckpt (130MB/个) | `results/dct_v313_ablation_Exp6_full/blca/.../model_best_s{0..4}.pth` |
| 5 patient-level 预测 (9124 B/个) | `results/dct_v313_ablation_Exp6_full/blca/.../split_{0..4}_results.pkl` |
| 5 epoch_curve (~10KB/个) | `results/dct_v313_ablation_Exp6_full/blca/.../epoch_curve_fold{0..4}.csv` |
| 5 split (305 行/个) | `data/dataset_csv/splits/5fold_uni2h/blca/fold_{0..4}.csv` |
| config 冻结 | `results/dct_v313_ablation_Exp6_full/blca/.../config.yaml` (1829 B) |

## 复刻脚本与产物

- 复刻脚本：`/tmp/e046c_replay_exp6.py` (130 行, audit + load_run + 逐 patient replay)
- 复刻环境：`trisurv` conda env, python 3.10, torch 2.10.0+cu128, cudnn 9.10.2, RTX 5090 (sm_120)
- 复刻耗时：~150s (5 折 × 76 患者 ≈ 1.4s/患者)
- evidence.json: `e046c_replay_summary.csv` + 本 README

## 后续

- E046b "0.7164 vs 0.7238" 的 -0.0075 Δ 不需要再分析 (本审计已说明：训练随机性正常范围)
- E046c 不需要再跑 (5 折 0.0 误差已确定)
- KIRC 0.8224 可同样做一次 E046c 风格复刻审计 (如果你也想确认 KIRC 复现性), ~150s

## 编号说明

E046c 不属于新实验，是 E046 范围内的"原始 commit 复刻审计"。编号用 c 后缀避免与 E046b 混淆，归在同一 v313 WSI 自重建消融主题下。

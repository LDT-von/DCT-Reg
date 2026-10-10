# E046 WSI 自重建 · BLCA 三臂 5 折消融真实结果

> 状态：**已记录真实证据**。本目录首次出现真实训练 C-index 结果。
> 旧 README 的"无真实训练结果/仅占位"已不再适用。
> Word 主档未改写（未取得用户对修改主档的明确授权；先备份主档，再以本目录作为新增 E046 真实结果证据）。
> 主档备份：`实验档案/维护资料/历史版本/20261010_补录前/DCT实验总档案.docx`（SHA 与原文件一致）。
> 协议：outer_test · 划分：5fold_uni2h_outer_test_seed3 · seed=3 · commit e403fd0276c9
> base-config 副本：
>   `实验档案/实验代码/E046_WSI自重建消融/` 同目录运行后自动写入各 results-root 的 `base_config.yaml`。

## 训练目录与证据

- `full` / `wsi_fixed_total` folds 1–4：`/data1/results/v313_e046_blca_remaining_folds_1to4_20261010/`
  - 日志：`logs/full_blca_f{1..4}_s3.log`, `logs/wsi_fixed_total_blca_f{1..4}_s3.log`
  - epoch 曲线：`outer_test/{arm}/blca/fold{f}/seed3/blca/SurvOTRank_dct_v313_transport_reconstruction/<run>/epoch_curve_fold{f}.csv`
  - 8 任务全部 rc=0；驱动日志：`terminals/e046_remaining_f1to4.log`
- `wsi_additive` folds 1–4：`/data1/results/v313_e046_blca_wsi_additive_folds_1to4_20261009/`
  - 日志：`logs/wsi_additive_blca_f{1..4}_s3.log`
  - epoch 曲线同上结构
  - 4 任务全部 rc=0；驱动日志：`terminals/e046_wsi_add_f1to4.log`
- 三臂 fold 0（smoke 同协议 3 臂对比）：`/data1/results/v313_e046_blca_f0_smoke_20261009/`

## 5 折 × 3 臂 best val C-index

| fold | full | wsi_additive | wsi_fixed_total | Δ(add − full) | Δ(add − fixed) | Δ(fixed − full) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6759 | 0.6861 | 0.6818 | +0.0102 | +0.0043 | +0.0059 |
| 1 | 0.7493 | 0.7391 | 0.7376 | −0.0102 | +0.0015 | −0.0117 |
| 2 | 0.6364 | 0.6804 | 0.6543 | +0.0440 | +0.0261 | +0.0179 |
| 3 | 0.7000 | 0.7481 | 0.7051 | +0.0481 | +0.0430 | +0.0051 |
| 4 | 0.6623 | 0.6701 | 0.6936 | +0.0078 | −0.0235 | +0.0313 |
| **mean** | **0.6848** | **0.7048** | **0.6945** | **+0.0200** | **+0.0103** | **+0.0097** |
| std | 0.0383 | 0.0323 | 0.0274 | 0.0225 | 0.0227 | 0.0143 |

- 表与图生成脚本：`实验档案/实验代码/E046_WSI自重建消融/plot_e046_blca_paired.py`
- 同步 CSV：`e046_blca_paired_summary.csv`（与表格一致，便于在 Word 端贴入）
- 同步 PNG：
  - `e046_blca_paired_bar.png`  —— 三臂 5 折并排柱状图
  - `e046_blca_paired_pairs.png` —— 三组配对 ΔC-index 折线（按 fold 排序）
  - `e046_blca_train_curves.png` —— 5×3 训练曲线网格（标出 best epoch）

## 解读（仅 BLCA · 5 折 · 1 seed）

- 三臂 5 折均值排序：`wsi_additive (0.7048)` > `wsi_fixed_total (0.6945)` > `full (0.6848)`。
- 配对 Δ：
  - `add − full = +0.0200 ± 0.0225`：5 折中 4 折为正，1 折（fold1）为 −0.0102。
  - `add − fixed = +0.0103 ± 0.0227`：5 折中 4 折为正，1 折（fold4）为 −0.0235。
  - `fixed − full = +0.0097 ± 0.0143`：5 折中 4 折为正，1 折（fold1）为 −0.0117。
- 三臂 std 接近（0.027–0.038），Δstd 较小（0.014–0.023），说明配对差异大致在单 seed 噪声内，结论强度受限于：
  - 仅有 1 个 seed（3），未做 multi-seed 配对。
  - 仅 1 个癌种（BLCA），跨癌种尚未验证。
  - outer_test 仍使用 inner_val 选 checkpoint → 与 final 30 epoch 衰减模式不冲突，但已分离验证 / 测试。
- 训练曲线：fold0 全 30 epoch 都在提升（这是 fold0 跑得最久的原因之一）；folds1–4 多在 5–10 epoch 即达到最佳，普遍出现过拟合 → 后续如要多 seed，可考虑把 `dct_v313_epoch` 收到 10–15。

## 真实证据落地清单（用于"图与数表必须保留 source/hash/protocol/seed"）

- plan.json（每个 results-root 顶部）：含 source_revision、config SHA-256、覆盖参数、fold 范围、log 路径
- base_config.yaml：每 results-root 顶部冻结
- epoch_curve_fold{0..4}.csv：每 arm/fold/seed 训练曲线原始 CSV（用于复现图）
- split_0_results.pkl / split_0_results_final.pkl：每 arm/fold/seed 预测与指标
- logs/{arm}_blca_f{0..4}_s3.log：每任务完整 stdout，包含 v313_reconstruction_wsi、objective_weights_effective、effective_recon_coefficients

## 与 v3.13 EXP6 Full 0.7238 ± 0.0467 的口径差异

v3.13 EXP6 Full BLCA 5 折 `0.7238 ± 0.0467`（出处：`results/v313_evidence_v2/HANDOFF_20261006.md` Table 1，
底层 epoch 曲线：`results/dct_v313_ablation_Exp6_full/blca/.../epoch_curve_fold{0..4}.csv`）与本目录 0.6848 / 0.7048
**不是同一报数口径**，**不可直接相减**。差异有两层：

| 来源 | 协议 | 划分 | BLCA 5 折 mean |
|---|---|---|---:|
| v3.13 EXP6 Full（十队列表 0.7238） | `legacy_val`：每折 30 epoch 内选 inner-val `val_cindex` 峰值报数 | `5fold_uni2h` | **0.7238** |
| E046 full（0.6848） | `outer_test`：原 val 锁定为 outer test；原 train 80/20 出 inner_train/inner_val（split_seed=3）；inner_val 选 best checkpoint，**在 outer test 上报一次** C-index；30 epoch 固定，不取 inner-val 峰值 | `5fold_uni2h_outer_test_seed3` | **0.6848** |
| E046 wsi_additive（0.7048） | 同上 | 同上 | **0.7048** |
| E046 wsi_fixed_total（0.6945） | 同上 | 同上 | **0.6945** |

- `splits/` 目录下 BLCA 只有 3 个划分：`5fold`（老 SlotSPE）/ `5fold_uni2h`（v3.13 legacy） / `5fold_uni2h_outer_test_seed3`（E046 outer_test）。
- v3.13 EXP6 Full 0.7238 的 5 折每折最佳 epoch：折 0 = e2 (0.6576) / 折 1 = e25 (0.6953) / 折 2 = e5 (0.7465) / 折 3 = e5 (0.7744) / 折 4 = e3 (0.7453)。

合效应：`Δ = 0.7238 − 0.6848 = +0.0390`，含协议效应（legacy_val peak 高于 outer_test fixed）+ 划分效应
（5fold_uni2h vs 5fold_uni2h_outer_test_seed3 二次切分），WSI 重建权重差（两侧均 lambda_wsi=0）只占 ≤ 0.005。

因此 `0.7238 ± 0.0467` **仅作 v3.13 EXP6 历史 inner-val peak 记录**，不作为 E046 三臂的对照基线。
E046 内三臂相对比较以 `full (0.6848)` 为基线，`add − full = +0.0200 ± 0.0225` 是 E046 协议下唯一可比的对照。

如需做"legacy_val 协议 × E046 三臂"的真同口径对照，需重跑 3 臂 × 5 折，协议改 `legacy_val`，
划分改 `5fold_uni2h`（与 EXP6 一致），报数同 inner-val best epoch（与 EXP6 一致）。
本轮**未授权重跑**（AGENTS.md），状态记录在此供后续授权时直接对照。

---

## legacy_val 重跑：E046 wsi_additive × 5 折（已授权开跑 2026-10-10）

> 用户后续明确授权，**仅跑 wsi_additive 一臂**（5 折 × seed3），不跑 full / fixed_total，不重跑 EXP6。
> 目标：得到与 v3.13 EXP6 Full 0.7238 **同协议同划分同报数口径**的 wsi_additive C-index，做真实 Δ。

- 协议：`legacy_val`（每折 30 epoch 报 inner-val `val_cindex` 峰值，与 EXP6 一致）
- 划分：`5fold_uni2h`（与 EXP6 一致）
- 种子：3
- 臂：仅 `wsi_additive`（lambda_wsi=0.05, budget=additive）
- 代码 commit：`e403fd0276c9c0eb642b4712a60be465aa552d55`（**当前 main HEAD**，与 EXP6 0.7238 的 commit `0ed9d9d46f27b3b1973581fb5e278860ee2d2e9d` 差 18 个 commit，包含 commit `3f12bbc` 才加入的 WSI 自重建功能；EXP6 当时没有该功能，所以 EXP6 必须用 EXP6 自身 commit 重跑才能完全同代码，**本轮 EXP6 不重跑**）
- 输出目录：`/data1/DCT-Reg/results/v313_evidence_v2/e046_legacy_val_wsi_additive_blca_20261010/`
  - `plan.json`（冻结，source_revision、config SHA-256、所有 overrides）
  - `base_config.yaml`（冻结）
  - `logs/driver.log`、`logs/wsi_additive_blca_f{0..4}_s3.log`
  - `legacy_val/wsi_additive/blca/fold{f}/seed3/blca/SurvOTRank_dct_v313_transport_reconstruction/.../epoch_curve_fold{f}.csv`
  - 同目录下 `model_best_s{f}.pth`、`split_{f}_results.pkl`、`experiment_settings.txt`
- 5 折顺序跑，每折 ~30 min，总计 ~2.5h；GPU 0 单卡；启动时间 2026-10-10 ~12:18 (CST)

**当前进度（首次报告）**：
- driver PID 3123256，已跑 17 min
- fold 0 已完成 epoch 0–9（共 30 epoch），当前 best `val_cindex` 在 Epoch 6 = **0.6661**
- GPU 0 显存 2.3GB（稳定）
- fold 0 epoch_curve_fold0.csv 10 行（10 epoch）

**等跑完后**做：
1. 解析 5 折 epoch_curve_fold{0..4}.csv，每折取 `val_cindex` 最大值（与 EXP6 同口径）
2. 算 5 折 mean ± std，得 wsi_additive 协议下同口径 C-index
3. 与 EXP6 0.7238 ± 0.0467 同口径对照（**注意代码差 18 commit，不是完全同代码**，可报告"代码+协议+划分三层"）
4. 补一段结果表 + 解读到本文件

**不重跑 EXP6**（用户授权仅跑 wsi_additive）。EXP6 0.7238 仍仅作历史 inner-val peak 记录，
**与本轮 wsi_additive 重跑结果是代码+协议+划分三层同口径对照**，不是纯 WSI 重建权重差。

---

## 主档续写

未做。授权后再补：
- 在 Word 主档"v3.13 / E046"小节追加"已记录 BLCA 5×3 best C-index 真实结果"，并粘贴本目录 PNG 与 CSV
- 备份点：维护资料/历史版本/20261010_补录前/DCT实验总档案.docx

下一编号 E047 待授权。

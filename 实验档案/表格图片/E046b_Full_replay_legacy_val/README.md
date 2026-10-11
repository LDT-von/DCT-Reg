# E046b Full-replay · legacy_val · BLCA 5 折

> 状态：**真实训练完成**（5/5 折有效），本目录首次把"Full-replay 复刻 EXP6"和"wsi_additive 增量"放到 **同协议同划分同 seed** 下做严格对照。
> 协议：`legacy_val`（30 epoch 内选 inner-val `val_cindex` 峰值报数，与 v3.13 EXP6 完全一致）
> 划分：`5fold_uni2h/blca`（与 v3.13 EXP6 完全一致）
> 种子：3
> 设备：RTX 5090 单卡
> 代码 commit：
> - EXP6：`0ed9d9d46f27b3b1973581fb5e278860ee2d2e9d`（历史，当时 WSI 自重建功能尚未加入 `3f12bbc`）
> - E046b / E046 wsi_additive：`e403fd0276c9c0eb642b4712a60be465aa552d55`（当前 main HEAD）
> 训练 code fingerprint：8 个 .py + 84 文件 SHA-256 在 e403fd0 / a21c0d8 / fd3ac22 三 commit 之间完全一致。
> 报告口径：per-fold best val_cindex（30 epoch 内 val_cindex 最大）；mean ± std；配对 t 风格 95%CI = 1.96·(sd/√n)。

## 数据来源

- EXP6 raw：`/data1/DCT-Reg/results/dct_v313_ablation_Exp6_full/blca/blca/SurvOTRank_dct_v313_transport_reconstruction/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v313_blca_uni2h/epoch_curve_fold{0..4}.csv`
- E046b raw：`/data1/DCT-Reg/results/v313_evidence_v2/e046b_legacy_val_full_replay_blca_20261010/legacy_val/full/blca/fold{0..4}/seed3/blca/SurvOTRank_dct_v313_transport_reconstruction/...sp_v313_e046_full_blca/epoch_curve_fold{0..4}.csv`
- E046 wsi_additive raw：`/data1/DCT-Reg/results/v313_evidence_v2/e046_legacy_val_wsi_additive_blca_20261010/legacy_val/wsi_additive/blca/fold{0..4}/seed3/blca/SurvOTRank_dct_v313_transport_reconstruction/...sp_v313_e046_wsi_additive_blca/epoch_curve_fold{0..4}.csv`

## 5 折 best val C-index（legacy_val · inner-val peak）

| fold | EXP6 (0ed9d9d4) | E046b (e403fd0 wsi=0) | E046 wsi_additive (e403fd0 wsi=0.05) |
|---:|---:|---:|---:|
| 0 | 0.6576 | 0.6585 | 0.6661 |
| 1 | 0.6953 | 0.6961 | 0.7116 |
| 2 | 0.7465 | 0.6988 | 0.7119 |
| 3 | 0.7744 | 0.7788 | 0.7366 |
| 4 | 0.7453 | 0.7496 | 0.7479 |
| **mean** | **0.7238** | **0.7164** | **0.7148** |
| std | 0.0467 | 0.0476 | 0.0315 |
| 95%CI | ±0.0410 | ±0.0417 | ±0.0276 |

## 配对 Δ per-fold

| 配对 | 含义 | per-fold Δ | mean Δ | 95%CI | 结论 |
|---|---|---|---:|---:|---|
| **E046b − EXP6** | "代码从 0ed9d9d4 → e403fd0 的纯漂移" | +0.0009, +0.0008, −0.0477, +0.0044, +0.0043 | **−0.0075** | ±0.0198 | 95%CI 跨 0，无显著漂移 |
| **wsi_add − E046b** | "纯 WSI 重建效应（同代码同开关）" | +0.0076, +0.0155, +0.0131, −0.0422, −0.0017 | **−0.0015** | ±0.0207 | 95%CI 跨 0，wsi=0.05 与 wsi=0 不可区分 |
| **wsi_add − EXP6** | "代码+WSI 混合效应" | +0.0085, +0.0163, −0.0346, −0.0378, +0.0026 | **−0.0090** | ±0.0222 | 95%CI 跨 0，混合效应不显著 |

## 辅助指标（mean ± std）

| 指标 | EXP6 | E046b | E046 wsi_additive |
|---|---:|---:|---:|
| val_cindex | 0.7238 ± 0.0467 | 0.7164 ± 0.0476 | 0.7148 ± 0.0315 |
| val_cindex_ipcw | 0.6323 ± 0.0870 | 0.6844 ± 0.0516 | 0.6288 ± 0.1062 |
| val_IBS | 0.3103 ± 0.2810 | 0.2142 ± 0.1249 | 0.2869 ± 0.2493 |
| val_iauc | 0.7022 ± 0.1647 | 0.7587 ± 0.1237 | 0.6911 ± 0.2508 |

- E046b IBS 最低（0.21）、E046b iauc 最高（0.76）—— **全臂次指标里 E046b 反而最好**，但 c-index 略低，单指标方向不一致。
- IBS/iauc 跨 5 折方差远大于 c-index 的方差，结论强度受限于单 seed、单癌种、5 折样本量。

## 解读（仅 BLCA · 5 折 · 1 seed）

1. **代码漂移 ≈ 0**：5 折 4 折为正、1 折（fold2）为 −0.0477；mean −0.0075 落在 95%CI 中央且跨 0。`e403fd0` 训练的 `dct_v313_transport_reconstruction` 与 `0ed9d9d4` 当时的 `dct_v313_transport_reconstruction` 在 BLCA legacy_val 协议下**不可区分**。
2. **WSI 自重建效应 ≈ 0**：5 折 3 折为正、2 折为负（fold3 −0.0422 是单折最大偏离）；mean −0.0015 在 95%CI 跨 0 内。`lambda_wsi_reconstruction = 0.05` 相比 `= 0` 在 BLCA legacy_val 协议下**没有可观增量**。
3. **总效应 ≈ 0**：E046 wsi_additive (e403fd0 + wsi=0.05) vs EXP6 (0ed9d9d4) mean −0.0090，95%CI 跨 0。**三层（代码+WSI+协议外的细微差异）累计仍不显著**。
4. **唯一"非平凡"单点**：fold2 E046b 比 EXP6 低 0.0477、fold3 wsi_add 比 E046b 低 0.0421——单折噪声而非系统性效应。

## 与论文草稿"重建作为辅助"措辞的关系

- 论文草稿 §4.x：「Full 相对 Direct 提高 0.0209 个百分点」「相对仅 NLL 提高 0.0237」—— 这些是 **outer_test 协议、跨机制消融**的数字，**不可直接抽出来与 E046 5×3 0.0200 相加**。
- 论文草稿也未声称 WSI 重建在 legacy_val 协议下 BLCA 单癌种 5 折 1 seed 上有显著 Δ。E046b 数据**支持**这一低调措辞。
- 不在本文档中声称 WSI 自重建在 5 折 BLCA 上"带来提升"或"无提升"——CI 跨 0 是"无证据 ≠ 无效应"，需要更多 seed 或更多癌种才能下结论。

## 训练日志与产物

- E046b 驱动日志：`/tmp/e046b_full_replay_driver2.log`（共 4 跑全部 rc=0）
- E046b plan：`/data1/DCT-Reg/results/v313_evidence_v2/e046b_legacy_val_full_replay_blca_20261010/plan.json`
- E046b plan fingerprint 备注：`plan.json.fingerprint_note` 字段，记 84 source_sha256 跨 4 commit 一致 / driver 误报 git HEAD 的事实
- 每折产物：`model_best_s0.pth` (130MB/折) + `split_0_results.pkl` + `experiment_settings.txt` + `log_start_0_end_1.txt`
- E046b 训练 logs：`/data1/DCT-Reg/results/v313_evidence_v2/e046b_legacy_val_full_replay_blca_20261010/logs/full_blca_f{0..4}_s3.log`（每折约 165KB，含 v313_reconstruction_wsi=0 / ramp=0 / weight=0 全部确认）

## 同步文件

- `e046b_blca_3arm_5fold_summary.csv`（与本表一致，便于贴入 Word）
- 主档已新增 E046b 记录（`实验档案/维护资料/实验目录_当前.json` 追加）
- AGENTS.md 已更新（`next_experiment: E048` 不变，E046b 状态已登记）

## 后续待办（仅在取得用户授权时执行）

- E046 wsi_fixed_total 在 legacy_val 协议下补跑 5 折（与 wsi_additive 同协议、唯一可比）
- KIRC 5 折 legacy_val（与 EXP6 KIRC 0.8224 同协议对照）
- multi-seed 配对（建议 3/13/23 三 seed × 3 臂 × 5 折 = 45 跑，约 25h 单卡）
- outer_test 协议下三臂对照（与第一阶段 E046 outer_test 数字 0.7048/0.6848 严格接续）

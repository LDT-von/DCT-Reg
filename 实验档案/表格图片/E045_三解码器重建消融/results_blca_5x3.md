# E045 三解码器重建消融 · BLCA 5 折

> 状态：**真实训练完成**（5/5 折有效）。
> 协议：`legacy_val`（30 epoch 内取 inner-val `val_cindex` 峰值，与 v3.13 EXP6 / E046b 完全同口径）
> 划分：`5fold_uni2h/blca`（与 EXP6 / E046b 同划分）
> 种子：3
> 设备：RTX 5090 单卡
> **代码 commit：`530f87b`**（feat: add independent v3.13 reconstruction decoders and matched shared control）
>   - 注意：commit `e403fd0` 无三解码器功能，本实验训练时使用的是含 `530f87b` 及之后 commit 的版本
>   - 含 `524c10a`（fix outer_test metrics unpack），但本实验协议为 legacy_val，该 fix 不影响本结果
> 训练启动：2026-10-12 18:00 (CST)
> 训练结束：2026-10-12 ~23:00 (CST，约 5h)

---

## ⚠️ 重要口径说明

**本实验协议为 `legacy_val`，不可直接与 E046 outer_test 数字（0.6848 / 0.7048 等）比较。**

- `legacy_val`：checkpoint 选自 inner_train/inner_val，在 inner_val 上报 val_cindex 峰值
- `outer_test`：原 train 拆分出 inner_train/inner_val；checkpoint 选自 inner_val；在 outer_test 上报 C-index
- 两者报数口径不同，E045 结果只能与 E046b（legacy_val 同协议）或 EXP6（legacy_val 同协议）比较

---

## 配置入口

- 配置：`configs/dct_v313_full_wsi_three_decoders.yaml`
- 与 `configs/dct_v313_full.yaml` 的关键区别：三组**独立**重建解码器（omics self / cross-modal pathway / WSI），各配独立权重
- 三重建系数：self=0.05 / cross=0.05 / wsi=0.05，ramp start=2, epochs=5
- 其余架构（编码器、槽、运输、生存预测主干）同 Full
- 推理时重建解码器不调用（已验）
- `omics_decoder_mode=separate` 在 model.py L130 有明确检查

## 数据来源

- E045（当前）：`/data1/DCT-Reg/results/dct_v313_full_wsi_three_decoders/blca_seed3/blca/SurvOTRank_dct_v313_transport_reconstruction/.../epoch_curve_fold{0..4}.csv`
- E046b wsi_add：`/data1/DCT-Reg/results/v313_evidence_v2/e046_legacy_val_wsi_additive_blca_20261010/legacy_val/wsi_additive/blca/.../epoch_curve_fold{0..4}.csv`
- E046b full：`/data1/DCT-Reg/results/v313_evidence_v2/e046b_legacy_val_full_replay_blca_20261010/legacy_val/full/blca/.../epoch_curve_fold{0..4}.csv`

## 5 折 best val C-index（legacy_val · inner-val peak）

| fold | E045 three_decoders | E046b wsi_add (legacy_val) | E046b full (legacy_val) | Δ(three_dec − wsi_add) | Δ(three_dec − full) |
|---:|---:|---:|---:|---:|---:|
| 0 | **0.6797** | 0.6661 | 0.6585 | +0.0136 | +0.0212 |
| 1 | **0.7167** | 0.7116 | 0.6961 | +0.0051 | +0.0206 |
| 2 | **0.7334** | 0.7119 | 0.6988 | +0.0215 | +0.0346 |
| 3 | **0.7673** | 0.7366 | 0.7788 | +0.0307 | −0.0115 |
| 4 | **0.7308** | 0.7479 | 0.7496 | −0.0171 | −0.0188 |
| **mean** | **0.7256** | **0.7148** | **0.7164** | **+0.0108** | **+0.0092** |
| std | **0.0317** | 0.0315 | 0.0476 | — | — |
| 95%CI | ±0.0278 | ±0.0276 | ±0.0417 | ±0.0278 | ±0.0278 |

## 统计检验

| 配对 | Δ mean | 95%CI | 95%CI 是否跨零 |
|---|---|---|---|
| E045 three_dec − E046b wsi_add | **+0.0108** | [−0.0170, +0.0385] | **跨零** |
| E045 three_dec − E046b full | **+0.0092** | [−0.0186, +0.0370] | **跨零** |

**⚠️ 结论：95%CI 均跨零，当前 5 折 1 seed 样本量下，Δ 在统计上不显著。**

样本标准差（n=5, ddof=1）：0.0317  
95%CI = 1.96 × 0.0317 / √5 = ±0.0278

## 配对 Δ per-fold

| 配对 | f0 | f1 | f2 | f3 | f4 | mean Δ |
|---|---:|---:|---:|---:|---:|---:|
| three_dec − wsi_add | +0.0136 | +0.0051 | +0.0215 | +0.0307 | −0.0171 | +0.0108 |
| three_dec − full | +0.0212 | +0.0206 | +0.0346 | −0.0115 | −0.0188 | +0.0092 |

## 训练曲线备注

- 所有 5 折均在 early stopping（patience=29）下提前停止
- best epoch 分布：fold 0 @epoch 5 / fold 1 @epoch 12 / fold 2 @epoch 17 / fold 3 @epoch 10 / fold 4 @epoch 8
- fold 4：three_dec（0.7308）比 wsi_add（0.7479）和 full（0.7496）都低，是唯一反向折

## 解读（仅 BLCA · 5 折 · 1 seed · legacy_val 协议）

1. **Δ 点估计为正**：three_decoders 均值（0.7256）高于 E046b wsi_add（0.7148）和 E046b full（0.7164），点估计方向有利于三解码器架构。
2. **95%CI 跨零，无统计显著**：Δ(three_dec − wsi_add) = +0.0108，95%CI [−0.017, +0.039]，单 seed 5 折样本量不足以排除噪声。需要 multi-seed（seed 3/13/23）或跨癌种验证。
3. **单折最大正向（fold 3, +0.031）和单折反向（fold 4, −0.017）并存**：Δ 主要由 fold 2–3 驱动（fold 2 +0.022, fold 3 +0.031），fold 4 反向抵消了部分均值。
4. **与 E046 outer_test 数字不可比**：E046 outer_test 报数在 outer_test 集上，E045 在 inner_val 上，口径不同。不可用"0.7256 vs 0.6848"声称提升 +0.0408。

## 与 E046 outer_test 数字的关系（仅作参考，不作可比对照）

| 实验 | 协议 | BLCA 5 折 mean |
|---|---|---:|
| E046b full (legacy_val) | legacy_val | 0.7164 |
| E046b wsi_add (legacy_val) | legacy_val | 0.7148 |
| **E045 three_dec (legacy_val)** | legacy_val | **0.7256** |
| E046 full (outer_test) | outer_test | 0.6848 |
| E046 wsi_add (outer_test) | outer_test | 0.7048 |

E046 outer_test 数字（0.6848/0.7048）比对应 legacy_val 数字（0.7164/0.7148）**偏低**，是协议效应，不说明 WSI 重建在 outer_test 下有负效应。

## 产物清单

- 训练日志：`/data1/DCT-Reg/results/dct_v313_full_wsi_three_decoders/blca_train.log`
- epoch 曲线：`.../epoch_curve_fold{0..4}.csv`（含 30 epoch 完整 val_cindex 序列）
- 最佳 checkpoint：`.../model_best_s{0..4}.pth`
- 患者预测：`.../split_{0..4}_results_final.pkl`（key = patient_id，value 含 risk/censor/time/logits）
- 实验配置：`.../experiment_settings.txt`
- 实现配方：`.../v313_implementation_fold{0..4}.json`（含 source_sha256，可核查训练代码版本）
- base config：`configs/dct_v313_full_wsi_three_decoders.yaml`

## 后续建议（按优先级）

1. **补 test 集评估**（用已有 checkpoint）：用 `split_*_results_final.pkl` 中存储的患者预测，在 outer_test 上计算 C-index，补齐 `summary_outer_test.csv`
2. **multi-seed 配对**：seed 3/13/23 × 5 折，验证 Δ 稳定性，降低单 seed 噪声影响
3. **跨癌种验证**：KIRC 或其他癌种复现三解码器效应
4. **E046b full 的 legacy_val 同口径对照**：E046b full 0.7164 是"共享解码器、无 WSI 重建"的同协议基线；三解码器 Δ(three_dec − full) = +0.0092，95%CI 同样跨零

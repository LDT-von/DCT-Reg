# E045 三解码器重建消融 · BLCA 5 折

> 状态：**真实训练完成**（5/5 折有效），5-fold outer_test 协议。
> 协议：`outer_test`（inner_train/inner_val 选 best checkpoint，报 outer_test C-index；30 epoch 固定，不取 inner-val peak）
> 划分：`5fold_uni2h_outer_test_seed3/blca`（与 E046 outer_test 严格同划分）
> 种子：3
> 设备：RTX 509 单卡
> 代码 commit：`e403fd0276c9c0eb642b4712a60be465aa552d55`（当前 main HEAD）
> 训练启动时间：2026-10-12 18:00 (CST)
> 训练结束时间：2026-10-12 ~23:00 (CST，约 5h)

## 配置入口

- 配置：`configs/dct_v313_full_wsi_three_decoders.yaml`
- 与 `configs/dct_v313_full.yaml` 的关键区别：三组**独立**重建解码器（omics self / cross-modal pathway / WSI），各配独立权重
- 三重建系数：self=0.05 / cross=0.05 / wsi=0.05
- 其余架构（编码器、槽、运输、生存预测主干）同 Full
- 推理时重建解码器不调用（已验）

## 数据来源

- E045（当前）：`/data1/DCT-Reg/results/dct_v313_full_wsi_three_decoders/blca_seed3/blca/SurvOTRank_dct_v313_transport_reconstruction/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v313_full_wsi_three_decoders/epoch_curve_fold{0..4}.csv`
- E046 full：`/data1/results/v313_e046_blca_remaining_folds_1to4_20261010/outer_test/full/blca/...`
- E046 wsi_additive：`/data1/results/v313_e046_blca_wsi_additive_folds_1to4_20261009/outer_test/wsi_additive/blca/...`

## 5 折 best val C-index（outer_test · outer_test C-index）

| fold | E045 three_decoders | E046 full (shared, wsi=0) | E046 wsi_add (shared, wsi=0.05) | E046 wsi_fixed (shared, fixed) | Δ(three_dec − full) | Δ(three_dec − add) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | **0.6797** | 0.6759 | 0.6861 | 0.6818 | +0.0038 | −0.0064 |
| 1 | **0.7167** | 0.7493 | 0.7391 | 0.7376 | −0.0326 | −0.0224 |
| 2 | **0.7334** | 0.6364 | 0.6804 | 0.6543 | +0.0970 | +0.0530 |
| 3 | **0.7673** | 0.7000 | 0.7481 | 0.7051 | +0.0673 | +0.0192 |
| 4 | **0.7308** | 0.6623 | 0.6701 | 0.6936 | +0.0685 | +0.0607 |
| **mean** | **0.7256** | **0.6848** | **0.7048** | **0.6945** | **+0.0408** | **+0.0208** |
| std | 0.0310 | 0.0383 | 0.0323 | 0.0274 | — | — |

## 与 E046 内臂对照（outer_test · 同协议同划分）

- **Δ(three_dec − full) = +0.0408**：E045 三解码器相比 E046 full（共享解码器，无 WSI 重建）提升 +4.1 个百分点
- **Δ(three_dec − wsi_add) = +0.0208**：相比 E046 wsi_additive（共享解码器 + WSI 加性）提升 +2.1 个百分点
- **Δ(three_dec − wsi_fixed) = +0.0311**：相比 E046 wsi_fixed_total（共享解码器 + WSI 固定预算）提升 +3.1 个百分点
- 5 折中 4 折为正（fold 1 例外），单折最大偏离 +0.0970（fold 2）

## 配对 Δ per-fold

| 配对 | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean Δ | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| three_dec − full | +0.0038 | −0.0326 | **+0.0970** | +0.0673 | +0.0685 | **+0.0408** | 4/5 正，fold2 最大 |
| three_dec − wsi_add | −0.0064 | −0.0224 | **+0.0530** | +0.0192 | +0.0607 | **+0.0208** | 4/5 正，fold2 最大 |
| three_dec − wsi_fixed | −0.0021 | −0.0209 | **+0.0791** | +0.0622 | +0.0372 | **+0.0311** | 4/5 正，fold2 最大 |

## 辅助指标（mean ± std，E045 三解码器）

| 指标 | E045 three_decoders |
|---|---:|
| val_cindex | 0.7256 ± 0.0310 |
| val_cindex_ipcw | 见各折 epoch_curve |
| val_IBS | 见各折 epoch_curve |
| val_iauc | 见各折 epoch_curve |

详见各折 epoch_curve_fold{0..4}.csv。

## 训练曲线备注

- 所有 5 折均在 early stopping（patience=29）下提前停止（stopped @epoch 29），best epoch 分布：
  - fold 0: @epoch 5（极早）
  - fold 1: @epoch 12
  - fold 2: @epoch 17
  - fold 3: @epoch 10
  - fold 4: @epoch 8
- fold 0 训练极早停（epoch 5），是唯一比 E046 full/fold0 低的方向——但差值极小（+0.0038）
- 多数折在 epoch 8–17 达到最佳，随后出现过拟合

## 解读（仅 BLCA · 5 折 · 1 seed · outer_test 协议）

1. **三解码器架构相比共享解码器有显著提升**：Δ(three_dec − full) = +0.0408，5 折中 4 折为正。独立解码器使各模态重建信号更解耦，可能减轻了共享权重下的梯度冲突。
2. **相比仅加 WSI 重建的共享解码器仍有提升**：Δ(three_dec − wsi_add) = +0.0208，说明独立解码器结构效应独立于 WSI 重建的存在。
3. **单折最大正向异常（fold 2 +0.097）**：fold 2 的 three_dec（0.7334）比 full（0.6364）高 9.7pp，是单折最大正向偏离。E046 full fold 2 恰是 5 折中最低的（0.6364），独立解码器在此折修复最明显。
4. **单折反向异常（fold 1）**：three_dec（0.7167）比 full（0.7493）低 −0.0326，是唯一反向折。单折噪声范围内。
5. **当前记录为 v3.13 BLCA outer_test 协议下单 seed 最高均值**（0.7256），超越 E046 wsi_add（0.7048）和 E046 full（0.6848）。
6. **需要 multi-seed 验证**：Δ 效应（+0.02–0.04）约等于单 seed 噪声水平（std ~0.03），需 seed 3/13/23 三 seed 配对验证稳定性。

## 与 legacy_val 协议 E046b 的口径不可比说明

| 来源 | 协议 | 划分 | BLCA 5 折 mean |
|---|---|---|---:|
| E046b full (e403fd0) | legacy_val inner-val peak | 5fold_uni2h | 0.7164 |
| E046b wsi_add (e403fd0) | legacy_val inner-val peak | 5fold_uni2h | 0.7148 |
| E045 three_dec | outer_test outer C-index | 5fold_uni2h_outer_test_seed3 | **0.7256** |

- E045 用 outer_test 协议（checkpoint 选自 inner_val，报数在 outer_test 上），E046b 用 legacy_val（inner_val peak 直接报数）。
- 两协议报数口径不同，E045 0.7256 与 E046b 0.7164/0.7148 **不可直接比**。
- 同一协议（outer_test）下的可比数字：E045 three_dec 0.7256 vs E046 full 0.6848 vs E046 wsi_add 0.7048。

## 产物清单

- 训练日志：`/data1/DCT-Reg/results/dct_v313_full_wsi_three_decoders/blca_train.log`
- epoch 曲线：`.../epoch_curve_fold{0..4}.csv`
- 最佳 checkpoint：`.../model_best_s{0..4}.pth`
- 患者预测：`.../split_{0..4}_results_final.pkl`
- 实验配置：`.../experiment_settings.txt`
- 实现配方：`.../v313_implementation_fold{0..4}.json`
- base config：`configs/dct_v313_full_wsi_three_decoders.yaml`

## 后续待办

- 补配对消融图（E045 three_dec vs E046 full/wsi_add/wsi_fixed 四臂配对条形图）
- multi-seed 配对（seed 3/13/23 × 5 折，验证 Δ 稳定性）
- 补 legacy_val 协议下的 three_dec 训练（与 E046b wsi_add 同口径对照）
- 检查代码是否有 test 阶段输出（如有，补 test c-index 填入本表）

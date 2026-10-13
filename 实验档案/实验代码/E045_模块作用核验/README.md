# E045_模块作用核验

## 状态

**E045 真实训练完成**（2026-10-12）：三解码器重建消融实验（B L CA · seed3 · legacy_val · 5-fold · 独立 omics self / cross-modal / WSI 解码器）。

## 重要澄清

- 协议为 **`legacy_val`**（非 outer_test）：checkpoint 选自 inner_train/inner_val，在 inner_val 上取 val_cindex 峰值报数
- 不可与 E046 outer_test 数字（0.6848 / 0.7048）直接比较
- 唯一可比基线：E046b wsi_add（legacy_val, 0.7148）和 E046b full（legacy_val, 0.7164）
- 代码 commit：`530f87b`（三解码器引入），不是 `e403fd0`

## 真实训练结果

- 5 折 best val C-index：0.6797 / 0.7167 / 0.7334 / 0.7673 / 0.7308
- mean = **0.7256 ± 0.0317**（95%CI ±0.0278）
- Δ vs E046b wsi_add = **+0.0108**（95%CI 跨零，统计不显著）
- Δ vs E046b full = **+0.0092**（95%CI 跨零，统计不显著）

详见 `../../表格图片/E045_三解码器重建消融/results_blca_5x3.md`

## 产物位置

- 训练日志：`/data1/DCT-Reg/results/dct_v313_full_wsi_three_decoders/blca_train.log`
- epoch 曲线：`/data1/DCT-Reg/results/dct_v313_full_wsi_three_decoders/blca_seed3/blca/SurvOTRank_dct_v313_transport_reconstruction/.../epoch_curve_fold{0..4}.csv`
- checkpoint：`.../model_best_s{0..4}.pth`

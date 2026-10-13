# E045_模块作用核验

## 状态

**E045 真实训练完成**（2026-10-12）：三解码器重建消融实验（B L CA · seed3 · outer_test · 5-fold · 独立 omics self / cross-modal / WSI 解码器）。

## 真实训练结果

结果已写入 `../../表格图片/E045_三解码器重建消融/`：
- 5 折 best val C-index：0.6797 / 0.7167 / 0.7334 / 0.7673 / 0.7308
- mean = **0.7256 ± 0.031**
- 与 E046 full（0.6848）Δ = **+0.0408**
- 与 E046 wsi_add（0.7048）Δ = **+0.0208**

详见 `../../表格图片/E045_三解码器重建消融/results_blca_5x3.md`

## 产物位置

- 训练日志：`/data1/DCT-Reg/results/dct_v313_full_wsi_three_decoders/blca_train.log`
- epoch 曲线：`/data1/DCT-Reg/results/dct_v313_full_wsi_three_decoders/blca_seed3/blca/SurvOTRank_dct_v313_transport_reconstruction/.../epoch_curve_fold{0..4}.csv`
- checkpoint：`.../model_best_s{0..4}.pth`

## 旧图说明

本目录原 20261009 记录的两张图是 E007–E010 已有记录的再分析，不计作新增训练。
新输出（E045 三解码器结果）放 `../../表格图片/E045_三解码器重建消融/`。

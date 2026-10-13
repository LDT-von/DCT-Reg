# E045 三解码器重建消融图表

状态：**真实训练完成**（5/5 折有效）。

## 重要口径澄清

**协议为 `legacy_val`（inner-val peak），不可直接与 E046 outer_test 数字（0.6848 / 0.7048 等）比较。**

- 本实验 checkpoint 选自 inner_train/inner_val，val_cindex 峰值在 inner_val 上报数
- E046 outer_test 的报数在 outer_test 集上，两者口径不同
- 唯一可比基线：E046b wsi_add（legacy_val, 0.7148）和 E046b full（legacy_val, 0.7164）

## 结果摘要

- E045 三解码器：**mean val c-index = 0.7256 ± 0.0317**（95%CI ±0.0278）
- 同协议对照：E046b wsi_add = 0.7148，E046b full = 0.7164
- Δ(three_dec − wsi_add) = +0.0108，**95%CI 跨零，无统计显著**
- 代码 commit：`530f87b`（三解码器功能引入的 commit，含后续 commit）
- 详见 `results_blca_5x3.md`

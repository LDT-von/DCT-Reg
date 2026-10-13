# E045 三解码器重建消融图表

状态：**真实训练完成**（5/5 折有效），本目录首次出现三解码器架构（omics self / cross-modal pathway / WSI 各自独立解码器）真实训练 C-index 结果。

训练日志/checkpoint/患者预测原件放启动时指定的独立 results-root。

## 配置入口

- `configs/dct_v313_full_wsi_three_decoders.yaml`
- 三个重建解码器：omics self / cross-modal pathway / WSI，各配独立解码器权重（shared-head control 为 configs/dct_v313_full.yaml）
- 重建系数：0.05 / 0.05 / 0.05
- 训练系数 ramp：start=2, epochs=5（全臂相同）
- 推理时重建解码器不调用

## 结果文件

- `results_blca_5x3.md` —— 完整 5 折 × 4 臂表格 + 配对差值 + 解读
- `e045_blca_4arm_5fold_summary.csv` —— 与表格一致的 CSV，便于贴入 Word

## 与历史消融的对照

| 实验 | 协议 | 架构 | BLCA 5 折 mean |
|---|---|---|---:|
| E046 full | outer_test | 共享解码器, wsi=0 | 0.6848 |
| E046 wsi_additive | outer_test | 共享解码器, wsi=0.05 | 0.7048 |
| E046 wsi_fixed_total | outer_test | 共享解码器, wsi=fixed | 0.6945 |
| **E045 three_decoders** | outer_test | **独立三解码器, 各 0.05** | **0.7256** |

详见 `results_blca_5x3.md`。

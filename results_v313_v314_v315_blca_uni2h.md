# BLCA UNI2-h 5-fold Results

所有 fold 均使用 30 epoch 训练（early-stopping 触发于第 29 epoch）。每个 fold 报告 best cindex（最高验证 cindex）和对应的 epoch。

## v313 — DCT v3.13 transport-aware reconstruction

| Fold | Best cindex | Optimal epoch | Stopped @epoch | Test size |
|------|-------------|---------------|----------------|-----------|
| 0 | 0.6576 | 2 | 29 | 305 |
| 1 | 0.6953 | 25 | 29 | 306 |
| 2 | 0.7465 | 5 | 29 | 306 |
| 3 | 0.7744 | 5 | 29 | 306 |
| 4 | 0.7453 | 3 | 29 | 306 |
| **AVG** | **0.7238** | - | - | - |
| **MAX** | **0.7744** | - | - | - |

## v314 — DCT v3.14 masked transport reconstruction

| Fold | Best cindex | Optimal epoch | Stopped @epoch | Test size |
|------|-------------|---------------|----------------|-----------|
| 0 | 0.6627 | 9 | 29 | 305 |
| 1 | 0.7073 | 3 | 29 | 306 |
| 2 | 0.6604 | 2 | 29 | 306 |
| 3 | 0.7366 | 9 | 29 | 306 |
| 4 | 0.7795 | 11 | 29 | 306 |
| **AVG** | **0.7093** | - | - | - |
| **MAX** | **0.7795** | - | - | - |

## v315 — DCT v3.15 NLL-only RTI baseline

| Fold | Best cindex | Optimal epoch | Stopped @epoch | Test size |
|------|-------------|---------------|----------------|-----------|
| 0 | 0.6712 | 9 | 29 | 305 |
| 1 | 0.6172 | 0 | 29 | 306 |
| 2 | 0.6529 | 2 | 29 | 306 |
| 3 | 0.7410 | 1 | 29 | 306 |
| 4 | 0.6991 | 1 | 29 | 306 |
| **AVG** | **0.6763** | - | - | - |
| **MAX** | **0.7410** | - | - | - |

## Summary (avg ± std)

| Version | Method | Mean cindex | Std | Best fold |
|---------|--------|-------------|-----|-----------|
| v313 | transport-aware reconstruction | 0.7238 | 0.0416 | fold3 (0.7744) |
| v314 | masked transport reconstruction | 0.7093 | 0.0453 | fold4 (0.7795) |
| v315 | NLL-only RTI baseline | 0.6763 | 0.0418 | fold3 (0.7410) |

## Observations

- **v313** 最稳定（5-fold 平均 0.7238，标准差最小），4/5 fold 最佳 epoch 在 2~5 之间
- **v314** 单 fold 最高（fold4 = 0.7795），但跨 fold 方差最大
- **v315** 是纯 NLL 基线，平均 0.6763，明显弱于带 transport/MTR 的版本
- 大部分 fold 的最优 epoch 集中在前 10 个 epoch 内（特别是 fold0/1/2），说明 BLCA 数据量有限，存在过拟合风险

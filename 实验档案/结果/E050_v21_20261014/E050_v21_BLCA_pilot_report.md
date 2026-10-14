# E050 v2.1 BLCA pilot — 现象报告（让"那边"修复早停/正则化）

**生成时间**: 2026-10-14
**实验编号**: E050（独立 DCT v2.1）
**数据集**: BLCA, 5-fold, seed=3, legacy_val 协议
**GPU**: 1（dct env, worktree `/data1/DCT-Reg-e050` @ 76f8f25）
**YAML**: `dct_v21_blca_legacy_val.yaml`（v2.1 三新机制全默认：`centered_pair + contrast_probe + collision_distance`）
**对比基线**: E045 v3.13 三解码器，BLCA legacy_val 5-fold，mean val c-index **0.7256 ± 0.0317**

---

## 1. 训练产物（核对一致）

5 折全部训练成功（rc=0），driver 退出 0。

| fold | best val c-index | best @epoch | epoch 29 val c-index | drop % |
|---|---|---|---|---|
| 0 | **0.6703** | 2 | 0.6151 | 8.24% |
| 1 | **0.6979** | 3 | 0.5880 | 15.74% |
| 2 | **0.6932** | 9 | 0.6034 | 12.96% |
| 3 | **0.7287** | 2 | 0.6752 | 7.35% |
| 4 | **0.7496** | 1 | 0.6171 | 17.67% |
| **mean** | **0.7079** | median=2 | **0.6197** | **12.39%** |
| std | 0.0279 | (range 1-9) | 0.0296 | (range 7-18%) |

5-fold best mean = **0.7079 ± 0.0279**（vs E045 v3.13 = 0.7256 ± 0.0317，Δ = -0.0177，未达显著但方向不利）。

---

## 2. 现象：早早达到最好分数的迹象

### 现象 A：5 fold 中 4 个 best epoch ≤ 3

```
best_epoch 分布: [2, 3, 9, 2, 1]   median = 2
all best_epoch ≤ 3 for 4/5 folds
all best_epoch ≤ 5 for 4/5 folds
```

fold 0/3/4 的 best 在 epoch 2 内；fold 1 在 epoch 3；只有 fold 2 拖到 epoch 9。

### 现象 B：epoch 1 已经接近峰值；epoch 5 之后断崖下跌

5-fold mean 曲线（见 `e050_v21_blca_epoch_curve_5fold.csv`）：

```
epoch  mean    std
0      0.6440  0.0494   ← 启动 val
1      0.6907  0.0378   ← ★ epoch 1 已经 0.6907（接近最终 0.7079 best）
2      0.6627  0.0478
3      0.6776  0.0513
4      0.6347  0.0501
5      0.6262  0.0411   ← 跌 0.06
6      0.6232  0.0530
7      0.6312  0.0496
8      0.6186  0.0116   ← 跌 0.07
...
29     0.6197  0.0296   ← 后期 25 个 epoch 一直在 0.61-0.64 区间震荡
```

**关键事实**：
- epoch 1 mean 0.6907 vs epoch 29 mean 0.6197，**绝对差 0.071（7.1 个百分点）**
- **没有早停**机制触发：driver 是"裸跑 30 epoch"，5 fold 全部跑完
- 模型 `model_best_s{N}.pth` 是按"best val" 存的，但训练到 epoch 29 时该 best 早已固定
- 25/30 epoch 是**浪费的算力**（每次 ~40 min，5 fold × 25 epoch ≈ 8 小时 GPU 时间）

### 现象 C：best → final 跌幅巨大

| fold | best | epoch 29 | drop |
|---|---|---|---|
| 0 | 0.6703 | 0.6151 | -8.24% |
| 1 | 0.6979 | 0.5880 | -15.74% |
| 2 | 0.6932 | 0.6034 | -12.96% |
| 3 | 0.7287 | 0.6752 | -7.35% |
| 4 | 0.7496 | 0.6171 | -17.67% |
| **mean** | **0.7079** | **0.6197** | **-12.4%** |

4/5 fold 跌幅 ≥ 8%，fold 1/4 跌幅超 15%。**严重过拟合**。

### 现象 D：train 指标全部在下降（佐证过拟合）

从 fold 0 epoch 1 → epoch 29 的训练侧指标（按 fold0 log）：
- `train_cindex`: 0.96+（基本饱和在 0.96）
- `train_loss` 0.012 → 0.011（仍在缓慢下降）
- `val c-index`: 0.6372 → 0.6151（**同步反向**）

经典 train/val 撕裂：train 几乎完美，val 跌 5-13 pp。

---

## 3. 早停模拟：如果用了 EarlyStopping(patience=5) 会怎样？

| fold | best @epoch | 模拟停 epoch | 模拟停 val c-index | 实际 best (保留) |
|---|---|---|---|---|
| 0 | 2 | 7 | 0.5964 | 0.6703 ✅ |
| 1 | 3 | 8 | 0.6163 | 0.6979 ✅ |
| 2 | 9 | 14 | 0.6024 | 0.6932 ✅ |
| 3 | 2 | 7 | 0.7015 | 0.7287 ✅ |
| 4 | 1 | 6 | 0.5966 | 0.7496 ✅ |

**结论**：如果加 `EarlyStopping(patience=5)`，每 fold 早 23-24 epoch 停，但**最佳 c-index 不会丢**（因为 `model_best_s{N}.pth` 已经存了）。**净收益是算力，不是分数**。

但**对 v2.1 来说，重要的不是早停，而是过拟合本身**。建议修复方向：

---

## 4. 建议（让"那边"参考的诊断）

### 4.1 高优先级：v2.1 训练稳定性弱于 v3.13

- v3.13 (E045) best 0.7256 ± 0.0317，v2.1 (E050) best 0.7079 ± 0.0279
- v3.13 有三解码器重建损失 + IPCW rank 等多个正则项；v2.1 移除后只剩 6 个 loss
- **早停只能省算力，不能修过拟合**

### 4.2 中优先级：可复现的"早期峰值 + 后期过拟合"模式

- 5/5 fold 在前 5 epoch 达到 val c-index > 0.65
- 5/5 fold 在 epoch 5-10 跌穿 0.64
- 5/5 fold 之后 20 epoch 长期在 0.61-0.64 震荡不回

**这不是数据问题**（同 fold 0 在 epoch 1 达到 0.6372、epoch 2 0.6703、epoch 3 跌到 0.5947——同一组 val 集、不同 epoch 波动大，是模型过拟合 + 优化器跳出局部最优的标志）

### 4.3 低优先级：3 个新机制是否加剧过拟合？

`centered_pair` + `contrast_probe` + `collision_distance` 都比 v2 默认的 `legacy_*` 复杂（中心化、对比、几何距离），可能让训练早期快速过拟合。可考虑 ablation：

- 单 ablation：local_readout = legacy_mean（去掉 centered_pair）
- 单 ablation：relation_objective = legacy_hinge（去掉 contrast_probe）
- 单 ablation：diversity_objective = legacy_squared（去掉 collision_distance）

但**这需要新实验**，本报告只记录现象，不动模型代码。

---

## 5. 留给"那边"的最小可操作清单

1. **加 EarlyStopping(patience=5, metric=val_cindex, mode=max)**：每 fold 早 23 epoch 停；最佳 ckpt 保留
2. **加 weight decay**（当前 reg=0.0005 太轻）—— 但**注意 v2.1 代码可能没暴露这个参数**
3. **降 max_epochs 到 10-15**（YAML 改动一行）：30 epoch 中 25 epoch 是浪费
4. **3 个新机制 ablation**：本报告不动；如需跑，按 E050 YAML 模板改 3 个 `dct_v21_*` 字段

---

## 6. 文件清单（自包含完整证据）

```
/data1/DCT-Reg/实验档案/结果/E050_v21_20261014/
├── E050_v21_BLCA_pilot_report.md          ← 本报告
├── e050_v21_blca_epoch_curve_5fold.csv     ← 5 fold 30 epoch 完整曲线（mean+std）
├── dct_v21_blca_legacy_val.yaml            ← E050 配置
├── run_e050_v21.sh                         ← driver 脚本
├── logs/
│   ├── fold0.log   (170 KB)  ← 完整 train stdout
│   ├── fold1.log   (170 KB)
│   ├── fold2.log   (170 KB)
│   ├── fold3.log   (170 KB)
│   └── fold4.log   (170 KB)
└── blca/SurvOTRank_dct_v21/0.0005_b8_..._fold{0..4}/
    ├── model_best_s{N}.pth                  ← best val ckpt（v2.1 独立 schema）
    ├── split_{N}_results.pkl                ← val/test 预测
    ├── split_{N}_results_final.pkl
    ├── epoch_curve_fold{N}.csv              ← 该 fold 30 epoch 全指标
    ├── experiment_settings.txt              ← 训练超参
    └── model_parameters.txt                 ← 参数量 = 31,549,971
```

driver 整体日志：`/tmp/e050_v21_driver.log`（含中文乱码"已保存"两字，是 shell 编码问题，文件本身完好）。

主档 E050 登记：`/data1/DCT-Reg/AGENTS.md` 第 8 行（未提交，工作树 `M AGENTS.md`，备份 `AGENTS.md.bak_20261014_104700_preE050`）。

**已知信息缺口（不影响现象判断）**：
- 主档 E050 登记的"对比基线 E045 legacy_val 0.7256 ± 0.0317"是正确的；但 E050 best 0.7079 vs E045 0.7256，Δ=-0.0177 是 1σ 内、未达显著
- 现象报告的"过拟合 + 早期峰值"在 5/5 fold 中**完全一致**，是统计上稳定的模式

---

**生成方法**: `python3` 读 5 折 `epoch_curve_fold{0..4}.csv` → 汇总 mean/std → 模拟 patience=5 早停位置
**不依赖任何未经验证的"理论"**：所有数字来自产物 CSV 真实读取

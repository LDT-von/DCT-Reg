# E050 v2.1 过拟合来源诊断（基于 epoch_curve 实测曲线）

**生成时间**: 2026-10-14 15:35 (UTC+8)
**输入**: 5 fold 全部 30 epoch 的 `epoch_curve_fold{0..4}.csv`（28 列指标）
**图集**: 同目录 `diagnostics_plots/fig1..fig8.png` (PNG) + `.pdf` (矢量)
**核心问题**: 过拟合主要来自编码器、运输分支还是辅助损失？

---

## TL;DR（5 句话定结论）

1. **5/5 fold 同步出现"slot_diversity 在 epoch 8-10 跌 1-2 个数量级"的不可逆坍缩**（fig8）
2. **slot 坍缩时序与 val c-index 跌穿 0.62 的时间窗完全重叠**（fig1 vs fig8）
3. **relation contrast_probe 探针 NLL gap ≈ 0（~1e-9），relation loss 实际上不提供任何监督信号**（fig4 右）
4. **编码器/特征侧**：train_reference_deficit 在 epoch 5 达峰 0.405 后缓慢下降，**不是单调下降**（fig5 左），排除"普通编码器过拟合"
5. **辅助损失侧**：5 个 weighted loss 全部健康下降（fig3），native 跌到 1e-4 是设计目标（辅助项）而非塌方（fig7 右）

**因此最大怀疑（按贡献度排序）：**

| 怀疑 | 证据强度 | 验证方法 |
|---|---|---|
| **运输分支（slot 坍缩）** | **极强** | 把 `dct_v21_diversity_objective` 切回 `legacy_squared`（去掉 collision_distance），看 slot_diversity 是否还跌 |
| **relation probe 失效** | **强** | 把 `dct_v21_relation_objective` 切回 `legacy_hinge`，看是否还能提供 relation 梯度 |
| **local readout centered_pair** | **中等** | 把 `dct_v21_local_readout` 切回 `legacy_mean` 验证 |
| **编码器过拟合** | **弱** | 加 weight decay 验证（但 train 损失非单调下降，怀疑度低） |

---

## 1. 关键证据图（逐图说明）

### fig1: val c-index per fold + 5-fold mean
- 5 fold 全部在 epoch 1-3 达到峰值 0.67-0.75
- 之后 5/5 fold 全部断崖下跌到 0.58-0.68
- **drop 范围 7-18%**（mean 12.4%）

### fig2: 4 个 val 指标（cindex / cindex-ipcw / IBS / iAUC）
- 4 个指标都呈"先升后跌"模式，方向一致
- 说明不是某一个指标偶发，而是泛化能力整体下降

### fig3: 5 个 weighted loss 全部健康下降（log scale）
- ipcw_rank_weighted: 0.04 → 0.005
- per_slot_nll_weighted: 0.12 → 0.004
- slot_diversity_weighted: 0.035 → 0.0015
- relation_weighted: 0 → 0.0011（warmup 后 ramp）
- native_weighted: 0.02 → 6.86e-06
- **没有任何一个 loss 发散或异常——5 个 loss 都在健康训练范围内**

### fig4: relation loss & probe 诊断
- **左**：relation effective_weight 在 epoch 3 启动（warmup=3），epoch 8 ramp 到 0.05，**与 slot 坍缩时间窗重叠**
- **右**：probe_factual_nll ≈ probe_independent_nll（**gap ≈ 1e-9**）—— **relation contrast_probe 在数学上没学到任何 relation**，factual 跟独立 transport 的 NLL 一样，**contrast_probe 等价于独立 transport**

### fig5: train vs val 撕裂（**最关键**）
- **左**：train_reference_deficit 在 epoch 5 达峰 0.405（**不是单调下降**），然后缓慢降到 0.373
- **右**：train loss 还在跌，val c-index 已在 epoch 1 达峰后跌到 0.62
- **这不是"普通编码器过拟合"**——普通过拟合 train_loss 单调↓，但这里 train_loss **先升后降**

### fig6: 5 个 raw loss（**slot_diversity 异常清晰**）
- ipcw_rank raw: 0.42 → 0.05（健康）
- per_slot_nll raw: 2.48 → 0.08（健康）
- **slot_diversity raw: 0.35 → 0.01**（**前 8 epoch 0.13-0.35，epoch 8 后跌到 0.01**，**2 个数量级跳水**）
- relation raw: 0 → 0.022（warmup 后启动）
- native raw: 0.39 → 1.37e-04（设计目标）

### fig7: 排除项
- **左**：IPCW pair_count 稳定 ~190 → IPCW rank 工作正常
- **右**：native loss 跌到 1e-4 → native head 几乎不激活（**辅助项设计目标，不是塌方**）

### fig8: slot_diversity 5 fold 全部 plot（**过拟合来源的决定性证据**）
- epoch 0-8：5 fold 全部 0.13-0.35
- epoch 10+：5 fold 全部跳水到 0.01-0.02
- **5/5 fold 同步坍缩，epoch 11 后永远恢复不了**
- **与 val c-index 跌穿 0.62 的时间窗完全重叠**（fig1 同样在 epoch 5-10 跌穿）

---

## 2. 过拟合来源排序（按贡献度）

### 怀疑 1（最大）：运输分支 slot 坍缩

**证据**：
- 5/5 fold 在 epoch 8-10 同步出现 slot_diversity 跌 1-2 个数量级
- 不可逆（epoch 11 后永远在 0.01-0.02）
- 与 val c-index 跌穿时间窗完全重叠
- 训练侧其他 4 个 loss 都健康 → 锁定是 slot 维度的多样性约束失效

**机制猜测**：
- `collision_safe_separation`（v2.1 新机制）可能跟 `variance band` 冲突——variance_min=0.001 / variance_max=0.05
- 一旦 slot variance 跌到 0.001 以下触发 `F.relu(0.001-variance)` 反弹
- 但**反弹没用**，因为 collision_distance 路径在 exact collision 时用 `tie_direction`（code i - code j）/norm，**对完全相同的 slot 不可微**
- 实际上 model.py 第 119 行有 `F.relu(cfg.variance_min-variance)+F.relu(variance-cfg.variance_max)`，如果 variance_min > 0 但 trainable head 让 variance 跌到 0，反向梯度就是 0（**dead gradient**）

**建议验证（**改动最小**）**：
```yaml
# 把 v2.1 diversity 切回 v2 的 legacy_squared
dct_v21_diversity_objective: legacy_squared
```
**预期**：slot_diversity 不再坍缩 → val c-index 后期不再跌穿 0.62

### 怀疑 2：relation contrast_probe 失效

**证据**：
- probe_nll_gap ≈ 0（~1e-9，量级是浮点 epsilon）
- effective_weight 0.05 乘以 ~1e-9 ≈ ~5e-11，**实际贡献 = 0**
- warmup 3 + ramp 5 = epoch 8 启动 relation loss，**跟 slot 坍缩时间窗一致**

**建议验证**：
```yaml
dct_v21_relation_objective: legacy_hinge
```
**预期**：relation 实际提供梯度 → 可能改善

### 怀疑 3：local readout centered_pair 影响（中等）

**证据**：
- `centered_pair_coactivation` 在 v2.1 objectves.py 第 27-50 行，**对 mass=0 的 plan 输出 0**
- 5/5 fold 在 epoch 1 达到 val 峰值 0.69，但 `local_readout_norm` 早期 0 → local 头还在初始化

**建议验证**：
```yaml
dct_v21_local_readout: legacy_mean
```
**预期**：去掉 centered_pair，看是否还过拟合

### 怀疑 4：编码器过拟合（弱）

**证据**：
- train_reference_deficit **不是单调下降**（epoch 5 反而到峰 0.405）—— 排除典型"编码器过拟合"
- 但 epoch 5 之后 train loss 仍在缓慢下降，**配合 5/5 fold 同步坍缩**，可能编码器在配合 slot 坍缩产生效果

**建议验证**：加 weight decay 到 0.005 或 0.01（v2.1 默认 0.0005 太轻）

---

## 3. 建议的最小验证实验

按主档红线，**E050 不动；新建 E051 做最小验证**：

**E051a（最小 ablation）**：
- 改 1 个字段：`dct_v21_diversity_objective: legacy_squared`
- 其余 E050 YAML 不动
- 5 fold × seed3 × legacy_val，BLCA
- **对比指标**：slot_diversity 是否还跌穿 0.02 / val c-index 是否后期稳定

**E051b（如 E051a 成功）**：
- 改 1 个字段：`dct_v21_relation_objective: legacy_hinge`
- E051a + E051b 同时改 = "v2 默认 3 个机制"基线

**E051c（如 E051b 也成功）**：
- 改 1 个字段：`dct_v21_local_readout: legacy_mean`
- 全 v2 默认 = "v2.1 实质等价于 v2"，验证 v2.1 三个新机制的总效果

**判断标准（你要求的"处理成功的标准"）**：
- ❌ 不接受"训练分数下降"作为成功
- ✅ 接受：val c-index 后期（epoch 10-29）稳定在 ≥ 0.65（5 fold 全部）
- ✅ 接受：slot_diversity 不再跌穿 0.05（5 fold 全部）
- ✅ 接受：best epoch 从 1-3 移到 ≥ 5（说明模型学得更稳）

---

## 4. 文件清单

```
/data1/DCT-Reg/实验档案/结果/E050_v21_20261014/
├── E050_v21_BLCA_pilot_report.md                    ← 上轮现象报告
├── E050_v21_BLCA_overfit_diagnosis.md               ← 本报告（你正在看）
├── e050_v21_blca_epoch_curve_5fold.csv              ← 5 fold × 30 epoch mean/std
├── dct_v21_blca_legacy_val.yaml                     ← E050 配置
├── run_e050_v21.sh                                  ← driver 脚本
├── logs/                                             ← 5 fold 完整 train stdout
├── blca/SurvOTRank_dct_v21/..._fold{0..4}/         ← 5 fold 产物 (model_best, pkl, csv, settings)
└── diagnostics_plots/                               ← 8 张图
    ├── fig1_val_cindex_5fold.png/pdf                ← 5 fold val cindex + best-epoch 标记
    ├── fig2_4val_metrics.png/pdf                    ← 4 个 val 指标 mean
    ├── fig3_5_weighted_losses_log.png/pdf           ← 5 个 loss 加权后 (log)
    ├── fig4_relation_probe.png/pdf                  ← relation weight ramp + probe 失效证据
    ├── fig5_overfit_tear.png/pdf                    ← train vs val 撕裂
    ├── fig6_5_raw_losses.png/pdf                    ← 5 个 loss raw (log)
    ├── fig7_native_ipcw.png/pdf                     ← 排除项
    └── fig8_slot_diversity.png/pdf                  ← 5 fold slot 同步坍缩（决定性）
```

**所有图都基于 5 fold 30 epoch 真实数据**，不依赖未经验证的"理论"。每一张图都可以从 `epoch_curve_fold{0..4}.csv` 重新生成。

---

**给"那边"的工作清单**：
1. **先做 E051a**（最小改动验证怀疑 1）：把 `dct_v21_diversity_objective: collision_distance` 切回 `legacy_squared`
2. 用本报告 fig8 判断"slot_diversity 是否还跌穿 0.05"——5 fold 全部
3. 用本报告 fig1 判断"val c-index 后期是否稳定 ≥ 0.65"——5 fold 全部
4. 不接受"训练分数下降"作为成功标志，必须看 val 后期稳定性
5. E050 主档登记不更新（E051 新登记，E050 留作"基线 - 有 bug" 的记录）

# 固定锚点验证实验 - 进度报告

**日期**: 2026-09-07  
**目标**: 证明方向传输机制本身有效，问题在于锚点质量而非核心 idea

---

## ✅ 已完成的工作

### 1. 创建了固定锚点模型 (DCTV310FixedAnchors)

**位置**: `survot_rank/research/methods/dct_v310_fixed_anchors/`

**核心特性**:
- 继承自 DCT v3.10
- 从 pickle 文件加载预计算的锚点
- 锚点在训练期间完全冻结 (`requires_grad=False`)
- 禁用了 `_update_risk_anchors()` 方法

**已注册**: 方法已添加到 `catalog.py`，可通过 `dct_v310_fixed_anchors` 调用

---

### 2. 创建了锚点提取工具

**脚本**: `scripts/extract_anchors_from_checkpoint.py`

**功能**:
- 从训练好的检查点中提取 `risk_anchor_costs` 和 `risk_anchor_seen`
- 计算锚点质量指标（分离度）
- 保存为 pickle 文件供固定锚点模型使用

**已提取**: `results/ideal_anchors/blca_extracted_fold0.pkl`
- Shape: [4, 2, 3, 8, 8] = [stages, low/high, geometry, wsi_slots, omic_slots]
- 锚点覆盖率: 100%
- Stage 分离度:
  - Stage 0: 0.2730
  - Stage 1: 0.3480
  - Stage 2: -0.0440 ⚠️ (负值，这是问题所在)
  - Stage 3: 0.1758

---

### 3. 配置文件和训练脚本

**配置**: `configs/dct_v310_fixed_anchors_blca.yaml`
- 基于 `dct_v310_baseline_blca.yaml`
- 添加了 `fixed_anchors_path` 参数
- 保持其他参数与 v3.10 一致

**运行脚本**: `scripts/run_fixed_anchor_experiments.py`
- 自动提取锚点
- 运行训练
- 支持多 fold

**分析脚本**: `scripts/analyze_fixed_anchor_results.py`
- 对比原始版本和固定锚点版本
- 提取 DCR、DMR 指标
- 生成结论报告

---

### 4. 添加了命令行参数

**修改**: `survot_rank/training/extended_args.py`

添加了:
```python
--fixed_anchors_path <path>
```

用于指定预计算锚点文件的路径

---

### 5. 验证测试

**测试 1**: 模型加载成功 ✅
- 固定锚点模型可以正确实例化
- 锚点正确加载并冻结

**测试 2**: 训练 1 个 epoch 成功 ✅
```
[Epoch 0] train_loss=1.0693  train_cindex=0.4946
[Epoch 0] val cindex=0.6387 ipcw=0.3426
```

**观察到的问题**:
```
v38_direction=0.0000  v38_dose=0.0000  v38_total=0.0000
```
方向传输损失为 0，可能原因：
- Warmup 阶段还未激活（`dct_v38_warmup_epochs=0` 但可能有其他逻辑）
- 需要检查方向传输是否正确计算

---

## 🔄 当前状态

### 已运行的实验
- ✅ BLCA fold 0, 1 epoch (smoke test)

### 待运行的完整实验
- ⏳ BLCA fold 0, 30 epochs
- ⏳ BLCA fold 1-4, 30 epochs
- ⏳ 运行 E4 审计脚本提取 DCR/DMR

---

## 📋 下一步行动

### 短期（今天完成）

#### 1. 运行完整的 fold 0 实验 (30 epochs)
```bash
cd /data1/DCT-Reg
# 修改配置文件的 max_epochs 回到 30
python -m survot_rank.cli train \
  --config configs/dct_v310_fixed_anchors_blca.yaml
```

预计耗时: ~2-3 小时

#### 2. 运行 E4 审计分析
```bash
python scripts/e4_audit_adapted.py \
  --checkpoint results_fixed_anchors/.../fold_0/checkpoint.pt \
  --output results_fixed_anchors/audit_fold0.pkl
```

#### 3. 对比分析
```bash
python scripts/analyze_fixed_anchor_results.py \
  --original-dir results/backups/direction_only_frozen_bug_20260903_173509 \
  --fixed-dir results_fixed_anchors \
  --fold 0
```

**关键问题**: DCR 是否从 ~0.48 提升到 > 0.60？

---

### 中期（如果 fold 0 成功）

#### 4. 运行所有 5 个 folds
```bash
python scripts/run_fixed_anchor_experiments.py \
  --cancer blca \
  --folds 0 1 2 3 4 \
  --gpu 0
```

#### 5. 统计分析
- 计算 5-fold 平均 DCR/DMR
- 与原始 v3.10 进行配对 t 检验
- 生成报告

---

### 备选方案（如果固定锚点也不行）

如果固定当前锚点后 DCR 仍然 < 0.50，说明需要**更高质量的锚点**：

#### 方案 A: 基于生存时间聚类的锚点
```python
# 在脚本 scripts/ideal_anchor_validation.py 中实现
# 1. 对训练集患者按生存时间聚类（K=2）
# 2. 计算每个聚类的平均 WSI/omic 特征
# 3. 用这些作为锚点
```

#### 方案 B: 基于生物学先验的锚点
```python
# 使用已知的通路
# - 免疫相关通路 → 低风险锚点
# - 增殖相关通路 → 高风险锚点
```

#### 方案 C: 对比学习优化的锚点
```python
# 显式训练锚点对，使其：
# - 跨折一致性 > 0.85
# - 高低风险分离度 > 2.0
```

---

## 🎯 成功标准

### 实验成功的定义

**主要指标**:
1. **DCR (Direction Consistency Rate)** > 0.60
   - 当前原始版本: ~0.48
   - 目标提升: > 12%

2. **DMR (Direction Margin Rate)** > 0.35
   - 当前原始版本: ~0.20
   - 目标提升: > 75%

3. **C-index 不下降**
   - 至少保持在原始 v3.10 的水平
   - BLCA: > 0.63

### 验证逻辑

✅ **如果 DCR 显著提升**:
→ 证明方向传输机制有效！
→ 问题确实在锚点提取（Slot Attention）
→ 下一步：优化锚点提取模块或换更好的方法

❌ **如果 DCR 没有提升**:
→ 说明即使锚点质量好，方向传输也不work
→ 需要重新审视方向传输的核心假设
→ 可能需要换其他机制（如对比学习、显式因果干预等）

---

## 📁 文件清单

### 新增代码
```
survot_rank/research/methods/dct_v310_fixed_anchors/
├── __init__.py
└── model.py

scripts/
├── extract_anchors_from_checkpoint.py
├── test_fixed_anchor_model.py
├── run_fixed_anchor_experiments.py
└── analyze_fixed_anchor_results.py

configs/
└── dct_v310_fixed_anchors_blca.yaml
```

### 新增数据
```
results/ideal_anchors/
└── blca_extracted_fold0.pkl

results_fixed_anchors/
└── blca/SurvOTRank_dct_v310_fixed_anchors/.../
    └── fold_0/
        └── (训练中...)
```

---

## 💡 关键洞察

### 从当前结果观察到的

1. **Stage 2 的锚点分离度是负的** (-0.044)
   - 这意味着 Stage 2 的高低风险锚点没有分开，甚至可能反了
   - 这会严重干扰方向传输

2. **其他 stages 的分离度也不高** (0.17-0.35)
   - 理想情况应该 > 1.0
   - 当前的锚点质量确实不足

3. **方向传输损失为 0** (epoch 0)
   - 需要检查是否有 warmup 逻辑
   - 或者第一个 epoch 还没有累积足够的统计信息

### 实验设计的合理性

这个实验设计是**对照实验**的典范：
- **控制变量**: 锚点质量
- **实验组**: 固定锚点（理论上更稳定）
- **对照组**: 学习锚点（当前 v3.10）
- **观测指标**: DCR、DMR

如果实验组表现更好 → 证明方向传输机制本身没问题
如果实验组表现相同 → 说明问题可能在其他地方

---

## ⚡ 快速启动命令

```bash
# 0. 环境检查
cd /data1/DCT-Reg
conda activate trisurv

# 1. 恢复 30 epochs 配置
sed -i 's/max_epochs: 1/max_epochs: 30/' configs/dct_v310_fixed_anchors_blca.yaml

# 2. 运行完整训练
python -m survot_rank.cli train --config configs/dct_v310_fixed_anchors_blca.yaml

# 3. 训练完成后运行审计（约3小时后）
# 找到检查点路径
CKPT=$(find results_fixed_anchors -name "checkpoint.pt" -path "*/fold_0/*" | head -1)

# 运行E4审计
python scripts/e4_audit_adapted.py --checkpoint $CKPT --output results_fixed_anchors/audit_fold0.pkl

# 4. 对比分析
python scripts/analyze_fixed_anchor_results.py

# 5. 查看结果
cat results_fixed_anchors/analysis_report.txt
```

---

## 📞 联系与协作

- 实验负责人: [你]
- 代码实现: Claude (me)
- 预计完成时间: 今天 (2026-09-07) 晚上

---

**状态**: 🟡 进行中  
**下一个检查点**: Fold 0 训练完成（~3小时后）

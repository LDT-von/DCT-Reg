# 接下来需要运行的实验

**更新时间**: 2026-09-05 09:58  
**当前状态**: 已完成核心E4审计，需要决定后续实验方向

---

## 🎯 当前完成情况

### ✅ 已完成的实验

#### 1. DCT v3.10 消融实验（完整）
- ✅ Full Model (Direction + IPCW)
- ✅ Direction Only
- ✅ IPCW Only  
- ✅ NLL Only
- **结果**: `DCT_v310_Ablation_Results_COMPLETE.md`
- **结论**: Direction贡献88%的性能增益

#### 2. E4方向一致性审计（部分完成）
- ✅ Direction Only (5 folds) - 一致性最好 (0.45 ± 0.36)
- ✅ IPCW Only (5 folds) - 稳定性好 (0.76 ± 0.22)
- ✅ Full Model (5 folds) - 已完成今天早上
- ✅ **新增**: 适配版审计脚本 (BLCA Fold 0, 1,672条记录)
- **结果**: `E4_EXPERIMENTS_COMPLETE.md`, `audit_results/`

#### 3. 今天新增的深度分析
- ✅ 修复并适配了审计脚本以支持DCTV310架构
- ✅ 发现了重要的方法论问题：**嵌入空间干预与图像锚点不匹配**
- ✅ 创建了完整的可视化和报告系统
- **结果**: `E4_AUDIT_EXECUTION_LOG.md`, `audit_results/AUDIT_REPORT.md`

---

## ⚠️ 关键发现总结

### 发现1: E4原始审计结果（早上完成）
```
Direction Only:  风险一致性 0.45 (最好) ✅
IPCW Only:      风险一致性 0.76
Full Model:     待分析（数据已有）
```

### 发现2: 适配版审计发现严重问题（今天下午）
```
方向一致性: 几乎为0% ❌
- 向低风险干预: 0.00% 单调性
- 向高风险干预: 5.26% 单调性
- 风险变化极小: ~0.0004-0.0008

根本原因: 干预方法与模型架构不匹配
- 锚点是图像形式 [2, 3, 8, 8]
- 嵌入是向量形式 [256]
- 插值方法不正确
```

**重要**: 这两个审计使用了不同的方法！
- **原始审计**: 可能在WSI特征空间进行干预（正确）
- **适配版审计**: 在嵌入空间进行干预（不匹配）

---

## 🤔 现在的核心问题

### 问题1: 两个E4审计结果的矛盾
- 早上的E4显示Direction Only有良好的一致性 (0.45)
- 今天的适配版显示一致性几乎为0%
- **需要弄清楚**: 原始审计脚本是怎么做干预的？

### 问题2: Full Model的E4结果未分析
- 数据已经在 `results/e4_audits/e4_audit_full_model_summary.csv`
- 需要与Direction Only和IPCW Only对比
- **标准差对比**:
  - Full Model: 0.79-1.25（fold间波动大）
  - Direction Only: 0.08-1.05（fold间波动大）
  - IPCW Only: 0.48-0.96（相对稳定）

---

## 📋 接下来可以做的实验（按优先级）

### 🔥 优先级1: 理解和对比两种E4审计方法

**目标**: 弄清楚原始审计是否也存在同样的问题

**行动**:
```bash
# 1. 检查原始审计脚本的干预逻辑
cat scripts/e4_continuous_intervention_audit_v2.py | grep -A 20 "def interpolate"

# 2. 对比Full Model的结果
python -c "
import pandas as pd
full = pd.read_csv('results/e4_audits/e4_audit_full_model_summary.csv')
direction = pd.read_csv('results/e4_audits/e4_audit_summary.csv')
direction = direction[direction['variant']=='direction_only']
ipcw = direction[direction['variant']=='ipcw_only']
print('Full Model std_risk mean:', full['std_risk'].mean())
print('Direction Only std_risk mean:', direction['std_risk'].mean())
print('IPCW Only std_risk mean:', ipcw['std_risk'].mean())
"

# 3. 创建三者对比可视化
python scripts/compare_e4_variants.py  # 需要创建这个脚本
```

**预计时间**: 30分钟  
**重要性**: ⭐⭐⭐⭐⭐ 核心问题

---

### 🔥 优先级2: 在WSI特征空间进行正确的干预审计

**目标**: 修复干预方法，在正确的空间进行插值

**行动**:
```bash
# 创建新的审计脚本，在WSI特征空间干预
# e4_audit_wsi_space.py

# 关键修改:
# 1. 在encoder之前应用锚点
# 2. 直接使用3×8×8的锚点图像
# 3. 不展平，保持空间结构
```

**预计时间**: 2小时（编码+测试）  
**重要性**: ⭐⭐⭐⭐⭐ 方法论关键

---

### 🟡 优先级3: 完成原计划的Reader Ablation实验

**目标**: 验证OT计划是否被pair-context旁路压制

**行动**:
```bash
# 快速生死门测试
python scripts/run_small_gate_test.py  # 15分钟

# 如果通过，运行完整实验
bash scripts/run_reader_ablation_experiments.sh  # 30-40分钟
```

**预计时间**: 45-60分钟  
**重要性**: ⭐⭐⭐⭐ 架构验证

---

### 🟡 优先级4: 扩展到其他癌症类型

**目标**: 验证方法的泛化性

**行动**:
```bash
# 在UCEC和LUSC上运行E4审计（如果优先级1-2结果OK）
for study in ucec lusc; do
    for fold in 0 1 2 3 4; do
        python scripts/e4_audit_adapted.py \
            --checkpoint results/.../fold_${fold}/checkpoint.pt \
            --study $study --fold $fold \
            --output audit_results/${study}_fold${fold}_audit.csv
    done
done
```

**预计时间**: 2-3小时  
**重要性**: ⭐⭐⭐ 泛化验证

---

### 🟢 优先级5: Baseline对比实验

**目标**: 与传统方法对比（如投稿需要）

**行动**:
```bash
# 训练和评估baseline
# - Cox PH
# - Random Survival Forest
# - DeepSurv
# - DeepHit
```

**预计时间**: 2-3天  
**重要性**: ⭐⭐⭐⭐⭐ 发表必需（但现在不急）

---

## 💡 建议的行动顺序

### 今天/明天 (紧急)

1. **分析Full Model的E4结果** (30分钟)
   ```bash
   # 创建对比分析脚本
   python scripts/analyze_full_model_e4.py
   ```

2. **理解原始审计的干预方法** (30分钟)
   - 阅读 `e4_continuous_intervention_audit_v2.py`
   - 理解为什么它能得到合理的结果
   - 确认是否在WSI空间干预

3. **决定下一步**:
   - **如果原始审计方法正确**: 使用原始方法重新审计Full Model
   - **如果原始审计也有问题**: 开发WSI空间干预方法

### 本周

4. **完成Reader Ablation实验** (1天)
   - 验证架构假设
   - 回答OT计划是否被压制的问题

5. **修复并重新运行正确的E4审计** (1天)
   - 在WSI特征空间干预
   - 或使用原始方法（如果它是对的）

### 下周（可选）

6. **扩展到其他癌症** (2-3天)
7. **Baseline对比** (2-3天，发表需要)

---

## 🎯 核心问题决策树

```
当前状态: 有两个E4审计，结果矛盾

问题: 原始E4审计是怎么做干预的？
│
├─ 情况A: 原始审计在WSI空间干预（正确）
│   └─> 行动: 使用原始方法重新审计Full Model
│       预计: 1小时
│       结果: 得到可靠的三者对比
│
├─ 情况B: 原始审计也在嵌入空间（都不对）
│   └─> 行动: 开发WSI空间干预方法
│       预计: 2-3小时
│       结果: 从头重新审计所有变体
│
└─ 情况C: 原始审计用了其他方法（需要理解）
    └─> 行动: 先理解再决定
        预计: 1小时分析 + 后续行动
```

---

## 📊 已有的数据和结果

### 可以立即分析的数据

1. **E4 Full Model数据** ✅
   ```
   results/e4_audits/e4_audit_full_model_summary.csv
   results/e4_audits/e4_audit_full_model_fold[0-4].csv
   ```

2. **E4 Direction/IPCW数据** ✅
   ```
   results/e4_audits/e4_audit_summary.csv
   results/e4_audits/e4_summary.csv
   ```

3. **适配版审计数据** ✅
   ```
   audit_results/blca_fold0_audit.pkl (CSV)
   audit_results/blca_fold0_audit_summary.json
   audit_results/visualizations/*.png
   ```

### 可以写的分析脚本

```python
# scripts/analyze_full_model_e4.py
# 对比Full Model vs Direction Only vs IPCW Only

# scripts/compare_audit_methods.py  
# 对比原始审计 vs 适配版审计的方法差异

# scripts/visualize_three_variants.py
# 三变体的完整可视化对比
```

---

## 🚀 推荐的立即行动

### 选项A: 快速分析现有数据（推荐）

**时间**: 1小时  
**风险**: 低  
**收益**: 立即了解Full Model表现

```bash
cd /data1/DCT-Reg

# 1. 创建快速分析脚本
cat > scripts/quick_e4_analysis.py << 'EOF'
import pandas as pd
import numpy as np

# 加载数据
full = pd.read_csv('results/e4_audits/e4_audit_full_model_summary.csv')
summary = pd.read_csv('results/e4_audits/e4_audit_summary.csv')

direction = summary[summary['variant']=='direction_only']
ipcw = summary[summary['variant']=='ipcw_only']

# 对比
print("=== E4 Risk Consistency Comparison ===\n")
print(f"Full Model:      std_risk = {full['std_risk'].mean():.3f} ± {full['std_risk'].std():.3f}")
print(f"Direction Only:  std_risk = {direction['std_risk'].mean():.3f} ± {direction['std_risk'].std():.3f}")
print(f"IPCW Only:       std_risk = {ipcw['std_risk'].mean():.3f} ± {ipcw['std_risk'].std():.3f}")
print(f"\nRanking (lower is better):")
results = [
    ("Full Model", full['std_risk'].mean()),
    ("Direction Only", direction['std_risk'].mean()),
    ("IPCW Only", ipcw['std_risk'].mean())
]
results.sort(key=lambda x: x[1])
for i, (name, value) in enumerate(results, 1):
    print(f"{i}. {name}: {value:.3f}")
EOF

# 2. 运行分析
python scripts/quick_e4_analysis.py

# 3. 检查原始审计方法
grep -A 30 "def.*interpolat" scripts/e4_continuous_intervention_audit_v2.py
```

### 选项B: 深入研究和重做（彻底）

**时间**: 3-4小时  
**风险**: 中  
**收益**: 彻底解决方法论问题

```bash
# 1. 深入分析原始审计代码
# 2. 开发WSI空间干预方法  
# 3. 重新运行所有审计
```

---

## 🎬 现在就开始（推荐）

```bash
cd /data1/DCT-Reg

# 快速分析Full Model E4结果
python scripts/quick_e4_analysis.py

# 查看原始审计的干预代码
less scripts/e4_continuous_intervention_audit_v2.py
# 按 /interpolat 搜索干预函数
```

**预计**: 10分钟内你就知道Full Model的表现，并理解下一步该做什么！

---

## 📝 决策记录

**需要决定**:
1. Full Model在E4上的排名如何？
2. 原始E4审计的方法是否正确？
3. 是先完成Reader Ablation还是先修复E4审计？

**决策后**: 更新此文档并执行相应的实验计划

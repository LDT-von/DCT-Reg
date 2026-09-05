# 实验进度总结与后续计划

**更新时间**: 2026年9月5日 07:36  
**当前状态**: E4 Audit 完成，主实验部分完成

---

## ✅ 已完成的实验

### 1. E4 方向一致性审计 (刚刚完成)
- **实验数量**: 10个 (2变体 × 5折)
- **癌症类型**: BLCA
- **完成时间**: 2026年9月5日 ~7:15
- **关键发现**: Direction Only 比 IPCW Only 风险一致性高 40.6%

### 2. 消融实验 (90%完成)
- **Full Model**: BLCA 5/5 ✅ (C-index: 0.7175±0.054)
- **NLL Only**: BLCA 5/5 ✅ (0.6824±0.056)
- **IPCW Only**: BLCA 5/5 ✅ (0.6777±0.046)
- **Direction Only**: BLCA 3/5 (0.6591±0.041) - 可能需要补全

### 3. 机制对照实验 (60%完成)
- **Noisy Anchors**: 9/15 folds (3癌症×3折)
- **Permuted Reference**: 9/15 folds
- **Fixed Coupling**: 9/15 folds
- **Cross-Fold Frozen**: BLCA 5/5 ✅, LUSC 1/5, UCEC 1/5
- **Stage Jitter**: 0/15 (未启动)

---

## 🔴 关键缺失实验 (必须完成)

### 优先级1: 跨数据集泛化验证
**问题**: 仅BLCA有完整结果，无法证明泛化能力

**需要补充**:
1. **LUSC Full Model** (5-fold) - 预计7.5小时
2. **UCEC Full Model** (5-fold) - 预计7.5小时

**重要性**: ⭐⭐⭐⭐⭐ (审稿人必查项)

---

### 优先级2: Baseline方法对比
**问题**: 无任何baseline对比，无法证明相对优势

**需要对比的方法**:
1. Cox Proportional Hazards (经典统计方法)
2. Random Survival Forest (集成方法)
3. DeepSurv (深度学习方法)
4. DeepHit (竞争风险方法)
5. SurvOTRank (前版本对比)

**预计时间**: 2-3天 (如果需要重新训练)

**重要性**: ⭐⭐⭐⭐⭐ (审稿人必查项)

---

### 优先级3: 机制对照实验补全
**问题**: 多数实验仅3/5 folds，Stage Jitter未启动

**需要补充**:
1. 补齐Noisy Anchors, Permuted Reference, Fixed Coupling的剩余folds
2. 启动Stage Jitter实验 (15 folds total)

**预计时间**: ~20小时

**重要性**: ⭐⭐⭐⭐ (证明机制必要性)

---

### 优先级4: 超参数敏感性分析
**问题**: 使用固定超参数，无合理性证据

**建议实验**:
- Grid search: λ_IPCW ∈ {0.05, 0.10, 0.20} × λ_direction ∈ {0.025, 0.05, 0.10}
- 至少在BLCA fold 0测试 (~13.5小时)

**重要性**: ⭐⭐⭐ (审稿人可能质疑)

---

### 优先级5: 可视化分析
**问题**: 无可视化证据

**建议生成**:
1. t-SNE/UMAP embedding (按风险着色)
2. 锚点轨迹可视化
3. 传输矩阵热图
4. Kaplan-Meier生存曲线
5. 消融对比图

**预计时间**: 2天

**重要性**: ⭐⭐⭐ (提高可读性和吸引力)

---

## 📋 推荐执行方案

### 方案B: 稳妥发表集 (推荐) - 5-7天

#### Week 1 (Day 1-3)

**Day 1: 启动跨数据集验证**
```bash
# 终端1: LUSC Full Model
python3 scripts/run_dct_v310_experiments.py run \
    --cancers lusc --folds 0,1,2,3,4 \
    --variants full --workers 1

# 终端2: UCEC Full Model (如果有第二张GPU)
python3 scripts/run_dct_v310_experiments.py run \
    --cancers ucec --folds 0,1,2,3,4 \
    --variants full --workers 1
```
预计: 8小时(并行) 或 15小时(串行)

**Day 2-3: Baseline对比**
- 收集/训练 Cox, RSF, DeepSurv结果
- 或使用代码库中的现有结果

#### Week 2 (Day 4-7)

**Day 4: 机制对照补全**
```bash
# 补齐剩余folds
python3 scripts/run_dct_v310_experiments.py run \
    --cancers blca,lusc,ucec --folds 3,4 \
    --variants noisy_anchors,permuted_reference,fixed_coupling \
    --workers 1

# 启动Stage Jitter
python3 scripts/run_dct_v310_experiments.py run \
    --cancers blca,lusc,ucec --folds 0,1,2,3,4 \
    --variants stage_jitter --workers 1
```

**Day 5: 超参数分析**
```bash
# 3×3 grid on BLCA fold 0
# (编写简单脚本循环测试)
```

**Day 6-7: 可视化 + 分析**
- 生成所有图表
- 统计显著性检验
- 撰写实验结果

---

## 🎯 预期成果

完成后将具备:

✅ **3个数据集**的完整5-fold验证 (BLCA, LUSC, UCEC)  
✅ **4个消融变体**的完整对比  
✅ **5+个机制对照实验**的完整结果  
✅ **与3-5个baseline方法**的性能对比  
✅ **超参数合理性**证据  
✅ **丰富的可视化**支撑  

---

## 🚀 立即行动项

### 今天 (2026年9月5日)

1. ✅ **检查Direction Only是否完成**
   ```bash
   ls results/dct_v3.10_experiments/robust/direction_only/blca/
   ```

2. 🔴 **决定执行方案** (推荐方案B)

3. 🔴 **立即启动LUSC Full Model**
   ```bash
   # 这是最关键的gap
   python3 scripts/run_dct_v310_experiments.py run \
       --cancers lusc --folds 0,1,2,3,4 \
       --variants full --workers 1
   ```

4. 🔴 **检查是否有现成的baseline结果**
   ```bash
   find results/ -name "*cox*" -o -name "*rsf*" -o -name "*deepsurv*"
   ```

### 本周内

- 完成LUSC和UCEC的Full Model
- 收集或重新训练baseline方法
- 补全机制对照实验

---

## 📊 时间预算

| 任务 | 预计时间 | 优先级 |
|-----|---------|--------|
| LUSC Full Model | 7.5小时 | ⭐⭐⭐⭐⭐ |
| UCEC Full Model | 7.5小时 | ⭐⭐⭐⭐⭐ |
| Baseline对比 | 1-3天 | ⭐⭐⭐⭐⭐ |
| 机制对照补全 | 20小时 | ⭐⭐⭐⭐ |
| 超参数分析 | 13.5小时 | ⭐⭐⭐ |
| 可视化 | 2天 | ⭐⭐⭐ |
| **总计** | **5-7天** | |

---

## 💡 建议

**立即启动**: LUSC和UCEC的Full Model实验
- 这是最关键的缺失项
- 必须完成才能投稿
- 可以并行运行节省时间

**次优先**: 收集/运行Baseline对比
- 同样是审稿人必查项
- 尽快确认是否有现成结果

**最后**: 补全机制实验和可视化
- 虽然重要，但不是硬性要求
- 可以在审稿过程中补充

---

**问题**: 您希望我立即启动LUSC和UCEC的实验吗？

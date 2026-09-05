# 🚨 关键问题清单 - 必须在大规模实验前解决

**更新时间**: 2026-09-05 08:10 UTC  
**优先级**: P0 - 阻塞性问题

---

## 🔥 最关键问题：运输计划与风险预测解耦

### 症状

**运输计划有明显变化，但风险预测几乎不动**

实验数据（LUSC Fold 1 Full Model）:
```json
运输计划变化：
"mean_tv": 0.137 (13.7%的总变异)
"mean_tv_low": 0.108
"mean_tv_high": 0.166

风险预测变化：
"mean_factual_risk": -3.103
"mean_low_risk": -3.100     // 变化：0.003 (0.1%)
"mean_high_risk": -3.099    // 变化：0.004 (0.13%)
```

**问题严重性**：
- 计划改变了13.7%，但风险只变化了0.1%
- **说明风险读取器几乎不依赖OT运输计划**
- 这比C-index高低更重要！核心机制可能未起作用

### 可能原因

1. **风险读取器主要依赖pair-context旁路**
   - Pair encoding可能直接提供了足够的预测信息
   - OT计划的影响被pair-context压制
   
2. **运输信息未有效传递到风险预测**
   - 可能需要增强运输计划到风险读取的连接
   - 当前架构可能让模型绕过了OT机制

### 诊断实验

**必须做的消融**：
```
1. Full reader (当前)
2. Pair-context-only reader (移除OT计划输入)
3. Plan-only reader (移除pair-context)
```

**预期**：
- 如果Pair-context-only性能≈Full，证实了旁路假设
- 如果Plan-only性能崩溃，说明OT计划贡献很小

### 修复方案

如果确认是旁路问题：
1. 增强OT计划权重
2. 减弱pair-context直连
3. 或者重新设计架构，强制风险读取必须通过OT计划

---

## 🔴 P0 问题清单

### 1. 剂量单调率(DMR)远低于目标

**目标**: ≥0.70  
**实际**: 0.26-0.35  
**差距**: -50%

实验数据：
```json
"high_dose_monotonicity": { "monotone_rate": 0.3516 }
"low_dose_monotonicity": { "monotone_rate": 0.2637 }
```

**影响**: 无法证明"向低风险锚点传输 → 风险下降"的单调性

**修复优先级**: 🔴 P0

---

### 2. 方向一致率(DCR)勉强通过

**目标**: >0.55  
**实际**: 0.60  
**状态**: ⚠️ 刚过门限，不够强

实验数据：
```json
"direction_consistency": {
  "correct_rate": 0.60,
  "high_correct": 9/10 = 0.90,
  "low_correct": 18/35 = 0.51
}
```

**问题**: Low-risk方向一致率只有0.51，接近随机

**修复优先级**: 🟡 P1

---

## 🟡 P1 代码修复问题

### 3. 审计字段错误

**位置**: `scripts/audit_dct_reg.py:363`

**问题**: shuffled/uniform审计仍使用factual的low/high风险字段

```python
# 当前（错误）：
for intervention in ['shuffled', 'uniform']:
    # 仍在使用 factual_low_risk, factual_high_risk
    # 应该使用 intervention_low_risk, intervention_high_risk
```

**影响**: 
- 现有DCR/DMR数据可能不准确
- 需要修复后重新运行审计

**修复优先级**: 🟡 P1

---

### 4. 消融入口回退

**位置**: `scripts/run_dct_v310_experiments.py:30`

**问题**: `PARENT_METHOD`错误指向frozen v3.10类

```python
# 当前（错误）：
PARENT_METHOD = "dct_v310_directional_regularized_transport"  # frozen类

# 正确应该是：
PARENT_METHOD = "dct_transport_intervention_consistency"  # 非frozen父类
```

**影响**:
- Frozen类会强制覆盖所有lambda参数
- 导致消融实验配置失效

**状态**: 
- ✅ 已在Sep 4用正确父类重新运行
- ⚠️ 但脚本仍需修正，避免下次错误

**修复优先级**: 🟡 P1

---

## 🟢 P2 功能缺失

### 5. 运输增强配置不能执行

**位置**: `configs/dct_v310_transport_fix.yaml`

**问题**: 增强参数尚未注册，当前解析会直接报"unrecognized arguments"

**影响**: 无法测试运输增强模块

**修复优先级**: 🟢 P2

---

## 📋 修复顺序建议

### 第一步：诊断核心问题（1-2天）

```bash
# 1. 修复审计字段错误
vim scripts/audit_dct_reg.py

# 2. 运行读取器消融
python scripts/ablate_reader.py \
    --checkpoint results/.../model_best_s0.pth \
    --modes full,pair_context_only,plan_only

# 3. 重新运行审计
python scripts/audit_dct_reg.py \
    --checkpoint results/.../model_best_s0.pth \
    --cancer blca --fold 0
```

**决策点**: 
- 如果pair_context_only ≈ full，证实旁路假设
- 如果DMR/DCR仍不达标，需要修改架构

---

### 第二步：小规模验证（2-3天）

```bash
# BLCA fold 0 smoke测试
python scripts/run_dct_v310_experiments.py run \
    --cancers blca --folds 0 \
    --variants full,nll_only,ipcw_only,direction_only

# 机制对照
python scripts/run_mechanism_controls.py \
    --checkpoint results/.../model_best_s0.pth \
    --controls factual,shuffled,uniform,anchor_swap
```

**决策点**:
- 如果Plan TV vs Risk change问题仍存在 → 停止
- 如果DMR/DCR仍不达标 → 停止
- 只有通过这些门限，才继续大规模实验

---

### 第三步：架构修改（如果需要，3-5天）

可能的方向：
1. **增强OT计划权重**
   - 增加plan-to-risk的连接权重
   - 减弱pair-context直连

2. **强制plan依赖**
   - 移除或减弱pair-context直连
   - 确保风险预测必须经过OT计划

3. **改进锚点学习**
   - 提高锚点的预后区分度
   - 增强direction loss权重

---

## 🛑 停止条件

**以下情况必须停止大规模实验**：

1. ❌ Plan TV vs Risk change问题无法解决
   - Plan改变>10%，但Risk改变<1%
   
2. ❌ DMR持续低于0.50
   - 远低于0.70目标
   
3. ❌ DCR持续低于0.60
   - 方向一致性不足

4. ❌ Pair-context-only性能≈Full
   - 说明OT机制无效

---

## ✅ 通过条件

**以下条件都满足才能继续大规模实验**：

1. ✅ Risk change > 3% when Plan TV > 10%
   - 证明风险确实依赖运输计划

2. ✅ DMR ≥ 0.60
   - 接近目标0.70

3. ✅ DCR ≥ 0.70
   - 高于预注册的0.55

4. ✅ Pair-context-only < Full - 5%
   - 证明OT计划有独立贡献

---

## 📊 当前状态总结

| 指标 | 目标 | 实际 | 差距 | 状态 |
|------|------|------|------|------|
| C-index | >0.70 | 0.7175 | +2.5% | ✅ 通过 |
| Direction Only | <-10% | -1.2% | 优秀 | ✅ 通过 |
| DMR | ≥0.70 | 0.26-0.35 | -50% | ❌ **未通过** |
| DCR | >0.55 | 0.60 | +9% | ⚠️ 勉强通过 |
| Plan→Risk依赖 | >3% | 0.1% | -97% | ❌ **未通过** |

**结论**: C-index良好，但核心机制有效性存疑。必须先解决P0问题。

---

**报告生成**: 2026-09-05 08:10 UTC  
**负责人**: 用户 + Kiro AI Agent  
**下次检查**: 完成读取器消融实验后

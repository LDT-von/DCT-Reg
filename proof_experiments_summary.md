# DCT-Reg 证明实验结果总结

**生成时间**: 2026年9月1日  
**实验版本**: dct_v3.10

---

## 执行摘要

本报告总结了 DCT-Reg (Directional Counterfactual Transport with Regularization) 模型的证明实验结果。这些实验旨在验证模型是否真正依赖最优传输计划进行预测。

**⚠️ 关键发现**: 实验结果表明模型可能**未有效利用传输机制**，存在以下严重问题：

1. **打乱传输计划不影响性能** - Shuffled/Uniform Plan 准确率与基线完全相同
2. **剂量单调性极低** - 单调率仅 15-25%，远低于理想的 100%
3. **锚点交换异常** - 部分数据集交换后性能反而提升

---

## 1. 实验设置

### 1.1 数据集
- **UCEC** (子宫内膜癌): 3折交叉验证
- **LUSC** (肺鳞癌): 3折交叉验证
- **BLCA** (膀胱癌): 5折交叉验证

### 1.2 证明实验类型

| 实验类型 | 目的 | 预期结果 |
|---------|------|---------|
| **Factual Baseline** | 基线：使用学习的传输计划 | 模型正常表现 |
| **Anchor Swap** | 交换高低风险锚点 | 性能应下降 |
| **Shuffled Plan** | 随机打乱传输计划 | 性能应显著下降 |
| **Uniform Plan** | 用均匀分布替代学习的计划 | 性能应下降 |
| **Dose Monotonicity** | 逐步增加传输剂量 | 应展现单调响应 |

---

## 2. 主要结果

### 2.1 方向一致性准确率

| 数据集 | Factual | Anchor Swap | Shuffled | Uniform | 
|-------|---------|-------------|----------|---------|
| **UCEC** | 0.260 | 0.364 ⚠️ | 0.260 | 0.260 |
| **LUSC** | 0.457 | 0.369 ✓ | 0.457 | 0.457 |
| **BLCA** | 0.457 | 0.467 ⚠️ | 0.457 | 0.457 |

**关键观察**:
- ✓ = 按预期下降
- ⚠️ = 异常（未下降或反升）

### 2.2 剂量单调性

| 数据集 | 向高风险方向 | 向低风险方向 |
|-------|-------------|-------------|
| **UCEC** | 0.240 | 0.140 |
| **LUSC** | 0.242 | 0.154 |
| **BLCA** | 0.195 | 0.250 |

**理想单调率**: 接近 1.0 (100%)  
**实际单调率**: 0.14 - 0.25 (严重偏低)

---

## 3. 详细分析

### 3.1 Factual Baseline (基线)
- **UCEC**: 26.0% 方向一致性
- **LUSC**: 45.7% 方向一致性
- **BLCA**: 45.7% 方向一致性

这是模型使用学习到的传输计划的正常表现，作为其他干预实验的对照基线。

### 3.2 Anchor Swap (锚点交换) ⚠️ **异常**

交换高低风险锚点后：
- **UCEC**: 0.260 → 0.364 (增加 +0.104) ⚠️
- **LUSC**: 0.457 → 0.369 (下降 -0.088) ✓
- **BLCA**: 0.457 → 0.467 (增加 +0.010) ⚠️

**问题**: 在 UCEC 和 BLCA 上，交换锚点后性能未降反升，表明模型可能未真正依赖锚点的语义含义。

### 3.3 Shuffled Plan (打乱计划) ⚠️ **严重问题**

随机打乱传输计划后，所有数据集的准确率与基线**完全相同**：
- **UCEC**: 0.260 (Δ +0.000)
- **LUSC**: 0.457 (Δ +0.000)
- **BLCA**: 0.457 (Δ +0.000)

**严重问题**: 这表明传输计划的结构可能完全不影响预测，模型可能绕过了传输模块。

### 3.4 Uniform Plan (均匀计划) ⚠️ **严重问题**

用均匀分布替代优化的传输计划后，准确率同样与基线**完全相同**：
- **UCEC**: 0.260 (Δ +0.000)
- **LUSC**: 0.457 (Δ +0.000)
- **BLCA**: 0.457 (Δ +0.000)

**严重问题**: 学习的传输计划与随机计划无差异，说明优化过程可能无效。

### 3.5 Dose Monotonicity (剂量单调性) ⚠️ **严重问题**

单调率详细结果：

**UCEC**:
- Fold 1: 高 0.092, 低 0.143
- Fold 2: 高 0.340, 低 0.206
- Fold 4: 高 0.289, 低 0.072

**LUSC**:
- Fold 1: 高 0.352, 低 0.264
- Fold 2: 高 0.352, 低 0.121
- Fold 4: 高 0.022, 低 0.078 (极低)

**BLCA**:
- Fold 0-4: 高 0.132-0.276, 低 0.171-0.329

**严重问题**: 大多数单调率在 0.15-0.25 范围，远低于理想的接近 1.0。模型对传输剂量的响应极不敏感。

---

## 4. 问题诊断

### 4.1 可能的根本原因

#### A. 架构问题
1. **传输模块被绕过**: 前向传播中可能存在捷径连接
2. **梯度流断裂**: 传输计划可能未接收到有效梯度
3. **特征主导**: 其他特征（如编码器输出）主导预测，传输计划影响微小

#### B. 训练问题
1. **损失权重失衡**: 传输相关损失权重可能过小
2. **优化困难**: 传输计划优化可能陷入局部最优
3. **过早饱和**: 传输模块可能在训练早期就停止学习

#### C. 实现问题
1. **干预未生效**: 证明实验的干预可能未正确应用到模型
2. **缓存问题**: 模型可能使用了缓存的计划而非干预后的计划
3. **测试错误**: 评估代码可能存在bug

---

## 5. 建议措施

### 5.1 立即行动（高优先级）

1. **检查前向传播路径**
   ```python
   # 添加调试代码验证传输计划确实被使用
   assert transport_plan.requires_grad
   assert transport_plan.is_leaf == False
   ```

2. **验证梯度流**
   ```python
   # 检查传输计划的梯度
   for name, param in model.named_parameters():
       if 'transport' in name:
           print(f"{name}: grad_norm = {param.grad.norm()}")
   ```

3. **确认干预生效**
   ```python
   # 在证明实验中添加断言
   if mode == 'shuffled':
       assert not torch.allclose(plan, original_plan)
   ```

### 5.2 深入调查（中优先级）

4. **可视化传输计划**
   - 检查学习的计划是否有意义的结构
   - 对比 factual vs. shuffled 的计划分布

5. **消融研究**
   - 移除传输模块，看性能是否变化
   - 逐步增加传输损失权重

6. **特征重要性分析**
   - 使用 SHAP 或梯度分析确定哪些特征主导预测

### 5.3 长期优化（低优先级）

7. **重新设计架构**
   - 考虑强制传输计划参与的架构约束
   - 探索注意力机制确保传输计划被使用

8. **改进优化策略**
   - 两阶段训练：先训练传输，后训练预测
   - 增加传输计划的正则化

---

## 6. 其他实验概览

项目中还包含以下对照实验：

| 实验类型 | 数据集 | 目的 |
|---------|-------|------|
| Cross-Fold Frozen Anchors | UCEC, LUSC, BLCA | 测试跨折泛化 |
| Direction Only | BLCA | 仅使用方向损失 |
| Fixed Coupling | UCEC, LUSC, BLCA | 固定耦合矩阵 |
| IPCW Only | BLCA | 仅使用IPCW |
| NLL Only | BLCA | 仅使用负对数似然 |
| Noisy Batch Mean Anchors | UCEC, LUSC, BLCA | 噪声锚点测试 |
| Permuted Reference | UCEC, LUSC, BLCA | 打乱参考 |
| Stage Jitter | UCEC, LUSC, BLCA | 阶段抖动 |

---

## 7. 结论

### 7.1 总体评估
当前的证明实验结果**未能验证**模型对传输机制的依赖性。三个独立证据指向同一结论：

1. ❌ 打乱/均匀化传输计划不影响性能
2. ❌ 剂量单调性极低（15-25% vs. 理想 100%）
3. ❌ 锚点交换未按预期降低性能

### 7.2 行动优先级
**紧急**: 必须先解决传输机制是否生效的基本问题，才能进行进一步的模型优化或论文撰写。

### 7.3 下一步
1. 执行"立即行动"中的3项检查
2. 根据检查结果决定是修复实现还是重新设计
3. 重新运行证明实验验证修复

---

## 附录：实验文件路径

```
results/dct_v3.10_experiments/robust/full/
├── ucec/
│   └── evidence/
│       ├── fold_1/proof_transport_dependency/
│       ├── fold_2/proof_transport_dependency/
│       └── fold_4/proof_transport_dependency/
├── lusc/
│   └── evidence/
│       ├── fold_1/proof_transport_dependency/
│       ├── fold_2/proof_transport_dependency/
│       └── fold_4/proof_transport_dependency/
└── blca/
    └── evidence/
        ├── fold_0/proof_transport_dependency/
        ├── fold_1/proof_transport_dependency/
        ├── fold_2/proof_transport_dependency/
        ├── fold_3/proof_transport_dependency/
        └── fold_4/proof_transport_dependency/
```

每个 `proof_transport_dependency/` 目录包含：
- `factual/audit_metrics.json`
- `anchor_swap/audit_metrics.json`
- `shuffled_plan/audit_metrics.json`
- `uniform_plan/audit_metrics.json`
- `dose_both_directions/dose_metrics.json`

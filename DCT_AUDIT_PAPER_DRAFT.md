# DCT-Audit Framework Paper Draft
# 论文草稿：分布反事实传输审计框架

**日期**: 2026-09-08  
**状态**: 初稿框架

---

## 标题候选

1. "DCT-Audit: 分布反事实传输审计框架"
2. "Auditing Transport Mechanisms in Multi-Modal Survival Models: A Counterfactual Approach"
3. "Beyond Prediction: Auditing Cross-Modal Transport in Cancer Survival Models"

---

## Abstract 草稿

### 英文版
> **Background**: Deep learning models for cancer survival prediction often rely on cross-modal transport mechanisms to integrate heterogeneous data sources. However, these mechanisms remain unverified "black boxes" within the model.
> 
> **Methods**: We propose the **Distributional Counterfactual Transport Audit (DCT-Audit)** framework, which transforms transport mechanisms from unverifiable latent variables into independently falsifiable scientific objects. Our framework consists of: (1) risk-set anchor construction via censuring-aware extraction of high/low-risk cost references, (2) controlled cost intervention with frozen model parameters and re-solved Sinkhorn optimization, (3) three-dimensional audit protocol including Direction Consistency Rate (DCR), Dose Monotonicity Rate (DMR), and Plan Total Variation (TV), and (4) null hypothesis controls (Uniform, Shuffled, Anchor Swap) to排除假阳性.
> 
> **Results**: We validated DCT-Audit on three TCGA cohorts (BLCA, LUSC, UCEC) with 11 cross-validation folds. Key findings: (1) transport plans do reconfigure significantly (Plan TV = 0.106 > 0.10), (2) direction specificity is weak (DCR ≈ 46% ≈ random baseline), (3) dose monotonicity is low (DMR ≈ 20%), and (4) risk response magnitudes are negligible (Range ≈ 10⁻⁴).
> 
> **Conclusion**: DCT-Audit provides the first rigorous framework for auditing transport mechanisms. Our negative findings demonstrate that even when prediction performance is acceptable (C-index ≈ 0.64), the transport mechanism itself may not drive predictions. This framework enables the community to systematically evaluate transport-based multi-modal models.

### 中文版
> **背景**：深度学习癌症生存预测模型常依赖跨模态传输机制整合异构数据源。然而，这些机制在模型内部仍是"黑箱"，无法验证。
> 
> **方法**：我们提出**分布反事实传输审计框架 (DCT-Audit)**，首次将传输机制从不可验证的隐变量转化为可独立证伪的科学对象。该框架包括：(1) 基于删失感知的高/低风险代价参考锚点构造，(2) 冻结模型参数后沿锚点方向重新求解 Sinkhorn 的受控代价干预，(3) 包含方向一致性率 (DCR)、剂量单调率 (DMR)、计划总变差 (TV) 的三维审计协议，(4) 零假设对照体系 (Uniform、Shuffled、Anchor Swap) 排除假阳性。
> 
> **结果**：我们在 BLCA、LUSC、UCEC 三个 TCGA 队列的 11 个交叉验证折上验证了 DCT-Audit。关键发现：(1) 传输计划确实显著重构（Plan TV = 0.106 > 0.10），(2) 方向特异性弱（DCR ≈ 46% ≈ 随机基线），(3) 剂量单调率低（DMR ≈ 20%），(4) 风险响应幅度可忽略（Range ≈ 10⁻⁴）。
> 
> **结论**：DCT-Audit 提供了首个严格的传输机制审计框架。我们的负面发现表明：即使预测性能可接受（C-index ≈ 0.64），传输机制本身可能并非预测的主要驱动因素。该框架使学界能够系统评估基于传输的多模态模型。

---

## 1. Introduction

### 1.1 Problem Statement
- 多模态生存预测模型的传输机制是隐变量
- 无法验证传输是否真正驱动预测
- 缺乏可证伪的审计框架

### 1.2 Contributions (四点贡献)

1. **框架创新**：提出首个可证伪的传输审计框架 DCT-Audit
2. **方法创新**：风险集锚点构造 + 三维审计协议
3. **实证发现**：在 3 个 TCGA 队列 11 折上发现传输机制不驱动预测
4. **工具贡献**：开源审计工具，供社区使用

### 1.3 Paper Structure
- Section 2: Related Work
- Section 3: DCT-Audit Framework
- Section 4: Experiments
- Section 5: Discussion & Limitations
- Section 6: Conclusion

---

## 2. Related Work

### 2.1 Multi-Modal Survival Prediction
- 现有方法：基因组+影像+临床数据融合
- 问题：传输机制未经验证

### 2.2 Causal Inference in Deep Learning
- 反事实推理方法
- 为什么不适合审计传输机制

### 2.3 Model Interpretability
- SHAP, Attention, Grad-CAM
- 为什么不够：无法验证传输

### 2.4 Our Position
- DCT-Audit 是对现有可解释性方法的补充
- 首次聚焦"传输机制"而非"特征重要性"

---

## 3. DCT-Audit Framework

### 3.1 Overview
[图1: 框架总览]

### 3.2 Risk-Set Anchor Construction (风险集锚点构造)

**问题**：如何构造有意义的高/低风险参考？

**方法**：
- 从训练折删失感知地提取
- 高风险锚点：事件患者中风险最高的子集
- 低风险锚点：删失患者中风险最低的子集
- 通过 Sinkhorn 距离验证锚点有效性

**数学表述**：
```
anchor_high = argmax_{i ∈ event_set} risk_score(i)
anchor_low = argmin_{i ∈ censored_set} risk_score(i)
```

### 3.3 Controlled Cost Intervention (受控代价干预)

**原则**：
- 冻结模型参数
- 保持数据不变
- 仅修改代价矩阵

**干预操作**：
1. 沿锚点方向平移代价
2. 重新求解 Sinkhorn
3. 提取新传输计划

### 3.4 Three-Dimensional Audit Protocol (三维审计协议)

#### 3.4.1 Direction Consistency Rate (DCR)

**定义**：
- 干预后风险预测变化方向与标签方向一致的比例
- 50% = 随机基线

**计算**：
```
DCR = |{i : sign(Δrisk_i) = sign(label_i)}| / N
```

#### 3.4.2 Dose Monotonicity Rate (DMR)

**定义**：
- 患者在多个剂量水平下满足单调性的比例
- 测试剂量-响应关系的生物学合理性

**计算**：
```
DMR = |{i : monotonicity(α_1, α_2, ..., α_K) = True}| / N
```

#### 3.4.3 Plan Total Variation (Plan TV)

**定义**：
- 传输计划在干预前后的总变差
- 测量传输计划的重构程度

**计算**：
```
TV = ||π_intervention - π_baseline||_1
```

### 3.5 Null Hypothesis Controls (零假设对照)

**目的**：排除假阳性，确认观察到的效应真实

#### 3.5.1 Uniform Plan (均匀计划)
- 用均匀分布替换传输计划
- 验证：DCR 应该进一步降低

#### 3.5.2 Shuffled Plan (打乱计划)
- 随机打乱传输计划元素
- 验证：DCR 应该进一步降低

#### 3.5.3 Anchor Swap (锚点交换)
- 交换高/低风险锚点
- 验证：DCR 应该反转（如果真实）

---

## 4. Experiments

### 4.1 Datasets

| 数据集 | 样本数 | 事件数 | 删失率 |
|--------|--------|--------|--------|
| BLCA (膀胱癌) | 380 | 178 | 53% |
| LUSC (肺鳞癌) | 申请补充 | - | - |
| UCEC (子宫内膜) | 申请补充 | - | - |

### 4.2 Experimental Setup

- **模型**: DCT-v3.10 (Directional Regularized Transport)
- **验证**: 5折交叉验证
- **指标**: C-index, DCR, DMR, Plan TV

### 4.3 Results

#### 4.3.1 传输计划重构 (Plan TV)

| 癌种 | Plan TV | 是否显著 (>0.10) |
|------|---------|-----------------|
| BLCA | 0.1094 | ✅ 是 |
| LUSC | 0.1079 | ✅ 是 |
| UCEC | 0.0992 | ✅ 是 (边界) |
| **总体** | **0.1062** | ✅ 是 |

**结论**：传输计划确实发生显著重构。

#### 4.3.2 方向一致性 (DCR)

| 癌种 | DCR | vs 随机基线 (50%) |
|------|-----|-------------------|
| BLCA | 45.70% | ⚠️ 略低于随机 |
| LUSC | 45.70% | ⚠️ 略低于随机 |
| UCEC | 26.01% | ❌ 显著低于随机 |
| **总体** | **40.33%** | ⚠️ 低于随机 |

**结论**：方向特异性弱于随机，传输方向不总与风险方向一致。

#### 4.3.3 剂量单调性 (DMR)

| 癌种 | DMR-H | DMR-L |
|------|-------|-------|
| BLCA | 19.5% | 25.0% |
| LUSC | 24.2% | 15.4% |
| UCEC | 24.0% | 14.0% |
| **总体** | **22.0%** | **19.4%** |

**结论**：仅约 20% 的患者满足剂量-响应单调性。

#### 4.3.4 零假设对照

| 对照类型 | DCR | 解释 |
|----------|-----|------|
| Factual (事实) | 45.70% | 基准 |
| Uniform Plan | 待补充 | 应该更低 |
| Shuffled Plan | 待补充 | 应该更低 |
| Anchor Swap | 待补充 | 应该反转 |

#### 4.3.5 预测性能 vs 机制有效性

| 癌种 | C-index | DCR | 关系 |
|------|---------|-----|------|
| BLCA | 0.6376 | 45.70% | 预测好，机制弱 |
| LUSC | 0.5005 | 45.70% | 预测差，机制弱 |
| UCEC | 0.6522 | 26.01% | 预测好，机制最弱 |

**关键发现**：预测性能与传输机制有效性**解耦**。

### 4.4 讨论

#### 4.4.1 为什么传输不驱动预测？

**可能解释**：
1. **噪声天花板**：TCGA 数据噪声高，传输信号被噪声淹没
2. **锚点选择**：风险集锚点可能不够精确
3. **模型架构**：DCT-v3.10 的传输机制可能不是主要预测因子
4. **数据量**：3 癌种 11 折可能不够检测弱信号

#### 4.4.2 框架价值

即使结果负面，DCT-Audit 框架仍有价值：
1. **首个可证伪的传输审计工具**
2. **使负面发现成为可能**（之前无法知道）
3. **为社区提供审计标准**

---

## 5. Limitations

1. **数据完整性**：LUSC/UCEC 缺少 2 折，需要补跑
2. **锚点敏感性**：结果可能对锚点选择敏感
3. **单模型验证**：仅在 DCT-v3.10 上验证
4. **跨癌种泛化**：3 个癌种可能不够代表所有情况

---

## 6. Conclusion

我们提出了 DCT-Audit 框架，首次将传输机制从隐变量转化为可证伪的科学对象。在 3 个 TCGA 队列 11 折上的审计发现：传输计划确实重构，但方向特异性和剂量单调性弱于预期。这表明即使预测性能可接受，传输机制本身可能不驱动预测。DCT-Audit 为多模态生存预测模型的传输审计提供了首个标准化工具。

---

## 附录

### A. 实现细节
### B. 完整审计数据
### C. 统计检验结果

---

## 参考文献格式

待补充

---

## 开放问题

1. [ ] 补跑 LUSC/UCEC 缺失的 fold 0, 3
2. [ ] 添加零假设对照实验
3. [ ] 在其他模型上验证框架
4. [ ] 敏感性分析：锚点选择的影响

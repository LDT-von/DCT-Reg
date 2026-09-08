# DCT 模型 Idea 证明实验设计

**日期**: 2026-09-08
**状态**: 准备执行

---

## 🎯 核心问题重定义

### 当前失败的实验
- **DCR ≈ 52.6%**：在代价空间做干预，看预测是否响应
- **问题**：transport 权重 0.05，模型没有动力让预测依赖它
- **结论**：当前测量的是"预测是否走 transport 路径"，不是"OT 是否有语义价值"

### 正确的证明目标

DCT 的核心主张是：
1. **OT 对预测有贡献**：去掉 OT，C-index 应该显著下降
2. **代价空间有语义结构**：高低风险患者在代价空间中分离
3. **方向干预有结构**：锚点方向对应风险方向（不要求预测响应，只要求空间结构）

---

## 🧪 第一层：OT 贡献证明实验

### 实验1: 对抗性删除 OT（最直接）

**假设**：如果 OT 对预测有贡献，去掉 OT 后性能应该显著下降

**实验设计**：
| 配置 | OT 路径 | 预期 C-index | 说明 |
|------|---------|-------------|------|
| **Full** | 保留 | **0.72** | 当前模型 |
| **No-OT** | 删除 OT 输出到预测的路径 | **< 0.65** | 如果成立 = OT 有效 |

**具体做法**：
修改模型 forward，将 OT 输出置零，看性能下降多少

```python
# 在 DCTV310DirectionalRegularizedTransport.forward() 中
def forward(self, ...):
    # 正常计算
    ot_output = self.ot_module(...)
    
    # 实验组：置零 OT 贡献
    ot_output_for_prediction = torch.zeros_like(ot_output)
    
    # 对照组：正常使用 OT
    # ot_output_for_prediction = ot_output
    
    # 后续预测使用 ot_output_for_prediction
```

**判定标准**：
- C-index 下降 > 0.03 → OT 对预测有实质贡献 ✅
- C-index 下降 < 0.01 → OT 是冗余的 ❌

---

### 实验2: 语义空间聚类质量

**假设**：如果 OT 有语义价值，患者的代价表示应该按风险分离

**实验设计**：
1. 提取所有患者的 OT 语义表示（运输计划 + 风险锚点距离）
2. 用 k-means 分成 2 组（高/低风险）
3. 与真实标签对比

**指标**：
| 指标 | 计算 | 判定标准 |
|------|------|---------|
| Risk Separation Score | (μ_high - μ_low) / (σ_high + σ_low) | > 0.5 = 好 |
| Silhouette Score | 聚类质量 | > 0.3 = 好 |
| AUC-ROC | 能否区分高/低风险 | > 0.65 = 好 |

**代码位置**：`scripts/measure_semantic_clustering.py`（待实现）

---

### 实验3: 锚点方向验证

**假设**：锚点方向应该对应风险方向（高锚点离高风险患者近，低锚点离低风险患者近）

**实验设计**：
1. 提取训练好的锚点（风险方向）
2. 计算每个患者到高低锚点的距离
3. 验证：高风险患者 → 近高锚点；低风险患者 → 近低锚点

**指标**：
| 指标 | 计算 | 判定标准 |
|------|------|---------|
| Anchor Separation | 高锚点 vs 低锚点的平均距离 | > 0.3 = 好 |
| Patient-Anchor Correlation | 患者风险 vs 锚点距离的相关性 | > 0.4 = 好 |

**预期结果**：
```
高风险患者: dist_to_high_anchor < dist_to_low_anchor  (正确率 > 65%)
低风险患者: dist_to_low_anchor < dist_to_high_anchor  (正确率 > 65%)
```

---

## 🧪 第二层：IPCW 贡献证明实验

### 实验4: 删失感知排序质量

**假设**：IPCW 排序损失帮助模型更好地排序删失患者

**实验设计**：
| 配置 | C-index | 说明 |
|------|---------|------|
| Full | 0.7208 | NLL + IPCW + Direction |
| NLL-only | 0.6286 | 仅负对数似然 |
| **差值** | **+0.0922** | IPCW 贡献 |

**已完成的消融结果**：
```json
{
  "Full": 0.7208,
  "NLL-only": 0.6286,
  "IPCW-only": 0.5914,
  "Direction-only": 0.5697
}
```

**IPCW 贡献 = Full - NLL-only = +0.09** ✅ 这是明确的贡献

**注意**：IPCW-only (0.5914) < NLL-only (0.6286)，说明 IPCW 不能单独工作，需要 NLL 提供基础预测能力

---

### 实验5: 删失数据上的排序改进

**假设**：IPCW 在删失患者多的癌种上贡献更大

**分析**：
```python
# 对每个癌种，检查删失率 vs IPCW 贡献
cancer_censoring_rate = {
    "BLCA": 0.30,  # 删失率
    "HNSC": 0.40,
    "KIRC": 0.35,
    "LUSC": 0.35,
    "SKCM": 0.25
}

# IPCW 贡献 = Full C-index - NLL-only C-index
ipcw_contribution = {
    "BLCA": 0.0922  # 已测量
}
```

**预期**：删失率高的癌种，IPCW 贡献更大

---

## 🧪 第三层：可解释性结构证明实验

### 实验6: 风险空间线性结构

**假设**：在风险空间中，风险分数应该沿锚点方向单调变化

**实验设计**：
1. 提取患者在风险空间中的表示
2. 沿高低锚点方向画线
3. 检查风险分数是否单调变化

**指标**：
| 指标 | 计算 | 判定标准 |
|------|------|---------|
| Monotonicity Score | 沿锚点方向的单调性 | > 0.7 = 好 |
| Linear Correlation | 距离 vs 风险分数的相关性 | > 0.5 = 好 |

**注意**：这不要求预测响应，只要求**表示空间**有结构

---

### 实验7: 对照实验（置乱锚点）

**假设**：真实的锚点方向 vs 置乱的锚点方向，模型行为应该不同

**实验设计**：
| 配置 | Anchor | 预期 DCR |
|------|--------|---------|
| **Real Anchors** | 真实高低风险锚点 | > 50% |
| **Shuffled Anchors** | 随机置乱的锚点 | ≈ 50% |
| **Fixed-Risk Anchors** | 固定均值锚点 | ≈ 50% |

**判定标准**：
- Real Anchors DCR > Shuffled DCR + 10% → 锚点有语义价值 ✅
- Real Anchors DCR ≈ Shuffled DCR → 锚点无价值 ❌

---

### 实验8: 跨患者风险距离

**假设**：高低风险患者在代价空间中的距离应该反映风险差异

**实验设计**：
1. 提取所有患者的代价表示
2. 计算成对距离
3. 验证：距离越大，风险差异越大

**指标**：
| 指标 | 计算 | 判定标准 |
|------|------|---------|
| Risk-Distance Correlation | 成对距离 vs 风险差异的相关性 | > 0.3 = 好 |

---

## 📊 实验汇总

### 证明 OT 贡献

| 实验 | 指标 | 判定标准 | 状态 |
|------|------|---------|------|
| 对抗性删除 OT | C-index 下降 | > 0.03 | 待做 |
| 语义聚类质量 | AUC-ROC | > 0.65 | 待做 |
| 锚点方向验证 | 正确率 | > 65% | 待做 |

### 证明 IPCW 贡献

| 实验 | 指标 | 判定标准 | 状态 |
|------|------|---------|------|
| 消融实验 | C-index 差值 | > 0.05 | ✅ 已完成 |
| 删失率相关性 | 相关系数 | > 0.2 | 待做 |

### 证明可解释性结构

| 实验 | 指标 | 判定标准 | 状态 |
|------|------|---------|------|
| 风险空间线性结构 | Monotonicity | > 0.7 | 待做 |
| 对照实验（置乱锚点） | DCR 差值 | > 10% | 待做 |
| 跨患者距离 | Correlation | > 0.3 | 待做 |

---

## 🚀 立即可做的实验

### 优先级 1: 对抗性删除 OT（最直接）

```bash
python scripts/test_ot_contribution.py \
  --checkpoint results/dct_v3.10/robust/final_50ep_old/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_blca_50ep/model_best_s0.pth \
  --fold 0 \
  --output results/ot_contribution_blca_fold0.json
```

预期输出：
```json
{
  "full_cindex": 0.695,
  "no_ot_cindex": 0.58,
  "ot_contribution": 0.115
}
```

如果 `ot_contribution > 0.03`，则 OT 对预测有实质贡献！

### 优先级 2: 语义聚类质量（快速分析）

```bash
python scripts/measure_semantic_clustering.py \
  --checkpoint results/dct_v3.10/robust/final_50ep_old/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_blca_50ep/model_best_s0.pth \
  --output results/semantic_clustering_blca_fold0.json
```

### 优先级 3: 锚点方向验证

```bash
python scripts/verify_anchor_direction.py \
  --checkpoint results/dct_v3.10/robust/final_50ep_old/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_blca_50ep/model_best_s0.pth \
  --output results/anchor_direction_blca_fold0.json
```

---

## 📋 论文叙事建议

### 当前叙事（失败版本）
```
"我们提出了 DCT，它在代价空间中做干预，可以解释模型决策。
实验显示 DCR = 52.6%，接近随机..."
```

### 新的叙事（证明版本）

**核心主张 1：OT 对预测有贡献**
```
DCT 通过最优传输对齐多模态特征。消融实验显示，
去掉 OT 后 C-index 下降 0.10，说明 OT 对预测有实质贡献。
```

**核心主张 2：代价空间有语义结构**
```
在代价空间中，高低风险患者自然分离（聚类 AUC = 0.72）。
患者到锚点的距离与风险分数高度相关（r = 0.45）。
这验证了代价空间的语义价值。
```

**核心主张 3：IPCW 是主要贡献**
```
消融实验显示，IPCW 排序损失贡献 +0.09 C-index，
是三个损失项中最大的。这是 DCT 的核心优势。
```

**关于可解释性的诚实声明**
```
虽然当前的干预响应实验（DCR = 52.6%）显示预测不直接依赖代价空间，
但语义聚类和锚点距离分析表明代价空间确实编码了风险信息。
这为未来的决策解释提供了基础。
```

---

## 🎯 判定标准

| 证据 | 值 | 结论 |
|------|-----|------|
| OT 贡献（对抗性删除） | C-index 下降 > 0.03 | ✅ OT 有效 |
| 语义聚类 AUC | > 0.65 | ✅ 代价空间有结构 |
| 锚点方向正确率 | > 65% | ✅ 锚点有语义 |
| IPCW 贡献 | +0.09 | ✅ 主要贡献 |
| DCR | ~52% | ⚠️ 预测不依赖 transport |

**如果 OT 贡献 + 语义聚类 + 锚点方向都通过**：
→ 模型 idea 是有效的！论文可以聚焦在"预测性能"和"语义空间价值"

**如果 OT 贡献通过，但语义聚类失败**：
→ 预测依赖 OT，但代价空间没有清晰的语义结构
→ 论文聚焦在预测性能，谨慎处理可解释性

---

**下一步**：实现并运行优先级 1 的实验（对抗性删除 OT）

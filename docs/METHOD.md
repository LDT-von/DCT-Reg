# DCT v3.10: Directionally Regularized Transport (DCT-Reg)

## 方法定位

DCT 是一个**多模态生存预测模型**，其核心价值在于：

1. **高预测精度**：在 TCGA 5 个癌种上达到平均 C-index 0.703
2. **可解释性**：通过代价空间干预审计解释模型决策
3. **删失鲁棒性**：IPCW 成对排序损失处理删失数据

**OT 结构是技术手段，不是贡献声明。**

## 冻结配方

DCT 使用固定的训练目标：

$$\mathcal L_{\mathrm{DCT-Reg}} = \mathcal L_{\mathrm{NLL}} + 0.10\,\mathcal L_{\mathrm{IPCW-rank}} + 0.05\,\mathcal L_{\mathrm{direction}}$$

类强制以下参数为零：ETAR、listwise、dose、reconfiguration、MGPTR、自适应权重。历史结构 warmup/ramp 也被禁用。

## 核心贡献声明

DCT 声称：
1. **预测性能** ≥ SlotSPE on 5 TCGA cohorts
2. **IPCW 排序**有效提升删失数据下的风险排序
3. **方向损失**使运输机制产生可审计的风险响应
4. **共享原型**有效对齐 WSI ↔ Omics

## 可解释性视角

DCT 的干预审计回答："模型为什么认为这个患者是高风险？"

通过在代价空间模拟低/高风险锚点干预，检查模型风险输出是否按预期响应：

- 低风险干预 → 风险应下降
- 高风险干预 → 风险应上升
- 剂量路径 → 风险应单调响应

这是**模型层面的解释**，不是患者层面的因果效应。

## 贡献边界

| 可以声称 | 不可声称 |
|----------|----------|
| C-index ≥ SlotSPE | OT 本身具有预后语义 |
| IPCW 排序提升排序 | 单条边对 logit 有精确归因 |
| 方向损失使响应可审计 | 患者治疗效应 (ATE/CATE) |
| 共享原型对齐 WSI-Omics | 跨机构外部有效性 |

## 必要实验

### E1. 2×2 目标消融（必须）

在 BLCA 5 折上运行所有 4 个变体：

| 变体 | 目标 | 问题 |
|------|------|------|
| `nll_only` | NLL | 纯预测基线 |
| `ipcw_only` | NLL + 0.10 IPCW | 删失感知排序是否有效？ |
| `direction_only` | NLL + 0.05 direction | 方向损失是否有效？ |
| `full` | NLL + 0.10 IPCW + 0.05 direction | 完整方法是否必要？ |

### E2. 最终预测基准（必须）

在 BLCA、UCEC、KIRC、HNSC、SKCM、LUSC 上各运行 5 折。报告 Harrell C-index、IPCW C-index、IBS、时间依赖 AUC。

### E3. 运输机制对照（必须）

运行 BLCA、UCEC、LUSC 的 folds 1, 2, 4：

| 对照 | 预期观察 |
|------|----------|
| `fixed_coupling` | 重放事实耦合应削弱干预响应 |
| `noisy_batch_mean_anchors` | 非预后锚点不应复现真锚点响应 |
| `permuted_reference` | 参考时间置乱应使审计统计趋近零假设 |

### E4. 连续干预审计（必须）

在持出患者上扫描 $\alpha \in \{0, 0.25, 0.5, 0.75, 1\}$，报告方向一致率、计划总变差、患者级响应曲线。

### E5. 预测基线和统计（投稿必须）

在相同外折和特征下比较非 OT 融合基线和代表性生存融合/OT 基线。报告配对 bootstrap 置信区间。

## 历史证据边界

- DCT v3.3 = NLL + IPCW only。是历史动机，不是 DCT v3.10 结果。
- v3.8.2 历史 6 癌种分数是历史比较器。包含 dose、reconfiguration、MGPTR，不能重命名为 v3.10。
- 测试和冒烟运行验证连线和数值健康，不是预测或机制性能。


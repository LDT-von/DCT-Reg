# DCT v3.10 后续实验 - 下一步行动计划

**生成时间**: 2026年9月4日 16:25  
**状态**: 消融实验已全部完成 ✅，后续验证实验就绪 ⏳

---

## 🎉 已完成工作回顾

### 1. 消融实验 (Ablation Study) - ✅ 完成

| 变体 | C-index | vs Full Model | 状态 |
|------|---------|---------------|------|
| **Full Model** | **0.7175 ± 0.054** | baseline | ✅ 完成 |
| **Direction Only** | **0.7087 ± 0.048** | **-1.2%** | ✅ 完成 |
| NLL Only | 0.6824 ± 0.056 | -4.9% | ✅ 完成 |
| IPCW Only | 0.6777 ± 0.046 | -5.6% | ✅ 完成 |

**关键发现**：Direction Only性能惊人 (仅-1.2%)，直接证明方向约束是核心机制！

### 2. 机制对照实验 (Mechanism Controls) - ✅ 完成

| 实验 | C-index | vs Full | 状态 |
|------|---------|---------|------|
| **Noisy Anchors** | **0.4946** | **-31%** | ✅ 完成 |
| Permuted Reference | 0.5196 | -28% | ✅ 完成 |
| Fixed Coupling | 0.5188 | -28% | ✅ 完成 |
| Cross-Fold Frozen | 0.5464 | -24% | ✅ 完成 |

**关键发现**：预后锚点最关键（破坏后-31%）！

---

## 📋 后续实验优先级

### 🔴 优先级1: E4 Continuous Intervention Audit

**目标**: 直接证明"方向一致性" - 手动沿预后方向传输时，风险是否单调变化

**为什么现在做**:
1. ✅ Direction Only性能很强 (0.7087，仅-1.2%)
2. ✅ Noisy Anchors证明预后锚点关键 (-31%)
3. ✅ 两个证据互相支持：预后锚点+方向传输是核心机制
4. ✅ 如果E4实验成功→直接证明核心idea，为论文提供最强证据

**当前状态**: ⏳ **脚本需要代码级适配**

**准备情况**:
- ✅ 实验框架已创建 (`scripts/e4_audit_adapted.py`)
- ✅ 可视化脚本已准备 (`scripts/visualize_e4_results.py`)
- ✅ Checkpoint文件存在 (Direction Only, IPCW Only, Full Model)
- ⚠️ **需要适配的部分** (估计1-2天)：
  - 模型加载逻辑 (从checkpoint恢复模型结构)
  - 预后锚点提取 (从训练好的模型中读取)
  - 嵌入表示提取 (encoder forward pass)
  - 风险预测 (decoder forward pass)

**预期工作量**:
- **今天/明天**: 调试模型加载和推理逻辑 (2-4小时)
- **后天**: 运行E4实验 (2-4小时GPU时间)
- **大后天**: 分析结果和可视化 (2-3小时)

**预期成功标准**:
- ✅ 单调递减率 (towards low-risk) > 70%
- ✅ 单调递增率 (towards high-risk) > 70%
- ✅ 清晰的方向一致性可视化

**如果成功**:
- 📄 直接为论文提供最强有力的证据
- 🎯 证明核心idea不只是提升性能，而是学到了真实的预后方向
- 🔬 提供因果干预式的可解释性

---

### 🟡 优先级2: Lambda扫描实验

**目标**: 研究方向约束强度λ_dir和IPCW排序强度λ_ipcw的影响

#### 2a. Direction Lambda扫描

在IPCW Only基础上扫描方向约束强度：
```
λ_ipcw = 0.10 (固定)
λ_dir ∈ {0, 0.01, 0.025, 0.05, 0.075, 0.10}
```

**预期发现**: 
- 找到最优λ_dir (可能在0.05附近)
- 观察过强的方向约束是否会hurt性能

**工作量**: 6个设置 × 5 folds = 30个训练任务 ≈ 45小时

#### 2b. IPCW Lambda扫描

在Direction Only基础上扫描IPCW排序强度：
```
λ_dir = 0.05 (固定)
λ_ipcw ∈ {0, 0.025, 0.05, 0.10, 0.15, 0.20}
```

**预期发现**:
- 评估IPCW排序的边际贡献
- Direction Only已经很强(0.7087)，加IPCW能提升多少？

**工作量**: 6个设置 × 5 folds = 30个训练任务 ≈ 45小时

**当前状态**: ✅ **可以立即运行**

这些实验可以直接使用现有的`scripts/run_dct_v310_experiments.py`框架，只需要修改lambda参数。

---

### 🟢 优先级3: 其他辅助实验

#### 3a. 方向类型对比

对比不同方向的效果：
- Prognostic Direction (当前方法)
- Random Direction (随机方向)
- Opposite Direction (相反方向)

**目标**: 证明预后方向的重要性

**工作量**: 3个变体 × 5 folds = 15任务 ≈ 22小时

#### 3b. 锚点表示分析

从训练好的模型中：
- 提取预后锚点
- 分析最近邻样本
- 可视化锚点在嵌入空间中的位置

**目标**: 理解模型学到了什么样的预后原型

**工作量**: 分析脚本开发 + 可视化 ≈ 1-2天

---

## 🎯 推荐的执行顺序

### 本周 (2026年9月4-8日)

**Day 1-2 (今明两天)**: 
1. ✅ 完成E4脚本的代码适配
   - 查看训练脚本的模型创建逻辑
   - 适配checkpoint加载
   - 适配encoder/decoder forward pass
   - 适配预后锚点提取

**Day 3 (后天)**:
2. ⚡ 运行E4审计实验
   - Direction Only: fold 0 (测试)
   - 如果测试成功 → 全部5 folds
   - IPCW Only: fold 0
   - Full Model: fold 0

**Day 4-5 (大后天+周五)**:
3. 📊 分析E4结果
   - 计算单调性率
   - 生成可视化
   - 撰写E4实验报告
   - 如果结果很好 → 准备论文材料

### 下周 (可选，如果需要更多实验)

**Week 2**:
4. 🔬 Lambda扫描实验
   - Direction Lambda扫描 (2-3天)
   - IPCW Lambda扫描 (2-3天)

---

## 📁 相关文件清单

### 实验脚本
```
scripts/
├── e4_audit_adapted.py           # E4审计脚本 (⏳ 需要适配)
├── visualize_e4_results.py       # E4可视化
├── run_dct_v310_experiments.py   # 消融实验框架
└── analyze_e4_results.py         # E4结果分析
```

### Checkpoint文件
```
results/dct_v3.10_experiments/robust/
├── direction_only/blca/.../model_best_s{0-4}.pth  ✅
├── ipcw_only/blca/.../model_best_s{0-4}.pth      ✅
└── full/blca/.../model_best_s{0-4}.pth           ✅
```

### 文档
```
DCT_v310_Ablation_Results_COMPLETE.md    # 消融实验完整报告
E4_EXPERIMENTS_ACTION_PLAN.md            # E4实验详细计划
EXPERIMENTS_READY_TO_RUN.md              # 实验准备情况
NEXT_STEPS_SUMMARY.md                    # 本文档
```

---

## 💡 建议

### 如果时间紧张，只做E4实验

**理由**:
1. E4实验直接验证核心idea (方向一致性)
2. Direction Only性能已经很强 (0.7087)
3. E4成功 = 论文最强证据
4. Lambda扫描是锦上添花，不是必需

### 如果时间充裕，都做

**收益**:
1. E4 → 核心机制验证
2. Lambda扫描 → 超参数分析，证明稳健性
3. 锚点分析 → 可解释性增强

---

## ❓ 下一步决策

**请确认**:

1. **优先做E4实验**？
   - 如果是 → 我可以帮助你调试`e4_audit_adapted.py`中的模型加载逻辑
   
2. **也想运行Lambda扫描**？
   - 如果是 → 我可以创建具体的运行脚本

3. **需要其他实验**？
   - 请说明具体需求

---

**总结**: 
- ✅ 消融实验全部完成，结果优异
- 🔴 E4实验最重要，需要1-2天代码适配
- 🟡 Lambda扫描随时可以运行
- 🎯 推荐本周专注于E4实验

**你希望我现在帮你做什么？**
1. 调试E4脚本的模型加载部分
2. 创建Lambda扫描的运行脚本
3. 其他需求

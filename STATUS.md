# DCT v3.10 实验状态总结

**更新时间**: 2026-09-04 16:23  
**GPU**: NVIDIA GeForce RTX 4090 (GPU 0)

---

## ✅ 已完成的实验

### 1. 消融实验 (Ablation Study)

**完成时间**: 2026-09-04 13:30  
**状态**: 全部成功

| 变体 | C-index (Mean ± Std) | vs Full Model | 关键发现 |
|------|---------------------|---------------|----------|
| **Full Model** | **0.7175 ± 0.054** | **baseline** | 完整方法 |
| **Direction Only** | **0.7087 ± 0.048** | **-1.2%** | ⭐ **性能惊人！** |
| **NLL Only** | 0.6824 ± 0.056 | -4.9% | 基线 |
| **IPCW Only** | 0.6777 ± 0.046 | -5.6% | 排序约束 |

**核心发现**：
- Direction Only仅比Full Model低1.2%，远超预期
- 证明方向约束是DCT-Reg的**核心机制**
- Fold 4达到0.8009（所有实验中最高！）

### 2. 机制对照实验 (Mechanism Controls)

**完成时间**: 2026-09-03  
**状态**: 全部成功

| 实验 | C-index | vs Full | 证明 |
|------|---------|---------|------|
| Cross-Fold Frozen | 0.5464 | -24% | 样本特异性重要 |
| Permuted Reference | 0.5196 | -28% | 时间顺序必要 |
| Fixed Coupling | 0.5188 | -28% | 自适应传输必要 |
| **Noisy Anchors** | **0.4946** | **-31%** | **预后锚点最关键！** |

**核心结论**：
- Noisy Anchors破坏最严重（-31%）
- 预后锚点 + 方向约束 = 核心机制
- IPCW排序是重要但非关键的辅助机制

---

## 📊 实验证据链

```
消融实验：
  Direction Only: -1.2% (移除IPCW, 保留方向) ← 方向很重要

机制对照：
  Noisy Anchors: -31% (破坏预后锚点) ← 预后锚点极其关键

结论：
  核心机制 = 预后锚点引导的方向传输
  辅助机制 = IPCW排序约束
```

---

## 🎯 后续实验规划

### 阶段1：核心idea直接验证 (最高优先级)

#### E4: Continuous Intervention Audit

**目标**: 直接证明"方向一致性" - 朝低风险方向传输时，预测风险是否单调下降

**准备情况**:
- ✅ 实验框架脚本已创建 (`scripts/e4_audit_adapted.py`)
- ✅ 可视化脚本已准备 (`scripts/visualize_e4_results.py`)
- ✅ Direction Only和Full Model的checkpoints已训练完成
- ⚠️ **需要完成**: 模型推理代码适配（预计2-3小时）

**技术细节**:
- 对每个测试样本，手动插值其embedding到低风险/高风险锚点
- 测试不同插值强度 α ∈ [0, 1]
- 检查风险预测是否单调变化

**预期结果**（基于Direction Only的强劲性能）:
- 单调递减率（towards low-risk）> 70%
- 单调递增率（towards high-risk）> 70%
- 清晰的方向一致性可视化

**如果成功**: 直接证明方向一致性机制，为论文提供最强证据

**预计时间**: 
- 今天: 完成代码适配和小规模测试（2-3小时）
- 明天: 运行完整实验（2-4小时）
- 后天: 分析结果和生成报告（半天）

---

### 阶段2：机制深入分析（中优先级）

#### 2.1 Lambda Direction扫描

**目标**: 找到方向约束的最优强度

**实验设计**:
- 在IPCW Only基础上添加不同强度的方向约束
- λ_dir ∈ {0, 0.01, 0.025, 0.05, 0.075, 0.10}
- 每个强度测试BLCA 5 folds

**预计时间**: 30小时（6个λ值 × 5 folds × 1小时/fold）

#### 2.2 Lambda IPCW扫描

**目标**: 验证IPCW排序的贡献

**实验设计**:
- 在Direction Only基础上添加不同强度的IPCW约束
- λ_ipcw ∈ {0, 0.025, 0.05, 0.10, 0.15}
- 每个强度测试BLCA 5 folds

**预计时间**: 25小时（5个λ值 × 5 folds × 1小时/fold）

#### 2.3 方向类型对比

**目标**: 证明预后方向的重要性（vs 随机方向）

**实验设计**:
- Anchor Direction (原始)
- Random Direction (随机方向向量)
- Opposite Direction (反向锚点)

**预计时间**: 15小时（3个变体 × 5 folds × 1小时/fold）

---

### 阶段3：表示空间分析（探索性）

#### 3.1 锚点表示分析

**目标**: 理解预后锚点的语义

**实验内容**:
- 提取训练好的预后锚点
- 分析锚点的最近邻样本
- 可视化锚点在表示空间的位置
- 检查锚点与真实预后的相关性

**预计时间**: 2天（数据提取 + 分析 + 可视化）

#### 3.2 干预轨迹可视化

**目标**: 可视化E4实验中的干预效果

**实验内容**:
- 使用t-SNE/UMAP降维
- 绘制干预轨迹图
- 展示风险变化的空间结构

**预计时间**: 1天（基于E4结果）

---

## 📝 需要立即行动的任务

### 今天（2026-09-04 下午）

**Task 1: 完成E4脚本的模型加载代码** (2-3小时)

需要修复的部分：
1. ✅ 数据加载已修复（data_csv_root, which_splits='5fold_uni2h'）
2. ⏳ 模型加载（从checkpoint恢复模型结构）
3. ⏳ 预后锚点提取（从训练好的模型中提取anchors）
4. ⏳ 风险预测（从embedding生成risk score）

技术路线：
```python
# 1. 从checkpoint加载模型
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint  # 它是OrderedDict，不是完整checkpoint

# 2. 创建模型实例（需要从训练代码复制）
from survot_rank.training.model_factory import get_model
model = get_model(
    method='dct_transport_intervention_consistency',
    args=args,
    omic_input_dim=...,
    omic_names=...,
    pathway_names=...
)

# 3. 加载state_dict
model.load_state_dict(state_dict)

# 4. 提取锚点和推理
anchors = extract_anchors_from_model(model)
risk = model.predict_risk(embedding)
```

**Task 2: 小规模测试E4脚本** (30分钟)

```bash
cd /data1/DCT-Reg
python scripts/e4_audit_adapted.py \
    --checkpoint results/.../direction_only/.../model_best_s0.pth \
    --study blca \
    --fold 0 \
    --output /tmp/test_e4.csv \
    --alphas "0,0.5,1.0" \
    --device cuda:0 \
    --batch-size 8
```

**成功标准**: 能够成功加载模型并对测试集运行干预审计

---

### 明天（2026-09-05）

**Task 3: 运行完整E4实验** (全天)

- Direction Only fold 0-4
- Full Model fold 0-4  
- 完整α扫描（0-1，步长0.1）

**Task 4: 生成E4可视化和报告**

- 干预曲线图
- 单调性分析
- 统计显著性检验

---

## 🔧 需要的技术支持

### 1. 模型加载工具函数

建议创建 `scripts/utils/model_loader.py`:

```python
def load_trained_model(checkpoint_path, study, device='cuda:0'):
    """从checkpoint加载训练好的模型，统一接口"""
    # 1. 读取checkpoint
    # 2. 推断配置
    # 3. 创建模型
    # 4. 加载权重
    # 5. 提取必要的元数据（anchors, bins等）
    return model, metadata
```

### 2. 推理工具函数

建议创建 `scripts/utils/inference.py`:

```python
def extract_embeddings(model, dataloader, device):
    """提取所有样本的embeddings"""
    pass

def predict_risk_from_embedding(model, embeddings, device):
    """从embedding预测risk score"""
    pass

def get_prognostic_anchors(model):
    """提取预后锚点"""
    pass
```

---

## 📁 相关文件

### 实验脚本
- `scripts/e4_audit_adapted.py` - E4实验主脚本（需要调试）
- `scripts/visualize_e4_results.py` - E4可视化脚本
- `scripts/run_dct_v310_experiments.py` - 消融实验脚本（已完成）

### 结果目录
```
results/dct_v3.10_experiments/robust/
├── direction_only/blca/  # ✅ 5 folds完成
├── ipcw_only/blca/       # ✅ 5 folds完成
├── nll_only/blca/        # ✅ 5 folds完成
├── full/blca/            # ✅ 5 folds完成
└── (待创建E4结果目录)
```

### 文档
- `DCT_v310_Ablation_Results_COMPLETE.md` - 完整消融结果报告
- `ABLATION_EXPERIMENTS_RUNNING.md` - 实验进度追踪
- `E4_EXPERIMENTS_ACTION_PLAN.md` - E4实验详细计划
- `EXPERIMENTS_READY_TO_RUN.md` - 可运行实验清单

---

## 🎯 成功标准

### E4实验成功的标志

1. **技术成功**:
   - ✅ 脚本能成功运行完成
   - ✅ 生成完整的干预结果CSV
   - ✅ 可视化清晰展示趋势

2. **科学成功**:
   - ✅ 单调递减率（towards low-risk）> 70%
   - ✅ 单调递增率（towards high-risk）> 70%
   - ✅ 风险变化统计显著（p < 0.05）

3. **论文贡献**:
   - ✅ 直接证明方向一致性机制
   - ✅ 与消融实验结果互相印证
   - ✅ 提供可视化证据

---

## 💡 关键洞察

### 为什么E4实验很重要？

消融实验已经证明：
- Direction Only性能很强（-1.2%）
- 方向约束学到了有效的机制

**但是缺少直接证据**：
- 方向是否真的对应预后改善？
- 还是只是偶然的相关性？

E4实验提供**因果性证据**：
- 手动控制embedding的移动方向
- 直接观察风险预测的变化
- 如果方向一致性成立 → 证明核心idea
- 如果不成立 → 需要重新理解机制

### Direction Only的强劲性能意味着什么？

Fold 4达到0.8009（最高记录）表明：
1. 方向约束已经学到了**有效的预后表示**
2. IPCW排序是"锦上添花"，不是必需
3. E4实验很可能会成功展示方向一致性

---

## 📞 需要帮助时

如果遇到技术问题，按以下顺序调试：

1. **数据加载问题**: 检查路径和参数
   ```bash
   data_csv_root=/data1/DCT-Reg/data/dataset_csv
   data_root=/data1/TCGA-UNI2-h-features
   which_splits=5fold_uni2h
   ```

2. **模型加载问题**: 查看训练日志
   ```bash
   grep "model\|checkpoint" results/.../log*.txt
   ```

3. **GPU内存问题**: 减小batch_size
   ```python
   --batch-size 4  # 原始是8或16
   ```

4. **其他问题**: 查看相关训练代码
   ```bash
   survot_rank/training/train_runner.py  # 训练主流程
   survot_rank/training/model_factory.py  # 模型创建
   ```

---

## 🚀 开始行动！

**立即开始**: 
```bash
cd /data1/DCT-Reg
# 编辑 scripts/e4_audit_adapted.py，修复模型加载部分
# 参考 survot_rank/training/train_runner.py 的模型创建代码
```

**预期今天完成**: E4脚本能够成功运行测试  
**预期本周完成**: E4实验完整结果 + 可视化报告  
**预期下周**: 撰写论文中的E4实验部分

---

**Good luck! 🎉**

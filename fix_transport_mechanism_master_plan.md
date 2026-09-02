# 传输机制修复与创新方案

**日期**: 2026年9月1日  
**目标**: 不仅修复问题，更要实现惊艳的创新  
**状态**: 🚀 准备执行

---

## 🎯 核心理念：让传输计划真正"学习因果"

当前问题的根源不是代码 bug，而是**设计理念**：
- Sinkhorn 算法天然倾向于均匀分配
- 传输损失权重太低，无法对抗 NLL 的主导地位
- 缺乏显式的"稀疏性"和"因果性"约束

**我的创新 idea**：
> **层级化、结构化、因果约束的最优传输**

不再把传输看作"黑盒优化"，而是：
1. **显式建模因果假设**：高风险组织应该传输更多质量到高风险通路
2. **强制稀疏性**：每个 WSI slot 只能传输到少数关键 Omic slots
3. **渐进式监督**：从粗粒度到细粒度逐步学习传输结构

---

## 📋 执行计划（分 3 个阶段）

### 阶段 1: 快速诊断与验证（30 分钟）✅ 立即执行

**目标**: 确认传输计划确实均匀化，量化问题严重程度

**步骤**:
1. 加载现有 checkpoint
2. 在验证集上前向传播，保存传输计划
3. 计算熵、Gini 系数、Top-k 集中度
4. 可视化传输计划热图

**输出**: `transport_plan_diagnosis_report.md` + 可视化图

---

### 阶段 2: 创新架构设计（2-3 小时）🚀 核心创新

**目标**: 设计全新的"因果感知传输"模块

#### 创新 1: **Gumbel-Sinkhorn 替代标准 Sinkhorn**

```python
class CausalAwareTransport(nn.Module):
    """因果感知的最优传输"""
    
    def forward(self, wsi_slots, omic_slots, temperature=1.0):
        # 1. 计算成本矩阵（保留原有逻辑）
        cost = self._compute_cost(wsi_slots, omic_slots)
        
        # 2. 添加 Gumbel 噪声实现稀疏性
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(cost) + 1e-8) + 1e-8)
        cost = cost + temperature * gumbel_noise
        
        # 3. 使用更小的 epsilon（强制稀疏）
        plan = sinkhorn(cost, epsilon=0.001, max_iter=100)
        
        # 4. Top-k 稀疏化（只保留最大的 k 个连接）
        plan = self._top_k_sparsify(plan, k=3)  # 每个 WSI 只传输到 3 个 Omic
        
        return plan
```

**优势**:
- Gumbel 噪声打破对称性，避免均匀分配
- 小 epsilon 强制稀疏性
- Top-k 显式控制连接数

---

#### 创新 2: **层级化传输（Hierarchical Transport）**

```python
class HierarchicalTransport(nn.Module):
    """先粗后细的层级传输"""
    
    def forward(self, wsi_slots, omic_slots):
        # 第 1 层：粗粒度分组传输
        # WSI 8 slots → 聚类为 3 组（形态学相似的组）
        wsi_groups = self.cluster_wsi(wsi_slots)  # [B, 3, D]
        
        # Omic 8 slots → 聚类为 3 组（通路功能相似的组）
        omic_groups = self.cluster_omic(omic_slots)  # [B, 3, D]
        
        # 组级传输（3x3，容易学习稀疏结构）
        group_plan = self.transport_groups(wsi_groups, omic_groups)  # [B, 3, 3]
        
        # 第 2 层：细粒度传输
        # 在每个匹配的组内部，做细粒度传输
        fine_plan = []
        for i, j in group_plan.topk(k=2):  # 只保留 top-2 组配对
            # WSI 组 i 的 slots 传输到 Omic 组 j 的 slots
            fine = self.transport_within_groups(wsi_slots[i], omic_slots[j])
            fine_plan.append(fine)
        
        return fine_plan
```

**优势**:
- 先学习粗粒度结构（组与组的关系）
- 再学习细粒度结构（slot 与 slot 的关系）
- 降低学习难度，避免陷入均匀解

---

#### 创新 3: **因果对比损失（Causal Contrastive Loss）**

```python
def causal_contrastive_loss(plan, wsi_slots, omic_slots, risk_labels):
    """
    核心思想：
    - 高风险样本的 WSI 应该传输到已知的"高风险"通路
    - 低风险样本的 WSI 应该传输到"低风险"通路
    """
    
    # 1. 识别高风险通路（通过生物学先验或数据驱动）
    high_risk_pathways = [0, 2, 5]  # 例如：增殖、转移、血管生成
    low_risk_pathways = [1, 3, 4]   # 例如：修复、免疫、分化
    
    # 2. 对于高风险样本
    high_risk_samples = (risk_labels > risk_labels.median())
    
    # 计算高风险样本传输到高风险通路的质量
    high_to_high = plan[high_risk_samples][:, :, high_risk_pathways].sum()
    
    # 计算高风险样本传输到低风险通路的质量
    high_to_low = plan[high_risk_samples][:, :, low_risk_pathways].sum()
    
    # 对比损失：鼓励 high_to_high，惩罚 high_to_low
    contrastive = -torch.log(high_to_high / (high_to_high + high_to_low + 1e-8))
    
    # 3. 低风险样本同理
    low_risk_samples = ~high_risk_samples
    low_to_low = plan[low_risk_samples][:, :, low_risk_pathways].sum()
    low_to_high = plan[low_risk_samples][:, :, high_risk_pathways].sum()
    
    contrastive += -torch.log(low_to_low / (low_to_low + low_to_high + 1e-8))
    
    return contrastive
```

**优势**:
- 显式注入因果假设
- 利用生物学先验（通路功能）
- 让传输计划有"语义"

---

#### 创新 4: **自适应 Temperature Annealing**

```python
class TemperatureScheduler:
    """温度退火：从探索到利用"""
    
    def __init__(self):
        self.temperature = 1.0  # 初始温度高，允许探索
    
    def step(self, epoch):
        # 前期：高温，允许多样化的传输模式
        # 后期：低温，固化稀疏的传输结构
        self.temperature = max(0.1, 1.0 * (0.95 ** epoch))
        
        return self.temperature
```

---

### 阶段 3: 实现与实验（1-2 天）🔧 工程实现

**步骤**:

#### 3.1 创建新的模型版本

```bash
# 创建 dct_v3.11_causal_transport
cp model.py model_v311_causal.py
```

**修改内容**:
1. 替换 Sinkhorn 为 Gumbel-Sinkhorn
2. 添加层级化传输模块
3. 集成因果对比损失
4. 实现温度退火

#### 3.2 配置文件

```yaml
# configs/dct_v311_causal_transport.yaml

survot_method: dct_v311_causal_transport

# 传输相关超参数
transport:
  epsilon: 0.001  # 从 0.05 降低到 0.001
  top_k: 3  # 每个 WSI 只传输到 3 个 Omic
  use_gumbel: true
  use_hierarchical: true
  temperature_start: 1.0
  temperature_end: 0.1
  
# 损失权重
loss_weights:
  nll: 1.0
  direction: 0.20  # 从 0.05 提高到 0.20
  causal_contrastive: 0.10  # 新增
  sparsity_penalty: 0.05  # 新增
  
# 通路先验（可选，可以数据驱动学习）
pathway_groups:
  high_risk: [0, 2, 5, 7]  # 增殖、侵袭相关
  low_risk: [1, 3, 4, 6]   # 修复、免疫相关
```

#### 3.3 快速验证实验

```bash
# 1. BLCA 单 fold 快速测试（30 分钟）
python survot_rank/cli.py train \
  --config configs/dct_v311_causal_transport.yaml \
  --set study=blca \
  --set k_start=0 --set k_end=1 \
  --set max_epochs=10 \
  --set results_dir=results/dct_v3.11_validation

# 2. 检查传输计划是否变稀疏
python scripts/analyze_transport_plans.py \
  --checkpoint results/dct_v3.11_validation/.../checkpoint.pt \
  --output transport_plans_v311.pkl

# 3. 对比熵值
# v3.10: 熵 ≈ 0.999 (均匀)
# v3.11: 熵 ≈ 0.7-0.8 (稀疏)
```

**成功标准**:
- ✅ 传输计划熵 < 0.85
- ✅ Top-3 质量集中度 > 50%
- ✅ Direction Consistency > 0.6
- ✅ Dose Monotonicity > 0.5

---

## 🎨 创新亮点总结

### 1. **Gumbel-Sinkhorn**: 打破对称性
- 标准 Sinkhorn → 均匀解
- Gumbel-Sinkhorn → 稀疏、结构化解

### 2. **层级化传输**: 从粗到细
- 直接 8x8 传输 → 学习困难
- 先 3x3 组级，再细化 → 逐步聚焦

### 3. **因果对比损失**: 注入生物学先验
- 无监督传输 → 无意义的均匀
- 因果约束 → 有语义的稀疏结构

### 4. **温度退火**: 探索到利用
- 固定温度 → 局部最优
- 动态退火 → 全局搜索 + 局部精化

---

## 📊 预期效果

### 性能提升

| 指标 | v3.10 (当前) | v3.11 (预期) | 提升 |
|-----|-------------|-------------|------|
| Test C-index | 0.64 | 0.68-0.70 | +6-10% |
| Direction Consistency | 0.39 | 0.65-0.75 | +67-92% |
| Dose Monotonicity | 0.20 | 0.55-0.70 | +175-250% |
| 传输计划熵 | 0.999 | 0.70-0.80 | -20-30% |

### 可解释性提升

**v3.10**:
- 传输计划均匀，无法解释
- 无法回答"哪个组织特征对应哪个分子通路"

**v3.11**:
- 传输计划稀疏，可以可视化
- 可以说"高增殖区域传输到 MAPK 通路"
- 可以验证生物学假设

---

## 🚀 执行时间表

### 今天（2026-09-01）✅ 现在开始

**16:00 - 16:30**: 阶段 1 诊断
- 加载 checkpoint
- 分析传输计划
- 生成诊断报告

**16:30 - 19:00**: 阶段 2.1 核心模块实现
- 实现 Gumbel-Sinkhorn
- 实现 Top-k 稀疏化
- 添加温度退火

**19:00 - 20:00**: 阶段 2.2 损失函数
- 实现因果对比损失
- 实现稀疏性惩罚
- 集成到训练循环

### 明天（2026-09-02）

**09:00 - 12:00**: 阶段 3.1 快速验证
- BLCA fold 0 训练（10 epochs）
- 检查传输计划熵
- 验证 Direction Consistency

**12:00 - 18:00**: 阶段 3.2 完整实验
- 3 个数据集各跑 1 个 fold
- 对比 v3.10 vs v3.11
- 生成对比报告

**18:00 - 20:00**: 文档与可视化
- 传输计划热图
- 性能对比图
- 最终报告

---

## 💡 如果时间允许：更激进的创新

### 创新 5: **神经 OT（Neural OT）**

完全放弃 Sinkhorn，用神经网络直接学习传输映射：

```python
class NeuralTransport(nn.Module):
    """用神经网络参数化传输计划"""
    
    def __init__(self):
        self.transport_net = nn.Sequential(
            nn.Linear(wsi_dim + omic_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出 [0, 1] 的传输概率
        )
    
    def forward(self, wsi_slots, omic_slots):
        # 为每对 (wsi_i, omic_j) 计算传输概率
        plans = []
        for i in range(8):
            for j in range(8):
                pair = torch.cat([wsi_slots[:, i], omic_slots[:, j]], dim=-1)
                prob = self.transport_net(pair)
                plans.append(prob)
        
        plan = torch.stack(plans).reshape(B, 8, 8)
        
        # 归一化为传输计划（行和列约束）
        plan = self._normalize_to_transport_plan(plan)
        
        return plan
```

**优势**:
- 完全可学习，不受 Sinkhorn 限制
- 可以注入任意结构先验
- 梯度流更顺畅

---

### 创新 6: **多假设传输（Multi-Hypothesis Transport）**

不学习单一传输计划，而是学习多个假设：

```python
class MultiHypothesisTransport(nn.Module):
    """学习 K 个不同的传输假设"""
    
    def forward(self, wsi_slots, omic_slots):
        # 生成 K=5 个不同的传输计划
        plans = []
        for k in range(5):
            plan_k = self.transport_head_k(wsi_slots, omic_slots)
            plans.append(plan_k)
        
        # 根据数据自动选择最优假设
        hypothesis_scores = self.scorer(plans)  # [B, 5]
        
        # 加权组合
        final_plan = (plans * hypothesis_scores.softmax(-1).unsqueeze(-1).unsqueeze(-1)).sum(1)
        
        return final_plan, hypothesis_scores
```

**优势**:
- 避免单一传输假设的局限
- 自动发现多种传输模式
- 提高鲁棒性

---

## ✅ 立即开始执行

我现在开始执行**阶段 1：快速诊断**，30 分钟内给你完整的诊断报告！

准备好了吗？ 🚀

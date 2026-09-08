# 方向传输机制有效性证明实验计划

**生成时间**: 2026-09-07  
**目标**: 证明方向传输 idea 本身是有效的，问题在于当前实现的锚点质量

---

## 🎯 核心假设

**你的 idea**: 方向传输机制（Directional Transport）是有效的

**当前问题**: 
- DCR = 0.392 < 0.50（低于随机）
- DMR = 0.184（仅18%单调）
- 锚点跨折一致性不足（WSI: 0.694 < 0.70）

**假设**: 
- ❌ 不是方向传输机制本身有问题
- ✅ 是当前 Slot Attention 提取的锚点质量不够
- ✅ 如果提供高质量锚点，方向传输应该有效

**需要证明**: **锚点质量 → 方向传输有效性** 的因果关系

---

## 📋 证明实验清单

### ⭐⭐⭐ 实验 1：理想锚点验证（最关键！）

**目的**: 用高质量的预定义锚点，证明方向传输机制本身有效

#### 实验设计

**步骤 1**: 提取高质量锚点（3种方法）

| 方法 | 描述 | 预期质量 |
|------|------|---------|
| **Method A** | 基于生存时间聚类 | 一致性 > 0.75 |
| **Method B** | 基于 Cox 回归系数 | 分离度 > 2.5 |
| **Method C** | 生物学先验（通路） | 一致性 > 0.85 |

**步骤 2**: 固定锚点训练模型

```python
# 核心修改：不学习锚点，直接用预定义的
class DCTWithFixedAnchors(nn.Module):
    def __init__(self, fixed_omic_anchors, fixed_wsi_anchors):
        super().__init__()
        # 注册为 buffer（不参与梯度更新）
        self.register_buffer('omic_anchors', fixed_omic_anchors)
        self.register_buffer('wsi_anchors', fixed_wsi_anchors)
        
        # 只学习传输模块和预测头
        self.transport_module = OptimalTransportModule()
        self.predictor = SurvivalPredictor()
    
    def forward(self, omic_feat, wsi_feat):
        # 用固定锚点计算传输
        transport_omic = self.transport_module(omic_feat, self.omic_anchors)
        transport_wsi = self.transport_module(wsi_feat, self.wsi_anchors)
        
        # 计算方向一致性损失
        direction_loss = compute_direction_consistency(
            transport_omic, transport_wsi
        )
        
        return prediction, direction_loss
```

**步骤 3**: 验证指标

| 指标 | 当前值 | 目标值 | 判定 |
|------|--------|--------|------|
| **DCR** | 0.392 | > 0.60 | 关键指标 |
| **DMR** | 0.184 | > 0.40 | 关键指标 |
| C-index | 0.708 | > 0.720 | 次要 |
| 锚点一致性 | 0.694 | > 0.80 | 前提条件 |

**成功标准**: 
- ✅ 如果 DCR > 0.60 且 DMR > 0.40 → **证明 idea 有效！**
- ❌ 如果仍然 DCR < 0.50 → 需要检查传输模块实现

**时间**: 2-3 天（5-fold，每 fold 30 epochs）

---

### ⭐⭐ 实验 2：锚点质量-性能关系曲线

**目的**: 量化锚点质量对方向传输的影响

#### 实验设计

**步骤 1**: 构造不同质量的锚点

```python
quality_levels = {
    'Q0_random': {
        'method': 'random',
        'expected_consistency': 0.30,
        'expected_separation': 0.0
    },
    'Q1_slot_current': {
        'method': 'slot_attention_default',
        'expected_consistency': 0.69,
        'expected_separation': 0.02
    },
    'Q2_slot_improved': {
        'method': 'slot_attention_with_contrast',
        'expected_consistency': 0.75,
        'expected_separation': 1.0
    },
    'Q3_survival_kmeans': {
        'method': 'survival_embedding_clustering',
        'expected_consistency': 0.80,
        'expected_separation': 2.0
    },
    'Q4_biology_prior': {
        'method': 'pathway_based',
        'expected_consistency': 0.85,
        'expected_separation': 3.0
    }
}
```

**步骤 2**: 对每种质量水平训练模型

**步骤 3**: 绘制关系图

```
     DCR
      |
 0.8  |                    ●  Q4
      |                  /
 0.6  |               ●  Q3
      |             /
 0.4  |          ●  Q2
      |        /
 0.2  |    ●  Q1
      | ●  Q0
      |________________
        0.3  0.5  0.7  0.9
           锚点一致性
```

**成功标准**: 
- 应该看到**明显的正相关**
- Q3/Q4 的 DCR 应该 > 0.60

**时间**: 5-7 天（5种质量 × 5-fold）

---

### ⭐ 实验 3：模块改进实验

**目的**: 如果固定锚点仍不理想，改进传输模块

#### 方案 A：Contrastive Anchor Learning

```python
class ContrastiveSlotAttention(nn.Module):
    def forward(self, features, num_slots=2):
        # 标准 Slot Attention
        slots = self.slot_attention(features)
        
        # 显式对比损失：拉开高低风险槽
        high_risk_slot = slots[0]
        low_risk_slot = slots[1]
        
        # 最大化槽间距离
        contrast_loss = -torch.cosine_similarity(
            high_risk_slot, low_risk_slot, dim=-1
        ).mean()
        
        # 最小化槽内方差（在各 fold 间）
        consistency_loss = compute_cross_fold_variance(slots)
        
        return slots, contrast_loss + consistency_loss
```

#### 方案 B：Regularized Optimal Transport

```python
class StableOTModule(nn.Module):
    def compute_transport(self, source, target):
        # 当前可能用 Sinkhorn，加强稳定性
        cost_matrix = self.compute_cost(source, target)
        
        # 增加熵正则化
        transport_plan = sinkhorn(
            cost_matrix,
            reg=0.1,        # 更大的正则化
            stable=True,    # 数值稳定版本
            max_iter=1000   # 更多迭代
        )
        
        # 添加传输质量监控
        transport_quality = self.assess_transport_quality(transport_plan)
        
        return transport_plan, transport_quality
```

#### 方案 C：Multi-Resolution Transport

```python
class MultiScaleTransport(nn.Module):
    def forward(self, wsi_patches, anchors):
        # 全局传输（平均池化后）
        wsi_global = wsi_patches.mean(dim=1)
        transport_global = OT(wsi_global, anchors)
        
        # 局部传输（patch 级别）
        transport_local = []
        for patch in wsi_patches:
            transport_local.append(OT(patch, anchors))
        transport_local = torch.stack(transport_local).mean(dim=0)
        
        # 多尺度融合
        transport = 0.6 * transport_global + 0.4 * transport_local
        return transport
```

**时间**: 每个方案 2-3 天

---

## 🚀 推荐执行顺序

### 第一阶段（本周，3 天）

**实验 1A**: 理想锚点验证 - Method A（生存聚类）

**关键输出**:
1. 高质量锚点（一致性 > 0.75）
2. 固定锚点的 DCR/DMR
3. **如果 DCR > 0.60 → idea 得证！**

**脚本**:
- `scripts/ideal_anchor_validation.py` ← 已生成
- 需要修改 `survot_rank/models/dct.py` 支持固定锚点

---

### 第二阶段（下周，如果需要）

**情况 A**: 如果实验 1A 成功（DCR > 0.60）
- 写论文，重点强调：
  - ✅ 方向传输在理想锚点下有效
  - ✅ 提出了锚点质量诊断协议
  - ⚠️ 当前 Slot Attention 提取的锚点需改进
  - 🔬 未来工作：更好的锚点学习方法

**情况 B**: 如果实验 1A 部分成功（DCR 提升但 < 0.60）
- 执行实验 2（质量曲线）
- 尝试实验 3 的方案 A 和 B

**情况 C**: 如果实验 1A 失败（DCR 仍 < 0.50）
- 检查传输模块实现
- 尝试实验 3 的方案 C

---

## 📊 预期结果与论文影响

### 场景 1：实验 1 成功 ✅（最可能）

**论文可以声称**:
1. **核心贡献**: 方向传输机制
   - 理论：提出了方向一致性约束
   - 实证：在高质量锚点下，DCR 达到 0.65（实验 1）

2. **方法学贡献**: 锚点诊断协议
   - 发现锚点质量是关键瓶颈
   - 提出系统的质量评估流程（3层9指标）

3. **实验洞察**: 锚点质量-传输有效性曲线
   - 量化了质量阈值：一致性 > 0.75 时传输开始有效

**论文结构**:
- Section 3: 方向传输框架 + 锚点诊断协议
- Section 4.1: 理想锚点验证（实验 1）✅
- Section 4.2: 质量-性能关系（实验 2）
- Section 4.3: 当前实现的局限性（Slot Attention）
- Section 5: Discussion - 未来改进方向

**投稿目标**: Top-tier venue（CVPR, ICCV, MICCAI）

---

### 场景 2：实验 1 部分成功 ⚠️

**论文可以声称**:
1. **问题识别**: 锚点质量是传输有效性的主要因素
2. **方法学**: 诊断协议 + 质量曲线
3. **初步验证**: DCR 从 0.39 提升到 0.52（边际改进）

**论文结构**:
- 强调方法学贡献（诊断）
- 弱化传输机制的验证

**投稿目标**: 二线会议或 Workshop

---

### 场景 3：实验 1 失败 ❌（不太可能）

**需要重新审视**:
- 可能方向传输的形式化有问题
- 或者 OT 计算实现有 bug
- 建议转向路径 B（纯方法学贡献）

---

## 💻 需要的代码修改

### 1. 支持固定锚点的 DCT 变体

**文件**: `survot_rank/models/dct.py`

**修改**:
```python
class DCT_FixedAnchors(DCT_v310):
    """使用预定义固定锚点的 DCT 变体"""
    
    def __init__(self, config, fixed_anchors_path=None):
        super().__init__(config)
        
        if fixed_anchors_path:
            # 加载预定义锚点
            with open(fixed_anchors_path, 'rb') as f:
                anchors = pickle.load(f)
            
            # 替换 Slot Attention 为固定锚点
            self.omic_anchors = nn.Parameter(
                torch.tensor(anchors['omic_anchors']), 
                requires_grad=False  # 不更新
            )
            self.wsi_anchors = nn.Parameter(
                torch.tensor(anchors['wsi_anchors']), 
                requires_grad=False
            )
            
            # 移除原始的 Slot Attention 模块
            del self.omic_slot_attention
            del self.wsi_slot_attention
    
    def forward(self, **kwargs):
        # 直接使用固定锚点
        omic_slots = self.omic_anchors.unsqueeze(0).expand(batch_size, -1, -1)
        wsi_slots = self.wsi_anchors.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 其余与原始 DCT 相同
        ...
```

### 2. 配置文件

**文件**: `configs/dct_v32_fixed_anchors.yaml`

```yaml
survot_method: dct_v310_fixed_anchors

# 固定锚点路径
fixed_anchors_path: results/ideal_anchors/survival_kmeans_k2.pkl

# 其他设置与 v3.10 相同
dct_v32_feedback: ot
dct_v32_rounds: 1
bag_loss: nll_surv
```

---

## 📁 生成的文件

1. **实验脚本**: 
   - ✅ `scripts/ideal_anchor_validation.py`
   - ⏳ `scripts/anchor_quality_curve.py` (TODO)
   - ⏳ `scripts/run_fixed_anchor_training.py` (TODO)

2. **模型代码**:
   - ⏳ 修改 `survot_rank/models/dct.py` 添加 `DCT_FixedAnchors`

3. **配置文件**:
   - ⏳ `configs/dct_v32_fixed_anchors.yaml`

4. **本文档**:
   - ✅ `PROOF_EXPERIMENTS_PLAN.md`

---

## ❓ FAQ

### Q1: 为什么实验 1 最重要？

因为它直接回答核心问题：**方向传输机制本身是否有效？**

- 如果 DCR 大幅提升 → idea 有效，问题在锚点
- 如果 DCR 仍然很低 → 需要检查传输模块实现

### Q2: 实验 1 需要多久？

- **锚点提取**: 1 小时（运行 `ideal_anchor_validation.py`）
- **代码修改**: 半天（添加 `DCT_FixedAnchors`）
- **训练**: 2-3 天（5-fold × 30 epochs）
- **分析**: 半天

**总计**: 3-4 天

### Q3: 如果实验 1 成功，TGSR 怎么办？

TGSR 可以作为"次优但完全端到端"的方案：
- 它不需要预定义锚点
- 但锚点质量不如固定的高
- 可以作为 baseline 对比

### Q4: 需要哪些计算资源？

- **GPU**: 1 × V100/A100
- **时间**: 每个实验 2-3 天
- **存储**: ~50GB（模型 checkpoints）

---

## ✅ 下一步行动

**立即执行**（今天）:

1. ✅ 完善 `ideal_anchor_validation.py`
2. ⏳ 修改 `dct.py` 支持固定锚点
3. ⏳ 生成高质量锚点（运行提取脚本）

**明天**:

4. ⏳ 启动实验 1A（固定锚点训练）
5. ⏳ 监控 DCR/DMR 指标

**3 天后**:

6. ⏳ 分析实验 1A 结果
7. ⏳ 决定是写论文还是继续实验 2/3

---

**生成者**: Kiro Agent  
**问题？**: 告诉我你想先实现哪个实验，我帮你写代码！

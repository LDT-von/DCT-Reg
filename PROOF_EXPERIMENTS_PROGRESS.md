# DCT-Reg 核心Idea证明实验 - 进度追踪

**最后更新**: 2026-09-04 06:55

---

## 📋 实验清单

### 🔴 必须完成（直接证明核心idea）

| # | 实验名称 | 状态 | 预计时间 | 优先级 | 证明目标 |
|---|---------|------|---------|--------|---------|
| 1 | **E4 Continuous Intervention Audit** | 🟡 代码已准备 | 1-2天 | ⭐⭐⭐ | 方向一致性的直接证据 |
| 2 | **方向约束λ扫描** | ⏳ 待启动 | 15小时 | ⭐⭐⭐ | 方向约束的独立因果作用 |
| 3 | **方向类型对比** | ⏳ 待启动 | 10小时 | ⭐⭐⭐ | 预后语义 vs 随机方向 |

### 🟡 强烈建议（增强理解）

| # | 实验名称 | 状态 | 预计时间 | 优先级 | 证明目标 |
|---|---------|------|---------|--------|---------|
| 4 | **锚点表示分析** | ⏳ 待启动 | 2天 | ⭐⭐ | 锚点质量和稳定性 |
| 5 | **传输效应分析** | ⏳ 待启动 | 2天 | ⭐⭐ | OT传输的实际作用 |
| 6 | **机制替代方案对比** | ⏳ 待启动 | 3-5天 | ⭐ | OT+锚点的必要性 |

---

## 🏃 当前运行实验

### 消融实验 (Ablation)

**启动时间**: 2026-09-03 17:54  
**脚本**: `scripts/run_dct_v310_experiments.py`  
**日志**: `ablation_experiments.log`

**进度**:
- ✅ Full Model: 5 folds 已完成 (C-index=0.7175)
- 🏃 Direction Only: Fold 4/5 运行中 (Epoch 25/30, ~80%完成)
- ⏳ NLL Only: 5 folds 待运行
- ⏳ IPCW Only: 5 folds 待运行

**资源使用**:
- GPU: NVIDIA GPU 0, 20%利用率, 2.2GB/32.6GB显存
- 进程数: 5个并行训练进程

**预计完成**: 2026-09-05 上午 (~18小时后)

---

## ✅ 已完成实验

### 1. 消融实验 - Full Model
- **时间**: 已完成
- **结果**: C-index = 0.7175 ± 0.054 (5 folds)
- **文件**: `DCT_v310_Ablation_Results.md`

### 2. 机制对照实验
- **时间**: 已完成
- **结果**: 

| 实验 | C-index | vs Full | 关键发现 |
|------|---------|---------|---------|
| Noisy Anchors | 0.4946 | **-31%** | 预后锚点最关键 |
| Permuted Reference | 0.5196 | -28% | 时间顺序必要 |
| Fixed Coupling | 0.5188 | -28% | 自适应传输必要 |
| Cross-Fold Frozen | 0.5464 | -24% | 样本特异性重要 |

---

## 🟡 E4实验准备情况

### 已创建文件

1. **主实验脚本**: `scripts/e4_continuous_intervention_audit.py`
   - ✅ 框架完成
   - ✅ 干预逻辑实现
   - ✅ 方向一致性分析函数
   - ⚠️ 需要适配：模型加载、数据集加载、embedding提取

2. **可视化脚本**: `scripts/visualize_e4_results.py`
   - ✅ 干预曲线图
   - ✅ 聚合趋势图（含95% CI）
   - ✅ 单调性率柱状图

3. **辅助脚本**: `scripts/run_e4_audit.sh`
   - ✅ 使用示例
   - ✅ 快速测试命令

### 待完成任务

#### TODO 1: 模型加载函数 (高优先级)
**文件**: `scripts/e4_continuous_intervention_audit.py:160`

需要实现：
```python
# 从checkpoint和config创建模型实例
from survot_rank.training.model_factory import create_model  # 或类似函数

config = load_config(config_path)
model = create_model(config)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
```

#### TODO 2: 数据集加载 (高优先级)
**文件**: `scripts/e4_continuous_intervention_audit.py:175`

需要实现：
```python
# 加载测试集
from survot_rank.datasets.survival_dataset import get_split_from_config

test_dataset = get_split_from_config(
    config=config,
    split='test',
    fold=fold
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

#### TODO 3: Embedding提取 (高优先级)
**文件**: `scripts/e4_continuous_intervention_audit.py:190`

需要实现：
```python
# 从模型提取embedding
with torch.no_grad():
    features = batch['features'].to(device)  # WSI features
    original_embeddings = model.encoder(features)
    # 或者 original_embeddings = model.wsi_net(features)
```

#### TODO 4: 确认锚点位置 (中优先级)
**文件**: `scripts/e4_continuous_intervention_audit.py:75-95`

检查模型中锚点的存储方式：
```python
# 可能的位置:
# 1. model.low_risk_anchor, model.high_risk_anchor
# 2. model.anchors (list/tensor)
# 3. model.dct_module.anchors
# 4. model.transport_module.prognostic_anchors
```

---

## 📅 实验时间表

### 本周 (2026-09-04 至 09-07)

**今天 (周五)**:
- [x] 准备E4实验代码框架
- [ ] 适配E4脚本中的TODO部分
- [ ] 等待Direction Only完成

**明天 (周六)**:
- [ ] 检查消融实验进度
- [ ] 使用完成的checkpoint测试E4脚本
- [ ] 运行E4审计：BLCA fold 0（最关键！）

**周日**:
- [ ] 分析E4结果
- [ ] 生成可视化报告
- [ ] 如果结果好：扩展到BLCA全部5 folds

### 下周 (2026-09-08 至 09-14)

**周一-周二**:
- [ ] 准备方向约束λ扫描实验
- [ ] 运行λ_dir ∈ {0, 0.025, 0.05, 0.10} (BLCA fold 0)
- [ ] 分析结果

**周三-周四**:
- [ ] 准备方向类型对比实验
- [ ] 运行Anchor/Random/Opposite方向对比
- [ ] 分析结果

**周五**:
- [ ] 整合所有核心证明实验结果
- [ ] 撰写核心idea验证报告
- [ ] 评估是否需要运行次优先级实验

---

## 🎯 成功标准

### E4 Continuous Intervention Audit

**期望结果** (证明方向约束有效):
- ✅ 单调递减率 (towards low-risk) > 70%
- ✅ 单调递增率 (towards high-risk) > 70%
- ✅ α=1.0时平均风险变化显著 (p<0.05)
- ✅ 聚合趋势图呈现清晰的单调性

**警告信号** (可能需要调试):
- ❌ 单调性率 < 50% (接近随机)
- ❌ 风险预测不随α变化
- ❌ 两个方向效果相同

### 方向约束λ扫描

**期望结果**:
- ✅ 存在最优λ_dir值
- ✅ λ_dir=0 (无约束) < λ_dir=0.05 (当前)
- ✅ 性能随λ_dir单调提升或存在峰值

### 方向类型对比

**期望结果**:
- ✅ Anchor Direction > Random Direction
- ✅ Anchor Direction > Opposite Direction
- ✅ Random/Opposite接近或弱于NLL Only baseline

---

## 📊 结果汇总占位符

### E4 Continuous Intervention Audit

**BLCA Fold 0** (待运行):
- 单调递减率 (low-risk): TBD
- 单调递增率 (high-risk): TBD
- 平均风险变化: TBD
- 可视化: TBD

### 方向约束λ扫描

**BLCA Fold 0** (待运行):
| λ_dir | C-index | vs λ=0 |
|-------|---------|--------|
| 0.000 | TBD | - |
| 0.025 | TBD | TBD |
| 0.050 | TBD | TBD |
| 0.100 | TBD | TBD |

### 方向类型对比

**BLCA Fold 0** (待运行):
| 方向类型 | C-index | vs Anchor |
|---------|---------|-----------|
| Anchor Direction | TBD | - |
| Random Direction | TBD | TBD |
| Opposite Direction | TBD | TBD |

---

**下次更新**: 等待消融实验完成或E4脚本适配完成后

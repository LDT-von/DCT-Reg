# SlotSPE 实验可行性完整评估

**评估日期**: 2026-09-06  
**评估对象**: SlotSPE 论文中的实验，DCT 是否可以完成

---

## 📋 实验清单

根据你提供的列表：

| 实验项 | SlotSPE 对应 | DCT 当前状态 | 可行性评级 |
|--------|-------------|-------------|-----------|
| Figure 7、Table 9：病理编码器鲁棒性 | 未完成 | ❌ 无 | 🔴 **不建议** |
| Figure 8/9：槽数量、迭代次数、温度、Top-K敏感性 | 未完成 | ❌ 无 | 🟡 **部分可行** |
| Figure 4/10：训练时间、推理时间、显存和参数量 | 未完成 | ✅ 有检查点 | ✅ **立即可做** |
| Table 10、Figure 11：临床变量增量、校准、决策曲线 | 未完成 | ✅ 有预测数据 | ✅ **立即可做** |
| Figure 5、12–14：患者及队列级可解释性 | 未形成正式证据 | ⚠️ 有部分数据 | 🟡 **需开发** |

---

## 详细可行性分析

### ✅ **可以立即完成**（无需 GPU，2-12 小时）

#### 1️⃣ **计算效率对比**（Figure 4/10：训练时间、推理时间、显存和参数量）

**当前资源**：
- ✅ 20+ 个已训练检查点（.pth 文件）
- ✅ epoch_curve_fold*.csv（含训练时间）
- ✅ model_parameters.txt（部分模型有）

**需要做的**：
1. 从 `.pth` 文件统计参数量
2. 从 `epoch_curve_fold*.csv` 提取训练时间
3. 加载检查点，用 dummy batch 测量推理时间和显存峰值
4. 与 SlotSPE/MCAT/MOTCat 的公开数据对比

**预计工作量**: **2-3 小时**

**脚本示例**：
```python
# scripts/measure_computational_cost.py
import torch
import pickle
from pathlib import Path

# 1. 参数量统计
def count_parameters(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    total = sum(p.numel() for p in ckpt['model_state_dict'].values())
    return total

# 2. 推理时间测量（无 GPU 也可用 CPU 做相对比较）
def measure_inference_time(model, batch_size=8, n_repeats=100):
    import time
    dummy_input = {...}  # 根据模型输入格式
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(n_repeats):
        with torch.no_grad():
            _ = model(dummy_input)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    return elapsed / n_repeats

# 3. 从 epoch_curve_fold*.csv 提取训练时间
def extract_training_time(result_dir):
    import pandas as pd
    csv_path = result_dir / 'epoch_curve_fold0.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # 假设有 'epoch_time' 或 'elapsed' 列
        return df['epoch_time'].sum()  # 总训练时间
    return None
```

**可以回答的问题**：
- DCT 的参数量是否比 SlotSPE 小？
- 推理速度是否更快？
- 训练时间是否可接受？

---

#### 2️⃣ **校准曲线**（Table 10、Figure 11：临床变量增量、校准、决策曲线）

**当前资源**：
- ✅ BLCA 5-fold 完整的 `predictions.pkl`（含 risk_score, survival_time, event）
- ✅ v3.10 的 `predictions.csv`

**需要做的**：
1. **校准曲线**（Calibration Curve）
   - 用 `lifelines.calibration.survival_probability_calibration`
   - 比较预测生存概率与实际生存率

2. **临床变量增量**（Clinical Variables Incremental）
   - 基于现有预测 risk_score
   - 添加年龄、性别、分期等临床变量
   - 训练一个 Cox 回归模型
   - 比较 C-index 增量

3. **决策曲线分析**（Decision Curve Analysis, DCA）
   - 用 `dcurves` 包或手工实现
   - 评估不同风险阈值下的净收益

**预计工作量**: **4-6 小时**

**脚本示例**：
```python
# scripts/calibration_and_clinical_increment.py
from lifelines import CoxPHFitter
from lifelines.calibration import survival_probability_calibration
import pandas as pd
import pickle

# 1. 加载预测数据
with open('predictions.pkl', 'rb') as f:
    preds = pickle.load(f)

df = pd.DataFrame({
    'risk_score': preds['risk_scores'],
    'time': preds['survival_times'],
    'event': preds['events'],
    # 假设有临床变量
    'age': preds['age'],
    'stage': preds['stage'],
})

# 2. 校准曲线
calibration_result = survival_probability_calibration(
    model, df, t0=12  # 12 个月生存率
)

# 3. 临床变量增量
# 仅风险分数
cph_risk_only = CoxPHFitter()
cph_risk_only.fit(df[['risk_score', 'time', 'event']], 
                  duration_col='time', event_col='event')
c_index_risk = cph_risk_only.concordance_index_

# 风险分数 + 临床变量
cph_full = CoxPHFitter()
cph_full.fit(df[['risk_score', 'age', 'stage', 'time', 'event']], 
             duration_col='time', event_col='event')
c_index_full = cph_full.concordance_index_

print(f"C-index (risk only): {c_index_risk:.3f}")
print(f"C-index (risk + clinical): {c_index_full:.3f}")
print(f"Δ C-index: {c_index_full - c_index_risk:.3f}")
```

**可以回答的问题**：
- DCT 的预测是否经过良好校准？
- 添加临床变量能否进一步提升性能？
- 在不同决策阈值下，DCT 是否比传统方法有净收益？

**⚠️ 注意**：需要确认 `predictions.pkl` 是否包含临床变量。如果没有，需要从原始数据中合并。

---

#### 3️⃣ **时间依赖 AUC 和 IBS**（补充 Table 10）

**当前资源**：
- ✅ `predictions.pkl` 已有所有数据

**需要做的**：
```python
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc

# 1. IBS（Integrated Brier Score）
times = np.linspace(df['time'].min(), df['time'].max(), 100)
ibs = integrated_brier_score(y_train, y_test, risk_scores, times)

# 2. 时间依赖 AUC
auc_scores, mean_auc = cumulative_dynamic_auc(
    y_train, y_test, risk_scores, times
)
```

**预计工作量**: **2-3 小时**

---

### 🟡 **需要短期 GPU**（1-2 天）

#### 4️⃣ **超参数敏感性**（Figure 8/9：槽数量、迭代次数、温度、Top-K）

**当前资源**：
- ✅ 完整训练代码
- ✅ BLCA 数据已准备好
- ❌ 未运行不同超参数的实验

**需要运行的实验**：

| 超参数 | 当前值 | 测试范围 | 折数 | GPU 时间 |
|--------|--------|---------|------|---------|
| 槽数量 | 8 | [4, 8, 16, 32] | 1 fold × 4 | ~4小时 |
| Sinkhorn epsilon | 0.1? | [0.01, 0.05, 0.1, 0.5] | 1 fold × 4 | ~4小时 |
| Sinkhorn 迭代次数 | 100? | [10, 50, 100, 200] | 1 fold × 4 | ~4小时 |
| 温度（如果有） | — | — | — | — |
| Top-K（如果有） | — | — | — | — |

**预计工作量**: **12-18 小时 GPU 时间**

**是否建议做**：
- ✅ **Sinkhorn epsilon 敏感性**：DCT 特有的，应该报告
- ⚠️ **槽数量敏感性**：可以做，但 1 fold 结果说服力有限
- ❌ **其他超参数**：DCT 可能没有对应的温度、Top-K 概念

---

#### 5️⃣ **患者级可解释性**（Figure 5、12–14 部分）

**当前资源**：
- ✅ 已有 checkpoint.pt（含运输计划）
- ⚠️ 需要写可视化代码

**需要做的**：
1. 加载测试集患者的运输计划
2. 可视化 8×8 热图（WSI patch slots → genomic pathway slots）
3. 叠加患者的实际生存结果
4. 选择代表性案例（高风险正确、低风险正确、预测失败）

**预计工作量**: **1-2 天**（含代码开发 + 案例筛选）

**脚本示例**：
```python
# scripts/visualize_transport_plan.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_patient_transport_plan(patient_id, transport_plan, 
                                  pathway_names, outcome):
    """
    transport_plan: [8 WSI slots, 8 genomic slots]
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(transport_plan, annot=True, fmt='.3f', 
                cmap='YlOrRd', ax=ax)
    ax.set_xlabel('Genomic Pathway Slots')
    ax.set_ylabel('WSI Patch Slots')
    ax.set_title(f'Patient {patient_id} | Outcome: {outcome}')
    plt.tight_layout()
    plt.savefig(f'patient_{patient_id}_transport.png', dpi=300)
```

**可以回答的问题**：
- 哪些病理特征映射到了哪些通路？
- 高风险患者的传输模式与低风险患者是否不同？

---

### 🔴 **不建议做**（投入产出比低）

#### 6️⃣ **病理编码器鲁棒性**（Figure 7、Table 9）

**SlotSPE 做了什么**：
- 测试不同预训练编码器（ResNet-50, ViT, CTransPath）
- 测试冻结 vs 微调编码器
- 测试不同分辨率

**DCT 当前情况**：
- ✅ 使用 UNI2-h（固定）
- ❌ 没有测试其他编码器的基础设施

**为什么不建议**：
1. **DCT 的贡献不在编码器**：论文核心是传输机制审计，不是编码器选择
2. **需要重新训练多个模型**：每个编码器 × 5 fold = 大量 GPU 时间
3. **SlotSPE 的编码器鲁棒性是为了证明槽注意力的通用性**，但 DCT 的重点是"审计传输是否真的有效"

**建议的替代方案**：
- 在论文中写一句："We use UNI2-h as the pathology encoder throughout this study. While encoder robustness is important for clinical deployment, our focus is on the validity of the transport mechanism itself, which is orthogonal to encoder choice."

---

#### 7️⃣ **队列级可解释性**（Figure 12–14 高级部分）

**SlotSPE 做了什么**：
- 聚合所有患者的运输计划
- 识别高频传输路径
- 分析不同癌种的传输模式差异

**为什么不建议**：
1. **需要大量患者数据**：单个癌种的 76 例测试集不足以识别稳定的模式
2. **需要额外的聚类/降维分析**：超出当前审计协议的范围
3. **论文已经有患者级审计（DCR/DMR）**：队列级可视化是锦上添花，不是必需

**建议**：
- 如果审稿人要求，可以作为 Supplementary Figure
- 但不应该在主文中占据大量篇幅

---

## 🎯 推荐的优先级

### **高优先级**（今天就可以做，无需 GPU）

1. ✅ **计算效率对比**（2-3 小时）
   - 统计参数量
   - 测量推理时间
   - 从 epoch_curve 提取训练时间

2. ✅ **校准曲线**（3-4 小时）
   - 用 lifelines 绘制校准曲线
   - 在 BLCA 5-fold 上评估

3. ✅ **临床变量增量**（2-3 小时）
   - 基于现有预测 + 临床变量训练 Cox 模型
   - 报告 Δ C-index

**总计**: **7-10 小时**，全部可在今天完成

---

### **中优先级**（如果有短期 GPU）

4. 🟡 **Sinkhorn epsilon 敏感性**（12-18 小时 GPU）
   - 测试 [0.01, 0.05, 0.1, 0.5]
   - 在 BLCA fold 0 上运行
   - 报告 C-index 和 DCR 的变化

5. 🟡 **患者级可解释性**（1-2 天）
   - 写可视化代码
   - 选择 3-5 个代表性案例
   - 绘制运输计划热图

---

### **低优先级**（不建议现在做）

6. 🔴 **病理编码器鲁棒性**（5-10 天 GPU + 大量工程）
   - DCT 贡献不在此
   - 可用文字说明替代

7. 🔴 **槽数量敏感性**（1-2 天 GPU）
   - 1 fold 结果说服力有限
   - 可用文字说明"我们选择 8 个槽是基于..."

8. 🔴 **队列级可解释性**（2-3 天开发 + 分析）
   - 超出论文范围
   - 可作为未来工作

---

## 📝 我可以立即为你做的

我现在可以写这 3 个脚本：

### **脚本 1**: `scripts/measure_computational_efficiency.py`
- 从检查点统计参数量
- 测量推理时间（CPU 或 GPU）
- 从 epoch_curve 提取训练时间
- 生成对比表格

### **脚本 2**: `scripts/plot_calibration_curves.py`
- 加载 BLCA 5-fold 的 predictions.pkl
- 用 lifelines 绘制校准曲线
- 计算校准指标（Expected vs Observed）

### **脚本 3**: `scripts/clinical_variable_incremental.py`
- 加载预测 + 临床变量
- 训练 Cox 模型（risk only vs risk + clinical）
- 报告 Δ C-index

---

## 🤔 你的决策

**选项 A**: 我立即写这 3 个脚本（计算效率、校准、临床增量）  
→ 预计 **2 小时**写完，你运行后 **今天**就能得到结果

**选项 B**: 你先确认一下哪些实验真的需要（比如是否有 GPU 做敏感性分析）  
→ 我根据你的优先级调整

**选项 C**: 我先检查一下 `predictions.pkl` 的具体格式，确保包含所需数据  
→ 避免写完脚本后发现数据不全

你希望我做什么？
